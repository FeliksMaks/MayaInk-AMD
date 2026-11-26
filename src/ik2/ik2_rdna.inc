// src/ik2/ik2_rdna.inc - maya-ik2's kernels for AMD's matrix cores (RDNA 3 / 3.5 gfx11, RDNA 4 gfx12, wave32), the
// paths where the CUDA kernels' mma.sync has no AMD instruction to map to (ik2_kernels.cu includes this inside
// ik2::k; the interface is in ik2_kernels.cuh).  The products: maya::kernels::wmma_f16 (16 x 16 x 16, F16 in, F32
// sums; include/strata/kernels/rdna_wmma.hpp).  Every sum runs in a fixed order, so a GPU's answers repeat; they are
// not the CUDA kernels' bits (another order of the same sums).
//
// The prompt's routed experts: a layer's experts over a chunk in three launches, straight from their blocks (NVFP4 or
// Q8_0, made FP16 in shared memory - exact: a code times its scale is an FP16), instead of a decode to FP16 and two
// BLAS products an expert.  The rows (a token's pick each) lie grouped by expert, in expert order; a tile is up to 32
// rows of one expert.  gate / up: a block 64 rows of gate and the same 64 of up against a tile's rows, SwiGLU in the
// registers (a lane holds a row's gate and up); down: 128 rows of down.  Then each token's picks summed in expert
// order (the order the CUDA path adds them in).
// (AMD builds only: ik2_kernels.cu includes it under STRATA_USE_HIP, after rdna_wmma.hpp)

namespace {

constexpr int kRmBM = 32, kRmBN = 128, kRmBK = 64, kRmLd = kRmBK + 8;   // (row pad: the lanes' rows on other banks)

// a block's W tile row: 64 values of one row (k-step ks) from its blocks into FP16 in shared memory
template <int TY>
__device__ __forceinline__ void rm_load_w(const uint8_t* row, int ks, uint32_t (&w)[17]) {
    if constexpr (TY == kNVFP4) {   // a 64-value block: 4 scales, 32 bytes of codes
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 36);
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(p + i);
    } else {   // Q8_0: two 32-value blocks (34 bytes each)
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 68);
#pragma unroll
        for (int i = 0; i < 17; ++i) w[i] = __ldg(p + i);
    }
}
template <int TY>
__device__ __forceinline__ void rm_store_w(const uint32_t (&w)[17], uint16_t* dst) {
    uint32_t h[32];
    if constexpr (TY == kNVFP4) {
#pragma unroll
        for (int sb = 0; sb < 4; ++sb) {   // sub-block sb: values 0..7 the low nibbles of its 8 bytes, 8..15 the high
            const uint32_t sc = ue4m3_h2((w[0] >> (8 * sb)) & 0xFFu);
            const uint32_t w0 = w[1 + 2 * sb], w1 = w[2 + 2 * sb];
            uint32_t* o = h + 8 * sb;
            o[0] = nv_pair_h2(w0, sc), o[1] = nv_pair_h2(w0 >> 16, sc);
            o[2] = nv_pair_h2(w1, sc), o[3] = nv_pair_h2(w1 >> 16, sc);
            o[4] = nv_pair_h2(w0 >> 4, sc), o[5] = nv_pair_h2(w0 >> 20, sc);
            o[6] = nv_pair_h2(w1 >> 4, sc), o[7] = nv_pair_h2(w1 >> 20, sc);
        }
    } else {
#pragma unroll
        for (int b = 0; b < 2; ++b) {   // block b at byte 34 b: its scale, then 32 int8 from byte 34 b + 2
            const int o0 = 34 * b;
            const uint32_t d = (w[o0 >> 2] >> ((o0 & 2) * 8)) & 0xFFFFu;
            const uint32_t sc = d | (d << 16);
#pragma unroll
            for (int p = 0; p < 16; ++p) {
                const int o = o0 + 2 + 2 * p;
                h[16 * b + p] = q8_h2((w[o >> 2] >> ((o & 2) * 8)) & 0xFFFFu, sc);
            }
        }
    }
    uint4* d4 = (uint4*) dst;
#pragma unroll
    for (int i = 0; i < 8; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}

#if defined(MAYA_RDNA_WMMA)
__device__ __forceinline__ maya::kernels::wmma_h16 rm_frag(const uint16_t* p) {
    union {
        uint4 u[2];
        maya::kernels::wmma_h16 v;
    } f;
    f.u[0] = *(const uint4*) p, f.u[1] = *(const uint4*) (p + 8);
    return f.v;
}
#endif

// GU: h [S][F] = SwiGLU(x . gate^T * sg, x . up^T * su) of a tile's rows; else y [S][C] = h . down^T.  grid (N / 64
// (GU) or C / 128, tiles), 128 threads: warp w the tile's rows 16 (w & 1) .., columns 64 (w >> 1) ..
template <int TY, bool GU>
__global__ void __launch_bounds__(128) experts_rdna_kernel(const uint8_t* const* __restrict__ eptr,
                                                           const int4* __restrict__ tiles, size_t off, size_t rb,
                                                           const uint16_t* __restrict__ x16, int K, int F, int C,
                                                           const float* __restrict__ sg, const float* __restrict__ su,
                                                           float gcap, float ucap, uint16_t* __restrict__ h16,
                                                           float* __restrict__ y32) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Xs[kRmBM * kRmLd];
    __shared__ __align__(16) uint16_t Ws[kRmBN * kRmLd];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mh = warp & 1, nh = warp >> 1;
    const int4 tl = tiles[blockIdx.y];   // (expert, first row, rows)
    const int row0 = tl.y, nrows = tl.z;
    const uint8_t* blob = eptr[tl.x];
    // this thread's W row: GU 64 gate rows then the same 64 of up, else 128 rows of down
    const uint8_t* wrow;
    if constexpr (GU) {
        const int r0 = blockIdx.x * 64;
        wrow = blob + (tid < 64 ? 0 : off) + (size_t) (r0 + (tid & 63)) * rb;
    } else {
        wrow = blob + off + (size_t) (blockIdx.x * 128 + tid) * rb;
    }
    // this thread's part of the X tile: row tid / 4, 16 values from column 16 (tid % 4)
    const int xr = tid >> 2, xc = (tid & 3) * 16;
    const bool xok = xr < nrows;
    const uint4* xsrc = (const uint4*) (x16 + (size_t) (row0 + (xok ? xr : 0)) * K + xc);
    const int nk = K / kRmBK;
    uint32_t w[17];
    uint4 xa, xb;
    rm_load_w<TY>(wrow, 0, w);
    xa = xok ? xsrc[0] : make_uint4(0u, 0u, 0u, 0u);
    xb = xok ? xsrc[1] : make_uint4(0u, 0u, 0u, 0u);
    wmma_f8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wmma_f8{};
    for (int ks = 0; ks < nk; ++ks) {
        __syncthreads();   // (the previous step's reads of the tiles are done)
        *(uint4*) (Xs + xr * kRmLd + xc) = xa;
        *(uint4*) (Xs + xr * kRmLd + xc + 8) = xb;
        rm_store_w<TY>(w, Ws + tid * kRmLd);
        __syncthreads();
        if (ks + 1 < nk) {   // (the next step's bytes in flight under this one's products)
            rm_load_w<TY>(wrow, ks + 1, w);
            xa = xok ? xsrc[(ks + 1) * (kRmBK / 8)] : make_uint4(0u, 0u, 0u, 0u);
            xb = xok ? xsrc[(ks + 1) * (kRmBK / 8) + 1] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int kk = 0; kk < kRmBK / 16; ++kk) {
            const wmma_h16 a = rm_frag(Xs + (16 * mh + (lane & 15)) * kRmLd + kk * 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int wr = GU ? (j < 2 ? 32 * nh + 16 * j : 64 + 32 * nh + 16 * (j - 2)) : 64 * nh + 16 * j;
                acc[j] = wmma_f16(a, rm_frag(Ws + (wr + (lane & 15)) * kRmLd + kk * 16), acc[j]);
            }
        }
    }
    const int hi = lane >> 4;
    if constexpr (GU) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = blockIdx.x * 64 + 32 * nh + 16 * j + (lane & 15);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 16 * mh + wmma_row(i, hi);
                if (m >= nrows) continue;
                const int r = row0 + m;
                const float g = acc[j][i] * sg[r], u = acc[j + 2][i] * su[r];
                h16[(size_t) r * F + col] =
                    f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = blockIdx.x * 128 + 64 * nh + 16 * j + (lane & 15);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 16 * mh + wmma_row(i, hi);
                if (m < nrows) y32[(size_t) (row0 + m) * C + col] = acc[j][i];
            }
        }
    }
#else
    __trap();   // (never launched: experts_rdna_available() is false without the matrix cores)
#endif
}

// (AMD port; the 128-row tiles and the skipped padding fragments are the design of CIRU AI's
// hip_moe_prefill.hip, Apache-2.0, Copyright (c) 2026 CIRU AI) experts_rdna_kernel with tiles of up to 128 rows of one
// expert: a 128-row weight tile is read and decoded once for four times the rows.  256 threads: warp w the tile's rows
// 32 (w >> 1) .. (two fragments), columns 64 (w & 1) ..; every output is the same WMMA chain in the same K order - the
// same bits.  A fragment past the tile's rows is skipped.
template <int TY>
__device__ __forceinline__ void rm_load_half(const uint8_t* row, int ks, int half, uint32_t (&w)[9]) {
    if constexpr (TY == kNVFP4) {
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 36);
        w[0] = __ldg(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[1 + i] = __ldg(p + 1 + 4 * half + i);
    } else {
        const uint32_t* p = (const uint32_t*) (row + (size_t) ks * 68) + 8 * half;
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = __ldg(p + i);
    }
}
template <int TY>
__device__ __forceinline__ void rm_store_half(const uint32_t (&w)[9], int half, uint16_t* dst) {
    uint32_t h[16];
    if constexpr (TY == kNVFP4) {
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            const uint32_t sc = ue4m3_h2((w[0] >> (8 * (2 * half + sb))) & 0xFFu);
            const uint32_t w0 = w[1 + 2 * sb], w1 = w[2 + 2 * sb];
            uint32_t* o = h + 8 * sb;
            o[0] = nv_pair_h2(w0, sc), o[1] = nv_pair_h2(w0 >> 16, sc);
            o[2] = nv_pair_h2(w1, sc), o[3] = nv_pair_h2(w1 >> 16, sc);
            o[4] = nv_pair_h2(w0 >> 4, sc), o[5] = nv_pair_h2(w0 >> 20, sc);
            o[6] = nv_pair_h2(w1 >> 4, sc), o[7] = nv_pair_h2(w1 >> 20, sc);
        }
    } else {   // (block `half` starts at byte 34 half; w from byte 32 half)
        const int o0 = 2 * half;
        const uint32_t d = (w[0] >> (8 * o0)) & 0xFFFFu;
        const uint32_t sc = d | (d << 16);
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            const int o = o0 + 2 + 2 * p;
            h[p] = q8_h2((w[o >> 2] >> ((o & 2) * 8)) & 0xFFFFu, sc);
        }
    }
    uint4* d4 = (uint4*) (dst + 32 * half);
#pragma unroll
    for (int i = 0; i < 4; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}
// (AMD port) the other formats of UD-IQ4_XS's experts (layers 20-23): 32 values of a row (its block / sub-block b32), the
// words they need, then the decode's own FP16 values (decode_value's, rounded once).  Values 0 .. 15 are the low nibbles of
// 16 bytes and 16 .. 31 the high ones (Q5_0, IQ4_NL, IQ4_XS); the K formats' 32 are one nibble of 32 bytes.
template <int TY>
constexpr int rm_words() {
    return TY == kQ5_K ? 20 : TY == kQ4_K ? 12 : TY == kIQ4_XS ? 6 : TY == kQ5_0 ? 6 : TY == kIQ4_NL ? 5 : 9;
}
template <int TY>
__device__ __forceinline__ void rm_load32(const uint8_t* row, int b32, uint32_t* w) {
    if constexpr (TY == kQ5_K || TY == kQ4_K) {
        const uint32_t* p = (const uint32_t*) (row + (size_t) (b32 >> 3) * (TY == kQ5_K ? 176 : 144));
        const int is = b32 & 7;
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = __ldg(p + i);
        const uint32_t* q = p + (TY == kQ5_K ? 12 : 4) + 8 * (is >> 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[4 + i] = __ldg(q + i);
        if constexpr (TY == kQ5_K) {
#pragma unroll
            for (int i = 0; i < 8; ++i) w[12 + i] = __ldg(p + 4 + i);
        }
    } else if constexpr (TY == kIQ4_XS) {
        const uint32_t* p = (const uint32_t*) (row + (size_t) (b32 >> 3) * 136);
        w[0] = __ldg(p), w[1] = __ldg(p + 1);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[2 + i] = __ldg(p + 2 + 4 * (b32 & 7) + i);
    } else {
        constexpr int BB = TY == kQ5_0 ? 22 : 18;
        constexpr int n = TY == kQ5_0 ? 6 : 5;
        const uint32_t* p = (const uint32_t*) (row + ((size_t) b32 * BB & ~(size_t) 3));
        uint32_t v[n];
#pragma unroll
        for (int i = 0; i < n; ++i) v[i] = __ldg(p + i);
        // (a block starts on a word or in the middle of one; w: from the block's first byte)
        const bool odd = (b32 * BB) & 2;
#pragma unroll
        for (int i = 0; i < n; ++i) w[i] = odd ? (v[i] >> 16) | (i + 1 < n ? v[i + 1 < n ? i + 1 : i] << 16 : 0u) : v[i];
    }
}
__device__ __forceinline__ uint32_t rm_iq4_bytes(uint32_t n) {   // four codes (a byte each, 0 .. 15) -> four int8
    // {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
    const uint32_t a = __builtin_amdgcn_perm(0xF6EADDCFu, 0xBFAD9881u, n & 0x07070707u);
    const uint32_t b = __builtin_amdgcn_perm(0x71594535u, 0x26190D01u, n & 0x07070707u);
    const uint32_t m = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (a & ~m) | (b & m);
}
template <int TY>
__device__ __forceinline__ void rm_store32(const uint32_t* w, int b32, uint16_t* dst) {
    uint32_t h[16];
    if constexpr (TY == kQ5_K || TY == kQ4_K) {
        const int is = b32 & 7;
        const float d = half_bits_to_float((uint16_t) (w[0] & 0xFFFFu)), mn = half_bits_to_float((uint16_t) (w[0] >> 16));
        auto by = [&](int i) { return (w[1 + (i >> 2)] >> (8 * (i & 3))) & 0xFFu; };   // (scales: 12 bytes, words 1 .. 3)
        uint32_t sc, m;
        const int i4 = is & 3;
        if (is < 4) {
            sc = by(i4) & 63u, m = by(i4 + 4) & 63u;
        } else {
            sc = (by(i4 + 8) & 0xFu) | ((by(i4) >> 6) << 4), m = (by(i4 + 8) >> 4) | ((by(i4 + 4) >> 6) << 4);
        }
        const float dsc = d * (float) sc, mnm = mn * (float) m;
        const int sh = (is & 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t v = (w[4 + i] >> sh) & 0x0F0F0F0Fu;
            if constexpr (TY == kQ5_K) v |= ((w[12 + i] >> is) & 0x01010101u) << 4;
            const float f0 = dsc * (float) (v & 0xFFu) - mnm, f1 = dsc * (float) ((v >> 8) & 0xFFu) - mnm;
            const float f2 = dsc * (float) ((v >> 16) & 0xFFu) - mnm, f3 = dsc * (float) (v >> 24) - mnm;
            h[2 * i] = f2_to_u32(f0, f1), h[2 * i + 1] = f2_to_u32(f2, f3);
        }
    } else {
        float dl = half_bits_to_float((uint16_t) (w[0] & 0xFFFFu));
        if constexpr (TY == kIQ4_XS) {
            const int ib = b32 & 7;
            const int ls = (int) ((w[1] >> (4 * ib)) & 0xFu) | (int) (((w[0] >> (16 + 2 * ib)) & 3u) << 4);
            dl = dl * (float) (ls - 32);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t c;   // 4 bytes of codes
            if constexpr (TY == kIQ4_XS) c = w[2 + i];
            else if constexpr (TY == kIQ4_NL) c = (w[i] >> 16) | (w[i + 1] << 16);
            else c = (w[1 + i] >> 16) | (w[1 + i + 1 < 6 ? 2 + i : 5] << 16);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                uint32_t v = (c >> (4 * half)) & 0x0F0F0F0Fu;
                int q[4];
                if constexpr (TY == kQ5_0) {
                    const uint32_t qh = (w[0] >> 16) | (w[1] << 16);
                    const uint32_t n = qh >> (16 * half + 4 * i);
#pragma unroll
                    for (int k = 0; k < 4; ++k) q[k] = (int) (((v >> (8 * k)) & 0xFu) | (((n >> k) & 1u) << 4)) - 16;
                } else {
                    v = rm_iq4_bytes(v);
#pragma unroll
                    for (int k = 0; k < 4; ++k) q[k] = (int) (int8_t) (v >> (8 * k));
                }
                h[8 * half + 2 * i] = f2_to_u32(dl * (float) q[0], dl * (float) q[1]);
                h[8 * half + 2 * i + 1] = f2_to_u32(dl * (float) q[2], dl * (float) q[3]);
            }
        }
    }
    uint4* d4 = (uint4*) dst;
#pragma unroll
    for (int i = 0; i < 4; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}
template <int TY>
constexpr bool rm_mixed() { return TY != kNVFP4 && TY != kQ8_0; }

template <int TY, bool GU>
__global__ void __launch_bounds__(256) experts_rdna128_kernel(const uint8_t* const* __restrict__ eptr,
                                                              const int4* __restrict__ tiles, size_t off, size_t rb,
                                                              const uint16_t* __restrict__ x16, int K, int F, int C,
                                                              const float* __restrict__ sg, const float* __restrict__ su,
                                                              float gcap, float ucap, uint16_t* __restrict__ h16,
                                                              float* __restrict__ y32) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Xs[128 * kRmLd];
    __shared__ __align__(16) uint16_t Ws[128 * kRmLd];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, nh = warp & 1;
    const int4 tl = tiles[blockIdx.y];
    const int row0 = tl.y, nrows = tl.z;
    const uint8_t* blob = eptr[tl.x];
    const int wr_i = tid & 127, half = tid >> 7;
    const uint8_t* wrow;
    if constexpr (GU) {
        const int r0 = blockIdx.x * 64;
        wrow = blob + (wr_i < 64 ? 0 : off) + (size_t) (r0 + (wr_i & 63)) * rb;
    } else {
        wrow = blob + off + (size_t) (blockIdx.x * 128 + wr_i) * rb;
    }
    const int xr = tid >> 1, xc = (tid & 1) * 32;   // this thread's part of the X tile: 32 values of a row
    const bool xok = xr < nrows;
    const uint4* xsrc = (const uint4*) (x16 + (size_t) (row0 + (xok ? xr : 0)) * K + xc);
    const bool on0 = 32 * wm < nrows, on1 = 32 * wm + 16 < nrows;
    const int nk = K / kRmBK;
    uint32_t w[9];
    uint4 xv[4];
    auto load_x = [&](int ks) {
#pragma unroll
        for (int i = 0; i < 4; ++i) xv[i] = xok ? xsrc[ks * (kRmBK / 8) + i] : make_uint4(0u, 0u, 0u, 0u);
    };
    rm_load_half<TY>(wrow, 0, half, w);
    load_x(0);
    wmma_f8 acc[2][4];
#pragma unroll
    for (int im = 0; im < 2; ++im)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[im][j] = wmma_f8{};
    for (int ks = 0; ks < nk; ++ks) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) *(uint4*) (Xs + xr * kRmLd + xc + 8 * i) = xv[i];
        rm_store_half<TY>(w, half, Ws + wr_i * kRmLd);
        __syncthreads();
        if (ks + 1 < nk) {
            rm_load_half<TY>(wrow, ks + 1, half, w);
            load_x(ks + 1);
        }
        if (on0) {
#pragma unroll
            for (int kk = 0; kk < kRmBK / 16; ++kk) {
                const wmma_h16 a0 = rm_frag(Xs + (32 * wm + (lane & 15)) * kRmLd + kk * 16);
                wmma_h16 a1{};
                if (on1) a1 = rm_frag(Xs + (32 * wm + 16 + (lane & 15)) * kRmLd + kk * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int wr = GU ? (j < 2 ? 32 * nh + 16 * j : 64 + 32 * nh + 16 * (j - 2)) : 64 * nh + 16 * j;
                    const wmma_h16 b = rm_frag(Ws + (wr + (lane & 15)) * kRmLd + kk * 16);
                    acc[0][j] = wmma_f16(a0, b, acc[0][j]);
                    if (on1) acc[1][j] = wmma_f16(a1, b, acc[1][j]);
                }
            }
        }
    }
    const int hi = lane >> 4;
#pragma unroll
    for (int im = 0; im < 2; ++im) {
        if (im ? !on1 : !on0) continue;
        if constexpr (GU) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = blockIdx.x * 64 + 32 * nh + 16 * j + (lane & 15);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int m = 32 * wm + 16 * im + wmma_row(i, hi);
                    if (m >= nrows) continue;
                    const int r = row0 + m;
                    const float g = acc[im][j][i] * sg[r], u = acc[im][j + 2][i] * su[r];
                    h16[(size_t) r * F + col] =
                        f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = blockIdx.x * 128 + 64 * nh + 16 * j + (lane & 15);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int m = 32 * wm + 16 * im + wmma_row(i, hi);
                    if (m < nrows) y32[(size_t) (row0 + m) * C + col] = acc[im][j][i];
                }
            }
        }
    }
#else
    __trap();
#endif
}

// (AMD port; after CIRU AI's 256-column tiles, Apache-2.0) ... 128 rows x 256 weight rows (GU: 128 of gate and
// the same of up), K in steps of 32: 30 KB of shared memory; warp w the rows 64 (w >> 2) .., the weight rows 64 (w & 3) ..
template <int TY, bool GU>
__global__ void __launch_bounds__(256) experts_rdna256_kernel(const uint8_t* const* __restrict__ eptr,
                                                              const int4* __restrict__ tiles, size_t off, size_t rb,
                                                              const uint16_t* __restrict__ x16, int K, int F, int C,
                                                              const float* __restrict__ sg, const float* __restrict__ su,
                                                              float gcap, float ucap, uint16_t* __restrict__ h16,
                                                              float* __restrict__ y32, const int* __restrict__ gidx) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    constexpr int LD = 40;
    __shared__ __align__(16) uint16_t Xs[128 * LD];
    __shared__ __align__(16) uint16_t Ws[256 * LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
    const int4 tl = tiles[blockIdx.y];
    const int row0 = tl.y, nrows = tl.z;
    const uint8_t* blob = eptr[tl.x];
    const uint8_t* wrow;
    if constexpr (GU) wrow = blob + (tid < 128 ? 0 : off) + (size_t) (blockIdx.x * 128 + (tid & 127)) * rb;
    else wrow = blob + off + (size_t) (blockIdx.x * 256 + tid) * rb;
    const int xr = tid >> 1, xc = (tid & 1) * 16;
    const bool xok = xr < nrows;
    // (gidx: the rows' places in x16, else they lie in order)
    const int xrow = row0 + (xok ? xr : 0);
    const uint4* xsrc = (const uint4*) (x16 + (size_t) (gidx ? gidx[xrow] : xrow) * K + xc);
    // (this warp's 16-row fragments that hold rows)
    int nf = 0;
#pragma unroll
    for (int im = 0; im < 4; ++im) nf += 64 * wm + 16 * im < nrows ? 1 : 0;
    const int nk = K / 32;
    uint32_t w[rm_words<TY>()];
    uint4 xa, xb;
    if constexpr (rm_mixed<TY>()) rm_load32<TY>(wrow, 0, w);
    else rm_load_half<TY>(wrow, 0, 0, w);
    xa = xok ? xsrc[0] : make_uint4(0u, 0u, 0u, 0u), xb = xok ? xsrc[1] : make_uint4(0u, 0u, 0u, 0u);
    wmma_f8 acc[4][4];
#pragma unroll
    for (int im = 0; im < 4; ++im)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[im][j] = wmma_f8{};
    for (int ks = 0; ks < nk; ++ks) {
        __syncthreads();
        *(uint4*) (Xs + xr * LD + xc) = xa;
        *(uint4*) (Xs + xr * LD + xc + 8) = xb;
        if constexpr (rm_mixed<TY>()) rm_store32<TY>(w, ks, Ws + tid * LD);
        else rm_store_half<TY>(w, ks & 1, Ws + tid * LD - 32 * (ks & 1));
        __syncthreads();
        if (ks + 1 < nk) {
            if constexpr (rm_mixed<TY>()) rm_load32<TY>(wrow, ks + 1, w);
            else rm_load_half<TY>(wrow, (ks + 1) >> 1, (ks + 1) & 1, w);
            xa = xok ? xsrc[(ks + 1) * 4] : make_uint4(0u, 0u, 0u, 0u);
            xb = xok ? xsrc[(ks + 1) * 4 + 1] : make_uint4(0u, 0u, 0u, 0u);
        }
        if (nf) {
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                wmma_h16 a[4];
#pragma unroll
                for (int im = 0; im < 4; ++im)
                    if (im < nf) a[im] = rm_frag(Xs + (64 * wm + 16 * im + (lane & 15)) * LD + kk * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int wr = GU ? (j < 2 ? 32 * wn + 16 * j : 128 + 32 * wn + 16 * (j - 2)) : 64 * wn + 16 * j;
                    const wmma_h16 b = rm_frag(Ws + (wr + (lane & 15)) * LD + kk * 16);
#pragma unroll
                    for (int im = 0; im < 4; ++im)
                        if (im < nf) acc[im][j] = wmma_f16(a[im], b, acc[im][j]);
                }
            }
        }
    }
    const int hi = lane >> 4;
#pragma unroll
    for (int im = 0; im < 4; ++im) {
        if (im >= nf) continue;
        if constexpr (GU) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = blockIdx.x * 128 + 32 * wn + 16 * j + (lane & 15);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int m = 64 * wm + 16 * im + wmma_row(i, hi);
                    if (m >= nrows) continue;
                    const int r = row0 + m;
                    const float g = acc[im][j][i] * sg[r], u = acc[im][j + 2][i] * su[r];
                    h16[(size_t) r * F + col] =
                        f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = blockIdx.x * 256 + 64 * wn + 16 * j + (lane & 15);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int m = 64 * wm + 16 * im + wmma_row(i, hi);
                    if (m < nrows) y32[(size_t) (row0 + m) * C + col] = acc[im][j][i];
                }
            }
        }
    }
#else
    __trap();
#endif
}

// (AMD port) a dense matrix's prompt product on the matrix cores straight from its planar Q8_0 (int8
// values [N][K], FP16 scales [N][K / 32]): y [T][ldy] = x16 [T][K] . W^T, F32 sums - no FP16 decode of W and no BLAS.
// The FP16 values are the decode's own (f16(d) x q rounded once, as decode_q8p_f16).  grid (ceil(N / 128),
// ceil(T / 32)), 128 threads: warp w the tile's rows 16 (w & 1) .., columns 64 (w >> 1) ..
// F16W: W is FP16 [N][K] (q its values, qd unused) - the same kernel for an FP16 matrix (gemm16's job)
template <bool F16W>
__global__ void __launch_bounds__(128) dense_q8p_rdna_kernel(const int8_t* __restrict__ q,
                                                             const uint16_t* __restrict__ qd, int N, int K,
                                                             const uint16_t* __restrict__ x16, int T,
                                                             float* __restrict__ y, int ldy) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Xs[kRmBM * kRmLd];
    __shared__ __align__(16) uint16_t Ws[kRmBN * kRmLd];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, mh = warp & 1, nh = warp >> 1;
    const int t0 = blockIdx.y * kRmBM, nrows = min(kRmBM, T - t0);
    const int wrow_i = min((int) blockIdx.x * 128 + tid, N - 1);
    const uint32_t* qrow = (const uint32_t*) (q + (size_t) wrow_i * K * (F16W ? 2 : 1));
    const uint16_t* drow = qd + (size_t) wrow_i * (K / 32);
    const int xr = tid >> 2, xc = (tid & 3) * 16;
    const bool xok = xr < nrows;
    const uint4* xsrc = (const uint4*) (x16 + (size_t) (t0 + (xok ? xr : 0)) * K + xc);
    const int nk = K / kRmBK;
    uint4 wq[8];
    uint32_t wd = 0;
    auto load_w = [&](int ks) {
        if constexpr (F16W) {   // (64 FP16 values: 8 x 16 bytes)
            const uint4* p = (const uint4*) (qrow + ks * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) wq[i] = __ldg(p + i);
        } else {
            const uint4* p = (const uint4*) (qrow + ks * 16);
            wq[0] = __ldg(p), wq[1] = __ldg(p + 1), wq[2] = __ldg(p + 2), wq[3] = __ldg(p + 3);
            wd = __ldg((const uint32_t*) (drow + 2 * ks));   // (the two 32-value blocks' scales)
        }
    };
    load_w(0);
    uint4 xa = xok ? xsrc[0] : make_uint4(0u, 0u, 0u, 0u), xb = xok ? xsrc[1] : make_uint4(0u, 0u, 0u, 0u);
    wmma_f8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = wmma_f8{};
    for (int ks = 0; ks < nk; ++ks) {
        __syncthreads();
        *(uint4*) (Xs + xr * kRmLd + xc) = xa;
        *(uint4*) (Xs + xr * kRmLd + xc + 8) = xb;
        if constexpr (F16W) {
            uint4* d4 = (uint4*) (Ws + tid * kRmLd);
#pragma unroll
            for (int i = 0; i < 8; ++i) d4[i] = wq[i];
        } else {
            const uint32_t s0 = (wd & 0xFFFFu) * 0x10001u, s1 = (wd >> 16) * 0x10001u;
            const uint32_t ww[16] = {wq[0].x, wq[0].y, wq[0].z, wq[0].w, wq[1].x, wq[1].y, wq[1].z, wq[1].w,
                                     wq[2].x, wq[2].y, wq[2].z, wq[2].w, wq[3].x, wq[3].y, wq[3].z, wq[3].w};
            uint32_t h[32];
#pragma unroll
            for (int p = 0; p < 32; ++p) h[p] = q8_h2((ww[p >> 1] >> (16 * (p & 1))) & 0xFFFFu, p < 16 ? s0 : s1);
            uint4* d4 = (uint4*) (Ws + tid * kRmLd);
#pragma unroll
            for (int i = 0; i < 8; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
        }
        __syncthreads();
        if (ks + 1 < nk) {
            load_w(ks + 1);
            xa = xok ? xsrc[(ks + 1) * (kRmBK / 8)] : make_uint4(0u, 0u, 0u, 0u);
            xb = xok ? xsrc[(ks + 1) * (kRmBK / 8) + 1] : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int kk = 0; kk < kRmBK / 16; ++kk) {
            const wmma_h16 a = rm_frag(Xs + (16 * mh + (lane & 15)) * kRmLd + kk * 16);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[j] = wmma_f16(a, rm_frag(Ws + (64 * nh + 16 * j + (lane & 15)) * kRmLd + kk * 16), acc[j]);
        }
    }
    const int hi = lane >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = blockIdx.x * 128 + 64 * nh + 16 * j + (lane & 15);
        if (col >= N) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = 16 * mh + wmma_row(i, hi);
            if (m < nrows) y[(size_t) (t0 + m) * ldy + col] = acc[j][i];
        }
    }
#else
    __trap();
#endif
}

// (AMD port) ... the same in tiles of 128 rows of X: a weight tile decoded once for four times the rows (experts_rdna128's
// layout; the same WMMA chains, the same bits)
template <bool F16W>
__global__ void __launch_bounds__(256) dense_rdna128_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ qd,
                                                            int N, int K, const uint16_t* __restrict__ x16, int T,
                                                            float* __restrict__ y, int ldy) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Xs[128 * kRmLd];
    __shared__ __align__(16) uint16_t Ws[128 * kRmLd];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, nh = warp & 1;
    const int t0 = blockIdx.y * 128, nrows = min(128, T - t0);
    const int wr_i = tid & 127, half = tid >> 7, wrow_i = min((int) blockIdx.x * 128 + wr_i, N - 1);
    const uint4* qrow = (const uint4*) ((const uint8_t*) q + (size_t) wrow_i * K * (F16W ? 2 : 1));
    const uint16_t* drow = qd + (size_t) wrow_i * (K / 32);
    const int xr = tid >> 1, xc = (tid & 1) * 32;
    const bool xok = xr < nrows;
    const uint4* xsrc = (const uint4*) (x16 + (size_t) (t0 + (xok ? xr : 0)) * K + xc);
    const bool on0 = 32 * wm < nrows, on1 = 32 * wm + 16 < nrows;
    const int nk = K / kRmBK;
    uint4 wq[4], xv[4];
    uint32_t wd = 0;
    auto load = [&](int ks) {
        if constexpr (F16W) {
#pragma unroll
            for (int i = 0; i < 4; ++i) wq[i] = __ldg(qrow + ks * 8 + half * 4 + i);
        } else {
            wq[0] = __ldg(qrow + ks * 4 + half * 2), wq[1] = __ldg(qrow + ks * 4 + half * 2 + 1);
            wd = __ldg(drow + 2 * ks + half);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) xv[i] = xok ? xsrc[ks * (kRmBK / 8) + i] : make_uint4(0u, 0u, 0u, 0u);
    };
    load(0);
    wmma_f8 acc[2][4];
#pragma unroll
    for (int im = 0; im < 2; ++im)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[im][j] = wmma_f8{};
    for (int ks = 0; ks < nk; ++ks) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) *(uint4*) (Xs + xr * kRmLd + xc + 8 * i) = xv[i];
        uint4* d4 = (uint4*) (Ws + wr_i * kRmLd + 32 * half);
        if constexpr (F16W) {
#pragma unroll
            for (int i = 0; i < 4; ++i) d4[i] = wq[i];
        } else {
            const uint32_t sc = wd * 0x10001u;
            const uint32_t ww[8] = {wq[0].x, wq[0].y, wq[0].z, wq[0].w, wq[1].x, wq[1].y, wq[1].z, wq[1].w};
            uint32_t h[16];
#pragma unroll
            for (int p = 0; p < 16; ++p) h[p] = q8_h2((ww[p >> 1] >> (16 * (p & 1))) & 0xFFFFu, sc);
#pragma unroll
            for (int i = 0; i < 4; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
        }
        __syncthreads();
        if (ks + 1 < nk) load(ks + 1);
        if (on0) {
#pragma unroll
            for (int kk = 0; kk < kRmBK / 16; ++kk) {
                const wmma_h16 a0 = rm_frag(Xs + (32 * wm + (lane & 15)) * kRmLd + kk * 16);
                wmma_h16 a1{};
                if (on1) a1 = rm_frag(Xs + (32 * wm + 16 + (lane & 15)) * kRmLd + kk * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const wmma_h16 b = rm_frag(Ws + (64 * nh + 16 * j + (lane & 15)) * kRmLd + kk * 16);
                    acc[0][j] = wmma_f16(a0, b, acc[0][j]);
                    if (on1) acc[1][j] = wmma_f16(a1, b, acc[1][j]);
                }
            }
        }
    }
    const int hi = lane >> 4;
#pragma unroll
    for (int im = 0; im < 2; ++im) {
        if (im ? !on1 : !on0) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = blockIdx.x * 128 + 64 * nh + 16 * j + (lane & 15);
            if (col >= N) continue;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 32 * wm + 16 * im + wmma_row(i, hi);
                if (m < nrows) y[(size_t) (t0 + m) * ldy + col] = acc[im][j][i];
            }
        }
    }
#else
    __trap();
#endif
}

// out [T][C] = sum of a token's picks' wtr[row] y[row], in row order (= expert order); a block a token
__global__ void moe_rows_combine_kernel(const float* __restrict__ y, const int* __restrict__ inv,
                                        const float* __restrict__ wtr, int K, int C, float* __restrict__ out) {
    const int t = blockIdx.x;
    int r[16];
    const int kk = K < 16 ? K : 16;
    for (int k = 0; k < kk; ++k) {   // (insertion sort of a token's few rows)
        int v = inv[(size_t) t * K + k], j = k;
        while (j > 0 && r[j - 1] > v) r[j] = r[j - 1], --j;
        r[j] = v;
    }
    for (int c4 = threadIdx.x; c4 < C / 4; c4 += blockDim.x) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int k = 0; k < kk; ++k) {
            const float w = wtr[r[k]];
            const float4 v = ((const float4*) (y + (size_t) r[k] * C))[c4];
            a.x = fmaf(w, v.x, a.x), a.y = fmaf(w, v.y, a.y), a.z = fmaf(w, v.z, a.z), a.w = fmaf(w, v.w, a.w);
        }
        ((float4*) (out + (size_t) t * C))[c4] = a;
    }
}

// The prompt's fused attention (attn_flash_f16_kernel's job and outputs: a split's unnormalised rows and (max, sum),
// merged with the sink by attn_flash_combine_kernel) on the matrix cores.  A block: 64 queries (4 warps x 16) of one
// head against a split of its keys in tiles of 32; S = Q K^T (Q in the registers, K as it lies), the online softmax
// in the log2 domain on S's layout (a lane: 8 query rows x one key a 16-key column; the relative bias read where it
// lies), P through the warp's own shared memory into the A layout, O += P V from V stored transposed.  24 KB of
// shared memory a block, so that several blocks share a CU (64 keys a tile with the bias in shared memory: 54 KB,
// one block a CU, 1.3x slower).
constexpr int kRfBN = 32, kRfLd = kFaD + 8, kRfVld = kRfBN + 8;
template <class KVT, int kRfNW, bool INL = false>
__global__ void __launch_bounds__(kRfNW * 32)
attn_flash_rdna_kernel(const uint16_t* __restrict__ q16, int q_ld, const KVT* __restrict__ kc,
                       const KVT* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                       const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                       float2* __restrict__ ml_part, const float* __restrict__ rq = nullptr,
                       const float* __restrict__ proj = nullptr) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    __shared__ __align__(16) uint16_t Ks[kRfBN * kRfLd];          // [key][dim]
    __shared__ __align__(16) uint16_t Vt[kFaD * kRfVld];          // [dim][key]
    __shared__ __align__(16) uint16_t Pss[kRfNW * 16 * kRfVld];   // each warp's P [16][key]
    __shared__ __align__(16) float Rs[INL ? 16 * kRfNW * 16 : 4];   // (INL: the block's queries' 16 relative values)
    constexpr bool F8 = sizeof(KVT) == 1;
    const int tid = threadIdx.x, warp = tid >> 5, hi = (tid & 31) >> 4, ln = tid & 15;
    uint16_t* Ps = Pss + warp * 16 * kRfVld;
    const int h = blockIdx.y, split = blockIdx.z, g = h / (H / KV), KD = KV * kFaD;
    constexpr int BM = 16 * kRfNW;
    // (the tiles are the CUDA kernel's 64-key ones cut in two: the same keys, so attn_flash_splits' counts hold)
    const int i0 = blockIdx.x * BM, i_last = min(Tq, i0 + BM) - 1;
    const int k_hi = qpos0 + i_last + 1, k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kRfBN - 1) / kRfBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);
    const int wi = i0 + warp * 16;
    if constexpr (INL) {
        for (int c = tid; c < BM * 4; c += kRfNW * 32) {
            const int qi = i0 + (c >> 2);
            ((float4*) Rs)[c] = qi < Tq ? ((const float4*) (rq + ((size_t) qi * H + h) * 16))[c & 3] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
    }
    wmma_h16 qf[kFaD / 16];   // (the lane's query row wi + ln, its 128 dims)
    {
        const int iq = wi + ln;
        const uint16_t* qr = q16 + (size_t) (iq < Tq ? iq : 0) * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) qf[kk] = iq < Tq ? rm_frag(qr + kk * 16) : wmma_h16{};
    }
    wmma_f8 o[kFaD / 16];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f8{};
    float mx[8], ls[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) mx[i] = -INFINITY, ls[i] = 0.0f;
    const float sc = qk_scale * kLog2e;
    // A tile's values: K by (key c / 16, 8 dims from 8 (c % 16)), V by a pair of keys (tid % 16) x 16 dims (tid / 16),
    // so that the transposed V goes in as words, a half-warp a row's 16 words.
    using Raw = typename std::conditional<F8, uint2, uint4>::type;
    Raw zr;
    __builtin_memset(&zr, 0, sizeof zr);   // (AMD port on Windows: the builtin in device code)
    const int vp = tid & 15, vg = tid >> 4;
    auto h8 = [&](const Raw& r) -> uint4 {
        if constexpr (F8) return e4m3x8_h8(r);
        else return r;
    };
    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * kRfBN;
        // does this warp's rows see any key of the tile, and is any distance within the bias's extent?
        const bool any = !(kt > qpos0 + wi + 15 || kt + kRfBN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + kRfBN - 1) < extent;
        // (AMD port: the bias asked for before the tile's barriers and products - the wait for it in their middle was more
        //  than half of this kernel on the 8060S)
        float bv[8][kRfBN / 16];
        if (!INL && any && near) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int ia = wi + wmma_row(i, hi);
                const float* brow = bias + ((size_t) (ia < Tq ? ia : 0) * H + h) * extent;
#pragma unroll
                for (int j = 0; j < kRfBN / 16; ++j) {
                    const int dist = qpos0 + ia - (kt + 16 * j + ln);
                    bv[i][j] = brow[dist < 0 ? 0 : (dist < extent ? dist : extent - 1)];
                }
            }
        }
        __syncthreads();   // (the previous tile's reads done)
#pragma unroll
        for (int it = 0; it < (kRfBN * kFaD / 8) / (kRfNW * 32); ++it) {
            const int c = it * kRfNW * 32 + tid, row = c >> 4, col = (c & 15) * 8, kp = kt + row;
            const bool ok = kp < k_hi;
            const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
            *(uint4*) (Ks + row * kRfLd + col) = h8(ok ? *(const Raw*) (kc + (size_t) pr * KD + (size_t) g * kFaD + col) : zr);
        }
        if (tid < 128) {   // (V: 128 threads' work; more queries a block share it)
            Raw vr[2][2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int kp = kt + 2 * vp + j;
                const bool ok = kp < k_hi;
                const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
                const KVT* src = vc + (size_t) pr * KD + (size_t) g * kFaD + 16 * vg;
                vr[j][0] = ok ? *(const Raw*) src : zr;
                vr[j][1] = ok ? *(const Raw*) (src + 8) : zr;
            }
            uint32_t* Vt32 = (uint32_t*) Vt;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint4 a = h8(vr[0][i]), b = h8(vr[1][i]);
                const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int d = 16 * vg + 8 * i + 2 * m;
                    Vt32[d * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x5410);
                    Vt32[(d + 1) * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x7632);
                }
            }
        }
        __syncthreads();
        if (!any) continue;
        wmma_f8 s[kRfBN / 16];
#pragma unroll
        for (int j = 0; j < kRfBN / 16; ++j) s[j] = wmma_f8{};
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk)
#pragma unroll
            for (int j = 0; j < kRfBN / 16; ++j)
                s[j] = wmma_f16(qf[kk], rm_frag(Ks + (16 * j + ln) * kRfLd + kk * 16), s[j]);
        // logits (log2 units): scale, bias (the table's value at the distance, 0 out of its extent), mask; the rows'
        // new maxima, the output's rescale
        float u[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = wmma_row(i, hi), ia = wi + m, qa = qpos0 + ia;
            float mxi = -INFINITY;
#pragma unroll
            for (int j = 0; j < kRfBN / 16; ++j) {
                const int kp = kt + 16 * j + ln, dist = qa - kp;
                float v = s[j][i] * sc;
                if (near && ia < Tq && dist >= 0 && dist < extent) {
                    if constexpr (INL) {   // (relb_kernel's order: the same bits as the table)
                        const float4* pp = (const float4*) (proj + (size_t) dist * 16);
                        const float4* rr = (const float4*) (Rs + (warp * 16 + wmma_row(i, hi)) * 16);
                        float b = 0.0f;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float4 x = pp[e], y = rr[e];
                            b = fmaf(x.w, y.w, fmaf(x.z, y.z, fmaf(x.y, y.y, fmaf(x.x, y.x, b))));
                        }
                        v += b * kLog2e;
                    } else {
                        v += bv[i][j] * kLog2e;
                    }
                }
                v = (dist < 0 || kp < qa - window + 1 || ia >= Tq) ? -INFINITY : v;
                s[j][i] = v;
                mxi = fmaxf(mxi, v);
            }
#pragma unroll
            for (int sh = 1; sh < 16; sh <<= 1) mxi = fmaxf(mxi, __shfl_xor_sync(0xffffffffu, mxi, sh));
            const float mn = fmaxf(mx[i], mxi);
            const float cf = mn == -INFINITY ? 1.0f : exp2f(mx[i] - mn);
            u[i] = mn == -INFINITY ? 0.0f : mn;
            mx[i] = mn;
            ls[i] *= cf;
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j][i] *= cf;
        }
#pragma unroll
        for (int j = 0; j < kRfBN / 16; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = exp2f(s[j][i] - u[i]);
                ls[i] += p;
                Ps[wmma_row(i, hi) * kRfVld + 16 * j + ln] = f2h(p);
            }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < kRfBN / 16; ++kk) {
            const wmma_h16 a = rm_frag(Ps + ln * kRfVld + kk * 16);
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f16(a, rm_frag(Vt + (16 * j + ln) * kRfVld + kk * 16), o[j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int sh = 1; sh < 16; sh <<= 1) ls[i] += __shfl_xor_sync(0xffffffffu, ls[i], sh);
    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H] = (max in log2 units, sum)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int ia = wi + wmma_row(i, hi);
        if (ia >= Tq) continue;
        float* op = o_part + (((size_t) split * Tq + ia) * H + h) * kFaD + ln;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) op[16 * j] = o[j][i];
        if (ln == 0) ml_part[((size_t) split * Tq + ia) * H + h] = make_float2(mx[i], ls[i]);
    }
#else
    __trap();   // (never launched: attn_flash_available() is false without the matrix cores)
#endif
}

// (AMD port; the swapped products - S^T = K Q^T, O^T = V^T P^T, P put together in the registers - are the design of CIRU AI's
// hip_attn_prefill.hip, Apache-2.0, Copyright (c) 2026 CIRU AI) attn_flash_rdna_kernel's job with a lane per query: a lane
// holds 8 of a 16-key column's logits of its own query, so a tile's maximum and sum are one exchange between the half-warps
// and P needs no shared memory.  FP16 cache.  The next tile's K and V are asked for before this tile's products.
template <int NW, bool INL>
__global__ void __launch_bounds__(NW * 32)
attn_flash2_rdna_kernel(const uint16_t* __restrict__ q16, int q_ld, const uint16_t* __restrict__ kc,
                        const uint16_t* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                        const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                        float2* __restrict__ ml_part, const float* __restrict__ rq, const float* __restrict__ proj) {
#if defined(MAYA_RDNA_WMMA)
    using namespace maya::kernels;
    constexpr int BN = 32, NT = NW * 32, BM = 16 * NW, NJ = BN / 16;
    constexpr int NKL = BN * 16 / NT, NVL = BN / 2 * 16 / NT;   // a thread's K words (uint4) and V pairs of keys
    __shared__ __align__(16) uint16_t Ks[BN * kRfLd];
    __shared__ __align__(16) uint16_t Vt[kFaD * kRfVld];
    const int tid = threadIdx.x, warp = tid >> 5, hi = (tid & 31) >> 4, ln = tid & 15;
    const int h = blockIdx.y, split = blockIdx.z, g = h / (H / KV), KD = KV * kFaD;
    const int i0 = blockIdx.x * BM, i_last = min(Tq, i0 + BM) - 1;
    const int k_hi = qpos0 + i_last + 1, k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + BN - 1) / BN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);
    const int wi = i0 + warp * 16, iq = wi + ln, qa = qpos0 + iq;
    const bool qok = iq < Tq;
    wmma_h16 qf[kFaD / 16];
    {
        const uint16_t* qr = q16 + (size_t) (qok ? iq : 0) * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) qf[kk] = qok ? rm_frag(qr + kk * 16) : wmma_h16{};
    }
    float4 rv[4];   // (INL: the lane's query's 16 relative values)
    if constexpr (INL) {
        const float4* rp = (const float4*) (rq + ((size_t) (qok ? iq : 0) * H + h) * 16);
#pragma unroll
        for (int v = 0; v < 4; ++v) rv[v] = rp[v];
    }
    const float* brow = bias + ((size_t) (qok ? iq : 0) * H + h) * extent;
    wmma_f8 o[kFaD / 16];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j] = wmma_f8{};
    float mx = -INFINITY, ls = 0.0f;
    const float sc = qk_scale * kLog2e;
    uint4 kr[NKL], vr[NVL][2];
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    auto row = [&](int kp) { return (size_t) (kv_cap > 0 ? kp % kv_cap : kp) * KD; };
    auto load = [&](int t) {
        const int kt = k_lo + t * BN;
#pragma unroll
        for (int it = 0; it < NKL; ++it) {
            const int c = it * NT + tid, kp = kt + (c >> 4);
            const bool ok = kp < k_hi;
            kr[it] = ok ? *(const uint4*) (kc + row(ok ? kp : 0) + (size_t) g * kFaD + (c & 15) * 8) : z4;
        }
#pragma unroll
        for (int it = 0; it < NVL; ++it) {   // (a pair of keys x 8 dims)
            const int c = it * NT + tid;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int kp = kt + 2 * (c >> 4) + j;
                const bool ok = kp < k_hi;
                vr[it][j] = ok ? *(const uint4*) (vc + row(ok ? kp : 0) + (size_t) g * kFaD + (c & 15) * 8) : z4;
            }
        }
    };
    if (t0 < t1) load(t0);
    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * BN;
        const bool any = !(kt > qpos0 + wi + 15 || kt + BN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + BN - 1) < extent;
        // the bias of the lane's 8 keys a column: every second distance of a row's 16
        float bv[NJ][8];
        if (!INL && any && near) {
            const bool inside = qpos0 + wi - (kt + BN - 1) >= 0 && qpos0 + wi + 15 - kt < extent && wi + 15 < Tq;
            if (inside) {
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float* bp = brow + (qa - (kt + 16 * j) - 15);   // (bp[15 - k]: the column's key k)
                    float b16[16];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const float4 x = *(const float4*) (bp + 4 * v);
                        b16[4 * v] = x.x, b16[4 * v + 1] = x.y, b16[4 * v + 2] = x.z, b16[4 * v + 3] = x.w;
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) bv[j][i] = hi ? b16[14 - 2 * i] : b16[15 - 2 * i];
                }
            } else {
#pragma unroll
                for (int j = 0; j < NJ; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int dist = qa - (kt + 16 * j + wmma_row(i, hi));
                        bv[j][i] = brow[dist < 0 ? 0 : (dist < extent ? dist : extent - 1)];
                    }
            }
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < NKL; ++it) {
            const int c = it * NT + tid;
            *(uint4*) (Ks + (c >> 4) * kRfLd + (c & 15) * 8) = kr[it];
        }
#pragma unroll
        for (int it = 0; it < NVL; ++it) {
            const int c = it * NT + tid, vp = c >> 4, d0 = (c & 15) * 8;
            const uint32_t aw[4] = {vr[it][0].x, vr[it][0].y, vr[it][0].z, vr[it][0].w};
            const uint32_t bw[4] = {vr[it][1].x, vr[it][1].y, vr[it][1].z, vr[it][1].w};
            uint32_t* Vt32 = (uint32_t*) Vt;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                Vt32[(d0 + 2 * m) * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x5410);
                Vt32[(d0 + 2 * m + 1) * (kRfVld / 2) + vp] = __byte_perm(aw[m], bw[m], 0x7632);
            }
        }
        __syncthreads();
        if (t + 1 < t1) load(t + 1);
        if (!any) continue;
        float sv[NJ][8];
        float mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            wmma_f8 st = wmma_f8{};
#pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk)
                st = wmma_f16(rm_frag(Ks + (16 * j + ln) * kRfLd + kk * 16), qf[kk], st);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int kp = kt + 16 * j + wmma_row(i, hi), dist = qa - kp;
                float v = st[i] * sc;
                if (near && dist >= 0 && dist < extent) {
                    if constexpr (INL) {
                        const float4* pp = (const float4*) (proj + (size_t) dist * 16);
                        float b = 0.0f;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float4 x = pp[e];
                            b = fmaf(x.w, rv[e].w, fmaf(x.z, rv[e].z, fmaf(x.y, rv[e].y, fmaf(x.x, rv[e].x, b))));
                        }
                        v += b * kLog2e;
                    } else
                        v += bv[j][i] * kLog2e;
                }
                v = (dist < 0 || kp < qa - window + 1 || !qok) ? -INFINITY : v;
                sv[j][i] = v;
                mt = fmaxf(mt, v);
            }
        }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 16));
        const float mn = fmaxf(mx, mt);
        const float cf = mn == -INFINITY ? 1.0f : exp2f(mx - mn), u = mn == -INFINITY ? 0.0f : mn;
        mx = mn;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[j][i] *= cf;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            uint32_t my[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p0 = exp2f(sv[j][2 * i] - u), p1 = exp2f(sv[j][2 * i + 1] - u);
                sum += p0 + p1;
                my[i] = f2_to_u32(p0, p1);
            }
            union {
                uint32_t w[8];
                wmma_h16 v;
            } pf;
#pragma unroll
            for (int i = 0; i < 4; ++i) {   // (the half-warps hold the even and the odd keys)
                const uint32_t ot = __shfl_xor_sync(0xffffffffu, my[i], 16);
                const uint32_t ev = hi ? ot : my[i], od = hi ? my[i] : ot;
                pf.w[2 * i] = (ev & 0xFFFFu) | (od << 16);
                pf.w[2 * i + 1] = (ev >> 16) | (od & 0xFFFF0000u);
            }
#pragma unroll
            for (int d = 0; d < kFaD / 16; ++d)
                o[d] = wmma_f16(rm_frag(Vt + (16 * d + ln) * kRfVld + 16 * j), pf.v, o[d]);
        }
        ls = ls * cf + sum + __shfl_xor_sync(0xffffffffu, sum, 16);
    }
    if (!qok) return;
    float* op = o_part + (((size_t) split * Tq + iq) * H + h) * kFaD;
#pragma unroll
    for (int d = 0; d < kFaD / 16; ++d)
#pragma unroll
        for (int i = 0; i < 8; ++i) op[16 * d + wmma_row(i, hi)] = o[d][i];
    if (hi == 0) ml_part[((size_t) split * Tq + iq) * H + h] = make_float2(mx, ls);
#else
    __trap();
#endif
}

#ifdef IK2_HIP
// (AMD port) attn_decode_mma_kernel's job (the same keys a warp, the same partials) on RDNA's matrix cores, FP16 cache, up to 4
// heads a K / V head: S = Q K^T with the heads as A's rows and 16 keys as B's, so that a lane holds a key's logits of
// two heads (hi = 0: heads 0, 2; hi = 1: heads 1, 3); the probabilities go to the warp as scalars (readlane) and P V is
// plain FMAs, a lane 4 dims.  No shared memory, no barriers.
__device__ __forceinline__ float rl_f(float v, int lane) {
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(uint32_t, v), lane));
}
#if defined(MAYA_RDNA_WMMA)
// two E4M3 values (bytes 0, 1 of x) as FP16 / 256 (exact: the exponent and mantissa bits land in the FP16's; NaN is never stored)
__device__ __forceinline__ uint32_t e4m3x2_h2_div256(uint32_t x) {
    x = (x & 0xFFu) | ((x & 0xFF00u) << 8);
    return ((x & 0x007F007Fu) << 7) | ((x & 0x00800080u) << 8);
}
// 16 E4M3 values as FP16 / 256 (a K row's 16 dims)
__device__ __forceinline__ maya::kernels::wmma_h16 rm_frag8(const uint8_t* p) {
    const uint4 q = *(const uint4*) p;
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
    uint32_t h[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) h[2 * i] = e4m3x2_h2_div256(w[i]), h[2 * i + 1] = e4m3x2_h2_div256(w[i] >> 16);
    maya::kernels::wmma_h16 r;
    __builtin_memcpy(&r, h, sizeof r);
    return r;
}
template <class KVT>
__device__ __forceinline__ void attn_wmma_part(const uint16_t* __restrict__ q16, const KVT* __restrict__ kc,
                                               const KVT* __restrict__ vc, int kv_cap, int H, int Hkv, const int* pos_d,
                                               int qpos, float qk_scale, const float* __restrict__ r,
                                               const float* __restrict__ proj, int d_rel, int extent, int window, int n_split,
                                               float* __restrict__ o_part, float2* __restrict__ ml_part, int bx, int by) {
    using namespace maya::kernels;
    if (pos_d) qpos = *pos_d;
    const int lane = threadIdx.x & 31, ln = lane & 15, hi = lane >> 4;
    const int g = bx, R = H / Hkv, KD = Hkv * kFaD;
    const int w = by * kWaNW + (threadIdx.x >> 5);
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);
    wmma_h16 qf[kFaD / 16];   // (A's row ln: head ln of the group)
#pragma unroll
    for (int kk = 0; kk < kFaD / 16; ++kk) qf[kk] = ln < R ? rm_frag(q16 + (size_t) (g * R + ln) * kFaD + kk * 16) : wmma_h16{};
    const int hd[2] = {hi, 2 + hi};
    const bool hok[2] = {hd[0] < R, hd[1] < R};
    float o[4][4], m[4], l[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        m[h] = -INFINITY;
#pragma unroll
        for (int d = 0; d < 4; ++d) o[h][d] = 0.0f;
    }
    const float sc = qk_scale * kLog2e * (sizeof(KVT) == 1 ? 256.0f : 1.0f);   // (FP8: K is / 256)
    const KVT* kg = kc + (size_t) g * kFaD;
    const KVT* vg = vc + (size_t) g * kFaD + lane * 4;
    const float* rr[2] = {r + (size_t) (g * R + (hok[0] ? hd[0] : 0)) * d_rel, r + (size_t) (g * R + (hok[1] ? hd[1] : 0)) * d_rel};
    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16, kmax = qpos - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        auto row = [&](int key) {   // (a key past the token's: the token's row, its logits masked below)
            int rw = row0 + min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t) rw * KD;
        };
        const int key = kt + ln, dist = qpos - key;
        const KVT* kp = kg + row(ln);
        wmma_h16 kf[kFaD / 16];
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            if constexpr (sizeof(KVT) == 1) kf[kk] = rm_frag8(kp + kk * 16);
            else kf[kk] = rm_frag(kp + kk * 16);
        }
        uint2 vv[16];   // (the V rows' 4 dims asked for with the keys)
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            if constexpr (sizeof(KVT) == 1) {
                const uint32_t q = *(const uint32_t*) (vg + row(k));
                vv[k] = make_uint2(e4m3x2_h2_div256(q), e4m3x2_h2_div256(q >> 16));
            } else {
                vv[k] = *(const uint2*) (vg + row(k));
            }
        }
        const bool near = dist >= 0 && dist < extent;
        float b[2] = {0.0f, 0.0f};
        if (qpos - (kt + 15) < extent) {
            const float* pj = proj + (size_t) (near ? dist : 0) * d_rel;
            for (int e = 0; e < d_rel; e += 4) {
                const float4 x = *(const float4*) (pj + e);
                const float4 y0 = *(const float4*) (rr[0] + e), y1 = *(const float4*) (rr[1] + e);
                b[0] = fmaf(x.w, y0.w, fmaf(x.z, y0.z, fmaf(x.y, y0.y, fmaf(x.x, y0.x, b[0]))));
                b[1] = fmaf(x.w, y1.w, fmaf(x.z, y1.z, fmaf(x.y, y1.y, fmaf(x.x, y1.x, b[1]))));
            }
        }
        wmma_f8 sa = wmma_f8{};
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) sa = wmma_f16(qf[kk], kf[kk], sa);
        float c[2], mx[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            c[i] = sa[i] * sc;   // (A's row 2 i + hi: the head hd[i])
            if (near) c[i] += b[i] * kLog2e;
            c[i] = (dist < 0 || !hok[i]) ? -INFINITY : c[i];
            mx[i] = c[i];
#pragma unroll
            for (int sh = 1; sh < 16; sh <<= 1) mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], sh));
        }
        float u[4], cf[4];
        bool any = false;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float mh = rl_f(mx[h >> 1], 16 * (h & 1)), mn = fmaxf(m[h], mh);
            cf[h] = mn > m[h] ? exp2f(m[h] - mn) : 1.0f;
            any = any || mn > m[h];
            m[h] = mn;
            u[h] = mn == -INFINITY ? 0.0f : mn;
        }
        if (any) {
#pragma unroll
            for (int h = 0; h < 4; ++h)
#pragma unroll
                for (int d = 0; d < 4; ++d) o[h][d] *= cf[h];
            l[0] *= cf[hi], l[1] *= cf[2 + hi];
        }
        float p[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            p[i] = exp2f(c[i] - u[hd[i]]);
            p[i] = c[i] == -INFINITY ? 0.0f : p[i];
            l[i] += p[i];
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const float2 a = u32_to_f2(vv[k].x), z = u32_to_f2(vv[k].y);
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                if (h >= R) continue;
                const float ph = rl_f(p[h >> 1], k + 16 * (h & 1));
                o[h][0] = fmaf(ph, a.x, o[h][0]), o[h][1] = fmaf(ph, a.y, o[h][1]);
                o[h][2] = fmaf(ph, z.x, o[h][2]), o[h][3] = fmaf(ph, z.y, o[h][3]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int sh = 1; sh < 16; sh <<= 1) l[i] += __shfl_xor_sync(0xffffffffu, l[i], sh);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        if (h >= R) continue;
        *(float4*) (o_part + ((size_t) w * H + g * R + h) * kFaD + lane * 4) = make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
    }
    if (ln == 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
            if (hok[i]) ml_part[(size_t) w * H + g * R + hd[i]] = make_float2(m[hd[i]], l[i]);
    }
}
#endif
template <class KVT>
__global__ void __launch_bounds__(kWaNW * 32)
attn_decode_wmma_kernel(const uint16_t* __restrict__ q16, const KVT* __restrict__ kc, const KVT* __restrict__ vc,
                        int kv_cap, int H, int Hkv, const int* pos_d, int qpos, float qk_scale,
                        const float* __restrict__ r, const float* __restrict__ proj, int d_rel, int extent, int window,
                        int n_split, float* __restrict__ o_part, float2* __restrict__ ml_part) {
#if defined(MAYA_RDNA_WMMA)
    attn_wmma_part(q16, kc, vc, kv_cap, H, Hkv, pos_d, qpos, qk_scale, r, proj, d_rel, extent, window, n_split, o_part, ml_part,
                   blockIdx.x, blockIdx.y);
#else
    __trap();
#endif
}
// attn_decode_wmma_kernel's blocks (first in dispatch order), then attn_decode_mma_combine_kernel's, a block a head, which wait
// for every partial; cnt: [0] partials done, [1] heads done (the last head sets both back to 0)
__global__ void __launch_bounds__(kWaNW * 32)
attn_pipe_kernel(const uint16_t* __restrict__ q16, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                 int kv_cap, int H, int Hkv, const int* pos_d, int qpos, float qk_scale, const float* __restrict__ r,
                 const float* __restrict__ proj, int d_rel, int extent, int window, int n_split, float* o_part,
                 float2* ml_part, const float* __restrict__ sink, const float* __restrict__ gate, float* out, uint16_t* out16,
                 int* cnt) {
#if defined(MAYA_RDNA_WMMA)
    extern __shared__ float cm_sm[];
    const int nP = Hkv * n_split, bid = blockIdx.x;
    if (bid < nP) {
        attn_wmma_part(q16, kc, vc, kv_cap, H, Hkv, pos_d, qpos, qk_scale, r, proj, d_rel, extent, window, n_split, o_part,
                       ml_part, bid % Hkv, bid / Hkv);
        pipe_signal(cnt);
        return;
    }
    const bool late = pipe_wait(cnt, nP, (int*) cm_sm);
    float* red = cm_sm;
    float* cw = cm_sm + kFaD;
    if (pos_d) qpos = *pos_d;
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    const int h = bid - nP, d = threadIdx.x;
    const float sk = sink[h] * kLog2e;
    float m = d == 0 ? sk : -INFINITY;
    for (int s = d; s < n_act; s += kFaD) m = fmaxf(m, ml_part[(size_t) s * H + h].x);
    m = bmax(m, red);
    float den = d == 0 ? exp2f(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const float2 ml = ml_part[(size_t) s * H + h];
        const float c = ml.x == -INFINITY ? 0.0f : exp2f(ml.x - m);
        cw[s] = c;
        den += ml.y * c;
    }
    den = bsum(den, red);
    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) acc += o_part[((size_t) s * H + h) * kFaD + d] * cw[s];
    float v = acc / den;
    if (late) v = NAN;
    if (gate) v = v * (1.0f / (1.0f + expf(-gate[(size_t) h * kFaD + d])));
    out[(size_t) h * kFaD + d] = v;
    if (out16) out16[(size_t) h * kFaD + d] = f2h(v);
    if (pipe_signal(cnt + 1) == H - 1) cnt[0] = 0, cnt[1] = 0;
#else
    __trap();
#endif
}
#endif

// an RDNA 3 / 3.5 / 4 GPU (the current one), whose matrix-core kernels this build has
bool rdna_wmma_gpu() {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    const std::string a = p.gcnArchName;
    return a.rfind("gfx11", 0) == 0 || a.rfind("gfx12", 0) == 0;
}

// an RDNA 3.5 APU (its shared memory's reads are what its kernels wait for)
bool rdna_igpu() {
    int dev = 0;
    cudaDeviceProp p{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
    return std::string(p.gcnArchName).rfind("gfx115", 0) == 0;
}

static const float *g_flash_rq = nullptr, *g_flash_proj = nullptr;
template <class KVT>
void attn_flash_rdna(const uint16_t* q16, int q_ld, const KVT* kc, const KVT* vc, int kv_cap, int H, int KV, int Tq,
                     int qpos0, float qk_scale, const float* bias, int extent, int window, int n_split, float* o_part,
                     float2* ml_part, cudaStream_t s) {
    static const int f2 = std::getenv("IK2_FLASH2") ? std::atoi(std::getenv("IK2_FLASH2")) : 0;   // (0: the earlier kernel; 4 or 8: the lane-per-query kernel with that many warps a block; off by default)
    static const int f2nw = f2 == 8 ? 8 : 4;
    if constexpr (sizeof(KVT) == 2) {
        if (f2 && (((uintptr_t) kc | (uintptr_t) vc | (uintptr_t) q16) & 15) == 0) {
            const bool inl = g_flash_rq != nullptr;
            const dim3 g2((unsigned) ((Tq + 16 * f2nw - 1) / (16 * f2nw)), (unsigned) H, (unsigned) n_split);
#define IK2_FL2(NW, INL) attn_flash2_rdna_kernel<NW, INL><<<g2, NW * 32, 0, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0,             qk_scale, bias, extent, window, n_split, o_part, ml_part, g_flash_rq, g_flash_proj)
            if (f2nw == 8) { if (inl) IK2_FL2(8, true); else IK2_FL2(8, false); }
            else { if (inl) IK2_FL2(4, true); else IK2_FL2(4, false); }
#undef IK2_FL2
            check("attn_flash2_rdna");
            return;
        }
    }
    // (queries a block: 64, or 128 that share one read of the keys and values - IK2_FLASH_NW=4 / 8)
    static const int nw = std::getenv("IK2_FLASH_NW") ? std::atoi(std::getenv("IK2_FLASH_NW")) : (rdna_igpu() ? 8 : 4);
    const dim3 grid((unsigned) ((Tq + 16 * nw - 1) / (16 * nw)), (unsigned) H, (unsigned) n_split);
    if (g_flash_rq && nw == 8)
        attn_flash_rdna_kernel<KVT, 8, true><<<grid, 256, 0, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias,
                                                                  extent, window, n_split, o_part, ml_part, g_flash_rq, g_flash_proj);
    else if (g_flash_rq)
        attn_flash_rdna_kernel<KVT, 4, true><<<dim3((unsigned) ((Tq + 63) / 64), (unsigned) H, (unsigned) n_split), 128, 0, s>>>(
            q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part, g_flash_rq,
            g_flash_proj);
    else if (nw == 8)
        attn_flash_rdna_kernel<KVT, 8><<<grid, 256, 0, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias,
                                                            extent, window, n_split, o_part, ml_part);
    else
        attn_flash_rdna_kernel<KVT, 4><<<dim3((unsigned) ((Tq + 63) / 64), (unsigned) H, (unsigned) n_split), 128, 0, s>>>(
            q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part);
    check("attn_flash_rdna");
}

// y [T][ldy] = x [T][K] . W [N][K]^T in BF16 for a few tokens (TT >= T): a warp R rows, a lane 8 values a step (16
// bytes of each row and of each token's x), every token's sum in its own register, then the warp's sum.
template <int WARPS, int R, int TT>
__global__ void __launch_bounds__(WARPS * 32)
gemm_bf16_cols_kernel(const uint16_t* __restrict__ W, int N, int K, const uint16_t* __restrict__ x, int T,
                      float* __restrict__ y, int ldy) {
    const int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (row0 >= N) return;
    const uint4* wr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) wr[i] = (const uint4*) (W + (size_t) min(row0 + i, N - 1) * K);
    float acc[R][TT];
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[i][t] = 0.0f;
    auto lo = [](uint32_t v) { return __uint_as_float(v << 16); };
    auto hi = [](uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); };
    for (int c = lane; c < (K >> 3); c += 32) {
        float wf[R][8];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint4 w = __ldg(wr[i] + c);
            wf[i][0] = lo(w.x), wf[i][1] = hi(w.x), wf[i][2] = lo(w.y), wf[i][3] = hi(w.y);
            wf[i][4] = lo(w.z), wf[i][5] = hi(w.z), wf[i][6] = lo(w.w), wf[i][7] = hi(w.w);
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= T) break;
            const uint4 v = __ldg((const uint4*) (x + (size_t) t * K) + c);
            const float xf[8] = {lo(v.x), hi(v.x), lo(v.y), hi(v.y), lo(v.z), hi(v.z), lo(v.w), hi(v.w)};
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][t] = fmaf(wf[i][j], xf[j], acc[i][t]);
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= T) break;
            const float a = wsum(acc[i][t]);
            if (lane == 0 && row0 + i < N) y[(size_t) t * ldy + row0 + i] = a;
        }
}
}  // namespace

bool gemm_bf16_cols(const uint16_t* W, int N, int K, const uint16_t* x, int T, float* y, int ldy, cudaStream_t s) {
    if (N <= 0 || T <= 0 || T > 8 || K % 8 || ((uintptr_t) W & 15) || ((uintptr_t) x & 15)) return false;
    constexpr int WARPS = 4, R = 4;
    const unsigned g = (unsigned) ((N + WARPS * R - 1) / (WARPS * R));
    if (T <= 4) gemm_bf16_cols_kernel<WARPS, R, 4><<<g, WARPS * 32, 0, s>>>(W, N, K, x, T, y, ldy);
    else gemm_bf16_cols_kernel<WARPS, R, 8><<<g, WARPS * 32, 0, s>>>(W, N, K, x, T, y, ldy);
    check("gemm_bf16_cols");
    return true;
}


int experts_rdna_rows() {   // (a tile's rows of one expert: 128, or 32 with IK2_RDNA_M128=0)
    static const int r = std::getenv("IK2_RDNA_M128") && !std::atoi(std::getenv("IK2_RDNA_M128")) ? 32 : 128;
    return r;
}
// (AMD port) Y [T][ldy] = X [T][K] . W^T in F32, W(n, k) = W[n ldn + k ldk] with ldk 1 or ldn 1: a block
// 32 rows of X by 64 columns, K in steps of 64 through shared memory, a thread 2 rows x 4 columns
template <bool KFAST>   // ldk == 1 (else ldn == 1)
__global__ void __launch_bounds__(256) gemm_f32_kernel(const float* __restrict__ X, int ldx, int T,
                                                       const float* __restrict__ W, int N, int K, int ldo,
                                                       float* __restrict__ Y, int ldy) {
    __shared__ __align__(16) float Xs[32][64];
    __shared__ __align__(16) float Ws[64][64];
    const int tid = threadIdx.x, t0 = blockIdx.y * 32, n0 = blockIdx.x * 64, nt = min(32, T - t0);
    const int xr = tid >> 3, xc = (tid & 7) * 8;          // 8 values of an X row
    const int tt = (tid >> 4) * 2, nn = (tid & 15) * 4;   // this thread's outputs: rows tt, tt + 1, columns nn .. nn + 3
    float acc[2][4] = {};
    for (int k0 = 0; k0 < K; k0 += 64) {
        if (k0) __syncthreads();
        if (xr < nt) {
            const float4* p = (const float4*) (X + (size_t) (t0 + xr) * ldx + k0 + xc);
            *(float4*) &Xs[xr][xc] = __ldg(p);
            *(float4*) &Xs[xr][xc + 4] = __ldg(p + 1);
        }
        const int a = tid >> 2, b = (tid & 3) * 16;       // 16 values of a W line: a its slow index, b .. its fast one
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (KFAST) {
                *(float4*) &Ws[a][b + 4 * i] = __ldg((const float4*) (W + (size_t) (n0 + a) * ldo + k0 + b + 4 * i));
            } else {
                const float4 v = __ldg((const float4*) (W + (size_t) (k0 + a) * ldo + n0 + b + 4 * i));
                Ws[b + 4 * i][a] = v.x, Ws[b + 4 * i + 1][a] = v.y, Ws[b + 4 * i + 2][a] = v.z, Ws[b + 4 * i + 3][a] = v.w;
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < 64; k += 4) {
            float4 x[2];
            x[0] = *(const float4*) &Xs[tt][k];
            x[1] = *(const float4*) &Xs[tt + 1][k];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 w = *(const float4*) &Ws[nn + j][k];
#pragma unroll
                for (int i = 0; i < 2; ++i)
                    acc[i][j] = fmaf(x[i].w, w.w, fmaf(x[i].z, w.z, fmaf(x[i].y, w.y, fmaf(x[i].x, w.x, acc[i][j]))));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
        if (tt + i < nt)
            *(float4*) (Y + (size_t) (t0 + tt + i) * ldy + n0 + nn) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
}
bool gemm_f32_rdna(const float* X, int ldx, int T, const float* W, int N, int K, int ldn, int ldk, float* Y, int ldy,
                   cudaStream_t s) {
    static const bool on = (std::getenv("IK2_GATE_GEMM") && std::atoi(std::getenv("IK2_GATE_GEMM"))) && rdna_wmma_gpu();
    if (!on || T < 16 || N % 64 || K % 64 || (ldn != 1 && ldk != 1)) return false;
    if (((uintptr_t) X | (uintptr_t) W | (uintptr_t) Y) & 15 || ldx % 4 || ldy % 4 || (ldk == 1 ? ldn : ldk) % 4) return false;
    const dim3 g((unsigned) (N / 64), (unsigned) ((T + 31) / 32));
    if (ldk == 1) gemm_f32_kernel<true><<<g, 256, 0, s>>>(X, ldx, T, W, N, K, ldn, Y, ldy);
    else gemm_f32_kernel<false><<<g, 256, 0, s>>>(X, ldx, T, W, N, K, ldk, Y, ldy);
    check("gemm_f32_rdna");
    return true;
}
// (AMD port) the relative bias of a prompt chunk's queries: relb [rows][extent] = r [rows][K] . proj
// [extent][K]^T in F32, kRbR rows a block: a lane a distance, r's values uniform over the warp (scalar registers)
constexpr int kRbR = 16;
__global__ void __launch_bounds__(256) relb_kernel(const float* __restrict__ r, const float* __restrict__ proj, int rows,
                                                   int extent, int K, float* __restrict__ out) {
    const int e = blockIdx.x * 256 + threadIdx.x, r0 = blockIdx.y * kRbR;
    const int nr = rows - r0 < kRbR ? rows - r0 : kRbR;
    if (e >= extent) return;
    float a[kRbR] = {};
    const float* pr = proj + (size_t) e * K;
    for (int k = 0; k < K; k += 4) {
        const float4 w = __ldg((const float4*) (pr + k));
        const float wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int t = 0; t < kRbR; ++t) {
            const float* rr = r + (size_t) (r0 + (t < nr ? t : 0)) * K + k;
#pragma unroll
            for (int i = 0; i < 4; ++i) a[t] = fmaf(wv[i], rr[i], a[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < kRbR; ++t)
        if (t < nr) out[(size_t) (r0 + t) * extent + e] = a[t];
}
bool relb_rdna(const float* r, const float* proj, int rows, int extent, int K, float* out, cudaStream_t s) {
    static const bool on = !(std::getenv("IK2_RELB_RDNA") && !std::atoi(std::getenv("IK2_RELB_RDNA")));
    if (!on || rows <= 0 || K % 4 || ((uintptr_t) proj & 15)) return false;
    relb_kernel<<<dim3((unsigned) ((extent + 255) / 256), (unsigned) ((rows + kRbR - 1) / kRbR)), 256, 0, s>>>(r, proj, rows,
                                                                                                          extent, K, out);
    check("relb_rdna");
    return true;
}
// (AMD port) a gate's up products and blend in one launch, nothing materialized (the "zero-mat" idea of
// CIRU AI's hip_glue_prefill.hip, Apache-2.0): f = g F, gn = g G, y = (1 - tanh f) x + tanh(gn) m over NTK tokens a block;
// a lane a channel, g's values uniform over the warp (scalar registers), F's and G's rows read side by side
constexpr int kGuT = 16;
__global__ void __launch_bounds__(256) loop_gate_up_kernel(const float* __restrict__ x, const float* __restrict__ m,
                                                           const float* __restrict__ g, const float* __restrict__ F,
                                                           const float* __restrict__ G, int T, int C, int R, float* y) {
    const int c = blockIdx.x * 256 + threadIdx.x, t0 = blockIdx.y * kGuT;
    const int nt = T - t0 < kGuT ? T - t0 : kGuT;
    if (c >= C) return;
    float f[kGuT] = {}, gn[kGuT] = {};
    for (int r = 0; r < R; ++r) {
        const float fw = F[(size_t) r * C + c], gw = G[(size_t) r * C + c];
#pragma unroll
        for (int t = 0; t < kGuT; ++t) {
            const float gv = g[(size_t) (t0 + (t < nt ? t : 0)) * R + r];
            f[t] = fmaf(fw, gv, f[t]), gn[t] = fmaf(gw, gv, gn[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < kGuT; ++t)
        if (t < nt) {
            const size_t i = (size_t) (t0 + t) * C + c;
            y[i] = (1.0f - tanhf(f[t])) * x[i] + tanhf(gn[t]) * m[i];
        }
}
bool loop_gate_up_rdna(const float* x, const float* m, const float* g, const float* F, const float* G, int T, int C, int R,
                       float* y, cudaStream_t s) {
    static const bool on = !(std::getenv("IK2_GATE_UP") && !std::atoi(std::getenv("IK2_GATE_UP")));
    if (!on) return false;
    loop_gate_up_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) ((T + kGuT - 1) / kGuT)), 256, 0, s>>>(x, m, g, F, G,
                                                                                                              T, C, R, y);
    check("loop_gate_up_rdna");
    return true;
}
static bool rdna_n256(bool gu) {   // (IK2_RDNA_N256: bit 0 gate / up, bit 1 down)
    static const int v = std::getenv("IK2_RDNA_N256") ? std::atoi(std::getenv("IK2_RDNA_N256")) : 3;
    return (v & (gu ? 1 : 2)) != 0;
}
// (AMD port) the next flash launches' relative inputs: r [Tq][H][16] and the projection [extent][16]; false: the table is needed
bool flash_inline_bias(const float* rq, const float* proj, int d_rel, bool fp8) {
    static const bool on = !(std::getenv("IK2_FLASH_INL") && !std::atoi(std::getenv("IK2_FLASH_INL"))) &&
                           (std::getenv("IK2_FLASH2") ? std::atoi(std::getenv("IK2_FLASH2")) != 0 : false) && rdna_wmma_gpu();
    // (the default kernel computes the bias itself too, for either cache: on an RDNA 3.5 APU, whose kernels wait for memory -
    //  8060S, 4K: 1,426 -> 1,621 tok/s; a 7900 XT: 3,615 -> 3,387, so off there. IK2_FLASH_INL1=0 / 1: off / on)
    static const bool on1 = (std::getenv("IK2_FLASH_INL1") ? std::atoi(std::getenv("IK2_FLASH_INL1")) != 0 : rdna_igpu()) && rdna_wmma_gpu();
    g_flash_rq = d_rel == 16 && ((on && !fp8) || on1) ? rq : nullptr, g_flash_proj = proj;
    return g_flash_rq != nullptr;
}
bool experts_rdna_in_place() {   // (the gate / up tiles can read a gidx)
    return experts_rdna_rows() == 128 && rdna_n256(true);
}
bool experts_rdna_available(int ty_gate, int ty_up, int ty_down, int C, int F) {
    static const int on = [] {   // (the current GPU: an RDNA 3 / 3.5 / 4 one of this build; IK2_RDNA_MOE=0: off)
        if (std::getenv("IK2_RDNA_MOE") && !std::atoi(std::getenv("IK2_RDNA_MOE"))) return 0;
        return rdna_wmma_gpu() ? 1 : 0;
    }();
    const bool fmt = (ty_gate == kNVFP4 || ty_gate == kQ8_0) && ty_up == ty_gate && ty_down == ty_gate;
    static const bool mix_on = !(std::getenv("IK2_RDNA_MIX") && !std::atoi(std::getenv("IK2_RDNA_MIX")));
    // (the other formats: the 256-column tiles only)
    const bool mix = mix_on && ty_up == ty_gate && (ty_gate == kQ5_K || ty_gate == kQ4_K || ty_gate == kIQ4_XS) &&
                     (ty_down == kQ5_0 || ty_down == kIQ4_NL) && experts_rdna_rows() == 128 && rdna_n256(true) &&
                     rdna_n256(false) && C % 256 == 0 && F % 128 == 0;
    return on && (fmt || mix) && C % kRmBN == 0 && F % kRmBN == 0 && C % kRmBK == 0 && F % kRmBK == 0;
}
void experts_rdna_gu(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_up,
                     const uint16_t* x16, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                     uint16_t* h16, cudaStream_t s, const int* gidx) {
    if (n_tiles <= 0) return;
    const dim3 g((unsigned) (F / 64), (unsigned) n_tiles);
    const size_t rb = mv_row_bytes(ty, C);
    if (experts_rdna_rows() == 128 && rdna_n256(true) && C % 256 == 0 && F % 128 == 0) {
        const dim3 g2((unsigned) (F / 128), (unsigned) n_tiles);
        if (ty == kNVFP4)
            experts_rdna256_kernel<kNVFP4, true><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg, su, gcap, ucap, h16, nullptr, gidx);
        else if (ty == kQ5_K)
            experts_rdna256_kernel<kQ5_K, true><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg, su, gcap, ucap, h16, nullptr, gidx);
        else if (ty == kQ4_K)
            experts_rdna256_kernel<kQ4_K, true><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg, su, gcap, ucap, h16, nullptr, gidx);
        else if (ty == kIQ4_XS)
            experts_rdna256_kernel<kIQ4_XS, true><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg, su, gcap, ucap, h16, nullptr, gidx);
        else
            experts_rdna256_kernel<kQ8_0, true><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg, su, gcap, ucap, h16, nullptr, gidx);
        check("experts_rdna256");
        return;
    }
    if (experts_rdna_rows() == 128) {
        if (ty == kNVFP4)
            experts_rdna128_kernel<kNVFP4, true><<<g, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg,
                                                                   su, gcap, ucap, h16, nullptr);
        else
            experts_rdna128_kernel<kQ8_0, true><<<g, 256, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg,
                                                                  su, gcap, ucap, h16, nullptr);
        check("experts_rdna_gu");
        return;
    }
    if (ty == kNVFP4)
        experts_rdna_kernel<kNVFP4, true><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg,
                                                            su, gcap, ucap, h16, nullptr);
    else
        experts_rdna_kernel<kQ8_0, true><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_up, rb, x16, C, F, C, sg,
                                                           su, gcap, ucap, h16, nullptr);
    check("experts_rdna_gu");
}
void experts_rdna_down(const uint8_t* const* eptr, const int* tiles, int n_tiles, int ty, size_t off_down,
                       const uint16_t* h16, int C, int F, float* y32, cudaStream_t s) {
    const int* gidx = nullptr;
    if (n_tiles <= 0) return;
    const dim3 g((unsigned) (C / 128), (unsigned) n_tiles);
    const size_t rb = mv_row_bytes(ty, F);
    if (experts_rdna_rows() == 128 && rdna_n256(false) && C % 256 == 0 && F % 128 == 0) {
        const dim3 g2((unsigned) (C / 256), (unsigned) n_tiles);
        if (ty == kNVFP4)
            experts_rdna256_kernel<kNVFP4, false><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C, nullptr, nullptr, 0.0f, 0.0f, nullptr, y32, gidx);
        else if (ty == kQ5_0)
            experts_rdna256_kernel<kQ5_0, false><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C, nullptr, nullptr, 0.0f, 0.0f, nullptr, y32, gidx);
        else if (ty == kIQ4_NL)
            experts_rdna256_kernel<kIQ4_NL, false><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C, nullptr, nullptr, 0.0f, 0.0f, nullptr, y32, gidx);
        else
            experts_rdna256_kernel<kQ8_0, false><<<g2, 256, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C, nullptr, nullptr, 0.0f, 0.0f, nullptr, y32, gidx);
        check("experts_rdna256");
        return;
    }
    if (experts_rdna_rows() == 128) {
        if (ty == kNVFP4)
            experts_rdna128_kernel<kNVFP4, false><<<g, 256, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C,
                                                                    nullptr, nullptr, 0.0f, 0.0f, nullptr, y32);
        else
            experts_rdna128_kernel<kQ8_0, false><<<g, 256, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C,
                                                                   nullptr, nullptr, 0.0f, 0.0f, nullptr, y32);
        check("experts_rdna_down");
        return;
    }
    if (ty == kNVFP4)
        experts_rdna_kernel<kNVFP4, false><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C,
                                                             nullptr, nullptr, 0.0f, 0.0f, nullptr, y32);
    else
        experts_rdna_kernel<kQ8_0, false><<<g, 128, 0, s>>>(eptr, (const int4*) tiles, off_down, rb, h16, F, F, C,
                                                            nullptr, nullptr, 0.0f, 0.0f, nullptr, y32);
    check("experts_rdna_down");
}
void moe_rows_combine(const float* y32, const int* inv, const float* wtr, int T, int K, int C, float* out,
                      cudaStream_t s) {
    if (T <= 0) return;
    moe_rows_combine_kernel<<<T, 256, 0, s>>>(y32, inv, wtr, K, C, out);
    check("moe_rows_combine");
}
// (AMD port) y [T][ldy] = x16 [T][K] . W^T, W planar Q8_0; false (nothing launched): not this shape or GPU
// (IK2_RDNA_DENSE=0: never)
bool dense_q8p_rdna(const int8_t* q, const uint16_t* qd, int N, int K, const uint16_t* x16, int T, float* y, int ldy,
                    cudaStream_t s) {
    static const bool on = !(std::getenv("IK2_RDNA_DENSE") && !std::atoi(std::getenv("IK2_RDNA_DENSE"))) && rdna_wmma_gpu();
    if (!on || T < 16 || N <= 0 || K % kRmBK || ((uintptr_t) q & 15) || ((uintptr_t) x16 & 15) || (K * 2) % 16) return false;
    static const int m128 = std::getenv("IK2_DENSE_M128") ? std::atoi(std::getenv("IK2_DENSE_M128")) : 1;
    if (m128 && T > 32) {
        dense_rdna128_kernel<false><<<dim3((unsigned) ((N + 127) / 128), (unsigned) ((T + 127) / 128)), 256, 0, s>>>(
            q, qd, N, K, x16, T, y, ldy);
        check("dense_rdna128");
        return true;
    }
    dense_q8p_rdna_kernel<false><<<dim3((unsigned) ((N + 127) / 128), (unsigned) ((T + kRmBM - 1) / kRmBM)), 128, 0, s>>>(
        q, qd, N, K, x16, T, y, ldy);
    check("dense_q8p_rdna");
    return true;
}
// (AMD port) y [T][ldy] = x16 [T][K] . W^T, W FP16 [N][K] (gemm16's product); false: not taken
// (IK2_RDNA_GEMM=0: never)
bool dense_f16_rdna(const uint16_t* W, int N, int K, const uint16_t* x16, int T, float* y, int ldy, cudaStream_t s) {
    static const bool on = !(std::getenv("IK2_RDNA_GEMM") && !std::atoi(std::getenv("IK2_RDNA_GEMM"))) && rdna_wmma_gpu();
    if (!on || T < 16 || N <= 0 || K % kRmBK || ((uintptr_t) W & 15) || ((uintptr_t) x16 & 15)) return false;
    static const int m128 = std::getenv("IK2_DENSE_M128") ? std::atoi(std::getenv("IK2_DENSE_M128")) : 1;
    if (m128 && T > 32) {
        dense_rdna128_kernel<true><<<dim3((unsigned) ((N + 127) / 128), (unsigned) ((T + 127) / 128)), 256, 0, s>>>(
            (const int8_t*) W, nullptr, N, K, x16, T, y, ldy);
        check("dense_rdna128");
        return true;
    }
    dense_q8p_rdna_kernel<true><<<dim3((unsigned) ((N + 127) / 128), (unsigned) ((T + kRmBM - 1) / kRmBM)), 128, 0, s>>>(
        (const int8_t*) W, nullptr, N, K, x16, T, y, ldy);
    check("dense_f16_rdna");
    return true;
}
