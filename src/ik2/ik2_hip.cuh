// src/ik2/ik2_hip.cuh - maya-ik2 on AMD (HIP, wave32 RDNA3 / RDNA3.5): software forms of the PTX the CUDA kernels use.
// mma.sync m16n8k16 (F16 in, F32 accumulate), ldmatrix x4 (plain and transposed) and cp.async are emulated with
// wave32 lane shuffles and plain loads, with the same per-lane fragment layouts, so every CUDA kernel runs unchanged
// (correct, not fast). The decode hot paths have their own RDNA kernels (ik2_rdna.inc); these keep the rest working.
#pragma once
#ifdef IK2_HIP
#include <hip/hip_fp16.h>
#include <cstdint>

#ifndef __trap
#define __trap() __builtin_trap()
#endif

namespace ik2::hip {

__device__ __forceinline__ int lane() { return (int) __lane_id(); }
__device__ __forceinline__ float2 h2_to_f2(uint32_t u) {
    __half2 h;
    __builtin_memcpy(&h, &u, 4);
    return __half22float2(h);
}
__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
    __half2 x, y;
    __builtin_memcpy(&x, &a, 4);
    __builtin_memcpy(&y, &b, 4);
    const __half2 r = __hmul2(x, y);
    uint32_t o;
    __builtin_memcpy(&o, &r, 4);
    return o;
}
__device__ __forceinline__ uint32_t hsub2_u32(uint32_t a, uint32_t b) {
    __half2 x, y;
    __builtin_memcpy(&x, &a, 4);
    __builtin_memcpy(&y, &b, 4);
    const __half2 r = __hsub2(x, y);
    uint32_t o;
    __builtin_memcpy(&o, &r, 4);
    return o;
}
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float acc) {
    const float2 x = h2_to_f2(a), y = h2_to_f2(b);
    return fmaf(x.y, y.y, fmaf(x.x, y.x, acc));
}

// D[16x8] += A[16x16] (row) x B[16x8] (col), mma.sync's fragments: a0 (row g, k 2t..2t+1), a1 (row g+8), a2 (k+8),
// a3 (row g+8, k+8); b0 (k 2t..2t+1, col g), b1 (k+8); c0,c1 (row g, cols 2t, 2t+1), c2,c3 (row g+8), g = lane/4,
// t = lane%4.  All 32 lanes must call it together (as mma.sync).
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    const int L = lane(), g = L >> 2, t = L & 3;
    const int n0 = 2 * t, n1 = 2 * t + 1;
    float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int sa = g * 4 + j;
        const uint32_t A0 = __shfl(a[0], sa), A1 = __shfl(a[1], sa), A2 = __shfl(a[2], sa), A3 = __shfl(a[3], sa);
        const uint32_t B00 = __shfl(b0, n0 * 4 + j), B10 = __shfl(b1, n0 * 4 + j);
        const uint32_t B01 = __shfl(b0, n1 * 4 + j), B11 = __shfl(b1, n1 * 4 + j);
        c0 = dot2(A2, B10, dot2(A0, B00, c0));
        c1 = dot2(A2, B11, dot2(A0, B01, c1));
        c2 = dot2(A3, B10, dot2(A1, B00, c2));
        c3 = dot2(A3, B11, dot2(A1, B01, c3));
    }
    c[0] = c0, c[1] = c1, c[2] = c2, c[3] = c3;
}
// ldmatrix .x4: lanes 8i..8i+7 give the row addresses of matrix i; lane L gets, of each matrix, row L/4's halves
// 2(L%4), 2(L%4)+1 (.trans: column L/4's halves at rows 2(L%4), 2(L%4)+1).
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const int L = lane();
    const unsigned long long a = (unsigned long long) (uintptr_t) p;
    uint32_t r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned long long row = __shfl(a, 8 * i + (L >> 2));
        r[i] = *(const uint32_t*) ((const char*) (uintptr_t) row + (L & 3) * 4);
    }
    r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3];
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const int L = lane();
    const unsigned long long a = (unsigned long long) (uintptr_t) p;
    uint32_t r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned long long ra = __shfl(a, 8 * i + 2 * (L & 3));
        const unsigned long long rb = __shfl(a, 8 * i + 2 * (L & 3) + 1);
        const uint32_t lo = *(const uint16_t*) ((const char*) (uintptr_t) ra + (L >> 2) * 2);
        const uint32_t hi = *(const uint16_t*) ((const char*) (uintptr_t) rb + (L >> 2) * 2);
        r[i] = lo | (hi << 16);
    }
    r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3];
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    *(uint4*) dst = valid ? *(const uint4*) src : make_uint4(0, 0, 0, 0);
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
    *(uint32_t*) dst = valid ? *(const uint32_t*) src : 0u;
}

}  // namespace ik2::hip

#define __cvta_generic_to_shared(p) ((unsigned) (uintptr_t) (p))
#endif  // IK2_HIP
