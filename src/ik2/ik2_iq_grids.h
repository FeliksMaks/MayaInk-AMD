// The i-quant grids of ggml's ggml-common.h (third_party/ggml, MIT License, Copyright (c) 2023-2024 The ggml
// authors) for the GSQ-RCO files' expert types (IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S): the header read twice, once
// through its GPU branch (ik2_grid::dev, read with __ldg) and once through its C++ branch (ik2_grid::host).
#pragma once
#include <cstdint>

namespace ik2_grid {
namespace dev {
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_CUDA
#undef GGML_COMMON_IMPL
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END
#undef GGML_COMMON_DECL
}  // namespace dev
namespace host {
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_CPP
#undef GGML_COMMON_IMPL
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END
}  // namespace host
}  // namespace ik2_grid
