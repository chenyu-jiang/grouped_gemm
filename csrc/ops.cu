#include "grouped_gemm.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm_cutlass", &GroupedGemmCutlass, "Grouped GEMM using CUTLASS library.");
  m.def("gmm_cublas", &GroupedGemmCublas, "Grouped GEMM using CUBLAS library.");
  m.def("gmm_cublas_get_ptrs", &CublasGroupedGemmGetPtrs, "Helper function for GroupedGemmCublas");
}

}  // namespace grouped_gemm
