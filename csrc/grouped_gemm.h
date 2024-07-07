#include <torch/extension.h>

namespace grouped_gemm {

void GroupedGemmCutlass(torch::Tensor a,
         torch::Tensor b,
         torch::Tensor c,
         torch::Tensor batch_sizes,
         bool trans_a, bool trans_b);

void GroupedGemmCublas(torch::Tensor a,
         torch::Tensor b,
         torch::Tensor c,
         torch::Tensor batch_sizes,
         torch::Tensor a_ptrs,
         torch::Tensor b_ptrs,
         torch::Tensor c_ptrs,
         bool trans_a, bool trans_b);

void CublasGroupedGemmGetPtrs(torch::Tensor a,
           torch::Tensor b,
           torch::Tensor c,
           torch::Tensor batch_sizes,
           bool trans_b,
           torch::Tensor a_ptrs,
           torch::Tensor b_ptrs,
		   torch::Tensor c_ptrs);

}  // namespace grouped_gemm
