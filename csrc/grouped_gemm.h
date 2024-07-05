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
		 bool trans_a, bool trans_b);

}  // namespace grouped_gemm
