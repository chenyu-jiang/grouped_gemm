import unittest

from absl.testing import parameterized

from grouped_gemm import ops
import torch


def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5, atol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


def add_true_false_flags(x):
    out = []
    for y in x:
        for f in [False, True,]:
            out.append(y + (f, ))
    return out

def add_backend_type_flags(x):
    out = []
    for y in x:
        for f in ["cutlass", "cublas"]:
            out.append(y + (f,))
    return out


_TEST_PROBLEMS = add_true_false_flags(add_backend_type_flags(add_true_false_flags((
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
))))


def randn(bs, x, y):
    out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    return out.cuda().to(torch.bfloat16)


def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)


@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    def testGroupedGemm_FixedSizes(self, z, m, k, n, trans_b, backend_type, check_gradient):
        if backend_type == "cutlass" and (trans_b or check_gradient):
            return
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)
        batch_sizes = torch.tensor([m] * z)

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b, backend_type)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        if check_gradient:
            # Check gradients.
            out.sum().backward()
            expected_out.sum().backward()
            self.assertTrue(allclose(a.grad, a_ref.grad))
            self.assertTrue(allclose(b.grad, b_ref.grad))

    def testGroupedGemm_VariableSizes(self, z, m, k, n, trans_b, backend_type, check_gradient):
        if backend_type == "cutlass" and (trans_b or check_gradient):
            return
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b, backend_type)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        if check_gradient:
            out.sum().backward()
            expected_out.sum().backward()
            self.assertTrue(allclose(a.grad, a_ref.grad))
            self.assertTrue(allclose(b.grad, b_ref.grad))



if __name__ == '__main__':
    unittest.main()