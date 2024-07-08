# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import grouped_gemm_backend as backend

def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)

def get_ptrs(a, b, c, batch_sizes, trans_b):
    a_ptrs = torch.empty(batch_sizes.shape[0], dtype=torch.uint64, device="cpu")
    b_ptrs = torch.empty(batch_sizes.shape[0], dtype=torch.uint64, device="cpu")
    c_ptrs = torch.empty(batch_sizes.shape[0], dtype=torch.uint64, device="cpu")
    backend.gmm_cublas_get_ptrs(a, b, c, batch_sizes, trans_b, a_ptrs, b_ptrs, c_ptrs)
    return a_ptrs, b_ptrs, c_ptrs

def gmm(a, b, batch_sizes, trans_a=False, trans_b=False, c=None, backend_type="cutlass", a_ptrs=None, b_ptrs=None, c_ptrs=None):
    if c is None:
        c = _allocate_output(a, b, batch_sizes, trans_a, trans_b)
    if backend_type == "cutlass":
        backend.gmm_cutlass(a, b, c, batch_sizes, trans_a, trans_b)
    elif backend_type == "cublas":
        # backend_f = backend.gmm_cublas
        if a_ptrs is None:
            if not trans_a:
                # compute ptrs
                a_ptrs, b_ptrs, c_ptrs = get_ptrs(a, b, c, batch_sizes, trans_b)
                a_ptrs = a_ptrs.to(a.device)
                b_ptrs = b_ptrs.to(a.device)
                c_ptrs = c_ptrs.to(a.device)
            else:
                # ptrs are unused
                a_ptrs = torch.empty(0, dtype=torch.uint64, device=a.device)
                b_ptrs = torch.empty(0, dtype=torch.uint64, device=a.device)
                c_ptrs = torch.empty(0, dtype=torch.uint64, device=a.device)
        backend.gmm_cublas(a, b, c, batch_sizes, a_ptrs, b_ptrs, c_ptrs, trans_a, trans_b)
    return c

