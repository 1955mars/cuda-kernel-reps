import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols


    #load row, pad out-of-bounds with -inf
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=float('-inf'))


    #pass 1 - max (numerical stability)
    x_max = tl.max(x, axis=0)

    #pass 2 - shift, exp, sum
    x = tl.exp(x - x_max)
    x_sum = tl.sum(x, axis=0)

    #pass 3 - normalize
    x = x/x_sum

    tl.store(y_ptr + row * n_cols + offsets, x, mask=mask)



def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols) # e.g. n_cols = 1024, BLOCK=1024
    y = torch.empty_like(x)

    softmax_kernel[(n_rows,)](x, y, n_cols, BLOCK)
    return y


if __name__ == "__main__":
    x = torch.randn(1024, 1024, device="cuda")

    y_triton = softmax(x)
    y_torch = torch.softmax(x, dim=1)

    max_error = (y_torch - y_triton).abs().max().item()
    print(f"Max Error: {max_error}")