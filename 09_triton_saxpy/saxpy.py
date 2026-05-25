import torch
import triton
import triton.language as tl


@triton.jit
def saxpy_kernel(x_ptr, y_ptr, a, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis = 0) # while tile (like blockIdx.x)
    offsets = pid * BLOCK + tl.arange(0, BLOCK) # absolute indices for this tile
    mask = offsets < n  # bounds guard for last tile

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    tl.store(y_ptr + offsets, a * x + y, mask=mask)



def saxpy(x, y, a):
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),) # ceil(n/BLOCK) program instances

    saxpy_kernel[grid](x, y, a, n, BLOCK)



if __name__ == "__main__":
    N = 1 << 20
    x = torch.ones(N, device="cuda")
    y = torch.ones(N, device="cuda")
    a = 2.0

    saxpy(x, y, a)

    expected = a * 1.0 + 1.0
    max_error = (y-expected).abs().max().item()
    print(f"Max Error  :{max_error}")




