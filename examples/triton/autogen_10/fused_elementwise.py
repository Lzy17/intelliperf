#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def fused_elementwise_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    # Fused operations: (x * y) + z -> ReLU
    result = x * y + z
    output = tl.where(result > 0, result, 0) # ReLU activation

    tl.store(output_ptr + offsets, output, mask=mask)

def fused_elementwise(x, y, z):
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    fused_elementwise_kernel[grid](
        x, y, z, output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output

def main(size=2**16):
    x = torch.randn(size, device='cuda', dtype=torch.float16)
    y = torch.randn(size, device='cuda', dtype=torch.float16)
    z = torch.randn(size, device='cuda', dtype=torch.float16)

    rep = 100
    
    for _ in range(10):
        output_triton = fused_elementwise(x, y, z)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        output_triton = fused_elementwise(x, y, z)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton fused element-wise time: {triton_time:.4f} ms")
    
    # output_torch = torch.nn.functional.relu(x * y + z)
    # print(f"Triton output: {output_triton}")
    # print(f"Torch output: {output_torch}")
    # assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0), "Triton and PyTorch results differ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Fused Element-wise Benchmark")
    parser.add_argument("--size", type=int, default=2**16, help="Size of the input tensors")
    args = parser.parse_args()
    
    main(args.size) 