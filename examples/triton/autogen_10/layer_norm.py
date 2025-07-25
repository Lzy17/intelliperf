#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def layer_norm_kernel(
    x_ptr, y_ptr,
    w_ptr, b_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_y_m, stride_y_n,
    eps,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    offset_m = pid_m * stride_x_m
    
    x_row_ptr = x_ptr + offset_m
    y_row_ptr = y_ptr + offset_m
    
    mean = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        x = tl.load(x_row_ptr + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
        
        mean += tl.sum(x, axis=0)
        var += tl.sum(x * x, axis=0)

    mean = mean / N
    var = var / N - mean * mean
    
    rstd = 1 / tl.sqrt(var + eps)
    
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        x = tl.load(x_row_ptr + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        
        y = (x - mean) * rstd * w + b
        
        tl.store(y_row_ptr + cols * stride_y_n, y.to(y_ptr.dtype.element_ty), mask=mask)

def layer_norm(x, w, b, eps):
    M, N = x.shape
    y = torch.empty_like(x)

    grid = (M,)
    
    layer_norm_kernel[grid](
        x, y, w, b,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_SIZE_N=triton.next_power_of_2(N)
    )
    return y

def main(M=8192, N=8192):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    w = torch.randn((N,), device='cuda', dtype=torch.float16)
    b = torch.randn((N,), device='cuda', dtype=torch.float16)
    eps = 1e-5
    
    rep = 100
    
    for _ in range(10):
        y_triton = layer_norm(x, w, b, eps)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        y_triton = layer_norm(x, w, b, eps)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton layer norm time: {triton_time:.4f} ms")

    # y_torch = torch.nn.functional.layer_norm(x, (N,), w, b, eps)
    # print(f"Triton output: {y_triton}")
    # print(f"Torch output: {y_torch}")
    # assert torch.allclose(y_triton, y_torch, atol=1e-1, rtol=0), "Triton and PyTorch results differ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton LayerNorm Benchmark")
    parser.add_argument("--M", type=int, default=8192, help="Number of rows")
    parser.add_argument("--N", type=int, default=8192, help="Number of columns")
    args = parser.parse_args()
    
    main(args.M, args.N) 