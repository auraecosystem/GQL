import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def softmax_kernel_v1(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    in_max = -float('inf')
    in_exp_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_max_new = tl.maximum(in_max, tl.max(in_data, axis=-1))
        in_exp_sum = in_exp_sum * tl.exp(in_max - in_max_new) + tl.sum(tl.exp(in_data - in_max_new), axis=-1)
        in_max = in_max_new
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask)
        in_exp = tl.exp(in_data - in_max)
        tl.store(output_ptr + pid * row_stride + col_range + offset, in_exp / in_exp_sum, mask=col_mask)

torch.manual_seed(0)
x = torch.randn(8192, 8192, device=DEVICE)
n_rows, n_cols = x.shape
output = torch.empty_like(x)
BLOCK_SIZE = 256
temp = torch.randn(n_rows, n_cols, device=DEVICE)
softmax_kernel_v1[(n_rows,)](
        temp,
        output,
        temp.stride(0),
        n_cols,
        BLOCK_SIZE
)#warmup
torch.cuda.empty_cache() #clean cache

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
softmax_kernel_v1[(n_rows,)](
        x,
        output,
        x.stride(0),
        n_cols,
        BLOCK_SIZE
)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)

print(f'Softmax Triton V1 Version Elapsed: {elapsed_time_ms:.3f}ms')
