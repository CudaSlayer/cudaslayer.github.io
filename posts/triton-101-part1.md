# 🚀 Triton 101 - Part 1 - (NOT COMPLETE)

Introduction to GPU Kernel Programming Made Easy

## What is Triton?

Triton is a Python-based GPU programming framework that makes it easier
to write efficient GPU kernels. Unlike CUDA, which requires managing
low-level details like thread synchronization and shared memory, Triton
provides a higher-level abstraction that handles these complexities
automatically.

## Setup and Utilities

Before we dive into kernels, let's set up our environment and define
some helpful utilities:

    import os
    import torch
    import triton
    import triton.language as tl
    from IPython.core.debugger import set_trace

    # Enable interpretation mode for debugging
    os.environ['TRITON_INTERPRET'] = '1'

### GPU Readiness Check

When working with GPU kernels, we need to ensure our tensors are
properly configured:

    def check_tensors_gpu_ready(*tensors):
        """
        Validates whether PyTorch tensors are ready for GPU operations.

        Performs two critical checks:
        1. Contiguity: Tensors must be stored contiguously in memory
        2. CUDA: Tensors must be on GPU (skipped in TRITON_INTERPRET mode)
        """
        for t in tensors:
            assert t.is_contiguous, "A tensor is not contiguous"
            if not os.environ.get('TRITON_INTERPRET') == '1':
                assert t.is_cuda, "A tensor is not on cuda"

> **Info:**
>
> **Why this matters:** GPU kernels require contiguous memory layout for
> efficient access. This function catches configuration issues early,
> preventing cryptic runtime errors.
>

### Ceiling Division Helper

    def cdiv(a, b):
        """Ceiling division: returns ⌈a/b⌉"""
        return (a + b - 1) // b

    # Examples:
    # cdiv(10, 2) = 5
    # cdiv(10, 3) = 4

This is essential for calculating how many blocks we need when the data
size doesn't divide evenly by the block size.

## The Programming Model: CUDA vs Triton

Understanding the difference between CUDA and Triton's programming
models is crucial. Let's break it down:

### 🔴 CUDA: Two-Level Decomposition

**How the work is split**

- **Blocks** – groups of threads that run on the same Streaming Multiprocessor (SM)
- **Threads** – individual execution units that operate on **scalars**

> **Memory mindset:** every block shares the same chunk of shared memory, and you are responsible for coordinating how threads use it.

### 🔵 Triton: One-Level Decomposition

**How the work is split**

- **Blocks** (called "programs") – that's it!

> **Memory mindset:** programs act on vectors; Triton takes care of shared memory management behind the scenes, so you focus on masked vector math instead of thread bookkeeping.

> **Key Insight:** In Triton lingo, each kernel that processes a block
> is called a "program". So when we say "program ID" (pid), it's
> equivalent to "block ID" in CUDA terminology.
>

## Visualization: CUDA vs Triton

Let's visualize how CUDA and Triton handle vector addition differently
with a concrete example:

<div class="triton-widget" data-widget="cuda-triton"></div>

## Example 1: Adding Two Vectors (Size 8)

Let's say we want to compute `z = x + y` where all vectors have size 8,
using blocks of size 4.

> **Note:**
>
> **Setup:**
>
> - Input size: 8 elements
> - Block size: 4 elements
> - Number of blocks: 8 ÷ 4 = 2 blocks
>

### How CUDA Handles This

    Block 0: [Thread 0, Thread 1, Thread 2, Thread 3]
    Block 1: [Thread 4, Thread 5, Thread 6, Thread 7]

- Runs **2 blocks**, each with **4 threads**
- Total: **8 threads**
- Each thread computes **one scalar**: `z[i] = x[i] + y[i]`

### How Triton Handles This

    Block 0: processes elements [0:4]
    Block 1: processes elements [4:8]

- Runs **2 blocks**
- Each block performs **vectorized addition** on 4 elements
- Block 0: `z[0:3] = x[0:3] + y[0:3]`
- Block 1: `z[4:7] = x[4:7] + y[4:7]`

## Example 2: Handling Boundaries (Size 6)

Now let's add a complication: what if our data size doesn't divide
evenly by the block size?

> **Note:**
>
> **Setup:**
>
> - Input size: 6 elements
> - Block size: 4 elements
> - Number of blocks: ⌈6 ÷ 4⌉ = 2 blocks
>

    x = torch.tensor([1, 2, 3, 4, 5, 6])
    y = torch.tensor([10, 20, 30, 40, 50, 60])
    # Expected: z = [11, 22, 33, 44, 55, 66]

### CUDA C Kernel

    __global__ void add_cuda_k(float* x, float* y, float* z, int n) {
        // Identify this thread's position
        int block_id = blockIdx.x;      // 0 or 1
        int thread_id = threadIdx.x;    // 0, 1, 2, or 3
        int bs = blockDim.x;            // 4

        // Calculate global index
        int offs = block_id * bs + thread_id;

        // Scalar guard clause
        if (offs < n) {
            // Scalar operations
            float x_value = x[offs];
            float y_value = y[offs];
            float z_value = x_value + y_value;
            z[offs] = z_value;
        }
    }

    // Launch: add_cuda_k<<<2, 4>>>(d_x, d_y, d_z, 6);

> **Info:**
>
> **Key points:**
>
> - `offs`, `x_value`, `y_value`, `z_value` are all **scalars**
> - The guard condition `offs < n` is a **scalar check**
> - Each thread independently decides whether to execute
>

### Triton Kernel

    @triton.jit
    def add_triton_k(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
        # Identify this block's position
        block_id = tl.program_id(0)  # 0 or 1

        # Calculate vector of offsets
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        # Block 0: [0, 1, 2, 3]
        # Block 1: [4, 5, 6, 7]  <- indices 6,7 are out of bounds!

        # Create vector mask
        mask = offs < n
        # Block 0: [True, True, True, True]
        # Block 1: [True, True, False, False]

        # Vectorized loads (masked)
        x_values = tl.load(x_ptr + offs, mask=mask)
        y_values = tl.load(y_ptr + offs, mask=mask)

        # Vectorized operation
        z_values = x_values + y_values

        # Vectorized store (masked)
        tl.store(z_ptr + offs, z_values, mask=mask)

    # Launch the kernel
    def add_triton(x, y, z):
        n = x.shape[0]
        BLOCK_SIZE = 4
        num_blocks = triton.cdiv(n, BLOCK_SIZE)  # ⌈6/4⌉ = 2
        add_triton_k[(num_blocks,)](x, y, z, n, BLOCK_SIZE)

> **Info:**
>
> **Key points:**
>
> - `offs` is a **vector**: `[0,1,2,3]` or `[4,5,6,7]`
> - `mask` is a **vector of booleans**
> - `x_values`, `y_values`, `z_values` are all **vectors**
> - All operations are vectorized
>

## Comparison Table

| Aspect                | CUDA                       | Triton                        |
|-----------------------|----------------------------|-------------------------------|
| **Abstraction Level** | 2-level (blocks → threads) | 1-level (blocks only)         |
| **Computation Unit**  | Scalar (per thread)        | Vector (per block)            |
| **Memory Management** | Manual shared memory       | Automatic                     |
| **Masking**           | Scalar `if` statements     | Vectorized masks              |
| **Complexity**        | Lower-level, more control  | Higher-level, more productive |
| **Launch Syntax**     | `kernel<<>>()`             | `kernel[grid]()`              |

## Complete Example: Tensor Copy Kernel

Let's put everything together with a practical example: copying a
tensor.

**Goal:** Given a tensor `x` of shape `(n)`, copy it into another tensor
`z`.

    def copy(x, block_size, kernel_fn):
        """
        Copy a tensor using a custom Triton kernel.

        Parameters:
        - x: Input tensor (must be contiguous and on CUDA)
        - block_size: Number of elements each block processes
        - kernel_fn: The Triton kernel function

        Returns:
        - z: Output tensor (copy of x)
        """
        # Allocate output tensor
        z = torch.zeros_like(x)

        # Validate tensors are GPU-ready
        check_tensors_gpu_ready(x, z)

        # Calculate grid dimensions
        n = x.numel()
        n_blocks = cdiv(n, block_size)
        grid = (n_blocks,)  # Can be 1D, 2D, or 3D tuple

        # Launch kernel
        kernel_fn[grid](x, z, n, block_size)

        return z

> **Warning:**
>
> **Example usage:**
>
>     # For a tensor with 1000 elements and block_size=256:
>     # - n_blocks = ⌈1000/256⌉ = 4 blocks
>     # - Block 0: [0:256]
>     # - Block 1: [256:512]
>     # - Block 2: [512:768]
>     # - Block 3: [768:1000] ← needs masking!
>

## Summary: The Triton Advantage

> **Key Insight:**
>
> **All operations in Triton kernels are vectorized:**
>
> - [x] Loading data
> - [x] Operating on data
> - [x] Storing data
> - [x] Creating masks
>

### Benefits of Triton's approach:

1.  **Higher productivity:** No need to think about individual threads
2.  **Automatic optimization:** Triton handles shared memory and
    synchronization
3.  **Cleaner code:** Vectorized operations are more intuitive
4.  **Better performance:** Compiler optimizations at the vector level

### When to use Triton:

- Custom GPU operations not available in PyTorch
- Performance-critical kernels
- When you want GPU programming without CUDA complexity

## What's Next?

In Part 2, we'll dive deeper into:

- More complex Triton kernels
- 2D and 3D grid configurations
- Reduction operations
- Memory access patterns and optimization

Stay tuned for more Triton tutorials!

## Resources

- <a href="https://triton-lang.org/" target="_blank">Triton
  Documentation</a>
- <a href="https://github.com/openai/triton" target="_blank">OpenAI Triton
  GitHub</a>
- <a
  href="https://triton-lang.org/main/getting-started/tutorials/index.html"
  target="_blank">Triton Tutorials</a>
