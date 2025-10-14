# CUDA - Day 1: Hello World and Vector Addition

## Why CUDA Matters (Friendly Framing)

Imagine a classroom packed with hundreds of students. A CPU works like one brilliant student who solves every exercise alone—accurate but slow. CUDA lets you hand each student a single question so the entire class finishes the worksheet in seconds. In GPU terms, each “student” is a *thread*, and CUDA coordinates thousands of them at once.

> **Analogy:** Threads = students, blocks = rows of desks, the grid = the whole classroom.

## Step 0 · Install & Confirm

- Download the CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads) and follow the installer.
- Open a terminal and verify both your GPU driver and compiler:

   ```bash
   nvidia-smi      # Shows GPU model and temperature
   nvcc --version  # Confirms the CUDA compiler is available
   ```

If both commands succeed, the lab is ready.

---

## Program 1 · “Hello from the GPU”

This first kernel does nothing but introduce itself, yet it demonstrates how CUDA arranges blocks and threads.

<div data-widget="cuda-hello"></div>

```cpp
// hello.cu  — minimal CUDA kernel launch

// CUDA Hello World Program
// This program demonstrates basic CUDA kernel execution with multiple threads and blocks

#include <iostream> // For input/output operations (like printing to console)

// Kernel function: This runs on the GPU
// __global__ is a CUDA keyword that tells the compiler this function runs on the GPU
// and can be called from the CPU (host)
__global__ void helloKernel(){
    // Print a message from each thread, showing which block and thread it is
    // blockIdx.x = which block this thread is in (0, 1, 2, ...)
    // threadIdx.x = which thread within the current block (0, 1, 2, ...)
    printf("Hello from block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

int main(){
    // Launch the kernel on the GPU
    // <<<2,4>>> is CUDA's execution configuration
    // The first number (2) = number of blocks in the grid
    // The second number (4) = number of threads per block
    // So we'll have: 2 blocks × 4 threads = 8 total threads running on the GPU
    helloKernel<<<2,4>>>();

    // Wait for the GPU to finish executing the kernel
    // This is important because CUDA kernel launches are asynchronous
    // Without this, the CPU might continue and exit before the GPU finishes
    cudaDeviceSynchronize();

    // Print a message from the CPU to show the program is done
    std::cout<<"Kernel execution done!\n";

    return 0;
}

```

### Vocabulary in Plain English

| CUDA Term            | What students hear                                  |
|---------------------|-----------------------------------------------------|
| `__global__`        | “This function runs on the GPU.”                    |
| `helloKernel<<<2,4>>>` | “Create 2 desk rows (blocks), each with 4 students (threads).” |
| `blockIdx.x`        | Row number                                          |
| `threadIdx.x`       | Seat number inside the row                          |
| `cudaDeviceSynchronize()` | “Teacher waits until every student stops talking.” |

```
Grid (classroom)
┌───────────────┐
│ Block 0       │  four seats → threads 0–3
│ Block 1       │  four seats → threads 0–3
└───────────────┘
Each seat prints one greeting. Order of greetings is not guaranteed.
```

### Compile & Run

```bash
nvcc hello.cu -o hello -std=c++17
./hello
```

Sample output (order may vary):

```
Hello from block 1, thread 0!
Hello from block 1, thread 1!
Hello from block 0, thread 2!
...
```

> **NOTE:** GPU scheduling is free to rearrange block order—students call out in any sequence.

---

## Program 2 · Vector Addition

With the basics in place, let every thread add one pair of numbers. The GPU holds two input vectors and produces a third, element by element.

<div data-widget="cuda-vector"></div>

```cpp
// vector_add.cu — add two vectors on the GPU

// CUDA Vector Addition Program
// This program demonstrates basic CUDA programming by adding two vectors on the GPU

#include <iostream>     // For input/output operations (like printing to console)
#include <vector>       // For using std::vector to store data on CPU
#include <cuda_runtime.h> // CUDA runtime API for GPU operations

// Kernel function: This runs on the GPU
// __global__ is a CUDA keyword that tells the compiler this function runs on the GPU
// and can be called from the CPU (host)
__global__ void vectorAdd(const float *a, const float *b, float *c, int n){
    // Calculate the global thread index
    // blockIdx.x = which block this thread is in (0, 1, 2, ...)
    // blockDim.x = how many threads are in each block (e.g., 256)
    // threadIdx.x = which thread within the current block (0 to 255)
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Boundary check: make sure we don't go out of vector bounds
    // This is important because we might launch more threads than needed
    if(idx<n){
        // Perform the vector addition: c[i] = a[i] + b[i]
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    // === STEP 1: SETUP AND DATA PREPARATION ON CPU ===

    // Define the size of our vectors (1000 elements)
    const int n = 1000;

    // Calculate memory needed for each vector (in bytes)
    // sizeof(float) = 4 bytes, so bytes = 1000 * 4 = 4000 bytes
    const size_t bytes = n*sizeof(float);

    // Create three vectors on the CPU (host)
    // h_ prefix indicates "host" (CPU) memory
    std::vector<float> h_a(n); // First input vector
    std::vector<float> h_b(n); // Second input vector
    std::vector<float> h_c(n); // Result vector

    // Initialize the input vectors with sample data
    for(int i=0; i<n; i++){
        h_a[i] = static_cast<float>(i);        // Vector A: [0, 1, 2, 3, ..., 999]
        h_b[i] = static_cast<float>(i) * 2.0f; // Vector B: [0, 2, 4, 6, ..., 1998]
    }

    // === STEP 2: ALLOCATE MEMORY ON GPU ===

    // Declare pointers for GPU (device) memory
    // d_ prefix indicates "device" (GPU) memory
    float *d_a, *d_b, *d_c;

    // Allocate memory on the GPU for each vector
    // cudaMalloc() allocates memory on the GPU device
    cudaMalloc(&d_a, bytes); // Allocate memory for vector A on GPU
    cudaMalloc(&d_b, bytes); // Allocate memory for vector B on GPU
    cudaMalloc(&d_c, bytes); // Allocate memory for result vector C on GPU

    // === STEP 3: COPY DATA FROM CPU TO GPU ===

    // Transfer data from CPU (host) to GPU (device)
    // cudaMemcpy() copies data between host and device memory
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice); // Copy A to GPU
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice); // Copy B to GPU

    // === STEP 4: CONFIGURE AND LAUNCH GPU KERNEL ===

    // Define how many threads we want per block
    // ThreadsPerBlock should be a multiple of 32 (warp size) for efficiency
    const int threadsPerBlock = 256;

    // Calculate how many blocks we need in the grid
    // This ensures we have enough threads to cover all vector elements
    // Formula: ceil(n / threadsPerBlock)
    const int blocksPerGrid = (n + threadsPerBlock - 1)/ threadsPerBlock;

    // Launch the kernel on the GPU
    // <<<blocksPerGrid, threadsPerBlock>>> is CUDA's execution configuration
    // It tells CUDA to launch 'blocksPerGrid' blocks, each with 'threadsPerBlock' threads
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // === STEP 5: COPY RESULTS BACK FROM GPU TO CPU ===

    // Transfer the result from GPU back to CPU
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // === STEP 6: VERIFY RESULTS ===

    // Print the first 50 results to verify the calculation is correct
    std::cout<<"Verification \n";
    for(int i=0;i<50;i++){
        std::cout<<h_a[i]<<" + "<<h_b[i]<<" : "<<h_c[i]<<"\n";
    }

    // === STEP 7: CLEAN UP GPU MEMORY ===

    // Free the allocated GPU memory
    // This is important to prevent memory leaks
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

```

### What the GPU is Doing

```
CPU arrays (h_a, h_b)
        │ cudaMemcpy (Host ➜ Device)
        ▼
GPU arrays (d_a, d_b) ──> vectorAdd kernel (each thread: c[i] = a[i] + b[i])
        │ cudaMemcpy (Device ➜ Host)
        ▼
CPU result (h_c)
```

| Scenario                    | Threads involved                 | Result stored in                |
|----------------------------|----------------------------------|---------------------------------|
| `idx = 0` (first element)  | Block 0 · Thread 0               | `c[0]`                          |
| `idx = 255`                | Block 0 · Thread 255             | `c[255]`                        |
| `idx = 256`                | Block 1 · Thread 0               | `c[256]`                        |
| `idx = 356`                | Block 1 · Thread 100             | `c[356]`                        |
| `idx = 999` (last element) | Block 3 · Thread 231             | `c[999]`                        |
| `idx = 1000` (extra thread)| Block 3 · Thread 232             | Guarded by `if (idx < n)` — no write |

```
Block 3 (covers indices 768‒1023)
┌──────────────────────────────┐
│ idx 768 … idx 999   → work ✓ │
│ idx 1000 … idx 1023 → skipped✗│
└──────────────────────────────┘
```

> **Why the guard matters:** the launch creates more threads than elements so that the grid is a tidy multiple of `threadsPerBlock`. The check `if (idx < n)` keeps those surplus threads from reading or writing past the end of the vectors.

---

### That's it for Day 1. Stay tuned for Day 2.
