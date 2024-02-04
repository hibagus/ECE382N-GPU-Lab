
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Lab 2: Workload Characterization and Performance Evaluation on GPUs</h3>

  <p align="center">
    ECE382N: Computer Performance Evaluation and Benchmarking (Spring 2024) 
    <br />
    Lizy K. John, Lecturer
    <br />
    Bagus Hanindhito and Ruihao Li, Teaching Assistants
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About Lab

In this lab, you will have hands-on experience profiling tools for GPU-accelerated applications. Specifically, we will use Nsight Compute (NCU) to perform kernel profiling and application profiling. For kernel profiling, we will take a look at the GEMM kernel provided by CUTLASS and cuBLAS. For application profiling, we will investigate machine learning training as well as inference as a bonus. **If you plan to use TACC to run your experiment, we strongly recommended to start early since there are a limited number of GPU-equipped compute node and the queues can be long.**


<!-- GETTING STARTED -->
## Introduction (0 Point)

### Basic GPU Architecture

In this section, we will briefly explain the basics of GPU architecture. Since we will use NVIDIA GPUs equipped with Tensor Cores in our lab, our explanation will be more focused on recent NVIDIA GPUs (Volta or newer generations). Still, the concepts can easily be extended to GPUs from other manufacturers (e.g., AMD, Intel). 

Since most operations in graphics applications are highly parallel (i.e., primitives, fragments, and pixels can be processed in parallel during each stage of the graphics pipeline), GPUs have been designed as massively parallel processors to extract these parallelisms [^2]. While early GPUs before the 2000s were fixed-function accelerators, they evolved to become more programmable: programmable shaders (early 2000s), unified shaders (early 2006), and general-purpose GPUs (2007) [^4], [^13], [^14], [^15]. The latter was started with the introduction of groundbreaking Tesla architecture in 2007 [^6] which became the foundation of NVIDIA GPUs for almost two decades. Nowadays, GPUs are popular not only for graphics applications but also for accelerating diverse workloads with abundant parallelism, such as high-performance computing and machine learning applications. Figure 2 shows the relation between hardware and software perspectives in GPUs.  

#### Hardware perspective
To signify its massively parallel architecture, manufacturers often advertise GPUs to have thousands of cores (i.e., CUDA Cores (CC) in NVIDIA GPUs, Stream Processors (SP) in AMD GPUs, or Vector Engines (XVE) in Intel GPUs.). However, the term _cores_ in GPUs is not the same as in CPUs; the term _cores_ in GPUs refers to the execution units (i.e., ALUs). These _cores_ are then grouped into one processor (Streaming Multiprocessor (SM) in NVIDIA GPUs, Compute Unit (CU) in AMD GPUs, Compute Slice (SLC) in Intel GPUs), which is called Streaming Multiprocessor (SM) in NVIDIA GPUs. NVIDIA GPU can have 100s of SMs, each with many _cores_, for a total of thousands of _cores_. A simplified illustration of an SM inside NVIDIA Volta is given in Figure 1 [^3]. Interested readers should consult the whitepaper released by NVIDIA to find detailed SM architecture for each generation of NVIDIA GPUs: Volta [^7], Turing [^8], Ampere [^9], Hopper [^10], and Ada Lovelace [^12]. 

Each SM contains four SM sub-partitions (SMSP), L1 instruction cache, L1 data cache, shared memory, and texture cache. Starting from Volta, L1 data cache, texture cache, and shared memory are implemented as unified on-chip memory, giving the users flexibility on how on-chip memory should be managed; either 100% as cache (hardware-managed) or some portions of it as shared memory (user-managed). Users who choose to use shared memory must carefully manage its use since it will reduce the amount of L1 and texture cache, possibly degrading the performance. The L2 cache is shared for all SMs in GPU, and it interfaces directly with off-chip memory (e.g., GDDR or HBM). 

Each SMSP has its on-chip memory: L0 instruction cache, constant cache, and register files. CUDA Cores (CC) are the default computation units inside the SMSP, which consist of FP64 units, FP32 units, INT32 units, and Special Function Units (SFUs). The number of FP32 units is used to advertise the number of CUDA Cores in GPUs. SFUs are used to compute transcendental functions such as trigonometric functions (e.g., sine, cosine), reciprocal, and square root. Compared to consumer-class GPUs, data-center-class GPUs feature significantly more FP64 units to handle HPC applications that use double precision for accuracy-sensitive computation. Starting from Volta architecture, specialized computation units called Tensor Cores (TC) are added to accelerate GEMM computation, which is abundant in many machine learning workloads. Tensor Cores will be briefly explained in Section~\ref{Introduction:Tensor_Cores}. 


#### Software perspective
GPU execution model is called Single-Instruction Multiple-Thread (SIMT), which is a modification to Single-Instruction Multiple-Data (SIMD) [^6]. In addition to executing one instruction on multiple data, SIMT applies this one instruction to multiple independent threads in parallel. This allows programmers to write thread-parallel code for individual threads and data-parallel code for coordinated threads. A GPU-accelerated application, called kernel, can have millions of threads. This collection of threads is called a grid. Inside, the threads are grouped into thread blocks or cooperative thread arrays (CTAs). A grid can have as many as $2^{31}-1$ thread blocks, each contains up to 1024 threads. Theoretically, a kernel can have as many as 2 trillion threads! (i.e., $(2^{31}-1) \times 1024 = 2,199,023,254,528$) 

The global scheduler (i.e., Gigathread Engine) schedules each thread block into the available SM and manages the context switches of the thread blocks for each SM. Ideally, each SM should hold multiple thread blocks to allow for aggressive context switching; when one thread block stalls (e.g., due to memory access), it can run another thread block to hide the latency and keep the SM busy. Finally, each SMSP executes a warp of threads; the warp scheduler maps the threads within the warp to the cores, which then execute in a lock-step fashion. Any differences in the thread execution path (e.g., due to different branch outcomes) within the warp will cause thread divergence; instead of running in parallel, the threads within the warp will run serially based on their execution path, reducing computation efficiency. Note that thread divergence only occurs within the warp since each warp can be independently executed. 


### Tensor Cores
In addition to CUDA Cores, Volta (2017) or newer generations of GPUs are equipped with Tensor Cores. Tensor Cores provide significant performance improvements over CUDA Cores for GEMM and GEMM-like operations. These operations are abundant in Machine Learning workloads, making GPUs popular accelerators. Other manufacturers included their version of Tensor Cores in their GPUs: AMD with Matrix Core (2020) [^1] and Intel with XMX Matrix Engine (2022) [^5].

Unlike CUDA Cores that run the instruction at the thread level, Tensor Cores run the instruction at warp level, performing **matrix multiply-accumulate (MMA)** operations per instruction. For instance, Volta [^7] is equipped with first-generation Tensor Cores capable of multiplying two half-precision (FP16) 4x4 matrices, resulting in a 4x4 matrix, either in half- or full-precision (FP16 or FP32), in each clock cycle. This is accomplished using 64 matrix-accumulate (MAC) units arranged in 4x4x4 three-dimensional cube (i.e., four layers, each with four columns and four rows of MAC units). The instruction supported by Volta's Tensor Cores is `hmma.884`: half-precision MMA with instruction shape 8x8x4. Each of `hmma.884` instruction performs 512 floating-point operations (i.e., 256 fused-multiply-add operations). Two Tensor Cores in each Volta's SMSP work in tandem to execute one `hmma.884` instruction in four clock cycles (i.e., Peak performance calculation as follows: each SMSP has two Tensor Cores for a total of eight Tensor Cores per SM. With 80 SMs, there are 640 Tensor Cores in NVIDIA Tesla V100 running at 1597 MHz boost clock. Each Tensor Core can execute 64 FMAs per clock cycle or 128 flops per cycle, and hence, theoretical peak performance is $640 \times 64 \times 2 \times 1597 \times 10^6 = 130$ TFLOP/s. This is four times higher than CUDA Cores peak performance for FP16 at $32$ TFLOP/s).

Subsequent generations of Tensor Cores support more precisions and larger instruction shapes. Turing [8] features second-generation Tensor Cores that add `hmma.1688` and `hmma.16816` that take 8 and 16 clock cycles to execute, respectively, using the same 4x4x4 MAC structure and two Tensor Cores per SMSP. They also support INT8, INT4, and INT1 precision accessible through the new `imma.8816`, `imma.8832`, and `bmma.88128` instructions, respectively. Ampere [9] features newly designed Tensor Cores with 256 MAC units arranged in 8x4x8 three-dimensional cuboid, making it incompatible with executing older `hmma.884` instruction. Although each SMSP only has one Tensor Core, it can execute `hmma.1688` and `hmma.16816` in 4 and 8 clock cycles, respectively. In addition, they add FP64 (`dmma.884`), TF32 (`hmma.1684` and `hmma.1688`), and BF16 (`hmma.1688` and `hmma.16816`) precisions and support for sparse matrix multiplication. Finally, four-generation Tensor Cores in Hopper [^10] and Ada Lovelace [^12] double the number of Ampere's Tensor Cores MAC units with 512 MAC units arranged in 8x4x16 three-dimensional cuboid. They have support for new quarter-precision (FP8) and new Tensor Cores instruction that run on warp-group (`GMMA`).

### Kernel Profiling using NSight Compute
While developing applications that run on GPUs is not a straightforward task, characterizing the runtime behavior, identifying the bottleneck, and optimizing the performance of such applications are quite another. It is crucial to have insight into the hardware activities to understand the runtime behavior of the codes, which is often a challenging but rewarding effort. NVIDIA provides a kernel profiling tool called NSight Compute (NCU) [^11] to help collect hardware metrics to understand the runtime behavior of GPU kernels. NCU is included with CUDA Toolkit distribution and supports Volta and newer generations of GPUs.

Snippet below shows the command used to check the NCU version and the available metrics. Note that every GPU and every version of NCU will have different metrics, and thus, it is wise to check them. **Specifically, we will use NCU shipped with CUDA Toolkit 12.2 or newer for this lab (i.e., NCU version 2023.2.0.0 or newer)**, making it easier to profile application that uses Tensor Cores.

```bash
# Check NCU Version
$ ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2023 NVIDIA Corporation
Version 2023.2.0.0 (build 32895467) (public-release)

# Query available metrics for GPU 0 and output to text file.
$ ncu --devices 0 --query-metrics > metrics.txt
$ cat metrics.txt
Device NVIDIA A100-PCIE-40GB (GA100)
----------------- ------------ ------------ ---------------------------
Metric Name       Metric Type  Metric Unit  Metric Description
----------------- ------------ ------------ ---------------------------
dram__bytes       Counter      byte         # of bytes accessed in DRAM
dram__bytes_read  Counter      byte         # of bytes read from DRAM
....
```

Let's say we want to collect the total bytes transferred between the GPU and off-chip memory (DRAM), the kernel execution time, and the number of floating-point instructions executed in vector addition application. This application performs element-wise addition of 100 pairs of vectors (i.e., 200 vectors) where each vector has 2 billion elements. The snippet below shows the command how to achieve that scenario in NVIDIA A100 GPU.

```bash
$ ncu --replay-mode application --metrics gpu__time_duration,dram__bytes,sm__sass_thread_inst_executed_op_fp32_pred_on --csv --print-summary per-kernel   --log-file profile_data.csv ./vectoradd 2000000000
```

The snippet below shows the result of this profiling. 

```bash
$ cat profile_data.csv
==PROF== Connected to process 3998654 (/raid/bagus/GPU_Lab/vectoradd)
==PROF== Disconnected from process 3998654
==PROF== Connected to process 3999805 (/raid/bagus/GPU_Lab/vectoradd)
==PROF== Disconnected from process 3999805
==PROF== Connected to process 4000302 (/raid/bagus/GPU_Lab/vectoradd)
==PROF== Disconnected from process 4000302
==PROF== Creating report from application replay data: 0%.....................100%
...,"Kernel Name","...","Invocations","...,","Metric Name","Metric Unit","Minimum","Maximum","Average"
...,"add_vector.","...,"100","...","dram__bytes.avg","byte","599,747,068.80","599,767,049.60","599,752,096.58"
...,"add_vector.","...,"100","...","dram__bytes.max","byte","599,769,344.00","599,832,960.00","599,779,247.36"
...,"add_vector.","...,"100","...","dram__bytes.min","byte","599,718,272.00","599,734,656.00","599,728,037.12"
...,"add_vector.","...,"100","...","dram__bytes.sum","byte","23,989,882,752.00","23,990,681,984.00","23,990,083,863.04"
...,"add_vector.","...,"100","...","gpu__time_duration.avg","nsecond","17,466,880.00","17,480,032.00","17,473,124.16"
...,"add_vector.","...,"100","...","gpu__time_duration.max","nsecond","17,466,880.00","17,480,032.00","17,473,124.16"
...,"add_vector.","...,"100","...","gpu__time_duration.min","nsecond","17,466,880.00","17,480,032.00","17,473,124.16"
...,"add_vector.","...,"100","...","gpu__time_duration.sum","nsecond","17,466,880.00","17,480,032.00","17,473,124.16"
...,"add_vector.","...,"100","...","sm__sass_thread_inst_executed_op_fp32_pred_on.avg","inst","18,518,518.52","18,518,518.52","18,518,518.52"
...,"add_vector.","...,"100","...","sm__sass_thread_inst_executed_op_fp32_pred_on.max","inst","29,498,880.00","29,586,944.00","29,548,072.96"
...,"add_vector.","...,"100","...","sm__sass_thread_inst_executed_op_fp32_pred_on.min","inst","7,369,216.00","7,416,320.00","7,387,453.44"
...,"add_vector.","...,"100","...","sm__sass_thread_inst_executed_op_fp32_pred_on.sum","inst","2,000,000,000.00","2,000,000,000.00","2,000,000,000.00"
```

The names of the metrics we are interested in are put after `--metrics` argument, separated with a comma. Note that there are no spaces between metric names. Since there are a limited number of hardware performance counters inside the GPU, NCU may need to replay the kernel depending on what metrics and the number of metrics that are collected. The replay behavior is defined using `--replay-mode` argument. By default, NCU will use kernel replay, which incurs significant overhead. It is recommended to have application replay instead, as long as the application has deterministic behavior in its overall execution. The `--csv` is used to tell NCU to output the result in comma-separated value format, which can then be imported to a spreadsheet (e.g., Microsoft Excel) or other data analysis libraries (e.g., pandas). The output is dumped into a file whose name is specified after `--log-file` argument instead of printed out to the standard output of the terminal. The `--print-summary per-kernel` argument groups kernels with the same name and launch configuration, making the profiling report shorter. Finally, the path to the application we want to profile is put at the end of the command, including its arguments (if any).



If you choose to summarize the profiling report using `--print-summary per-kernel`, you will see four additional columns on your profiling report: `Invocations`, `Minimum`, `Maximum`, and `Average`. The `Invocations` indicates how many times the kernel is launched on GPU. In addition, the `Minimum`, `Maximum`, and `Average` indicate the minimum, maximum, and average value of a particular metric across all invocations of this kernel. In the case of the vector addition application mentioned before, the `add_vector` kernel is launched 100 times since 100 pairs of vectors need to be processed. Therefore, to find the aggregate value of a particular metric across all invocations, the number of invocations should be multiplied by the average value. For example, the total execution time of `add_vector` kernel is $100 \times 17,473,124.16 ns = 1,747,312,416 ns$.

You may also notice that each collected metric has four sub-metrics consisting of `.avg`, `.max`, `.min`, and `.sum` values. While these values are equal for `gpu__time_duration`, they are not for the other two metrics in this example. For `dram__bytes`, the `.sum` is the total bytes of data accessed in DRAM collected from all 40 memory channels (i.e., NVIDIA A100 GPU has five stacks of HBM2/HBM2e memory. Each stack has eight 128-bit memory channels.) in NVIDIA A100 GPU. The `.avg` gives the average bytes of data accessed in each memory channel (i.e., `.sum` divided by 40) while `.max` and `.min` give the maximum and minimum bytes of data accessed through the memory channels, respectively. On the other hand, the `.sum` sub-metric in `sm__sass_thread_inst_executed_op_fp32_pred_on` gives the total of all FP32 instructions executed from all 108 SMs in NVIDIA A100 GPU while `.avg` gives the average number of FP32 executed in each SM  (i.e., `.sum` divided by 108).


## Preparing Working Space (0 Point)
Although this section is worth zero points, you must complete it to prepare your working directory for the experiment :). Please kindly follow the snippet below to clone the GitHub repository into your work directory at TACC and download the necessary files. It may take 10-15 minutes to run the `prepare_workspace.sh` script, so sit back and relax. When running the script, make sure that you have a stable connection. If you use system other than TACC, you may need to modify the provided scripts to fit your system. 

```bash
# Go to your work directory at TACC
$ cd $WORK

# Clone GitHub repository
$ git clone https://github.com/hibagus/ECE382N-GPU-Lab.git

# Run preparation script
$ cd ECE382N-GPU-Lab
$ ./prepare_workspace.sh

# Source environment variable
source set_environment
```
Since TACC only provides older NCU shipped with CUDA Toolkit 12.0, the script will download the necessary version of NCU for this experiment. After setting up your workspace, make sure that you source the script `set_environment` every time you open a terminal or submit a job to TACC using SLURM. It will set the correct path for NCU 2023.2.0 instead of using NCU provided by TACC. It will also activate the Python Virtual Environment required for Application Characterization (Section~\ref{sec:Application_Characterization}). Although you can prepare your working space using login node, all of the experiments must be done in GPU-equipped compute node (i.e., `gpu-a100` or `gpu-a100-dev` in Lonestar6).



## Kernel Characterization (50 Points)
In the first experiment, you are asked to profile GEMM and GEMV kernels. GEMM stands for general matrix multiply and is abundant in many workloads, including machine learning. The problem size of GEMM is usually denoted by $\{m,n,k\}$, which represents multiplication between matrix $A_{m \times n}$ and matrix $B_{n \times k}$ resulting in matrix $C_{m \times k}$. In addition, GEMV stands for general matrix-vector multiply and is a special case of GEMM where $n=1$ (i.e., $\{m,1,k\}$). 

You will compare the characteristics of GEMM kernel provided by cuBLAS and CUTLASS. Although NVIDIA develops both libraries, only CUTLASS is open-source, while cuBLAS is close-source. CUTLASS provides a collection of C++ template headers for implementing GEMM operations. On the other hand, cuBLAS contains the collection of GEMM kernels hand-tuned at the assembly level for specific devices, problem dimensions, and target precision. cuBLAS uses heuristics to choose the most performant kernels for each usage case. 

We have provided you with a wrapper program to run the GEMM kernel, so you don't have to create your own wrapper. Assuming your working space is correctly configured (Section~\ref{sec:Preparing_Working_Space}), this wrapper should have been compiled as binary at `kernel/bin/gemm_cuda_bench`. Snippet below gives some sample commands to interact with the wrapper. 

```bash
# CUTLASS GEMM Kernel; problem size {2048,2048,2048}; FP16; Tensor Cores
$ ./gemm_cuda_bench -M fp16 -A fp16 2048 2048 2048

# cuBLAS GEMM Kernel; problem size {2048,2048,2048}; FP16; Tensor Cores
$ ./gemm_cuda_bench --usecublas -M fp16 -A fp16 2048 2048 2048

# CUTLASS GEMM Kernel; problem size {2048,2048,2048}; FP32; CUDA Cores
$ ./gemm_cuda_bench --cudacoresonly -M fp32 -A fp32 2048 2048 2048

# cuBLAS GEMM Kernel; problem size {2048,2048,2048}; FP32; CUDA Cores
$ ./gemm_cuda_bench --usecublas --cudacoresonly -M fp32 -A fp32 2048 2048 2048

# Multiple kernel launches; useful for measuring average kernel execution time.
$ ./gemm_cuda_bench -M fp16 -A fp16 --iterations 100 2048 2048 2048

# Help Menu
$ ./gemm_cuda_bench --help
```

For the experiment, use the following problem sizes: GEMM $\{1024,1024,1024\}$, GEMM $\{32768,32768,32768\}$, GEMV $\{1024,1,1024\}$, and GEMV $\{32768,1,32768\}$. You are free to add another problem size if you wish. In your report, please state which metrics on NCU you use for collecting the data and write the name of the GEMM kernel (no need to put the full name of the kernel; the first 32 characters of the name are sufficient). With these four problem sizes, below are your tasks for this experiment. 
Table~\ref{tab:sample_table_kernel_1_4} is a sample table for Problem 1 to Problem 4, while Table~\ref{tab:sample_table_kernel_5} is a sample table for Problem 5 for your reference. Feel free to modify the table as you wish. 

1. (**10 points**) Compare the average kernel execution time between CUTLASS and cuBLAS for each problem size. Comparison should be done in both FP32 (CUDA Cores) and FP16 (Tensor Cores). Tabulate the data and make a bar chart showing this comparison. 
2. (**10 points**) Calculate the achieved throughput (GFLOP/s) for CUTLASS and cuBLAS GEMM kernels for each problem size in both FP32 (CUDA Cores) and FP16 (Tensor Cores). Don't forget to count all floating-point operations on CUDA and Tensor Cores to derive the throughput. Tabulate the data and make a bar chart showing this comparison.
3. (**10 points**) Calculate the arithmetic intensity (FLOP/byte) for CUTLASS and cuBLAS GEMM Kernels for each problem size in both FP32 (CUDA Cores) and FP16 (Tensor Cores). Use the total data transferred from/to DRAM as the data volume. Tabulate the data and make a bar chart showing this comparison.
4. (**10 points**) Calculate L2 Cache MPKI (level-two-cache/LT\$) for CUTLASS and cuBLAS GEMM Kernels for each problem size in both FP32 (CUDA Cores) and FP16 (Tensor Cores). Tabulate the data and make a bar chart showing this comparison. 
5. (**10 points**) For GEMM $\{32768,32768,32768\}$, present the instruction mix (branch, integer, floating-point CUDA Cores, floating-point Tensor Cores, load/store) of both CUTLASS and cuBLAS GEMM kernels in both FP32 (CUDA Cores) and FP16 (Tensor Cores). You are free to add more classes of instructions. Tabulate the data and make a pie chart.

We also provide an example for SLURM script to submit to TACC Lonestar6 (i.e., `run_ncu_part1.slurm`. Please start early, as the job queue in TACC can be quite unpredictable. It will take around 4 hours to collect all of the metrics in this experiment, so it is recommended that you first plan which metrics should be collected before writing the SLURM script. 


## Application Characterization (50 Points)
In the second experiment, you are asked to profile the training of a machine learning model. To shorten the time needed, we will not train the model from scratch. Instead, we will take the pre-trained model and fine-tune it for our needs. We take the ResNet-50 model trained using the ImageNet dataset with 1000 classes and fine-tune it to classify the images of dogs and cats. You have been provided with a Python script, `application/train.py`, to fine-tune the model. Snippet below provides examples of using the training script. 

```bash
# Run fine-tuning for 1 Epoch using full-precision (FP32)
$ python train.py --precision fp32 --num_epoch 1

# Run fine-tuning for 1 Epoch using mixed-precision (FP32+FP16)
$ python train.py --precision fp16 --num_epoch 1

# Disable the validation step and checkpoint storage; 
# this is useful to shorten profiling time.
$ python train.py --profile

# Help Menu
$ python train.py --help
```

Below are your tasks for this experiment. Table~\ref{tab:sample_table_apps_1_3} is a sample table for Problem 1 to Problem 3, while Table~\ref{tab:sample_table_apps_5} is a sample table for Problem 5 for your reference. Feel free to modify the table as you wish. **For profiling purposes, you only need to run the training for one epoch**.

1. (**10 points**) Compare the application's GPU runtime (i.e., the sum of the duration of all kernels) for full-precision and mixed-precision training. 
2. (**10 points**) Calculate the application's achieved throughput (GFLOP/s) full-precision and mixed-precision training. Don't forget to count all floating-point operations on CUDA and Tensor Cores to derive the throughput.
3. (**10 points**) Calculate the arithmetic intensity (FLOP/byte) for full-precision and mixed-precision training. Use the total data transferred from/to DRAM as the data volume.
4. (**10 points**) For mixed-precision training: 
   - Calculate the percentage of floating-point operations executed in CUDA Cores and Tensor Cores.
   - Calculate the percentage of the total time of kernels that use Tensor Cores.
   - List the top 5 kernels that use Tensor Cores the most in terms of duration and flops (no need to put the full name; first 32 characters of kernel name are sufficient) and the top 5 kernels that use Tensor Cores the least (including kernels that do not use Tensor Cores at all).
    
5. (**10 points**) Classify the kernel based on their operations (GEMM, Convolution, Element-Wise, Softmax, etc.) for both full-precision and mixed-precision training. You can infer it from the kernel name; don't worry if it is inaccurate. Make a pie chart for the total kernel execution time for each class of kernels. 
    
6. (**Bonus 10 points**) We also provide inference script (`application/infer.py`). To use this script, you must run the fine-tuning for at least 15-20 epochs (without any profiling) to generate the model checkpoint. Find out the inference GPU's runtime, achieved throughput, and arithmetic intensity for full-precision and mixed-precision inference. 
    
Even with only one epoch, profiling can take a long time (2-4 hours), so it is highly recommended to use the SLURM script or tmux (interactive session) to avoid any interruption in case you lose connection to the TACC cluster. We also provide an example for SLURM script to submit to TACC Lonestar6 (i.e., `run_ncu_part2.slurm`).


<!-- ACKNOWLEDGEMENTS -->
## References
[^1] Advanced Micro Devices. 2020. AMD CDNA Architecture. Whitepaper. Advanced Micro Devices, California, US. https:
//www.amd.com/system/files/documents/amd-cdna-whitepaper.pdf
[^2] David Blythe. 2008. Rise of the Graphics Processor. Proc. IEEE 96, 5 (2008), 761–778. https://doi.org/10.1109/JPROC.2008.917718
[^3] Jack Choquette, Olivier Giroux, and Denis Foley. 2018. Volta: Performance and Programmability. IEEE Micro 38, 2 (2018), 42–52.
https://doi.org/10.1109/MM.2018.022071134
[^4] William J. Dally, Stephen W. Keckler, and David B. Kirk. 2021. Evolution of the Graphics Processing Unit (GPU). IEEE Micro 41, 6
(2021), 42–51. https://doi.org/10.1109/MM.2021.3113475
[^5] H. Jiang. 2022. Intel’s Ponte Vecchio GPU : Architecture, Systems & Software. In 2022 IEEE Hot Chips 34 Symposium (HCS). IEEE
Computer Society, Los Alamitos, CA, USA, 1–29. https://doi.org/10.1109/HCS55958.2022.9895631
[^6] Erik Lindholm, John Nickolls, Stuart Oberman, and John Montrym. 2008. NVIDIA Tesla: A Unified Graphics and Computing
Architecture. IEEE Micro 28, 2 (2008), 39–55. https://doi.org/10.1109/MM.2008.31
[^7] NVIDIA Corporation. 2017. NVIDIA Tesla V100 GPU Architecture: The World’s Most Advanced Data Center GPU. Whitepaper.
NVIDIA Corporation, California, US. https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.
pdf
[^8] NVIDIA Corporation. 2018. NVIDIA Turing GPU Architecture: Graphics Reinvented. Whitepaper. NVIDIA Corporation, California,
US. https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-
Architecture-Whitepaper.pdf
[^9] NVIDIA Corporation. 2020. NVIDIA A100 Tensor Core GPU Architecture: Unprecedented Acceleration at Every Scale. Whitepa-
per. NVIDIA Corporation, California, US. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-
architecture-whitepaper.pdf
[^10] NVIDIA Corporation. 2022. NVIDIA H100 Tensor Core GPU Architecture: Exceptional Performance, Scalability, and Security for
The Data Center. Whitepaper. NVIDIA Corporation, California, US. https://resources.nvidia.com/en-us-tensor-core/gtc22-
whitepaper-hopper
[^11] NVIDIA Corporation. 2023. Nsight Compute CLI. https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html.
[^12] NVIDIA Corporation. 2023. NVIDIA Ada GPU Architecture: Designed to deliver outstanding gaming and creating, professional
graphics, AI, and compute performance. Whitepaper. NVIDIA Corporation, California, US. https://images.nvidia.com/aem-
dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.0.pdf
[^13] Jon Peddie. 2023. The History of the GPU - Eras and Environment. Springer International Publishing, Cham. https://doi.org/10.
1007/978-3-031-13581-1
[^14] Jon Peddie. 2023. The History of the GPU - New Developments. Springer International Publishing, Cham. https://doi.org/10.1007/
978-3-031-14047-1
[^15] Jon Peddie. 2023. The History of the GPU - Steps to Invention. Springer International Publishing, Cham. https://doi.org/10.1007/978-
3-031-10968-3
