
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

Since most operations in graphics applications are highly parallel (i.e., primitives, fragments, and pixels can be processed in parallel during each stage of the graphics pipeline), GPUs have been designed as massively parallel processors to extract these parallelisms [2]. While early GPUs before the 2000s were fixed-function accelerators, they evolved to become more programmable: programmable shaders (early 2000s), unified shaders (early 2006), and general-purpose GPUs (2007) [4, 13-15]. The latter was started with the introduction of groundbreaking Tesla architecture in 2007 [6] which became the foundation of NVIDIA GPUs for almost two decades. Nowadays, GPUs are popular not only for graphics applications but also for accelerating diverse workloads with abundant parallelism, such as high-performance computing and machine learning applications. Figure 2 shows the relation between hardware and software perspectives in GPUs.  

#### Hardware perspective
To signify its massively parallel architecture, manufacturers often advertise GPUs to have thousands of cores (i.e., CUDA Cores (CC) in NVIDIA GPUs, Stream Processors (SP) in AMD GPUs, or Vector Engines (XVE) in Intel GPUs.). However, the term _cores_ in GPUs is not the same as in CPUs; the term _cores_ in GPUs refers to the execution units (i.e., ALUs). These _cores_ are then grouped into one processor (Streaming Multiprocessor (SM) in NVIDIA GPUs, Compute Unit (CU) in AMD GPUs, Compute Slice (SLC) in Intel GPUs), which is called Streaming Multiprocessor (SM) in NVIDIA GPUs. NVIDIA GPU can have 100s of SMs, each with many _cores_, for a total of thousands of _cores_. A simplified illustration of an SM inside NVIDIA Volta is given in Figure 1 [3]. Interested readers should consult the whitepaper released by NVIDIA to find detailed SM architecture for each generation of NVIDIA GPUs: Volta [7], Turing [8], Ampere [9], Hopper [10], and Ada Lovelace [12]. 

Each SM contains four SM sub-partitions (SMSP), L1 instruction cache, L1 data cache, shared memory, and texture cache. Starting from Volta, L1 data cache, texture cache, and shared memory are implemented as unified on-chip memory, giving the users flexibility on how on-chip memory should be managed; either 100\% as cache (hardware-managed) or some portions of it as shared memory (user-managed). Users who choose to use shared memory must carefully manage its use since it will reduce the amount of L1 and texture cache, possibly degrading the performance. The L2 cache is shared for all SMs in GPU, and it interfaces directly with off-chip memory (e.g., GDDR or HBM). 

Each SMSP has its on-chip memory: L0 instruction cache, constant cache, and register files. CUDA Cores (CC) are the default computation units inside the SMSP, which consist of FP64 units, FP32 units, INT32 units, and Special Function Units (SFUs). The number of FP32 units is used to advertise the number of CUDA Cores in GPUs. SFUs are used to compute transcendental functions such as trigonometric functions (e.g., sine, cosine), reciprocal, and square root. Compared to consumer-class GPUs, data-center-class GPUs feature significantly more FP64 units to handle HPC applications that use double precision for accuracy-sensitive computation. Starting from Volta architecture, specialized computation units called Tensor Cores (TC) are added to accelerate GEMM computation, which is abundant in many machine learning workloads. Tensor Cores will be briefly explained in Section~\ref{Introduction:Tensor_Cores}. 


#### Software perspective
GPU execution model is called Single-Instruction Multiple-Thread (SIMT), which is a modification to Single-Instruction Multiple-Data (SIMD) [6]. In addition to executing one instruction on multiple data, SIMT applies this one instruction to multiple independent threads in parallel. This allows programmers to write thread-parallel code for individual threads and data-parallel code for coordinated threads. A GPU-accelerated application, called kernel, can have millions of threads. This collection of threads is called a grid. Inside, the threads are grouped into thread blocks or cooperative thread arrays (CTAs). A grid can have as many as $2^{31}-1$ thread blocks, each contains up to 1024 threads. Theoretically, a kernel can have as many as 2 trillion threads! (i.e., $(2^{31}-1) \times 1024 = 2,199,023,254,528$) 

The global scheduler (i.e., Gigathread Engine) schedules each thread block into the available SM and manages the context switches of the thread blocks for each SM. Ideally, each SM should hold multiple thread blocks to allow for aggressive context switching; when one thread block stalls (e.g., due to memory access), it can run another thread block to hide the latency and keep the SM busy. Finally, each SMSP executes a warp of threads; the warp scheduler maps the threads within the warp to the cores, which then execute in a lock-step fashion. Any differences in the thread execution path (e.g., due to different branch outcomes) within the warp will cause thread divergence; instead of running in parallel, the threads within the warp will run serially based on their execution path, reducing computation efficiency. Note that thread divergence only occurs within the warp since each warp can be independently executed. 


### Tensor Cores
In addition to CUDA Cores, Volta (2017) or newer generations of GPUs are equipped with Tensor Cores. Tensor Cores provide significant performance improvements over CUDA Cores for GEMM and GEMM-like operations. These operations are abundant in Machine Learning workloads, making GPUs popular accelerators. Other manufacturers included their version of Tensor Cores in their GPUs: AMD with Matrix Core (2020) [1] and Intel with XMX Matrix Engine (2022) [5].

Unlike CUDA Cores that run the instruction at the thread level, Tensor Cores run the instruction at warp level, performing **matrix multiply-accumulate (MMA)** operations per instruction. For instance, Volta [7] is equipped with first-generation Tensor Cores capable of multiplying two half-precision (FP16) 4x4 matrices, resulting in a 4x4 matrix, either in half- or full-precision (FP16 or FP32), in each clock cycle. This is accomplished using 64 matrix-accumulate (MAC) units arranged in 4x4x4 three-dimensional cube (i.e., four layers, each with four columns and four rows of MAC units). The instruction supported by Volta's Tensor Cores is `hmma.884`: half-precision MMA with instruction shape 8x8x4. Each of `hmma.884` instruction performs 512 floating-point operations (i.e., 256 fused-multiply-add operations). Two Tensor Cores in each Volta's SMSP work in tandem to execute one `hmma.884` instruction in four clock cycles\footnote{Peak performance calculation as follows: each SMSP has two Tensor Cores for a total of eight Tensor Cores per SM. With 80 SMs, there are 640 Tensor Cores in NVIDIA Tesla V100 running at 1597 MHz boost clock. Each Tensor Core can execute 64 FMAs per clock cycle or 128 flops per cycle, and hence, theoretical peak performance is $640 \times 64 \times 2 \times 1597 \times 10^6 = 130$ TFLOP/s. This is four times higher than CUDA Cores peak performance for FP16 at $32$ TFLOP/s.}.

Subsequent generations of Tensor Cores support more precisions and larger instruction shapes. Turing [8] features second-generation Tensor Cores that add `hmma.1688` and `hmma.16816` that take 8 and 16 clock cycles to execute, respectively, using the same 4x4x4 MAC structure and two Tensor Cores per SMSP. They also support INT8, INT4, and INT1 precision accessible through the new `imma.8816`, `imma.8832`, and `bmma.88128` instructions, respectively. Ampere [9] features newly designed Tensor Cores with 256 MAC units arranged in 8x4x8 three-dimensional cuboid, making it incompatible with executing older `hmma.884` instruction. Although each SMSP only has one Tensor Core, it can execute `hmma.1688` and `hmma.16816` in 4 and 8 clock cycles, respectively. In addition, they add FP64 (`dmma.884`), TF32 (`hmma.1684` and `hmma.1688`), and BF16 (`hmma.1688` and `hmma.16816`) precisions and support for sparse matrix multiplication. Finally, four-generation Tensor Cores in Hopper [10] and Ada Lovelace [12] double the number of Ampere's Tensor Cores MAC units with 512 MAC units arranged in 8x4x16 three-dimensional cuboid. They have support for new quarter-precision (FP8) and new Tensor Cores instruction that run on warp-group (`GMMA`).

### Kernel Profiling using NSight Compute
While developing applications that run on GPUs is not a straightforward task, characterizing the runtime behavior, identifying the bottleneck, and optimizing the performance of such applications are quite another. It is crucial to have insight into the hardware activities to understand the runtime behavior of the codes, which is often a challenging but rewarding effort. NVIDIA provides a kernel profiling tool called NSight Compute (NCU) \cite{NCU} to help collect hardware metrics to understand the runtime behavior of GPU kernels. NCU is included with CUDA Toolkit distribution and supports Volta and newer generations of GPUs.

```
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

Listing~\ref{lst:ncu_version_metrics} shows the command used to check the NCU version and the available metrics. Note that every GPU and every version of NCU will have different metrics, and thus, it is wise to check them. **Specifically, we will use NCU shipped with CUDA Toolkit 12.2 or newer for this lab (i.e., NCU version 2023.2.0.0 or newer)**, making it easier to profile application that uses Tensor Cores.

Let's say we want to collect the total bytes transferred between the GPU and off-chip memory (DRAM), the kernel execution time, and the number of floating-point instructions executed in vector addition application. This application performs element-wise addition of 100 pairs of vectors (i.e., 200 vectors) where each vector has 2 billion elements. The Listing~\ref{lst:sample_command} shows the command how to achieve that scenario in NVIDIA A100 GPU while Listing~\ref{lst:sample_result} shows the result of this profiling. 

The names of the metrics we are interested in are put after `--metrics` argument, separated with a comma. Note that there are no spaces between metric names. Since there are a limited number of hardware performance counters inside the GPU, NCU may need to replay the kernel depending on what metrics and the number of metrics that are collected. The replay behavior is defined using `--replay-mode` argument. By default, NCU will use kernel replay, which incurs significant overhead. It is recommended to have application replay instead, as long as the application has deterministic behavior in its overall execution. The `--csv` is used to tell NCU to output the result in comma-separated value format, which can then be imported to a spreadsheet (e.g., Microsoft Excel) or other data analysis libraries (e.g., pandas). The output is dumped into a file whose name is specified after `--log-file` argument instead of printed out to the standard output of the terminal. The `--print-summary per-kernel` argument groups kernels with the same name and launch configuration, making the profiling report shorter. Finally, the path to the application we want to profile is put at the end of the command, including its arguments (if any).

```
$ ncu --replay-mode application --metrics gpu__time_duration,dram__bytes,sm__sass_thread_inst_executed_op_fp32_pred_on --csv --print-summary per-kernel   --log-file profile_data.csv ./vectoradd 2000000000
```

If you choose to summarize the profiling report using `--print-summary per-kernel`, you will see four additional columns on your profiling report: `Invocations`, `Minimum`, `Maximum`, and `Average`. The `Invocations` indicates how many times the kernel is launched on GPU. In addition, the `Minimum`, `Maximum`, and `Average` indicate the minimum, maximum, and average value of a particular metric across all invocations of this kernel. In the case of the vector addition application mentioned before, the `add_vector` kernel is launched 100 times since 100 pairs of vectors need to be processed. Therefore, to find the aggregate value of a particular metric across all invocations, the number of invocations should be multiplied by the average value. For example, the total execution time of `add_vector` kernel is $100 \times 17,473,124.16~ns = 1,747,312,416~ns$.

You may also notice that each collected metric has four sub-metrics consisting of `.avg`, `.max`, `.min`, and `.sum` values. While these values are equal for `gpu__time_duration`, they are not for the other two metrics in this example. For `dram__bytes`, the `.sum` is the total bytes of data accessed in DRAM collected from all 40 memory channels (i.e., NVIDIA A100 GPU has five stacks of HBM2/HBM2e memory. Each stack has eight 128-bit memory channels.) in NVIDIA A100 GPU. The `.avg` gives the average bytes of data accessed in each memory channel (i.e., `.sum` divided by 40) while `.max` and `.min` give the maximum and minimum bytes of data accessed through the memory channels, respectively. On the other hand, the `.sum` sub-metric in `sm__sass_thread_inst_executed_op_fp32_pred_on` gives the total of all FP32 instructions executed from all 108 SMs in NVIDIA A100 GPU while `.avg` gives the average number of FP32 executed in each SM  (i.e., `.sum` divided by 108).

```
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


This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```JS
   const API_KEY = 'ENTER YOUR API';
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
