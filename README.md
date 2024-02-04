
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
GPU execution model is called Single-Instruction Multiple-Thread (SIMT), which is a modification to Single-Instruction Multiple-Data (SIMD) [6]. In addition to executing one instruction on multiple data, SIMT applies this one instruction to multiple independent threads in parallel. This allows programmers to write thread-parallel code for individual threads and data-parallel code for coordinated threads. A GPU-accelerated application, called kernel, can have millions of threads. This collection of threads is called a grid. Inside, the threads are grouped into thread blocks or cooperative thread arrays (CTAs). A grid can have as many as $2^{31}-1$ thread blocks, each contains up to 1024 threads. Theoretically, a kernel can have as many as 2 trillion threads\footnote{i.e., $(2^{31}-1) \times 1024 = 2,199,023,254,528$}! 

The global scheduler (i.e., Gigathread Engine) schedules each thread block into the available SM and manages the context switches of the thread blocks for each SM. Ideally, each SM should hold multiple thread blocks to allow for aggressive context switching; when one thread block stalls (e.g., due to memory access), it can run another thread block to hide the latency and keep the SM busy. Finally, each SMSP executes a warp of threads; the warp scheduler maps the threads within the warp to the cores, which then execute in a lock-step fashion. Any differences in the thread execution path (e.g., due to different branch outcomes) within the warp will cause thread divergence; instead of running in parallel, the threads within the warp will run serially based on their execution path, reducing computation efficiency. Note that thread divergence only occurs within the warp since each warp can be independently executed. 

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
