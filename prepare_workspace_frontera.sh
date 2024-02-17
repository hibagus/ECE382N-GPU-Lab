#!/bin/bash
echo "Begin Preparing Your Workspace"
echo "Downloading Nsight Compute 2023.2.0..."
wget -O ncu_2023.2.0.tar.gz https://utexas.box.com/shared/static/zf8w3ioehjuy7qojaglnkqm4njvf65e4.gz -q
echo "Extracting Nsight Compute 2023.2.0..."
tar -xf ncu_2023.2.0.tar.gz -C nsight-compute/
echo "Cleaning up..."
rm ncu_2023.2.0.tar.gz

echo "Downloading Cat and Dog Dataset..."
wget -O catdog.tar.gz https://utexas.box.com/shared/static/my4goc3kxa9crea0pmb9iqgsuoj461io.gz -q
echo "Extracting Cat and Dog Dataset..."
tar -xf catdog.tar.gz -C application/
echo "Cleaning up..."
rm catdog.tar.gz

echo "Setting-up Python Virtual Environment..."
module load python3/3.9.2
python3 -m venv application/venv
echo "Activating Virtual Environment..."
source application/venv/bin/activate
echo "Downloading Python Packages..."
pip3 install -r application/requirements.txt

echo "Cloning CUDA Bench Repository..."
git submodule update --init --recursive
echo "Loading Toolkit Needed on TACC Frontera"
module load cuda/12.2
module load cmake
module load gcc/9.1.0
export CC=/opt/apps/gcc/9.1.0/bin/gcc
export CXX=/opt/apps/gcc/9.1.0/bin/g++
echo "Preparing CUDA Bench Build Environment..."
cd kernel && mkdir -p build && cd build
echo "Generating Makefile for CUDA Bench..."
cmake -DBUILD_MODE=Release ..
echo "Starting Compilation using 8 Parallel Threads..."
make

echo "Finished Preparing Workspace. Good Luck! :)"