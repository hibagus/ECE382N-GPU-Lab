#!/bin/bash
echo "Begin Preparing Your Workspace"
echo "Downloading Nsight Compute 2023.2.0..."
wget -O ncu_2023.2.0.tar.gz https://utexas.box.com/shared/static/zf8w3ioehjuy7qojaglnkqm4njvf65e4.gz -q --show-progress
echo "Extracting Nsight Compute 2023.2.0..."
tar -xf ncu_2023.2.0.tar.gz -C nsight-compute/
echo "Cleaning up..."
rm ncu_2023.2.0.tar.gz

echo "Downloading Cat and Dog Dataset..."
wget -O catdog.tar.gz https://utexas.box.com/shared/static/my4goc3kxa9crea0pmb9iqgsuoj461io.gz -q --show-progress
echo "Extracting Cat and Dog Dataset..."
tar -xf catdog.tar.gz -C application/
echo "Cleaning up..."
rm catdog.tar.gz

echo "Setting-up Python Virtual Environment..."
python -m venv application/venv
echo "Activating Virtual Environment..."
source application/venv/bin/activate
echo "Downloading Python Packages..."
pip3 install -r application/requirements.txt

echo "Cloning CUDA Bench Repository..."
git submodule update --init --recursive
echo "Loading CUDA Toolkit on TACC Lonestar6"
module load cuda/11.4
echo "Preparing CUDA Bench Build Environment..."
cd kernel && mkdir -p build && cd build
echo "Generating Makefile for CUDA Bench..."
cmake -DBUILD_MODE=Debug ..
echo "Starting Compilation using 8 Parallel Threads..."
make

echo "Finished Preparing Workspace. Good Luck! :)"