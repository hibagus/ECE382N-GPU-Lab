#!/bin/bash
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


#wget  -q --show-progress --progress=dot 