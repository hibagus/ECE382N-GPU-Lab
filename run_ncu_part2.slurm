#!/bin/bash
#----------------------------------------------------

#SBATCH -J NCU_Part2_GPULab                # Job name
#SBATCH -o runjob/NCU_Part2_GPULab.o%j     # Name of stdout output file
#SBATCH -e runjob/NCU_Part2_GPULab.e%j     # Name of stderr error file
#SBATCH -p gpu-a100                        # Queue (partition) name (use gpu-a100 for job >2 hours, use gpu-a100-dev for job <2 hours)
#SBATCH -N 1                               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:59:59                        # Run time (hh:mm:ss) !! Adjust the Time !!
#SBATCH --mail-user=                       # Put your email
#SBATCH --mail-type=all                    #
#SBATCH -A                                 # Put your allocation name

# Other commands must follow all #SBATCH directives.
source set_environment
which ncu

###### Problem 1
# FP32
ncu ... python ./application/train.py ...
# FP16
ncu ... python ./application/train.py ...

###### Problem 2
# FP32
ncu ... python ./application/train.py ...
# FP16
ncu ... python ./application/train.py ...

###### Problem 3
# FP32
ncu ... python application/train.py ...
# FP16
ncu ... python application/train.py ...

###### Problem 6 (BONUS)
# Run actual fine tuning not in profiling mode to generate model checkpoint

# FP32
python application/train.py ...
# FP16
python application/train.py ...

# FP32
ncu ... python application/train.py ...
# FP16
ncu ... python application/train.py ...