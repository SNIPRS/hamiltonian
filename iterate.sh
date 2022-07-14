#!/bin/bash
#SBATCH -t 0-14:00
#SBATCH --mem=8000
#SBATCH -c 32

source ~/hamiltonian/bin/activate
python ./script.py
