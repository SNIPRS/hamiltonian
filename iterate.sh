#!/bin/bash
#SBATCH -t 0-32:00
#SBATCH --mem=32000
#SBATCH -c 48

source ~/hamiltonian/bin/activate
python ./script.py
