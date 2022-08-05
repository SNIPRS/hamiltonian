#!/bin/bash
#SBATCH -t 0-36:00
#SBATCH --mem=48000
#SBATCH -c 48

source ~/hamiltonian/bin/activate
python ./script.py
