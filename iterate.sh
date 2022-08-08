#!/bin/bash
#SBATCH -t 0-24:00
#SBATCH --mem=12000
#SBATCH -c 48

source ~/hamiltonian/bin/activate
python ./script.py
