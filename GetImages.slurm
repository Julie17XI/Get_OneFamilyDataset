#!/bin/bash

#SBATCH -J GetImages.py
#SBATCH -o GetImages.txt
#SBATCH -p gpu2v100
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xi.tang@my.utsa.edu

module load anaconda3
yes | conda create -n get_facial_images python=3.8.8
conda activate get_facial_images
pip install -r required_libraries.txt
python GetImages.py LanceOHara
conda deactivate
yes | conda env remove -n get_facial_images
