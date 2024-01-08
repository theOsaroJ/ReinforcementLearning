#!/bin/bash
#$ -q hpc@@colon
#$ -N RL
#$ -pe smp 16

module load python

python3 RL.py > output_RL.txt

if [[ ! -d Results ]]; then
mkdir Results
fi

mv Prior_data* Results
mv Predicted* Results
mv R2* Results
