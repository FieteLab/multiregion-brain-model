#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

# Activate your conda environment
source /cm/shared/openmind/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate towertask

# Define the Python script to run
# SCRIPT="plot_data_mat.py"
SCRIPT="non_linear_reduction.py"

# Common arguments
COMMON_ARGS="--sequence_length 20 --fov 5 --max_towers 5 --hidden_size 32 --alpha 0.1 --lambdas 7 8 11 --gcpc p --Np 800 --policy_type RNN --learning_rate 5e-4 --q 1"

# Submit jobs for each model with the respective arguments
for model in M1 M2 M3 M4 M5; do
  # Define the output and error log file paths
  OUTPUT_LOG="logs/NLR/${model}_output.log"
  ERROR_LOG="logs/NLR/${model}_error.log"

  # Ensure the logs directory exists
  mkdir -p logs/NLR/

  # Prepare the job submission command with 12 hours runtime and log files
  sbatch --gres=gpu:1 \
         --time=12:00:00 \
         --output=$OUTPUT_LOG \
         --error=$ERROR_LOG \
         --wrap="python $SCRIPT $COMMON_ARGS --model_type $model"
  
  echo "Submitted job for model $model with arguments: $COMMON_ARGS --model_type $model"
done
