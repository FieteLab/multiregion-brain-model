#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

# Path to your conda setup script (adjust if needed)
CONDA_ENV="towertask"
BASE_SEED=25

# Create a variable for the log directory
LOG_error_DIR="logs/model/error"
LOG_output_DIR="logs/model/output"

# Ensure the log directories exist
mkdir -p "$LOG_error_DIR" "$LOG_output_DIR"

# Function to submit jobs
submit_job() {
    local trial=$1
    local model_cmd=$2

    sbatch --gres=gpu:1 --mem=20G --time=12:00:00 \
           --job-name="${trial}" \
           --output="$LOG_output_DIR/${trial}.log" \
           --error="$LOG_error_DIR/${trial}.log" \
           --wrap="source /cm/shared/openmind/anaconda/3-2022.05/etc/profile.d/conda.sh && conda activate $CONDA_ENV && \
            sleep 10 && \
                   echo 'Job started at: ' \$(date) > $LOG_output_DIR/${trial}.log && \
                   echo 'SLURM Job ID: ' \$SLURM_JOB_ID >> $LOG_output_DIR/${trial}.log && \
                   echo 'SLURM Node List: ' \$SLURM_NODELIST >> $LOG_output_DIR/${trial}.log && \
                   echo 'Memory usage at start:' >> $LOG_output_DIR/${trial}.log && free -h >> $LOG_output_DIR/${trial}.log && \
                   echo 'GPU usage at start:' >> $LOG_output_DIR/${trial}.log && nvidia-smi >> $LOG_output_DIR/${trial}.log && \
                   $model_cmd && \
                   echo 'Memory usage at end:' >> $LOG_output_DIR/${trial}.log && free -h >> $LOG_output_DIR/${trial}.log && \
                   echo 'GPU usage at end:' >> $LOG_output_DIR/${trial}.log && nvidia-smi >> $LOG_output_DIR/${trial}.log && \
                   echo 'Job finished at: ' \$(date) >> $LOG_output_DIR/${trial}.log"
}

# ---------------------------------------------------------------------
# DEFINE BASE COMMANDS FOR EACH MODEL
# ---------------------------------------------------------------------
M1="python3 train.py --reset_data --grid_assignment position position position --Np 800"
M2="python3 train.py --reset_data --grid_assignment position position position --Np 800 --new_model"
M3="python3 train.py --reset_data --grid_assignment position position position --Np 800 --with_mlp --mlp_input_type sensory"
M4="python3 train.py --reset_data --grid_assignment position position evidence  --Np 800 --with_mlp --mlp_input_type sensory --new_model"
M5="python3 train.py --reset_data --grid_assignment position position position --Np 800 --with_mlp --mlp_input_type sensory --new_model"

# M0plus (with velocity); M0 (no velocity)
M0plus="python3 train.py --reset_data --rnn_only --larger_rnn_with_scalffold_size --larger_rnn_with_LEC_size --rnn_add_pos --rnn_add_evi --with_mlp"
M0="python3 train.py --reset_data --rnn_only --larger_rnn_with_scalffold_size --larger_rnn_with_LEC_size"

# One ICML reviewer brought up: what if RNNs match #parameters instead of #neurons in M5?
M0plus_matchNumParams="python3 train.py --reset_data --rnn_only --hidden_size 157 --rnn_add_pos --rnn_add_evi --with_mlp" 
M0_matchNumParams="python3 train.py --reset_data --rnn_only --hidden_size 158" 

# To test with CA3 recurrence, use the following command (with appropriate adjustment wrt your job script):
# Use --modified_mixture means we use only the hippocampal cells projected by g -> p for updating Wps, Wsp, i.e., torch.relu(p_g)
# instead of using the mix of projection from g and s, i.e., p_for_update = torch.relu(p_g+p_s) 
M1_with_CA3_recur="$M1 --add_recurrence"
M2_with_CA3_recur="$M2 --add_recurrence"
M3_with_CA3_recur="$M3 --add_recurrence"
M4_with_CA3_recur="$M4 --add_recurrence"
M5_with_CA3_recur="$M5 --add_recurrence"

# ---------------------------------------------------------------------
# LAUNCH THE LEARNING-RATE RUNS
# ---------------------------------------------------------------------
for i in 1 2 3; do
    SEED=$((BASE_SEED + i))

    # ----------------------------------------
    # M1–M5: Paper default uses 5e-4
    # ----------------------------------------
    for lr in 0.001; do
        submit_job "M1_trial_${i}_lr${lr}" \
            "$M1 --trial_name icml_M1_trial_${i}_lr${lr} --seed $SEED --learning_rate $lr --num_episodes 20000"
        
        # submit_job "M2_trial_${i}_lr${lr}" \
        #     "$M2 --trial_name icml_M2_trial_${i}_lr${lr} --seed $SEED --learning_rate $lr --num_episodes 20000"
        
        # submit_job "M3_trial_${i}_lr${lr}" \
        #     "$M3 --trial_name icml_M3_trial_${i}_lr${lr} --seed $SEED --learning_rate $lr --num_episodes 20000"
        
        # submit_job "M4_trial_${i}_lr${lr}" \
        #     "$M4 --trial_name icml_M4_trial_${i}_lr${lr} --seed $SEED --learning_rate $lr --num_episodes 20000"
        
        # submit_job "M5_trial_${i}_lr${lr}" \
        #     "$M5 --trial_name icml_M5_trial_${i}_lr${lr} --seed $SEED --learning_rate $lr --num_episodes 20000"
    done

    # -----------------------------------------------------
    # M0plus and M0: Paper default uses 1e-4
    # -----------------------------------------------------
    # for lr in 0.001 0.0001 0.0005 0.00005; do
    #     # M0plus
    #     submit_job "rebuttal_m5size_bothv_M0_trial_${i}_lr${lr}" \
    #         "$M0plus --trial_name rebuttal_m5size_bothv_M0_trial_${i}_lr${lr} \
    #          --seed $SEED --learning_rate $lr --num_episodes 20000"

    #     # M0
    #     submit_job "rebuttal_m5size_nov_M0_trial_${i}_lr${lr}" \
    #         "$M0 --trial_name rebuttal_m5size_nov_M0_trial_${i}_lr${lr} \
    #          --seed $SEED --learning_rate $lr --num_episodes 20000"
    # done

done
