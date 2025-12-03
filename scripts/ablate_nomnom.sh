#!/bin/bash
# Ablate training and visualization over hyperparameter values
# Usage: ./ablate_nomnom.sh <param_name> <value1> [value2] [value3] ...
# Example: ./ablate_nomnom.sh max_energy 4 6 8 10 12
# 
# Parallel execution: Set PARALLELISM environment variable to control max concurrent jobs
# Example: PARALLELISM=8 ./ablate_nomnom.sh max_energy 4 6 8 10 12
#
# Model type: Set USE_LINEAR_MODEL environment variable to control model type
# Example: USE_LINEAR_MODEL=1 ./ablate_nomnom.sh max_energy 4 6 8 10 12  (uses linear_model)
# Example: USE_LINEAR_MODEL=0 ./ablate_nomnom.sh max_energy 4 6 8 10 12  (uses nomnom_model, default)
#
# RL options: Set ENABLE_RL and ENABLE_COMMUNICATION to enable RL features
# Example: ENABLE_RL=true ./ablate_nomnom.sh max_energy 4 6 8 10 12  (enables RL adaptation)
# Example: ENABLE_RL=true ENABLE_COMMUNICATION=true ./ablate_nomnom.sh max_energy 4 6 8  (enables both RL and communication)
# When ENABLE_RL=false (default), uses standard evolution-only training (train_nomnom.py)
# When ENABLE_RL=true, uses RL-enabled training (train_nomnom_rl.py)
#
# RL parameters: Can also set RL-specific hyperparameters
# Example: ENABLE_RL=true RL_LEARNING_RATE=1e-4 RL_BUFFER_SIZE=16 ./ablate_nomnom.sh max_energy 4 6 8

# Check if parameter name and at least one value are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <param_name> <value1> [value2] [value3] ..."
    echo "Example: $0 max_energy 4 6 8 10 12"
    exit 1
fi

# Get script directory first (needed by the function)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Function to clear all contents in visual_output directory without deleting the folder itself
clear_visual_output() {
    VISUAL_OUTPUT_DIR="$SCRIPT_DIR/../dirt/examples/nomnom/visual_output"
    if [ -d "${VISUAL_OUTPUT_DIR}" ]; then
        echo "Clearing contents of ${VISUAL_OUTPUT_DIR}..."
        # Remove all files and subdirectories but keep the parent directory
        find "${VISUAL_OUTPUT_DIR}" -mindepth 1 -delete
        echo "Cleared visual_output directory."
    else
        echo "visual_output directory does not exist, creating it..."
        mkdir -p "${VISUAL_OUTPUT_DIR}"
    fi
}

# root level folder to clear : e.g. linear_model, but don't delete folder
# Usage: clear_folder [folder_path]
# If no path provided, defaults to linear_model in the root dirt directory
clear_folder() {
    local folder_path="${1:-$SCRIPT_DIR/../linear_model}"
    
    if [ -d "${folder_path}" ]; then
        echo "Clearing contents of ${folder_path}..."
        # Remove all files and subdirectories but keep the parent directory
        find "${folder_path}" -mindepth 1 -delete
        echo "Cleared ${folder_path} directory."
    else
        echo "${folder_path} directory does not exist, creating it..."
        mkdir -p "${folder_path}"
    fi
}

PARAM_NAME="$1"
shift  # Remove param_name from arguments, leaving only values
PARAM_VALUES=("$@")  # All remaining arguments are values
RAW_DIR="$SCRIPT_DIR/../dirt/examples/nomnom/raw"
mkdir -p "$RAW_DIR"

# Set parallelism (default to 4 if not specified)
PARALLELISM=${PARALLELISM:-4}

# Set model type (default to 1 for linear_model)
USE_LINEAR_MODEL=${USE_LINEAR_MODEL:-1}

# Set RL options (default to false for backward compatibility)
ENABLE_RL=${ENABLE_RL:-false}
ENABLE_COMMUNICATION=${ENABLE_COMMUNICATION:-false}

# RL hyperparameters (only used if ENABLE_RL=true)
RL_LEARNING_RATE=${RL_LEARNING_RATE:-1e-4}
RL_BUFFER_SIZE=${RL_BUFFER_SIZE:-16}
RL_ADAPT_FREQUENCY=${RL_ADAPT_FREQUENCY:-8}
SOCIAL_REWARD_WEIGHT=${SOCIAL_REWARD_WEIGHT:-0.5}
SURVIVAL_REWARD_WEIGHT=${SURVIVAL_REWARD_WEIGHT:-1.0}
ENERGY_REWARD_WEIGHT=${ENERGY_REWARD_WEIGHT:-0.1}
COMMUNICATION_RADIUS=${COMMUNICATION_RADIUS:-5}

# Determine output directory based on model type and RL settings
if [ "${USE_LINEAR_MODEL}" = "1" ] || [ "${USE_LINEAR_MODEL}" = "true" ]; then
    if [ "${ENABLE_RL}" = "true" ] || [ "${ENABLE_RL}" = "1" ]; then
        if [ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ]; then
            OUTPUT_DIR_BASE="$SCRIPT_DIR/../rl_experiments/linear_model_rl_comm"
        else
            OUTPUT_DIR_BASE="$SCRIPT_DIR/../rl_experiments/linear_model_rl"
        fi
    else
        OUTPUT_DIR_BASE="$SCRIPT_DIR/../linear_model"
    fi
else
    if [ "${ENABLE_RL}" = "true" ] || [ "${ENABLE_RL}" = "1" ]; then
        if [ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ]; then
            OUTPUT_DIR_BASE="$SCRIPT_DIR/../rl_experiments/nomnom_model_rl_comm"
        else
            OUTPUT_DIR_BASE="$SCRIPT_DIR/../rl_experiments/nomnom_model_rl"
        fi
    else
        OUTPUT_DIR_BASE="$SCRIPT_DIR/../nomnom_model"
    fi
fi

echo "Using model type: ${USE_LINEAR_MODEL} (0=nomnom_model, 1=linear_model)"
echo "RL enabled: ${ENABLE_RL}"
echo "Communication enabled: ${ENABLE_COMMUNICATION}"
echo "Output directory: ${OUTPUT_DIR_BASE}"

# Clear visual_output directory before starting ablation
clear_visual_output
clear_folder "$OUTPUT_DIR_BASE"

# Function to run a single experiment (train + visualize)
run_experiment() {
    local param_value="$1"
    
    # Create exp_id with RL suffix if using RL
    if [ "${ENABLE_RL}" = "true" ] || [ "${ENABLE_RL}" = "1" ]; then
        if [ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ]; then
            local exp_id="${PARAM_NAME}_${param_value}_rl_comm"
        else
            local exp_id="${PARAM_NAME}_${param_value}_rl"
        fi
    else
        local exp_id="${PARAM_NAME}_${param_value}"
    fi
    
    local log_file="$RAW_DIR/train_nomnom_${exp_id}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting experiment: $PARAM_NAME=$param_value (RL=$ENABLE_RL, Comm=$ENABLE_COMMUNICATION, PID: $$)"
    
    # Build training command with all parameters.
    # NOTE: We do NOT pass enable_rl as a CLI arg because train_nomnom.sh
    # uses ENABLE_RL (env var) only to select the RL script. Passing
    # enable_rl=... would be misinterpreted as an env_params-* argument.
    local train_cmd="USE_LINEAR_MODEL=${USE_LINEAR_MODEL} ENABLE_RL=${ENABLE_RL} ENABLE_COMMUNICATION=${ENABLE_COMMUNICATION}"
    train_cmd="${train_cmd} $SCRIPT_DIR/train_nomnom.sh exp_id=${exp_id} use_linear_model=${USE_LINEAR_MODEL} ${PARAM_NAME}=${param_value}"
    
    # Add RL-specific hyperparameters if RL is enabled
    if [ "${ENABLE_RL}" = "true" ] || [ "${ENABLE_RL}" = "1" ]; then
        train_cmd="${train_cmd} rl_learning_rate=${RL_LEARNING_RATE}"
        train_cmd="${train_cmd} rl_buffer_size=${RL_BUFFER_SIZE}"
        train_cmd="${train_cmd} rl_adapt_frequency=${RL_ADAPT_FREQUENCY}"
        train_cmd="${train_cmd} survival_reward_weight=${SURVIVAL_REWARD_WEIGHT}"
        train_cmd="${train_cmd} energy_reward_weight=${ENERGY_REWARD_WEIGHT}"
        
        if [ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ]; then
            train_cmd="${train_cmd} enable_communication=True"
            train_cmd="${train_cmd} social_reward_weight=${SOCIAL_REWARD_WEIGHT}"
            train_cmd="${train_cmd} communication_radius=${COMMUNICATION_RADIUS}"
        fi
    fi
    
    # Execute training command
    eval "$train_cmd" >> "$log_file" 2>&1
    
    # Determine the output directory for this experiment
    local exp_output_dir="${OUTPUT_DIR_BASE}/${exp_id}"
    
    # Visualize the results (use param_name and value for subdirectory)
    # Pass the correct output directory based on model type and RL settings
    CAPTURE_VIDEO=true VISUAL_OUTPUT_SUBDIR=${exp_id} $SCRIPT_DIR/visualize_nomnom.sh "${exp_output_dir}" >> "$log_file" 2>&1
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed experiment: $PARAM_NAME=$param_value"
}

# Function to wait for a job slot to become available
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge "$PARALLELISM" ]; do
        sleep 0.5
    done
}

# Start timer
SECONDS=0

# Loop through each parameter value and run in parallel
for param_value in "${PARAM_VALUES[@]}"; do
    # Wait if we've reached the parallelism limit
    wait_for_slot
    
    # Start experiment in background
    run_experiment "$param_value" &
done

# Wait for all background jobs to complete
echo "Waiting for all experiments to complete..."
wait

echo ""
echo "✅ All experiments completed. Total time elapsed: $((SECONDS / 60)) minutes $((SECONDS % 60)) seconds"
echo "Results saved to: ${OUTPUT_DIR_BASE}"
if [ "${ENABLE_RL}" = "true" ] || [ "${ENABLE_RL}" = "1" ]; then
    echo "RL training mode: enabled"
    echo "  Learning rate: ${RL_LEARNING_RATE}"
    echo "  Buffer size: ${RL_BUFFER_SIZE}"
    if [ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ]; then
        echo "  Communication: enabled (radius=${COMMUNICATION_RADIUS})"
    fi
else
    echo "Training mode: evolution only (standard)"
fi