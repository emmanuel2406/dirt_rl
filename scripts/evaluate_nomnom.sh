#!/bin/bash
# Evaluation script for NomNom checkpoints (CPU by default)
# Usage:
#   ./evaluate_nomnom.sh
#   or override defaults:
#   FOLDER=/path/to/checkpoints MAX_POP=512 TRIALS=5 STEPS=5 ./evaluate_nomnom.sh

# Load Mambaforge module (required for mamba/conda to work)
module load Mambaforge/23.11.0-fasrc01

# Initialize conda for bash scripting
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate dirt

# Verify environment is actually activated and Python path is correct
PYTHON_PATH=$(which python)

# Default to CPU for evaluation (can be overridden by user before calling script)
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}

# Check if we're actually in the dirt environment
if [ "$CONDA_DEFAULT_ENV" != "dirt" ] || ([[ "$PYTHON_PATH" != *"dirt"* ]] && [[ "$PYTHON_PATH" != *".conda"* ]]); then
    echo "WARNING: Environment may not be activated correctly."
    echo "Attempting to find and activate dirt environment..."
    
    # Try to find the dirt environment
    DIRT_ENV_PATH=$(conda env list | grep -E "^\s*dirt" | awk '{print $NF}' | head -1)
    if [ -n "$DIRT_ENV_PATH" ] && [ -d "$DIRT_ENV_PATH" ]; then
        # Activate using the full path
        source "$DIRT_ENV_PATH/bin/activate"
        conda activate dirt
    else
        echo "ERROR: Could not find 'dirt' environment."
        echo "Available environments:"
        conda env list
        echo ""
        echo "Please create or locate the 'dirt' environment first."
        exit 1
    fi
fi

# Double-check Python path after activation
PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" != *"dirt"* ]] && [[ "$PYTHON_PATH" != *".conda"* ]]; then
    echo "ERROR: Python is still pointing to base installation!"
    echo "This means the environment is not properly activated."
    echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    exit 1
fi

# Verify jax can be imported
python -c "import jax" 2>/dev/null || {
    echo "✗ ERROR: Failed to import jax. Please check that jax is installed in the 'dirt' environment."
    exit 1
}

# Verify mechagogue can be imported (required for evaluation)
python -c "from mechagogue.ecology.natural_selection import natural_selection; from mechagogue.breed.normal import normal_mutate; from mechagogue.static import static_data; from mechagogue.serial import save_leaf_data, load_example_data" 2>/dev/null || {
    echo "✗ ERROR: Failed to import mechagogue modules. Please check that mechagogue is installed in the 'dirt' environment."
    exit 1
}

echo ""
echo "=== NomNom Evaluation ==="
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo ""

# Allow overrides via environment variables, otherwise use README defaults
FOLDER_DEFAULT="/n/netscratch/kdbrantley_lab/Lab/erassou/dirt/linear_model"
FOLDER="${FOLDER:-$FOLDER_DEFAULT}"
MAX_POP="${MAX_POP:-512}"
TRIALS="${TRIALS:-5}"
STEPS="${STEPS:-5}"

echo "Using settings:"
echo "  Folder:           ${FOLDER}"
echo "  Max population:   ${MAX_POP}"
echo "  Trials per agent: ${TRIALS}"
echo "  Steps per trial:  ${STEPS}"
echo ""

python nomnom_evaluate.py \
  --folder "${FOLDER}" \
  --max_population "${MAX_POP}" \
  --trials_per_agent "${TRIALS}" \
  --steps_per_trial "${STEPS}" > raw/evaluate_nomnom.log 2>&1

echo "Evaluation complete. Last 10 lines of log:"
tail -n 10 raw/evaluate_nomnom.log