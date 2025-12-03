#!/bin/bash
# Visualization script for NomNom training runs (CPU by default)
# Usage:
#   Headless (Xvfb, Slurm/batch, default):
#     ./visualize_nomnom.sh
#     ./visualize_nomnom.sh /path/to/output_directory
#     OUTPUT_DIR=/path/to/output_directory ./visualize_nomnom.sh
#     VISUAL_OUTPUT_SUBDIR=my_subdir ./visualize_nomnom.sh
#     RELATIVE_DIR=some/path ./visualize_nomnom.sh
#     CAPTURE_VIDEO=true ./visualize_nomnom.sh
#   Interactive (X11 forwarding / real display):
#     MODE=interactive ./visualize_nomnom.sh
#     MODE=interactive OUTPUT_DIR=/path/to/output_directory ./visualize_nomnom.sh
#     MODE=interactive VISUAL_OUTPUT_SUBDIR=my_subdir ./visualize_nomnom.sh
#     MODE=interactive RELATIVE_DIR=some/path ./visualize_nomnom.sh
#     MODE=interactive CAPTURE_VIDEO=true ./visualize_nomnom.sh
#
# In both modes, the output directory must contain:
#   - train_params.state
#   - one or more report_XXXXXXXX.state files
#
# VISUAL_OUTPUT_SUBDIR: Optional subdirectory name within visual_output for saving frames.
#                       If not provided, defaults to the basename of output_directory.
# RELATIVE_DIR: Optional relative or absolute path to change working directory before running.
#               If not provided, defaults to the scripts directory.
# CAPTURE_VIDEO: If set to "true", converts PNG frames to video after visualization (default: false).

# Load Mambaforge module (required for mamba/conda to work)
module load Mambaforge/23.11.0-fasrc01

# Initialize conda for bash scripting
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate dirt

# Verify environment is actually activated and Python path is correct
PYTHON_PATH=$(which python)

# Default to CPU for visualization (can be overridden by user before calling script)
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

# Verify mechagogue can be imported (required for data loading)
python -c "from mechagogue.ecology.natural_selection import natural_selection; from mechagogue.breed.normal import normal_mutate; from mechagogue.static import static_data; from mechagogue.serial import save_leaf_data, load_example_data" 2>/dev/null || {
    echo "✗ ERROR: Failed to import mechagogue modules. Please check that mechagogue is installed in the 'dirt' environment."
    exit 1
}

echo ""
echo "=== NomNom Visualization ==="
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo ""

# Resolve script directory so we can call visualize_nomnom.py reliably
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
# Calculate path to visualize_nomnom.py (in dirt/dirt/examples/nomnom/ relative to scripts/)
VISUALIZE_SCRIPT="${SCRIPT_DIR}/../dirt/examples/nomnom/visualize_nomnom.py"
# Path to convert_video.sh script
CONVERT_VIDEO_SCRIPT="${SCRIPT_DIR}/convert_video.sh"

# Allow RELATIVE_DIR to change working directory before running
# Usage: RELATIVE_DIR=some/path ./visualize_nomnom.sh
if [ -n "${RELATIVE_DIR}" ]; then
    cd "${RELATIVE_DIR}"
else
    cd "${SCRIPT_DIR}"
fi

# Allow overrides via CLI argument or environment variable
# Precedence: CLI arg > OUTPUT_DIR env var > default
OUTPUT_DIR_DEFAULT="/n/netscratch/kdbrantley_lab/Lab/erassou/dirt/linear_model"

if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
fi

echo "Using output directory:"
echo "  ${OUTPUT_DIR}"
echo ""

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "✗ ERROR: Output directory does not exist: ${OUTPUT_DIR}"
    exit 1
fi

# Check for train_params.state - first in OUTPUT_DIR, then in subdirectories.
# New structure: train_params.state is stored inside per-experiment folders
# under the chosen output root (e.g. linear_model/{exp_id}/ for evolution-only
# runs, or rl_experiments/linear_model_rl/{exp_id}/ for RL runs).
TRAIN_PARAMS_PATH="${OUTPUT_DIR}/train_params.state"
if [ ! -f "${TRAIN_PARAMS_PATH}" ]; then
    # Look for train_params.state in subdirectories (new structure:
    # */{exp_id}/train_params.state, supporting both evolution-only and RL layouts)
    echo "train_params.state not found in ${OUTPUT_DIR}, checking subdirectories..."
    FOUND_PARAMS=""
    # Check if there are any subdirectories
    if [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; then
        for subdir in "${OUTPUT_DIR}"/*; do
            if [ -d "${subdir}" ] && [ -f "${subdir}/train_params.state" ]; then
                TRAIN_PARAMS_PATH="${subdir}/train_params.state"
                OUTPUT_DIR="${subdir}"  # Update OUTPUT_DIR to the subdirectory
                FOUND_PARAMS="true"
                echo "Found train_params.state in subdirectory: ${OUTPUT_DIR}"
                break
            fi
        done
    fi
    if [ -z "${FOUND_PARAMS}" ]; then
        echo "✗ ERROR: train_params.state not found in ${OUTPUT_DIR} or its subdirectories"
        echo "  Expected location: ${OUTPUT_DIR}/train_params.state"
        echo "  Or in subdirectory: ${OUTPUT_DIR}/*/train_params.state"
        exit 1
    fi
fi

REPORT_COUNT=$(ls "${OUTPUT_DIR}"/report_*.state 2>/dev/null | wc -l)
if [ "${REPORT_COUNT}" -eq 0 ]; then
    echo "✗ ERROR: No report_XXXXXXXX.state files found in ${OUTPUT_DIR}"
    exit 1
fi
echo ""

# MODE selection:
#   - MODE=interactive : assume DISPLAY is a real X server (e.g., via ssh -X / Slurm --x11)
#                        and run with effectively unlimited frames for interactive use.
#   - default (headless): start Xvfb if DISPLAY is not set and run for a finite number
#                         of frames in auto_step mode (headless/Slurm-friendly).
MODE="${MODE:-headless}"

# Allow visual_output_subdir to be set via environment variable
# Usage: VISUAL_OUTPUT_SUBDIR=my_subdir ./visualize_nomnom.sh
VISUAL_OUTPUT_SUBDIR_ARGS=()
if [ -n "${VISUAL_OUTPUT_SUBDIR}" ]; then
    VISUAL_OUTPUT_SUBDIR_ARGS=("--visual_output_subdir" "${VISUAL_OUTPUT_SUBDIR}")
fi

if [ "${MODE}" = "interactive" ]; then
    echo "Running in INTERACTIVE mode (no Xvfb; relying on existing DISPLAY=${DISPLAY:-unset})"
    if [ -z "${DISPLAY}" ]; then
        echo "✗ ERROR: MODE=interactive but DISPLAY is not set."
        echo "  Make sure you are using X11 forwarding or a real display (e.g., ssh -X, salloc --x11)."
        exit 1
    fi
    python "${VISUALIZE_SCRIPT}" "${OUTPUT_DIR}" "${VISUAL_OUTPUT_SUBDIR_ARGS[@]}" > raw/visualize_nomnom.log 2>&1
else
    echo "Running in HEADLESS mode (Xvfb + auto_step)."
    # For headless mode, let visualize_nomnom.py decide about Xvfb and max_frames.
    python "${VISUALIZE_SCRIPT}" "${OUTPUT_DIR}" --backend matplotlib --step_stride 128 "${VISUAL_OUTPUT_SUBDIR_ARGS[@]}"
fi

# Determine visualization result status
VISUALIZATION_EXIT_CODE=$?

# Handle video capture if requested
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"
if [ "${CAPTURE_VIDEO}" = "true" ]; then
    echo ""
    echo "=== Converting frames to video ==="
    
    # Determine the folder suffix (subdirectory name in visual_output)
    if [ -n "${VISUAL_OUTPUT_SUBDIR}" ]; then
        FOLDER_SUFFIX="${VISUAL_OUTPUT_SUBDIR}"
    else
        # Use basename of OUTPUT_DIR as default (matching visualize_nomnom.py behavior)
        FOLDER_SUFFIX=$(basename "$(realpath "${OUTPUT_DIR}")")
    fi
    
    echo "Converting frames from folder: ${FOLDER_SUFFIX}"
    
    # Check if convert_video.sh exists
    if [ ! -f "${CONVERT_VIDEO_SCRIPT}" ]; then
        echo "✗ WARNING: convert_video.sh not found at ${CONVERT_VIDEO_SCRIPT}"
        echo "  Skipping video conversion."
    else
        # Call convert_video.sh with the folder suffix
        "${CONVERT_VIDEO_SCRIPT}" "${FOLDER_SUFFIX}"
        VIDEO_CONVERSION_EXIT_CODE=$?
        
        if [ "${VIDEO_CONVERSION_EXIT_CODE}" -eq 0 ]; then
            echo "✓ Video conversion completed successfully"
        else
            echo "✗ WARNING: Video conversion failed (exit code: ${VIDEO_CONVERSION_EXIT_CODE})"
        fi
    fi
fi

# Exit with the visualization exit code (preserve original exit status)
exit ${VISUALIZATION_EXIT_CODE}
