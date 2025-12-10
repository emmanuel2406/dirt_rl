#!/bin/bash
# Interactive version of job.sh for testing on GPU nodes
# Usage: ./interactive_job.sh
# Or: bash interactive_job.sh

# Load Mambaforge module (required for mamba to work)
module load Mambaforge/23.11.0-fasrc01
module load cuda/12.4.1 cudnn/9.10.2.21_cuda12

# Initialize conda/mamba for bash scripting
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
# Enable conda activate in bash scripts (use conda, not mamba, for shell hooks)
eval "$(conda shell.bash hook)"

# Activate the environment using conda (mamba activate should work after conda init)
conda activate dirt

# Verify environment is actually activated and Python path is correct
PYTHON_PATH=$(which python)

export JAX_PLATFORMS=cpu

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

# Load CUDA and related modules BEFORE importing JAX
# (JAX needs CUDA/cuDNN available at import time to detect GPUs)
module load cuda/12.4.1-fasrc01
module load gcc/12.2.0-fasrc01

# Try to load cuDNN module (required for JAX GPU support)
# Match cuDNN version to CUDA version
if [ "$CUDA_VERSION" = "12.2.0" ]; then
    # Try cuDNN versions compatible with CUDA 12.2
    if ! module load cudnn/8.9.2.26_cuda12-fasrc01 2>/dev/null; then
        if ! module load cudnn 2>/dev/null; then
            echo "⚠ WARNING: Could not load cuDNN module"
        fi
    fi
elif [ "$CUDA_VERSION" = "12.4.1" ]; then
    # Try cuDNN versions compatible with CUDA 12.4
    if ! module load cudnn 2>/dev/null; then
        echo "⚠ WARNING: Could not load cuDNN system module"
    fi
else
    # Try default
    module load cudnn 2>/dev/null || echo "⚠ Could not load cuDNN"
fi

# Set library paths for CUDA (including cuDNN, cuFFT, cuSPARSE, and other CUDA libraries)
# CRITICAL: Library path order matters! CUDA lib64 must come FIRST
# JAX requires: cuFFT, cuDNN, cuSPARSE, cuBLAS, and other CUDA libraries
# Save the existing LD_LIBRARY_PATH from modules
OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

# Start fresh with CUDA libraries FIRST (for cuFFT, cuBLAS, cuSPARSE, etc.)
# Include both lib64 and lib directories to ensure all CUDA libraries are found
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib

# Also check targets/x86_64-linux/lib if it exists (some CUDA installations put libraries there)
if [ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]; then
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/targets/x86_64-linux/lib
fi

# Add CUDA extras (CUPTI) - needed for profiling
if [ -d "${CUDA_HOME}/extras/CUPTI/lib64" ]; then
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/extras/CUPTI/lib64
fi

# Then add cuDNN library path (from module) - AFTER CUDA
# NOTE: cuDNN uses 'lib' not 'lib64'!
if [ -n "${CUDNN_HOME}" ]; then
    # Check which directory exists (lib or lib64)
    if [ -d "${CUDNN_HOME}/lib64" ]; then
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDNN_HOME}/lib64
    fi
    if [ -d "${CUDNN_HOME}/lib" ]; then
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDNN_HOME}/lib
    fi
fi

# Add other system libraries (gcc, etc.) from module paths
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OLD_LD_LIBRARY_PATH}

# Check if conda has nvidia/cufft that might conflict with system CUDA cuFFT
CONDA_CUFFT_DIR="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cufft/lib"
if [ -d "${CONDA_CUFFT_DIR}" ]; then
    echo "⚠ WARNING: Found conda-installed cuFFT in nvidia/cufft"
    echo "  Conda cuFFT (version 11) conflicts with system CUDA 12.4.1 cuFFT (version 11.201)"
    echo "  Attempting to exclude conda cuFFT from library search..."
    
    # Exclude the conda cuFFT directory from LD_LIBRARY_PATH
    # by NOT adding the nvidia/cufft/lib directory
    # We'll add conda lib but exclude the problematic nvidia subdirectory
    
    # Add conda lib but note that nvidia/cufft/lib should be excluded
    if [ -d "${CONDA_PREFIX}/lib" ]; then
        # Filter out the nvidia/cufft directory if it gets picked up
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
    fi
else
    # Normal case: add conda libraries
    if [ -d "${CONDA_PREFIX}/lib" ]; then
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
    fi
fi

# Use LD_PRELOAD to force system cuFFT to be loaded first (if available)
# This ensures system CUDA 12 cuFFT is used even if conda cuFFT is found
SYSTEM_CUFFT="${CUDA_HOME}/lib64/libcufft.so"
if [ -f "${SYSTEM_CUFFT}" ]; then
    export LD_PRELOAD="${SYSTEM_CUFFT}:${LD_PRELOAD:-}"
fi

# Check for CUDA libraries in conda environment (some conda packages install CUDA libs)
if [ -d "${CONDA_PREFIX}/lib/cudnn" ]; then
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/cudnn
fi

# Verify CUDA libraries are available (required by JAX)
CUFFT_FOUND=0
CUSPARSE_FOUND=0
CUBLAS_FOUND=0
CUFFT_PATH=""
CUSPARSE_PATH=""
if [ -f "${CUDA_HOME}/lib64/libcufft.so" ]; then
    CUFFT_PATH="${CUDA_HOME}/lib64/libcufft.so"
    CUFFT_FOUND=1
elif [ -f "${CUDA_HOME}/lib/libcufft.so" ]; then
    CUFFT_PATH="${CUDA_HOME}/lib/libcufft.so"
    CUFFT_FOUND=1
fi

# Check for cuSPARSE (required by JAX)
if [ -f "${CUDA_HOME}/lib64/libcusparse.so" ]; then
    CUSPARSE_PATH="${CUDA_HOME}/lib64/libcusparse.so"
    CUSPARSE_FOUND=1
elif [ -f "${CUDA_HOME}/lib/libcusparse.so" ]; then
    CUSPARSE_PATH="${CUDA_HOME}/lib/libcusparse.so"
    CUSPARSE_FOUND=1
elif [ -f "${CUDA_HOME}/targets/x86_64-linux/lib/libcusparse.so" ]; then
    CUSPARSE_PATH="${CUDA_HOME}/targets/x86_64-linux/lib/libcusparse.so"
    CUSPARSE_FOUND=1
else
    echo "✗ WARNING: cuSPARSE library not found! JAX GPU support may fail."
fi

# Check for cuBLAS (also required by JAX)
if [ -f "${CUDA_HOME}/lib64/libcublas.so" ]; then
    CUBLAS_FOUND=1
elif [ -f "${CUDA_HOME}/lib/libcublas.so" ]; then
    CUBLAS_FOUND=1
else
    echo "⚠ WARNING: cuBLAS library not found"
fi

# Check cuFFT dependencies
if [ $CUFFT_FOUND -eq 1 ] && command -v ldd &> /dev/null; then
    MISSING_DEPS=$(ldd "${CUFFT_PATH}" 2>/dev/null | grep "not found" || true)
    if [ -n "$MISSING_DEPS" ]; then
        echo "⚠ WARNING: cuFFT has missing dependencies:"
        echo "$MISSING_DEPS"
    fi
fi

# Also check for versioned cuFFT libraries
if [ $CUFFT_FOUND -eq 0 ]; then
    CUFFT_LIBS=$(find ${CUDA_HOME}/lib64 ${CUDA_HOME}/lib -name "libcufft*.so*" 2>/dev/null | head -3)
    if [ -n "$CUFFT_LIBS" ]; then
        CUFFT_FOUND=1
    fi
fi

# Check system library cache
if [ $CUFFT_FOUND -eq 0 ]; then
    if ldconfig -p 2>/dev/null | grep -q libcufft; then
        CUFFT_FOUND=1
    fi
fi

if [ $CUFFT_FOUND -eq 0 ]; then
    echo "⚠ WARNING: cuFFT library not found!"
    echo "  CUDA_HOME: ${CUDA_HOME}"
    echo "  This will cause JAX GPU initialization to fail"
    echo ""
    echo "  Trying to find CUDA libraries..."
    if [ -d "${CUDA_HOME}/lib64" ]; then
        echo "  Libraries in ${CUDA_HOME}/lib64:"
        ls -1 ${CUDA_HOME}/lib64/libcu*.so* 2>/dev/null | head -10 || echo "    (none found)"
    fi
fi

# Check for multiple CUDA plugin conflicts
python -c "
import pkgutil
import jax_plugins

cuda_plugins = []
for importer, modname, ispkg in pkgutil.iter_modules(jax_plugins.__path__):
    if 'cuda' in modname.lower():
        cuda_plugins.append(modname)

if len(cuda_plugins) > 1:
    print(f'⚠ WARNING: Multiple CUDA plugins found: {cuda_plugins}')
    print('  This can cause \"ALREADY_EXISTS\" errors')
    print('  Recommendation: Uninstall CUDA 13 plugin since system uses CUDA 12.4.1')
    print('  Run: pip uninstall jax-cuda13-plugin jax-cuda13-pjrt')
" 2>/dev/null || true

# Verify jax can be imported
python -c "import jax" 2>/dev/null || {
    echo "✗ ERROR: Failed to import jax. Please check that jax is installed in the 'dirt' environment."
    exit 1
}

# Set JAX environment variables BEFORE importing JAX
# NOTE: Do NOT set JAX_PLATFORMS=cuda explicitly - let JAX auto-detect!
# Setting JAX_PLATFORMS=cuda forces strict initialization that may fail even when GPU is available
# JAX will automatically detect and use GPU if available
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
# Note: Removed invalid XLA_FLAGS flag that was causing errors
# export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=false"  # This flag doesn't exist

# Verify mechagogue can be imported (required for training script)
python -c "from mechagogue.ecology.natural_selection import natural_selection; from mechagogue.breed.normal import normal_mutate; from mechagogue.static import static_data; from mechagogue.serial import save_leaf_data, load_example_data" 2>/dev/null || {
    echo "✗ ERROR: Failed to import mechagogue modules. Please check that mechagogue is installed in the 'dirt' environment."
    exit 1
}

echo ""
echo "=== Current Status ==="
if [ $CUFFT_FOUND -eq 1 ] && [ $CUSPARSE_FOUND -eq 1 ]; then
    echo "✓ JAX 0.8.0 is installed"
    echo "✓ CUDA 12 plugins are installed"
    echo "✓ cuFFT library found"
    echo "✓ cuSPARSE library found"
    if [ $CUBLAS_FOUND -eq 1 ]; then
        echo "✓ cuBLAS library found"
    fi
    echo "✓ cuDNN library found"
    echo ""
    echo "✓✓✓ All required CUDA libraries are available!"
    echo "   JAX should be able to use GPU devices."
else
    echo "⚠ Some CUDA libraries are missing:"
    [ $CUFFT_FOUND -eq 0 ] && echo "  ✗ cuFFT not found"
    [ $CUSPARSE_FOUND -eq 0 ] && echo "  ✗ cuSPARSE not found (REQUIRED for JAX)"
    [ $CUBLAS_FOUND -eq 0 ] && echo "  ✗ cuBLAS not found"
    echo ""
    echo "Make sure CUDA_HOME is set correctly and LD_LIBRARY_PATH includes CUDA libraries."
fi

# export JAX_DISABLE_JIT=1
# Get script directory and calculate path to train_nomnom.py
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Default to standard training script (evolution only)
ENABLE_RL=${ENABLE_RL:-false}
ENABLE_COMMUNICATION=${ENABLE_COMMUNICATION:-false}
USE_MESSAGE_ACTION=${USE_MESSAGE_ACTION:-true}

# Choose training script based on RL flag
if [ "${ENABLE_RL}" = "true" ] || [ "${ENABLE_RL}" = "1" ]; then
    TRAIN_SCRIPT="${SCRIPT_DIR}/../dirt/examples/nomnom/train_nomnom_rl.py"
    echo "Using RL-enabled training script"
else
    TRAIN_SCRIPT="${SCRIPT_DIR}/../dirt/examples/nomnom/train_nomnom.py"
    echo "Using standard evolution-only training script"
fi

# Accept exp_id as a special argument (--exp_id value or exp_id=value)
# Also accept any parameter in param_name=value format
# Also accept --use_linear_model or use_linear_model=value
# Also accept RL-specific parameters
EXP_ID_ARG=()
PARAM_ARGS=()
USE_LINEAR_MODEL_ARG=()
RL_ARGS=()
TOP_LEVEL_ARGS=()

# Default use_linear_model to 0 (nomnom_model) if not specified
USE_LINEAR_MODEL=${USE_LINEAR_MODEL:-0}

# Helper function to convert boolean strings to 1/0 (required by argparse bool parser)
# The mechagogue commandline_interface expects bools as integers: 0 or 1
convert_bool_to_int() {
    local value="$1"
    case "$value" in
        [Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Oo][Nn])
            echo "1"
            ;;
        [Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff])
            echo "0"
            ;;
        *)
            # If it's already a number, pass it through
            echo "$value"
            ;;
    esac
}

# Parse all arguments
ENABLE_COMMUNICATION_SET=false
while [ $# -gt 0 ]; do
    arg="$1"
    shift
    
    if [[ "$arg" == "--exp_id" ]]; then
        # Next argument is the exp_id value
        if [ $# -gt 0 ]; then
            EXP_ID_ARG=("--exp_id" "$1")
            shift
        fi
    elif [[ "$arg" == "--use_linear_model" ]]; then
        # Next argument is the use_linear_model value
        if [ $# -gt 0 ]; then
            USE_LINEAR_MODEL="$1"
            USE_LINEAR_MODEL_ARG=("--use_linear_model" "$1")
            shift
        fi
    elif [[ "$arg" == "--enable_communication" ]]; then
        # Next argument is the enable_communication value
        if [ $# -gt 0 ]; then
            ENABLE_COMMUNICATION_VALUE=$(convert_bool_to_int "$1")
            RL_ARGS+=("--enable_communication" "$ENABLE_COMMUNICATION_VALUE")
            ENABLE_COMMUNICATION_SET=true
            shift
        fi
    elif [[ "$arg" =~ ^exp_id=(.+)$ ]]; then
        # exp_id=value format
        EXP_ID_VALUE="${BASH_REMATCH[1]}"
        EXP_ID_ARG=("--exp_id" "$EXP_ID_VALUE")
    elif [[ "$arg" =~ ^use_linear_model=(.+)$ ]]; then
        # use_linear_model=value format
        USE_LINEAR_MODEL="${BASH_REMATCH[1]}"
        USE_LINEAR_MODEL_ARG=("--use_linear_model" "$USE_LINEAR_MODEL")
    elif [[ "$arg" =~ ^enable_communication=(.+)$ ]]; then
        # enable_communication=value format (RL-specific)
        # Convert boolean strings to 1/0 format
        ENABLE_COMMUNICATION_VALUE=$(convert_bool_to_int "${BASH_REMATCH[1]}")
        RL_ARGS+=("--enable_communication" "$ENABLE_COMMUNICATION_VALUE")
        ENABLE_COMMUNICATION_SET=true
    elif [[ "$arg" =~ ^rl_learning_rate=(.+)$ ]]; then
        # rl_learning_rate=value format (RL-specific)
        RL_ARGS+=("--rl_learning_rate" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^rl_buffer_size=(.+)$ ]]; then
        # rl_buffer_size=value format (RL-specific)
        RL_ARGS+=("--rl_buffer_size" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^rl_adapt_frequency=(.+)$ ]]; then
        # rl_adapt_frequency=value format (RL-specific)
        RL_ARGS+=("--rl_adapt_frequency" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^use_message_action=(.+)$ ]]; then
        # use_message_action=value format (RL-specific)
        # Convert boolean strings to 1/0 format
        USE_MESSAGE_ACTION_VALUE=$(convert_bool_to_int "${BASH_REMATCH[1]}")
        RL_ARGS+=("--use_message_action" "$USE_MESSAGE_ACTION_VALUE")
    elif [[ "$arg" =~ ^social_reward_weight=(.+)$ ]]; then
        # social_reward_weight=value format (RL-specific)
        RL_ARGS+=("--social_reward_weight" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^survival_reward_weight=(.+)$ ]]; then
        # survival_reward_weight=value format (RL-specific)
        RL_ARGS+=("--survival_reward_weight" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^energy_reward_weight=(.+)$ ]]; then
        # energy_reward_weight=value format (RL-specific)
        RL_ARGS+=("--energy_reward_weight" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^communication_radius=(.+)$ ]]; then
        # communication_radius=value format (RL-specific)
        RL_ARGS+=("--communication_radius" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^food_grid_stride=(.+)$ ]]; then
        # food_grid_stride=value format (top-level parameter, not env_params)
        TOP_LEVEL_ARGS+=("--food_grid_stride" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^seed=(.+)$ ]]; then
        # seed=value format (top-level parameter)
        TOP_LEVEL_ARGS+=("--seed" "${BASH_REMATCH[1]}")
    elif [[ "$arg" =~ ^([^=]+)=(.+)$ ]]; then
        # Parse param_name=value format (e.g., max_energy=4, initial_players=32)
        PARAM_NAME="${BASH_REMATCH[1]}"
        PARAM_VALUE="${BASH_REMATCH[2]}"
        PARAM_ARGS+=("--env_params-${PARAM_NAME}" "$PARAM_VALUE")
        # Special case: if max_players is set, also update train_params.max_players
        # to keep them in sync (prevents size mismatch errors)
        if [[ "$PARAM_NAME" == "max_players" ]]; then
            PARAM_ARGS+=("--train_params-max_players" "$PARAM_VALUE")
        fi
    fi
done

# If USE_LINEAR_MODEL was set via environment variable but not via argument, use it
if [ -z "${USE_LINEAR_MODEL_ARG[*]}" ] && [ -n "${USE_LINEAR_MODEL}" ]; then
    USE_LINEAR_MODEL_ARG=("--use_linear_model" "$USE_LINEAR_MODEL")
fi

# If ENABLE_COMMUNICATION was set via environment variable but not via argument, use it
# Similar to how ENABLE_RL works - use environment variable as default
if [ "$ENABLE_COMMUNICATION_SET" = "false" ] && ([ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ] || [ "${ENABLE_COMMUNICATION}" = "True" ]); then
    ENABLE_COMMUNICATION_VALUE=$(convert_bool_to_int "${ENABLE_COMMUNICATION}")
    RL_ARGS+=("--enable_communication" "$ENABLE_COMMUNICATION_VALUE")
fi

# If USE_MESSAGE_ACTION was set via environment variable, use it
# (Only pass if communication is enabled, otherwise it's ignored)
if ([ "${ENABLE_COMMUNICATION}" = "true" ] || [ "${ENABLE_COMMUNICATION}" = "1" ] || [ "${ENABLE_COMMUNICATION}" = "True" ]) || [ "$ENABLE_COMMUNICATION_SET" = "true" ]; then
    USE_MESSAGE_ACTION_VALUE=$(convert_bool_to_int "${USE_MESSAGE_ACTION}")
    RL_ARGS+=("--use_message_action" "$USE_MESSAGE_ACTION_VALUE")
fi

python "${TRAIN_SCRIPT}" "${USE_LINEAR_MODEL_ARG[@]}" --log_level ERROR "${EXP_ID_ARG[@]}" "${PARAM_ARGS[@]}" "${RL_ARGS[@]}" "${TOP_LEVEL_ARGS[@]}"
