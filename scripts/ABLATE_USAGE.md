# Ablation Script Usage Guide

## Overview

The `ablate_nomnom.sh` script performs hyperparameter ablation studies by training and visualizing multiple experiments in parallel. It now supports both evolution-only training (original) and RL-enabled training (new).

## Basic Usage

```bash
./ablate_nomnom.sh <param_name> <value1> [value2] [value3] ...
```

## Training Modes

### Mode 1: Evolution Only (Default)

Standard evolutionary training using `train_nomnom.py`:

```bash
# Ablate over max_energy values
./ablate_nomnom.sh max_energy 4 6 8 10 12

# Ablate over food growth rates
./ablate_nomnom.sh mean_food_growth 2 4 8 16 32
```

### Mode 2: RL Adaptation Only

Enable within-lifetime learning (Stage 1):

```bash
# Enable RL for all experiments
ENABLE_RL=true ./ablate_nomnom.sh max_energy 4 6 8 10 12

# With custom RL hyperparameters
ENABLE_RL=true RL_LEARNING_RATE=1e-3 RL_BUFFER_SIZE=32 \
  ./ablate_nomnom.sh max_energy 4 6 8 10
```

### Mode 3: RL + Communication

Enable both RL and communication (Stage 1 + Stage 2):

```bash
# Enable both stages
ENABLE_RL=true ENABLE_COMMUNICATION=true \
  ./ablate_nomnom.sh max_energy 4 6 8 10

# With custom parameters
ENABLE_RL=true ENABLE_COMMUNICATION=true \
  RL_LEARNING_RATE=1e-4 SOCIAL_REWARD_WEIGHT=0.8 \
  COMMUNICATION_RADIUS=10 \
  ./ablate_nomnom.sh mean_food_growth 4 8 16
```

## Environment Variables

### Core Options

| Variable | Default | Description |
|----------|---------|-------------|
| `PARALLELISM` | `4` | Max concurrent experiments |
| `USE_LINEAR_MODEL` | `1` | Use linear model (1) or full model (0) |
| `ENABLE_RL` | `false` | Enable RL adaptation |
| `ENABLE_COMMUNICATION` | `false` | Enable agent communication |

### RL Hyperparameters

Only used when `ENABLE_RL=true`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RL_LEARNING_RATE` | `1e-4` | Learning rate for policy gradients |
| `RL_BUFFER_SIZE` | `16` | Experience buffer size |
| `RL_ADAPT_FREQUENCY` | `8` | Steps between RL updates |
| `SURVIVAL_REWARD_WEIGHT` | `1.0` | Reward for staying alive |
| `ENERGY_REWARD_WEIGHT` | `0.1` | Reward for energy changes |

### Communication Parameters

Only used when `ENABLE_COMMUNICATION=true`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOCIAL_REWARD_WEIGHT` | `0.5` | Weight for social rewards |
| `COMMUNICATION_RADIUS` | `5` | Grid cells for communication range |

## Common Use Cases

### 1. Compare Evolution vs RL

Run same ablation with and without RL:

```bash
# Evolution only
./ablate_nomnom.sh max_energy 4 6 8 10 12

# RL enabled
ENABLE_RL=true ./ablate_nomnom.sh max_energy 4 6 8 10 12

# Compare results in different output directories:
# - Evolution: ../linear_model/max_energy_*
# - RL: ../rl_experiments/linear_model_rl/max_energy_*_rl
```

### 2. Test Communication Impact

Compare RL with and without communication:

```bash
# RL only
ENABLE_RL=true ./ablate_nomnom.sh mean_food_growth 4 8 16

# RL + Communication
ENABLE_RL=true ENABLE_COMMUNICATION=true \
  ./ablate_nomnom.sh mean_food_growth 4 8 16

# Results:
# - RL only: ../rl_experiments/linear_model_rl/mean_food_growth_*_rl
# - RL + Comm: ../rl_experiments/linear_model_rl_comm/mean_food_growth_*_rl_comm
```

### 3. Parallel Processing

Control how many experiments run simultaneously:

```bash
# Run 8 experiments in parallel
PARALLELISM=8 ./ablate_nomnom.sh max_energy 2 4 6 8 10 12 14 16

# Single experiment at a time (serial)
PARALLELISM=1 ./ablate_nomnom.sh max_energy 4 6 8
```

### 4. Ablate RL Hyperparameters

Test different RL learning rates:

```bash
# Note: To ablate RL hyperparameters, you need to run the script multiple times
# with different environment variable values

# Learning rate 1e-3
ENABLE_RL=true RL_LEARNING_RATE=1e-3 \
  ./ablate_nomnom.sh max_energy 4 6 8

# Learning rate 1e-4
ENABLE_RL=true RL_LEARNING_RATE=1e-4 \
  ./ablate_nomnom.sh max_energy 4 6 8

# Learning rate 1e-5
ENABLE_RL=true RL_LEARNING_RATE=1e-5 \
  ./ablate_nomnom.sh max_energy 4 6 8
```

### 5. Test Social Reward Weights

```bash
# Low social reward
ENABLE_RL=true ENABLE_COMMUNICATION=true SOCIAL_REWARD_WEIGHT=0.1 \
  ./ablate_nomnom.sh mean_food_growth 4 8 16

# Medium social reward
ENABLE_RL=true ENABLE_COMMUNICATION=true SOCIAL_REWARD_WEIGHT=0.5 \
  ./ablate_nomnom.sh mean_food_growth 4 8 16

# High social reward
ENABLE_RL=true ENABLE_COMMUNICATION=true SOCIAL_REWARD_WEIGHT=1.0 \
  ./ablate_nomnom.sh mean_food_growth 4 8 16
```

## Output Organization

The script automatically organizes outputs based on training mode:

```
dirt/
├── linear_model/              # Evolution only
│   └── max_energy_4/
│       ├── train_state_*.state
│       └── report_*.state
│
├── rl_experiments/
│   ├── linear_model_rl/       # RL only
│   │   └── max_energy_4_rl/
│   │       ├── train_state_*.state
│   │       └── report_*.state
│   │
│   └── linear_model_rl_comm/  # RL + Communication
│       └── max_energy_4_rl_comm/
│           ├── train_state_*.state
│           └── report_*.state
│
└── examples/nomnom/
    ├── raw/                   # Training logs
    │   ├── train_nomnom_max_energy_4.log
    │   ├── train_nomnom_max_energy_4_rl.log
    │   └── train_nomnom_max_energy_4_rl_comm.log
    │
    └── visual_output/         # Videos and visualizations
        ├── max_energy_4/
        ├── max_energy_4_rl/
        └── max_energy_4_rl_comm/
```

## Complete Examples

### Example 1: Quick Test

Test a single parameter with 3 values:

```bash
# Evolution only
./ablate_nomnom.sh max_energy 8 12 16

# With RL
ENABLE_RL=true ./ablate_nomnom.sh max_energy 8 12 16
```

### Example 2: Comprehensive Ablation

Test multiple food growth rates with all three modes:

```bash
# 1. Evolution only (baseline)
./ablate_nomnom.sh mean_food_growth 2 4 8 16 32 64

# 2. RL adaptation
ENABLE_RL=true ./ablate_nomnom.sh mean_food_growth 2 4 8 16 32 64

# 3. RL + Communication
ENABLE_RL=true ENABLE_COMMUNICATION=true \
  ./ablate_nomnom.sh mean_food_growth 2 4 8 16 32 64
```

### Example 3: High-Performance Setup

Maximum parallelism with RL:

```bash
PARALLELISM=16 ENABLE_RL=true RL_BUFFER_SIZE=32 \
  ./ablate_nomnom.sh max_players 32 64 128 256 512
```

### Example 4: Communication Study

Focus on communication parameters:

```bash
# Test different communication radii
ENABLE_RL=true ENABLE_COMMUNICATION=true COMMUNICATION_RADIUS=3 \
  ./ablate_nomnom.sh max_energy 4 6 8
  
ENABLE_RL=true ENABLE_COMMUNICATION=true COMMUNICATION_RADIUS=5 \
  ./ablate_nomnom.sh max_energy 4 6 8
  
ENABLE_RL=true ENABLE_COMMUNICATION=true COMMUNICATION_RADIUS=10 \
  ./ablate_nomnom.sh max_energy 4 6 8
```

## Parameters You Can Ablate

Common parameters to sweep:

- `max_energy`: Maximum agent energy (4, 6, 8, 10, 12, 16)
- `mean_food_growth`: Average food spawn rate (2, 4, 8, 16, 32, 64)
- `max_food_growth`: Maximum food in cell (4, 16, 64, 256, 1024)
- `initial_players`: Starting population (8, 16, 32, 64, 128)
- `max_players`: Population capacity (64, 128, 256, 512, 1024)
- `world_size`: Grid dimensions (need special handling for tuples)
- `senescence`: Aging rate (0.0, 0.01, 0.02, 0.05, 0.1)
- `food_metabolism`: Energy from food (1, 2, 4, 8, 16)

## Tips and Best Practices

### 1. Start Small

Test with a few values first to ensure everything works:

```bash
# Small test
ENABLE_RL=true ./ablate_nomnom.sh max_energy 4 6 8

# Then scale up
ENABLE_RL=true PARALLELISM=8 \
  ./ablate_nomnom.sh max_energy 4 6 8 10 12 14 16 20
```

### 2. Monitor Progress

Check logs in real-time:

```bash
# In another terminal
tail -f dirt/examples/nomnom/raw/train_nomnom_max_energy_4_rl.log
```

### 3. Clear Old Results

The script automatically clears the output directory. To keep previous results:

```bash
# Move old results before running
mv linear_model linear_model_backup_$(date +%Y%m%d)
```

### 4. Combine with Model Types

```bash
# Test both linear and full model with RL
USE_LINEAR_MODEL=1 ENABLE_RL=true ./ablate_nomnom.sh max_energy 4 6 8
USE_LINEAR_MODEL=0 ENABLE_RL=true ./ablate_nomnom.sh max_energy 4 6 8
```

## Troubleshooting

### Issue: Experiments not starting

**Solution**: Check that train_nomnom.sh has execute permissions:
```bash
chmod +x dirt/scripts/train_nomnom.sh
chmod +x dirt/scripts/ablate_nomnom.sh
```

### Issue: RL not being used

**Solution**: Verify ENABLE_RL is set correctly:
```bash
ENABLE_RL=true ./ablate_nomnom.sh ...  # Correct
ENABLE_RL=1 ./ablate_nomnom.sh ...     # Also correct
ENABLE_RL=false ./ablate_nomnom.sh ... # Uses evolution only
```

### Issue: Out of memory

**Solution**: Reduce parallelism:
```bash
PARALLELISM=2 ./ablate_nomnom.sh ...
```

### Issue: Training crashes

**Solution**: Check the log files:
```bash
cat dirt/examples/nomnom/raw/train_nomnom_max_energy_4_rl.log
```

## Advanced: Custom Ablation Script

For more complex ablations (e.g., sweeping multiple parameters), create a wrapper:

```bash
#!/bin/bash
# custom_ablation.sh

# Sweep learning rates
for lr in 1e-3 1e-4 1e-5; do
    # Sweep buffer sizes
    for buf in 8 16 32; do
        echo "Testing lr=$lr, buffer=$buf"
        ENABLE_RL=true \
        RL_LEARNING_RATE=$lr \
        RL_BUFFER_SIZE=$buf \
        ./ablate_nomnom.sh max_energy 4 6 8
        
        # Wait between batches
        sleep 10
    done
done
```

## Comparison with Original

**Original behavior (ENABLE_RL=false)**:
- Uses `train_nomnom.py`
- Pure evolutionary training
- Saves to `linear_model/` or `nomnom_model/`

**New behavior (ENABLE_RL=true)**:
- Uses `train_nomnom_rl.py`
- RL + evolution hybrid
- Saves to `rl_experiments/linear_model_rl/`
- Additional RL parameters available

**Backward compatible**: Running without ENABLE_RL flag behaves exactly as before.

## See Also

- `QUICK_START.md` - Getting started with RL expansion
- `RL_EXPANSION_SUMMARY.md` - Complete implementation overview
- `mechagogue/DESIGN_RL_EXPANSION.md` - Technical design details
- `mechagogue/ecology/README_RL.md` - API reference

