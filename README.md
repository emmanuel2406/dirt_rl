# DIRT-rl: A DIRT fork for RL ablations
![Fig1](https://github.com/aaronwalsman/dirt/blob/main/images/fig1.png?raw=true)
_Simulator Overview: This map contains 1024 × 1024
grid cells and 10,000 individual agents, which are too small to see
when fully zoomed out. The inset areas progressively zoom in on
one patch of terrain to reveal detail._

## Overview
DIRT-rl is a fork of DIRT: The Distributed Intelligent Replicator Toolkit, focused on running RL-enabled ablations in the NomNom environment. It retains the original simulator while adding scripts to reproduce our RL experiments.

## Setup

Clone this fork:
```
git clone https://github.com/emmanuel2406/dirt_rl.git
```

Create a conda env with Python 3.11:
```
conda create --name dirt python=3.11 pip
```

Activate the environment and install `dirt` dependencies:
```
cd dirt
conda activate dirt
pip install --upgrade pip
pip install -e .
```

Keep the conda env active for the remainder of setup and development.

Clone the `macos` branch of our 3D renderer, `splendor-render`, and install the package:
```
cd ..
git clone --branch macos https://github.com/aaronwalsman/splendor-render.git
cd splendor-render
pip install -e .
```

Now, run
```
splendor_asset_installer
```

Next, clone `mechagogue` and install the package:
```
cd ..
git clone https://github.com/aaronwalsman/mechagogue.git
cd mechagogue
pip install -e .
```

Ensure you have a CUDA-compatible GPU available to JAX.

Test your setup with the `landscape` simulation example (the output is execution time):
```
cd ../dirt/dirt/examples/landscape
python landscape.py
```

## Reproducibility (NomNom ablations)
The RL ablation and visualization pipeline is driven by `scripts/ablate_nomnom.sh`.

1) Ensure the environment is activated and from the repo root run:
```
cd scripts
```

2) Launch an ablation sweep over a parameter (example: varying `max_energy`):
```
./ablate_nomnom.sh max_energy 4 6 8 10 12
```

Key options (set as environment variables when calling the script):
- `PARALLELISM`: max concurrent experiments (default 4).  
- `USE_LINEAR_MODEL=0|1`: choose `nomnom_model` (0) or `linear_model` (1, default).  
- `ENABLE_RL=true|false`: toggle RL training; `ENABLE_COMMUNICATION=true` adds message passing.  
- `ALL=1`: run vanilla, RL-only, and RL+communication variants for each value.  
- `TRIES=N`: repeat each config with seeds `SEED..SEED+N-1` (default `SEED=1234`).  
- RL hyperparameters: `RL_LEARNING_RATE`, `RL_BUFFER_SIZE`, `RL_ADAPT_FREQUENCY`, `SURVIVAL_REWARD_WEIGHT`, `ENERGY_REWARD_WEIGHT`, `SOCIAL_REWARD_WEIGHT`, `COMMUNICATION_RADIUS`.  
- Visualization stride: `STEP_STRIDE` (default 128; passed through to training/visualization).  
- Messaging toggle: `USE_MESSAGE_ACTION` when communication is enabled.

Outputs:
- Raw logs: `dirt/examples/nomnom/raw/train_nomnom_<exp_id>.log`.
- Experiment folders: under `linear_model`, `nomnom_model`, or `rl_experiments/*` depending on RL/communication choices.
- Aggregated plots: generated at the end via `dirt/examples/nomnom/aggregate_players_timeseries.py`; ablation runs are tagged with a random `ABLATE_ID`.

Checkpointing & cleanup:
- Completed experiments (presence of `players_time_series.json`) are skipped on reruns.
- `visual_output` is cleared before each ablation (except `visual_output/ablate_results`).
- Partial/incomplete runs in output directories are pruned; completed runs are preserved.



## Landscape (for DIRT)
### Rock
The rock layer makes up the baseline height of each grid cell in the system.  Rock is not modified, except by erosion, which is usually only used to initialize more complex starting terrain, but not used during simulation.  The rock is initialized using Perlin noise.  The units of rock are such that a rock value of is as tall as a single grid cell is wide.  It's resolution is determined by the `rock_downsample` parameter.

### Water
Water is initialized by filling all areas below a sea level parameter, and an additional `initial_water_per_cell` parameter.  Water uses the same units as rock, where one unit of water is as tall as one grid cell is wide.  The sum of the water and rock grids are used to compute the total altitude at each grid cell.  Water flows downhill.  Additionally, water sinks will drain water from certain grid cells and distribute it to water sources.  The locations of these water sinks and sources can be generated randomly using the `water_sink_density` and `water_source_density` parameters.  The weather system also modifies the water by evaporating it, which turns it into air moisture, and then raining it back down later.  Agents can also consume water as a resource.  In general, the water system was designed to be a closed system where the total volume of water remains constant throughout simulation, although numerical rounding issues may affect this, and have not been accounted for.  The resolution of the water is is determined by the `terrain_downsample` parameter.

### Light
The lighting model keeps track of a global light direction, and computes local light values using a dot-product with the altitude normals.  As an optimization detail, the normals are computed for the rock layer once at the beginning of simulation, as the rock values do not change, and any location with standing water is approximated to have flat normals.  This avoids recomputing normals at each time step.  If the weather system is active, moisture in the atmosphere and rain will reduce the ammount of light entering each grid cell.  The resolution of the light is determined by the `light_downsample` parameter.

### Weather
The weather system maintains a global __wind__ direction, as well as grids for __temperature__, __moisture__ and __rain__.

__Wind__ is modelled as an [OU](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) process that tracks a 2D direction.

__Temperature__ is increased by the light coming into each grid cell.  This is computed using an exponential moving average that uses a target temperature based on the incoming light, altitude, and presence of standing water.  Additionally the rate at which the temperature is modified by the incoming light depends on the presence of standing water, so that areas with water will change temperature more slowly.  The temperature values diffuse spatially and are offset by the wind.  The resolution of the temperature is determined by the `temperature_downsample` parameter.

__Moisture__ represents water that has been evaporated into the air.  In areas with high temperature, water is removed from the environment and added to the moisture grid.  When the moisture reaches a predefined threshold, which varies with altitude, it begins raining, and continues to do so until the moisture drops below a second threshold.  Moisture is diffused by spatially and offset by the wind.  The resolution is determined by the `rain_downsample` parameter.

__Rain__ is a binary grid representing locations that are currently raining.  Wherever it is raining, water is removed from the moisture and added to the surface water.  The resolution is determined by the `rain_downsample` parameter.

### Resources
In addition to water, the landscape also tracks __biomass__ and __energy__.  Both can be consumed by agents.  The biomass is designed to be mass preserving, while energy is not.  Biomass can produce energy via photosynthesis using the `biomass_photosynthesis` parameter.

### Smell and Audio
Smell and audio are two separate communication media that can be produced and sensed by agents.  Smell perists in the environment, is blown by the wind and slowly diffused, while audio does not persist, is not affected by the wind but is heavily diffused on a very coarse grid in order to propagate quickly.

## Implementation Details
Data in DIRT is associated either with grid cells (2D), individual agents (1D) or the global state of the system (0D).  The grid data can be stored and computed at different resolutions so that some effects, for example the temperature, can be simulated at a coarser resolution for efficiency purposes.  This is also used for functional purposes, such as in the audio system, which uses aggressive downsampling so that audio can be propagated over long distances using a relatively small filter.
