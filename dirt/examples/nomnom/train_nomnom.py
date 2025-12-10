'''
This example is designed to test the nomnom environment in a training loop
with a random policy.
'''
import time
import argparse
import os
import json
import numpy as np
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.ecology.natural_selection import (
    natural_selection, NaturalSelectionParams)
from mechagogue.breed.normal import normal_mutate
from mechagogue.static import static_data as static_dataclass
from mechagogue.commandline import commandline_interface
from mechagogue.serial import save_leaf_data, load_example_data

#from dirt.examples.nomnom.nomnom_env import nomnom, NomNomParams, NomNomAction
from dirt.envs.nomnom import nomnom, NomNomParams, NomNomAction
from dirt.examples.nomnom.nomnom_model import (
    NomNomModelParams, nomnom_model, nomnom_linear_model)
import matplotlib.pyplot as plt

@static_dataclass
@commandline_interface()
class NomNomTrainParams:
    max_players : int = 256
    use_linear_model : bool = True
    log_level : str = "INFO"
    exp_id : Optional[str] = None
    env_params : Any = NomNomParams(
        mean_initial_food=8**2,
        max_initial_food=32**2,
        mean_food_growth=2**2,
        max_food_growth=16**2,
        initial_players=32,
        max_players=max_players,
        world_size=(32,32)
    )
    train_params : Any = NaturalSelectionParams(
        max_players=max_players,
    )
    epochs : int = 100
    steps_per_epoch : int = 1000
    output_directory : str = '/n/netscratch/kdbrantley_lab/Lab/erassou/dirt'
    load_from_file : Optional[str] = None
    # Memory optimization: store food_grid only every Nth step (for visualization)
    # Set to 1 to store every step, or higher to save memory
    # This should match the step_stride used in visualize_nomnom.py
    food_grid_stride : int = 128  # Store food_grid every Nth step
    # Random seed
    seed : int = 1234

@static_dataclass
class NomNomReport:
    actions : Any = NomNomAction(0,0,0)
    players : jnp.array = False
    player_x : jnp.array = False
    player_r : jnp.array = False
    player_energy : jnp.array = False
    food_grid : jnp.array = False
    family_tree_parents : jnp.array = False
    family_tree_birthdays : jnp.array = False  # Birth step for each agent
    family_tree_current_time : int = 0  # Current step/time

def make_report(
    state, actions, next_state, players, parent_locations, child_locations, food_grid_stride=1
):
    #jax.debug.print('player_x {x}', x=next_state.env_state.player_x)
    #jax.debug.print('player_r {r}', r=next_state.env_state.player_r)
    #jax.debug.print('action {a}', a=actions)
    # Only store food_grid for steps that match the stride (to save memory)
    # Use curr_step to determine if we should store this step's food_grid.
    # This must be JAX-friendly since make_report is called inside a jitted scan.
    curr_step = next_state.env_state.curr_step
    should_store_food_grid = (curr_step % food_grid_stride == 0)
    
    # Store food_grid if stride matches, otherwise store zeros (same shape to maintain array structure).
    # Use jax.lax.cond instead of a Python if to avoid TracerBoolConversionError.
    food_grid = jax.lax.cond(
        should_store_food_grid,
        lambda *_: next_state.env_state.food_grid,
        lambda *_: jnp.zeros_like(next_state.env_state.food_grid),
        operand=None,
    )
    
    # Extract birthdays and current time from family_tree
    birthdays = next_state.env_state.family_tree.player_state.players[..., 0]
    current_time = next_state.env_state.family_tree.player_state.current_time
    
    return NomNomReport(
        actions,
        players,
        next_state.env_state.player_x,
        next_state.env_state.player_r,
        next_state.env_state.player_energy,
        food_grid,
        next_state.env_state.family_tree.parents,
        birthdays,
        current_time,
    )

def train(key, params):
    if params.log_level == "INFO":
        import wandb
        wandb.init(project="dirt",
                entity="lunameme"
                )
    
    # build the necessary functions
    # - build the environment functions
    poeg = nomnom(params.env_params)
    # Pass the POEG object directly - natural_selection will extract methods
    reset_env = poeg.init
    step_env = poeg.step
    
    # - build mutate function
    # Wrap to match expected signature: breed(key, parent_state)
    _mutate = normal_mutate(learning_rate=3e-4)
    def mutate(key, parent_state):
        # normal_mutate expects (key, state), so we pass parent_state as state
        return _mutate(key, parent_state)
    
    # - build the model functions
    model_params = NomNomModelParams(
        view_width=params.env_params.view_width,
        view_distance=params.env_params.view_distance,
    )
    # NOTE: NomNomTrainParams is a frozen static dataclass, so we avoid
    # mutating it in-place (which would raise FrozenInstanceError) and
    # instead derive a local output_directory used for saving.
    if params.use_linear_model:
        init_model, model = nomnom_linear_model(model_params)
        output_directory = params.output_directory + '/linear_model'
    else:
        init_model, model = nomnom_model(model_params)
        output_directory = params.output_directory + '/nomnom_model'
    
    # Append exp_id to output_directory if provided
    if params.exp_id is not None:
        output_directory = output_directory + '/' + params.exp_id
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # - build the training functions
    # Create a make_report function that respects the food_grid_stride parameter
    def make_report_with_stride(state, actions, next_state, players, parent_locations, child_locations):
        return make_report(state, actions, next_state, players, parent_locations, child_locations,
                          food_grid_stride=params.food_grid_stride)
    
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model,
        model,
        mutate,
        make_report_with_stride
    )
    
    # get the initial state of the training function
    key, reset_key = jrng.split(key)
    train_state, _ = jax.jit(reset_train)(reset_key)
    epoch = 0
    if params.load_from_file is not None:
        key, epoch, train_state = load_example_data(
            (key, epoch, train_state), params.load_from_file)
    
    # precompile the primary epoch train computation
    def train_epoch(epoch_key, train_state):
        def scan_body(train_state, step_key):
            next_train_state, report = step_train(step_key, train_state)
            return next_train_state, report
        
        train_state, reports = jax.lax.scan(
            scan_body,
            train_state,
            jrng.split(epoch_key, params.steps_per_epoch),
        )
        
        return train_state, reports
    train_epoch = jax.jit(train_epoch)
    
    save_leaf_data(
        params,
        f'{output_directory}/train_params.state',
    )
    
    # Initialize time-series tracking for active players
    players_time_series = []
    
    # Initialize time-series tracking for average lifespan per epoch
    # This will be a list where index i contains the average lifespan for agents that died in epoch i
    avg_lifespan_per_epoch = []
    
    # the outer loop is not scanned because it will have side effects
    while epoch < params.epochs:
        print(f'Epoch: {epoch}')
        key, epoch_key = jrng.split(key)
        
        train_state, reports = train_epoch(epoch_key, train_state)
        
        save_leaf_data(
            (key, epoch, train_state),
            f'{output_directory}/train_state_{epoch:08}.state',
        )
        save_leaf_data(
            reports,
            f'{output_directory}/report_{epoch:08}.state',
        )
        epoch += 1
        
        #actions, players = reports
        actions = reports.actions
        players = reports.players
        # For the 3 components of actions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].hist(jnp.mean(actions.forward * players, axis=0), bins=20, color="blue", alpha=0.7)
        axes[0].set_title("Action distribution 1: forward")

        axes[1].hist(jnp.mean(actions.rotate * players, axis=0), bins=20, color="red", alpha=0.7)
        axes[1].set_title("Action distribution 2: rotate")

        axes[2].hist(jnp.mean(actions.reproduce * players, axis=0), bins=20, color="green", alpha=0.7)
        axes[2].set_title("Action distribution 3: reproduce")

        fig.tight_layout()
        if params.log_level == "INFO":
            wandb.log({"plot/actions": wandb.Image(fig)})
            wandb.log({"active players": players.sum()})
        
        # Check if there are 0 agents at any point in this epoch
        # players shape is (steps_per_epoch, max_players)
        active_players_per_step = jnp.sum(players, axis=-1)  # Sum over player dimension
        
        # Accumulate time-series data (convert to numpy for JSON serialization)
        active_players_array = np.array(active_players_per_step)
        players_time_series.extend(active_players_array.tolist())
        
        # Track agent deaths and calculate lifespans
        # Compare consecutive steps to detect deaths (was active, now inactive)
        lifespans_this_epoch = []
        
        # Track deaths within this epoch by comparing consecutive steps
        players_array = np.array(reports.players)  # Shape: (steps_per_epoch, max_players)
        birthdays_array = np.array(reports.family_tree_birthdays)  # Shape: (steps_per_epoch, max_players)
        
        # Calculate absolute step numbers for this epoch
        # Step 0 of epoch 0 is step 0, step 0 of epoch 1 is step steps_per_epoch, etc.
        epoch_start_step = (epoch - 1) * params.steps_per_epoch
        
        # For each step after the first, check for deaths
        for step_idx in range(1, players_array.shape[0]):
            prev_active = players_array[step_idx - 1]  # Active in previous step
            curr_active = players_array[step_idx]  # Active in current step
            
            # Deaths: was active, now inactive
            # Handle both boolean and integer arrays (1/0 or True/False)
            prev_active_bool = prev_active.astype(bool)
            curr_active_bool = curr_active.astype(bool)
            deaths = prev_active_bool & ~curr_active_bool
            death_indices = np.where(deaths)[0]
            
            if len(death_indices) > 0:
                # Get birthdays for dead agents (use previous step's birthdays)
                # Birthdays are absolute step numbers when agents were born
                dead_birthdays = birthdays_array[step_idx - 1][death_indices]
                # Death time is the absolute step number when death occurred
                death_time = epoch_start_step + step_idx
                
                # Calculate lifespans: death_time - birthday
                lifespans = death_time - dead_birthdays
                lifespans_this_epoch.extend(lifespans.tolist())
        
        # Calculate average lifespan for this epoch
        if len(lifespans_this_epoch) > 0:
            avg_lifespan = np.mean(lifespans_this_epoch)
            avg_lifespan_per_epoch.append(float(avg_lifespan))
            print(f"  Average lifespan (agents that died this epoch): {avg_lifespan:.2f} steps ({len(lifespans_this_epoch)} deaths)")
        else:
            avg_lifespan_per_epoch.append(None)  # No deaths this epoch
            print(f"  Average lifespan: N/A (no deaths this epoch)")
        
        # Check if any step had 0 agents
        min_active_players = jnp.min(active_players_per_step)
        
        if min_active_players == 0:
            # Find the first timestep with 0 agents
            zero_agent_steps = jnp.where(active_players_per_step == 0)[0]
            first_zero_step = int(zero_agent_steps[0])
            total_timestep = (epoch - 1) * params.steps_per_epoch + first_zero_step
            
            print(f"\n{'='*60}")
            print(f"Training ended: 0 agents detected")
            print(f"Epoch: {epoch - 1}")
            print(f"Timestep within epoch: {first_zero_step}")
            print(f"Total timestep: {total_timestep}")
            print(f"{'='*60}\n")
            
            if params.log_level == "INFO":
                wandb.log({
                    "training_ended": True,
                    "ending_epoch": epoch - 1,
                    "ending_timestep": total_timestep,
                    "ending_timestep_in_epoch": first_zero_step
                })
            
            break
    
    # Save time-series data to JSON file
    time_series_file = f'{output_directory}/players_time_series.json'
    time_series_data = {
        'active_players_per_step': players_time_series,
        'avg_lifespan_per_epoch': avg_lifespan_per_epoch,
        'total_steps': len(players_time_series),
        'steps_per_epoch': params.steps_per_epoch,
        'epochs_completed': epoch,
        'exp_id': params.exp_id,
    }
    with open(time_series_file, 'w') as f:
        json.dump(time_series_data, f, indent=2)
    print(f"Saved players time-series to: {time_series_file}")
    
    return train_state

if __name__ == '__main__':
    
    max_players = 128 #*16
    env_params = NomNomParams(
        max_energy=2,
        mean_initial_food=100000, #8**2,
        max_initial_food= 100000, #100000, #32**2,
        mean_food_growth=16, #16, #16*16, #2**2,
        max_food_growth=512, #1000, #16**2,
        initial_players=32,
        max_players=max_players,
        world_size=(512,512), #(64,64),
        senescence=0.01,
        food_metabolism=8,
    )
    train_params = NaturalSelectionParams(
        max_players=max_players,
    )
    params = NomNomTrainParams(
        env_params=env_params,
        train_params=train_params,
        epochs=20,
        steps_per_epoch=1024,
        use_linear_model=True,
    )
    
    # update these defaults with commandline arguments
    parser = argparse.ArgumentParser()
    params.add_commandline_args(parser)
    args = parser.parse_args()
    params = params.update_from_commandline(args)
    
    # Initialize random key with seed from params
    key = jrng.key(params.seed)
    
    start = time.time()
    train(key, params)
    print(time.time() - start)
