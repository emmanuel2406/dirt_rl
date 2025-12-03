'''
This example is designed to test the nomnom environment in a training loop
with a random policy.
'''
import time
import argparse
import os
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

@static_dataclass
class NomNomReport:
    actions : Any = NomNomAction(0,0,0)
    players : jnp.array = False
    player_x : jnp.array = False
    player_r : jnp.array = False
    player_energy : jnp.array = False
    food_grid : jnp.array = False
    family_tree_parents : jnp.array = False

def make_report(
    state, actions, next_state, players, parent_locations, child_locations
):
    #jax.debug.print('player_x {x}', x=next_state.env_state.player_x)
    #jax.debug.print('player_r {r}', r=next_state.env_state.player_r)
    #jax.debug.print('action {a}', a=actions)
    return NomNomReport(
        actions,
        players,
        next_state.env_state.player_x,
        next_state.env_state.player_r,
        next_state.env_state.player_energy,
        next_state.env_state.food_grid,
        next_state.env_state.family_tree.parents,
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
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model,
        model,
        mutate,
        make_report
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
    
    return train_state

if __name__ == '__main__':
    
    #key = jrng.key(5432)
    key = jrng.key(1234)
    
    max_players = 128 #*16
    env_params = NomNomParams(
        max_energy=16,
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
        epochs=10,
        steps_per_epoch=4096,
        use_linear_model=True,
    )
    
    # update these defaults with commandline arguments
    parser = argparse.ArgumentParser()
    params.add_commandline_args(parser)
    args = parser.parse_args()
    params = params.update_from_commandline(args)
    
    start = time.time()
    train(key, params)
    print(time.time() - start)
