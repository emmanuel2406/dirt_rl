'''
Train NomNom with RL Adaptation

This script demonstrates how to train NomNom agents with:
- Stage 1: Individual RL adaptation (within-lifetime learning)
- Stage 2: Local communication (optional)

The agents evolve through natural selection while also learning via RL.
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

# RL components
from mechagogue.ecology.rl_adaptation import RLAdaptationParams
from mechagogue.ecology.communication import CommunicationParams
from mechagogue.ecology.rl_policy import RLPolicyParams
from mechagogue.ecology.policy_rl_integration import make_rl_enabled_policy

# NomNom environment and models
from dirt.envs.nomnom import nomnom, NomNomParams, NomNomAction
from dirt.examples.nomnom.nomnom_model import (
    NomNomModelParams, nomnom_linear_model)
from dirt.examples.nomnom.nomnom_model_rl import (
    create_rl_nomnom_model, NomNomRLModelParams)

import matplotlib.pyplot as plt


@static_dataclass
@commandline_interface()
class NomNomRLTrainParams:
    '''Training parameters for RL-enabled NomNom.'''
    max_players: int = 256
    use_linear_model: bool = True
    log_level: str = "INFO"
    exp_id: Optional[str] = None
    
    # Environment parameters
    env_params: Any = NomNomParams(
        mean_initial_food=8**2,
        max_initial_food=32**2,
        mean_food_growth=2**2,
        max_food_growth=16**2,
        initial_players=32,
        max_players=max_players,
        world_size=(32, 32)
    )
    
    # Training parameters
    train_params: Any = NaturalSelectionParams(
        max_players=max_players,
    )
    
    # RL parameters
    enable_rl: bool = True
    rl_learning_rate: float = 1e-4
    rl_discount_factor: float = 0.95
    rl_buffer_size: int = 16
    rl_adapt_frequency: int = 8
    
    # Communication parameters
    enable_communication: bool = False
    communication_radius: int = 5
    max_neighbors: int = 8
    
    # Reward weights
    survival_reward_weight: float = 1.0
    energy_reward_weight: float = 0.1
    social_reward_weight: float = 0.5
    
    # Training loop
    epochs: int = 100
    steps_per_epoch: int = 1000
    output_directory: str = '/n/netscratch/kdbrantley_lab/Lab/erassou/dirt/rl_experiments'
    load_from_file: Optional[str] = None


@static_dataclass
class NomNomRLReport:
    '''Extended report with RL metrics.'''
    actions: Any = NomNomAction(0, 0, 0)
    players: jnp.array = False
    player_x: jnp.array = False
    player_r: jnp.array = False
    player_energy: jnp.array = False
    food_grid: jnp.array = False
    family_tree_parents: jnp.array = False
    
    # RL-specific metrics
    avg_episode_return: float = 0.0
    avg_buffer_fullness: float = 0.0


def make_report(
    state, actions, next_state, players, parent_locations, child_locations
):
    '''Create training report.'''
    return NomNomRLReport(
        actions=actions,
        players=players,
        player_x=next_state.env_state.player_x,
        player_r=next_state.env_state.player_r,
        player_energy=next_state.env_state.player_energy,
        food_grid=next_state.env_state.food_grid,
        family_tree_parents=next_state.env_state.family_tree.parents,
    )


def train(key, params):
    '''Main training loop.'''
    
    # Setup logging
    if params.log_level == "INFO":
        try:
            import wandb
            wandb.init(
                project="dirt-rl",
                entity="lunameme",
                config={
                    "enable_rl": params.enable_rl,
                    "enable_communication": params.enable_communication,
                    "rl_learning_rate": params.rl_learning_rate,
                    "rl_buffer_size": params.rl_buffer_size,
                    "exp_id": params.exp_id,
                }
            )
        except ImportError:
            print("Warning: wandb not available, skipping logging")
            params = params.replace(log_level="WARN")
    
    # Build environment
    poeg = nomnom(params.env_params)
    reset_env = poeg.init
    step_env = poeg.step
    
    # Build model
    model_params = NomNomModelParams(
        view_width=params.env_params.view_width,
        view_distance=params.env_params.view_distance,
    )
    
    if params.use_linear_model:
        # RL experiments with the linear model are stored under `linear_model_rl`
        # to match the expectations in `scripts/ablate_nomnom.sh`.
        init_model, model = nomnom_linear_model(model_params)
        output_directory = params.output_directory + '/linear_model_rl'
    else:
        # Use RL-compatible model if communication enabled.
        # Directory names are chosen to stay in sync with OUTPUT_DIR_BASE
        # in `scripts/ablate_nomnom.sh`.
        if params.enable_communication:
            init_model, model = create_rl_nomnom_model(
                use_communication=True,
                view_width=params.env_params.view_width,
                view_distance=params.env_params.view_distance,
            )
            output_directory = params.output_directory + '/nomnom_model_rl_comm'
        else:
            init_model, model = nomnom_linear_model(model_params)
            output_directory = params.output_directory + '/nomnom_model_rl'
    
    # Append exp_id
    if params.exp_id is not None:
        output_directory = output_directory + '/' + params.exp_id
    os.makedirs(output_directory, exist_ok=True)
    
    # Create RL-enabled policy if enabled
    if params.enable_rl:
        print("Creating RL-enabled policy...")
        
        # Configure RL adaptation
        rl_adaptation_params = RLAdaptationParams(
            learning_rate=params.rl_learning_rate,
            discount_factor=params.rl_discount_factor,
            buffer_size=params.rl_buffer_size,
            adapt_frequency=params.rl_adapt_frequency,
            enable_adaptation=True,
        )
        
        # Create reward shaper
        from mechagogue.ecology.rl_adaptation import simple_reward_shaper
        reward_fn = simple_reward_shaper(
            survival_weight=params.survival_reward_weight,
            energy_weight=params.energy_reward_weight,
        )
        
        # Create RL policy
        policy = make_rl_enabled_policy(
            init_model,
            model,
            rl_params=rl_adaptation_params,
            compute_reward=reward_fn,
        )
        
        # Override init_model and model with policy versions
        # Note: This is a simplification. In a full implementation,
        # we'd integrate this more deeply with natural_selection
        print("RL policy created with learning_rate={}, buffer_size={}".format(
            params.rl_learning_rate, params.rl_buffer_size))
    
    # Build mutation function
    _mutate = normal_mutate(learning_rate=3e-4)
    def mutate(key, parent_state):
        return _mutate(key, parent_state)
    
    # Build training functions
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model if not params.enable_rl else policy.init,
        model if not params.enable_rl else policy.act,
        mutate,
        make_report
    )
    
    # Initialize training state
    key, reset_key = jrng.split(key)
    train_state, _ = jax.jit(reset_train)(reset_key)
    epoch = 0
    
    if params.load_from_file is not None:
        key, epoch, train_state = load_example_data(
            (key, epoch, train_state), params.load_from_file)
    
    # Precompile training epoch
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
    
    # Save training parameters
    save_leaf_data(
        params,
        f'{output_directory}/train_params.state',
    )
    
    # Training loop
    print(f"Starting training: {params.epochs} epochs, {params.steps_per_epoch} steps per epoch")
    print(f"RL enabled: {params.enable_rl}")
    print(f"Communication enabled: {params.enable_communication}")
    
    while epoch < params.epochs:
        epoch_start = time.time()
        print(f'\nEpoch: {epoch}')
        
        key, epoch_key = jrng.split(key)
        train_state, reports = train_epoch(epoch_key, train_state)
        
        epoch_time = time.time() - epoch_start
        
        # Save state and reports
        save_leaf_data(
            (key, epoch, train_state),
            f'{output_directory}/train_state_{epoch:08}.state',
        )
        save_leaf_data(
            reports,
            f'{output_directory}/report_{epoch:08}.state',
        )
        
        # Logging
        actions = reports.actions
        players = reports.players
        active_count = jnp.sum(players)
        
        print(f"  Active agents: {active_count:.0f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Plot action distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(jnp.mean(actions.forward * players, axis=0), bins=20, color="blue", alpha=0.7)
        axes[0].set_title("Forward actions")
        
        axes[1].hist(jnp.mean(actions.rotate * players, axis=0), bins=20, color="red", alpha=0.7)
        axes[1].set_title("Rotate actions")
        
        axes[2].hist(jnp.mean(actions.reproduce * players, axis=0), bins=20, color="green", alpha=0.7)
        axes[2].set_title("Reproduce actions")
        
        fig.tight_layout()
        
        if params.log_level == "INFO":
            try:
                wandb.log({
                    "plot/actions": wandb.Image(fig),
                    "active_players": int(active_count),
                    "epoch": epoch,
                    "epoch_time": epoch_time,
                })
            except:
                pass
        
        plt.close(fig)
        
        # Check for extinction
        active_players_per_step = jnp.sum(players, axis=-1)
        min_active_players = jnp.min(active_players_per_step)
        
        if min_active_players == 0:
            zero_agent_steps = jnp.where(active_players_per_step == 0)[0]
            first_zero_step = int(zero_agent_steps[0])
            total_timestep = epoch * params.steps_per_epoch + first_zero_step
            
            print(f"\n{'='*60}")
            print(f"Training ended: 0 agents detected")
            print(f"Epoch: {epoch}")
            print(f"Timestep: {first_zero_step}")
            print(f"Total timestep: {total_timestep}")
            print(f"{'='*60}\n")
            
            if params.log_level == "INFO":
                try:
                    wandb.log({
                        "training_ended": True,
                        "ending_epoch": epoch,
                        "ending_timestep": total_timestep,
                    })
                except:
                    pass
            
            break
        
        epoch += 1
    
    print("\nTraining complete!")
    return train_state


if __name__ == '__main__':
    key = jrng.key(1234)
    
    # Default parameters
    max_players = 128
    env_params = NomNomParams(
        max_energy=16,
        mean_initial_food=100000,
        max_initial_food=100000,
        mean_food_growth=16,
        max_food_growth=512,
        initial_players=32,
        max_players=max_players,
        world_size=(512, 512),
        senescence=0.01,
        food_metabolism=8,
    )
    
    train_params = NaturalSelectionParams(
        max_players=max_players,
    )
    
    params = NomNomRLTrainParams(
        env_params=env_params,
        train_params=train_params,
        epochs=10,
        steps_per_epoch=4096,
        use_linear_model=True,
        enable_rl=True,  # ← Enable RL!
        rl_learning_rate=1e-4,
        rl_buffer_size=16,
        rl_adapt_frequency=8,
        enable_communication=False,  # Start without communication
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    params.add_commandline_args(parser)
    args = parser.parse_args()
    params = params.update_from_commandline(args)
    
    # Train
    start = time.time()
    train(key, params)
    print(f"Total time: {time.time() - start:.2f}s")

