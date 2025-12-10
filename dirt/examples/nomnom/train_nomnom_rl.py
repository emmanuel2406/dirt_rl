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
    create_rl_nomnom_model, NomNomRLModelParams,
    nomnom_linear_model_with_communication)

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
    use_message_action: bool = True  # Whether agents output message actions (separate from communication infrastructure)
    communication_radius: int = 5
    max_neighbors: int = 8
    
    # Reward weights
    survival_reward_weight: float = 1.0
    energy_reward_weight: float = 0.1
    social_reward_weight: float = 0.5
    
    # Memory optimization: store food_grid only every Nth step (for visualization)
    # Set to 1 to store every step, or higher to save memory
    # This should match the step_stride used in visualize_nomnom.py
    food_grid_stride: int = 1  # Store food_grid every Nth step
    
    # Training loop
    epochs: int = 100
    steps_per_epoch: int = 1000
    output_directory: str = '/n/netscratch/kdbrantley_lab/Lab/erassou/dirt/rl_experiments'
    load_from_file: Optional[str] = None
    
    # Random seed
    seed: int = 1234


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
    family_tree_birthdays: jnp.array = False  # Birth step for each agent
    family_tree_current_time: int = 0  # Current step/time
    
    # RL-specific metrics
    avg_episode_return: float = 0.0
    avg_buffer_fullness: float = 0.0


def make_report(
    state, actions, next_state, players, parent_locations, child_locations, food_grid_stride=1
):
    '''Create training report.
    
    BUG FIX: When communication is enabled, the env_state structure may differ.
    This function handles both cases:
    1. Standard: env_state is a NomNomState object with food_grid attribute
    2. Communication: env_state might be a tuple (env_state, comm_state) if the
       environment is wrapped for communication
    
    Counterfactual debugging: Compare behavior with enable_communication=True vs False
    to identify where the state structure changes.
    '''
    # Extract env_state from next_state
    # next_state is always a NaturalSelectionState with env_state attribute
    env_state_raw = next_state.env_state
    
    # Handle case where env_state might be a tuple (env_state, comm_state) when communication wraps the environment
    # Use hasattr to check if it's a dataclass/object with attributes (JAX-compatible)
    # If it doesn't have the expected attribute, it might be a tuple
    if hasattr(env_state_raw, 'food_grid'):
        env_state = env_state_raw
    elif isinstance(env_state_raw, tuple) and len(env_state_raw) > 0:
        # Communication enabled: env_state is a tuple (env_state, comm_state)
        env_state = env_state_raw[0]
    else:
        # Fallback: assume it's the env_state directly
        env_state = env_state_raw
    
    # Only store food_grid for steps that match the stride (to save memory)
    # Use curr_step to determine if we should store this step's food_grid
    curr_step = env_state.curr_step
    should_store_food_grid = (curr_step % food_grid_stride == 0)
    
    # Store food_grid if stride matches, otherwise store zeros (same shape to maintain array structure)
    # Use jnp.where instead of Python if/else to work with JAX tracing
    # Ensure we're accessing the correct food_grid field
    food_grid_data = env_state.food_grid
    food_grid = jnp.where(
        should_store_food_grid,
        food_grid_data,
        jnp.zeros_like(food_grid_data)
    )
    
    # Extract birthdays and current time from family_tree
    birthdays = env_state.family_tree.player_state.players[..., 0]
    current_time = env_state.family_tree.player_state.current_time
    
    # Extract RL rewards from population state if available
    # The population state contains policy states from make_rl_enabled_policy
    # Policy state structure: {'rl_state': RLAdaptationState, 'last_obs': ..., 'last_action': ...}
    # After vmapping in make_ecology_population, population_state is a dict where values are vmapped
    # So population_state['rl_state'] gives us vmapped RLAdaptationState with shape (max_players, ...)
    avg_episode_return = 0.0
    avg_buffer_fullness = 0.0
    
    # Extract RL metrics - use direct dict access which works with JAX PyTree dicts
    try:
        if hasattr(next_state, 'population_state'):
            pop_state = next_state.population_state
            
            # Direct key access works for JAX dict PyTrees (even in traced contexts)
            # Check if 'rl_state' key exists by trying to access it
            try:
                rl_states = pop_state['rl_state']
                
                # rl_states is vmapped RLAdaptationState
                # Extract episode_return field (shape: (max_players,))
                if hasattr(rl_states, 'episode_return'):
                    episode_returns = jnp.asarray(rl_states.episode_return)
                    active_mask = jnp.asarray(players).astype(jnp.float32)
                    
                    # Ensure compatible shapes
                    max_len = min(episode_returns.shape[0], active_mask.shape[0])
                    episode_returns = episode_returns[:max_len]
                    active_mask = active_mask[:max_len]
                    
                    total_active = jnp.sum(active_mask)
                    if total_active > 0:
                        masked_returns = episode_returns * active_mask
                        avg_episode_return = float(jnp.sum(masked_returns) / total_active)
                
                # Extract buffer_full field
                if hasattr(rl_states, 'buffer_full'):
                    buffer_full = jnp.asarray(rl_states.buffer_full).astype(jnp.float32)
                    active_mask = jnp.asarray(players).astype(jnp.float32)
                    
                    max_len = min(buffer_full.shape[0], active_mask.shape[0])
                    buffer_full = buffer_full[:max_len]
                    active_mask = active_mask[:max_len]
                    
                    total_active = jnp.sum(active_mask)
                    if total_active > 0:
                        masked_buffer = buffer_full * active_mask
                        avg_buffer_fullness = float(jnp.sum(masked_buffer) / total_active)
            except (KeyError, TypeError):
                # 'rl_state' key doesn't exist or pop_state is not dict-like
                # This is expected when RL is not enabled
                pass
    except (AttributeError, TypeError, ValueError, IndexError):
        # If extraction fails for any reason, use defaults
        # This ensures make_report doesn't crash
        pass
    
    return NomNomRLReport(
        actions=actions,
        players=players,
        player_x=env_state.player_x,
        player_r=env_state.player_r,
        player_energy=env_state.player_energy,
        food_grid=food_grid,
        family_tree_parents=env_state.family_tree.parents,
        family_tree_birthdays=birthdays,
        family_tree_current_time=current_time,
        avg_episode_return=avg_episode_return,
        avg_buffer_fullness=avg_buffer_fullness,
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
    
    # Store message_dim for reward function (default to 4)
    message_dim = 4
    
    if params.use_linear_model:
        # RL experiments with the linear model are stored under `linear_model_rl`
        # to match the expectations in `scripts/ablate_nomnom.sh`.
        if params.enable_communication:
            # Use linear model with communication support
            rl_model_params = NomNomRLModelParams(
                view_width=params.env_params.view_width,
                view_distance=params.env_params.view_distance,
                enable_communication=True,
                use_augmented_obs=True,
                use_message_action=params.use_message_action,
            )
            message_dim = rl_model_params.message_dim
            init_model, model = nomnom_linear_model_with_communication(rl_model_params)
            output_directory = params.output_directory + '/linear_model_rl_comm'
        else:
            # Use RL-compatible model if RL is enabled, otherwise use vanilla model
            if params.enable_rl:
                # RL-compatible linear model without communication
                rl_model_params = NomNomRLModelParams(
                    view_width=params.env_params.view_width,
                    view_distance=params.env_params.view_distance,
                    enable_communication=False,
                    use_augmented_obs=False,
                )
                init_model, model = nomnom_linear_model_with_communication(rl_model_params)
            else:
                # Standard linear model without communication (no RL)
                init_model, model = nomnom_linear_model(model_params)
            output_directory = params.output_directory + '/linear_model_rl'
    else:
        # Use RL-compatible model if communication enabled.
        # Directory names are chosen to stay in sync with OUTPUT_DIR_BASE
        # in `scripts/ablate_nomnom.sh`.
        if params.enable_communication:
            # For non-linear model, message_dim defaults to 4 in NomNomRLModelParams
            # We could extract it from the created model, but 4 is the standard default
            message_dim = 4  # Default for non-linear models
            init_model, model = create_rl_nomnom_model(
                use_communication=True,
                view_width=params.env_params.view_width,
                view_distance=params.env_params.view_distance,
                use_message_action=params.use_message_action,
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
        from mechagogue.ecology.communication import (
            social_reward_shaper, AugmentedObservation, CommunicationAction
        )
        
        # Individual reward function
        individual_reward_fn = simple_reward_shaper(
            survival_weight=params.survival_reward_weight,
            energy_weight=params.energy_reward_weight,
        )
        
        # Social reward function (only used if communication enabled)
        if params.enable_communication:
            # message_dim is set above when creating the model
            social_reward_fn = social_reward_shaper(
                clustering_weight=0.1 * params.social_reward_weight,
                communication_use_weight=0.05 * params.social_reward_weight,
                energy_sharing_weight=0.2 * params.social_reward_weight,
            )
            
            # Combined reward function that handles both individual and social rewards
            def compute_combined_reward(obs, action, next_obs):
                # Extract base observations
                base_obs = obs.base_obs if hasattr(obs, 'base_obs') else obs
                next_base_obs = next_obs.base_obs if hasattr(next_obs, 'base_obs') else next_obs
                
                # Compute individual reward
                individual_reward = individual_reward_fn(base_obs, action, next_base_obs)
                
                # Compute social reward if observations are augmented
                social_reward = 0.0
                if (hasattr(obs, 'base_obs') and hasattr(next_obs, 'base_obs') and 
                    isinstance(obs, AugmentedObservation) and isinstance(next_obs, AugmentedObservation)):
                    # Extract communication action from NomNomAction
                    # If action has a message, create CommunicationAction
                    if action.message is not None:
                        # Handle message shape - should be (message_dim,) for single agent
                        message = action.message
                        # Flatten if needed and ensure correct size
                        if message.shape == ():
                            message = jnp.zeros(message_dim)
                        elif len(message.shape) > 1:
                            # If batched, take first (shouldn't happen in per-agent reward)
                            message = message.flatten()[:message_dim]
                        elif message.shape[0] != message_dim:
                            # Wrong size, pad or truncate
                            if message.shape[0] < message_dim:
                                padding = jnp.zeros(message_dim - message.shape[0])
                                message = jnp.concatenate([message, padding])
                            else:
                                message = message[:message_dim]
                        
                        comm_action = CommunicationAction(
                            message=message,
                            emit_smell=0.0,  # Not used in current implementation
                            emit_sound=0.0,  # Not used in current implementation
                        )
                    else:
                        # No message, use zero communication action
                        comm_action = CommunicationAction(
                            message=jnp.zeros(message_dim),
                            emit_smell=0.0,
                            emit_sound=0.0,
                        )
                    
                    # Compute social reward
                    social_reward = social_reward_fn(obs, action, comm_action, next_obs)
                
                return individual_reward + social_reward
            
            reward_fn = compute_combined_reward
            print(f"Using combined reward function with social_reward_weight={params.social_reward_weight}")
        else:
            # No communication, use only individual rewards
            reward_fn = individual_reward_fn
            print("Using individual reward function only (no social rewards)")
        
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
    # Create a make_report function that respects the food_grid_stride parameter
    def make_report_with_stride(state, actions, next_state, players, parent_locations, child_locations):
        return make_report(state, actions, next_state, players, parent_locations, child_locations,
                          food_grid_stride=params.food_grid_stride)
    
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model if not params.enable_rl else policy.init,
        model if not params.enable_rl else policy.act,
        mutate,
        make_report_with_stride,
        # When RL is enabled, pass the full policy class so that its adapt()
        # method is used by the ecology population, which in turn calls
        # rl_adaptation.RLAdaptation.adapt via policy_rl_integration.
        policy_cls=policy if params.enable_rl else None,
    )
    
    # Initialize training state
    key, reset_key = jrng.split(key)
    train_state, _ = jax.jit(reset_train)(reset_key)
    epoch = 0

    # JAX PyTree stability fix for RL-enabled training:
    # RLEnabledPolicy.state includes fields like 'last_obs' and 'last_action'.
    # If these start as None and later become NomNomObservation/NomNomAction,
    # the carry structure in jax.lax.scan changes and raises a TypeError.
    #
    # To ensure the carry PyTree structure is stable across all scan
    # iterations, we perform a single warm-up step with step_train when RL
    # is enabled. This initializes the policy state so that 'last_obs' and
    # 'last_action' have their final (non-None) types before entering the
    # jitted training epoch scan.
    if params.enable_rl:
        key, warmup_key = jrng.split(key)
        train_state, _ = step_train(warmup_key, train_state)
    
    if params.load_from_file is not None:
        key, epoch, train_state = load_example_data(
            (key, epoch, train_state), params.load_from_file)
    
    # Define training epoch
    #
    # For pure evolutionary training (enable_rl == False), we use a JAX scan +
    # jit for maximum performance.
    #
    # For RL-enabled training, the policy and RL adaptation state include
    # fields that change Python types over time (e.g., None -> observation
    # dataclasses inside the carry). JAX's lax.scan requires the carry PyTree
    # structure and Python types to be identical across iterations, which is
    # violated in this case and leads to scan carry mismatch errors.
    #
    # To keep the RL path robust and JAX-compatible without a large refactor
    # of all state types, we fall back to a simple Python loop for epochs when
    # RL is enabled. The inner operations (env step, policy act/adapt, RL
    # updates) remain JAX-jittable, but the outer epoch loop is executed in
    # Python.
    def train_epoch_scan(epoch_key, train_state):
        def scan_body(train_state, step_key):
            next_train_state, report = step_train(step_key, train_state)
            return next_train_state, report

        train_state, reports = jax.lax.scan(
            scan_body,
            train_state,
            jrng.split(epoch_key, params.steps_per_epoch),
        )

        return train_state, reports

    def train_epoch_python(epoch_key, train_state):
        """Epoch loop implemented as a Python for-loop for RL-enabled runs."""
        step_keys = jrng.split(epoch_key, params.steps_per_epoch)
        state = train_state
        reports_list = []

        for step_key in step_keys:
            state, report = step_train(step_key, state)
            reports_list.append(report)

        # Stack reports into the same structure produced by lax.scan
        reports = jax.tree.map(lambda *xs: jnp.stack(xs), *reports_list)
        return state, reports

    # Choose implementation based on RL flag
    if params.enable_rl:
        # No outer jit/scan: avoid PyTree carry structure issues in RL state.
        train_epoch = train_epoch_python
    else:
        # Fast, fully-jitted epoch for non-RL training.
        train_epoch = jax.jit(train_epoch_scan)
    
    # Save training parameters
    save_leaf_data(
        params,
        f'{output_directory}/train_params.state',
    )
    
    # Initialize time-series tracking for active players
    players_time_series = []
    
    # Initialize time-series tracking for average rewards per epoch
    # This will be a list where index i contains the average reward for epoch i
    avg_reward_per_epoch = []
    
    # Initialize time-series tracking for average buffer fullness per epoch
    # This will be a list where index i contains the average buffer fullness for epoch i
    avg_buffer_fullness_per_epoch = []
    
    # Initialize time-series tracking for average lifespan per epoch
    # This will be a list where index i contains the average lifespan for agents that died in epoch i
    avg_lifespan_per_epoch = []
    
    # Training loop
    print(f"Starting training: {params.epochs} epochs, {params.steps_per_epoch} steps per epoch")
    print(f"RL enabled: {params.enable_rl}")
    print(f"Communication enabled: {params.enable_communication}")
    if params.enable_communication:
        print(f"Social reward weight: {params.social_reward_weight}")
        print(f"Use message action: {params.use_message_action}")
    if params.food_grid_stride > 1:
        world_size = params.env_params.world_size
        total_steps = params.epochs * params.steps_per_epoch
        stored_steps = total_steps // params.food_grid_stride
        memory_saved = (total_steps - stored_steps) * world_size[0] * world_size[1] * 4 / 1e6
        print(f"Memory optimization: food_grid stored every {params.food_grid_stride} steps (saves ~{memory_saved:.1f}MB total)")
    
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
        
        # Extract average reward and buffer fullness from reports (computed during RL adaptation)
        if params.enable_rl:
            # reports.avg_episode_return has shape (steps_per_epoch,)
            # episode_return is cumulative, so we compute per-step rewards from differences
            if hasattr(reports, 'avg_episode_return') and reports.avg_episode_return is not None:
                episode_returns = jnp.array(reports.avg_episode_return)  # Shape: (steps_per_epoch,)
                if episode_returns.shape[0] > 0:
                    # Compute per-step rewards as differences (reward[t] = episode_return[t] - episode_return[t-1])
                    # For first step, reward is just the episode_return value
                    if episode_returns.shape[0] == 1:
                        step_rewards = episode_returns
                    else:
                        step_rewards = jnp.concatenate([
                            episode_returns[0:1],  # First step
                            episode_returns[1:] - episode_returns[:-1]  # Subsequent steps
                        ])
                    # Average across all steps in the epoch
                    avg_reward = float(np.array(jnp.mean(step_rewards)))
                    avg_reward_per_epoch.append(avg_reward)
                    print(f"  Average reward per agent per step: {avg_reward:.4f} (from RL adaptation)")
                else:
                    avg_reward_per_epoch.append(0.0)
            else:
                # Fallback if reports don't have avg_episode_return
                avg_reward_per_epoch.append(0.0)
            
            # Extract average buffer fullness
            if hasattr(reports, 'avg_buffer_fullness') and reports.avg_buffer_fullness is not None:
                buffer_fullness = jnp.array(reports.avg_buffer_fullness)  # Shape: (steps_per_epoch,)
                if buffer_fullness.shape[0] > 0:
                    # Average across all steps in the epoch
                    avg_buffer_fullness = float(np.array(jnp.mean(buffer_fullness)))
                    avg_buffer_fullness_per_epoch.append(avg_buffer_fullness)
                else:
                    avg_buffer_fullness_per_epoch.append(0.0)
            else:
                # Fallback if reports don't have avg_buffer_fullness
                avg_buffer_fullness_per_epoch.append(0.0)
        else:
            avg_reward_per_epoch.append(0.0)
            avg_buffer_fullness_per_epoch.append(0.0)
        
        # Track agent deaths and calculate lifespans
        # Compare consecutive steps to detect deaths (was active, now inactive)
        lifespans_this_epoch = []
        
        # Track deaths within this epoch by comparing consecutive steps
        players_array = np.array(reports.players)  # Shape: (steps_per_epoch, max_players)
        birthdays_array = np.array(reports.family_tree_birthdays)  # Shape: (steps_per_epoch, max_players)
        
        # Calculate absolute step numbers for this epoch
        # Step 0 of epoch 0 is step 0, step 0 of epoch 1 is step steps_per_epoch, etc.
        epoch_start_step = epoch * params.steps_per_epoch
        
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
        
        # Check for extinction
        active_players_per_step = jnp.sum(players, axis=-1)
        
        # Accumulate time-series data (convert to numpy for JSON serialization)
        active_players_array = np.array(active_players_per_step)
        players_time_series.extend(active_players_array.tolist())
        
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
    
    # Save time-series data to JSON file
    time_series_file = f'{output_directory}/players_time_series.json'
    time_series_data = {
        'active_players_per_step': players_time_series,
        'avg_reward_per_epoch': avg_reward_per_epoch,
        'avg_buffer_fullness_per_epoch': avg_buffer_fullness_per_epoch,
        'avg_lifespan_per_epoch': avg_lifespan_per_epoch,
        'total_steps': len(players_time_series),
        'steps_per_epoch': params.steps_per_epoch,
        'epochs_completed': epoch,
        'exp_id': params.exp_id,
        'enable_rl': params.enable_rl,
        'enable_communication': params.enable_communication,
    }
    with open(time_series_file, 'w') as f:
        json.dump(time_series_data, f, indent=2)
    print(f"Saved players time-series to: {time_series_file}")
    
    print("\nTraining complete!")
    return train_state


if __name__ == '__main__':
    # Default parameters
    max_players = 128
    env_params = NomNomParams(
        max_energy=2,
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
        epochs=20,
        steps_per_epoch=1024,
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
    
    # Initialize random key with seed from params
    key = jrng.key(params.seed)
    
    print(f"Learning rate: {params.rl_learning_rate}")
    print(f"Buffer size: {params.rl_buffer_size}")
    print(f"Adapt frequency: {params.rl_adapt_frequency}")
    # Train
    start = time.time()
    train(key, params)
    print(f"Total time: {time.time() - start:.2f}s")

