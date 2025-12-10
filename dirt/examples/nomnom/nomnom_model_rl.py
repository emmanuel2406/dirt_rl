'''
Extended NomNom Models with RL and Communication Support

This module provides enhanced versions of the NomNom models that are compatible
with the RL adaptation and communication systems from mechagogue.

Key additions:
- Extended observation processing for augmented observations
- Communication action outputs
- Compatibility with both evolution and RL training
'''

import jax
import jax.random as jrng
import jax.numpy as jnp
import jax.nn as jnn
from jax.tree import map as tree_map

from mechagogue.static import static_data as static_dataclass
from mechagogue.nn.linear import linear_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.structure import dict_layer as parallel_dict_layer
from mechagogue.nn.distributions import categorical_sampler_layer

from dirt.envs.nomnom import NomNomObservation, NomNomAction


@static_dataclass
class NomNomRLModelParams:
    '''Extended parameters for RL-enabled models.'''
    num_input_classes: int = 4
    view_width: int = 5
    view_distance: int = 5
    
    # Communication parameters
    max_neighbors: int = 8
    message_dim: int = 4
    enable_communication: bool = True
    
    # Whether observations are augmented
    use_augmented_obs: bool = True
    
    # Whether agents output message actions (separate from communication infrastructure)
    # When False, agents can still receive messages but don't output them
    use_message_action: bool = True


def nomnom_linear_model_with_communication(params=NomNomRLModelParams()):
    '''
    Extended linear model that handles augmented observations and communication.
    
    This model:
    - Accepts standard NomNomObservation or AugmentedObservation
    - Outputs standard NomNomAction (forward, rotate, reproduce)
    - Can optionally output communication actions
    
    Compatible with both evolution-only and RL training.
    '''
    
    # Calculate input dimensions
    base_obs_dim = (
        (params.num_input_classes - 1) *
        params.view_width *
        params.view_distance +
        1  # energy
    )
    
    # Additional dimensions if using augmented observations
    if params.enable_communication and params.use_augmented_obs:
        # Each neighbor: (dx, dy, energy, message_dim, distance, is_valid)
        neighbor_feature_size = 2 + 1 + params.message_dim + 1 + 1
        neighbor_total = params.max_neighbors * neighbor_feature_size
        # Environmental cues: smell, sound, num_nearby
        env_cue_size = 3
        in_dim = base_obs_dim + neighbor_total + env_cue_size
    else:
        in_dim = base_obs_dim
    
    def encoder_forward(x):
        '''
        Encode observation into feature vector.
        
        Handles both standard NomNomObservation and AugmentedObservation.
        '''
        # Check if observation is augmented
        is_augmented = hasattr(x, 'base_obs')
        
        if is_augmented:
            # Extract base observation
            base_obs = x.base_obs
        else:
            base_obs = x
        
        # Process base observation
        food = (base_obs.view == 1).reshape(-1).astype(jnp.float32)
        players = (base_obs.view == 2).reshape(-1).astype(jnp.float32)
        out_of_bounds = (base_obs.view == 3).reshape(-1).astype(jnp.float32)
        
        energy_flat = jnp.atleast_1d(base_obs.energy).flatten()
        energy = energy_flat[0:1].astype(jnp.float32)
        
        base_features = jnp.concatenate(
            (food, players, out_of_bounds, energy), axis=-1
        )
        
        # If augmented, add communication features
        if is_augmented and params.enable_communication:
            # Process nearby agents
            def flatten_agent_info(info):
                return jnp.concatenate([
                    info.relative_x.astype(jnp.float32),
                    jnp.array([info.energy], dtype=jnp.float32),
                    info.message.astype(jnp.float32),
                    jnp.array([info.distance], dtype=jnp.float32),
                    jnp.array([info.is_valid], dtype=jnp.float32),
                ])
            
            # Vectorize over neighbors
            nearby_features = jax.vmap(flatten_agent_info)(x.nearby_agents)
            nearby_features = nearby_features.flatten()
            
            # Environmental cues
            env_cues = jnp.array([
                x.local_smell,
                x.local_sound,
                x.num_nearby / params.max_neighbors,
            ], dtype=jnp.float32)
            
            # Concatenate all features
            all_features = jnp.concatenate([
                base_features,
                nearby_features,
                env_cues,
            ])
        else:
            all_features = base_features
        
        # Ensure correct dimension (pad or truncate)
        current_dim = all_features.shape[-1]
        if current_dim > in_dim:
            all_features = all_features[..., :in_dim]
        elif current_dim < in_dim:
            padding = jnp.zeros((in_dim - current_dim,), dtype=jnp.float32)
            all_features = jnp.concatenate([all_features, padding], axis=-1)
        
        # Return features for each action head
        return {
            'forward': all_features,
            'rotate': all_features,
            'reproduce': all_features,
            'message': all_features if (params.enable_communication and params.use_message_action) else jnp.zeros(0),
        }
    
    encoder = (lambda: None, encoder_forward)
    
    # Action decoder heads
    action_heads = {
        'forward': layer_sequence((
            linear_layer(in_dim, 2, use_bias=True),
            categorical_sampler_layer(),
        )),
        'rotate': layer_sequence((
            linear_layer(in_dim, 3, use_bias=True),
            categorical_sampler_layer(choices=jnp.array([-1, 0, 1])),
        )),
        'reproduce': layer_sequence((
            linear_layer(in_dim, 2, use_bias=True),
            categorical_sampler_layer(),
        )),
    }
    
    # Add communication head if enabled and message action is used
    if params.enable_communication and params.use_message_action:
        action_heads['message'] = layer_sequence((
            linear_layer(in_dim, params.message_dim, use_bias=True),
            (lambda: None, lambda key, x, state: jnn.tanh(x)),  # Bound messages to [-1, 1]
        ))
    
    decoder_heads = parallel_dict_layer(action_heads)
    
    # Combine into action
    def make_action(x):
        '''Convert decoder outputs to NomNomAction.'''
        # Build action dict with standard fields
        action_dict = {
            'forward': x['forward'],
            'rotate': x['rotate'],
            'reproduce': x['reproduce'],
        }
        # Include message if communication is enabled AND message action is used
        # The decoder only outputs 'message' when both enable_communication=True and use_message_action=True
        if params.enable_communication and params.use_message_action and 'message' in x:
            message = x['message']
            # Check if message is non-empty (not a zero-length array)
            if message.shape == () or (len(message.shape) > 0 and message.shape[-1] > 0):
                action_dict['message'] = message
        # If message not included, NomNomAction will use default None (backward compatible)
        return NomNomAction(**action_dict)
    
    decoder = layer_sequence((
        decoder_heads,
        (lambda: None, make_action),
    ))
    
    model_class = layer_sequence([
        encoder,
        decoder,
    ])
    
    return model_class.init, model_class.forward


def extract_communication_from_action(action, message_dim=4):
    '''
    Extract communication action from model output.
    
    This is a helper for when the model outputs communication actions
    as part of its forward pass.
    
    Note: The base NomNomAction doesn't include communication fields,
    so this is a placeholder for when we extend the action space.
    '''
    from mechagogue.ecology.communication import CommunicationAction
    
    # For now, return zero communication (placeholder)
    # In a full implementation, the model would output communication separately
    return CommunicationAction(
        message=jnp.zeros(message_dim),
        emit_smell=0.0,
        emit_sound=0.0,
    )


def create_rl_nomnom_model(
    use_communication=True,
    view_width=5,
    view_distance=5,
    use_message_action=True,
):
    '''
    Convenience function to create a complete RL-enabled NomNom model.
    
    Args:
        use_communication: Whether to enable communication features (infrastructure)
        view_width: Width of agent's view
        view_distance: Distance of agent's view
        use_message_action: Whether agents output message actions (defaults to True if communication enabled)
    
    Returns:
        model_init, model_forward: Model functions
    '''
    params = NomNomRLModelParams(
        view_width=view_width,
        view_distance=view_distance,
        enable_communication=use_communication,
        use_augmented_obs=use_communication,
        use_message_action=use_message_action if use_communication else False,
    )
    
    return nomnom_linear_model_with_communication(params)


# Backward compatibility: also export a version that matches the original API
def nomnom_linear_model_rl_compatible(params=None):
    '''
    Drop-in replacement for nomnom_linear_model with RL support.
    
    This can be used in place of the original nomnom_linear_model
    in train_nomnom.py without other changes.
    '''
    from dirt.examples.nomnom.nomnom_model import NomNomModelParams
    
    if params is None:
        params = NomNomModelParams()
    
    rl_params = NomNomRLModelParams(
        num_input_classes=params.num_input_classes,
        view_width=params.view_width,
        view_distance=params.view_distance,
        enable_communication=False,  # Disabled for compatibility
        use_augmented_obs=False,
    )
    
    return nomnom_linear_model_with_communication(rl_params)


if __name__ == '__main__':
    # Test the models
    import jax
    
    print("Testing RL-compatible linear model...")
    
    # Test 1: Standard observation (backward compatible)
    print("\n1. Testing with standard observation...")
    model_init, model_forward = create_rl_nomnom_model(use_communication=False)
    
    key = jrng.key(42)
    model_state = model_init(key)
    
    obs = NomNomObservation(
        view=jnp.zeros((5, 5), dtype=jnp.int32),
        energy=jnp.array([1.0]),
    )
    
    action = model_forward(key, obs, model_state)
    print(f"Action: forward={action.forward}, rotate={action.rotate}, reproduce={action.reproduce}")
    
    # Test 2: Augmented observation
    print("\n2. Testing with augmented observation...")
    from mechagogue.ecology.communication import (
        AugmentedObservation,
        LocalAgentInfo,
    )
    
    model_init, model_forward = create_rl_nomnom_model(use_communication=True)
    model_state = model_init(key)
    
    # Create dummy augmented observation
    nearby_agents = jax.vmap(lambda i: LocalAgentInfo(
        relative_x=jnp.array([float(i), float(i)]),
        energy=0.5,
        message=jnp.zeros(4),
        distance=float(i),
        is_valid=(i < 3),
    ))(jnp.arange(8))
    
    aug_obs = AugmentedObservation(
        base_obs=obs,
        nearby_agents=nearby_agents,
        num_nearby=3,
        local_smell=0.0,
        local_sound=0.0,
    )
    
    action = model_forward(key, aug_obs, model_state)
    print(f"Action: forward={action.forward}, rotate={action.rotate}, reproduce={action.reproduce}")
    
    print("\n✓ All tests passed!")

