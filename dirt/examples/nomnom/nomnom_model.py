import jax.random as jrng
import jax.numpy as jnp
import jax.nn as jnn
from jax.tree import map as tree_map

from mechagogue.static import static_data as static_dataclass
from mechagogue.nn.linear import (
    embedding_layer, linear_layer, conv_layer)
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.mlp import mlp
from mechagogue.nn.structure import dict_layer as parallel_dict_layer
from mechagogue.nn.distributions import categorical_sampler_layer
from mechagogue.nn.debug import print_activations_layer

from dirt.envs.nomnom import NomNomObservation, NomNomAction

@static_dataclass
class NomNomModelParams:
    num_input_classes : int = 4
    view_width : int = 5
    view_distance : int = 5

def nomnom_linear_model(params=NomNomModelParams()):
    in_dim = (
        (params.num_input_classes-1) *
        params.view_width *
        params.view_distance +
        1   # energy
    )
    out_dim = 2+3+2
    
    def encoder_forward(x):
        food = (x.view == 1).reshape(-1).astype(jnp.float32)
        players = (x.view == 2).reshape(-1).astype(jnp.float32)
        out_of_bounds = (x.view == 3).reshape(-1).astype(jnp.float32)
        # Ensure energy is exactly 1 dimension - take the first element if it's an array
        energy_flat = jnp.atleast_1d(x.energy).flatten()
        energy = energy_flat[0:1].astype(jnp.float32)  # Take first element, keep as array for concat
        x = jnp.concatenate(
            (food, players, out_of_bounds, energy), axis=-1)
        # Ensure output dimension matches expected in_dim (pad or truncate if needed)
        current_dim = x.shape[-1]
        if current_dim > in_dim:
            x = x[..., :in_dim]
        elif current_dim < in_dim:
            padding = jnp.zeros((in_dim - current_dim,), dtype=jnp.float32)
            x = jnp.concatenate([x, padding], axis=-1)
        return {'forward' : x, 'rotate' : x, 'reproduce' : x}
    
    encoder = (lambda: None, encoder_forward)
    
    decoder_heads = parallel_dict_layer({
        'forward' : layer_sequence((
            linear_layer(in_dim, 2, use_bias=True),
            #print_activations_layer('forward:'),
            categorical_sampler_layer(),
            #print_activations_layer('forward_sample:'),
        )),
        'rotate' : layer_sequence((
            linear_layer(in_dim, 3, use_bias=True),
            #print_activations_layer('rotate:'),
            categorical_sampler_layer(choices=jnp.array([-1,0,1])),
            #print_activations_layer('rotate_sample:'),
        )),
        'reproduce' : layer_sequence((
            linear_layer(in_dim, 2, use_bias=True),
            #print_activations_layer('reproduce:'),
            categorical_sampler_layer(),
            #print_activations_layer('reproduce_sample:'),
        )),
    })
    
    decoder = layer_sequence((
        decoder_heads,
        (lambda: None, lambda x : NomNomAction(**x)),
        #print_activations_layer('action:'),
    ))
    
    model_class = layer_sequence([
        encoder,
        decoder,
    ])
    
    # Extract init and forward methods from the class to return as functions
    return model_class.init, model_class.forward


def mutate(params, mutation_rate=0.1):
    """Apply Gaussian noise to parameters for mutation."""
    return tree_map(lambda p: p + mutation_rate * jrng.normal(jrng.PRNGKey(0), p.shape), params)


def crossover(parent1, parent2):
    """Blend two parent parameter sets to create a child."""
    return tree_map(lambda p1, p2: (p1 + p2) / 2, parent1, parent2)


def nomnom_model(params=NomNomModelParams()):
    
    # utility
    # - make a relu
    relu = (lambda: None, jnn.relu)
    
    # encoder
    #   this section builds the components that convert the inputs to a fixed
    #   hidden dimension that can be processed by the backbone
    # - view encoder
    view_embedding = embedding_layer(params.num_input_classes, 32)
    view_conv1 = conv_layer(32, 64, padding='VALID', use_bias=True)
    view_conv2 = conv_layer(64, 128, padding='VALID', use_bias=True)
    flattened_channels = (params.view_width-4) * (params.view_distance-4) * 128
    # Flatten preserving batch dimensions
    # Handle various input shapes:
    # - (1, 1, 128) -> (1, 128)  [single example, 3D]
    # - (batch, 1, 1, 128) -> (batch, 128)  [batched, 4D]
    # - (batch, 1, 128) -> (batch, 128)  [batched, 3D]
    # - (1, 128) -> (1, 128) (already flat)
    def flatten_view(x):
        x = jnp.asarray(x)
        shape = x.shape
        # If already 2D with correct last dimension, return as-is
        if len(shape) == 2 and shape[-1] == flattened_channels:
            return x
        # For 3D input - could be (batch, spatial, channels) or (H, W, C)
        if len(shape) == 3:
            # If last dimension matches flattened_channels, flatten preceding dimensions
            # This handles both (batch, spatial, channels) and (H, W, channels)
            if shape[-1] == flattened_channels:
                # Flatten all dimensions before the last: (any, any, channels) -> (-1, channels)
                return x.reshape(-1, flattened_channels)
            # Otherwise, compute total size and reshape
            total_size = shape[0] * shape[1] * shape[2]
            if total_size % flattened_channels == 0:
                batch_size = total_size // flattened_channels
                return x.reshape(batch_size, flattened_channels)
            else:
                # Single example case - take first flattened_channels elements
                flat = x.reshape(-1)
                if flat.shape[0] >= flattened_channels:
                    return flat[:flattened_channels].reshape(1, flattened_channels)
                else:
                    # Pad if smaller
                    padding = jnp.zeros(flattened_channels - flat.shape[0], dtype=x.dtype)
                    return jnp.concatenate([flat, padding]).reshape(1, flattened_channels)
        # For 4D input (batch, H, W, C) - preserve batch dimension
        if len(shape) == 4:
            batch_size = shape[0]
            return x.reshape(batch_size, flattened_channels)
        # For other cases, use -1 to compute
        return x.reshape(-1, flattened_channels)
    flatten = (lambda: None, flatten_view)
    view_fc1 = linear_layer(flattened_channels, 128, use_bias=True)
    view_encoder = layer_sequence(
        (view_embedding, view_conv1, relu, view_conv2, flatten, relu, view_fc1)
    )
    
    # - energy encoder
    energy_encoder = layer_sequence((
        (lambda: None, lambda x : x.reshape(-1)),
        mlp(
            hidden_layers=1,
            in_channels=1,
            hidden_channels=32,
            out_channels=128,
        ),
    ))
    
    # - combine the encoders
    #   this first converts an observation object to a dictionary
    #   then passes that dictionary to the corresponding encoders
    #   then sums the output
    def combine_encoders(x=None):
        # Explicitly add the encoder outputs to ensure correct broadcasting
        # Both should be (batch, 128), so this should work
        # x should be a dictionary from parallel_dict_layer with 'view' and 'energy' keys
        if x is None:
            raise ValueError("combine_encoders received None input - check previous layer output")
        if not isinstance(x, dict):
            raise TypeError(f"combine_encoders expected a dictionary, got {type(x)}: {x}")
        if 'view' not in x or 'energy' not in x:
            raise KeyError(f"combine_encoders expected dictionary with 'view' and 'energy' keys, got keys: {list(x.keys()) if isinstance(x, dict) else 'N/A'}")
        view_out = x['view']
        energy_out = x['energy']
        # Ensure both are 2D with shape (batch, 128)
        # If view_out has extra dimensions, flatten them
        if len(view_out.shape) > 2:
            view_out = view_out.reshape(view_out.shape[0], -1)
            # If it's too large, take only the first 128 elements
            if view_out.shape[-1] > 128:
                view_out = view_out[..., :128]
            elif view_out.shape[-1] < 128:
                # Pad if needed
                padding = jnp.zeros((view_out.shape[0], 128 - view_out.shape[-1]), dtype=view_out.dtype)
                view_out = jnp.concatenate([view_out, padding], axis=-1)
        # Same for energy_out
        if len(energy_out.shape) > 2:
            energy_out = energy_out.reshape(energy_out.shape[0], -1)
            if energy_out.shape[-1] > 128:
                energy_out = energy_out[..., :128]
            elif energy_out.shape[-1] < 128:
                padding = jnp.zeros((energy_out.shape[0], 128 - energy_out.shape[-1]), dtype=energy_out.dtype)
                energy_out = jnp.concatenate([energy_out, padding], axis=-1)
        return view_out + energy_out
    
    encoder = layer_sequence((
        (lambda: None, lambda x : {'view': x.view.astype(jnp.int32), 'energy': x.energy}),
        parallel_dict_layer({'view':view_encoder, 'energy':energy_encoder}),
        (lambda: None, combine_encoders),
        relu,
    ))
    
    # backbone
    #   this is a small MLP
    backbone = layer_sequence((
        linear_layer(128, 64, use_bias=True),
        relu,
        linear_layer(64, 32, use_bias=True),
        relu,
        (lambda: None, lambda x : {'forward':x, 'rotate':x, 'reproduce':x}),
    ))
    
    # decoder heads
    #   the decoders convert the output of the MLP to the individual
    #   components of the action
    # - these are three linear layers followed by categorical samplers
    # - ensure outputs are 1D arrays (not 2D) for proper broadcasting in dynamics
    # CRITICAL FIX: Categorical sampler should output (batch,) but if it outputs (batch, classes),
    # we need to apply argmax to get the sampled index. This fixes the broadcasting error.
    def ensure_1d(x):
        # Force 1D output: (batch,) 
        # Handle various input shapes and always return (batch,)
        x = jnp.asarray(x)
        shape = x.shape
        
        # If already 1D, return as-is
        if len(shape) == 1:
            return x
        
        # If 2D with shape (batch, classes), apply argmax to get indices
        if len(shape) == 2:
            if shape[-1] > 1:
                # (batch, classes) - apply argmax to convert to (batch,)
                return jnp.argmax(x, axis=-1)
            else:
                # (batch, 1) - squeeze to (batch,)
                return x.squeeze(-1)
        
        # For 3D input (batch, extra_dim, classes), take the last dimension
        # This handles cases where we have (batch, 128, 2) -> should be (batch, 128)
        # by taking argmax along the last axis
        if len(shape) == 3:
            # (batch, extra_dim, classes) -> (batch, extra_dim) by argmax
            if shape[-1] > 1:
                return jnp.argmax(x, axis=-1)
            else:
                return x.squeeze(-1)
        
        # For higher dimensions, flatten and handle
        # Reshape to (batch, -1) and then apply argmax or squeeze
        x_flat = x.reshape(shape[0], -1)
        if x_flat.shape[-1] > 1:
            return jnp.argmax(x_flat, axis=-1)
        else:
            return x_flat.squeeze(-1)
    
    decoder_heads = parallel_dict_layer({
        'forward' : layer_sequence((
            linear_layer(32, 2, use_bias=True),
            # print_activations_layer('forward:'),
            categorical_sampler_layer(),
            (lambda: None, ensure_1d),
            # print_activations_layer('forward_sample:'),
        )),
        'rotate' : layer_sequence((
            linear_layer(32, 3, use_bias=True),
            #print_activations_layer('rotate:'),
            categorical_sampler_layer(choices=jnp.array([-1,0,1])),
            (lambda: None, ensure_1d),
            #print_activations_layer('rotate_sample:'),
        )),
        'reproduce' : layer_sequence((
            linear_layer(32, 2, use_bias=True),
            #print_activations_layer('reproduce:'),
            categorical_sampler_layer(),
            (lambda: None, ensure_1d),
            #print_activations_layer('reproduce_sample:'),
        )),
    })
    
    # - this runs the three decoders and then combines the result into a
    #   new action
    decoder = layer_sequence((
        decoder_heads,
        (lambda: None, lambda x : NomNomAction(**x)),
        #print_activations_layer('action:'),
    ))
    
    # combine the encoder, backbone and decoder
    model_class = layer_sequence([
        encoder,
        backbone,
        decoder,
    ])
    
    # Extract init and forward methods from the class to return as functions
    return model_class.init, model_class.forward

def test_model(key):
    init_model, model = nomnom_model()
    
    model_state = init_model(key)
    observation = NomNomObservation(
        view=jnp.zeros((5,5), dtype=jnp.int32),
        energy=jnp.array([1.]),
    )
    action = model(key, observation, model_state)
    print(action)

if __name__ == '__main__':
    test_model(jrng.key(12345))
