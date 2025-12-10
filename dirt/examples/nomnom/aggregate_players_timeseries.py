#!/usr/bin/env python3
"""
Aggregate and plot players time-series data from multiple experiments.

Usage:
    python aggregate_players_timeseries.py <output_dir_base> <ablate_id> [--output-dir <output_dir>]

This script:
1. Finds all players_time_series.json files in subdirectories of output_dir_base
2. Aggregates them into four plots:
   - Active players over time (players_time_series.png)
   - Average reward per epoch (avg_reward_per_epoch.png, if RL enabled)
   - Average buffer fullness per epoch (avg_buffer_fullness_per_epoch.png, if RL enabled)
   - Average lifespan per epoch (avg_lifespan_per_epoch.png)
3. Saves the plots to visual_output/ablate_results/{ablate_id}/
"""

import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def find_timeseries_files(output_dir_base):
    """Find all players_time_series.json files in subdirectories."""
    timeseries_files = []
    
    # Handle multiple directories (for combining variants)
    if isinstance(output_dir_base, list):
        dirs = output_dir_base
    else:
        dirs = [output_dir_base]
    
    for base_dir in dirs:
        # Search for players_time_series.json in all subdirectories
        pattern = os.path.join(base_dir, "**", "players_time_series.json")
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract exp_id from the directory structure
                    exp_dir = os.path.dirname(file_path)
                    exp_id = os.path.basename(exp_dir)
                    timeseries_files.append((exp_id, file_path, data, base_dir))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
    
    return timeseries_files


def extract_base_exp_id(exp_id):
    """Extract base exp_id by removing seed suffix (e.g., _seed_1234)."""
    # Remove seed suffix pattern: _seed_<number>
    import re
    base_id = re.sub(r'_seed_\d+$', '', exp_id)
    return base_id


def detect_variant(exp_id):
    """Detect variant type from exp_id: vanilla, rl, or rl_comm."""
    # Use base exp_id to detect variant (strip seed suffix first)
    base_id = extract_base_exp_id(exp_id)
    if base_id.endswith('_rl_comm'):
        return 'rl_comm'
    elif base_id.endswith('_rl'):
        return 'rl'
    else:
        return 'vanilla'


def extract_param_value(exp_id, param_name):
    """Extract parameter value from exp_id."""
    # Use base exp_id to extract param value (strip seed suffix first)
    base_id = extract_base_exp_id(exp_id)
    if not param_name or param_name not in base_id:
        return None
    
    parts = base_id.split('_')
    for i, part in enumerate(parts):
        if part == param_name and i + 1 < len(parts):
            try:
                value = parts[i + 1]
                # Remove RL suffixes if present
                if value.endswith('rl') or value.endswith('comm'):
                    value = value.rstrip('rl').rstrip('comm').rstrip('_')
                return value
            except:
                pass
    return None


def plot_epoch_metric(timeseries_files, output_path, data_key, ylabel, title_base, 
                      param_name=None, use_variant_shapes=False, filter_none=True):
    """
    Generalized function to plot epoch-based metrics (rewards, lifespans, etc.).
    
    Args:
        timeseries_files: List of (exp_id, file_path, data, base_dir) tuples
        output_path: Path to save the plot
        data_key: Key in JSON data to extract (e.g., 'avg_reward_per_epoch', 'avg_lifespan_per_epoch')
        ylabel: Y-axis label
        title_base: Base title (e.g., 'Average Reward per Epoch', 'Average Lifespan per Epoch')
        param_name: Parameter name for labeling
        use_variant_shapes: Whether to use different markers for variants
        filter_none: Whether to filter out None values (for lifespans where None means no deaths)
    """
    if not timeseries_files:
        print(f"No time-series files found to plot {data_key}.")
        return
    
    # Filter to only experiments with the requested data
    metric_files = []
    for item in timeseries_files:
        if len(item) == 4:  # (exp_id, file_path, data, base_dir)
            exp_id, file_path, data, base_dir = item
        else:  # backward compatibility
            exp_id, file_path, data = item
            base_dir = None
        
        metric_data = data.get(data_key, [])
        # Filter based on data type
        if data_key == 'avg_reward_per_epoch':
            # Only include if there's actual reward data (not all zeros or empty)
            if metric_data and any(r != 0.0 for r in metric_data):
                metric_files.append(item)
        elif data_key == 'avg_lifespan_per_epoch':
            # Include if there's any non-None data
            if metric_data and any(l is not None for l in metric_data):
                metric_files.append(item)
        else:
            # Generic: include if data exists
            if metric_data:
                metric_files.append(item)
    
    if not metric_files:
        print(f"No {data_key} data found in any experiments. Skipping plot.")
        return
    
    # Group experiments by base_exp_id (to aggregate across seeds)
    from collections import defaultdict
    grouped_experiments = defaultdict(list)
    for item in metric_files:
        if len(item) == 4:  # (exp_id, file_path, data, base_dir)
            exp_id, file_path, data, base_dir = item
        else:  # backward compatibility
            exp_id, file_path, data = item
            base_dir = None
        base_exp_id = extract_base_exp_id(exp_id)
        grouped_experiments[base_exp_id].append((exp_id, file_path, data, base_dir))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Variant markers: circle for vanilla, 'x' for RL, 's' (square) for RL+comm
    variant_markers = {
        'vanilla': 'o',
        'rl': 'x',
        'rl_comm': 's'
    }
    
    # Group by parameter value to use same color
    if use_variant_shapes and param_name:
        # Extract unique parameter values from base_exp_ids
        param_values = set()
        for base_exp_id in grouped_experiments.keys():
            value = extract_param_value(base_exp_id, param_name)
            if value:
                param_values.add(value)
        
        param_values = sorted(param_values, key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x)
        # Use colormap for parameter values
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))
        value_to_color = {val: colors[i] for i, val in enumerate(param_values)}
    else:
        # Use a colormap for different experiments
        # Convert to list to ensure consistent indexing
        base_exp_ids_list = list(grouped_experiments.keys())
        num_groups = len(base_exp_ids_list)
        colors = plt.cm.tab10(np.linspace(0, 1, num_groups))
        value_to_color = None
    
    for idx, (base_exp_id, items) in enumerate(grouped_experiments.items()):
        # Ensure idx is within bounds (safety check)
        color_idx = idx % len(colors) if len(colors) > 0 else 0
        
        # Get all metric arrays for this base experiment (across seeds)
        all_metric_arrays = []
        all_epochs = []
        
        for item in items:
            if len(item) == 4:  # (exp_id, file_path, data, base_dir)
                exp_id, file_path, data, base_dir = item
            else:  # backward compatibility
                exp_id, file_path, data = item
                base_dir = None
            
            metric_data = data.get(data_key, [])
            if not metric_data:
                continue
            
            # Handle None values (for lifespans where None means no deaths in that epoch)
            if filter_none:
                # For lifespans, we need to preserve the epoch structure even if some are None
                # Convert None to NaN but keep all epochs to maintain alignment
                # This ensures experiments that end early are still included
                metric_array = np.array([np.nan if v is None else v for v in metric_data])
                epochs = np.arange(len(metric_array))  # Keep all epochs, even if some are NaN
                
                # Only skip if ALL values are None/NaN (no data at all)
                if np.all(np.isnan(metric_array)):
                    continue
            else:
                # Convert None to NaN for plotting (will be skipped)
                metric_array = np.array([np.nan if v is None else v for v in metric_data])
                epochs = np.arange(len(metric_array))
            
            all_metric_arrays.append(metric_array)
            all_epochs.append(epochs)
        
        if not all_metric_arrays:
            continue
        
        # Compute mean and std across seeds
        # First, align all arrays to the same epoch indices
        if len(all_metric_arrays) > 1:
            # Find common epoch range - use the full range from all experiments
            # This ensures experiments that ended early are still included
            min_epoch = min(ep[0] for ep in all_epochs if len(ep) > 0)
            max_epoch = max(ep[-1] for ep in all_epochs if len(ep) > 0)
            common_epochs = np.arange(min_epoch, max_epoch + 1)
            
            # Interpolate/extend arrays to common epochs
            aligned_arrays = []
            for metric_array, epochs in zip(all_metric_arrays, all_epochs):
                aligned = np.full(len(common_epochs), np.nan)
                for i, epoch in enumerate(common_epochs):
                    # Check if this epoch exists in the original metric_array
                    # For the new approach (filter_none=True with full structure):
                    #   epochs is [0, 1, 2, ..., N-1] and metric_array has length N
                    #   We can directly index if epoch < len(metric_array)
                    if epoch < len(metric_array):
                        # Direct indexing since epochs should be sequential starting from 0
                        aligned[i] = metric_array[epoch]  # Will be NaN if original was None/NaN
                    # Also handle case where epochs might be sparse (backward compatibility)
                    elif len(epochs) > 0 and epoch >= epochs[0] and epoch <= epochs[-1]:
                        # Check if epoch is in the sparse epochs array
                        if epoch in epochs:
                            epoch_idx = np.where(epochs == epoch)[0][0]
                            if epoch_idx < len(metric_array):
                                aligned[i] = metric_array[epoch_idx]
                aligned_arrays.append(aligned)
            
            # Compute mean and std
            stacked = np.stack(aligned_arrays, axis=0)
            mean_array = np.nanmean(stacked, axis=0)
            std_array = np.nanstd(stacked, axis=0)
            
            # Filter out NaN values
            valid_mask = ~np.isnan(mean_array)
            mean_array = mean_array[valid_mask]
            std_array = std_array[valid_mask]
            epochs = common_epochs[valid_mask]
            
            show_std = len(all_metric_arrays) > 5  # Show std shaded region when TRIES > 5
        else:
            # Single run, no aggregation needed
            mean_array = all_metric_arrays[0]
            epochs = all_epochs[0]
            std_array = None
            show_std = False
        
        # Detect variant and determine marker
        variant = detect_variant(base_exp_id)
        marker = variant_markers.get(variant, 'o') if use_variant_shapes else None
        
        # Extract parameter value and label
        param_value = extract_param_value(base_exp_id, param_name)
        if param_value:
            variant_suffix = ''
            if use_variant_shapes:
                if variant == 'rl':
                    variant_suffix = ' (RL)'
                elif variant == 'rl_comm':
                    variant_suffix = ' (RL+Comm)'
                else:
                    variant_suffix = ' (Vanilla)'
            label = f"{param_name}={param_value}{variant_suffix}"
            if len(all_metric_arrays) > 1:
                label += f" (n={len(all_metric_arrays)})"
        else:
            label = base_exp_id
            if len(all_metric_arrays) > 1:
                label += f" (n={len(all_metric_arrays)})"
        
        # Determine color
        if use_variant_shapes and value_to_color and param_value:
            color = value_to_color[param_value]
        else:
            color = colors[color_idx]
        
        # Plot mean with std shaded region if applicable
        if show_std and std_array is not None:
            ax.fill_between(epochs, mean_array - std_array, mean_array + std_array, 
                          color=color, alpha=0.2, label='_nolegend_')
        
        # Plot mean line
        if marker:
            ax.plot(epochs, mean_array, label=label, color=color, 
                   alpha=0.7, linewidth=1.5, marker=marker, markersize=4, markevery=max(1, len(epochs)//20))
        else:
            ax.plot(epochs, mean_array, label=label, color=color, alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if param_name:
        title = f'{title_base} - {param_name} Ablation'
        if use_variant_shapes:
            title += ' (All Variants)'
    else:
        title = f'{title_base} - All Experiments'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {data_key} plot to: {output_path}")
    
    plt.close(fig)


def plot_rewards(timeseries_files, output_path, param_name=None, use_variant_shapes=False):
    """Plot average reward per epoch as time series (one value per epoch for each experiment)."""
    plot_epoch_metric(
        timeseries_files, output_path,
        data_key='avg_reward_per_epoch',
        ylabel='Average Reward per Agent per Step',
        title_base='Average Reward per Epoch',
        param_name=param_name,
        use_variant_shapes=use_variant_shapes,
        filter_none=False  # Rewards don't have None values
    )


def plot_lifespans(timeseries_files, output_path, param_name=None, use_variant_shapes=False):
    """Plot average lifespan per epoch as time series (one value per epoch for each experiment)."""
    plot_epoch_metric(
        timeseries_files, output_path,
        data_key='avg_lifespan_per_epoch',
        ylabel='Average Lifespan (steps)',
        title_base='Average Lifespan per Epoch',
        param_name=param_name,
        use_variant_shapes=use_variant_shapes,
        filter_none=True  # Lifespans can be None when no deaths occur
    )


def plot_buffer_fullness(timeseries_files, output_path, param_name=None, use_variant_shapes=False):
    """Plot average buffer fullness per epoch as time series (one value per epoch for each experiment)."""
    plot_epoch_metric(
        timeseries_files, output_path,
        data_key='avg_buffer_fullness_per_epoch',
        ylabel='Average Buffer Fullness',
        title_base='Average Buffer Fullness per Epoch',
        param_name=param_name,
        use_variant_shapes=use_variant_shapes,
        filter_none=False  # Buffer fullness doesn't have None values
    )


def plot_timeseries(timeseries_files, output_path, param_name=None, use_variant_shapes=False):
    """Plot all time series on the same graph."""
    if not timeseries_files:
        print("No time-series files found to plot.")
        return
    
    # Group experiments by base_exp_id (to aggregate across seeds)
    from collections import defaultdict
    grouped_experiments = defaultdict(list)
    for item in timeseries_files:
        if len(item) == 4:  # (exp_id, file_path, data, base_dir)
            exp_id, file_path, data, base_dir = item
        else:  # backward compatibility
            exp_id, file_path, data = item
            base_dir = None
        base_exp_id = extract_base_exp_id(exp_id)
        grouped_experiments[base_exp_id].append((exp_id, file_path, data, base_dir))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Variant markers: circle for vanilla, 'x' for RL, 's' (square) for RL+comm
    variant_markers = {
        'vanilla': 'o',
        'rl': 'x',
        'rl_comm': 's'
    }
    
    # Group by parameter value to use same color
    if use_variant_shapes and param_name:
        # Extract unique parameter values from base_exp_ids
        param_values = set()
        for base_exp_id in grouped_experiments.keys():
            value = extract_param_value(base_exp_id, param_name)
            if value:
                param_values.add(value)
        
        param_values = sorted(param_values, key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x)
        # Use colormap for parameter values
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))
        value_to_color = {val: colors[i] for i, val in enumerate(param_values)}
    else:
        # Use a colormap for different experiments
        # Convert to list to ensure consistent indexing
        base_exp_ids_list = list(grouped_experiments.keys())
        num_groups = len(base_exp_ids_list)
        colors = plt.cm.tab10(np.linspace(0, 1, num_groups))
        value_to_color = None
    
    for idx, (base_exp_id, items) in enumerate(grouped_experiments.items()):
        # Ensure idx is within bounds (safety check)
        color_idx = idx % len(colors) if len(colors) > 0 else 0
        
        # Get all player arrays for this base experiment (across seeds)
        all_player_arrays = []
        all_timesteps = []
        
        for item in items:
            if len(item) == 4:  # (exp_id, file_path, data, base_dir)
                exp_id, file_path, data, base_dir = item
            else:  # backward compatibility
                exp_id, file_path, data = item
                base_dir = None
            
            active_players = data.get('active_players_per_step', [])
            if not active_players:
                continue
            
            # Convert to numpy array for easier handling
            players_array = np.array(active_players)
            timesteps = np.arange(len(players_array))
            
            # Downsample to every 10 steps to reduce plot density
            downsample_step = 10
            players_array = players_array[::downsample_step]
            timesteps = timesteps[::downsample_step]
            
            all_player_arrays.append(players_array)
            all_timesteps.append(timesteps)
        
        if not all_player_arrays:
            continue
        
        # Compute mean and std across seeds
        # First, align all arrays to the same timestep indices
        if len(all_player_arrays) > 1:
            # Find common timestep range
            min_timestep = min(ts[0] for ts in all_timesteps if len(ts) > 0)
            max_timestep = max(ts[-1] for ts in all_timesteps if len(ts) > 0)
            # Use the same downsampling
            common_timesteps = np.arange(min_timestep, max_timestep + 1, 10)
            
            # Interpolate/extend arrays to common timesteps
            aligned_arrays = []
            for players_array, timesteps in zip(all_player_arrays, all_timesteps):
                aligned = np.full(len(common_timesteps), np.nan)
                for i, timestep in enumerate(common_timesteps):
                    if timestep in timesteps:
                        timestep_idx = np.where(timesteps == timestep)[0][0]
                        aligned[i] = players_array[timestep_idx]
                aligned_arrays.append(aligned)
            
            # Compute mean and std
            stacked = np.stack(aligned_arrays, axis=0)
            mean_array = np.nanmean(stacked, axis=0)
            std_array = np.nanstd(stacked, axis=0)
            
            # Filter out NaN values
            valid_mask = ~np.isnan(mean_array)
            mean_array = mean_array[valid_mask]
            std_array = std_array[valid_mask]
            timesteps = common_timesteps[valid_mask]
            
            show_std = len(all_player_arrays) > 5  # Show std shaded region when TRIES > 5
        else:
            # Single run, no aggregation needed
            mean_array = all_player_arrays[0]
            timesteps = all_timesteps[0]
            std_array = None
            show_std = False
        
        # Detect variant and determine marker
        variant = detect_variant(base_exp_id)
        marker = variant_markers.get(variant, 'o') if use_variant_shapes else None
        
        # Extract parameter value and label
        param_value = extract_param_value(base_exp_id, param_name)
        if param_value:
            variant_suffix = ''
            if use_variant_shapes:
                if variant == 'rl':
                    variant_suffix = ' (RL)'
                elif variant == 'rl_comm':
                    variant_suffix = ' (RL+Comm)'
                else:
                    variant_suffix = ' (Vanilla)'
            label = f"{param_name}={param_value}{variant_suffix}"
            if len(all_player_arrays) > 1:
                label += f" (n={len(all_player_arrays)})"
        else:
            label = base_exp_id
            if len(all_player_arrays) > 1:
                label += f" (n={len(all_player_arrays)})"
        
        # Determine color
        if use_variant_shapes and value_to_color and param_value:
            color = value_to_color[param_value]
        else:
            color = colors[color_idx]
        
        # Plot mean with std shaded region if applicable
        if show_std and std_array is not None:
            ax.fill_between(timesteps, mean_array - std_array, mean_array + std_array, 
                          color=color, alpha=0.2, label='_nolegend_')
        
        # Plot mean line
        if marker:
            ax.plot(timesteps, mean_array, label=label, color=color, 
                   alpha=0.7, linewidth=1.5, marker=marker, markersize=4, markevery=max(1, len(timesteps)//50))
        else:
            ax.plot(timesteps, mean_array, label=label, color=color, alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Active Players', fontsize=12)
    
    if param_name:
        title = f'Active Players Over Time - {param_name} Ablation'
        if use_variant_shapes:
            title += ' (All Variants)'
    else:
        title = 'Active Players Over Time - All Experiments'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved aggregated time-series plot to: {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and plot players time-series data from multiple experiments'
    )
    parser.add_argument('output_dir_base', type=str, nargs='+',
                       help='Base directory(ies) containing experiment subdirectories (can specify multiple for combining variants)')
    parser.add_argument('ablate_id', type=str,
                       help='Ablation run ID for output directory naming')
    parser.add_argument('--param-name', type=str, default=None,
                       help='Parameter name being ablated (for better labels)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plot (default: visual_output/ablate_results/{ablate_id})')
    parser.add_argument('--use-variant-shapes', action='store_true',
                       help='Use different markers for variants (circle=vanilla, x=RL, square=RL+comm)')
    parser.add_argument('--rl-enabled', action='store_true',
                       help='Indicate that RL experiments are present (enables reward plotting)')
    
    args = parser.parse_args()
    
    # Find all time-series files
    if len(args.output_dir_base) == 1:
        print(f"Searching for time-series files in: {args.output_dir_base[0]}")
    else:
        print(f"Searching for time-series files in {len(args.output_dir_base)} directories:")
        for dir in args.output_dir_base:
            print(f"  - {dir}")
    
    timeseries_files = find_timeseries_files(args.output_dir_base)
    
    if not timeseries_files:
        dirs_str = ', '.join(args.output_dir_base) if isinstance(args.output_dir_base, list) else args.output_dir_base
        print(f"No players_time_series.json files found in {dirs_str}")
        return
    
    print(f"Found {len(timeseries_files)} time-series files:")
    for item in timeseries_files:
        if len(item) == 4:
            exp_id, file_path, _, _ = item
        else:
            exp_id, file_path, _ = item
        print(f"  - {exp_id}: {file_path}")
    
    # Determine output path
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Use the same structure as visualize_nomnom.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "visual_output", "ablate_results", args.ablate_id)
    
    players_output_path = os.path.join(output_dir, "players_time_series.png")
    rewards_output_path = os.path.join(output_dir, "avg_reward_per_epoch.png")
    buffer_fullness_output_path = os.path.join(output_dir, "avg_buffer_fullness_per_epoch.png")
    lifespans_output_path = os.path.join(output_dir, "avg_lifespan_per_epoch.png")
    
    # Plot aggregated data for active players
    plot_timeseries(timeseries_files, players_output_path, param_name=args.param_name, 
                   use_variant_shapes=args.use_variant_shapes)
    
    # Plot aggregated data for average rewards (only if RL is enabled)
    plots_saved = [players_output_path]
    if args.rl_enabled:
        plot_rewards(timeseries_files, rewards_output_path, param_name=args.param_name, 
                    use_variant_shapes=args.use_variant_shapes)
        plots_saved.append(rewards_output_path)
        
        # Plot aggregated data for average buffer fullness (only if RL is enabled)
        plot_buffer_fullness(timeseries_files, buffer_fullness_output_path, param_name=args.param_name, 
                            use_variant_shapes=args.use_variant_shapes)
        plots_saved.append(buffer_fullness_output_path)
    
    # Plot aggregated data for average lifespans (always enabled if data exists)
    plot_lifespans(timeseries_files, lifespans_output_path, param_name=args.param_name, 
                  use_variant_shapes=args.use_variant_shapes)
    plots_saved.append(lifespans_output_path)
    
    print(f"\n✅ Aggregation complete.")
    for plot_path in plots_saved:
        print(f"   Plot saved to: {plot_path}")
    if not args.rl_enabled:
        print(f"   (Rewards plot skipped - RL not enabled)")


if __name__ == '__main__':
    main()

