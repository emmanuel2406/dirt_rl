#!/usr/bin/env python3
"""
Aggregate and plot players time-series data from multiple experiments.

Usage:
    python aggregate_players_timeseries.py <output_dir_base> <ablate_id> [--output-dir <output_dir>]

This script:
1. Finds all players_time_series.json files in subdirectories of output_dir_base
2. Aggregates them into a single plot
3. Saves the plot to visual_output/ablate_results/{ablate_id}/players_time_series.png
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


def detect_variant(exp_id):
    """Detect variant type from exp_id: vanilla, rl, or rl_comm."""
    if exp_id.endswith('_rl_comm'):
        return 'rl_comm'
    elif exp_id.endswith('_rl'):
        return 'rl'
    else:
        return 'vanilla'


def extract_param_value(exp_id, param_name):
    """Extract parameter value from exp_id."""
    if not param_name or param_name not in exp_id:
        return None
    
    parts = exp_id.split('_')
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


def plot_timeseries(timeseries_files, output_path, param_name=None, use_variant_shapes=False):
    """Plot all time series on the same graph."""
    if not timeseries_files:
        print("No time-series files found to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Variant markers: circle for vanilla, 'x' for RL, 's' (square) for RL+comm
    variant_markers = {
        'vanilla': 'o',
        'rl': 'x',
        'rl_comm': 's'
    }
    
    # Group by parameter value to use same color
    if use_variant_shapes and param_name:
        # Extract unique parameter values
        param_values = set()
        for item in timeseries_files:
            if len(item) == 4:  # (exp_id, file_path, data, base_dir)
                exp_id = item[0]
            else:  # backward compatibility
                exp_id = item[0]
            value = extract_param_value(exp_id, param_name)
            if value:
                param_values.add(value)
        
        param_values = sorted(param_values, key=lambda x: float(x) if x.replace('.', '').replace('-', '').isdigit() else x)
        # Use colormap for parameter values
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_values)))
        value_to_color = {val: colors[i] for i, val in enumerate(param_values)}
    else:
        # Use a colormap for different experiments
        colors = plt.cm.tab10(np.linspace(0, 1, len(timeseries_files)))
        value_to_color = None
    
    for idx, item in enumerate(timeseries_files):
        if len(item) == 4:  # (exp_id, file_path, data, base_dir)
            exp_id, file_path, data, base_dir = item
        else:  # backward compatibility
            exp_id, file_path, data = item
            base_dir = None
        
        active_players = data.get('active_players_per_step', [])
        if not active_players:
            print(f"Warning: No data in {exp_id}")
            continue
        
        # Convert to numpy array for easier handling
        players_array = np.array(active_players)
        timesteps = np.arange(len(players_array))
        
        # Detect variant and determine marker
        variant = detect_variant(exp_id)
        marker = variant_markers.get(variant, 'o') if use_variant_shapes else None
        
        # Extract parameter value and label
        param_value = extract_param_value(exp_id, param_name)
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
        else:
            label = exp_id
        
        # Determine color
        if use_variant_shapes and value_to_color and param_value:
            color = value_to_color[param_value]
        else:
            color = colors[idx]
        
        # Plot with or without marker
        if marker:
            ax.plot(timesteps, players_array, label=label, color=color, 
                   alpha=0.7, linewidth=1.5, marker=marker, markersize=4, markevery=max(1, len(timesteps)//50))
        else:
            ax.plot(timesteps, players_array, label=label, color=color, alpha=0.5, linewidth=1.5)
    
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
    
    output_path = os.path.join(output_dir, "players_time_series.png")
    
    # Plot aggregated data
    plot_timeseries(timeseries_files, output_path, param_name=args.param_name, 
                   use_variant_shapes=args.use_variant_shapes)
    
    print(f"\n✅ Aggregation complete. Plot saved to: {output_path}")


if __name__ == '__main__':
    main()

