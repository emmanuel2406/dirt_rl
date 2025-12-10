import os
import argparse
import subprocess
import atexit
from collections import Counter

import numpy as np

import jax.numpy as jnp

from dirt.visualization.viewer import Viewer
from dirt.examples.nomnom.train_nomnom import NomNomTrainParams, NomNomReport
from mechagogue.serial import load_example_data

# Matplotlib backend for headless, file-based visualization
import matplotlib
matplotlib.use('Agg')  # no GUI
import matplotlib.pyplot as plt

def terrain_texture(report, shape, display_mode):
    th, tw = shape
    food_grid = report.food_grid
    world_size = food_grid.shape
    h, w = world_size
    assert th % h == 0
    assert tw % w == 0
    
    ry = th//h 
    rx = tw//w
    
    texture = food_grid.astype(jnp.uint8) * 128 + 127
    texture = jnp.repeat(texture, ry, axis=0)
    texture = jnp.repeat(texture, rx, axis=1)
    texture = jnp.repeat(texture[:,:,None], 3, axis=2)
    return np.array(texture)

def make_get_player_energy(max_energy):
    def get_player_energy(report):
        return report.player_energy / max_energy
    return get_player_energy

def make_get_terrain_map(world_size):
    def get_terrain_map(report):
        return jnp.zeros(world_size)
    return get_terrain_map

def start_viewer(output_directory, max_frames=None):
    print(f"Loading data from: {output_directory}")
    report_paths = sorted([
        f'{output_directory}/{file_path}'
        for file_path in os.listdir(output_directory)
        if file_path.startswith('report') and file_path.endswith('.state')
    ])
    
    if not report_paths:
        raise ValueError(f"No report files found in {output_directory}")
    
    print(f"Found {len(report_paths)} report files")

    # Load the first report to infer world_size and (optionally) max_energy.
    # This is more robust than relying on train_params.state, which can have
    # PyTree compatibility issues if the training params class has changed.
    # Note: train_params.state is now stored inside per‑experiment subdirectories
    # under the chosen output root (e.g. linear_model/{exp_id}/ or
    # rl_experiments/linear_model_rl/{exp_id}/), but we infer metadata from
    # reports instead so that both evolution-only and RL runs are supported
    # without caring about the specific directory prefix.
    first_report_path = report_paths[0]
    print(f"Loading first report for metadata: {first_report_path}")
    first_report = load_example_data(NomNomReport(), first_report_path)

    # Infer world_size from the food grid in the report.
    # Reports are typically shaped (T, H, W); the viewer expects a 2D (H, W)
    # terrain map, so we take only the spatial dimensions.
    food_shape = first_report.food_grid.shape
    if len(food_shape) >= 2:
        world_size = food_shape[-2:]
    else:
        # Fallback: if shape is unexpected, treat entire shape as (H, W)
        world_size = food_shape
    print(f"Inferred world size from report (H, W): {world_size}")

    # Infer max_energy directly from the first report.
    # Use the maximum player energy observed as a proxy.
    # If all energies are zero, default to 1.0 to avoid division by zero.
    max_energy_observed = float(jnp.max(first_report.player_energy))
    if max_energy_observed <= 0:
        max_energy_observed = 1.0
    params_max_energy = max_energy_observed
    print(f"Inferred max_energy from first report: {params_max_energy}")

    # Create get_player_energy function with chosen max_energy
    get_player_energy_func = make_get_player_energy(params_max_energy)
    
    # Create get_terrain_map function with world_size
    get_terrain_map_func = make_get_terrain_map(world_size)
    
    # Check for display - required for GLFW/OpenGL
    # If not set, automatically start Xvfb (like cleanrl does)
    display = os.environ.get('DISPLAY')
    xvfb_process = None
    
    if not display:
        print("DISPLAY not set. Attempting to start Xvfb (virtual framebuffer)...")
        try:
            # Start Xvfb on display :1 with 1024x768x24
            xvfb_process = subprocess.Popen(
                ['Xvfb', ':1', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX', '+render', '-noreset'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Give it more time to start
            import time
            time.sleep(1.0)
            
            # Check if it's still running (didn't crash)
            if xvfb_process.poll() is None:
                os.environ['DISPLAY'] = ':1'
                display = ':1'
                print(f"Xvfb started successfully on {display}")
                
                # Set environment variables for headless OpenGL rendering
                # Force software rendering to avoid GPU driver issues
                os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
                os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
                
                # Register cleanup function
                def cleanup_xvfb():
                    if xvfb_process and xvfb_process.poll() is None:
                        xvfb_process.terminate()
                        xvfb_process.wait()
                atexit.register(cleanup_xvfb)
            else:
                raise RuntimeError("Xvfb failed to start")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"\nFailed to start Xvfb: {e}")
            print("\n" + "="*70)
            print("ERROR: DISPLAY environment variable not set and Xvfb unavailable!")
            print("="*70)
            print("\nThe visualization requires a display to run. Options:")
            print("\n1. Install Xvfb: sudo apt-get install xvfb (or equivalent)")
            print("2. Use X11 forwarding (if SSH'ing): ssh -X user@server")
            print("3. Use xvfb-run: xvfb-run -a python visualize_nomnom.py <output_directory>")
            print("4. Set DISPLAY manually: export DISPLAY=:0.0")
            print("\n" + "="*70)
            raise RuntimeError("DISPLAY environment variable not set. Cannot initialize OpenGL/GLFW viewer.")
    
    print(f"DISPLAY: {display}")
    
    # Set OpenGL environment variables for headless rendering
    if 'LIBGL_ALWAYS_SOFTWARE' not in os.environ:
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    if 'GALLIUM_DRIVER' not in os.environ:
        os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
    
    print("Initializing Viewer...")
    print("(Note: If this segfaults, it may be an OpenGL/GLFW driver issue)")
    if max_frames is not None:
        print(f"Viewer will run for at most {max_frames} frames (headless/Slurm-friendly).")
        print("Viewer will automatically advance steps (auto_step=True).")
    
    # Note: Segfaults in native code (GLFW/OpenGL) can't be caught with try-except
    # If this crashes, it's likely a display/OpenGL driver issue
    viewer = Viewer(
        NomNomReport(),
        report_paths,
        world_size,
        window_size=(1024, 1024),
        get_player_energy=get_player_energy_func,
        get_terrain_map=get_terrain_map_func,
        get_terrain_texture=terrain_texture,
    )
    print("Viewer initialized successfully. Starting...")
    # On Slurm/headless, we want the simulation to progress automatically
    # rather than waiting for key presses, so we use auto_step=True.
    viewer.start(max_frames=max_frames, auto_step=True)


def render_with_matplotlib(output_directory, max_frames=None, step_stride=1, experiment_name=None):
    """Headless visualization that saves frames as PNGs using matplotlib.

    This avoids any OpenGL/GLFW requirements and works well on Slurm/batch.
    
    Args:
        output_directory: Directory containing report files
        max_frames: Maximum number of frames to render
        step_stride: Save every Nth step
        experiment_name: Experiment name used for subdirectory and title (default: None, uses output_directory basename)
    """
    print(f"[matplotlib] Loading data from: {output_directory}")
    report_paths = sorted([
        f'{output_directory}/{file_path}'
        for file_path in os.listdir(output_directory)
        if file_path.startswith('report') and file_path.endswith('.state')
    ])

    if not report_paths:
        raise ValueError(f"No report files found in {output_directory}")

    print(f"[matplotlib] Found {len(report_paths)} report files")

    # Determine the base directory for visual output
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # visual_output is at dirt/examples/nomnom/visual_output (same directory as this script)
    base_visual_output = os.path.join(script_dir, "visual_output")
    
    # Use provided experiment name, or default to output_directory basename
    if experiment_name is None:
        experiment_name = os.path.basename(os.path.abspath(output_directory))
    
    images_dir = os.path.join(base_visual_output, experiment_name)
    os.makedirs(images_dir, exist_ok=True)
    print(f"[matplotlib] Saving frames into: {images_dir}")

    # frame_idx increments by 1 for each saved frame (consecutive numbering for proper sorting)
    # step_stride controls which steps from the data are processed (e.g., every 100th step)
    # The actual step index 't' is shown in the plot title
    frame_idx = 0
    # Track the last valid food grid to handle cases where food_grid_stride > 1
    # (when food_grid_stride > 1, non-strided steps store zeros to save memory)
    last_valid_fg = None
    for block_idx, path in enumerate(report_paths):
        print(f"[matplotlib] Loading report block: {path}")
        # Use non‑strict loading to tolerate legacy report files that may have
        # additional fields compared to the current NomNomReport definition.
        # Extra leaves are discarded by the relaxed loader in
        # mechagogue.serial.unpack_tree_data.
        reports = load_example_data(NomNomReport(), path, strict=False)

        food_grid = reports.food_grid
        players = reports.players

        T = food_grid.shape[0]
        
        # Infer food_grid_stride by finding the pattern of non-zero food_grids
        # This helps us understand which steps have valid data
        valid_food_grid_indices = []
        for i in range(min(500, T)):  # Check first 500 steps to infer pattern
            if np.sum(np.array(food_grid[i])) > 0:
                valid_food_grid_indices.append(i)
        
        inferred_stride = None
        if len(valid_food_grid_indices) >= 2:
            # Find the most common interval between valid food_grids
            intervals = [valid_food_grid_indices[i+1] - valid_food_grid_indices[i] 
                        for i in range(len(valid_food_grid_indices) - 1)]
            if intervals:
                # Use the most common interval as the inferred stride
                most_common_interval = Counter(intervals).most_common(1)[0][0]
                inferred_stride = most_common_interval
                print(f"[matplotlib] Inferred food_grid_stride: {inferred_stride} (from pattern: {intervals[:10]}...)")
        
        # Process every step_stride-th step (e.g., if step_stride=100, process steps 0, 100, 200, ...)
        for t in range(0, T, step_stride):
            if max_frames is not None and frame_idx >= max_frames:
                print(f"[matplotlib] Reached max_frames={max_frames}, stopping.")
                return

            fg = np.array(food_grid[t])
            # If food_grid was stored with stride > 1, non-strided steps contain zeros
            # Find the nearest valid food_grid (preferring the most recent one)
            if np.sum(fg) == 0:
                found_valid = False
                
                # If we inferred the stride, use it to find the nearest valid food_grid more efficiently
                if inferred_stride is not None and inferred_stride > 1:
                    # Round down to nearest multiple of stride
                    nearest_stride_index = (t // inferred_stride) * inferred_stride
                    # Check a few candidates around the expected stride-aligned index
                    candidates = [nearest_stride_index]
                    for offset in [-inferred_stride, inferred_stride]:
                        candidate = nearest_stride_index + offset
                        if 0 <= candidate < T:
                            candidates.append(candidate)
                    
                    # Try candidates in order: nearest stride-aligned, then nearby ones
                    for candidate_t in candidates:
                        if 0 <= candidate_t < T:
                            scan_fg = np.array(food_grid[candidate_t])
                            if np.sum(scan_fg) > 0:
                                fg = scan_fg
                                found_valid = True
                                break
                
                # If stride-based lookup didn't work, scan backwards (most recent valid)
                if not found_valid:
                    for scan_t in range(t - 1, max(-1, t - 200), -1):  # Scan up to 200 steps back
                        scan_fg = np.array(food_grid[scan_t])
                        if np.sum(scan_fg) > 0:
                            fg = scan_fg
                            found_valid = True
                            break
                
                # If no valid food_grid found backwards, try scanning forward (for very early steps)
                if not found_valid:
                    for scan_t in range(t + 1, min(t + 200, T)):  # Scan up to 200 steps ahead
                        scan_fg = np.array(food_grid[scan_t])
                        if np.sum(scan_fg) > 0:
                            fg = scan_fg
                            found_valid = True
                            break
                
                # Update last_valid_fg for future reference
                if found_valid:
                    last_valid_fg = fg.copy()
                elif last_valid_fg is not None:
                    # Fallback: use last valid food grid we saw (from previous block or earlier)
                    fg = last_valid_fg
            else:
                # Update last_valid_fg when we encounter a non-zero food grid
                last_valid_fg = fg.copy()
            pl = np.array(players[t])
            
            # Get player positions and other data
            # Handle both time-indexed arrays and direct access
            try:
                if hasattr(reports, 'player_x') and reports.player_x is not False:
                    px_data = np.array(reports.player_x)
                    # Check if it has a time dimension
                    if len(px_data.shape) == 3:  # (T, num_players, 2)
                        player_x = px_data[t]
                    elif len(px_data.shape) == 2:  # (num_players, 2) - same for all steps
                        player_x = px_data
                    else:
                        player_x = None
                        if frame_idx == 0:  # Debug on first frame
                            print(f"[matplotlib] Warning: player_x has unexpected shape: {px_data.shape}")
                else:
                    player_x = None
                    if frame_idx == 0:  # Debug on first frame
                        print(f"[matplotlib] Warning: player_x not available in report")
            except Exception as e:
                player_x = None
                if frame_idx == 0:  # Debug on first frame
                    print(f"[matplotlib] Warning: Error loading player_x: {e}")
                
            try:
                if hasattr(reports, 'player_energy') and reports.player_energy is not False:
                    pe_data = np.array(reports.player_energy)
                    if len(pe_data.shape) == 2:  # (T, num_players)
                        player_energy = pe_data[t]
                    elif len(pe_data.shape) == 1:  # (num_players) - same for all steps
                        player_energy = pe_data
                    else:
                        player_energy = None
                else:
                    player_energy = None
            except Exception:
                player_energy = None
                
            try:
                if hasattr(reports, 'player_r') and reports.player_r is not False:
                    pr_data = np.array(reports.player_r)
                    if len(pr_data.shape) == 2:  # (T, num_players)
                        player_r = pr_data[t]
                    elif len(pr_data.shape) == 1:  # (num_players) - same for all steps
                        player_r = pr_data
                    else:
                        player_r = None
                else:
                    player_r = None
            except Exception:
                player_r = None
                
            # Get family tree data for lineage tracking
            try:
                if hasattr(reports, 'family_tree_parents') and reports.family_tree_parents is not False:
                    ft_data = np.array(reports.family_tree_parents)
                    if len(ft_data.shape) == 3:  # (T, max_players, parents_per_child)
                        family_tree_parents = ft_data[t]
                    elif len(ft_data.shape) == 2:  # (max_players, parents_per_child) - same for all steps
                        family_tree_parents = ft_data
                    else:
                        family_tree_parents = None
                        if frame_idx == 0:
                            print(f"[matplotlib] Warning: family_tree_parents has unexpected shape: {ft_data.shape}")
                else:
                    family_tree_parents = None
                    if frame_idx == 0:
                        print(f"[matplotlib] Warning: family_tree_parents not available in report (using fallback coloring)")
            except Exception as e:
                family_tree_parents = None
                if frame_idx == 0:
                    print(f"[matplotlib] Warning: Error loading family_tree_parents: {e}")
            

            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot food grid as translucent background
            im = ax.imshow(fg, cmap='Greens', origin='lower', alpha=0.3, vmin=0, vmax=fg.max() if fg.max() > 0 else 1)
            
            # Get active player indices
            active_mask = pl.astype(bool)
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 0 and player_x is not None:
                # Extract positions of active players
                # player_x is typically (y, x) coordinates in grid space
                try:
                    active_positions = player_x[active_indices]
                    
                    # Convert to (x, y) for plotting (matplotlib uses x, y)
                    if len(active_positions.shape) >= 2 and active_positions.shape[1] >= 2:
                        plot_x = active_positions[:, 1]  # x coordinate
                        plot_y = active_positions[:, 0]   # y coordinate
                    elif len(active_positions.shape) == 1:
                        # Fallback: assume positions are interleaved or single dimension
                        plot_x = active_positions
                        plot_y = active_positions
                    else:
                        plot_x = None
                        plot_y = None
                except (IndexError, TypeError) as e:
                    print(f"[matplotlib] Warning: Could not extract player positions: {e}")
                    plot_x = None
                    plot_y = None
                
                # Plot players if we have valid positions
                if plot_x is not None and plot_y is not None:
                    # Color players by family lineage
                    # Use union-find to group players by their family tree
                    if family_tree_parents is not None:
                        # Build family groups using union-find
                        # Each player's parent ID points to their parent's player ID (birthday, location)
                        # We'll trace lineages by following parent links
                        
                        # Initialize: each player is in their own family
                        family_roots = {idx: idx for idx in active_indices}
                        
                        # Union-find helper functions
                        def find_root(idx):
                            """Find root of family tree for player index."""
                            if family_roots[idx] != idx:
                                family_roots[idx] = find_root(family_roots[idx])
                            return family_roots[idx]
                        
                        def union(idx1, idx2):
                            """Merge two players into same family."""
                            if idx1 not in active_indices or idx2 not in active_indices:
                                return
                            root1 = find_root(idx1)
                            root2 = find_root(idx2)
                            if root1 != root2:
                                family_roots[root2] = root1
                        
                        # Build a map: for each active player, store their parent ID
                        # Then try to match parent IDs to current player indices
                        # We'll use the parent ID's location component as a heuristic
                        player_to_parent_id = {}
                        for player_idx in active_indices:
                            parent_id = family_tree_parents[player_idx, 0]
                            if parent_id[0] != -1 and parent_id[0] >= 0:
                                player_to_parent_id[player_idx] = tuple(parent_id)
                        
                        # Try to match parent IDs to current players
                        # Strategy: use parent ID's location component as potential parent index
                        for player_idx in active_indices:
                            if player_idx in player_to_parent_id:
                                parent_id = player_to_parent_id[player_idx]
                                # Try to find parent by location component
                                parent_location = int(parent_id[1])
                                if (parent_location >= 0 and 
                                    parent_location < len(family_tree_parents) and 
                                    parent_location in active_indices):
                                    # Link child to parent
                                    union(player_idx, parent_location)
                        
                        # Assign family IDs based on roots
                        family_id_map = {}
                        next_family_id = 0
                        for player_idx in active_indices:
                            root = find_root(player_idx)
                            if root not in family_id_map:
                                family_id_map[root] = next_family_id
                                next_family_id += 1
                        
                        # Get family IDs for active players
                        family_ids = np.array([family_id_map[find_root(idx)] for idx in active_indices])
                        max_families = max(len(family_id_map), 1)
                        normalized_family_ids = family_ids.astype(float) / max_families
                        player_colors = plt.cm.hsv(normalized_family_ids)
                    else:
                        # Fallback: use player indices if family tree not available
                        max_expected_players = 512
                        normalized_indices = active_indices.astype(float) / max(max_expected_players, active_indices.max() + 1)
                        player_colors = plt.cm.hsv(normalized_indices)
                    
                    scatter = ax.scatter(
                        plot_x, plot_y,
                        c=player_colors,
                        s=100,
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.9,
                        zorder=10
                    )
                    
                    # Optionally show player direction if rotation is available
                    if player_r is not None and len(active_indices) > 0:
                        try:
                            active_rotations = player_r[active_indices].astype(int)
                            # Convert rotation to direction vector (assuming 0=up, 1=right, 2=down, 3=left)
                            direction_map = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])  # (dy, dx)
                            directions = direction_map[active_rotations % 4]
                            # Draw direction arrows
                            ax.quiver(
                                plot_x, plot_y,
                                directions[:, 1], directions[:, 0],  # (dx, dy)
                                scale=30,  # Larger scale = shorter arrows
                                width=0.003,
                                headwidth=2,
                                headlength=3,
                                color='black',
                                alpha=0.2,
                                zorder=11
                            )
                        except Exception:
                            pass  # Skip arrows if rotation data is invalid

            ax.set_title(f"NomNom – {experiment_name} | epoch {block_idx}, step {t} | Active players: {len(active_indices)}", fontsize=12)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set axis limits to match world size (always 512x512)
            h, w = 512, 512
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(-0.5, h - 0.5)

            # Use consecutive frame_idx for filenames (ensures proper sorting)
            # The step index 't' is already shown in the plot title
            out_path = os.path.join(images_dir, f"frame_{frame_idx:06d}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[matplotlib] Saved {out_path} (step {t}, frame {frame_idx})")
            frame_idx += 1  # Increment by 1 for each saved frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', type=str)
    parser.add_argument(
        '--max_frames',
        type=int,
        default=600,
        help='Maximum number of frames to render before exiting '
             '(for OpenGL mode: frames; for matplotlib mode: saved images).',
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['opengl', 'matplotlib'],
        default='opengl',
        help='Visualization backend: "opengl" for live viewer (default), '
             '"matplotlib" for headless PNG export.',
    )
    parser.add_argument(
        '--step_stride',
        type=int,
        default=1,
        help='For matplotlib backend: save every Nth step (default: 1).',
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name used for subdirectory and graph title. '
             'If not provided, uses the basename of output_directory.',
    )
    args = parser.parse_args()

    if args.backend == 'opengl':
        start_viewer(args.output_directory, max_frames=args.max_frames)
    else:
        render_with_matplotlib(
            args.output_directory,
            max_frames=args.max_frames if args.max_frames > 0 else None,
            step_stride=args.step_stride,
            experiment_name=args.experiment_name,
        )
