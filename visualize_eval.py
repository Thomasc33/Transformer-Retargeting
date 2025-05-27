import argparse
import torch
import numpy as np
import os
import pickle
import sys
import plotly.graph_objects as go
import plotly.io
import pandas as pd
from tqdm import tqdm
from PIL import Image
import glob

# Add project root to sys.path to allow imports
# Assumes the script is run from the root of the Transformer-Retargeting directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from data import load_data, datasets, parse_file_name
    from visualize.render import render_video # Use existing for GIF
    # Import necessary functions from eval_model.py
    from eval_model import (
        load_anonymizer,
        prep_data,
        get_anonymized_paired_raw,
        get_anonymized_paired_transformer,
        get_anonymized_paired_dmr_pmr
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the 'Transformer-Retargeting' directory,")
    print("or that the necessary modules are in your Python path.")
    sys.exit(1)

# Define connections globally
NTU_CONNECTIONS = [[0, 1], [1, 20], [20, 2], [2, 3], [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22], [0, 16], [16, 17], [17, 18], [18, 19], [0, 12], [12, 13], [13, 14], [14, 15]]

def setup_scene(d):
    """Calculates bounds and returns scene configuration."""
    # Flatten the tensor to find global bounds
    # Handle potential empty tensor
    if d.numel() == 0:
        print("Warning: Empty tensor passed to setup_scene. Using default bounds.")
        x_range, y_range, z_range = [-1, 1], [-1, 1], [-1, 1]
    else:
        d_flattened = d.reshape(-1, 3)
        x_min, x_max = d_flattened[:, 0].min().item(), d_flattened[:, 0].max().item()
        y_min, y_max = d_flattened[:, 1].min().item(), d_flattened[:, 1].max().item()
        z_min, z_max = d_flattened[:, 2].min().item(), d_flattened[:, 2].max().item()

        # Expand the bounds a bit
        padding = 0.5
        x_range = [x_min - padding, x_max + padding]
        y_range = [y_min - padding, y_max + padding]
        z_range = [z_min - padding, z_max + padding]

    scene = dict(
        xaxis=dict(range=x_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=y_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
        zaxis=dict(range=z_range, autorange=False, showgrid=False, zeroline=False, showticklabels=False, title=''),
        camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5), # Adjusted camera angle
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)
            ),
        aspectmode='cube',
        bgcolor='rgba(255,255,255,1)' # White background
    )
    return scene

def save_single_frame_image(d, frame_idx, filename, cons):
    """Saves a single frame of the skeleton animation as a PNG image."""
    if d is None or d.numel() == 0:
        print(f"Error: Cannot save frame {frame_idx}. Input data is empty or None.")
        return
    if frame_idx >= d.shape[0]:
        print(f"Error: frame_index {frame_idx} is out of bounds for skeleton with {d.shape[0]} frames.")
        return

    scene = setup_scene(d)
    layout = go.Layout(
        scene=scene,
        width=800,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=False,
        # paper_bgcolor='rgba(0,0,0,0)', # Transparent background if needed
        # plot_bgcolor='rgba(0,0,0,0)'
    )

    # Get data for the specific frame
    frame_data = d[frame_idx].reshape(-1, 3)
    x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    num_points = x.shape[0]

    # Create traces for this frame
    frame_traces = []
    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=3, color=np.linspace(1, num_points, num_points), colorscale='Rainbow')
    )
    frame_traces.append(scatter)

    for con in cons:
        # Check if connection indices are valid for the number of points
        if con[0] < num_points and con[1] < num_points:
            lx = [x[con[0]], x[con[1]]]
            ly = [y[con[0]], y[con[1]]]
            lz = [z[con[0]], z[con[1]]]
            line_trace = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=3))
            frame_traces.append(line_trace)
        else:
            print(f"Warning: Skipping connection {con} due to invalid indices for {num_points} points.")


    # Create figure and save
    fig = go.Figure(data=frame_traces, layout=layout)
    try:
        plotly.io.write_image(fig, filename, width=800, height=600, scale=1)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        print("Ensure you have 'kaleido' installed (`pip install -U kaleido`).")

def main():
    parser = argparse.ArgumentParser(description="Visualize skeleton data from datasets or files.")
    parser.add_argument('--dataset', required=True, choices=['ntu120', 'ntu', 'etri'],
                        help='Dataset name')
    parser.add_argument('--setting', default='cs', choices=['cs', 'cv'],
                        help='Cross-subject (cs) or cross-view (cv) setting (used if loading from dataset)')
    parser.add_argument('--skeleton_file', default=None, type=str,
                        help='Optional path to a .pkl file containing a dictionary of skeleton sequences.')
    parser.add_argument('--sample_index', default=0, type=int,
                        help='Index of the sample to visualize (from dataset or skeleton_file)')
    parser.add_argument('--output_gif', default=None, type=str,
                        help='Filename for the output GIF (e.g., "my_animation"). ".gif" will be appended. Provide "default" for auto-naming.')
    parser.add_argument('--output_frame', default=None, type=str,
                        help='Filename for the output single frame PNG image (e.g., "my_frame"). ".png" will be appended. Provide "default" for auto-naming.')
    parser.add_argument('--frame_index', default=0, type=int,
                        help='Index of the frame to save if --output_frame is used.')
    parser.add_argument('--duration', default=100, type=int,
                        help='Duration (ms) between frames in the GIF.')
    parser.add_argument('--T', default=64, type=int,
                        help='Target sequence length T for loading data and anonymizer (Transformer=64, PMR/DMR=75).')
    # Add anonymizer arguments
    parser.add_argument('--model_type', default='raw', choices=['raw','transformer','pmr','dmr'],
                      help="Type of anonymizer model to use ('raw' means no anonymization)")
    parser.add_argument('--transformer_model_path', default='model.pth', type=str,
                      help='Path to transformer model weights (if model_type is transformer)')
    parser.add_argument('--batch_size', default=1, type=int, # Default to 1 as we process one sample
                      help='Batch size (needed for PMR/DMR anonymizers)')


    args = parser.parse_args()

    # --- Argument Validation ---
    if not args.output_gif and not args.output_frame:
        print("Error: Please specify at least one output type: --output_gif or --output_frame.")
        sys.exit(1)

    # Configure device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is NOT available. Using CPU.")
    args.device = device
    args.loading_transformer = False # Flag needed by load_anonymizer

    # --- Load Skeleton Data ---
    skeleton_data = None
    skeleton_key = None
    data_source = ""
    original_actor = 0
    original_action = 0

    if args.skeleton_file:
        data_source = args.skeleton_file
        if not os.path.exists(args.skeleton_file):
            print(f"Error: Skeleton file not found: {args.skeleton_file}")
            sys.exit(1)
        if not args.skeleton_file.endswith('.pkl'):
            print(f"Error: Expected a .pkl file for --skeleton_file.")
            sys.exit(1)
        try:
            with open(args.skeleton_file, 'rb') as f:
                data_dict = pickle.load(f)
            if not isinstance(data_dict, dict):
                print(f"Error: Expected a dictionary in {args.skeleton_file}")
                sys.exit(1)

            keys = list(data_dict.keys())
            if not keys:
                print(f"Error: No keys found in the dictionary within {args.skeleton_file}")
                sys.exit(1)
            if args.sample_index >= len(keys):
                print(f"Error: sample_index {args.sample_index} out of range for file {args.skeleton_file} (size: {len(keys)})")
                sys.exit(1)

            skeleton_key = keys[args.sample_index]
            skeleton_data = data_dict[skeleton_key]
            print(f"Loaded skeleton '{skeleton_key}' (index {args.sample_index}) from {args.skeleton_file}")
            # Parse key to get original actor/action
            try:
                parts = parse_file_name(skeleton_key, dataset=args.dataset)
                original_actor = parts['P']
                original_action = parts['A']
            except Exception as parse_e:
                print(f"Warning: Could not parse filename '{skeleton_key}' for actor/action info: {parse_e}")

        except Exception as e:
            print(f"Error loading skeleton file {args.skeleton_file}: {e}")
            sys.exit(1)
    else:
        data_source = f"{args.dataset}_dataset"
        print(f"Loading dataset '{args.dataset}' with T={args.T}...")
        try:
            # Use load_data which handles padding/truncating and returns a dict
            all_data = load_data(args.dataset, T=args.T)
            keys = list(all_data.keys())
            if not keys:
                 print(f"Error: No data loaded for dataset '{args.dataset}'. Check data path and format.")
                 sys.exit(1)
            if args.sample_index >= len(keys):
                print(f"Error: sample_index {args.sample_index} out of range for dataset {args.dataset} (size: {len(keys)})")
                sys.exit(1)

            skeleton_key = keys[args.sample_index]
            skeleton_data = all_data[skeleton_key]
            print(f"Loaded sample {args.sample_index} ('{skeleton_key}') from dataset '{args.dataset}'")
            # Parse key to get original actor/action
            try:
                parts = parse_file_name(skeleton_key, dataset=args.dataset)
                original_actor = parts['P']
                original_action = parts['A']
            except Exception as parse_e:
                print(f"Warning: Could not parse filename '{skeleton_key}' for actor/action info: {parse_e}")

        except FileNotFoundError:
             print(f"Error: Data file not found for dataset '{args.dataset}'. Expected at: {datasets[args.dataset]['path']}")
             sys.exit(1)
        except Exception as e:
            print(f"Error loading dataset {args.dataset}: {e}")
            sys.exit(1)

    if skeleton_data is None:
        print("Error: Failed to load skeleton data.")
        sys.exit(1)

    # --- Ensure data is Torch Tensor ---
    if isinstance(skeleton_data, np.ndarray):
        skeleton_data = torch.from_numpy(skeleton_data).float()
    elif not isinstance(skeleton_data, torch.Tensor):
        print(f"Error: Loaded data is not a NumPy array or Torch Tensor (type: {type(skeleton_data)})")
        sys.exit(1)

    # --- Load Anonymizer ---
    ds_config = datasets[args.dataset]
    anonymizer_model = None
    anonymizer_path = None
    if args.model_type != 'raw':
        if args.model_type == 'transformer':
            anonymizer_path = args.transformer_model_path
        else: # PMR/DMR - Construct path based on convention if needed, or use a default
            # Example: anonymizer_path = f'./eval/{args.model_type}/{args.dataset}.pt'
            # Using a placeholder path for now, adjust as needed
            anonymizer_path = f'trained_models/{args.model_type}_{args.dataset}_{args.setting}_final.pth' # Adjust this path
            if not os.path.exists(anonymizer_path):
                 print(f"Warning: Anonymizer model path not found: {anonymizer_path}. Check path or use --transformer_model_path.")
                 # Fallback or exit if necessary
                 # anonymizer_path = None # Or sys.exit(1)

        if anonymizer_path and os.path.exists(anonymizer_path):
            print(f"Loading anonymizer model ({args.model_type}) from {anonymizer_path}")
            anonymizer_model = load_anonymizer(
                args.model_type, anonymizer_path, args.device, args, ds=ds_config
            )
        elif args.model_type != 'raw':
             print(f"Warning: Could not load anonymizer model ({args.model_type}). Path '{anonymizer_path}' not found or not specified correctly.")
             print("Proceeding without anonymization.")
             args.model_type = 'raw' # Fallback to raw if model loading fails

    # --- Anonymize Data ---
    processed_skeleton_data = None
    if args.model_type != 'raw' and anonymizer_model is not None:
        print(f"Anonymizing skeleton using {args.model_type}...")
        # Create a dummy batch for the anonymizer functions
        # We need (x1, x2, y1, y2, actors, actions)
        # Let x1 be our skeleton. Use x1 for x2, y1, y2.
        # Use original actor/action and dummy values for the second pair.
        dummy_actor2 = 0 # Use 0 or another placeholder

        dummy_action2 = 0

        # Ensure data is on the correct device
        x1 = skeleton_data.unsqueeze(0).to(args.device) # Add batch dim
        x2 = x1.clone()
        y1 = x1.clone()
        y2 = x1.clone()

        # Actors and actions need to be tensors for the functions
        actors = torch.tensor([[original_actor, dummy_actor2]], dtype=torch.float).to(args.device)
        actions = torch.tensor([[original_action, dummy_action2]], dtype=torch.float).to(args.device)

        dummy_batch = (x1, x2, y1, y2, actors, actions)

        # Call the appropriate anonymization function
        anonymized_output_list = []
        if args.model_type == 'transformer':
            anonymized_output_list = get_anonymized_paired_transformer(dummy_batch, anonymizer_model, prep_data)
        elif args.model_type in ['pmr', 'dmr']:
             # PMR/DMR expect T=75 by default in their original eval, adjust T if needed
             anonymizer_T = 75 if args.model_type in ['pmr', 'dmr'] else args.T
             # If T=64, we're likely using this for Mixformer, so trim the output
             mixformer_mode = args.T == 64
             anonymized_output_list = get_anonymized_paired_dmr_pmr(
                 dummy_batch,
                 anonymizer_model,
                 T=anonymizer_T,
                 mixformer_mode=mixformer_mode
             )

        # Extract the anonymized skeleton (x1_hat, which is the first item's 'skeleton')
        if anonymized_output_list:
            processed_skeleton_data = anonymized_output_list[0]['skeleton'].cpu() # Move back to CPU
            print("Anonymization complete.")
        else:
            print("Warning: Anonymization failed. Using original data.")
            processed_skeleton_data = skeleton_data.cpu() # Use original if anonymization fails
    else:
        print("No anonymization applied (model_type is 'raw' or model failed to load).")
        processed_skeleton_data = skeleton_data.cpu() # Use original data

    if processed_skeleton_data is None or processed_skeleton_data.numel() == 0:
         print("Error: Processed skeleton data is empty or None after anonymization step.")
         sys.exit(1)


    # --- Generate Output ---
    # Create output directories if they don't exist
    gif_dir = os.path.join('results', 'gif')
    img_dir = os.path.join('results', 'img')
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Generate default filenames if needed
    # Sanitize key for filename
    safe_key = "".join(c if c.isalnum() else "_" for c in str(skeleton_key))
    # Add model type to filename if anonymized
    model_tag = f"_{args.model_type}" if args.model_type != 'raw' else ""
    base_filename = f"{args.dataset}_{safe_key}{model_tag}"

    output_gif_name = args.output_gif
    if output_gif_name == 'default':
        output_gif_name = base_filename

    output_frame_name = args.output_frame
    if output_frame_name == 'default':
        output_frame_name = f"{base_filename}_frame{args.frame_index}"

    # Render GIF using the processed (potentially anonymized) data
    if output_gif_name:
        # Pass the base name to render_video, it will handle the path and extension
        gif_base_path = os.path.join(gif_dir, output_gif_name)
        print(f"Rendering GIF to {gif_base_path}.gif...")
        # Note: render_video expects the *base* path without extension for its 'gif' arg
        render_video(processed_skeleton_data, gif=gif_base_path, show_render=False, duration=args.duration)
        print("GIF rendering complete.")

    # Render Single Frame using the processed (potentially anonymized) data
    if output_frame_name:
        img_path = os.path.join(img_dir, f"{output_frame_name}.png")
        print(f"Rendering frame {args.frame_index} to {img_path}...")
        save_single_frame_image(processed_skeleton_data, args.frame_index, img_path, NTU_CONNECTIONS)
        print("Frame rendering complete.")

if __name__ == "__main__":
    main()
