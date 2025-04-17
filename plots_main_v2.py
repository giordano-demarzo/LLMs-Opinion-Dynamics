import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PathCollection
from matplotlib.patches import Circle, FancyArrowPatch
from collections import defaultdict

#%%FIG 1 PAPER
simulation_data = [
    ["150"],
    ["125", "25"],
    ["108", "17", "25"],
    ["99", "9", "17", "25"],
    ["87", "12", "9", "17", "25"],
    ["75", "12", "12", "9", "17", "25"],
    ["61", "14", "12", "12", "9", "17", "25"],
    ["51", "10", "14", "12", "12", "9", "17", "22", "3"],
    ["48", "3", "10", "14", "12", "12", "9", "17", "20", "2", "3"],
    ["48", "3", "10", "14", "12", "12", "9", "17", "20", "2", "3"],
    ["48", "3", "10", "14", "12", "12", "9", "17", "20", "2", "3"]
]

parent_relations = {
    (1,0):(0,0), (1,1):(0,0),
    (2,0):(1,0), (2,1):(1,0), (2,2):(1,1),
    (3,0):(2,0), (3,1):(2,0), (3,2):(2,1), (3,3):(2,2),
    (4,0):(3,0), (4,1):(3,0), (4,2):(3,1), (4,3):(3,2), (4,4):(3,3),
    (5,0):(4,0), (5,1):(4,0), (5,2):(4,1), (5,3):(4,2), (5,4):(4,3), (5,5):(4,4),
    (6,0):(5,0), (6,1):(5,0), (6,2):(5,1), (6,3):(5,2), (6,4):(5,3), (6,5):(5,4), (6,6):(5,5),
    (7,0):(6,0), (7,1):(6,0), (7,2):(6,1), (7,3):(6,2), (7,4):(6,3), (7,5):(6,4), (7,6):(6,5), (7,7):(6,6), (7,8):(6,6),
    (8,0):(7,0), (8,1):(7,0), (8,2):(7,1), (8,3):(7,2), (8,4):(7,3), (8,5):(7,4), (8,6):(7,5), (8,7):(7,6), (8,8):(7,7), (8,9):(7,7), (8,10):(7,8),
    (9,0):(8,0), (9,1):(8,1), (9,2):(8,2), (9,3):(8,3), (9,4):(8,4), (9,5):(8,5), (9,6):(8,6), (9,7):(8,7), (9,8):(8,8), (9,9):(8,9), (9,10):(8,10),
    (10,0):(9,0), (10,1):(9,1), (10,2):(9,2), (10,3):(9,3), (10,4):(9,4), (10,5):(9,5), (10,6):(9,6), (10,7):(9,7), (10,8):(9,8), (10,9):(9,9), (10,10):(9,10)
}

def circle_radius(size):
    return np.sqrt(int(size.replace('*', ''))) * 0.2

def identify_stable_groups(simulation_data, stability_threshold=3):
    """
    Identify groups that remain stable (unchanged) for 'stability_threshold' consecutive iterations.
    Returns a set of (iteration, index) tuples that should be frozen (not shown after stability is achieved).
    """
    stable_groups = set()
    
    # Create a map from group size to their positions at each iteration
    size_to_position = defaultdict(list)
    for iter_idx, iteration in enumerate(simulation_data):
        for group_idx, size in enumerate(iteration):
            size_to_position[size].append((iter_idx, group_idx))
    
    # Track stability of each group
    group_stability = {}
    
    # For each iteration starting from the third
    for iter_idx in range(2, len(simulation_data)):
        # Check each group in this iteration
        for group_idx, size in enumerate(simulation_data[iter_idx]):
            current = (iter_idx, group_idx)
            
            # Check if this group has the same size in the previous two iterations
            # by tracing its lineage through parent relations
            if current in parent_relations:
                parent = parent_relations[current]
                parent_size = simulation_data[parent[0]][parent[1]]
                
                if parent in parent_relations:
                    grandparent = parent_relations[parent]
                    grandparent_size = simulation_data[grandparent[0]][grandparent[1]]
                    
                    # If sizes match for 3 consecutive generations
                    if size == parent_size == grandparent_size:
                        # Initialize or increment stability counter
                        if current not in group_stability:
                            group_stability[current] = 1
                        else:
                            group_stability[current] = group_stability.get(parent, 0) + 1
                        
                        # If this group has been stable for 'stability_threshold' iterations
                        if group_stability[current] >= stability_threshold - 2:  # -2 because we've already checked 2 previous iterations
                            # Mark this group and all its future descendants as frozen
                            for future_iter in range(iter_idx + 1, len(simulation_data)):
                                # Find descendants in future iterations (potentially recursive search)
                                for future_idx in range(len(simulation_data[future_iter])):
                                    future_pos = (future_iter, future_idx)
                                    # Check if this is a descendant of our stable group
                                    # For simplicity, we'll just check direct parentage one level at a time
                                    if future_pos in parent_relations:
                                        if (
                                            parent_relations[future_pos] == current or
                                            (parent_relations[future_pos] in stable_groups)
                                        ):
                                            stable_groups.add(future_pos)
    
    return stable_groups

def compute_positions(simulation_data, frozen_groups=None, horizontal_gap=6, vertical_gap=5):
    """
    Compute positions for each group, skipping frozen groups.
    """
    if frozen_groups is None:
        frozen_groups = set()
        
    positions = {}
    y_levels = {}
    
    # Active groups at each iteration (filtering out frozen ones)
    active_groups = []
    for iter_idx, iteration in enumerate(simulation_data):
        active_in_iteration = []
        for group_idx, size in enumerate(iteration):
            if (iter_idx, group_idx) not in frozen_groups:
                active_in_iteration.append((group_idx, size))
        active_groups.append(active_in_iteration)

    # First pass: assign positions for non-frozen groups
    for iteration, groups in enumerate(simulation_data):
        x = iteration * horizontal_gap
        for idx, size in enumerate(groups):
            current = (iteration, idx)
            
            # Skip frozen groups
            if current in frozen_groups:
                continue
                
            if current in parent_relations:
                parent = parent_relations[current]
                
                # If parent is frozen, find the last active ancestor
                ancestor = parent
                while ancestor in frozen_groups and ancestor in parent_relations:
                    ancestor = parent_relations[ancestor]
                
                # If we found an active ancestor, use its position as reference
                if ancestor not in frozen_groups and ancestor in positions:
                    siblings = [
                        child for child, p in parent_relations.items() 
                        if p == parent and child not in frozen_groups
                    ]
                    num_siblings = len(siblings)
                    if num_siblings == 1:
                        y_levels[current] = y_levels.get(ancestor, 0)
                    else:
                        # Slightly larger vertical offset for iteration == 1
                        offset_scale = 1 if iteration == 1 else 1
                        offset = offset_scale*vertical_gap * (siblings.index(current) - (num_siblings - 1)/2)
                        y_levels[current] = y_levels.get(ancestor, 0) + offset
                else:
                    # If no active ancestor found, just use default position
                    y_levels[current] = 0
            else:
                y_levels[current] = 0
                
            positions[current] = (x, y_levels[current])

    # Resolve any minor collisions by shifting upwards a little
    for iter_idx in range(len(simulation_data)):
        seen_y = set()
        for idx in range(len(simulation_data[iter_idx])):
            current = (iter_idx, idx)
            
            # Skip frozen groups
            if current in frozen_groups or current not in positions:
                continue
                
            x, y = positions[current]
            while y in seen_y:
                y += 0.1
            seen_y.add(y)
            positions[current] = (x, y)

    return positions

def generate_grid_points_in_circle(n, cx, cy, r):
    points = []
    side = int(np.ceil(np.sqrt(n)))
    grid_x = np.linspace(cx - r, cx + r, side)
    grid_y = np.linspace(cy - r, cy + r, side)
    for x in grid_x:
        for y in grid_y:
            if len(points) < n and (x - cx)**2 + (y - cy)**2 <= r**2:
                points.append((x, y))
    return zip(*points)

def load_magnetization_data(filename):
    """
    Load magnetization data from a file.
    """
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Extract iteration and magnetization value
                parts = line.strip().split(',')
                if len(parts) == 2:
                    iteration, magnetization = int(parts[0]), float(parts[1])
                    data.append((iteration, magnetization))
    except Exception as e:
        print(f"Error loading magnetization data from {filename}: {e}")
    return data

def get_font_size(size):
    """
    Get appropriate font size based on group size
    """
    font_sizes = [18, 14, 12, 8, 6]
    if int(size.replace('*', '')) > 80:
        return font_sizes[0]
    elif int(size.replace('*', '')) > 50:
        return font_sizes[1]
    elif int(size.replace('*', '')) > 20:
        return font_sizes[2]
    elif int(size.replace('*', '')) > 10:
        return font_sizes[3]
    else:
        return font_sizes[4]

def plot_visualization(
    simulation_data, 
    positions, 
    frozen_groups=None,
    horizontal_gap=7, 
    vertical_gap=2,
    style="normal", 
    save_path=None,
    mag_file_1="magnetization_0_0_150_0.7_0.0_meta-llama-3-70b-instruct.txt",
    mag_file_2="magnetization_3_0_25_0.7_-0.04_meta-llama-3-70b-instruct.txt"
):
    """
    Plots the simulation data with two possible styles:
      - style='normal': standard output with black text/arrows and white background.
      - style='transparent': figure saved with transparent background, 
        white text/arrows (useful for dark slides).
    Optionally specify a save_path to save the figure as an image.
    """
    if frozen_groups is None:
        frozen_groups = set()

    # Decide colors for text, arrows, circle edges, background
    if style.lower() == "transparent":
        text_color = "white"
        arrow_color = "white"
        edge_color = "white"
        # We'll set the figure facecolor to 'none' at save time
    else:
        text_color = "black"
        arrow_color = "gray"
        edge_color = "black"

    # Create a simple figure with just one main axis
    fig, ax = plt.subplots(figsize=(22, 12))

    # Build a larger color palette (40 colors) from tab20 + tab20b
    base_colors = np.vstack([
        plt.cm.tab20(np.linspace(0,1,20)),
        plt.cm.tab20b(np.linspace(0,1,20))
    ])

    # Dictionary for counting children
    children_count = defaultdict(int)
    for child, par in parent_relations.items():
        if child not in frozen_groups:  # Only count non-frozen children
            children_count[par] += 1

    group_colors = {}
    color_idx = 0

    # Assign colors for each group
    for (iteration, idx), (x, y) in positions.items():
        if (iteration, idx) in frozen_groups:
            continue  # Skip frozen groups
            
        size = simulation_data[iteration][idx]
        r = circle_radius(size)

        parent = parent_relations.get((iteration, idx))
        if parent:
            n_children = children_count[parent]
            if n_children == 1:
                # Only one child => child keeps the parent's color
                group_colors[(iteration, idx)] = group_colors.get(parent, base_colors[color_idx])
                if parent not in group_colors:
                    color_idx += 1
            else:
                # 2+ children => each child gets a new color
                group_colors[(iteration, idx)] = base_colors[color_idx]
                color_idx += 1
        else:
            # No parent => new color
            group_colors[(iteration, idx)] = base_colors[color_idx]
            color_idx += 1

        circle = Circle(
            (x, y), 
            r, 
            color=group_colors[(iteration, idx)],
            alpha=1, 
            ec=edge_color
        )
        ax.add_patch(circle)

        # Print group size in the center
        font_size = get_font_size(size)
            
        ax.text(
            x, y, 
            str(size), 
            ha='center', va='center',
            fontsize=font_size, fontweight='bold',
            color='black'
        )

        # Label iteration if idx == 0
        if idx == 0:
            ax.text(
                x,
                max(p[1] for p in positions.values()) + vertical_gap*1.5,
                "i=0" if iteration == 0 else f"i={iteration}",  # Changed from "Initial"/"Iter. X" to "t=X"
                ha='center', 
                fontsize=16, fontweight='bold',
                color=text_color
            )

    # Draw arrows - only for non-frozen groups
    for child, parent in parent_relations.items():
        # Skip if either child or parent is frozen or not in positions
        if (child in frozen_groups or parent in frozen_groups or 
            child not in positions or parent not in positions):
            continue
            
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        r0 = circle_radius(simulation_data[parent[0]][parent[1]])
        r1 = circle_radius(simulation_data[child[0]][child[1]])

        dx, dy = x1 - x0, y1 - y0
        dist = np.hypot(dx, dy)
        dx, dy = dx/dist, dy/dist

        start = (x0 + dx*r0, y0 + dy*r0)
        end = (x1 - dx*r1, y1 - dy*r1)

        arrow = FancyArrowPatch(
            start, end, 
            arrowstyle='-|>',
            color=arrow_color,
            lw=1, 
            mutation_scale=10
        )
        ax.add_patch(arrow)

    # Adjust main axis
    ax.set_xlim(-horizontal_gap, horizontal_gap * len(simulation_data))
    if positions:  # Only if we have positions (not all groups might be frozen)
        y_vals = [pos[1] for pos in positions.values()]
        y_margin = max(np.abs(y_vals)) + vertical_gap*4 if y_vals else vertical_gap*4
        ax.set_ylim(-y_margin, y_margin)
    else:
        ax.set_ylim(-vertical_gap*4, vertical_gap*4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    
    # Load magnetization data for the two specified groups
    mag_data_1 = load_magnetization_data(mag_file_1)
    mag_data_2 = load_magnetization_data(mag_file_2)
    
    # Create two small subplots for magnetization at the bottom
    left, bottom, width, height = 0.2, 0.2, 0.12, 0.15  # Bottom left corner for group with size 150
    ax_mag1 = fig.add_axes([left, bottom, width, height])
    
    # Left side, between iteration label and bottom groups
    left, bottom, width, height = 0.65, 0.4, 0.12, 0.15  # Bottom center
    ax_mag2 = fig.add_axes([left, bottom, width, height])
    
    # Plot magnetization for group with size 150 (initial group)
    if mag_data_1:
        # Get the position and color for group (0,0)
        group_0_0_pos = positions.get((0, 0))
        color_1 = group_colors.get((0, 0), 'blue')  # Default to blue if color not found
        
        iterations_1, magnetizations_1 = zip(*mag_data_1)
        iterations_1 = np.array(iterations_1)
        
        # Set colored frame for inset
        # for spine in ax_mag1.spines.values():
        #     spine.set_color(color_1)
        #     spine.set_linewidth(2)
        
        ax_mag1.plot(iterations_1/150, magnetizations_1, '-', lw=2, color=color_1)
        ax_mag1.set_xlabel("Time", fontsize=15, color=text_color)
        ax_mag1.set_ylabel("Average\ngroup opinion", fontsize=15, color=text_color)
        ax_mag1.set_ylim([-1, 1])
        ax_mag1.set_xlim([-0.1, 10])
        ax_mag1.tick_params(axis='both', colors=text_color, labelsize=12)
        
       
        
    # Plot magnetization for group with size 25 from iteration 3
    if mag_data_2:
        # Get the position and color for group (1, 1) - the orange "25" group
        group_1_1_pos = positions.get((1, 1))
        color_2 = group_colors.get((1, 1), 'red')  # Default to red if color not found
        
        iterations_2, magnetizations_2 = zip(*mag_data_2)
        iterations_2 = np.array(iterations_2)
        
        # Set colored frame for inset
        # for spine in ax_mag2.spines.values():
        #     spine.set_color(color_2)
        #     spine.set_linewidth(2)
        
        ax_mag2.plot(iterations_2/25, magnetizations_2, '-', lw=2, color=color_2)
        ax_mag2.set_xlabel("Time", fontsize=15, color=text_color)
        ax_mag2.set_ylabel("Average\ngroup opinion", fontsize=15, color=text_color)
        ax_mag2.set_ylim([-1, 1])
        ax_mag2.set_xlim([-0.1, 10])
        ax_mag2.tick_params(axis='both', colors=text_color, labelsize=12)
        
      

    # Save the figure if requested
    if save_path is not None:
        if style.lower() == "transparent":
            plt.savefig(save_path, transparent=True, facecolor='none', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Identify stable groups (groups that remain unchanged for 3 iterations)
    frozen_groups = identify_stable_groups(simulation_data, stability_threshold=3)
    print(f"Found {len(frozen_groups)} groups to freeze after 3 iterations of stability")
    
    # Compute positions for non-frozen groups
    positions = compute_positions(simulation_data, frozen_groups)
    
    # Plot the visualization with frozen groups hidden and magnetization plots
    plot_visualization(
        simulation_data, 
        positions, 
        frozen_groups=frozen_groups,
        style="normal", 
        save_path="plot1_with_magnetization.png"
    )

#%%FIG 2 PAPER
# Function to extract size and model name from the file name
def extract_info(filename):
    parts = filename.split('_')
    size = float(parts[2])
    model_name = filename.split('_')[-1].replace('.txt', '')
    return size, model_name

# Mapping from long filenames to legend names
filename_to_legend = {
    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "claude-2.0": "Claude 2.0",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "gpt-3.5-turbo-1106": "GPT-3.5 Turbo",
    "gpt-4-0613": "GPT-4",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "meta-llama-3-70b-instruct": "Llama 3 70B"
}

# Directories containing the files
fully_connected_dirs = ['fully_connected_50']
final_values_dirs = ['final_values_fully_connected_50']

# Define line styles, colors, and markers
line_styles = ['-', '-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
colors = list(mcolors.TABLEAU_COLORS.keys())
markers = ['o', 's', 'D', '>', 'v', 'p', '*', 'h', '+', 'x']

# Map models to their specific color and marker based on previous plots
color_marker_map = {
    "Claude 3.5 Sonnet": (colors[7], 'o'),
    "GPT-4o": (colors[8], 's'),
    "Claude 3 Opus": (colors[4], 'D'),
    "GPT-4 Turbo": (colors[3], '>'),
    "GPT-4": (colors[5], 'v'),
    "Llama 3 70B": (colors[0], 'p'),
    "Claude 3 Sonnet": (colors[6], '*'),
    "Claude 2.0": (colors[2], 'h'),
    "Claude 3 Haiku": (colors[1], '+'),
    "GPT-3.5 Turbo": (colors[9], 'x')
}

# Increase font size for all text elements
plt.rcParams.update({
    'font.size': 35,
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'legend.fontsize': 25,
})

# Initialize plot for left side only
fig = plt.figure(figsize=(14, 12))

# Create main axis
ax1 = fig.add_subplot(111)

# Create a mapping for size to line style and model to color
size_to_line_style = {}
model_to_color = {}

# Lists of unique sizes and models
unique_sizes = []
unique_models = list(color_marker_map.keys())

# Function to plot data
def plot_data(directory, ax):
    global unique_sizes, unique_models, color_marker_map, filename_to_legend
    files = glob.glob(os.path.join(directory, 'magnetization_*.txt'))
    model_lines = []  # To store lines for legend
    
    for file in files:
        # Extract size and model name from the file name
        size, model_name = extract_info(os.path.basename(file))
        
        # Map the long model name to the legend name
        if model_name in filename_to_legend:
            model_name = filename_to_legend[model_name]
        
        # Only plot data for sizes 50
        if size not in [50]:
            continue
        
        # Assign line style and color if not already assigned
        if size not in size_to_line_style:
            size_to_line_style[size] = line_styles[len(size_to_line_style) % len(line_styles)]
            unique_sizes.append(size)
        if model_name in color_marker_map:
            color, marker = color_marker_map[model_name]
            model_to_color[model_name] = color
        else:
            color = colors[len(model_to_color) % len(colors)]
            marker = markers[len(model_to_color) % len(markers)]
            model_to_color[model_name] = color
            color_marker_map[model_name] = (color, marker)
            unique_models.append(model_name)
        
        # Read the file
        data = pd.read_csv(file, header=None, names=['time', 'magnetization'])
        
        # Ensure the columns are numeric
        data['time'] = pd.to_numeric(data['time'], errors='coerce')
        data['magnetization'] = pd.to_numeric(data['magnetization'], errors='coerce')
        
        # Drop any rows with NaN values (in case of errors in conversion)
        data.dropna(inplace=True)
        
        # Rescale time
        data['rescaled_time'] = data['time'] / size
        
        # Convert pandas Series to numpy arrays before plotting to avoid multi-dimensional indexing issues
        x_data = data['rescaled_time'].to_numpy()
        y_data = np.abs(data['magnetization'].to_numpy())
        
        # Plot the data
        line = ax.plot(x_data, y_data, 
                linestyle=size_to_line_style[size], 
                color=color,
                linewidth=2.5,
                label=model_name)
                
        # Store the line for legend
        model_lines.append((model_name, line[0]))

    return model_lines

# Plot data for fully_connected N=50
model_lines = plot_data(fully_connected_dirs[0], ax1)

# Set axes labels with large font size
ax1.set_xlabel(r'Time $t$', fontsize=35)
ax1.set_ylabel(r'Coordination level $|m(t)|$', fontsize=35)
ax1.set_ylim([-0.01, 1.01])
ax1.set_xlim([0, 10])

# Increase tick label font sizes
ax1.tick_params(axis='both', which='major', labelsize=35)

# Create an inset axis for the box plot N=50
ax3 = inset_axes(ax1, width="20%", height="100%", loc='center right', borderpad=-6.75)

# Collect final values of magnetization N=50
final_values = {}
files = glob.glob(os.path.join(final_values_dirs[0], '*.txt'))
for file in files:
    model_name = os.path.basename(file).partition('0.0_')[2].partition('_MF')[0]
    if model_name in filename_to_legend:
        model_name = filename_to_legend[model_name]
    size = os.path.basename(file).partition('magnetization_')[2].partition('_')[0]
    key = f"{model_name}_{int(size)}"
    if key not in final_values:
        final_values[key] = []
    final_values[key].extend(pd.read_csv(file, header=None, names=['final_magnetization'])['final_magnetization'].tolist())

# Prepare data for the box plot N=50
i = 1
for model_name in unique_models:
    key = f"{model_name}_50"
    if key in final_values:
        color = model_to_color[model_name]
        ax3.boxplot(np.abs(final_values[key]), positions=[i], widths=[0.75], patch_artist=True, 
                    boxprops=dict(facecolor=color, alpha=0.3), 
                    medianprops=dict(color=color, linewidth=10), 
                    flierprops=dict(markerfacecolor=color))
        i += 1

ax3.set_yticks([])
ax3.set_xlim([0, 6])  # Adjust this limit based on the number of models
ax3.set_xticks([])
ax3.set_ylim([-0.01, 1.01])

# Sort models for legend
unique_models.sort()

# Create legend with consistent colors
handles = []
labels = []
for model_name in unique_models:
    if model_name in model_to_color:
        color = model_to_color[model_name]
        marker = color_marker_map[model_name][1]
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='-', markersize=10))
        labels.append(model_name)

# Add legend with increased font size
legend = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                    fontsize=25, ncol=3)

# Adjust layout to make room for the larger font sizes
plt.tight_layout()

# Show plot
plt.savefig('plot2.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

#%%FIG 3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Define the tanh fitting function
def tanh_fit(x, beta):
    return 0.5 * (np.tanh(beta * x) + 1)

# Mapping from long filenames to legend names
filename_to_legend = {
    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "claude-2.0": "Claude 2.0",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo",
    "gpt-3.5-turbo-1106": "GPT-3.5 Turbo",
    "gpt-4-0613": "GPT-4",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "meta-llama-3-70b-instruct": "Llama 3 70B"
}

# Define line styles, colors, and markers
line_styles = ['-', '-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
colors = list(mcolors.TABLEAU_COLORS.keys())
markers = ['o', 's', 'D', '>', 'v', 'p', '*', 'h', '+', 'x']

# Map models to their specific color and marker based on previous plots
color_marker_map = {
    "Claude 3.5 Sonnet": (colors[7], 'o'),
    "GPT-4o": (colors[8], 's'),
    "Claude 3 Opus": (colors[4], 'D'),
    "GPT-4 Turbo": (colors[3], '>'),
    "GPT-4": (colors[5], 'v'),
    "Llama 3 70B": (colors[0], 'p'),
    "Claude 3 Sonnet": (colors[6], '*'),
    "Claude 2.0": (colors[2], 'h'),
    "Claude 3 Haiku": (colors[1], '+'),
    "GPT-3.5 Turbo": (colors[9], 'x')
}

# Directories
second_plot_dir = 'transition_prob_various_models_kz/50_kz'

# Increase font size
plt.rcParams.update({
    'font.size': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 22,
})

# Create the figure with two subplots side by side
fig = plt.figure(figsize=(24, 8))
gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

# LEFT PLOT: Adoption probability plot (right part from first code)
ax1 = fig.add_subplot(gs[0, 0])

# Function to plot adoption probability data (right part of the original plot)
def plot_adoption_probability(ax, inset, directory, plot_inset=True):
    files_info = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            model_name = filename.split('_')[4].partition(".tx")[0]
            if model_name in filename_to_legend:
                model_name = filename_to_legend[model_name]
            files_info.append((filename, model_name))
            
    # Sort files alphabetically by model name
    files_info.sort(key=lambda x: x[1])

    for i, (filename, model_name) in enumerate(files_info):
        # Read the data
        data = pd.read_csv(os.path.join(directory, filename), header=None)

        # Extract x and y data
        x = data.iloc[:, 0].values
        y = data.iloc[:, -1].values

        # Fit the data to the tanh function
        popt, _ = curve_fit(tanh_fit, x, y)
        beta = popt[0]

        # Generate x values for the fit line
        x_fit = np.linspace(-1, 1, 100)
        y_fit = tanh_fit(x_fit, beta)

        # Use consistent color and marker
        color, marker = color_marker_map[model_name]

        # Plot the original data
        ax.scatter(x, y, label=fr'{model_name}', color=color, marker=marker, s=100)

        # Plot the fit line
        ax.plot(x_fit, y_fit, color=color)

        # If plot_inset and beta > 0.5, add the data to the inset
        if plot_inset:
            x_scaled = x * beta
            inset.scatter(x_scaled, y, color=color, marker=marker)

# Create an inset axis for the first plot
ax_inset = ax1.inset_axes([0.1, 0.6, 0.35, 0.35])
ax_inset.set_facecolor('white')
ax_inset.grid(True)
ax_inset.set_yticks([0, 0.5, 1])

# Plot the adoption probability data in the first subplot
plot_adoption_probability(ax1, ax_inset, second_plot_dir)

# Set axis labels for the first plot
ax1.set_xlabel(r'Collective opinion $m$')
ax1.set_ylabel(r'Adoption probability $P(m)$')
ax1.set_ylim([-0.1, 1.3])

# RIGHT PLOT: Group size vs Beta plot (right part from second code)
ax2 = fig.add_subplot(gs[0, 1])

# Function to process each directory and fit the tanh function
def process_directory(directory):
    allowed_N_values = {10, 20, 30, 50, 100, 200, 500, 1000}
    files_info = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            full_path = os.path.join(directory, filename)
            data = pd.read_csv(full_path, header=None)
            # Extract x and y data
            x = data.iloc[:, 0].values
            y = data.iloc[:, -1].values
            # Fit the data to the tanh function
            popt, pcov = curve_fit(tanh_fit, x, y)
            beta = popt[0]
            beta_err = np.sqrt(np.diag(pcov))[0]  # Extract the standard deviation (error) for beta
            # Extract N from filename
            N = int(filename.split('_')[2])

            # Only include allowed N values
            if N in allowed_N_values:
                files_info.append((N, beta, beta_err))
    
    # Sort files by N
    files_info.sort(key=lambda x: x[0])
    
    return files_info

# Dictionary to hold the results for each model
results = {}

# Process each directory and store the results
directories = [
    "transition_prob_various_models_kz/various_N_Claude 2.0",
    "transition_prob_various_models_kz/various_N_Claude 3 Haiku",
    "transition_prob_various_models_kz/various_N_Claude 3.5 Sonnet",
    "transition_prob_various_models_kz/various_N_Claude 3 Sonnet",
    "transition_prob_various_models_kz/various_N_GPT-4",
    "transition_prob_various_models_kz/various_N_GPT-4o",
    "transition_prob_various_models_kz/various_N_GPT-4 Turbo",
    "transition_prob_various_models_kz/various_N_GPT-3.5 Turbo",
    "transition_prob_various_models_kz/various_N_Llama 3 70B"
]

for directory in directories:
    results[directory] = process_directory(directory)

# Plot data for the second subplot
for idx, (model, data) in enumerate(results.items()):
    N_values, beta_values, beta_errors = zip(*data)
    model_name = model.split('_')[-1]
    color, marker = color_marker_map[model_name]
    ax2.errorbar(N_values, beta_values, yerr=beta_errors, fmt=marker, color=color, 
                capsize=5, label=model_name, linestyle='-', lw=2, markersize=10)

#beta treshold
x = np.arange(1, 1200)
y = 0.5*np.log(x)
ax2.plot(x, y, color='grey', ls='--')

# Configure the second plot
ax2.set_xscale('log')
ax2.set_xlabel('Group size $N$')
ax2.set_ylabel(r'Majority Force $\beta$')
ax2.plot([5, 1200], [1, 1], color='grey', ls='--')
ax2.set_xlim([8, 1200])
ax2.set_ylim([-2, 14.5])

# Define range of N for shading
N_vals = np.logspace(np.log10(8), np.log10(1200), 500)
beta_threshold = 0.5 * np.log(N_vals)

# Fill disordered region (beta < 1)
ax2.fill_between(N_vals, -2, 1, color='lightblue', alpha=0.3)

# Fill partial order region (1 < beta < beta_t)
ax2.fill_between(N_vals, 1, beta_threshold, color='khaki', alpha=0.3)

# Fill coordination region (beta > beta_t)
ax2.fill_between(N_vals, beta_threshold, 14.5, color='lightgreen', alpha=0.3)

# Annotate regions
ax2.text(500, -1.5, 'Uncoordinated', fontsize=20, color='black', ha='center', va='bottom')
ax2.text(320, 1.6, 'Partially coordinated', fontsize=20, color='black', ha='center', va='bottom')
ax2.text(100, 12, 'Coordinated', fontsize=20, color='black', ha='center', va='top')


# Collect all legend handles and labels
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine all handles and labels
all_handles = handles1 + handles2
all_labels = labels1 + labels2

# Remove duplicate labels while preserving order
seen = set()
unique_handles = []
unique_labels = []
for h, l in zip(all_handles, all_labels):
    if l not in seen:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)

# Add the legend at the top of the figure
fig.legend(unique_handles, unique_labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=22)

# Set subplot backgrounds to white
ax1.set_facecolor('white')
ax2.set_facecolor('white')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the legend at the top

# Save the figure
plt.savefig('plot3.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

#%%FIG 4

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# Function to extract the system size N and model name from the filename
def extract_info(filename):
    parts = filename.split('_')
    N = int(parts[2])
    model_name = parts[5]
    return N, model_name

# Initialize dictionaries to hold the data
data = {}
# Path to the directory containing the files
path = 'consensus_time/'

# Read all files with the specified pattern
file_pattern = os.path.join(path, 'list_times_*.txt')
files = glob.glob(file_pattern)

# Proimport os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# Function to extract the system size N and model name from the filename
def extract_info(filename):
    parts = filename.split('_')
    N = int(parts[2])
    model_name = parts[5]
    return N, model_name

# Initialize dictionaries to hold the data
data = {}
# Path to the directory containing the files
path = 'consensus_time/'

# Read all files with the specified pattern
file_pattern = os.path.join(path, 'list_times_*.txt')
files = glob.glob(file_pattern)

# Process each file
for file in files:
    N, model_name = extract_info(os.path.basename(file))
    if model_name not in data:
        data[model_name] = []
    
    # Read the consensus times from the file
    consensus_times = pd.read_csv(file, header=None).squeeze()
    
    # Normalize by N
    consensus_times_normalized = consensus_times / N
    
    # Store the normalized times with their corresponding N
    data[model_name].append((N, consensus_times_normalized))

# Prepare data for plotting
plot_data = {}

for model_name, values in data.items():
    plot_data[model_name] = {}
    for N, times in values:
        if N not in plot_data[model_name]:
            plot_data[model_name][N] = []
        plot_data[model_name][N].extend(times)
        
# Calculate mean and standard deviation
mean_std_data = {model_name: {'N': [], 'mean': [], 'min': [], 'max': []} for model_name in plot_data}

for model_name, values in plot_data.items():
    for N, times in sorted(values.items()):
        mean_std_data[model_name]['N'].append(N)
        mean_std_data[model_name]['mean'].append(np.mean(times))
        mean_std_data[model_name]['min'].append(np.min(times))
        mean_std_data[model_name]['max'].append(np.max(times))


# Load Curie-Weiss model data
cw_data_path = 'consensus_time/times_curie_weiss_3_75/'
cw_files1 = glob.glob(os.path.join(cw_data_path, 'N*_CW.txt'))


curie_weiss_data1 = []


for file in cw_files1:
    N = int(os.path.basename(file).split('_')[0][1:])
    data = pd.read_csv(file, delim_whitespace=True, header=None)
    consensus_times = data.iloc[:, -1]  # Last column
    mean_time = consensus_times.mean()
    std_time = consensus_times.std()
    curie_weiss_data1.append((N, mean_time, std_time))
    


curie_weiss_data1.sort(key=lambda x: x[0])

# Convert Curie-Weiss data to a DataFrame
curie_weiss_df1 = pd.DataFrame(curie_weiss_data1, columns=['N', 'mean', 'std'])

# Plotting
fig = plt.figure(figsize=(18, 9))
# Increase font size
plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
})
gs = GridSpec(2, 2, width_ratios=[1, 1],height_ratios=[15, 1], hspace=0.4, wspace=0.3)  # The last row is for the legend
ax2 = fig.add_subplot(gs[0, 0])
# ax1 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])


# Bottom consensus time plot (all T values, only GPT-4 models and specified times)
for model_name, values in mean_std_data.items():
    print(model_name)
    # if "gpt-4" in model_name:
    times_to_plot = np.isin(values['N'], curie_weiss_df1['N']) | np.isin(values['N'], [26, 30, 34, 55, 120, 140, 500, 1000])
    mean_values = np.array(values['mean'])[times_to_plot]
    min_values = np.array(values['min'])[times_to_plot]
    max_values = np.array(values['max'])[times_to_plot]
    
    # Calculate the difference between mean and min/max for error bars
    lower_error = mean_values - min_values
    upper_error = max_values - mean_values
    error = [lower_error, upper_error]  # tuple for asymmetric error bars

    label = filename_to_legend[model_name]
    ax2.errorbar(np.array(values['N'])[times_to_plot], mean_values, 
                 yerr=error, label=label, capsize=5, marker=color_marker_map[label][1], color=color_marker_map[label][0])
# Adding Curie-Weiss model data
ax2.errorbar(curie_weiss_df1['N'], curie_weiss_df1['mean'], yerr=curie_weiss_df1['std'], label=r'Curie-Weiss model', capsize=5, marker='x', color='grey')

ax2.set_xlabel(r'Group Size $N$', fontsize=20)
ax2.set_ylabel(r'Coordination Time $T_c$', fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim([0.4, 100])
ax2.grid(True)

# Get the handles and labels from ax1
handles, labels = ax2.get_legend_handles_labels()

# Place the legend below the left column plots
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.29, 0.05), fontsize=20)

# MMLU vs Critical N
data_points = [
    ("GPT-3.5 Turbo", 70, 2),
    ("GPT-4 Turbo", 85.4, 1000),
    ("GPT-4", 82.7, 600),
    ("Claude 3.5 Sonnet", 88.7, 1000),
    ("Claude 3 Sonnet", 79.0, 80),
    ("Llama 3 70B", 82, 30),
    ("Claude 2.0", 78.5, 15),
    ("Claude 3 Haiku", 75.2, 10),
    ("GPT-4o", 87.2, 80),
    ("Humans", 89, 150)
]

# Unpacking the data
labels, x, y = zip(*data_points)

# Convert x and y to NumPy arrays
x = np.array(x)
y = np.array(y)

# Define the exponential function
def exp_func(x, a, b):
    return a*x-b

# Fit the exponential function to the data (excluding GPT4, GPT4 Turbo, and Humans)
# fit_mask = (x != 85.4) & (x != 88.7)  & (x != 89) & (x != 82.7) 
fit_mask = (x != 88.7)  & (x != 89)
popt, pcov = curve_fit(exp_func, x[fit_mask], np.log(y[fit_mask]), maxfev=100000)

# Generate points for plotting the fitted curve
x_fit = np.linspace(0.5*min(x), 2*max(x), 500)
y_fit = np.exp(exp_func(x_fit, *popt))

# Different markers for different models
arrows = {
    "GPT 4 Turbo": '^'
}

# Models that have lower bounds
lower_bound_models = ["Claude 3.5 Sonnet"]

# Plotting the exponential fit
ax3.plot(x_fit, y_fit, color='tab:orange', lw=2, label='Exponential fit')

# Define offsets for annotations
offsets = {
    "GPT-3.5 Turbo": (15, 15),
    "GPT-4 Turbo": (-120, -0),
    "GPT-4": (10, -0),
    "Claude 3.5 Sonnet": (-150, 35),
    "Claude 3 Sonnet": (-130, 15),
    "Llama 3 70B": (5, -25),
    "Claude 2.0": (-100, 5),
    "Claude 3 Haiku": (5, -25),
    "GPT-4o":(5, -25),
    "Humans": (0, 10)
}

# Plotting the data
for i in range(len(labels)):
    print(labels[i])
    if labels[i] in lower_bound_models:
        # For these models, plot the data point
        color = color_marker_map[labels[i]][0]
        marker = color_marker_map[labels[i]][1]
        ax3.scatter(x[i], y[i], marker=marker, s=200, color=color)
        used_color = color
        
        # Compute arrowhead position in log scale
        log_yi = np.log10(y[i])
        arrowhead_log_y = log_yi + 0.2  # Adjust this value as needed
        arrowhead_y = 10 ** arrowhead_log_y
        
        # Add an upward arrow
        ax3.annotate(
            '',
            xy=(x[i], arrowhead_y),
            xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle='->', color=used_color, lw=2)
        )
    else:
        # Existing code for other models
        if labels[i] in arrows:
            ax3.scatter(x[i], y[i], marker=arrows[labels[i]], s=300, color='black')
            used_color = 'black'
        elif labels[i] == "Humans":
            ax3.scatter(x[i], y[i], marker='H', s=300, color='pink')
            used_color = 'pink'
        else:
            color = color_marker_map[labels[i]][0]
            marker = color_marker_map[labels[i]][1]
            ax3.scatter(x[i], y[i], marker=marker, s=200, color=color)
            used_color = color

    # Now, add text label next to the data point
    # Use annotate with offset
    xytext = offsets.get(labels[i], (5, 5))
    ax3.annotate(
        labels[i],
        (x[i], y[i]),
        textcoords="offset points",
        xytext=xytext,
        ha='left',
        color=used_color,
        fontsize=16
    )

# Rest of your plotting code remains the same
ax3.set_xlabel('Intelligence (MMLU Benchmark score)')
ax3.set_ylabel(r'Maximal group size $N_{c}$')
ax3.set_yscale('log')
ax3.set_xlim([69, 94])
ax3.set_ylim([1, 2200])
ax3.grid(True)


plt.tight_layout(pad=0)
plt.savefig('plot4.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()



