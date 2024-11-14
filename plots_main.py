#%%PLOT 1 PAPER

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
second_plot_dir = 'transition_prob_various_models_kz/50_kz'

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

# Increase font size
plt.rcParams.update({'font.size': 25})

# Initialize plot with main axis
fig = plt.figure(figsize=(24, 12))
gs = GridSpec(3, 2, width_ratios=[1, 1.5], height_ratios=[1, 3, 3], hspace=0.4, wspace=0.4)

ax1 = fig.add_subplot(gs[1:3, 0])
ax2 = fig.add_subplot(gs[1:3, 1])  
ax_legend = fig.add_subplot(gs[0, 0:2])

# Hide the legend axis
ax_legend.axis('off')

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
    for file in files:
        # Extract size and model name from the file name
        size, model_name = extract_info(os.path.basename(file))
        
        # Map the long model name to the legend name
        if model_name in filename_to_legend:
            model_name = filename_to_legend[model_name]
        
        # Only plot data for sizes 20 and 50
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
        
        # Plot the data
        ax.plot(data['rescaled_time'], np.abs(data['magnetization']), 
                linestyle=size_to_line_style[size], 
                color=color,
                linewidth=2.5)

# Collect final values of magnetization N=20
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


# Plot data for fully_connected N=50
plot_data(fully_connected_dirs[0], ax1)
ax1.set_xlabel(r'Time $t$')
ax1.set_ylabel(r'Consensus level $|m(t)|$')
ax1.set_ylim([-0.01, 1.01])
ax1.set_xlim([0, 10])

# Create an inset axis for the box plot N=50
ax3 = inset_axes(ax1, width="20%", height="100%", loc='center right', borderpad=-3.75)

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
                    medianprops=dict(color=color, linewidth=7), 
                    flierprops=dict(markerfacecolor=color))
        i += 1

ax3.set_yticks([])
ax3.set_xlim([0, 11])  # Adjust this limit based on the number of models
ax3.set_xticks([])
ax3.set_ylim([-0.01, 1.01])

# Second plot: Scatter and fit lines

# Define the tanh fitting function
def tanh_fit(x, beta):
    return 0.5 * (np.tanh(beta * x) + 1)

# Initialize the inset for the second subplot
ax_inset = ax2.inset_axes([0.1, 0.6, 0.35, 0.35])

# Set individual subplot backgrounds to white
ax2.set_facecolor('white')
ax_inset.set_facecolor('white')
ax_inset.grid(True)
ax_inset.set_yticks([0, 0.5, 1])


# Function to plot data
def plot_data_second(ax, inset, directory, plot_inset=False):
    global filename_to_legend
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
        print(model_name, beta)

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
            inset.scatter(x_scaled, y, label=fr'{model_name} ($\beta={beta:.1f}$)', color=color, marker=marker)

# Plot the data for the second subplot
plot_data_second(ax2, ax_inset, second_plot_dir, plot_inset=True)
ax2.set_xlabel(r'Collective opinion $m$')
ax2.set_ylabel(r'Adoption probability $P(m)$')
ax2.set_ylim([-0.1, 1.3])

# Sort sizes and models
unique_sizes.sort()
unique_models.sort()

# Add legend directly from the scatter plot with both markers and lines
handles, labels = ax2.get_legend_handles_labels()
new_handles = []
for handle, label in zip(handles, labels):
    if isinstance(handle, plt.Line2D):
        color = handle.get_color()
        marker = handle.get_marker()
    elif isinstance(handle, PathCollection):
        color = handle.get_facecolors()[0]
        marker = handle.get_paths()[0]

    new_handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle='-', label=label, markersize=10))

ax_legend.legend(new_handles, labels, loc='upper center', fontsize=22, ncol=5, bbox_to_anchor=(0.47, 0.75))

# Hide the custom legend subplot axis
ax_legend.axis('off')


plt.tight_layout(rect=[0, 0.05, 1, 1])

# Show plot
plt.savefig('plot1.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


#%%PLOT 2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import seaborn as sns

# Define the tanh fitting function
def tanh_fit(x, beta):
    return 0.5 * (np.tanh(beta * x) + 1)

# Data for MMLU and Beta
data = [
    ("Claude 3.5 Sonnet", 88.7, 9.1),  
    ("GPT-4o", 88.7, 2.9),
    ("Claude 3 Opus", 86.8, 8.6),
    ("GPT-4 Turbo", 86.5, 4.8),
    ("GPT-4", 86.4, 3.7),
    ("Llama 3 70B", 82, 1.0),
    ("Claude 3 Sonnet", 79.0, 2.7),
    ("Claude 2.0", 78.5, 0.8),
    ("Claude 3 Haiku", 75.2, 0.11),
    ("GPT-3.5 Turbo", 70, 0.15)
]

# Create a dictionary for MMLU benchmarks
mmlu_dict = {item[0]: item[1] for item in data}

# Directory containing the data
second_plot_dir = 'transition_prob_various_models_kz/50_kz'

# Extract beta values and their errors
def extract_beta_and_error(directory):
    global filename_to_legend
    beta_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            model_name = filename.split('_')[4].partition(".tx")[0]
            if model_name in filename_to_legend:
                model_name = filename_to_legend[model_name]

            # Read the data
            data = pd.read_csv(os.path.join(directory, filename), header=None)
            x = data.iloc[:, 0].values
            y = data.iloc[:, -1].values

            # Fit the data to the tanh function
            popt, pcov = curve_fit(tanh_fit, x, y)
            beta = popt[0]
            beta_err = np.sqrt(np.diag(pcov))[0]  # Error on beta
            
            # Store the result
            beta_dict[model_name] = (beta, beta_err)
    
    return beta_dict

# Get the beta values and errors
beta_dict = extract_beta_and_error(second_plot_dir)

# Prepare data for plot
plot_data = []
for model_name, mmlu, _ in data:
    if model_name in beta_dict:
        beta, beta_err = beta_dict[model_name]
        plot_data.append((model_name, mmlu, beta, beta_err))
        
# Unpack the plot data
labels, mmlu_values, beta_values, beta_errors = zip(*plot_data)

# Convert to numpy arrays for plotting
mmlu_values = np.array(mmlu_values)
beta_values = np.array(beta_values)
beta_errors = np.array(beta_errors)

# Convert x and y to NumPy arrays
x = np.array(mmlu_values)
y = np.array(beta_values)

# Create the figure with a larger height to accommodate the legend
fig = plt.figure(figsize=(24, 14))
plt.rcParams.update({'font.size': 25})

# Plotting the first row
ax1 = plt.subplot(2, 2, 1)
for i, label in enumerate(labels):
    # plt.scatter(x[labels.index(label)], y[labels.index(label)], label=label, color=color_marker_map[label][0], marker=color_marker_map[label][1], s=100)
    plt.errorbar(mmlu_values[i], beta_values[i], yerr=beta_errors[i], capsize=5, label=label, color=color_marker_map[label][0], marker=color_marker_map[label][1], markersize='15')

# Adding the linear regression line with confidence interval
sns.regplot(x=x, y=y, ci=95, scatter=False, color='tab:orange', line_kws={"lw": 2})
plt.xlabel('Languange understanding score (MMLU)')
plt.ylabel(r'Majority force $\beta$')
plt.ylim([-2, 10])

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
            # print(filename, N, beta, beta_err)

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

# Plotting the second row
ax2 = plt.subplot(2, 2, 2)
for idx, (model, data) in enumerate(results.items()):
    N_values, beta_values, beta_errors = zip(*data)
    model_name = model.split('_')[-1]
    color, marker = color_marker_map[model_name]
    ax2.errorbar(N_values, beta_values, yerr=beta_errors, fmt=marker, color=color, capsize=5, label=model_name, linestyle='-', lw=2, markersize='10')


ax2.set_xscale('log')
ax2.set_xlabel('Group size $N$')
ax2.set_ylabel(r'Majority Force $\beta$')
ax2.plot([5, 1000], [1, 1], color='grey', ls='--')
ax2.set_xlim([8, 1200])
ax2.set_ylim([-2, 14.5])

# Create a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles.extend(handles2)
labels.extend(labels2)

# Remove duplicate labels while preserving order
seen = set()
unique_handles = []
unique_labels = []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)

# Add the legend below both plots
plt.figlegend(unique_handles, unique_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.97), fontsize='22')

# Adjust the layout to make room for the legend
plt.tight_layout()

# Set figure background to transparent
# fig.patch.set_alpha(0)  # Makes the figure background (outside the axes) transparent

# Ensure axes background remains opaque
ax1.set_facecolor('white')
ax2.set_facecolor('white')

# Save and show the plot
plt.savefig('plot2.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

#%%PLOT 3 PAPER 


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
plt.rcParams.update({'font.size': 20})
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
ax2.set_ylabel(r'Consensus Time $T_c$', fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim([0.4, 100])
ax2.grid(True)

# Get the handles and labels from ax1
handles, labels = ax2.get_legend_handles_labels()

# Place the legend below the left column plots
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.29, 0.05))

# MMLU vs Critical N
data_points = [
    ("GPT-3.5 Turbo", 70, 2),
    ("GPT-4 Turbo", 86.5, 1000),
    ("Claude 3.5 Sonnet", 88.7, 1000),
    ("Claude 3 Sonnet", 79.0, 150),
    ("Llama 3 70B", 82, 50),
    ("Claude 2.0", 78.5, 20),
    ("Claude 3 Haiku", 75.2, 10),
    ("GPT-4o", 88.71, 150),
    ("Humans", 89, 200)
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
fit_mask = (x != 86.5) & (x != 88.7)  & (x != 89) 
popt, pcov = curve_fit(exp_func, x[fit_mask], np.log(y[fit_mask]), maxfev=100000)

# Generate points for plotting the fitted curve
x_fit = np.linspace(0.5*min(x), 2*max(x), 500)
y_fit = np.exp(exp_func(x_fit, *popt))

# Different markers for different models
arrows = {
    "GPT 4 Turbo": '^'
}

# Models that have lower bounds
lower_bound_models = ["GPT-4 Turbo", "Claude 3.5 Sonnet"]

# Plotting the exponential fit
ax3.plot(x_fit, y_fit, color='tab:orange', lw=2, label='Exponential fit')

# Define offsets for annotations
offsets = {
    "GPT-3.5 Turbo": (15, 15),
    "GPT-4 Turbo": (-100, -25),
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
ax3.set_xlabel('Language Understanding score (MMLU)', fontsize=20)
ax3.set_ylabel(r'Maximal group size $N_{c}$', fontsize=20)
ax3.set_yscale('log')
ax3.set_xlim([69, 94])
ax3.set_ylim([1, 2200])
ax3.grid(True)


plt.tight_layout(pad=0)
plt.savefig('plot3.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()
