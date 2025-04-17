#%%PLOT 1 SI

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folders
folders = [
    "data_SI/bias/no_shuffling",
    "data_SI/bias/shuffling"
]

# Initialize a figure for plotting with two subplots (one for each folder)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.rcParams.update({'font.size': 20})


# Initialize a dictionary to store the data by opinion name
data_dict = {}

# Loop over the folders and gather the data
for folder_path in folders:
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Extract opinion names from the filename
            name_parts = filename.split('_')
            opinion_k = name_parts[-2]
            opinion_z = name_parts[-1].split('.')[0]
            legend_label = f"{opinion_k}_{opinion_z}"
            
            # Load the data
            data = pd.read_csv(file_path, delimiter=',', header=None)
            
            # Store the data in the dictionary
            if legend_label not in data_dict:
                data_dict[legend_label] = {}
            data_dict[legend_label][folder_path] = data

# Plot the data, ensuring consistent legend order
for i, folder_path in enumerate(folders):
    ax = axes[i]
    
    for legend_label in sorted(data_dict.keys()):
        # Check if the data exists for the current folder and legend label
        if folder_path in data_dict[legend_label]:
            # Plot the data from the two-column GPT-4 files
            ax.plot(data_dict[legend_label][folder_path].iloc[:, 0], 
                    data_dict[legend_label][folder_path].iloc[:, 1], label=legend_label)
    
    # Add labels and legend to the subplot
    ax.set_xlabel(r'Collective opinion $m$', fontsize=20)
    ax.set_ylabel(r'Adoption probability $P(m)$', fontsize=20)
    ax.legend(fontsize=15)
    # ax.set_title(f'Data from {folder_path.split("/")[1]}')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('plot1_SI.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

#%%PLOT 2 SI

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folders for the first row (different_names)
folders_row1 = [
    "data_SI/claude_3.5_sonnet/different_names",
    "data_SI/gpt4turbo/different_names",
    "data_SI/llama3_70b/different_names"
]

# Define the folders for the second row (different_T)
folders_row2 = [
    "data_SI/claude_3.5_sonnet/different_T",
    "data_SI/gpt4turbo/different_T",
    "data_SI/llama3_70b/different_T"
]

# Define the folders for the third row (different_prompts)
folders_row3 = [
    "data_SI/claude_3.5_sonnet/different_prompts",
    "data_SI/gpt4turbo/different_prompts",
    "data_SI/llama3_70b/different_prompts"
]

# Titles for the first and third rows
titles = [
    "Claude 3.5 Sonnet",
    "GPT-4 Turbo",
    "Llama 3 70B"
]

# Initialize a figure for plotting with nine subplots (3 rows, 3 columns)
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Plotting for the first row (different_names)
data_dict_row1 = {}

for i, folder_path in enumerate(folders_row1):
    ax = axes[0, i]  # First row
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            name_parts = filename.split('_')
            k = name_parts[-2]
            z = name_parts[-1].split('.')[0]
            legend_label = f"{k}_{z}"
            data = pd.read_csv(file_path, delimiter=',', header=None)
            if legend_label not in data_dict_row1:
                data_dict_row1[legend_label] = {}
            data_dict_row1[legend_label][folder_path] = data

    for legend_label in sorted(data_dict_row1.keys()):
        if folder_path == "gpt4turbo/different_names":
            ax.plot(data_dict_row1[legend_label][folder_path].iloc[:, 0], 
                    data_dict_row1[legend_label][folder_path].iloc[:, 1], label=legend_label)
        else:
            ax.plot(data_dict_row1[legend_label][folder_path].iloc[:, 0], 
                    data_dict_row1[legend_label][folder_path].iloc[:, -1], label=legend_label)
    
    ax.set_xlabel(r'Collective opinion $m$', fontsize=22)
    ax.set_ylabel(r'Adoption probability $P(m)$', fontsize=22)
    ax.legend(fontsize=17)
    ax.set_title(titles[i], fontsize=24)  # Add titles only to the first row

# Plotting for the second row (different_T)
data_dict_row2 = {}

for i, folder_path in enumerate(folders_row2):
    ax = axes[1, i]  # Second row
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            name_parts = filename.split('_')
            temperature = name_parts[-4]
            data = pd.read_csv(file_path, delimiter=',', header=None)
            if temperature not in data_dict_row2:
                data_dict_row2[temperature] = {}
            data_dict_row2[temperature][folder_path] = data

    for temperature in sorted(data_dict_row2.keys(), key=float):
        if folder_path in data_dict_row2[temperature]:
            if folder_path == "gpt4turbo/different_T":
                ax.plot(data_dict_row2[temperature][folder_path].iloc[:, 0], 
                        data_dict_row2[temperature][folder_path].iloc[:, 1], label=f"T={temperature}")
            else:
                ax.plot(data_dict_row2[temperature][folder_path].iloc[:, 0], 
                        data_dict_row2[temperature][folder_path].iloc[:, -1], label=f"T={temperature}")
    
    ax.set_xlabel(r'Collective opinion $m$', fontsize=22)
    ax.set_ylabel(r'Adoption probability $P(m)$', fontsize=22)
    ax.legend(fontsize=17)

# Plotting for the third row (different_prompts)
data_dict_row3 = {}

for i, folder_path in enumerate(folders_row3):
    ax = axes[2, i]  # Third row
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            name_parts = filename.split('_')
            prompt_number = int(name_parts[-1].replace("pr", "").replace(".txt", ""))  # Extract prompt number
            legend_label = f"Prompt {prompt_number}"
            data = pd.read_csv(file_path, delimiter=',', header=None)
            if legend_label not in data_dict_row3:
                data_dict_row3[legend_label] = {}
            data_dict_row3[legend_label][folder_path] = data

    for legend_label in sorted(data_dict_row3.keys()):
        if folder_path == "gpt4turbo/different_prompts":
            ax.plot(data_dict_row3[legend_label][folder_path].iloc[:, 0], 
                    data_dict_row3[legend_label][folder_path].iloc[:, 1], label=legend_label)
        else:
            ax.plot(data_dict_row3[legend_label][folder_path].iloc[:, 0], 
                    data_dict_row3[legend_label][folder_path].iloc[:, -1], label=legend_label)
    
    ax.set_xlabel(r'Collective opinion $m$', fontsize=22)
    ax.set_ylabel(r'Adoption probability $P(m)$', fontsize=22)
    ax.legend(fontsize=17)
    # ax.set_title(titles[i], fontsize=24)  # Add titles only to the first and third rows

# Adjust layout and display the combined plot
plt.tight_layout()
plt.savefig('plot2_SI.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

#%%PLOT 3 SI

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
import pandas as pd

second_plot_dir = 'data_SI/transition_prob_majority_gpt4_kz'
gpt_4_dir = 'data_SI/various_N_GPT-4 Turbo'
fig = plt.figure(figsize=(18, 8))
plt.rcParams.update({'font.size': 20})
gs = GridSpec(1, 1) 
ax1 = fig.add_subplot(gs[0, 0])
#ax2 = fig.add_subplot(gs[0, 1])


# Data provided by the user
data = [
    ("Claude3.5Sonnet", 200000, 9.1),  
    ("GPT4o", 128000, 2.9),
    ("Claude3Opus", 200000, 8.6),
    ("GPT4Turbo", 128000, 4.8),
    ("GPT4", 8200, 3.7),
    ("Llama370b", 8200, 1.0),
    ("Claude3Sonnet", 200000, 2.7),
    ("Claude2.0", 100000, 0.8),
    ("Claude3Haiku", 200000, 0.11),
    ("GPT3.5", 16385, 0.15)
]



# Unpacking the data
labels, x, y = zip(*data)

# Convert x and y to NumPy arrays
x = np.array(x)
y = np.array(y)

# Calculate the Pearson correlation coefficient and the p-value
correlation_coefficient, p_value = pearsonr(x, y)
print(f"Correlation Coefficient: {correlation_coefficient:.2f}")
print(f"P-value: {p_value:.4f}")

# Different markers for different models
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', '*', '+']

# Plotting the data
for i in range(len(labels)):
    ax1.scatter(x[i], y[i], label=labels[i], marker=markers[i], s=200)

# Adding the linear regression line with confidence interval
sns.regplot(x=x, y=y, ax=ax1, ci=95, scatter=False, color='tab:orange', line_kws={"lw": 2})

ax1.set_xlabel('Context Window Length')
ax1.set_ylabel(r'Majority force $\beta$')

# # Set x-axis to scientific notation
# fig.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# fig.gca().ticklabel_format(style='sci', axis='x', scilimits=(0,0))

ax1.set_ylim([-2, 10])
ax1.legend(fontsize='14')
ax1.grid(True)



color = list(mcolors.TABLEAU_COLORS.keys())[:5]
marker = ['o', 's', 'D', '>', 'v']

# Define the tanh fitting function
def tanh_fit(x, beta):
    return 0.5 * (np.tanh(beta * x) + 1)

# Function to plot data
def plot_data_second(ax, directory):
    files_info = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            size = int(filename.split('_')[2].partition(".tx")[0])
            files_info.append((filename, size))
            
    # Sort files alphabetically by model name
    files_info.sort(key=lambda x: x[1])

    for i, (filename, size) in enumerate(files_info):
        # Read the data
        data = pd.read_csv(os.path.join(directory, filename), header=None)

        # Extract x and y data
        x = data.iloc[:, 0].values
        y = data.iloc[:, -1].values

        # Fit the data to the tanh function
        popt, _ = curve_fit(tanh_fit, x, y)
        beta = popt[0]
        print(size, beta)

        # Generate x values for the fit line
        x_fit = np.linspace(-1, 1, 100)
        y_fit = tanh_fit(x_fit, beta)

        # Plot the original data
        ax.scatter(x, y, label=fr'$N={size}$', color=color[i], marker=marker[i], s=500)

        # Plot the fit line
        ax.plot(x_fit, y_fit, color=color[i])

# # Plot the data for the second subplot
# plot_data_second(ax2, second_plot_dir)
# ax2.set_xlabel(r'Collective opinion $m$')
# ax2.set_ylabel(r'Adoption probability $P(m)$')
# ax2.set_ylim([-0.1, 1.1])
# ax2.legend()


# # Function to process each directory and fit the tanh function
# def process_directory(directory):
#     allowed_N_values = {30, 50, 100, 200, 500}
#     files_info = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.txt'):
#             full_path = os.path.join(directory, filename)
#             data = pd.read_csv(full_path, header=None)
#             # Extract x and y data
#             x = data.iloc[:, 0].values
#             y = data.iloc[:, -1].values
#             # Fit the data to the tanh function
#             popt, pcov = curve_fit(tanh_fit, x, y)
#             beta = popt[0]
#             beta_err = np.sqrt(np.diag(pcov))[0]  # Extract the standard deviation (error) for beta
#             # Extract N from filename
#             N = int(filename.split('_')[2])
#             # print(filename, N, beta, beta_err)

#             # Only include allowed N values
#             if N in allowed_N_values:
#                 files_info.append((N, beta, beta_err))
    
#     # Sort files by N
#     files_info.sort(key=lambda x: x[0])
    
#     return files_info

# results = {}
# for directory in [second_plot_dir, gpt_4_dir]:
#     results[directory] = process_directory(directory)

# ax3 = ax2.inset_axes([0.6, 0.15, 0.35, 0.35])

# # Plotting the second row
# for idx, (model, data) in enumerate(results.items()):
#     N_values, beta_values, beta_errors = zip(*data)
#     model_name = model.split('_')[-1]
#     ax3.errorbar(N_values, beta_values, yerr=beta_errors, fmt=marker[idx], color=color[idx], capsize=5, label=model_name, linestyle='-', lw=2, markersize='10')

# ax3.set_xscale('log')
# ax3.set_xlabel(r'$N$')
# ax3.set_ylabel(r'$\beta$')
# ax3.set_xlim([20, 1000])

plt.savefig('plot3_SI.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Data
data = [
    ("Claude 3.5 Sonnet", 88.7, 65, 92.0, 71.1, 91.6, 1283, 1000),  
    ("GPT-4o", 87.2, 49.9, 91, 76.6, 89.9, 1285, 80),
    #("Claude 3 Opus", 86.8, 50.4, 84.9, 60.1, 90.7, 1247, 8.6),
    ("GPT-4 Turbo", 85.4, 41.4, 86.6, 64.5, 85.1, 1245, 1000),
    ("GPT-4", 82.7, -1, 66.4, -1, -1, 1163, 600),
    ("Llama 3 70B", 82, 39.5, 81.7, 50.4, -1, 1206, 30),
    ("Claude 3 Sonnet", 79.0, 40.4, 73, 43.1, 83.5, 1201, 80),
    ("Claude 2.0", 78.5, -1, 71.2, -1, -1, -1, 15),
    ("Claude 3 Haiku", 75.2, 33.3, 75.9, 38.9, 75.1, 1179, 10),
    ("GPT-3.5 Turbo", 70, -1, -1, -1, -1, 1106, 2)
]

columns = ["Model", "MMLU", "GPQA", "HumanEval", "Math", "MGSM", "ELO", "Beta"]
df = pd.DataFrame(data, columns=columns)

# Plot style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16
})

# Benchmarks and visual setup
benchmarks = ["MMLU", "GPQA", "HumanEval", "Math", "MGSM", "ELO"]
benchmark_titles = {
    "MMLU": "MMLU",
    "GPQA": "GPQA",
    "HumanEval": "HumanEval",
    "Math": "Math",
    "MGSM": "MGSM",
    "ELO": "ELO Score"
}

colors = list(mcolors.TABLEAU_COLORS.values())
model_order = [
    "Claude 3.5 Sonnet", "GPT-4o", "Claude 3 Opus", "GPT-4 Turbo",
    "GPT-4", "Llama 3 70B", "Claude 3 Sonnet", "Claude 2.0",
    "Claude 3 Haiku", "GPT-3.5 Turbo"
]
markers = ['o', 's', 'D', '>', 'v', 'p', '*', 'h', '+', 'x']
color_marker_map = dict(zip(model_order, zip(colors, markers)))

# Plot
fig = plt.figure(figsize=(16, 8))
axes = GridSpec(3, 3, height_ratios=[3, 3, 0.2], hspace=0.3)
axes = [axes[i, j] for i in range(3) for j in range(3)]


for i, benchmark in enumerate(benchmarks):
    ax = fig.add_subplot(axes[i])
    valid = df[df[benchmark] != -1]
    for _, row in valid.iterrows():
        model = row["Model"]
        color, marker = color_marker_map[model]
        ax.plot(row[benchmark], row["Beta"], marker=marker, color=color,
                linestyle='None', markersize=8, label=model)
    ax.set_xlabel(benchmark_titles[benchmark])
    ax.set_ylabel("Critical group size $N_c$")
    #ax.set_ylim(-1, 10)
    #ax.set_title(f"{benchmark_titles[benchmark]} vs Beta")
    ax.set_yscale('log')

# Remove unused bottom-right plots
#for j in range(6, 9):
#    axes[j].axis("off")

# Legend
handles = [
    plt.Line2D([0], [0], marker=marker, color='w', label=model,
               markerfacecolor=color, markersize=10)
    for model, (color, marker) in color_marker_map.items()
]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=5, fontsize=12)

# Save and show
plt.tight_layout()
plt.savefig("plot4_SI.png", dpi=300)
plt.show()
