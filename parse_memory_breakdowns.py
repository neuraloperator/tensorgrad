import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_cuda_mem_file(file_path):
    memory_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Max ") or line.startswith("Peak total memory"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    key = parts[1] if parts[0] == "Max" else "Peak_Total"
                    if key != "Peak_Total":
                        key = key[0] + key[1:].lower()
                    value = float(parts[-1])
                    memory_data[key] = value
    return memory_data

def collect_all_runs(base_dir, regex=''):
    pattern = re.compile(regex)
    all_data = {}
    for run_name in os.listdir(base_dir):
        if pattern.match(run_name):
            run_path = os.path.join(base_dir, run_name)
            mem_file = os.path.join(run_path, "max_mem.txt")
            if os.path.isdir(run_path) and os.path.isfile(mem_file):
                memory_data = parse_cuda_mem_file(mem_file)
                all_data[run_name] = memory_data
    return pd.DataFrame.from_dict(all_data, orient="index").fillna(0)

############
# Plotting #
############
# Figure sizes for paper quality (in inches)
FIG_SIZE = {
    'small': (8, 6),
    'medium': (10, 7),
    'large': (12, 8),
    'wide': (15, 9),    
    'square': (9, 9)    
}

def apply_mpl_settings():
    """Apply matplotlib settings for consistent plot appearance"""
    
    # Set font to DejaVu Serif
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # Use DejaVu Serif for math text
    
    # Font settings for paper publication quality
    FONT_SIZE = {
        'small': 22,       
        'medium': 24,      
        'large': 26,       
        'x-large': 28,     
        'xx-large': 30     
    }


    # example : plt.figure(figsize=sf.FIG_SIZE['large'])
    
    plt.rcParams['font.size'] = FONT_SIZE['large']
    plt.rcParams['axes.titlesize'] = FONT_SIZE['large']
    plt.rcParams['axes.labelsize'] = FONT_SIZE['large']
    plt.rcParams['xtick.labelsize'] = FONT_SIZE['large']
    plt.rcParams['ytick.labelsize'] = FONT_SIZE['large']
    plt.rcParams['legend.fontsize'] = FONT_SIZE['small']
    plt.rcParams['figure.titlesize'] = FONT_SIZE['large']
    
    # Additional settings for paper quality
    plt.rcParams['axes.linewidth'] = 1.5
    # marker size
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    
    # Disable LaTeX since it's not available
    plt.rcParams['text.usetex'] = False

def get_color_palette(categories, cmap_name='tab20'):
    cmap = plt.get_cmap(cmap_name)
    return {category: cmap(i % cmap.N) for i, category in enumerate(categories)}

def compute_correct_intermediates(df, extra_cols_to_drop=[]):
    # I added this to correct a bug instead of rerunning everything
    # importantly, if the intermediates were computed correctly at profile time,
    # this would return the same result.
    cols_to_drop = ["Peak_Total"] + extra_cols_to_drop
    df['Intermediate'] = df["Peak_Total"] - df.drop(columns=cols_to_drop).sum(axis=1)
    df = df.drop(columns=extra_cols_to_drop, errors='ignore')
    return df


def plot_stacked_bars(df, save_name):
    df = df.sort_values(by='Peak_Total',ascending=False)
    df_plot = df.drop(columns=["Peak_Total"],\
                    errors='ignore')  # exclude Peak_Total
    
    # Get colormap
    categories = df_plot.columns
    colors = get_color_palette(categories, cmap_name='tab20b')

    # Map each column to a consistent color
    #color_list = [colors[col] for col in categories]
    color_list = ['#687d4e', '#2f3e22', '#f4ecd4', '#d9a460', '#b67635', '#8c5b3e', '#5f4030']

    df_plot.plot(kind='bar', stacked=True, figsize=(12, 6), color=color_list)
    plt.ylabel("Memory (GB)")
    plt.title("Stacked Memory Usage by Category per Run")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    plt.savefig(save_name)

def plot_hbars(df, save_name):
    df = df.sort_values(by='Peak_Total',ascending=True)
    df_plot = df.drop(columns=["Peak_Total"],\
                    errors='ignore')  # exclude Peak_Total
    
    # Get colormap
    categories = df_plot.columns
    # Map each column to a consistent color
    color_list = ['#687d4e', '#2f3e22', '#f4ecd4', '#d9a460', '#b67635', '#8c5b3e', '#5f4030']

    ax = df_plot.plot(
        kind='barh',                 # <-- horizontal bars
        stacked=True,
        figsize=FIG_SIZE['large'],
        color=color_list,
        #edgecolor='none'
    )
    ax.set_xlabel("Memory (GB)")
    ax.set_xlim([0,100.])
    ax.set_title("Memory Usage by Category")
    
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    ax.legend(
        title="Category",
        loc='upper right',
        bbox_to_anchor=(0.97, 0.55),  # shift slightly inward
        bbox_transform=ax.transAxes,  # make it relative to axes box
        frameon=True,
        #borderpad=0.3,
        #handletextpad=0.5
    )

    # Don't use tight_layout; use manual layout
    #plt.subplots_adjust(right=1.4)  # shift right edge inward
    #plt.tight_layout()
    plt.savefig(save_name)



if __name__ == "__main__":
    import sys
    apply_mpl_settings()
    reg = "^(?!.*c256).*128modes_.*"
    all_runs = collect_all_runs(sys.argv[1], reg)
    print(all_runs)
    all_runs_processed = compute_correct_intermediates(all_runs, 
                                                       extra_cols_to_drop=[
                                                           "Temp",
                                                           "Input",
                                                           "Autograd_detail",
                                                           #"Activation",
                                                       ])
    print(all_runs_processed)
    plot_stacked_bars(all_runs_processed, 'plots/all_bars.png')

    # plot just a subset
    final_plot_df = all_runs_processed.loc[[
        "adam_256C_128modes_full", # baseline
        #"adam_256C_128modes_mixed_half", # baseline half
        "adam_256C_128modes_mixed_half_activation_ckpt", # baseline half
        "TensorGRaD_25%_256C_128modes_full", # ours full
        "TensorGRaD_25%_256C_128modes_mixed_half", # ours full
        "TensorGRaD_25%_256C_128modes_mixed_half_activation_ckpt", # ours full
        ]]

    label_map = {
        "adam_256C_128modes_full": "AdamW",
        #"adam_256C_128modes_mixed_half": "16-bit AdamW",
        "adam_256C_128modes_mixed_half_activation_ckpt": "AdamW-H+",
        "TensorGRaD_25%_256C_128modes_full": "TensorGRaD", # ours full
        "TensorGRaD_25%_256C_128modes_mixed_half": "TensorGRaD-H", # ours full
        "TensorGRaD_25%_256C_128modes_mixed_half_activation_ckpt": "TensorGRaD-H+", # ours full
    }

    final_plot_df = final_plot_df.rename(index=label_map)
    print(final_plot_df)

    plot_stacked_bars(final_plot_df, save_name="./plots/main_fig.png")
    plot_hbars(final_plot_df, save_name="./plots/main_fig_horiz.pdf")
    plot_hbars(final_plot_df, save_name="./plots/main_fig_horiz.png")

