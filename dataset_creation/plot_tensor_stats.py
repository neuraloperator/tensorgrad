import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import defaultdict

def load_json_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_sequential_ids(data):
    """Convert timestamp IDs to sequential numbers (0,1,2,...) based on their original values"""
    unique_ids = sorted(set(d['projector_id'] for d in data))
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    return id_map

def group_by_projector(data):
    """Group data by projector_id"""
    grouped = defaultdict(list)
    id_map = get_sequential_ids(data)
    for entry in data:
        # Convert timestamp ID to sequential ID
        entry['sequential_id'] = id_map[entry['projector_id']]
        grouped[entry['sequential_id']].append(entry)
    return grouped

def get_layer_label(projector_id, projector_data):
    """Get a label for the layer, including shape if available"""
    label = f'Layer {projector_id}'
    if projector_data and 'shape' in projector_data[0]:
        label += f' {projector_data[0]["shape"]}'
    return label

def plot_statistics_over_time(data, stat_name, title, ylabel, log_scale=False):
    plt.figure(figsize=(12, 6))
    grouped_data = group_by_projector(data)
    
    # Use a different color for each projector
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_data)))
    
    for (projector_id, projector_data), color in zip(grouped_data.items(), colors):
        steps = [d['step'] for d in projector_data]
        values = [d[stat_name] for d in projector_data]
        label = get_layer_label(projector_id, projector_data)
        plt.plot(steps, values, marker='o', linestyle='-', markersize=4, 
                label=label, color=color)
    
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()

def plot_histograms(data, step_idx=-1):
    """Plot separate histograms for each layer"""
    grouped_data = group_by_projector(data)
    n_layers = len(grouped_data)
    
    # Calculate grid dimensions
    n_cols = min(3, n_layers)  # Max 3 columns
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Use a different color for each projector
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    for idx, ((projector_id, projector_data), color) in enumerate(zip(grouped_data.items(), colors)):
        ax = axes[idx]
        hist_data = projector_data[step_idx]['histogram']
        bins = hist_data['bins']
        bin_edges = np.array(hist_data['bin_edges'])  # Convert to numpy array
        label = get_layer_label(projector_id, projector_data)
        
        # Calculate bin centers and widths
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Set x-axis labels to show bin ranges
        x = np.arange(len(bins))
        ax.set_xticks(x)
        bin_labels = [f'[{edges[0]:.2e},\n{edges[1]:.2e}]' 
                     for edges in zip(bin_edges[:-1], bin_edges[1:])]
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        
        ax.set_title(f'{label}\nStep {projector_data[step_idx]["step"]}')
        ax.set_xlabel(f'Value Range (bin width: {bin_width:.2e})')
        ax.set_ylabel('Count (log scale)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add min/max/mean as text
        stats_text = (f'Min: {projector_data[step_idx]["min"]:.2e}\n'
                     f'Max: {projector_data[step_idx]["max"]:.2e}\n'
                     f'Mean: {projector_data[step_idx]["mean"]:.2e}\n'
                     f'Bin width: {bin_width:.2e}')
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Compute top-5% bins
        total = sum(bins)
        threshold = total * 0.05
        cum = 0
        top_5_bins = []
        
        # Count from right to left
        for idx in range(len(bins)-1, -1, -1):
            if cum + bins[idx] > threshold:
                break
            cum += bins[idx]
            top_5_bins.append(idx)
        
        # Draw bars, outlining the "top-5%" bins
        color_top_5 = 'black'
        for i, count in enumerate(bins):
            if i in top_5_bins:
                ax.bar(i, count, width=0.8, color=color, 
                      hatch='////', edgecolor=color_top_5, linewidth=1)
            else:
                ax.bar(i, count, width=0.8, color=color)
        
        # Optional single label
        if top_5_bins:
            ax.text(
                0.5, 0.9, 'Top 5 %', transform=ax.transAxes,
                color=color_top_5, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    # Hide unused subplots
    for idx in range(len(grouped_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution of Reconstructed Tensor Values\n(10 equal-width bins)', y=1.02)
    plt.tight_layout()
    return fig

def plot_min_max_mean(data):
    """Plot min, max, and mean values over time for each projector"""
    plt.figure(figsize=(15, 8))
    grouped_data = group_by_projector(data)
    
    # Use a different color for each projector
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_data)))
    
    for (projector_id, projector_data), color in zip(grouped_data.items(), colors):
        steps = [d['step'] for d in projector_data]
        label = get_layer_label(projector_id, projector_data)
        
        # Plot min, max, and mean with different line styles
        plt.plot(steps, [d['min'] for d in projector_data], 
                label=f'Min ({label})', 
                color=color, linestyle='--', marker='o', markersize=4)
        plt.plot(steps, [d['max'] for d in projector_data], 
                label=f'Max ({label})', 
                color=color, linestyle=':', marker='o', markersize=4)
        plt.plot(steps, [d['mean'] for d in projector_data], 
                label=f'Mean ({label})', 
                color=color, linestyle='-', marker='o', markersize=4)
    
    plt.title('Min, Max, and Mean Values Over Time')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()

def plot_tensor_histograms(data, step_idx=-1):
    """Plot separate histograms for tensor statistics using quantile information"""
    grouped_data = group_by_projector(data)
    n_layers = len(grouped_data)
    
    # Calculate grid dimensions
    n_cols = min(3, n_layers)  # Max 3 columns
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Use a different color for each projector
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    
    for idx, ((projector_id, projector_data), color) in enumerate(zip(grouped_data.items(), colors)):
        ax = axes[idx]
        stats = projector_data[step_idx]
        label = get_layer_label(projector_id, projector_data)
        
        # Create a box plot using the quantile information
        box_data = [
            stats['min'],
            stats['q1'],
            stats['median'],
            stats['q3'],
            stats['max']
        ]
        
        # Plot box plot
        ax.boxplot([box_data], vert=False, widths=0.6, patch_artist=True)
        
        # Add mean as a red diamond
        ax.scatter([stats['mean']], [1], color='red', marker='D', s=100, label='Mean')
        
        # Add outliers count
        outliers_percent = (stats['outliers_count'] / stats['total_elements']) * 100
        
        ax.set_title(f'{label}\nStep {stats["step"]}')
        ax.set_xlabel('Value (log scale)')
        ax.set_xscale('log')
        ax.set_yticks([])  # Hide y-axis ticks
        ax.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (f'Min: {stats["min"]:.2e}\n'
                     f'Q1: {stats["q1"]:.2e}\n'
                     f'Median: {stats["median"]:.2e}\n'
                     f'Q3: {stats["q3"]:.2e}\n'
                     f'Max: {stats["max"]:.2e}\n'
                     f'Mean: {stats["mean"]:.2e}\n'
                     f'Std: {stats["std"]:.2e}\n'
                     f'Outliers: {outliers_percent:.1f}%')
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(grouped_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution of Tensor Values\n(Box plots showing min, Q1, median, Q3, max)', y=1.02)
    plt.tight_layout()
    return fig

def main():
    # Create output directories for different types of plots
    base_dir = Path('tensor_stats_plots')
    recon_dir = base_dir / 'reconstruction'
    diff_dir = base_dir / 'difference'
    tensor_dir = base_dir / 'tensor'
    
    for dir_path in [base_dir, recon_dir, diff_dir, tensor_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Load data for each type
    recon_data = load_json_data('tensor_stats/reconstruction_stats_[64, 64, 64, 33].json')
    diff_data = load_json_data('tensor_stats/difference_stats_[64, 64, 64, 33].json')
    tensor_data = load_json_data('tensor_stats/tensor_stats_[64, 64, 64, 33].json')
    
    # Plot basic statistics for each type
    stats_to_plot = [
        ('mean', 'Mean Value Over Time', 'Mean Value', True),
        ('std', 'Standard Deviation Over Time', 'Standard Deviation', True),
    ]
    
    # Plot reconstruction statistics
    for stat, title, ylabel, log_scale in stats_to_plot:
        fig = plot_statistics_over_time(recon_data, stat, f'Reconstruction {title}', ylabel, log_scale)
        fig.savefig(recon_dir / f'{stat}_over_time.png', bbox_inches='tight')
        plt.close(fig)
    
    # Plot reconstruction histograms
    fig = plot_histograms(recon_data)
    fig.suptitle('Distribution of Reconstructed Tensor Values\n(10 equal-width bins)', y=1.02)
    fig.savefig(recon_dir / 'histograms.png', bbox_inches='tight')
    plt.close(fig)
    
    # Plot reconstruction min, max, mean
    fig = plot_min_max_mean(recon_data)
    fig.suptitle('Min, Max, and Mean Values of Reconstructed Tensors Over Time', y=1.02)
    fig.savefig(recon_dir / 'min_max_mean.png', bbox_inches='tight')
    plt.close(fig)
    
    # Plot difference statistics
    for stat, title, ylabel, log_scale in stats_to_plot:
        fig = plot_statistics_over_time(diff_data, stat, f'Difference {title}', ylabel, log_scale)
        fig.savefig(diff_dir / f'{stat}_over_time.png', bbox_inches='tight')
        plt.close(fig)
    
    # Plot difference histograms
    fig = plot_histograms(diff_data)
    fig.suptitle('Distribution of Tensor Difference Values\n(10 equal-width bins)', y=1.02)
    fig.savefig(diff_dir / 'histograms.png', bbox_inches='tight')
    plt.close(fig)
    
    # Plot difference min, max, mean
    fig = plot_min_max_mean(diff_data)
    fig.suptitle('Min, Max, and Mean Values of Tensor Differences Over Time', y=1.02)
    fig.savefig(diff_dir / 'min_max_mean.png', bbox_inches='tight')
    plt.close(fig)
    
    # Plot tensor statistics
    for stat, title, ylabel, log_scale in stats_to_plot:
        fig = plot_statistics_over_time(tensor_data, stat, f'Original Tensor {title}', ylabel, log_scale)
        fig.savefig(tensor_dir / f'{stat}_over_time.png', bbox_inches='tight')
        plt.close(fig)
    
    # Plot tensor histograms
    fig = plot_histograms(tensor_data)
    fig.suptitle('Distribution of Original Tensor Values\n(10 equal-width bins)', y=1.02)
    fig.savefig(tensor_dir / 'histograms.png', bbox_inches='tight')
    plt.close(fig)
    
    # Plot tensor min, max, mean
    fig = plot_min_max_mean(tensor_data)
    fig.suptitle('Min, Max, and Mean Values of Original Tensors Over Time', y=1.02)
    fig.savefig(tensor_dir / 'min_max_mean.png', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plots have been saved to:")
    print(f"- Reconstruction plots: {recon_dir}")
    print(f"- Difference plots: {diff_dir}")
    print(f"- Tensor plots: {tensor_dir}")

if __name__ == "__main__":
    main() 