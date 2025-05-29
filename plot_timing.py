import os
import re
import matplotlib.pyplot as plt

def extract_total_time(timing_file):
    with open(timing_file, 'r') as f:
        for line in f:
            if line.startswith('total_time:'):
                match = re.search(r'total_time: ([\d.]+)\+/-([\d.]+)ms', line)
                if match:
                    mean, std = map(float, match.groups())
                    return mean, std
    return None, None

def main(parent_dir):
    run_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))]
    
    run_names = []
    means = []
    stds = []

    for run_dir in run_dirs:
        timing_path = os.path.join(run_dir, 'cuda_timing.txt')
        if os.path.exists(timing_path):
            mean, std = extract_total_time(timing_path)
            if mean is not None:
                run_names.append(os.path.basename(run_dir))
                means.append(mean)
                stds.append(std)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(run_names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Total Time (us)')
    plt.title('CUDA time per iteration')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig("./plots/timings.png")

if __name__ == "__main__":
    main('timing_outputs')  # Replace 'results' with your parent directory if needed