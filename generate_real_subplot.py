import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.rcParams['font.family'] = 'DeJavu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Load data
configs = ['4AP_6UAV', '4AP_9UAV', '4AP_12UAV']
data = {}

for config in configs:
    with open(f'/home/hzl/hyd/virtualForce/results/config_{config}_run2_20251115_*.json', 'r') as f:
        # Use glob to find the file
        import glob
        files = glob.glob(f'/home/hzl/hyd/virtualForce/results/config_{config}_run2_*.json')
        if files:
            with open(files[0], 'r') as f:
                data[config] = json.load(f)

# Algorithm names and colors
algorithms = {
    'VF': 'Balanced Virtual Force Algorithm',
    'GA': 'Discrete CF Genetic Algorithm', 
    'PSO': 'Distributed CF Particle Swarm Optimization'
}

colors = {
    'VF': '#e74c3c',   # Red
    'GA': '#3498db',   # Blue
    'PSO': '#2ecc71'   # Green
}

# ============================================================================
# Figure 1: Minimum Rate Comparison
# ============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(configs))
width = 0.25

# Plot 1a: Minimum Rate Values
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    min_rates = [data[cfg][alg_key]['min_rate'] for cfg in configs]
    ax1.bar(x + i*width, min_rates, width, label=alg_name, color=colors[alg_key], alpha=0.8)
        # Add value labels on bars
    for j, val in enumerate(min_rates):
        ax1.text(x[j] + i*width, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Configuration (Number of Ground APs and UAVs)', fontweight='bold')
ax1.set_ylabel('Minimum User Rate (Mbps)', fontweight='bold')
ax1.set_title('(a) Minimum User Rate Comparison', fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels([f'{cfg.replace("_", " ")}' for cfg in configs])
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max([data[cfg]['VF']['min_rate'] for cfg in configs]) * 1.15)

# Plot 1b: Minimum Rate Improvement Percentage
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    improvements = []
    for cfg in configs:
        initial = data[cfg]['initial']['min_rate']
        final = data[cfg][alg_key]['min_rate']
        improvement = ((final - initial) / initial) * 100
        improvements.append(improvement)
        bars = ax2.bar(x + i*width, improvements, width, label=alg_name, color=colors[alg_key], alpha=0.8)
            # Add value labels on bars
    for j, val in enumerate(improvements):
        ax2.text(x[j] + i*width, val + 1, f'{val:+.1f}%', ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Configuration (Number of Ground APs and UAVs)', fontweight='bold')
ax2.set_ylabel('Minimum Rate Improvement (%)', fontweight='bold')
ax2.set_title('(b) Minimum User Rate Improvement Percentage', fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels([f'{cfg.replace("_", " ")}' for cfg in configs])
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('/home/hzl/hyd/virtualForce/results/minimum_rate_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: minimum_rate_comparison.png")
plt.close()

# ============================================================================
# Figure 2: Total Sum Rate Improvement Percentage
# ============================================================================
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    improvements = []
    for cfg in configs:
        initial = data[cfg]['initial']['sum_rate']
        final = data[cfg][alg_key]['sum_rate']
        improvement = ((final - initial) / initial) * 100
        improvements.append(improvement)
        bars = ax.bar(x + i*width, improvements, width, label=alg_name, color=colors[alg_key], alpha=0.8)
            # Add value labels on bars
    for j, val in enumerate(improvements):
        y_pos = val + 0.15 if val >= 0 else val - 0.15
        va = 'bottom' if val >= 0 else 'top'
        ax.text(x[j] + i*width, y_pos, f'{val:+.1f}%', ha='center', va=va, fontsize=9)

ax.set_xlabel('Configuration (Number of Ground APs and UAVs)', fontweight='bold')
ax.set_ylabel('Sum Rate Improvement (%)', fontweight='bold')
ax.set_title('Total System Sum Rate Improvement Percentage', fontweight='bold', fontsize=13)
ax.set_xticks(x + width)
ax.set_xticklabels([f'{cfg.replace("_", " ")}' for cfg in configs])
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('/home/hzl/hyd/virtualForce/results/sum_rate_improvement_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sum_rate_improvement_comparison.png")
plt.close()

# ============================================================================
# Figure 3: UAV Position Changes
# ============================================================================
fig3 = plt.figure(figsize=(18, 5 * len(configs)))
gs = GridSpec(len(configs), 3, figure=fig3, hspace=0.35, wspace=0.25)

config_titles = {
    '4AP_6UAV': '4 Ground APs, 6 UAVs',
    '4AP_9UAV': '4 Ground APs, 9 UAVs',
    '4AP_12UAV': '4 Ground APs, 12 UAVs'
}

for row, config in enumerate(configs):
    for col, (alg_key, alg_name) in enumerate(algorithms.items()):
        ax = fig3.add_subplot(gs[row, col])
        
        # Get positions
        initial_pos = np.array(data[config]['initial']['UAV_pos'])
        final_pos = np.array(data[config][alg_key]['UAV_pos'])
        ground_ap_pos = np.array(data[config]['ground_AP_pos'])
        
        # Plot Ground APs
        ax.scatter(ground_ap_pos[:, 0], ground_ap_pos[:, 1], 
                  c='black', marker='^', s=200, label='Ground AP', zorder=5, edgecolors='white', linewidths=1.5)
        
        # Plot UAV trajectories
        for i in range(len(initial_pos)):
            # Initial position
            ax.scatter(initial_pos[i, 0], initial_pos[i, 1], 
                      c='lightgray', marker='o', s=100, zorder=3, edgecolors='gray', linewidths=1)
            
            # Final position
            ax.scatter(final_pos[i, 0], final_pos[i, 1], 
                      c=colors[alg_key], marker='o', s=150, zorder=4, edgecolors='white', linewidths=1.5)
            
            # Movement arrow
            ax.annotate('', xy=final_pos[i, :2], xytext=initial_pos[i, :2],
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=colors[alg_key], alpha=0.6))
            
            # UAV labels
            ax.text(final_pos[i, 0], final_pos[i, 1] + 25, f'U{i+1}', 
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # Add legend
        initial_patch = mpatches.Patch(color='lightgray', label='Initial Position')
        final_patch = mpatches.Patch(color=colors[alg_key], label='Final Position')
        ap_patch = mpatches.Patch(color='black', label='Ground AP')
        ax.legend(handles=[initial_patch, final_patch, ap_patch], loc='upper right', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('X Coordinate (m)', fontweight='bold')
        ax.set_ylabel('Y Coordinate (m)', fontweight='bold')
        
        if row == 0:
            ax.set_title(f'{alg_name.split()[0]} {alg_name.split()[1]} {"".join(alg_name.split()[2:])}', 
                        fontweight='bold', fontsize=11)
        
        # Add configuration label on the left
        if col == 0:
            ax.text(-0.25, 0.5, config_titles[config], 
                   transform=ax.transAxes, fontsize=11, fontweight='bold',
                   ha='right', va='center', rotation=90)
        
        # Grid and limits
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')

plt.suptitle('UAV Position Changes Before and After Optimization', 
            fontsize=14, fontweight='bold', y=0.995)
plt.savefig('/home/hzl/hyd/virtualForce/results/uav_position_changes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: uav_position_changes.png")
plt.close()   
print("\n" + "="*80)
print("All comparison plots generated successfully!")
print("="*80)
print("\nGenerated files:")
print("  1. minimum_rate_comparison.png")
print("  2. sum_rate_improvement_comparison.png")
print("  3. uav_position_changes.png")
print("\nLocation: /home/hzl/hyd/virtualForce/result/")
print("="*80)