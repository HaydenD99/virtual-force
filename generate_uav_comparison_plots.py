"""
UAV配置对比可视化 - 专业格式
参考generate_real_subplot.py的格式
对比6/9/12 UAVs配置下的性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Load data
configs = ['6UAV', '9UAV', '12UAV']
data = {}

# 6 UAVs数据
with open('result/original_6uav_seed51.json', 'r') as f:
    data['6UAV'] = json.load(f)

# 9 UAVs数据（从batch2_partial_11.json）
data['9UAV'] = {
    "initial": {
        "min_rate": 30.593229480255477,
        "sum_rate": 2613.6417776916123,
    },
    "VF": {
        "min_rate": 38.45973938484597,
        "sum_rate": 2747.3818989444912,
    },
    "GA": {
        "min_rate": 34.61927199968264,
        "sum_rate": 2607.085592381084,
    },
    "PSO": {
        "min_rate": 34.884425617079174,
        "sum_rate": 2651.1635258693295,
    },
    "NewSSA": {
        "min_rate": 31.01316688365497,
        "sum_rate": 2502.491888999317,
    }
}

# 12 UAVs数据
with open('result/original_12uav_seed51.json', 'r') as f:
    data['12UAV'] = json.load(f)

# Algorithm names and colors
algorithms = {
    'VF': 'Balanced Virtual Force',
    'GA': 'Genetic Algorithm',
    'PSO': 'Particle Swarm Optimization',
    'NewSSA': 'New Sparrow Search Algorithm'
}

colors = {
    'VF': '#e74c3c',    # Red
    'GA': '#3498db',    # Blue
    'PSO': '#2ecc71',   # Green
    'NewSSA': '#f39c12' # Orange
}

# ============================================================================
# Figure 1: Minimum Rate Comparison
# ============================================================================
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(configs))
width = 0.2

# Plot 1a: Minimum Rate Values
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    min_rates = [data[cfg][alg_key]['min_rate'] for cfg in configs]
    ax1.bar(x + i*width, min_rates, width, label=alg_name, color=colors[alg_key], alpha=0.8)
    # Add value labels on bars
    for j, val in enumerate(min_rates):
        ax1.text(x[j] + i*width, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Configuration (4 Ground APs + N UAVs)', fontweight='bold')
ax1.set_ylabel('Minimum User Rate (Mbps)', fontweight='bold')
ax1.set_title('(a) Minimum User Rate Comparison', fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels([f'4AP + {cfg}' for cfg in configs])
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

ax2.set_xlabel('Configuration (4 Ground APs + N UAVs)', fontweight='bold')
ax2.set_ylabel('Minimum Rate Improvement (%)', fontweight='bold')
ax2.set_title('(b) Minimum User Rate Improvement Percentage', fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels([f'4AP + {cfg}' for cfg in configs])
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('result/minimum_rate_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: result/minimum_rate_comparison.png")
plt.close()

# ============================================================================
# Figure 2: Sum Rate Comparison
# ============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 2a: Sum Rate Values
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    sum_rates = [data[cfg][alg_key]['sum_rate'] for cfg in configs]
    ax1.bar(x + i*width, sum_rates, width, label=alg_name, color=colors[alg_key], alpha=0.8)
    # Add value labels on bars
    for j, val in enumerate(sum_rates):
        ax1.text(x[j] + i*width, val + 30, f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Configuration (4 Ground APs + N UAVs)', fontweight='bold')
ax1.set_ylabel('System Sum Rate (Mbps)', fontweight='bold')
ax1.set_title('(a) System Sum Rate Comparison', fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels([f'4AP + {cfg}' for cfg in configs])
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max([data[cfg]['VF']['sum_rate'] for cfg in configs]) * 1.1)

# Plot 2b: Sum Rate Improvement Percentage
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    improvements = []
    for cfg in configs:
        initial = data[cfg]['initial']['sum_rate']
        final = data[cfg][alg_key]['sum_rate']
        improvement = ((final - initial) / initial) * 100
        improvements.append(improvement)
    bars = ax2.bar(x + i*width, improvements, width, label=alg_name, color=colors[alg_key], alpha=0.8)
    # Add value labels on bars
    for j, val in enumerate(improvements):
        y_pos = val + 0.15 if val >= 0 else val - 0.15
        va = 'bottom' if val >= 0 else 'top'
        ax2.text(x[j] + i*width, y_pos, f'{val:+.1f}%', ha='center', va=va, fontsize=8)

ax2.set_xlabel('Configuration (4 Ground APs + N UAVs)', fontweight='bold')
ax2.set_ylabel('Sum Rate Improvement (%)', fontweight='bold')
ax2.set_title('(b) System Sum Rate Improvement Percentage', fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels([f'4AP + {cfg}' for cfg in configs])
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('result/sum_rate_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: result/sum_rate_comparison.png")
plt.close()

# ============================================================================
# Figure 3: Combined Performance Metrics
# ============================================================================
fig3 = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.25)

# Plot 3a: Absolute Min Rate by UAV Configuration
ax1 = fig3.add_subplot(gs[0, 0])
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    min_rates = [data[cfg][alg_key]['min_rate'] for cfg in configs]
    ax1.plot(range(len(configs)), min_rates, marker='o', linewidth=2.5, 
            markersize=8, label=alg_name, color=colors[alg_key])
    # Add value labels
    for j, val in enumerate(min_rates):
        ax1.text(j, val + 0.8, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# Add initial line
initial_rates = [data[cfg]['initial']['min_rate'] for cfg in configs]
ax1.plot(range(len(configs)), initial_rates, marker='s', linewidth=2, 
        markersize=7, label='Initial', color='gray', linestyle='--')

ax1.set_xlabel('Configuration', fontweight='bold')
ax1.set_ylabel('Minimum User Rate (Mbps)', fontweight='bold')
ax1.set_title('(a) Minimum Rate vs UAV Number', fontweight='bold')
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels([f'4AP+{cfg}' for cfg in configs])
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 3b: Absolute Sum Rate by UAV Configuration
ax2 = fig3.add_subplot(gs[0, 1])
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    sum_rates = [data[cfg][alg_key]['sum_rate'] for cfg in configs]
    ax2.plot(range(len(configs)), sum_rates, marker='o', linewidth=2.5,
            markersize=8, label=alg_name, color=colors[alg_key])
    # Add value labels
    for j, val in enumerate(sum_rates):
        ax2.text(j, val + 30, f'{val:.0f}', ha='center', va='bottom', fontsize=8)

# Add initial line
initial_rates = [data[cfg]['initial']['sum_rate'] for cfg in configs]
ax2.plot(range(len(configs)), initial_rates, marker='s', linewidth=2,
        markersize=7, label='Initial', color='gray', linestyle='--')

ax2.set_xlabel('Configuration', fontweight='bold')
ax2.set_ylabel('System Sum Rate (Mbps)', fontweight='bold')
ax2.set_title('(b) Sum Rate vs UAV Number', fontweight='bold')
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels([f'4AP+{cfg}' for cfg in configs])
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')

# Plot 3c: Min Rate Gain (Absolute Improvement)
ax3 = fig3.add_subplot(gs[1, 0])
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    gains = []
    for cfg in configs:
        initial = data[cfg]['initial']['min_rate']
        final = data[cfg][alg_key]['min_rate']
        gain = final - initial
        gains.append(gain)
    ax3.bar(x + i*width, gains, width, label=alg_name, color=colors[alg_key], alpha=0.8)
    # Add value labels
    for j, val in enumerate(gains):
        ax3.text(x[j] + i*width, val + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

ax3.set_xlabel('Configuration (4 Ground APs + N UAVs)', fontweight='bold')
ax3.set_ylabel('Absolute Improvement (Mbps)', fontweight='bold')
ax3.set_title('(c) Minimum Rate Absolute Gain', fontweight='bold')
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels([f'4AP + {cfg}' for cfg in configs])
ax3.legend(loc='best')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Plot 3d: Sum Rate Gain (Absolute Improvement)
ax4 = fig3.add_subplot(gs[1, 1])
for i, (alg_key, alg_name) in enumerate(algorithms.items()):
    gains = []
    for cfg in configs:
        initial = data[cfg]['initial']['sum_rate']
        final = data[cfg][alg_key]['sum_rate']
        gain = final - initial
        gains.append(gain)
    ax4.bar(x + i*width, gains, width, label=alg_name, color=colors[alg_key], alpha=0.8)
    # Add value labels
    for j, val in enumerate(gains):
        y_pos = val + 10 if val >= 0 else val - 10
        va = 'bottom' if val >= 0 else 'top'
        ax4.text(x[j] + i*width, y_pos, f'{val:.0f}', ha='center', va=va, fontsize=8)

ax4.set_xlabel('Configuration (4 Ground APs + N UAVs)', fontweight='bold')
ax4.set_ylabel('Absolute Improvement (Mbps)', fontweight='bold')
ax4.set_title('(d) Sum Rate Absolute Gain', fontweight='bold')
ax4.set_xticks(x + width * 1.5)
ax4.set_xticklabels([f'4AP + {cfg}' for cfg in configs])
ax4.legend(loc='best')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.suptitle('Comprehensive Performance Comparison Across UAV Configurations',
            fontsize=14, fontweight='bold', y=0.995)
plt.savefig('result/comprehensive_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: result/comprehensive_performance_comparison.png")
plt.close()

# ============================================================================
# Print Summary
# ============================================================================
print("\n" + "="*80)
print("All comparison plots generated successfully!")
print("="*80)
print("\nGenerated files:")
print("  1. minimum_rate_comparison.png - Min rate values and improvements")
print("  2. sum_rate_comparison.png - Sum rate values and improvements")
print("  3. comprehensive_performance_comparison.png - Combined metrics")
print("\nLocation: result/")
print("="*80)

# Print data summary
print("\n" + "="*80)
print(" Performance Summary Table ".center(80))
print("="*80)

for cfg in configs:
    print(f"\n{'='*80}")
    print(f" Configuration: 4 Ground APs + {cfg} ".center(80))
    print('='*80)
    print(f"\n{'Algorithm':<15} {'Min Rate':<12} {'Sum Rate':<12} {'Min Δ%':<12} {'Sum Δ%':<12}")
    print("-" * 80)
    
    initial_min = data[cfg]['initial']['min_rate']
    initial_sum = data[cfg]['initial']['sum_rate']
    
    print(f"{'Initial':<15} {initial_min:<12.2f} {initial_sum:<12.0f} {'-':<12} {'-':<12}")
    
    for alg_key, alg_name in algorithms.items():
        min_rate = data[cfg][alg_key]['min_rate']
        sum_rate = data[cfg][alg_key]['sum_rate']
        min_improve = ((min_rate - initial_min) / initial_min) * 100
        sum_improve = ((sum_rate - initial_sum) / initial_sum) * 100
        
        print(f"{alg_name:<15} {min_rate:<12.2f} {sum_rate:<12.0f} "
              f"{min_improve:>+10.1f}% {sum_improve:>+10.1f}%")

print("\n" + "="*80)
