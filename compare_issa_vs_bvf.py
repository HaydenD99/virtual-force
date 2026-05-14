"""
ISSA vs BVF 对比实验
基于相同的BVF信道模型进行比较
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from balanced_virtual_force_optimizer_v3 import BalancedVirtualForceOptimizerV3, create_balanced_config
from issa_optimizer_bvf_channel import ISSAOptimizerBVFChannel


def run_comparison(num_runs: int = 3, config: Dict = None):
    """
    运行ISSA和BVF的对比实验
    
    Parameters:
    -----------
    num_runs : int
        运行次数（用于统计）
    config : Dict
        配置字典
    """
    if config is None:
        config = create_balanced_config()
        # 调整ISSA参数使其与BVF的计算量相当
        config['issa_n_sparrows'] = 20
        config['issa_max_iter'] = 50
        config['max_iterations'] = 50  # BVF迭代次数
    
    print("=" * 80)
    print("ISSA vs BVF 对比实验")
    print("=" * 80)
    print(f"配置参数:")
    print(f"  - UE数量: {config['num_UE']}")
    print(f"  - UAV数量: {config['num_UAV']}")
    print(f"  - 地面AP数量: {config['num_ground_AP']}")
    print(f"  - ISSA种群大小: {config.get('issa_n_sparrows', 20)}")
    print(f"  - ISSA最大迭代: {config.get('issa_max_iter', 50)}")
    print(f"  - BVF最大迭代: {config.get('max_iterations', 50)}")
    print(f"  - 运行次数: {num_runs}")
    print("=" * 80)
    
    # 存储结果
    bvf_results = {
        'sum_rates': [],
        'min_rates': [],
        'times': [],
        'final_rates': [],
        'all_history': []
    }
    
    issa_results = {
        'sum_rates': [],
        'min_rates': [],
        'times': [],
        'final_rates': [],
        'all_history': []
    }
    
    # 使用相同的随机种子确保公平对比
    for run in range(num_runs):
        print(f"\n{'='*80}")
        print(f"运行 {run+1}/{num_runs}")
        print(f"{'='*80}")
        
        # 设置随机种子
        np.random.seed(42 + run)
        seed = 42 + run
        
        # 初始化BVF优化器
        bvf_config = config.copy()
        bvf_config['random_seed'] = seed
        bvf_optimizer = BalancedVirtualForceOptimizerV3(bvf_config)
        
        # 初始化ISSA优化器
        issa_config = config.copy()
        issa_config['random_seed'] = seed
        issa_optimizer = ISSAOptimizerBVFChannel(issa_config)
        
        # 生成相同的初始位置
        UE_pos, ground_AP_pos, UAV_pos = bvf_optimizer.initialize_positions()
        
        print(f"\n初始状态:")
        print(f"  UE数量: {len(UE_pos)}")
        print(f"  地面AP数量: {len(ground_AP_pos)}")
        print(f"  UAV数量: {len(UAV_pos)}")
        
        # 运行BVF优化
        print(f"\n{'─'*80}")
        print("运行 BVF 优化...")
        print(f"{'─'*80}")
        bvf_start = time.time()
        bvf_result = bvf_optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
        bvf_time = time.time() - bvf_start
        
        bvf_results['sum_rates'].append(bvf_result['final_sum_rate'])
        bvf_results['min_rates'].append(bvf_result['final_min_rate'])
        bvf_results['times'].append(bvf_result['optimization_time'])
        bvf_results['final_rates'].append(bvf_result['final_rates'])
        bvf_results['all_history'].append({
            'iterations': bvf_result['history']['iterations'],
            'sum_rates': bvf_result['history']['sum_rates'],
            'min_rates': bvf_result['history']['min_rates']
        })
        
        print(f"\nBVF结果:")
        print(f"  总速率: {bvf_result['final_sum_rate']:.2f} Mbps")
        print(f"  最小速率: {bvf_result['final_min_rate']:.4f} Mbps")
        print(f"  优化时间: {bvf_result['optimization_time']:.2f} 秒")
        
        # 运行ISSA优化
        print(f"\n{'─'*80}")
        print("运行 ISSA 优化...")
        print(f"{'─'*80}")
        issa_start = time.time()
        issa_result = issa_optimizer.optimize(UE_pos, ground_AP_pos, UAV_pos.copy())
        issa_time = time.time() - issa_start
        
        issa_results['sum_rates'].append(issa_result['final_sum_rate'])
        issa_results['min_rates'].append(issa_result['final_min_rate'])
        issa_results['times'].append(issa_result['optimization_time'])
        issa_results['final_rates'].append(issa_result['final_rates'])
        issa_results['all_history'].append({
            'iterations': issa_result['history']['iterations'],
            'sum_rates': issa_result['history']['sum_rates'],
            'min_rates': issa_result['history']['min_rates']
        })
        
        print(f"\nISSA结果:")
        print(f"  总速率: {issa_result['final_sum_rate']:.2f} Mbps")
        print(f"  最小速率: {issa_result['final_min_rate']:.4f} Mbps")
        print(f"  优化时间: {issa_result['optimization_time']:.2f} 秒")
        
        # 对比
        print(f"\n{'─'*80}")
        print("对比结果:")
        print(f"{'─'*80}")
        print(f"总速率: BVF={bvf_result['final_sum_rate']:.2f}, "
              f"ISSA={issa_result['final_sum_rate']:.2f}, "
              f"差异={issa_result['final_sum_rate']-bvf_result['final_sum_rate']:.2f} "
              f"({(issa_result['final_sum_rate']/bvf_result['final_sum_rate']-1)*100:.1f}%)")
        print(f"最小速率: BVF={bvf_result['final_min_rate']:.4f}, "
              f"ISSA={issa_result['final_min_rate']:.4f}, "
              f"差异={issa_result['final_min_rate']-bvf_result['final_min_rate']:.4f} "
              f"({(issa_result['final_min_rate']/bvf_result['final_min_rate']-1)*100:.1f}%)")
        print(f"优化时间: BVF={bvf_result['optimization_time']:.2f}, "
              f"ISSA={issa_result['optimization_time']:.2f}, "
              f"比率={issa_result['optimization_time']/bvf_result['optimization_time']:.2f}x")
    
    # 统计分析
    print(f"\n{'='*80}")
    print("统计分析（所有运行的平均值）")
    print(f"{'='*80}")
    
    bvf_avg_sum = np.mean(bvf_results['sum_rates'])
    bvf_std_sum = np.std(bvf_results['sum_rates'])
    bvf_avg_min = np.mean(bvf_results['min_rates'])
    bvf_std_min = np.std(bvf_results['min_rates'])
    bvf_avg_time = np.mean(bvf_results['times'])
    
    issa_avg_sum = np.mean(issa_results['sum_rates'])
    issa_std_sum = np.std(issa_results['sum_rates'])
    issa_avg_min = np.mean(issa_results['min_rates'])
    issa_std_min = np.std(issa_results['min_rates'])
    issa_avg_time = np.mean(issa_results['times'])
    
    print(f"\nBVF (平均 ± 标准差):")
    print(f"  总速率: {bvf_avg_sum:.2f} ± {bvf_std_sum:.2f} Mbps")
    print(f"  最小速率: {bvf_avg_min:.4f} ± {bvf_std_min:.4f} Mbps")
    print(f"  优化时间: {bvf_avg_time:.2f} 秒")
    
    print(f"\nISSA (平均 ± 标准差):")
    print(f"  总速率: {issa_avg_sum:.2f} ± {issa_std_sum:.2f} Mbps")
    print(f"  最小速率: {issa_avg_min:.4f} ± {issa_std_min:.4f} Mbps")
    print(f"  优化时间: {issa_avg_time:.2f} 秒")
    
    print(f"\n对比 (ISSA vs BVF):")
    print(f"  总速率提升: {(issa_avg_sum/bvf_avg_sum - 1)*100:.2f}%")
    print(f"  最小速率提升: {(issa_avg_min/bvf_avg_min - 1)*100:.2f}%")
    print(f"  时间比率: {issa_avg_time/bvf_avg_time:.2f}x")
    
    # 绘制对比图
    plot_comparison(bvf_results, issa_results)
    
    return {
        'bvf_results': bvf_results,
        'issa_results': issa_results,
        'statistics': {
            'bvf': {
                'avg_sum_rate': bvf_avg_sum,
                'std_sum_rate': bvf_std_sum,
                'avg_min_rate': bvf_avg_min,
                'std_min_rate': bvf_std_min,
                'avg_time': bvf_avg_time
            },
            'issa': {
                'avg_sum_rate': issa_avg_sum,
                'std_sum_rate': issa_std_sum,
                'avg_min_rate': issa_avg_min,
                'std_min_rate': issa_std_min,
                'avg_time': issa_avg_time
            }
        }
    }


def plot_comparison(bvf_results: Dict, issa_results: Dict):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 收敛曲线 - 总速率
    ax1 = axes[0, 0]
    for i, history in enumerate(bvf_results['all_history']):
        iterations = history['iterations']
        sum_rates = history['sum_rates']
        if i == 0:
            ax1.plot(iterations, sum_rates, 'b-', alpha=0.6, linewidth=1.5, label='BVF')
        else:
            ax1.plot(iterations, sum_rates, 'b-', alpha=0.6, linewidth=1.5)
    
    for i, history in enumerate(issa_results['all_history']):
        iterations = history['iterations']
        sum_rates = history['sum_rates']
        if i == 0:
            ax1.plot(iterations, sum_rates, 'r--', alpha=0.6, linewidth=1.5, label='ISSA')
        else:
            ax1.plot(iterations, sum_rates, 'r--', alpha=0.6, linewidth=1.5)
    
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('总速率 (Mbps)')
    ax1.set_title('收敛曲线 - 总速率')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 收敛曲线 - 最小速率
    ax2 = axes[0, 1]
    for i, history in enumerate(bvf_results['all_history']):
        iterations = history['iterations']
        min_rates = history['min_rates']
        if i == 0:
            ax2.plot(iterations, min_rates, 'b-', alpha=0.6, linewidth=1.5, label='BVF')
        else:
            ax2.plot(iterations, min_rates, 'b-', alpha=0.6, linewidth=1.5)
    
    for i, history in enumerate(issa_results['all_history']):
        iterations = history['iterations']
        min_rates = history['min_rates']
        if i == 0:
            ax2.plot(iterations, min_rates, 'r--', alpha=0.6, linewidth=1.5, label='ISSA')
        else:
            ax2.plot(iterations, min_rates, 'r--', alpha=0.6, linewidth=1.5)
    
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('最小速率 (Mbps)')
    ax2.set_title('收敛曲线 - 最小速率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 总速率对比（箱线图）
    ax3 = axes[1, 0]
    data = [bvf_results['sum_rates'], issa_results['sum_rates']]
    bp = ax3.boxplot(data, labels=['BVF', 'ISSA'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('总速率 (Mbps)')
    ax3.set_title('总速率分布对比')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 最小速率对比（箱线图）
    ax4 = axes[1, 1]
    data = [bvf_results['min_rates'], issa_results['min_rates']]
    bp = ax4.boxplot(data, labels=['BVF', 'ISSA'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('最小速率 (Mbps)')
    ax4.set_title('最小速率分布对比')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('issa_vs_bvf_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: issa_vs_bvf_comparison.png")
    plt.close()


if __name__ == "__main__":
    # 运行对比实验
    results = run_comparison(num_runs=3)
    
    print(f"\n{'='*80}")
    print("对比实验完成!")
    print(f"{'='*80}")

