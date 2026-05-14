"""
四种UAV位置优化算法的快速对比
对比算法：Simplified VF, Balanced VF, PSO, GWO
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入所有优化算法
from balanced_virtual_force_optimizer import BalancedVirtualForceOptimizer, create_balanced_config
from genetic_algorithm_optimizer import GeneticAlgorithmOptimizer, create_ga_config
from pso_optimizer import PSOOptimizer, create_pso_config
from gwo_optimizer import GWOOptimizer, create_gwo_config

class FourAlgorithmsComparison:
    """四种算法快速对比"""
    
    def __init__(self):
        self.algorithms = {
            'Balanced VF': {
                'class': BalancedVirtualForceOptimizer,
                'config': create_balanced_config(),
                'color': '#4ECDC4',
                'marker': 's',
                'name_short': 'BVF'
            },
            'GA': {
                'class': GeneticAlgorithmOptimizer,
                'config': create_ga_config(),
                'color': '#FF6B6B',
                'marker': 'o',
                'name_short': 'GA'
            },
            'PSO': {
                'class': PSOOptimizer,
                'config': create_pso_config(),
                'color': '#45B7D1',
                'marker': '^',
                'name_short': 'PSO'
            },
            'GWO': {
                'class': GWOOptimizer,
                'config': create_gwo_config(),
                'color': '#96CEB4',
                'marker': 'D',
                'name_short': 'GWO'
            }
        }
        
        # 统一配置参数（快速对比）
        base_config = {
            'square_length': 1000,
            'num_UE': 60,
            'num_UAV': 9,
            'num_ground_AP': 4,
            'M': 4,
            'max_iterations': 50,  # 减少迭代次数
            'nbrOfRealizations': 50,
        }
        
        for alg_name in self.algorithms:
            config_update = base_config.copy()
            # GA算法使用max_generations而不是max_iterations
            if alg_name == 'GA':
                config_update['max_generations'] = config_update.pop('max_iterations')
            self.algorithms[alg_name]['config'].update(config_update)

    def run_comparison(self, seed: int = 42) -> Dict:
        """运行单次对比"""
        np.random.seed(seed)
        
        print(f"\n🚀 Four Algorithms Comparison")
        print(f"="*50)
        
        # 使用相同的初始化位置
        base_optimizer = BalancedVirtualForceOptimizer(self.algorithms['Balanced VF']['config'])
        UE_pos, ground_AP_pos, initial_UAV_pos = base_optimizer.initialize_positions()
        
        print(f"Network Setup:")
        print(f"  Users: {len(UE_pos)}, Ground APs: {len(ground_AP_pos)}, UAVs: {len(initial_UAV_pos)}")
        print(f"  Area: {base_optimizer.square_length}×{base_optimizer.square_length} m²")
        
        results = {}
        
        for alg_name, alg_info in self.algorithms.items():
            print(f"\n{'='*20} {alg_name} {'='*20}")
            
            start_time = time.time()
            
            # 创建优化器并运行
            optimizer = alg_info['class'](alg_info['config'])
            
            # 计算初始性能
            initial_performance = self._evaluate_performance(UE_pos, ground_AP_pos, initial_UAV_pos, optimizer)
            
            # 执行优化（GA算法接口不同）
            if alg_name == 'GA':
                opt_result = optimizer.optimize(UE_pos, ground_AP_pos)
            else:
                opt_result = optimizer.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
            
            total_time = time.time() - start_time
            
            # 存储结果
            results[alg_name] = {
                'initial': initial_performance,
                'final': {
                    'sum_rate': opt_result['final_sum_rate'],
                    'min_rate': opt_result['final_min_rate'],
                    'rates': opt_result['final_rates'],
                    'fairness': self._calculate_fairness(opt_result['final_rates'])
                },
                'time': opt_result['optimization_time'],
                'total_time': total_time,
                'iterations': opt_result.get('total_iterations', opt_result.get('total_generations', 50)),
                'history': opt_result['history'],
                'UAV_pos': opt_result['optimized_UAV_pos'],
                'improvement': {
                    'sum_rate': ((opt_result['final_sum_rate'] - initial_performance['sum_rate']) / initial_performance['sum_rate']) * 100,
                    'min_rate': ((opt_result['final_min_rate'] - initial_performance['min_rate']) / initial_performance['min_rate']) * 100
                }
            }
            
            # 输出结果
            print(f"Results:")
            print(f"  Sum Rate: {initial_performance['sum_rate']:.2f} → {opt_result['final_sum_rate']:.2f} Mbps "
                  f"({results[alg_name]['improvement']['sum_rate']:+.1f}%)")
            print(f"  Min Rate: {initial_performance['min_rate']:.4f} → {opt_result['final_min_rate']:.4f} Mbps "
                  f"({results[alg_name]['improvement']['min_rate']:+.1f}%)")
            print(f"  Fairness: {initial_performance['fairness']:.4f} → {results[alg_name]['final']['fairness']:.4f}")
            print(f"  Time: {opt_result['optimization_time']:.2f}s")
        
        return results

    def _evaluate_performance(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray, 
                            UAV_pos: np.ndarray, optimizer) -> Dict:
        """评估性能"""
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
        _, _, betas = optimizer.compute_channel_model(UE_pos, all_AP_pos)
        mask = optimizer.compute_AP_selection_mask(betas)
        rates, sum_rate = optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
        
        return {
            'sum_rate': sum_rate,
            'min_rate': rates.min(),
            'fairness': self._calculate_fairness(rates)
        }

    def _calculate_fairness(self, rates: np.ndarray) -> float:
        """计算Jain公平性指数"""
        if len(rates) == 0 or np.sum(rates) == 0:
            return 0.0
        return (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2))

    def create_comparison_visualization(self, results: Dict, save_path: str = None):
        """创建对比可视化"""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Four UAV Positioning Algorithms Comparison\n(Balanced VF, GA, PSO, GWO)', fontsize=18, fontweight='bold')
        
        alg_names = list(results.keys())
        colors = [self.algorithms[name]['color'] for name in alg_names]
        short_names = [self.algorithms[name]['name_short'] for name in alg_names]
        
        # 1. 最小速率对比
        ax1 = axes[0, 0]
        final_min_rates = [results[name]['final']['min_rate'] for name in alg_names]
        bars1 = ax1.bar(short_names, final_min_rates, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Minimum Rate (Mbps)')
        ax1.set_title('Minimum Rate Performance')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars1, final_min_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_min_rates)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 总速率对比
        ax2 = axes[0, 1]
        final_sum_rates = [results[name]['final']['sum_rate'] for name in alg_names]
        bars2 = ax2.bar(short_names, final_sum_rates, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Sum Rate (Mbps)')
        ax2.set_title('Sum Rate Performance')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars2, final_sum_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_sum_rates)*0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 公平性对比
        ax3 = axes[0, 2]
        fairness_values = [results[name]['final']['fairness'] for name in alg_names]
        bars3 = ax3.bar(short_names, fairness_values, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Jain Fairness Index')
        ax3.set_title('Fairness Performance')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars3, fairness_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 优化时间对比
        ax4 = axes[1, 0]
        opt_times = [results[name]['time'] for name in alg_names]
        bars4 = ax4.bar(short_names, opt_times, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Optimization Time (s)')
        ax4.set_title('Computational Efficiency')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars4, opt_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(opt_times)*0.02,
                    f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 5. 性能改善对比（最小速率）
        ax5 = axes[1, 1]
        min_rate_improvements = [results[name]['improvement']['min_rate'] for name in alg_names]
        bars5 = ax5.bar(short_names, min_rate_improvements, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Min Rate Improvement (%)')
        ax5.set_title('Min Rate Improvement')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 添加数值标签
        for bar, val in zip(bars5, min_rate_improvements):
            height = bar.get_height()
            label_y = height + max(min_rate_improvements)*0.02 if height >= 0 else height - max(min_rate_improvements)*0.05
            ax5.text(bar.get_x() + bar.get_width()/2, label_y,
                    f'{val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # 6. 收敛曲线对比（最小速率）
        ax6 = axes[1, 2]
        for alg_name in alg_names:
            history = results[alg_name]['history']
            # GA算法使用不同的字段名
            min_rates_key = 'best_min_rates' if alg_name == 'GA' else 'min_rates'
            
            if min_rates_key in history and len(history[min_rates_key]) > 0:
                color = self.algorithms[alg_name]['color']
                marker = self.algorithms[alg_name]['marker']
                short_name = self.algorithms[alg_name]['name_short']
                
                iterations = np.arange(len(history[min_rates_key]))
                ax6.plot(iterations, history[min_rates_key], 
                        color=color, linewidth=2.5, marker=marker, 
                        markersize=4, markevery=max(1, len(iterations)//8),
                        label=short_name, alpha=0.9)
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Minimum Rate (Mbps)')
        ax6.set_title('Convergence Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n📊 Comparison plot saved: {save_path}")
        
        plt.show()

    def generate_comparison_table(self, results: Dict) -> pd.DataFrame:
        """生成对比表格"""
        data = []
        
        for alg_name, result in results.items():
            row = {
                'Algorithm': self.algorithms[alg_name]['name_short'],
                'Min Rate (Mbps)': f"{result['final']['min_rate']:.4f}",
                'Sum Rate (Mbps)': f"{result['final']['sum_rate']:.2f}",
                'Fairness Index': f"{result['final']['fairness']:.4f}",
                'Opt Time (s)': f"{result['time']:.2f}",
                'Min Rate Improve (%)': f"{result['improvement']['min_rate']:+.1f}",
                'Sum Rate Improve (%)': f"{result['improvement']['sum_rate']:+.1f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df


def main():
    """主函数"""
    print("🚀 Four Algorithms Comparison Starting...")
    
    # 创建对比器
    comparator = FourAlgorithmsComparison()
    
    # 运行对比
    results = comparator.run_comparison(seed=42)
    
    # 生成可视化
    print(f"\n🎨 Creating visualization...")
    comparator.create_comparison_visualization(
        results, 
        save_path='/home/hzl/hyd/virtualForce/four_algorithms_comparison.png'
    )
    
    # 生成对比表格
    print(f"\n📋 Comparison Results Table:")
    df = comparator.generate_comparison_table(results)
    print(df.to_string(index=False))
    
    # 保存表格
    df.to_csv('/home/hzl/hyd/virtualForce/four_algorithms_results.csv', index=False)
    
    # 性能排名
    print(f"\n🏆 Performance Ranking:")
    
    # 按最小速率排名
    min_rate_ranking = sorted(results.items(), key=lambda x: x[1]['final']['min_rate'], reverse=True)
    print(f"\nMinimum Rate Ranking:")
    for i, (alg_name, result) in enumerate(min_rate_ranking):
        short_name = comparator.algorithms[alg_name]['name_short']
        min_rate = result['final']['min_rate']
        print(f"  #{i+1}. {short_name:4s}: {min_rate:.4f} Mbps")
    
    # 按总速率排名
    sum_rate_ranking = sorted(results.items(), key=lambda x: x[1]['final']['sum_rate'], reverse=True)
    print(f"\nSum Rate Ranking:")
    for i, (alg_name, result) in enumerate(sum_rate_ranking):
        short_name = comparator.algorithms[alg_name]['name_short']
        sum_rate = result['final']['sum_rate']
        print(f"  #{i+1}. {short_name:4s}: {sum_rate:.2f} Mbps")
    
    # 按公平性排名
    fairness_ranking = sorted(results.items(), key=lambda x: x[1]['final']['fairness'], reverse=True)
    print(f"\nFairness Ranking:")
    for i, (alg_name, result) in enumerate(fairness_ranking):
        short_name = comparator.algorithms[alg_name]['name_short']
        fairness = result['final']['fairness']
        print(f"  #{i+1}. {short_name:4s}: {fairness:.4f}")
    
    # 按效率排名（时间）
    time_ranking = sorted(results.items(), key=lambda x: x[1]['time'])
    print(f"\nEfficiency Ranking (by time):")
    for i, (alg_name, result) in enumerate(time_ranking):
        short_name = comparator.algorithms[alg_name]['name_short']
        time_val = result['time']
        print(f"  #{i+1}. {short_name:4s}: {time_val:.2f} seconds")
    
    print(f"\n✅ Four algorithms comparison completed!")
    print(f"🔬 Algorithms: Balanced VF, GA, PSO, GWO")
    print(f"📊 Results saved to: /home/hzl/hyd/virtualForce/four_algorithms_comparison.png")
    print(f"📋 Table saved to: /home/hzl/hyd/virtualForce/four_algorithms_results.csv")


if __name__ == "__main__":
    main()
