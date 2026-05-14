"""
Comprehensive Algorithm Comparison
四种UAV位置优化算法的全面对比：简化虚拟力、平衡虚拟力、PSO、GWO
包含性能指标对比和计算复杂度分析
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入所有优化算法
from simplified_virtual_force_optimizer import SimplifiedVirtualForceOptimizer, create_simplified_config
from balanced_virtual_force_optimizer import BalancedVirtualForceOptimizer, create_balanced_config
from pso_optimizer import PSOOptimizer, create_pso_config
from gwo_optimizer import GWOOptimizer, create_gwo_config

class AlgorithmComparison:
    """算法综合对比分析器"""
    
    def __init__(self, num_runs: int = 5):
        self.num_runs = num_runs
        self.algorithms = {
            'Simplified VF': {
                'class': SimplifiedVirtualForceOptimizer,
                'config': create_simplified_config(),
                'color': '#FF6B6B',
                'marker': 'o'
            },
            'Balanced VF': {
                'class': BalancedVirtualForceOptimizer,
                'config': create_balanced_config(),
                'color': '#4ECDC4',
                'marker': 's'
            },
            'PSO': {
                'class': PSOOptimizer,
                'config': create_pso_config(),
                'color': '#45B7D1',
                'marker': '^'
            },
            'GWO': {
                'class': GWOOptimizer,
                'config': create_gwo_config(),
                'color': '#96CEB4',
                'marker': 'D'
            }
        }
        
        # 确保所有算法使用相同的基本参数
        base_config = {
            'square_length': 1000,
            'num_UE': 60,
            'num_UAV': 9,
            'num_ground_AP': 4,
            'M': 4,
            'max_iterations': 80,  # 减少迭代次数以便快速对比
            'nbrOfRealizations': 50,
        }
        
        for alg_name in self.algorithms:
            self.algorithms[alg_name]['config'].update(base_config)

    def run_single_comparison(self, seed: int = None) -> Dict:
        """运行单次对比实验"""
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n{'='*50}")
        print(f"Running Comparison (Seed: {seed})")
        print(f"{'='*50}")
        
        results = {}
        
        # 为所有算法使用相同的初始位置
        base_optimizer = SimplifiedVirtualForceOptimizer(self.algorithms['Simplified Virtual Force']['config'])
        UE_pos, ground_AP_pos, initial_UAV_pos = base_optimizer.initialize_positions()
        
        print(f"Network Configuration:")
        print(f"  - Users (UE): {len(UE_pos)}")
        print(f"  - Ground APs: {len(ground_AP_pos)}")
        print(f"  - UAVs: {len(initial_UAV_pos)}")
        print(f"  - Coverage Area: {base_optimizer.square_length}×{base_optimizer.square_length} m²")
        
        for alg_name, alg_info in self.algorithms.items():
            print(f"\n{'─'*30}")
            print(f"Running {alg_name}...")
            print(f"{'─'*30}")
            
            start_time = time.time()
            
            # 创建优化器
            optimizer = alg_info['class'](alg_info['config'])
            
            # 记录初始性能
            initial_result = self._evaluate_performance(UE_pos, ground_AP_pos, initial_UAV_pos, optimizer)
            
            # 执行优化
            optimization_result = optimizer.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
            
            total_time = time.time() - start_time
            
            # 记录结果
            results[alg_name] = {
                'initial_performance': initial_result,
                'final_performance': {
                    'sum_rate': optimization_result['final_sum_rate'],
                    'min_rate': optimization_result['final_min_rate'],
                    'rate_std': optimization_result['final_rates'].std(),
                    'fairness_index': self._calculate_fairness_index(optimization_result['final_rates']),
                    'coverage_efficiency': self._calculate_coverage_efficiency(optimization_result['final_rates'])
                },
                'optimization_time': optimization_result['optimization_time'],
                'total_time': total_time,
                'convergence_info': {
                    'iterations': optimization_result['total_iterations'],
                    'improvement_achieved': optimization_result['improvement_achieved']
                },
                'history': optimization_result['history'],
                'optimized_UAV_pos': optimization_result['optimized_UAV_pos'],
                'complexity_metrics': self._calculate_complexity_metrics(alg_name, alg_info['config'])
            }
            
            # 输出简要结果
            print(f"Results Summary:")
            print(f"  Sum Rate: {results[alg_name]['initial_performance']['sum_rate']:.2f} → "
                  f"{results[alg_name]['final_performance']['sum_rate']:.2f} Mbps "
                  f"({((results[alg_name]['final_performance']['sum_rate'] - results[alg_name]['initial_performance']['sum_rate'])/results[alg_name]['initial_performance']['sum_rate']*100):+.1f}%)")
            print(f"  Min Rate: {results[alg_name]['initial_performance']['min_rate']:.4f} → "
                  f"{results[alg_name]['final_performance']['min_rate']:.4f} Mbps "
                  f"({((results[alg_name]['final_performance']['min_rate'] - results[alg_name]['initial_performance']['min_rate'])/results[alg_name]['initial_performance']['min_rate']*100):+.1f}%)")
            print(f"  Fairness Index: {results[alg_name]['initial_performance']['fairness_index']:.4f} → "
                  f"{results[alg_name]['final_performance']['fairness_index']:.4f}")
            print(f"  Optimization Time: {results[alg_name]['optimization_time']:.2f}s")
            print(f"  Iterations: {results[alg_name]['convergence_info']['iterations']}")
        
        return results

    def _evaluate_performance(self, UE_pos: np.ndarray, ground_AP_pos: np.ndarray, 
                            UAV_pos: np.ndarray, optimizer) -> Dict:
        """评估给定位置的性能"""
        all_AP_pos = np.vstack([ground_AP_pos, UAV_pos])
        _, _, betas = optimizer.compute_channel_model(UE_pos, all_AP_pos)
        mask = optimizer.compute_AP_selection_mask(betas)
        rates, sum_rate = optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
        
        return {
            'sum_rate': sum_rate,
            'min_rate': rates.min(),
            'rate_std': rates.std(),
            'fairness_index': self._calculate_fairness_index(rates),
            'coverage_efficiency': self._calculate_coverage_efficiency(rates)
        }

    def _calculate_fairness_index(self, rates: np.ndarray) -> float:
        """计算Jain公平性指数"""
        if len(rates) == 0 or np.sum(rates) == 0:
            return 0.0
        return (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2))

    def _calculate_coverage_efficiency(self, rates: np.ndarray) -> float:
        """计算覆盖效率（满足最小速率要求的用户比例）"""
        min_required_rate = 0.1  # Mbps
        return np.sum(rates >= min_required_rate) / len(rates)

    def _calculate_complexity_metrics(self, alg_name: str, config: Dict) -> Dict:
        """计算算法复杂度指标"""
        K = config['num_UE']
        L = config['num_UAV']
        G = config['num_ground_AP']
        M = config['M']
        max_iter = config['max_iterations']
        
        if 'Virtual Force' in alg_name:
            # 虚拟力算法复杂度
            # 主要计算：信道计算 + 力计算 + 位置更新
            channel_complexity = K * (L + G) * M**2  # 信道模型计算
            force_complexity = L * K  # 虚拟力计算
            per_iteration = channel_complexity + force_complexity
            total_complexity = per_iteration * max_iter
            
        elif alg_name == 'PSO':
            # PSO复杂度
            swarm_size = config['swarm_size']
            per_iteration = swarm_size * K * (L + G) * M**2  # 每个粒子都要评估
            total_complexity = per_iteration * max_iter
            
        elif alg_name == 'GWO':
            # GWO复杂度
            pack_size = config['pack_size']
            per_iteration = pack_size * K * (L + G) * M**2  # 每只狼都要评估
            total_complexity = per_iteration * max_iter
        
        return {
            'per_iteration_complexity': per_iteration,
            'total_complexity': total_complexity,
            'complexity_order': f"O({max_iter} × {per_iteration:.0e})",
            'memory_complexity': K * (L + G) * M**2  # 主要是信道矩阵存储
        }

    def run_comprehensive_comparison(self) -> Dict:
        """运行全面对比实验（多次运行取平均）"""
        print(f"\n🚀 Starting Comprehensive Algorithm Comparison")
        print(f"🔢 Number of runs: {self.num_runs}")
        print(f"🏗️ Algorithms: {list(self.algorithms.keys())}")
        
        all_results = []
        
        for run in range(self.num_runs):
            print(f"\n📊 Run {run + 1}/{self.num_runs}")
            single_result = self.run_single_comparison(seed=42 + run)
            all_results.append(single_result)
        
        # 聚合结果
        aggregated_results = self._aggregate_results(all_results)
        
        return aggregated_results

    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """聚合多次运行的结果"""
        aggregated = {}
        
        for alg_name in self.algorithms.keys():
            # 收集所有运行的指标
            sum_rates = [r[alg_name]['final_performance']['sum_rate'] for r in all_results]
            min_rates = [r[alg_name]['final_performance']['min_rate'] for r in all_results]
            fairness_indices = [r[alg_name]['final_performance']['fairness_index'] for r in all_results]
            optimization_times = [r[alg_name]['optimization_time'] for r in all_results]
            iterations = [r[alg_name]['convergence_info']['iterations'] for r in all_results]
            
            # 计算统计量
            aggregated[alg_name] = {
                'performance': {
                    'sum_rate': {
                        'mean': np.mean(sum_rates),
                        'std': np.std(sum_rates),
                        'min': np.min(sum_rates),
                        'max': np.max(sum_rates)
                    },
                    'min_rate': {
                        'mean': np.mean(min_rates),
                        'std': np.std(min_rates),
                        'min': np.min(min_rates),
                        'max': np.max(min_rates)
                    },
                    'fairness_index': {
                        'mean': np.mean(fairness_indices),
                        'std': np.std(fairness_indices)
                    }
                },
                'efficiency': {
                    'optimization_time': {
                        'mean': np.mean(optimization_times),
                        'std': np.std(optimization_times)
                    },
                    'iterations': {
                        'mean': np.mean(iterations),
                        'std': np.std(iterations)
                    }
                },
                'complexity_metrics': all_results[0][alg_name]['complexity_metrics'],
                'best_run': all_results[np.argmax(min_rates)][alg_name],  # 最佳最小速率的运行
                'color': self.algorithms[alg_name]['color'],
                'marker': self.algorithms[alg_name]['marker']
            }
        
        return aggregated

    def create_comprehensive_visualization(self, results: Dict, save_path: str = None):
        """创建综合可视化图表"""
        # 设置matplotlib参数
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11
        })
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 性能对比 - 最小速率 vs 总速率
        ax1 = plt.subplot(3, 4, 1)
        for alg_name, alg_results in results.items():
            perf = alg_results['performance']
            plt.scatter(perf['sum_rate']['mean'], perf['min_rate']['mean'],
                       s=150, c=alg_results['color'], marker=alg_results['marker'],
                       label=alg_name, alpha=0.7, edgecolors='black', linewidth=1)
            
            # 添加误差棒
            plt.errorbar(perf['sum_rate']['mean'], perf['min_rate']['mean'],
                        xerr=perf['sum_rate']['std'], yerr=perf['min_rate']['std'],
                        color=alg_results['color'], alpha=0.5, capsize=3)
        
        plt.xlabel('Sum Rate (Mbps)')
        plt.ylabel('Minimum Rate (Mbps)')
        plt.title('Performance Trade-off: Sum Rate vs Min Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 最小速率对比（柱状图）
        ax2 = plt.subplot(3, 4, 2)
        alg_names = list(results.keys())
        min_rates_mean = [results[name]['performance']['min_rate']['mean'] for name in alg_names]
        min_rates_std = [results[name]['performance']['min_rate']['std'] for name in alg_names]
        colors = [results[name]['color'] for name in alg_names]
        
        bars = plt.bar(range(len(alg_names)), min_rates_mean, yerr=min_rates_std,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('Minimum Rate (Mbps)')
        plt.title('Minimum Rate Comparison')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, val, std) in enumerate(zip(bars, min_rates_mean, min_rates_std)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. 总速率对比（柱状图）
        ax3 = plt.subplot(3, 4, 3)
        sum_rates_mean = [results[name]['performance']['sum_rate']['mean'] for name in alg_names]
        sum_rates_std = [results[name]['performance']['sum_rate']['std'] for name in alg_names]
        
        bars = plt.bar(range(len(alg_names)), sum_rates_mean, yerr=sum_rates_std,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('Sum Rate (Mbps)')
        plt.title('Sum Rate Comparison')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, val, std) in enumerate(zip(bars, sum_rates_mean, sum_rates_std)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 公平性对比
        ax4 = plt.subplot(3, 4, 4)
        fairness_mean = [results[name]['performance']['fairness_index']['mean'] for name in alg_names]
        fairness_std = [results[name]['performance']['fairness_index']['std'] for name in alg_names]
        
        bars = plt.bar(range(len(alg_names)), fairness_mean, yerr=fairness_std,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('Jain Fairness Index')
        plt.title('Fairness Comparison')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1)
        
        # 5. 优化时间对比
        ax5 = plt.subplot(3, 4, 5)
        opt_times_mean = [results[name]['efficiency']['optimization_time']['mean'] for name in alg_names]
        opt_times_std = [results[name]['efficiency']['optimization_time']['std'] for name in alg_names]
        
        bars = plt.bar(range(len(alg_names)), opt_times_mean, yerr=opt_times_std,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('Optimization Time (s)')
        plt.title('Computational Efficiency')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 6. 收敛性对比
        ax6 = plt.subplot(3, 4, 6)
        iterations_mean = [results[name]['efficiency']['iterations']['mean'] for name in alg_names]
        iterations_std = [results[name]['efficiency']['iterations']['std'] for name in alg_names]
        
        bars = plt.bar(range(len(alg_names)), iterations_mean, yerr=iterations_std,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('Iterations to Convergence')
        plt.title('Convergence Speed')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 7. 计算复杂度对比
        ax7 = plt.subplot(3, 4, 7)
        complexities = [results[name]['complexity_metrics']['total_complexity'] for name in alg_names]
        
        bars = plt.bar(range(len(alg_names)), np.log10(complexities), 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('log₁₀(Total Complexity)')
        plt.title('Computational Complexity')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 8. 收敛曲线对比（最小速率）
        ax8 = plt.subplot(3, 4, 8)
        for alg_name, alg_results in results.items():
            history = alg_results['best_run']['history']
            if 'min_rates' in history and len(history['min_rates']) > 0:
                iterations = np.arange(len(history['min_rates']))
                plt.plot(iterations, history['min_rates'], 
                        color=alg_results['color'], linewidth=2,
                        marker=alg_results['marker'], markersize=4, markevery=max(1, len(iterations)//10),
                        label=alg_name)
        
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Rate (Mbps)')
        plt.title('Convergence: Minimum Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. 收敛曲线对比（总速率）
        ax9 = plt.subplot(3, 4, 9)
        for alg_name, alg_results in results.items():
            history = alg_results['best_run']['history']
            if 'sum_rates' in history and len(history['sum_rates']) > 0:
                iterations = np.arange(len(history['sum_rates']))
                plt.plot(iterations, history['sum_rates'],
                        color=alg_results['color'], linewidth=2,
                        marker=alg_results['marker'], markersize=4, markevery=max(1, len(iterations)//10),
                        label=alg_name)
        
        plt.xlabel('Iteration')
        plt.ylabel('Sum Rate (Mbps)')
        plt.title('Convergence: Sum Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 10. 雷达图 - 综合性能对比
        ax10 = plt.subplot(3, 4, 10, projection='polar')
        categories = ['Min Rate\n(normalized)', 'Sum Rate\n(normalized)', 
                     'Fairness', 'Speed\n(1/time)', 'Efficiency\n(1/complexity)']
        
        # 归一化指标
        max_min_rate = max([results[name]['performance']['min_rate']['mean'] for name in alg_names])
        max_sum_rate = max([results[name]['performance']['sum_rate']['mean'] for name in alg_names])
        max_time = max([results[name]['efficiency']['optimization_time']['mean'] for name in alg_names])
        max_complexity = max([results[name]['complexity_metrics']['total_complexity'] for name in alg_names])
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for alg_name, alg_results in results.items():
            values = [
                alg_results['performance']['min_rate']['mean'] / max_min_rate,
                alg_results['performance']['sum_rate']['mean'] / max_sum_rate,
                alg_results['performance']['fairness_index']['mean'],
                1 - (alg_results['efficiency']['optimization_time']['mean'] / max_time),
                1 - (alg_results['complexity_metrics']['total_complexity'] / max_complexity)
            ]
            values += values[:1]  # 闭合图形
            
            plt.plot(angles, values, color=alg_results['color'], linewidth=2, 
                    marker=alg_results['marker'], label=alg_name)
            plt.fill(angles, values, color=alg_results['color'], alpha=0.1)
        
        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 1)
        plt.title('Overall Performance Radar Chart')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 11. UAV位置可视化（最佳运行）
        ax11 = plt.subplot(3, 4, 11)
        # 选择平衡虚拟力算法的最佳运行进行可视化
        best_alg = 'Balanced VF'
        if best_alg in results:
            best_run = results[best_alg]['best_run']
            
            # 模拟UE和地面AP位置（用于可视化）
            np.random.seed(42)
            K, G, square_length = 60, 4, 1000
            UE_pos = np.random.uniform(50, square_length-50, (K, 2))
            
            # 地面AP位置
            grid_size = 2
            spacing = square_length / (grid_size + 1)
            ground_AP_pos = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = (i + 1) * spacing
                    y = (j + 1) * spacing
                    ground_AP_pos.append([x, y])
            ground_AP_pos = np.array(ground_AP_pos)
            
            UAV_pos = best_run['optimized_UAV_pos']
            
            # 绘制网络拓扑
            plt.scatter(UE_pos[:, 0], UE_pos[:, 1], c='lightblue', s=20, alpha=0.6, label='Users')
            plt.scatter(ground_AP_pos[:, 0], ground_AP_pos[:, 1], c='red', s=100, marker='s', 
                       label='Ground APs', edgecolors='black')
            plt.scatter(UAV_pos[:, 0], UAV_pos[:, 1], c='orange', s=150, marker='^',
                       label='UAVs (Optimized)', edgecolors='black', linewidth=2)
            
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Optimized Network Topology\n(Balanced VF Algorithm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 12. 性能改善对比
        ax12 = plt.subplot(3, 4, 12)
        improvements = []
        for alg_name in alg_names:
            best_run = results[alg_name]['best_run']
            initial = best_run['initial_performance']['min_rate']
            final = best_run['final_performance']['min_rate']
            improvement = ((final - initial) / initial) * 100 if initial > 0 else 0
            improvements.append(improvement)
        
        bars = plt.bar(range(len(alg_names)), improvements, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1)
        plt.xlabel('Algorithm')
        plt.ylabel('Min Rate Improvement (%)')
        plt.title('Performance Improvement')
        plt.xticks(range(len(alg_names)), [name.replace(' ', '\n') for name in alg_names])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(improvements)*0.01,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n📊 Comprehensive visualization saved to: {save_path}")
        
        plt.show()

    def generate_comparison_report(self, results: Dict, save_path: str = None):
        """生成对比分析报告"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE ALGORITHM COMPARISON REPORT")
        report.append("="*80)
        report.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of Evaluation Runs: {self.num_runs}")
        report.append("")
        
        # 性能排名
        report.append("PERFORMANCE RANKING")
        report.append("-" * 40)
        
        # 按最小速率排名
        min_rate_ranking = sorted(results.items(), 
                                 key=lambda x: x[1]['performance']['min_rate']['mean'], 
                                 reverse=True)
        
        report.append("1. Minimum Rate Performance:")
        for i, (alg_name, alg_results) in enumerate(min_rate_ranking):
            perf = alg_results['performance']['min_rate']
            report.append(f"   #{i+1:2d}. {alg_name:25s}: {perf['mean']:.4f} ± {perf['std']:.4f} Mbps")
        
        # 按总速率排名
        sum_rate_ranking = sorted(results.items(), 
                                 key=lambda x: x[1]['performance']['sum_rate']['mean'], 
                                 reverse=True)
        
        report.append("\n2. Sum Rate Performance:")
        for i, (alg_name, alg_results) in enumerate(sum_rate_ranking):
            perf = alg_results['performance']['sum_rate']
            report.append(f"   #{i+1:2d}. {alg_name:25s}: {perf['mean']:.2f} ± {perf['std']:.2f} Mbps")
        
        # 按公平性排名
        fairness_ranking = sorted(results.items(), 
                                 key=lambda x: x[1]['performance']['fairness_index']['mean'], 
                                 reverse=True)
        
        report.append("\n3. Fairness Performance:")
        for i, (alg_name, alg_results) in enumerate(fairness_ranking):
            fairness = alg_results['performance']['fairness_index']
            report.append(f"   #{i+1:2d}. {alg_name:25s}: {fairness['mean']:.4f} ± {fairness['std']:.4f}")
        
        # 效率分析
        report.append("\nCOMPUTATIONAL EFFICIENCY ANALYSIS")
        report.append("-" * 40)
        
        time_ranking = sorted(results.items(), 
                             key=lambda x: x[1]['efficiency']['optimization_time']['mean'])
        
        report.append("Optimization Time Ranking (faster is better):")
        for i, (alg_name, alg_results) in enumerate(time_ranking):
            time_info = alg_results['efficiency']['optimization_time']
            report.append(f"   #{i+1:2d}. {alg_name:25s}: {time_info['mean']:.2f} ± {time_info['std']:.2f} seconds")
        
        # 复杂度分析
        report.append("\nCOMPLEXITY ANALYSIS")
        report.append("-" * 40)
        
        for alg_name, alg_results in results.items():
            complexity = alg_results['complexity_metrics']
            report.append(f"{alg_name}:")
            report.append(f"   Complexity Order: {complexity['complexity_order']}")
            report.append(f"   Total Operations: {complexity['total_complexity']:.2e}")
            report.append(f"   Memory Requirement: {complexity['memory_complexity']:.2e}")
            report.append("")
        
        # 综合评分
        report.append("OVERALL ALGORITHM SCORE")
        report.append("-" * 40)
        
        scores = {}
        for alg_name, alg_results in results.items():
            # 归一化各项指标并计算综合得分
            min_rates = [r['performance']['min_rate']['mean'] for r in results.values()]
            sum_rates = [r['performance']['sum_rate']['mean'] for r in results.values()]
            fairness_vals = [r['performance']['fairness_index']['mean'] for r in results.values()]
            times = [r['efficiency']['optimization_time']['mean'] for r in results.values()]
            
            min_rate_norm = alg_results['performance']['min_rate']['mean'] / max(min_rates)
            sum_rate_norm = alg_results['performance']['sum_rate']['mean'] / max(sum_rates)
            fairness_norm = alg_results['performance']['fairness_index']['mean']
            time_norm = min(times) / alg_results['efficiency']['optimization_time']['mean']
            
            # 加权综合得分 (最小速率权重最大)
            overall_score = (0.4 * min_rate_norm + 0.25 * sum_rate_norm + 
                           0.2 * fairness_norm + 0.15 * time_norm) * 100
            
            scores[alg_name] = overall_score
        
        score_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (alg_name, score) in enumerate(score_ranking):
            report.append(f"#{i+1:2d}. {alg_name:25s}: {score:.2f}/100")
        
        # 推荐结论
        report.append("\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        best_min_rate = min_rate_ranking[0][0]
        best_sum_rate = sum_rate_ranking[0][0] 
        best_fairness = fairness_ranking[0][0]
        best_time = time_ranking[0][0]
        best_overall = score_ranking[0][0]
        
        report.append(f"• For Maximum Fairness (Min Rate): {best_min_rate}")
        report.append(f"• For Maximum Throughput (Sum Rate): {best_sum_rate}")
        report.append(f"• For Best User Fairness: {best_fairness}")
        report.append(f"• For Fastest Optimization: {best_time}")
        report.append(f"• Overall Best Algorithm: {best_overall}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📋 Comparison report saved to: {save_path}")
        
        print("\n" + report_text)
        
        return report_text


def main():
    """主函数"""
    print("🚀 Starting Comprehensive Algorithm Comparison")
    print("="*60)
    
    # 创建对比分析器（减少运行次数以便快速对比）
    comparator = AlgorithmComparison(num_runs=2)
    
    # 运行全面对比
    results = comparator.run_comprehensive_comparison()
    
    # 生成可视化
    print("\n🎨 Creating comprehensive visualization...")
    comparator.create_comprehensive_visualization(
        results, 
        save_path='/home/hzl/hyd/virtualForce/comprehensive_algorithm_comparison.png'
    )
    
    # 生成报告
    print("\n📋 Generating comparison report...")
    comparator.generate_comparison_report(
        results,
        save_path='/home/hzl/hyd/virtualForce/algorithm_comparison_report.txt'
    )
    
    print(f"\n✅ Comprehensive comparison completed successfully!")
    print(f"📊 Results summary:")
    for alg_name, alg_results in results.items():
        min_rate = alg_results['performance']['min_rate']['mean']
        sum_rate = alg_results['performance']['sum_rate']['mean']
        opt_time = alg_results['efficiency']['optimization_time']['mean']
        print(f"   {alg_name:25s}: Min={min_rate:.4f} Mbps, Sum={sum_rate:.2f} Mbps, Time={opt_time:.2f}s")


if __name__ == "__main__":
    main()
