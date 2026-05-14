#!/usr/bin/env python3
"""
四算法对比测试：简化虚拟力 vs 平衡虚拟力 vs 增强虚拟力 vs 遗传算法
找到最适合最小速率优化的最佳算法
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

from simplified_virtual_force_optimizer import SimplifiedVirtualForceOptimizer, create_simplified_config
from balanced_virtual_force_optimizer import BalancedVirtualForceOptimizer, create_balanced_config
from enhanced_virtual_force_optimizer import EnhancedVirtualForceOptimizer, create_enhanced_config
from genetic_algorithm_optimizer import GeneticAlgorithmOptimizer, create_ga_config

def create_unified_test_config() -> Dict:
    """创建统一的测试配置"""
    return {
        # 基本参数
        'square_length': 1000,
        'num_UE': 45,  # 适中的规模
        'num_UAV': 7,
        'num_ground_AP': 4,
        'M': 4,
        
        # 高度设置
        'UE_height': 1.65,
        'ground_AP_height': 15.0,
        'UAV_height': 50.0,
        
        # 信道参数
        'alpha': 3.67,
        'constant_term': -30.5,
        'B': 20e6,
        'Pmax': 1000,
        'noise_figure': 7,
        'distance_vertical': 150,
        'tau_p': 45,
        'tau_c': 200,
        'ASD_deg': 10,
        'num_serving_APs': 3,
        'nbrOfRealizations': 35,
        
        # 简化虚拟力参数
        'K_min_rate': 8e4,
        'K_separation': 2e4,
        'K_boundary': 3e4,
        'separation_distance': 120,
        'boundary_margin': 60,
        
        # 平衡虚拟力参数
        'K_universal': 3e4,
        'K_cooperation': 2e4,
        'cooperation_distance': 200,
        
        # 增强虚拟力参数
        'K_base': 3e4,
        'w_fairness': 1.2,
        'w_capacity': 0.4,
        'w_separation': 1.2,
        'w_cooperation': 0.5,
        'w_boundary': 0.8,
        'w_load_balance': 0.8,
        
        # 优化参数
        'step_size': 25,
        'max_iterations': 80,
        'convergence_threshold': 2e-3,
        
        # 遗传算法参数
        'population_size': 25,
        'max_generations': 80,
        'crossover_rate': 0.8,
        'mutation_rate': 0.15,
        'elite_size': 5,
        'tournament_size': 4,
    }

def run_four_algorithm_comparison() -> Dict:
    """运行四种算法的对比测试"""
    print("🚀 四算法终极对比：简化 vs 平衡 vs 增强 vs 遗传算法")
    print("🎯 目标：找到最小速率优化的最佳算法")
    print("=" * 80)
    
    # 设置随机种子确保公平对比
    np.random.seed(42)
    
    # 创建统一配置
    base_config = create_unified_test_config()
    
    # 创建四种优化器
    algorithms = {
        'simplified': ('简化虚拟力', SimplifiedVirtualForceOptimizer(base_config), 'red'),
        'balanced': ('平衡虚拟力', BalancedVirtualForceOptimizer(base_config), 'green'),
        'enhanced': ('增强虚拟力', EnhancedVirtualForceOptimizer(base_config), 'blue'),
        'genetic': ('遗传算法', GeneticAlgorithmOptimizer(base_config), 'orange')
    }
    
    # 使用第一个算法初始化相同的位置
    first_optimizer = list(algorithms.values())[0][1]
    UE_pos, ground_AP_pos, initial_UAV_pos = first_optimizer.initialize_positions()
    
    print(f"系统配置：{base_config['num_UE']}用户，{base_config['num_UAV']}UAV，{base_config['num_ground_AP']}地面AP")
    
    # 计算初始性能
    all_AP_pos = np.vstack([ground_AP_pos, initial_UAV_pos])
    _, _, betas = first_optimizer.compute_channel_model(UE_pos, all_AP_pos)
    mask = first_optimizer.compute_AP_selection_mask(betas)
    initial_rates, initial_sum_rate = first_optimizer.compute_user_rates(UE_pos, all_AP_pos, mask)
    initial_min_rate = initial_rates.min()
    
    print(f"初始性能：总速率={initial_sum_rate:.2f}Mbps，最小速率={initial_min_rate:.4f}Mbps")
    print()
    
    results = {}
    
    # 依次测试四种算法
    for algo_key, (algo_name, optimizer, color) in algorithms.items():
        print(f"🔥 测试 {algo_name}...")
        start_time = time.time()
        
        try:
            if algo_key == 'genetic':
                # 遗传算法不需要初始UAV位置
                result = optimizer.optimize(UE_pos, ground_AP_pos)
            else:
                # 虚拟力算法使用相同的初始位置
                result = optimizer.optimize(UE_pos, ground_AP_pos, initial_UAV_pos.copy())
            
            actual_time = time.time() - start_time
            
            print(f"✅ {algo_name}完成：{actual_time:.2f}秒")
            print(f"   最终：总速率={result['final_sum_rate']:.2f}Mbps，"
                  f"最小速率={result['final_min_rate']:.4f}Mbps")
            print(f"   改善：总速率+{(result['final_sum_rate']/initial_sum_rate-1)*100:.1f}%，"
                  f"最小速率+{(result['final_min_rate']/initial_min_rate-1)*100:.1f}%")
            
            # 添加公共信息
            result['algorithm'] = algo_name
            result['algo_key'] = algo_key
            result['color'] = color
            result['actual_time'] = actual_time
            result['success'] = True
            result['initial_sum_rate'] = initial_sum_rate
            result['initial_min_rate'] = initial_min_rate
            result['UE_pos'] = UE_pos
            result['ground_AP_pos'] = ground_AP_pos
            result['initial_UAV_pos'] = initial_UAV_pos
            
            results[algo_key] = result
            
        except Exception as e:
            print(f"❌ {algo_name}失败：{str(e)}")
            results[algo_key] = {
                'success': False, 
                'error': str(e),
                'algorithm': algo_name,
                'algo_key': algo_key,
                'color': color
            }
        
        print()
    
    return results

def analyze_four_algorithm_results(results: Dict):
    """分析四算法对比结果"""
    print("📊 四算法终极对比分析")
    print("=" * 80)
    
    successful_algorithms = [key for key in results if results[key].get('success', False)]
    
    if not successful_algorithms:
        print("❌ 所有算法都失败了！")
        return
    
    print(f"成功运行的算法：{len(successful_algorithms)}/4")
    print()
    
    # 详细性能表格
    print(f"{'算法':<12} {'最小速率':<12} {'最小改善%':<12} {'总速率':<12} {'总改善%':<12} {'时间(秒)':<10} {'评级'}")
    print("-" * 85)
    
    # 收集所有成功算法的数据
    algorithm_data = []
    for algo_key in successful_algorithms:
        result = results[algo_key]
        min_rate = result['final_min_rate']
        sum_rate = result['final_sum_rate']
        actual_time = result['actual_time']
        min_improvement = (min_rate / result['initial_min_rate'] - 1) * 100
        sum_improvement = (sum_rate / result['initial_sum_rate'] - 1) * 100
        
        algorithm_data.append({
            'key': algo_key,
            'name': result['algorithm'],
            'min_rate': min_rate,
            'sum_rate': sum_rate,
            'min_improvement': min_improvement,
            'sum_improvement': sum_improvement,
            'time': actual_time
        })
    
    # 按最小速率排序（最重要的指标）
    algorithm_data.sort(key=lambda x: x['min_rate'], reverse=True)
    
    # 输出排序后的结果
    for i, data in enumerate(algorithm_data):
        # 简单的评级系统
        if i == 0:
            rating = "🥇优秀"
        elif i == 1:
            rating = "🥈良好"
        elif i == 2:
            rating = "🥉一般"
        else:
            rating = "📉较差"
        
        print(f"{data['name']:<12} {data['min_rate']:<12.4f} {data['min_improvement']:<12.1f} "
              f"{data['sum_rate']:<12.2f} {data['sum_improvement']:<12.1f} {data['time']:<10.2f} {rating}")
    
    print()
    
    # 最佳算法分析
    best_algo = algorithm_data[0]
    print(f"🏆 最小速率优化冠军：{best_algo['name']}")
    print(f"   🎯 最小速率：{best_algo['min_rate']:.4f} Mbps ({best_algo['min_improvement']:+.1f}%)")
    print(f"   📊 总速率：{best_algo['sum_rate']:.2f} Mbps ({best_algo['sum_improvement']:+.1f}%)")
    print(f"   ⏱️  优化时间：{best_algo['time']:.2f} 秒")
    
    # 各项指标最佳
    best_min_rate = max(algorithm_data, key=lambda x: x['min_rate'])
    best_sum_rate = max(algorithm_data, key=lambda x: x['sum_rate'])
    best_time = min(algorithm_data, key=lambda x: x['time'])
    
    print(f"\n🏅 单项最佳：")
    print(f"   最小速率最佳：{best_min_rate['name']} ({best_min_rate['min_rate']:.4f} Mbps)")
    print(f"   总速率最佳：{best_sum_rate['name']} ({best_sum_rate['sum_rate']:.2f} Mbps)")
    print(f"   效率最佳：{best_time['name']} ({best_time['time']:.2f} 秒)")
    
    # 算法特点分析
    print(f"\n💡 算法特点分析：")
    for data in algorithm_data:
        key = data['key']
        name = data['name']
        
        if key == 'simplified':
            print(f"   🔴 {name}：简洁高效，专注核心目标")
        elif key == 'balanced':
            print(f"   🟢 {name}：平衡全面，考虑多种因素")
        elif key == 'enhanced':
            print(f"   🔵 {name}：功能丰富，复杂度较高")
        elif key == 'genetic':
            print(f"   🟠 {name}：全局搜索，进化优化")
    
    # 绘制对比图
    if len(successful_algorithms) > 1:
        plot_four_algorithm_comparison(results, successful_algorithms)

def plot_four_algorithm_comparison(results: Dict, successful_algorithms: List):
    """绘制四算法对比图"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('四算法终极性能对比分析', fontsize=16)
    
    # 获取第一个成功算法的位置数据
    first_algo = successful_algorithms[0]
    UE_pos = results[first_algo]['UE_pos']
    ground_AP_pos = results[first_algo]['ground_AP_pos']
    initial_UAV_pos = results[first_algo]['initial_UAV_pos']
    
    colors = {key: results[key]['color'] for key in successful_algorithms}
    labels = {key: results[key]['algorithm'] for key in successful_algorithms}
    
    # 1. UAV最终位置对比
    ax = axes[0, 0]
    ax.scatter(UE_pos[:, 0], UE_pos[:, 1], c='lightblue', s=12, alpha=0.6, label='用户')
    ax.scatter(ground_AP_pos[:, 0], ground_AP_pos[:, 1], c='black', s=60, marker='s', label='地面AP')
    ax.scatter(initial_UAV_pos[:, 0], initial_UAV_pos[:, 1], c='gray', s=40, marker='^', 
              label='初始UAV', alpha=0.4)
    
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            optimized_pos = results[algo_key]['optimized_UAV_pos']
            ax.scatter(optimized_pos[:, 0], optimized_pos[:, 1], 
                      c=colors[algo_key], s=70, marker='^', 
                      label=labels[algo_key], alpha=0.8)
    
    ax.set_title('UAV最终位置对比')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    
    # 2. 最小速率收敛曲线
    ax = axes[0, 1]
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            result = results[algo_key]
            history = result['history']
            
            if algo_key == 'genetic':
                x_data = history.get('generations', [])
                y_data = history.get('best_min_rates', [])
            else:
                x_data = history.get('iterations', [])
                y_data = history.get('min_rates', [])
            
            if x_data and y_data:
                ax.plot(x_data, y_data, color=colors[algo_key], linewidth=2, 
                       label=labels[algo_key], alpha=0.8)
    
    ax.set_xlabel('迭代次数/代数')
    ax.set_ylabel('最小速率 (Mbps)')
    ax.set_title('最小速率收敛对比')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. 总速率收敛曲线
    ax = axes[0, 2]
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            result = results[algo_key]
            history = result['history']
            
            if algo_key == 'genetic':
                x_data = history.get('generations', [])
                y_data = history.get('best_sum_rates', [])
            else:
                x_data = history.get('iterations', [])
                y_data = history.get('sum_rates', [])
            
            if x_data and y_data:
                ax.plot(x_data, y_data, color=colors[algo_key], linewidth=2, 
                       label=labels[algo_key], alpha=0.8)
    
    ax.set_xlabel('迭代次数/代数')
    ax.set_ylabel('总速率 (Mbps)')
    ax.set_title('总速率收敛对比')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 最小速率性能对比（柱状图）
    ax = axes[1, 0]
    algorithms = []
    min_rates = []
    
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            algorithms.append(labels[algo_key])
            min_rates.append(results[algo_key]['final_min_rate'])
    
    bars = ax.bar(algorithms, min_rates, color=[colors[key] for key in successful_algorithms], alpha=0.7)
    ax.set_ylabel('最小速率 (Mbps)')
    ax.set_title('最小速率对比（越高越好）')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, rate in zip(bars, min_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
               f'{rate:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 5. 改善百分比对比
    ax = axes[1, 1]
    min_improvements = []
    sum_improvements = []
    
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            result = results[algo_key]
            min_imp = (result['final_min_rate'] / result['initial_min_rate'] - 1) * 100
            sum_imp = (result['final_sum_rate'] / result['initial_sum_rate'] - 1) * 100
            min_improvements.append(min_imp)
            sum_improvements.append(sum_imp)
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, min_improvements, width, label='最小速率改善%', alpha=0.7)
    bars2 = ax.bar(x + width/2, sum_improvements, width, label='总速率改善%', alpha=0.7)
    
    ax.set_ylabel('改善百分比 (%)')
    ax.set_title('性能改善对比')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. 优化时间对比
    ax = axes[1, 2]
    times = []
    
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            times.append(results[algo_key]['actual_time'])
    
    bars = ax.bar(algorithms, times, color=[colors[key] for key in successful_algorithms], alpha=0.7)
    ax.set_ylabel('优化时间 (秒)')
    ax.set_title('效率对比（越低越好）')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, time_val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
               f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 7. 速率分布对比（箱线图）
    ax = axes[2, 0]
    rate_distributions = []
    box_labels = []
    
    for algo_key in successful_algorithms:
        if results[algo_key].get('success', False):
            rate_distributions.append(results[algo_key]['final_rates'])
            box_labels.append(labels[algo_key])
    
    bp = ax.boxplot(rate_distributions, labels=box_labels, patch_artist=True)
    
    # 设置箱线图颜色
    for patch, algo_key in zip(bp['boxes'], successful_algorithms):
        patch.set_facecolor(colors[algo_key])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('速率 (Mbps)')
    ax.set_title('速率分布对比')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15)
    
    # 8. 综合性能雷达图
    ax = axes[2, 1]
    
    if len(successful_algorithms) >= 2:
        categories = ['最小速率', '总速率', '收敛速度', '稳定性', '效率']
        
        # 计算各算法的评分
        algo_scores = {}
        for algo_key in successful_algorithms:
            if results[algo_key].get('success', False):
                result = results[algo_key]
                
                # 标准化评分（0-1）
                min_rate_score = min(1.0, result['final_min_rate'] / 0.8)  # 以0.8为满分
                sum_rate_score = min(1.0, result['final_sum_rate'] / 250)  # 以250为满分
                speed_score = max(0, 1.0 - result['actual_time'] / 60)     # 以60秒为基准
                stability_score = max(0, 1.0 - result['final_rates'].std() / 8)
                efficiency_score = min_rate_score * 0.6 + sum_rate_score * 0.4  # 综合效率
                
                algo_scores[algo_key] = [min_rate_score, sum_rate_score, speed_score, 
                                       stability_score, efficiency_score]
        
        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for algo_key, scores in algo_scores.items():
            scores += scores[:1]  # 闭合
            ax.plot(angles, scores, 'o-', linewidth=2, label=labels[algo_key], 
                   color=colors[algo_key])
            ax.fill(angles, scores, alpha=0.15, color=colors[algo_key])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图')
        ax.legend(fontsize=8)
        ax.grid(True)
    
    # 9. 算法优势分析
    ax = axes[2, 2]
    ax.axis('off')
    
    # 创建总结文本
    summary_text = "🏆 四算法对比总结\n\n"
    
    # 按性能排序
    performance_ranking = sorted(successful_algorithms, 
                               key=lambda x: results[x]['final_min_rate'], reverse=True)
    
    for i, algo_key in enumerate(performance_ranking):
        result = results[algo_key]
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣"][i] if i < 4 else f"{i+1}️⃣"
        summary_text += f"{rank_emoji} {result['algorithm']}\n"
        summary_text += f"   最小速率: {result['final_min_rate']:.4f}\n"
        summary_text += f"   改善率: {(result['final_min_rate']/result['initial_min_rate']-1)*100:+.1f}%\n\n"
    
    # 推荐结论
    best_algo = results[performance_ranking[0]]
    summary_text += f"💡 最小速率优化推荐:\n"
    summary_text += f"   🏅 {best_algo['algorithm']}\n"
    summary_text += f"   🎯 专注目标优化\n"
    summary_text += f"   ⚡ 效果最佳"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/hzl/hyd/virtualForce/four_algorithm_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("🎯 四算法终极对比测试")
    print("💫 目标：找到最小速率优化的最强算法")
    print()
    
    # 运行对比测试
    results = run_four_algorithm_comparison()
    
    # 分析结果
    analyze_four_algorithm_results(results)
    
    print(f"\n🎉 四算法终极对比完成！")
    print("🏆 冠军已诞生，详细对比图表已保存")

if __name__ == "__main__":
    main()









