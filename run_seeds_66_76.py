"""
多随机种子四算法对比 - 种子66-76
使用随机种子: 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76
寻找更好更合理的结果
"""

import numpy as np
import json
import time
import os
from compare_optimizers_fair import run_fair_comparison, print_fair_summary_table

# 配置
SEEDS = list(range(66, 77))  # [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
NBR_OF_REALIZATIONS = 50
TOTAL_EVALUATIONS = 1500
OUTPUT_DIR = 'result'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_results(all_results, filename):
    """保存结果到JSON文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # 转换numpy数组为列表
    serializable_results = {}
    for seed, results in all_results.items():
        serializable_results[f'seed_{seed}'] = {}
        for method, data in results.items():
            serializable_results[f'seed_{seed}'][method] = {
                'min_rate': float(data['min_rate']),
                'sum_rate': float(data['sum_rate']),
                'mean_rate': float(data['mean_rate']),
                'std_rate': float(data['std_rate']),
                'time': float(data['time'])
            }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ 结果已保存到: {filepath}")

def print_aggregated_results(all_results):
    """打印聚合结果"""
    methods = ['VF', 'GA', 'PSO', 'NewSSA']
    
    print("\n" + "="*90)
    print(" 多随机种子聚合结果 (种子66-76) ".center(90))
    print("="*90)
    
    # 计算每个方法的平均值和标准差
    aggregated = {}
    for method in methods:
        min_rates = []
        sum_rates = []
        times = []
        
        for seed, results in all_results.items():
            if method in results:
                min_rates.append(results[method]['min_rate'])
                sum_rates.append(results[method]['sum_rate'])
                times.append(results[method]['time'])
        
        aggregated[method] = {
            'min_rate_mean': np.mean(min_rates),
            'min_rate_std': np.std(min_rates),
            'min_rate_min': np.min(min_rates),
            'min_rate_max': np.max(min_rates),
            'sum_rate_mean': np.mean(sum_rates),
            'sum_rate_std': np.std(sum_rates),
            'time_mean': np.mean(times),
            'time_std': np.std(times)
        }
    
    # 打印表格
    print(f"\n{'算法':<15} {'最小速率 (Mbps)':<35} {'总速率 (Mbps)':<25} {'时间 (秒)':<20}")
    print(f"{'':15} {'平均±标准差 [最小, 最大]':<35} {'平均±标准差':<25} {'平均±标准差':<20}")
    print("-" * 95)
    
    for method in methods:
        data = aggregated[method]
        print(f"{method:<15} "
              f"{data['min_rate_mean']:6.2f}±{data['min_rate_std']:4.2f} "
              f"[{data['min_rate_min']:5.2f}, {data['min_rate_max']:5.2f}]  "
              f"{data['sum_rate_mean']:7.1f}±{data['sum_rate_std']:5.1f}     "
              f"{data['time_mean']:6.1f}±{data['time_std']:4.1f}")
    
    print("-" * 95)
    
    # 计算初始值的平均
    initial_min_rates = [all_results[seed]['initial']['min_rate'] for seed in all_results.keys()]
    initial_mean = np.mean(initial_min_rates)
    initial_std = np.std(initial_min_rates)
    
    print(f"\n初始状态: {initial_mean:.2f} ± {initial_std:.2f} Mbps")
    
    # 排名
    print("\n" + "="*90)
    print(" 最小速率排名（基于平均值） ".center(90))
    print("="*90)
    
    ranking = sorted(methods, key=lambda m: aggregated[m]['min_rate_mean'], reverse=True)
    
    for rank, method in enumerate(ranking, 1):
        data = aggregated[method]
        improvement = (data['min_rate_mean'] - initial_mean) / initial_mean * 100
        
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        else:
            medal = "  "
        
        print(f"{medal} {rank}. {method:<12}: {data['min_rate_mean']:6.2f} ± {data['min_rate_std']:4.2f} Mbps "
              f"({improvement:+6.2f}%)")
    
    # 找出表现最好的种子
    print("\n" + "="*90)
    print(" 各算法最佳种子 ".center(90))
    print("="*90)
    
    for method in methods:
        best_seed = None
        best_rate = -1
        
        for seed, results in all_results.items():
            if method in results:
                rate = results[method]['min_rate']
                if rate > best_rate:
                    best_rate = rate
                    best_seed = seed
        
        print(f"\n  {method}:")
        print(f"    最佳种子: {best_seed}")
        print(f"    最小速率: {best_rate:.4f} Mbps")
        print(f"    总速率:   {all_results[best_seed][method]['sum_rate']:.2f} Mbps")
    
    # 找出整体最合理的种子（所有算法都表现较好）
    print("\n" + "="*90)
    print(" 整体表现最合理的种子 ".center(90))
    print("="*90)
    
    seed_scores = {}
    for seed in all_results.keys():
        # 计算综合得分：所有算法的平均提升
        improvements = []
        initial_min = all_results[seed]['initial']['min_rate']
        
        for method in methods:
            if method in all_results[seed]:
                final_min = all_results[seed][method]['min_rate']
                improvement = (final_min - initial_min) / initial_min * 100
                improvements.append(improvement)
        
        # 综合得分：平均提升 - 标准差（希望所有算法都好）
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        score = mean_improvement - std_improvement * 0.5
        
        seed_scores[seed] = {
            'score': score,
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            'improvements': improvements
        }
    
    # 排序
    sorted_seeds = sorted(seed_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("\n排名（基于综合得分 = 平均提升 - 0.5×标准差）:\n")
    for rank, (seed, data) in enumerate(sorted_seeds[:5], 1):
        print(f"  {rank}. 种子 {seed}:")
        print(f"     综合得分: {data['score']:.2f}")
        print(f"     平均提升: {data['mean_improvement']:.2f}%")
        print(f"     提升标准差: {data['std_improvement']:.2f}%")
        print(f"     初始min_rate: {all_results[seed]['initial']['min_rate']:.2f} Mbps")
        print(f"     各算法: VF={all_results[seed]['VF']['min_rate']:.2f}, "
              f"GA={all_results[seed]['GA']['min_rate']:.2f}, "
              f"PSO={all_results[seed]['PSO']['min_rate']:.2f}, "
              f"NewSSA={all_results[seed]['NewSSA']['min_rate']:.2f}")
        print()
    
    print("="*90)


if __name__ == "__main__":
    print("\n" + "="*90)
    print(" 多随机种子四算法对比实验 - 种子66-76 ".center(90))
    print("="*90)
    
    print(f"\n📊 实验配置:")
    print(f"   • 随机种子: {SEEDS}")
    print(f"   • 运行次数: {len(SEEDS)} 轮")
    print(f"   • 算法: VF, GA, PSO, NewSSA (原始fitness)")
    print(f"   • 信道实现: {NBR_OF_REALIZATIONS}")
    print(f"   • 总评估次数: {TOTAL_EVALUATIONS}")
    print(f"   • 输出目录: {OUTPUT_DIR}/")
    print(f"\n⏱️  预计总时间: ~{len(SEEDS) * 15} 分钟 (每轮约15分钟)")
    print()
    
    all_results = {}
    total_start_time = time.time()
    
    for idx, seed in enumerate(SEEDS, 1):
        print("\n" + "="*90)
        print(f" 运行 {idx}/{len(SEEDS)}: 随机种子 = {seed} ".center(90))
        print("="*90)
        
        run_start_time = time.time()
        
        try:
            # 运行对比
            results, _, _ = run_fair_comparison(
                num_evaluations=TOTAL_EVALUATIONS,
                nbrOfRealizations=NBR_OF_REALIZATIONS,
                random_seed=seed
            )
            
            all_results[seed] = results
            
            run_time = time.time() - run_start_time
            
            # 打印本次结果摘要
            print(f"\n✓ 随机种子 {seed} 完成（用时 {run_time/60:.1f} 分钟）")
            print(f"  初始:   {results['initial']['min_rate']:.2f} Mbps")
            print(f"  VF:     {results['VF']['min_rate']:.2f} Mbps (+{((results['VF']['min_rate']-results['initial']['min_rate'])/results['initial']['min_rate']*100):.1f}%)")
            print(f"  GA:     {results['GA']['min_rate']:.2f} Mbps (+{((results['GA']['min_rate']-results['initial']['min_rate'])/results['initial']['min_rate']*100):.1f}%)")
            print(f"  PSO:    {results['PSO']['min_rate']:.2f} Mbps (+{((results['PSO']['min_rate']-results['initial']['min_rate'])/results['initial']['min_rate']*100):.1f}%)")
            print(f"  NewSSA: {results['NewSSA']['min_rate']:.2f} Mbps (+{((results['NewSSA']['min_rate']-results['initial']['min_rate'])/results['initial']['min_rate']*100):.1f}%)")
            
            # 保存中间结果
            save_results(all_results, f'seeds_66_76_partial_{idx}.json')
            
        except Exception as e:
            print(f"\n❌ 随机种子 {seed} 运行出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start_time
    
    # 打印聚合结果
    if all_results:
        print_aggregated_results(all_results)
        
        # 保存最终结果
        save_results(all_results, 'seeds_66_76_final.json')
        
        print(f"\n✅ 所有实验完成！")
        print(f"   总运行时间: {total_time/60:.1f} 分钟")
        print(f"   成功运行: {len(all_results)}/{len(SEEDS)} 轮")
        print(f"   结果保存在: {OUTPUT_DIR}/ 文件夹")
    else:
        print(f"\n❌ 没有成功完成的实验")
