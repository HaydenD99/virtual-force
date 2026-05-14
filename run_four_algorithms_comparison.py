"""
运行四算法公平对比（使用3D版本ISSA + 均匀分布Ground AP）
"""

import sys
import time
from compare_optimizers_fair import run_fair_comparison, print_fair_summary_table, plot_fair_comparison

if __name__ == "__main__":
    print("\n" + "="*90)
    print(" 四算法公平对比实验 ".center(90))
    print(" (使用3D版本ISSA + 均匀分布Ground AP) ".center(90))
    print("="*90)
    
    print("\n⚠️  注意：完整实验大约需要 15-20 分钟")
    print("   - VF:   ~50 秒")
    print("   - GA:   ~300 秒")
    print("   - PSO:  ~300 秒")
    print("   - ISSA: ~280 秒")
    print()
    
    # 配置参数
    TOTAL_EVALUATIONS = 1500
    NBR_OF_REALIZATIONS = 50
    RANDOM_SEED = 43  # 使用42作为随机种子（与之前测试一致）
    
    print(f"📊 实验配置:")
    print(f"   • 总评估次数: {TOTAL_EVALUATIONS} (30个体 × 50迭代)")
    print(f"   • 信道实现: {NBR_OF_REALIZATIONS}")
    print(f"   • 随机种子: {RANDOM_SEED}")
    print(f"   • Ground AP: 均匀分布在地面区域")
    print(f"   • ISSA版本: 3D标准版（严格遵循论文公式14-20）")
    print()
    
    # 开始计时
    total_start = time.time()
    
    # 运行对比
    print("🚀 开始四算法公平对比...")
    print()
    
    try:
        results, UE_pos, ground_AP_pos = run_fair_comparison(
            num_evaluations=TOTAL_EVALUATIONS,
            nbrOfRealizations=NBR_OF_REALIZATIONS,
            random_seed=RANDOM_SEED
        )
        
        total_time = time.time() - total_start
        
        # 打印结果
        print_fair_summary_table(results)
        
        # 绘制图表
        print("\n[5/5] Generating comparison plot...")
        plot_fair_comparison(results, save_path='optimizer_comparison_fair_3d.png')
        
        # 最终总结
        print("\n" + "="*90)
        print(" 实验完成！ ".center(90))
        print("="*90)
        
        print(f"\n⏱️  总运行时间: {total_time/60:.2f} 分钟 ({total_time:.1f} 秒)")
        
        print(f"\n📊 最终排名（最小用户速率）:")
        opt_methods = ['VF', 'GA', 'PSO', 'ISSA']
        ranking = sorted(opt_methods, key=lambda m: results[m]['min_rate'], reverse=True)
        
        for rank, method in enumerate(ranking, 1):
            min_rate = results[method]['min_rate']
            improvement = (min_rate - results['initial']['min_rate']) / results['initial']['min_rate'] * 100
            
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"
            else:
                medal = "  "
            
            print(f"   {medal} {rank}. {method:6s}: {min_rate:7.4f} Mbps ({improvement:+6.2f}%)")
        
        print(f"\n✅ 结果图表已保存到: optimizer_comparison_fair_3d.png")
        
        # 检查ISSA性能
        issa_improvement = (results['ISSA']['min_rate'] - results['initial']['min_rate']) / results['initial']['min_rate'] * 100
        
        print(f"\n🎯 ISSA (3D标准版) 性能评估:")
        if issa_improvement > 15:
            print(f"   🎉🎉🎉 优秀！ISSA达到了{issa_improvement:+.2f}%的改进")
        elif issa_improvement > 5:
            print(f"   ✅ 良好！ISSA实现了{issa_improvement:+.2f}%的改进")
        elif issa_improvement > 0:
            print(f"   📈 及格！ISSA实现了{issa_improvement:+.2f}%的改进")
        else:
            print(f"   ❌ 需要改进！ISSA表现为{issa_improvement:+.2f}%")
        
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
