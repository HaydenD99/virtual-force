#!/bin/bash
# 检查所有后台实验的状态

echo "=========================================="
echo "    后台实验状态检查"
echo "=========================================="
echo ""

# 检查种子55-65的实验
if [ -f "result/seeds_55_65.log" ]; then
    echo "📊 种子55-65实验 (11轮):"
    lines=$(wc -l < result/seeds_55_65.log)
    echo "   日志行数: $lines"
    
    # 检查已完成的轮次
    completed=$(ls result/seeds_55_65_partial_*.json 2>/dev/null | wc -l)
    echo "   已完成轮次: $completed / 11"
    
    # 显示最新几行
    echo "   最新日志:"
    tail -5 result/seeds_55_65.log | sed 's/^/     /'
    echo ""
fi

# 检查种子66-76的实验
if [ -f "result/seeds_66_76.log" ]; then
    echo "📊 种子66-76实验 (11轮):"
    lines=$(wc -l < result/seeds_66_76.log)
    echo "   日志行数: $lines"
    
    # 检查已完成的轮次
    completed=$(ls result/seeds_66_76_partial_*.json 2>/dev/null | wc -l)
    echo "   已完成轮次: $completed / 11"
    
    # 显示最新几行
    echo "   最新日志:"
    tail -5 result/seeds_66_76.log | sed 's/^/     /'
    echo ""
fi

# 检查种子62实验
if [ -f "result/seed62_experiments.log" ]; then
    echo "📊 种子62实验 (6 UAV + 12 UAV):"
    lines=$(wc -l < result/seed62_experiments.log)
    echo "   日志行数: $lines"
    
    # 检查结果文件
    if [ -f "result/original_6uav_seed62.json" ]; then
        echo "   ✅ 6 UAV - 已完成"
    else
        echo "   ⏳ 6 UAV - 进行中"
    fi
    
    if [ -f "result/original_12uav_seed62.json" ]; then
        echo "   ✅ 12 UAV - 已完成"
    else
        echo "   ⏳ 12 UAV - 等待中"
    fi
    
    # 显示最新几行
    echo "   最新日志:"
    tail -5 result/seed62_experiments.log | sed 's/^/     /'
    echo ""
fi

# 检查种子63实验
if [ -f "result/seed63_experiments.log" ]; then
    echo "📊 种子63实验 (6 UAV + 12 UAV) - 原始Fitness:"
    lines=$(wc -l < result/seed63_experiments.log)
    echo "   日志行数: $lines"
    
    # 检查结果文件
    if [ -f "result/original_6uav_seed63.json" ]; then
        echo "   ✅ 6 UAV - 已完成"
    else
        echo "   ⏳ 6 UAV - 进行中"
    fi
    
    if [ -f "result/original_12uav_seed63.json" ]; then
        echo "   ✅ 12 UAV - 已完成"
    else
        echo "   ⏳ 12 UAV - 等待中"
    fi
    
    # 显示最新几行
    echo "   最新日志:"
    tail -5 result/seed63_experiments.log | sed 's/^/     /'
    echo ""
fi

# 检查种子62加权实验
if [ -f "result/weighted_6uav_seed62.log" ]; then
    echo "📊 种子62实验 (6 UAV) - 加权Fitness:"
    lines=$(wc -l < result/weighted_6uav_seed62.log)
    echo "   日志行数: $lines"
    
    if [ -f "result/weighted_6uav_seed62.json" ]; then
        echo "   ✅ 已完成"
    else
        echo "   ⏳ 进行中"
    fi
    
    echo "   最新日志:"
    tail -5 result/weighted_6uav_seed62.log | sed 's/^/     /'
    echo ""
fi

# 检查种子63加权实验
if [ -f "result/weighted_6uav_seed63.log" ]; then
    echo "📊 种子63实验 (6 UAV) - 加权Fitness:"
    lines=$(wc -l < result/weighted_6uav_seed63.log)
    echo "   日志行数: $lines"
    
    if [ -f "result/weighted_6uav_seed63.json" ]; then
        echo "   ✅ 已完成"
    else
        echo "   ⏳ 进行中"
    fi
    
    echo "   最新日志:"
    tail -5 result/weighted_6uav_seed63.log | sed 's/^/     /'
    echo ""
fi

# 检查选定种子实验 (71, 75, 76)
if [ -f "result/selected_seeds_71_75_76.log" ]; then
    echo "📊 选定种子实验 (71, 75, 76) - 6 UAV + 12 UAV:"
    lines=$(wc -l < result/selected_seeds_71_75_76.log)
    echo "   日志行数: $lines"
    
    # 检查完成的实验
    completed_71_6=$([ -f "result/original_6uav_seed71.json" ] && echo "✅" || echo "⏳")
    completed_71_12=$([ -f "result/original_12uav_seed71.json" ] && echo "✅" || echo "⏳")
    completed_75_6=$([ -f "result/original_6uav_seed75.json" ] && echo "✅" || echo "⏳")
    completed_75_12=$([ -f "result/original_12uav_seed75.json" ] && echo "✅" || echo "⏳")
    completed_76_6=$([ -f "result/original_6uav_seed76.json" ] && echo "✅" || echo "⏳")
    completed_76_12=$([ -f "result/original_12uav_seed76.json" ] && echo "✅" || echo "⏳")
    
    echo "   种子71: 6UAV $completed_71_6  12UAV $completed_71_12"
    echo "   种子75: 6UAV $completed_75_6  12UAV $completed_75_12"
    echo "   种子76: 6UAV $completed_76_6  12UAV $completed_76_12"
    
    echo "   最新日志:"
    tail -5 result/selected_seeds_71_75_76.log | sed 's/^/     /'
    echo ""
fi

echo "=========================================="
echo ""
echo "💡 提示:"
echo "   - 查看完整日志: tail -f result/xxx.log"
echo "   - 检查结果文件: ls -lt result/*.json | head"
echo ""
