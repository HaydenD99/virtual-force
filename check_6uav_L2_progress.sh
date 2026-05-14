#!/bin/bash
# 检查6 UAV L=2实验进度

echo "======================================"
echo "6 UAV + L=2 实验进度检查"
echo "======================================"
echo ""

LOG_FILE="result/6uav_L2_multi_seeds.log"
EXPECTED_SEEDS=(51 62 63 $(seq 77 87))
TOTAL_SEEDS=${#EXPECTED_SEEDS[@]}

echo "📊 实验配置:"
echo "   • 目标种子: ${EXPECTED_SEEDS[@]}"
echo "   • 总数: $TOTAL_SEEDS 个"
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "📄 日志文件: $LOG_FILE"
    echo "📏 日志行数: $(wc -l < $LOG_FILE)"
    echo ""
    
    echo "🔍 最新日志:"
    tail -30 "$LOG_FILE"
    echo ""
else
    echo "⚠️  日志文件不存在"
    echo ""
fi

echo "======================================"
echo "已完成的结果文件:"
COMPLETED=0
for seed in "${EXPECTED_SEEDS[@]}"; do
    FILE="result/6uav_L2_seed${seed}.json"
    if [ -f "$FILE" ]; then
        FILESIZE=$(ls -lh "$FILE" | awk '{print $5}')
        FILETIME=$(ls -lh "$FILE" | awk '{print $6, $7, $8}')
        echo "  ✓ seed $seed: $FILESIZE ($FILETIME)"
        ((COMPLETED++))
    fi
done

if [ $COMPLETED -eq 0 ]; then
    echo "  暂无完成的文件"
fi

echo ""
echo "======================================"
echo "完成状态:"
echo "   已完成: $COMPLETED / $TOTAL_SEEDS 个种子"

if [ $COMPLETED -eq $TOTAL_SEEDS ]; then
    echo "   状态: ✅ 所有实验已完成！"
else
    PROGRESS=$((COMPLETED * 100 / TOTAL_SEEDS))
    echo "   状态: ⏳ 实验进行中... ($PROGRESS%)"
    
    REMAINING=$((TOTAL_SEEDS - COMPLETED))
    EST_TIME=$((REMAINING * 15))
    echo "   剩余: $REMAINING 个种子"
    echo "   预计剩余时间: ~$EST_TIME 分钟"
fi

echo ""
echo "======================================"
echo "未完成的种子:"
MISSING=""
for seed in "${EXPECTED_SEEDS[@]}"; do
    FILE="result/6uav_L2_seed${seed}.json"
    if [ ! -f "$FILE" ]; then
        MISSING="$MISSING $seed"
    fi
done

if [ -z "$MISSING" ]; then
    echo "  无（全部完成）"
else
    echo "  $MISSING"
fi

echo ""
echo "======================================"
