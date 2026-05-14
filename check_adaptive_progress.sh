#!/bin/bash
# 检查自适应固定AP选择实验进度

echo "======================================"
echo "自适应固定AP选择实验进度检查"
echo "======================================"
echo ""

LOG_FILE="result/adaptive_fixed_comparison.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在"
    exit 1
fi

echo "📄 日志文件: $LOG_FILE"
echo "📏 日志行数: $(wc -l < $LOG_FILE)"
echo ""

echo "🔍 最新进展:"
tail -30 "$LOG_FILE"
echo ""

echo "======================================"
echo "已完成的结果文件:"
ls -lh result/adaptive_*uav_seed*.json 2>/dev/null | awk '{print $9, $5}' || echo "暂无结果文件"
echo ""

echo "======================================"
echo "完成状态:"
EXPECTED_FILES=6  # 9 UAV * 3 seeds + 12 UAV * 3 seeds
COMPLETED_FILES=$(ls result/adaptive_*uav_seed*.json 2>/dev/null | wc -l)
echo "已完成: $COMPLETED_FILES / $EXPECTED_FILES 个配置"

if [ $COMPLETED_FILES -eq $EXPECTED_FILES ]; then
    echo "✅ 所有实验已完成！"
else
    echo "⏳ 实验进行中..."
    PROGRESS=$((COMPLETED_FILES * 100 / EXPECTED_FILES))
    echo "进度: $PROGRESS%"
fi
