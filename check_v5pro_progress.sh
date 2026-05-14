#!/bin/bash
echo "--- V5-Pro 全方位对比进度监控 ---"
total=$(ls result/v5pro_comparison/v5pro_comp_*.json 2>/dev/null | wc -l)
echo "已完成实验数量: $total"
echo "最近生成的 5 个结果:"
ls -lhrt result/v5pro_comparison/v5pro_comp_*.json | tail -n 5
echo "---------------------------------"
