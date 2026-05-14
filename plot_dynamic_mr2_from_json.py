"""
仅重绘脚本：从已有 raw_results_mr2_quick.json 直接出图
====================================================
不重新运行任何实验，仅根据结果文件绘制时序图。

当前版本仅做以下改动：
1. DPI 调整为 900
2. 英文统一 Times New Roman
3. 中文统一 SimSong（macOS 可识别）
4. x/y label 改为中文
5. 导出 png / pdf / svg，其中 svg 适合导入 Word 作为矢量图
6. 能效比图强制显示顶部 0.5 刻度
7. 累计能耗图强制显示顶部 5000 刻度

注意：
- 为了保证“中文是宋体、英文是 Times New Roman”，不要对整条 xlabel/ylabel 再单独传 fontproperties。
- 采用全局字体回退：英文优先 Times New Roman，中文自动回退到 SimSong。
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= 字体设置（只改字体，不改原有风格） =================
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimSong']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.antialiased'] = True

OUT_DIR = 'result/dynamic_large_scale_plus'
JSON_PATH = os.path.join(OUT_DIR, 'raw_results_mr2_quick.json')

NUM_STEPS = 10
DT = 5.0

# 统一图例与配色（保持你原来的风格）
ALG_ORDER = ['DE2VF', 'DGA-CF', 'DPSO-CF', 'NSSA-CF']
COLORS = {
    'DE2VF': '#ee6f63',
    'DGA-CF': '#5fa8e8',
    'DPSO-CF': '#63cfa0',
    'NSSA-CF': '#b08adf'
}
MARKS = {
    'DE2VF': 'X',
    'DGA-CF': 's',
    'DPSO-CF': '^',
    'NSSA-CF': 'D'
}


def _normalize_key(raw_data):
    """兼容老结果里 'DE2VF-MR2' 这个 key。"""
    out = {}
    for s, rec in raw_data.items():
        if 'DE2VF' not in rec and 'DE2VF-MR2' in rec:
            rec['DE2VF'] = rec['DE2VF-MR2']
        out[s] = rec
    return out


def _collect(all_rec, key):
    seeds = sorted(all_rec.keys(), key=lambda x: int(x))
    arr_map = {}
    for alg in ALG_ORDER:
        series = []
        for s in seeds:
            if alg not in all_rec[s]:
                continue
            series.append(all_rec[s][alg][key])
        if len(series) > 0:
            arr_map[alg] = np.array(series)
    return arr_map


def plot_one(all_rec, key, ylabel_cn, out_tag, scale=1.0, show_title=False):
    t = np.arange(NUM_STEPS + 1) * DT
    arr_map = _collect(all_rec, key)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    for alg in ALG_ORDER:
        if alg not in arr_map:
            continue
        arr = arr_map[alg] * scale
        m = arr.mean(axis=0)
        ax.plot(
            t, m,
            color=COLORS[alg],
            marker=MARKS[alg],
            linewidth=2.3,
            markersize=6,
            label=alg
        )

    ax.set_xlabel('时间 (s)')
    ax.set_ylabel(ylabel_cn)

    if key == 'jfi':
        ax.set_ylim(0.75, 0.95)

    if key == 'energy_cumul':
        ax.set_ylim(0, 5000)
        ax.set_yticks(np.arange(0, 5001, 1000))

    if key == 'min_rate':
        ax.set_ylim(32, 65)

    if key == 'joint_score':
        ax.set_ylim(0.2, 1)

    if show_title:
        ax.set_title(out_tag, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, prop={'family': 'Times New Roman'})
    plt.tight_layout()

    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(
            os.path.join(OUT_DIR, f'{out_tag}.{ext}'),
            dpi=900,
            bbox_inches='tight'
        )
    plt.close(fig)


def plot_energy_eff(all_rec):
    # 使用 JS / cumulative energy（跳过 t=0，避免分母为 0）
    t = np.arange(1, NUM_STEPS + 1) * DT
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))

    seeds = sorted(all_rec.keys(), key=lambda x: int(x))
    for alg in ALG_ORDER:
        series = []
        for s in seeds:
            if alg not in all_rec[s]:
                continue

            js = np.array(all_rec[s][alg]['joint_score'], dtype=float)[1:]
            # cumulative energy: J -> MJ，统一量纲并提升可读性
            ec_mj = (np.array(all_rec[s][alg]['energy_cumul'], dtype=float) / 1e6)[1:]

            ee = js / (ec_mj + 1e-9)
            series.append(ee)

        if not series:
            continue

        arr = np.array(series)
        m = arr.mean(axis=0)
        ax.plot(
            t, m,
            color=COLORS[alg],
            marker=MARKS[alg],
            linewidth=2.3,
            markersize=6,
            label=alg
        )

    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('能效比')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, prop={'family': 'Times New Roman'})
    plt.tight_layout()

    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(
            os.path.join(OUT_DIR, f'fig_energy_eff.{ext}'),
            dpi=900,
            bbox_inches='tight'
        )
    plt.close(fig)


def main():
    if not os.path.exists(JSON_PATH):
        print(f'缺少结果文件: {JSON_PATH}')
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    all_rec = _normalize_key(raw)

    plot_one(all_rec, 'min_rate', '最小用户速率 (Mbps)', 'fig_mr2_min_rate')
    plot_one(all_rec, 'jfi', r'$JFI_{\mathrm{eff}}$', 'fig_mr2_jfi')
    plot_one(all_rec, 'energy_cumul', '累计能耗 (kJ)', 'fig_mr2_energy', scale=1 / 1000)
    plot_one(all_rec, 'joint_score', '综合JS', 'fig_mr2_joint_score')
    plot_energy_eff(all_rec)

    print(f'重绘完成，输出目录: {OUT_DIR}/')
    print('已导出格式: png / pdf / svg')
    print('导入 Word 建议优先使用 svg 作为矢量图。')


if __name__ == '__main__':
    main()