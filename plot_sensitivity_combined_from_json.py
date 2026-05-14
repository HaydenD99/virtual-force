"""
从 sensitivity_keyparams.json 直接绘制合并图（3x2）
=================================================
- 不重跑实验
- 将 theta_l / beta / s0 三组敏感性结果合并到一张图中
- 参数名称按需求替换为: tau_l, u_d, S_base
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'result/sensitivity_keyparams'
JSON_PATH = os.path.join(OUT_DIR, 'sensitivity_keyparams.json')


def _get_series(block):
    xs = [d['value'] for d in block['static']]

    s_js = [d['js'] for d in block['static']]
    s_mr = [d['min_rate'] for d in block['static']]
    s_jfi = [d['jfi'] for d in block['static']]

    d_js = [d['avg_js'] for d in block['dynamic']]
    d_mr = [d['avg_min_rate'] for d in block['dynamic']]
    d_jfi = [d['avg_jfi'] for d in block['dynamic']]

    return xs, s_js, s_mr, s_jfi, d_js, d_mr, d_jfi


def _plot_one_row(ax_l, ax_r, block, x_label, title_tag):
    xs, s_js, s_mr, s_jfi, d_js, d_mr, d_jfi = _get_series(block)

    # static: 左轴(JS/JFI), 右轴(Min-Rate)
    ax_l.plot(xs, s_js, marker='o', color='#ee6f63', label='JS', linewidth=2)
    ax_l.plot(xs, s_jfi, marker='^', color='#63cfa0', label='JFI', linewidth=2)
    ax_l.set_title(f'Static ({title_tag})', fontweight='bold')
    ax_l.set_xlabel(x_label)
    ax_l.set_ylabel('JS / JFI')
    ax_l.grid(True, alpha=0.3)

    ax_l2 = ax_l.twinx()
    ax_l2.plot(xs, s_mr, marker='s', color='#5fa8e8', label='Min-Rate', linewidth=2)
    ax_l2.set_ylabel('Min-Rate (Mbps)')

    # 扩大显示范围，弱化视觉波动
    ax_l.set_ylim(0.60, 1.00)
    ax_l2.set_ylim(35, 65)

    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_l2.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc='best', fontsize=9)

    # dynamic: 左轴(Avg JS/JFI), 右轴(Avg Min-Rate)
    ax_r.plot(xs, d_js, marker='o', color='#ee6f63', label='Avg JS', linewidth=2)
    ax_r.plot(xs, d_jfi, marker='^', color='#63cfa0', label='Avg JFI', linewidth=2)
    ax_r.set_title(f'Dynamic ({title_tag})', fontweight='bold')
    ax_r.set_xlabel(x_label)
    ax_r.set_ylabel('Avg JS / Avg JFI')
    ax_r.grid(True, alpha=0.3)

    ax_r2 = ax_r.twinx()
    ax_r2.plot(xs, d_mr, marker='s', color='#5fa8e8', label='Avg Min-Rate', linewidth=2)
    ax_r2.set_ylabel('Avg Min-Rate (Mbps)')

    # 扩大显示范围，弱化视觉波动
    ax_r.set_ylim(0.60, 1.00)
    ax_r2.set_ylim(35, 65)

    h1, l1 = ax_r.get_legend_handles_labels()
    h2, l2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(h1 + h2, l1 + l2, loc='best', fontsize=9)


def main():
    if not os.path.exists(JSON_PATH):
        print(f'缺少结果文件: {JSON_PATH}')
        return

    with open(JSON_PATH, 'r') as f:
        all_res = json.load(f)

    fig, axes = plt.subplots(3, 2, figsize=(13, 15))

    _plot_one_row(
        axes[0, 0], axes[0, 1],
        all_res['theta_l'],
        r'$\tau_l$ (load overload threshold)',
        r'$\tau_l$'
    )

    _plot_one_row(
        axes[1, 0], axes[1, 1],
        all_res['beta'],
        r'$u_d$ (damping coefficient)',
        r'$u_d$'
    )

    _plot_one_row(
        axes[2, 0], axes[2, 1],
        all_res['s0'],
        r'$S_{base}$ (base step size)',
        r'$S_{base}$'
    )

    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, 'fig4_combined_sensitivity.png')
    out_pdf = os.path.join(OUT_DIR, 'fig4_combined_sensitivity.pdf')
    plt.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=220, bbox_inches='tight')
    plt.close(fig)

    print(f'已生成: {out_png}')
    print(f'已生成: {out_pdf}')


if __name__ == '__main__':
    main()
