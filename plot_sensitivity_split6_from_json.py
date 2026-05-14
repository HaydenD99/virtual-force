"""
从 sensitivity_keyparams.json 直接绘制 6 张拆分图
=================================================
- 不重跑实验
- 每个参数两张图：Static / Dynamic
- 与当前合并图保持一致的纵轴范围（更平滑）
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


def _plot_static(block, x_label, title_tag, out_tag):
    xs, s_js, s_mr, s_jfi, _, _, _ = _get_series(block)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    ax.plot(xs, s_js, marker='o', color='#ee6f63', label='JS', linewidth=2)
    ax.plot(xs, s_jfi, marker='^', color='#63cfa0', label='JFI', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('JS / JFI')
    ax.set_title(f'Static ({title_tag})', fontweight='bold')
    ax.set_ylim(0.60, 1.00)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(xs, s_mr, marker='s', color='#5fa8e8', label='Min-Rate', linewidth=2)
    ax2.set_ylabel('Min-Rate (Mbps)')
    ax2.set_ylim(35, 65)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUT_DIR, f'{out_tag}.{ext}'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_dynamic(block, x_label, title_tag, out_tag):
    xs, _, _, _, d_js, d_mr, d_jfi = _get_series(block)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    ax.plot(xs, d_js, marker='o', color='#ee6f63', label='Avg JS', linewidth=2)
    ax.plot(xs, d_jfi, marker='^', color='#63cfa0', label='Avg JFI', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Avg JS / Avg JFI')
    ax.set_title(f'Dynamic ({title_tag})', fontweight='bold')
    ax.set_ylim(0.60, 1.00)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(xs, d_mr, marker='s', color='#5fa8e8', label='Avg Min-Rate', linewidth=2)
    ax2.set_ylabel('Avg Min-Rate (Mbps)')
    ax2.set_ylim(35, 65)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUT_DIR, f'{out_tag}.{ext}'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    if not os.path.exists(JSON_PATH):
        print(f'缺少结果文件: {JSON_PATH}')
        return

    with open(JSON_PATH, 'r') as f:
        all_res = json.load(f)

    # theta_l
    _plot_static(all_res['theta_l'], r'$\tau_l$ (load overload threshold)', r'$\tau_l$', 'fig4x_theta_static')
    _plot_dynamic(all_res['theta_l'], r'$\tau_l$ (load overload threshold)', r'$\tau_l$', 'fig4x_theta_dynamic')

    # beta -> u_d
    _plot_static(all_res['beta'], r'$u_d$ (damping coefficient)', r'$u_d$', 'fig4y_ud_static')
    _plot_dynamic(all_res['beta'], r'$u_d$ (damping coefficient)', r'$u_d$', 'fig4y_ud_dynamic')

    # s0 -> S_base
    _plot_static(all_res['s0'], r'$S_{base}$ (base step size)', r'$S_{base}$', 'fig4z_sbase_static')
    _plot_dynamic(all_res['s0'], r'$S_{base}$ (base step size)', r'$S_{base}$', 'fig4z_sbase_dynamic')

    print(f'已生成 6 张拆分图到: {OUT_DIR}/')


if __name__ == '__main__':
    main()
