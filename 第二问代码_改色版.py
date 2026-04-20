import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE_DIR = '/mnt/data'
OUT_DIR = BASE_DIR


def pick_cjk_font() -> None:
    candidates = [
        'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC',
        'Source Han Sans SC', 'Source Han Sans CN', 'SimHei',
        'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 160
    plt.rcParams['savefig.dpi'] = 240
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


pick_cjk_font()


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def load_attack_data() -> pd.DataFrame:
    attack = pd.read_csv(os.path.join(BASE_DIR, 'problem2_attack_dataset.csv'))
    return attack


def load_defense_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE_DIR, 'problem2_defense_dataset.csv'))


def load_match_matrix(attack: pd.DataFrame, defense: pd.DataFrame) -> pd.DataFrame:
    matrix_df = pd.read_csv(os.path.join(BASE_DIR, 'problem2_match_matrix.csv'))
    matrix_df = matrix_df.set_index('动作')
    matrix_df = matrix_df.reindex(index=attack['动作'], columns=defense['防守动作']).fillna(0.0)
    return matrix_df


def compute_scores(attack: pd.DataFrame, defense: pd.DataFrame, match_matrix: pd.DataFrame) -> pd.DataFrame:
    alpha_by_proto = {
        1: (0.35, 0.30, 0.35),
        2: (0.40, 0.35, 0.25),
        3: (0.25, 0.55, 0.20),
        4: (0.40, 0.40, 0.20),
        5: (0.45, 0.20, 0.35),
        6: (0.30, 0.35, 0.35),
    }

    tmax = 145.0
    records = []
    for _, atk in attack.iterrows():
        a1, a2, a3 = alpha_by_proto[int(atk['原型编号'])]
        for _, d in defense.iterrows():
            raw_margin = d['n_j'] * tmax / (atk['冲量_J'] * d['lever_arm_l'])
            t_res = float(np.clip(raw_margin / 20.0, 0, 1))
            b_eff = 0.5 * d['b_base'] + 0.5 * (0.6 * d['block_speed_hat'] + 0.4 * t_res)
            d_ij = 0.6 * b_eff + 0.4 * d['z_j']
            p_ij = match_matrix.loc[atk['动作'], d['防守动作']] * (a1 * d_ij + a2 * d['u_j'] + a3 * d['T_j'])
            q_ij = d['R_j'] * atk['扰动风险_Ri']
            c_ij = d['k_j'] * atk['反制窗口_Wi']
            total = 0.5 * p_ij + 0.3 * q_ij + 0.2 * c_ij
            records.append({
                '攻击动作': atk['动作'],
                '攻击原型': atk['攻击原型'],
                '防守动作': d['防守动作'],
                '防守类别': d['类别'],
                '匹配矩阵_Mij': match_matrix.loc[atk['动作'], d['防守动作']],
                '抗冲击余量_tres': t_res,
                '直接防护_Dij': d_ij,
                '主防匹配_Pij': p_ij,
                '稳定保障_Qij': q_ij,
                '反制机会_Cij': c_ij,
                '总评分_Dij': total,
            })
    return pd.DataFrame(records)


def compute_recommendations(attack: pd.DataFrame, defense: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    aux_candidates = ['重心补偿', '卸力缓冲', '步点调整']
    ground_candidates = ['受控倒地', '快速起身', '倒地防御']

    aux_pref_shock = {'重心补偿': 0.92, '卸力缓冲': 1.00, '步点调整': 0.82}
    aux_pref_leg = {'重心补偿': 0.65, '卸力缓冲': 0.55, '步点调整': 1.00}
    aux_synergy = {
        '格挡': {'重心补偿': 0.88, '卸力缓冲': 1.00, '步点调整': 0.78},
        '闪避': {'重心补偿': 0.76, '卸力缓冲': 0.68, '步点调整': 1.00},
        '姿态': {'重心补偿': 0.94, '卸力缓冲': 0.90, '步点调整': 0.82},
        '平衡': {'重心补偿': 0.92, '卸力缓冲': 0.92, '步点调整': 0.92},
        '倒地': {'重心补偿': 0.60, '卸力缓冲': 0.65, '步点调整': 0.70},
        '组合': {'重心补偿': 0.82, '卸力缓冲': 0.78, '步点调整': 0.96},
    }
    ground_pref_special = {'受控倒地': 0.82, '快速起身': 0.88, '倒地防御': 0.86}
    ground_pref_spin = {'受控倒地': 1.00, '快速起身': 0.72, '倒地防御': 0.74}
    ground_pref_close = {'受控倒地': 0.70, '快速起身': 0.72, '倒地防御': 1.00}

    recommend_rows = []
    for _, atk in attack.iterrows():
        sub = scores[scores['攻击动作'] == atk['动作']].sort_values('总评分_Dij', ascending=False).reset_index(drop=True)
        best = sub.iloc[0]
        primary = best['防守动作']
        primary_cat = best['防守类别']

        aux_scores = {}
        for aux in aux_candidates:
            base_r = defense.loc[defense['防守动作'] == aux, 'R_j'].iloc[0]
            val = (
                0.40 * base_r * atk['扰动风险_Ri']
                + 0.25 * aux_synergy[primary_cat][aux]
                + 0.20 * atk['攻击强度_Gi'] * aux_pref_shock[aux]
                + 0.15 * atk['旋转扰动_Yi'] * aux_pref_leg[aux]
            )
            if atk['动作'] == '冲撞' and aux in ['重心补偿', '卸力缓冲']:
                val += 0.08
            if int(atk['原型编号']) in [3, 4, 6] and aux == '步点调整':
                val += 0.05
            aux_scores[aux] = val
        auxiliary = max(aux_scores, key=aux_scores.get)

        ground_scores = {}
        for g in ground_candidates:
            r_val = defense.loc[defense['防守动作'] == g, 'r_j'].iloc[0]
            if int(atk['原型编号']) == 6:
                pref = ground_pref_special[g]
            elif int(atk['原型编号']) in [3, 4] or atk['扰动风险_Ri'] >= 0.7 or atk['攻击强度_Gi'] >= 0.7:
                pref = ground_pref_spin[g]
            elif int(atk['原型编号']) == 5:
                pref = ground_pref_close[g]
            else:
                pref = ground_pref_special[g]
            val = 0.55 * r_val + 0.25 * pref + 0.20 * atk['扰动风险_Ri']
            if atk['动作'] == '倒地反击' and g == '倒地防御':
                val += 0.10
            if atk['动作'] == '冲撞' and g == '受控倒地':
                val += 0.08
            ground_scores[g] = val
        fallback = max(ground_scores, key=ground_scores.get)

        top3 = sub.head(3).copy()
        top3_txt = ' / '.join((top3['防守动作'] + '(' + top3['总评分_Dij'].round(3).astype(str) + ')').tolist())
        recommend_rows.append({
            '攻击动作': atk['动作'],
            '攻击原型': atk['攻击原型'],
            '主防动作': primary,
            '主防类别': primary_cat,
            '辅助动作': auxiliary,
            '兜底动作': fallback,
            '闭环方案': f'{primary} → {auxiliary} → {fallback}',
            'Top3评分': top3_txt,
            '最优总分': round(float(best['总评分_Dij']), 4),
            '扰动风险_Ri': round(float(atk['扰动风险_Ri']), 4),
            '反制窗口_Wi': round(float(atk['反制窗口_Wi']), 4),
            '攻击强度_Gi': round(float(atk['攻击强度_Gi']), 4),
        })
    return pd.DataFrame(recommend_rows)


def save_tables(attack: pd.DataFrame, defense: pd.DataFrame, match_matrix: pd.DataFrame,
                scores: pd.DataFrame, recommend: pd.DataFrame) -> None:
    scores.to_csv(os.path.join(OUT_DIR, 'problem2_defense_scores_long_revised_v2.csv'), index=False, encoding='utf-8-sig')
    recommend.to_csv(os.path.join(OUT_DIR, 'problem2_recommendations_revised_v2.csv'), index=False, encoding='utf-8-sig')
    top3 = scores.sort_values(['攻击动作', '总评分_Dij'], ascending=[True, False]).groupby('攻击动作').head(3)
    top3[['攻击动作', '防守动作', '防守类别', '总评分_Dij']].assign(
        总评分_Dij=lambda df: df['总评分_Dij'].round(4)
    ).to_csv(os.path.join(OUT_DIR, 'problem2_top3_per_attack_revised_v2.csv'), index=False, encoding='utf-8-sig')


def style_axes(ax) -> None:
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)


def relative_luminance(hex_color: str) -> float:
    r, g, b = to_rgb(hex_color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_blue_gradient(n: int) -> List[str]:
    # 根据用户给出的色号修正明显录入错误：O -> 0
    supplied = [
        '#60B0F4', '#7EBCF5', '#98C9F1', '#B4DCEC', '#D0ECE9',
        '#5A97D0', '#8CC6ED', '#C0E0F8', '#D0E8FF', '#EDF7FF',
        '#2F74B8', '#79B0D7', '#A0C8E8', '#C6E0F2', '#6DA6D4',
    ]
    ordered = sorted(supplied, key=relative_luminance)
    cmap = LinearSegmentedColormap.from_list('custom_blue_gradient', ordered)
    colors = [cmap(v) for v in np.linspace(0, 1, n)]
    return [matplotlib.colors.to_hex(c) for c in colors]


def build_heatmap_cmap() -> LinearSegmentedColormap:
    # 将用户色号中的 #lccde 解释为更符合序列位置的 #F1CCDE
    heatmap_colors = [
        '#28437d', '#4675b0', '#6892c5', '#a6badf', '#e7e4f8',
        '#f4dced', '#f1ccde', '#eab7ca', '#d2798c', '#b34557',
    ]
    return LinearSegmentedColormap.from_list('attack_defense_heatmap', heatmap_colors, N=256)


def quadrant_tag(x: float, y: float) -> str:
    if x < 0.5 and y >= 0.5:
        return 'UL'
    if x >= 0.5 and y >= 0.5:
        return 'UR'
    if x < 0.5 and y < 0.5:
        return 'LL'
    return 'LR'


def plot_score_heatmap(scores: pd.DataFrame) -> None:
    pivot_scores = scores.pivot(index='攻击动作', columns='防守动作', values='总评分_Dij')
    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(pivot_scores.values, aspect='auto', cmap=build_heatmap_cmap())
    ax.set_xticks(range(len(pivot_scores.columns)))
    ax.set_xticklabels(pivot_scores.columns, rotation=75, ha='right')
    ax.set_yticks(range(len(pivot_scores.index)))
    ax.set_yticklabels(pivot_scores.index)
    ax.set_title('问题2：攻击-防守总评分热力图')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'problem2_score_heatmap_recolored.png'), bbox_inches='tight')
    plt.close(fig)


def plot_best_score_bar(recommend: pd.DataFrame) -> None:
    bar_df = recommend.sort_values('最优总分', ascending=False).reset_index(drop=True)
    x = np.arange(len(bar_df))
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    bars = ax.bar(x, bar_df['最优总分'], width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_df['攻击动作'], rotation=45, ha='right')
    ax.set_ylabel('最优总分')
    ax.set_title('各攻击动作的最优主防得分')
    style_axes(ax)
    ymax = bar_df['最优总分'].max()
    for rect, row in zip(bars, bar_df.itertuples(index=False)):
        h = rect.get_height()
        cx = rect.get_x() + rect.get_width() / 2
        ax.text(
            cx,
            h * 0.55,
            f'{row.最优总分:.3f}',
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            rotation=0,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='black', alpha=0.22, edgecolor='none'),
        )
        ax.text(
            cx,
            h + ymax * 0.015,
            row.主防动作,
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=0,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'problem2_best_score_bar_revised_v2.png'), bbox_inches='tight')
    plt.close(fig)


def plot_attack_risk_scatter(attack: pd.DataFrame, recommend: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    sizes = 320 * (0.40 + attack['攻击强度_Gi'])

    quadrant_colors = {
        'UR': '#00008B',  # 右上：高风险 + 高空档
        'LR': '#FF69B4',  # 右下：高风险 + 低空档
        'LL': '#90EE90',  # 左下：低风险 + 低空档
    }
    point_colors = [quadrant_colors.get(quadrant_tag(x, y), '#00008B') for x, y in zip(attack['扰动风险_Ri'], attack['反制窗口_Wi'])]

    ax.scatter(
        attack['扰动风险_Ri'],
        attack['反制窗口_Wi'],
        s=sizes,
        c=point_colors,
        alpha=0.88,
        edgecolors='white',
        linewidths=0.9,
    )
    for _, row in recommend.iterrows():
        x = float(attack.loc[attack['动作'] == row['攻击动作'], '扰动风险_Ri'].iloc[0])
        y = float(attack.loc[attack['动作'] == row['攻击动作'], '反制窗口_Wi'].iloc[0])
        ax.annotate(f"{row['攻击动作']}\n{row['主防动作']}", (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.axvline(0.5, linestyle='--', linewidth=1.0)
    ax.axhline(0.5, linestyle='--', linewidth=1.0)
    ax.text(0.76, 0.93, '右上：高风险 + 高空档（绕）', transform=ax.transAxes, fontsize=8, ha='center', va='center')
    ax.text(0.24, 0.93, '左上：低风险 + 高空档（反）', transform=ax.transAxes, fontsize=8, ha='center', va='center')
    ax.text(0.24, 0.08, '左下：低风险 + 低空档（挡）', transform=ax.transAxes, fontsize=8, ha='center', va='center')
    ax.text(0.76, 0.08, '右下：高风险 + 低空档（稳）', transform=ax.transAxes, fontsize=8, ha='center', va='center')
    ax.set_xlabel('扰动风险 $R_i$')
    ax.set_ylabel('反制窗口 $W_i$')
    ax.set_title('攻击风险分布与推荐主防动作')
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'problem2_attack_risk_scatter_recolored.png'), bbox_inches='tight')
    plt.close(fig)


def plot_top3_frequency(scores: pd.DataFrame) -> None:
    top3 = scores.sort_values(['攻击动作', '总评分_Dij'], ascending=[True, False]).groupby('攻击动作').head(3)
    freq = top3['防守动作'].value_counts().sort_values(ascending=False)

    labels = freq.index.tolist()
    values = freq.values.astype(int)
    x = np.arange(len(labels))

    unique_vals = sorted(pd.unique(values), reverse=True)

    # 直接选用区分度更大的离散蓝色，而不是连续插值，避免后几档看起来太接近
    stepped_blues = [
        '#2F74B8',  # 最深：最高频
        '#5A97D0',
        '#7EBCF5',
        '#B4DCEC',
        '#EDF7FF',  # 最浅：最低频
    ]
    if len(unique_vals) <= len(stepped_blues):
        chosen = stepped_blues[:len(unique_vals)]
    else:
        chosen = build_blue_gradient(len(unique_vals))
    value_color_map = {val: chosen[i] for i, val in enumerate(unique_vals)}
    bar_colors = [value_color_map[v] for v in values]

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars = ax.bar(x, values, width=0.72, color=bar_colors, edgecolor='none')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.set_ylabel('进入Top3次数')
    ax.set_xlabel('防守动作')
    ax.set_title('防守动作在各攻击中的Top3出现频次')
    style_axes(ax)
    ax.set_ylim(0, values.max() + 1.0)

    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.08,
            str(int(val)),
            ha='center',
            va='bottom',
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'problem2_top3_frequency_vertical_grouped_v2.png'), bbox_inches='tight')
    plt.close(fig)


def plot_best_defense_decomposition(scores: pd.DataFrame, recommend: pd.DataFrame) -> None:
    best_df = []
    for _, rec in recommend.iterrows():
        sub = scores[
            (scores['攻击动作'] == rec['攻击动作'])
            & (scores['防守动作'] == rec['主防动作'])
        ].iloc[0]
        best_df.append({
            '攻击动作': rec['攻击动作'],
            '攻击原型': rec['攻击原型'],
            '主防动作': rec['主防动作'],
            '主防匹配项': 0.5 * sub['主防匹配_Pij'],
            '稳定保障项': 0.3 * sub['稳定保障_Qij'],
            '反制机会项': 0.2 * sub['反制机会_Cij'],
            '最优总分': sub['总评分_Dij'],
        })
    best_df = pd.DataFrame(best_df)

    proto_order = ['直线快攻型', '弧线摆动型', '低位破坏型', '贴身压迫型', '旋转腿击型', '特殊场景型']
    proto_rank = {name: idx for idx, name in enumerate(proto_order)}
    best_df['原型排序'] = best_df['攻击原型'].map(proto_rank)
    best_df = best_df.sort_values(['原型排序', '最优总分'], ascending=[True, False]).reset_index(drop=True)

    bar_colors = {
        '主防匹配项': '#4E79A7',
        '稳定保障项': '#76B7B2',
        '反制机会项': '#F28E2B',
    }
    band_colors = {
        '直线快攻型': '#DCE8F4',
        '弧线摆动型': '#EDE4DB',
        '低位破坏型': '#E2F0E1',
        '贴身压迫型': '#F0E4E8',
        '旋转腿击型': '#E2F1F1',
        '特殊场景型': '#EEE5EE',
    }
    label_colors = {
        '直线快攻型': '#4E79A7',
        '弧线摆动型': '#F28E2B',
        '低位破坏型': '#59A14F',
        '贴身压迫型': '#E15759',
        '旋转腿击型': '#76B7B2',
        '特殊场景型': '#B07AA1',
    }

    fig, ax = plt.subplots(figsize=(13.8, 8.1))
    y = np.arange(len(best_df))
    height = 0.68

    for proto in proto_order:
        sub = best_df[best_df['攻击原型'] == proto]
        if sub.empty:
            continue
        start = int(sub.index.min())
        end = int(sub.index.max())
        ax.axhspan(start - 0.5, end + 0.5, color=band_colors[proto], alpha=0.65, zorder=0)
        center = (start + end) / 2
        ax.text(
            -0.08,
            center,
            proto,
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='center',
            fontsize=11,
            color=label_colors[proto],
            fontweight='bold',
            clip_on=False,
        )
        if end < len(best_df) - 1:
            ax.axhline(end + 0.5, color='#cfcfcf', linewidth=0.8, zorder=1)

    left = np.zeros(len(best_df))
    for col in ['主防匹配项', '稳定保障项', '反制机会项']:
        ax.barh(
            y,
            best_df[col],
            left=left,
            height=height,
            color=bar_colors[col],
            edgecolor='white',
            linewidth=0.8,
            label=col,
            zorder=3,
        )
        left += best_df[col].to_numpy()

    ax.scatter(best_df['最优总分'], y, s=22, color='black', zorder=4)
    for yi, row in best_df.iterrows():
        ax.text(
            row['最优总分'] + 0.010,
            yi,
            f"{row['主防动作']} ({row['最优总分']:.3f})",
            ha='left',
            va='center',
            fontsize=9,
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(best_df['攻击动作'])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.80, float(best_df['最优总分'].max()) + 0.18))
    ax.set_xlabel(r'最优总分分解  ($D^* = 0.5P + 0.3Q + 0.2C$)')
    ax.set_ylabel('')
    ax.set_title('各攻击动作最优主防得分的机制分解图', pad=20)
    ax.grid(axis='x', alpha=0.35, linestyle='--', linewidth=0.7)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.79, 1.14))

    fig.subplots_adjust(left=0.20, right=0.96, top=0.84, bottom=0.10)
    fig.savefig(os.path.join(OUT_DIR, 'problem2_best_defense_decomposition_revised_v2_modified.png'), bbox_inches='tight')
    plt.close(fig)


def plot_score_3d_scatter(scores: pd.DataFrame, topk: int = 3) -> None:
    marker_map = {'格挡': 's', '闪避': 'o', '姿态': '^', '平衡': 'D', '倒地': 'P', '组合': 'X'}
    top_candidates = scores.sort_values(['攻击动作', '总评分_Dij'], ascending=[True, False]).groupby('攻击动作').head(topk).copy()
    best_idx = scores.groupby('攻击动作')['总评分_Dij'].idxmax()
    best_scores = scores.loc[best_idx].copy()

    fig = plt.figure(figsize=(12.0, 9.2))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(top_candidates['总评分_Dij'].min(), top_candidates['总评分_Dij'].max())
    cmap = plt.cm.viridis

    for cat, marker in marker_map.items():
        sub = top_candidates[top_candidates['防守类别'] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub['主防匹配_Pij'], sub['稳定保障_Qij'], sub['反制机会_Cij'], c=sub['总评分_Dij'], cmap=cmap,
                   norm=norm, marker=marker, s=44, alpha=0.28, depthshade=True)

    ax.scatter(best_scores['主防匹配_Pij'], best_scores['稳定保障_Qij'], best_scores['反制机会_Cij'],
               c=best_scores['总评分_Dij'], cmap=cmap, norm=norm, marker='*', s=250,
               edgecolors='black', linewidths=0.9, alpha=0.98, depthshade=False)

    for _, row in best_scores.iterrows():
        ax.text(row['主防匹配_Pij'] + 0.006, row['稳定保障_Qij'] + 0.004, row['反制机会_Cij'] + 0.008,
                f"{row['攻击动作']}\n{row['防守动作']}", fontsize=7)

    ax.set_xlabel('主防匹配得分 (P_{ij})', labelpad=12)
    ax.set_ylabel('稳定保障得分 (Q_{ij})', labelpad=12)
    ax.set_zlabel('反制机会得分 (C_{ij})', labelpad=20)
    ax.text2D(0.02, 0.80, '反制机会得分 (C_{ij})', transform=ax.transAxes, fontsize=10)
    ax.set_title('Top 候选方案图', pad=18)
    ax.view_init(elev=24, azim=38)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label('综合评分 (D_{ij})')

    handles = [
        Line2D([0], [0], marker=marker_map[cat], color='w', label=cat, markerfacecolor='gray', markersize=8, markeredgecolor='gray')
        for cat in marker_map
    ]
    handles.append(Line2D([0], [0], marker='*', color='w', label='最优主防', markerfacecolor='gray', markeredgecolor='black', markersize=12))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.02, 1.06), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'problem2_score_3d_scatter_revised_v2_modified.png'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    attack = load_attack_data()
    defense = load_defense_data()
    match_matrix = load_match_matrix(attack, defense)
    scores = compute_scores(attack, defense, match_matrix)
    recommend = compute_recommendations(attack, defense, scores)
    save_tables(attack, defense, match_matrix, scores, recommend)
    plot_score_heatmap(scores)
    plot_best_score_bar(recommend)
    plot_attack_risk_scatter(attack, recommend)
    plot_top3_frequency(scores)
    plot_best_defense_decomposition(scores, recommend)
    plot_score_3d_scatter(scores, topk=3)
    print('done')


if __name__ == '__main__':
    main()
