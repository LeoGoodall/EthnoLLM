import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import numpy as np
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec
from matplotlib import transforms

# Set style parameters for Nature-style figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# ---------------- Load results for each model ----------------
all_data = []
model_names = []

for model in ["gptoss20b", "gptoss120b", "deepseekv31671b"]:
    # Include the new combined condition
    for condition in ["single", "ensemble", "mtp", "ensemble_mtp"]:
        model_name = f'{model}{f"_{condition}" if condition != "single" else ""}'
        results_file = f"comparison/agreement_analysis_{model_name}.csv"
        if not os.path.exists(results_file):
            print(f"Results file {results_file} not found")
            continue
        df = pd.read_csv(results_file)
        df['Model'] = model_name
        all_data.append(df)
        model_names.append(model_name)

if not all_data:
    raise SystemExit("No comparison/agreement_analysis_*.csv files found.")

# Combine all dataframes
all_results = pd.concat(all_data, ignore_index=True)
os.makedirs('comparison', exist_ok=True)
all_results.to_csv('comparison/model_cross_comparison.csv', index=False)

# ---------------- Ordering & pretty labels ----------------
base_order = ["gptoss20b", "gptoss120b", "deepseekv31671b"]
cond_order = ["single", "ensemble", "mtp", "ensemble_mtp"]

cond_label_map = {
    "single": "Baseline",
    "ensemble": "Ensemble",
    "mtp": "MTP",
    "ensemble_mtp": "MTP+Ensemble",
}

model_display = {
    'gptoss20b': 'GPT-oss 20B',
    'gptoss120b': 'GPT-oss 120B',
    'deepseekv31671b': 'DeepSeek V3.1 671B',
}

def split_base_cond(model_key: str):
    """Return (base, cond) from keys like 'gptoss20b_ensemble', 'gptoss20b' -> ('gptoss20b','single')."""
    parts = model_key.split("_", 1)
    if len(parts) == 1:
        return parts[0], "single"
    base, cond = parts[0], parts[1]
    return base, cond

# Build an ordered list of models grouped by base and following cond_order,
# but only include models that actually exist in all_results.
present_models = sorted(all_results['Model'].unique().tolist(),
                        key=lambda k: (base_order.index(split_base_cond(k)[0]) if split_base_cond(k)[0] in base_order else 999,
                                       cond_order.index(split_base_cond(k)[1]) if split_base_cond(k)[1] in cond_order else 999))
# Rebuild per-base grouped order
ordered_models = []
group_spans = []  # list of (start_idx, end_idx, base)
for base in base_order:
    start_idx = len(ordered_models)
    for cond in cond_order:
        key = f"{base}_{cond}" if cond != "single" else base
        if key in present_models:
            ordered_models.append(key)
    end_idx = len(ordered_models) - 1
    if end_idx >= start_idx:
        group_spans.append((start_idx, end_idx, base))

# ---------------- Model-level metrics ----------------
model_metrics = (all_results.groupby('Model', as_index=False)
                 .agg(Agreement=('Agreement','mean'),
                      Balanced_Accuracy=('Balanced_Accuracy','mean'),
                      Sensitivity=('Sensitivity','mean'),
                      Specificity=('Specificity','mean')))

model_metrics['Model'] = pd.Categorical(model_metrics['Model'], categories=ordered_models, ordered=True)
model_metrics = model_metrics.sort_values('Model')

metrics_melted = pd.melt(model_metrics, id_vars=['Model'], var_name='Metric', value_name='Score')

# ---------------- Load per-sample correlation data ----------------
# Length–Correctness (per model/condition)
length_corr_rows = []
for model_name in ordered_models:
    length_file = f"comparison/sample_level_length_{model_name}.csv"
    if not os.path.exists(length_file):
        continue
    sdf = pd.read_csv(length_file)
    if {'Text_Length', 'Correct'}.issubset(sdf.columns):
        sdf = sdf[['Text_Length', 'Correct']].dropna()
        if len(sdf) > 1 and sdf['Text_Length'].var() > 0:
            r, p = pearsonr(sdf['Text_Length'].values, sdf['Correct'].values)
            n = len(sdf)
            length_corr_rows.append({'Model': model_name, 'Correlation': r, 'P': p, 'N': n})
length_corr_df = pd.DataFrame(length_corr_rows)

# Certainty–Correctness (covers ensemble and ensemble_mtp)
certainty_corr_rows = []
for model_name in ordered_models:
    if "ensemble" not in model_name:
        continue
    cert_file = f"comparison/sample_level_certainty_{model_name}.csv"
    if not os.path.exists(cert_file):
        # fallback for older naming (without condition)
        base = split_base_cond(model_name)[0]
        alt = f"comparison/sample_level_certainty_{base}.csv"
        cert_file = alt if os.path.exists(alt) else None
    if cert_file is None or not os.path.exists(cert_file):
        continue
    cdf = pd.read_csv(cert_file)
    if {'Certainty', 'Correct'}.issubset(cdf.columns):
        cdf = cdf[['Certainty', 'Correct']].dropna()
        if len(cdf) > 1 and cdf['Certainty'].var() > 0:
            r, p = pearsonr(cdf['Certainty'].values, cdf['Correct'].values)
            n = len(cdf)
            certainty_corr_rows.append({'Model': model_name, 'Correlation': r, 'P': p, 'N': n})
certainty_corr_df = pd.DataFrame(certainty_corr_rows)


fig = plt.figure(figsize=(15, 25), dpi=300)
# Taller top panel, then heatmap, then two correlations side-by-side
gs = GridSpec(nrows=3, ncols=2, figure=fig, height_ratios=[9.5, 5.0, 5.0], hspace=0.28, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])  # tall top row
ax2 = fig.add_subplot(gs[1, :])  # full-width middle
ax3 = fig.add_subplot(gs[2, 0])  # bottom-left
ax4 = fig.add_subplot(gs[2, 1])  # bottom-right

# ---------------- Color palettes (4-step gradient: base < ensemble < mtp < ensemble_mtp) ----------------
# Blue gradient
base_blue        = '#C6D6EB'  # light
ensemble_blue    = '#9AB3D5'  # medium
mtp_blue         = '#4C72B0'  # dark
ensemble_mtp_blue= '#2F4A8A'  # darker
# Green gradient
base_green        = '#CAE6D0'
ensemble_green    = '#A8D3B3'
mtp_green         = '#55A868'
ensemble_mtp_green= '#2E7A4A'
# Red gradient
base_red          = '#F1B8BA'
ensemble_red      = '#E19699'
mtp_red           = '#C44E52'
ensemble_mtp_red  = '#8E2F34'

base_colors = {
    'gptoss20b': base_blue,
    'gptoss120b': base_green,
    'deepseekv31671b': base_red
}
ensemble_colors = {
    'gptoss20b_ensemble': ensemble_blue,
    'gptoss120b_ensemble': ensemble_green,
    'deepseekv31671b_ensemble': ensemble_red
}
mtp_colors = {
    'gptoss20b_mtp': mtp_blue,
    'gptoss120b_mtp': mtp_green,
    'deepseekv31671b_mtp': mtp_red
}
ensemble_mtp_colors = {
    'gptoss20b_ensemble_mtp': ensemble_mtp_blue,
    'gptoss120b_ensemble_mtp': ensemble_mtp_green,
    'deepseekv31671b_ensemble_mtp': ensemble_mtp_red
}

palette = {**base_colors, **ensemble_colors, **mtp_colors, **ensemble_mtp_colors}

# --- ax1: model-level metrics ---
sns.barplot(
    data=metrics_melted.replace({'Balanced_Accuracy': 'Balanced Accuracy'}),
    x='Metric', y='Score', hue='Model',
    hue_order=ordered_models, palette=palette, ax=ax1
)

# Remove seaborn's default legend and add our custom, per-model legends inside the plot
if ax1.legend_:
    ax1.legend_.remove()

def add_model_legend(ax, base_key, x_anchor):
    keys = [base_key, f"{base_key}_ensemble", f"{base_key}_mtp", f"{base_key}_ensemble_mtp"]
    labels = ["Baseline", "Ensemble", "MTP", "MTP+Ensemble"]
    handles = [Patch(facecolor=palette[k], edgecolor='none') for k in keys if k in palette]
    leg = ax.legend(
        handles, labels[:len(handles)],
        title=model_display[base_key],
        loc='upper left',
        bbox_to_anchor=(x_anchor, 0.98),  # place INSIDE axes to avoid top gap
        frameon=False, fontsize=8, title_fontsize=9,
        ncol=1, handlelength=1.6, handletextpad=0.6, borderaxespad=0.0
    )
    ax.add_artist(leg)

# place three legends along the top inside the plot area
add_model_legend(ax1, 'gptoss20b',       x_anchor=0.01)
add_model_legend(ax1, 'gptoss120b',      x_anchor=0.32)
add_model_legend(ax1, 'deepseekv31671b', x_anchor=0.63)

ax1.set_ylim(0, 1)
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison', pad=6, fontsize=10, fontweight='bold')

# --- ax2: feature heatmap ---
# Use ordered model list for columns
feature_metrics = (all_results.pivot_table(values='Balanced_Accuracy', index='Feature', columns='Model', aggfunc='mean')
                   .reindex(columns=ordered_models))

sns.heatmap(feature_metrics, cmap='YlOrRd', annot=True, fmt='.2f', vmax=1.0, ax=ax2)
for start, end, base in group_spans[:-1]:  # skip last group
    ax2.axvline(end + 1, color='black', linewidth=2)
ax2.set_title('Feature-level Balanced Accuracy by Model', pad=6, fontsize=10, fontweight='bold')
ax2.set_xlabel('')

# Replace raw tick labels with condition labels, and add centered bold model names under groups
def relabel_axis_with_groups(ax, ordered_keys, group_spans, rotate=0, tick_offset=0.0, group_label_offset=0.0):
    # map each key -> condition label
    pretty = []
    for k in ordered_keys:
        base, cond = split_base_cond(k)
        pretty.append(cond_label_map.get(cond, cond))
    positions = np.arange(len(ordered_keys)) + tick_offset
    # Choose alignment based on rotation
    ha = 'center' if rotate == 0 else 'right'
    ax.set_xticks(positions)
    ax.set_xticklabels(pretty, rotation=rotate, ha=ha)
    # add centered model names below tick labels
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for start, end, base in group_spans:
        center = (start + end) / 2.0 + tick_offset
        ax.text(center, group_label_offset, model_display[base], transform=trans,
                ha='center', va='top', fontsize=9, fontweight='bold')

# Heatmap uses cell centers at x = 0.5, 1.5, ..., so offset ticks and group labels by +0.5
relabel_axis_with_groups(ax2, ordered_models, group_spans, rotate=0, tick_offset=0.5, group_label_offset=-0.07)

# helper for significance labels
def sig_label(p):
    if pd.isna(p):
        return 'n/a'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'

# --- ax3: Length–Correctness correlation ---
if not length_corr_df.empty:
    length_corr_df['Model'] = pd.Categorical(length_corr_df['Model'], categories=ordered_models, ordered=True)
    length_corr_df = length_corr_df.sort_values('Model')
    g3 = sns.barplot(data=length_corr_df, x='Model', y='Correlation',
                     palette=palette, order=ordered_models, ax=ax3)
    ax3.axhline(0, linewidth=0.8)
    ax3.set_ylim(-1, 1)
    ax3.set_xlabel('')
    ax3.set_ylabel('r (Correctness vs Text Length)')
    ax3.set_title('Correctness vs Text Length (Pearson r, p, N)', pad=6, fontsize=10, fontweight='bold')
    # annotate r and p above bars
    for patch, (_, row) in zip(g3.patches, length_corr_df.iterrows()):
        x = patch.get_x() + patch.get_width()/2
        y = patch.get_height()
        txt = f"{row['Correlation']:.2f}\n{sig_label(row.get('P'))}\nN={int(row.get('N', 0))}"
        va = 'bottom' if y >= 0 else 'top'
        y_off = 0.02 if y >= 0 else -0.02
        ax3.text(x, y + y_off, txt, ha='center', va=va, fontsize=7)
    # rotate condition labels to avoid crowding
    relabel_axis_with_groups(ax3, ordered_models, group_spans, rotate=90, tick_offset=0.0, group_label_offset=-0.22)
else:
    ax3.set_visible(False)

# --- ax4: Certainty–Correctness correlation (ensemble & ensemble_mtp) ---
if not certainty_corr_df.empty:
    present_models = [m for m in certainty_corr_df['Model'].unique().tolist() if m in ordered_models]
    order = [m for m in ordered_models if m in present_models]
    certainty_corr_df['Model'] = pd.Categorical(certainty_corr_df['Model'], categories=order, ordered=True)
    certainty_corr_df = certainty_corr_df.sort_values('Model')

    g4 = sns.barplot(data=certainty_corr_df, x='Model', y='Correlation',
                     palette=palette, order=order, ax=ax4)
    ax4.axhline(0, linewidth=0.8)
    ax4.set_ylim(-1, 1)
    ax4.set_xlabel('')
    ax4.set_ylabel('r (Correctness vs Certainty)')
    ax4.set_title('Correctness vs Certainty (Pearson r, p, N)', pad=6, fontsize=10, fontweight='bold')
    for patch, (_, row) in zip(g4.patches, certainty_corr_df.iterrows()):
        x = patch.get_x() + patch.get_width()/2
        y = patch.get_height()
        txt = f"{row['Correlation']:.2f}\n{sig_label(row.get('P'))}\nN={int(row.get('N', 0))}"
        va = 'bottom' if y >= 0 else 'top'
        y_off = 0.02 if y >= 0 else -0.02
        ax4.text(x, y + y_off, txt, ha='center', va=va, fontsize=7)
    # replace x tick labels and add grouped base labels (only for those present in this subplot)
    # Build spans for 'order' subset
    subset_spans = []
    idx_map = {m:i for i,m in enumerate(order)}
    for start, end, base in group_spans:
        # find indices in this subset
        ids = [idx_map[m] for m in order if split_base_cond(m)[0] == base]
        if ids:
            subset_spans.append((min(ids), max(ids), base))
    relabel_axis_with_groups(ax4, order, subset_spans, rotate=90, tick_offset=0.0, group_label_offset=-0.22)
else:
    ax4.set_visible(False)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/model_cross_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()