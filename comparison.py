import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, balanced_accuracy_score
from utils import models
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

def main(model, condition):

    model_name = models[model]

    # Accept new combined condition explicitly; also keep backward compatibility
    suffix = "" if condition == "single" else f"_{condition}"
    results_file = f"results/results_{model_name}{suffix}.csv"

    if not os.path.exists(results_file):
        # Fallbacks (e.g., user passes "ensemble" but only combined exists)
        if condition == "ensemble" and os.path.exists(f"results/results_{model_name}_ensemble_mtp.csv"):
            results_file = f"results/results_{model_name}_ensemble_mtp.csv"
            condition = "ensemble_mtp"
        elif condition == "mtp" and os.path.exists(f"results/results_{model_name}_ensemble_mtp.csv"):
            results_file = f"results/results_{model_name}_ensemble_mtp.csv"
            condition = "ensemble_mtp"
        else:
            return

    results = pd.read_csv(results_file)
    features = pd.read_csv("data/features.csv")

    # Ritual text lengths (join by ritual_number rather than relying on positional index)
    rituals_df = pd.read_csv("data/rituals_codes.csv")[["ritual_number", "text"]].copy()
    rituals_df["text"] = rituals_df["text"].fillna("").astype(str)
    rituals_df.loc[rituals_df["text"] == "-", "text"] = ""
    rituals_df["Text_Length"] = rituals_df["text"].str.len()

    feature_name_map = dict(zip(features['feature_variable'], features['feature_name']))

    human_cols = [col for col in results.columns if '_human' in col and col != 'ritual_number']
    model_cols = [col.replace('_human', '_llm') for col in human_cols]

    # New: detect certainty presence from columns, not just condition name
    has_certainty = any(col.endswith('_certainty') for col in results.columns)

    agreements = []
    incorrect_lengths = []
    certainty_analysis = []

    # Per-sample records for later visualisation
    length_records = []     # Feature, ritual_number, Text_Length, Correct
    certainty_records = []  # Feature, ritual_number, Certainty, Correct

    for human_col, model_col in zip(human_cols, model_cols):
        if human_col not in results.columns or model_col not in results.columns:
            print(f"Warning: Columns {human_col} and/or {model_col} not found in data")
            continue

        valid_rows = results[[human_col, model_col, 'ritual_number']].replace(r'^\s*$', np.nan, regex=True).dropna()
        valid_rows[human_col] = pd.to_numeric(valid_rows[human_col], errors='coerce')
        valid_rows[model_col] = pd.to_numeric(valid_rows[model_col], errors='coerce')
        valid_rows = valid_rows.dropna()
        valid_rows = valid_rows[
            ((valid_rows[human_col] == 0) | (valid_rows[human_col] == 1)) &
            ((valid_rows[model_col] == 0) | (valid_rows[model_col] == 1))
        ]
        if valid_rows.empty:
            continue

        merged = valid_rows.merge(rituals_df[['ritual_number', 'Text_Length']], on='ritual_number', how='left')
        correctness = (merged[human_col] == merged[model_col]).astype(int)

        length_records.append(pd.DataFrame({
            'Feature': feature_name_map[human_col],
            'ritual_number': merged['ritual_number'],
            'Text_Length': merged['Text_Length'].fillna(0),
            'Correct': correctness
        }))

        incorrect_mask = (correctness == 0)
        for tl in merged.loc[incorrect_mask, 'Text_Length'].tolist():
            incorrect_lengths.append({'Feature': feature_name_map[human_col], 'Text_Length': tl})

        # Certainty analysis if certainty columns exist (covers ensemble and ensemble_mtp)
        if has_certainty:
            certainty_col = f"{model_col}_certainty"
            if certainty_col in results.columns:
                cert_series = results.loc[merged.index, certainty_col].astype(float)
                certainty_scores = results[certainty_col].dropna().astype(float)
                correct_certainty = cert_series[correctness == 1].dropna()
                incorrect_certainty = cert_series[correctness == 0].dropna()
                corr_val = np.corrcoef(correctness, cert_series.fillna(cert_series.mean()))[0, 1] if len(cert_series) > 1 else np.nan
                certainty_analysis.append({
                    'Feature': feature_name_map[human_col],
                    'Mean_Certainty': certainty_scores.mean(),
                    'Std_Certainty': certainty_scores.std(),
                    'Mean_Correct_Certainty': correct_certainty.mean() if not correct_certainty.empty else np.nan,
                    'Mean_Incorrect_Certainty': incorrect_certainty.mean() if not incorrect_certainty.empty else np.nan,
                    'Correlation': corr_val
                })
                certainty_records.append(pd.DataFrame({
                    'Feature': feature_name_map[human_col],
                    'ritual_number': merged['ritual_number'],
                    'Certainty': cert_series.values,
                    'Correct': correctness.values
                }))

        # Agreement stats
        agreement = correctness.mean()
        kappa = cohen_kappa_score(merged[human_col], merged[model_col])
        balanced_acc = balanced_accuracy_score(merged[human_col], merged[model_col])
        cm = confusion_matrix(merged[human_col], merged[model_col])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        class_dist = merged[human_col].value_counts(normalize=True)

        agreements.append({
            'Feature': feature_name_map[human_col],
            'Agreement': agreement,
            'Kappa': kappa,
            'Balanced_Accuracy': balanced_acc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Prop_Present': class_dist.get(1, 0),
            'N': len(merged)
        })

    agreement_df = pd.DataFrame(agreements)

    # Combined metrics frame for plotting (unchanged below this point)
    metrics_df = pd.DataFrame({
        'Feature': agreement_df['Feature'],
        'Raw Agreement': agreement_df['Agreement'],
        'Balanced Accuracy': agreement_df['Balanced_Accuracy'],
        'Sensitivity': agreement_df['Sensitivity'],
        'Specificity': agreement_df['Specificity']
    })
    metrics_melted = pd.melt(metrics_df, id_vars=['Feature'], var_name='Metric', value_name='Value')

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    # Treat combined condition like ensemble for plotting extra panel if you have that path elsewhere
    if has_certainty:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 16), dpi=300)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), dpi=300)

    sns.barplot(data=metrics_melted, x='Value', y='Feature', hue='Metric', palette=colors, ax=ax1)
    ax1.set_xlim(0, 1); ax1.set_xlabel('Score'); ax1.set_ylabel('')
    ax1.set_title('Agreement Metrics by Feature', pad=10, fontsize=10, fontweight='bold')

    sns.barplot(data=agreement_df, x='Prop_Present', y='Feature', color='#4C72B0', ax=ax2)
    ax2.set_xlim(0, 1); ax2.set_xlabel('Proportion of Present (1) Cases'); ax2.set_ylabel('')
    ax2.set_title('Class Distribution in Ground Truth', pad=10, fontsize=10, fontweight='bold')

    incorrect_df = pd.DataFrame(incorrect_lengths)
    sns.boxplot(data=incorrect_df, x='Text_Length', y='Feature', color='#4C72B0', ax=ax3)
    ax3.set_xlabel('Text Length (characters)'); ax3.set_ylabel('')
    ax3.set_title('Distribution of Text Lengths for Incorrect Annotations', pad=10, fontsize=10, fontweight='bold')

    if has_certainty:
        certainty_summary_df = pd.DataFrame(certainty_analysis)
        certainty_melted = pd.melt(certainty_summary_df, id_vars=['Feature'],
                                   value_vars=['Mean_Correct_Certainty', 'Mean_Incorrect_Certainty'],
                                   var_name='Prediction_Type', value_name='Certainty')
        sns.barplot(data=certainty_melted, x='Certainty', y='Feature',
                    hue='Prediction_Type', palette=['#55A868', '#C44E52'], ax=ax4)
        ax4.set_xlim(0, 1); ax4.set_xlabel('Mean Certainty Score'); ax4.set_ylabel('')
        ax4.set_title('Certainty Scores by Prediction Correctness', pad=10, fontsize=10, fontweight='bold')

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'figures/agreement_analysis_{model_name}{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Console summaries
    print("\nAgreement Analysis Summary:")
    print("===========================")
    print(f"Average raw agreement across features: {agreement_df['Agreement'].mean():.3f}")
    print(f"Average balanced accuracy: {agreement_df['Balanced_Accuracy'].mean():.3f}")
    print(f"Average sensitivity: {agreement_df['Sensitivity'].mean():.3f}")
    print(f"Average specificity: {agreement_df['Specificity'].mean():.3f}")
    print(f"Average proportion of present cases: {agreement_df['Prop_Present'].mean():.3f}")

    # Save detailed results
    os.makedirs("comparison", exist_ok=True)
    agreement_df.to_csv(f"comparison/agreement_analysis_{model_name}{suffix}.csv", index=False)

    # Save per-sample datasets for later visualisation across models
    length_df = pd.concat(length_records, ignore_index=True) if length_records else pd.DataFrame(columns=['Feature','ritual_number','Text_Length','Correct'])
    length_df.to_csv(f"comparison/sample_level_length_{model_name}{suffix}.csv", index=False)
    length_corr_by_feature = (
        length_df.groupby('Feature')
        .apply(lambda g: g[['Text_Length','Correct']].corr().iloc[0,1] if len(g) > 1 else np.nan)
        .rename('Length_Correct_Corr')
        .reset_index()
    ) if not length_df.empty else pd.DataFrame(columns=['Feature','Length_Correct_Corr'])
    length_corr_by_feature.to_csv(f"comparison/length_corr_by_feature_{model_name}{suffix}.csv", index=False)

    if has_certainty and certainty_records:
        certainty_df_samples = pd.concat(certainty_records, ignore_index=True)
        certainty_df_samples.to_csv(f"comparison/sample_level_certainty_{model_name}{suffix}.csv", index=False)
        cert_corr_by_feature = (
            certainty_df_samples.groupby('Feature')
            .apply(lambda g: g[['Certainty','Correct']].corr().iloc[0,1] if len(g) > 1 else np.nan)
            .rename('Certainty_Correct_Corr')
            .reset_index()
        )
        # Include condition suffix to keep datasets separate
        pd.DataFrame(certainty_analysis).to_csv(f"comparison/certainty_analysis_{model_name}{suffix}.csv", index=False)
        cert_corr_by_feature.to_csv(f"comparison/certainty_corr_by_feature_{model_name}{suffix}.csv", index=False)

if __name__ == "__main__":
    for model in ["gpt-oss:20b", "gpt-oss:120b", "deepseek-v3.1:671b"]:
        # Include the new combined condition explicitly
        for condition in ["single", "ensemble", "mtp", "ensemble_mtp"]:
            main(model, condition)