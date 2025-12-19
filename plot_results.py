import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

def abbreviate_label(label):
    """Abbreviates long condition labels for the heatmap."""
    if label == 'Clean':
        return 'Clean'
    label = label.replace('Gaussian Noise', 'GN')
    label = label.replace('sigma=', 's=')
    label = label.replace('k=', 'k=')
    label = label.replace('factor=', 'f=')
    return label

def plot_all_results():
    results_dir = 'results1'
    output_dir = 'plots'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_files = glob.glob(os.path.join(results_dir, '*_stress_test_results.csv'))
    
    all_data = []
    for f in csv_files:
        model_name = os.path.basename(f).replace('_stress_test_results.csv', '')
        df = pd.read_csv(f)
        df['model'] = model_name
        all_data.append(df)
        
    df_all = pd.concat(all_data, ignore_index=True)
    
    # --- Generate Line Plots ---
    line_plot_conditions = df_all[df_all['Condition'] != 'Clean']['Condition'].unique()
    
    for condition in line_plot_conditions:
        condition_df = df_all[df_all['Condition'] == condition]

        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        for model in condition_df['model'].unique():
            model_df = condition_df[condition_df['model'] == model].copy()
            if condition == 'Gaussian Noise':
                model_df['param'] = model_df['Parameter'].str.extract('=(\d+\.?\d*)').astype(float)
                xlabel = 'Noise Sigma'
            elif condition == 'Blur':
                model_df['param'] = model_df['Parameter'].str.extract('=(\d+)').astype(int)
                xlabel = 'Kernel Size'
            elif condition == 'Contrast':
                model_df['param'] = model_df['Parameter'].str.extract('=(\d+\.?\d*)').astype(float)
                xlabel = 'Contrast Factor'
            model_df = model_df.sort_values('param')
            plt.plot(model_df['param'], model_df['Accuracy'], marker='o', label=model)
        plt.title(f'Model Accuracy under {condition}')
        plt.xlabel(xlabel)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(output_dir, f'{condition.lower().replace(" ", "_")}_accuracy_comparison.png')
        plt.savefig(output_path)
        plt.close()

        # Plot F1-Score
        plt.figure(figsize=(10, 6))
        for model in condition_df['model'].unique():
            model_df = condition_df[condition_df['model'] == model].copy()
            if condition == 'Gaussian Noise':
                model_df['param'] = model_df['Parameter'].str.extract('=(\d+\.?\d*)').astype(float)
                xlabel = 'Noise Sigma'
            elif condition == 'Blur':
                model_df['param'] = model_df['Parameter'].str.extract('=(\d+)').astype(int)
                xlabel = 'Kernel Size'
            elif condition == 'Contrast':
                model_df['param'] = model_df['Parameter'].str.extract('=(\d+\.?\d*)').astype(float)
                xlabel = 'Contrast Factor'
            model_df = model_df.sort_values('param')
            plt.plot(model_df['param'], model_df['F1-Score'], marker='o', label=model)
        plt.title(f'Model F1-Score under {condition}')
        plt.xlabel(xlabel)
        plt.ylabel('F1-Score')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(output_dir, f'{condition.lower().replace(" ", "_")}_f1_score_comparison.png')
        plt.savefig(output_path)
        plt.close()

    # --- Generate Heatmaps ---
    df_all_copy = df_all.copy()
    df_all_copy['Test Condition'] = df_all_copy.apply(
        lambda row: 'Clean' if row['Condition'] == 'Clean' else f"{row['Condition']} ({row['Parameter']})",
        axis=1
    )
    df_all_copy['Test Condition'] = df_all_copy['Test Condition'].apply(abbreviate_label)

    # Reorder columns to have 'Clean' first
    all_conditions = sorted([col for col in df_all_copy['Test Condition'].unique() if col != 'Clean'])
    ordered_columns = ['Clean'] + all_conditions

    # Accuracy Heatmap
    accuracy_pivot = df_all_copy.pivot(index='model', columns='Test Condition', values='Accuracy')
    accuracy_pivot = accuracy_pivot[ordered_columns] # Apply column order
    plt.figure(figsize=(16, 8))
    sns.heatmap(accuracy_pivot, annot=True, fmt=".3f", cmap="RdBu_r", linewidths=.5)
    plt.title('Model Accuracy Under Different Conditions', fontsize=16)
    plt.xlabel('Test Condition', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    accuracy_heatmap_path = os.path.join(output_dir, 'accuracy_heatmap.png')
    plt.savefig(accuracy_heatmap_path)
    plt.close()

    # F1-Score Heatmap
    f1_score_pivot = df_all_copy.pivot(index='model', columns='Test Condition', values='F1-Score')
    f1_score_pivot = f1_score_pivot[ordered_columns] # Apply column order
    plt.figure(figsize=(16, 8))
    sns.heatmap(f1_score_pivot, annot=True, fmt=".3f", cmap="RdBu_r", linewidths=.5)
    plt.title('Model F1-Score Under Different Conditions', fontsize=16)
    plt.xlabel('Test Condition', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    f1_score_heatmap_path = os.path.join(output_dir, 'f1_score_heatmap.png')
    plt.savefig(f1_score_heatmap_path)
    plt.close()

    print(f"All plots, including updated heatmaps, have been saved to the '{output_dir}' directory.")

if __name__ == '__main__':
    plot_all_results()
