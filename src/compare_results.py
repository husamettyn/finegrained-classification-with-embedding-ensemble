import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def load_metrics(models_dir):
    results = []
    
    # Iterate over all subdirectories in the models directory
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        metrics_file = os.path.join(model_path, "metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Warning: No metrics.json found for {model_name}")
            continue
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1']
        })
        
    return pd.DataFrame(results)

def plot_comparison(df, save_dir):
    # Melt dataframe for seaborn
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create comparison plot
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, rotation=90)
        
    plt.title("Model Comparison Metrics", fontsize=16, pad=20)
    # Fix text alignment for rotated labels
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)  # Scores are between 0 and 1, add some headroom for labels
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)  # Add extra space for rotated labels
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()

def plot_f1_comparison(df, save_dir):
    plt.figure(figsize=(12, 6))
    
    # Sort by F1 Score
    df_sorted = df.sort_values('F1 Score', ascending=False)
    
    # Fix FutureWarning: use hue parameter and set legend=False
    ax = sns.barplot(data=df_sorted, x="Model", y="F1 Score", hue="Model", palette="magma", legend=False)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
        
    plt.title("F1 Score Comparison (Sorted)", fontsize=16, pad=20)
    # Fix text alignment for rotated labels
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add extra space for rotated labels
    save_path = os.path.join(save_dir, "f1_score_ranking.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"F1 ranking plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare model results from metrics.json files")
    parser.add_argument("--models_dir", required=True, help="Path to the directory containing model subdirectories")
    args = parser.parse_args()
    
    if not os.path.exists(args.models_dir):
        print(f"Error: Directory {args.models_dir} does not exist")
        return

    print(f"Loading metrics from {args.models_dir}...")
    df = load_metrics(args.models_dir)
    
    if df.empty:
        print("No metrics found.")
        return
        
    print("\nResults Summary:")
    print(df.sort_values('F1 Score', ascending=False).to_string(index=False))
    
    # Save comparison CSV
    csv_path = os.path.join(args.models_dir, "comparison_summary.csv")
    df.sort_values('F1 Score', ascending=False).to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved to {csv_path}")
    
    # Generate plots
    plot_comparison(df, args.models_dir)
    plot_f1_comparison(df, args.models_dir)

if __name__ == "__main__":
    main()

