import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_study_results(results_file):
    """Load the Optuna study results from a JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_trial_dataframe(study_results):
    """Create a pandas DataFrame from the trial results."""
    trials = study_results['all_trials']
    data = []

    for trial in trials:
        if trial['value'] == float('inf'):
            continue  # Skip failed trials

        row = {'trial': trial['trial'], 'value': trial['value']}
        row.update(trial['params'])
        data.append(row)

    return pd.DataFrame(data)

def plot_parameter_importances(df, output_dir):
    """Plot the importance of each hyperparameter."""
    # Calculate correlation with the objective value
    correlations = df.drop('trial', axis=1).corr()['value'].drop('value')

    # Sort by absolute correlation
    correlations = correlations.abs().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.values, y=correlations.index)
    plt.title('Hyperparameter Importance (Correlation with Objective)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_importance.png'), dpi=300)
    plt.close()

def plot_parallel_coordinates(df, output_dir):
    """Create a parallel coordinates plot to visualize the hyperparameter space."""
    # Normalize the data for better visualization
    df_norm = df.copy()
    for col in df_norm.columns:
        if col not in ['trial', 'value']:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

    # Sort by objective value
    df_norm = df_norm.sort_values('value')

    # Create a colormap based on the objective value
    colormap = plt.cm.viridis
    colors = colormap(np.linspace(0, 1, len(df_norm)))

    plt.figure(figsize=(15, 8))
    pd.plotting.parallel_coordinates(
        df_norm, 'trial',
        color=colors,
        cols=[col for col in df_norm.columns if col != 'trial']
    )
    plt.title('Parallel Coordinates Plot of Hyperparameters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parallel_coordinates.png'), dpi=300)
    plt.close()

def plot_pairwise_relationships(df, output_dir):
    """Plot pairwise relationships between hyperparameters and the objective."""
    # Select a subset of the most important parameters
    correlations = df.drop('trial', axis=1).corr()['value'].drop('value')
    top_params = correlations.abs().sort_values(ascending=False).head(4).index

    # Create a subset dataframe with the top parameters and the objective
    subset_df = df[list(top_params) + ['value']]

    # Create pairplot
    plt.figure(figsize=(12, 10))
    g = sns.pairplot(subset_df, diag_kind='kde', corner=True)
    g.fig.suptitle('Pairwise Relationships Between Top Parameters', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairwise_relationships.png'), dpi=300)
    plt.close()

def plot_objective_history(df, output_dir):
    """Plot the history of the objective value across trials."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['trial'], df['value'], 'o-', color='blue')
    plt.axhline(y=df['value'].min(), color='red', linestyle='--',
                label=f'Best value: {df["value"].min():.4f}')
    plt.title('Objective Value History')
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'objective_history.png'), dpi=300)
    plt.close()

def create_summary_table(df, study_results, output_dir):
    """Create a summary table of the best hyperparameters and metrics."""
    best_params = study_results['best_params']
    best_value = study_results['best_value']

    # Create a summary table for hyperparameters
    summary = pd.DataFrame({
        'Parameter': list(best_params.keys()) + ['Combined Score'],
        'Best Value': list(best_params.values()) + [best_value]
    })

    # Save to CSV
    summary.to_csv(os.path.join(output_dir, 'best_parameters.csv'), index=False)

    # Also save as a formatted markdown table
    with open(os.path.join(output_dir, 'best_parameters.md'), 'w') as f:
        f.write('# Best Hyperparameters\n\n')
        f.write(summary.to_markdown(index=False))
        f.write('\n\n')

    # Try to load the metrics from the best trial
    best_trial_number = study_results['best_trial']
    metrics_file = os.path.join(os.path.dirname(output_dir), 'results', f'trial_{best_trial_number}', 'metrics.json')

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Create a metrics summary table
        metrics_to_display = [
            ('Action Recognition Accuracy', metrics.get('action_recognition_accuracy', 'N/A'), 'Higher is better'),
            ('Re-identification Accuracy', metrics.get('reidentification_accuracy', 'N/A'), 'Lower is better'),
            ('MSE with Ground Truth', metrics.get('mse_gt', 'N/A'), 'Lower is better'),
            ('Bone Length Consistency', metrics.get('bone_length_consistency', 'N/A'), 'Lower is better'),
            ('Joint Angle Limits', metrics.get('joint_angle_limits', 'N/A'), 'Higher is better'),
            ('Temporal Smoothness', metrics.get('temporal_smoothness', 'N/A'), 'Lower is better'),
            ('Velocity Consistency', metrics.get('velocity_consistency', 'N/A'), 'Higher is better'),
            ('Foot Contact Consistency', metrics.get('foot_contact_consistency', 'N/A'), 'Higher is better'),
            ('Validation Loss', metrics.get('validation_loss', 'N/A'), 'Lower is better'),
            ('Combined Score', metrics.get('combined_score', 'N/A'), 'Lower is better')
        ]

        metrics_df = pd.DataFrame(metrics_to_display, columns=['Metric', 'Value', 'Interpretation'])

        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(output_dir, 'best_metrics.csv'), index=False)

        # Also save as a formatted markdown table
        with open(os.path.join(output_dir, 'best_metrics.md'), 'w') as f:
            f.write('# Best Trial Metrics\n\n')
            f.write(metrics_df.to_markdown(index=False))
            f.write('\n\n')

def main():
    parser = argparse.ArgumentParser(description='Analyze Optuna hyperparameter tuning results')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to the study results JSON file')
    parser.add_argument('--output-dir', type=str, default='experiments/3_hyperparameter/analysis',
                        help='Directory to save analysis results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load study results
    study_results = load_study_results(args.results_file)

    # Create DataFrame from trials
    df = create_trial_dataframe(study_results)

    # Save the processed DataFrame
    df.to_csv(os.path.join(args.output_dir, 'processed_trials.csv'), index=False)

    # Generate plots and analysis
    plot_parameter_importances(df, args.output_dir)
    plot_parallel_coordinates(df, args.output_dir)
    plot_pairwise_relationships(df, args.output_dir)
    plot_objective_history(df, args.output_dir)
    create_summary_table(df, study_results, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
