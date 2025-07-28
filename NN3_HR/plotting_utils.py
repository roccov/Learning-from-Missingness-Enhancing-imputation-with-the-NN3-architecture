# =============================================================================
# SIMPLIFIED PLOTTING WITH R² BAR CHARTS
# =============================================================================

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import seaborn as sns
from config import Config

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

class SimplifiedTrainingPlotter:
    """Simplified plotting for PyTorch ensemble only with R² bar charts."""
    
    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Create plots subfolder
        self.plots_folder = os.path.join(output_folder, 'plots')
        os.makedirs(self.plots_folder, exist_ok=True)
        
        # Create individual windows subfolder
        self.individual_plots_folder = os.path.join(self.plots_folder, 'individual_windows')
        os.makedirs(self.individual_plots_folder, exist_ok=True)
    
    def plot_individual_window_immediately(self, window_num: int, training_history: Dict[str, Any], 
                                      window_metrics: Dict[str, Any]):
        """Plot training curve for a single window immediately after training."""
        
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])
        
        if not train_losses or not val_losses:
            print(f"Warning: No training history for window {window_num}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Training curves
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        # Mark best epoch
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best Epoch: {best_epoch}')
        ax1.plot(best_epoch, best_val_loss, 'go', markersize=10)
        
        # Calculate overfitting metrics
        final_gap = ((val_losses[-1] - train_losses[-1]) / train_losses[-1]) * 100
        overfitting_status = "Severe" if final_gap > 50 else "Moderate" if final_gap > 20 else "Minimal"
        
        ax1.set_title(f'Window {window_num} Training Curves\n'
                    f'Train R² = {window_metrics["train_r2"]:.3f} | Test R² = {window_metrics["r2"]:.3f}\n'
                    f'Overfitting Gap = {final_gap:.1f}% ({overfitting_status})', 
                    fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set y-axis to log scale if losses vary widely
        if max(train_losses) / min(train_losses) > 10:
            ax1.set_yscale('log')
        
        # Plot 2: Overfitting analysis
        gap_percentages = ((np.array(val_losses) - np.array(train_losses)) / np.array(train_losses)) * 100
        
        ax2.plot(epochs, gap_percentages, 'purple', linewidth=2, label='Val-Train Gap %')
        ax2.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='No Gap')
        ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Moderate Overfitting')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Severe Overfitting')
        
        # Fill areas
        ax2.fill_between(epochs, 0, gap_percentages, 
                        where=(np.array(gap_percentages) <= 20), 
                        alpha=0.3, color='green', label='Good Zone')
        ax2.fill_between(epochs, 0, gap_percentages, 
                        where=(np.array(gap_percentages) > 20) & (np.array(gap_percentages) <= 50), 
                        alpha=0.3, color='orange', label='Caution Zone')
        ax2.fill_between(epochs, 0, gap_percentages, 
                        where=(np.array(gap_percentages) > 50), 
                        alpha=0.3, color='red', label='Danger Zone')
        
        ax2.set_title(f'Overfitting Analysis\nMax Gap = {np.max(gap_percentages):.1f}%', 
                    fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation-Training Gap (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.individual_plots_folder, f'window_{window_num:02d}_training.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  └─ Training plot saved: {plot_path}")
        print(f"  └─ Overfitting status: {overfitting_status} (Gap: {final_gap:.1f}%)")
    
    def plot_summary_at_end(self, training_history: List[Dict[str, Any]], 
                           window_metrics: List[Dict[str, Any]],
                           train_r2_values: List[float], 
                           test_r2_values: List[float]):
        """Plot summary statistics with R² bar charts."""
        
        print("\n=== GENERATING SUMMARY PLOTS ===")
        
        # Main summary plot
        self._plot_main_summary(training_history, window_metrics, train_r2_values, test_r2_values)
        
        # R² bar charts
        self._plot_r2_bar_charts(train_r2_values, test_r2_values)
        
        # Performance analysis
        self._plot_performance_analysis(window_metrics)
    
    def _plot_main_summary(self, training_history: List[Dict[str, Any]], 
                          window_metrics: List[Dict[str, Any]],
                          train_r2_values: List[float], 
                          test_r2_values: List[float]):
        """Plot main summary with training curves and R² progression."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        windows = list(range(1, len(window_metrics) + 1))
        
        # Plot 1: R² progression across windows
        ax1.plot(windows, test_r2_values, 'ro-', label='Test R²', linewidth=2, markersize=6)
        ax1.plot(windows, train_r2_values, 'bo-', label='Train R²', linewidth=2, markersize=6)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax1.set_title('R² Performance Across Windows', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('R² Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error metrics progression
        rmse_values = [m['rmse'] for m in window_metrics]
        mae_values = [m['mae'] for m in window_metrics]
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(windows, rmse_values, 'r-o', label='RMSE', linewidth=2)
        line2 = ax2_twin.plot(windows, mae_values, 'b-s', label='MAE', linewidth=2)
        
        ax2.set_xlabel('Window')
        ax2.set_ylabel('RMSE', color='red')
        ax2_twin.set_ylabel('MAE', color='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        ax2.set_title('Error Metrics Progression', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        # Plot 3: Training epochs per window
        final_epochs = [h['final_epoch'] for h in training_history]
        ax3.bar(windows, final_epochs, alpha=0.7, color='skyblue')
        ax3.set_title('Training Epochs per Window', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Number of Epochs')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        stats_text = f"""PyTorch Ensemble Summary:

Overall Performance:
• Mean Test R²: {np.mean(test_r2_values):.3f} ± {np.std(test_r2_values):.3f}
• Mean Train R²: {np.mean(train_r2_values):.3f} ± {np.std(train_r2_values):.3f}
• Mean RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}
• Mean MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}

Window Statistics:
• Total Windows: {len(windows)}
• Best Test R²: {max(test_r2_values):.3f} (Window {test_r2_values.index(max(test_r2_values))+1})
• Worst Test R²: {min(test_r2_values):.3f} (Window {test_r2_values.index(min(test_r2_values))+1})
• Best Train R²: {max(train_r2_values):.3f} (Window {train_r2_values.index(max(train_r2_values))+1})

Model Configuration:
• Architecture: {Config.HIDDEN_LAYERS}
• Ensemble Size: {Config.DEFAULT_N_ESTIMATORS} models
• Average Epochs: {np.mean(final_epochs):.1f}

Overfitting Analysis:
• Avg Train-Test Gap: {np.mean(train_r2_values) - np.mean(test_r2_values):.3f}
• R² Stability (Test): {np.std(test_r2_values):.3f}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'main_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Main summary saved to: {self.plots_folder}/main_summary.png")
    
    def _plot_r2_bar_charts(self, train_r2_values: List[float], test_r2_values: List[float]):
        """Plot separate bar charts for train and test R²."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        windows = list(range(1, len(train_r2_values) + 1))
        
        # Plot 1: Train R² bar chart
        colors_train = ['green' if r2 > 0.1 else 'orange' if r2 > 0 else 'red' for r2 in train_r2_values]
        bars1 = ax1.bar(windows, train_r2_values, color=colors_train, alpha=0.7, edgecolor='black')
        ax1.set_title('Train R² by Window', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Train R² Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars1, train_r2_values):
            height = bar.get_height()
            label_y = height + 0.01 if height >= 0 else height - 0.02
            ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        # Set y-axis limits
        y_min = min(min(train_r2_values) - 0.05, -0.1)
        y_max = max(max(train_r2_values) + 0.05, 0.1)
        ax1.set_ylim(y_min, y_max)
        
        # Plot 2: Test R² bar chart
        colors_test = ['green' if r2 > 0.1 else 'orange' if r2 > 0 else 'red' for r2 in test_r2_values]
        bars2 = ax2.bar(windows, test_r2_values, color=colors_test, alpha=0.7, edgecolor='black')
        ax2.set_title('Test R² by Window', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Test R² Score')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars2, test_r2_values):
            height = bar.get_height()
            label_y = height + 0.01 if height >= 0 else height - 0.02
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        # Set y-axis limits (same as train for comparison)
        ax2.set_ylim(y_min, y_max)
        
        # Add summary statistics
        train_mean = np.mean(train_r2_values)
        test_mean = np.mean(test_r2_values)
        ax1.axhline(y=train_mean, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean: {train_mean:.3f}')
        ax2.axhline(y=test_mean, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Mean: {test_mean:.3f}')
        
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'r2_bar_charts.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"R² bar charts saved to: {self.plots_folder}/r2_bar_charts.png")
    
    def _plot_performance_analysis(self, window_metrics: List[Dict[str, Any]]):
        """Plot performance analysis."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        r2_values = [m['r2'] for m in window_metrics]
        train_r2_values = [m['train_r2'] for m in window_metrics]
        rmse_values = [m['rmse'] for m in window_metrics]
        n_test_values = [m['n_test'] for m in window_metrics]
        feature_counts = [m['n_features_selected'] for m in window_metrics]
        windows = list(range(1, len(window_metrics) + 1))
        
        # Plot 1: R² distribution
        ax1.hist(r2_values, bins=min(10, len(r2_values)), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(r2_values), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(r2_values):.3f}')
        ax1.axvline(np.median(r2_values), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(r2_values):.3f}')
        ax1.set_title('Test R² Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('R² Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Train vs Test R² scatter
        ax2.scatter(train_r2_values, r2_values, alpha=0.7, s=50, c=windows, cmap='viridis')
        
        # Perfect correlation line
        min_val = min(min(train_r2_values), min(r2_values))
        max_val = max(max(train_r2_values), max(r2_values))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
        
        ax2.set_xlabel('Train R²')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Train vs Test R² Correlation', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Window Number')
        
        # Plot 3: Feature count progression
        ax3.bar(windows, feature_counts, alpha=0.7, color='lightgreen')
        ax3.set_title('Features Selected per Window', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Number of Features')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample size vs performance
        ax4.scatter(n_test_values, r2_values, alpha=0.7, s=50, c=windows, cmap='plasma')
        ax4.set_xlabel('Test Sample Size')
        ax4.set_ylabel('Test R² Score')
        ax4.set_title('Sample Size vs Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar2.set_label('Window Number')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'performance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance analysis saved to: {self.plots_folder}/performance_analysis.png")
    
    def plot_predictions_vs_actual(self, dates: List, actual_values: List[float], 
                                 predictions: List[float]):
        """Plot predictions vs actual values over time."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series plot
        ax1.plot(dates, actual_values, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
        ax1.plot(dates, predictions, 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
        
        # Calculate and display correlation
        correlation = np.corrcoef(actual_values, predictions)[0, 1]
        
        ax1.set_title(f'Predictions vs Actual Values Over Time (PyTorch Ensemble)\n'
                     f'Correlation: {correlation:.3f}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Target Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Scatter plot
        ax2.scatter(actual_values, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(min(actual_values), min(predictions))
        max_val = max(max(actual_values), max(predictions))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(actual_values, predictions)
        
        ax2.set_title(f'Predicted vs Actual (Scatter Plot)\nR² = {r2:.3f}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Statistics:
R² = {r2:.3f}
Correlation = {correlation:.3f}
RMSE = {np.sqrt(np.mean((np.array(actual_values) - np.array(predictions))**2)):.4f}
MAE = {np.mean(np.abs(np.array(actual_values) - np.array(predictions))):.4f}
N = {len(actual_values):,} samples"""
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_folder, 'predictions_vs_actual.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Predictions vs actual plot saved to: {self.plots_folder}/predictions_vs_actual.png")