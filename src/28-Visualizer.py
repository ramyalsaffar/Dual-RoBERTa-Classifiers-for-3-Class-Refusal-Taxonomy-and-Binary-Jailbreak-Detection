# Visualization Module
#---------------------
# Generate visualizations for results.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - All alpha values from VISUALIZATION_CONFIG
# - Uses safe_divide() from Utils for robust division
# - Improved comments and documentation
# All imports are in 01-Imports.py
###############################################################################


class Visualizer:
    """
    Generate visualizations for classifier results.
    
    Creates various plots including confusion matrices, F1 scores, confidence
    distributions, training curves, and vulnerability heatmaps. All visual 
    parameters (alpha, colors, DPI) are sourced from VISUALIZATION_CONFIG 
    to maintain consistency and avoid hardcoded values.
    """

    def __init__(self, class_names: List[str] = None):
        """
        Initialize visualizer with class names and configuration.
        
        Args:
            class_names: List of class names for labels (default: CLASS_NAMES from Setup)
        """
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.class_colors = PLOT_COLORS_LIST
        self.model_colors = MODEL_COLORS
        
        # Get visualization config - NO HARDCODING!
        self.dpi = REPORT_CONFIG['images']['save_dpi']  # CHANGED: Use 300 DPI from REPORT_CONFIG
        self.alpha_bar = VISUALIZATION_CONFIG['alpha_bar']
        self.alpha_line = VISUALIZATION_CONFIG['alpha_line']
        self.alpha_hist = VISUALIZATION_CONFIG['alpha_hist']
        self.alpha_grid = VISUALIZATION_CONFIG['alpha_grid']
        self.alpha_bbox = VISUALIZATION_CONFIG['alpha_bbox']
        
        sns.set_style(VISUALIZATION_CONFIG['style'])

    def plot_confusion_matrix(self, cm: np.ndarray, output_path: str):
        """
        Plot confusion matrix heatmap.
    
        Args:
            cm: Confusion matrix (numpy array)
            output_path: FULL PATH to save (including directory and filename)
    
        Returns:
            Matplotlib figure object
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        
        # Use figure size from REPORT_CONFIG
        img_cfg = REPORT_CONFIG['images']
        cm_size = img_cfg['figure_sizes']['confusion_matrix']  # e.g. (6.0, 5.0)
    
        fig = plt.figure(figsize=cm_size)
    
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            square=True
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
    
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {output_path}")
        return fig

    def plot_per_class_f1(self, f1_scores: Dict[str, float], output_path: str):
        """
        Plot per-class F1 scores as bar chart.
        
        Args:
            f1_scores: Dictionary mapping class names to F1 scores
            output_path: FULL PATH to save (including directory and filename)
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        
        plt.figure(figsize=(10, 6))
        classes = list(f1_scores.keys())
        scores = list(f1_scores.values())

        # Use modulo for color cycling if we have more classes than colors
        colors = [self.class_colors[i % len(self.class_colors)] for i in range(len(classes))]
        bars = plt.bar(classes, scores, color=colors, alpha=self.alpha_bar, edgecolor='black')
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Per-Class F1 Scores', fontsize=16, fontweight='bold')
        plt.ylim([0, 1])
        
        # Add target F1 reference line from config
        f1_target = VISUALIZATION_CONFIG['f1_target']
        plt.axhline(y=f1_target, color='red', linestyle='--', alpha=self.alpha_line, 
                   label=f'Target ({f1_target:.2f})')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-class F1 to {output_path}")

    def plot_per_model_f1(self, results: Dict, output_path: str):
        """
        Plot per-model F1 comparison bar chart.
        
        Args:
            results: Dictionary with model results (expects 'f1_macro' key per model)
            output_path: Path to save the plot
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        models = [m for m in results.keys() if m != 'analysis']
        f1_scores = [results[m]['f1_macro'] for m in models]

        plt.figure(figsize=(10, 6))
        colors = [self.model_colors.get(m, '#95a5a6') for m in models]
        bars = plt.bar(models, f1_scores, color=colors, alpha=self.alpha_bar, edgecolor='black')

        plt.ylabel('F1 Score (Macro)', fontsize=12)
        plt.title('Cross-Model Performance', fontsize=16, fontweight='bold')
        plt.ylim([0, 1])
        plt.axhline(y=np.mean(f1_scores), color='red', linestyle='--',
                   alpha=self.alpha_line, label=f'Mean ({np.mean(f1_scores):.3f})')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved per-model F1 to {output_path}")

    def plot_adversarial_robustness(self, results: Dict, output_path: str):
        """
        Plot adversarial robustness comparison (original vs paraphrased).
        
        Shows F1 score drops across different paraphrase dimensions.
        Uses safe_divide() from Utils to avoid division by zero.
        
        Args:
            results: Dictionary with 'original_f1' and 'paraphrased_f1' keys
            output_path: Path to save the plot
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        dimensions = list(results['paraphrased_f1'].keys())
        paraphrased_f1 = list(results['paraphrased_f1'].values())
        original_f1 = results['original_f1']

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(dimensions))
        width = 0.35

        # Original (repeated for comparison) - using green from PLOT_COLORS
        ax.bar(x - width/2, [original_f1] * len(dimensions), width,
               label='Original', color='#2ecc71', alpha=self.alpha_bar, edgecolor='black')

        # Paraphrased - using red to indicate degradation
        ax.bar(x + width/2, paraphrased_f1, width,
               label='Paraphrased', color='#e74c3c', alpha=self.alpha_bar, edgecolor='black')

        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Adversarial Robustness: Original vs Paraphrased',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions)
        ax.legend()
        ax.set_ylim([0, 1])

        # Add drop percentage using safe_divide from Utils
        for i, (orig, para) in enumerate(zip([original_f1] * len(dimensions), paraphrased_f1)):
            drop = safe_divide(orig - para, orig, default=0.0) * 100
            ax.text(i, max(orig, para) + 0.02, f'-{drop:.1f}%',
                   ha='center', fontsize=9, color='red')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved adversarial robustness to {output_path}")

    def plot_confidence_distributions(self, labels: List, confidences: List, output_path: str):
        """
        Plot confidence distributions per class.
        
        GENERIC: Works with any number of classes (2, 3, or more).
        Uses alpha_hist from config for histogram transparency.
        
        Args:
            labels: List of true labels
            confidences: List of prediction confidences
            output_path: Path to save the plot
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        # Dynamic subplot configuration based on number of classes
        fig, axes = plt.subplots(1, self.num_classes, figsize=(5 * self.num_classes, 4))

        # Handle single class case (axes won't be a list)
        if self.num_classes == 1:
            axes = [axes]

        for class_idx in range(self.num_classes):
            class_confidences = [c for l, c in zip(labels, confidences) if l == class_idx]

            # Use modulo for color cycling if we have more classes than colors
            color_idx = class_idx % len(self.class_colors)

            # Use confidence_bins from ANALYSIS_CONFIG and alpha_hist from VISUALIZATION_CONFIG
            axes[class_idx].hist(class_confidences, bins=ANALYSIS_CONFIG['confidence_bins'],
                               alpha=self.alpha_hist, color=self.class_colors[color_idx], edgecolor='black')
            axes[class_idx].set_title(f'{self.class_names[class_idx]}',
                                     fontsize=12, fontweight='bold')
            axes[class_idx].set_xlabel('Confidence', fontsize=10)
            axes[class_idx].set_ylabel('Count', fontsize=10)
            axes[class_idx].set_xlim([0, 1])

            if len(class_confidences) > 0:
                axes[class_idx].axvline(x=np.mean(class_confidences), color='red',
                                       linestyle='--', label=f'Mean: {np.mean(class_confidences):.3f}')
                axes[class_idx].legend()

        plt.suptitle('Confidence Distributions by Class', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confidence distributions to {output_path}")

    def plot_training_curves(self, history: Dict, output_path: str):
        """
        Plot training and validation curves (loss and F1).
    
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'val_f1' keys
            output_path: Path to save the plot
    
        Returns:
            Matplotlib figure object
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        img_cfg = REPORT_CONFIG['images']
        train_size = img_cfg['figure_sizes']['training_curves']  # e.g. (7.0, 5.0)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=train_size)
    
        epochs = range(1, len(history['train_loss']) + 1)
    
        # Loss subplot
        ax1.plot(epochs, history['train_loss'], label='Train Loss', alpha=self.alpha_line)
        ax1.plot(epochs, history['val_loss'], label='Val Loss', alpha=self.alpha_line)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(alpha=VISUALIZATION_CONFIG['alpha_grid'])
    
        # F1 subplot
        ax2.plot(epochs, history['val_f1'], label='Val F1', alpha=self.alpha_line)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1')
        ax2.legend()
        ax2.grid(alpha=VISUALIZATION_CONFIG['alpha_grid'])
    
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved training curves to {output_path}")
        return fig

    def plot_model_vulnerability_heatmap(self, matrix_df: pd.DataFrame, output_path: str):
        """
        Plot Model × Attack Type vulnerability heatmap.

        NEW (V09): Visualizes which jailbreak tactics work on which models.
        All visual parameters sourced from VISUALIZATION_CONFIG['heatmap'].

        Args:
            matrix_df: DataFrame with models as rows, attack types as columns, values = success rate %
            output_path: Path to save the plot
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        if matrix_df.empty:
            print("⚠️  No data for heatmap - skipping")
            return

        # Get heatmap config - NO HARDCODING!
        heatmap_config = VISUALIZATION_CONFIG['heatmap']

        # Calculate figure size based on matrix dimensions
        n_rows = len(matrix_df.index)
        n_cols = len(matrix_df.columns)
        figsize_per_cell = heatmap_config['figsize_per_cell']
        figsize = (n_cols * figsize_per_cell + 2, n_rows * figsize_per_cell + 1)

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap - all params from config
        sns.heatmap(
            matrix_df,
            annot=heatmap_config['annot'],
            fmt=heatmap_config['fmt'],
            cmap=heatmap_config['cmap'],
            vmin=heatmap_config['vmin'],
            vmax=heatmap_config['vmax'],
            linewidths=heatmap_config['linewidths'],
            linecolor=heatmap_config['linecolor'],
            cbar_kws={'label': heatmap_config['cbar_label']},
            ax=ax
        )

        # Title and labels
        ax.set_title('Model Vulnerability Heatmap: Jailbreak Success Rate (%)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Attack Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')

        # Rotate labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved vulnerability heatmap to {output_path}")

    def plot_model_vulnerability_comparison(self, vulnerability_stats: Dict, significance_results: Dict, output_path: str):
        """
        Plot bar chart comparing model vulnerability rates with significance markers.

        NEW (V09): Shows jailbreak success rate per model with statistical significance.
        All visual parameters sourced from VISUALIZATION_CONFIG['model_comparison'].

        Args:
            vulnerability_stats: Dict from JailbreakAnalysis._analyze_vulnerability_per_model()
            significance_results: Dict from JailbreakAnalysis._test_model_vulnerability_significance()
            output_path: Path to save the plot
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        if not vulnerability_stats:
            print("⚠️  No vulnerability data - skipping comparison chart")
            return

        # Get model comparison config - NO HARDCODING!
        comp_config = VISUALIZATION_CONFIG['model_comparison']

        # Extract data
        models = list(vulnerability_stats.keys())
        success_rates = [vulnerability_stats[m]['success_rate'] * 100 for m in models]  # Convert to percentage
        total_samples = [vulnerability_stats[m]['total_samples'] for m in models]

        # Create figure with configured dimensions
        fig, ax = plt.subplots(figsize=(comp_config['figsize_width'], comp_config['figsize_height']))

        # Create bar chart with color coding (red=worst, green=best, blue=middle)
        bars = ax.bar(
            range(len(models)),
            success_rates,
            width=comp_config['bar_width'],
            color=['#e74c3c' if rate == max(success_rates) else '#2ecc71' if rate == min(success_rates) else '#3498db'
                   for rate in success_rates],
            edgecolor='black',
            linewidth=1.5
        )

        # Add value labels on bars if configured
        if comp_config['show_values']:
            for i, (bar, rate, n) in enumerate(zip(bars, success_rates, total_samples)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{rate:{comp_config["value_format"]}}%\n(n={n})',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

        # Add significance marker if test shows significance
        if significance_results and not significance_results.get('error'):
            if significance_results['significant']:
                # Add marker above most vulnerable model
                vulnerable_idx = models.index(significance_results['vulnerable_model'])
                ax.text(
                    vulnerable_idx,
                    success_rates[vulnerable_idx] + max(success_rates) * 0.1,
                    comp_config['significance_marker'],
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    color='red',
                    fontweight='bold'
                )

                # Add note
                ax.text(
                    0.98,
                    0.98,
                    f'{comp_config["significance_marker"]} p < {significance_results["alpha"]}',
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=self.alpha_bbox)
                )

        # Labels and title
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Jailbreak Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Vulnerability Comparison\n(Lower is Better)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, max(success_rates) * 1.3)  # Add space for labels
        ax.grid(axis='y', alpha=self.alpha_grid)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved vulnerability comparison to {output_path}")

    def plot_class_distribution(self, labels: List, output_path: str):
        """
        Plot class distribution bar chart.
    
        Args:
            labels: List of class labels (integers)
            output_path: Path to save the plot
    
        Returns:
            Matplotlib figure object
        """
        # ADDED: Ensure parent directory exists
        ensure_dir_exists(os.path.dirname(output_path))
        

        img_cfg = REPORT_CONFIG['images']
        dist_size = img_cfg['figure_sizes']['class_distribution']  # e.g. (6.0, 4.0)
    
        fig = plt.figure(figsize=dist_size)
    
        # Count samples per class
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_names_subset = [self.class_names[i] for i in unique_labels]
    
        colors = [self.class_colors[i % len(self.class_colors)] for i in range(len(unique_labels))]
        bars = plt.bar(class_names_subset, counts, color=colors, alpha=self.alpha_bar, edgecolor='black')
    
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution', fontsize=16, fontweight='bold')
    
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved class distribution to {output_path}")
        return fig


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: Visualization Module
Created on October 28, 2025
@author: ramyalsaffar
"""
