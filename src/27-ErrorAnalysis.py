# Error Analysis Module
#-----------------------
# Comprehensive error analysis for classification models.
# Provides 7 detailed analysis modules to understand model failures.
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# ERROR ANALYZER CLASS
# =============================================================================

class ErrorAnalyzer:
    """
    Comprehensive error analysis for classification models.

    Modules:
    1. Confusion Matrix Deep Dive
    2. Per-Class Performance Breakdown
    3. Confidence Analysis
    4. Input Length Analysis
    5. Failure Case Extraction
    6. Token-Level Attribution (SHAP/Attention)
    7. Jailbreak Detection Error Analysis
    """

    def __init__(self,
                 model,
                 dataset,
                 tokenizer,
                 device: torch.device,
                 class_names: List[str],
                 task_type: str = 'refusal',
                 output_dir: str = None,
                 timestamp: str = None):
        """
        Initialize error analyzer.

        Args:
            model: Trained classification model
            dataset: Test dataset
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names
            task_type: 'refusal' or 'jailbreak'
            output_dir: Directory to save visualizations (default: error_analysis_path)
        """
        # Input Validation
        if model is None:
            raise ValueError("❌ ERROR: model cannot be None")
        
        if dataset is None or len(dataset) == 0:
            raise ValueError("❌ ERROR: dataset cannot be None or empty")
        
        if tokenizer is None:
            raise ValueError("❌ ERROR: tokenizer cannot be None")
        
        if class_names is None or len(class_names) == 0:
            raise ValueError("❌ ERROR: class_names cannot be None or empty")
        
        self.model = model.to(device)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names
        self.task_type = task_type
        
        # Set output directory - use error_analysis_path if not provided
        self.output_dir = output_dir if output_dir is not None else error_analysis_path
        self.timestamp = timestamp

        # Model predictions and metadata
        self.predictions = None
        self.true_labels = None
        self.confidence_scores = None
        self.logits = None
        self.texts = None
        self.token_lengths = None

        # Analysis results
        self.analysis_results = {}

        print_banner(f"ERROR ANALYZER INITIALIZED: {task_type.upper()}", width=60)
        print(f"Model: {model.__class__.__name__}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Classes: {class_names}")
        print(f"Device: {device}")
        print(f"Output directory: {self.output_dir}")
        print_banner("", width=60, char="=")


    def run_predictions(self):
        """Run model on full dataset and store predictions."""
        print_banner("RUNNING MODEL PREDICTIONS", width=60)

        self.model.eval()

        all_preds = []
        all_labels = []
        all_confidences = []
        all_logits = []
        all_texts = []
        all_token_lengths = []

        # CRITICAL FIX: Use TRAINING_CONFIG not API_CONFIG for batch_size!
        data_loader = DataLoader(self.dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Predicting") as pbar:
                for batch in data_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                    confidences, preds = torch.max(probs, dim=1)

                    # Store results
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_confidences.extend(confidences.cpu().numpy())
                    all_logits.extend(logits.cpu().numpy())

                    # Decode texts
                    for input_id in input_ids:
                        text = self.tokenizer.decode(input_id, skip_special_tokens=True)
                        all_texts.append(text)
                        # Count non-padding tokens
                        token_length = (input_id != self.tokenizer.pad_token_id).sum().item()
                        all_token_lengths.append(token_length)

                    pbar.update(1)

        # Store predictions
        self.predictions = np.array(all_preds)
        self.true_labels = np.array(all_labels)
        self.confidence_scores = np.array(all_confidences)
        self.logits = np.array(all_logits)
        self.texts = all_texts
        self.token_lengths = np.array(all_token_lengths)

        # Calculate accuracy
        accuracy = (self.predictions == self.true_labels).mean()
        print(f"\n✓ Predictions complete")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Total samples: {len(self.predictions)}")
        print(f"  Correct: {(self.predictions == self.true_labels).sum()}")
        print(f"  Incorrect: {(self.predictions != self.true_labels).sum()}\n")


    # =========================================================================
    # MODULE 1: CONFUSION MATRIX DEEP DIVE
    # =========================================================================

    def analyze_confusion_matrix(self, save_visualizations: bool = True) -> Dict:
        """
        Module 1: Detailed confusion matrix analysis.

        Returns:
            Dictionary with confusion matrix analysis
        """
        print_banner("MODULE 1: CONFUSION MATRIX DEEP DIVE", width=60)


        # Calculate confusion matrix with EXPLICIT labels to ensure correct size
        # CRITICAL: Must specify labels parameter to match self.class_names
        # Otherwise sklearn infers from unique values which may not match num_classes
        cm = confusion_matrix(
            self.true_labels, 
            self.predictions,
            labels=list(range(len(self.class_names)))  # FIXED: Explicit labels [0, 1, ...N-1]
        )

        # Normalize confusion matrix (row-wise) using safe_divide
        # Handle division by zero - rows with no samples get 0
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.array([[safe_divide(cm[i,j], row_sums[i,0], default=0.0) 
                                   for j in range(cm.shape[1])] 
                                  for i in range(cm.shape[0])])

        # Print raw counts
        print("Confusion Matrix (Raw Counts):")
        print(f"{'-'*60}")
        header = "True \\ Pred".ljust(15) + "".join([f"{name[:10]:>12}" for name in self.class_names])
        print(header)
        print(f"{'-'*60}")
        for i, class_name in enumerate(self.class_names):
            row = f"{class_name[:15]:<15}" + "".join([f"{cm[i, j]:>12}" for j in range(len(self.class_names))])
            print(row)
        print()

        # Print normalized percentages
        print("Confusion Matrix (Row-Normalized %):")
        print(f"{'-'*60}")
        print(header)
        print(f"{'-'*60}")
        for i, class_name in enumerate(self.class_names):
            row = f"{class_name[:15]:<15}" + "".join([f"{cm_normalized[i, j]*100:>11.1f}%" for j in range(len(self.class_names))])
            print(row)
        print()

        # Identify most confused class pairs
        print("Most Confused Class Pairs:")
        print(f"{'-'*60}")

        confused_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': cm[i, j],
                        'percentage': cm_normalized[i, j] * 100
                    })

        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)

        for pair in confused_pairs[:5]:  # Top 5
            print(f"  {pair['true_class']} → {pair['predicted_class']}: {pair['count']} ({pair['percentage']:.1f}%)")

        print()

        # Visualize if requested
        if save_visualizations:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Raw counts
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
            disp.plot(ax=axes[0], cmap='Blues', values_format='d')
            axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

            # Normalized
            disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=self.class_names)
            disp_norm.plot(ax=axes[1], cmap='Oranges', values_format='.2f')
            axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

            plt.tight_layout()
            
            save_path = os.path.join(self.output_dir, get_timestamped_filename("error_confusion_matrix.png", self.task_type, self.timestamp))
            
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            print(f"✓ Confusion matrix visualization saved: {save_path}\n")

        result = {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'confused_pairs': confused_pairs
        }

        self.analysis_results['confusion_matrix'] = result
        return result


    # =========================================================================
    # MODULE 2: PER-CLASS PERFORMANCE BREAKDOWN
    # =========================================================================

    def analyze_per_class_performance(self, save_visualizations: bool = True) -> Dict:
        """
        Module 2: Detailed per-class metrics.

        Returns:
            Dictionary with per-class analysis
        """
        print_banner("MODULE 2: PER-CLASS PERFORMANCE BREAKDOWN", width=60)


        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels,
            self.predictions,
            labels=list(range(len(self.class_names)))
        )

        # Print table
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print(f"{'-'*70}")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

        # Macro averages using safe_divide
        print(f"{'-'*70}")
        avg_precision = safe_divide(precision.sum(), len(precision), default=0.0)
        avg_recall = safe_divide(recall.sum(), len(recall), default=0.0)
        avg_f1 = safe_divide(f1.sum(), len(f1), default=0.0)
        print(f"{'Macro Average':<20} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f} {support.sum():<10}")
        print()

        # Identify best and worst performing classes
        best_f1_idx = np.argmax(f1)
        worst_f1_idx = np.argmin(f1)

        print(f"Best performing class:  {self.class_names[best_f1_idx]} (F1: {f1[best_f1_idx]:.4f})")
        print(f"Worst performing class: {self.class_names[worst_f1_idx]} (F1: {f1[worst_f1_idx]:.4f})")
        print()

        # Visualize if requested
        if save_visualizations:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            x = np.arange(len(self.class_names))
            width = 0.6

            # Precision
            axes[0].bar(x, precision, width, color='#3498db', alpha=0.8)
            axes[0].set_ylabel('Precision', fontsize=12)
            axes[0].set_title('Precision by Class', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[0].set_ylim([0, 1.0])
            axes[0].grid(axis='y', alpha=0.3)

            # Recall
            axes[1].bar(x, recall, width, color='#2ecc71', alpha=0.8)
            axes[1].set_ylabel('Recall', fontsize=12)
            axes[1].set_title('Recall by Class', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[1].set_ylim([0, 1.0])
            axes[1].grid(axis='y', alpha=0.3)

            # F1-Score
            axes[2].bar(x, f1, width, color='#e74c3c', alpha=0.8)
            axes[2].set_ylabel('F1-Score', fontsize=12)
            axes[2].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[2].set_ylim([0, 1.0])
            axes[2].grid(axis='y', alpha=0.3)

            plt.tight_layout()

            save_path = os.path.join(self.output_dir, get_timestamped_filename("error_per_class_metrics.png", self.task_type, self.timestamp))

            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            print(f"✓ Per-class metrics visualization saved: {save_path}\n")

        result = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'best_class': self.class_names[best_f1_idx],
            'worst_class': self.class_names[worst_f1_idx]
        }

        self.analysis_results['per_class'] = result
        return result


    # =========================================================================
    # MODULE 3: CONFIDENCE ANALYSIS
    # =========================================================================

    def analyze_confidence(self, save_visualizations: bool = True) -> Dict:
        """
        Module 3: Confidence score analysis for correct vs incorrect predictions.

        Returns:
            Dictionary with confidence analysis
        """
        print_banner("MODULE 3: CONFIDENCE ANALYSIS", width=60)

        # Separate correct and incorrect predictions
        correct_mask = self.predictions == self.true_labels
        incorrect_mask = ~correct_mask

        correct_confidences = self.confidence_scores[correct_mask]
        incorrect_confidences = self.confidence_scores[incorrect_mask]

        # Statistics
        print(f"Correct Predictions:")
        print(f"  Count: {len(correct_confidences)}")
        if len(correct_confidences) > 0:
            print(f"  Mean confidence: {correct_confidences.mean():.4f}")
            print(f"  Std: {correct_confidences.std():.4f}")
            print(f"  Median: {np.median(correct_confidences):.4f}")
            print(f"  Min: {correct_confidences.min():.4f}")
            print(f"  Max: {correct_confidences.max():.4f}")
        else:
            print(f"  No correct predictions to analyze")
        print()

        print(f"Incorrect Predictions:")
        print(f"  Count: {len(incorrect_confidences)}")
        if len(incorrect_confidences) > 0:
            print(f"  Mean confidence: {incorrect_confidences.mean():.4f}")
            print(f"  Std: {incorrect_confidences.std():.4f}")
            print(f"  Median: {np.median(incorrect_confidences):.4f}")
            print(f"  Min: {incorrect_confidences.min():.4f}")
            print(f"  Max: {incorrect_confidences.max():.4f}")
        else:
            print(f"  No incorrect predictions to analyze")
        print()

        # Identify "confident mistakes" (high confidence but wrong) - using config
        high_confidence_threshold = ERROR_ANALYSIS_CONFIG['high_confidence_threshold']
        confident_mistakes = (incorrect_mask) & (self.confidence_scores > high_confidence_threshold)
        num_confident_mistakes = confident_mistakes.sum()

        # Use safe_divide for percentage
        confident_mistake_pct = safe_divide(num_confident_mistakes, len(incorrect_confidences), default=0.0) * 100
        
        print(f"Confident Mistakes (confidence > {high_confidence_threshold}):")
        print(f"  Count: {num_confident_mistakes}")
        print(f"  Percentage of errors: {confident_mistake_pct:.2f}%")
        print()

        # Visualize if requested
        if save_visualizations:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            axes[0].hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='#2ecc71', density=True)
            axes[0].hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='#e74c3c', density=True)
            axes[0].set_xlabel('Confidence Score', fontsize=12)
            axes[0].set_ylabel('Density', fontsize=12)
            axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Box plot
            axes[1].boxplot([correct_confidences, incorrect_confidences],
                           labels=['Correct', 'Incorrect'],
                           patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            axes[1].set_ylabel('Confidence Score', fontsize=12)
            axes[1].set_title('Confidence Distribution Comparison', fontsize=14, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)

            plt.tight_layout()

            save_path = os.path.join(self.output_dir, get_timestamped_filename("error_confidence.png", self.task_type, self.timestamp))

            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            print(f"✓ Confidence visualization saved: {save_path}\n")

        result = {
            'correct_confidences': {
                'mean': float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0.0,
                'std': float(correct_confidences.std()) if len(correct_confidences) > 0 else 0.0,
                'median': float(np.median(correct_confidences)) if len(correct_confidences) > 0 else 0.0
            },
            'incorrect_confidences': {
                'mean': float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else 0.0,
                'std': float(incorrect_confidences.std()) if len(incorrect_confidences) > 0 else 0.0,
                'median': float(np.median(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0
            },
            'confident_mistakes': {
                'count': int(num_confident_mistakes),
                'percentage': float(safe_divide(num_confident_mistakes, len(incorrect_confidences), default=0.0) * 100)
            }
        }

        self.analysis_results['confidence'] = result
        return result


    # =========================================================================
    # MODULE 4: INPUT LENGTH ANALYSIS
    # =========================================================================

    def analyze_input_length(self, save_visualizations: bool = True) -> Dict:
        """
        Module 4: Analyze accuracy by input length.

        Returns:
            Dictionary with length analysis
        """
        print_banner("MODULE 4: INPUT LENGTH ANALYSIS", width=60)

        # Define length bins (using config)
        config_bins = ERROR_ANALYSIS_CONFIG['length_bins']
        bins = config_bins + [max(self.token_lengths)+1]  # Add max+1 for upper bound
        # Generate labels dynamically
        bin_labels = [f"{bins[i]}-{bins[i+1]-1}" if i < len(bins)-2 else f"{bins[i]}+"
                      for i in range(len(bins)-1)]

        # Assign each sample to a bin
        length_bins = np.digitize(self.token_lengths, bins) - 1

        # Calculate accuracy per bin
        print(f"{'Length Range':<15} {'Count':<10} {'Accuracy':<12} {'Avg Confidence':<15}")
        print(f"{'-'*55}")

        bin_accuracies = []
        bin_counts = []
        bin_confidences = []

        for i, label in enumerate(bin_labels):
            mask = length_bins == i
            count = mask.sum()

            if count > 0:
                accuracy = (self.predictions[mask] == self.true_labels[mask]).mean()
                avg_confidence = self.confidence_scores[mask].mean()
                bin_accuracies.append(accuracy)
                bin_counts.append(count)
                bin_confidences.append(avg_confidence)
                print(f"{label:<15} {count:<10} {accuracy:<12.4f} {avg_confidence:<15.4f}")
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
                bin_confidences.append(0)
                print(f"{label:<15} {count:<10} {'N/A':<12} {'N/A':<15}")

        print()

        # Identify trends
        best_length_idx = np.argmax(bin_accuracies)
        worst_length_idx = np.argmax([acc if count > 10 else 0 for acc, count in zip(bin_accuracies, bin_counts)])

        if bin_counts[best_length_idx] > 0:
            print(f"Best performing length range:  {bin_labels[best_length_idx]} (Accuracy: {bin_accuracies[best_length_idx]:.4f})")
        if bin_counts[worst_length_idx] > 0:
            print(f"Worst performing length range: {bin_labels[worst_length_idx]} (Accuracy: {bin_accuracies[worst_length_idx]:.4f})")
        print()

        # Visualize if requested
        if save_visualizations:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Bar plot - Accuracy by length
            x = np.arange(len(bin_labels))
            axes[0].bar(x, bin_accuracies, color='#3498db', alpha=0.8)
            axes[0].set_xlabel('Token Length Range', fontsize=12)
            axes[0].set_ylabel('Accuracy', fontsize=12)
            axes[0].set_title('Accuracy by Input Length', fontsize=14, fontweight='bold')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(bin_labels, rotation=45, ha='right')
            axes[0].set_ylim([0, 1.0])
            axes[0].grid(axis='y', alpha=0.3)

            # Add count labels on bars
            for i, (acc, count) in enumerate(zip(bin_accuracies, bin_counts)):
                if count > 0:
                    axes[0].text(i, acc + 0.02, f'n={count}', ha='center', va='bottom', fontsize=9)

            # Scatter plot - Length vs Confidence
            correct_mask = self.predictions == self.true_labels
            axes[1].scatter(self.token_lengths[correct_mask], self.confidence_scores[correct_mask],
                          alpha=0.3, s=10, label='Correct', color='#2ecc71')
            axes[1].scatter(self.token_lengths[~correct_mask], self.confidence_scores[~correct_mask],
                          alpha=0.5, s=10, label='Incorrect', color='#e74c3c')
            axes[1].set_xlabel('Token Length', fontsize=12)
            axes[1].set_ylabel('Confidence Score', fontsize=12)
            axes[1].set_title('Token Length vs Confidence', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()

            save_path = os.path.join(self.output_dir, get_timestamped_filename("error_length_analysis.png", self.task_type, self.timestamp))

            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            print(f"✓ Length analysis visualization saved: {save_path}\n")

        result = {
            'bin_labels': bin_labels,
            'bin_accuracies': [float(a) for a in bin_accuracies],
            'bin_counts': [int(c) for c in bin_counts],
            'bin_confidences': [float(c) for c in bin_confidences]
        }

        self.analysis_results['length'] = result
        return result


    # =========================================================================
    # MODULE 5: FAILURE CASE EXTRACTION
    # =========================================================================

    def extract_failure_cases(self, top_k: int = None, save_to_csv: bool = True) -> pd.DataFrame:
        """
        Module 5: Extract and analyze top failure cases.

        Args:
            top_k: Number of failure cases to extract (default: from config)
            save_to_csv: Whether to save to CSV

        Returns:
            DataFrame with failure cases
        """
        # Use config value if not provided
        if top_k is None:
            top_k = ERROR_ANALYSIS_CONFIG['top_k_errors']

        print_banner(f"MODULE 5: FAILURE CASE EXTRACTION (Top {top_k})", width=60)

        # Get incorrect predictions
        incorrect_mask = self.predictions != self.true_labels

        # Extract failure case data
        failure_data = {
            'text': [self.texts[i] for i in range(len(self.texts)) if incorrect_mask[i]],
            'true_label': [self.class_names[self.true_labels[i]] for i in range(len(self.true_labels)) if incorrect_mask[i]],
            'predicted_label': [self.class_names[self.predictions[i]] for i in range(len(self.predictions)) if incorrect_mask[i]],
            'confidence': self.confidence_scores[incorrect_mask],
            'token_length': self.token_lengths[incorrect_mask]
        }

        # Create DataFrame
        df = pd.DataFrame(failure_data)

        # Sort by confidence (most confident mistakes first)
        df = df.sort_values('confidence', ascending=False).reset_index(drop=True)

        # Take top k
        df_top = df.head(top_k)

        print(f"Total failures: {len(df)}")
        print(f"Showing top {min(top_k, len(df))} most confident mistakes\n")

        # Print summary
        print(f"Top 10 Confident Mistakes:")
        print(f"{'-'*60}")
        for i, row in df_top.head(10).iterrows():
            print(f"\n{i+1}. Confidence: {row['confidence']:.4f}")
            print(f"   True: {row['true_label']} | Predicted: {row['predicted_label']}")
            print(f"   Length: {row['token_length']} tokens")
            print(f"   Text: {row['text'][:100]}...")

        print()

        # Save to CSV if requested
        if save_to_csv:
            csv_path = os.path.join(self.output_dir, f"{self.task_type}_failure_cases_top{top_k}_{self.timestamp}.csv")

            df_top.to_csv(csv_path, index=False)
            print(f"✓ Failure cases saved to CSV: {csv_path}\n")

        self.analysis_results['failure_cases'] = df_top

        return df_top


    # =========================================================================
    # MODULE 6: TOKEN-LEVEL ATTRIBUTION (Simplified)
    # =========================================================================

    def analyze_token_attribution(self, num_samples: int = 10) -> Dict:
        """
        Module 6: Token-level attribution for failure cases using attention.

        Note: Full SHAP analysis is in ShapAnalyzer.py
        This provides simplified attention-based attribution.

        Args:
            num_samples: Number of failure cases to analyze

        Returns:
            Dictionary with attribution analysis
        """
        print_banner(f"MODULE 6: TOKEN-LEVEL ATTRIBUTION ({num_samples} samples)", width=60)

        print("⚠️  Note: This module provides simplified attention-based attribution.")
        print("   For full SHAP analysis, use ShapAnalyzer.py\n")

        # Get failure cases
        incorrect_mask = self.predictions != self.true_labels
        incorrect_indices = np.where(incorrect_mask)[0]

        if len(incorrect_indices) == 0:
            print("✓ No failures to analyze - model achieved 100% accuracy!")
            return {'message': 'No failures to analyze', 'failure_count': 0}

        # Sort by confidence (most confident mistakes)
        sorted_indices = incorrect_indices[np.argsort(self.confidence_scores[incorrect_mask])[::-1]]
        sample_indices = sorted_indices[:num_samples]

        attribution_results = []

        self.model.eval()

        for idx in sample_indices:
            # Get input
            sample = self.dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                pred_class = torch.argmax(logits, dim=1).item()

            # Decode text
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            # Store result
            attribution_results.append({
                'text': text,
                'tokens': tokens,
                'true_label': self.class_names[self.true_labels[idx]],
                'predicted_label': self.class_names[pred_class],
                'confidence': self.confidence_scores[idx]
            })

            print(f"Sample {len(attribution_results)}:")
            print(f"  True: {self.class_names[self.true_labels[idx]]}")
            print(f"  Predicted: {self.class_names[pred_class]} (confidence: {self.confidence_scores[idx]:.4f})")
            print(f"  Text: {text[:100]}...")
            print()

        print(f"✓ Analyzed {len(attribution_results)} failure cases")
        print(f"  For detailed SHAP attribution, run ShapAnalyzer.py\n")

        result = {
            'num_samples': len(attribution_results),
            'samples': attribution_results
        }

        self.analysis_results['attribution'] = result
        return result


    # =========================================================================
    # COMPREHENSIVE ANALYSIS RUNNER
    # =========================================================================

    def run_full_analysis(self, save_visualizations: bool = True, top_k_failures: int = None) -> Dict:
        """
        Run all error analysis modules.

        Args:
            save_visualizations: Whether to save visualizations
            top_k_failures: Number of failure cases to extract (default: from config)

        Returns:
            Complete analysis results
        """
        # Use config value if not provided
        if top_k_failures is None:
            top_k_failures = ERROR_ANALYSIS_CONFIG['top_k_errors']

        print(f"\n{'#'*60}")
        print(f"COMPREHENSIVE ERROR ANALYSIS: {self.task_type.upper()}")
        print(f"{'#'*60}\n")

        # Run predictions
        if self.predictions is None:
            self.run_predictions()

        # Run all modules
        print_banner("RUNNING ALL ANALYSIS MODULES", width=60)

        self.analyze_confusion_matrix(save_visualizations)
        self.analyze_per_class_performance(save_visualizations)
        self.analyze_confidence(save_visualizations)
        self.analyze_input_length(save_visualizations)
        self.extract_failure_cases(top_k=top_k_failures, save_to_csv=True)
        self.analyze_token_attribution(num_samples=10)

        # Save results
        self.save_analysis_results()

        print_banner("ERROR ANALYSIS COMPLETE", width=60)

        return self.analysis_results


    def save_analysis_results(self, output_path: str = None) -> str:
        """
        Save analysis results to pickle file.

        Args:
            output_path: Path to save results

        Returns:
            Path to saved file
        """
        if output_path is None:
            # Get base experiment name (strip timestamp if present)
            exp_name = EXPERIMENT_CONFIG['experiment_name']
            
            # Remove timestamp from experiment name if it has one
            # Pattern: dual_RoBERTa_classifier_20251117_1126 -> dual_RoBERTa_classifier
            parts = exp_name.split('_')
            if len(parts) >= 4:
                # Check if last 2 parts are timestamp (YYYYMMDD_HHMM)
                if len(parts[-2]) == 8 and parts[-2].isdigit() and len(parts[-1]) == 4 and parts[-1].isdigit():
                    # Remove timestamp parts
                    base_name = '_'.join(parts[:-2])
                else:
                    base_name = exp_name
            else:
                base_name = exp_name
            
            # Build filename with model timestamp (REQUIRED)
            if self.timestamp is None:
                raise ValueError(f"❌ ERROR: timestamp parameter is required for error analysis! Cannot create file without model training timestamp.")
            
            filename = f"{base_name}_{self.timestamp}_{self.task_type}_error_analysis.pkl"
            
            output_path = os.path.join(
                self.output_dir,
                filename
            )

        # Use ensure_dir_exists from Utils instead of os.makedirs
        ensure_dir_exists(os.path.dirname(output_path))

        with open(output_path, 'wb') as f:
            pickle.dump(self.analysis_results, f)

        print(f"✓ Error analysis results saved: {output_path}\n")

        return output_path


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_error_analysis(model, dataset, tokenizer, device, class_names, task_type='refusal', output_dir=None, timestamp=None) -> Dict:

    """
    Quick function to run complete error analysis.

    Args:
        model: Trained model
        dataset: Test dataset
        tokenizer: RoBERTa tokenizer
        device: torch device
        class_names: List of class names
        task_type: 'refusal' or 'jailbreak'
        output_dir: Directory to save visualizations (default: error_analysis_path)

    Returns:
        Complete analysis results
    """
    analyzer = ErrorAnalyzer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        class_names=class_names,
        task_type=task_type,
        output_dir=output_dir,
        timestamp=timestamp
    )

    # Use config value for top_k_failures
    results = analyzer.run_full_analysis(save_visualizations=True, top_k_failures=ERROR_ANALYSIS_CONFIG['top_k_errors'])

    return results


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 1, 2025
@author: ramyalsaffar
"""
