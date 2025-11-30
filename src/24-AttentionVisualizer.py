# Attention Visualizer Module
#----------------------------
# Visualizes RoBERTa attention weights to interpret model decisions.
# All imports are in 01-Imports.py
###############################################################################


class AttentionVisualizer:
    """Visualize attention weights from RoBERTa model."""

    def __init__(self, model, tokenizer, device, class_names: List[str] = None, task_type: str = 'refusal'):
        """
        Initialize attention visualizer.

        GENERIC: Works with any classifier (RefusalClassifier or JailbreakDetector).

        Args:
            model: Trained classification model (RefusalClassifier or JailbreakDetector)
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names (default: uses CLASS_NAMES from config)
            task_type: 'refusal' or 'jailbreak' for proper label column detection
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.dpi = VISUALIZATION_CONFIG['dpi']
        self.task_type = task_type
        self.model.eval()

    def get_attention_weights(self, text: str):
        """
        Get attention weights for a text input.

        Args:
            text: Input text

        Returns:
            Dictionary with tokens, attentions, prediction
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=MODEL_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get predictions and attention weights
        with torch.no_grad():
            result = self.model(
                input_ids,
                attention_mask,
                output_attentions=True
            )

            # Extract from dictionary
            logits = result['logits']
            attentions = result['attentions']

            # Get prediction
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        # Convert tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Get actual token count (before padding)
        actual_length = attention_mask[0].sum().item()
        tokens = tokens[:actual_length]

        # Process attention weights
        # attentions is a tuple of (num_layers,) where each is (batch, num_heads, seq_len, seq_len)
        attention_data = []
        for layer_idx, layer_attention in enumerate(attentions):
            # Average across heads
            layer_attention = layer_attention[0].mean(dim=0)  # (seq_len, seq_len)

            # Focus on [CLS] token attention (which is used for classification)
            cls_attention = layer_attention[0, :actual_length].cpu().numpy()

            attention_data.append({
                'layer': layer_idx,
                'cls_attention': cls_attention
            })

        return {
            'tokens': tokens,
            'attention_by_layer': attention_data,
            'predicted_class': predicted.item(),
            'confidence': confidence.item(),
            'probabilities': probs[0].cpu().numpy()
        }

    def visualize_attention(self, text: str, output_path: str,
                           layer_idx: int = -1, top_k: int = None):
        """
        Create attention visualization for a single text.

        Args:
            text: Input text to visualize
            output_path: Path to save visualization
            layer_idx: Which layer to visualize (-1 for last layer)
            top_k: Number of top tokens to highlight (default: from config)
        """
        # Use config value if not provided
        if top_k is None:
            top_k = INTERPRETABILITY_CONFIG['attention_top_k_tokens']

        # Get attention data
        attention_data = self.get_attention_weights(text)
        tokens = attention_data['tokens']
        predicted = attention_data['predicted_class']
        confidence = attention_data['confidence']

        # Get attention for specified layer
        if layer_idx == -1:
            layer_idx = len(attention_data['attention_by_layer']) - 1

        cls_attention = attention_data['attention_by_layer'][layer_idx]['cls_attention']

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Attention heatmap
        ax1.barh(range(len(tokens)), cls_attention, color=PLOT_COLORS['soft_refusal'])
        ax1.set_yticks(range(len(tokens)))
        ax1.set_yticklabels(tokens, fontsize=9)
        ax1.set_xlabel('Attention Weight', fontsize=11)
        ax1.set_title(
            f'Layer {layer_idx} - [CLS] Token Attention\n'
            f'Predicted: {self.class_names[predicted]} (confidence: {confidence:.3f})',
            fontsize=12, fontweight='bold'
        )
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Top-K tokens
        top_indices = np.argsort(cls_attention)[-top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_weights = [cls_attention[i] for i in top_indices]

        colors = [PLOT_COLORS['hard_refusal'] if w > np.mean(cls_attention)
                 else PLOT_COLORS['no_refusal'] for w in top_weights]

        ax2.barh(range(len(top_tokens)), top_weights, color=colors)
        ax2.set_yticks(range(len(top_tokens)))
        ax2.set_yticklabels(top_tokens, fontsize=10, fontweight='bold')
        ax2.set_xlabel('Attention Weight', fontsize=11)
        ax2.set_title(f'Top {top_k} Most Attended Tokens', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved attention visualization to {output_path}")

    def analyze_samples(self, test_df: pd.DataFrame, num_samples: int = None, output_dir: str = None, timestamp: str = None, classifier_name: str = None):

        """
        Analyze attention patterns for multiple samples.

        GENERIC: Works with any number of classes (2, 3, or more).

        Args:
            test_df: Test DataFrame
            num_samples: Number of samples per class to analyze (default: from config)
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with analysis results
        """
        # Use config value if not provided
        if num_samples is None:
            num_samples = INTERPRETABILITY_CONFIG['attention_samples_per_class']

        if output_dir is None:
            output_dir = attention_analysis_path
        ensure_dir_exists(output_dir)

        print_banner("ATTENTION ANALYSIS", width=60, char="=")

        # Smart label column detection (same logic as ConfidenceAnalyzer)
        if self.task_type == 'refusal':
            label_col = 'refusal_label' if 'refusal_label' in test_df.columns else 'label'
        else:
            label_col = 'jailbreak_label' if 'jailbreak_label' in test_df.columns else 'label'

        results = {'by_class': {}, 'examples': []}

        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            print(f"\nAnalyzing {class_name} samples...")

            # Get samples from this class
            class_samples = test_df[test_df[label_col] == class_idx]

            if len(class_samples) == 0:
                continue

            # Sample randomly
            sample_indices = np.random.choice(
                len(class_samples),
                min(num_samples, len(class_samples)),
                replace=False
            )

            class_attention_stats = []

            # Progress bar for analyzing samples in this class
            for i, idx in enumerate(tqdm(sample_indices, desc=f"Analyzing {class_name}", leave=False)):
                row = class_samples.iloc[idx]
                text = row['response']

                # Validate text is a string and not NaN
                if not isinstance(text, str) or pd.isna(text) or len(text) == 0:
                    print(f"\n⚠️  Skipping invalid text at index {idx} (NaN or empty)")
                    continue

                # Get attention
                attention_data = self.get_attention_weights(text)
                
                
                # Save visualization for first 3 samples
                if i < 3:
                    output_path = os.path.join(
                        output_dir,
                        f"attention_{class_name.replace(' ', '_').lower()}_sample_{i+1}_{timestamp}.png" if timestamp else f"attention_{class_name.replace(' ', '_').lower()}_sample_{i+1}.png"
                    )
                    self.visualize_attention(text, output_path)

                # Collect stats
                last_layer_attention = attention_data['attention_by_layer'][-1]['cls_attention']
                class_attention_stats.append({
                    'mean_attention': np.mean(last_layer_attention),
                    'max_attention': np.max(last_layer_attention),
                    'entropy': -np.sum(last_layer_attention * np.log(last_layer_attention + 1e-10))
                })

                results['examples'].append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'true_label': class_name,
                    'predicted_label': self.class_names[attention_data['predicted_class']],
                    'confidence': float(attention_data['confidence']),
                    'tokens': attention_data['tokens'][:20]  # First 20 tokens
                })

            # Aggregate stats
            if class_attention_stats:
                results['by_class'][class_name] = {
                    'avg_mean_attention': float(np.mean([s['mean_attention'] for s in class_attention_stats])),
                    'avg_max_attention': float(np.mean([s['max_attention'] for s in class_attention_stats])),
                    'avg_entropy': float(np.mean([s['entropy'] for s in class_attention_stats]))
                }
                
                print(f"  Mean attention: {results['by_class'][class_name]['avg_mean_attention']:.4f}")
                print(f"  Max attention: {results['by_class'][class_name]['avg_max_attention']:.4f}")
                print(f"  Attention entropy: {results['by_class'][class_name]['avg_entropy']:.4f}")
            else:
                print(f"  ⚠️  No valid attention data for {class_name}")
                results['by_class'][class_name] = {
                    'avg_mean_attention': 0.0,
                    'avg_max_attention': 0.0,
                    'avg_entropy': 0.0
                }

        # Save results
        results_json_path = os.path.join(output_dir, f"{classifier_name}_attention_analysis_results_{timestamp}.json")

        serializable_results = convert_to_serializable(results)

        with open(results_json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Saved attention analysis to {results_json_path}")

        return results

    def compare_layers(self, text: str, output_path: str):
        """
        Compare attention patterns across all layers.

        Args:
            text: Input text
            output_path: Path to save comparison plot
        """
        attention_data = self.get_attention_weights(text)
        tokens = attention_data['tokens']
        predicted = attention_data['predicted_class']

        # Get number of layers
        num_layers = len(attention_data['attention_by_layer'])

        # Create figure
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for layer_idx in range(min(num_layers, 12)):  # Show first 12 layers
            cls_attention = attention_data['attention_by_layer'][layer_idx]['cls_attention']

            ax = axes[layer_idx]
            ax.barh(range(len(tokens)), cls_attention, color=PLOT_COLORS['soft_refusal'], alpha=0.7)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=7)
            ax.set_xlabel('Attention', fontsize=8)
            ax.set_title(f'Layer {layer_idx}', fontsize=9, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle(
            f'Attention Across All Layers\nPredicted: {self.class_names[predicted]}',
            fontsize=14, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved layer comparison to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
