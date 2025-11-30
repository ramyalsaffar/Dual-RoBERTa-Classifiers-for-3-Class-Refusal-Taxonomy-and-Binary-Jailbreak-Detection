# Per-Model Analysis Module 
#---------------------------
# Analyze classifier performance per model with statistical testing.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Statistical significance testing between models
# - Confidence intervals for metrics
# - Better handling of jailbreak classification
# - Comprehensive reporting
# All imports are in 01-Imports.py
###############################################################################


class PerModelAnalyzer:
    """
    Enhanced analyzer for per-model performance with statistical testing.
    
    Features:
    - Per-model performance metrics
    - Statistical significance testing between models
    - Confidence intervals
    - Handles both refusal and jailbreak classification
    """
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer,
                 device: torch.device,
                 class_names: List[str] = None,
                 task_type: str = 'refusal'):
        """
        Initialize enhanced per-model analyzer.
        
        Args:
            model: Classification model (RefusalClassifier or JailbreakClassifier)
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names (uses config if None)
            task_type: 'refusal' or 'jailbreak' for proper handling
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.task_type = task_type
        
        # Set class names based on task type
        if class_names:
            self.class_names = class_names
        elif task_type == 'refusal':
            self.class_names = CLASS_NAMES  # From config
        else:  # jailbreak
            self.class_names = ['Failed', 'Succeeded']
        
        self.num_classes = len(self.class_names)
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        
        # For statistical testing
        self.alpha = HYPOTHESIS_TESTING_CONFIG.get('alpha', 0.05)
        self.confidence_level = HYPOTHESIS_TESTING_CONFIG.get('confidence_level', 0.95)
    
    def _get_model_short_key(self, model_name: str) -> str:
        """
        Convert full model name to short key for consistent storage.
        
        Args:
            model_name: Full model name (e.g., 'claude-sonnet-4-5-20250929')
            
        Returns:
            Short key (e.g., 'claude') or original name if no mapping found
        """
        # Build reverse mapping from API_CONFIG
        # 'claude-sonnet-4-5-20250929' -> 'claude'
        # 'gpt-5.1-2025-11-13' -> 'gpt5'
        reverse_map = {v: k for k, v in API_CONFIG['response_models'].items()}
        
        # Check if it's already a short key (like 'wildjailbreak')
        if model_name in API_CONFIG['response_models']:
            return model_name
        
        # Check WildJailbreak synthetic label
        if model_name == WILDJAILBREAK_CONFIG.get('synthetic_model_label', 'wildjailbreak'):
            return 'wildjailbreak'
        
        # Return mapped short key or original if not found
        return reverse_map.get(model_name, model_name)
    

    def analyze(self, test_df: pd.DataFrame) -> Dict:
        """
        Analyze performance per model with statistical testing.
        
        Args:
            test_df: Test dataframe with 'model' column and labels
        
        Returns:
            Dictionary with per-model results and statistical tests
        """
        # Get unique models (filter NaN)
        models = test_df['model'].dropna().unique()
        
        if len(models) == 0:
            print("❌ ERROR: No valid model names found in test data")
            return {}
        
        results = {
            'models': {},
            'statistical_tests': {},
            'summary': {}
        }
        
        if self.verbose:
            print_banner("PER-MODEL ANALYSIS", char="=")
            print(f"  Task: {self.task_type}")
            print(f"  Models to analyze: {len(models)}")
            print(f"  Total samples: {len(test_df):,}")
            print("=" * 60)
        
        # Analyze each model
        all_model_preds = {}  # Store for statistical testing
        
        for model_name in models:
            # Convert full model name to short key for consistent storage
            short_key = self._get_model_short_key(model_name)
            
            if self.verbose:
                print(f"\n📊 Evaluating {get_model_display_name(short_key)}...")
            
            model_df = test_df[test_df['model'] == model_name]

            
            # Get appropriate label column
            if self.task_type == 'refusal':
                label_col = 'refusal_label' if 'refusal_label' in model_df.columns else 'label'
            else:
                label_col = 'jailbreak_label' if 'jailbreak_label' in model_df.columns else 'label'
            
            # Filter out error labels (-1)
            valid_mask = model_df[label_col] != -1
            model_df = model_df[valid_mask]
            
            if len(model_df) == 0:
                if self.verbose:
                    print(f"  ⚠️  No valid samples for {model_name}, skipping...")
                continue
            
            # Create dataset and loader
            dataset = ClassificationDataset(
                texts=model_df['response'].tolist(),
                labels=model_df[label_col].tolist(),
                tokenizer=self.tokenizer,
                prompts=model_df['prompt'].tolist() if 'prompt' in model_df.columns else None,
                task_type=self.task_type,
                validate_labels=False  # Already validated
            )
            
            loader = DataLoader(
                dataset,
                batch_size=TRAINING_CONFIG.get('inference_batch_size', 32),
                shuffle=False,
                num_workers=0  # Use 0 for inference
            )
            
            
            # Evaluate
            preds, labels, confidences = self._evaluate(loader)
            all_model_preds[short_key] = (preds, labels)
            
            # Calculate metrics with confidence intervals
            metrics = self._calculate_metrics_with_ci(preds, labels, confidences)
            
            # Store results using short key for ReportGenerator compatibility
            results['models'][short_key] = {
                **metrics,
                'num_samples': len(model_df),
                'num_valid_samples': len(labels),
                'full_model_name': model_name  # Preserve full name for reference
            }
            
            if self.verbose:
                self._print_model_results(short_key, results['models'][short_key])
        
        
        # Statistical comparisons between models
        if len(results['models']) > 1:
            results['statistical_tests'] = self._compare_models_statistically(
                all_model_preds,
                results['models']
            )
        
        # Summary statistics
        results['summary'] = self._calculate_summary_stats(results['models'])
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _evaluate(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model and return predictions, labels, and confidences."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions with confidence
                result = self.model.predict_with_confidence(
                    input_ids,
                    attention_mask,
                    use_mc_dropout=False  # Faster for per-model analysis
                )
                
                all_preds.extend(result['predictions'].cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_confidences.extend(result['confidence'].cpu().numpy().tolist())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_confidences)
    
    def _calculate_metrics_with_ci(self, preds: np.ndarray, 
                                   labels: np.ndarray,
                                   confidences: np.ndarray) -> Dict:
        """Calculate metrics with confidence intervals using bootstrap."""
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        
        # Per-class F1
        label_indices = list(range(self.num_classes))
        f1_per_class = f1_score(labels, preds, average=None, 
                               labels=label_indices, zero_division=0)
        
        # Bootstrap confidence intervals for F1
        n_bootstrap = 1000
        f1_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(labels), len(labels), replace=True)
            boot_labels = labels[indices]
            boot_preds = preds[indices]
            
            f1_boot = f1_score(boot_labels, boot_preds, average='macro', zero_division=0)
            f1_bootstrap.append(f1_boot)
        
        # Calculate CI
        f1_bootstrap = np.array(f1_bootstrap)
        ci_lower = np.percentile(f1_bootstrap, (1 - self.confidence_level) * 100 / 2)
        ci_upper = np.percentile(f1_bootstrap, (1 + self.confidence_level) * 100 / 2)
        
        
        # Average confidence
        mean_confidence = np.mean(confidences)
        
        # Class distribution (count of samples per class)
        class_distribution = {}
        for i, class_name in enumerate(self.class_names):
            class_distribution[class_name] = int(np.sum(labels == i))
        
        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_macro_ci': (float(ci_lower), float(ci_upper)),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_per_class': {
                self.class_names[i]: float(f1_per_class[i])
                for i in range(len(self.class_names))
            },
            'mean_confidence': float(mean_confidence),
            'class_distribution': class_distribution
        }
    
    
    def _compare_models_statistically(self, all_model_preds: Dict,
                                     model_results: Dict) -> Dict:
        """Perform statistical tests between models."""
        
        comparisons = {}
        model_names = list(all_model_preds.keys())
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                preds1, labels1 = all_model_preds[model1]
                preds2, labels2 = all_model_preds[model2]
                
                # McNemar's test for paired samples (if same test set)
                if len(labels1) == len(labels2) and np.array_equal(labels1, labels2):
                    # Create contingency table
                    correct1_wrong2 = np.sum((preds1 == labels1) & (preds2 != labels2))
                    wrong1_correct2 = np.sum((preds1 != labels1) & (preds2 == labels2))
                    
                    # McNemar's test
                    if correct1_wrong2 + wrong1_correct2 > 0:
                        statistic = (abs(correct1_wrong2 - wrong1_correct2) - 1)**2 / (correct1_wrong2 + wrong1_correct2)
                        p_value = 1 - scipy_stats.chi2.cdf(statistic, df=1)
                    else:
                        p_value = 1.0
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    comparisons[comparison_key] = {
                        'test': 'McNemar',
                        'statistic': float(statistic) if correct1_wrong2 + wrong1_correct2 > 0 else 0.0,
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha,
                        'f1_diff': model_results[model1]['f1_macro'] - model_results[model2]['f1_macro']
                    }
        
        # Overall test (Friedman test if >2 models)
        if len(model_names) > 2:
            # Collect F1 scores
            f1_scores = [model_results[m]['f1_macro'] for m in model_names]
            
            # Friedman test (non-parametric)
            statistic, p_value = scipy_stats.friedmanchisquare(*[[f1] for f1 in f1_scores])
            
            comparisons['overall'] = {
                'test': 'Friedman',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'interpretation': 'Models differ significantly' if p_value < self.alpha else 'No significant difference'
            }
        
        return comparisons
    
    def _calculate_summary_stats(self, model_results: Dict) -> Dict:
        """Calculate summary statistics across models."""
        if not model_results:
            return {}
        
        f1_scores = [r['f1_macro'] for r in model_results.values()]
        accuracy_scores = [r['accuracy'] for r in model_results.values()]
        
        return {
            'f1_macro': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
                'range': float(np.max(f1_scores) - np.min(f1_scores))
            },
            'accuracy': {
                'mean': float(np.mean(accuracy_scores)),
                'std': float(np.std(accuracy_scores)),
                'min': float(np.min(accuracy_scores)),
                'max': float(np.max(accuracy_scores))
            },
            'hardest_model': min(model_results.items(), key=lambda x: x[1]['f1_macro'])[0],
            'easiest_model': max(model_results.items(), key=lambda x: x[1]['f1_macro'])[0],
            'num_models_analyzed': len(model_results)
        }
    
    def _print_model_results(self, model_name: str, results: Dict):
        """Print results for a single model."""
        print(f"  Results for {get_model_display_name(model_name)}:")
        print(f"    F1 (macro): {results['f1_macro']:.4f} "
              f"CI: [{results['f1_macro_ci'][0]:.4f}, {results['f1_macro_ci'][1]:.4f}]")
        print(f"    Accuracy: {results['accuracy']:.4f}")
        print(f"    Mean confidence: {results['mean_confidence']:.4f}")
        print(f"    Samples: {results['num_samples']:,}")
    
    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print_banner("ANALYSIS SUMMARY", char="=")
        
        summary = results['summary']
        print(f"Models analyzed: {summary['num_models_analyzed']}")
        print(f"\nF1 Score Statistics:")
        print(f"  Mean: {summary['f1_macro']['mean']:.4f}")
        print(f"  Std: {summary['f1_macro']['std']:.4f}")
        print(f"  Range: [{summary['f1_macro']['min']:.4f}, {summary['f1_macro']['max']:.4f}]")
        print(f"\nHardest model: {get_model_display_name(summary['hardest_model'])}")
        print(f"Easiest model: {get_model_display_name(summary['easiest_model'])}")
        
        # Statistical tests
        if 'statistical_tests' in results and results['statistical_tests']:
            print("\nStatistical Tests:")
            print("─" * 60)
            
            if 'overall' in results['statistical_tests']:
                overall = results['statistical_tests']['overall']
                print(f"Overall test ({overall['test']}):")
                print(f"  p-value: {overall['p_value']:.4f}")
                print(f"  Result: {overall['interpretation']}")
            
            # Pairwise comparisons
            pairwise = {k: v for k, v in results['statistical_tests'].items() if k != 'overall'}
            if pairwise:
                print("\nPairwise comparisons (McNemar's test):")
                for comparison, result in pairwise.items():
                    models = comparison.replace('_vs_', ' vs ')
                    sig = "✅ Significant" if result['significant'] else "❌ Not significant"
                    print(f"  {models}: p={result['p_value']:.4f} {sig}")
        
        print("=" * 60)
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save results to JSON."""
        if output_path is None:
            output_path = os.path.join(
                analysis_results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_per_model_results.json"
                #"per_model_results.json"
            )

        ensure_dir_exists(os.path.dirname(output_path))

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Saved per-model results to {output_path}")
        
        return output_path


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
