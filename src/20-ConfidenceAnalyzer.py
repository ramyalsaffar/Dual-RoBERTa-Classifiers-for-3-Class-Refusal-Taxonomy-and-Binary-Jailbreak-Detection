# Confidence Analysis Module 
#----------------------------
# Enhanced confidence analysis with calibration metrics and statistical testing.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
# - Statistical testing for confidence differences
# - Reliability diagrams data
# - Support for jailbreak classification
# - Monte Carlo dropout for uncertainty estimation
# All imports are in 01-Imports.py
###############################################################################


class ConfidenceAnalyzer:
    """
    Enhanced confidence analyzer with calibration metrics and statistical testing.
    
    Features:
    - Confidence calibration metrics (ECE, MCE)
    - Statistical testing for confidence differences
    - Reliability diagram data
    - Monte Carlo dropout for uncertainty
    - Support for both refusal and jailbreak tasks
    """
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 device: torch.device,
                 class_names: List[str] = None,
                 task_type: str = 'refusal'):
        """
        Initialize enhanced confidence analyzer.
        
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
            self.class_names = CLASS_NAMES
        else:  # jailbreak
            self.class_names = ['Failed', 'Succeeded']
        
        self.num_classes = len(self.class_names)
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        
        # Statistical testing parameters
        self.alpha = HYPOTHESIS_TESTING_CONFIG.get('alpha', 0.05)
        self.confidence_level = HYPOTHESIS_TESTING_CONFIG.get('confidence_level', 0.95)
    
    def analyze(self, test_df: pd.DataFrame, use_mc_dropout: bool = False) -> Dict:
        """
        Analyze confidence distributions with calibration metrics.
        
        Args:
            test_df: Test dataframe with labels
            use_mc_dropout: Use Monte Carlo dropout for uncertainty estimation
        
        Returns:
            Dictionary with confidence metrics and calibration analysis
        """
        if self.verbose:
            print_banner("CONFIDENCE ANALYSIS", char="=")
            print(f"  Task: {self.task_type}")
            print(f"  Samples: {len(test_df):,}")
            print(f"  MC Dropout: {use_mc_dropout}")
            print("=" * 60)
        
        # Filter for jailbreak attempts if analyzing jailbreak classifier
        if self.task_type == 'jailbreak' and 'is_jailbreak_attempt' in test_df.columns:
            test_df = test_df[test_df['is_jailbreak_attempt'] == 1].copy()
            if self.verbose:
                print(f"  Filtered to {len(test_df):,} jailbreak attempts")
        
        # Get appropriate label column
        if self.task_type == 'refusal':
            label_col = 'refusal_label' if 'refusal_label' in test_df.columns else 'label'
        else:
            label_col = 'jailbreak_label' if 'jailbreak_label' in test_df.columns else 'label'
        
        # Filter out error labels
        valid_mask = test_df[label_col] != -1
        test_df = test_df[valid_mask].copy()
        
        # Create dataset
        dataset = ClassificationDataset(
            texts=test_df['response'].tolist(),
            labels=test_df[label_col].tolist(),
            tokenizer=self.tokenizer,
            prompts=test_df['prompt'].tolist() if 'prompt' in test_df.columns else None,
            task_type=self.task_type,
            validate_labels=False
        )
        
        loader = DataLoader(
            dataset,
            batch_size=TRAINING_CONFIG.get('inference_batch_size', 32),
            shuffle=False,
            num_workers=0
        )
        
        # Get predictions with confidence
        if use_mc_dropout:
            eval_results = self._evaluate_with_uncertainty(loader)
        else:
            eval_results = self._evaluate(loader)
        
        preds = eval_results['predictions']
        labels = eval_results['labels']
        confidences = eval_results['confidences']
        all_probs = eval_results['probabilities']
        
        # Calculate metrics
        results = {
            'basic_metrics': self._calculate_basic_metrics(preds, labels, confidences),
            'calibration_metrics': self._calculate_calibration_metrics(preds, labels, confidences, all_probs),
            'per_class_metrics': self._calculate_per_class_metrics(preds, labels, confidences),
            'statistical_tests': self._perform_statistical_tests(preds, labels, confidences),
            'reliability_diagram': self._generate_reliability_diagram_data(preds, labels, confidences)
        }
        
        # Add uncertainty metrics if MC dropout was used
        if use_mc_dropout:
            results['uncertainty_metrics'] = {
                'mean_epistemic': float(np.mean(eval_results['epistemic_uncertainty'])),
                'mean_aleatoric': float(np.mean(eval_results['aleatoric_uncertainty'])),
                'mean_total': float(np.mean(eval_results['total_uncertainty']))
            }
        
        if self.verbose:
            self._print_analysis_results(results)
        
        return results
    
    def _evaluate(self, loader: DataLoader) -> Dict:
        """Standard evaluation with confidence scores."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                result = self.model.predict_with_confidence(
                    input_ids, attention_mask,
                    use_mc_dropout=False
                )
                
                all_preds.extend(result['predictions'].cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_confidences.extend(result['confidence'].cpu().numpy().tolist())
                all_probs.extend(result['probabilities'].cpu().numpy().tolist())
        
        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'confidences': np.array(all_confidences),
            'probabilities': np.array(all_probs)
        }
    
    def _evaluate_with_uncertainty(self, loader: DataLoader) -> Dict:
        """Evaluation with MC dropout for uncertainty estimation."""
        self.model.eval()  # Will be set to train mode in predict_with_confidence
        
        all_results = {
            'predictions': [],
            'labels': [],
            'confidences': [],
            'probabilities': [],
            'epistemic_uncertainty': [],
            'aleatoric_uncertainty': [],
            'total_uncertainty': []
        }
        
        for batch in tqdm(loader, desc="  MC Dropout evaluation", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            result = self.model.predict_with_confidence(
                input_ids, attention_mask,
                use_mc_dropout=True,
                n_samples=20  # More samples for better uncertainty estimation
            )
            
            all_results['predictions'].extend(result['predictions'].cpu().numpy().tolist())
            all_results['labels'].extend(labels.cpu().numpy().tolist())
            all_results['confidences'].extend(result['confidence'].cpu().numpy().tolist())
            all_results['probabilities'].extend(result['probabilities'].cpu().numpy().tolist())
            all_results['epistemic_uncertainty'].extend(result['epistemic_uncertainty'].cpu().numpy().tolist())
            all_results['aleatoric_uncertainty'].extend(result['aleatoric_uncertainty'].cpu().numpy().tolist())
            all_results['total_uncertainty'].extend(result['total_uncertainty'].cpu().numpy().tolist())
        
        # Convert to arrays
        for key in all_results:
            all_results[key] = np.array(all_results[key])
        
        return all_results
    
    def _calculate_basic_metrics(self, preds: np.ndarray, labels: np.ndarray,
                                confidences: np.ndarray) -> Dict:
        """Calculate basic confidence metrics."""
        correct = preds == labels
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(labels, preds)
        
        # Separate correct/incorrect confidences
        correct_confidences = confidences[correct]
        incorrect_confidences = confidences[~correct]
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_confidence_correct': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
            'mean_confidence_incorrect': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
            'confidence_gap': float(np.mean(correct_confidences) - np.mean(incorrect_confidences)) if len(correct_confidences) > 0 and len(incorrect_confidences) > 0 else 0.0,
            'cohen_kappa': float(kappa),
            'accuracy': float(np.mean(correct))
        }
    
    def _calculate_calibration_metrics(self, preds: np.ndarray, labels: np.ndarray,
                                      confidences: np.ndarray, probabilities: np.ndarray) -> Dict:
        """Calculate calibration metrics (ECE and MCE)."""
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (preds[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                bin_accuracies.append(float(accuracy_in_bin))
                bin_confidences.append(float(avg_confidence_in_bin))
                bin_counts.append(int(in_bin.sum()))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)
        
        # Brier Score (proper scoring rule)
        brier_score = 0.0
        for i in range(len(labels)):
            true_class_prob = probabilities[i, labels[i]]
            brier_score += (1 - true_class_prob) ** 2
        brier_score /= len(labels)
        
        # Log Loss (cross-entropy loss)
        # log_loss = -1/N * sum(log(P(true_class)))
        log_loss_value = 0.0
        epsilon = 1e-15  # Small value to avoid log(0)
        for i in range(len(labels)):
            true_class_prob = probabilities[i, labels[i]]
            # Clip probability to avoid log(0)
            true_class_prob = np.clip(true_class_prob, epsilon, 1 - epsilon)
            log_loss_value -= np.log(true_class_prob)
        log_loss_value /= len(labels)
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier_score),
            'log_loss': float(log_loss_value),  # NEW!
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'n_bins': n_bins
        }
    
    def _calculate_per_class_metrics(self, preds: np.ndarray, labels: np.ndarray,
                                    confidences: np.ndarray) -> Dict:
        """Calculate per-class confidence metrics."""
        per_class = {}
        
        for class_idx in range(self.num_classes):
            # True positives for this class
            class_mask = labels == class_idx
            pred_mask = preds == class_idx
            
            if class_mask.sum() > 0:
                class_confidences = confidences[class_mask]
                class_correct = (preds[class_mask] == class_idx)
                
                per_class[self.class_names[class_idx]] = {
                    'mean_confidence': float(np.mean(class_confidences)),
                    'std_confidence': float(np.std(class_confidences)),
                    'min_confidence': float(np.min(class_confidences)),
                    'max_confidence': float(np.max(class_confidences)),
                    'accuracy': float(np.mean(class_correct)),
                    'n_samples': int(class_mask.sum())
                }
        
        return per_class
    
    def _perform_statistical_tests(self, preds: np.ndarray, labels: np.ndarray,
                                  confidences: np.ndarray) -> Dict:
        """Perform statistical tests on confidence scores."""
        
        correct = preds == labels
        correct_confidences = confidences[correct]
        incorrect_confidences = confidences[~correct]
        
        tests = {}
        
        # Test if correct predictions have significantly higher confidence
        if len(correct_confidences) > 0 and len(incorrect_confidences) > 0:
            t_stat, p_value = scipy_stats.ttest_ind(
                correct_confidences,
                incorrect_confidences,
                alternative='greater'
            )
            
            variance_sum = np.var(correct_confidences) + np.var(incorrect_confidences)
            if variance_sum > 0:
                effect_size = float((np.mean(correct_confidences) - np.mean(incorrect_confidences)) / 
                                   np.sqrt(variance_sum / 2))
            else:
                effect_size = 0.0  # No variance means no measurable effect
            
            tests['correct_vs_incorrect'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'effect_size': effect_size
            }
        
        # Test if confidences are well-calibrated (uniformly distributed)
        # Kolmogorov-Smirnov test against uniform distribution
        ks_stat, ks_p = scipy_stats.kstest(confidences, 'uniform')
        
        tests['calibration_uniformity'] = {
            'ks_statistic': float(ks_stat),
            'p_value': float(ks_p),
            'well_calibrated': ks_p > self.alpha
        }
        
        return tests
    
    def _generate_reliability_diagram_data(self, preds: np.ndarray, labels: np.ndarray,
                                          confidences: np.ndarray) -> Dict:
        """Generate data for reliability diagram plotting."""
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        reliability_data = {
            'bin_boundaries': bin_boundaries.tolist(),
            'bin_centers': [],
            'bin_accuracies': [],
            'bin_mean_confidences': [],
            'bin_counts': [],
            'perfect_calibration': []  # y=x line
        }
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            bin_center = (bin_lower + bin_upper) / 2
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = (preds[in_bin] == labels[in_bin]).mean()
                bin_mean_conf = confidences[in_bin].mean()
                bin_count = in_bin.sum()
            else:
                bin_accuracy = bin_center
                bin_mean_conf = bin_center
                bin_count = 0
            
            reliability_data['bin_centers'].append(float(bin_center))
            reliability_data['bin_accuracies'].append(float(bin_accuracy))
            reliability_data['bin_mean_confidences'].append(float(bin_mean_conf))
            reliability_data['bin_counts'].append(int(bin_count))
            reliability_data['perfect_calibration'].append(float(bin_center))
        
        return reliability_data
    
    def _print_analysis_results(self, results: Dict):
        """Print confidence analysis results."""
        basic = results['basic_metrics']
        calib = results['calibration_metrics']
        
        print("\n📊 Basic Metrics:")
        print("─" * 60)
        print(f"  Mean Confidence: {basic['mean_confidence']:.4f} ± {basic['std_confidence']:.4f}")
        print(f"  Confidence (Correct): {basic['mean_confidence_correct']:.4f}")
        print(f"  Confidence (Incorrect): {basic['mean_confidence_incorrect']:.4f}")
        print(f"  Confidence Gap: {basic['confidence_gap']:.4f}")
        print(f"  Cohen's Kappa: {basic['cohen_kappa']:.4f}")
        
        # Interpret Kappa
        kappa = basic['cohen_kappa']
        kappa_thresh = INTERPRETABILITY_CONFIG['kappa_thresholds']
        if kappa > kappa_thresh['almost_perfect']:
            print(f"    ✅ Almost perfect agreement")
        elif kappa > kappa_thresh['substantial']:
            print(f"    ✅ Substantial agreement")
        elif kappa > kappa_thresh['moderate']:
            print(f"    ⚠️  Moderate agreement")
        elif kappa > kappa_thresh['fair']:
            print(f"    ⚠️  Fair agreement")
        else:
            print(f"    🚨 Poor agreement")
        
        print("\n📈 Calibration Metrics:")
        print("─" * 60)
        print(f"  Expected Calibration Error (ECE): {calib['ece']:.4f}")
        print(f"  Maximum Calibration Error (MCE): {calib['mce']:.4f}")
        print(f"  Brier Score: {calib['brier_score']:.4f}")
        
        # Interpret calibration
        if calib['ece'] < 0.05:
            print("    ✅ Well calibrated (ECE < 0.05)")
        elif calib['ece'] < 0.10:
            print("    ⚠️  Moderately calibrated (ECE < 0.10)")
        else:
            print("    🚨 Poorly calibrated (ECE ≥ 0.10)")
        
        # Statistical tests
        if 'statistical_tests' in results:
            print("\n🔬 Statistical Tests:")
            print("─" * 60)
            
            if 'correct_vs_incorrect' in results['statistical_tests']:
                test = results['statistical_tests']['correct_vs_incorrect']
                print(f"  Correct vs Incorrect Confidence:")
                print(f"    p-value: {test['p_value']:.4f}")
                print(f"    Effect size: {test['effect_size']:.3f}")
                sig = "✅ Significant" if test['significant'] else "❌ Not significant"
                print(f"    Result: {sig} (α={self.alpha})")
        
        # Uncertainty metrics if available
        if 'uncertainty_metrics' in results:
            unc = results['uncertainty_metrics']
            print("\n🎲 Uncertainty Metrics:")
            print("─" * 60)
            print(f"  Epistemic (model): {unc['mean_epistemic']:.4f}")
            print(f"  Aleatoric (data): {unc['mean_aleatoric']:.4f}")
            print(f"  Total: {unc['mean_total']:.4f}")
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save results to JSON."""
        if output_path is None:
            output_path = os.path.join(
                analysis_results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_confidence_analysis.json"
                #"confidence_analysis.json"
            )

        ensure_dir_exists(os.path.dirname(output_path))

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Saved confidence analysis to {output_path}")
        
        return output_path


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
