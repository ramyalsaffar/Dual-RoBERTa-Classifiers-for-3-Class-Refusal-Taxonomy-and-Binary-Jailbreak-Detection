# Jailbreak Analysis Module
#--------------------------
# Specialized analysis for jailbreak detector with security-critical metrics.
# Wraps existing analyzers with jailbreak-specific interpretation and cross-analysis.
# All imports are in 01-Imports.py
###############################################################################

class JailbreakAnalysis:
    """
    Security-critical analysis for jailbreak detector.

    Key Differences from Refusal Analysis:
    - Extreme class imbalance (~95% Failed, ~5% Succeeded)
    - False Negatives are CATASTROPHIC (missed breaches)
    - Recall on "Succeeded" class is primary metric
    - Cross-analysis with refusal classifier
    """

    def __init__(self, jailbreak_model, refusal_model, tokenizer, device):
        """
        Initialize jailbreak analyzer.

        Args:
            jailbreak_model: Trained JailbreakDetector model
            refusal_model: Trained RefusalClassifier model (for cross-analysis)
            tokenizer: RoBERTa tokenizer
            device: torch device
        """
        self.jailbreak_model = jailbreak_model
        self.refusal_model = refusal_model
        self.tokenizer = tokenizer
        self.device = device

        # Reuse existing analyzers (with jailbreak-specific class names and task_type)
        self.confidence_analyzer = ConfidenceAnalyzer(jailbreak_model, tokenizer, device, JAILBREAK_CLASS_NAMES, task_type='jailbreak')
        self.per_model_analyzer = PerModelAnalyzer(jailbreak_model, tokenizer, device, JAILBREAK_CLASS_NAMES, task_type='jailbreak')
        self.attention_viz = AttentionVisualizer(jailbreak_model, tokenizer, device, JAILBREAK_CLASS_NAMES, task_type='jailbreak')

    def analyze_full(self, test_df: pd.DataFrame) -> Dict:
        """
        Complete jailbreak detection analysis.

        Args:
            test_df: Test dataframe with columns:
                - 'response': LLM response text
                - 'jailbreak_label': Ground truth (0=Failed, 1=Succeeded)
                - 'refusal_label': Refusal classification (0/1/2)
                - 'model': Source model name
                - 'category': Prompt category

        Returns:
            Dictionary with all analysis results
        """
        print_banner("JAILBREAK DETECTOR ANALYSIS", width=60, char="=")

        results = {}

        # OPTIMIZATION: Run inference ONCE and cache results (4x speedup)
        print("\n--- Running Inference (cached for all analyses) ---")
        cached_predictions = self._run_inference_once(test_df)
        print(f"✓ Inference complete: {len(cached_predictions['jailbreak_preds'])} predictions cached")

        # 1. Security-Critical Metrics
        print("\n--- Security-Critical Metrics ---")
        results['security_metrics'] = self._calculate_security_metrics(cached_predictions)
        self._print_security_metrics(results['security_metrics'])

        # 2. Identify False Negatives (CRITICAL!)
        print("\n--- False Negative Analysis ---")
        results['false_negatives'] = self._identify_false_negatives(test_df, cached_predictions)
        print(f"Total False Negatives (missed jailbreaks): {len(results['false_negatives'])}")
        if len(results['false_negatives']) > 0:
            print("🚨 CRITICAL: These jailbreak successes were NOT detected!")
            print(f"   Review samples in: results/jailbreak_false_negatives.csv")

        # 3. Confidence Analysis
        print("\n--- Confidence Analysis ---")
        # Analyzers now have task_type='jailbreak' and will auto-detect jailbreak_label
        conf_results = self.confidence_analyzer.analyze(test_df)
        
        # Use cached predictions from earlier inference (line 61)
        preds = cached_predictions['jailbreak_preds']
        labels = cached_predictions['jailbreak_labels']
        confidences = cached_predictions['jailbreak_confidences']
        
        results['confidence'] = conf_results
        results['predictions'] = {'preds': preds, 
                                    'labels': labels, 
                                    'confidences': confidences,
                                    'accuracy': accuracy_score(labels, preds),
                                    'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
                                    'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0),
                                    'macro_precision': precision_score(labels, preds, average='macro', zero_division=0),
                                    'macro_recall': recall_score(labels, preds, average='macro', zero_division=0)
                                }


        # 4. Per-Model Analysis (NEW!)
        print("\n--- Per-Model Analysis ---")
        results['per_model'] = self.per_model_analyzer.analyze(test_df)

        # 5. Per-Model Vulnerability
        print("\n--- Per-Model Vulnerability Analysis ---")
        results['vulnerability'] = self._analyze_vulnerability_per_model(test_df)
        self._print_vulnerability_analysis(results['vulnerability'])

        # 6. Attack Type Analysis
        print("\n--- Attack Type Success Rate ---")
        results['attack_types'] = self._analyze_attack_types(test_df)
        self._print_attack_analysis(results['attack_types'])

        # 6.5. Model × Attack Type Matrix (NEW - V09)
        print("\n--- Model × Attack Type Matrix (Real Data Only) ---")
        results['model_attack_matrix'] = self._analyze_model_attack_matrix(test_df)
        self._print_model_attack_matrix(results['model_attack_matrix'])

        # 6.6. Statistical Significance Testing (NEW - V09)
        print("\n--- Statistical Significance Test (Real Data Only) ---")
        results['significance_test'] = self._test_model_vulnerability_significance(test_df)
        self._print_significance_test(results['significance_test'])

        # 7. Cross-Analysis with Refusal Classifier
        print("\n--- Cross-Analysis with Refusal Classifier ---")
        results['cross_analysis'] = self._cross_analyze_with_refusal(test_df, cached_predictions)
        self._print_cross_analysis(results['cross_analysis'])

        # 8. Precision-Recall Curve (better than ROC for imbalanced data)
        print("\n--- Precision-Recall Analysis ---")
        results['pr_curve'] = self._calculate_pr_curve(preds, labels, confidences)

        # 9. Attention Analysis on Failures
        print("\n--- Attention Analysis on False Negatives ---")
        if len(results['false_negatives']) > 0:
            results['attention_fn'] = self._analyze_attention_on_failures(results['false_negatives'])

        print_banner("✅ JAILBREAK ANALYSIS COMPLETE", width=60, char="=")
        
        return convert_to_serializable(results)
        #return results

    def _run_inference_once(self, test_df: pd.DataFrame) -> Dict:
        """
        Run inference ONCE and cache all predictions for efficiency.

        PERFORMANCE OPTIMIZATION: Previously ran inference 4 times (security metrics,
        false negatives, jailbreak preds, refusal preds). Now runs 2 times total.

        Returns:
            Dictionary with cached predictions:
                - jailbreak_preds: Jailbreak predictions (0/1)
                - jailbreak_confidences: Confidence scores
                - jailbreak_labels: Ground truth labels
                - refusal_preds: Refusal predictions (0/1/2)
        """
        # 1. Jailbreak model inference
        dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['jailbreak_label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

        jailbreak_preds = []
        jailbreak_confidences = []
        jailbreak_labels = []

        self.jailbreak_model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']

                logits = self.jailbreak_model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                confidences = torch.max(probs, dim=1)[0].cpu().numpy()

                jailbreak_preds.extend(preds.tolist())
                jailbreak_confidences.extend(confidences.tolist())
                jailbreak_labels.extend(labels.cpu().numpy().tolist())

        # 2. Refusal model inference (for cross-analysis)
        dataset = ClassificationDataset(
            test_df['response'].tolist(),
            test_df['refusal_label'].tolist(),
            self.tokenizer
        )
        loader = DataLoader(dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

        refusal_preds = []

        self.refusal_model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.refusal_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                refusal_preds.extend(preds)

        return {
            'jailbreak_preds': np.array(jailbreak_preds),
            'jailbreak_confidences': np.array(jailbreak_confidences),
            'jailbreak_labels': np.array(jailbreak_labels),
            'refusal_preds': np.array(refusal_preds)
        }

    def _calculate_security_metrics(self, cached_predictions: Dict) -> Dict:
        """
        Calculate security-critical metrics using cached predictions.

        Primary Metrics:
        - Recall on "Succeeded" (class 1) - MOST IMPORTANT
        - False Negative Rate (FNR) - Must be minimized
        - True Negative Rate (TNR) - Correctly identifying safe responses
        - F1 Score (weighted for imbalance)

        Args:
            cached_predictions: Cached predictions from _run_inference_once()
        """
        # Use cached predictions (no inference needed!)
        all_preds = cached_predictions['jailbreak_preds']
        all_labels = cached_predictions['jailbreak_labels']

        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)

        # Handle edge case: single-class predictions (all failed or all succeeded)
        if cm.shape != (2, 2):
            unique_preds = np.unique(all_preds)
            unique_labels = np.unique(all_labels)

            # All jailbreaks failed (perfect defense)
            if len(unique_labels) == 1 and unique_labels[0] == 0:
                print("✅ Perfect defense: All jailbreak attempts failed")
                return {
                    'confusion_matrix': cm,
                    'recall_succeeded': 0.0,
                    'precision_succeeded': 0.0,
                    'false_negative_rate': 0.0,
                    'true_negative_rate': 1.0,
                    'false_positive_rate': 0.0,
                    'f1_macro': 0.0,
                    'f1_weighted': 0.0,
                    'accuracy': 1.0,
                    'cohen_kappa': 0.0,
                    'counts': {
                        'true_negatives': int(cm[0, 0]),
                        'false_positives': 0,
                        'false_negatives': 0,
                        'true_positives': 0
                    },
                    'note': 'single_class_all_failed'
                }
            # All jailbreaks succeeded (catastrophic)
            elif len(unique_labels) == 1 and unique_labels[0] == 1:
                print("⚠️  CATASTROPHIC: All jailbreak attempts succeeded")
                return {
                    'confusion_matrix': cm,
                    'recall_succeeded': 1.0,
                    'precision_succeeded': 1.0,
                    'false_negative_rate': 0.0,
                    'true_negative_rate': 0.0,
                    'false_positive_rate': 1.0,
                    'f1_macro': 0.0,
                    'f1_weighted': 0.0,
                    'accuracy': 1.0,
                    'cohen_kappa': 0.0,
                    'counts': {
                        'true_negatives': 0,
                        'false_positives': 0,
                        'false_negatives': 0,
                        'true_positives': int(cm[0, 0])
                    },
                    'note': 'single_class_all_succeeded'
                }

        # Normal case: 2x2 confusion matrix
        # For binary classification:
        # TN (True Negative): Predicted Failed, Actually Failed
        # FP (False Positive): Predicted Succeeded, Actually Failed
        # FN (False Negative): Predicted Failed, Actually Succeeded - CATASTROPHIC!
        # TP (True Positive): Predicted Succeeded, Actually Succeeded

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        # Critical metrics
        recall_succeeded = recall_score(all_labels, all_preds, pos_label=1)  # Recall on class 1
        precision_succeeded = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)

        # Rates
        fnr = safe_divide(fn, fn + tp, default=0.0)  # False Negative Rate
        tnr = safe_divide(tn, tn + fp, default=0.0)  # True Negative Rate (Specificity)
        fpr = safe_divide(fp, fp + tn, default=0.0)  # False Positive Rate

        # F1 Scores
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        # Overall accuracy (less important for imbalanced data)
        accuracy = accuracy_score(all_labels, all_preds)

        # Cohen's Kappa (agreement beyond chance - important for imbalanced data)
        kappa = cohen_kappa_score(all_labels, all_preds)

        return {
            'confusion_matrix': cm.tolist(),
            'recall_succeeded': recall_succeeded,  # PRIMARY METRIC
            'precision_succeeded': precision_succeeded,
            'false_negative_rate': fnr,  # CRITICAL - Must be low
            'true_negative_rate': tnr,
            'false_positive_rate': fpr,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy,
            'cohen_kappa': float(kappa),
            'counts': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),  # These are breaches we missed!
                'true_positives': int(tp)
            }
        }

    def _print_security_metrics(self, metrics: Dict):
        """Print security metrics with alerts."""
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted Failed  Predicted Succeeded")
        print(f"Actually Failed        {metrics['counts']['true_negatives']:6d}            {metrics['counts']['false_positives']:6d}")
        print(f"Actually Succeeded     {metrics['counts']['false_negatives']:6d}            {metrics['counts']['true_positives']:6d}")

        print(f"\n🎯 PRIMARY METRIC:")
        print(f"   Recall on Succeeded (class 1): {metrics['recall_succeeded']:.4f}")
        min_recall = JAILBREAK_CONFIG['min_recall_succeeded']
        if metrics['recall_succeeded'] < min_recall:
            print(f"   🚨 WARNING: Recall below {min_recall*100:.0f}% target!")

        print(f"\n🚨 CRITICAL METRIC:")
        print(f"   False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)")
        print(f"   Missed Jailbreaks: {metrics['counts']['false_negatives']}")
        if metrics['counts']['false_negatives'] > 0:
            print(f"   🚨 ALERT: {metrics['counts']['false_negatives']} jailbreak successes were NOT detected!")

        print(f"\n📊 Other Metrics:")
        print(f"   Precision on Succeeded: {metrics['precision_succeeded']:.4f}")
        print(f"   True Negative Rate (TNR): {metrics['true_negative_rate']:.4f}")
        print(f"   F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Cohen's Kappa: {metrics['cohen_kappa']:.4f}", end="")
        kappa_thresh = INTERPRETABILITY_CONFIG['kappa_thresholds']
        if metrics['cohen_kappa'] > kappa_thresh['almost_perfect']:
            print(f" ✅ (Almost perfect)")
        elif metrics['cohen_kappa'] > kappa_thresh['substantial']:
            print(f" ✅ (Substantial)")
        elif metrics['cohen_kappa'] > kappa_thresh['moderate']:
            print(f" ⚠️  (Moderate)")
        else:
            print(f" 🚨 (Poor)")

    def _identify_false_negatives(self, test_df: pd.DataFrame, cached_predictions: Dict) -> pd.DataFrame:
        """
        Identify all false negatives (missed jailbreak successes) using cached predictions.
        These are CRITICAL failures that must be manually reviewed.

        Args:
            test_df: Test dataframe
            cached_predictions: Cached predictions from _run_inference_once()
        """
        # Use cached predictions (no inference needed!)
        test_df = test_df.copy()
        test_df['jailbreak_pred'] = cached_predictions['jailbreak_preds']
        test_df['jailbreak_confidence'] = cached_predictions['jailbreak_confidences']

        # False Negatives: Actually Succeeded (1) but predicted Failed (0)
        false_negatives = test_df[
            (test_df['jailbreak_label'] == 1) & (test_df['jailbreak_pred'] == 0)
        ].copy()

        # Save for review
        if len(false_negatives) > 0:
            fn_path = os.path.join(quality_review_path, f"jailbreak_{EXPERIMENT_CONFIG['experiment_name']}_false_negatives.csv")
            
            
            
            # Only include columns that exist
            columns_to_save = ['prompt', 'response', 'model', 'jailbreak_confidence']
            if 'category' in false_negatives.columns:
                columns_to_save.insert(3, 'category')  # Add category after model if it exists
            false_negatives[columns_to_save].to_csv(fn_path, index=False)

        return false_negatives

    def _analyze_vulnerability_per_model(self, test_df: pd.DataFrame) -> Dict:
        """
        Analyze which source models are most vulnerable to jailbreaks.
        """
        vulnerability = {}

        for model_name in test_df['model'].unique():
            model_df = test_df[test_df['model'] == model_name]

            # How many jailbreak attempts were there?
            total_samples = len(model_df)
            jailbreak_successes = (model_df['jailbreak_label'] == 1).sum()
            success_rate = safe_divide(jailbreak_successes, total_samples, default=0.0)

            vulnerability[model_name] = {
                'total_samples': total_samples,
                'jailbreak_successes': int(jailbreak_successes),
                'success_rate': success_rate
            }

        return vulnerability

    def _print_vulnerability_analysis(self, vulnerability: Dict):
        """Print vulnerability analysis by model."""
        print("\nVulnerability by Source Model:")
        for model, stats in vulnerability.items():
            print(f"  {model}:")
            print(f"    Jailbreak Successes: {stats['jailbreak_successes']}/{stats['total_samples']} ({stats['success_rate']*100:.2f}%)")

    def _analyze_attack_types(self, test_df: pd.DataFrame) -> Dict:
        """
        Analyze success rate by attack type (based on category).
        """
        # Check if category column exists (only present in real prompt data with metadata)
        if 'category' not in test_df.columns:
            return {
                'note': 'Attack type analysis skipped - no category metadata available',
                'reason': 'Generated prompts do not include category/attack type information'
            }

        attack_analysis = {}

        # Map categories to attack types
        jailbreak_categories = ['jailbreaks']  # Could expand if we track attack types

        # Check if category column exists
        if 'category' not in test_df.columns:
            print("⚠️  Attack type analysis skipped - no category metadata available")
            return attack_analysis
            
        for category in test_df['category'].unique():
            cat_df = test_df[test_df['category'] == category]

            total = len(cat_df)
            successes = (cat_df['jailbreak_label'] == 1).sum()
            success_rate = safe_divide(successes, total, default=0.0)

            attack_analysis[category] = {
                'total_samples': total,
                'jailbreak_successes': int(successes),
                'success_rate': success_rate
            }

        return attack_analysis

    def _print_attack_analysis(self, attack_analysis: Dict):
        """Print attack type analysis."""
        # Check if analysis was skipped due to missing category column
        if 'note' in attack_analysis:
            print(f"\n⚠️  {attack_analysis['note']}")
            return

        print("\nJailbreak Success Rate by Category:")
        for category, stats in sorted(attack_analysis.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            if stats['jailbreak_successes'] > 0:
                print(f"  {category}:")
                print(f"    Successes: {stats['jailbreak_successes']}/{stats['total_samples']} ({stats['success_rate']*100:.2f}%)")

    def _filter_real_only_data(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only real model responses (exclude WildJailbreak synthetic data).

        NEW (V09): Enables per-model vulnerability analysis on real data only,
        excluding synthetic WildJailbreak samples from model comparisons.

        Args:
            test_df: Full test dataframe

        Returns:
            DataFrame with only real model responses (data_source='real')
        """
        if not WILDJAILBREAK_CONFIG['exclude_from_model_analysis']:
            return test_df

        # Filter to real data only
        if 'data_source' in test_df.columns:
            real_only = test_df[test_df['data_source'] == 'real'].copy()

            if len(real_only) < len(test_df):
                excluded_count = len(test_df) - len(real_only)
                print(f"  ℹ️  Excluded {excluded_count} synthetic WildJailbreak samples from model analysis")
                print(f"  ℹ️  Analyzing {len(real_only)} real model responses only")

            return real_only
        else:
            # No data_source column - assume all real
            return test_df

    def _analyze_model_attack_matrix(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Model × Attack Type matrix showing jailbreak success rates.

        NEW (V09): Reveals which jailbreak tactics work on which models.

        Returns:
            DataFrame with models as rows, attack types as columns, success rates as values

        Example:
                          violence  hate  jailbreak  privacy
        Claude Sonnet:       1.2%   0.5%     2.1%     0.8%
        GPT-5:               5.3%   7.1%     8.9%     4.2%  ← Consistently vulnerable!
        Gemini 2.5:          0.2%   0.1%     0.8%     0.3%
        """
        # Filter to real data only
        real_df = self._filter_real_only_data(test_df)

        if len(real_df) == 0:
            print("  ⚠️  No real data available for model-attack matrix")
            return pd.DataFrame()

        # Check if category column exists
        if 'category' not in real_df.columns:
            print("  ⚠️  No category metadata available for model-attack matrix")
            return pd.DataFrame()

        # Create matrix
        models = sorted(real_df['model'].unique())
        categories = sorted(real_df['category'].unique())

        matrix_data = []

        for model in models:
            row = {'model': model}
            model_df = real_df[real_df['model'] == model]

            for category in categories:
                cat_df = model_df[model_df['category'] == category]

                if len(cat_df) > 0:
                    successes = (cat_df['jailbreak_label'] == 1).sum()
                    success_rate = safe_divide(successes, len(cat_df), default=0.0) * 100  # Convert to percentage
                    row[category] = success_rate
                else:
                    row[category] = 0.0

            matrix_data.append(row)

        matrix_df = pd.DataFrame(matrix_data)
        matrix_df = matrix_df.set_index('model')

        return matrix_df

    def _test_model_vulnerability_significance(self, test_df: pd.DataFrame) -> Dict:
        """
        Test if model vulnerability differences are statistically significant.

        NEW (V09): Uses chi-square test to determine if observed differences
        in jailbreak success rates across models are statistically significant.

        Returns:
            Dictionary with:
                - chi2_statistic: Chi-square test statistic
                - p_value: P-value from test
                - significant: Boolean (p < alpha)
                - alpha: Significance level from config
                - conclusion: Human-readable conclusion
                - effect_size: Cramér's V effect size
                - models_compared: List of model names
                - vulnerable_model: Model with highest jailbreak rate
                - robust_model: Model with lowest jailbreak rate
        """
        # Filter to real data only
        real_df = self._filter_real_only_data(test_df)

        if len(real_df) == 0:
            return {'error': 'No real data available for statistical testing'}

        # Get configuration values
        alpha = HYPOTHESIS_TESTING_CONFIG['alpha']
        min_samples = HYPOTHESIS_TESTING_CONFIG['min_samples_for_test']

        # Check if we have enough samples per model
        model_counts = real_df['model'].value_counts()
        models = [m for m in model_counts.index if model_counts[m] >= min_samples]

        if len(models) < 2:
            return {
                'error': f'Insufficient samples for testing (need ≥{min_samples} per model, have {len(models)} models)',
                'min_samples_required': min_samples
            }

        # Filter to models with enough samples
        test_df_filtered = real_df[real_df['model'].isin(models)].copy()

        # Create contingency table: Models (rows) × Jailbreak Status (columns)
        contingency = pd.crosstab(
            test_df_filtered['model'],
            test_df_filtered['jailbreak_label']
        )

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Calculate Cramér's V (effect size)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

        # Determine most/least vulnerable models
        vulnerability_rates = {}
        for model in models:
            model_df = test_df_filtered[test_df_filtered['model'] == model]
            rate = safe_divide((model_df['jailbreak_label'] == 1).sum(), len(model_df), default=0.0)
            vulnerability_rates[model] = rate

        vulnerable_model = max(vulnerability_rates.items(), key=lambda x: x[1])
        robust_model = min(vulnerability_rates.items(), key=lambda x: x[1])

        # Create conclusion
        significant = p_value < alpha
        if significant:
            conclusion = f"{vulnerable_model[0]} is significantly more vulnerable than {robust_model[0]} (p={p_value:.5f}, α={alpha})"
        else:
            conclusion = f"No statistically significant difference in model vulnerabilities (p={p_value:.3f}, α={alpha})"

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': significant,
            'alpha': alpha,
            'conclusion': conclusion,
            'effect_size_cramers_v': cramers_v,
            'models_compared': models,
            'vulnerable_model': vulnerable_model[0],
            'vulnerable_rate': vulnerable_model[1] * 100,
            'robust_model': robust_model[0],
            'robust_rate': robust_model[1] * 100,
            'contingency_table': contingency
        }

    def _print_model_attack_matrix(self, matrix_df: pd.DataFrame):
        """Print Model × Attack Type matrix."""
        if matrix_df.empty:
            return

        print_banner("MODEL × ATTACK TYPE VULNERABILITY MATRIX", width=60, char="=")
        print("(Shows jailbreak success rate % for each model-category combination)")
        print()

        # Print matrix with nice formatting
        print(matrix_df.to_string(float_format=lambda x: f"{x:5.1f}%"))
        print()

        # Identify most vulnerable model-category pairs
        print("Most Vulnerable Combinations:")
        # Flatten and sort
        flat_data = []
        for model in matrix_df.index:
            for category in matrix_df.columns:
                rate = matrix_df.loc[model, category]
                if rate > 0:
                    flat_data.append((model, category, rate))

        flat_data.sort(key=lambda x: x[2], reverse=True)

        # Show top 5
        for i, (model, category, rate) in enumerate(flat_data[:5], 1):
            print(f"  {i}. {model} + {category}: {rate:.1f}%")

        print("="*60)

    def _print_significance_test(self, sig_results: Dict):
        """Print statistical significance test results."""
        if 'error' in sig_results:
            print(f"\n⚠️  Statistical Test Skipped: {sig_results['error']}")
            return

        print_banner("STATISTICAL SIGNIFICANCE TEST", width=60, char="=")
        print(f"Test: Chi-Square Test for Model Vulnerability Differences")
        print(f"Models Compared: {', '.join(sig_results['models_compared'])}")
        print(f"Significance Level (α): {sig_results['alpha']}")
        print()
        print(f"Results:")
        print(f"  χ² = {sig_results['chi2_statistic']:.3f}")
        print(f"  p-value = {sig_results['p_value']:.6f}")
        print(f"  Effect Size (Cramér's V) = {sig_results['effect_size_cramers_v']:.3f}")
        print()

        if sig_results['significant']:
            print(f"✓ SIGNIFICANT: Differences are statistically significant (p < {sig_results['alpha']})")
            print(f"  Most Vulnerable: {sig_results['vulnerable_model']} ({sig_results['vulnerable_rate']:.2f}% jailbreak rate)")
            print(f"  Most Robust:     {sig_results['robust_model']} ({sig_results['robust_rate']:.2f}% jailbreak rate)")
        else:
            print(f"✗ NOT SIGNIFICANT: Differences could be due to chance (p ≥ {sig_results['alpha']})")

        print()
        print(f"Conclusion: {sig_results['conclusion']}")
        print("="*60)

    def _cross_analyze_with_refusal(self, test_df: pd.DataFrame, cached_predictions: Dict) -> Dict:
        """
        Cross-analyze jailbreak detection with refusal classification using cached predictions.

        Critical Question: Do jailbreak successes bypass refusal detection?

        Args:
            test_df: Test dataframe
            cached_predictions: Cached predictions from _run_inference_once()
        """
        # Use cached predictions (no inference needed!)
        jailbreak_preds = cached_predictions['jailbreak_preds']
        refusal_preds = cached_predictions['refusal_preds']

        # Cross-tabulation
        cross_tab = pd.crosstab(
            refusal_preds,
            jailbreak_preds,
            rownames=['Refusal'],
            colnames=['Jailbreak']
        )

        # Critical case: Jailbreak Succeeded + No Refusal
        dangerous = (refusal_preds == 0) & (jailbreak_preds == 1)
        dangerous_count = dangerous.sum()

        # Also check: Jailbreak Succeeded + Soft Refusal (partial bypass)
        partial_bypass = (refusal_preds == 2) & (jailbreak_preds == 1)
        partial_count = partial_bypass.sum()

        return {
            'cross_tab': cross_tab,
            'dangerous_combinations': dangerous_count,
            'partial_bypass': partial_count,
            'dangerous_samples': test_df[dangerous].copy() if dangerous_count > 0 else pd.DataFrame()
        }

    def _print_cross_analysis(self, cross_analysis: Dict):
        """Print cross-analysis results."""
        print("\nCross-Tabulation (Refusal vs Jailbreak):")
        print(cross_analysis['cross_tab'])

        if cross_analysis['dangerous_combinations'] > 0:
            print(f"\n🚨 CRITICAL: {cross_analysis['dangerous_combinations']} samples where:")
            print(f"   - Jailbreak SUCCEEDED")
            print(f"   - Refusal classifier says NO REFUSAL")
            print(f"   → Model completely bypassed!")

        if cross_analysis['partial_bypass'] > 0:
            print(f"\n⚠️  WARNING: {cross_analysis['partial_bypass']} samples where:")
            print(f"   - Jailbreak SUCCEEDED")
            print(f"   - Refusal classifier says SOFT REFUSAL")
            print(f"   → Partial bypass detected!")

    def _calculate_pr_curve(self, preds, labels, confidences) -> Dict:
        """
        Calculate Precision-Recall curve.
        Better than ROC for highly imbalanced data.
        """
        # Use confidence as score
        precision, recall, thresholds = precision_recall_curve(labels, confidences)
        avg_precision = average_precision_score(labels, confidences)

        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision
        }

    def _analyze_attention_on_failures(self, false_negatives: pd.DataFrame) -> Dict:
        """
        Analyze attention patterns on false negatives.
        Helps understand why we missed these jailbreaks.
        """
        if len(false_negatives) == 0:
            return {}

        # Analyze first N false negatives
        num_analyze = min(ANALYSIS_CONFIG['error_examples_count'], len(false_negatives))
        attention_results = []

        for idx in range(num_analyze):
            text = false_negatives.iloc[idx]['response']
            attention_data = self.attention_viz.get_attention_weights(text)
            attention_results.append(attention_data)

        return {'analyzed_samples': num_analyze, 'attention_data': attention_results}

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'false_negatives':
                serializable_results[key] = f"{len(value)} samples (see jailbreak_false_negatives.csv)"
            elif key == 'predictions':
                serializable_results[key] = {
                    'preds': [int(p) for p in value['preds']],        # ✅ Actual predictions
                    'labels': [int(l) for l in value['labels']],      # ✅ Actual labels
                    'confidences': [float(c) for c in value['confidences']],  # ✅ Actual confidences
                    'num_samples': len(value['preds'])                # ✅ Count too
                }
            elif key == 'cross_analysis':
                serializable_results[key] = {
                    'dangerous_combinations': int(value['dangerous_combinations']),
                    'partial_bypass': int(value['partial_bypass'])
                }
            elif isinstance(value, dict):
                serializable_results[key] = convert_to_serializable(value)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Saved jailbreak analysis to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 30, 2025
@author: ramyalsaffar
"""
