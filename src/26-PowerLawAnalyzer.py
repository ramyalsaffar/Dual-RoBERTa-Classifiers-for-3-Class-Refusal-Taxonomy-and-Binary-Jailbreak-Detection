# Power Law Analyzer Module
#--------------------------
# Analyzes power law distributions in classifier performance:
# 1. Error Concentration (Pareto): Do 20% of categories cause 80% of errors?
# 2. Confidence Distribution: Do confidence scores follow power law?
# 3. Attention Distribution: Do few tokens dominate attention?
# All imports are in 01-Imports.py
###############################################################################


class PowerLawAnalyzer:
    """
    Analyzes power law distributions in classifier behavior.

    Power Laws in ML:
    - Error Concentration: Few categories/models cause most errors (Pareto Principle)
    - Confidence Distribution: Scores often follow power law
    - Attention Weights: Few tokens receive most attention (Zipfian)

    These analyses help identify:
    - Where to focus improvement efforts (concentrated errors)
    - Model calibration issues (confidence power laws)
    - Feature importance patterns (attention distribution)
    """

    def __init__(self, model, tokenizer, device, class_names: List[str] = None, model_type: str = "Model"):
        """
        Initialize power law analyzer.

        Args:
            model: Classification model
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names
            model_type: Type of model for display (e.g., "Refusal Classifier", "Jailbreak Detector")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.model_type = model_type
        
        # Create filename prefix from model_type
        # "Refusal Classifier" → "refusal"
        # "Jailbreak Detector" → "jailbreak"
        self.file_prefix = model_type.lower().split()[0] if model_type else "model"
    

    def analyze_all(self, test_df: pd.DataFrame, predictions: np.ndarray,
                    confidences: np.ndarray, output_dir: str = None, timestamp: str = None) -> Dict:
        """
        Run all power law analyses.

        Args:
            test_df: Test dataframe with 'category', 'model', 'response', 'label' columns
            predictions: Model predictions
            confidences: Prediction confidences
            output_dir: Directory to save visualizations (default: visualizations_path)
            timestamp: Timestamp string for filename (e.g., '20250114_1430')

        Returns:
            Dictionary with all power law analysis results
        """
        # Input Validation
        if test_df is None or len(test_df) == 0:
            print("❌ ERROR: test_df is empty. Cannot perform power law analysis.")
            return {'error': 'Empty test dataframe'}
        
        # Smart label column detection (same logic as ConfidenceAnalyzer)
        # Determine which label column to use based on model_type
        if 'jailbreak' in self.model_type.lower():
            label_col = 'jailbreak_label' if 'jailbreak_label' in test_df.columns else 'label'
        else:
            label_col = 'refusal_label' if 'refusal_label' in test_df.columns else 'label'
        
        # Check required columns
        required_cols = [label_col, 'response']
        missing_cols = [col for col in required_cols if col not in test_df.columns]
        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            return {'error': f'Missing columns: {missing_cols}'}
        
        # Validate array shapes
        if len(predictions) != len(test_df):
            print(f"❌ ERROR: predictions length ({len(predictions)}) != test_df length ({len(test_df)})")
            return {'error': 'Mismatched array lengths'}
        
        if len(confidences) != len(test_df):
            print(f"❌ ERROR: confidences length ({len(confidences)}) != test_df length ({len(test_df)})")
            return {'error': 'Mismatched array lengths'}
        
        # Check for perfect predictions (no errors)
        labels = test_df[label_col].values
        predictions_arr = np.asarray(predictions, dtype=np.int32)
        errors = predictions_arr != labels
        error_count = errors.sum()
        
        if error_count == 0:
            print("⚠️  WARNING: Model has perfect predictions (0 errors)!")
            print("   Power law analysis for errors will be skipped.")
            print("   Confidence and attention analyses will still run.")
        
        # Ensure output directory exists using Utils
        if output_dir is None:
            output_dir = visualizations_path
        ensure_dir_exists(output_dir)

        print_banner(f"POWER LAW ANALYSIS - {self.model_type.upper()}", width=60)

        results = {}

        # 1. Error Concentration Analysis (Pareto)
        print("\n--- Error Concentration Analysis (Pareto Principle) ---")
        if error_count > 0:
            results['error_concentration'] = self._analyze_error_concentration(
                test_df, predictions, output_dir, label_col, timestamp
            )
            self._print_pareto_results(results['error_concentration'])
        else:
            print("   ⚠️  Skipped: No errors to analyze (perfect predictions)")
            results['error_concentration'] = {'skipped': True, 'reason': 'No errors'}

        # 2. Confidence Distribution Analysis
        print("\n--- Confidence Distribution (Power Law Check) ---")
        results['confidence_distribution'] = self._analyze_confidence_distribution(
            confidences, predictions, test_df[label_col].values, output_dir, timestamp
        )
        self._print_confidence_analysis(results['confidence_distribution'])

        # 3. Attention Power Law Analysis
        print("\n--- Attention Distribution (Token Importance) ---")
        results['attention_power_law'] = self._analyze_attention_power_law(
            test_df, output_dir, timestamp
        )
        self._print_attention_analysis(results['attention_power_law'])

        print_banner(f"✅ POWER LAW ANALYSIS COMPLETE - {self.model_type.upper()}", width=60)

        return results

    def _analyze_error_concentration(self, test_df: pd.DataFrame,
                                    predictions: np.ndarray,
                                    output_dir: str,
                                    label_col: str,
                                    timestamp: str = None) -> Dict:
        """
        Analyze error concentration (Pareto Principle).

        Question: Do 20% of categories/models cause 80% of errors?

        Args:
            test_df: Test dataframe
            predictions: Model predictions
            output_dir: Output directory for plots
            label_col: Name of label column to use

        Returns:
            Dictionary with Pareto analysis results
        """
        # Ensure predictions is a numpy array
        predictions = np.asarray(predictions, dtype=np.int32)
        labels = test_df[label_col].values
        errors = predictions != labels

        results = {
            'by_category': self._pareto_analysis_by_group(
                test_df, errors, 'category', 'Category'
            ),
            'by_model': self._pareto_analysis_by_group(
                test_df, errors, 'model', 'Model'
            ) if 'model' in test_df.columns else None,
            'by_class': self._pareto_analysis_by_class(
                labels, predictions
            )
        }

        # Create Pareto visualizations
        self._plot_pareto_chart(
            results['by_category'],
            os.path.join(output_dir, get_timestamped_filename("pareto_errors_by_category.png", self.file_prefix, timestamp)),
            "Error Concentration by Category (Pareto Analysis)"
        )

        if results['by_model'] is not None:
            self._plot_pareto_chart(
                results['by_model'],
                os.path.join(output_dir, get_timestamped_filename("pareto_errors_by_model.png", self.file_prefix, timestamp)),
                "Error Concentration by Model (Pareto Analysis)"
            )

        return results

    def _pareto_analysis_by_group(self, test_df: pd.DataFrame, errors: np.ndarray,
                                  group_col: str, group_name: str) -> Dict:
        """
        Perform Pareto analysis by grouping column.

        Returns:
            Dict with error counts, cumulative percentages, Pareto metrics
        """
        if group_col not in test_df.columns:
            return None

        # Count errors per group
        df_with_errors = test_df.copy()
        df_with_errors['error'] = errors

        error_counts = df_with_errors.groupby(group_col)['error'].agg(['sum', 'count'])
        error_counts['error_rate'] = error_counts.apply(
            lambda row: safe_divide(row['sum'], row['count'], default=0.0), axis=1
        )
        error_counts = error_counts.sort_values('sum', ascending=False)

        # Calculate cumulative percentages
        total_errors = error_counts['sum'].sum()
        error_counts['cumulative_errors'] = error_counts['sum'].cumsum()
        # Use safe_divide from Utils to protect against division by zero
        error_counts['cumulative_pct'] = error_counts['cumulative_errors'].apply(
            lambda x: safe_divide(x, total_errors, default=0.0) * 100
        )

        # Find what % of groups cause 80% of errors (using config threshold)
        pareto_threshold = ANALYSIS_CONFIG['pareto_threshold']
        groups_80pct = (error_counts['cumulative_pct'] <= pareto_threshold).sum()
        total_groups = len(error_counts)
        groups_80pct_ratio = safe_divide(groups_80pct, total_groups, default=0.0) * 100

        pareto_strict = ANALYSIS_CONFIG['pareto_strict_threshold']
        return {
            'group_name': group_name,
            'error_counts': error_counts,
            'total_groups': total_groups,
            'groups_causing_80pct_errors': groups_80pct,
            'groups_80pct_ratio': groups_80pct_ratio,
            'pareto_holds': groups_80pct_ratio <= pareto_strict  # Strict Pareto is ~20%, allow up to configured %
        }

    def _pareto_analysis_by_class(self, labels: np.ndarray,
                                  predictions: np.ndarray) -> Dict:
        """
        Analyze which classes contribute most to errors.

        Returns:
            Dict with per-class error analysis
        """
        # Ensure inputs are numpy arrays
        labels = np.asarray(labels, dtype=np.int32)
        predictions = np.asarray(predictions, dtype=np.int32)
        
        class_errors = {}
        total_errors = 0

        for class_idx in range(self.num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue

            class_error_count = ((predictions != labels) & class_mask).sum()
            total_errors += class_error_count

            class_errors[self.class_names[class_idx]] = {
                'error_count': int(class_error_count),
                'total_samples': int(class_mask.sum()),
                'error_rate': safe_divide(float(class_error_count), float(class_mask.sum()), default=0.0)
            }

        # Sort by error count
        sorted_classes = sorted(
            class_errors.items(),
            key=lambda x: x[1]['error_count'],
            reverse=True
        )

        # Calculate cumulative
        cumulative_errors = 0
        for class_name, stats in sorted_classes:
            cumulative_errors += stats['error_count']
            stats['cumulative_pct'] = safe_divide(cumulative_errors, total_errors, default=0.0) * 100

        return {
            'class_errors': dict(sorted_classes),
            'total_errors': int(total_errors)
        }

    def _analyze_confidence_distribution(self, confidences: np.ndarray,
                                        predictions: np.ndarray,
                                        labels: np.ndarray,
                                        output_dir: str,
                                        timestamp: str = None) -> Dict:
        """
        Analyze confidence score distribution for power law.

        Power law: P(x) ∝ x^(-α)
        On log-log plot, this appears as straight line with slope -α

        Also checks calibration: are high-confidence predictions actually correct?

        Returns:
            Dict with power law fit, calibration metrics
        """
        # Ensure all inputs are numpy arrays with correct types
        predictions = np.asarray(predictions, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32) 
        confidences = np.asarray(confidences, dtype=np.float32)
        
        # Now safe to do boolean indexing
        errors = predictions != labels
        correct_confidences = confidences[~errors]
        error_confidences = confidences[errors]

        # Bin confidences and count (using config)
        num_bins = ANALYSIS_CONFIG['confidence_bins']
        bins = np.linspace(0, 1, num_bins + 1)  # +1 because linspace includes both endpoints
        bin_centers = (bins[:-1] + bins[1:]) / 2

        hist_all, _ = np.histogram(confidences, bins=bins)
        hist_correct, _ = np.histogram(correct_confidences, bins=bins)
        hist_errors, _ = np.histogram(error_confidences, bins=bins)

        # Fit power law to high-confidence region (using config threshold)
        high_conf_threshold = ANALYSIS_CONFIG['high_confidence_threshold']
        high_conf_mask = bin_centers >= high_conf_threshold
        x_fit = bin_centers[high_conf_mask]
        y_fit = hist_all[high_conf_mask]

        # Log-log fit (avoid log(0))
        valid_mask = y_fit > 0
        if valid_mask.sum() > 2:
            log_x = np.log(x_fit[valid_mask])
            log_y = np.log(y_fit[valid_mask])
            slope, intercept = np.polyfit(log_x, log_y, 1)
            power_law_exponent = -slope
            
            # Statistical Test: Goodness-of-fit for power law
            # Use Kolmogorov-Smirnov test to validate power law fit
            power_law_test = self._test_power_law_fit(
                confidences, power_law_exponent, 'confidence'
            )
        else:
            power_law_exponent = None
            slope = None
            intercept = None
            power_law_test = {'skipped': True, 'reason': 'Insufficient valid bins'}

        # Calibration analysis: confidence bins vs accuracy
        calibration = []
        for i in range(len(bins) - 1):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if bin_mask.sum() > 0:
                bin_accuracy = (predictions[bin_mask] == labels[bin_mask]).mean()
                bin_conf = confidences[bin_mask].mean()
                calibration.append({
                    'confidence_range': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    'mean_confidence': float(bin_conf),
                    'accuracy': float(bin_accuracy),
                    'count': int(bin_mask.sum()),
                    'calibration_error': float(abs(bin_conf - bin_accuracy))
                })

        # Expected Calibration Error (ECE)
        ece = np.mean([c['calibration_error'] for c in calibration])
        
        # Statistical Test: Chi-square test for calibration
        # Test if observed accuracy matches expected confidence
        calibration_test = self._test_calibration(calibration)

        # Plot confidence distribution
        self._plot_confidence_distribution(
            confidences, correct_confidences, error_confidences,
            os.path.join(output_dir, get_timestamped_filename("confidence_distribution.png", self.file_prefix, timestamp))
        )

        # Plot calibration curve
        self._plot_calibration_curve(
            calibration,
            os.path.join(output_dir, get_timestamped_filename("confidence_calibration.png", self.file_prefix, timestamp))
        )

        return {
            'power_law_exponent': power_law_exponent,
            'power_law_slope': slope,
            'power_law_intercept': intercept,
            'power_law_test': power_law_test,  # NEW: Statistical test results
            'expected_calibration_error': float(ece),
            'calibration_test': calibration_test,  # NEW: Calibration test results
            'calibration_bins': calibration,
            'stats': {
                'mean_confidence_all': float(confidences.mean()),
                'mean_confidence_correct': float(correct_confidences.mean()),
                'mean_confidence_errors': float(error_confidences.mean()),
                'overconfidence_on_errors': float(error_confidences.mean() > 0.5)
            }
        }

    def _analyze_attention_power_law(self, test_df: pd.DataFrame,
                                    output_dir: str,
                                    timestamp: str = None) -> Dict:
        """
        Analyze attention weight distribution for power law.

        Question: Do few tokens receive most attention? (Zipfian distribution)

        Samples a subset of test data to analyze attention patterns.

        Returns:
            Dict with attention power law metrics
        """
        # Sample texts for analysis (attention is expensive)
        sample_size = min(ANALYSIS_CONFIG.get('attention_sample_size', 100), len(test_df))
        sample_df = test_df.sample(n=sample_size, random_state=42)

        all_token_attentions = []
        top_k_concentrations = []

        self.model.eval()
        with torch.no_grad():
            for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing attention"):
                text = row['response']

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

                # Get attention weights from model
                if hasattr(self.model, 'roberta'):
                    outputs = self.model.roberta(
                        input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                    attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

                    # Average across layers and heads, get attention to CLS token
                    # Shape: (layers, batch, heads, seq, seq) -> (seq,)
                    avg_attention = torch.stack([a[0].mean(0) for a in attentions]).mean(0)
                    cls_attention = avg_attention[0, :]  # Attention from CLS to all tokens

                    # Get valid tokens only
                    valid_mask = attention_mask[0].cpu().numpy().astype(bool)
                    valid_attention = cls_attention.cpu().numpy()[valid_mask]

                    all_token_attentions.extend(valid_attention.tolist())

                    # Top-k concentration: what % of attention goes to top K% tokens (using config)
                    k_divisor = ANALYSIS_CONFIG['attention_top_k_percentage']
                    k = max(1, len(valid_attention) // k_divisor)  # Top (1/k_divisor) percentage
                    top_k_attn = np.partition(valid_attention, -k)[-k:].sum()
                    top_k_concentrations.append(float(top_k_attn))

        if len(all_token_attentions) == 0:
            return {'error': 'No attention weights extracted'}

        # Analyze distribution
        all_token_attentions = np.array(all_token_attentions)

        # Fit power law (Zipf: frequency ∝ rank^(-s))
        # Sort attentions in descending order
        sorted_attentions = np.sort(all_token_attentions)[::-1]
        ranks = np.arange(1, len(sorted_attentions) + 1)

        # Log-log fit
        log_ranks = np.log(ranks[sorted_attentions > 0])
        log_attentions = np.log(sorted_attentions[sorted_attentions > 0])

        if len(log_ranks) > 10:
            # Fit power law using config limit
            fit_limit = ANALYSIS_CONFIG['attention_power_law_fit_limit']
            slope, intercept = np.polyfit(log_ranks[:fit_limit], log_attentions[:fit_limit], 1)
            zipf_exponent = -slope
            
            # Statistical Test: Goodness-of-fit for Zipf's law
            zipf_test = self._test_power_law_fit(
                all_token_attentions, zipf_exponent, 'attention'
            )
        else:
            zipf_exponent = None
            slope = None
            zipf_test = {'skipped': True, 'reason': 'Insufficient data points'}

        # Plot attention distribution
        self._plot_attention_distribution(
            sorted_attentions, ranks,
            os.path.join(output_dir, get_timestamped_filename("attention_power_law.png", self.file_prefix, timestamp))
        )

        return {
            'zipf_exponent': zipf_exponent,
            'zipf_slope': slope,
            'zipf_test': zipf_test,  # NEW: Statistical test results
            'mean_top20_concentration': float(np.mean(top_k_concentrations)),
            'attention_stats': {
                'mean_attention': float(all_token_attentions.mean()),
                'std_attention': float(all_token_attentions.std()),
                'max_attention': float(all_token_attentions.max()),
                'min_attention': float(all_token_attentions.min()),
                'total_tokens_analyzed': len(all_token_attentions)
            }
        }

    def _test_power_law_fit(self, data: np.ndarray, exponent: float, 
                           distribution_type: str) -> Dict:
        """
        Test if data significantly follows a power law distribution.
        
        Uses Kolmogorov-Smirnov test to compare empirical distribution
        against theoretical power law with fitted exponent.
        
        Args:
            data: Observed data (confidences or attention weights)
            exponent: Fitted power law exponent
            distribution_type: 'confidence' or 'attention' for context
            
        Returns:
            Dict with test statistic, p-value, and interpretation
        """
        # Remove zeros and negatives (not valid for power law)
        data_positive = data[data > 0]
        
        if len(data_positive) < HYPOTHESIS_TESTING_CONFIG['min_samples_for_test']:
            return {
                'test': 'Kolmogorov-Smirnov',
                'skipped': True,
                'reason': f'Insufficient samples (n={len(data_positive)} < {HYPOTHESIS_TESTING_CONFIG["min_samples_for_test"]})'
            }
        
        # Generate theoretical power law CDF
        # For power law: P(X ≤ x) = 1 - (x_min/x)^α
        x_min = data_positive.min()
        x_sorted = np.sort(data_positive)
        
        # Theoretical CDF for power law
        theoretical_cdf = 1 - (x_min / x_sorted) ** exponent
        
        # Empirical CDF
        empirical_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        
        # KS test: maximum difference between CDFs
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        # Critical value for KS test (approximate)
        # KS critical value at α=0.05: 1.36 / sqrt(n)
        n = len(x_sorted)
        critical_value = 1.36 / np.sqrt(n)
        
        # Calculate p-value using scipy's KS test
        # We compare against uniform distribution transformed by power law
        try:
            # Transform data to test against power law
            transformed = (data_positive / x_min) ** exponent
            ks_result = kstest(transformed, 'uniform')
            p_value = ks_result.pvalue
            use_scipy = True
        except:
            # Fallback: approximate p-value
            p_value = np.exp(-2 * n * ks_statistic**2) if ks_statistic < 0.5 else 0.0
            use_scipy = False
        
        # Interpretation
        alpha = HYPOTHESIS_TESTING_CONFIG['alpha']
        follows_power_law = p_value > alpha
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'distribution_type': distribution_type,
            'statistic': float(ks_statistic),
            'critical_value': float(critical_value),
            'p_value': float(p_value),
            'alpha': alpha,
            'follows_power_law': follows_power_law,
            'n_samples': int(n),
            'exponent_tested': float(exponent),
            'method': 'scipy' if use_scipy else 'approximate',
            'interpretation': (
                f"Data {'DOES' if follows_power_law else 'DOES NOT'} "
                f"follow power law distribution (p={p_value:.4f}, α={alpha})"
            )
        }
    
    def _test_calibration(self, calibration_bins: List[Dict]) -> Dict:
        """
        Test if model is well-calibrated using chi-square goodness-of-fit.
        
        Null hypothesis: Model confidence matches actual accuracy
        (observed accuracy = expected confidence)
        
        Args:
            calibration_bins: List of dicts with 'mean_confidence', 'accuracy', 'count'
            
        Returns:
            Dict with chi-square statistic, p-value, and interpretation
        """
        if len(calibration_bins) < 2:
            return {
                'test': 'Chi-square',
                'skipped': True,
                'reason': 'Insufficient bins for test'
            }
        
        # Extract data
        observed = np.array([bin['accuracy'] * bin['count'] 
                           for bin in calibration_bins])
        expected = np.array([bin['mean_confidence'] * bin['count'] 
                           for bin in calibration_bins])
        counts = np.array([bin['count'] for bin in calibration_bins])
        
        # Check if we have enough samples
        total_samples = counts.sum()
        if total_samples < HYPOTHESIS_TESTING_CONFIG['min_samples_for_test']:
            return {
                'test': 'Chi-square',
                'skipped': True,
                'reason': f'Insufficient samples (n={total_samples} < {HYPOTHESIS_TESTING_CONFIG["min_samples_for_test"]})'
            }
        
        # Chi-square test: sum((observed - expected)^2 / expected)
        # Only use bins with sufficient expected counts (≥5 is standard)
        valid_bins = expected >= 5
        
        if valid_bins.sum() < 2:
            return {
                'test': 'Chi-square',
                'skipped': True,
                'reason': 'Insufficient bins with expected count ≥ 5'
            }
        
        observed_valid = observed[valid_bins]
        expected_valid = expected[valid_bins]
        
        # Calculate chi-square statistic
        chi2_statistic = np.sum((observed_valid - expected_valid)**2 / expected_valid)
        
        # Degrees of freedom = number of bins - 1
        df = valid_bins.sum() - 1
        
        # Calculate p-value using scipy
        p_value = 1 - chi2.cdf(chi2_statistic, df)
        
        # Interpretation
        alpha = HYPOTHESIS_TESTING_CONFIG['alpha']
        well_calibrated = p_value > alpha
        
        return {
            'test': 'Chi-square goodness-of-fit',
            'statistic': float(chi2_statistic),
            'degrees_of_freedom': int(df),
            'p_value': float(p_value),
            'alpha': alpha,
            'well_calibrated': well_calibrated,
            'n_bins_tested': int(valid_bins.sum()),
            'total_samples': int(total_samples),
            'interpretation': (
                f"Model {'IS' if well_calibrated else 'IS NOT'} "
                f"well-calibrated (p={p_value:.4f}, α={alpha})"
            )
        }

    def _plot_pareto_chart(self, pareto_data: Dict, output_path: str, title: str):
        """Plot Pareto chart (bar + cumulative line)."""
        if pareto_data is None:
            return

        error_counts = pareto_data['error_counts']

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar chart for error counts
        x = np.arange(len(error_counts))
        ax1.bar(x, error_counts['sum'], color='steelblue', alpha=0.7)
        ax1.set_xlabel(pareto_data['group_name'])
        ax1.set_ylabel('Error Count', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(error_counts.index, rotation=45, ha='right')

        # Cumulative percentage line
        ax2 = ax1.twinx()
        ax2.plot(x, error_counts['cumulative_pct'], color='red', marker='o', linewidth=2)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_ylabel('Cumulative % of Errors', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 105])
        ax2.legend()

        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print(f"   Saved Pareto chart: {output_path}")

    def _plot_confidence_distribution(self, all_conf, correct_conf, error_conf, output_path: str):
        """Plot confidence distribution for correct vs incorrect predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram (using config bins)
        num_bins = ANALYSIS_CONFIG['confidence_bins']
        axes[0].hist(correct_conf, bins=num_bins, alpha=0.7, label='Correct', color='green', density=True)
        axes[0].hist(error_conf, bins=num_bins, alpha=0.7, label='Errors', color='red', density=True)
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Confidence Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Log-log plot (power law check) - using config bins
        bins = np.linspace(0, 1, num_bins + 1)
        hist_all, _ = np.histogram(all_conf, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        valid_mask = (hist_all > 0) & (bin_centers > 0)
        axes[1].loglog(bin_centers[valid_mask], hist_all[valid_mask], 'o-', label='All predictions')
        axes[1].set_xlabel('Confidence (log scale)')
        axes[1].set_ylabel('Frequency (log scale)')
        axes[1].set_title('Power Law Check (Log-Log Plot)')
        axes[1].legend()
        axes[1].grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print(f"   Saved confidence distribution: {output_path}")

    def _plot_calibration_curve(self, calibration: List[Dict], output_path: str):
        """Plot calibration curve (confidence vs accuracy)."""
        confidences = [c['mean_confidence'] for c in calibration]
        accuracies = [c['accuracy'] for c in calibration]

        plt.figure(figsize=(8, 8))
        plt.plot(confidences, accuracies, 'o-', linewidth=2, markersize=8, label='Model')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        plt.xlabel('Mean Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print(f"   Saved calibration curve: {output_path}")

    def _plot_attention_distribution(self, sorted_attentions: np.ndarray,
                                    ranks: np.ndarray, output_path: str):
        """Plot attention distribution on log-log scale (Zipf's law check)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Regular plot
        axes[0].plot(ranks[:500], sorted_attentions[:500], 'o-', alpha=0.6)
        axes[0].set_xlabel('Token Rank')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Attention Distribution (Top 500 Tokens)')
        axes[0].grid(alpha=0.3)

        # Log-log plot (Zipf's law) - using config limit
        valid_mask = sorted_attentions > 0
        fit_limit = ANALYSIS_CONFIG['attention_power_law_fit_limit']
        axes[1].loglog(ranks[valid_mask][:fit_limit], sorted_attentions[valid_mask][:fit_limit], 'o', alpha=0.5)
        axes[1].set_xlabel('Token Rank (log scale)')
        axes[1].set_ylabel('Attention Weight (log scale)')
        axes[1].set_title('Zipf\'s Law Check (Log-Log Plot)')
        axes[1].grid(alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print(f"   Saved attention distribution: {output_path}")

    def _print_pareto_results(self, results: Dict):
        """Print Pareto analysis results."""
        # Check if analysis was skipped
        if results.get('skipped', False):
            print(f"   ℹ️  {results.get('reason', 'Analysis skipped')}")
            return
        
        print("\n📊 Pareto Principle Analysis:")

        # By category
        if results['by_category']:
            cat = results['by_category']
            print(f"\n   By Category:")
            print(f"   - Total categories: {cat['total_groups']}")
            print(f"   - Categories causing 80% of errors: {cat['groups_causing_80pct_errors']} ({cat['groups_80pct_ratio']:.1f}%)")
            if cat['pareto_holds']:
                print(f"   ✅ Pareto Principle HOLDS: {cat['groups_80pct_ratio']:.1f}% of categories cause 80% of errors")
            else:
                print(f"   ⚠️  Pareto Principle does NOT hold: {cat['groups_80pct_ratio']:.1f}% > 30%")

        # By model
        if results['by_model']:
            model = results['by_model']
            print(f"\n   By Model:")
            print(f"   - Total models: {model['total_groups']}")
            print(f"   - Models causing 80% of errors: {model['groups_causing_80pct_errors']} ({model['groups_80pct_ratio']:.1f}%)")

        # By class
        if results['by_class']:
            print(f"\n   By Class (top 3 error producers):")
            for i, (class_name, stats) in enumerate(list(results['by_class']['class_errors'].items())[:3]):
                print(f"   {i+1}. {class_name}: {stats['error_count']} errors ({stats['cumulative_pct']:.1f}% cumulative)")

    def _print_confidence_analysis(self, results: Dict):
        """Print confidence distribution analysis."""
        print("\n📈 Confidence Distribution Analysis:")

        # Use config thresholds
        pl_range = INTERPRETABILITY_CONFIG['power_law_exponent_range']
        ece_thresh = INTERPRETABILITY_CONFIG['ece_thresholds']
        overconf_thresh = ANALYSIS_CONFIG['overconfidence_threshold']

        if results['power_law_exponent'] is not None:
            print(f"   Power law exponent (α): {results['power_law_exponent']:.3f}")
            if pl_range[0] <= results['power_law_exponent'] <= pl_range[1]:
                print(f"   ✅ Confidence follows power law distribution")
            else:
                print(f"   ⚠️  Unusual power law exponent")
            
            # Print statistical test results
            test = results.get('power_law_test', {})
            if not test.get('skipped', False):
                print(f"\n   Statistical Test: {test['test']}")
                print(f"   - {test['interpretation']}")
                print(f"   - Test statistic: {test['statistic']:.4f}")
                print(f"   - P-value: {test['p_value']:.4f}")
                if test['follows_power_law']:
                    print(f"   ✅ Power law fit is statistically significant")
                else:
                    print(f"   ⚠️  Power law fit is NOT statistically significant")

        print(f"\n   Calibration:")
        print(f"   - Expected Calibration Error (ECE): {results['expected_calibration_error']:.4f}")
        if results['expected_calibration_error'] < ece_thresh['excellent']:
            print(f"   ✅ Well calibrated (ECE < {ece_thresh['excellent']})")
        elif results['expected_calibration_error'] < ece_thresh['good']:
            print(f"   ⚠️  Moderate calibration (ECE < {ece_thresh['good']})")
        else:
            print(f"   🚨 Poor calibration (ECE ≥ {ece_thresh['good']})")
        
        # Print calibration test results
        cal_test = results.get('calibration_test', {})
        if not cal_test.get('skipped', False):
            print(f"\n   Statistical Test: {cal_test['test']}")
            print(f"   - {cal_test['interpretation']}")
            print(f"   - Chi-square statistic: {cal_test['statistic']:.4f}")
            print(f"   - P-value: {cal_test['p_value']:.4f}")
            if cal_test['well_calibrated']:
                print(f"   ✅ Calibration is statistically validated")
            else:
                print(f"   🚨 Significant miscalibration detected")

        stats = results['stats']
        print(f"\n   Confidence Statistics:")
        print(f"   - Mean confidence (all): {stats['mean_confidence_all']:.3f}")
        print(f"   - Mean confidence (correct): {stats['mean_confidence_correct']:.3f}")
        print(f"   - Mean confidence (errors): {stats['mean_confidence_errors']:.3f}")
        if stats['mean_confidence_errors'] > overconf_thresh:
            print(f"   🚨 WARNING: High confidence on errors (overconfidence)!")

    def _print_attention_analysis(self, results: Dict):
        """Print attention power law analysis."""
        if 'error' in results:
            print(f"   ⚠️  {results['error']}")
            return

        print("\n🔍 Attention Distribution Analysis:")

        # Use config thresholds
        zipf_range = INTERPRETABILITY_CONFIG['zipf_exponent_range']
        min_conc = ANALYSIS_CONFIG['min_attention_concentration']
        k_pct = 100 / ANALYSIS_CONFIG['attention_top_k_percentage']  # Convert divisor to percentage

        if results['zipf_exponent'] is not None:
            print(f"   Zipf exponent (s): {results['zipf_exponent']:.3f}")
            if zipf_range[0] <= results['zipf_exponent'] <= zipf_range[1]:
                print(f"   ✅ Attention follows Zipfian distribution (typical for language)")
            else:
                print(f"   ⚠️  Unusual Zipf exponent")
            
            # Print statistical test results
            test = results.get('zipf_test', {})
            if not test.get('skipped', False):
                print(f"\n   Statistical Test: {test['test']}")
                print(f"   - {test['interpretation']}")
                print(f"   - Test statistic: {test['statistic']:.4f}")
                print(f"   - P-value: {test['p_value']:.4f}")
                if test['follows_power_law']:
                    print(f"   ✅ Zipfian distribution is statistically significant")
                else:
                    print(f"   ⚠️  Zipfian distribution is NOT statistically significant")

        print(f"\n   Attention Concentration:")
        print(f"   - Top {k_pct:.0f}% tokens receive: {results['mean_top20_concentration']*100:.1f}% of attention")
        if results['mean_top20_concentration'] > min_conc:
            print(f"   ✅ Attention appropriately concentrated on key tokens")
        else:
            print(f"   ⚠️  Attention may be too diffuse")

        stats = results['attention_stats']
        print(f"\n   Attention Statistics:")
        print(f"   - Total tokens analyzed: {stats['total_tokens_analyzed']}")
        print(f"   - Mean attention: {stats['mean_attention']:.6f}")
        print(f"   - Max attention: {stats['max_attention']:.6f}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 31, 2025
@author: ramyalsaffar
"""
