# Hypothesis Testing Module
#---------------------------
# Statistical hypothesis tests for dataset and model validation.
# Includes class balance tests, distribution tests, and significance testing.
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# CLASS BALANCE HYPOTHESIS TESTING
# =============================================================================

class DatasetValidator:
    """
    Statistical hypothesis testing for classification datasets.

    Provides tests for:
    - Class balance (chi-square goodness-of-fit)
    - Distribution normality
    - Statistical documentation for reproducibility
    """

    def __init__(self, class_names: List[str] = None, alpha: float = None):
        """
        Initialize hypothesis tester.

        Args:
            class_names: List of class names for display
            alpha: Significance level (default: from HYPOTHESIS_TESTING_CONFIG)
        """
        self.class_names = class_names
        # Use config value if not provided - NO HARDCODING!
        self.alpha = alpha if alpha is not None else HYPOTHESIS_TESTING_CONFIG['alpha']
        self.test_results = {}

        print_banner("HYPOTHESIS TESTING SETUP", width=60)
        print(f"Significance level (α): {self.alpha}")
        print(f"Confidence level: {(1-self.alpha)*100:.0f}%")
        print_banner("", width=60, char="=")


    def test_class_balance(self,
                          class_counts: List[int],
                          expected_distribution: str = 'uniform',
                          expected_proportions: List[float] = None) -> Dict:
        """
        Test if class distribution is balanced using chi-square goodness-of-fit test.

        Null Hypothesis (H0): Classes are distributed according to expected distribution
        Alternative Hypothesis (H1): Classes deviate from expected distribution

        Args:
            class_counts: Observed counts for each class [count_0, count_1, ...]
            expected_distribution: 'uniform' or 'custom'
            expected_proportions: Custom proportions if expected_distribution='custom'

        Returns:
            Dictionary with test results
        """
        # Input Validation
        if not class_counts or len(class_counts) == 0:
            print("❌ ERROR: class_counts is empty")
            return {'error': 'Empty class_counts'}
        
        if any(count < 0 for count in class_counts):
            print("❌ ERROR: class_counts contains negative values")
            return {'error': 'Negative counts not allowed'}
        
        if sum(class_counts) == 0:
            print("❌ ERROR: Total samples is zero")
            return {'error': 'No samples'}

        num_classes = len(class_counts)
        total_samples = sum(class_counts)

        print_banner("CHI-SQUARE GOODNESS-OF-FIT TEST: CLASS BALANCE", width=60)

        # Calculate expected counts using safe_divide for proportions
        if expected_distribution == 'uniform':
            expected_counts = [safe_divide(total_samples, num_classes, default=0.0) for _ in range(num_classes)]
            print(f"Expected distribution: Uniform (equal classes)")
        elif expected_distribution == 'custom':
            if expected_proportions is None or len(expected_proportions) != num_classes:
                raise ValueError("Custom expected_proportions must be provided and match number of classes")
            if not np.isclose(sum(expected_proportions), 1.0):
                raise ValueError("Expected proportions must sum to 1.0")
            expected_counts = [p * total_samples for p in expected_proportions]
            print(f"Expected distribution: Custom")
        else:
            raise ValueError("expected_distribution must be 'uniform' or 'custom'")

        # Print observed vs expected
        print(f"{'Class':<25} {'Observed':<12} {'Expected':<12} {'Difference':<12}")
        print(f"{'-'*60}")
        for i in range(num_classes):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            diff = class_counts[i] - expected_counts[i]
            print(f"{class_name:<25} {class_counts[i]:<12} {expected_counts[i]:<12.1f} {diff:+12.1f}")

        # Perform chi-square test
        # Check if any expected count is too small (< 1)
        if any(e < 1 for e in expected_counts):
            print(f"\n⚠️  Warning: Some expected counts are < 1")
            print(f"   Chi-square test may be unreliable")
        
        chi2_statistic, p_value = chisquare(f_obs=class_counts, f_exp=expected_counts)

        # Interpret results
        is_balanced = p_value >= self.alpha

        print(f"\n{'-'*60}")
        print(f"Test Statistics:")
        print(f"{'-'*60}")
        print(f"Chi-square statistic (χ²): {chi2_statistic:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significance level (α): {self.alpha}")
        print(f"Degrees of freedom: {num_classes - 1}")

        print(f"\n{'-'*60}")
        print(f"Interpretation:")
        print(f"{'-'*60}")

        if is_balanced:
            print(f"✓ FAIL TO REJECT NULL HYPOTHESIS (p = {p_value:.4f} >= α = {self.alpha})")
            print(f"  → Classes are BALANCED (no significant deviation from expected)")
            print(f"  → The observed class distribution is consistent with the expected distribution")
            recommendation = "No class weighting needed, but you may still use it for better minority class performance."
        else:
            print(f"✗ REJECT NULL HYPOTHESIS (p = {p_value:.4f} < α = {self.alpha})")
            print(f"  → Classes are IMBALANCED (significant deviation from expected)")
            print(f"  → The observed class distribution significantly differs from expected")
            recommendation = "RECOMMENDATION: Use class weights in loss function to handle imbalance."

        print(f"\n{recommendation}")
        print_banner("", width=60, char="=")

        # Calculate class proportions using safe_divide
        proportions = [safe_divide(count, total_samples, default=0.0) for count in class_counts]

        # Calculate imbalance ratio (max / min) using safe_divide
        max_count = max(class_counts)
        min_count = min(class_counts)
        imbalance_ratio = safe_divide(max_count, min_count, default=1.0)

        # Store results
        result = {
            'test_name': 'chi_square_class_balance',
            'chi2_statistic': chi2_statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'is_balanced': is_balanced,
            'reject_null': not is_balanced,
            'degrees_of_freedom': num_classes - 1,
            'observed_counts': class_counts,
            'expected_counts': expected_counts,
            'proportions': proportions,
            'imbalance_ratio': imbalance_ratio,
            'recommendation': recommendation,
            'interpretation': 'balanced' if is_balanced else 'imbalanced'
        }

        self.test_results['class_balance'] = result

        return result


    def test_normality(self, data: np.ndarray, data_name: str = "Data") -> Dict:
        """
        Test if data follows normal distribution using Shapiro-Wilk test.

        Useful for validating assumptions in parametric statistical tests.

        Null Hypothesis (H0): Data is normally distributed
        Alternative Hypothesis (H1): Data is not normally distributed

        Args:
            data: 1D array of numerical values
            data_name: Name of the data for display

        Returns:
            Dictionary with test results
        """
        # Input Validation
        if data is None or len(data) == 0:
            print("❌ ERROR: data is empty")
            return {
                'test_name': 'shapiro_wilk_normality',
                'error': 'Empty data',
                'statistic': np.nan,
                'p_value': np.nan,
                'alpha': self.alpha,
                'is_normal': False,
                'reject_null': True,
                'sample_size': 0
            }
        
        # Convert to numpy array if needed
        data = np.asarray(data).flatten()
        
        # Check for non-finite values
        if not np.all(np.isfinite(data)):
            print("❌ ERROR: data contains NaN or infinite values")
            return {
                'test_name': 'shapiro_wilk_normality',
                'error': 'Non-finite values in data',
                'statistic': np.nan,
                'p_value': np.nan,
                'alpha': self.alpha,
                'is_normal': False,
                'reject_null': True,
                'sample_size': len(data)
            }

        print_banner(f"SHAPIRO-WILK NORMALITY TEST: {data_name}", width=60)
        
        # Check minimum sample size
        if len(data) < 3:
            print(f"⚠️  Insufficient samples for Shapiro-Wilk test")
            print(f"   Need at least 3 samples, have {len(data)}")
            print_banner("", width=60, char="=")
            return {
                'test_name': 'shapiro_wilk_normality',
                'statistic': np.nan,
                'p_value': np.nan,
                'alpha': self.alpha,
                'is_normal': False,
                'reject_null': True,
                'sample_size': len(data),
                'error': 'Insufficient samples (need >= 3)'
            }

        # Basic statistics
        print(f"Sample size: {len(data)}")
        print(f"Mean: {np.mean(data):.4f}")
        print(f"Std: {np.std(data):.4f}")
        print(f"Min: {np.min(data):.4f}")
        print(f"Max: {np.max(data):.4f}")

        # Perform Shapiro-Wilk test
        statistic, p_value = shapiro(data)

        # Interpret results
        is_normal = p_value >= self.alpha

        print(f"\n{'-'*60}")
        print(f"Test Statistics:")
        print(f"{'-'*60}")
        print(f"W-statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significance level (α): {self.alpha}")

        print(f"\n{'-'*60}")
        print(f"Interpretation:")
        print(f"{'-'*60}")

        if is_normal:
            print(f"✓ FAIL TO REJECT NULL HYPOTHESIS (p = {p_value:.4f} >= α = {self.alpha})")
            print(f"  → Data appears NORMALLY DISTRIBUTED")
            print(f"  → Parametric tests (t-test, ANOVA) are appropriate")
        else:
            print(f"✗ REJECT NULL HYPOTHESIS (p = {p_value:.4f} < α = {self.alpha})")
            print(f"  → Data is NOT NORMALLY DISTRIBUTED")
            print(f"  → Consider non-parametric tests (Mann-Whitney, Kruskal-Wallis)")

        print_banner("", width=60, char="=")

        # Store results
        result = {
            'test_name': 'shapiro_wilk_normality',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'is_normal': is_normal,
            'reject_null': not is_normal,
            'sample_size': len(data),
            'mean': np.mean(data),
            'std': np.std(data)
        }

        test_key = f'normality_{data_name.lower().replace(" ", "_")}'
        self.test_results[test_key] = result

        return result


    def analyze_dataset_statistics(self,
                                   dataset,
                                   task_type: str = 'refusal') -> Dict:
        """
        Comprehensive statistical analysis of dataset.

        Tests:
        1. Class balance (chi-square)
        2. Sample size adequacy
        3. Per-class statistics

        Args:
            dataset: Dataset object with labels attribute
            task_type: 'refusal' (3-class) or 'jailbreak' (2-class)

        Returns:
            Dictionary with complete statistical analysis
        """
        # Input Validation
        if dataset is None:
            print("❌ ERROR: dataset is None")
            return {'error': 'Dataset is None'}
        
        if len(dataset) == 0:
            print("❌ ERROR: dataset is empty")
            return {'error': 'Empty dataset'}
        
        print(f"\n{'#'*60}")
        print(f"COMPREHENSIVE DATASET STATISTICAL ANALYSIS: {task_type.upper()}")
        print(f"{'#'*60}\n")

        # Extract labels
        try:
            all_labels = dataset.labels if hasattr(dataset, 'labels') else [dataset[i]['label'] for i in range(len(dataset))]
            all_labels = np.array(all_labels)
        except Exception as e:
            print(f"❌ ERROR: Failed to extract labels from dataset: {e}")
            return {'error': f'Label extraction failed: {e}'}
        
        if len(all_labels) == 0:
            print("❌ ERROR: No labels found in dataset")
            return {'error': 'No labels in dataset'}

        # Determine class names
        if task_type == 'refusal':
            class_names = CLASS_NAMES
        elif task_type == 'jailbreak':
            class_names = JAILBREAK_CLASS_NAMES
        else:
            class_names = [f"Class {i}" for i in range(len(np.unique(all_labels)))]

        self.class_names = class_names

        # Calculate class counts
        num_classes = len(class_names)
        class_counts = [int(np.sum(all_labels == i)) for i in range(num_classes)]
        total_samples = len(all_labels)

        # Print dataset overview
        print_banner("DATASET OVERVIEW", width=60)
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {num_classes}")
        print(f"Task type: {task_type}")
        print(f"\nClass distribution:")
        for i, (name, count) in enumerate(zip(class_names, class_counts)):
            percentage = safe_divide(count, total_samples, default=0.0) * 100
            print(f"  {name:<25}: {count:>6} ({percentage:>5.2f}%)")

        # Calculate imbalance ratio using safe_divide
        max_count = max(class_counts) if class_counts else 0
        min_count = min(class_counts) if class_counts else 0
        imbalance_ratio = safe_divide(max_count, min_count, default=1.0)
        print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")

        if imbalance_ratio < 1.5:
            balance_assessment = "Well-balanced"
        elif imbalance_ratio < 3.0:
            balance_assessment = "Moderately imbalanced"
        else:
            balance_assessment = "Severely imbalanced"

        print(f"Balance assessment: {balance_assessment}")
        print_banner("", width=60, char="=")

        # TEST 1: Class balance (chi-square)
        balance_result = self.test_class_balance(
            class_counts=class_counts,
            expected_distribution='uniform'
        )

        # TEST 2: Sample size adequacy
        print_banner("SAMPLE SIZE ADEQUACY", width=60)

        # Rule of thumb: Use config minimum for meaningful statistics
        min_samples_per_class = ANALYSIS_CONFIG['min_samples_per_class']
        adequate_sample_size = all(count >= min_samples_per_class for count in class_counts)

        print(f"Minimum recommended samples per class: {min_samples_per_class}")
        print(f"Actual minimum: {min(class_counts)}")

        if adequate_sample_size:
            print(f"✓ ADEQUATE: All classes have sufficient samples")
        else:
            print(f"⚠ WARNING: Some classes have fewer than {min_samples_per_class} samples")
            for name, count in zip(class_names, class_counts):
                if count < min_samples_per_class:
                    print(f"  - {name}: {count} samples (need {min_samples_per_class - count} more)")

        print_banner("", width=60, char="=")

        # Compile results with safe_divide for proportions
        results = {
            'task_type': task_type,
            'total_samples': total_samples,
            'num_classes': num_classes,
            'class_names': class_names,
            'class_counts': class_counts,
            'class_proportions': [safe_divide(c, total_samples, default=0.0) for c in class_counts],
            'imbalance_ratio': imbalance_ratio,
            'balance_assessment': balance_assessment,
            'class_balance_test': balance_result,
            'sample_size_adequate': adequate_sample_size,
            'min_samples_per_class': min(class_counts) if class_counts else 0
        }

        # Store in test results
        self.test_results[f'{task_type}_dataset_analysis'] = results

        return results


    def generate_statistical_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive statistical report with all test results.

        Args:
            output_path: Path to save report (default: results/hypothesis_tests.txt)

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = os.path.join(
                analysis_results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_hypothesis_tests.txt"
            )

        # Use ensure_dir_exists from Utils instead of os.makedirs
        ensure_dir_exists(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL HYPOTHESIS TESTING REPORT\n")
            f.write("="*80 + "\n")
            # Use get_timestamp from Utils for consistent formatting
            f.write(f"Generated: {get_timestamp('display')}\n")
            f.write(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}\n")
            f.write(f"Significance Level (α): {self.alpha}\n")
            f.write(f"Confidence Level: {(1-self.alpha)*100:.0f}%\n")
            f.write("="*80 + "\n\n")

            for test_name, result in self.test_results.items():
                f.write("-"*80 + "\n")
                f.write(f"TEST: {test_name.upper().replace('_', ' ')}\n")
                f.write("-"*80 + "\n\n")

                if 'dataset_analysis' in test_name:
                    # Dataset analysis
                    f.write(f"Task Type: {result['task_type']}\n")
                    f.write(f"Total Samples: {result['total_samples']}\n")
                    f.write(f"Number of Classes: {result['num_classes']}\n\n")

                    f.write("Class Distribution:\n")
                    for name, count, prop in zip(result['class_names'], result['class_counts'], result['class_proportions']):
                        f.write(f"  {name:<25}: {count:>6} ({prop*100:>5.2f}%)\n")

                    f.write(f"\nImbalance Ratio: {result['imbalance_ratio']:.2f}\n")
                    f.write(f"Balance Assessment: {result['balance_assessment']}\n\n")

                    # Chi-square results
                    chi2_result = result['class_balance_test']
                    f.write(f"Chi-Square Test:\n")
                    f.write(f"  χ² statistic: {chi2_result['chi2_statistic']:.4f}\n")
                    f.write(f"  p-value: {chi2_result['p_value']:.6f}\n")
                    f.write(f"  Interpretation: {chi2_result['interpretation'].upper()}\n")
                    f.write(f"  Recommendation: {chi2_result['recommendation']}\n\n")

                else:
                    # Individual test
                    f.write(f"Test: {result.get('test_name', 'Unknown')}\n")
                    if 'chi2_statistic' in result:
                        f.write(f"χ² statistic: {result['chi2_statistic']:.4f}\n")
                    if 'statistic' in result:
                        f.write(f"Test statistic: {result['statistic']:.4f}\n")
                    f.write(f"P-value: {result['p_value']:.6f}\n")
                    f.write(f"Result: {'REJECT H0' if result.get('reject_null', False) else 'FAIL TO REJECT H0'}\n\n")

                f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"✓ Statistical report saved to: {output_path}")

        return output_path


    def save_results(self, output_path: str = None) -> str:
        """
        Save test results to pickle file.

        Args:
            output_path: Path to save results (default: results/hypothesis_results.pkl)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = os.path.join(
                analysis_results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_hypothesis_results.pkl"
            )

        # Use ensure_dir_exists from Utils instead of os.makedirs
        ensure_dir_exists(os.path.dirname(output_path))

        with open(output_path, 'wb') as f:
            pickle.dump(self.test_results, f)

        print(f"✓ Hypothesis test results saved to: {output_path}")

        return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_class_balance(dataset, task_type: str = 'refusal', alpha: float = None) -> Dict:
    """
    Quick function to analyze class balance with hypothesis testing.

    Args:
        dataset: Dataset with labels
        task_type: 'refusal' or 'jailbreak'
        alpha: Significance level (default: from HYPOTHESIS_TESTING_CONFIG)

    Returns:
        Statistical analysis results
    """
    # Use config value if not provided - NO HARDCODING!
    if alpha is None:
        alpha = HYPOTHESIS_TESTING_CONFIG['alpha']
    
    tester = DatasetValidator(alpha=alpha)
    results = tester.analyze_dataset_statistics(dataset, task_type=task_type)

    # Save results
    tester.save_results()
    tester.generate_statistical_report()

    return results


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 1, 2025
@author: ramyalsaffar
"""
