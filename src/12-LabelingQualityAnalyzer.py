# Labeling Quality Analyzer Module 
#----------------------------------
# Analyzes judge labeling quality and confidence scores.
# Helps identify low-quality labels and systematic issues.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Handles new is_jailbreak_attempt field
# - Better statistics with safe_divide
# - Visualization support
# - Export capabilities enhanced
# All imports are in 01-Imports.py
###############################################################################


class LabelingQualityAnalyzer:
    """Analyze judge labeling quality using confidence scores."""
    
    def __init__(self, verbose: bool = None):
        """
        Initialize analyzer.
        
        Args:
            verbose: Print detailed output (default: from EXPERIMENT_CONFIG)
        """
        # Use config values - NO HARDCODING!
        self.verbose = verbose if verbose is not None else EXPERIMENT_CONFIG.get('verbose', True)
        self.class_names = CLASS_NAMES
        self.threshold = LABELING_CONFIG['low_confidence_threshold']
        
        # Statistics tracking
        self.stats = {
            'total_analyzed': 0,
            'error_labels': 0,
            'low_confidence_samples': 0,
            'analysis_timestamp': None
        }

    def analyze_full(self, labeled_df: pd.DataFrame) -> Dict:
        """
        Run complete labeling quality analysis.
        
        Args:
            labeled_df: DataFrame with columns:
                - refusal_label, refusal_confidence
                - is_jailbreak_attempt, jailbreak_label, jailbreak_confidence
                - model (LLM that generated response)
                - category (prompt category)
        
        Returns:
            Dictionary with all analysis results
        """
        if self.verbose:
            print_banner("LABELING QUALITY ANALYSIS", char="=")
        
        self.stats['total_analyzed'] = len(labeled_df)
        self.stats['analysis_timestamp'] = get_timestamp('display')
        
        results = {
            'timestamp': self.stats['analysis_timestamp'],
            'total_samples': len(labeled_df)
        }
        
        # Overall confidence statistics
        results['overall'] = self._analyze_overall_confidence(labeled_df)
        
        # Jailbreak attempt analysis (NEW)
        if 'is_jailbreak_attempt' in labeled_df.columns:
            results['jailbreak_attempts'] = self._analyze_jailbreak_attempts(labeled_df)
        
        # Per-model analysis
        if 'model' in labeled_df.columns:
            results['per_model'] = self._analyze_per_model_confidence(labeled_df)
        
        # Per-category analysis
        if 'category' in labeled_df.columns:
            results['per_category'] = self._analyze_per_category_confidence(labeled_df)
        
        # Low confidence flags
        results['low_confidence'] = self._flag_low_confidence(labeled_df)
        
        # Task agreement analysis
        results['task_agreement'] = self._analyze_task_agreement(labeled_df)
        
        # Label distribution
        results['label_distribution'] = self._analyze_label_distribution(labeled_df)
        
        if self.verbose:
            self._print_summary(results)
        
        return results

    def _analyze_overall_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze overall confidence statistics."""
        if self.verbose:
            print_banner("Overall Confidence Statistics", char="─")
        
        # Filter out error labels
        valid_refusal = df[df['refusal_label'] != -1]
        valid_jailbreak = df[df['jailbreak_label'] != -1]

        # Calculate statistics safely
        results = {
            'refusal': self._calculate_confidence_stats(
                valid_refusal['refusal_confidence'],
                'Refusal'
            ),
            'jailbreak': self._calculate_confidence_stats(
                valid_jailbreak['jailbreak_confidence'],
                'Jailbreak'
            )
        }

        # Track error rates
        refusal_error_rate = safe_divide(
            (df['refusal_label'] == -1).sum(),
            len(df), 0
        ) * 100
        jailbreak_error_rate = safe_divide(
            (df['jailbreak_label'] == -1).sum(),
            len(df), 0
        ) * 100
        
        results['error_rates'] = {
            'refusal': refusal_error_rate,
            'jailbreak': jailbreak_error_rate
        }
        
        if self.verbose:
            print(f"  Refusal - Mean: {results['refusal']['mean']:.1f}%, "
                  f"Median: {results['refusal']['median']:.1f}%, "
                  f"Low (<{self.threshold}%): {results['refusal']['count_low']} "
                  f"({results['refusal']['percent_low']:.1f}%)")
            print(f"  Jailbreak - Mean: {results['jailbreak']['mean']:.1f}%, "
                  f"Median: {results['jailbreak']['median']:.1f}%, "
                  f"Low (<{self.threshold}%): {results['jailbreak']['count_low']} "
                  f"({results['jailbreak']['percent_low']:.1f}%)")
            print(f"  Error rates - Refusal: {refusal_error_rate:.1f}%, "
                  f"Jailbreak: {jailbreak_error_rate:.1f}%")
        
        return results

    def _calculate_confidence_stats(self, confidence_series: pd.Series, 
                                   label_name: str) -> Dict:
        """Calculate confidence statistics safely."""
        if len(confidence_series) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0, 'count_low': 0, 'percent_low': 0.0
            }
        
        count_low = (confidence_series < self.threshold).sum()
        percent_low = safe_divide(count_low, len(confidence_series), 0) * 100
        
        return {
            'mean': float(confidence_series.mean()),
            'std': float(confidence_series.std()),
            'median': float(confidence_series.median()),
            'min': float(confidence_series.min()),
            'max': float(confidence_series.max()),
            'count_low': int(count_low),
            'percent_low': float(percent_low)
        }

    def _analyze_jailbreak_attempts(self, df: pd.DataFrame) -> Dict:
        """Analyze jailbreak attempt detection (NEW)."""
        if self.verbose:
            print_banner("Jailbreak Attempt Analysis", char="─")
        
        total = len(df)
        attempts = df['is_jailbreak_attempt'].sum()
        attempt_rate = safe_divide(attempts, total, 0) * 100
        
        # Among attempts, how many succeeded?
        attempts_df = df[df['is_jailbreak_attempt'] == 1]
        if len(attempts_df) > 0:
            successes = (attempts_df['jailbreak_label'] == 1).sum()
            success_rate = safe_divide(successes, len(attempts_df), 0) * 100
        else:
            successes = 0
            success_rate = 0.0
        
        results = {
            'total_attempts': int(attempts),
            'attempt_rate': float(attempt_rate),
            'total_successes': int(successes),
            'success_rate': float(success_rate),
            'defense_rate': float(100 - success_rate)
        }
        
        if self.verbose:
            print(f"  Jailbreak attempts: {attempts:,}/{total:,} ({attempt_rate:.1f}%)")
            if attempts > 0:
                print(f"  Success rate: {successes:,}/{attempts:,} ({success_rate:.1f}%)")
                print(f"  Defense rate: {100 - success_rate:.1f}%")
        
        return results

    def _analyze_per_model_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence by LLM model."""
        if self.verbose:
            print_banner("Per-Model Confidence", char="─")
        
        results = {}
        valid_df = df[df['refusal_label'] != -1]
        
        for model in valid_df['model'].unique():
            model_df = valid_df[valid_df['model'] == model]
            
            refusal_low = (model_df['refusal_confidence'] < self.threshold).sum()
            jailbreak_low = (model_df['jailbreak_confidence'] < self.threshold).sum()
            
            results[model] = {
                'sample_count': len(model_df),
                'refusal': {
                    'mean': float(model_df['refusal_confidence'].mean()),
                    'count_low': int(refusal_low),
                    'percent_low': safe_divide(refusal_low, len(model_df), 0) * 100
                },
                'jailbreak': {
                    'mean': float(model_df['jailbreak_confidence'].mean()),
                    'count_low': int(jailbreak_low),
                    'percent_low': safe_divide(jailbreak_low, len(model_df), 0) * 100
                }
            }
            
            # Add jailbreak attempt stats if available
            if 'is_jailbreak_attempt' in model_df.columns:
                attempts = model_df['is_jailbreak_attempt'].sum()
                results[model]['jailbreak_attempts'] = int(attempts)
                results[model]['attempt_rate'] = safe_divide(attempts, len(model_df), 0) * 100
            
            if self.verbose:
                display_name = get_model_display_name(model) if 'get_model_display_name' in globals() else model
                print(f"\n  {display_name}:")
                print(f"    Samples: {len(model_df):,}")
                print(f"    Refusal - Mean: {results[model]['refusal']['mean']:.1f}%, "
                      f"Low: {refusal_low} ({results[model]['refusal']['percent_low']:.1f}%)")
                print(f"    Jailbreak - Mean: {results[model]['jailbreak']['mean']:.1f}%, "
                      f"Low: {jailbreak_low} ({results[model]['jailbreak']['percent_low']:.1f}%)")
        
        return results

    def _analyze_per_category_confidence(self, df: pd.DataFrame) -> Dict:
        """Analyze confidence by prompt category."""
        if self.verbose:
            print_banner("Per-Category Confidence", char="─")
        
        results = {}
        valid_df = df[df['refusal_label'] != -1]
        
        if 'category' not in valid_df.columns:
            if self.verbose:
                print("  No category column found")
            return results
        
        for category in valid_df['category'].unique():
            cat_df = valid_df[valid_df['category'] == category]
            
            refusal_low = (cat_df['refusal_confidence'] < self.threshold).sum()
            jailbreak_low = (cat_df['jailbreak_confidence'] < self.threshold).sum()
            
            results[category] = {
                'sample_count': len(cat_df),
                'refusal': {
                    'mean': float(cat_df['refusal_confidence'].mean()),
                    'count_low': int(refusal_low),
                    'percent_low': safe_divide(refusal_low, len(cat_df), 0) * 100
                },
                'jailbreak': {
                    'mean': float(cat_df['jailbreak_confidence'].mean()),
                    'count_low': int(jailbreak_low),
                    'percent_low': safe_divide(jailbreak_low, len(cat_df), 0) * 100
                }
            }
        
        # Print top 5 categories with lowest confidence
        if self.verbose and results:
            sorted_cats = sorted(results.items(),
                               key=lambda x: x[1]['refusal']['mean'])[:5]
            print("\n  Top 5 categories with lowest refusal confidence:")
            for cat, metrics in sorted_cats:
                print(f"    {cat}: {metrics['refusal']['mean']:.1f}% "
                      f"(n={metrics['sample_count']})")
        
        return results

    def _flag_low_confidence(self, df: pd.DataFrame) -> Dict:
        """Flag samples with low confidence for review."""
        if self.verbose:
            print_banner(f"Low Confidence Flags (threshold: {self.threshold}%)", char="─")
        
        valid_df = df[df['refusal_label'] != -1]
        
        # Find low confidence samples
        low_refusal = valid_df[valid_df['refusal_confidence'] < self.threshold]
        low_jailbreak = valid_df[valid_df['jailbreak_confidence'] < self.threshold]
        low_both = valid_df[(valid_df['refusal_confidence'] < self.threshold) &
                           (valid_df['jailbreak_confidence'] < self.threshold)]
        
        self.stats['low_confidence_samples'] = len(low_both)
        
        results = {
            'threshold': self.threshold,
            'low_refusal_count': len(low_refusal),
            'low_jailbreak_count': len(low_jailbreak),
            'low_both_count': len(low_both),
            'low_refusal_percent': safe_divide(len(low_refusal), len(valid_df), 0) * 100,
            'low_jailbreak_percent': safe_divide(len(low_jailbreak), len(valid_df), 0) * 100,
            'low_both_percent': safe_divide(len(low_both), len(valid_df), 0) * 100
        }
        
        if self.verbose:
            print(f"  Low refusal confidence: {results['low_refusal_count']:,} "
                  f"({results['low_refusal_percent']:.1f}%)")
            print(f"  Low jailbreak confidence: {results['low_jailbreak_count']:,} "
                  f"({results['low_jailbreak_percent']:.1f}%)")
            print(f"  Low confidence in BOTH: {results['low_both_count']:,} "
                  f"({results['low_both_percent']:.1f}%)")
        
        # Save flagged samples for review
        results['flagged_samples'] = []
        for idx, row in low_both.head(100).iterrows():  # Limit to 100 for JSON size
            results['flagged_samples'].append({
                'index': int(idx),
                'model': row.get('model', 'unknown'),
                'category': row.get('category', 'unknown'),
                'refusal_label': int(row['refusal_label']),
                'refusal_confidence': float(row['refusal_confidence']),
                'jailbreak_label': int(row.get('jailbreak_label', -1)),
                'jailbreak_confidence': float(row['jailbreak_confidence']),
                'is_jailbreak_attempt': int(row.get('is_jailbreak_attempt', 0))
            })
        
        if self.verbose and results['flagged_samples']:
            print(f"  Flagged {len(results['flagged_samples'])} samples for review")
        
        return results

    def _analyze_task_agreement(self, df: pd.DataFrame) -> Dict:
        """Analyze agreement between refusal and jailbreak labeling."""
        if self.verbose:
            print_banner("Task Agreement Analysis", char="─")

        valid_df = df[(df['refusal_label'] != -1) & (df['jailbreak_label'] != -1)]

        # Confidence correlation
        correlation = valid_df['refusal_confidence'].corr(valid_df['jailbreak_confidence'])


        alpha = HYPOTHESIS_TESTING_CONFIG.get('alpha', 0.05)
        n = len(valid_df)
        if n > 3:  # Need at least 4 samples
            # Test if correlation is significantly different from 0
            t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
            p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 2))
            correlation_significant = p_value < alpha
        else:
            p_value = 1.0
            correlation_significant = False




        # Logical consistency checks
        # No refusal (0) but jailbreak succeeded (1) - expected for successful jailbreaks
        no_refusal_jb_success = len(valid_df[
            (valid_df['refusal_label'] == 0) &
            (valid_df['jailbreak_label'] == 1)
        ])

        # Refusal (1 or 2) but jailbreak succeeded (1) - inconsistent!
        refusal_but_jb_success = len(valid_df[
            (valid_df['refusal_label'].isin([1, 2])) &
            (valid_df['jailbreak_label'] == 1)
        ])
        
        results = {
            'confidence_correlation': float(correlation) if not pd.isna(correlation) else 0.0,
            'correlation_p_value': float(p_value),                      
            'correlation_significant': bool(correlation_significant),
            'no_refusal_jb_success': int(no_refusal_jb_success),
            'refusal_but_jb_success': int(refusal_but_jb_success),  # This is problematic
            'inconsistency_rate': safe_divide(refusal_but_jb_success, len(valid_df), 0) * 100
        }
        
        if self.verbose:
            print(f"  Confidence correlation: {results['confidence_correlation']:.3f} "
                  f"(p={p_value:.4f}, {'significant' if correlation_significant else 'not significant'})")
            print(f"  No refusal + jailbreak success: {no_refusal_jb_success:,} (expected)")
            print(f"  Refusal + jailbreak success: {refusal_but_jb_success:,} "
                  f"({results['inconsistency_rate']:.1f}% - INCONSISTENT!)")
        
        return results

    def _analyze_label_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze label distribution."""
        results = {}
        
        # Refusal distribution
        if 'refusal_label' in df.columns:
            refusal_dist = df['refusal_label'].value_counts().to_dict()
            results['refusal'] = {
                (self.class_names[k] if 0 <= k < len(self.class_names) else f'Label_{k}'): v
                for k, v in refusal_dist.items()
            }
        
        # Jailbreak distribution
        if 'jailbreak_label' in df.columns:
            jb_dist = df['jailbreak_label'].value_counts().to_dict()
            results['jailbreak'] = {
                'Failed': jb_dist.get(0, 0),
                'Succeeded': jb_dist.get(1, 0),
                'Error': jb_dist.get(-1, 0)
            }
        
        # Jailbreak attempt distribution
        if 'is_jailbreak_attempt' in df.columns:
            attempt_dist = df['is_jailbreak_attempt'].value_counts().to_dict()
            results['jailbreak_attempts'] = {
                'Not Attempt': attempt_dist.get(0, 0),
                'Is Attempt': attempt_dist.get(1, 0)
            }
        
        return results

    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print_banner("ANALYSIS SUMMARY", char="=")
        print(f"Total samples analyzed: {results['total_samples']:,}")
        print(f"Timestamp: {results['timestamp']}")
        
        if 'jailbreak_attempts' in results:
            print(f"\nJailbreak Statistics:")
            print(f"  Attempt rate: {results['jailbreak_attempts']['attempt_rate']:.1f}%")
            print(f"  Success rate: {results['jailbreak_attempts']['success_rate']:.1f}%")
        
        if 'low_confidence' in results:
            print(f"\nLow Confidence Samples:")
            print(f"  Both tasks: {results['low_confidence']['low_both_count']:,} "
                  f"({results['low_confidence']['low_both_percent']:.1f}%)")
        
        if 'task_agreement' in results:
            print(f"\nTask Agreement:")
            print(f"  Inconsistency rate: {results['task_agreement']['inconsistency_rate']:.1f}%")

    def export_flagged_samples(self, labeled_df: pd.DataFrame, output_path: str):
        """Export low-confidence samples for manual review."""
        ensure_dir_exists(os.path.dirname(output_path))
        
        valid_df = labeled_df[labeled_df['refusal_label'] != -1]
        low_both = valid_df[
            (valid_df['refusal_confidence'] < self.threshold) &
            (valid_df['jailbreak_confidence'] < self.threshold)
        ]
        
        # Select relevant columns
        export_cols = [
            'prompt', 'response', 'model', 'category',
            'refusal_label', 'refusal_confidence',
            'is_jailbreak_attempt', 'jailbreak_label', 'jailbreak_confidence'
        ]
        export_cols = [col for col in export_cols if col in low_both.columns]
        
        low_both[export_cols].to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\n✅ Exported {len(low_both):,} low-confidence samples to {output_path}")

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        ensure_dir_exists(os.path.dirname(output_path))

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Saved labeling quality analysis to {output_path}")


class QualityAnalyzer:
    """
    Unified quality analysis for both labels and model predictions.
    
    Two-phase analysis:
    1. Label Quality (Step 4): Validates GPT-4 labeling quality after DataLabeler
    2. Model Quality (Step 8): Validates trained model predictions vs ground truth
    
    This class wraps LabelingQualityAnalyzer for label analysis and adds
    model prediction quality analysis including calibration and agreement metrics.
    """
    
    def __init__(self, verbose: bool = None):
        """
        Initialize quality analyzer.
        
        Args:
            verbose: Print detailed output (default: from EXPERIMENT_CONFIG)
        """
        # Use config values - NO HARDCODING!
        self.verbose = verbose if verbose is not None else EXPERIMENT_CONFIG.get('verbose', True)
        self.label_analyzer = LabelingQualityAnalyzer(verbose=verbose)
        self.threshold = LABELING_CONFIG['low_confidence_threshold']  # 70% from config
        
        # Statistics tracking
        self.stats = {
            'label_quality_checked': False,
            'model_quality_checked': False,
            'analysis_timestamp': None
        }
    
    # =========================================================================
    # PHASE 1: LABEL QUALITY ANALYSIS (Step 4 - After GPT-4 Labeling)
    # =========================================================================
    
    def analyze_label_quality(self, labeled_df: pd.DataFrame, timestamp: str = None, save_results: bool = True) -> Dict:
        """
        Phase 1: Analyze GPT-4 label quality after DataLabeler.
        
        This uses LabelingQualityAnalyzer to validate labeling quality.
        Reports warnings if quality is low but never blocks pipeline.
        
        Args:
            labeled_df: DataFrame with GPT-4 labels and confidence scores
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with label quality metrics and recommendations
        """
        if self.verbose:
            print_banner("PHASE 1: LABEL QUALITY ANALYSIS", char="=", width=60)
            print("Analyzing GPT-4 labeling quality...\n")
        
        # Use LabelingQualityAnalyzer for comprehensive label analysis
        label_results = self.label_analyzer.analyze_full(labeled_df)
        
        # Add quality assessment
        quality_assessment = self._assess_label_quality(label_results)
        label_results['quality_assessment'] = quality_assessment
        
        # Print warnings if quality is low
        if quality_assessment['overall_quality'] == 'poor':
            print()
            print_banner("⚠️  WARNING: LOW LABEL QUALITY DETECTED", char="!", width=60)
            print(f"  Overall quality: {quality_assessment['overall_quality'].upper()}")
            print(f"  Refusal confidence: {label_results['overall']['refusal']['mean']:.1f}%")
            print(f"  Jailbreak confidence: {label_results['overall']['jailbreak']['mean']:.1f}%")
            print(f"  Recommendation: {quality_assessment['recommendation']}")
            print("!" * 60)
            print()
        elif self.verbose:
            print()
            print(f"✅ Label quality: {quality_assessment['overall_quality'].upper()}")
            print(f"   Refusal confidence: {label_results['overall']['refusal']['mean']:.1f}%")
            print(f"   Jailbreak confidence: {label_results['overall']['jailbreak']['mean']:.1f}%")
            print()
        
        # Save results if requested
        if save_results:
            if timestamp is None:
                timestamp = get_timestamp('file')  # Fallback if not provided
            
            output_path = os.path.join(
                quality_review_path, 
                f"label_quality_analysis_{timestamp}.json"
            )
            self.label_analyzer.save_results(label_results, output_path)
            
            # Export flagged low-confidence samples
            flagged_path = os.path.join(
                quality_review_path,
                f"low_confidence_labels_{timestamp}.csv"
            )
            self.label_analyzer.export_flagged_samples(labeled_df, flagged_path)
        
        self.stats['label_quality_checked'] = True
        self.stats['analysis_timestamp'] = get_timestamp('display')
        
        return label_results
    
    def _assess_label_quality(self, label_results: Dict) -> Dict:
        """
        Assess overall label quality and provide recommendations.
        
        Args:
            label_results: Results from LabelingQualityAnalyzer
        
        Returns:
            Dictionary with quality assessment
        """
        # Get average confidences
        refusal_conf = label_results['overall']['refusal']['mean']
        jailbreak_conf = label_results['overall']['jailbreak']['mean']
        avg_conf = (refusal_conf + jailbreak_conf) / 2
        
        # Get error rates
        refusal_error_rate = label_results['overall']['error_rates']['refusal']
        jailbreak_error_rate = label_results['overall']['error_rates']['jailbreak']
        avg_error_rate = (refusal_error_rate + jailbreak_error_rate) / 2
        
        # Assess quality based on thresholds
        if avg_conf >= 85 and avg_error_rate < 5:
            quality = 'excellent'
            recommendation = 'Labels are high quality. Proceed with confidence.'
        elif avg_conf >= 70 and avg_error_rate < 10:
            quality = 'good'
            recommendation = 'Labels are acceptable quality. Proceed with training.'
        elif avg_conf >= 60 and avg_error_rate < 15:
            quality = 'fair'
            recommendation = 'Labels are marginal quality. Consider reviewing low-confidence samples.'
        else:
            quality = 'poor'
            recommendation = 'Labels are low quality. Consider relabeling or reviewing thresholds.'
        
        return {
            'overall_quality': quality,
            'avg_confidence': avg_conf,
            'avg_error_rate': avg_error_rate,
            'refusal_confidence': refusal_conf,
            'jailbreak_confidence': jailbreak_conf,
            'refusal_error_rate': refusal_error_rate,
            'jailbreak_error_rate': jailbreak_error_rate,
            'recommendation': recommendation,
            'meets_threshold': avg_conf >= self.threshold
        }
    
    # =========================================================================
    # PHASE 2: MODEL QUALITY ANALYSIS (Step 8 - After Training)
    # =========================================================================
    
    def analyze_model_agreement(self, 
                               test_df: pd.DataFrame,
                               refusal_confidences: np.ndarray,
                               jailbreak_confidences: np.ndarray,
                               timestamp: str = None,
                              classifier_name: str = 'dual_classifier',
                              save_results: bool = True) -> Dict:
        """
        Phase 2: Analyze model prediction quality and agreement with labels.
        
        Compares model predictions against ground truth labels, analyzes
        calibration, and identifies patterns where model and labels disagree.
        
        Args:
            test_df: Test DataFrame with ground truth labels
            refusal_confidences: Refusal model confidence scores (array)
            jailbreak_confidences: Jailbreak model confidence scores (array)
            timestamp: Timestamp to use in filename (uses current time if None)
            classifier_name: Name prefix for output file (default: 'dual_classifier')
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with model quality analysis results
        """
        if self.verbose:
            print_banner("PHASE 2: MODEL QUALITY ANALYSIS", char="=", width=60)
            print("Analyzing model prediction quality...\n")
        
        results = {
            'timestamp': get_timestamp('display'),
            'total_samples': len(test_df)
        }
        
        # Analyze refusal classifier quality
        if 'refusal_label' in test_df.columns and 'refusal_confidence' in test_df.columns:
            results['refusal_quality'] = self._analyze_classifier_quality(
                test_df,
                label_col='refusal_label',
                label_conf_col='refusal_confidence',
                model_conf=refusal_confidences,
                classifier_name='Refusal'
            )
        
        # Analyze jailbreak detector quality
        if 'jailbreak_label' in test_df.columns and 'jailbreak_confidence' in test_df.columns:
            results['jailbreak_quality'] = self._analyze_classifier_quality(
                test_df,
                label_col='jailbreak_label',
                label_conf_col='jailbreak_confidence',
                model_conf=jailbreak_confidences,
                classifier_name='Jailbreak'
            )
        
        # Cross-task agreement analysis
        results['cross_task'] = self._analyze_cross_task_agreement(
            test_df, refusal_confidences, jailbreak_confidences
        )
        
        # Overall quality summary
        results['summary'] = self._summarize_model_quality(results)
        
        if self.verbose:
            self._print_model_quality_summary(results)
        
        # Save results if requested
        if save_results:
            if timestamp is None:
                timestamp = get_timestamp('file')  # Fallback if not provided
            output_path = os.path.join(
                quality_review_path,
                f"{classifier_name}_model_quality_analysis_{timestamp}.json"
            )
            ensure_dir_exists(quality_review_path)
            
            serializable_results = convert_to_serializable(results)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            if self.verbose:
                print(f"\n✅ Saved model quality analysis to {output_path}")
        
        self.stats['model_quality_checked'] = True
        
        return results
    
    def _analyze_classifier_quality(self,
                                   test_df: pd.DataFrame,
                                   label_col: str,
                                   label_conf_col: str,
                                   model_conf: np.ndarray,
                                   classifier_name: str) -> Dict:
        """
        Analyze quality of a single classifier's predictions.
        
        Args:
            test_df: Test DataFrame
            label_col: Column with ground truth labels
            label_conf_col: Column with label confidence scores
            model_conf: Model confidence scores (array)
            classifier_name: Name of classifier for display
        
        Returns:
            Dictionary with classifier quality metrics
        """
        # Filter out error labels (-1)
        # CRITICAL: Ensure model_conf is numpy array for boolean indexing
        model_conf = np.asarray(model_conf)
        
        # DEBUG: Print types
        if self.verbose:
            print(f"  DEBUG: model_conf type = {type(model_conf)}, shape = {getattr(model_conf, 'shape', 'NO SHAPE ATTR')}")
            print(f"  DEBUG: model_conf dtype = {getattr(model_conf, 'dtype', 'NO DTYPE')}")
            print(f"  DEBUG: model_conf sample = {model_conf[:3] if len(model_conf) > 0 else 'EMPTY'}")
        
        valid_mask = test_df[label_col] != -1
        valid_df = test_df[valid_mask].copy()
        
        # Convert mask to numpy array explicitly
        mask_array = valid_mask.values if hasattr(valid_mask, 'values') else np.array(valid_mask)
        
        # DEBUG: Print mask info
        if self.verbose:
            print(f"  DEBUG: mask type = {type(mask_array)}, shape = {mask_array.shape}")
            print(f"  DEBUG: mask sum = {mask_array.sum()}")
        
        # BACKWARD COMPATIBILITY: Normalize label confidences if they're on 0-100 scale
        # This handles legacy data from before the DataLabeler fix (Nov 15, 2025)
        # New data should already be on 0-1 scale, but old pickled test data may have integers
        label_conf_values = valid_df[label_conf_col].values
        if label_conf_values.dtype in [np.int32, np.int64] or (len(label_conf_values) > 0 and np.max(label_conf_values) > 1.0):
            if self.verbose:
                print(f"  ⚠️  Legacy data detected: Normalizing {classifier_name} label confidences from 0-100 to 0-1 scale")
            # FIX: Convert to float64 array explicitly to avoid object dtype issues
            normalized_values = (valid_df[label_conf_col].values / 100.0).astype(np.float64)
            valid_df[label_conf_col] = normalized_values
        
        valid_model_conf = model_conf[mask_array]
        
        if len(valid_df) == 0:
            return {
                'error': 'No valid samples',
                'sample_count': 0
            }
        
        # Model confidence statistics
        model_conf_stats = {
            'mean': float(np.mean(valid_model_conf)),
            'std': float(np.std(valid_model_conf)),
            'median': float(np.median(valid_model_conf)),
            'min': float(np.min(valid_model_conf)),
            'max': float(np.max(valid_model_conf))
        }
        
        # Label confidence statistics (for comparison)
        label_conf_stats = {
            'mean': float(valid_df[label_conf_col].mean()),
            'std': float(valid_df[label_conf_col].std()),
            'median': float(valid_df[label_conf_col].median()),
            'min': float(valid_df[label_conf_col].min()),
            'max': float(valid_df[label_conf_col].max())
        }
        
        # Confidence correlation (model vs label)
        # Convert both to clean numpy arrays (avoid pandas Series issues)
        model_conf_array = np.asarray(valid_model_conf).flatten()
        label_conf_array = np.asarray(valid_df[label_conf_col].values).flatten()
        
        # DEBUG: Check array types and shapes
        if self.verbose:
            print(f"  DEBUG BEFORE CORRCOEF:")
            print(f"    model_conf_array: type={type(model_conf_array)}, shape={model_conf_array.shape}, dtype={model_conf_array.dtype}")
            print(f"    label_conf_array: type={type(label_conf_array)}, shape={label_conf_array.shape}, dtype={label_conf_array.dtype}")
        
        # Ensure both arrays have same length and are valid 1D arrays
        if len(model_conf_array) != len(label_conf_array):
            if self.verbose:
                print(f"  ⚠️  Length mismatch: model={len(model_conf_array)}, label={len(label_conf_array)}")
            correlation = 0.0
        elif len(model_conf_array) == 0 or len(label_conf_array) == 0:
            if self.verbose:
                print(f"  ⚠️  Empty arrays: model={len(model_conf_array)}, label={len(label_conf_array)}")
            correlation = 0.0
        elif model_conf_array.ndim != 1 or label_conf_array.ndim != 1:
            if self.verbose:
                print(f"  ⚠️  Invalid dimensions: model={model_conf_array.ndim}D, label={label_conf_array.ndim}D")
            correlation = 0.0
        else:
            # Reshape to ensure proper 1D arrays for corrcoef
            model_conf_array = model_conf_array.reshape(-1)
            label_conf_array = label_conf_array.reshape(-1)
            
            if self.verbose:
                print(f"  DEBUG CALLING CORRCOEF:")
                print(f"    model_conf_array after reshape: shape={model_conf_array.shape}")
                print(f"    label_conf_array after reshape: shape={label_conf_array.shape}")
            
            correlation = np.corrcoef(model_conf_array, label_conf_array)[0, 1]
        
        # Low confidence analysis
        low_model_conf = (valid_model_conf < self.threshold).sum()
        low_label_conf = (valid_df[label_conf_col] < self.threshold).sum()
        low_both = ((valid_model_conf < self.threshold) & 
                   (valid_df[label_conf_col] < self.threshold)).sum()
        
        results = {
            'sample_count': len(valid_df),
            'model_confidence': model_conf_stats,
            'label_confidence': label_conf_stats,
            'confidence_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'low_confidence_counts': {
                'model': int(low_model_conf),
                'label': int(low_label_conf),
                'both': int(low_both),
                'model_percent': safe_divide(low_model_conf, len(valid_df), 0) * 100,
                'label_percent': safe_divide(low_label_conf, len(valid_df), 0) * 100,
                'both_percent': safe_divide(low_both, len(valid_df), 0) * 100
            }
        }
        
        if self.verbose:
            print(f"\n{classifier_name} Classifier Quality:")
            print(f"  Model confidence: {model_conf_stats['mean']:.1f}% "
                  f"(±{model_conf_stats['std']:.1f}%)")
            print(f"  Label confidence: {label_conf_stats['mean']:.1f}% "
                  f"(±{label_conf_stats['std']:.1f}%)")
            print(f"  Confidence correlation: {correlation:.3f}")
            print(f"  Low model confidence: {low_model_conf}/{len(valid_df)} "
                  f"({results['low_confidence_counts']['model_percent']:.1f}%)")
        
        return results
    
    def _analyze_cross_task_agreement(self,
                                     test_df: pd.DataFrame,
                                     refusal_conf: np.ndarray,
                                     jailbreak_conf: np.ndarray) -> Dict:
        """
        Analyze agreement between refusal and jailbreak classifiers.
        
        Args:
            test_df: Test DataFrame
            refusal_conf: Refusal model confidences
            jailbreak_conf: Jailbreak model confidences
        
        Returns:
            Dictionary with cross-task agreement metrics
        """
        # CRITICAL: Ensure confidences are numpy arrays for boolean indexing
        refusal_conf = np.asarray(refusal_conf)
        jailbreak_conf = np.asarray(jailbreak_conf)
        
        # DEBUG: Print types
        if self.verbose:
            print(f"\n  CROSS-TASK DEBUG:")
            print(f"  refusal_conf type = {type(refusal_conf)}, shape = {getattr(refusal_conf, 'shape', 'NO SHAPE')}")
            print(f"  jailbreak_conf type = {type(jailbreak_conf)}, shape = {getattr(jailbreak_conf, 'shape', 'NO SHAPE')}")
        
        # Filter valid samples
        valid_mask = ((test_df['refusal_label'] != -1) & 
                     (test_df['jailbreak_label'] != -1))
        
        if valid_mask.sum() == 0:
            return {'error': 'No valid samples for cross-task analysis'}
        
        # Convert mask to numpy array explicitly
        mask_array = valid_mask.values if hasattr(valid_mask, 'values') else np.array(valid_mask)
        
        if self.verbose:
            print(f"  mask type = {type(mask_array)}, shape = {mask_array.shape}")
            print(f"  About to index refusal_conf...")
        
        valid_refusal_conf = refusal_conf[mask_array]
        
        if self.verbose:
            print(f"  ✓ Indexed refusal_conf successfully")
            print(f"  About to index jailbreak_conf...")
        
        valid_jailbreak_conf = jailbreak_conf[mask_array]
        
        if self.verbose:
            print(f"  ✓ Indexed jailbreak_conf successfully")
        
        # Correlation between model confidences
        # Ensure clean 1D arrays for corrcoef
        refusal_array = np.asarray(valid_refusal_conf).flatten()
        jailbreak_array = np.asarray(valid_jailbreak_conf).flatten()
        
        model_correlation = np.corrcoef(refusal_array, jailbreak_array)[0, 1]
        
        # Both models low confidence
        both_low = ((valid_refusal_conf < self.threshold) & 
                   (valid_jailbreak_conf < self.threshold)).sum()
        
        # Both models high confidence
        both_high = ((valid_refusal_conf >= 85) & 
                    (valid_jailbreak_conf >= 85)).sum()
        
        results = {
            'model_confidence_correlation': float(model_correlation) if not np.isnan(model_correlation) else 0.0,
            'both_low_confidence': int(both_low),
            'both_high_confidence': int(both_high),
            'both_low_percent': safe_divide(both_low, valid_mask.sum(), 0) * 100,
            'both_high_percent': safe_divide(both_high, valid_mask.sum(), 0) * 100
        }
        
        if self.verbose:
            print(f"\nCross-Task Agreement:")
            print(f"  Model confidence correlation: {model_correlation:.3f}")
            print(f"  Both low confidence: {both_low} ({results['both_low_percent']:.1f}%)")
            print(f"  Both high confidence: {both_high} ({results['both_high_percent']:.1f}%)")
        
        return results
    
    def _summarize_model_quality(self, results: Dict) -> Dict:
        """
        Summarize overall model quality.
        
        Args:
            results: Full analysis results
        
        Returns:
            Summary dictionary
        """
        summary = {
            'timestamp': results['timestamp'],
            'total_samples': results['total_samples']
        }
        
        # Refusal quality summary
        if 'refusal_quality' in results and 'error' not in results['refusal_quality']:
            ref_qual = results['refusal_quality']
            summary['refusal'] = {
                'avg_confidence': ref_qual['model_confidence']['mean'],
                'low_confidence_rate': ref_qual['low_confidence_counts']['model_percent'],
                'label_agreement': ref_qual['confidence_correlation']
            }
        
        # Jailbreak quality summary
        if 'jailbreak_quality' in results and 'error' not in results['jailbreak_quality']:
            jb_qual = results['jailbreak_quality']
            summary['jailbreak'] = {
                'avg_confidence': jb_qual['model_confidence']['mean'],
                'low_confidence_rate': jb_qual['low_confidence_counts']['model_percent'],
                'label_agreement': jb_qual['confidence_correlation']
            }
        
        # Overall assessment
        if 'refusal' in summary and 'jailbreak' in summary:
            avg_conf = (summary['refusal']['avg_confidence'] + 
                       summary['jailbreak']['avg_confidence']) / 2
            
            if avg_conf >= 85:
                quality = 'excellent'
            elif avg_conf >= 75:
                quality = 'good'
            elif avg_conf >= 65:
                quality = 'fair'
            else:
                quality = 'poor'
            
            summary['overall_quality'] = quality
            summary['avg_model_confidence'] = avg_conf
        
        return summary
    
    def _print_model_quality_summary(self, results: Dict):
        """Print model quality summary."""
        if 'summary' not in results:
            return
        
        summary = results['summary']
        
        print()
        print_banner("MODEL QUALITY SUMMARY", char="─", width=60)
        
        if 'overall_quality' in summary:
            print(f"  Overall quality: {summary['overall_quality'].upper()}")
            print(f"  Average model confidence: {summary['avg_model_confidence']:.1f}%")
        
        if 'refusal' in summary:
            print(f"\n  Refusal Classifier:")
            print(f"    Confidence: {summary['refusal']['avg_confidence']:.1f}%")
            print(f"    Low confidence rate: {summary['refusal']['low_confidence_rate']:.1f}%")
            print(f"    Label agreement: {summary['refusal']['label_agreement']:.3f}")
        
        if 'jailbreak' in summary:
            print(f"\n  Jailbreak Detector:")
            print(f"    Confidence: {summary['jailbreak']['avg_confidence']:.1f}%")
            print(f"    Low confidence rate: {summary['jailbreak']['low_confidence_rate']:.1f}%")
            print(f"    Label agreement: {summary['jailbreak']['label_agreement']:.3f}")
        
        print("─" * 60)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def export_flagged_samples(self,
                              test_df: pd.DataFrame,
                              output_path: str,
                              threshold: float = None):
        """
        Export samples with low model confidence for manual review.
        
        This identifies samples where BOTH classifiers have low confidence,
        which may indicate edge cases or labeling issues.
        
        Args:
            test_df: Test DataFrame
            output_path: Path to save CSV
            threshold: Confidence threshold (default: uses config)
        """
        threshold = threshold or self.threshold
        ensure_dir_exists(os.path.dirname(output_path))
        
        # This method expects 'refusal_pred_confidence' and 'jailbreak_pred_confidence'
        # to exist in test_df (added during prediction phase)
        if 'refusal_pred_confidence' not in test_df.columns:
            if self.verbose:
                print("⚠️  Warning: No prediction confidences in test_df, skipping export")
            return
        
        # Find samples with low confidence in both tasks
        low_both = test_df[
            (test_df['refusal_pred_confidence'] < threshold) &
            (test_df['jailbreak_pred_confidence'] < threshold)
        ]
        
        if len(low_both) == 0:
            if self.verbose:
                print(f"✅ No samples with both confidences below {threshold:.0f}%")
            return
        
        # Select relevant columns for review
        export_cols = [
            'prompt', 'response', 'model', 'category',
            'refusal_label', 'refusal_confidence', 'refusal_pred_confidence',
            'jailbreak_label', 'jailbreak_confidence', 'jailbreak_pred_confidence'
        ]
        export_cols = [col for col in export_cols if col in low_both.columns]
        
        low_both[export_cols].to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\n✅ Exported {len(low_both):,} low-confidence samples to {output_path}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 31, 2025
@author: ramyalsaffar
"""
