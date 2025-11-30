# ReportGenerator Module
#-----------------------
# Generates comprehensive PDF reports using ReportLab.
# Creates professional reports for model performance, interpretability,
# production monitoring, and executive summaries.
# 
# IMPROVEMENTS (V10 - FIXED):
# - FIXED: Corrected metric extraction from PerModelAnalyzer structure
# - FIXED: Added support calculation per class
# - FIXED: Added per-class precision/recall extraction
# - ENHANCED: Added Matthews correlation coefficient, log loss, specificity
# - Full Config/Utils integration where applicable
# - Uses safe_divide() from Utils for robust division
# - Uses get_timestamp() from Utils for consistent timestamps
# Requirements: pip install reportlab
# All imports are in 01-Imports.py
###############################################################################


def build_refusal_metrics_from_results(analysis_results):
    """
    Centralized builder for refusal classifier report metrics.
    
    FIXED (V10): Now correctly extracts from PerModelAnalyzer's actual structure:
      - analysis_results['per_model']['models'][model_name] - per model metrics
      - analysis_results['per_model']['summary'] - aggregated stats
      - analysis_results['confidence']['basic_metrics'] - confidence stats
    
    Args:
        analysis_results: Dictionary from ExperimentRunner.analyze_models()
    
    Returns:
        Dictionary with all metrics needed for report generation
    """
    
    metrics = {}
    
    # ------------------------------------------------------------------
    # Extract from actual PerModelAnalyzer structure (FIXED!)
    # ------------------------------------------------------------------
    per_model = analysis_results.get('per_model', {})
    models_dict = per_model.get('models', {})
    summary_stats = per_model.get('summary', {})
    
    confidence = analysis_results.get('confidence', {})
    basic_conf = confidence.get('basic_metrics', {})
    calib_conf = confidence.get('calibration_metrics', {})

    
    # ------------------------------------------------------------------
    # OVERALL METRICS (FIXED: Always calculate from predictions for consistency)
    # ------------------------------------------------------------------
    
    predictions_data = analysis_results.get('predictions', {})
    preds = predictions_data.get('preds', [])
    labels = predictions_data.get('labels', [])
    
    if len(preds) > 0 and len(labels) > 0:
        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        
        valid_mask = labels_arr != -1
        valid_preds = preds_arr[valid_mask]
        valid_labels = labels_arr[valid_mask]
        
        if len(valid_labels) > 0:
            metrics['accuracy'] = float(accuracy_score(valid_labels, valid_preds))
            metrics['macro_f1'] = float(f1_score(valid_labels, valid_preds, average='macro', zero_division=0))
        else:
            metrics['accuracy'] = 0.0
            metrics['macro_f1'] = 0.0
    else:
        # Fallback to other sources
        if summary_stats and 'f1_macro' in summary_stats:
            metrics['macro_f1'] = float(summary_stats['f1_macro'].get('mean', 0.0))
            metrics['accuracy'] = float(summary_stats['accuracy'].get('mean', 0.0))
        elif models_dict:
            all_f1 = [m.get('f1_macro', 0.0) for m in models_dict.values()]
            all_acc = [m.get('accuracy', 0.0) for m in models_dict.values()]
            metrics['macro_f1'] = float(np.mean(all_f1)) if all_f1 else 0.0
            metrics['accuracy'] = float(np.mean(all_acc)) if all_acc else 0.0
        elif basic_conf:
            metrics['accuracy'] = float(basic_conf.get('accuracy', 0.0))
            metrics['macro_f1'] = float(basic_conf.get('macro_f1', 0.0))
        else:
            metrics['accuracy'] = 0.0
            metrics['macro_f1'] = 0.0
    
    
    # Other overall metrics from confidence analyzer (or calculate if missing)
    metrics['weighted_f1'] = float(basic_conf.get('weighted_f1', 0.0))
    metrics['macro_precision'] = float(basic_conf.get('macro_precision', 0.0))
    metrics['macro_recall'] = float(basic_conf.get('macro_recall', 0.0))
    
    # If these are still 0, calculate from predictions
    if metrics['macro_precision'] == 0.0 or metrics['macro_recall'] == 0.0 or metrics['weighted_f1'] == 0.0:
        predictions_data = analysis_results.get('predictions', {})
        preds = predictions_data.get('preds', [])
        labels = predictions_data.get('labels', [])
        
        if len(preds) > 0 and len(labels) > 0:
            preds_arr = np.array(preds)
            labels_arr = np.array(labels)
            valid_mask = labels_arr != -1
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            if len(valid_labels) > 0:
                
                if metrics['macro_precision'] == 0.0:
                    metrics['macro_precision'] = float(precision_score(valid_labels, valid_preds, average='macro', zero_division=0))
                
                if metrics['macro_recall'] == 0.0:
                    metrics['macro_recall'] = float(recall_score(valid_labels, valid_preds, average='macro', zero_division=0))
                
                if metrics['weighted_f1'] == 0.0:
                    metrics['weighted_f1'] = float(f1_score(valid_labels, valid_preds, average='weighted', zero_division=0))
    
    # Agreement / robustness
    metrics['cohen_kappa'] = float(basic_conf.get('cohen_kappa', 0.0))
    metrics['matthews_corrcoef'] = float(basic_conf.get('matthews_corrcoef', 0.0))
    
    # Calculate matthews if missing and we have predictions
    if metrics['matthews_corrcoef'] == 0.0:
        predictions_data = analysis_results.get('predictions', {})
        preds = predictions_data.get('preds', [])
        labels = predictions_data.get('labels', [])
        
        if len(preds) > 0 and len(labels) > 0:
            preds_arr = np.array(preds)
            labels_arr = np.array(labels)
            valid_mask = labels_arr != -1
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            if len(valid_labels) > 0:
                try:
                    metrics['matthews_corrcoef'] = float(matthews_corrcoef(valid_labels, valid_preds))
                except:
                    metrics['matthews_corrcoef'] = 0.0
    
    # Confidence & calibration
    metrics['mean_confidence'] = basic_conf.get('mean_confidence', 0.0)
    metrics['std_confidence'] = basic_conf.get('std_confidence', 0.0)
    metrics['calibration_error'] = calib_conf.get('ece', 0.0)
    metrics['brier_score'] = calib_conf.get('brier_score', 0.0)
    metrics['log_loss'] = calib_conf.get('log_loss', 0.0)
    
    # PHASE 1: NEW - Additional confidence metrics (4 metrics)
    metrics['mce'] = calib_conf.get('mce', 0.0)  # Max Calibration Error
    metrics['confidence_gap'] = basic_conf.get('confidence_gap', 0.0)  # Correct - Incorrect
    metrics['mean_confidence_correct'] = basic_conf.get('mean_confidence_correct', 0.0)  # When right
    metrics['mean_confidence_incorrect'] = basic_conf.get('mean_confidence_incorrect', 0.0)  # When wrong
    
    # Calculate log_loss if missing and we have predictions + probabilities
    if metrics['log_loss'] == 0.0:
        # Check if we have predictions and probabilities
        # Probabilities needed: (n_samples, n_classes)
        metadata = analysis_results.get('metadata', {})
        preds_from_meta = metadata.get('predictions', [])
        labels_from_meta = metadata.get('true_labels', [])
        
        # Also check analysis_results['predictions'] (fallback)
        if not preds_from_meta:
            predictions_data = analysis_results.get('predictions', {})
            preds_from_meta = predictions_data.get('preds', [])
            labels_from_meta = predictions_data.get('labels', [])
        
        if len(preds_from_meta) > 0 and len(labels_from_meta) > 0:
            preds_arr = np.array(preds_from_meta)
            labels_arr = np.array(labels_from_meta)
            
            # Filter out error labels
            valid_mask = labels_arr != -1
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            # For log_loss, we need probability distributions, not just predictions
            # If we don't have probabilities, we can't calculate log_loss properly
            # Check if we have probability data from confidence analyzer
            confidence_data = analysis_results.get('confidence', {})
            
            # Try to get probabilities from saved results
            # This would require the full probability matrix which we may not have
            # For now, leave as 0.0 if we don't have proper probabilities
            # Log loss requires P(y|x) for all classes, not just argmax
            
            # NOTE: Proper log_loss calculation requires probability distributions
            # which are not stored in analysis_results. This would need to be
            # calculated during the ConfidenceAnalyzer step and saved.
            # Leaving as 0.0 is acceptable - log_loss is a secondary metric.
            pass
    
    # ------------------------------------------------------------------
    # DATASET SPLIT SIZES (FIXED V12 - 2025-11-18)
    # ------------------------------------------------------------------
    # Calculate train/val sizes from actual test data and class distribution
    # NO HARDCODING - derives from actual data
    
    metadata = analysis_results.get('metadata', {})
    num_test_samples = metadata.get('num_test_samples', 0)
    
    # Get the actual split information from analysis_results if available
    # This should be populated by ExperimentRunner when it saves metadata
    train_samples = metadata.get('num_train_samples', None)
    val_samples = metadata.get('num_val_samples', None)
    
    # If metadata has actual counts, use them directly
    if train_samples is not None and val_samples is not None:
        metrics['train_samples'] = train_samples
        metrics['val_samples'] = val_samples
        metrics['test_samples'] = num_test_samples
    else:
        # If not in metadata, mark as unavailable
        # DO NOT calculate - this would require config values
        metrics['train_samples'] = 'N/A'
        metrics['val_samples'] = 'N/A'
        metrics['test_samples'] = num_test_samples if num_test_samples > 0 else 'N/A'
    
    # ------------------------------------------------------------------
    # PER-CLASS METRICS (FIXED V11 - 2025-11-18)
    # ------------------------------------------------------------------
    # Extract per-class metrics from multiple sources with fallbacks:
    # 1. analysis_results['per_class_metrics'] (if exists - from ExperimentRunner)
    # 2. Calculate from predictions directly (sklearn classification_report)
    # 3. Confidence metrics from analysis_results['confidence']['per_class_metrics']
    
    class_names = metadata.get('refusal_class_names', CLASS_NAMES)
    
    # SOURCE 1: Try direct per-class metrics (may not exist in current pipeline)
    per_class_metrics_direct = analysis_results.get('per_class_metrics', {})
    
    # Get predictions for calculations
    predictions_data = analysis_results.get('predictions', {})
    preds_from_meta = predictions_data.get('preds', [])
    labels_from_meta = predictions_data.get('labels', [])
    
    # SOURCE 2: If no direct per-class metrics, calculate from predictions
    if not per_class_metrics_direct and len(preds_from_meta) > 0 and len(labels_from_meta) > 0:
        preds_arr = np.array(preds_from_meta)
        labels_arr = np.array(labels_from_meta)
        
        # Filter out error labels
        valid_mask = labels_arr != -1
        if valid_mask.sum() > 0:
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            # Calculate per-class metrics using sklearn
            per_class_report = classification_report(
                valid_labels, 
                valid_preds,
                labels=list(range(len(class_names))),
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Convert to expected format: {'class_0': {...}, 'class_1': {...}, ...}
            per_class_metrics_direct = {}
            for i, class_name in enumerate(class_names):
                if class_name in per_class_report:
                    per_class_metrics_direct[f'class_{i}'] = per_class_report[class_name]
    
    # Calculate confusion matrix for specificity
    confusion_matrix_for_spec = None
    if len(preds_from_meta) > 0 and len(labels_from_meta) > 0:
        
        preds_arr = np.array(preds_from_meta)
        labels_arr = np.array(labels_from_meta)
        
        # Filter out error labels
        valid_mask = labels_arr != -1
        if valid_mask.sum() > 0:
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            # Calculate confusion matrix
            # Labels parameter ensures all classes are included even if missing in predictions
            confusion_matrix_for_spec = confusion_matrix(
                valid_labels,
                valid_preds,
                labels=list(range(len(class_names)))
            )
    
    if per_class_metrics_direct:
        # CORRECT SOURCE: Direct per-class metrics from sklearn classification_report
        for i, class_name in enumerate(class_names):
            # Keys are 'class_0', 'class_1', 'class_2' (from ExperimentRunner line 616)
            class_key = f'class_{i}'
            class_data = per_class_metrics_direct.get(class_key, {})
            
            metrics[f'class_{i}_precision'] = float(class_data.get('precision', 0.0))
            metrics[f'class_{i}_recall'] = float(class_data.get('recall', 0.0))
            metrics[f'class_{i}_f1'] = float(class_data.get('f1-score', 0.0))
            metrics[f'class_{i}_support'] = int(class_data.get('support', 0))
            
            # PHASE 2: Extract per-class confidence metrics from ConfidenceAnalyzer
            per_class_confidence = confidence.get('per_class_metrics', {})
            class_conf_data = per_class_confidence.get(class_name, {})
            
            metrics[f'class_{i}_mean_confidence'] = float(class_conf_data.get('mean_confidence', 0.0))
            metrics[f'class_{i}_std_confidence'] = float(class_conf_data.get('std_confidence', 0.0))
            metrics[f'class_{i}_min_confidence'] = float(class_conf_data.get('min_confidence', 0.0))
            metrics[f'class_{i}_max_confidence'] = float(class_conf_data.get('max_confidence', 0.0))
            metrics[f'class_{i}_class_accuracy'] = float(class_conf_data.get('accuracy', 0.0))
            
            # Calculate specificity from confusion matrix
            # Specificity for class i = TN_i / (TN_i + FP_i)
            # TN_i = all correct predictions for other classes
            # FP_i = incorrect predictions as class i (when true label != i)
            if confusion_matrix_for_spec is not None:
                # True Negatives: sum of all cells except row i and column i
                tn = confusion_matrix_for_spec.sum() - confusion_matrix_for_spec[i, :].sum() - confusion_matrix_for_spec[:, i].sum() + confusion_matrix_for_spec[i, i]
                
                # False Positives: sum of column i except diagonal (predicted as i but wasn't)
                fp = confusion_matrix_for_spec[:, i].sum() - confusion_matrix_for_spec[i, i]
                
                # Specificity = TN / (TN + FP)
                if (tn + fp) > 0:
                    metrics[f'class_{i}_specificity'] = float(tn / (tn + fp))
                else:
                    metrics[f'class_{i}_specificity'] = 0.0
            else:
                metrics[f'class_{i}_specificity'] = 0.0
            
    else:
        # FALLBACK: Calculate from predictions directly (ConfidenceAnalyzer doesn't have precision/recall)
        predictions_data = analysis_results.get('predictions', {})
        preds = predictions_data.get('preds', [])
        labels = predictions_data.get('labels', [])
        
        if len(preds) > 0 and len(labels) > 0:
            # Calculate metrics from scratch
            preds_arr = np.array(preds)
            labels_arr = np.array(labels)
            
            # Filter out error labels
            valid_mask = labels_arr != -1
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            if len(valid_labels) > 0:
                
                precision, recall, f1, support = precision_recall_fscore_support(
                    valid_labels,
                    valid_preds,
                    labels=list(range(len(class_names))),
                    zero_division=0
                )
                
                # Calculate confusion matrix for specificity
                cm_fallback = confusion_matrix(
                    valid_labels,
                    valid_preds,
                    labels=list(range(len(class_names)))
                )
                
                for i in range(len(class_names)):
                    metrics[f'class_{i}_precision'] = float(precision[i])
                    metrics[f'class_{i}_recall'] = float(recall[i])
                    metrics[f'class_{i}_f1'] = float(f1[i])
                    metrics[f'class_{i}_support'] = int(support[i])
                    
                    # Calculate specificity
                    tn = cm_fallback.sum() - cm_fallback[i, :].sum() - cm_fallback[:, i].sum() + cm_fallback[i, i]
                    fp = cm_fallback[:, i].sum() - cm_fallback[i, i]
                    
                    if (tn + fp) > 0:
                        metrics[f'class_{i}_specificity'] = float(tn / (tn + fp))
                    else:
                        metrics[f'class_{i}_specificity'] = 0.0
            else:
                # No valid data - set all to 0
                for i in range(len(class_names)):
                    metrics[f'class_{i}_precision'] = 0.0
                    metrics[f'class_{i}_recall'] = 0.0
                    metrics[f'class_{i}_f1'] = 0.0
                    metrics[f'class_{i}_support'] = 0
                    metrics[f'class_{i}_specificity'] = 0.0
        else:
            # No predictions available - set all to 0
            for i in range(len(class_names)):
                metrics[f'class_{i}_precision'] = 0.0
                metrics[f'class_{i}_recall'] = 0.0
                metrics[f'class_{i}_f1'] = 0.0
                metrics[f'class_{i}_support'] = 0
                metrics[f'class_{i}_specificity'] = 0.0
    
    # ------------------------------------------------------------------
    # RECALCULATE macro_f1 if it's still 0 and we have per-class F1
    # ------------------------------------------------------------------
    if metrics['macro_f1'] == 0.0:
        class_f1_values = [metrics.get(f'class_{i}_f1', 0.0) for i in range(len(class_names))]
        if any(f1 > 0 for f1 in class_f1_values):
            metrics['macro_f1'] = float(np.mean(class_f1_values))
    
    # ------------------------------------------------------------------
    # METADATA EXTRACTION (for Model Configuration & Training Details section)
    # ------------------------------------------------------------------
    # Extract from config if available in analysis_results
    metadata = analysis_results.get('metadata', {})
    
    # Auto-detect versions if not in metadata
    try:
        pytorch_version = torch.__version__
    except:
        pytorch_version = 'N/A'
    
    try:
        transformers_version = transformers.__version__
    except:
        transformers_version = 'N/A'
    
    # Model configuration (with defaults from standard config)
    metrics['model_name'] = metadata.get('model_name', 'roberta-base')
    metrics['max_length'] = metadata.get('max_length', 512)
    metrics['dropout'] = metadata.get('dropout', 0.1)
    metrics['freeze_layers'] = metadata.get('freeze_layers', 6)
    
    # Training configuration
    metrics['batch_size'] = metadata.get('batch_size', 16)
    metrics['epochs'] = metadata.get('epochs', 3)
    metrics['learning_rate'] = metadata.get('learning_rate', 2e-5)
    metrics['warmup_steps'] = metadata.get('warmup_steps', 100)
    metrics['weight_decay'] = metadata.get('weight_decay', 0.01)
    metrics['gradient_clip'] = metadata.get('gradient_clip', 1.0)
    
    # Dataset information
    metrics['train_samples'] = metadata.get('train_samples', 'N/A')
    metrics['val_samples'] = metadata.get('val_samples', 'N/A')
    
    # Computational details
    metrics['device'] = metadata.get('device', 'cuda')
    metrics['training_time'] = metadata.get('training_time', 'N/A')
    metrics['hardware'] = metadata.get('hardware', 'CUDA GPU')
    metrics['random_seed'] = metadata.get('random_seed', 42)  # Default from config
    metrics['pytorch_version'] = metadata.get('pytorch_version', torch.__version__)
    metrics['transformers_version'] = metadata.get('transformers_version', transformers.__version__)
    
    
    # Extract CV metrics
    cv_data = analysis_results.get('cv_results', {})
    if cv_data:
        metrics['cv_overall_metrics'] = cv_data.get('overall', {})
        metrics['cv_significance'] = cv_data.get('significance', {})

    else:
        metadata = analysis_results.get('metadata', {})
        cv_from_meta = metadata.get('cv_results', {})
        if cv_from_meta:
            metrics['cv_overall_metrics'] = cv_from_meta.get('overall', {})
            metrics['cv_significance'] = cv_from_meta.get('significance', {})
    
    # CRITICAL FIX: Compute statistical significance from predictions if not in CV results
    if 'cv_significance' not in metrics or not metrics.get('cv_significance'):
        predictions_data = analysis_results.get('predictions', {})
        preds = predictions_data.get('preds', [])
        labels = predictions_data.get('labels', [])
        
        if len(preds) > 0 and len(labels) > 0:
            preds_arr = np.array(preds)
            labels_arr = np.array(labels)
            valid_mask = labels_arr != -1
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            if len(valid_labels) > 0:
                num_classes = len(np.unique(valid_labels))
                num_correct = (valid_preds == valid_labels).sum()
                num_total = len(valid_labels)
                accuracy = num_correct / num_total
                random_baseline = 1.0 / num_classes
                
                from scipy.stats import binomtest
                result = binomtest(num_correct, num_total, random_baseline, alternative='greater')
                p_value = result.pvalue
                is_significant = p_value < 0.05
                
                h = 2 * (np.arcsin(np.sqrt(accuracy)) - np.arcsin(np.sqrt(random_baseline)))
                
                if abs(h) < 0.2:
                    effect_interpretation = "Small"
                elif abs(h) < 0.5:
                    effect_interpretation = "Medium"
                else:
                    effect_interpretation = "Large"
                
                metrics['cv_significance'] = {
                    'vs_random': {
                        'significant': is_significant,
                        'p_value': float(p_value),
                        'test_statistic': 'Binomial Test',
                        'random_baseline': float(random_baseline),
                        'model_accuracy': float(accuracy),
                        'effect_size_h': float(h),
                        'effect_interpretation': effect_interpretation,
                        'num_correct': int(num_correct),
                        'num_total': int(num_total)
                    }
                }
    
    per_model = analysis_results.get('per_model', {})
    
    
    models_dict = per_model.get('models', {})
    
    if models_dict:
        # Store per-model data for report
        metrics['per_model_data'] = models_dict
        
        # Store summary stats if available
        summary_stats = per_model.get('summary', {})
        if summary_stats:
            metrics['per_model_summary'] = summary_stats
        
        # Store significance testing if available
        significance = per_model.get('significance', {})
        if significance:
            metrics['per_model_significance'] = significance
    
    
    predictions_data = analysis_results.get('predictions', {})
    if predictions_data:
        test_preds = predictions_data.get('preds', [])
        test_labels = predictions_data.get('labels', [])
        
        if len(test_preds) > 0 and len(test_labels) > 0:
            metrics['test_predictions'] = test_preds
            metrics['test_labels'] = test_labels
    
    
    return metrics


def build_jailbreak_metrics_from_results(analysis_results):
    """
    Centralized builder for jailbreak classifier report metrics.
    
    Similar to build_refusal_metrics_from_results but adapted for 2-class jailbreak detection.
    
    Args:
        analysis_results: Dictionary from ExperimentRunner.analyze_jailbreak_models()
    
    Returns:
        Dictionary with all metrics needed for report generation
    """
    
    metrics = {}
    class_names = ['Jailbreak Failed', 'Jailbreak Succeeded']
    
    # ------------------------------------------------------------------
    # Extract from PerModelAnalyzer structure
    # ------------------------------------------------------------------
    per_model = analysis_results.get('per_model', {})
    models_dict = per_model.get('models', {})
    summary_stats = per_model.get('summary', {})
    
    confidence = analysis_results.get('confidence', {})
    basic_conf = confidence.get('basic_metrics', {})
    calib_conf = confidence.get('calibration_metrics', {})
    
    # ------------------------------------------------------------------
    # OVERALL METRICS
    # ------------------------------------------------------------------
    # CRITICAL FIX: For jailbreak, use security_metrics accuracy (correct on all 78 samples)
    # NOT confidence analyzer accuracy (wrong - only on filtered subset)
    security_metrics = analysis_results.get('security_metrics', {})
    
    # Use security_metrics as primary source for jailbreak (it's correct!)
    metrics['accuracy'] = float(security_metrics.get('accuracy', basic_conf.get('accuracy', 0.0)))
    metrics['macro_f1'] = float(security_metrics.get('f1_macro', basic_conf.get('f1_macro', 0.0)))
    metrics['weighted_f1'] = float(security_metrics.get('f1_weighted', basic_conf.get('weighted_f1', 0.0)))
    metrics['cohen_kappa'] = float(security_metrics.get('cohen_kappa', basic_conf.get('cohen_kappa', 0.0)))
    
    
    # Store complete security_metrics for jailbreak-specific insights
    if security_metrics:
        metrics['security_metrics'] = {
            'false_negative_rate': float(security_metrics.get('false_negative_rate', 0.0)),
            'true_negative_rate': float(security_metrics.get('true_negative_rate', 0.0)),
            'false_positive_rate': float(security_metrics.get('false_positive_rate', 0.0)),
            'recall_succeeded': float(security_metrics.get('recall_succeeded', 0.0)),
            'precision_succeeded': float(security_metrics.get('precision_succeeded', 0.0))
        }
    
    
    # Matthews corrcoef from confidence analyzer (will calculate from predictions later if needed)
    metrics['matthews_corrcoef'] = float(basic_conf.get('matthews_corrcoef', 0.0))
    
    # For macro precision/recall, we need to calculate from confusion matrix
    # since security_metrics only has per-class (succeeded) values
    if 'confusion_matrix' in security_metrics:
        cm = np.array(security_metrics['confusion_matrix'])  # [[TN, FP], [FN, TP]]
        tn, fp = cm[0]
        fn, tp = cm[1]
        
        # Class 0 (Failed): precision = TN / (TN + FN), recall = TN / (TN + FP)
        prec_failed = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_failed = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Class 1 (Succeeded): from security_metrics
        prec_succeeded = float(security_metrics.get('precision_succeeded', 0.0))
        rec_succeeded = float(security_metrics.get('recall_succeeded', 0.0))
        
        # Macro averages
        metrics['macro_precision'] = float((prec_failed + prec_succeeded) / 2)
        metrics['macro_recall'] = float((rec_failed + rec_succeeded) / 2)
    else:
        # Fallback to confidence analyzer (may be wrong but better than nothing)
        metrics['macro_precision'] = float(basic_conf.get('macro_precision', 0.0))
        metrics['macro_recall'] = float(basic_conf.get('macro_recall', 0.0))
    
    # If overall metrics are missing but we have predictions, calculate them
    # Note: We'll extract predictions properly below in the PER-CLASS METRICS section
    # and use those for any needed calculations here
    
    # ------------------------------------------------------------------
    # PER-CLASS METRICS (2 classes: Failed, Succeeded)
    # ------------------------------------------------------------------
    per_class_metrics_direct = analysis_results.get('per_class_metrics', {})
    predictions_data = analysis_results.get('predictions', {})
    
    # Extract and convert to numpy arrays (handle both list and array inputs)
    preds_raw = predictions_data.get('preds', [])
    labels_raw = predictions_data.get('labels', [])
    confidences_raw = predictions_data.get('confidences', [])
    
    # Convert to numpy arrays if they're lists
    preds = np.array(preds_raw) if isinstance(preds_raw, list) else preds_raw
    labels = np.array(labels_raw) if isinstance(labels_raw, list) else labels_raw
    confidences = np.array(confidences_raw) if isinstance(confidences_raw, list) else confidences_raw
    
    # Ensure they are numpy arrays (handle edge cases)
    if not isinstance(preds, np.ndarray):
        preds = np.array([])
    if not isinstance(labels, np.ndarray):
        labels = np.array([])
    if not isinstance(confidences, np.ndarray):
        confidences = np.array([])
    
    
    # ------------------------------------------------------------------
    # CALCULATE CONFUSION MATRIX FOR SPECIFICITY
    # ------------------------------------------------------------------
    confusion_matrix_for_spec = None
    if len(preds) > 0 and len(labels) > 0:
        # Calculate confusion matrix for specificity
        # Labels parameter ensures all classes are included even if missing in predictions
        confusion_matrix_for_spec = confusion_matrix(
            labels,
            preds,
            labels=list(range(len(class_names)))
        )
    
    
    # ------------------------------------------------------------------
    # CALCULATE OVERALL METRICS if missing (now that we have proper arrays)
    # ------------------------------------------------------------------
    if len(preds) > 0 and len(labels) > 0 and metrics['macro_f1'] == 0.0:
        # Calculate overall metrics from predictions using sklearn
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        metrics['macro_f1'] = float(f1_macro)
        metrics['weighted_f1'] = float(f1_weighted)
        metrics['macro_precision'] = float(precision_macro)
        metrics['macro_recall'] = float(recall_macro)
        metrics['cohen_kappa'] = float(cohen_kappa_score(labels, preds))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(labels, preds))
        
        if metrics['accuracy'] == 0.0:
            metrics['accuracy'] = float(np.mean(preds == labels))
    
    # Calculate Matthews if it's 0 but we have predictions (not calculated above)
    if metrics['matthews_corrcoef'] == 0.0 and len(preds) > 0 and len(labels) > 0:
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(labels, preds))
    

    # ------------------------------------------------------------------
    # EXTRACT PER-CLASS METRICS (WITH SPECIFICITY CALCULATION)
    # ------------------------------------------------------------------
    if per_class_metrics_direct:
        for i, class_name in enumerate(class_names):
            class_key = f'class_{i}'
            class_data = per_class_metrics_direct.get(class_key, {})
            
            metrics[f'class_{i}_precision'] = float(class_data.get('precision', 0.0))
            metrics[f'class_{i}_recall'] = float(class_data.get('recall', 0.0))
            metrics[f'class_{i}_f1'] = float(class_data.get('f1-score', 0.0))
            metrics[f'class_{i}_support'] = int(class_data.get('support', 0))
            
            per_class_confidence = confidence.get('per_class_metrics', {})
            class_conf_data = per_class_confidence.get(class_name, {})
            
            metrics[f'class_{i}_mean_confidence'] = float(class_conf_data.get('mean_confidence', 0.0))
            metrics[f'class_{i}_std_confidence'] = float(class_conf_data.get('std_confidence', 0.0))
            metrics[f'class_{i}_min_confidence'] = float(class_conf_data.get('min_confidence', 0.0))
            metrics[f'class_{i}_max_confidence'] = float(class_conf_data.get('max_confidence', 0.0))
            metrics[f'class_{i}_class_accuracy'] = float(class_conf_data.get('accuracy', 0.0))
            
            # Calculate specificity from confusion matrix
            # Specificity for class i = TN_i / (TN_i + FP_i)
            if confusion_matrix_for_spec is not None:
                # True Negatives: sum of all cells except row i and column i
                tn = confusion_matrix_for_spec.sum() - confusion_matrix_for_spec[i, :].sum() - confusion_matrix_for_spec[:, i].sum() + confusion_matrix_for_spec[i, i]
                
                # False Positives: sum of column i except diagonal (predicted as i but wasn't)
                fp = confusion_matrix_for_spec[:, i].sum() - confusion_matrix_for_spec[i, i]
                
                # Specificity = TN / (TN + FP)
                if (tn + fp) > 0:
                    metrics[f'class_{i}_specificity'] = float(tn / (tn + fp))
                else:
                    metrics[f'class_{i}_specificity'] = 0.0
            else:
                metrics[f'class_{i}_specificity'] = 0.0
    
    elif len(preds) > 0 and len(labels) > 0:
        # CALCULATE from predictions - classification_report already imported
        report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            class_report = report.get(class_name, {})
            metrics[f'class_{i}_precision'] = float(class_report.get('precision', 0.0))
            metrics[f'class_{i}_recall'] = float(class_report.get('recall', 0.0))
            metrics[f'class_{i}_f1'] = float(class_report.get('f1-score', 0.0))
            metrics[f'class_{i}_support'] = int(class_report.get('support', 0))
        
        for i in range(len(class_names)):
            class_mask = (labels == i)
            class_confidences = confidences[class_mask]
            
            if len(class_confidences) > 0:
                metrics[f'class_{i}_mean_confidence'] = float(np.mean(class_confidences))
                metrics[f'class_{i}_std_confidence'] = float(np.std(class_confidences))
                metrics[f'class_{i}_min_confidence'] = float(np.min(class_confidences))
                metrics[f'class_{i}_max_confidence'] = float(np.max(class_confidences))
                class_preds = preds[class_mask]
                class_correct = np.sum(class_preds == labels[class_mask])
                metrics[f'class_{i}_class_accuracy'] = float(class_correct / len(class_confidences))
            else:
                metrics[f'class_{i}_mean_confidence'] = 0.0
                metrics[f'class_{i}_std_confidence'] = 0.0
                metrics[f'class_{i}_min_confidence'] = 0.0
                metrics[f'class_{i}_max_confidence'] = 0.0
                metrics[f'class_{i}_class_accuracy'] = 0.0
            
            # Calculate specificity from confusion matrix
            if confusion_matrix_for_spec is not None:
                tn = confusion_matrix_for_spec.sum() - confusion_matrix_for_spec[i, :].sum() - confusion_matrix_for_spec[:, i].sum() + confusion_matrix_for_spec[i, i]
                fp = confusion_matrix_for_spec[:, i].sum() - confusion_matrix_for_spec[i, i]
                
                if (tn + fp) > 0:
                    metrics[f'class_{i}_specificity'] = float(tn / (tn + fp))
                else:
                    metrics[f'class_{i}_specificity'] = 0.0
            else:
                metrics[f'class_{i}_specificity'] = 0.0
    
    else:
        # No predictions - all zeros
        for i in range(len(class_names)):
            metrics[f'class_{i}_precision'] = 0.0
            metrics[f'class_{i}_recall'] = 0.0
            metrics[f'class_{i}_f1'] = 0.0
            metrics[f'class_{i}_support'] = 0
            metrics[f'class_{i}_mean_confidence'] = 0.0
            metrics[f'class_{i}_std_confidence'] = 0.0
            metrics[f'class_{i}_min_confidence'] = 0.0
            metrics[f'class_{i}_max_confidence'] = 0.0
            metrics[f'class_{i}_class_accuracy'] = 0.0
            metrics[f'class_{i}_specificity'] = 0.0
    
    
    # ------------------------------------------------------------------
    # CONFIDENCE & CALIBRATION METRICS
    # ------------------------------------------------------------------
    metrics['mean_confidence'] = basic_conf.get('mean_confidence', 0.0)
    metrics['std_confidence'] = basic_conf.get('std_confidence', 0.0)
    metrics['calibration_error'] = calib_conf.get('ece', 0.0)
    metrics['brier_score'] = calib_conf.get('brier_score', 0.0)
    metrics['log_loss'] = calib_conf.get('log_loss', 0.0)
    
    # NEW: Additional confidence metrics
    metrics['mce'] = calib_conf.get('mce', 0.0)
    metrics['confidence_gap'] = basic_conf.get('confidence_gap', 0.0)
    metrics['mean_confidence_correct'] = basic_conf.get('mean_confidence_correct', 0.0)
    metrics['mean_confidence_incorrect'] = basic_conf.get('mean_confidence_incorrect', 0.0)
    
    # ------------------------------------------------------------------
    # METADATA EXTRACTION
    # ------------------------------------------------------------------
    metadata = analysis_results.get('metadata', {})
    
    # Auto-detect versions if not in metadata
    try:
        pytorch_version = torch.__version__
    except:
        pytorch_version = 'N/A'
    
    try:
        transformers_version = transformers.__version__
    except:
        transformers_version = 'N/A'
    
    metrics['model_name'] = metadata.get('model_name', 'roberta-base')
    metrics['max_length'] = metadata.get('max_length', 512)
    metrics['dropout'] = metadata.get('dropout', 0.1)
    metrics['freeze_layers'] = metadata.get('freeze_layers', 6)
    
    metrics['batch_size'] = metadata.get('batch_size', 16)
    metrics['epochs'] = metadata.get('epochs', 3)
    metrics['learning_rate'] = metadata.get('learning_rate', 2e-5)
    metrics['warmup_steps'] = metadata.get('warmup_steps', 100)
    metrics['weight_decay'] = metadata.get('weight_decay', 0.01)
    metrics['gradient_clip'] = metadata.get('gradient_clip', 1.0)
    
    metrics['train_samples'] = metadata.get('train_samples', 'N/A')
    metrics['val_samples'] = metadata.get('val_samples', 'N/A')
    
    num_test_samples = metadata.get('num_test_samples', 0)
    metrics['test_samples'] = num_test_samples if num_test_samples > 0 else 'N/A'
    
    metrics['device'] = metadata.get('device', 'cuda')
    metrics['training_time'] = metadata.get('training_time', 'N/A')
    metrics['hardware'] = metadata.get('hardware', 'CUDA GPU')
    metrics['random_seed'] = metadata.get('random_seed', 42)  # Default from config
    metrics['pytorch_version'] = metadata.get('pytorch_version', torch.__version__)
    metrics['transformers_version'] = metadata.get('transformers_version', transformers.__version__)
    
    
    # ------------------------------------------------------------------
    # CROSS-VALIDATION METRICS (CRITICAL FOR RESEARCH CREDIBILITY!)
    # ------------------------------------------------------------------
    # Extract CV metrics - same structure as refusal classifier
    cv_data = analysis_results.get('cv_results', {})
    if cv_data:
        metrics['cv_overall_metrics'] = cv_data.get('overall', {})
        metrics['cv_significance'] = cv_data.get('significance', {})
    
    
    else:
        # Fallback: check metadata for CV results
        cv_from_meta = metadata.get('cv_results', {})
        if cv_from_meta:
            metrics['cv_overall_metrics'] = cv_from_meta.get('overall', {})
            metrics['cv_significance'] = cv_from_meta.get('significance', {})
    
    # CRITICAL FIX: Compute statistical significance from predictions if not in CV results
    if 'cv_significance' not in metrics or not metrics.get('cv_significance'):
        predictions_data = analysis_results.get('predictions', {})
        preds = predictions_data.get('preds', [])
        labels = predictions_data.get('labels', [])
        
        if len(preds) > 0 and len(labels) > 0:
            preds_arr = np.array(preds)
            labels_arr = np.array(labels)
            valid_mask = labels_arr != -1
            valid_preds = preds_arr[valid_mask]
            valid_labels = labels_arr[valid_mask]
            
            if len(valid_labels) > 0:
                num_classes = len(np.unique(valid_labels))
                num_correct = (valid_preds == valid_labels).sum()
                num_total = len(valid_labels)
                accuracy = num_correct / num_total
                random_baseline = 1.0 / num_classes
                
                result = binomtest(num_correct, num_total, random_baseline, alternative='greater')
                p_value = result.pvalue
                is_significant = p_value < 0.05
                
                h = 2 * (np.arcsin(np.sqrt(accuracy)) - np.arcsin(np.sqrt(random_baseline)))
                
                if abs(h) < 0.2:
                    effect_interpretation = "Small"
                elif abs(h) < 0.5:
                    effect_interpretation = "Medium"
                else:
                    effect_interpretation = "Large"
                
                metrics['cv_significance'] = {
                    'vs_random': {
                        'significant': is_significant,
                        'p_value': float(p_value),
                        'test_statistic': 'Binomial Test',
                        'random_baseline': float(random_baseline),
                        'model_accuracy': float(accuracy),
                        'effect_size_h': float(h),
                        'effect_interpretation': effect_interpretation,
                        'num_correct': int(num_correct),
                        'num_total': int(num_total)
                    }
                }
    
    # ------------------------------------------------------------------
    # PER-MODEL ANALYSIS (ESSENTIAL FOR MULTI-MODEL TESTING!)
    # ------------------------------------------------------------------
    # Extract per-model data - same structure as refusal classifier
    per_model = analysis_results.get('per_model', {})
    models_dict = per_model.get('models', {})
    
    models_dict = {k.lower(): v for k, v in models_dict.items()}
    
    if models_dict:
        # Store per-model data for report
        metrics['per_model_data'] = models_dict
        
        # Store summary stats if available
        summary_stats = per_model.get('summary', {})
        if summary_stats:
            metrics['per_model_summary'] = summary_stats
        
        # Store significance testing if available
        significance = per_model.get('significance', {})
        if significance:
            metrics['per_model_significance'] = significance
    
    # ------------------------------------------------------------------
    # TEST PREDICTIONS (FOR ADDITIONAL ANALYSIS IF NEEDED)
    # ------------------------------------------------------------------
    predictions_data = analysis_results.get('predictions', {})
    if predictions_data:
        test_preds = predictions_data.get('preds', [])
        test_labels = predictions_data.get('labels', [])
        
        if len(test_preds) > 0 and len(test_labels) > 0:
            metrics['test_predictions'] = test_preds
            metrics['test_labels'] = test_labels


    # ------------------------------------------------------------------
    # CORRELATION ANALYSIS (REFUSAL-JAILBREAK RELATIONSHIP)
    # ------------------------------------------------------------------
    # Extract correlation analysis if available
    correlation_data = analysis_results.get('correlation', {})
    if correlation_data:
        # Agreement metrics
        agreement = correlation_data.get('agreement', {})
        if agreement:
            metrics['correlation_agreement_rate'] = float(agreement.get('agreement_rate', 0.0))
            metrics['correlation_agreement_count'] = int(agreement.get('agreement_count', 0))
            metrics['correlation_disagreement_count'] = int(agreement.get('disagreement_count', 0))
            metrics['correlation_interpretation'] = agreement.get('interpretation', '')
            metrics['correlation_recommendation'] = agreement.get('recommendation', '')
        
        # Per-refusal-class breakdown
        per_class_breakdown = correlation_data.get('per_class_breakdown', {})
        if per_class_breakdown:
            metrics['correlation_per_class_breakdown'] = per_class_breakdown
        
        # Independence test results
        independence = correlation_data.get('independence', {})
        if independence:
            metrics['correlation_chi2_statistic'] = float(independence.get('chi2_statistic', 0.0))
            metrics['correlation_chi2_pvalue'] = float(independence.get('p_value', 0.0))
            metrics['correlation_cramers_v'] = float(independence.get('cramers_v', 0.0))
            metrics['correlation_is_independent'] = independence.get('is_independent', False)
    
    # ------------------------------------------------------------------
    # ADVERSARIAL ROBUSTNESS METRICS (PARAPHRASE TESTING)
    # ------------------------------------------------------------------
    # Extract adversarial testing results if available
    adversarial_data = analysis_results.get('adversarial', {})
    if adversarial_data:
        # Overall robustness metrics
        metrics['adversarial_original_f1'] = float(adversarial_data.get('original_f1', 0.0))
        metrics['adversarial_avg_paraphrased_f1'] = float(adversarial_data.get('avg_paraphrased_f1', 0.0))
        metrics['adversarial_f1_drop'] = float(adversarial_data.get('f1_drop', 0.0))
        metrics['adversarial_relative_drop_pct'] = float(adversarial_data.get('relative_drop_pct', 0.0))
        metrics['adversarial_samples_tested'] = int(adversarial_data.get('samples_tested', 0))
        
        # Per-dimension robustness
        paraphrased_f1 = adversarial_data.get('paraphrased_f1', {})
        if paraphrased_f1:
            for dimension, f1_value in paraphrased_f1.items():
                metrics[f'adversarial_{dimension}_f1'] = float(f1_value)
        
        # Quality statistics
        quality_stats = adversarial_data.get('quality_stats', {})
        if quality_stats:
            metrics['adversarial_successful_paraphrases'] = int(quality_stats.get('successful_paraphrases', 0))
            metrics['adversarial_avg_semantic_similarity'] = float(quality_stats.get('avg_semantic_similarity', 0.0))
            metrics['adversarial_quality_pass_rate'] = float(quality_stats.get('quality_pass_rate', 0.0))
        
        # Hypothesis test results
        hypothesis_tests = adversarial_data.get('hypothesis_tests', {})
        if hypothesis_tests:
            metrics['adversarial_hypothesis_tests'] = hypothesis_tests
    
    
    return metrics


def calculate_per_class_metrics_from_predictions(y_true: np.ndarray, 
                                                 y_pred: np.ndarray,
                                                 class_names: List[str]) -> Dict:
    """
    Calculate comprehensive per-class metrics including support.
    
    NEW (V10): Standalone function to calculate all per-class metrics
    including support, specificity, and balanced accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        class_names: List of class names
    
    Returns:
        Dictionary with per-class metrics
    """
    
    # Get classification report with zero_division handling
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Get confusion matrix for specificity calculation
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    per_class = {}
    
    for i, class_name in enumerate(class_names):
        class_report = report.get(class_name, {})
        
        # Basic metrics from classification_report
        per_class[class_name] = {
            'precision': class_report.get('precision', 0.0),
            'recall': class_report.get('recall', 0.0),
            'f1-score': class_report.get('f1-score', 0.0),
            'support': int(class_report.get('support', 0))
        }
        
        # Calculate specificity (True Negative Rate)
        # Specificity = TN / (TN + FP)
        if len(cm) > i:
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = safe_divide(tn, tn + fp, 0.0)
            per_class[class_name]['specificity'] = float(specificity)
        else:
            per_class[class_name]['specificity'] = 0.0
        
        # Calculate balanced accuracy for this class
        # Balanced Accuracy = (Sensitivity + Specificity) / 2
        sensitivity = per_class[class_name]['recall']  # Recall = Sensitivity
        specificity = per_class[class_name]['specificity']
        per_class[class_name]['balanced_accuracy'] = (sensitivity + specificity) / 2.0
    
    return per_class


class ReportGenerator:
    """
    Generate professional PDF reports for any N-class classifier.
    
    Creates comprehensive reports using ReportLab with professional styling.
    Supports multiple report types for different audiences and use cases.

    Supported report types:
    - Model Performance Report: Technical deep-dive with metrics and visualizations
    - Jailbreak Security Report: Security-focused analysis for adversarial robustness
    - Production Monitoring Report: Real-time performance tracking and drift detection
    - Interpretability Report: Feature importance, attention, SHAP, and power law analysis
    - Executive Summary: High-level 1-2 page overview for stakeholders
    
    Uses DPI from VISUALIZATION_CONFIG for consistent image quality.
    """
    
    # Comprehensive metric descriptions with range and interpretation
    METRIC_DESCRIPTIONS = {
        # Overall Performance Metrics
        'Accuracy': {'range': '[0, 1]', 'better': '↑', 'note': '1.0 = perfect'},
        'Macro F1': {'range': '[0, 1]', 'better': '↑', 'note': 'Unweighted average'},
        'Weighted F1': {'range': '[0, 1]', 'better': '↑', 'note': 'Accounts for imbalance'},
        'Macro Precision': {'range': '[0, 1]', 'better': '↑', 'note': 'Avg across classes'},
        'Macro Recall': {'range': '[0, 1]', 'better': '↑', 'note': 'Avg across classes'},
        "Cohen's Kappa": {'range': '[-1, 1]', 'better': '↑', 'note': '0 = random'},
        'Matthews Corrcoef': {'range': '[-1, 1]', 'better': '↑', 'note': '0 = random'},
        'Log Loss': {'range': '[0, ∞)', 'better': '↓', 'note': '0 = perfect'},
        
        # Confidence & Calibration
        'Mean Confidence': {'range': '[0, 1]', 'better': '~', 'note': 'Should match accuracy'},
        'Std. Confidence': {'range': '[0, 1]', 'better': '~', 'note': 'Variation in confidence'},
        'Calibration Error (ECE)': {'range': '[0, 1]', 'better': '↓', 'note': '<0.1 = good'},
        'MCE (Max Calibration Error)': {'range': '[0, 1]', 'better': '↓', 'note': 'Worst-case calibration'},
        'Brier Score': {'range': '[0, 1]', 'better': '↓', 'note': '0 = perfect'},
        'Confidence Gap': {'range': '[-1, 1]', 'better': '↑', 'note': 'Correct - Incorrect'},
        'Mean Confidence (Correct)': {'range': '[0, 1]', 'better': '↑', 'note': 'When model is right'},
        'Mean Confidence (Incorrect)': {'range': '[0, 1]', 'better': '↓', 'note': 'When model is wrong'},
        
        # Per-Class Metrics
        'Precision': {'range': '[0, 1]', 'better': '↑', 'note': 'TP/(TP+FP)'},
        'Recall': {'range': '[0, 1]', 'better': '↑', 'note': 'TP/(TP+FN)'},
        'F1 Score': {'range': '[0, 1]', 'better': '↑', 'note': 'Harmonic mean'},
        'Support': {'range': '[0, ∞)', 'better': '~', 'note': '# samples'},
        'Specificity': {'range': '[0, 1]', 'better': '↑', 'note': 'TN/(TN+FP)'},
        # PHASE 2: Per-class confidence metrics
        'Class Mean Confidence': {'range': '[0, 1]', 'better': '↑', 'note': 'Avg confidence'},
        'Class Confidence Std': {'range': '[0, 1]', 'better': '~', 'note': 'Variation'},
        'Class Min Confidence': {'range': '[0, 1]', 'better': '~', 'note': 'Lowest'},
        'Class Max Confidence': {'range': '[0, 1]', 'better': '~', 'note': 'Highest'},
        'Class-Level Accuracy': {'range': '[0, 1]', 'better': '↑', 'note': 'For this class'}
    }

    def __init__(self, class_names: List[str] = None):
        """
        Initialize the report generator.

        Args:
            class_names: List of class names (default: CLASS_NAMES from Setup)
        """
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom paragraph styles for reports."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#d62728'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))

        # Metric style (for key numbers)
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            fontName='Helvetica-Bold'
        ))

        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))
        
        # MetaInfo style (for run timestamps and metadata)
        self.styles.add(ParagraphStyle(
            name='MetaInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#1f77b4'),
            alignment=TA_CENTER,
            spaceAfter=4,
            fontName='Helvetica'
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['BodyText'],
            fontSize=10,
            leftIndent=20,
            rightIndent=10,  # Add right margin
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        # Recommendation styles
        self.styles.add(ParagraphStyle(
            name='RecommendationGood',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#2ca02c'),
            leftIndent=20,
            rightIndent=10,  # Add right margin
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='RecommendationConcern',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#d62728'),
            leftIndent=20,
            rightIndent=10,  # Add right margin
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))

    def _generate_key_insights(self, metrics: Dict) -> List:
        """
        Enhanced automated insights that scale from test runs to full experiments.
        
        Adapts recommendations based on:
        - Dataset size (test vs production)
        - Statistical power (small N = less confident conclusions)
        - Class imbalance severity
        - Production readiness
        
        Args:
            metrics: Dictionary of all metrics
        
        Returns:
            List of ReportLab elements for the insights section
        """
        elements = []
        
        # Section header
        elements.append(Paragraph("Key Insights & Recommendations", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Automated analysis of model behavior with actionable recommendations.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        # Detect dataset size for adaptive thresholds
        test_size = metrics.get('test_samples', 0)
        if isinstance(test_size, str):
            test_size = 0
        
        is_small_dataset = test_size < 200  # Small = test run
        is_medium_dataset = 200 <= test_size < 500
        is_large_dataset = test_size >= 500  # Large = full experiment
        
        # Dataset size context
        if is_small_dataset and test_size > 0:
            context = f"<b>Dataset Context:</b> Small test set (n={test_size}). Recommendations are preliminary—verify on full experiment."
            elements.append(Paragraph(context, self.styles['BulletPoint']))
            elements.append(Spacer(1, 6))
        
        # =========================================================================
        # INSIGHT 1: Which class is hardest to classify?
        # =========================================================================
        elements.append(Paragraph("1. Which class is hardest to classify?", self.styles['SubsectionHeading']))
        
        class_data = []
        for i, class_name in enumerate(self.class_names):
            mean_conf = metrics.get(f'class_{i}_mean_confidence', 0.0)
            f1 = metrics.get(f'class_{i}_f1', 0.0)
            support = metrics.get(f'class_{i}_support', 0)
            class_data.append((class_name, mean_conf, f1, support))
        
        # Sort by F1 score (more reliable than confidence)
        class_data.sort(key=lambda x: x[2])
        
        if class_data[0][1] > 0:
            hardest = class_data[0]
            easiest = class_data[-1]
            
            text = (f"<b>{hardest[0]}</b> (F1={hardest[2]:.3f}, conf={hardest[1]:.3f}, n={hardest[3]}) is hardest. "
                    f"<b>{easiest[0]}</b> (F1={easiest[2]:.3f}, conf={easiest[1]:.3f}, n={easiest[3]}) is easiest.")
            elements.append(Paragraph(text, self.styles['BulletPoint']))
            
            # Adaptive recommendations based on dataset size
            if hardest[2] < 0.3:  # Very low F1
                if is_small_dataset:
                    rec = f"⚠️ {hardest[0]} shows poor performance but sample size is small (n={hardest[3]}). Verify on full dataset before major changes."
                else:
                    rec = f"⚠️ CRITICAL: {hardest[0]} F1={hardest[2]:.3f} is unacceptable. Immediate action: (1) Collect 3-5x more training data, (2) Review labeling quality, (3) Check for label noise."
                elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
            elif hardest[2] < 0.6:  # Low F1
                if is_small_dataset:
                    rec = f"⚡ {hardest[0]} performance is low. Wait for full experiment results before optimization."
                else:
                    rec = f"⚠️ {hardest[0]} needs improvement. Collect 2x more training samples and apply class-specific data augmentation."
                elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
            else:
                rec = f"✓ All classes achieve F1 > 0.6. Model generalizes well."
                elements.append(Paragraph(rec, self.styles['RecommendationGood']))
        
        elements.append(Spacer(1, 12))
        
        # =========================================================================
        # INSIGHT 2: Class imbalance severity
        # =========================================================================
        elements.append(Paragraph("2. Is class imbalance problematic?", self.styles['SubsectionHeading']))
        
        supports = [metrics.get(f'class_{i}_support', 0) for i in range(self.num_classes)]
        total_samples = sum(supports)
        
        if total_samples > 0:
            max_support = max(supports)
            min_support = min([s for s in supports if s > 0])  # Exclude zero-sample classes
            imbalance_ratio = safe_divide(max_support, min_support, 1.0)
            
            # Show class distribution
            dist_text = "Distribution: " + ", ".join([f"{self.class_names[i]}={supports[i]}" for i in range(self.num_classes)])
            elements.append(Paragraph(dist_text, self.styles['BulletPoint']))
            
            # Severity thresholds scale with dataset size
            if is_large_dataset:
                severe_threshold = 10.0  # Stricter for large datasets
                moderate_threshold = 5.0
            else:
                severe_threshold = 15.0  # More lenient for small datasets
                moderate_threshold = 7.0
            
            if imbalance_ratio > severe_threshold:
                rec = f"⚠️ SEVERE IMBALANCE: {imbalance_ratio:.1f}:1 ratio. Use class weights (already enabled) + oversampling minority classes."
                elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
            elif imbalance_ratio > moderate_threshold:
                rec = f"⚡ MODERATE IMBALANCE: {imbalance_ratio:.1f}:1 ratio. Class weights help but consider collecting more minority samples."
                elements.append(Paragraph(rec, self.styles['BulletPoint']))
            else:
                rec = f"✓ BALANCED: {imbalance_ratio:.1f}:1 ratio. Class distribution is reasonable."
                elements.append(Paragraph(rec, self.styles['RecommendationGood']))
        
        elements.append(Spacer(1, 12))
        
        # =========================================================================
        # INSIGHT 3: Statistical power check
        # =========================================================================
        elements.append(Paragraph("3. Is there sufficient data for reliable metrics?", self.styles['SubsectionHeading']))
        
        if total_samples > 0:
            min_samples_per_class = min([s for s in supports if s > 0])
            
            # Statistical power thresholds
            if is_large_dataset:
                text = f"✓ FULL EXPERIMENT: {total_samples} test samples provides high statistical power. Results are reliable."
                elements.append(Paragraph(text, self.styles['RecommendationGood']))
            elif is_medium_dataset:
                text = f"⚡ MEDIUM DATASET: {total_samples} test samples provides moderate power. Results are indicative but not definitive."
                elements.append(Paragraph(text, self.styles['BulletPoint']))
            else:
                text = f"⚠️ SMALL DATASET: {total_samples} test samples provides LOW statistical power. Metrics may not generalize—run full experiment for confident conclusions."
                elements.append(Paragraph(text, self.styles['RecommendationConcern']))
            
            # Check minimum class size
            if min_samples_per_class < 10:
                rec = f"⚠️ CRITICAL: Smallest class has only {min_samples_per_class} samples. Need minimum 30 samples per class for reliable per-class metrics."
                elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
            elif min_samples_per_class < 30:
                rec = f"⚡ WARNING: Smallest class has {min_samples_per_class} samples. Aim for 50+ per class for robust evaluation."
                elements.append(Paragraph(rec, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 12))
        
        # =========================================================================
        # INSIGHT 4: Calibration quality
        # =========================================================================
        elements.append(Paragraph("4. Are confidence scores trustworthy?", self.styles['SubsectionHeading']))
        
        ece = metrics.get('calibration_error', 0.0)
        mce = metrics.get('mce', 0.0)
        gap = metrics.get('confidence_gap', 0.0)
        
        text = f"ECE={ece:.3f}, MCE={mce:.3f}, Confidence Gap={gap:.3f}."
        elements.append(Paragraph(text, self.styles['BulletPoint']))
        
        # Calibration assessment (stricter for large datasets)
        excellent_threshold = 0.05 if is_large_dataset else 0.08
        good_threshold = 0.10 if is_large_dataset else 0.15
        
        if ece < excellent_threshold:
            rec = f"✓ EXCELLENT CALIBRATION: ECE={ece:.3f}. Confidence scores accurately reflect true accuracy. Safe for production."
            elements.append(Paragraph(rec, self.styles['RecommendationGood']))
        elif ece < good_threshold:
            rec = f"⚡ GOOD CALIBRATION: ECE={ece:.3f}. Acceptable but consider temperature scaling for production deployment."
            elements.append(Paragraph(rec, self.styles['BulletPoint']))
        else:
            rec = f"⚠️ POOR CALIBRATION: ECE={ece:.3f}. Confidence scores are unreliable. REQUIRED: Apply temperature scaling before deployment."
            elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
        
        
        # Confidence gap check (adjusted for high-accuracy models)
        accuracy = metrics.get('accuracy', 0.0)
        
        if gap < 0.05:
            if accuracy >= 0.90:
                rec = f"ℹ️ Confidence gap={gap:.3f} with accuracy={accuracy:.3f}. Small gap is expected when model makes very few errors."
                elements.append(Paragraph(rec, self.styles['BulletPoint']))
            else:
                rec = f"⚠️ CRITICAL: Confidence gap={gap:.3f}. Model cannot distinguish correct vs incorrect predictions. Review training process."
                elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
        elif gap < 0.15:
            if accuracy >= 0.85:
                rec = f"ℹ️ Moderate confidence gap ({gap:.3f}) is acceptable given high accuracy ({accuracy:.3f})."
                elements.append(Paragraph(rec, self.styles['BulletPoint']))
            else:
                rec = f"⚡ Model shows weak discrimination (gap={gap:.3f}). Consider adding uncertainty quantification."
                elements.append(Paragraph(rec, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 12))
        
        # =========================================================================
        # INSIGHT 5: Production readiness (only for large datasets)
        # =========================================================================
        if is_large_dataset:
            elements.append(Paragraph("5. Is the model production-ready?", self.styles['SubsectionHeading']))
            
            accuracy = metrics.get('accuracy', 0.0)
            macro_f1 = metrics.get('macro_f1', 0.0)
            
            criteria_met = 0
            total_criteria = 5
            
            # Criterion 1: Overall performance
            if accuracy > 0.8 and macro_f1 > 0.75:
                elements.append(Paragraph("✓ Criterion 1: Strong overall performance (Acc>0.8, F1>0.75)", self.styles['BulletPoint']))
                criteria_met += 1
            else:
                elements.append(Paragraph(f"✗ Criterion 1: Performance below production threshold (Acc={accuracy:.3f}, F1={macro_f1:.3f})", self.styles['BulletPoint']))
            
            # Criterion 2: All classes viable
            min_class_f1 = min([metrics.get(f'class_{i}_f1', 0.0) for i in range(self.num_classes)])
            if min_class_f1 > 0.6:
                elements.append(Paragraph(f"✓ Criterion 2: All classes viable (min F1={min_class_f1:.3f})", self.styles['BulletPoint']))
                criteria_met += 1
            else:
                elements.append(Paragraph(f"✗ Criterion 2: Weak class performance (min F1={min_class_f1:.3f})", self.styles['BulletPoint']))
            
            # Criterion 3: Calibration
            if ece < 0.15:
                elements.append(Paragraph(f"✓ Criterion 3: Acceptable calibration (ECE={ece:.3f})", self.styles['BulletPoint']))
                criteria_met += 1
            else:
                elements.append(Paragraph(f"✗ Criterion 3: Poor calibration (ECE={ece:.3f})", self.styles['BulletPoint']))
            
            # Criterion 4: Statistical significance
            # Check if we have CV results
            cv_sig = metrics.get('cv_significance', {})
            vs_random = cv_sig.get('vs_random', {})
            if vs_random.get('significant', False):
                elements.append(Paragraph("✓ Criterion 4: Significantly exceeds random baseline", self.styles['BulletPoint']))
                criteria_met += 1
            else:
                elements.append(Paragraph("✗ Criterion 4: Not statistically significant vs random", self.styles['BulletPoint']))
            
            # Criterion 5: Sufficient test data
            if test_size >= 300:
                elements.append(Paragraph(f"✓ Criterion 5: Sufficient test samples (n={test_size})", self.styles['BulletPoint']))
                criteria_met += 1
            else:
                elements.append(Paragraph(f"✗ Criterion 5: Insufficient test samples (n={test_size})", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 6))
            
            # Overall verdict
            if criteria_met == total_criteria:
                verdict = f"✓ PRODUCTION READY: All {total_criteria} criteria met. Model is suitable for deployment with standard monitoring."
                elements.append(Paragraph(verdict, self.styles['RecommendationGood']))
            elif criteria_met >= 4:
                verdict = f"⚡ NEAR PRODUCTION: {criteria_met}/{total_criteria} criteria met. Address remaining issues before deployment."
                elements.append(Paragraph(verdict, self.styles['BulletPoint']))
            else:
                verdict = f"⚠️ NOT READY: Only {criteria_met}/{total_criteria} criteria met. Significant improvements needed before production."
                elements.append(Paragraph(verdict, self.styles['RecommendationConcern']))
        
        
        # =========================================================================
        # INSIGHT 6: SECURITY-CRITICAL METRICS (JAILBREAK ONLY)
        # =========================================================================
        # Only add for binary jailbreak classifier (2 classes)
        if self.num_classes == 2:
            elements.append(Paragraph("6. Security-Critical Metrics", self.styles['SubsectionHeading']))
            
            # Extract security metrics (from JailbreakAnalysis)
            # Note: These should be in analysis_results['security_metrics']
            security_metrics = metrics.get('security_metrics', {})
            
            if security_metrics:
                fnr = security_metrics.get('false_negative_rate', 0.0)
                tnr = security_metrics.get('true_negative_rate', 0.0)
                recall_succeeded = security_metrics.get('recall_succeeded', 0.0)
                
                text = f"False Negative Rate: {fnr*100:.1f}%, True Negative Rate: {tnr*100:.1f}%, Recall (Jailbreak Succeeded): {recall_succeeded*100:.1f}%"
                elements.append(Paragraph(text, self.styles['BulletPoint']))
                
                # Security assessment
                if fnr < 0.05:
                    rec = f"✓ EXCELLENT SECURITY: FNR={fnr*100:.1f}%. Model catches 95%+ of jailbreak attempts."
                    elements.append(Paragraph(rec, self.styles['RecommendationGood']))
                elif fnr < 0.10:
                    rec = f"⚡ GOOD SECURITY: FNR={fnr*100:.1f}%. Acceptable but monitor false negatives closely."
                    elements.append(Paragraph(rec, self.styles['BulletPoint']))
                else:
                    rec = f"⚠️ SECURITY RISK: FNR={fnr*100:.1f}%. Too many jailbreaks are missed. CRITICAL: Increase recall on 'Jailbreak Succeeded' class through class weights or resampling."
                    elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
                
                # TNR assessment
                if tnr > 0.90:
                    rec = f"✓ Low false alarm rate (TNR={tnr*100:.1f}%). Safe responses correctly identified."
                    elements.append(Paragraph(rec, self.styles['RecommendationGood']))
                elif tnr < 0.70:
                    rec = f"⚠️ High false alarm rate. {(1-tnr)*100:.1f}% of safe responses flagged as jailbreaks."
                    elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
            
            elements.append(Spacer(1, 12))
            
            # =========================================================================
            # INSIGHT 7: CORRELATION WITH REFUSAL CLASSIFIER
            # =========================================================================
            correlation_agreement = metrics.get('correlation_agreement_rate', None)
            
            if correlation_agreement is not None:
                elements.append(Paragraph("7. Correlation with Refusal Classifier", self.styles['SubsectionHeading']))
                
                interpretation = metrics.get('correlation_interpretation', '')
                recommendation = metrics.get('correlation_recommendation', '')
                
                text = f"Agreement Rate: {correlation_agreement*100:.1f}%. {interpretation}"
                elements.append(Paragraph(text, self.styles['BulletPoint']))
                
                if correlation_agreement >= 0.85:
                    rec = f"✓ {recommendation}"
                    elements.append(Paragraph(rec, self.styles['RecommendationGood']))
                elif correlation_agreement >= 0.70:
                    rec = f"⚡ {recommendation}"
                    elements.append(Paragraph(rec, self.styles['BulletPoint']))
                else:
                    rec = f"✓ {recommendation}"
                    elements.append(Paragraph(rec, self.styles['RecommendationGood']))
                
                elements.append(Spacer(1, 12))
            
            # =========================================================================
            # INSIGHT 8: ADVERSARIAL ROBUSTNESS
            # =========================================================================
            adversarial_drop = metrics.get('adversarial_relative_drop_pct', None)
            
            if adversarial_drop is not None:
                elements.append(Paragraph("8. Adversarial Robustness", self.styles['SubsectionHeading']))
                
                original_f1 = metrics.get('adversarial_original_f1', 0.0)
                paraphrased_f1 = metrics.get('adversarial_avg_paraphrased_f1', 0.0)
                
                text = f"Original F1: {original_f1:.3f}, Paraphrased F1: {paraphrased_f1:.3f}, Drop: {adversarial_drop:.1f}%"
                elements.append(Paragraph(text, self.styles['BulletPoint']))
                
                # Robustness assessment
                if adversarial_drop < 5.0:
                    rec = f"✓ EXCELLENT ROBUSTNESS: <5% F1 drop under paraphrasing. Model is resilient to adversarial inputs."
                    elements.append(Paragraph(rec, self.styles['RecommendationGood']))
                elif adversarial_drop < 10.0:
                    rec = f"⚡ GOOD ROBUSTNESS: {adversarial_drop:.1f}% F1 drop is acceptable for production."
                    elements.append(Paragraph(rec, self.styles['BulletPoint']))
                elif adversarial_drop < 20.0:
                    rec = f"⚠️ MODERATE ROBUSTNESS: {adversarial_drop:.1f}% F1 drop. Consider adversarial training or data augmentation."
                    elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
                else:
                    rec = f"⚠️ POOR ROBUSTNESS: {adversarial_drop:.1f}% F1 drop. CRITICAL: Model fails under paraphrasing. Implement adversarial training immediately."
                    elements.append(Paragraph(rec, self.styles['RecommendationConcern']))
        
        
        elements.append(Spacer(1, 12))
        
        return elements

    
    def _generate_data_composition_section(self, metrics: Dict) -> List:
        """
        Generate data composition section showing real vs synthetic data breakdown.
        
        CRITICAL FOR TRANSPARENCY:
        - Shows WildJailbreak supplementation if used
        - Provides sample counts per source
        - Essential for reproducibility and research integrity
        
        Args:
            metrics: Dictionary containing per_model_data with sample counts
        
        Returns:
            List of ReportLab elements for data composition section
        """
        elements = []
        
        # Extract per-model data to calculate composition
        per_model_data = metrics.get('per_model_data', {})
        
        if not per_model_data:
            return elements
        
        # Calculate sample counts
        real_models = [k for k in API_CONFIG['response_models'].keys() if k != 'wildjailbreak']

        real_samples = 0
        synthetic_samples = 0
        
        model_counts = {}
        for model_key in real_models:
            if model_key in per_model_data:
                count = int(per_model_data[model_key].get('num_samples', 0))
                model_counts[model_key] = count
                real_samples += count
        
        if 'wildjailbreak' in per_model_data:
            synthetic_samples = int(per_model_data['wildjailbreak'].get('num_samples', 0))
        
        total_samples = real_samples + synthetic_samples
        
        # Only show section if WildJailbreak is present
        if synthetic_samples == 0 or self.num_classes == 3:
            return elements
        
        
        print(f"DEBUG: num_classes={self.num_classes}, synthetic_samples={synthetic_samples}")
        
        # Section header
        elements.append(Paragraph("Data Composition", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Breakdown of test set by data source. Synthetic data from WildJailbreak dataset "
            "is used to supplement real model responses and ensure sufficient jailbreak examples.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        # Summary paragraph
        synthetic_pct = (synthetic_samples / total_samples * 100) if total_samples > 0 else 0
        real_pct = (real_samples / total_samples * 100) if total_samples > 0 else 0
        
        summary_text = (
            f"<b>Total Samples:</b> {total_samples:,}<br/>"
            f"<b>Real Model Responses:</b> {real_samples:,} ({real_pct:.1f}%)<br/>"
            f"<b>Synthetic (WildJailbreak):</b> {synthetic_samples:,} ({synthetic_pct:.1f}%)"
        )
        elements.append(Paragraph(summary_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))
        
        # Detailed breakdown table
        data = [['Source', 'Type', 'Samples', 'Percentage']]
        
        # Model name mapping
        model_display_names = {k: get_model_display_name(k) for k in real_models}
        
        real_models = [k for k in API_CONFIG['response_models'].keys() if k != 'wildjailbreak']
        
        # Add real model rows
        for model_key in real_models:
            if model_key in model_counts:
                count = model_counts[model_key]
                pct = (count / total_samples * 100) if total_samples > 0 else 0
                data.append([
                    model_display_names.get(model_key, model_key),
                    'Real',
                    f"{count:,}",
                    f"{pct:.1f}%"
                ])
        
        # Add WildJailbreak row
        if synthetic_samples > 0:
            data.append([
                'WildJailbreak',
                'Synthetic',
                f"{synthetic_samples:,}",
                f"{synthetic_pct:.1f}%"
            ])
        
        # Add total row
        data.append([
            'TOTAL',
            '—',
            f"{total_samples:,}",
            '100.0%'
        ])
        
        # Create table
        table = Table(data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-2, -1), [colors.white, colors.lightgrey]),
            # Make total row bold
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e8e8e8')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        total_wj = metrics.get('total_wildjailbreak_samples', 0)
        if total_wj > 0:
            elements.append(Paragraph(
                f"<b>Total WildJailbreak Samples Used:</b> {total_wj:,} (across train/val/test splits)",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 6))
        
        
        # Transparency note
        note_text = (
            "<b>Note on Synthetic Data:</b> WildJailbreak samples are used exclusively for jailbreak class augmentation. "
            "WildJailbreak samples are pre-labeled successful jailbreak attempts "
            "from AllenAI's dataset of adversarial prompts that bypassed AI safety mechanisms. "
            "These samples supplement real model responses when insufficient positive examples exist, "
            "ensuring the jailbreak classifier has adequate training data for both classes. "
            "Performance metrics include both real and synthetic samples to provide comprehensive evaluation."
        )
        elements.append(Paragraph(note_text, self.styles['BulletPoint']))
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _generate_split_metrics_section(self, metrics: Dict) -> List:
        """
        Generate performance metrics split by real vs. real+synthetic data.
        
        CRITICAL FOR SCIENTIFIC RIGOR:
        - Shows performance on real data only (true generalization)
        - Shows performance with synthetic augmentation
        - Allows readers to assess impact of WildJailbreak supplementation
        
        Args:
            metrics: Dictionary containing per_model_data and overall metrics
        
        Returns:
            List of ReportLab elements for split metrics section
        """
        elements = []
        
        # Extract per-model data
        per_model_data = metrics.get('per_model_data', {})
        
        if not per_model_data or 'wildjailbreak' not in per_model_data or self.num_classes == 3:
            return elements  # Only show if WildJailbreak is present AND jailbreak report
        
        # Section header
        elements.append(Paragraph("Performance: Real vs. Real+Synthetic", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Comparison of classifier performance on real model responses only versus "
            "combined real and synthetic data. This shows the impact of WildJailbreak supplementation.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        # Calculate real-only metrics (weighted average across real models)
        real_models = [k for k in API_CONFIG['response_models'].keys() if k != 'wildjailbreak']
        
        real_accuracies = []
        real_f1s = []
        real_weights = []
        
        for model_key in real_models:
            if model_key in per_model_data:
                model_metrics = per_model_data[model_key]
                samples = int(model_metrics.get('num_samples', 0))
                if samples > 0:
                    real_weights.append(samples)
                    real_accuracies.append(model_metrics.get('accuracy', 0.0))
                    real_f1s.append(model_metrics.get('f1_macro', 0.0))
        
        # Calculate weighted averages for real-only
        if real_weights:
            total_weight = sum(real_weights)
            real_only_accuracy = sum(a * w for a, w in zip(real_accuracies, real_weights)) / total_weight
            real_only_f1 = sum(f * w for f, w in zip(real_f1s, real_weights)) / total_weight
        else:
            real_only_accuracy = 0.0
            real_only_f1 = 0.0
        
        # Get overall metrics (real + synthetic)
        overall_accuracy = metrics.get('accuracy', 0.0)
        overall_f1 = metrics.get('macro_f1', 0.0)
        
        # Calculate sample counts
        real_samples = sum(real_weights) if real_weights else 0
        synthetic_samples = int(per_model_data['wildjailbreak'].get('num_samples', 0))
        total_samples = real_samples + synthetic_samples
        
        # Create comparison table
        data = [
            ['Metric', 'Real Only', 'Real + Synthetic', 'Difference'],
            [
                'Test Samples',
                f"{real_samples:,}",
                f"{total_samples:,}",
                f"+{synthetic_samples:,}"
            ],
            [
                'Accuracy',
                f"{real_only_accuracy:.4f}",
                f"{overall_accuracy:.4f}",
                f"{(overall_accuracy - real_only_accuracy):+.4f}"
            ],
            [
                'Macro F1',
                f"{real_only_f1:.4f}",
                f"{overall_f1:.4f}",
                f"{(overall_f1 - real_only_f1):+.4f}"
            ]
        ]
        
        # Create table
        table = Table(data, colWidths=[1.8*inch, 1.5*inch, 1.5*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        # Interpretation
        f1_diff = overall_f1 - real_only_f1
        if abs(f1_diff) < 0.01:
            interpretation = "✓ Minimal impact from synthetic supplementation. Performance is consistent across real and augmented data."
            style = self.styles['RecommendationGood']
        elif f1_diff > 0:
            interpretation = f"⚡ Synthetic supplementation improved F1 by {f1_diff:.4f}. WildJailbreak data enhanced jailbreak class representation."
            style = self.styles['BulletPoint']
        else:
            interpretation = f"⚠️ Performance decreased by {abs(f1_diff):.4f} with synthetic data. Consider reviewing WildJailbreak sample quality."
            style = self.styles['RecommendationConcern']
        
        elements.append(Paragraph(interpretation, style))
        elements.append(Spacer(1, 12))
        
        return elements

    
    def _generate_statistical_significance_section(self, metrics: Dict) -> List:
        """
        Generate statistical significance testing section.
        
        Tests if model performance significantly exceeds random baseline.
        Uses binomial test for classification accuracy.
        """
        elements = []
        
        # Check if already computed in build_refusal_metrics_from_results
        cv_sig = metrics.get('cv_significance', {})
        vs_random = cv_sig.get('vs_random', {})
        
        if vs_random:
            # Use pre-computed values - NO RECALCULATION
            p_value = vs_random['p_value']
            random_baseline = vs_random['random_baseline']
            accuracy = vs_random['model_accuracy']
            h = vs_random['effect_size_h']
            effect_interpretation = vs_random['effect_interpretation']
            num_correct = vs_random['num_correct']
            num_total = vs_random['num_total']
            num_classes = self.num_classes
            
        else:
            # Fallback: compute from predictions (backward compatibility)
            test_preds = metrics.get('test_predictions', [])
            test_labels = metrics.get('test_labels', [])
            
            if len(test_preds) == 0 or len(test_labels) == 0:
                return elements
            
            preds = np.array(test_preds)
            labels = np.array(test_labels)
            
            valid_mask = labels != -1
            valid_preds = preds[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) == 0:
                return elements
            
            # Calculate metrics
            num_classes = self.num_classes
            num_correct = (valid_preds == valid_labels).sum()
            num_total = len(valid_labels)
            accuracy = num_correct / num_total
            random_baseline = 1.0 / num_classes
            
            # Binomial test
            result = binomtest(num_correct, num_total, random_baseline, alternative='greater')
            p_value = result.pvalue
            
            # Effect size (Cohen's h)
            h = 2 * (np.arcsin(np.sqrt(accuracy)) - np.arcsin(np.sqrt(random_baseline)))
            
            # Interpret effect size
            if abs(h) < 0.2:
                effect_interpretation = "Small"
            elif abs(h) < 0.5:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"
        
        # Section header (moved here so it's after the if/else)
        elements.append(Paragraph("Statistical Significance", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Hypothesis testing to verify model performance exceeds random baseline. "
            "Essential for demonstrating genuine learning.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        # Create results table
        data = [
            ['Metric', 'Value'],
            ['Test Set Size', f"{num_total} samples"],
            ['Number of Classes', str(num_classes)],
            ['Random Baseline', f"{random_baseline:.4f} ({random_baseline*100:.2f}%)"],
            ['Model Accuracy', f"{accuracy:.4f} ({accuracy*100:.2f}%)"],
            ['Correct Predictions', f"{num_correct} / {num_total}"],
            ['Improvement', f"{(accuracy - random_baseline):.4f} ({(accuracy - random_baseline)*100:.2f}% points)"],
            ['', ''],
            ['Test Statistic', 'Binomial Test'],
            ['P-value', f"{p_value:.6f}" if p_value >= 0.000001 else "< 0.000001"],
            ['Significance Level', 'α = 0.05'],
            ['Result', '✓ SIGNIFICANT' if p_value < 0.05 else '✗ NOT SIGNIFICANT'],
            ['', ''],
            ['Effect Size (Cohen\'s h)', f"{h:.4f}"],
            ['Effect Interpretation', effect_interpretation],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('BACKGROUND', (0, 11), (-1, 11), colors.HexColor('#90EE90') if p_value < 0.05 else colors.HexColor('#FFB6C1')),
            ('FONTNAME', (0, 11), (-1, 11), 'Helvetica-Bold'),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        # Interpretation
        elements.append(Paragraph("<b>Interpretation:</b>", self.styles['SubsectionHeading']))
        
        if p_value < 0.001:
            interp_text = (
                f"✓ <b>Highly Significant (p < 0.001):</b> The model's accuracy ({accuracy:.4f}) is "
                f"<b>significantly better</b> than random guessing ({random_baseline:.4f}). "
                f"There is <b>overwhelming evidence</b> that the model has learned meaningful patterns. "
                f"Effect size is {effect_interpretation.lower()} (h={h:.4f})."
            )
            elements.append(Paragraph(interp_text, self.styles['RecommendationGood']))
        elif p_value < 0.01:
            interp_text = (
                f"✓ <b>Very Significant (p < 0.01):</b> The model's accuracy ({accuracy:.4f}) is "
                f"<b>significantly better</b> than random guessing ({random_baseline:.4f}). "
                f"There is <b>strong evidence</b> of genuine learning. "
                f"Effect size is {effect_interpretation.lower()} (h={h:.4f})."
            )
            elements.append(Paragraph(interp_text, self.styles['RecommendationGood']))
        elif p_value < 0.05:
            interp_text = (
                f"✓ <b>Significant (p < 0.05):</b> The model's accuracy ({accuracy:.4f}) is "
                f"<b>significantly better</b> than random guessing ({random_baseline:.4f}). "
                f"There is <b>sufficient evidence</b> of learning. "
                f"Effect size is {effect_interpretation.lower()} (h={h:.4f})."
            )
            elements.append(Paragraph(interp_text, self.styles['RecommendationGood']))
        else:
            interp_text = (
                f"⚠️ <b>Not Significant (p = {p_value:.4f}):</b> The model's accuracy ({accuracy:.4f}) is "
                f"<b>not significantly better</b> than random guessing ({random_baseline:.4f}). "
                f"The model may not have learned meaningful patterns. "
                f"<b>Consider:</b> More training data, better features, or different architecture."
            )
            elements.append(Paragraph(interp_text, self.styles['RecommendationConcern']))
        
        elements.append(Spacer(1, 12))
        
        # Publication note
        pub_note = (
            "<b>For Publication:</b> "
            f"Report as: \"Model accuracy ({accuracy:.4f}) significantly exceeded random baseline "
            f"({random_baseline:.4f}, binomial test, p {'< 0.001' if p_value < 0.001 else f'= {p_value:.4f}'}, "
            f"Cohen's h = {h:.4f}).\""
        )
        elements.append(Paragraph(pub_note, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 12))
        
        return elements
            
    
    def _generate_per_model_section(self, metrics: Dict) -> List:
        """
        Generate per-model performance analysis section.
        
        Shows how classifier performs on responses from different LLMs:
        - Claude 
        - GPT
        - Gemini
        
        ESSENTIAL because:
        - Different models have different refusal patterns
        - Shows if classifier generalizes across model families
        - Identifies which models are hardest to classify
        
        Args:
            metrics: Dictionary containing 'per_model_data' with model-specific metrics
        
        Returns:
            List of ReportLab elements for per-model section
        """
        elements = []
        
        # Extract per-model data
        per_model_data = metrics.get('per_model_data', {})
        
        # If no per-model data available, skip this section
        if not per_model_data:
            return elements
        
        # Section header
        elements.append(Paragraph("Per-Model Analysis", self.styles['SectionHeading']))
        elements.append(Paragraph(
            f"Performance breakdown across the {len(API_CONFIG['response_models'])} tested LLMs. "

            "Shows how well the classifier generalizes to different model families.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        
        # Prepare table data
        data = [['Model', 'Accuracy', 'Macro F1', 'Precision', 'Recall', 'Samples']]
        
        # FIX: Iterate over actual keys in per_model_data instead of hardcoded config keys
        for model_key, model_metrics in per_model_data.items():
            # Use get_model_display_name for readable names
            display_name = get_model_display_name(model_key)
            accuracy = model_metrics.get('accuracy', 0.0)
            f1_macro = model_metrics.get('f1_macro', 0.0)
            precision = model_metrics.get('precision_macro', 0.0)
            recall = model_metrics.get('recall_macro', 0.0)
            samples = int(model_metrics.get('num_samples', 0))
            
            data.append([
                display_name,
                f"{accuracy:.4f}",
                f"{f1_macro:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                str(samples)
            ])

        
        # Create table
        table = Table(data, colWidths=[2.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 8))
        
        # Add explanatory note about WildJailbreak perfect scores (jailbreak report only)
        if self.num_classes == 2:
            # Add explanatory note about WildJailbreak perfect scores
            elements.append(Paragraph(
                "<i><b>Note on WildJailbreak Scores:</b> The WildJailbreak dataset supplements training data "
                "with successful jailbreak examples (from AllenAI's adversarial_harmful subset). Since these "
                "samples are exclusively successful jailbreaks, they all share the same ground-truth label. "
                "The perfect 1.0 scores reflect correct classification of a single-class subset, not "
                "exceptional discriminative performance. Real-world generalization is better assessed by "
                "Claude and GPT metrics, which include mixed refusal/compliance responses.</i>",
                self.styles['BodyText']
                ))
        
            elements.append(Spacer(1, 12))
        
        
        # Add class distribution table for jailbreak classifier (2 classes)
        if self.num_classes == 2:
            # FIX: Check actual keys in per_model_data, not hardcoded list
            has_class_dist = any(
                'class_distribution' in model_metrics
                for model_metrics in per_model_data.values()
            )
            
            if has_class_dist:
                elements.append(Paragraph(
                    "<b>Per-Model Class Distribution</b>",
                    self.styles['SubsectionHeading']
                ))
                elements.append(Paragraph(
                    "Shows how many samples of each class exist per model. "
                    "Low F1 scores are expected when a model has very few samples of one class.",
                    self.styles['BodyText']
                ))
                elements.append(Spacer(1, 8))
                
                analyzer_class_names = ['Jailbreak Failed', 'Jailbreak Succeeded']
                
                dist_data = [[
                    'Model', 
                    self.class_names[0] if len(self.class_names) > 0 else 'Class 0',
                    self.class_names[1] if len(self.class_names) > 1 else 'Class 1',
                    'Total'
                ]]
                
                # FIX: Iterate over actual keys in per_model_data
                for model_key, model_metrics in per_model_data.items():
                    class_dist = model_metrics.get('class_distribution', {})
                    
                    if class_dist:
                        # Use get_model_display_name for readable names
                        display_name = get_model_display_name(model_key)
                        class_0_count = class_dist.get(analyzer_class_names[0], 0)
                        class_1_count = class_dist.get(analyzer_class_names[1], 0)
                        total = class_0_count + class_1_count
                        
                        dist_data.append([
                            display_name,
                            str(class_0_count),
                            str(class_1_count),
                            str(total)
                        ])
                
                
                if len(dist_data) > 1:
                    dist_table = Table(dist_data, colWidths=[2.0*inch, 1.5*inch, 1.5*inch, 1.0*inch])
                    dist_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('TOPPADDING', (0, 1), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                    ]))
                    
                    elements.append(dist_table)
                    elements.append(Spacer(1, 8))
                    
                    elements.append(Paragraph(
                        "<i>Note: Claude and GPT have very few 'Jailbreak Succeeded' samples because "
                        "modern LLMs have strong safety guardrails. The low per-model Macro F1 (~0.5) "
                        "reflects this class imbalance, not poor classifier performance.</i>",
                        self.styles['BodyText']
                    ))
                    elements.append(Spacer(1, 12))
        
        # Add summary statistics if available
        summary_stats = metrics.get('per_model_summary', {})
        if summary_stats:
            # Find best and worst performing models
            f1_scores = {}

            all_models = list(API_CONFIG['response_models'].keys()) + ['wildjailbreak']
            
            real_models = [k for k in API_CONFIG['response_models'].keys() if k != 'wildjailbreak']
            
            # Model name mapping
            model_display_names = {k: get_model_display_name(k) for k in real_models}
            
            for model_key in all_models:
                if model_key in per_model_data:
                    f1_scores[model_key] = per_model_data[model_key].get('f1_macro', 0.0)
            
            if f1_scores:
                best_model = max(f1_scores, key=f1_scores.get)
                worst_model = min(f1_scores, key=f1_scores.get)
                best_f1 = f1_scores[best_model]
                worst_f1 = f1_scores[worst_model]
                
                best_display = model_display_names.get(best_model, best_model)
                worst_display = model_display_names.get(worst_model, worst_model)
                
                # Calculate variance across models
                f1_values = list(f1_scores.values())
                f1_std = np.std(f1_values)
                
                # Insights
                insights = []
                
                insights.append(f"<b>Best Performance:</b> {best_display} (F1={best_f1:.4f})")
                insights.append(f"<b>Worst Performance:</b> {worst_display} (F1={worst_f1:.4f})")
                insights.append(f"<b>F1 Std Dev:</b> {f1_std:.4f}")
                
                if f1_std < 0.05:
                    insights.append("✓ <b>Excellent generalization</b> - classifier performs consistently across all models.")
                elif f1_std < 0.10:
                    insights.append("✓ <b>Good generalization</b> - minor performance variations across models.")
                else:
                    insights.append("⚠️ <b>Variable performance</b> - classifier struggles with some model types. Consider model-specific fine-tuning.")
                
                for insight in insights:
                    elements.append(Paragraph(insight, self.styles['BulletPoint']))
                
                elements.append(Spacer(1, 6))
        
        # Statistical significance testing if available
        per_model_significance = metrics.get('per_model_significance', {})
        if per_model_significance:
            p_value = per_model_significance.get('anova_p_value', None)
            if p_value is not None:
                if p_value < 0.05:
                    sig_text = f"⚠️ <b>Significant performance differences detected</b> (ANOVA p={p_value:.4f}). Model-specific behavior exists."
                    elements.append(Paragraph(sig_text, self.styles['RecommendationConcern']))
                else:
                    sig_text = f"✓ <b>No significant differences</b> (ANOVA p={p_value:.4f}). Classifier generalizes well across models."
                    elements.append(Paragraph(sig_text, self.styles['RecommendationGood']))
        
        elements.append(Spacer(1, 12))
        
        return elements
        
    
    def _generate_cross_validation_section(self, metrics: Dict) -> List:
        """
        Generate cross-validation metrics section with mean ± std and 95% CI.
        
        CRITICAL FOR RESEARCH CREDIBILITY:
        - Shows model generalization via k-fold validation
        - Provides statistical rigor with confidence intervals
        - Essential for publication-quality reports
        
        Args:
            metrics: Dictionary containing 'cv_overall_metrics' with CV statistics
        
        Returns:
            List of ReportLab elements for CV section
        """
        elements = []
        
        # Extract cross-validation metrics
        cv_metrics = metrics.get('cv_overall_metrics', {})
        
        # If no CV metrics available, skip this section
        if not cv_metrics:
            return elements
        
        # Section header
        elements.append(Paragraph("Cross-Validation Metrics", self.styles['SectionHeading']))
        elements.append(Paragraph(
            f"K-fold cross-validation results demonstrating model generalization."
            f"Metrics are reported as mean ± standard deviation with 95% confidence intervals.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        # Prepare CV metrics table data
        # Header row
        data = [['Metric', 'Mean', 'Std Dev', '95% CI', 'Range', 'Better']]
        
        # Define metric order (most important first)
        metric_order = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'loss']
        
        # Add rows for each metric
        for metric_name in metric_order:
            if metric_name in cv_metrics:
                stats = cv_metrics[metric_name]
                
                # Extract values
                mean_val = stats.get('mean', 0.0)
                std_val = stats.get('std', 0.0)
                ci = stats.get('confidence_interval', (0.0, 0.0))
                min_val = stats.get('min', 0.0)
                max_val = stats.get('max', 0.0)
                
                # Format metric name
                if metric_name == 'f1_macro':
                    display_name = 'Macro F1'
                elif metric_name == 'f1_weighted':
                    display_name = 'Weighted F1'
                elif metric_name == 'precision_macro':
                    display_name = 'Macro Precision'
                elif metric_name == 'recall_macro':
                    display_name = 'Macro Recall'
                else:
                    display_name = metric_name.replace('_', ' ').title()
                
                # Format values
                mean_str = f"{mean_val:.4f}"
                std_str = f"{std_val:.4f}"
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                range_str = f"[{min_val:.4f}, {max_val:.4f}]"
                
                # Better direction
                better_direction = '↓' if metric_name == 'loss' else '↑'
                
                # Create row with Paragraph for text wrapping
                mean_para = Paragraph(mean_str, self.styles['Normal'])
                ci_para = Paragraph(ci_str, self.styles['Normal'])
                
                data.append([display_name, mean_para, std_str, ci_para, range_str, better_direction])
        
        # Create table with optimized column widths
        table = Table(data, colWidths=[1.6*inch, 0.8*inch, 0.8*inch, 1.4*inch, 1.2*inch, 0.6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),  
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),  # Center numerical columns
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        # =====================================================================
        # PER-FOLD F1 SCORES TABLE WITH BEST FOLD HIGHLIGHT
        # =====================================================================
        f1_macro_data = cv_metrics.get('f1_macro', {})
        fold_values = f1_macro_data.get('values', [])
        
        if fold_values and len(fold_values) > 0:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Per-Fold F1 Scores", self.styles['SubsectionHeading']))
            elements.append(Paragraph(
                "Individual F1 (macro) scores for each cross-validation fold. "
                "The best performing fold is highlighted in green.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 8))
            
            # Find best fold (handle both list and numpy array)
            fold_values_list = list(fold_values)  # Ensure it's a list
            best_fold_idx = fold_values_list.index(max(fold_values_list))
            best_fold_f1 = fold_values_list[best_fold_idx]
            
            # Build table data
            num_folds = len(fold_values_list)
            fold_header = ['Metric'] + [f'Fold {i+1}' for i in range(num_folds)] + ['Best']
            fold_scores = ['F1 (macro)'] + [f'{v:.4f}' for v in fold_values_list] + [f'Fold {best_fold_idx + 1}']
            
            fold_table_data = [fold_header, fold_scores]
            
            # Calculate column widths dynamically based on number of folds
            metric_col_width = 1.0 * inch
            fold_col_width = 0.75 * inch
            best_col_width = 0.8 * inch
            fold_table = Table(
                fold_table_data, 
                colWidths=[metric_col_width] + [fold_col_width] * num_folds + [best_col_width]
            )
            
            # Style with best fold highlighted
            style_commands = [
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                # First column (Metric label)
                ('BACKGROUND', (0, 1), (0, 1), colors.lightgrey),
                ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
                ('ALIGN', (0, 1), (0, 1), 'LEFT'),
                # Data cells
                ('BACKGROUND', (1, 1), (-2, -1), colors.white),
                # Highlight best fold column
                ('BACKGROUND', (best_fold_idx + 1, 0), (best_fold_idx + 1, -1), colors.HexColor('#2ecc71')),
                ('FONTNAME', (best_fold_idx + 1, 1), (best_fold_idx + 1, 1), 'Helvetica-Bold'),
                # Best column
                ('BACKGROUND', (-1, 1), (-1, -1), colors.HexColor('#d4edda')),
                ('FONTNAME', (-1, 1), (-1, -1), 'Helvetica-Bold'),
            ]
            
            fold_table.setStyle(TableStyle(style_commands))
            elements.append(fold_table)
            elements.append(Spacer(1, 8))
            
            # Best fold summary
            best_fold_text = (
                f"<b>Best Fold:</b> Fold {best_fold_idx + 1} achieved the highest F1 score of "
                f"<b>{best_fold_f1:.4f}</b>."
            )
            elements.append(Paragraph(best_fold_text, self.styles['RecommendationGood']))
        
        # Add interpretation note
        interpretation_text = (
            "<b>Interpretation:</b> "
            "Cross-validation provides a robust estimate of model generalization by training and evaluating "
            "on multiple data splits. Low standard deviation indicates stable performance across folds. "
            "Confidence intervals quantify uncertainty in the estimates (95% probability the true value lies within the interval)."
        )
        elements.append(Paragraph(interpretation_text, self.styles['BulletPoint']))
        elements.append(Spacer(1, 12))
        
        # Add statistical significance if available
        cv_significance = metrics.get('cv_significance', {})
        if cv_significance:
            vs_random = cv_significance.get('vs_random', {})
            if vs_random:
                p_value = vs_random.get('p_value', 1.0)
                is_significant = vs_random.get('significant', False)
                random_baseline = vs_random.get('random_baseline', 0.0)
                
                if is_significant:
                    sig_text = (
                        f"<b>Statistical Significance:</b> ✓ Performance significantly exceeds random baseline "
                        f"({random_baseline:.3f}) with p < {p_value:.4f}. Model demonstrates genuine learning."
                    )
                    elements.append(Paragraph(sig_text, self.styles['RecommendationGood']))
                else:
                    sig_text = (
                        f"<b>Statistical Significance:</b> ⚠️ Performance not significantly different from random baseline "
                        f"({random_baseline:.3f}) with p = {p_value:.4f}. Model may need improvement."
                    )
                    elements.append(Paragraph(sig_text, self.styles['RecommendationConcern']))
            
            # Add stability assessment
            stability = cv_significance.get('stability', {})
            if stability:
                cv_coef = stability.get('cv_coefficient', float('inf'))
                interpretation = stability.get('interpretation', 'Unknown')
                
                if cv_coef < float('inf'):
                    if interpretation == 'Stable':
                        stability_text = (
                            f"<b>Model Stability:</b> ✓ {interpretation} (CV coefficient = {cv_coef:.4f}). "
                            f"Consistent performance across folds."
                        )
                        elements.append(Paragraph(stability_text, self.styles['RecommendationGood']))
                    else:
                        stability_text = (
                            f"<b>Model Stability:</b> ⚠️ {interpretation} (CV coefficient = {cv_coef:.4f}). "
                            f"Performance varies across folds—consider more data or hyperparameter tuning."
                        )
                        elements.append(Paragraph(stability_text, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    
    def _create_simple_table(self, data_dict: Dict) -> List:
        """
        Create a simple 2-column table (Metric, Value) for metadata.
        No Range, Better, Note columns - just clean key-value pairs.
        
        Args:
            data_dict: Dictionary of metric_name: value pairs
        
        Returns:
            List of ReportLab elements
        """
        elements = []
        
        # Prepare data
        data = [['Metric', 'Value']]
        
        for key, value in data_dict.items():
            # Wrap long values in Paragraph for word wrapping
            value_para = Paragraph(str(value), self.styles['Normal'])
            data.append([key, value_para])
        
        # Create table with simple 2-column layout
        # Give most space to Value column
        table = Table(data, colWidths=[2.0*inch, 4.0*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('WORDWRAP', (0, 0), (-1, -1), True)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _generate_model_metadata(self, metrics: Dict) -> List:
        """
        Generate model metadata section with architecture and training details.
        
        Args:
            metrics: Dictionary containing metadata (if available)
        
        Returns:
            List of ReportLab elements for the metadata section
        """
        elements = []
        
        # Section header
        elements.append(Paragraph("Model Configuration & Training Details", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Complete model and training configuration for reproducibility.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        
        # Model Architecture
        elements.append(Paragraph("Model Architecture", self.styles['SubsectionHeading']))
        
        model_info = {
            'Base Model': metrics.get('model_name', 'roberta-base'),
            'Architecture': 'RoBERTa (Transformer)',
            'Number of Classes': str(self.num_classes),
            'Class Names': ', '.join(self.class_names),
            'Max Sequence Length': f"{metrics.get('max_length', 512)} tokens",
            'Dropout Rate': f"{metrics.get('dropout', 0.1):.2f}",
            'Frozen Layers': f"{metrics.get('freeze_layers', 6)} layers"
        }
        
        elements.extend(self._create_simple_table(model_info))
        
        # Training Configuration
        elements.append(Paragraph("Training Configuration", self.styles['SubsectionHeading']))
        
        training_info = {
            'Batch Size': str(metrics.get('batch_size', 16)),
            'Training Epochs': str(metrics.get('epochs', 3)),
            'Learning Rate': f"{metrics.get('learning_rate', 2e-5):.2e}",
            'Warmup Steps': str(metrics.get('warmup_steps', 100)),
            'Weight Decay': f"{metrics.get('weight_decay', 0.01):.3f}",
            'Gradient Clipping': f"{metrics.get('gradient_clip', 1.0):.1f}",
            'Optimizer': 'AdamW',
            'LR Scheduler': 'Linear with warmup'
        }
        
        elements.extend(self._create_simple_table(training_info))
        
        
        # Model Versions Table (shows short names → full version mapping)
        elements.append(Paragraph("Model Versions", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Models evaluated in this experiment with their full version identifiers.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 8))
        
        # Build model versions table from API_CONFIG
        version_data = [['Display Name', 'Full Model Version']]
        
        for short_key, full_version in API_CONFIG['response_models'].items():
            display_name = get_model_display_name(short_key)
            version_data.append([display_name, full_version])
        
        # Add WildJailbreak if present in data
        if metrics.get('per_model_data', {}).get('wildjailbreak') or \
           any('wildjailbreak' in str(k).lower() for k in metrics.get('per_model_data', {}).keys()):
            version_data.append([
                get_model_display_name('wildjailbreak'),
                f"{WILDJAILBREAK_CONFIG['dataset_name']} ({WILDJAILBREAK_CONFIG['data_type_filter']})"
            ])
        
        version_table = Table(version_data, colWidths=[2.0*inch, 4.0*inch])
        version_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), 'Courier'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        
        elements.append(version_table)
        elements.append(Spacer(1, 16))
    
        
        # Dataset Information
        elements.append(Paragraph("Dataset Information", self.styles['SubsectionHeading']))
        
        
        # Get split sizes
        total_samples = sum([metrics.get(f'class_{i}_support', 0) for i in range(self.num_classes)])
        
        dataset_info = {
            'Test Set Size': f"{total_samples} samples",
            'Class Distribution': ' | '.join([
                f"{self.class_names[i]}: {metrics.get(f'class_{i}_support', 0)}"
                for i in range(self.num_classes)
            ]),
            'Train Set Size': metrics.get('train_samples', 'N/A'),
            'Validation Set Size': metrics.get('val_samples', 'N/A')
        }
        
        elements.extend(self._create_simple_table(dataset_info))
        
        # Computational Details
        elements.append(Paragraph("Computational Details", self.styles['SubsectionHeading']))
        
        compute_info = {
            'Device': metrics.get('device', 'cuda'),
            'Training Time': metrics.get('training_time', 'N/A'),
            'Hardware': metrics.get('hardware', 'CUDA GPU'),
            'Random Seed': str(metrics.get('random_seed', 'Not specified')),
            'PyTorch Version': metrics.get('pytorch_version', 'N/A'),
            'Transformers Version': metrics.get('transformers_version', 'N/A')
        }
        
        elements.extend(self._create_simple_table(compute_info))
        
        return elements
    
    def _fig_to_image(self, fig, kind: str = None) -> Image:
        """
        Convert matplotlib figure to a high-quality ReportLab Image, using
        REPORT_CONFIG['images'] for sizes and DPI.

        Args:
            fig: Matplotlib figure
            kind: Optional key into REPORT_CONFIG['images']['figure_sizes']
                  (e.g., 'confusion_matrix', 'training_curves', 'class_distribution')

        Returns:
            ReportLab Image object with constrained dimensions
        """
        img_cfg = REPORT_CONFIG['images']

        # 1. Optionally set figure size from config
        figure_sizes = img_cfg.get('figure_sizes', {})
        if kind and kind in figure_sizes:
            w_in, h_in = figure_sizes[kind]
            fig.set_size_inches(w_in, h_in, forward=True)

        # 2. Render figure to an in-memory PNG buffer
        buf = BytesIO()
        fig.savefig(
            buf,
            format=img_cfg.get('format', 'png'),
            dpi=img_cfg.get('save_dpi', VISUALIZATION_CONFIG['dpi']),
            bbox_inches=img_cfg.get('bbox_inches', 'tight'),
            pad_inches=img_cfg.get('pad_inches', 0.1),
            facecolor=img_cfg.get('facecolor', 'white'),
            edgecolor=img_cfg.get('edgecolor', 'none'),
            transparent=img_cfg.get('transparent', False),
        )
        buf.seek(0)

        # 3. Compute target width/height in points from inches
        default_width_in = img_cfg.get('default_width_inches', 5.5)
        max_height_in = img_cfg.get('max_height_inches', 7.0)

        target_width = default_width_in * inch
        max_height = max_height_in * inch

        # 4. Create ReportLab Image and scale proportionally
        img = Image(buf)
        aspect = img.imageHeight / float(img.imageWidth)

        img.drawWidth = target_width
        img.drawHeight = target_width * aspect

        # 5. If height exceeds maximum, scale down to fit
        if img.drawHeight > max_height:
            img.drawHeight = max_height
            img.drawWidth = max_height / aspect

        return img

    def _create_header(self, title: str, subtitle: str = None, run_timestamp: str = None) -> List:
        """
        Create a professional header with title, subtitle, and timestamps.
        
        Args:
            title: Main report title
            subtitle: Optional subtitle
            run_timestamp: Timestamp of the run that generated this data (e.g., '20250114_1430')
        
        Returns:
            List of ReportLab elements for the header
        """
        elements = []
        
        # Title
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        
        # Subtitle
        if subtitle:
            elements.append(Paragraph(subtitle, self.styles['Normal']))
        
        # Run ID (Industry Best Practice: Show which run generated this data)
        if run_timestamp:
            run_id_text = f"<b>Run ID:</b> {run_timestamp}"
            elements.append(Paragraph(run_id_text, self.styles['MetaInfo']))
            
        # Generation timestamp (when this PDF was created)
        timestamp_str = get_timestamp('display')
        elements.append(Paragraph(
            f"<b>Report Generated:</b> {timestamp_str}",
            self.styles['Footer']
        ))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_metrics_table(self, metrics: Dict, columns: int = 2) -> List:
        """
        Create a formatted table of metrics with interpretability notes.
        
        ENHANCED: Now includes range, direction (↑/↓), and brief notes for each metric
        FIXED: Proper word wrapping to prevent text overflow
        
        Args:
            metrics: Dictionary of metric_name: value pairs
            columns: Number of columns (default: 2 for side-by-side display)
        
        Returns:
            List of ReportLab elements
        """
        elements = []
        
        # Prepare data with enhanced formatting
        data = [['Metric', 'Value', 'Range', 'Better', 'Note']]
        
        for key, value in metrics.items():
            # Don't modify key if it's already properly formatted (contains special chars)
            # Otherwise, format metric name (replace underscores with spaces, title case)
            if '_' in key:
                metric_name = key.replace('_', ' ').title()
            else:
                # Key is already formatted (e.g., "Cohen's Kappa", "Calibration Error (ECE)")
                metric_name = key
            
            # Format value
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, tuple) and len(value) == 2:
                # Confidence interval
                formatted_value = f"[{value[0]:.4f}, {value[1]:.4f}]"
            else:
                formatted_value = str(value)
            
            # Get metric description if available
            desc = self.METRIC_DESCRIPTIONS.get(metric_name, {})
            metric_range = desc.get('range', '-')
            better_direction = desc.get('better', '-')
            note = desc.get('note', '-')
            
            # Wrap long text in Paragraph for proper word wrapping
            metric_para = Paragraph(metric_name, self.styles['Normal'])
            note_para = Paragraph(note, self.styles['Normal'])
            
            data.append([metric_para, formatted_value, metric_range, better_direction, note_para])
        
        # Create table with optimized column widths
        # Total width ≈ 6 inches (fits in letter size with margins)
        table = Table(data, colWidths=[1.6*inch, 0.9*inch, 0.8*inch, 0.6*inch, 2.1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Top alignment for wrapped text
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),  # Center value column
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),  # Center range column
            ('ALIGN', (3, 0), (3, -1), 'CENTER'),  # Center better column
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('WORDWRAP', (0, 0), (-1, -1), True)  # Enable word wrapping
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 12))
        
        return elements

    def generate_model_performance_report(
        self,
        model_name: str,
        metrics: Dict,
        confusion_matrix_fig,
        training_curves_fig,
        class_distribution_fig,
        per_class_metrics_fig = None,
        output_path: str = None,
        run_timestamp: str = None
    ):
        """
        Generate comprehensive model performance report.
        
        ENHANCED (V10): Now properly displays all metrics including support.
        ENHANCED (V11): Now includes run timestamp in report content.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary from build_refusal_metrics_from_results()
            confusion_matrix_fig: Confusion matrix visualization
            training_curves_fig: Training curves visualization
            class_distribution_fig: Class distribution visualization
            per_class_metrics_fig: Optional per-class metrics visualization
            output_path: Path to save PDF report
            run_timestamp: Timestamp of the run (e.g., '20250114_1430')
        
        Returns:
            Path to the generated PDF report
        """
        if output_path is None:
            timestamp = run_timestamp or get_timestamp('file')
            output_path = os.path.join(reports_path, f"{model_name.lower().replace(' ', '_')}_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header (with run timestamp)
        elements.extend(self._create_header(
            f"{model_name} Performance Report",
            f"Comprehensive analysis of {self.num_classes}-class classification model",
            run_timestamp=run_timestamp
        ))

        # Executive Summary
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeading']))
        
        accuracy = metrics.get('accuracy', 0.0)
        macro_f1 = metrics.get('macro_f1', 0.0)
        
        summary_text = (
            f"This report presents a detailed performance analysis of the {model_name} classifier. "
            f"The model classifies text into {self.num_classes} categories: {', '.join(self.class_names)}. "
            f"Overall accuracy: {accuracy*100:.2f}%, Macro F1 Score: {macro_f1:.4f}."
        )
        elements.append(Paragraph(summary_text, self.styles['BodyText']))
        elements.append(Spacer(1, 12))

        # Key Insights & Recommendations (NEW! - Automated Analysis)
        elements.extend(self._generate_key_insights(metrics))
        elements.append(PageBreak())
        
        # Model Configuration & Training Details (NEW! - Metadata)
        elements.extend(self._generate_model_metadata(metrics))
        elements.append(PageBreak())
        
        # Data Composition (NEW! - Shows WildJailbreak usage)
        elements.extend(self._generate_data_composition_section(metrics))
        
        # Split Metrics (NEW! - Real vs Real+Synthetic)
        elements.extend(self._generate_split_metrics_section(metrics))


        # Overall Performance Metrics
        elements.append(Paragraph("Overall Performance Metrics", self.styles['SectionHeading']))
        
        overall_metrics = {
            'Accuracy': metrics.get('accuracy', 0.0),
            'Macro F1': metrics.get('macro_f1', 0.0),
            'Weighted F1': metrics.get('weighted_f1', 0.0),
            'Macro Precision': metrics.get('macro_precision', 0.0),
            'Macro Recall': metrics.get('macro_recall', 0.0),
            "Cohen's Kappa": metrics.get('cohen_kappa', 0.0),
            'Matthews Corrcoef': metrics.get('matthews_corrcoef', 0.0),  # NEW!
            'Log Loss': metrics.get('log_loss', 0.0)  # NEW!
        }
        
        
        elements.extend(self._create_metrics_table(overall_metrics))
        
        elements.extend(self._generate_cross_validation_section(metrics))
        
        # Per-Model Analysis (ESSENTIAL for multi-model testing!)
        elements.extend(self._generate_per_model_section(metrics))

        # Statistical Significance (ESSENTIAL for publication!)
        elements.extend(self._generate_statistical_significance_section(metrics))
        

        # Confidence & Calibration Metrics
        elements.append(Paragraph("Confidence & Calibration Metrics", self.styles['SectionHeading']))
        
        conf_metrics = {
            'Mean Confidence': metrics.get('mean_confidence', 0.0),
            'Std. Confidence': metrics.get('std_confidence', 0.0),
            'Calibration Error (ECE)': metrics.get('calibration_error', 0.0),
            'MCE (Max Calibration Error)': metrics.get('mce', 0.0),  # PHASE 1: NEW!
            'Brier Score': metrics.get('brier_score', 0.0),
            'Confidence Gap': metrics.get('confidence_gap', 0.0),  # PHASE 1: NEW!
            'Mean Confidence (Correct)': metrics.get('mean_confidence_correct', 0.0),  # PHASE 1: NEW!
            'Mean Confidence (Incorrect)': metrics.get('mean_confidence_incorrect', 0.0)  # PHASE 1: NEW!
        }
        
        elements.extend(self._create_metrics_table(conf_metrics))

        # Per-Class Performance (FIXED!)
        elements.append(PageBreak())
        elements.append(Paragraph("Per-Class Performance", self.styles['SectionHeading']))

        for i, class_name in enumerate(self.class_names):
            elements.append(Paragraph(f"Class: {class_name}", self.styles['SubsectionHeading']))
            
            class_metrics = {
                'Precision': metrics.get(f'class_{i}_precision', 0.0),
                'Recall': metrics.get(f'class_{i}_recall', 0.0),
                'F1 Score': metrics.get(f'class_{i}_f1', 0.0),
                'Support': int(metrics.get(f'class_{i}_support', 0)),
                'Specificity': metrics.get(f'class_{i}_specificity', 0.0),
                # PHASE 2: Per-class confidence metrics
                'Class Mean Confidence': metrics.get(f'class_{i}_mean_confidence', 0.0),
                'Class Confidence Std': metrics.get(f'class_{i}_std_confidence', 0.0),
                'Class Min Confidence': metrics.get(f'class_{i}_min_confidence', 0.0),
                'Class Max Confidence': metrics.get(f'class_{i}_max_confidence', 0.0),
                'Class-Level Accuracy': metrics.get(f'class_{i}_class_accuracy', 0.0)
            }
            
            elements.extend(self._create_metrics_table(class_metrics))

        # Confusion Matrix
        elements.append(PageBreak())
        elements.append(Paragraph("Confusion Matrix", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "The confusion matrix shows the model's predictions versus actual labels. "
            "Diagonal elements represent correct predictions.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(confusion_matrix_fig, kind='confusion_matrix'))

        # Training Curves
        elements.append(PageBreak())
        elements.append(Paragraph("Training Curves", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Training and validation metrics over epochs. "
            "Monitor for overfitting (validation diverging from training).",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(training_curves_fig, kind='training_curves'))

        # Class Distribution
        elements.append(PageBreak())
        elements.append(Paragraph("Class Distribution", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Distribution of samples across classes. "
            "Imbalanced datasets may require weighted loss or resampling.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(class_distribution_fig, kind='class_distribution'))

        # Per-Class Metrics Figure (if provided)
        if per_class_metrics_fig:
            elements.append(PageBreak())
            elements.append(Paragraph("Per-Class Metrics Visualization", self.styles['SectionHeading']))
            elements.append(Spacer(1, 12))
            elements.append(self._fig_to_image(per_class_metrics_fig, kind='per_class_metrics'))

        # Build PDF
        doc.build(elements)
        print(f"✓ Model Performance Report saved to: {output_path}")
        return output_path

    # REMOVED: generate_jailbreak_report() 
    # Now using generate_model_performance_report() for BOTH classifiers
    # This ensures consistent report structure and comprehensive metrics
    # The jailbreak detector report now includes:
    #   - Training curves, confusion matrix, class distribution
    #   - Per-class metrics (precision/recall/F1 for both classes)
    #   - Confidence analysis (calibration, ECE, MCE, Brier score)
    #   - Automated insights and recommendations
    # All driven by build_jailbreak_metrics_from_results() function (lines 464-655)

    def generate_interpretability_report(
        self,
        model_name: str,
        attention_figs: List,
        shap_figs: List = None,
        power_law_figs: List = None,
        output_path: str = None,
        run_timestamp: str = None
    ):
        """
        Generate interpretability report with attention, SHAP, and power law analysis.
        
        Args:
            model_name: Name of the model
            attention_figs: List of attention visualization figures
            shap_figs: Optional list of SHAP analysis figures
            power_law_figs: Optional list of power law analysis figures
            output_path: Path to save PDF report
            run_timestamp: Timestamp of the run (e.g., '20250114_1430')
        
        Returns:
            Path to the generated PDF report
        """
        if output_path is None:
            timestamp = run_timestamp or get_timestamp('file')
            output_path = os.path.join(reports_path, f"interpretability_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header (with run timestamp)
        elements.extend(self._create_header(
            f"{model_name} - Interpretability Report",
            "Model behavior analysis and feature importance",
            run_timestamp=run_timestamp
        ))

        # Introduction
        elements.append(Paragraph("Model Interpretability Analysis", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "This report provides insights into how the model makes decisions through "
            "attention analysis, SHAP values, and power law distributions.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))

        # Attention Analysis
        if attention_figs:
            elements.append(PageBreak())
            elements.append(Paragraph("Attention Analysis", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "Attention weights show which tokens the model focuses on when making predictions. "
                "Higher attention indicates greater importance for the classification decision.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

            for i, fig in enumerate(attention_figs):
                elements.append(Paragraph(f"Sample {i+1}", self.styles['SubsectionHeading']))
                elements.append(self._fig_to_image(fig))
                elements.append(Spacer(1, 12))

        # SHAP Analysis
        if shap_figs:
            elements.append(PageBreak())
            elements.append(Paragraph("SHAP Analysis", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "SHAP (SHapley Additive exPlanations) values show each token's contribution "
                "to the model's prediction. Red indicates positive contribution, blue indicates negative.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

            for i, fig in enumerate(shap_figs):
                elements.append(Paragraph(f"SHAP Analysis {i+1}", self.styles['SubsectionHeading']))
                elements.append(self._fig_to_image(fig))
                elements.append(Spacer(1, 12))

        # Power Law Analysis
        if power_law_figs:
            elements.append(PageBreak())
            elements.append(Paragraph("Power Law Analysis", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "Power law analysis reveals Pareto distributions in model behavior, "
                "showing if a small subset of features drives most predictions.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

            for i, fig in enumerate(power_law_figs):
                elements.append(self._fig_to_image(fig))
                elements.append(Spacer(1, 12))

        # Build PDF
        doc.build(elements)
        print(f"✓ Interpretability Report saved to: {output_path}")
        return output_path

    def generate_production_monitoring_report(
        self,
        predictions_df: pd.DataFrame,
        metrics_over_time_fig,
        confidence_distribution_fig,
        latency_distribution_fig,
        ab_test_comparison_fig = None,
        output_path: str = None,
        run_timestamp: str = None
    ):
        """
        Generate production monitoring report from logged predictions.
        
        Uses get_timestamp() from Utils for consistent filename generation.
        Uses safe_divide() from Utils for robust metric calculations.

        Args:
            predictions_df: DataFrame with prediction logs
            metrics_over_time_fig: Figure showing metrics over time
            confidence_distribution_fig: Figure showing confidence distributions
            latency_distribution_fig: Figure showing prediction latency
            ab_test_comparison_fig: Optional A/B test comparison figure
            output_path: Path to save PDF report
            run_timestamp: Timestamp of the run (e.g., '20250114_1430')
            
        Returns:
            Path to the generated PDF report
        """
        if output_path is None:
            timestamp = run_timestamp or get_timestamp('file')
            output_path = os.path.join(reports_path, f"production_monitoring_{timestamp}.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header (with run timestamp)
        elements.extend(self._create_header(
            "Production Monitoring Report",
            f"Analysis of {len(predictions_df)} predictions",
            run_timestamp=run_timestamp
        ))

        # Summary Statistics
        elements.append(Paragraph("Summary Statistics", self.styles['SectionHeading']))

        total_predictions = len(predictions_df)
        
        # Use safe defaults for empty dataframes
        avg_confidence = predictions_df['confidence'].mean() if len(predictions_df) > 0 else 0.0
        avg_latency = predictions_df['latency_ms'].mean() if len(predictions_df) > 0 else 0.0

        # Predictions per class
        class_counts = {}
        for i, class_name in enumerate(self.class_names):
            count = (predictions_df['prediction'] == i).sum()
            class_counts[f"{class_name} predictions"] = count

        summary_metrics = {
            'Total Predictions': total_predictions,
            'Average Confidence': f"{avg_confidence:.4f}",
            'Average Latency (ms)': f"{avg_latency:.2f}",
            **class_counts
        }

        elements.extend(self._create_metrics_table(summary_metrics))

        # Metrics Over Time
        elements.append(PageBreak())
        elements.append(Paragraph("Metrics Over Time", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Monitor for model drift and performance degradation over time.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(metrics_over_time_fig))

        # Confidence Distribution
        elements.append(PageBreak())
        elements.append(Paragraph("Confidence Distribution", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Distribution of prediction confidence scores. "
            "Low confidence predictions may require human review.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(confidence_distribution_fig))

        # Latency Analysis
        elements.append(PageBreak())
        elements.append(Paragraph("Latency Analysis", self.styles['SectionHeading']))
        elements.append(Paragraph(
            "Prediction latency distribution. Monitor for performance bottlenecks.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(latency_distribution_fig))

        # A/B Test Comparison (if available)
        if ab_test_comparison_fig is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("A/B Test Comparison", self.styles['SectionHeading']))
            elements.append(Paragraph(
                "Performance comparison between active and challenger models.",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))
            elements.append(self._fig_to_image(ab_test_comparison_fig))

        # Build PDF
        doc.build(elements)
        print(f"✓ Production Monitoring Report saved to: {output_path}")
        return output_path

    def generate_executive_summary(
        self,
        model_name: str,
        key_metrics: Dict,
        performance_chart_fig,
        recommendations: List[str],
        output_path: str,
        data_composition_stats: Dict = None
    ):
        """
        Generate 1-2 page executive summary for stakeholders.

        Args:
            model_name: Name of the model
            key_metrics: Dictionary of key performance indicators
            performance_chart_fig: Summary performance visualization
            recommendations: List of actionable recommendations
            output_path: Path to save PDF report
            data_composition_stats: Optional dict with WildJailbreak supplementation stats
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []

        # Header
        elements.extend(self._create_header(
            f"{model_name} - Executive Summary",
            "High-level performance overview and recommendations"
        ))

        # Data Composition (if applicable)
        if data_composition_stats and data_composition_stats.get('supplementation_used', False):
            elements.append(Paragraph("Data Composition", self.styles['SectionHeading']))
            elements.append(Paragraph(
                f"Training data includes {data_composition_stats.get('total_samples', 0):,} samples, "
                f"with {data_composition_stats.get('wildjailbreak_samples', 0):,} from WildJailbreak dataset "
                f"({data_composition_stats.get('wildjailbreak_percentage', 0):.1f}% supplementation).",
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 12))

        # Key Metrics
        elements.append(Paragraph("Key Performance Indicators", self.styles['SectionHeading']))
        elements.extend(self._create_metrics_table(key_metrics))

        # Performance Chart
        elements.append(Spacer(1, 12))
        elements.append(self._fig_to_image(performance_chart_fig))

        # Recommendations
        elements.append(PageBreak())
        elements.append(Paragraph("Recommendations", self.styles['SectionHeading']))

        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['BodyText']))
            elements.append(Spacer(1, 8))

        # Build PDF
        doc.build(elements)
        print(f"✓ Executive Summary saved to: {output_path}")
        return output_path


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: Report Generation Module
Created on October 31, 2025
FIXED: V10 - November 13, 2025
@author: ramyalsaffar
"""
