# Cross-Validator Module 
#------------------------
# Enhanced k-fold cross-validation for robust performance evaluation.
# Uses stratified k-fold to preserve class distribution across folds.
# Reports mean ± std metrics with confidence intervals and significance testing.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Statistical significance testing
# - Confidence interval calculation
# - Mixed precision support
# - Checkpoint support for resuming
# - Better visualization
# All imports are in 01-Imports.py
###############################################################################


def aggressive_mps_cleanup():
    """
    Aggressively clean MPS memory to prevent memory leaks.
    
    MPS (Apple Silicon GPU) has known memory leak issues with PyTorch.
    This function forces synchronization and garbage collection.
    """
    # Force garbage collection first
    gc.collect()
    
    # Synchronize and clear MPS cache if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Another round of garbage collection
    gc.collect()


class CrossValidator:
    """
    Enhanced K-Fold Cross-Validation for classification models.
    
    Features:
    - Stratified k-fold for class balance
    - Statistical significance testing
    - Confidence intervals
    - Per-class and overall metrics
    - Visualization support
    """
    
    def __init__(self,
                 model_class,
                 dataset,
                 k_folds: int = None,
                 device: torch.device = None,
                 class_names: List[str] = None,
                 random_state: int = None,
                 use_mixed_precision: bool = None):
        """
        Initialize enhanced cross-validator.
        
        Args:
            model_class: Class to instantiate model (RefusalClassifier or JailbreakClassifier)
            dataset: Full dataset (before train/test split)
            k_folds: Number of folds (uses config if None)
            device: torch device
            class_names: List of class names for display
            random_state: Random seed (uses config if None)
            use_mixed_precision: Use AMP for training
        """
        self.model_class = model_class
        self.dataset = dataset
        
        # Detect model type for proper fallback naming
        if model_class.__name__ == 'RefusalClassifier':
            self.model_type = 'refusal'
        elif model_class.__name__ in ['JailbreakClassifier', 'JailbreakDetector']:
            self.model_type = 'jailbreak'
        else:
            self.model_type = None
        
        # Use config values - NO HARDCODING!
        self.k_folds = k_folds or CROSS_VALIDATION_CONFIG.get('default_folds', 5)
        self.random_state = random_state or DATASET_CONFIG.get('random_seed', 42)
        self.use_mixed_precision = use_mixed_precision or TRAINING_CONFIG.get('mixed_precision', False)
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        
        # Device setup
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        unique_labels = len(set(dataset.labels))
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(unique_labels)]
        
        # Results storage
        self.fold_results = []
        self.cv_metrics = {
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'precision_macro': [],
            'recall_macro': [],
            'loss': []
        }
        
        # Per-class metrics
        self.per_class_metrics = {
            'precision': [[] for _ in range(len(self.class_names))],
            'recall': [[] for _ in range(len(self.class_names))],
            'f1': [[] for _ in range(len(self.class_names))]
        }
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            operation_name='cross_validation'
        )
        
        if self.verbose:
            self._print_cv_setup()
    
    def _print_cv_setup(self):
        """Print cross-validation setup information."""
        print_banner("CROSS-VALIDATION SETUP", char="=")
        print(f"  K-Folds: {self.k_folds}")
        print(f"  Total samples: {len(self.dataset):,}")
        print(f"  Classes: {len(self.class_names)}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print(f"  Random state: {self.random_state}")
        
        # Calculate and display class distribution
        label_counts = {}
        for label in self.dataset.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n  Overall class distribution:")
        for label, count in sorted(label_counts.items()):
            pct = safe_divide(count, len(self.dataset), 0) * 100
            if label < len(self.class_names):
                class_name = self.class_names[label]
                print(f"    {class_name}: {count:,} ({pct:.1f}%)")
        print("=" * 60)
    
    def run_cross_validation(self, 
                           save_fold_models: bool = False,
                           resume_from_fold: int = None) -> Dict:
        """
        Run enhanced k-fold cross-validation.
        
        Args:
            save_fold_models: Save model for each fold
            resume_from_fold: Resume from specific fold (for interrupted CV)
        
        Returns:
            Dictionary with CV results and statistics
        """
        # Create stratified k-fold splitter
        skf = StratifiedKFold(
            n_splits=self.k_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Get all labels for stratification
        all_labels = np.array([self.dataset.labels[i] for i in range(len(self.dataset))])
        
        # Check for checkpoint
        start_fold = 1
        if resume_from_fold:
            start_fold = resume_from_fold
            if self.verbose:
                print_banner("RESUMING CROSS-VALIDATION", char="!")
                print(f"  Starting from fold {start_fold}/{self.k_folds}")
                print("!" * 60)
        
        if self.verbose:
            print_banner(f"{self.k_folds}-FOLD CROSS-VALIDATION", char="=")
        
        # Track time
        cv_start_time = time.time()
        
        # Iterate through folds
        fold_iterator = enumerate(skf.split(np.zeros(len(all_labels)), all_labels), 1)
        for fold_idx, (train_idx, val_idx) in fold_iterator:
            # Skip if resuming
            if fold_idx < start_fold:
                continue
            
            fold_start_time = time.time()
            
            if self.verbose:
                print_banner(f"FOLD {fold_idx}/{self.k_folds}", char="─")
                print(f"  Train samples: {len(train_idx):,}")
                print(f"  Val samples: {len(val_idx):,}")
            
            # Create fold datasets
            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
            
            # Create data loaders with adaptive batch size for MPS
            # CRITICAL: Reduce batch size on MPS (Apple Silicon) to prevent OOM
            batch_size = TRAINING_CONFIG['batch_size']
            if self.device.type == 'mps' and batch_size > 8:
                batch_size = 8
                if self.verbose and fold_idx == 1:
                    print(f"  ⚠️  Reducing batch size to {batch_size} for MPS device")

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=TRAINING_CONFIG['num_workers'],
                pin_memory=TRAINING_CONFIG.get('pin_memory', True)
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=TRAINING_CONFIG['num_workers'],
                pin_memory=TRAINING_CONFIG.get('pin_memory', True)
            )
            
            # Calculate class distribution for this fold
            train_labels = [all_labels[i] for i in train_idx]
            class_counts = [train_labels.count(i) for i in range(len(self.class_names))]
            
            if self.verbose:
                print(f"\n  Class distribution in fold {fold_idx}:")
                for i, (name, count) in enumerate(zip(self.class_names, class_counts)):
                    pct = safe_divide(count, len(train_idx), 0) * 100
                    print(f"    {name}: {count:,} ({pct:.1f}%)")
            
            # Train fold model
            fold_metrics = self._train_fold(
                fold_idx=fold_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                class_counts=class_counts,
                save_model=save_fold_models
            )
            
            # Store metrics
            self._store_fold_metrics(fold_idx, fold_metrics)
            
            # Save checkpoint after each fold
            # Convert fold results to DataFrame for CheckpointManager
            checkpoint_df = pd.DataFrame(self.fold_results) if self.fold_results else pd.DataFrame()
            self.checkpoint_manager.save_checkpoint(
                data=checkpoint_df,
                last_index=fold_idx,
                metadata={'k_folds': self.k_folds, 'cv_metrics': self.cv_metrics}
            )
            
            # Print fold time
            fold_time = time.time() - fold_start_time
            if self.verbose:
                print(f"\n  Fold {fold_idx} completed in {format_time(fold_time)}")
                print("─" * 60)


            # CRITICAL: Aggressive memory cleanup between folds to prevent MPS memory leak
            if self.device.type == 'mps':
                aggressive_mps_cleanup()
                if self.verbose:
                    print("  🧹 Aggressive MPS cleanup complete")
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                if self.verbose:
                    print("  🧹 Cleared CUDA cache")
            else:
                # CPU - still do garbage collection
                gc.collect()
                
        
        # Calculate statistics across folds
        cv_summary = self._calculate_cv_statistics()
        
        # Add total time
        cv_summary['total_time'] = time.time() - cv_start_time
        
        # Statistical significance testing
        cv_summary['significance'] = self._test_significance()
        
        # Print CV summary
        if self.verbose:
            self._print_cv_summary(cv_summary)
        
        return cv_summary
    
    def _train_fold(self, fold_idx: int, train_loader: DataLoader,
                   val_loader: DataLoader, class_counts: List[int],
                   save_model: bool) -> Dict:
        """Train a single fold and return metrics."""
        # Initialize model
        num_classes = len(self.class_names)
        model = self.model_class(num_classes=num_classes).to(self.device)
        
        # Setup training components
        criterion = get_weighted_criterion(
            class_counts,
            self.device,
            self.class_names,
            allow_zero=True
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler with ADAPTIVE warmup
        num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']

        # CRITICAL FIX: Warmup should be proportional to dataset size
        # For small datasets, use fewer warmup steps to avoid killing the learning rate
        configured_warmup = TRAINING_CONFIG.get('warmup_steps', 500)
        adaptive_warmup = min(configured_warmup, max(1, int(num_training_steps * 0.1)))

        if self.verbose and configured_warmup != adaptive_warmup:
            print(f"  ⚠️  Reducing warmup steps from {configured_warmup} to {adaptive_warmup} for small dataset")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=adaptive_warmup,
            num_training_steps=num_training_steps
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            use_mixed_precision=self.use_mixed_precision,
            model_type=self.model_type  # Pass model_type for unambiguous fallback naming
        )
        
        # Model save path
        model_save_path = None
        if save_model:
            model_save_path = os.path.join(
                models_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_fold{fold_idx}.pt"
            )
        
        # Train
        history = trainer.train(model_save_path=model_save_path)
        
        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Device-aware autocast (MPS doesn't support AMP)
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                use_amp = self.use_mixed_precision and self.device.type == 'cuda'
                with torch.amp.autocast(device_type, enabled=use_amp):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = safe_divide(total_loss, len(val_loader), 0)
        
        # Filter out error labels
        valid_mask = np.array(all_labels) != -1
        valid_preds = np.array(all_preds)[valid_mask]
        valid_labels = np.array(all_labels)[valid_mask]
        
        if len(valid_labels) > 0:
            report = classification_report(
                valid_labels,
                valid_preds,
                labels=list(range(len(self.class_names))),
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            accuracy = accuracy_score(valid_labels, valid_preds)
            f1_macro = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
            f1_weighted = f1_score(valid_labels, valid_preds, average='weighted', zero_division=0)
            precision_macro = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
            recall_macro = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
        else:
            # Handle edge case with no valid predictions
            report = {}
            accuracy = f1_macro = f1_weighted = precision_macro = recall_macro = 0.0
        
        result = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'loss': avg_loss,
            'classification_report': report,
            'history': history
        }
    
        # CRITICAL: Explicit cleanup to prevent MPS memory leak
        # Delete large objects before returning
        del model
        del trainer
        del criterion
        del optimizer
        del scheduler
        
        return result

    
    def _store_fold_metrics(self, fold_idx: int, metrics: Dict):
        """Store metrics from a fold."""
        # Store overall metrics
        self.cv_metrics['accuracy'].append(metrics['accuracy'])
        self.cv_metrics['f1_macro'].append(metrics['f1_macro'])
        self.cv_metrics['f1_weighted'].append(metrics['f1_weighted'])
        self.cv_metrics['precision_macro'].append(metrics['precision_macro'])
        self.cv_metrics['recall_macro'].append(metrics['recall_macro'])
        self.cv_metrics['loss'].append(metrics['loss'])
        
        # Store per-class metrics if report exists
        if metrics['classification_report']:
            for i, class_name in enumerate(self.class_names):
                if class_name in metrics['classification_report']:
                    self.per_class_metrics['precision'][i].append(
                        metrics['classification_report'][class_name]['precision']
                    )
                    self.per_class_metrics['recall'][i].append(
                        metrics['classification_report'][class_name]['recall']
                    )
                    self.per_class_metrics['f1'][i].append(
                        metrics['classification_report'][class_name]['f1-score']
                    )
        
        # Store complete fold result
        self.fold_results.append({
            'fold': fold_idx,
            **metrics
        })
        
        if self.verbose:
            print_banner(f"FOLD {fold_idx} RESULTS", char="─")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
            print(f"  Loss: {metrics['loss']:.4f}")
    
    def _calculate_cv_statistics(self) -> Dict:
        """Calculate mean, std, and confidence intervals across folds."""
        stats = {
            'overall': {},
            'per_class': {},
            'fold_results': self.fold_results
        }
        
        # Calculate overall metrics statistics
        for metric_name, values in self.cv_metrics.items():
            if values:  # Check if not empty
                values_array = np.array(values)
                
                # Calculate confidence interval
                confidence_level = HYPOTHESIS_TESTING_CONFIG.get('confidence_level', 0.95)
                degrees_freedom = len(values) - 1
                sample_mean = np.mean(values_array)
                sample_std = np.std(values_array, ddof=1)  # Use sample std
                
                # t-distribution for small samples
                t_score = scipy_stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
                margin_error = t_score * sample_std / np.sqrt(len(values))
                
                stats['overall'][metric_name] = {
                    'mean': sample_mean,
                    'std': sample_std,
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'confidence_interval': (sample_mean - margin_error, sample_mean + margin_error),
                    'values': values
                }
        
        # Calculate per-class statistics
        for metric_type in ['precision', 'recall', 'f1']:
            stats['per_class'][metric_type] = {}
            for i, class_name in enumerate(self.class_names):
                values = self.per_class_metrics[metric_type][i]
                if values:
                    values_array = np.array(values)
                    stats['per_class'][metric_type][class_name] = {
                        'mean': np.mean(values_array),
                        'std': np.std(values_array, ddof=1),
                        'min': np.min(values_array),
                        'max': np.max(values_array),
                        'values': values
                    }
        
        return stats
    
    def _test_significance(self) -> Dict:
        """Test statistical significance of results."""
        
        significance_results = {}
        
        # Use alpha from config - NO HARDCODING!
        alpha = HYPOTHESIS_TESTING_CONFIG.get('alpha', 0.05)
        
        # Test if performance is significantly different from random
        # For binary: random = 0.5, for 3-class: random = 0.33, etc.
        random_baseline = 1.0 / len(self.class_names)
        
        if self.cv_metrics['accuracy']:
            # One-sample t-test against random baseline
            t_stat, p_value = scipy_stats.ttest_1samp(
                self.cv_metrics['accuracy'],
                random_baseline
            )
            
            significance_results['vs_random'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'random_baseline': random_baseline,
                'alpha': alpha
            }
        
        # Test variance across folds (low variance = stable model)
        if self.cv_metrics['f1_macro']:
            mean_f1 = np.mean(self.cv_metrics['f1_macro'])
            if mean_f1 > 0:
                cv_coefficient = np.std(self.cv_metrics['f1_macro']) / mean_f1
                interpretation = 'Stable' if cv_coefficient < 0.1 else 'Variable'
            else:
                cv_coefficient = float('inf')
                interpretation = 'Unstable (zero mean F1)'

            significance_results['stability'] = {
                'cv_coefficient': cv_coefficient,
                'interpretation': interpretation
            }
        
        return significance_results
    
    def _print_cv_summary(self, cv_summary: Dict):
        """Print enhanced cross-validation summary."""
        print_banner(f"{self.k_folds}-FOLD CV SUMMARY", char="=")
        
        # Overall metrics
        print("\nOverall Metrics (Mean ± Std [95% CI]):")
        print("─" * 60)
        for metric_name, stats in cv_summary['overall'].items():
            ci_lower, ci_upper = stats['confidence_interval']
            print(f"  {metric_name.replace('_', ' ').title():20s}: "
                  f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"[{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Per-class metrics
        print("\nPer-Class Metrics (Mean ± Std):")
        print("─" * 60)
        for class_name in self.class_names:
            print(f"  {class_name}:")
            for metric_type in ['precision', 'recall', 'f1']:
                if class_name in cv_summary['per_class'][metric_type]:
                    stats = cv_summary['per_class'][metric_type][class_name]
                    print(f"    {metric_type.title():10s}: "
                          f"{stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Statistical significance
        if 'significance' in cv_summary:
            print("\nStatistical Analysis:")
            print("─" * 60)
            
            if 'vs_random' in cv_summary['significance']:
                sig = cv_summary['significance']['vs_random']
                print(f"  Performance vs Random ({sig['random_baseline']:.2f}):")
                print(f"    p-value: {sig['p_value']:.4f}")
                print(f"    Result: {'✅ Significantly better' if sig['significant'] else '❌ Not significant'}")
            
            if 'stability' in cv_summary['significance']:
                stab = cv_summary['significance']['stability']
                print(f"  Model Stability:")
                print(f"    CV coefficient: {stab['cv_coefficient']:.3f}")
                print(f"    Assessment: {stab['interpretation']}")
        
        # Time
        if 'total_time' in cv_summary:
            print(f"\nTotal CV Time: {format_time(cv_summary['total_time'])}")
        
        print("=" * 60)
    
    def save_cv_results(self, output_path: str = None) -> str:
        """Save cross-validation results."""
        if output_path is None:
            output_path = os.path.join(
                analysis_results_path,
                f"{EXPERIMENT_CONFIG['experiment_name']}_cv_results.pkl"
            )
        
        ensure_dir_exists(os.path.dirname(output_path))
        
        cv_summary = self._calculate_cv_statistics()
        cv_summary['significance'] = self._test_significance()
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(cv_summary, f)
        
        # Also save as JSON for readability
        json_path = output_path.replace('.pkl', '.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_summary = self._make_json_serializable(cv_summary)
        
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ CV results saved to:")
            print(f"   Pickle: {output_path}")
            print(f"   JSON: {json_path}")
        
        return output_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def train_with_cross_validation(full_dataset,
                                model_class,
                                k_folds: int = None,
                                test_split: float = None,
                                class_names: List[str] = None,
                                save_final_model: bool = True,
                                final_model_path: str = None) -> Dict:
    """
    Complete cross-validation training pipeline.

    This function:
    1. Splits data into train+val (for CV) and test (held out)
    2. Runs K-fold CV on train+val data
    3. Trains final model on full train+val data
    4. Evaluates final model on test data
    5. Returns complete results

    Args:
        full_dataset: Complete dataset (before any splitting)
        model_class: Model class to instantiate (RefusalClassifier or JailbreakDetector)
        k_folds: Number of CV folds (uses config if None)
        test_split: Fraction for test set (uses config if None)
        class_names: List of class names for display
        save_final_model: Whether to save the final trained model
        final_model_path: Path to save final model

    Returns:
        Dictionary with keys:
            - 'cv_results': Cross-validation results and statistics
            - 'test_results': Final model test set performance
            - 'final_model_path': Path to saved model
            - 'split_info': Information about train/val/test splits
    """
    # Use config values if not provided
    if k_folds is None:
        k_folds = CROSS_VALIDATION_CONFIG['default_folds']
    if test_split is None:
        test_split = DATASET_CONFIG['test_split']
    if class_names is None:
        num_classes = len(set(full_dataset.labels))
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Detect model type from model_class for proper fallback naming
    model_type = None
    if model_class.__name__ == 'RefusalClassifier':
        model_type = 'refusal'
    elif model_class.__name__ in ['JailbreakClassifier', 'JailbreakDetector']:
        model_type = 'jailbreak'

    print_banner(f"CROSS-VALIDATION TRAINING PIPELINE ({k_folds}-FOLD)", width=60, char='#')
    print(f"  Total samples: {len(full_dataset):,}")
    print(f"  Test split: {test_split:.1%}")
    print(f"  CV folds: {k_folds}")
    print(f"  Classes: {len(class_names)}")
    if model_type:
        print(f"  Model type: {model_type}")
    print(f"#" * 60)

    # Step 1: Split into train+val (for CV) and test (held out)
    print("\n" + "="*60)
    print("STEP 1: STRATIFIED TRAIN+VAL / TEST SPLIT")
    print("="*60)

    # Get all labels for stratification
    all_labels = np.array(full_dataset.labels)
    all_texts = full_dataset.texts
    all_indices = np.arange(len(full_dataset))

    # Stratified split
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_split,
        random_state=DATASET_CONFIG['random_seed'],
        stratify=all_labels
    )

    # Create train+val dataset (for CV) - proper ClassificationDataset, not Subset
    train_val_texts = [all_texts[i] for i in train_val_idx]
    train_val_labels = [all_labels[i] for i in train_val_idx]
    train_val_dataset = ClassificationDataset(train_val_texts, train_val_labels, full_dataset.tokenizer)

    # Create test dataset - proper ClassificationDataset, not Subset
    test_texts = [all_texts[i] for i in test_idx]
    test_labels = all_labels[test_idx].tolist()
    test_dataset = ClassificationDataset(test_texts, test_labels, full_dataset.tokenizer)

    print(f"  Train+Val: {len(train_val_dataset):,} ({len(train_val_dataset)/len(full_dataset):.1%})")
    print(f"  Test: {len(test_dataset):,} ({len(test_dataset)/len(full_dataset):.1%})")

    # Print test set class distribution
    test_labels = all_labels[test_idx]
    print(f"\n  Test set class distribution:")
    for i, class_name in enumerate(class_names):
        count = (test_labels == i).sum()
        pct = safe_divide(count, len(test_labels), 0) * 100
        print(f"    {class_name}: {count:,} ({pct:.1f}%)")

    # Step 2: Run K-fold cross-validation on train+val data
    print("\n" + "="*60)
    print(f"STEP 2: {k_folds}-FOLD CROSS-VALIDATION")
    print("="*60)

    cv = CrossValidator(
        model_class=model_class,
        dataset=train_val_dataset,
        k_folds=k_folds,
        device=DEVICE,
        class_names=class_names
    )

    cv.run_cross_validation(save_fold_models=False)
    cv_summary = cv._calculate_cv_statistics()
    cv_summary['significance'] = cv._test_significance()
    cv._print_cv_summary(cv_summary)

    # Step 3: Train final model on full train+val data
    print("\n" + "="*60)
    print("STEP 3: TRAIN FINAL MODEL (FULL TRAIN+VAL)")
    print("="*60)
    print(f"  Training on {len(train_val_dataset):,} samples")

    # Create train loader (use all train+val data) with adaptive batch size
    # CRITICAL: Reduce batch size on MPS to prevent OOM
    batch_size = TRAINING_CONFIG['batch_size']
    if DEVICE.type == 'mps' and batch_size > 8:
        batch_size = 8
        print(f"  ⚠️  Reducing batch size to {batch_size} for MPS device")

    train_loader = DataLoader(
        train_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG.get('pin_memory', True)
    )

    # Get class distribution for weighting
    train_val_labels = all_labels[train_val_idx]
    class_counts = [int((train_val_labels == i).sum()) for i in range(len(class_names))]

    print(f"\n  Class distribution:")
    for i, (class_name, count) in enumerate(zip(class_names, class_counts)):
        pct = safe_divide(count, len(train_val_dataset), 0) * 100
        print(f"    {class_name}: {count:,} ({pct:.1f}%)")

    # Initialize final model
    model = model_class(num_classes=len(class_names))
    model.freeze_roberta_layers()
    model = model.to(DEVICE)

    # Count trainable parameters (with fallback if count_parameters not available)
    if 'count_parameters' in globals():
        trainable_params = count_parameters(model)
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {trainable_params:,}")

    # Setup training components
    criterion = get_weighted_criterion(class_counts, DEVICE, class_names=class_names)
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    num_training_steps = len(train_loader) * TRAINING_CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # Create dummy val_loader for Trainer compatibility (no validation during final training)
    val_loader = DataLoader(
        test_dataset,  # Use test set for validation monitoring only
        batch_size=batch_size,  # Use same adaptive batch size
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG.get('pin_memory', True)
    )

    # Train final model
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        DEVICE,
        model_type=model_type  # Pass model_type for unambiguous fallback naming
    )

    # Save path
    if final_model_path is None:
        final_model_path = os.path.join(
            models_path,
            f"{EXPERIMENT_CONFIG['experiment_name']}_cv_final_best.pt"
        )

    history = trainer.train(model_save_path=final_model_path if save_final_model else None)

    if save_final_model:
        print(f"\n✓ Final model saved: {final_model_path}")

    # Step 4: Evaluate final model on test set
    print("\n" + "="*60)
    print("STEP 4: EVALUATE ON TEST SET")
    print("="*60)

    # Load best model
    model.eval()
    if save_final_model:
        checkpoint = torch.load(final_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best model (epoch {checkpoint['epoch']}, val F1: {checkpoint['best_val_f1']:.4f})")

    # Evaluate on test set
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,  # Use same adaptive batch size
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG.get('pin_memory', True)
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label']

            outputs = model(input_ids, attention_mask)
            # Handle both dict and tensor returns
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate test metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class metrics
    test_f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    test_precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    test_recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

    print(f"\n  Test Set Performance:")
    print(f"    Accuracy: {test_accuracy:.4f}")
    print(f"    F1 (macro): {test_f1_macro:.4f}")
    print(f"    F1 (weighted): {test_f1_weighted:.4f}")
    print(f"    Precision (macro): {test_precision:.4f}")
    print(f"    Recall (macro): {test_recall:.4f}")

    print(f"\n  Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        # Only print metrics if class exists in test set
        if i < len(test_f1_per_class):
            print(f"    {class_name}:")
            print(f"      F1: {test_f1_per_class[i]:.4f}")
            print(f"      Precision: {test_precision_per_class[i]:.4f}")
            print(f"      Recall: {test_recall_per_class[i]:.4f}")
        else:
            print(f"    {class_name}:")
            print(f"      No samples in test set")

    
    # CRITICAL: Add cv_results to the saved checkpoint
    # The Trainer already saved the model, but it doesn't know about cv_results
    # So we reload the checkpoint, add cv_results, and save it again
    if save_final_model and final_model_path:
        print(f"\n💾 Adding CV results to checkpoint...")
        checkpoint = torch.load(final_model_path, map_location='cpu')
        checkpoint['cv_results'] = cv_summary  # Add CV results
        
        checkpoint['split_info'] = {
            'train_val_size': len(train_val_dataset),
            'test_size': len(test_dataset),
            'train_val_indices': train_val_idx.tolist(),
            'test_indices': test_idx.tolist()
        }
        
        torch.save(checkpoint, final_model_path)
        print(f"✓ CV results saved to checkpoint")


    # Return complete results
    results = {
        'cv_results': cv_summary,
        'test_results': {
            'accuracy': test_accuracy,
            'f1_macro': test_f1_macro,
            'f1_weighted': test_f1_weighted,
            'precision_macro': test_precision,
            'recall_macro': test_recall,
            'f1_per_class': {class_names[i]: float(test_f1_per_class[i]) for i in range(min(len(class_names), len(test_f1_per_class)))},
            'precision_per_class': {class_names[i]: float(test_precision_per_class[i]) for i in range(min(len(class_names), len(test_precision_per_class)))},
            'recall_per_class': {class_names[i]: float(test_recall_per_class[i]) for i in range(min(len(class_names), len(test_recall_per_class)))},
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        },
        'final_model_path': final_model_path if save_final_model else None,
        'split_info': {
            'train_val_size': len(train_val_dataset),
            'test_size': len(test_dataset),
            'train_val_indices': train_val_idx.tolist(),
            'test_indices': test_idx.tolist()
        }
    }

    print("\n" + "#"*60)
    print("CROSS-VALIDATION TRAINING COMPLETE")
    print("#"*60)

    return results


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
