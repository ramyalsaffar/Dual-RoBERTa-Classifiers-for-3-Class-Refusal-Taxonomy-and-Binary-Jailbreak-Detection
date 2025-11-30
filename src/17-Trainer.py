# Trainer Module 
#----------------
# Enhanced training loop with progress tracking, early stopping, and checkpointing.
# Includes weighted loss, mixed precision, and comprehensive logging.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Mixed precision training support
# - Gradient accumulation
# - Learning rate tracking
# - Test set evaluation
# - Better logging with Utils
# All imports are in 01-Imports.py
###############################################################################


# =============================================================================
# WEIGHTED LOSS FUNCTIONS (for class imbalance)
# =============================================================================

def calculate_class_weights(class_counts: List[int], 
                           device: torch.device,
                           allow_zero: bool = False, 
                           zero_weight: float = 1.0) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Formula: weight_i = total_samples / (num_classes * count_i)
    
    Args:
        class_counts: List of sample counts per class
        device: torch device
        allow_zero: If True, allows zero counts (assigns zero_weight)
        zero_weight: Weight to assign to classes with zero samples
    
    Returns:
        torch.Tensor of class weights
    
    Raises:
        ValueError: If any class has zero samples and allow_zero=False
    """
    # Check for zero counts
    zero_indices = [i for i, count in enumerate(class_counts) if count == 0]
    
    if zero_indices and not allow_zero:
        raise ValueError(
            f"Class counts contain zeros: {class_counts}. "
            f"Zero counts found at indices: {zero_indices}. "
            f"Set allow_zero=True to handle zero-sample classes."
        )
    
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    # Calculate weights with zero handling
    weights = []
    for count in class_counts:
        if count == 0:
            weights.append(zero_weight)
        else:
            weight = safe_divide(total_samples, num_classes * count, 1.0)
            weights.append(weight)
    
    return torch.FloatTensor(weights).to(device)


def get_weighted_criterion(class_counts: List[int], 
                          device: torch.device,
                          class_names: List[str] = None,
                          allow_zero: bool = False,
                          zero_weight: float = 1.0) -> nn.CrossEntropyLoss:
    """
    Get weighted CrossEntropyLoss criterion.
    
    Args:
        class_counts: List of sample counts per class
        device: torch device
        class_names: Optional list of class names for display
        allow_zero: If True, allows zero counts
        zero_weight: Weight for zero-sample classes
    
    Returns:
        nn.CrossEntropyLoss with class weights
    """
    class_weights = calculate_class_weights(class_counts, device, allow_zero, zero_weight)
    
    if EXPERIMENT_CONFIG.get('verbose', True):
        print_banner("CLASS WEIGHTS", char="─")
        for i, weight in enumerate(class_weights):
            class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            count = class_counts[i] if i < len(class_counts) else 0
            print(f"  {class_label}: {weight:.3f} (count: {count:,})")
        print("─" * 60)
    
    return nn.CrossEntropyLoss(weight=class_weights)


# =============================================================================
# ENHANCED TRAINER CLASS
# =============================================================================

class Trainer:
    """Enhanced trainer with mixed precision and comprehensive logging."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: torch.device,
                 test_loader: DataLoader = None,
                 use_mixed_precision: bool = None,
                 gradient_accumulation_steps: int = 1,
                 checkpoint_manager: CheckpointManager = None,
                 model_type: str = None):
        """
        Initialize enhanced trainer.
        
        Args:
            model: Classification model (RefusalClassifier or JailbreakClassifier)
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: torch device
            test_loader: Optional test data loader
            use_mixed_precision: Use automatic mixed precision
            gradient_accumulation_steps: Steps for gradient accumulation
            checkpoint_manager: Optional checkpoint manager
            model_type: Type identifier for fallback naming ('refusal', 'jailbreak', or None)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model_type = model_type  # Store for fallback naming
        
        # Configuration from config - NO HARDCODING!
        self.epochs = TRAINING_CONFIG['epochs']
        self.gradient_clip = TRAINING_CONFIG['gradient_clip']
        self.early_stopping_patience = TRAINING_CONFIG['early_stopping_patience']
        self.save_best_only = TRAINING_CONFIG['save_best_only']
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision or TRAINING_CONFIG.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Checkpoint manager
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            operation_name='training'
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Early stopping tracking
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Count parameters
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.total_params = sum(p.numel() for p in model.parameters())
        
        if self.verbose:
            self._print_training_config()
    
    def _print_training_config(self):
        """Print training configuration."""
        print_banner("TRAINING CONFIGURATION", char="=")
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Device: {self.device}")
        print(f"  Total parameters: {self.total_params:,}")
        print(f"  Trainable parameters: {self.trainable_params:,}")
        print(f"\n  Training Settings:")
        print(f"    Epochs: {self.epochs}")
        print(f"    Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"    Learning rate: {TRAINING_CONFIG['learning_rate']}")
        print(f"    Gradient clipping: {self.gradient_clip}")
        print(f"    Early stopping patience: {self.early_stopping_patience}")
        print(f"    Mixed precision: {self.use_mixed_precision}")
        print(f"    Gradient accumulation: {self.gradient_accumulation_steps} steps")
        print(f"\n  Data:")
        print(f"    Train batches: {len(self.train_loader)}")
        print(f"    Val batches: {len(self.val_loader)}")
        if self.test_loader:
            print(f"    Test batches: {len(self.test_loader)}")
        print("=" * 60)
    
    def train_epoch(self) -> float:
        """
        Train one epoch with mixed precision support.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, 
                 desc="Training", 
                 leave=False) as pbar:
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Device-aware autocast (MPS doesn't support AMP)
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                use_amp = self.use_mixed_precision and self.device.type == 'cuda'
                with torch.amp.autocast(device_type, enabled=use_amp):
                    # Forward pass
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_mixed_precision:
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip
                    )
                    
                    if self.use_mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_description(f"Training (LR: {current_lr:.2e})")
                
                # Track loss (unscaled)
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = safe_divide(total_loss, num_batches, 0)
        return avg_loss
    
    def validate_epoch(self, loader: DataLoader = None, 
                      desc: str = "Validation") -> Dict[str, float]:
        """
        Validate with comprehensive metrics.
        
        Args:
            loader: Data loader (uses val_loader if None)
            desc: Description for progress bar
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        loader = loader or self.val_loader
        
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = len(loader)
        
        with torch.no_grad():
            with tqdm(total=num_batches, desc=desc, leave=False) as pbar:
                for batch in loader:
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Device-aware autocast (MPS doesn't support AMP)
                    device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                    use_amp = self.use_mixed_precision and self.device.type == 'cuda'
                    with torch.amp.autocast(device_type, enabled=use_amp):
                        logits = self.model(input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=1)
                    
                    # Track metrics
                    total_loss += loss.item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    pbar.update(1)
        
        # Calculate metrics
        avg_loss = safe_divide(total_loss, num_batches, 0)
        
        # Filter out error labels (-1) for metrics
        valid_mask = np.array(all_labels) != -1
        valid_preds = np.array(all_preds)[valid_mask]
        valid_labels = np.array(all_labels)[valid_mask]
        
        if len(valid_labels) > 0:
            f1 = f1_score(valid_labels, valid_preds, average='macro')
            accuracy = accuracy_score(valid_labels, valid_preds)
            precision = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
            recall = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)
        else:
            f1 = accuracy = precision = recall = 0.0
        
        return {
            'loss': avg_loss,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, 
             model_save_path: str = None,
             resume_from_checkpoint: str = None):
        """
        Full training with early stopping and checkpoint support.
        
        Args:
            model_save_path: Path to save best model
            resume_from_checkpoint: Path to resume from checkpoint
        
        Returns:
            Training history dictionary
        """
        # Create unambiguous fallback path if none provided
        if model_save_path is None:
            # Use model_type to create descriptive filename
            if self.model_type:
                filename = f"{EXPERIMENT_CONFIG['experiment_name']}_{self.model_type}_best.pt"
            else:
                filename = f"{EXPERIMENT_CONFIG['experiment_name']}_best.pt"
            
            model_save_path = os.path.join(models_path, filename)
            
            if self.verbose:
                print(f"⚠️  No model_save_path provided, using fallback: {os.path.basename(model_save_path)}")
        
        # Resume from checkpoint if specified
        start_epoch = 1
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print_banner("RESUMING FROM CHECKPOINT", char="!")
            self.load_checkpoint(resume_from_checkpoint)
            start_epoch = self.best_epoch + 1
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Previous best F1: {self.best_val_f1:.4f}")
            print("!" * 60)
        
        print_banner("TRAINING START", char="=")
        print(f"Training for {self.epochs} epochs (starting from {start_epoch})")
        
        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.time()
            
            print_banner(f"EPOCH {epoch}/{self.epochs}", char="─")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch(desc="Validation")
            
            # Track epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(epoch_time)
            
            # Print metrics
            print(f"\n  Training:")
            print(f"    Loss: {train_loss:.4f}")
            print(f"    LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"\n  Validation:")
            print(f"    Loss: {val_metrics['loss']:.4f}")
            print(f"    F1: {val_metrics['f1']:.4f}")
            print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall: {val_metrics['recall']:.4f}")
            print(f"\n  Time: {format_time(epoch_time)}")
            
            # Check for improvement
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(model_save_path, epoch=epoch)
                print(f"\n  ✅ New best model! F1: {val_metrics['f1']:.4f}")
            else:
                self.patience_counter += 1
                print(f"\n  No improvement ({self.patience_counter}/{self.early_stopping_patience})")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print_banner("EARLY STOPPING", char="!")
                print(f"  Stopped at epoch {epoch}")
                print(f"  Best epoch: {self.best_epoch}")
                print(f"  Best F1: {self.best_val_f1:.4f}")
                break
        
        # Test evaluation if available
        if self.test_loader:
            print_banner("TEST EVALUATION", char="=")
            test_metrics = self.validate_epoch(self.test_loader, desc="Testing")
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Test F1: {test_metrics['f1']:.4f}")
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            self.history['test_metrics'] = test_metrics
        
        # Training complete
        print_banner("TRAINING COMPLETE", char="=")
        print(f"  Best Validation F1: {self.best_val_f1:.4f}")
        print(f"  Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"  Best Epoch: {self.best_epoch}")
        print(f"  Total Training Time: {format_time(sum(self.history['epoch_times']))}")
        print(f"  Model saved to: {model_save_path}")
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int = None):
        """Save enhanced checkpoint with all training state."""
        ensure_dir_exists(os.path.dirname(path))
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'epoch': epoch if epoch is not None else self.best_epoch,
            'history': self.history,
            'config': {
                'experiment_name': EXPERIMENT_CONFIG['experiment_name'],
                'model_name': MODEL_CONFIG['model_name'],
                'learning_rate': TRAINING_CONFIG['learning_rate'],
                'batch_size': TRAINING_CONFIG['batch_size']
            }
        }
        
        if self.use_mixed_precision and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load enhanced checkpoint."""
        checkpoint = safe_load_checkpoint(path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.verbose:
            print(f"✅ Loaded checkpoint from {path}")
            print(f"   Best Val F1: {self.best_val_f1:.4f}")
            print(f"   Best Epoch: {self.best_epoch}")


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
