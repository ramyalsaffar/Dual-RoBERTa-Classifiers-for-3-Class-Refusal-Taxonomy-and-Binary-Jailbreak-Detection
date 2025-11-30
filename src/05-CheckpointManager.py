# CheckpointManager Utility
#---------------------------
# Handles checkpoint saving/loading for error recovery during long-running operations.
# Provides automatic cleanup and resumption capabilities.
# All imports are in 01-Imports.py
###############################################################################


class CheckpointManager:
    """
    Manages checkpoints for error recovery during long-running API operations.

    Features:
    - Automatic checkpoint saving at configurable intervals (time or count based)
    - Resume from latest checkpoint with fallback to older ones
    - Automatic cleanup of old checkpoints
    - Size management to prevent disk overflow
    - Version compatibility checking
    - Emergency checkpoint on errors
    
    IMPORTANT: Uses settings from CHECKPOINT_CONFIG, never hardcodes values!
    """

    def __init__(self, checkpoint_dir: str = None, operation_name: str = None,
                 checkpoint_every: int = None, auto_cleanup: bool = None,
                 keep_last_n: int = None, max_age_hours: int = None,
                 checkpoint_interval_seconds: int = 300,
                 max_total_size_mb: int = 1000):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints (default: from paths)
            operation_name: Name of operation (e.g., 'labeling', 'collection')
            checkpoint_every: Save checkpoint every N items (default: from CHECKPOINT_CONFIG)
            auto_cleanup: Automatically delete old checkpoints (default: from CHECKPOINT_CONFIG)
            keep_last_n: Number of recent checkpoints to keep (default: from CHECKPOINT_CONFIG)
            max_age_hours: Maximum checkpoint age in hours (default: from CHECKPOINT_CONFIG)
            checkpoint_interval_seconds: Time interval for checkpoints (default: 300 = 5 minutes)
            max_total_size_mb: Maximum total size of checkpoints in MB (default: 1000)
        """
        # Use Config values as defaults - NO HARDCODING!
        self.checkpoint_dir = checkpoint_dir or data_checkpoints_path
        self.operation_name = operation_name or 'general'
        
        # Get settings from CHECKPOINT_CONFIG based on operation type
        if operation_name == 'labeling':
            self.checkpoint_every = checkpoint_every or CHECKPOINT_CONFIG.get('labeling_checkpoint_every', 100)
        elif operation_name == 'collection':
            self.checkpoint_every = checkpoint_every or CHECKPOINT_CONFIG.get('collection_checkpoint_every', 500)
        else:
            # Default to smaller value for unknown operations
            self.checkpoint_every = checkpoint_every or 100
            
        self.auto_cleanup = auto_cleanup if auto_cleanup is not None else CHECKPOINT_CONFIG.get('auto_cleanup', True)
        self.keep_last_n = keep_last_n or CHECKPOINT_CONFIG.get('keep_last_n', 2)
        self.max_age_hours = max_age_hours or CHECKPOINT_CONFIG.get('max_checkpoint_age_hours', 48)
        
        # Time-based checkpointing
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        
        # Size management
        self.max_total_size_mb = max_total_size_mb
        
        # Track checkpoint history for recovery
        self.checkpoint_history = []
        
        # Version for compatibility checking
        self.checkpoint_version = "1.0"

        # Create checkpoint directory using Utils function
        ensure_dir_exists(self.checkpoint_dir)
        
        # Print configuration using Utils
        if EXPERIMENT_CONFIG.get('verbose', True):
            print_banner(f"CHECKPOINT MANAGER: {self.operation_name.upper()}", width=60, char="-")
            print(f"  Directory: {self.checkpoint_dir}")
            print(f"  Save every: {self.checkpoint_every} items OR {self.checkpoint_interval_seconds}s")
            print(f"  Auto cleanup: {self.auto_cleanup}")
            print(f"  Keep last: {self.keep_last_n} checkpoints")
            print(f"  Max age: {self.max_age_hours} hours")
            print(f"  Max total size: {self.max_total_size_mb} MB")
            print("-" * 60)

    def should_checkpoint(self, current_index: int, force: bool = False) -> bool:
        """
        Enhanced check if a checkpoint should be saved.
        
        Args:
            current_index: Current processing index
            force: Force checkpoint regardless of intervals
            
        Returns:
            True if checkpoint should be saved
        """
        if force:
            return True
        
        # Check index-based interval
        if current_index > 0 and current_index % self.checkpoint_every == 0:
            return True
        
        # Check time-based interval
        time_since_last = time.time() - self.last_checkpoint_time
        if time_since_last >= self.checkpoint_interval_seconds:
            return True
        
        return False

    def save_checkpoint(self, data: pd.DataFrame, last_index: int,
                       metadata: Dict = None, verbose: bool = None,
                       force: bool = False) -> str:
        """
        Save checkpoint to disk with size management and versioning.

        Args:
            data: DataFrame with processed data
            last_index: Last processed index
            metadata: Optional metadata dictionary
            verbose: Override verbosity setting
            force: Force checkpoint regardless of intervals

        Returns:
            Path to saved checkpoint or None if skipped
        """
        # Check if we should checkpoint
        if not force and not self.should_checkpoint(last_index):
            return None
        
        # Use get_timestamp from Utils
        timestamp = get_timestamp('file') + "_" + datetime.now().strftime("%S")  # Add seconds for uniqueness
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.operation_name}_{timestamp}.pkl"
        )

        checkpoint_data = {
            'data': data.copy(),
            'last_index': last_index,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'experiment': EXPERIMENT_CONFIG.get('experiment_name', 'unknown'),
            'created_at': get_timestamp('display'),  # Human-readable timestamp
            'version': self.checkpoint_version,
            'operation': self.operation_name
        }

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Update last checkpoint time
            self.last_checkpoint_time = time.time()
            
            # Add to history
            self.checkpoint_history.append(checkpoint_path)
            
            # Use verbose setting from config
            verbose = verbose if verbose is not None else EXPERIMENT_CONFIG.get('verbose', True)
            if verbose:
                file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")
                print(f"   Progress: {last_index:,} items completed")
                print(f"   Data shape: {data.shape}")
                print(f"   File size: {file_size_mb:.2f} MB")
            
            # Auto-cleanup if enabled (using config value)
            if self.auto_cleanup:
                self.cleanup_checkpoints(keep_last_n=self.keep_last_n)
            
            # Size-based cleanup
            self._manage_checkpoint_size()
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return None

    def load_latest_checkpoint(self, max_age_hours: int = None, 
                              fallback_attempts: int = 3) -> Optional[Dict]:
        """
        Load the most recent checkpoint with fallback to older ones if needed.
        
        Args:
            max_age_hours: Maximum age of checkpoint in hours (default: from CHECKPOINT_CONFIG)
            fallback_attempts: Number of older checkpoints to try if latest fails
            
        Returns:
            Checkpoint data dictionary or None if no valid checkpoint found
        """
        max_age_hours = max_age_hours or self.max_age_hours
        
        # Find all checkpoints for this operation
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        
        if not checkpoints:
            if EXPERIMENT_CONFIG.get('verbose', True):
                print(f"ℹ️  No checkpoints found for operation: {self.operation_name}")
            return None
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # Try multiple checkpoints if needed
        for i, checkpoint_path in enumerate(checkpoints[:fallback_attempts]):
            file_age_seconds = time.time() - os.path.getmtime(checkpoint_path)
            file_age_hours = file_age_seconds / 3600
            
            # Skip if too old
            if file_age_hours > max_age_hours:
                if i == 0 and EXPERIMENT_CONFIG.get('verbose', True):
                    print(f"⚠️  Latest checkpoint is {format_time(file_age_seconds)} old")
                    print(f"   Maximum allowed: {max_age_hours} hours")
                continue
            
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Check version compatibility
                if not self._check_version_compatibility(checkpoint_data):
                    print(f"⚠️  Version mismatch in checkpoint: {os.path.basename(checkpoint_path)}")
                    continue
                
                # Validate checkpoint
                if not self._validate_checkpoint(checkpoint_data):
                    print(f"⚠️  Validation failed for: {os.path.basename(checkpoint_path)}")
                    continue
                
                # Success!
                if EXPERIMENT_CONFIG.get('verbose', True):
                    print_banner("CHECKPOINT LOADED", width=60, char="=")
                    print(f"✅ File: {os.path.basename(checkpoint_path)}")
                    if i > 0:
                        print(f"   Note: Using backup checkpoint (attempt {i+1}/{fallback_attempts})")
                    print(f"   Resuming from index: {checkpoint_data['last_index']:,}")
                    print(f"   Data shape: {checkpoint_data['data'].shape}")
                    print(f"   Age: {format_time(file_age_seconds)}")
                    if 'experiment' in checkpoint_data:
                        print(f"   Original experiment: {checkpoint_data['experiment']}")
                    print("=" * 60)
                
                return checkpoint_data
                
            except Exception as e:
                print(f"⚠️  Error loading checkpoint {i+1}/{fallback_attempts}: {e}")
                continue
        
        print(f"❌ No valid checkpoints found after {fallback_attempts} attempts")
        return None

    def cleanup_checkpoints(self, keep_last_n: int = None) -> None:
        """
        Delete old checkpoints, keeping only the N most recent.

        Args:
            keep_last_n: Number of recent checkpoints to keep (default: from CHECKPOINT_CONFIG)
        """
        # Use config value if not specified
        keep_last_n = keep_last_n or self.keep_last_n
        
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))

        if len(checkpoints) <= keep_last_n:
            return

        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)

        # Delete old checkpoints
        to_delete = checkpoints[keep_last_n:]
        deleted_count = 0
        
        for checkpoint in to_delete:
            try:
                os.remove(checkpoint)
                deleted_count += 1
                if EXPERIMENT_CONFIG.get('verbose', True):
                    print(f"🗑️  Deleted old checkpoint: {os.path.basename(checkpoint)}")
            except Exception as e:
                print(f"⚠️  Could not delete {os.path.basename(checkpoint)}: {e}")
        
        if deleted_count > 0 and EXPERIMENT_CONFIG.get('show_progress', True):
            print(f"   Cleaned up {deleted_count} old checkpoint(s)")

    def _manage_checkpoint_size(self) -> None:
        """Manage total checkpoint size to prevent disk overflow."""
        stats = self.get_checkpoint_stats()
        
        if stats['total_size_mb'] > self.max_total_size_mb:
            print(f"⚠️  Total checkpoint size ({stats['total_size_mb']:.1f}MB) exceeds limit ({self.max_total_size_mb}MB)")
            
            # Get all checkpoints sorted by age
            pattern = f"checkpoint_{self.operation_name}_*.pkl"
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))
            checkpoints.sort(key=os.path.getmtime)  # Oldest first
            
            # Delete oldest until under limit
            for checkpoint in checkpoints[:-self.keep_last_n]:  # Keep minimum N
                try:
                    size_mb = os.path.getsize(checkpoint) / (1024 * 1024)
                    os.remove(checkpoint)
                    print(f"🗑️  Deleted checkpoint to save space: {os.path.basename(checkpoint)} ({size_mb:.1f}MB)")
                    
                    # Check if we're under limit now
                    new_stats = self.get_checkpoint_stats()
                    if new_stats['total_size_mb'] < self.max_total_size_mb:
                        break
                        
                except Exception as e:
                    print(f"⚠️  Could not delete {os.path.basename(checkpoint)}: {e}")

    def _validate_checkpoint(self, checkpoint_data: Dict) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            checkpoint_data: Checkpoint dictionary

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['data', 'last_index', 'timestamp']

        # Check required keys
        if not all(key in checkpoint_data for key in required_keys):
            return False

        # Check data types
        if not isinstance(checkpoint_data['data'], pd.DataFrame):
            return False

        if not isinstance(checkpoint_data['last_index'], int):
            return False

        # Check DataFrame not empty
        if checkpoint_data['data'].empty:
            return False

        return True
    
    def _check_version_compatibility(self, checkpoint_data: Dict) -> bool:
        """
        Check if checkpoint version is compatible with current code.
        
        Args:
            checkpoint_data: Checkpoint dictionary
            
        Returns:
            True if compatible, False otherwise
        """
        if 'version' not in checkpoint_data:
            # Old checkpoint without version - warn but allow
            print(f"⚠️  Checkpoint has no version info (assuming v0.x)")
            return True
        
        checkpoint_version = checkpoint_data['version']
        current_version = self.checkpoint_version
        
        # Simple compatibility: major version must match
        checkpoint_major = checkpoint_version.split('.')[0]
        current_major = current_version.split('.')[0]
        
        return checkpoint_major == current_major
    
    def create_recovery_checkpoint(self, data: pd.DataFrame, last_index: int,
                                  error_info: str = None) -> str:
        """
        Create an emergency checkpoint on error/exception.
        
        Args:
            data: Current data state
            last_index: Last successfully processed index
            error_info: Error description
            
        Returns:
            Path to recovery checkpoint
        """
        timestamp = get_timestamp('file') + "_RECOVERY"
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.operation_name}_{timestamp}.pkl"
        )
        
        checkpoint_data = {
            'data': data.copy(),
            'last_index': last_index,
            'timestamp': timestamp,
            'created_at': get_timestamp('display'),
            'error_info': error_info,
            'recovery': True,
            'version': self.checkpoint_version,
            'experiment': EXPERIMENT_CONFIG.get('experiment_name', 'unknown')
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"🚨 Emergency checkpoint saved: {os.path.basename(checkpoint_path)}")
            return checkpoint_path
        except Exception as e:
            print(f"❌ Failed to save emergency checkpoint: {e}")
            return None
    
    def get_checkpoint_stats(self) -> Dict:
        """
        Get statistics about existing checkpoints.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        
        if not checkpoints:
            return {
                'count': 0,
                'total_size_mb': 0,
                'oldest_age_hours': 0,
                'newest_age_hours': 0,
                'average_size_mb': 0
            }
        
        # Calculate stats
        sizes = [os.path.getsize(cp) for cp in checkpoints]
        ages = [(time.time() - os.path.getmtime(cp)) / 3600 for cp in checkpoints]
        
        return {
            'count': len(checkpoints),
            'total_size_mb': sum(sizes) / (1024 * 1024),
            'oldest_age_hours': max(ages),
            'newest_age_hours': min(ages),
            'average_size_mb': (sum(sizes) / len(sizes)) / (1024 * 1024)
        }
    
    def list_checkpoints(self, show_details: bool = True) -> List[str]:
        """
        List all checkpoints for this operation.
        
        Args:
            show_details: Print detailed information about each checkpoint
            
        Returns:
            List of checkpoint paths
        """
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))
        
        if not checkpoints:
            print(f"No checkpoints found for operation: {self.operation_name}")
            return []
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        if show_details and EXPERIMENT_CONFIG.get('verbose', True):
            print_banner(f"CHECKPOINTS: {self.operation_name.upper()}", width=60, char="-")
            for i, cp in enumerate(checkpoints, 1):
                size_mb = os.path.getsize(cp) / (1024 * 1024)
                age_seconds = time.time() - os.path.getmtime(cp)
                print(f"{i:2}. {os.path.basename(cp)}")
                print(f"    Size: {size_mb:.2f} MB | Age: {format_time(age_seconds)}")
            print("-" * 60)
        
        return checkpoints
    
    def delete_all_checkpoints(self, confirm: bool = False) -> int:
        """
        Delete all checkpoints for this operation.

        Args:
            confirm: Must be True to actually delete (safety check)

        Returns:
            Number of checkpoints deleted
        """
        if not confirm:
            print("⚠️  Set confirm=True to delete all checkpoints")
            return 0

        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))

        deleted = 0
        for cp in checkpoints:
            try:
                os.remove(cp)
                deleted += 1
            except Exception as e:
                print(f"⚠️  Could not delete {os.path.basename(cp)}: {e}")

        if deleted > 0:
            print(f"🗑️  Deleted {deleted} checkpoint(s) for {self.operation_name}")

        return deleted

    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict]:
        """
        Get detailed information about a specific checkpoint without loading full data.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata or None if error
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            info = {
                'path': checkpoint_path,
                'basename': os.path.basename(checkpoint_path),
                'size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
                'age_seconds': time.time() - os.path.getmtime(checkpoint_path),
                'age_hours': (time.time() - os.path.getmtime(checkpoint_path)) / 3600,
                'last_index': checkpoint_data.get('last_index', 0),
                'data_shape': checkpoint_data['data'].shape if 'data' in checkpoint_data else (0, 0),
                'timestamp': checkpoint_data.get('timestamp', 'unknown'),
                'created_at': checkpoint_data.get('created_at', 'unknown'),
                'version': checkpoint_data.get('version', 'unknown'),
                'experiment': checkpoint_data.get('experiment', 'unknown'),
                'operation': checkpoint_data.get('operation', self.operation_name),
                'metadata': checkpoint_data.get('metadata', {}),
                'is_recovery': checkpoint_data.get('recovery', False)
            }

            return info

        except Exception as e:
            print(f"⚠️  Error reading checkpoint info: {e}")
            return None

    def list_all_checkpoints_with_info(self) -> List[Dict]:
        """
        Get detailed information about all checkpoints for this operation.

        Returns:
            List of dictionaries with checkpoint information
        """
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))

        if not checkpoints:
            return []

        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)

        checkpoint_info_list = []
        for cp in checkpoints:
            info = self.get_checkpoint_info(cp)
            if info:
                checkpoint_info_list.append(info)

        return checkpoint_info_list

    def load_checkpoint_by_name(self, checkpoint_name: str) -> Optional[Dict]:
        """
        Load a specific checkpoint by its filename.

        Args:
            checkpoint_name: Checkpoint filename (e.g., 'checkpoint_labeling_20250111_143022.pkl')

        Returns:
            Checkpoint data dictionary or None if not found
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_name}")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Validate checkpoint
            if not self._validate_checkpoint(checkpoint_data):
                print(f"⚠️  Checkpoint validation failed: {checkpoint_name}")
                return None

            if EXPERIMENT_CONFIG.get('verbose', True):
                print_banner("CHECKPOINT LOADED", width=60, char="=")
                print(f"✅ File: {checkpoint_name}")
                print(f"   Resuming from index: {checkpoint_data['last_index']:,}")
                print(f"   Data shape: {checkpoint_data['data'].shape}")
                if 'experiment' in checkpoint_data:
                    print(f"   Original experiment: {checkpoint_data['experiment']}")
                print("=" * 60)

            return checkpoint_data

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return None

    def delete_checkpoints_after_timestamp(self, timestamp: str) -> int:
        """
        Delete all checkpoints created after a specific timestamp.
        Useful for removing checkpoints from later steps when restarting from earlier step.

        Args:
            timestamp: Timestamp string (format: YYYYMMDD_HHMMSS)

        Returns:
            Number of checkpoints deleted
        """
        pattern = f"checkpoint_{self.operation_name}_*.pkl"
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, pattern))

        deleted = 0
        for cp in checkpoints:
            basename = os.path.basename(cp)
            # Extract timestamp from filename: checkpoint_operation_TIMESTAMP.pkl
            parts = basename.replace('.pkl', '').split('_')
            if len(parts) >= 3:
                cp_timestamp = '_'.join(parts[2:])  # Get everything after operation name

                # Compare timestamps as strings (works for YYYYMMDD_HHMMSS format)
                if cp_timestamp > timestamp:
                    try:
                        os.remove(cp)
                        deleted += 1
                        if EXPERIMENT_CONFIG.get('verbose', True):
                            print(f"🗑️  Deleted checkpoint: {basename} (timestamp: {cp_timestamp})")
                    except Exception as e:
                        print(f"⚠️  Could not delete {basename}: {e}")

        if deleted > 0:
            print(f"✓ Deleted {deleted} checkpoint(s) created after {timestamp}")

        return deleted


    @staticmethod
    def save_pipeline_checkpoint(step_number: int, experiment_name: str, timestamp: str):
        """Save pipeline checkpoint after step completes."""
        checkpoint_data = {
            'step_completed': step_number,
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'saved_at': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(data_checkpoints_path, f"pipeline_checkpoint_{timestamp}.json")
        ensure_dir_exists(data_checkpoints_path)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"✓ Pipeline checkpoint saved: Step {step_number} complete")


    @staticmethod
    def load_pipeline_checkpoint():
        """Load most recent pipeline checkpoint."""
        checkpoint_pattern = os.path.join(data_checkpoints_path, "pipeline_checkpoint_*.json")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        with open(latest_checkpoint, 'r') as f:
            return json.load(f)
    

#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
