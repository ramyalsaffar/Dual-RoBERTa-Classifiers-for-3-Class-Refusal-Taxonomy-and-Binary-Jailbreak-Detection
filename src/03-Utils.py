# Utils
#------
# Utility functions and classes for the project.
# Includes helper functions and the KeepAwake context manager.
###############################################################################


#------------------------------------------------------------------------------


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_timestamp(format_type: str = 'file') -> str:
    """
    Get standard timestamp for consistent naming across the project.

    Matches the format used in EXPERIMENT_CONFIG['experiment_name'] for consistency.

    Args:
        format_type: 'file' for filenames (YYYYmmdd_HHMM),
                     'display' for human-readable (YYYY-MM-DD HH:MM:SS)

    Returns:
        Formatted timestamp string

    Examples:
        >>> get_timestamp('file')
        '20250115_1430'
        >>> get_timestamp('display')
        '2025-01-15 14:30:00'
    """
    if format_type == 'file':
        return datetime.now().strftime("%Y%m%d_%H%M")
    elif format_type == 'display':
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return datetime.now().strftime("%Y%m%d_%H%M")


def safe_load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """
    Safely load PyTorch checkpoint, handling MPS alignment bug on Mac.
    
    On Mac with Apple Silicon (M1/M2/M3), loading checkpoints directly to MPS device
    causes an alignment error in PyTorch's MPS backend. This function works around
    that bug by loading to CPU first, then moving to the target device.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt, .pth)
        device: Target device (cuda, mps, or cpu)
    
    Returns:
        Loaded checkpoint dictionary
    
    Example:
        >>> checkpoint = safe_load_checkpoint('model.pt', DEVICE)
        >>> model.load_state_dict(checkpoint['model_state_dict'])
        >>> model = model.to(DEVICE)
    
    Technical Details:
        - PyTorch Issue: MPS backend has alignment bug in load_state_dict
        - Workaround: Load to device specified in TRAINING_CONFIG (typically CPU)
        - Performance Impact: None (model still runs on GPU/MPS after loading)
        - Configuration: TRAINING_CONFIG['checkpoint_load_device']
    """
    # Load to device specified in config (no hardcoding!)
    load_device = TRAINING_CONFIG.get('checkpoint_load_device', 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=load_device, weights_only=False)

    return checkpoint


# =============================================================================
# DYNAMIC RATE LIMIT MANAGEMENT
# =============================================================================

class DynamicRateLimiter:
    """
    Thread-safe adaptive rate limiter that automatically adjusts workers and delays
    based on API rate limit responses.
    
    Strategy:
    1. Start with 5 workers, 0.2s delay
    2. On rate limit: reduce workers (5→4→3→2)
    3. If at min workers, increase delay (0.2→0.3→0.5→1.0s)
    4. Track success rate and can gradually increase if stable
    
    Thread Safety:
        All state modifications are protected by a threading.Lock to ensure
        safe concurrent access from multiple threads.
    """
    
    def __init__(self, initial_workers: int = 5, initial_delay: float = 0.2):
        """
        Initialize dynamic rate limiter.
        
        Args:
            initial_workers: Starting number of parallel workers (default: 5)
            initial_delay: Starting delay between API calls in seconds (default: 0.2)
            
        Raises:
            ValueError: If parameters are out of valid range
            TypeError: If parameters are not correct type
        """
        # Validate inputs
        if not isinstance(initial_workers, int):
            raise TypeError(f"initial_workers must be int, got {type(initial_workers).__name__}")
        if not isinstance(initial_delay, (int, float)):
            raise TypeError(f"initial_delay must be numeric, got {type(initial_delay).__name__}")
        if initial_workers < 1:
            raise ValueError(f"initial_workers must be >= 1, got {initial_workers}")
        if initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {initial_delay}")
        
        # Thread safety lock
        self._lock = threading.Lock()
        
        self.workers = initial_workers
        self.delay = initial_delay
        self.min_workers = 2
        self.max_workers = 10 if IS_AWS else 5
        self.min_delay = 0.2
        self.max_delay = 2.0
        
        # Track rate limit hits
        self.consecutive_hits = 0
        self.total_requests = 0
        self.rate_limit_hits = 0
        self.last_adjustment = datetime.now()
        self.stable_duration_minutes = 10  # Minutes without hits before increasing
        
        # Delay progression when at minimum workers
        self.delay_steps = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        self.current_delay_index = 0
        
        print(f"🚀 Rate Limiter initialized: {self.workers} workers, {self.delay}s delay")
    
    def hit_rate_limit(self):
        """
        Called when a rate limit (429) error is encountered.
        Adjusts workers and delay accordingly.
        
        Thread-safe operation.
        """
        with self._lock:
            self.consecutive_hits += 1
            self.rate_limit_hits += 1
            self.last_adjustment = datetime.now()
            
            # First try reducing workers
            if self.workers > self.min_workers:
                self.workers -= 1
                print(f"⚠️  Rate limit hit! Reducing workers: {self.workers + 1} → {self.workers}")
                self.consecutive_hits = 0  # Reset counter after adjustment
                
            # If at minimum workers, increase delay
            else:
                if self.current_delay_index < len(self.delay_steps) - 1:
                    self.current_delay_index += 1
                    old_delay = self.delay
                    self.delay = self.delay_steps[self.current_delay_index]
                    print(f"⚠️  Rate limit hit! Increasing delay: {old_delay}s → {self.delay}s")
                    self.consecutive_hits = 0
                else:
                    print(f"⚠️  At maximum throttling: {self.workers} workers, {self.delay}s delay")
                    print(f"   Consider upgrading API tier or reducing request volume")
    
    def success(self):
        """
        Called on successful API request.
        May gradually increase capacity if stable.
        
        Thread-safe operation.
        """
        with self._lock:
            self.total_requests += 1
            self.consecutive_hits = 0
            
            # Check if we can increase capacity (after stable period)
            minutes_since_adjustment = (datetime.now() - self.last_adjustment).total_seconds() / 60
            
            if minutes_since_adjustment > self.stable_duration_minutes:
                success_rate = 1 - (self.rate_limit_hits / max(self.total_requests, 1))
                
                # If very stable (>99% success), try increasing capacity
                if success_rate > 0.99:
                    if self.delay > self.min_delay and self.current_delay_index > 0:
                        # First reduce delay
                        self.current_delay_index -= 1
                        old_delay = self.delay
                        self.delay = self.delay_steps[self.current_delay_index]
                        print(f"✅ Stable performance! Reducing delay: {old_delay}s → {self.delay}s")
                        self.last_adjustment = datetime.now()
                        
                    elif self.workers < self.max_workers:
                        # Then increase workers
                        self.workers += 1
                        print(f"✅ Stable performance! Increasing workers: {self.workers - 1} → {self.workers}")
                        self.last_adjustment = datetime.now()
    
    def get_settings(self) -> Dict:
        """
        Get current rate limiter settings.
        
        Thread-safe operation.
        
        Returns:
            Dictionary with current workers and delay settings
        """
        with self._lock:
            return {
                'workers': self.workers,
                'delay': self.delay,
                'success_rate': 1 - (self.rate_limit_hits / max(self.total_requests, 1)),
                'total_requests': self.total_requests,
                'rate_limit_hits': self.rate_limit_hits
            }
    
    def reset_stats(self) -> None:
        """
        Reset statistics while keeping current settings.
        
        Thread-safe operation.
        """
        with self._lock:
            self.total_requests = 0
            self.rate_limit_hits = 0
            self.consecutive_hits = 0
            print(f"📊 Stats reset. Current: {self.workers} workers, {self.delay}s delay")


# Global rate limiter instance (singleton)
_rate_limiter = None

def get_rate_limiter() -> DynamicRateLimiter:
    """
    Get or create the global rate limiter instance.
    
    Returns:
        DynamicRateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        # Initialize based on environment
        initial_workers = 5 if not IS_AWS else 10
        _rate_limiter = DynamicRateLimiter(initial_workers=initial_workers)
    return _rate_limiter


# =============================================================================
# MACOS SLEEP PREVENTION
# =============================================================================

class KeepAwake:
    """
    Context manager to prevent macOS from sleeping during long operations.

    Uses the native 'caffeinate' command which prevents:
    - Display sleep
    - System sleep
    - Disk sleep

    Automatically restores normal sleep behavior when done or on error.
    Only active on macOS - no-op on other platforms.

    Usage:
        with KeepAwake():
            # Your long-running code here
            pipeline.run_full_pipeline()
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize KeepAwake context manager.

        Args:
            verbose: Print status messages (default: True)
        """
        self.verbose = verbose
        self.process = None
        self.is_mac = sys.platform == 'darwin'

    def __enter__(self):
        """Start preventing sleep when entering context."""
        if not self.is_mac:
            if self.verbose:
                print("ℹ️  Sleep prevention only available on macOS")
            return self

        try:
            # Start caffeinate process
            # -d: Prevent display sleep
            # -i: Prevent system idle sleep
            # -m: Prevent disk sleep
            # -s: Prevent system sleep when on AC power
            self.process = subprocess.Popen(
                ['caffeinate', '-dims'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if self.verbose:
                print("☕ caffeinate activated - Mac will stay awake during execution")

            # Register cleanup in case of unexpected termination
            atexit.register(self._cleanup)

        except FileNotFoundError:
            if self.verbose:
                print("⚠️  caffeinate command not found - sleep prevention unavailable")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not activate sleep prevention: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop preventing sleep when exiting context."""
        self._cleanup()
        return False  # Don't suppress exceptions

    def _cleanup(self):
        """Terminate caffeinate process and restore normal sleep behavior."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                if self.verbose:
                    print("☕ caffeinate deactivated - Mac sleep settings restored")
            except Exception:
                # Force kill if terminate doesn't work
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
                # Unregister from atexit to avoid double cleanup
                try:
                    atexit.unregister(self._cleanup)
                except Exception:
                    pass


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string like "2h 15m 30s" or "45.2s"
        
    Examples:
        >>> format_time(7890)
        '2h 11m 30s'
        >>> format_time(45.234)
        '45.2s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def print_banner(text: str, width: int = 60, char: str = "=") -> None:
    """
    Print a formatted banner for section headers.
    
    Args:
        text: Banner text
        width: Total width of banner (default: 60)
        char: Character to use for lines (default: "=")
        
    Example:
        >>> print_banner("TRAINING")
        ============================================================
        TRAINING
        ============================================================
    """
    print(char * width)
    print(text)
    print(char * width)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if denominator is zero (default: 0.0)
        
    Returns:
        Result of division or default value
        
    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def get_model_display_name(model_key: str) -> str:
    """
    Convert model key to display-friendly name.
    
    Args:
        model_key: Model key from API_CONFIG (e.g., 'claude', 'gpt5')
        
    Returns:
        Display-friendly name
        
    Examples:
        >>> get_model_display_name('claude')
        'Claude Sonnet 4.5'
        >>> get_model_display_name('gpt5')
        'GPT-5'
    """
    display_names = {
        'claude': 'Claude Sonnet 4.5',
        'gpt5': 'GPT-5.1', 
        'gemini': 'Gemini 3 Pro',
        'wildjailbreak': 'WildJailbreak (Synthetic)'
    }
    return display_names.get(model_key, model_key.title())


def ensure_dir_exists(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to check/create
    """
    os.makedirs(path, exist_ok=True)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count approximate tokens in text for a given model.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: "gpt-4o")
        
    Returns:
        Approximate token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate of 1 token per 4 characters
        return len(text) // 4


def get_dynamic_workers() -> int:
    """
    Get the current number of parallel workers from the rate limiter.
    
    Returns:
        Current number of workers to use for parallel processing
    """
    rate_limiter = get_rate_limiter()
    return rate_limiter.workers


def get_dynamic_delay() -> float:
    """
    Get the current API delay from the rate limiter.

    Returns:
        Current delay in seconds between API calls
    """
    rate_limiter = get_rate_limiter()
    return rate_limiter.delay


def convert_to_serializable(obj):
    """
    Recursively convert numpy/pandas types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, pandas DataFrame, etc.)

    Returns:
        Object with all numpy/pandas types converted to native Python types

    Examples:
        >>> convert_to_serializable({'a': np.int64(5), 'b': np.bool_(True)})
        {'a': 5, 'b': True}
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')  # Convert DataFrame to list of dicts
    elif isinstance(obj, pd.Series):
        return obj.to_list()  # Convert Series to list
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):  # Handle pandas NA/NaN
        return None
    return obj


def get_interpretability_config(total_prompts: int = None, test_split: float = None) -> dict:
    """
    Dynamically calculate optimal interpretability settings based on dataset size.
    
    Automatically selects appropriate settings tier to balance:
    - Statistical rigor (larger samples for publication-ready work)
    - Computational efficiency (smaller samples for quick testing)
    - Industry standards (meets academic requirements automatically)
    
    Tier Selection:
        - MINIMAL: Very small datasets (<30 test samples) - exploratory only
        - EXPLORATORY: Small datasets (30-100 test samples) - quick testing
        - BALANCED: Medium datasets (100-200 test samples) - serious work
        - PUBLICATION: Large datasets (200+ test samples) - publication-ready
    
    Args:
        total_prompts: Total prompts in experiment (defaults to 2000)
        test_split: Test set fraction (defaults to 0.15)
    
    Returns:
        Dictionary with optimal interpretability settings
    
    Examples:
        >>> # Test mode: 50 prompts → 7 test samples → MINIMAL tier
        >>> config = get_interpretability_config(50)
        >>> config['shap_samples']
        10
        
        >>> # Full mode: 2000 prompts → 300 test samples → PUBLICATION tier
        >>> config = get_interpretability_config(2000)
        >>> config['shap_samples']
        50
    
    Academic References:
        - Ribeiro et al. 2016 (LIME): 20-50 examples
        - Lundberg et al. 2020 (SHAP): 100 examples for robust analysis
        - Jain & Wallace 2019 (Attention): 50-100 examples
    """
    # Use sensible defaults if not provided
    # Config file will pass actual values from DATASET_CONFIG when calling
    if total_prompts is None:
        total_prompts = 2000  # Standard full experiment size
    if test_split is None:
        test_split = 0.15  # Standard test split
    
    # Calculate expected test set size
    test_set_size = int(total_prompts * test_split)
    
    # Define scaling rules based on dataset size
    # Thresholds derived from academic standards (Ribeiro 2016, Lundberg 2020)
    if test_set_size < 30:
        # MINIMAL: Very small test sets - exploratory only
        config_tier = 'minimal'
        shap_samples = max(10, min(15, test_set_size - 5))
        attention_per_class = 3
        shap_background = 30
        shap_visualize_per_class = 1
        
    elif test_set_size < 100:
        # EXPLORATORY: Small test sets - quick testing, not publication
        config_tier = 'exploratory'
        shap_samples = max(15, min(20, test_set_size - 10))
        attention_per_class = 5
        shap_background = 50
        shap_visualize_per_class = 2
        
    elif test_set_size < 200:
        # BALANCED: Medium test sets - serious work, approaching publication quality
        config_tier = 'balanced'
        shap_samples = 30
        attention_per_class = 10
        shap_background = 75
        shap_visualize_per_class = 3
        
    else:
        # PUBLICATION: Large test sets - publication-ready, meets academic standards
        config_tier = 'publication'
        shap_samples = 50
        attention_per_class = 20
        shap_background = 100
        shap_visualize_per_class = 3
    
    # Safety: Ensure we don't request more samples than available
    max_safe_samples = max(10, test_set_size - 10)  # Reserve buffer
    shap_samples = min(shap_samples, max_safe_samples)
    
    return {
        # Metadata (for debugging/logging)
        '_config_tier': config_tier,
        '_test_set_size': test_set_size,
        '_total_prompts': total_prompts,
        
        # Attention Visualization
        'attention_samples_per_class': attention_per_class,
        'attention_layer_index': -1,
        'attention_top_k_tokens': 15,
        'visualize_all_layers': False,
        
        # SHAP Analysis
        'shap_enabled': True,
        'shap_samples': shap_samples,
        'shap_samples_per_class': shap_visualize_per_class,
        'shap_background_samples': shap_background,
        'shap_max_display': 20,
        
        # Statistical Agreement Thresholds (Cohen's Kappa, etc.)
        'kappa_thresholds': {
            'almost_perfect': 0.80,
            'substantial': 0.60,
            'moderate': 0.40,
            'fair': 0.20,
            'slight': 0.0
        },
        
        # Power Law Analysis Thresholds
        'power_law_exponent_range': (1.0, 3.0),
        'zipf_exponent_range': (0.8, 1.5),
        
        # Calibration Thresholds (Expected Calibration Error)
        'ece_thresholds': {
            'excellent': 0.05,
            'good': 0.10,
            'acceptable': 0.15
        },
        
        # General Interpretability
        'min_confidence_threshold': 0.5
    }


def get_timestamped_filename(base_name: str, prefix: str = None, timestamp: str = None) -> str:
    """
    Generate standardized filename with optional prefix and timestamp.
    
    Args:
        base_name: Base filename (e.g., "confusion_matrix.png")
        prefix: Optional prefix (e.g., "refusal", "jailbreak", "correlation")
        timestamp: Optional timestamp (e.g., "20251116_2348")
    
    Returns:
        Formatted filename (e.g., "refusal_confusion_matrix_20251116_2348.png")
    
    Examples:
        >>> get_timestamped_filename("heatmap.png", "correlation", "20251116_2348")
        'correlation_heatmap_20251116_2348.png'
    """
    name, ext = os.path.splitext(base_name)
    
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(name)
    if timestamp:
        parts.append(timestamp)
    
    return "_".join(parts) + ext


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: Utility Functions and Classes
Created on October 28, 2025
@author: ramyalsaffar
"""
