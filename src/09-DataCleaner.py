# DataCleaner Module 
#--------------------
# Comprehensive data quality management for text classification datasets.
# Detects and cleans duplicates, outliers, invalid data, and label inconsistencies.
# 
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Checkpoint support for large datasets
# - Better duplicate detection
# - Comprehensive statistics
# - Handles new is_jailbreak_attempt field
# All imports are in 01-Imports.py
###############################################################################


class DataCleaner:
    """
    Comprehensive data cleaning and quality validation for refusal classification.
    
    Handles:
    - Duplicate detection (exact and near-duplicates)
    - Empty/null value validation
    - Length outliers (too short/long responses)
    - Invalid character detection
    - Label consistency checks
    - Response quality indicators
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager = None):
        """
        Initialize DataCleaner.
        
        Args:
            checkpoint_manager: Optional checkpoint manager for large datasets
        """
        # Use config values - NO HARDCODING!
        self.verbose = EXPERIMENT_CONFIG.get('verbose', True)
        self.cleaning_log = []
        
        # Quality thresholds from config
        self.min_response_length = DATA_CLEANING_CONFIG['min_response_length']
        self.max_response_length = DATA_CLEANING_CONFIG['max_response_length']
        self.min_prompt_length = DATA_CLEANING_CONFIG['min_prompt_length']
        self.max_prompt_length = DATA_CLEANING_CONFIG['max_prompt_length']
        self.similarity_threshold = DATA_CLEANING_CONFIG['similarity_threshold']
        
        # Checkpoint manager for large datasets
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            operation_name='data_cleaning'
        )
        
        # Error indicators
        self.error_patterns = [
            r'\[ERROR\]',  # Escaped brackets for literal "[ERROR]" match
            'Failed to generate',
            r'^<!DOCTYPE html>',  # HTML error pages (must be at start)
            r'^<html>',  # HTML error pages (must be at start)
            'An error occurred',
            'Rate limit',
            'API error'
        ]
        
        # Statistics tracking
        self.stats = {
            'original_size': 0,
            'final_size': 0,
            'total_removed': 0,
            'null_removed': 0,
            'duplicate_removed': 0,
            'length_outlier_removed': 0,
            'error_response_removed': 0,
            'invalid_label_removed': 0,
            'processing_time': 0
        }

    def clean_dataset(self, df: pd.DataFrame, strategy: str = 'auto', 
                     use_checkpoint: bool = False) -> pd.DataFrame:
        """
        Main cleaning pipeline - runs all cleaning steps.
        
        Args:
            df: DataFrame with columns: prompt, response, model, etc.
            strategy: Cleaning strategy ('auto', 'conservative', 'aggressive', 'none')
            use_checkpoint: Whether to save checkpoints during cleaning
        
        Returns:
            Cleaned DataFrame with quality report
        """
        start_time = time.time()
        
        if self.verbose:
            print_banner("DATA CLEANING PIPELINE", char="=")
            print(f"  Strategy: {strategy}")
            print(f"  Initial dataset size: {len(df):,} samples")
            print("=" * 60)
        
        self.stats['original_size'] = len(df)
        df_clean = df
        
        # Step 1: Validate basic data integrity
        if self.verbose:
            print("\n🔍 Step 1: Validating data integrity...")
        df_clean = self._validate_data_integrity(df_clean)
        
        if use_checkpoint:
            self.checkpoint_manager.save_checkpoint(
                data=df_clean,
                last_index=1,
                metadata={'step': 'integrity_validation', 'stats': self.stats}
            )
        
        # Step 2: Remove exact duplicates
        if self.verbose:
            print("\n🔍 Step 2: Removing duplicates...")
        df_clean = self._remove_duplicates(df_clean, strategy)
        
        if use_checkpoint:
            self.checkpoint_manager.save_checkpoint(
                data=df_clean,
                last_index=2,
                metadata={'step': 'duplicate_removal', 'stats': self.stats}
            )
        
        # Step 3: Filter length outliers
        if self.verbose:
            print("\n🔍 Step 3: Filtering length outliers...")
        df_clean = self._filter_length_outliers(df_clean, strategy)
        
        if use_checkpoint:
            self.checkpoint_manager.save_checkpoint(
                data=df_clean,
                last_index=3,
                metadata={'step': 'length_filtering', 'stats': self.stats}
            )
        
        # Step 4: Detect and handle error responses
        if self.verbose:
            print("\n🔍 Step 4: Handling error responses...")
        df_clean = self._handle_error_responses(df_clean)
        
        # Step 5: Validate label consistency (if labeled)
        if 'refusal_label' in df_clean.columns:
            if self.verbose:
                print("\n🔍 Step 5: Validating label consistency...")
            df_clean = self._validate_label_consistency(df_clean, strategy)
        
        # Final statistics
        self.stats['final_size'] = len(df_clean)
        self.stats['total_removed'] = self.stats['original_size'] - self.stats['final_size']
        self.stats['processing_time'] = time.time() - start_time
        
        # Step 6: Quality assessment
        quality_report = self._assess_quality(df_clean)
        
        # Final checkpoint - ALWAYS save at completion (force=True)
        if use_checkpoint:
            self.checkpoint_manager.save_checkpoint(
                data=df_clean,
                last_index=len(df_clean),
                metadata={
                    'step': 'final_clean',
                    'stats': self.stats,
                    'quality_report': quality_report
                },
                force=True  # Force final checkpoint regardless of interval
            )
            print(f"\n💾 Final checkpoint saved: {len(df_clean):,} cleaned samples")
        
        if self.verbose:
            self._print_cleaning_report(quality_report)
        
        return df_clean

    def _validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for null values and required columns."""
        # Base columns that must ALWAYS exist
        required_cols = ['prompt', 'response', 'model']

        # Optional columns for labeled data
        optional_cols = ['refusal_label', 'is_jailbreak_attempt', 'jailbreak_label',
                        'expected_label', 'refusal_confidence', 'jailbreak_confidence']
        
        existing_optional = [col for col in optional_cols if col in df.columns]
        check_cols = required_cols + existing_optional
        
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Count nulls
        null_counts = df[check_cols].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            if self.verbose:
                print(f"  ⚠️  Found {total_nulls:,} null values:")
                for col, count in null_counts[null_counts > 0].items():
                    print(f"     {col}: {count:,} nulls")
            
            # Remove rows with nulls in critical columns
            before = len(df)
            df = df.dropna(subset=check_cols)
            removed = before - len(df)
            
            self.stats['null_removed'] = removed
            
            if removed > 0:
                self.cleaning_log.append({
                    'step': 'null_removal',
                    'removed': removed,
                    'reason': 'Null values in required columns'
                })
                if self.verbose:
                    print(f"  ✓ Removed {removed:,} rows with null values")
        
        return df

    def _remove_duplicates(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Remove exact and near-duplicate entries."""
        before = len(df)
        
        # Exact duplicates (prompt + response)
        df = df.drop_duplicates(subset=['prompt', 'response'], keep='first')
        exact_removed = before - len(df)
        
        self.stats['duplicate_removed'] += exact_removed
        
        if exact_removed > 0:
            self.cleaning_log.append({
                'step': 'exact_duplicates',
                'removed': exact_removed,
                'reason': 'Identical prompt-response pairs'
            })
            if self.verbose:
                print(f"  ✓ Removed {exact_removed:,} exact duplicates")
        
        # Near-duplicates (same prompt, very similar response)
        if strategy in ['aggressive', 'auto']:
            before_near = len(df)
            df = self._remove_near_duplicates(df)
            near_removed = before_near - len(df)
            
            self.stats['duplicate_removed'] += near_removed
            
            if near_removed > 0:
                self.cleaning_log.append({
                    'step': 'near_duplicates',
                    'removed': near_removed,
                    'reason': 'Very similar responses to same prompt'
                })
                if self.verbose:
                    print(f"  ✓ Removed {near_removed:,} near-duplicates")
        
        return df

    def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove responses that are too similar for the same prompt using MinHash LSH.
        
        Fast O(n) algorithm using Locality-Sensitive Hashing instead of O(n²) pairwise comparison.
        Falls back to slower method if datasketch not installed.
        """
        
        
        drop_indices = []
        
        # Group by prompt for efficiency
        prompt_groups = df.groupby('prompt')
        
        with tqdm(total=len(prompt_groups), desc="  Checking near-duplicates (LSH)", 
                  disable=not self.verbose) as pbar:
            for prompt, group in prompt_groups:
                pbar.update(1)
                
                if len(group) <= 1:
                    continue
                
                # Create LSH index for this prompt group
                lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
                
                responses = group['response'].tolist()
                indices = group.index.tolist()
                
                # Process each response
                for idx, response in enumerate(responses):
                    original_idx = indices[idx]
                    
                    # Skip if already marked for removal
                    if original_idx in drop_indices:
                        continue
                    
                    # Create MinHash signature
                    m = MinHash(num_perm=128)
                    words = response.lower().split()
                    
                    # Skip empty responses
                    if not words:
                        continue
                    
                    for word in words:
                        m.update(word.encode('utf-8'))
                    
                    # Query LSH for similar documents
                    result = lsh.query(m)
                    
                    if result:
                        # Found similar document - mark for removal (keep first occurrence)
                        drop_indices.append(original_idx)
                    else:
                        # No similar document - add to index
                        lsh.insert(str(original_idx), m)
        
        if drop_indices:
            df = df.drop(index=drop_indices)
        
        return df


    def _remove_near_duplicates_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback O(n²) implementation when datasketch is not available.
        This is the original implementation for compatibility.
        """
        drop_indices = []
        
        # Group by prompt for efficiency
        prompt_groups = df.groupby('prompt')
        
        with tqdm(total=len(prompt_groups), desc="  Checking near-duplicates", 
                  disable=not self.verbose) as pbar:
            for prompt, group in prompt_groups:
                pbar.update(1)
                
                if len(group) <= 1:
                    continue
                
                responses = group['response'].tolist()
                indices = group.index.tolist()
                
                # Check pairwise similarity
                for i in range(len(responses)):
                    if indices[i] in drop_indices:
                        continue
                    for j in range(i + 1, len(responses)):
                        if indices[j] in drop_indices:
                            continue
                        
                        # Calculate similarity (Jaccard on words)
                        words_i = set(responses[i].lower().split())
                        words_j = set(responses[j].lower().split())
                        
                        if len(words_i | words_j) == 0:
                            continue
                        
                        jaccard = safe_divide(len(words_i & words_j), len(words_i | words_j), 0)
                        
                        # If very similar, keep first occurrence
                        if jaccard > self.similarity_threshold:
                            drop_indices.append(indices[j])
        
        if drop_indices:
            df = df.drop(index=drop_indices)
        
        return df
    

    def _filter_length_outliers(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Filter responses/prompts that are too short or too long."""
        before = len(df)
        
        # Calculate lengths
        df['response_len'] = df['response'].str.len()
        df['prompt_len'] = df['prompt'].str.len()
        
        # Calculate statistics
        response_stats = df['response_len'].describe()
        prompt_stats = df['prompt_len'].describe()
        
        if self.verbose:
            print(f"  📊 Length Statistics:")
            print(f"     Response: mean={response_stats['mean']:.0f}, "
                  f"median={response_stats['50%']:.0f}, "
                  f"std={response_stats['std']:.0f}")
            print(f"     Prompt: mean={prompt_stats['mean']:.0f}, "
                  f"median={prompt_stats['50%']:.0f}")
        
        # Determine bounds based on strategy
        if strategy == 'conservative':
            # Only remove extreme outliers
            q1, q3 = response_stats['25%'], response_stats['75%']
            iqr = q3 - q1
            lower_bound = max(self.min_response_length, q1 - 3 * iqr)
            upper_bound = min(self.max_response_length, q3 + 3 * iqr)
        elif strategy == 'aggressive':
            # Tighter bounds
            q1, q3 = response_stats['25%'], response_stats['75%']
            iqr = q3 - q1
            lower_bound = max(self.min_response_length, q1 - 1.5 * iqr)
            upper_bound = min(self.max_response_length, q3 + 1.5 * iqr)
        else:  # auto or none
            # Use config thresholds
            lower_bound = self.min_response_length
            upper_bound = self.max_response_length
        
        # Apply filters
        mask = (
            (df['response_len'] >= lower_bound) &
            (df['response_len'] <= upper_bound) &
            (df['prompt_len'] >= self.min_prompt_length) &
            (df['prompt_len'] <= self.max_prompt_length)
        )
        df = df[mask]
        
        # Clean up temporary columns
        df = df.drop(columns=['response_len', 'prompt_len'])
        
        removed = before - len(df)
        self.stats['length_outlier_removed'] = removed
        
        if removed > 0:
            self.cleaning_log.append({
                'step': 'length_outliers',
                'removed': removed,
                'reason': f'Length outside bounds ({lower_bound:.0f}-{upper_bound:.0f})'
            })
            if self.verbose:
                print(f"  ✓ Removed {removed:,} length outliers")
        
        return df

    def _handle_error_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove error responses from API failures."""
        before = len(df)
        
        # Check for error patterns
        error_mask = df['response'].str.contains('|'.join(self.error_patterns), 
                                                case=False, na=False, regex=True)
        error_count = error_mask.sum()
        
        if error_count > 0:
            if self.verbose:
                print(f"  ⚠️  Found {error_count:,} error responses")
                # Show examples
                error_samples = df[error_mask]['response'].head(3)
                for i, sample in enumerate(error_samples, 1):
                    print(f"     Example {i}: {sample[:100]}...")
            
            df = df[~error_mask]
            self.stats['error_response_removed'] = error_count
            
            self.cleaning_log.append({
                'step': 'error_responses',
                'removed': error_count,
                'reason': 'API error or failed generation'
            })
            
            if self.verbose:
                print(f"  ✓ Removed {error_count:,} error responses")
        
        return df

    def _validate_label_consistency(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Validate and clean label values."""
        before = len(df)
        invalid_removed = 0
        
        # Check refusal labels (0, 1, 2, -1)
        if 'refusal_label' in df.columns:
            valid_refusal = df['refusal_label'].isin([0, 1, 2, -1])
            invalid_refusal = (~valid_refusal).sum()
            
            if invalid_refusal > 0:
                df = df[valid_refusal]
                invalid_removed += invalid_refusal
                if self.verbose:
                    print(f"  ✓ Removed {invalid_refusal:,} invalid refusal labels")
        
        # Check jailbreak attempt labels (0, 1)
        if 'is_jailbreak_attempt' in df.columns:
            valid_attempt = df['is_jailbreak_attempt'].isin([0, 1])
            invalid_attempt = (~valid_attempt).sum()
            
            if invalid_attempt > 0:
                df = df[valid_attempt]
                invalid_removed += invalid_attempt
                if self.verbose:
                    print(f"  ✓ Removed {invalid_attempt:,} invalid jailbreak attempt labels")
        
        # Check jailbreak labels (0, 1, -1)
        if 'jailbreak_label' in df.columns:
            valid_jailbreak = df['jailbreak_label'].isin([0, 1, -1])
            invalid_jailbreak = (~valid_jailbreak).sum()

            if invalid_jailbreak > 0:
                df = df[valid_jailbreak]
                invalid_removed += invalid_jailbreak
                if self.verbose:
                    print(f"  ✓ Removed {invalid_jailbreak:,} invalid jailbreak labels")
        
        self.stats['invalid_label_removed'] = invalid_removed
        
        if invalid_removed > 0:
            self.cleaning_log.append({
                'step': 'invalid_labels',
                'removed': invalid_removed,
                'reason': 'Invalid label values'
            })
        
        # Check label distribution
        if 'refusal_label' in df.columns and self.verbose:
            self._print_label_distribution(df)
        
        return df

    def _print_label_distribution(self, df: pd.DataFrame):
        """Print label distribution statistics."""
        print(f"\n  📊 Label Distribution:")
        
        # Refusal labels
        if 'refusal_label' in df.columns:
            refusal_dist = df['refusal_label'].value_counts()
            print(f"     Refusal Classification:")
            for label in [0, 1, 2, -1]:
                if label in refusal_dist.index:
                    count = refusal_dist[label]
                    pct = safe_divide(count, len(df), 0) * 100
                    name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else 'Error'
                    print(f"       {name}: {count:,} ({pct:.1f}%)")
        
        # Jailbreak statistics
        if 'is_jailbreak_attempt' in df.columns:
            attempt_count = df['is_jailbreak_attempt'].sum()
            attempt_pct = safe_divide(attempt_count, len(df), 0) * 100
            print(f"     Jailbreak Attempts: {attempt_count:,} ({attempt_pct:.1f}%)")

            if 'jailbreak_label' in df.columns and attempt_count > 0:
                # Only count successes among attempts
                attempts_df = df[df['is_jailbreak_attempt'] == 1]
                success_count = (attempts_df['jailbreak_label'] == 1).sum()
                success_rate = safe_divide(success_count, attempt_count, 0) * 100
                print(f"       Success Rate: {success_count:,}/{attempt_count:,} ({success_rate:.1f}%)")

    def _assess_quality(self, df: pd.DataFrame) -> Dict:
        """Assess overall data quality after cleaning."""
        removal_pct = safe_divide(self.stats['total_removed'], 
                                 self.stats['original_size'], 0) * 100
        
        # Quality score calculation
        if removal_pct < 2:
            quality = "Excellent"
        elif removal_pct < 5:
            quality = "Good"
        elif removal_pct < 10:
            quality = "Acceptable"
        else:
            quality = "Concerning"
        
        return {
            'original_size': self.stats['original_size'],
            'cleaned_size': self.stats['final_size'],
            'removed_total': self.stats['total_removed'],
            'removal_percentage': removal_pct,
            'quality_rating': quality,
            'cleaning_log': self.cleaning_log,
            'processing_time': self.stats['processing_time']
        }

    def _print_cleaning_report(self, report: Dict):
        """Print comprehensive cleaning report using Utils functions."""
        print_banner("DATA CLEANING REPORT", char="=")
        
        print(f"Original dataset: {report['original_size']:,} samples")
        print(f"Cleaned dataset:  {report['cleaned_size']:,} samples")
        print(f"Removed:          {report['removed_total']:,} samples ({report['removal_percentage']:.2f}%)")
        print(f"Quality Rating:   {report['quality_rating']}")
        print(f"Processing Time:  {format_time(report['processing_time'])}")
        
        if report['cleaning_log']:
            print(f"\nCleaning Steps:")
            for entry in report['cleaning_log']:
                print(f"   • {entry['step']}: {entry['removed']:,} samples")
                print(f"     Reason: {entry['reason']}")
        
        print("\nDetailed Statistics:")
        print(f"   Nulls removed:         {self.stats['null_removed']:,}")
        print(f"   Duplicates removed:    {self.stats['duplicate_removed']:,}")
        print(f"   Length outliers:       {self.stats['length_outlier_removed']:,}")
        print(f"   Error responses:       {self.stats['error_response_removed']:,}")
        print(f"   Invalid labels:        {self.stats['invalid_label_removed']:,}")
        
        print("=" * 60)

    def get_outlier_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate outlier analysis report without modifying data.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with outlier statistics and recommendations
        """
        report = {
            'total_samples': len(df),
            'issues_found': [],
            'timestamp': get_timestamp('display')
        }
        
        # Check for nulls
        null_check_cols = ['prompt', 'response']
        if 'refusal_label' in df.columns:
            null_check_cols.extend(['refusal_label', 'is_jailbreak_attempt', 'jailbreak_label'])
        
        null_counts = df[null_check_cols].isnull().sum()
        if null_counts.sum() > 0:
            report['issues_found'].append({
                'type': 'null_values',
                'count': int(null_counts.sum()),
                'percentage': safe_divide(null_counts.sum(), len(df), 0) * 100
            })
        
        # Check for duplicates
        duplicate_count = df.duplicated(subset=['prompt', 'response']).sum()
        if duplicate_count > 0:
            report['issues_found'].append({
                'type': 'exact_duplicates',
                'count': int(duplicate_count),
                'percentage': safe_divide(duplicate_count, len(df), 0) * 100
            })
        
        # Check length outliers
        df_temp = df.copy()
        df_temp['response_len'] = df_temp['response'].str.len()
        q1 = df_temp['response_len'].quantile(0.25)
        q3 = df_temp['response_len'].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df_temp['response_len'] < q1 - 3 * iqr) |
                   (df_temp['response_len'] > q3 + 3 * iqr)).sum()
        
        if outliers > 0:
            report['issues_found'].append({
                'type': 'length_outliers',
                'count': int(outliers),
                'percentage': safe_divide(outliers, len(df), 0) * 100
            })
        
        # Recommendation based on total issues
        total_issues = sum(issue['count'] for issue in report['issues_found'])
        issue_pct = safe_divide(total_issues, len(df), 0) * 100
        
        if issue_pct < 2.5:
            report['recommendation'] = 'conservative'
            report['reasoning'] = 'Very few issues - use conservative cleaning'
        elif issue_pct < 10:
            report['recommendation'] = 'auto'
            report['reasoning'] = 'Moderate issues - use auto cleaning'
        else:
            report['recommendation'] = 'aggressive'
            report['reasoning'] = 'Many issues detected - aggressive cleaning recommended'
        
        return report


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
