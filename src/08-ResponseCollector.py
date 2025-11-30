# ResponseCollector Module
#-------------------------
# Collects responses from multiple LLM APIs (Claude, GPT-5, Gemini).
# Features:
# - Full Config/Utils integration (NO HARDCODING!)
# - Dynamic rate limiting per model
# - Enhanced parallel processing with model-specific worker pools
# - Checkpoint support with rate limiter state preservation
# - Comprehensive statistics tracking
# All imports are in 01-Imports.py
###############################################################################


class ResponseCollector:
    """Collect responses from multiple LLM models with enhanced parallel processing."""

    def __init__(self, claude_key: str, openai_key: str, google_key: None):
        """
        Initialize response collector.

        Args:
            claude_key: Anthropic API key
            openai_key: OpenAI API key
            google_key: Google API key
        """
        self.anthropic_client = Anthropic(api_key=claude_key)
        self.openai_client = OpenAI(api_key=openai_key)
        
        #genai.configure(api_key=google_key)
        #self.gemini_model = genai.GenerativeModel(API_CONFIG['response_models']['gemini'])
        
        if google_key and 'gemini' in API_CONFIG['response_models']:
            genai.configure(api_key=google_key)
            self.gemini_model = genai.GenerativeModel(API_CONFIG['response_models']['gemini'])
        else:
            self.gemini_model = None
        
        # Use config values - NO HARDCODING!
        self.models = list(API_CONFIG['response_models'].values())
        self.model_keys = list(API_CONFIG['response_models'].keys())  # For display names
        self.max_completion_tokens = API_CONFIG['max_tokens_response']
        self.max_retries = API_CONFIG['max_retries']
        self.rate_limit_backoff = API_CONFIG['rate_limit_backoff']
        
        self.model_rate_limiters = {}
        for model_key in self.model_keys:
            if model_key == 'claude':
                self.model_rate_limiters[model_key] = DynamicRateLimiter(
                    initial_workers=API_CONFIG.get('claude_workers', 3),
                    initial_delay=API_CONFIG.get('claude_delay', 0.2)
                )
            elif model_key == 'gpt5':
                self.model_rate_limiters[model_key] = DynamicRateLimiter(
                    initial_workers=API_CONFIG.get('gpt5_workers', 5),
                    initial_delay=API_CONFIG.get('gpt5_delay', 0.1)
                )
            elif model_key == 'gemini':
                self.model_rate_limiters[model_key] = DynamicRateLimiter(
                    initial_workers=API_CONFIG.get('gemini_workers', 10),
                    initial_delay=API_CONFIG.get('gemini_delay', 0.05)
                )
        
# =============================================================================
#         # Model-specific rate limiters (each API has different limits!)
#         self.model_rate_limiters = {
#             'claude': DynamicRateLimiter(
#                 initial_workers=API_CONFIG.get('claude_workers', 3),
#                 initial_delay=API_CONFIG.get('claude_delay', 0.2)
#             ),
#             'gpt5': DynamicRateLimiter(
#                 initial_workers=API_CONFIG.get('gpt5_workers', 5),
#                 initial_delay=API_CONFIG.get('gpt5_delay', 0.1)
#             ),
#             'gemini': DynamicRateLimiter(
#                 initial_workers=API_CONFIG.get('gemini_workers', 10),
#                 initial_delay=API_CONFIG.get('gemini_delay', 0.05)
#             )
#         }
# =============================================================================
        
        # Token counter for API usage tracking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Statistics tracking
        self.stats = {
            'total_api_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rate_limit_hits': 0,
            'total_tokens': 0,
            'per_model_success': {key: 0 for key in self.model_keys},
            'per_model_errors': {key: 0 for key in self.model_keys},
            'per_model_tokens': {key: 0 for key in self.model_keys},
            'per_model_rate_limits': {key: 0 for key in self.model_keys}
        }

    def collect_all_responses(self, prompts: Dict[str, List[str]], 
                             parallel: bool = True,
                             resume_from_checkpoint: bool = True) -> pd.DataFrame:
        """
        Main entry point for collecting responses.
        
        Args:
            prompts: Dictionary with keys: 'hard_refusal', 'soft_refusal', 'no_refusal'
            parallel: If True, use enhanced parallel processing
            resume_from_checkpoint: If True, resume from existing checkpoint
        
        Returns:
            DataFrame with columns: [prompt, response, model, expected_label, timestamp]
        """
        if parallel:
            return self.collect_all_responses_parallel_enhanced(prompts, resume_from_checkpoint)
        else:
            return self.collect_all_responses_sequential(prompts)

    def collect_all_responses_parallel_enhanced(self, prompts: Dict[str, List[str]],
                                               resume_from_checkpoint: bool = True) -> pd.DataFrame:
        """
        Enhanced parallel collection with separate worker pools per model.
        
        This allows true parallel processing:
        - Each model gets its own thread pool
        - Models process independently
        - Better utilization of rate limits
        """
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=data_checkpoints_path,
            operation_name='response_collection'
        )
        
        # Flatten prompts
        prompt_data = []
        for label_name, prompt_list in prompts.items():
            for prompt in prompt_list:
                prompt_data.append({
                    'prompt': prompt,
                    'expected_label': label_name
                })
        
        total_prompts = len(prompt_data)
        total_calls = total_prompts * len(self.models)
        
        print_banner("COLLECTING RESPONSES (ENHANCED PARALLEL)", char="=")
        print(f"  Total prompts: {total_prompts:,}")
        print(f"  Models: {', '.join([get_model_display_name(k) for k in self.model_keys])}")
        print(f"  Total API calls: {total_calls:,}")
        print(f"  Parallelization: Model-specific worker pools")
        for model_key in self.model_keys:
            limiter = self.model_rate_limiters[model_key]
            print(f"    {get_model_display_name(model_key)}: {limiter.workers} workers, {limiter.delay:.2f}s delay")
        print("=" * 60)
        
        # Check for existing checkpoint
        checkpoint_data = None
        if resume_from_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()
        
        # Resume or start fresh
        if checkpoint_data:
            responses_df = checkpoint_data['data'].copy()
            completed_tasks = checkpoint_data['metadata'].get('completed_tasks', set())
            self.stats = checkpoint_data['metadata'].get('stats', self.stats)
            
            # Restore rate limiter states
            for model_key, limiter_state in checkpoint_data['metadata'].get('rate_limiters', {}).items():
                if model_key in self.model_rate_limiters:
                    self.model_rate_limiters[model_key].workers = limiter_state.get('workers', 5)
                    self.model_rate_limiters[model_key].delay = limiter_state.get('delay', 0.1)
            
            print(f"✅ Resuming from checkpoint: {len(responses_df):,} responses already collected")
            print(f"   Completed tasks: {len(completed_tasks):,}")
        else:
            responses_df = pd.DataFrame()
            completed_tasks = set()
        
        # Thread-safe collection
        lock = threading.Lock()
        all_results = list(responses_df.to_dict('records')) if not responses_df.empty else []
        
        # Create model-specific task queues
        model_tasks = {model_key: [] for model_key in self.model_keys}
        
        for prompt_info in prompt_data:
            for model_key, model_name in zip(self.model_keys, self.models):
                task_id = f"{hash(prompt_info['prompt'])}_{model_key}"
                
                if task_id in completed_tasks:
                    continue
                
                model_tasks[model_key].append({
                    'prompt': prompt_info['prompt'],
                    'expected_label': prompt_info['expected_label'],
                    'model_key': model_key,
                    'model_name': model_name,
                    'task_id': task_id
                })
        
        # Calculate total remaining tasks
        total_remaining = sum(len(tasks) for tasks in model_tasks.values())
        
        if total_remaining == 0:
            print("✅ All tasks already completed!")
            return responses_df
        
        print(f"\n📊 Tasks to process:")
        for model_key in self.model_keys:
            print(f"    {get_model_display_name(model_key)}: {len(model_tasks[model_key]):,} tasks")
        
        # Start time
        start_time = time.time()
        last_checkpoint_count = len(all_results)
        
        # ENHANCED: Separate ThreadPoolExecutor per model for true isolation
        # This prevents thread pool exhaustion and potential deadlocks
        # Each model gets its own dedicated thread pool
        model_executors = {}
        model_futures = {}
        
        try:
            # Create isolated executor for each model
            for model_key in self.model_keys:
                if model_tasks[model_key]:  # Only if tasks exist
                    # Each model gets its own thread pool sized to its rate limiter
                    executor = ThreadPoolExecutor(max_workers=self.model_rate_limiters[model_key].workers)
                    model_executors[model_key] = executor
                    
                    # Submit batch processing to this model's executor
                    future = executor.submit(
                        self._process_model_batch,
                        model_key,
                        model_tasks[model_key],
                        self.model_rate_limiters[model_key],
                        lock,
                        all_results,
                        completed_tasks
                    )
                    model_futures[model_key] = future
            
            # Progress tracking
            with tqdm(total=total_remaining, desc="Total Progress") as pbar:
                
                # Monitor all model processors
                while model_futures:
                    # Brief sleep to not hammer CPU
                    time.sleep(0.5)
                    
                    # Check completed futures
                    done_futures = []
                    for model_key, future in model_futures.items():
                        if future.done():
                            done_futures.append(model_key)
                            try:
                                # Get results from this model
                                model_results = future.result()
                                print(f"\n✅ {get_model_display_name(model_key)} complete: {len(model_results)} responses")
                            except Exception as e:
                                print(f"\n❌ {get_model_display_name(model_key)} failed: {str(e)[:100]}")
                    
                    # Remove completed
                    for model_key in done_futures:
                        del model_futures[model_key]
                    
                    # Update progress
                    current_completed = len(all_results)
                    new_items = current_completed - len(responses_df) if not responses_df.empty else current_completed
                    pbar.n = new_items
                    pbar.refresh()
                    
                    # Periodic checkpoint
                    if current_completed - last_checkpoint_count >= CHECKPOINT_CONFIG['collection_checkpoint_every']:
                        with lock:
                            # Build DataFrame incrementally to prevent memory leak
                            if all_results:
                                # Only convert new results since last checkpoint
                                if responses_df.empty:
                                    # First checkpoint - convert all
                                    responses_df = pd.DataFrame(all_results)
                                else:
                                    # Incremental update - only new results
                                    new_results_start = len(responses_df)
                                    if new_results_start < len(all_results):
                                        new_results = all_results[new_results_start:]
                                        new_df = pd.DataFrame(new_results)
                                        responses_df = pd.concat([responses_df, new_df], ignore_index=True)
                                
                                # Save checkpoint with rate limiter states
                                rate_limiter_states = {
                                    k: v.get_settings() 
                                    for k, v in self.model_rate_limiters.items()
                                }
                                
                                self.checkpoint_manager.save_checkpoint(
                                    data=responses_df,
                                    last_index=len(responses_df),
                                    metadata={
                                        'completed_tasks': completed_tasks.copy(),
                                        'rate_limiters': rate_limiter_states,
                                        'stats': self.stats.copy()
                                    }
                                )
                                last_checkpoint_count = current_completed
                                print(f"\n💾 Checkpoint saved: {len(responses_df):,} responses")
        
        finally:
            # Cleanup: Shutdown all model-specific executors
            for model_key, executor in model_executors.items():
                try:
                    executor.shutdown(wait=True)
                    if EXPERIMENT_CONFIG.get('verbose', True):
                        print(f"  ✓ {get_model_display_name(model_key)} executor shutdown")
                except Exception as e:
                    print(f"  ⚠️  Error shutting down {model_key} executor: {e}")

        # Final DataFrame - build from any remaining results not yet in responses_df
        if all_results:
            new_results_start = len(responses_df) if not responses_df.empty else 0
            if new_results_start < len(all_results):
                new_results = all_results[new_results_start:]
                if new_results:
                    new_df = pd.DataFrame(new_results)
                    if responses_df.empty:
                        responses_df = new_df
                    else:
                        responses_df = pd.concat([responses_df, new_df], ignore_index=True)

        # Final checkpoint - ALWAYS save at completion (force=True)
        rate_limiter_states = {k: v.get_settings() for k, v in self.model_rate_limiters.items()}
        self.checkpoint_manager.save_checkpoint(
            data=responses_df,
            last_index=len(responses_df),
            metadata={
                'completed_tasks': completed_tasks,
                'rate_limiters': rate_limiter_states,
                'stats': self.stats
            },
            force=True  # Force final checkpoint regardless of interval
        )
        print(f"\n💾 Final checkpoint saved: {len(responses_df):,} responses")
        
        # Summary
        elapsed = time.time() - start_time
        self._print_final_summary(responses_df, elapsed)
        
        return responses_df

    def _process_model_batch(self, model_key: str, tasks: List[Dict], 
                             rate_limiter: DynamicRateLimiter, lock, 
                             all_results: List, completed_tasks: set) -> List[Dict]:
        """
        Process all tasks for a specific model with its own worker pool.
        This runs in its own thread and manages its own sub-workers.
        """
        model_results = []
        
        # Model-specific thread pool
        with ThreadPoolExecutor(max_workers=rate_limiter.workers) as executor:
            
            # Submit all tasks for this model
            future_to_task = {
                executor.submit(self._collect_with_rate_limiter, task, model_key, rate_limiter): task
                for task in tasks
            }
            
            # Process as they complete
            with tqdm(total=len(tasks), desc=f"  {get_model_display_name(model_key)}", position=1, leave=False) as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        
                        if result:
                            # Thread-safe append to shared results
                            with lock:
                                all_results.append(result)
                                completed_tasks.add(task['task_id'])
                                model_results.append(result)
                            
                            # Update stats
                            if result['response'] != ERROR_RESPONSE:
                                rate_limiter.success()
                                self.stats['successful_calls'] += 1
                                self.stats['per_model_success'][model_key] += 1
                            else:
                                self.stats['failed_calls'] += 1
                                self.stats['per_model_errors'][model_key] += 1
                                
                                # Check if rate limit
                                if 'rate limit' in result.get('error', '').lower():
                                    rate_limiter.hit_rate_limit()
                                    self.stats['rate_limit_hits'] += 1
                                    self.stats['per_model_rate_limits'][model_key] += 1
                        
                    except Exception as e:
                        print(f"\n❌ Error processing task for {model_key}: {str(e)[:100]}")
                    
                    pbar.update(1)
        
        return model_results

    def _collect_with_rate_limiter(self, task: Dict, model_key: str, 
                                   rate_limiter: DynamicRateLimiter) -> Dict:
        """
        Collect single response with model-specific rate limiter.
        """
        # Wait for rate limiter
        time.sleep(rate_limiter.delay)
        
        try:
            # Query model
            response = self._query_model_with_key(model_key, task['model_name'], task['prompt'])
            
            # Count tokens
            token_count = self._count_tokens(response)
            self.stats['total_tokens'] += token_count
            self.stats['per_model_tokens'][model_key] += token_count
            
            return {
                'prompt': task['prompt'],
                'response': response,
                'model': model_key,
                'model_name': task['model_name'],
                'expected_label': task['expected_label'],
                'timestamp': get_timestamp('display'),
                'tokens': token_count
            }
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Print detailed error information
            print(f"\n❌ RESPONSE COLLECTION ERROR - {get_model_display_name(model_key)}")
            print(f"   Model: {task.get('model_name', 'unknown')}")
            print(f"   Model Key: {model_key}")
            print(f"   Error Type: {error_type}")
            print(f"   Error Message: {error_msg[:300]}")
            print(f"   Prompt preview: {task['prompt'][:100]}...")
            if 'rate limit' in error_msg.lower() or '429' in error_msg:
                print(f"   ⚠️  This is a RATE LIMIT error")
            
            return {
                'prompt': task['prompt'],
                'response': ERROR_RESPONSE,
                'model': model_key,
                'model_name': task['model_name'],
                'expected_label': task['expected_label'],
                'timestamp': get_timestamp('display'),
                'error': error_msg,
                'tokens': 0
            }

    def _query_model_with_key(self, model_key: str, model_name: str, prompt: str) -> str:
        """
        Query a specific model with retry logic.
        """
        for attempt in range(self.max_retries):
            try:
                self.stats['total_api_calls'] += 1
                
                # Route to appropriate API
                if model_key == 'claude':
                    return self._query_claude(prompt)
                elif model_key == 'gpt5':
                    return self._query_gpt5(prompt)
                
                elif model_key == 'gemini':
                    return self._query_gemini(prompt)
                
                else:
                    raise ValueError(f"Unknown model key: {model_key}")
                    
            except Exception as e:
                # Check if rate limit error
                error_str = str(e).lower()
                error_type = type(e).__name__
                is_rate_limit = any(keyword in error_str for keyword in ['rate limit', '429', 'quota'])
                
                if attempt < self.max_retries - 1:
                    # Print retry information
                    print(f"\n⚠️  API ERROR - {get_model_display_name(model_key)} (Attempt {attempt + 1}/{self.max_retries})")
                    print(f"   Error Type: {error_type}")
                    print(f"   Error: {str(e)[:300]}")
                    
                    if is_rate_limit:
                        wait_time = self.rate_limit_backoff * (attempt + 1)
                        print(f"   ⚠️  RATE LIMIT detected - waiting {wait_time}s before retry")
                    else:
                        wait_time = min(2 ** attempt, 30)  # Exponential backoff, cap at 30s
                        print(f"   Retrying in {wait_time}s...")
                    
                    time.sleep(wait_time)
                else:
                    # Final failure
                    print(f"\n❌ API CALL FAILED - {get_model_display_name(model_key)}")
                    print(f"   Model: {model_name}")
                    print(f"   Attempts: {self.max_retries}")
                    print(f"   Error Type: {error_type}")
                    print(f"   Final Error: {str(e)[:300]}")
                    if is_rate_limit:
                        print(f"   ⚠️  This is a persistent RATE LIMIT error")
                        print(f"   💡 Consider: Upgrade API tier, reduce parallel workers, or increase delays")
                    raise e

    def collect_all_responses_sequential(self, prompts: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Simple sequential collection (fallback option).
        """
        all_data = []
        
        # Flatten prompts
        prompt_data = []
        for label_name, prompt_list in prompts.items():
            for prompt in prompt_list:
                prompt_data.append({
                    'prompt': prompt,
                    'expected_label': label_name
                })
        
        total_prompts = len(prompt_data)
        total_calls = total_prompts * len(self.models)
        
        print_banner("COLLECTING RESPONSES (SEQUENTIAL)", char="=")
        print(f"  Total prompts: {total_prompts:,}")
        print(f"  Models: {', '.join([get_model_display_name(k) for k in self.model_keys])}")
        print(f"  Total API calls: {total_calls:,}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Collect responses
        with tqdm(total=total_calls, desc="API Calls") as pbar:
            for prompt_info in prompt_data:
                for model_key, model_name in zip(self.model_keys, self.models):
                    try:
                        # Use single shared rate limiter for sequential
                        rate_limiter = self.model_rate_limiters[model_key]
                        time.sleep(rate_limiter.delay)
                        
                        response = self._query_model_with_key(model_key, model_name, prompt_info['prompt'])
                        
                        token_count = self._count_tokens(response)
                        self.stats['total_tokens'] += token_count
                        self.stats['per_model_tokens'][model_key] += token_count
                        
                        all_data.append({
                            'prompt': prompt_info['prompt'],
                            'response': response,
                            'model': model_key,
                            'model_name': model_name,
                            'expected_label': prompt_info['expected_label'],
                            'timestamp': get_timestamp('display'),
                            'tokens': token_count
                        })
                        
                        self.stats['successful_calls'] += 1
                        self.stats['per_model_success'][model_key] += 1
                        rate_limiter.success()
                        
                    except Exception as e:
                        all_data.append({
                            'prompt': prompt_info['prompt'],
                            'response': ERROR_RESPONSE,
                            'model': model_key,
                            'model_name': model_name,
                            'expected_label': prompt_info['expected_label'],
                            'timestamp': get_timestamp('display'),
                            'tokens': 0
                        })
                        
                        self.stats['failed_calls'] += 1
                        self.stats['per_model_errors'][model_key] += 1
                        
                        if "429" in str(e) or "rate" in str(e).lower():
                            self.stats['rate_limit_hits'] += 1
                            self.stats['per_model_rate_limits'][model_key] += 1
                            rate_limiter.hit_rate_limit()
                    
                    pbar.update(1)
        
        df = pd.DataFrame(all_data)
        elapsed = time.time() - start_time
        self._print_final_summary(df, elapsed)
        
        return df

    def _query_claude(self, prompt: str) -> str:
        """Query Claude Sonnet 4.5."""
        response = self.anthropic_client.messages.create(
            model=API_CONFIG['response_models']['claude'],
            max_tokens=self.max_completion_tokens,
            temperature=API_CONFIG['temperature_response'],
            messages=[{"role": "user", "content": prompt}]
        )
        
        if not response.content or len(response.content) == 0:
            return ERROR_RESPONSE
        return response.content[0].text

    def _query_gpt5(self, prompt: str) -> str:
        """Query GPT-5."""
        response = self.openai_client.chat.completions.create(
            model=API_CONFIG['response_models']['gpt5'],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_completion_tokens,  # GPT-5 uses max_completion_tokens!
            temperature=API_CONFIG['temperature_response']
        )
        
        if not response.choices:
            return ERROR_RESPONSE
        return response.choices[0].message.content

    def _query_gemini(self, prompt: str) -> str:
        """Query Gemini 2.5."""
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_completion_tokens,
            temperature=API_CONFIG['temperature_response']
        )
        
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not response.text:
            return ERROR_RESPONSE
        return response.text

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            # Token counting is non-critical, log only if verbose
            if EXPERIMENT_CONFIG.get('verbose', False):
                print(f"⚠️  Token counting failed: {e}")
            return 0

    def _print_final_summary(self, df: pd.DataFrame, elapsed_time: float):
        """Print final collection summary."""
        print_banner("RESPONSE COLLECTION COMPLETE", char="=")
        
        total_responses = len(df)
        error_responses = len(df[df['response'] == ERROR_RESPONSE])
        success_responses = total_responses - error_responses
        
        print(f"  Total responses: {total_responses:,}")
        print(f"  Successful: {success_responses:,} ({safe_divide(success_responses, total_responses, 0) * 100:.1f}%)")
        print(f"  Errors: {error_responses:,} ({safe_divide(error_responses, total_responses, 0) * 100:.1f}%)")
        print(f"  Time elapsed: {format_time(elapsed_time)}")
        print(f"  Rate: {safe_divide(total_responses, elapsed_time / 60, 0):.1f} responses/min")
        print(f"  Total tokens: {self.stats['total_tokens']:,}")
        
        # Per-model breakdown
        print("\n  Per-model statistics:")
        for model_key in self.model_keys:
            success = self.stats['per_model_success'][model_key]
            errors = self.stats['per_model_errors'][model_key]
            tokens = self.stats['per_model_tokens'][model_key]
            rate_limits = self.stats['per_model_rate_limits'][model_key]
            
            total_model = success + errors
            success_rate = safe_divide(success, total_model, 0) * 100
            
            print(f"\n    {get_model_display_name(model_key)}:")
            print(f"      Success: {success:,} ({success_rate:.1f}%)")
            print(f"      Errors: {errors:,}")
            print(f"      Rate limits: {rate_limits:,}")
            print(f"      Tokens: {tokens:,}")
        
        # Rate limiter final states
        print("\n  Final rate limiter states:")
        for model_key, limiter in self.model_rate_limiters.items():
            settings = limiter.get_settings()
            print(f"    {get_model_display_name(model_key)}:")
            print(f"      Workers: {settings['workers']}, Delay: {settings['delay']:.2f}s")
            print(f"      Success rate: {settings['success_rate']:.1%}")
        
        print("=" * 60)


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 28, 2025
@author: ramyalsaffar
"""
