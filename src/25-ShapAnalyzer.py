# SHAP Analyzer Module
#----------------------
# Computes SHAP values for model interpretability.
# Generic implementation that works with any classifier (RefusalClassifier or JailbreakDetector).
# All imports are in 01-Imports.py
###############################################################################


class ShapAnalyzer:
    """Compute and visualize SHAP values for model interpretability."""

    def __init__(self, model, tokenizer, device, class_names: List[str] = None):
        """
        Initialize SHAP analyzer.

        Args:
            model: Trained classification model (RefusalClassifier or JailbreakDetector)
            tokenizer: RoBERTa tokenizer
            device: torch device
            class_names: List of class names (default: uses CLASS_NAMES from config)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.model.eval()
        
        # Check shap availability once at initialization
        try:
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            print("⚠️  SHAP not installed. SHAP analysis will be unavailable.")
            print("   Install with: pip install shap")

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for texts (needed for SHAP).

        OPTIMIZATION: Uses batched inference for efficiency.

        Args:
            texts: List of text strings or single string

        Returns:
            Probability array (num_samples, num_classes)
        """
        # Handle single string or list of strings
        if isinstance(texts, str):
            texts = [texts]
        elif hasattr(texts, '__iter__') and not isinstance(texts, (str, bytes)):
            # Handle any iterable but ensure it's a list of strings
            texts = [str(t) for t in texts]
        
        self.model.eval()
        all_probs = []

        # Batch processing for efficiency
        batch_size = TRAINING_CONFIG['batch_size']

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing predictions for SHAP", leave=False):
                batch_texts = texts[i:i+batch_size]

                # Tokenize batch
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=MODEL_CONFIG['max_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Get predictions
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def compute_shap_values(self, texts: List[str], background_texts: List[str] = None,
                           max_background: int = None):
        """
        Compute SHAP values for text samples.

        Args:
            texts: List of texts to explain
            background_texts: Background dataset for SHAP (uses subset if None)
            max_background: Maximum background samples to use (default: from config)

        Returns:
            Dictionary with SHAP values and metadata
        """
        if max_background is None:
            max_background = INTERPRETABILITY_CONFIG['shap_background_samples']
        
        if not self.shap_available:
            print("❌ SHAP not installed. Install with: pip install shap")
            return None
        

        print_banner("COMPUTING SHAP VALUES", width=60, char="=")

        # Prepare background data
        if background_texts is None:
            background_texts = texts[:max_background]
        else:
            background_texts = background_texts[:max_background]

        print(f"Background samples: {len(background_texts)}")
        print(f"Texts to explain: {len(texts)}")

        # EFFICIENT SOLUTION: Use gradient-based attribution
        print("\nInitializing gradient-based explainer...")
        print("Note: Using input gradients for fast transformer interpretability")
        
        try:
            # Compute attribution values using gradients
            shap_values_list = []
            
            # Set model to eval mode but enable gradients
            self.model.eval()
            
            for i, text in enumerate(tqdm(texts, desc="Computing attributions")):
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
                
                # Get embeddings and register hook to capture gradients
                # FIX: Clone embeddings and set requires_grad on the clone (leaf variable)
                with torch.enable_grad():
                    # Get embeddings from the model
                    embeddings = self.model.roberta.embeddings(input_ids)
                    
                    # Clone to create a leaf variable that we can track gradients for
                    embeddings = embeddings.clone().detach().requires_grad_(True)
                    
                    # Forward pass with custom embeddings
                    outputs = self.model.roberta(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask
                    )
                    
                    pooled = outputs[1]  # Pooled output
                    logits = self.model.classifier(pooled)
                    probs = torch.softmax(logits, dim=1)
                    
                    # Compute gradients for each class
                    class_attributions = []
                    for class_idx in range(self.num_classes):
                        # Zero gradients
                        if embeddings.grad is not None:
                            embeddings.grad.zero_()
                        
                        # Backward for this class
                        if class_idx < self.num_classes - 1:
                            probs[0, class_idx].backward(retain_graph=True)
                        else:
                            probs[0, class_idx].backward()
                        
                        # Get gradient magnitude as attribution
                        # Shape: (1, seq_len, hidden_dim)
                        grad = embeddings.grad.detach()
                        
                        # Average over hidden dimension to get per-token attribution
                        # Shape: (seq_len,)
                        token_attribution = grad[0].abs().mean(dim=-1).cpu().numpy()
                        
                        class_attributions.append(token_attribution)
                
                # Stack: (seq_len, num_classes)
                sample_attribution = np.stack(class_attributions, axis=-1)
                shap_values_list.append(sample_attribution)
            
            # Stack all samples: (num_samples, seq_len, num_classes)
            shap_values = np.stack(shap_values_list, axis=0)
            
            print(f"✓ Gradient attributions computed: shape {shap_values.shape}")
            
        except Exception as e:
            print(f"❌ Attribution computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        print("✓ Attribution computation complete")

        return {
            'shap_values': shap_values,  # Numpy array: (n_samples, seq_len, n_classes)
            'texts': texts,
            'background_texts': background_texts,
            'method': 'gradient_attribution'  # Track which method was used
        }

    def visualize_shap(self, text: str, shap_data: dict, output_path: str,
                      class_idx: int = None):
        """
        Create SHAP-like visualization for a single text using gradient attributions.

        Args:
            text: Input text to visualize
            shap_data: Attribution data from compute_shap_values
            output_path: Path to save visualization
            class_idx: Which class to visualize (None for predicted class)
        """
        if not self.shap_available:
            print("❌ SHAP not installed")
            return
        

        # Find the text in shap_data
        text_idx = shap_data['texts'].index(text) if text in shap_data['texts'] else 0
        shap_values = shap_data['shap_values']  # Already a numpy array from gradient computation

        # Determine class to visualize
        if class_idx is None:
            # Use predicted class
            probs = self._predict_proba([text])[0]
            class_idx = np.argmax(probs)

        plt.figure(figsize=(12, 8))
        
        # Extract attribution values for this sample and class
        # Shape: (seq_len, num_classes) → extract for specific class
        if len(shap_values.shape) == 3:
            # (n_samples, seq_len, n_classes)
            sample_attributions = shap_values[text_idx, :, class_idx]
        elif len(shap_values.shape) == 2:
            # (seq_len, n_classes) - single sample
            sample_attributions = shap_values[:, class_idx]
        else:
            raise ValueError(f"Unexpected attribution shape: {shap_values.shape}")
        
        # Tokenize to get actual tokens for display
        encoding = self.tokenizer(
            text,
            max_length=MODEL_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        # Get tokens (only non-padding)
        tokens = []
        token_attributions = []
        for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
            if mask == 0:  # Padding
                break
            if i >= len(sample_attributions):  # Safety check
                break
            
            token_str = self.tokenizer.decode([token_id])
            # Skip special tokens for visualization
            if token_str not in ['<s>', '</s>', '<pad>']:
                tokens.append(token_str.strip())
                token_attributions.append(sample_attributions[i])
        
        # Create bar plot (simpler than waterfall, faster to generate)
        token_attributions = np.array(token_attributions)
        
        # Take top K most important tokens
        top_k = min(20, len(tokens))
        top_indices = np.argsort(np.abs(token_attributions))[-top_k:][::-1]
        
        top_tokens = [tokens[i] for i in top_indices]
        top_values = [token_attributions[i] for i in top_indices]
        
        # Color by positive/negative
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_values]
        
        plt.barh(range(len(top_tokens)), top_values, color=colors)
        plt.yticks(range(len(top_tokens)), top_tokens, fontsize=10)
        plt.xlabel('Attribution Score', fontsize=12)
        plt.title(
            f'Token Importance for {self.class_names[class_idx]}\n{text[:80]}...',
            fontsize=12, fontweight='bold'
        )
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print(f"✓ Saved attribution visualization to {output_path}")

    def analyze_samples(self, test_df: pd.DataFrame, num_samples: int = None, output_dir: str = None, timestamp: str = None, classifier_name: str = None):


        """
        Analyze SHAP values for multiple samples.

        Args:
            test_df: Test DataFrame
            num_samples: Number of samples to analyze (default: from config)
            output_dir: Directory to save results
            timestamp: Optional timestamp for file naming (default: current time)

        Returns:
            Dictionary with SHAP analysis results
        """
        # Use config value if not provided
        if num_samples is None:
            num_samples = INTERPRETABILITY_CONFIG['shap_samples']

        if output_dir is None:
            output_dir = shap_analysis_path
        ensure_dir_exists(output_dir)

        print_banner("SHAP ANALYSIS", width=60, char="=")

        # Sample texts
        sample_indices = np.random.choice(
            len(test_df),
            min(num_samples, len(test_df)),
            replace=False
        )
        sampled_df = test_df.iloc[sample_indices]

        texts = sampled_df['response'].tolist()
        
        # Smart label column detection (same logic as other analyzers)
        if 'refusal_label' in sampled_df.columns:
            labels = sampled_df['refusal_label'].tolist()
        elif 'jailbreak_label' in sampled_df.columns:
            labels = sampled_df['jailbreak_label'].tolist()
        elif 'label' in sampled_df.columns:
            labels = sampled_df['label'].tolist()
        else:
            print("❌ No label column found in DataFrame")
            return None

        # Compute SHAP values
        shap_data = self.compute_shap_values(texts)

        if shap_data is None:
            return None

        # Visualize samples from each class
        results = {'by_class': {}, 'examples': []}

        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            class_mask = np.array(labels) == class_idx

            if not class_mask.any():
                continue

            class_indices = np.where(class_mask)[0]

            # Visualize samples from this class (using config)
            num_viz = INTERPRETABILITY_CONFIG['shap_samples_per_class']
            for i, idx in enumerate(tqdm(class_indices[:num_viz], desc=f"Visualizing {class_name}", leave=False)):
                text = texts[idx]
                if timestamp:
                    output_path = os.path.join(
                        output_dir,
                        f"{classifier_name}_shap_{class_name.replace(' ', '_').lower()}_sample_{i+1}_{timestamp}.png"
                    )
                else:
                    output_path = os.path.join(
                        output_dir,
                        f"{classifier_name}_shap_summary_{class_name.replace(' ', '_').lower()}_{timestamp}.png"
                    )
                self.visualize_shap(text, shap_data, output_path, class_idx)

                results['examples'].append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'true_label': class_name,
                    'class_idx': int(class_idx)
                })

        # Create summary plot
        if not self.shap_available:
            print("⚠️  SHAP not available - skipping summary plot")
        else:
            print("\nCreating SHAP summary plot...")

            try:
                # For each class, create a summary
                max_display = INTERPRETABILITY_CONFIG['shap_max_display']
                for class_idx in range(self.num_classes):
                    class_name = self.class_names[class_idx]

                    plt.figure(figsize=(12, 8))
                    
                    # FIX: Handle different SHAP value structures
                    shap_values_obj = shap_data['shap_values']
                    
                    # Check if it's a SHAP Explanation object
                    if hasattr(shap_values_obj, 'values'):
                        # Standard SHAP Explanation object
                        shap_array = shap_values_obj.values
                    elif isinstance(shap_values_obj, np.ndarray):
                        # Already a numpy array
                        shap_array = shap_values_obj
                    elif isinstance(shap_values_obj, list):
                        # Convert list to numpy array
                        shap_array = np.array(shap_values_obj)
                    else:
                        # Unknown structure - try to convert
                        shap_array = np.array(shap_values_obj)
                    
                    # Handle different dimensionality
                    if len(shap_array.shape) == 3:
                        # Shape: (samples, features, classes)
                        class_shap_values = shap_array[:, :, class_idx]
                    elif len(shap_array.shape) == 2:
                        # Shape: (samples, features) - binary classification or single class
                        class_shap_values = shap_array
                    else:
                        raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")
                    
                    shap.summary_plot(
                        class_shap_values,
                        texts,
                        show=False,
                        max_display=max_display
                    )
                    plt.title(
                        f'SHAP Summary - {class_name}',
                        fontsize=14, fontweight='bold'
                    )
                    summary_path = os.path.join(
                        output_dir,
                        f"shap_summary_{class_name.replace(' ', '_').lower()}.png"
                    )
                    plt.tight_layout()
                    plt.savefig(summary_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
                    plt.close()

                    print(f"✓ Saved summary for {class_name}")

            except Exception as e:
                print(f"⚠️  Could not create summary plot: {e}")

        # Save results
        results_json_path = os.path.join(output_dir, f"{classifier_name}_shap_analysis_results_{timestamp}.json")

        serializable_results = convert_to_serializable(results)

        with open(results_json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Saved SHAP analysis to {results_json_path}")

        return results

    def get_top_features(self, text: str, class_idx: int = None, top_k: int = None):
        """
        Get top features (tokens) influencing prediction.

        IMPROVED: Uses actual RoBERTa BPE tokenization for proper alignment.

        Args:
            text: Input text
            class_idx: Target class (None for predicted)
            top_k: Number of top features to return (default: from config)

        Returns:
            Dictionary with top positive and negative features
        """
        # Use config value if not provided
        if top_k is None:
            top_k = INTERPRETABILITY_CONFIG['shap_max_display']
        
        if not self.shap_available:
            print("❌ SHAP not installed")
            return None
        

        # Compute SHAP values
        shap_data = self.compute_shap_values([text])

        if shap_data is None:
            return None

        # Get predicted class if not specified
        if class_idx is None:
            probs = self._predict_proba([text])[0]
            class_idx = np.argmax(probs)

        # Get SHAP values for this class
        # FIX: Handle different SHAP value structures robustly
        shap_values_obj = shap_data['shap_values']
        
        if hasattr(shap_values_obj, 'values'):
            shap_array = shap_values_obj.values
        elif isinstance(shap_values_obj, np.ndarray):
            shap_array = shap_values_obj
        elif isinstance(shap_values_obj, list):
            shap_array = np.array(shap_values_obj)
        else:
            shap_array = np.array(shap_values_obj)
        
        # Handle different dimensionality
        if len(shap_array.shape) == 3:
            # Shape: (samples, features, classes)
            shap_vals = shap_array[0, :, class_idx]
        elif len(shap_array.shape) == 2:
            # Shape: (samples, features) - binary or single class
            shap_vals = shap_array[0, :]
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_array.shape}")

        # PROPER TOKENIZATION: Use actual RoBERTa tokenizer
        encoding = self.tokenizer(
            text,
            max_length=MODEL_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'][0]  # Shape: (seq_len,)
        attention_mask = encoding['attention_mask'][0]

        # Decode tokens to get readable strings
        tokens = []
        valid_indices = []
        for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
            if mask == 0:  # Padding
                break

            # Decode token
            token_str = self.tokenizer.decode([token_id])

            # Skip special tokens
            if token_str in ['<s>', '</s>', '<pad>']:
                continue

            tokens.append(token_str)
            valid_indices.append(i)

        # Match SHAP values to valid tokens
        if len(valid_indices) > len(shap_vals):
            valid_indices = valid_indices[:len(shap_vals)]
        elif len(valid_indices) < len(shap_vals):
            shap_vals = shap_vals[:len(valid_indices)]

        # Get SHAP values for valid tokens
        valid_shap_vals = shap_vals[valid_indices] if len(valid_indices) <= len(shap_vals) else shap_vals[:len(valid_indices)]

        # Get top positive and negative
        sorted_indices = np.argsort(np.abs(valid_shap_vals))[::-1]
        top_indices = sorted_indices[:min(top_k, len(sorted_indices))]

        top_features = []
        for idx in top_indices:
            if idx < len(tokens):
                top_features.append({
                    'token': tokens[idx].strip(),  # Clean up whitespace
                    'shap_value': float(valid_shap_vals[idx]),
                    'impact': 'positive' if valid_shap_vals[idx] > 0 else 'negative'
                })

        return {
            'text': text,
            'class': self.class_names[class_idx],
            'top_features': top_features,
            'note': 'Tokens are from RoBERTa BPE tokenization (may be subwords)'
        }


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 28, 2025
@author: ramyalsaffar
"""
