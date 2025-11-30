# Jailbreak Classifier Model 
#----------------------------
# Fine-tuned RoBERTa for binary jailbreak success detection.
# Detects when adversarial attacks successfully bypass AI safety mechanisms.
#
# WHY RoBERTa?
# 1. State-of-the-art performance on binary classification
# 2. Superior contextual understanding for attack patterns
# 3. Proven effectiveness on safety-critical tasks
# 4. Robust to adversarial inputs with fine-tuning
# 5. Bidirectional attention captures attack/response patterns
#
# IMPROVEMENTS:
# - Full Config/Utils integration (NO HARDCODING!)
# - Renamed to JailbreakClassifier for consistency
# - Model statistics reporting
# - Temperature scaling for calibration
# - MC Dropout for uncertainty estimation
# - Attention visualization support
#
# All imports are in 01-Imports.py
###############################################################################


class JailbreakClassifier(nn.Module):
    """Enhanced RoBERTa-based binary classifier for jailbreak success detection."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 dropout: float = None,
                 model_name: str = None,
                 use_class_weights: bool = False,
                 temperature: float = 1.0):
        """
        Initialize enhanced jailbreak classifier.
        
        Args:
            num_classes: Number of output classes (2: Failed/Succeeded)
            dropout: Dropout probability (uses config if None)
            model_name: Pre-trained model name (uses config if None)
            use_class_weights: Whether to use class weights for imbalanced data
            temperature: Temperature for confidence calibration (must be > 0, default: 1.0)
        """
        super(JailbreakClassifier, self).__init__()
        
        # Validate temperature (used for division in forward pass)
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        # Use config values - NO HARDCODING!
        self.num_classes = num_classes
        self.model_name = model_name or JAILBREAK_CONFIG['model_name']
        self.dropout_prob = dropout or JAILBREAK_CONFIG['dropout']
        self.temperature = temperature
        self.use_class_weights = use_class_weights
        
        # Load pre-trained RoBERTa
        # Note: All imports are in 01-Imports.py
        self.roberta = RobertaModel.from_pretrained(
            self.model_name,
            attn_implementation="eager",  # Suppress compatibility warnings
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob
        )
        
        # Get hidden size from config
        self.hidden_size = self.roberta.config.hidden_size
        
        # Enhanced binary classification head with layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        # Model statistics
        self.total_params = self._count_parameters(trainable_only=False)
        self.trainable_params = self._count_parameters(trainable_only=True)
        
        if EXPERIMENT_CONFIG.get('verbose', True):
            self._print_model_info()
    
    def _count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def _print_model_info(self):
        """Print model information using Utils."""
        print_banner("JAILBREAK CLASSIFIER MODEL", char="=")
        print(f"  Model: {self.model_name}")
        print(f"  Task: Binary jailbreak detection")
        print(f"  Classes: {self.num_classes} (Failed=0, Succeeded=1)")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Dropout: {self.dropout_prob}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Total parameters: {self.total_params:,}")
        print(f"  Trainable parameters: {self.trainable_params:,}")
        print(f"  Class weights: {self.use_class_weights}")
        print("=" * 60)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                output_attentions: bool = False,
                output_hidden_states: bool = False):
        """
        Enhanced forward pass with optional outputs.
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            output_attentions: Return attention weights
            output_hidden_states: Return hidden states
        
        Returns:
            If no extra outputs: Logits (batch_size, 2)
            Otherwise: Dictionary with logits and requested outputs
        """
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Enhanced classification head
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Apply temperature scaling for calibration
        logits = logits / self.temperature
        
        # Return based on requested outputs
        if output_attentions or output_hidden_states:
            result = {'logits': logits}
            if output_attentions:
                result['attentions'] = outputs.attentions
            if output_hidden_states:
                result['hidden_states'] = outputs.hidden_states
            return result
        
        return logits
    
    def predict_with_confidence(self, 
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               use_mc_dropout: bool = False,
                               n_samples: int = 10):
        """
        Get predictions with calibrated confidence scores.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            use_mc_dropout: Use Monte Carlo dropout for uncertainty
            n_samples: Number of MC dropout samples
        
        Returns:
            Dictionary with predictions, confidences, and uncertainties
            Labels: 0 = Jailbreak Failed, 1 = Jailbreak Succeeded
        """
        if use_mc_dropout:
            # Monte Carlo Dropout for uncertainty estimation
            self.train()  # Enable dropout
            
            all_probs = []
            with torch.no_grad():
                for _ in range(n_samples):
                    logits = self.forward(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.unsqueeze(0))
            
            # Stack and compute statistics
            all_probs = torch.cat(all_probs, dim=0)  # (n_samples, batch, 2)
            mean_probs = all_probs.mean(dim=0)
            std_probs = all_probs.std(dim=0)
            
            # Predictions and confidence
            confidence, predicted = torch.max(mean_probs, dim=1)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = std_probs.gather(1, predicted.unsqueeze(1)).squeeze()
            
            # Aleatoric uncertainty (data uncertainty) 
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)
            
            self.eval()  # Restore eval mode
            
            return {
                'predictions': predicted,
                'confidence': confidence,
                'probabilities': mean_probs,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': entropy,
                'total_uncertainty': epistemic_uncertainty + entropy,
                'jailbreak_probability': mean_probs[:, 1]  # Probability of success
            }
        else:
            # Standard prediction
            self.eval()
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
            
            return {
                'predictions': predicted,
                'confidence': confidence,
                'probabilities': probs,
                'jailbreak_probability': probs[:, 1]  # Probability of success
            }
    
    def freeze_roberta_layers(self, num_layers_to_freeze: int = None):
        """
        Freeze bottom RoBERTa layers for efficient training.
        
        Args:
            num_layers_to_freeze: Number of encoder layers to freeze
        """
        num_layers_to_freeze = num_layers_to_freeze or JAILBREAK_CONFIG.get('freeze_layers', 0)
        
        if num_layers_to_freeze == 0:
            return
        
        # Freeze embeddings
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified encoder layers
        total_layers = len(self.roberta.encoder.layer)
        layers_to_freeze = min(num_layers_to_freeze, total_layers)
        
        for i in range(layers_to_freeze):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Update trainable parameters count
        self.trainable_params = self._count_parameters(trainable_only=True)
        
        if EXPERIMENT_CONFIG.get('verbose', True):
            print_banner("FREEZING LAYERS", char="─")
            print(f"  Froze {layers_to_freeze}/{total_layers} RoBERTa layers")
            print(f"  Froze embedding layers: Yes")
            print(f"  New trainable parameters: {self.trainable_params:,}")
            print("─" * 60)
    
    def set_temperature(self, temperature: float):
        """
        Set temperature for confidence calibration.
        
        Args:
            temperature: Temperature value (>1 makes model less confident)
        """
        self.temperature = temperature
        if EXPERIMENT_CONFIG.get('verbose', True):
            print(f"Set temperature to {temperature:.2f}")
    
    def get_attention_weights(self, 
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            layer: int = -1):
        """
        Get attention weights for visualization.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            layer: Which layer's attention to return (-1 for last)
        
        Returns:
            Attention weights (batch, heads, seq_len, seq_len)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, 
                attention_mask,
                output_attentions=True
            )
            
            if isinstance(outputs, dict):
                attentions = outputs['attentions']
            else:
                # Backward compatibility
                _, attentions = outputs
            
            return attentions[layer]
    
    def get_interpretable_prediction(self,
                                    input_ids: torch.Tensor,
                                    attention_mask: torch.Tensor):
        """
        Get prediction with interpretable output for jailbreak detection.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Dictionary with interpretable results
        """
        results = self.predict_with_confidence(
            input_ids, 
            attention_mask,
            use_mc_dropout=True
        )
        
        # Add interpretable labels
        batch_size = results['predictions'].shape[0]
        interpretations = []
        
        for i in range(batch_size):
            pred = results['predictions'][i].item()
            conf = results['confidence'][i].item()
            jb_prob = results['jailbreak_probability'][i].item()
            uncertainty = results['total_uncertainty'][i].item()
            
            interpretation = {
                'prediction': 'Jailbreak Succeeded' if pred == 1 else 'Jailbreak Failed',
                'confidence': f"{conf*100:.1f}%",
                'jailbreak_probability': f"{jb_prob*100:.1f}%",
                'defense_probability': f"{(1-jb_prob)*100:.1f}%",
                'uncertainty': 'Low' if uncertainty < 0.5 else 'Medium' if uncertainty < 1.0 else 'High',
                'raw_values': {
                    'prediction': pred,
                    'confidence': conf,
                    'jailbreak_prob': jb_prob,
                    'uncertainty': uncertainty
                }
            }
            interpretations.append(interpretation)
        
        return interpretations


# Backward compatibility - keep old name as alias
JailbreakDetector = JailbreakClassifier


#------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RoBERTa Classifiers: 3-Class Refusal Taxonomy & Binary Jailbreak Detection
Created on October 30, 2025
@author: ramyalsaffar
"""
