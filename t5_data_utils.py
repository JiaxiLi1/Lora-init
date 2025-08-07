"""
T5 data preprocessing utilities for proper denoising pretraining
"""
import torch
import random
import numpy as np
from typing import List, Dict, Any

class T5DataPreprocessor:
    """T5 denoising data preprocessor"""
    
    def __init__(self, tokenizer, noise_density=0.15, mean_noise_span_length=3.0, max_length=512):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.max_length = max_length
        
        # T5 special tokens
        self.sentinel_tokens = [f"<extra_id_{i}>" for i in range(100)]
        
    def add_noise_to_sequence(self, tokens: List[int]) -> tuple:
        """
        Add noise to a sequence of tokens for T5 denoising pretraining
        
        Returns:
            encoder_input: tokens with noise spans replaced by sentinel tokens
            decoder_target: sentinel tokens followed by the original noise spans
        """
        if len(tokens) < 2:
            return tokens, tokens
            
        # Calculate number of tokens to corrupt
        num_to_corrupt = max(1, int(len(tokens) * self.noise_density))
        
        # Calculate average span length
        avg_span_length = max(1, int(self.mean_noise_span_length))
        
        # Generate noise spans
        noise_spans = []
        corrupted_tokens = 0
        
        while corrupted_tokens < num_to_corrupt:
            # Random span length around average
            span_length = max(1, np.random.poisson(avg_span_length))
            span_length = min(span_length, num_to_corrupt - corrupted_tokens)
            
            # Random start position
            max_start = len(tokens) - span_length
            if max_start <= 0:
                break
                
            start = random.randint(0, max_start)
            end = start + span_length
            
            # Check for overlap with existing spans
            overlap = False
            for existing_start, existing_end in noise_spans:
                if not (end <= existing_start or start >= existing_end):
                    overlap = True
                    break
                    
            if not overlap:
                noise_spans.append((start, end))
                corrupted_tokens += span_length
                
            if len(noise_spans) >= 10:  # Limit number of spans
                break
                
        # Sort spans by start position
        noise_spans.sort()
        
        if not noise_spans:
            return tokens, tokens
            
        # Create encoder input (with sentinel tokens)
        encoder_tokens = []
        decoder_tokens = []
        
        last_end = 0
        sentinel_id = 0
        
        for start, end in noise_spans:
            # Add unchanged tokens before this span
            encoder_tokens.extend(tokens[last_end:start])
            
            # Add sentinel token to encoder
            sentinel_token = self.tokenizer.encode(self.sentinel_tokens[sentinel_id], add_special_tokens=False)
            if sentinel_token:
                encoder_tokens.extend(sentinel_token)
                
            # Add sentinel token and original span to decoder target
            if sentinel_token:
                decoder_tokens.extend(sentinel_token)
            decoder_tokens.extend(tokens[start:end])
            
            last_end = end
            sentinel_id += 1
            
        # Add remaining tokens to encoder
        encoder_tokens.extend(tokens[last_end:])
        
        # Add final sentinel and EOS to decoder
        if sentinel_id < len(self.sentinel_tokens):
            final_sentinel = self.tokenizer.encode(self.sentinel_tokens[sentinel_id], add_special_tokens=False)
            if final_sentinel:
                decoder_tokens.extend(final_sentinel)
                
        return encoder_tokens, decoder_tokens
        
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch for T5 denoising pretraining
        """
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        
        encoder_inputs = []
        decoder_inputs = []
        decoder_labels = []
        
        for i in range(batch_size):
            # Get tokens for this sequence (remove padding)
            tokens = input_ids[i].tolist()
            # Remove pad tokens
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]
            
            if len(tokens) < 2:
                # Fallback for very short sequences
                encoder_inputs.append(tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
                start_token = getattr(self.tokenizer, 'decoder_start_token_id', self.tokenizer.pad_token_id)
                eos_token = getattr(self.tokenizer, 'eos_token_id', self.tokenizer.pad_token_id)
                decoder_inputs.append([start_token] + tokens)
                decoder_labels.append(tokens + [eos_token])
                continue
                
            # Apply denoising
            encoder_tokens, decoder_target_tokens = self.add_noise_to_sequence(tokens)
            
            # Truncate if too long
            encoder_tokens = encoder_tokens[:self.max_length]
            decoder_target_tokens = decoder_target_tokens[:self.max_length-1]  # Leave space for start token
            
            # Pad encoder input
            encoder_input = encoder_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(encoder_tokens))
            
            # Create decoder input (starts with pad_token_id for T5)
            start_token = getattr(self.tokenizer, 'decoder_start_token_id', self.tokenizer.pad_token_id)
            decoder_input = [start_token] + decoder_target_tokens
            decoder_input = decoder_input[:self.max_length]
            decoder_input += [self.tokenizer.pad_token_id] * (self.max_length - len(decoder_input))
            
            # Create decoder labels (target tokens + EOS)
            eos_token = getattr(self.tokenizer, 'eos_token_id', self.tokenizer.pad_token_id)
            decoder_label = decoder_target_tokens + [eos_token]
            decoder_label = decoder_label[:self.max_length]
            decoder_label += [-100] * (self.max_length - len(decoder_label))  # Use -100 for padding in labels
            
            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            decoder_labels.append(decoder_label)
            
        return {
            "input_ids": torch.tensor(encoder_inputs, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_inputs, dtype=torch.long),
            "labels": torch.tensor(decoder_labels, dtype=torch.long),
            "attention_mask": (torch.tensor(encoder_inputs) != self.tokenizer.pad_token_id).long(),
            "decoder_attention_mask": (torch.tensor(decoder_inputs) != self.tokenizer.pad_token_id).long()
        }


def create_t5_denoising_batch(batch, tokenizer, noise_density=0.15, mean_noise_span_length=3.0, max_length=512):
    """
    Simple function to create T5 denoising batch
    """
    processor = T5DataPreprocessor(tokenizer, noise_density, mean_noise_span_length, max_length)
    return processor.process_batch(batch)