#!/usr/bin/env python3
"""
Test Gemma 3 1B with proper tokenizer decoding
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path("/home/peterjangminho/dev/Project_A/app/src/main/assets/models/gemma-3-1b")

class SimpleTokenizer:
    def __init__(self):
        """Load tokenizer vocabulary"""
        with open(MODEL_DIR / "tokenizer.json", 'r') as f:
            tokenizer_data = json.load(f)
        
        # Extract vocabulary
        self.vocab = tokenizer_data['model']['vocab']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.bos_token_id = 2
        self.eos_token_ids = [1, 106]
        
        print(f"✅ Loaded tokenizer with {len(self.vocab)} tokens")
    
    def encode(self, text):
        """Simple encode (for testing)"""
        tokens = [self.bos_token_id]
        
        # Add special prefix for Gemma
        text = "▁" + text.replace(" ", "▁")
        
        # Simple greedy tokenization
        i = 0
        while i < len(text):
            found = False
            # Try longest match first
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.vocab:
                    tokens.append(self.vocab[substr])
                    i += length
                    found = True
                    break
            
            if not found:
                # Use unknown token or skip
                i += 1
        
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        text = ""
        for token_id in token_ids:
            if token_id in self.eos_token_ids:
                break
            if token_id == self.bos_token_id:
                continue
            
            if token_id in self.reverse_vocab:
                token_text = self.reverse_vocab[token_id]
                # Handle SentencePiece formatting
                if token_text.startswith("▁"):
                    text += " " + token_text[1:]
                else:
                    text += token_text
            else:
                text += f"[UNK_{token_id}]"
        
        return text.strip()

def generate_response(prompt, max_tokens=50):
    """Generate a response to the prompt"""
    print(f"\n🎯 GENERATING RESPONSE")
    print(f"📝 Prompt: '{prompt}'")
    print("=" * 60)
    
    # Load model
    session = ort.InferenceSession(
        str(MODEL_DIR / "model_q4f16.onnx"),
        providers=['CPUExecutionProvider']
    )
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    print(f"📊 Input tokens ({len(input_ids)}): {input_ids}")
    print(f"📝 Input decoded: '{tokenizer.decode(input_ids)}'")
    
    # Generate tokens
    generated_ids = input_ids.copy()
    
    # Initialize KV cache
    kv_cache = {}
    for i in range(26):
        kv_cache[f"past_key_values.{i}.key"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
        kv_cache[f"past_key_values.{i}.value"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
    
    print(f"\n🔄 Generating up to {max_tokens} tokens...")
    print("-" * 60)
    
    for step in range(max_tokens):
        # Prepare inputs
        if step == 0:
            # First pass: all prompt tokens
            curr_input_ids = np.array([generated_ids], dtype=np.int64)
            seq_len = len(generated_ids)
        else:
            # Subsequent: only last token
            curr_input_ids = np.array([[generated_ids[-1]]], dtype=np.int64)
            seq_len = 1
        
        position_ids = np.arange(
            len(generated_ids) - seq_len, 
            len(generated_ids), 
            dtype=np.int64
        ).reshape(1, seq_len)
        
        inputs = {
            "input_ids": curr_input_ids,
            "position_ids": position_ids,
            **kv_cache
        }
        
        # Run inference
        outputs = session.run(None, inputs)
        logits = outputs[0]
        
        # Get next token (using sampling)
        next_token_logits = logits[0, -1, :]
        
        # Apply temperature
        temperature = 0.7
        next_token_logits = next_token_logits / temperature
        
        # Softmax
        exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample from top-k
        top_k = 50
        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)
        
        next_token = np.random.choice(top_indices, p=top_probs)
        generated_ids.append(int(next_token))
        
        # Decode current token
        token_text = tokenizer.decode([next_token])
        print(f"Step {step+1}: Token {next_token} → '{token_text}'", end=" ")
        
        # Update KV cache
        for i in range(26):
            kv_cache[f"past_key_values.{i}.key"] = outputs[1 + i*2]
            kv_cache[f"past_key_values.{i}.value"] = outputs[2 + i*2]
        
        # Check for EOS
        if next_token in tokenizer.eos_token_ids:
            print("\n🛑 EOS token reached")
            break
        
        # Print partial response every 10 tokens
        if (step + 1) % 10 == 0:
            partial = tokenizer.decode(generated_ids[len(input_ids):])
            print(f"\n  📄 Partial: '{partial}'")
    
    print("-" * 60)
    
    # Final response
    response_ids = generated_ids[len(input_ids):]
    response_text = tokenizer.decode(response_ids)
    
    print(f"\n✅ GENERATION COMPLETE")
    print(f"📊 Generated {len(response_ids)} tokens")
    print(f"\n🤖 RESPONSE: '{response_text}'")
    
    return response_text

def main():
    print("=" * 60)
    print("GEMMA 3 1B CONVERSATION TEST")
    print("=" * 60)
    
    # Test queries
    prompts = [
        "tell me about korea",
        "Hello, how are you?",
        "What is machine learning?"
    ]
    
    for prompt in prompts:
        response = generate_response(prompt, max_tokens=30)
        print("\n" + "=" * 60)
        print(f"💬 CONVERSATION:")
        print(f"👤 User: {prompt}")
        print(f"🤖 Gemma: {response}")
        print("=" * 60)
        print("\n")

if __name__ == "__main__":
    main()