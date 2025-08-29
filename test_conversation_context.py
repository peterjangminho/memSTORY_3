#!/usr/bin/env python3
"""
Test conversation context and KV cache memory usage
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
import psutil
import os

MODEL_DIR = Path("/home/peterjangminho/dev/Project_A/app/src/main/assets/models/gemma-3-1b")

class ConversationTester:
    def __init__(self):
        print("🚀 LOADING MODEL...")
        self.session = ort.InferenceSession(
            str(MODEL_DIR / "model_q4f16.onnx"),
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        with open(MODEL_DIR / "tokenizer.json", 'r') as f:
            tokenizer_data = json.load(f)
        
        self.vocab = tokenizer_data['model']['vocab']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.bos_token_id = 2
        self.eos_token_ids = [1, 106]
        
        # Conversation state
        self.conversation_history = []
        self.kv_cache = {}
        self.total_tokens = 0
        
        # Initialize KV cache
        self.reset_kv_cache()
        
        print(f"✅ Model loaded with {len(self.vocab)} tokens")
    
    def reset_kv_cache(self):
        """Initialize empty KV cache"""
        self.kv_cache = {}
        for i in range(26):
            self.kv_cache[f"past_key_values.{i}.key"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
            self.kv_cache[f"past_key_values.{i}.value"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
    
    def get_kv_cache_memory(self):
        """Calculate KV cache memory usage"""
        total_bytes = 0
        for key, tensor in self.kv_cache.items():
            total_bytes += tensor.nbytes
        return total_bytes / (1024 * 1024)  # MB
    
    def get_process_memory(self):
        """Get current process memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024)
        }
    
    def encode(self, text):
        """Encode text to token IDs"""
        tokens = [self.bos_token_id] if not self.conversation_history else []
        
        text = "▁" + text.replace(" ", "▁")
        
        i = 0
        while i < len(text):
            found = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.vocab:
                    tokens.append(self.vocab[substr])
                    i += length
                    found = True
                    break
            if not found:
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
                if token_text.startswith("▁"):
                    text += " " + token_text[1:]
                else:
                    text += token_text
        return text.strip()
    
    def generate_response(self, user_input, max_tokens=50):
        """Generate response maintaining conversation context"""
        print(f"\n📝 USER: {user_input}")
        print("=" * 50)
        
        # Track memory before
        mem_before = self.get_process_memory()
        kv_mem_before = self.get_kv_cache_memory()
        
        # Encode user input
        input_tokens = self.encode(user_input)
        print(f"📊 Input tokens: {len(input_tokens)}")
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Generate response tokens
        generated_tokens = []
        
        print(f"🧠 KV Cache before: {kv_mem_before:.2f} MB")
        print(f"🧠 Process memory: {mem_before['rss_mb']:.2f} MB")
        print(f"🔄 Total conversation tokens so far: {self.total_tokens}")
        
        # Generation loop
        for step in range(max_tokens):
            if step == 0:
                # First step: process user input
                curr_input_ids = np.array([input_tokens], dtype=np.int64)
                seq_len = len(input_tokens)
            else:
                # Subsequent steps: only last generated token
                if not generated_tokens:
                    break
                curr_input_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                seq_len = 1
            
            # Position IDs continue from total conversation position
            start_pos = self.total_tokens + (0 if step == 0 else len(input_tokens) + len(generated_tokens) - 1)
            position_ids = np.arange(start_pos, start_pos + seq_len, dtype=np.int64).reshape(1, seq_len)
            
            # Prepare inputs
            inputs = {
                "input_ids": curr_input_ids,
                "position_ids": position_ids,
                **self.kv_cache
            }
            
            # Run inference
            outputs = self.session.run(None, inputs)
            logits = outputs[0]
            
            # Sample next token
            next_token_logits = logits[0, -1, :] / 0.7  # temperature
            exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Top-k sampling
            top_k = 50
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices] / np.sum(probs[top_indices])
            next_token = int(np.random.choice(top_indices, p=top_probs))
            
            generated_tokens.append(next_token)
            
            # Update KV cache from outputs
            for i in range(26):
                self.kv_cache[f"past_key_values.{i}.key"] = outputs[1 + i*2]
                self.kv_cache[f"past_key_values.{i}.value"] = outputs[2 + i*2]
            
            # Check for EOS
            if next_token in self.eos_token_ids:
                break
        
        # Update total token count
        self.total_tokens += len(input_tokens) + len(generated_tokens)
        
        # Decode response
        response_text = self.decode(generated_tokens)
        
        # Track memory after
        mem_after = self.get_process_memory()
        kv_mem_after = self.get_kv_cache_memory()
        
        # Add to conversation history
        self.conversation_history.append(f"Assistant: {response_text}")
        
        # Print results
        print(f"🤖 ASSISTANT: {response_text}")
        print("-" * 50)
        print(f"📊 Generated tokens: {len(generated_tokens)}")
        print(f"🧠 KV Cache after: {kv_mem_after:.2f} MB (Δ +{kv_mem_after-kv_mem_before:.2f} MB)")
        print(f"🧠 Process memory: {mem_after['rss_mb']:.2f} MB (Δ +{mem_after['rss_mb']-mem_before['rss_mb']:.2f} MB)")
        print(f"🔄 Total conversation tokens: {self.total_tokens}")
        
        return response_text

def main():
    print("=" * 60)
    print("GEMMA 3 1B CONVERSATION CONTEXT TEST")
    print("=" * 60)
    
    tester = ConversationTester()
    
    # First question
    print("\n🎯 FIRST QUESTION")
    response1 = tester.generate_response("tell me about korea", max_tokens=40)
    
    # Second question - testing context retention
    print("\n🎯 SECOND QUESTION - CONTEXT TEST")
    response2 = tester.generate_response("tell me more", max_tokens=40)
    
    # Third question - memory growth test
    print("\n🎯 THIRD QUESTION - MEMORY GROWTH TEST")
    response3 = tester.generate_response("what about the culture?", max_tokens=40)
    
    print("\n" + "=" * 60)
    print("📝 FULL CONVERSATION SUMMARY:")
    print("=" * 60)
    for i, msg in enumerate(tester.conversation_history, 1):
        print(f"{i}. {msg}")
    
    print(f"\n💾 FINAL MEMORY STATS:")
    print(f"   - KV Cache: {tester.get_kv_cache_memory():.2f} MB")
    print(f"   - Total tokens: {tester.total_tokens}")
    print(f"   - Process memory: {tester.get_process_memory()['rss_mb']:.2f} MB")

if __name__ == "__main__":
    main()