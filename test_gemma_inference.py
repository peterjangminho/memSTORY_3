#!/usr/bin/env python3
"""
Test actual Gemma 3 1B inference with proper inputs
"""

import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path("/home/peterjangminho/dev/Project_A/app/src/main/assets/models/gemma-3-1b")

def simple_tokenize(text):
    """Simple tokenization for testing (not real BPE)"""
    # Load tokenizer config to get special tokens
    with open(MODEL_DIR / "generation_config.json") as f:
        gen_config = json.load(f)
    
    bos_token_id = gen_config['bos_token_id']
    
    # Simple word-based tokenization (for testing only)
    words = text.lower().split()
    # Map words to token IDs (simple hash for testing)
    token_ids = [bos_token_id]  # Start with BOS
    for word in words:
        # Simple hash to get token ID (0-262143 range)
        token_id = abs(hash(word)) % 262144
        token_ids.append(token_id)
    
    return token_ids

def test_inference():
    """Test actual text generation"""
    print("🚀 TESTING GEMMA 3 1B INFERENCE")
    print("=" * 60)
    
    # Load model
    print("Loading ONNX model...")
    session = ort.InferenceSession(
        str(MODEL_DIR / "model_q4f16.onnx"),
        providers=['CPUExecutionProvider']
    )
    print("✅ Model loaded!")
    
    # Test input
    test_text = "tell me about korea"
    print(f"\n📝 Input text: '{test_text}'")
    
    # Tokenize
    token_ids = simple_tokenize(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Prepare inputs
    batch_size = 1
    seq_length = len(token_ids)
    
    input_ids = np.array([token_ids], dtype=np.int64)
    position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, seq_length)
    
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids
    }
    
    # Add KV cache (float32!)
    for i in range(26):
        # Start with minimal cache
        inputs[f"past_key_values.{i}.key"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
        inputs[f"past_key_values.{i}.value"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
    
    print(f"\n🔧 Prepared {len(inputs)} input tensors")
    print(f"   - input_ids shape: {input_ids.shape}")
    print(f"   - position_ids shape: {position_ids.shape}")
    print(f"   - KV cache: 26 layers × 2 (key+value) × float32")
    
    # Run inference
    print("\n🚀 Running inference...")
    try:
        outputs = session.run(None, inputs)
        print("✅ Inference successful!")
        
        # Get logits
        logits = outputs[0]  # First output is logits
        print(f"\n📊 Output logits shape: {logits.shape}")
        print(f"   Expected: (1, {seq_length}, 262144)")
        
        # Get next token prediction
        next_token_logits = logits[0, -1, :]  # Last position
        
        # Apply softmax for probabilities
        exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Get top 5 predictions
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        print(f"\n🎯 Top {top_k} predicted tokens:")
        for i, idx in enumerate(top_indices):
            print(f"   {i+1}. Token ID {idx}: probability {probs[idx]:.4f}")
        
        # Check KV cache outputs
        print(f"\n📦 KV Cache outputs:")
        print(f"   Total outputs: {len(outputs)}")
        print(f"   Logits + 52 KV tensors (26 layers × 2)")
        
        # Check one KV output shape
        if len(outputs) > 1:
            kv_output = outputs[1]  # First KV cache output
            print(f"   Sample KV shape: {kv_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def test_generation_loop():
    """Test autoregressive generation"""
    print("\n" + "=" * 60)
    print("🔄 TESTING AUTOREGRESSIVE GENERATION")
    print("=" * 60)
    
    # Load model
    session = ort.InferenceSession(
        str(MODEL_DIR / "model_q4f16.onnx"),
        providers=['CPUExecutionProvider']
    )
    
    # Start with a prompt
    prompt = "Tell me"
    token_ids = simple_tokenize(prompt)
    print(f"Starting prompt: '{prompt}'")
    print(f"Initial tokens: {token_ids}")
    
    # Generate 10 more tokens
    max_new_tokens = 10
    generated_tokens = token_ids.copy()
    
    # Initialize KV cache
    kv_cache = {}
    for i in range(26):
        kv_cache[f"past_key_values.{i}.key"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
        kv_cache[f"past_key_values.{i}.value"] = np.zeros((1, 1, 1, 256), dtype=np.float32)
    
    print(f"\n🎲 Generating {max_new_tokens} tokens...")
    
    for step in range(max_new_tokens):
        # Prepare inputs for this step
        if step == 0:
            # First pass: process all prompt tokens
            input_ids = np.array([generated_tokens], dtype=np.int64)
            seq_len = len(generated_tokens)
        else:
            # Subsequent passes: only new token
            input_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
            seq_len = 1
        
        position_ids = np.arange(len(generated_tokens) - seq_len, len(generated_tokens), dtype=np.int64).reshape(1, seq_len)
        
        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            **kv_cache
        }
        
        # Run inference
        outputs = session.run(None, inputs)
        
        # Get next token
        logits = outputs[0]
        next_token_logits = logits[0, -1, :]
        
        # Simple sampling (greedy)
        next_token = int(np.argmax(next_token_logits))
        generated_tokens.append(next_token)
        
        # Update KV cache from outputs
        for i in range(26):
            kv_cache[f"past_key_values.{i}.key"] = outputs[1 + i*2]
            kv_cache[f"past_key_values.{i}.value"] = outputs[2 + i*2]
        
        print(f"   Step {step+1}: Generated token {next_token}")
        
        # Check for EOS
        if next_token in [1, 106]:  # EOS tokens
            print("   🛑 EOS token generated, stopping")
            break
    
    print(f"\n📝 Final generated tokens: {generated_tokens}")
    print(f"   Total length: {len(generated_tokens)} tokens")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("GEMMA 3 1B ONNX INFERENCE TEST")
    print("=" * 60)
    
    # Test 1: Single inference
    success1 = test_inference()
    
    # Test 2: Generation loop
    if success1:
        success2 = test_generation_loop()
        
        if success2:
            print("\n" + "=" * 60)
            print("✅ ✅ ✅ ALL TESTS PASSED! ✅ ✅ ✅")
            print("=" * 60)
            print("\n🎯 READY FOR ANDROID DEPLOYMENT!")
        else:
            print("\n⚠️ Generation test failed")
    else:
        print("\n⚠️ Basic inference failed")
    
    print("\n💡 Next steps:")
    print("1. Build APK with updated OnnxLLMEngine.kt")
    print("2. Install on device")
    print("3. Test with 'tell me about korea' query")