# KV Cache Debug Analysis Report - memSTORY Project

## Executive Summary

Through comprehensive debugging analysis of the memSTORY Android app's ONNX LLM Engine, we have identified the root cause of the token count expansion issue where simple 48-token inputs were being processed as 500+ token sequences, causing shape mismatch errors and empty AI responses.

**Key Finding**: The implementation was performing **non-incremental generation** with **KV cache double accumulation**, leading to exponential memory growth and ONNX Runtime buffer reuse conflicts.

## Problem Identification

### Initial Symptoms
- User input: "hell" (48 tokens) resulted in processing 500+ tokens
- Shape mismatch errors: `{1,1,512,256} != {1,1,523,256}`
- Empty AI responses despite successful model loading
- Memory inefficiency with each generation step

### Root Cause Analysis

#### 1. Non-Incremental Generation Pattern
```kotlin
// WRONG: Reprocessing entire sequence every step
val inputIds = LongArray(windowedTokens.size) { windowedTokens[it].toLong() }
val positionIds = LongArray(windowedTokens.size) { it.toLong() }
```

**Evidence from logs:**
```
Step 1: inputIds.size: 48, KV cache: [1,1,1,256] → [1,1,49,256]
Step 2: inputIds.size: 49, KV cache: [1,1,49,256] → [1,1,98,256] ❌
Step 3: inputIds.size: 50, KV cache: [1,1,98,256] → [1,1,148,256] ❌
```

#### 2. KV Cache Double Accumulation
- **Expected**: KV cache grows by 1 token per step (incremental)
- **Actual**: KV cache accumulates entire sequence each step
- **Result**: 48 → 49 → 98 → 148 → ... exponential growth

#### 3. Position IDs Misalignment
- **Current**: Always starts from 0 `[0, 1, 2, ..., n-1]`
- **Required**: Should continue from past length `[past_len, past_len+1, ...]`
- **Impact**: ONNX Runtime internal length calculation confusion

## Technical Deep Dive

### Current Implementation Issues

1. **Full Sequence Reprocessing**
   ```kotlin
   // Every step processes ALL tokens from beginning
   for (i in 0 until maxTokens) {
       val windowedTokens = generatedTokens // ENTIRE sequence
       val inputIds = LongArray(windowedTokens.size) // ALL tokens
   }
   ```

2. **Incorrect KV Cache Management**
   ```kotlin
   // KV cache receives outputs for ENTIRE sequence
   // Instead of just the new token
   val keyOutput = outputs?.get(1 + layer * 2) as OnnxTensor
   // This contains [B, H, past_len + cur_len, D]
   ```

3. **Memory Inefficiency**
   - Each step: O(n²) memory growth
   - Token count: 48 → 98 → 148 → 198 → ... → 568+
   - Eventually exceeds sliding window limit (512)

### Correct Implementation Pattern

#### 1. Incremental Generation
```kotlin
// CORRECT: Process only the last token after first inference
val inputIds = if (isFirstInference) {
    // First: process entire prompt
    LongArray(initialTokens.size) { initialTokens[it].toLong() }
} else {
    // Subsequent: process only last generated token
    LongArray(1) { lastGeneratedToken.toLong() }
}
```

#### 2. Proper Position IDs
```kotlin
// CORRECT: Continue from past sequence length
val positionIds = if (isFirstInference) {
    LongArray(inputIds.size) { it.toLong() }
} else {
    LongArray(1) { (pastSequenceLength).toLong() }
}
```

#### 3. KV Cache Sliding Window
```kotlin
// CORRECT: Apply sliding window to KV cache, not input tokens
if (totalSequenceLength > modelConfig.sliding_window) {
    // Trim oldest KV cache entries, keep recent ones
    val keepLength = modelConfig.sliding_window
    // Slice KV cache tensors to keep only recent tokens
}
```

## Solution Implementation

### Phase 1: Incremental Generation Fix
1. **First Inference**: Process entire initial prompt
2. **Subsequent Steps**: Process only 1 new token at a time
3. **Position IDs**: Start from 0 for first, continue from past_length for subsequent

### Phase 2: KV Cache Optimization
1. **Proper Accumulation**: KV cache grows by exactly 1 token per step
2. **Sliding Window**: Apply to KV cache when total length exceeds 512
3. **Memory Management**: Release old tensors, maintain only active window

### Phase 3: ONNX Runtime Integration
1. **Input Consistency**: Ensure input shapes match KV cache expectations
2. **Buffer Management**: Prevent buffer reuse conflicts
3. **Hybrid Cache**: Leverage ONNX Runtime's hybrid cache implementation

## Performance Impact

### Before Fix
- **Memory Growth**: O(n²) per generation step
- **Processing Time**: Increases quadratically with sequence length
- **Token Efficiency**: Reprocesses entire history every step
- **Cache Size**: Unlimited growth until memory exhaustion

### After Fix
- **Memory Growth**: O(n) with sliding window cap
- **Processing Time**: Constant per step (1 token)
- **Token Efficiency**: Processes only new information
- **Cache Size**: Capped at 512 tokens (sliding window)

## Implementation Checklist

### Critical Fixes Required
- [ ] Implement incremental generation (1 token per step after first)
- [ ] Fix position IDs calculation for autoregressive pattern
- [ ] Apply sliding window to KV cache, not input tokens
- [ ] Ensure KV cache grows by exactly 1 token per step
- [ ] Add comprehensive logging for cache size validation

### Optimization Opportunities
- [ ] Implement proper tensor memory management
- [ ] Leverage ONNX Runtime hybrid cache features
- [ ] Add memory usage monitoring and alerts
- [ ] Optimize for Snapdragon 8 Elite NPU capabilities

## Testing Strategy

### Unit Tests
1. **Token Count Validation**: Verify 1-token-per-step growth
2. **KV Cache Shape**: Ensure [B,H,correct_len,D] at each step
3. **Position IDs**: Validate sequential continuation
4. **Memory Usage**: Monitor for memory leaks and growth patterns

### Integration Tests
1. **End-to-End Generation**: Test complete conversation flows
2. **Sliding Window**: Verify proper context preservation
3. **Performance Benchmarks**: Measure generation speed improvements
4. **Memory Stress Testing**: Test with long conversations

## Conclusion

The debugging analysis revealed a fundamental architectural issue in the autoregressive generation implementation. The fix requires shifting from full-sequence reprocessing to proper incremental generation with correct KV cache management.

**Expected Outcomes:**
- **Immediate**: Elimination of shape mismatch errors
- **Performance**: 90%+ reduction in memory usage and processing time
- **Reliability**: Consistent AI response generation
- **Scalability**: Support for longer conversations within memory limits

This analysis provides a clear roadmap for implementing efficient on-device LLM inference for the memSTORY application, ensuring optimal performance on Android devices with limited memory resources.

---

**Report Generated**: August 30, 2025  
**Debug Session**: KV Cache Token Expansion Investigation  
**Status**: Root Cause Identified, Solution Defined  
**Next Phase**: Implementation of Incremental Generation Pattern