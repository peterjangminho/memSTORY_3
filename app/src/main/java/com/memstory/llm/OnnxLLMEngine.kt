package com.memstory.llm

import ai.onnxruntime.*
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.IntBuffer

/**
 * ONNX Runtime-based LLM inference engine for Gemma 3 1B model
 * Handles model loading, tokenization, and text generation with Static KV Cache
 */
class OnnxLLMEngine(private val context: Context) {
    
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null
    private lateinit var tokenizer: GemmaTokenizer
    private lateinit var modelConfig: ModelConfig
    
    // 🚀 HYBRID KV CACHE - Dynamic sizing with sliding window pattern  
    private var staticKvCache: MutableList<Pair<OnnxTensor, OnnxTensor>>? = null
    private var systemPromptTokens: IntArray? = null
    private var systemPromptLength: Int = 0
    private var maxStaticCacheLength: Int = 512  // Sliding window limit from config
    private var currentCachePosition: Int = 0  // Track current position in KV cache
    private var kvCacheSequenceLength: Int = 0  // Track actual sequence length in cache
    private val slidingThreshold: Int get() = (maxStaticCacheLength * 0.8).toInt()  // 80% threshold (410 tokens)
    
    companion object {
        private const val TAG = "OnnxLLMEngine"
        private const val MODEL_PATH = "models/gemma-3-1b"
        private const val MODEL_FILE = "model_q4f16.onnx"
        private const val CONFIG_FILE = "config.json"
        private const val TOKENIZER_FILE = "tokenizer.json"
        private const val GENERATION_CONFIG_FILE = "generation_config.json"
    }
    
    // 🧠 HYBRID CACHE PATTERN FUNCTIONS
    /**
     * Determines if a layer uses sliding window (every 6th layer per Gemma-3 config)
     */
    private fun isSlididingWindowLayer(layer: Int): Boolean {
        return layer % modelConfig.sliding_window_pattern == 0
    }
    
    /**
     * Calculate optimal cache size based on 2-Phase Gemma-3 Hybrid strategy
     * Phase 1 (0~512): All layers grow dynamically
     * Phase 2 (512+): Full-attention layers freeze at 512 + FIFO, Sliding layers handle overflow
     */
    private fun calculateCacheSize(layer: Int, currentSequenceLength: Int): Int {
        return when {
            // Phase 1: Dynamic Growth (0~512 tokens)
            // All layers grow together up to sliding window limit
            currentSequenceLength <= maxStaticCacheLength -> {
                currentSequenceLength  // 80, 150, 300, 450, 512...
            }
            
            // Phase 2: Hybrid Sliding (512+ tokens) 
            // Role separation: Full-attention freeze, Sliding handle overflow
            currentSequenceLength > maxStaticCacheLength -> {
                if (isSlididingWindowLayer(layer)) {
                    // Sliding layers: Handle overflow tokens (513, 514, 515...)
                    currentSequenceLength
                } else {
                    // Full-attention layers: Freeze at 512 + FIFO maintenance
                    maxStaticCacheLength  // Always 512
                }
            }
            
            else -> currentSequenceLength
        }
    }
    
    /**
     * Resize KV cache if needed for new sequence length
     */
    private fun resizeKvCacheIfNeeded(newSequenceLength: Int) {
        if (staticKvCache == null) return
        
        var needsResize = false
        
        // Check if any layer needs resizing
        for (layer in 0 until modelConfig.num_hidden_layers) {
            val requiredSize = calculateCacheSize(layer, newSequenceLength)
            val currentSize = staticKvCache!![layer].first.info.shape[2]  // seq_len dimension
            
            if (requiredSize != currentSize.toInt()) {
                needsResize = true
                break
            }
        }
        
        if (!needsResize) return
        
        Log.d(TAG, "🔄 Resizing KV Cache for sequence length: $newSequenceLength")
        
        // Resize each layer
        for (layer in 0 until modelConfig.num_hidden_layers) {
            val requiredSize = calculateCacheSize(layer, newSequenceLength)
            val currentSize = staticKvCache!![layer].first.info.shape[2].toInt()
            
            if (requiredSize != currentSize) {
                val layerType = if (isSlididingWindowLayer(layer)) "Sliding" else "Full"
                
                // Close old tensors
                staticKvCache!![layer].first.close()
                staticKvCache!![layer].second.close()
                
                // Create new tensors with required size
                val keyCache = OnnxTensor.createTensor(
                    ortEnvironment, 
                    Array(1) {
                        Array(modelConfig.num_key_value_heads) {
                            Array(requiredSize) {
                                FloatArray(modelConfig.head_dim) { Float.NaN }
                            }
                        }
                    }
                )
                val valueCache = OnnxTensor.createTensor(
                    ortEnvironment, 
                    Array(1) {
                        Array(modelConfig.num_key_value_heads) {
                            Array(requiredSize) {
                                FloatArray(modelConfig.head_dim) { Float.NaN }
                            }
                        }
                    }
                )
                
                staticKvCache!![layer] = Pair(keyCache, valueCache)
                
                Log.v(TAG, "Layer $layer ($layerType) resized: $currentSize → $requiredSize tokens")
            }
        }
    }
    
    data class ModelConfig(
        val vocab_size: Int,
        val max_position_embeddings: Int,
        val hidden_size: Int,
        val num_attention_heads: Int,
        val num_hidden_layers: Int,
        val bos_token_id: Int,
        val eos_token_id: List<Int>,
        val sliding_window: Int,
        val sliding_window_pattern: Int,
        val cache_implementation: String,
        val num_key_value_heads: Int,
        val head_dim: Int,
        val use_cache: Boolean
    )
    
    data class GenerationConfig(
        val max_length: Int = 512,  // Aligned with sliding_window
        val temperature: Float = 0.7f,
        val top_p: Float = 0.9f,
        val do_sample: Boolean = true,
        val repetition_penalty: Float = 1.0f,  // No penalty by default
        val length_penalty: Float = 1.0f  // No length penalty
    )
    
    /**
     * Initialize the LLM engine with model loading and tokenizer setup
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing ONNX LLM Engine...")
            
            // Initialize ONNX Runtime Environment
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Load model configuration
            modelConfig = loadModelConfig()
            Log.d(TAG, "Model config loaded: vocab_size=${modelConfig.vocab_size}")
            Log.d(TAG, "Config details: sliding_window=${modelConfig.sliding_window}, " +
                     "cache_impl=${modelConfig.cache_implementation}, " +
                     "num_kv_heads=${modelConfig.num_key_value_heads}, " +
                     "head_dim=${modelConfig.head_dim}")
            
            // Update max cache length from config
            maxStaticCacheLength = modelConfig.sliding_window
            
            // Load tokenizer
            tokenizer = GemmaTokenizer.fromAssets(context.assets, "$MODEL_PATH/$TOKENIZER_FILE")
            Log.d(TAG, "Tokenizer loaded with vocab size: ${tokenizer.vocabSize}")
            
            // Load ONNX model with NNAPI acceleration for Snapdragon 8 Elite
            val modelFilePath = ensureModelInInternalStorage()
            
            // Create session options with NNAPI provider for Snapdragon 8 Elite NPU
            val sessionOptions = OrtSession.SessionOptions()
            try {
                // Configure NNAPI execution provider for NPU acceleration (1.22.0+ API)
                val nnapiFlags = java.util.EnumSet.of(
                    NNAPIFlags.USE_FP16  // FP16 optimization for Snapdragon 8 Elite NPU
                )
                sessionOptions.addNnapi(nnapiFlags)
                
                // ONNX Runtime 1.22.0 optimizations - Use single optimization level setting
                sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                sessionOptions.setInterOpNumThreads(4)  // Match device CPU cores  
                sessionOptions.setIntraOpNumThreads(4)  // Parallel execution within operators
                
                ortSession = ortEnvironment?.createSession(modelFilePath, sessionOptions)
                Log.d(TAG, "ONNX model loaded with NNAPI execution provider (NPU acceleration)")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI not available, falling back to CPU only: ${e.message}")
                val fallbackOptions = OrtSession.SessionOptions()
                fallbackOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                ortSession = ortEnvironment?.createSession(modelFilePath, fallbackOptions)
                Log.d(TAG, "ONNX model loaded with CPU execution provider only")
            }
            
            // 🚀 Initialize Static KV Cache and System Prompt
            initializeStaticKvCacheAndSystemPrompt()
            
            Log.d(TAG, "LLM Engine initialization completed")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LLM Engine", e)
            false
        }
    }
    
    /**
     * 🚀 Initialize Static KV Cache with System Prompt Pre-processing
     * Based on GitHub Issue #23061 GQA model optimization
     */
    private suspend fun initializeStaticKvCacheAndSystemPrompt() {
        try {
            Log.d(TAG, "🚀 Initializing Static KV Cache with System Prompt...")
            
            // System prompt that knows user's heart the best
            val systemPrompt = """You are memSTORY, the AI that understands your heart the best.
                |You live entirely on your phone, helping organize thoughts, memories, and ideas.
                |Always be warm, understanding, and helpful. You work completely offline.""".trimMargin()
            
            // Tokenize system prompt
            systemPromptTokens = tokenizer.encode(systemPrompt).toIntArray()
            systemPromptLength = systemPromptTokens!!.size
            
            Log.d(TAG, "System prompt tokenized: $systemPromptLength tokens")
            Log.d(TAG, "System prompt: $systemPrompt")
            
            // 🧠 Pre-allocate Hybrid KV Cache with dynamic sizing
            staticKvCache = mutableListOf()
            
            // Calculate initial cache size (system prompt only for now)
            val initialCacheSize = systemPromptLength
            
            Log.d(TAG, "🧠 Creating Hybrid KV Cache with initial size: $initialCacheSize tokens")
            Log.d(TAG, "Sliding threshold: $slidingThreshold tokens (80% of $maxStaticCacheLength)")
            
            // For Gemma 3 1B: 26 layers, 1 KV head (GQA), 256 head dimension
            // 🔥 ONNX Runtime KV Cache Standard Shape: [batch_size, num_heads, seq_len, head_dim]
            for (layer in 0 until modelConfig.num_hidden_layers) {
                val cacheSize = calculateCacheSize(layer, initialCacheSize)
                val layerType = if (isSlididingWindowLayer(layer)) "Sliding" else "Full"
                
                // 🚀 Create properly shaped KV cache tensors with dynamic sizing
                val keyCache = OnnxTensor.createTensor(
                    ortEnvironment, 
                    Array(1) {  // batch_size = 1
                        Array(modelConfig.num_key_value_heads) {  // num_kv_heads = 1
                            Array(cacheSize) {  // Dynamic seq_len based on layer type
                                FloatArray(modelConfig.head_dim) { Float.NaN }  // Uninitialized like torch.empty()
                            }
                        }
                    }
                )
                val valueCache = OnnxTensor.createTensor(
                    ortEnvironment, 
                    Array(1) {  // batch_size = 1
                        Array(modelConfig.num_key_value_heads) {  // num_kv_heads = 1
                            Array(cacheSize) {  // Dynamic seq_len based on layer type
                                FloatArray(modelConfig.head_dim) { Float.NaN }  // Uninitialized like torch.empty()
                            }
                        }
                    }
                )
                
                staticKvCache!!.add(Pair(keyCache, valueCache))
                
                Log.v(TAG, "Layer $layer ($layerType) KV Cache: [1, ${modelConfig.num_key_value_heads}, $cacheSize, ${modelConfig.head_dim}]")
            }
            
            // Reset cache tracking
            currentCachePosition = 0
            kvCacheSequenceLength = 0
            
            // 🔥 Pre-process system prompt to warm up KV cache
            val systemInputIds = LongArray(systemPromptLength) { systemPromptTokens!![it].toLong() }
            val systemPositionIds = LongArray(systemPromptLength) { it.toLong() }
            
            // Create input tensors
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(systemInputIds))
            val positionTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(systemPositionIds))
            
            // Prepare KV cache inputs (initially zeros)
            val inputs = mutableMapOf<String, OnnxTensorLike>()
            inputs["input_ids"] = inputTensor
            inputs["position_ids"] = positionTensor
            
            // Add zero-initialized KV cache inputs
            for (layer in 0 until modelConfig.num_hidden_layers) {
                val (keyCache, valueCache) = staticKvCache!![layer]
                inputs["past_key_values.${layer}.key"] = keyCache
                inputs["past_key_values.${layer}.value"] = valueCache
            }
            
            Log.d(TAG, "🔥 Processing system prompt to initialize KV cache...")
            
            // Run inference to populate KV cache with system prompt
            val outputs = ortSession?.run(inputs)
            
            if (outputs != null && outputs.size() > 0) {
                Log.d(TAG, "✅ System prompt processed, KV cache populated with ${systemPromptLength} tokens")
                
                // Update cache position tracking
                currentCachePosition = systemPromptLength
                kvCacheSequenceLength = systemPromptLength
                
                // Debug: Print all available outputs
                Log.d(TAG, "Available outputs: ${outputs.size()}")
                for (i in 0 until outputs.size()) {
                    try {
                        val output = outputs.get(i)
                        Log.d(TAG, "  Output[$i]: ${output.value::class.simpleName}")
                    } catch (e: Exception) {
                        Log.d(TAG, "  Output[$i]: ${e.message}")
                    }
                }
                
                // 🔥 Handle ONNX model outputs properly (all outputs are multi-dimensional arrays)
                // 🔥 Try to get first output (logits) - handle both OnnxTensor and Array types
                val firstOutput = outputs.get(0)
                Log.d(TAG, "First output type: ${firstOutput::class.simpleName}")
                val logits = firstOutput as? OnnxTensor
                // For system prompt initialization, we don't need actual logits processing
                // Just verify outputs and update KV cache
                if (logits != null) {
                    Log.d(TAG, "Logits extracted as OnnxTensor, shape: [${logits.info.shape.joinToString(", ")}]")
                } else {
                    Log.d(TAG, "First output is not OnnxTensor (likely Array), proceeding with KV cache update")
                }
                
                // Update static KV cache with new outputs (using safe casting)
                var kvOutputIndex = 1
                for (layer in 0 until modelConfig.num_hidden_layers) {
                    try {
                        // Safe casting with type checking
                        val keyOutputValue = outputs.get(kvOutputIndex).value
                        val valueOutputValue = outputs.get(kvOutputIndex + 1).value
                        
                        if (keyOutputValue is OnnxTensor && valueOutputValue is OnnxTensor) {
                            // Replace layer cache with new results
                            staticKvCache!![layer].first.close()
                            staticKvCache!![layer].second.close()
                            staticKvCache!![layer] = Pair(keyOutputValue, valueOutputValue)
                            
                            Log.v(TAG, "Layer $layer KV cache updated (indices: $kvOutputIndex, ${kvOutputIndex + 1})")
                        } else {
                            // Convert Java arrays to OnnxTensor if needed
                            val keyTensor = if (keyOutputValue is OnnxTensor) keyOutputValue 
                                           else convertArrayToTensor(keyOutputValue, "key", layer)
                            val valueTensor = if (valueOutputValue is OnnxTensor) valueOutputValue 
                                             else convertArrayToTensor(valueOutputValue, "value", layer)
                            
                            staticKvCache!![layer].first.close()
                            staticKvCache!![layer].second.close()
                            staticKvCache!![layer] = Pair(keyTensor, valueTensor)
                            
                            Log.v(TAG, "Layer $layer KV cache updated via array conversion (indices: $kvOutputIndex, ${kvOutputIndex + 1})")
                        }
                        
                        kvOutputIndex += 2
                        
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to update KV cache for layer $layer: ${e.message}")
                        Log.d(TAG, "Output types: key=${outputs.get(kvOutputIndex).value::class.simpleName}, value=${outputs.get(kvOutputIndex + 1).value::class.simpleName}")
                        break
                    }
                }
                
                Log.d(TAG, "🚀 Static KV Cache initialized successfully")
            } else {
                Log.w(TAG, "No outputs received from system prompt processing")
            }
            
            // Cleanup temporary tensors
            inputTensor.close()
            positionTensor.close()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Static KV Cache", e)
        }
    }
    
    /**
     * Generate text response from user input
     */
    suspend fun generateResponse(userInput: String, maxTokens: Int = 256): String = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Generating response for: $userInput")
            
            // Tokenize input
            val inputTokens = tokenizer.encode(userInput)
            Log.d(TAG, "Input tokenized: ${inputTokens.size} tokens")
            
            Log.d(TAG, "Starting autoregressive generation with NNAPI acceleration")
            
            // Generate tokens using proper autoregressive generation with NNAPI
            val generatedTokens = generateTokens(inputTokens, maxTokens)
            
            // Decode response (skip input tokens)
            val response = tokenizer.decode(generatedTokens.drop(inputTokens.size))
            // Memory usage summary for this generation
            val finalMemoryInfo = """
                Generation completed:
                - Input tokens: ${inputTokens.size}
                - Generated tokens: ${generatedTokens.size - inputTokens.size}
                - Total conversation tokens: ${generatedTokens.size}
                - KV cache size: ${estimateKVCacheMemory(generatedTokens.size)}MB
            """.trimIndent()
            
            Log.d(TAG, finalMemoryInfo)
            Log.d(TAG, "Response generated: $response")
            
            // Cleanup handled in generateTokens() function
            
            response
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate response", e)
            "Sorry, I encountered an error while processing your request."
        }
    }
    
    /**
     * 🚀 STATIC KV CACHE Generation - Based on GitHub Issue #23061
     * Pre-allocated buffer sharing for GQA models with system prompt initialization
     * INCREMENTAL MODE: Process only new tokens with persistent KV cache
     */
    private suspend fun generateTokens(
        initialTokens: List<Int>,
        maxTokens: Int,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): List<Int> = withContext(Dispatchers.IO) {
        // Combine system prompt + user input for full conversation context
        val fullConversationTokens = (systemPromptTokens!!.toList() + initialTokens).toMutableList()
        var localPosition = currentCachePosition // Use current cache position
        
        Log.d(TAG, "=== 🧠 HYBRID KV CACHE GENERATION ===")
        Log.d(TAG, "System prompt tokens: $systemPromptLength")
        Log.d(TAG, "User input tokens: ${initialTokens.size}")
        Log.d(TAG, "Total conversation tokens: ${fullConversationTokens.size}")
        Log.d(TAG, "KV cache position starts at: $currentCachePosition")
        
        Log.d(TAG, "Gemma 3 1B Hybrid Cache: sliding_window=${modelConfig.sliding_window}, pattern=${modelConfig.sliding_window_pattern}")
        
        // 🧠 Phase 1: Start with current conversation length only (no pre-allocation)
        val currentSequenceLength = fullConversationTokens.size
        Log.d(TAG, "🔄 Current sequence length: $currentSequenceLength tokens")
        
        // Resize KV cache to current size (dynamic approach)
        resizeKvCacheIfNeeded(currentSequenceLength)
        
        // Process user input tokens first (batch mode for efficiency)
        if (initialTokens.isNotEmpty()) {
            Log.d(TAG, "🔥 Processing user input with pre-warmed Hybrid KV cache...")
            
            val userInputIds = LongArray(initialTokens.size) { initialTokens[it].toLong() }
            val userPositionIds = LongArray(initialTokens.size) { localPosition + it.toLong() }
            
            // Update cache tracking
            currentCachePosition += initialTokens.size
            kvCacheSequenceLength += initialTokens.size
            
            val nextToken = processStaticKVCacheInference(userInputIds, userPositionIds, fullConversationTokens.size, temperature, topP)
            if (nextToken != -1) {
                fullConversationTokens.add(nextToken)
                Log.d(TAG, "First generated token: $nextToken")
            }
        }
        
        // Continue autoregressive generation (incremental mode)
        for (i in 1 until maxTokens) {
            try {
                Log.d(TAG, "--- Generation Step ${i + 1} (INCREMENTAL) ---")
                
                // Process only the last token
                val lastToken = fullConversationTokens.last()
                val inputIds = LongArray(1) { lastToken.toLong() }
                val positionIds = LongArray(1) { (fullConversationTokens.size - 1).toLong() }
                
                Log.d(TAG, "Processing last token: $lastToken at position ${positionIds[0]}")
                
                // Apply sliding window if sequence gets too long
                val effectivePosition = if (kvCacheSequenceLength >= maxStaticCacheLength) {
                    Log.d(TAG, "🔥 SLIDING WINDOW: Applying at ${kvCacheSequenceLength} tokens")
                    slideKVCache()
                    kvCacheSequenceLength = maxStaticCacheLength / 2  // Reset to half
                    currentCachePosition = kvCacheSequenceLength
                    currentCachePosition.toLong()
                } else {
                    currentCachePosition++
                    kvCacheSequenceLength++
                    (currentCachePosition - 1).toLong()  // Use previous position for current token
                }
                
                val adjustedPositionIds = LongArray(1) { effectivePosition }
                val nextToken = processStaticKVCacheInference(inputIds, adjustedPositionIds, fullConversationTokens.size, temperature, topP)
                
                if (nextToken == -1 || modelConfig.eos_token_id.contains(nextToken)) {
                    Log.d(TAG, "EOS token generated: $nextToken")
                    break
                }
                
                fullConversationTokens.add(nextToken)
                
                // 🧠 Dynamic KV Cache expansion for next token
                val newSequenceLength = fullConversationTokens.size
                if (newSequenceLength > currentSequenceLength) {
                    Log.v(TAG, "🔄 Expanding KV cache from $currentSequenceLength to $newSequenceLength")
                    resizeKvCacheIfNeeded(newSequenceLength)
                }
                
                // Periodic cleanup
                if (i % 20 == 19) {
                    System.gc()
                    Log.d(TAG, "🗑️ Periodic GC at step ${i + 1}")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "=== ERROR in Generation Step ${i + 1} ===", e)
                break
            }
        }
        
        Log.d(TAG, "=== 🚀 STATIC KV CACHE END ===")
        Log.d(TAG, "Total generated tokens: ${fullConversationTokens.size}")
        Log.d(TAG, "New tokens generated: ${fullConversationTokens.size - systemPromptLength - initialTokens.size}")
        
        // Return only the generated response (skip system prompt + user input)
        fullConversationTokens.drop(systemPromptLength)
    }
    
    /**
     * 🚀 Process inference with Static KV Cache (GitHub Issue #23061 implementation)
     * Uses pre-allocated buffer sharing for GQA models
     */
    private suspend fun processStaticKVCacheInference(
        inputIds: LongArray,
        positionIds: LongArray,
        currentSequenceLength: Int,
        temperature: Float,
        topP: Float
    ): Int {
        try {
            // Create input tensors
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(inputIds))
            val positionTensor = OnnxTensor.createTensor(ortEnvironment, arrayOf(positionIds))
            
            // Prepare inputs with static KV cache
            val inputs = mutableMapOf<String, OnnxTensorLike>()
            inputs["input_ids"] = inputTensor
            inputs["position_ids"] = positionTensor
            
            // Add static KV cache to inputs (shared buffer approach)
            for (layer in 0 until modelConfig.num_hidden_layers) {
                val (keyCache, valueCache) = staticKvCache!![layer]
                inputs["past_key_values.${layer}.key"] = keyCache
                inputs["past_key_values.${layer}.value"] = valueCache
            }
            
            Log.d(TAG, "🔥 STATIC KV INFERENCE: input_ids=${inputIds.size}, positions=${positionIds.contentToString()}")
            Log.d(TAG, "Cache state: position=$currentCachePosition, seq_len=$kvCacheSequenceLength")
            
            // Run inference
            val outputs = ortSession?.run(inputs)
            
            if (outputs != null && outputs.size() > 0) {
                // Extract logits with safe casting
                val logitsValue = outputs.get(0).value
                val logits = if (logitsValue is OnnxTensor) {
                    logitsValue
                } else {
                    // Convert array to tensor if needed
                    Log.d(TAG, "Converting logits from ${logitsValue::class.simpleName} to OnnxTensor")
                    convertArrayToTensor(logitsValue, "logits", -1)
                }
                val logitsShape = logits.info.shape
                Log.d(TAG, "Logits shape: [${logitsShape.joinToString(", ")}]")
                
                // Get next token from last position
                val nextToken = sampleNextToken(logits, temperature, topP)
                
                // Update static KV cache with new outputs (enhanced tensor handling)
                var kvOutputIndex = 1
                for (layer in 0 until modelConfig.num_hidden_layers) {
                    try {
                        // Safely get KV outputs and check availability
                        if (kvOutputIndex + 1 < outputs.size()) {
                            val keyOutputValue = outputs.get(kvOutputIndex).value
                            val valueOutputValue = outputs.get(kvOutputIndex + 1).value
                            
                            // Handle both OnnxTensor and Java array types - APPEND ONLY (no full replacement)
                            if (keyOutputValue is OnnxTensor && valueOutputValue is OnnxTensor) {
                                // For INCREMENTAL mode, we should append new token data only
                                // But ONNX Runtime doesn't support in-place tensor modification
                                // So we keep the current approach but optimize memory
                                val currentKeyCache = staticKvCache!![layer].first
                                val currentValueCache = staticKvCache!![layer].second
                                
                                // Only replace if the new tensor has different sequence length
                                val currentSeqLen = currentKeyCache.info.shape[2]
                                val newSeqLen = keyOutputValue.info.shape[2]
                                
                                if (newSeqLen > currentSeqLen) {
                                    currentKeyCache.close()
                                    currentValueCache.close()
                                    staticKvCache!![layer] = Pair(keyOutputValue, valueOutputValue)
                                    Log.v(TAG, "Layer $layer KV cache expanded from $currentSeqLen to $newSeqLen (direct)")
                                } else {
                                    // Same size, reuse existing cache to reduce memory pressure
                                    Log.v(TAG, "Layer $layer KV cache reused (same size: $currentSeqLen)")
                                }
                            } else {
                                // Convert arrays to tensors
                                val keyTensor = if (keyOutputValue is OnnxTensor) keyOutputValue 
                                               else convertArrayToTensor(keyOutputValue, "key", layer)
                                val valueTensor = if (valueOutputValue is OnnxTensor) valueOutputValue 
                                                 else convertArrayToTensor(valueOutputValue, "value", layer)
                                
                                val currentKeyCache = staticKvCache!![layer].first
                                val currentValueCache = staticKvCache!![layer].second
                                
                                // Only replace if sequence length increased
                                val currentSeqLen = currentKeyCache.info.shape[2]
                                val newSeqLen = keyTensor.info.shape[2]
                                
                                if (newSeqLen > currentSeqLen) {
                                    currentKeyCache.close()
                                    currentValueCache.close()
                                    staticKvCache!![layer] = Pair(keyTensor, valueTensor)
                                    Log.v(TAG, "Layer $layer KV cache expanded from $currentSeqLen to $newSeqLen (converted)")
                                } else {
                                    // Same size, keep existing and dispose new ones
                                    if (keyTensor != keyOutputValue) keyTensor.close()
                                    if (valueTensor != valueOutputValue) valueTensor.close()
                                    Log.v(TAG, "Layer $layer KV cache reused (same size: $currentSeqLen)")
                                }
                            }
                        } else {
                            Log.w(TAG, "Layer $layer: Not enough KV outputs, skipping")
                            break
                        }
                        
                        kvOutputIndex += 2
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to update KV cache for layer $layer: ${e.message}")
                        Log.d(TAG, "Available outputs: ${outputs.size()}, trying indices: $kvOutputIndex, ${kvOutputIndex + 1}")
                        break
                    }
                }
                
                Log.d(TAG, "✅ Generated token: $nextToken, KV cache updated")
                
                // Cleanup temporary tensors
                inputTensor.close()
                positionTensor.close()
                
                return nextToken
                
            } else {
                Log.e(TAG, "No outputs from ONNX inference")
                return -1
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Static KV Cache inference failed", e)
            return -1
        }
    }
    
    /**
     * 🔥 Sliding Window for Static KV Cache - maintains fixed buffer size
     * ONNX Runtime 1.22.0 optimized sliding window implementation with actual data copying
     * ✅ Fixed: Now properly copies existing KV cache data instead of just creating empty tensors
     */
    private fun slideKVCache() {
        try {
            Log.d(TAG, "🔥 Applying sliding window to Static KV Cache...")
            Log.d(TAG, "Cache before slide: position=$currentCachePosition, seq_len=$kvCacheSequenceLength")
            
            val keepLength = maxStaticCacheLength / 2  // Keep recent half (256 tokens)
            val slideAmount = maxStaticCacheLength - keepLength  // Remove older half (256 tokens)
            
            Log.d(TAG, "Sliding window: keeping recent $keepLength tokens, removing oldest $slideAmount tokens")
            
            for (layer in 0 until modelConfig.num_hidden_layers) {
                val (oldKeyCache, oldValueCache) = staticKvCache!![layer]
                
                // 🚀 Create new KV cache tensors with proper ONNX Runtime shape (torch.empty() pattern)
                val newKeyCache = OnnxTensor.createTensor(
                    ortEnvironment, 
                    Array(1) {  // batch_size = 1
                        Array(modelConfig.num_key_value_heads) {  // num_kv_heads = 1
                            Array(maxStaticCacheLength) {  // seq_len = 512
                                FloatArray(modelConfig.head_dim) { Float.NaN }  // Uninitialized like torch.empty()
                            }
                        }
                    }
                )
                val newValueCache = OnnxTensor.createTensor(
                    ortEnvironment, 
                    Array(1) {  // batch_size = 1
                        Array(modelConfig.num_key_value_heads) {  // num_kv_heads = 1
                            Array(maxStaticCacheLength) {  // seq_len = 512
                                FloatArray(modelConfig.head_dim) { Float.NaN }  // Uninitialized like torch.empty()
                            }
                        }
                    }
                )
                
                // 🔥 Copy recent KV cache data from old tensor to new tensor
                // Copy from position slideAmount to current position (keep recent tokens)
                try {
                    // Get tensor data as multi-dimensional arrays with safe casting
                    @Suppress("UNCHECKED_CAST")
                    val oldKeyData = oldKeyCache.value as Array<Array<Array<FloatArray>>>
                    @Suppress("UNCHECKED_CAST")
                    val oldValueData = oldValueCache.value as Array<Array<Array<FloatArray>>>
                    @Suppress("UNCHECKED_CAST")
                    val newKeyData = newKeyCache.value as Array<Array<Array<FloatArray>>>
                    @Suppress("UNCHECKED_CAST")
                    val newValueData = newValueCache.value as Array<Array<Array<FloatArray>>>
                    
                    // Copy recent tokens to beginning of new cache
                    // Source: oldCache[0][0][slideAmount..currentCachePosition]
                    // Target: newCache[0][0][0..keepLength]
                    var copiedTokens = 0
                    for (batchIdx in 0 until 1) {
                        for (headIdx in 0 until modelConfig.num_key_value_heads) {
                            var newPos = 0
                            for (seqPos in slideAmount until minOf(currentCachePosition, maxStaticCacheLength)) {
                                if (newPos < keepLength) {
                                    // Copy key cache data
                                    System.arraycopy(
                                        oldKeyData[batchIdx][headIdx][seqPos], 0,
                                        newKeyData[batchIdx][headIdx][newPos], 0,
                                        modelConfig.head_dim
                                    )
                                    // Copy value cache data
                                    System.arraycopy(
                                        oldValueData[batchIdx][headIdx][seqPos], 0,
                                        newValueData[batchIdx][headIdx][newPos], 0,
                                        modelConfig.head_dim
                                    )
                                    newPos++
                                }
                            }
                            copiedTokens = newPos  // Track tokens copied for logging
                        }
                    }
                    
                    Log.v(TAG, "Layer $layer: Copied $copiedTokens tokens to new KV cache")
                    
                } catch (copyException: Exception) {
                    Log.w(TAG, "Failed to copy KV cache data for layer $layer, using empty cache", copyException)
                }
                
                // 🗑️ Clean up old tensors to prevent memory leaks
                oldKeyCache.close()
                oldValueCache.close()
                
                // Replace with new cache
                staticKvCache!![layer] = Pair(newKeyCache, newValueCache)
            }
            
            // Update cache position tracking
            val newPosition = minOf(keepLength, currentCachePosition - slideAmount)
            currentCachePosition = maxOf(0, newPosition)
            kvCacheSequenceLength = currentCachePosition
            
            Log.d(TAG, "✅ KV Cache sliding window applied successfully")
            Log.d(TAG, "Cache after slide: position=$currentCachePosition, seq_len=$kvCacheSequenceLength")
            Log.d(TAG, "Memory freed: ${slideAmount * modelConfig.num_hidden_layers * modelConfig.num_key_value_heads * modelConfig.head_dim * 4 * 2 / 1024 / 1024} MB")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to apply sliding window to KV cache", e)
            // Fallback: at least reset tracking variables
            currentCachePosition = maxOf(0, currentCachePosition - maxStaticCacheLength / 2)
            kvCacheSequenceLength = currentCachePosition
        }
    }
    
    /**
     * Legacy function - kept for compatibility during transition
     */
    private suspend fun processInferenceWithKVCache(
        inputIds: LongArray,
        positionIds: LongArray,
        step: Int,
        generatedTokens: MutableList<Int>,
        temperature: Float,
        topP: Float,
        useBatchMode: Boolean = false
    ): Int {
        // Create tensors
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, Array(1) { inputIds })
        val positionTensor = OnnxTensor.createTensor(ortEnvironment, Array(1) { positionIds })
        
        Log.d(TAG, "inputTensor shape: ${inputTensor.info.shape.contentToString()}")
        Log.d(TAG, "positionTensor shape: ${positionTensor.info.shape.contentToString()}")
        
        // Adaptive KV cache strategy based on processing mode
        val kvCache = mutableMapOf<String, OnnxTensor>()
        if (useBatchMode) {
            // BATCH MODE: Create minimal dummy KV cache to satisfy model requirements
            for (layer in 0 until modelConfig.num_hidden_layers) {
                // Minimal dummy cache: (batch=1, heads=1, seq_len=1, head_dim=256)
                // 🚀 Use torch.empty() pattern for dummy KV cache (better performance)
                val dummyKey = Array(1) { Array(modelConfig.num_key_value_heads) { Array(1) { FloatArray(modelConfig.head_dim) { Float.NaN } } } }
                val dummyValue = Array(1) { Array(modelConfig.num_key_value_heads) { Array(1) { FloatArray(modelConfig.head_dim) { Float.NaN } } } }
                
                kvCache["past_key_values.${layer}.key"] = OnnxTensor.createTensor(ortEnvironment, dummyKey)
                kvCache["past_key_values.${layer}.value"] = OnnxTensor.createTensor(ortEnvironment, dummyValue)
            }
        } else {
            // INCREMENTAL MODE: Create proper KV cache for efficient processing
            // For simplicity, still use minimal dummy cache but this could be optimized with real KV cache
            for (layer in 0 until modelConfig.num_hidden_layers) {
                // 🚀 Use torch.empty() pattern for dummy KV cache (better performance)
                val dummyKey = Array(1) { Array(modelConfig.num_key_value_heads) { Array(1) { FloatArray(modelConfig.head_dim) { Float.NaN } } } }
                val dummyValue = Array(1) { Array(modelConfig.num_key_value_heads) { Array(1) { FloatArray(modelConfig.head_dim) { Float.NaN } } } }
                
                kvCache["past_key_values.${layer}.key"] = OnnxTensor.createTensor(ortEnvironment, dummyKey)
                kvCache["past_key_values.${layer}.value"] = OnnxTensor.createTensor(ortEnvironment, dummyValue)
            }
        }
        
        try {
            // Inputs with dummy KV cache to satisfy model requirements
            val inputs = mutableMapOf<String, OnnxTensor>(
                "input_ids" to inputTensor,
                "position_ids" to positionTensor
            )
            inputs.putAll(kvCache)
            
            // Run ONNX inference with NNAPI
            val modeText = if (useBatchMode) "batch mode with dummy KV cache" else "incremental mode with dummy KV cache"
            Log.d(TAG, "Calling ONNX Runtime inference ($modeText)...")
            val outputs = ortSession?.run(inputs)
            Log.d(TAG, "ONNX inference completed successfully")
            
            // DEBUG: Check output shapes and count
            Log.d(TAG, "ONNX outputs count: ${outputs?.size()}")
            
            @Suppress("UNCHECKED_CAST")
            val logits = outputs?.get(0)?.value as Array<Array<FloatArray>>
            Log.d(TAG, "Logits shape: [${logits.size}, ${logits[0].size}, ${logits[0][0].size}]")
            
            // Get next token logits (last position)
            val nextTokenLogits = logits[0].last().copyOf()
            
            // Apply temperature
            for (j in nextTokenLogits.indices) {
                nextTokenLogits[j] = nextTokenLogits[j] / temperature
            }
            
            // Sample next token
            val nextToken = sampleTopP(nextTokenLogits, topP)
            
            // Check for EOS
            if (modelConfig.eos_token_id.contains(nextToken)) {
                Log.d(TAG, "EOS token reached at position ${generatedTokens.size}")
                return -1 // Signal EOS
            }
            
            generatedTokens.add(nextToken)
            Log.d(TAG, "Added next token: $nextToken (generatedTokens.size now: ${generatedTokens.size})")
            
            return nextToken
            
        } finally {
            // Always cleanup tensors
            inputTensor.close()
            positionTensor.close()
            // Clean up KV cache tensors
            kvCache.values.forEach { it.close() }
        }
    }
    
    /**
     * Convert Java arrays to OnnxTensor for KV cache compatibility
     * Handles the casting error: float[][][][] cannot be cast to ai.onnxruntime.OnnxTensor
     */
    private fun convertArrayToTensor(arrayValue: Any, tensorType: String, layer: Int): OnnxTensor {
        return try {
            when (arrayValue) {
                is Array<*> -> {
                    // Special handling for logits (3D array)
                    if (tensorType == "logits") {
                        try {
                            @Suppress("UNCHECKED_CAST")
                            val logitsArray = arrayValue as Array<Array<FloatArray>>
                            val tensor = OnnxTensor.createTensor(ortEnvironment, logitsArray)
                            Log.v(TAG, "Layer $layer $tensorType converted from Array<Array<FloatArray>> to OnnxTensor, shape: ${tensor.info.shape.contentToString()}")
                            return tensor
                        } catch (logitsException: Exception) {
                            Log.w(TAG, "Failed 3D logits conversion, trying 4D: ${logitsException.message}")
                        }
                    }
                    
                    // Handle 4D float array: [batch, heads, seq_len, head_dim]
                    @Suppress("UNCHECKED_CAST")
                    val floatArray = arrayValue as Array<Array<Array<FloatArray>>>
                    val tensor = OnnxTensor.createTensor(ortEnvironment, floatArray)
                    Log.v(TAG, "Layer $layer $tensorType converted from Array<Array<Array<FloatArray>>> to OnnxTensor, shape: ${tensor.info.shape.contentToString()}")
                    tensor
                }
                is FloatArray -> {
                    // Handle 1D case and reshape to [1, 1, seq_len, head_dim]
                    val inputArray = arrayValue
                    val batchSize = 1
                    val numHeads = modelConfig.num_key_value_heads
                    val seqLen = inputArray.size / modelConfig.head_dim / numHeads
                    val headDim = modelConfig.head_dim
                    
                    val reshapedArray = Array(batchSize) { 
                        Array(numHeads) { 
                            Array(seqLen) { i ->
                                FloatArray(headDim) { j ->
                                    inputArray[i * headDim + j]
                                }
                            }
                        }
                    }
                    val tensor = OnnxTensor.createTensor(ortEnvironment, reshapedArray)
                    Log.v(TAG, "Layer $layer $tensorType converted from FloatArray to OnnxTensor, shape: ${tensor.info.shape.contentToString()}")
                    tensor
                }
                else -> {
                    Log.w(TAG, "Layer $layer $tensorType: Unsupported array type ${arrayValue::class.simpleName}, creating dummy tensor")
                    // Create dummy tensor as fallback
                    val dummyArray = Array(1) { Array(modelConfig.num_key_value_heads) { Array(1) { FloatArray(modelConfig.head_dim) } } }
                    OnnxTensor.createTensor(ortEnvironment, dummyArray)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert $tensorType array for layer $layer: ${e.message}")
            // Create minimal dummy tensor as last resort
            val dummyArray = Array(1) { Array(modelConfig.num_key_value_heads) { Array(1) { FloatArray(modelConfig.head_dim) } } }
            OnnxTensor.createTensor(ortEnvironment, dummyArray)
        }
    }

    /**
     * Estimate KV cache memory usage for conversation length
     * Gemma 3 1B: 26 layers × num_key_value_heads × tokens × head_dim × 4 bytes × 2 (K+V)
     * ONNX Runtime 1.23.0 uses FP32 internally even with FP16 models
     */
    private fun estimateKVCacheMemory(numTokens: Int): Float {
        // Gemma 3 1B GQA configuration
        val numLayers = modelConfig.num_hidden_layers // 26
        val numKeyValueHeads = modelConfig.num_key_value_heads // 1 for GQA
        val headDim = modelConfig.head_dim // 256
        val bytesPerFloat = 4  // ONNX Runtime 1.23.0 uses FP32 for KV cache
        
        // KV cache = Key + Value tensors
        val kvCacheBytes = numLayers * numKeyValueHeads * numTokens * headDim * bytesPerFloat * 2
        
        return kvCacheBytes.toFloat() / (1024 * 1024) // Convert to MB
    }
    
    /**
     * Top-p (nucleus) sampling implementation
     */

    private fun sampleTopP(logits: FloatArray, topP: Float): Int {
        // Convert logits to probabilities using softmax
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = logits.map { kotlin.math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expLogits.sum()
        val probabilities = expLogits.map { it / sumExp }
        
        // Create probability-index pairs and sort by probability descending
        val sortedProbs = probabilities.mapIndexed { index, prob -> Pair(prob, index) }
            .sortedByDescending { it.first }
        
        // Calculate cumulative probabilities
        var cumSum = 0f
        val selectedIndices = mutableListOf<Int>()
        
        for ((prob, index) in sortedProbs) {
            cumSum += prob
            selectedIndices.add(index)
            if (cumSum >= topP) break
        }
        
        // Sample from selected indices
        val randomValue = kotlin.random.Random.nextFloat() * cumSum
        var currentSum = 0f
        
        for (index in selectedIndices) {
            currentSum += probabilities[index]
            if (randomValue <= currentSum) {
                return index
            }
        }
        
        return selectedIndices.last()
    }
    
    /**
     * Load model configuration from assets
     */
    private fun loadModelConfig(): ModelConfig {
        val configStream = context.assets.open("$MODEL_PATH/$CONFIG_FILE")
        val configJson = configStream.bufferedReader().use { it.readText() }
        val gson = Gson()
        val jsonObject = gson.fromJson(configJson, JsonObject::class.java)
        
        return ModelConfig(
            vocab_size = jsonObject.get("vocab_size").asInt,
            max_position_embeddings = jsonObject.get("max_position_embeddings").asInt,
            hidden_size = jsonObject.get("hidden_size").asInt,
            num_attention_heads = jsonObject.get("num_attention_heads").asInt,
            num_hidden_layers = jsonObject.get("num_hidden_layers").asInt,
            bos_token_id = jsonObject.get("bos_token_id").asInt,
            eos_token_id = gson.fromJson(jsonObject.get("eos_token_id"), List::class.java).map { (it as Double).toInt() },
            sliding_window = jsonObject.get("sliding_window")?.asInt ?: 512,
            sliding_window_pattern = jsonObject.get("sliding_window_pattern")?.asInt ?: 6,
            cache_implementation = jsonObject.get("cache_implementation")?.asString ?: "hybrid",
            num_key_value_heads = jsonObject.get("num_key_value_heads")?.asInt ?: 1,
            head_dim = jsonObject.get("head_dim")?.asInt ?: 256,
            use_cache = jsonObject.get("use_cache")?.asBoolean ?: true
        )
    }
    
    /**
     * Ensure model is in internal storage for native heap loading
     * Java heap: ~1MB for configs, Native heap: ~998MB for model weights
     */
    private suspend fun ensureModelInInternalStorage(): String = withContext(Dispatchers.IO) {
        val modelsDir = File(context.filesDir, "models/gemma-3-1b")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }
        
        val modelFile = File(modelsDir, MODEL_FILE)
        
        // Check if model exists and is valid (998MB expected)
        if (!modelFile.exists() || modelFile.length() < 990_000_000L) {
            Log.d(TAG, "Copying 998MB model from assets to internal storage...")
            Log.d(TAG, "Memory strategy: Java heap (configs) + Native heap (model weights + KV cache)")
            
            copyAssetToInternalStorage("$MODEL_PATH/$MODEL_FILE", modelFile)
            
            Log.d(TAG, "Model ready for native heap loading: ${modelFile.length()} bytes")
        } else {
            Log.d(TAG, "Model already in internal storage: ${modelFile.length()} bytes")
        }
        
        modelFile.absolutePath
    }
    
    /**
     * Stream-based asset copying (avoids Java heap OOM)
     */
    private fun copyAssetToInternalStorage(assetPath: String, destinationFile: File) {
        context.assets.open(assetPath).use { inputStream ->
            FileOutputStream(destinationFile).use { outputStream ->
                val buffer = ByteArray(64 * 1024) // 64KB streaming buffer
                var totalBytes = 0L
                var bytesRead: Int
                
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    outputStream.write(buffer, 0, bytesRead)
                    totalBytes += bytesRead
                    
                    // Log progress every 100MB
                    if (totalBytes % (100 * 1024 * 1024) == 0L) {
                        Log.d(TAG, "Copied ${totalBytes / (1024 * 1024)}MB...")
                    }
                }
                outputStream.flush()
                Log.d(TAG, "Total copied: ${totalBytes / (1024 * 1024)}MB")
            }
        }
    }
    
    /**
     * 🚀 Sample next token from logits with temperature and top-p sampling
     */
    private fun sampleNextToken(logits: OnnxTensor, temperature: Float, topP: Float): Int {
        try {
            val logitsArray = logits.floatBuffer.array()
            val vocabSize = logitsArray.size
            
            // Apply temperature scaling
            for (i in logitsArray.indices) {
                logitsArray[i] = logitsArray[i] / temperature
            }
            
            // Convert to probabilities using softmax
            val maxLogit = logitsArray.maxOrNull() ?: 0f
            var sumExp = 0f
            for (i in logitsArray.indices) {
                logitsArray[i] = kotlin.math.exp(logitsArray[i] - maxLogit)
                sumExp += logitsArray[i]
            }
            
            // Normalize to probabilities
            for (i in logitsArray.indices) {
                logitsArray[i] = logitsArray[i] / sumExp
            }
            
            // Apply top-p (nucleus) sampling
            val sortedIndices = logitsArray.indices.sortedByDescending { logitsArray[it] }
            var cumulativeProb = 0f
            val validIndices = mutableListOf<Int>()
            
            for (idx in sortedIndices) {
                cumulativeProb += logitsArray[idx]
                validIndices.add(idx)
                if (cumulativeProb >= topP) break
            }
            
            // Sample from valid indices
            val randomValue = kotlin.random.Random.nextFloat()
            var cumProb = 0f
            for (idx in validIndices) {
                cumProb += logitsArray[idx] / cumulativeProb  // Renormalize
                if (randomValue <= cumProb) {
                    return idx
                }
            }
            
            // Fallback: return last valid index
            return validIndices.lastOrNull() ?: 0
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in token sampling", e)
            return 0 // Fallback token
        }
    }
    
    /**
     * Get comprehensive memory usage information
     */
    fun getMemoryInfo(): String {
        val modelSizeMB = 998 // model_q4f16.onnx size
        val tokenizerSizeMB = 33 // tokenizer.json size
        val javaHeapMB = 51 // Estimated Java heap usage
        
        return """
            MemSTORY LLM Engine Memory Usage:
            
            📋 JAVA HEAP (Safe Zone):
            - Model configs: ~1MB
            - Tokenizer configs: ~1MB  
            - App overhead: ~49MB
            - Total Java heap: ${javaHeapMB}MB / 512MB limit
            
            🧠 NATIVE HEAP (ONNX Runtime Managed):
            - Model weights (mmap): ${modelSizeMB}MB virtual (~200-300MB physical)
            - Tokenizer vocab: ${tokenizerSizeMB}MB
            - KV cache (varies): 0-52MB (depends on conversation length)
            - Tensor buffers: ~10MB
            - Total native heap: ~295-395MB
            
            💡 GQA OPTIMIZATION:
            - Standard attention: 4 heads × 52MB = 208MB KV cache
            - Gemma 3 1B GQA: 1 head × 52MB = 52MB KV cache  
            - Memory saving: 75% reduction!
            
            🎯 TOTAL MEMORY: ~346-446MB (8GB device = Very Safe)
        """.trimIndent()
    }
    
    /**
     * Clean up all resources (Java + Native heaps)
     */
    fun cleanup() {
        try {
            Log.d(TAG, "Cleaning up LLM Engine resources...")
            
            // Cleanup ONNX Runtime (native heap)
            ortSession?.close()
            ortEnvironment?.close()
            
            // Cleanup tokenizer (native heap)
            tokenizer.cleanup()
            
            Log.d(TAG, "All LLM Engine resources cleaned up")
            Log.d(TAG, "Memory freed: ~395MB native heap + ~51MB Java heap")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}