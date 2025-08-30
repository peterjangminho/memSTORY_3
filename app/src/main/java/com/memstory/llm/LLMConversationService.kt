package com.memstory.llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import org.koin.core.component.KoinComponent
import org.koin.core.component.inject

/**
 * Conversation message data class
 */
data class ConversationMessage(
    val content: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * LLM response with generation info
 */
data class LLMResponse(
    val content: String,
    val tokensGenerated: Int,
    val generationTimeMs: Long,
    val isComplete: Boolean = true
)

/**
 * LLM Conversation Service with Koin DI integration
 * Manages conversation state and handles text generation
 */
class LLMConversationService(
    private val context: Context
) : KoinComponent {
    
    private val llmEngine: OnnxLLMEngine by inject()
    private var isInitialized = false
    private var isGenerating = false
    private val conversationHistory = mutableListOf<ConversationMessage>()
    
    companion object {
        private const val TAG = "LLMConversationService"
        
        // Conversation settings
        private const val MAX_CONVERSATION_LENGTH = 8192
        private const val MAX_RESPONSE_TOKENS = 512
        private const val TEMPERATURE = 0.7f
        private const val TOP_P = 0.9f
        
        // Prompt templates
        private const val SYSTEM_PROMPT = """You are memSTORY, a helpful AI assistant that lives entirely on the user's phone. 
You help users organize their thoughts, memories, and ideas. Always be friendly, concise, and helpful.
Remember that you work completely offline - you have no internet connection."""

        private const val USER_PREFIX = "User: "
        private const val ASSISTANT_PREFIX = "memSTORY: "
    }
    
    /**
     * Initialize the LLM service
     */
    suspend fun initialize(): Boolean {
        if (isInitialized) return true
        
        return withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing LLM Conversation Service...")
                val success = llmEngine.initialize()
                if (success) {
                    isInitialized = true
                    Log.d(TAG, "LLM Service initialized successfully")
                } else {
                    Log.e(TAG, "Failed to initialize LLM engine")
                }
                success
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing LLM service", e)
                false
            }
        }
    }

    
    /**
     * 🚀 Warm up the LLM service in background (non-blocking)
     * This triggers the full initialization pipeline:
     * APK assets → internal storage copy → native heap loading
     * Called from MainActivity.onCreate() for instant user experience
     */
    fun warmUp() {
        if (isInitialized) {
            Log.d(TAG, "LLM service already initialized")
            return
        }
        
        // Launch initialization in background coroutine
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d(TAG, "🚀 Starting background LLM warm-up...")
                val startTime = System.currentTimeMillis()
                
                val success = initialize()
                
                val warmUpTime = System.currentTimeMillis() - startTime
                if (success) {
                    Log.d(TAG, "✅ LLM warm-up completed successfully in ${warmUpTime}ms")
                } else {
                    Log.e(TAG, "❌ LLM warm-up failed after ${warmUpTime}ms")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during LLM warm-up", e)
            }
        }
    }
    
    /**
     * Generate response for user input
     */
    suspend fun generateResponse(userInput: String): LLMResponse {
        if (!isInitialized) {
            throw IllegalStateException("LLM service not initialized")
        }
        
        if (isGenerating) {
            throw IllegalStateException("Already generating response")
        }
        
        return withContext(Dispatchers.IO) {
            try {
                isGenerating = true
                val startTime = System.currentTimeMillis()
                
                Log.d(TAG, "Generating response for: $userInput")
                
                // Build conversation prompt first (before adding to history to avoid duplication)
                val prompt = buildConversationPrompt(userInput)
                Log.d(TAG, "Built prompt with ${prompt.length} characters")
                
                // Add user message to history after prompt building
                addToHistory(ConversationMessage(userInput, isUser = true))
                
                // Generate response using LLM engine
                val response = llmEngine.generateResponse(prompt, MAX_RESPONSE_TOKENS)
                
                val generationTime = System.currentTimeMillis() - startTime
                Log.d(TAG, "Generated response in ${generationTime}ms: $response")
                
                // Add AI response to history
                addToHistory(ConversationMessage(response, isUser = false))
                
                LLMResponse(
                    content = response,
                    tokensGenerated = estimateTokens(response),
                    generationTimeMs = generationTime
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Error generating response", e)
                LLMResponse(
                    content = "I'm sorry, I encountered an error while processing your request. Please try again.",
                    tokensGenerated = 0,
                    generationTimeMs = 0,
                    isComplete = false
                )
            } finally {
                isGenerating = false
            }
        }
    }
    
    /**
     * Generate streaming response (Flow-based)
     */
    fun generateStreamingResponse(userInput: String): Flow<String> = flow {
        if (!isInitialized) {
            emit("Service not initialized")
            return@flow
        }
        
        try {
            // Build conversation prompt first (before adding to history to avoid duplication)
            val prompt = buildConversationPrompt(userInput)
            
            // Add user message to history after prompt building
            addToHistory(ConversationMessage(userInput, isUser = true))
            
            // For now, emit the full response at once
            // TODO: Implement actual streaming when ONNX Runtime supports it
            val response = llmEngine.generateResponse(prompt, MAX_RESPONSE_TOKENS)
            
            // Simulate streaming by emitting word by word
            val words = response.split(" ")
            var currentText = ""
            
            for (word in words) {
                currentText += if (currentText.isEmpty()) word else " $word"
                emit(currentText)
                delay(50) // Simulate streaming delay
            }
            
            // Add final response to history
            addToHistory(ConversationMessage(response, isUser = false))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in streaming response", e)
            emit("Error generating response: ${e.message}")
        }
    }
    
    /**
     * Build conversation prompt from history
     */
    private fun buildConversationPrompt(currentInput: String): String {
        val builder = StringBuilder()
        
        // Add system prompt
        builder.append(SYSTEM_PROMPT)
        builder.append("\n\n")
        
        // Add conversation history (last N messages to stay within limits)
        val recentHistory = getRecentHistory()
        for (message in recentHistory) {
            val prefix = if (message.isUser) USER_PREFIX else ASSISTANT_PREFIX
            builder.append(prefix)
            builder.append(message.content)
            builder.append("\n")
        }
        
        // Add current user input
        builder.append(USER_PREFIX)
        builder.append(currentInput)
        builder.append("\n")
        builder.append(ASSISTANT_PREFIX)
        
        return builder.toString()
    }
    
    /**
     * Get recent conversation history within token limits
     */
    private fun getRecentHistory(): List<ConversationMessage> {
        val maxHistoryTokens = MAX_CONVERSATION_LENGTH - MAX_RESPONSE_TOKENS
        var currentTokens = 0
        val recentMessages = mutableListOf<ConversationMessage>()
        
        // Add messages from most recent, stopping when we hit token limit
        for (message in conversationHistory.reversed()) {
            val messageTokens = estimateTokens(message.content)
            if (currentTokens + messageTokens > maxHistoryTokens) break
            
            recentMessages.add(0, message) // Insert at beginning
            currentTokens += messageTokens
        }
        
        return recentMessages
    }
    
    /**
     * Add message to conversation history
     */
    private fun addToHistory(message: ConversationMessage) {
        conversationHistory.add(message)
        
        // Limit history size to prevent memory issues
        if (conversationHistory.size > 100) {
            conversationHistory.removeAt(0)
        }
    }
    
    /**
     * Estimate token count for text (rough approximation)
     */
    private fun estimateTokens(text: String): Int {
        // Rough estimate: 1 token ≈ 4 characters for English
        return (text.length / 4).coerceAtLeast(1)
    }
    
    /**
     * Clear conversation history
     */
    fun clearHistory() {
        conversationHistory.clear()
        Log.d(TAG, "Conversation history cleared")
    }
    
    /**
     * Get conversation history
     */
    fun getConversationHistory(): List<ConversationMessage> {
        return conversationHistory.toList()
    }
    
    /**
     * Check if service is ready
     */
    fun isReady(): Boolean {
        return isInitialized && !isGenerating
    }
    
    /**
     * Check if currently generating
     */
    fun isCurrentlyGenerating(): Boolean {
        return isGenerating
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        try {
            llmEngine.cleanup()
            conversationHistory.clear()
            isInitialized = false
            Log.d(TAG, "LLM service cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}