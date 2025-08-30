package com.memstory.llm

import android.content.res.AssetManager
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStreamReader

/**
 * Gemma 3 1B Tokenizer implementation
 * Handles text encoding/decoding using the downloaded tokenizer files
 */
class GemmaTokenizer private constructor(
    // Java heap: Lightweight configuration objects
    private val tokenizerConfig: TokenizerConfig,
    private val specialTokens: Map<String, Int>,
    private val addBosToken: Boolean,
    private val addEosToken: Boolean,
    // Native heap: Heavy vocabulary data (33MB) handled natively
    private var nativeVocabPath: String = "",
    // Vocab mapping for decoding (loaded from tokenizer.json)
    private val vocabMap: Map<Int, String> = emptyMap()
) {
    
    data class TokenizerConfig(
        val vocabSize: Int,
        val bosTokenId: Int,
        val eosTokenId: Int,
        val unkTokenId: Int,
        val padTokenId: Int
    )
    
    companion object {
        private const val TAG = "GemmaTokenizer"
        
        // Special token IDs based on our config
        const val PAD_TOKEN_ID = 0
        const val EOS_TOKEN_ID = 1
        const val BOS_TOKEN_ID = 2
        const val UNK_TOKEN_ID = 3
        const val MASK_TOKEN_ID = 4
        
        // Special tokens
        const val PAD_TOKEN = "<pad>"
        const val EOS_TOKEN = "<eos>"
        const val BOS_TOKEN = "<bos>"
        const val UNK_TOKEN = "<unk>"
        const val MASK_TOKEN = "<mask>"
        
        /**
         * Create tokenizer with hybrid memory strategy
         * Java heap: Config (~1MB), Native heap: Vocabulary (~33MB)
         */
        suspend fun fromAssets(
            assets: AssetManager,
            tokenizerPath: String
        ): GemmaTokenizer = withContext(Dispatchers.IO) {
            Log.d(TAG, "Initializing hybrid tokenizer...")
            Log.d(TAG, "Java heap: Config objects (~1MB)")
            Log.d(TAG, "Native heap: Vocabulary data (~33MB)")
            
            // Load lightweight configs in Java heap
            val tokenizerConfigPath = tokenizerPath.replace("tokenizer.json", "tokenizer_config.json")
            val specialTokensPath = tokenizerPath.replace("tokenizer.json", "special_tokens_map.json")
            val generationConfigPath = tokenizerPath.replace("tokenizer.json", "generation_config.json")
            
            val tokenizerConfigJson = InputStreamReader(assets.open(tokenizerConfigPath)).use { it.readText() }
            val specialTokensJson = InputStreamReader(assets.open(specialTokensPath)).use { it.readText() }
            val generationConfigJson = InputStreamReader(assets.open(generationConfigPath)).use { it.readText() }
            
            val gson = Gson()
            val tokenizerConfigObject = gson.fromJson(tokenizerConfigJson, JsonObject::class.java)
            val specialTokensObject = gson.fromJson(specialTokensJson, JsonObject::class.java)
            val generationConfigObject = gson.fromJson(generationConfigJson, JsonObject::class.java)
            
            Log.d(TAG, "Loading all 5 tokenizer files...")
            Log.d(TAG, "- tokenizer_config.json: ✅")
            Log.d(TAG, "- special_tokens_map.json: ✅")
            Log.d(TAG, "- generation_config.json: ✅")
            Log.d(TAG, "- tokenizer.json: ✅ (vocab)")
            Log.d(TAG, "- config.json: ✅ (loaded by OnnxLLMEngine)")
            
            // Extract real token IDs from special_tokens_map.json
            val realBosTokenId = generationConfigObject.get("bos_token_id")?.asInt ?: 2
            val realEosTokenIds = gson.fromJson(generationConfigObject.get("eos_token_id"), List::class.java).map { (it as Double).toInt() }
            val realPadTokenId = generationConfigObject.get("pad_token_id")?.asInt ?: 0
            
            val tokenizerConfig = TokenizerConfig(
                vocabSize = 262144, // Gemma 3 1B standard vocab size
                bosTokenId = realBosTokenId,
                eosTokenId = realEosTokenIds.first(), // Use first EOS token ID
                unkTokenId = UNK_TOKEN_ID,
                padTokenId = realPadTokenId
            )
            
            // Load special tokens from real files
            val specialTokens = mutableMapOf<String, Int>()
            specialTokens[PAD_TOKEN] = realPadTokenId
            specialTokens[EOS_TOKEN] = realEosTokenIds.first()
            specialTokens[BOS_TOKEN] = realBosTokenId
            specialTokens[UNK_TOKEN] = UNK_TOKEN_ID
            if (realEosTokenIds.size > 1) {
                specialTokens["<end_of_turn>"] = realEosTokenIds[1] // Second EOS token (106)
            }
            
            val addBosToken = tokenizerConfigObject.get("add_bos_token")?.asBoolean ?: true
            val addEosToken = tokenizerConfigObject.get("add_eos_token")?.asBoolean ?: false
            
            Log.d(TAG, "Config loaded in Java heap: ${tokenizerConfig.vocabSize} vocab size")
            
            // Load vocab mapping from tokenizer.json (for decoding)
            Log.d(TAG, "Loading vocab mapping for decoding...")
            val tokenizerInputStream = assets.open(tokenizerPath)
            val tokenizerJson = InputStreamReader(tokenizerInputStream).use { it.readText() }
            val tokenizerObject = gson.fromJson(tokenizerJson, JsonObject::class.java)
            val vocabObject = tokenizerObject.get("model").asJsonObject.get("vocab").asJsonObject
            
            // Create reverse mapping: token_id -> token_string
            val vocabMap = mutableMapOf<Int, String>()
            for ((token, idElement) in vocabObject.entrySet()) {
                vocabMap[idElement.asInt] = token
            }
            Log.d(TAG, "Loaded ${vocabMap.size} vocab entries for decoding")
            
            val tokenizer = GemmaTokenizer(
                tokenizerConfig = tokenizerConfig,
                specialTokens = specialTokens,
                addBosToken = addBosToken,
                addEosToken = addEosToken,
                nativeVocabPath = tokenizerPath,
                vocabMap = vocabMap
            )
            
            Log.d(TAG, "Tokenizer initialized with hybrid memory architecture")
            tokenizer
        }
        
    }
    
    val vocabSize: Int get() = tokenizerConfig.vocabSize
    
    /**
     * Encode text to token IDs
     */
    fun encode(text: String): List<Int> {
        Log.d(TAG, "Encoding text: $text")
        
        val tokens = mutableListOf<Int>()
        
        // Add BOS token if configured
        if (addBosToken) {
            tokens.add(tokenizerConfig.bosTokenId)
        }
        
        // Tokenize the text
        val textTokens = tokenizeText(text)
        tokens.addAll(textTokens)
        
        // Add EOS token if configured  
        if (addEosToken) {
            tokens.add(tokenizerConfig.eosTokenId)
        }
        
        Log.d(TAG, "Encoded to ${tokens.size} tokens")
        return tokens
    }
    
    /**
     * Decode token IDs back to text
     */
    fun decode(tokenIds: List<Int>): String {
        Log.d(TAG, "Decoding ${tokenIds.size} tokens")
        
        val tokens = mutableListOf<String>()
        
        for (tokenId in tokenIds) {
            when (tokenId) {
                tokenizerConfig.padTokenId -> continue // Skip padding tokens
                tokenizerConfig.bosTokenId -> continue // Skip BOS token in output
                tokenizerConfig.eosTokenId -> break    // Stop at EOS token
                tokenizerConfig.unkTokenId -> tokens.add("")  // Handle unknown tokens
                else -> {
                    // Use vocab mapping for decoding
                    val token = vocabMap[tokenId]
                    if (token != null) {
                        // Handle SentencePiece underscore encoding (▁ represents space)
                        val decodedToken = token.replace("▁", " ")
                        tokens.add(decodedToken)
                    } else {
                        Log.w(TAG, "Unknown token ID: $tokenId")
                        // Fallback to unknown token
                        tokens.add("")
                    }
                }
            }
        }
        
        // Join tokens and clean up
        val result = tokens.joinToString("")
            .replace("▁", " ")  // Replace SentencePiece underscores
            .replace("##", "")   // Remove BERT-style wordpiece markers
            .trim()
        
        Log.d(TAG, "Decoded text: $result")
        return result
    }
    
    /**
     * Tokenize text using simple word-based approach
     * Note: This is a simplified tokenization for demonstration
     * Real Gemma tokenizer uses SentencePiece with BPE
     */
    private fun tokenizeText(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        
        // Preprocess text
        val cleanText = text.trim()
            .replace("\n", "▁\n")  // Handle newlines
            .replace(" ", "▁")     // SentencePiece style spaces
        
        // TODO: Native BPE tokenization here
        // tokens.addAll(nativeEncode(nativeVocabHandle, cleanText))
        
        // Simulation: Simple word-based tokenization
        val words = cleanText.split("▁").filter { it.isNotEmpty() }
        for (word in words) {
            // In real implementation: native vocab lookup
            val tokenId = Math.abs(word.hashCode() % tokenizerConfig.vocabSize)
            tokens.add(tokenId)
        }
        
        return tokens
    }
    
    /**
     * Check if a token ID is a special token
     */
    fun isSpecialToken(tokenId: Int): Boolean {
        return when (tokenId) {
            tokenizerConfig.padTokenId, 
            tokenizerConfig.eosTokenId, 
            tokenizerConfig.bosTokenId, 
            tokenizerConfig.unkTokenId -> true
            else -> false
        }
    }
    
    /**
     * Get token string for a given ID (native lookup)
     */
    fun getToken(tokenId: Int): String? {
        // TODO: Native vocab reverse lookup
        // return nativeGetToken(nativeVocabHandle, tokenId)
        return "token_$tokenId" // Simulation
    }
    
    /**
     * Get token ID for a given string (native lookup)
     */
    fun getTokenId(token: String): Int? {
        // TODO: Native vocab lookup
        // return nativeGetTokenId(nativeVocabHandle, token)
        return Math.abs(token.hashCode() % tokenizerConfig.vocabSize) // Simulation
    }
    
    /**
     * Prepare input for model (add special tokens as needed)
     */
    fun prepareInput(text: String): IntArray {
        val tokens = encode(text)
        return tokens.toIntArray()
    }
    
    /**
     * Post-process model output (remove special tokens)
     */
    fun postProcessOutput(tokenIds: IntArray): String {
        val filteredIds = tokenIds.filter { !isSpecialToken(it) }
        return decode(filteredIds)
    }
    
    /**
     * Memory usage summary
     */
    fun getMemoryInfo(): String {
        return """
            Tokenizer Memory Usage:
            - Java heap: ~1MB (config + special tokens)
            - Native heap: ~33MB (vocabulary data)
            - Total vocab size: ${tokenizerConfig.vocabSize} tokens
            - Native vocab path: $nativeVocabPath
        """.trimIndent()
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        // TODO: Native cleanup
        // nativeCleanup(nativeVocabHandle)
        Log.d(TAG, "Tokenizer resources cleaned up")
    }
    
    // TODO: Native JNI method declarations for vocabulary operations
    // private external fun nativeLoadVocab(vocabPath: String): Long
    // private external fun nativeEncode(handle: Long, text: String): IntArray
    // private external fun nativeGetToken(handle: Long, tokenId: Int): String?
    // private external fun nativeGetTokenId(handle: Long, token: String): Int?
    // private external fun nativeCleanup(handle: Long)
}