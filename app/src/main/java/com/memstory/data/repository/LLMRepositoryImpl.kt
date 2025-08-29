package com.memstory.data.repository

import com.memstory.data.llm.LLMEngine
import com.memstory.domain.repository.LLMRepository

class LLMRepositoryImpl(
    private val llmEngine: LLMEngine
) : LLMRepository {
    
    override suspend fun generateText(prompt: String): String {
        return llmEngine.generateText(prompt)
    }
    
    override suspend fun isModelLoaded(): Boolean {
        return llmEngine.isModelLoaded()
    }
    
    override suspend fun loadModel(): Boolean {
        return llmEngine.loadModel()
    }
}