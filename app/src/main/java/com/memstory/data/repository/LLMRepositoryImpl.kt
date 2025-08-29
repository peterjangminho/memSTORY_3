package com.memstory.data.repository

import com.memstory.llm.LLMConversationService
import com.memstory.domain.repository.LLMRepository

class LLMRepositoryImpl(
    private val llmConversationService: LLMConversationService
) : LLMRepository {
    
    override suspend fun generateText(prompt: String): String {
        val response = llmConversationService.generateResponse(prompt)
        return response.content
    }
    
    override suspend fun isModelLoaded(): Boolean {
        return llmConversationService.isReady()
    }
    
    override suspend fun loadModel(): Boolean {
        return llmConversationService.initialize()
    }
}