package com.memstory.domain.repository

interface LLMRepository {
    suspend fun generateText(prompt: String): String
    suspend fun isModelLoaded(): Boolean
    suspend fun loadModel(): Boolean
}