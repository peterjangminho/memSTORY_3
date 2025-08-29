package com.memstory.domain.usecase

import com.memstory.domain.repository.LLMRepository

class GenerateTextUseCase(
    private val llmRepository: LLMRepository
) {
    suspend operator fun invoke(prompt: String): String {
        return if (llmRepository.isModelLoaded()) {
            llmRepository.generateText(prompt)
        } else {
            if (llmRepository.loadModel()) {
                llmRepository.generateText(prompt)
            } else {
                throw Exception("Failed to load LLM model")
            }
        }
    }
}