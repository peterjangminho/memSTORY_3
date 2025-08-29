package com.memstory.di

import com.memstory.llm.OnnxLLMEngine
import com.memstory.llm.LLMConversationService
import org.koin.android.ext.koin.androidContext
import org.koin.dsl.module

/**
 * Koin DI module for LLM components
 */
val llmModule = module {
    
    // LLM Engine - Singleton
    single<OnnxLLMEngine> { 
        OnnxLLMEngine(androidContext()) 
    }
    
    // LLM Conversation Service - Singleton
    single<LLMConversationService> { 
        LLMConversationService(androidContext()) 
    }
}