package com.memstory.di

import com.memstory.data.repository.LLMRepositoryImpl
import com.memstory.domain.repository.LLMRepository
import com.memstory.domain.usecase.GenerateTextUseCase
import com.memstory.data.llm.LLMEngine
import org.koin.dsl.module

val appModule = module {
    // Data Layer
    single<LLMRepository> { LLMRepositoryImpl(get()) }
    single { LLMEngine(get()) }
    
    // Domain Layer
    single { GenerateTextUseCase(get()) }
    
    // Presentation Layer
    single { com.memstory.presentation.viewmodel.HelloWorldViewModel(get()) }
    single { com.memstory.presentation.viewmodel.TextChatViewModel(get()) }
}