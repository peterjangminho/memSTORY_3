package com.memstory.di

import com.memstory.data.repository.LLMRepositoryImpl
import com.memstory.domain.repository.LLMRepository
import com.memstory.domain.usecase.GenerateTextUseCase
import org.koin.core.module.dsl.viewModel
import org.koin.dsl.module

val appModule = module {
    // Data Layer  
    single<LLMRepository> { LLMRepositoryImpl(get()) }
    
    // Domain Layer
    single { GenerateTextUseCase(get()) }
    
    // Presentation Layer (ViewModels)
    viewModel { com.memstory.presentation.viewmodel.HelloWorldViewModel(get()) }
    viewModel { com.memstory.presentation.viewmodel.TextChatViewModel(get()) }
}