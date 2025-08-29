package com.memstory.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.memstory.domain.usecase.GenerateTextUseCase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class HelloWorldUiState(
    val isLoading: Boolean = false,
    val generatedText: String = "",
    val errorMessage: String? = null
)

class HelloWorldViewModel(
    private val generateTextUseCase: GenerateTextUseCase
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(HelloWorldUiState())
    val uiState: StateFlow<HelloWorldUiState> = _uiState.asStateFlow()
    
    fun generateHelloWorld() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isLoading = true,
                errorMessage = null
            )
            
            try {
                val generatedText = generateTextUseCase("Hello World")
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    generatedText = generatedText
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    errorMessage = e.message
                )
            }
        }
    }
}