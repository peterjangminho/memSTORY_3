package com.memstory.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.memstory.domain.model.Message
import com.memstory.domain.usecase.GenerateTextUseCase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.*

data class TextChatUiState(
    val messages: List<Message> = emptyList(),
    val currentMessage: String = "",
    val isLoading: Boolean = false
)

class TextChatViewModel(
    private val generateTextUseCase: GenerateTextUseCase
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(TextChatUiState())
    val uiState: StateFlow<TextChatUiState> = _uiState.asStateFlow()
    
    init {
        // Add welcome message
        val welcomeMessage = Message(
            id = UUID.randomUUID().toString(),
            content = "Hello! I'm memSTORY, your personal AI assistant. How can I help you today?",
            isFromUser = false
        )
        _uiState.value = _uiState.value.copy(
            messages = listOf(welcomeMessage)
        )
    }
    
    fun updateCurrentMessage(message: String) {
        _uiState.value = _uiState.value.copy(currentMessage = message)
    }
    
    fun sendMessage() {
        val currentMessage = _uiState.value.currentMessage.trim()
        if (currentMessage.isEmpty() || _uiState.value.isLoading) return
        
        // Add user message
        val userMessage = Message(
            id = UUID.randomUUID().toString(),
            content = currentMessage,
            isFromUser = true
        )
        
        val updatedMessages = _uiState.value.messages + userMessage
        _uiState.value = _uiState.value.copy(
            messages = updatedMessages,
            currentMessage = "",
            isLoading = true
        )
        
        // Generate AI response
        viewModelScope.launch {
            try {
                val aiResponse = generateTextUseCase(currentMessage)
                val aiMessage = Message(
                    id = UUID.randomUUID().toString(),
                    content = aiResponse,
                    isFromUser = false
                )
                
                _uiState.value = _uiState.value.copy(
                    messages = _uiState.value.messages + aiMessage,
                    isLoading = false
                )
            } catch (e: Exception) {
                val errorMessage = Message(
                    id = UUID.randomUUID().toString(),
                    content = "Sorry, I encountered an error: ${e.message}",
                    isFromUser = false
                )
                
                _uiState.value = _uiState.value.copy(
                    messages = _uiState.value.messages + errorMessage,
                    isLoading = false
                )
            }
        }
    }
}