package com.memstory.presentation.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.memstory.R
import com.memstory.domain.model.Message
import com.memstory.presentation.viewmodel.TextChatViewModel
import org.koin.androidx.compose.koinViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TextChatScreen(
    viewModel: TextChatViewModel = koinViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val listState = rememberLazyListState()
    
    // Auto-scroll to bottom when new message is added
    LaunchedEffect(uiState.messages.size) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }
    
    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Chat Header
        TopAppBar(
            title = { Text(stringResource(R.string.text_chat)) }
        )
        
        // Messages List
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            state = listState,
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = PaddingValues(vertical = 16.dp)
        ) {
            items(uiState.messages) { message ->
                MessageBubble(message = message)
            }
            
            if (uiState.isLoading) {
                item {
                    TypingIndicator()
                }
            }
        }
        
        // Message Input
        MessageInput(
            value = uiState.currentMessage,
            onValueChange = viewModel::updateCurrentMessage,
            onSend = viewModel::sendMessage,
            isLoading = uiState.isLoading
        )
    }
}

@Composable
fun MessageBubble(message: Message) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (message.isFromUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            modifier = Modifier.widthIn(max = 280.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (message.isFromUser) 
                    MaterialTheme.colorScheme.primary 
                else 
                    MaterialTheme.colorScheme.surfaceVariant
            ),
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (message.isFromUser) 16.dp else 4.dp,
                bottomEnd = if (message.isFromUser) 4.dp else 16.dp
            )
        ) {
            Text(
                text = message.content,
                modifier = Modifier.padding(12.dp),
                color = if (message.isFromUser) 
                    MaterialTheme.colorScheme.onPrimary 
                else 
                    MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
fun TypingIndicator() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
            shape = RoundedCornerShape(16.dp, 16.dp, 16.dp, 4.dp)
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "memSTORY is thinking...",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun MessageInput(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    isLoading: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text(stringResource(R.string.type_message)) },
                maxLines = 3,
                enabled = !isLoading
            )
            
            Spacer(modifier = Modifier.width(8.dp))
            
            FloatingActionButton(
                onClick = onSend,
                modifier = Modifier.size(48.dp),
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary
            ) {
                Icon(
                    imageVector = Icons.Default.Send,
                    contentDescription = stringResource(R.string.send)
                )
            }
        }
    }
}