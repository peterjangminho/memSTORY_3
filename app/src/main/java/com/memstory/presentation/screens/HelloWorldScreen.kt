package com.memstory.presentation.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.memstory.R
import com.memstory.presentation.viewmodel.HelloWorldViewModel
import org.koin.androidx.compose.koinViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HelloWorldScreen(
    onNavigateToTextChat: () -> Unit,
    viewModel: HelloWorldViewModel = koinViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "memSTORY",
            style = MaterialTheme.typography.headlineLarge
        )
        
        Spacer(modifier = Modifier.height(32.dp))
        
        if (uiState.generatedText.isNotEmpty()) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                Text(
                    text = uiState.generatedText,
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        if (uiState.errorMessage != null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                )
            ) {
                Text(
                    text = "Error: ${uiState.errorMessage}",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        Button(
            onClick = { viewModel.generateHelloWorld() },
            enabled = !uiState.isLoading
        ) {
            if (uiState.isLoading) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(
                text = if (uiState.isLoading) stringResource(R.string.generating_text) 
                      else stringResource(R.string.hello_world)
            )
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        OutlinedButton(
            onClick = onNavigateToTextChat
        ) {
            Text(stringResource(R.string.text_chat))
        }
    }
}