package com.memstory.presentation

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.rememberNavController
import com.memstory.presentation.navigation.MemStoryNavigation
import com.memstory.presentation.theme.MemSTORYTheme
import org.koin.androidx.compose.KoinAndroidContext
import org.koin.core.component.KoinComponent
import org.koin.core.component.get
import com.memstory.llm.LLMConversationService

class MainActivity : ComponentActivity(), KoinComponent {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // 🚀 Background LLM initialization for instant user experience
        initializeLLMService()
        
        setContent {
            KoinAndroidContext {
                MemSTORYTheme {
                    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                        val navController = rememberNavController()
                        MemStoryNavigation(
                            navController = navController,
                            modifier = Modifier.padding(innerPadding)
                        )
                    }
                }
            }
        }
    }
    
    /**
     * 🚀 Initialize LLM service in background during app startup
     * This moves the 7-second initialization from first user interaction to app launch
     */
    private fun initializeLLMService() {
        val llmService = get<LLMConversationService>()
        // LLMConversationService will automatically initialize OnnxLLMEngine on first access
        // This triggers: APK assets → internal storage copy → native heap loading
        // Total time: ~7 seconds in background while user sees UI
        // 🚀 Trigger actual background initialization
        llmService.warmUp() // This starts the 7-second initialization process
        android.util.Log.d("MainActivity", "🚀 Background LLM warm-up initiated")
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MemSTORYTheme {
        // Preview content
    }
}