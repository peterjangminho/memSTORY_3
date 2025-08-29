package com.memstory.presentation.navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.memstory.presentation.screens.HelloWorldScreen
import com.memstory.presentation.screens.TextChatScreen

@Composable
fun MemStoryNavigation(
    navController: NavHostController,
    modifier: Modifier = Modifier
) {
    NavHost(
        navController = navController,
        startDestination = "hello_world",
        modifier = modifier
    ) {
        composable("hello_world") {
            HelloWorldScreen(
                onNavigateToTextChat = {
                    navController.navigate("text_chat")
                }
            )
        }
        
        composable("text_chat") {
            TextChatScreen()
        }
    }
}