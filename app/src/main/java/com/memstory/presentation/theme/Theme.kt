package com.memstory.presentation.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColorScheme = lightColorScheme(
    primary = PastelBlue,
    onPrimary = White,
    primaryContainer = PastelBlueLight,
    onPrimaryContainer = DarkBlue,
    background = White,
    onBackground = DarkGray,
    surface = White,
    onSurface = DarkGray
)

@Composable
fun MemSTORYTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}