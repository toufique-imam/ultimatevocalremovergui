package com.uvr.android.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// UVR color palette — matches the dark theme used in the desktop app
private val UVRDarkPrimary = Color(0xFF6B4EFF)        // Purple accent
private val UVRDarkSecondary = Color(0xFF03DAC5)       // Teal
private val UVRDarkBackground = Color(0xFF1A1A2E)      // Dark navy
private val UVRDarkSurface = Color(0xFF16213E)
private val UVRDarkOnPrimary = Color.White
private val UVRDarkError = Color(0xFFCF6679)

private val UVRLightPrimary = Color(0xFF5B3ECC)
private val UVRLightBackground = Color(0xFFF5F5F5)
private val UVRLightSurface = Color.White

val DarkColorScheme = darkColorScheme(
    primary = UVRDarkPrimary,
    secondary = UVRDarkSecondary,
    background = UVRDarkBackground,
    surface = UVRDarkSurface,
    onPrimary = UVRDarkOnPrimary,
    onBackground = Color.White,
    onSurface = Color(0xFFE0E0E0),
    error = UVRDarkError
)

val LightColorScheme = lightColorScheme(
    primary = UVRLightPrimary,
    secondary = UVRDarkSecondary,
    background = UVRLightBackground,
    surface = UVRLightSurface,
    onPrimary = Color.White,
    onBackground = Color(0xFF1A1A2E),
    onSurface = Color(0xFF333333)
)

@Composable
fun UVRTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme
    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
