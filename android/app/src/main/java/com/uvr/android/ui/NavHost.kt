package com.uvr.android.ui

import android.net.Uri
import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.uvr.android.viewmodel.MainViewModel

/**
 * App-level navigation graph.
 *
 * Routes:
 *  home       — File picker + model selector
 *  separate   — Separation progress + output
 *  settings   — App/model settings
 *  models     — Model management (add/remove ONNX models)
 */
@Composable
fun UVRNavHost(
    incomingAudioUri: Uri? = null,
    onIncomingAudioHandled: () -> Unit = {},
    viewModel: MainViewModel = viewModel()
) {
    val navController = rememberNavController()
    val context = LocalContext.current

    // App opened with an audio URI ("Open with UVR"): load it and go to home
    LaunchedEffect(incomingAudioUri) {
        incomingAudioUri?.let {
            if (viewModel.selectAudioFile(it)) {
                navController.popBackStack("home", inclusive = false)
            } else {
                Toast.makeText(context, "Finish or cancel the current separation first",
                    Toast.LENGTH_LONG).show()
            }
            onIncomingAudioHandled()
        }
    }

    NavHost(
        navController = navController,
        startDestination = "home"
    ) {
        composable("home") {
            HomeScreen(
                viewModel = viewModel,
                onNavigateToSettings = { navController.navigate("settings") },
                onNavigateToModels = { navController.navigate("models") },
                onStartSeparation = { navController.navigate("separate") }
            )
        }

        composable("separate") {
            SeparationScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }

        composable("settings") {
            SettingsScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }

        composable("models") {
            ModelsScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }
    }
}
