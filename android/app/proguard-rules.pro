# ONNX Runtime — keep all native binding classes
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# FFmpegKit
-keep class com.arthenica.ffmpegkit.** { *; }
-dontwarn com.arthenica.ffmpegkit.**

# DataStore / Protobuf
-keepclassmembers class * extends com.google.protobuf.GeneratedMessageLite {
    <fields>;
}

# Keep model data classes (used with Gson)
-keep class com.uvr.android.ml.** { *; }
-keep class com.uvr.android.audio.** { *; }

# Kotlin serialization
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
