// MNN isn't built for this ABI: MnnCpuEngine.isAvailable is false and the app uses ONNX Runtime.
#include <jni.h>

extern "C" JNIEXPORT jboolean JNICALL
Java_com_uvr_android_ml_MnnCpuEngine_nativeIsStub(JNIEnv*, jclass) { return JNI_TRUE; }
