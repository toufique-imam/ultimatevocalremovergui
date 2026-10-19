// JNI bridge for MnnCpuEngine: one MNN model on the CPU backend in FP32, single input and output.
#include <jni.h>
#include <android/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define LOG_TAG "MnnCpuEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

struct Engine {
    std::unique_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    MNN::Tensor* input = nullptr;
    std::unique_ptr<MNN::Tensor> hostIn;
    std::vector<int> shape;
};

void throwJava(JNIEnv* env, const std::string& msg) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(cls, msg.c_str());
}

std::string shapeString(const std::vector<int>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); i++) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

// The session keeps the model's input shape: the model buffer is released after the session is
// created (saving its size in memory), and MNN can't resize without it.
bool prepare(Engine* e, const std::vector<int>& shape) {
    if (shape != e->shape) return false;
    if (!e->hostIn) e->hostIn.reset(new MNN::Tensor(e->input, MNN::Tensor::CAFFE));
    return true;
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_uvr_android_ml_MnnCpuEngine_nativeIsStub(JNIEnv*, jclass) { return JNI_FALSE; }

JNIEXPORT jlong JNICALL
Java_com_uvr_android_ml_MnnCpuEngine_nativeCreate(JNIEnv* env, jclass, jstring jpath, jint threads) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(path));
    env->ReleaseStringUTFChars(jpath, path);
    if (!net) {
        throwJava(env, "MNN could not load the model");
        return 0;
    }
    // Plain convolutions instead of Winograd's 3x3 shortcut: Winograd's transforms cost accuracy
    // (Kim_Vocal_2: ~81 dB SNR against ONNX Runtime with it, ~118 dB without, with Strassen off too)
    net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = threads;
    MNN::BackendConfig backend;
    backend.precision = MNN::BackendConfig::Precision_High;  // FP32 (Normal/Low may use FP16 on ARMv8.2)
    backend.power = MNN::BackendConfig::Power_High;
    backend.memory = MNN::BackendConfig::Memory_Normal;
    config.backendConfig = &backend;
    MNN::Session* session = net->createSession(config);
    if (!session) {
        throwJava(env, "MNN could not create a CPU session");
        return 0;
    }
    auto* e = new Engine();
    e->session = session;
    e->input = net->getSessionInput(session, nullptr);
    e->shape = e->input->shape();
    float memoryMb = 0;
    net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryMb);
    LOGI("CPU session: %d threads, input %s, %.0f MB", (int)threads, shapeString(e->shape).c_str(), memoryMb);
    net->releaseModel();  // the session holds its own copy of the weights
    e->net = std::move(net);
    return reinterpret_cast<jlong>(e);
}

JNIEXPORT jintArray JNICALL
Java_com_uvr_android_ml_MnnCpuEngine_nativeInputShape(JNIEnv* env, jclass, jlong handle) {
    auto* e = reinterpret_cast<Engine*>(handle);
    jintArray out = env->NewIntArray((jsize)e->shape.size());
    env->SetIntArrayRegion(out, 0, (jsize)e->shape.size(), e->shape.data());
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_com_uvr_android_ml_MnnCpuEngine_nativeRun(JNIEnv* env, jclass, jlong handle, jfloatArray jdata, jintArray jshape) {
    auto* e = reinterpret_cast<Engine*>(handle);
    std::vector<int> shape(env->GetArrayLength(jshape));
    env->GetIntArrayRegion(jshape, 0, (jsize)shape.size(), shape.data());
    if (!prepare(e, shape)) {
        throwJava(env, "MNN model takes input " + shapeString(e->shape) + ", not " + shapeString(shape));
        return nullptr;
    }
    const jsize n = env->GetArrayLength(jdata);
    if (n != e->hostIn->elementSize()) {
        throwJava(env, "input has " + std::to_string(n) + " values, shape " + shapeString(shape) + " needs " +
                           std::to_string(e->hostIn->elementSize()));
        return nullptr;
    }
    env->GetFloatArrayRegion(jdata, 0, n, e->hostIn->host<float>());
    e->input->copyFromHostTensor(e->hostIn.get());
    if (e->net->runSession(e->session) != MNN::NO_ERROR) {
        throwJava(env, "MNN inference failed");
        return nullptr;
    }
    MNN::Tensor* output = e->net->getSessionOutput(e->session, nullptr);
    MNN::Tensor hostOut(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(&hostOut);
    const jsize m = hostOut.elementSize();
    jfloatArray result = env->NewFloatArray(m);
    if (result) env->SetFloatArrayRegion(result, 0, m, hostOut.host<float>());
    return result;
}

JNIEXPORT void JNICALL
Java_com_uvr_android_ml_MnnCpuEngine_nativeRelease(JNIEnv*, jclass, jlong handle) {
    auto* e = reinterpret_cast<Engine*>(handle);
    if (!e) return;
    e->hostIn.reset();
    if (e->session) e->net->releaseSession(e->session);
    delete e;
}

}  // extern "C"
