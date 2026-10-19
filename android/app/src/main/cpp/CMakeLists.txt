# libuvr_mnn.so: MNN's CPU backend (built from source, static) behind a small JNI API (MnnCpuEngine).
#
# MNN is downloaded once per build directory (pinned release + hash) and patched so its CPU
# convolutions keep FP32 accuracy: see patch_mnn.cmake. Pass -DMNN_SOURCE_DIR=<dir> to use a
# local checkout instead (it is patched the same way).
#
# Only arm64-v8a gets MNN; other ABIs (the x86_64 emulator) build a stub whose engine reports
# itself unavailable, and the app stays on ONNX Runtime.
cmake_minimum_required(VERSION 3.22.1)
project(uvr_mnn CXX C)

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    add_library(uvr_mnn SHARED uvr_mnn_stub.cpp)
    target_link_libraries(uvr_mnn log)
    return()
endif()

set(MNN_VERSION 3.6.1)
if(NOT MNN_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(mnn
        URL https://github.com/alibaba/MNN/archive/refs/tags/${MNN_VERSION}.tar.gz
        URL_HASH SHA256=4b6065c4e2674318f5bf1dc75836ce4d30c17bfe598c4a1b11b7d0b2092b06e6
        PATCH_COMMAND ${CMAKE_COMMAND} -DMNN_DIR=<SOURCE_DIR> -P ${CMAKE_CURRENT_LIST_DIR}/patch_mnn.cmake
    )
    FetchContent_GetProperties(mnn)
    if(NOT mnn_POPULATED)
        FetchContent_Populate(mnn)
    endif()
    set(MNN_SOURCE_DIR ${mnn_SOURCE_DIR})
else()
    execute_process(COMMAND ${CMAKE_COMMAND} -DMNN_DIR=${MNN_SOURCE_DIR} -P ${CMAKE_CURRENT_LIST_DIR}/patch_mnn.cmake
                    COMMAND_ERROR_IS_FATAL ANY)
endif()

# CPU only, one static library
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(MNN_SME2 OFF CACHE BOOL "" FORCE)
set(MNN_USE_LOGCAT ON CACHE BOOL "" FORCE)
add_subdirectory(${MNN_SOURCE_DIR} ${CMAKE_BINARY_DIR}/mnn EXCLUDE_FROM_ALL)
# Debug app builds would otherwise compile MNN at -O0, which runs the model many times slower
target_compile_options(MNN PRIVATE -O3)

add_library(uvr_mnn SHARED uvr_mnn.cpp)
target_include_directories(uvr_mnn PRIVATE ${MNN_SOURCE_DIR}/include)
# Whole archive: MNN registers its CPU ops and backend from static initialisers
target_link_libraries(uvr_mnn -Wl,--whole-archive MNN -Wl,--no-whole-archive log)
target_link_options(uvr_mnn PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL)
