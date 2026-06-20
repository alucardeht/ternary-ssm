cmake_minimum_required(VERSION 3.16)
project(ternary_engine C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(ENGINE_NATIVE "Build optimized for this machine" ON)
option(ENGINE_AVX2 "Enable AVX2 (x86)" ON)
option(ENGINE_AVX512 "Enable AVX-512 (x86)" OFF)
option(ENGINE_NEON "Enable NEON (ARM)" ON)
option(ENGINE_WASM_SIMD "Enable WASM SIMD (Emscripten)" OFF)
option(ENGINE_VULKAN "Enable Vulkan compute backend" OFF)
option(ENGINE_BUILD_TESTS "Build tests" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x86")
    set(ARCH_X86 ON)
    message(STATUS "Architecture: x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ARCH_ARM ON)
    message(STATUS "Architecture: ARM64")
else()
    message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR} (scalar fallback)")
endif()

# SIMD flags
include(CheckCCompilerFlag)

if(ARCH_X86)
    if(ENGINE_AVX2)
        if(MSVC)
            check_c_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                add_compile_definitions(ENGINE_AVX2)
                add_compile_options(/arch:AVX2)
                message(STATUS "AVX2: enabled (MSVC)")
            endif()
        else()
            check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
            if(COMPILER_SUPPORTS_AVX2)
                add_compile_definitions(ENGINE_AVX2)
                add_compile_options(-mavx2 -mfma -mf16c)
                message(STATUS "AVX2: enabled")
            endif()
        endif()
    endif()

    if(ENGINE_AVX512)
        if(MSVC)
            check_c_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
            if(COMPILER_SUPPORTS_AVX512)
                add_compile_definitions(ENGINE_AVX512)
                add_compile_options(/arch:AVX512)
                message(STATUS "AVX-512: enabled (MSVC)")
            endif()
        else()
            check_c_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
            if(COMPILER_SUPPORTS_AVX512)
                add_compile_definitions(ENGINE_AVX512)
                add_compile_options(-mavx512f -mavx512bw -mavx512vl)
                message(STATUS "AVX-512: enabled")
            endif()
        endif()
    endif()
endif()

if(ARCH_ARM AND ENGINE_NEON)
    add_compile_definitions(ENGINE_NEON)
    message(STATUS "NEON: enabled (ARM64 default)")
endif()

if(EMSCRIPTEN)
    set(ENGINE_WASM_SIMD ON CACHE BOOL "" FORCE)
    set(ENGINE_AVX2 OFF CACHE BOOL "" FORCE)
    set(ENGINE_AVX512 OFF CACHE BOOL "" FORCE)
    set(ENGINE_NEON OFF CACHE BOOL "" FORCE)
    message(STATUS "Emscripten detected: WASM SIMD mode")
endif()

if(ENGINE_WASM_SIMD)
    add_compile_definitions(ENGINE_WASM_SIMD)
    if(EMSCRIPTEN)
        add_compile_options(-msimd128)
    endif()
endif()

# Native optimization
if(ENGINE_NATIVE)
    if(NOT MSVC)
        add_compile_options(-march=native)
        message(STATUS "Native optimization: enabled")
    endif()
endif()

# Warning flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Core library
add_library(ternary_engine STATIC
    src/simd_detect.c
    src/ternary_pack.c
    src/ternary_gemv.c
    src/popcount_gemv.c
    src/activations.c
    src/float_gemv.c
    src/ssm_scan.c
    src/attention.c
    src/model_io.c
    src/inference.c
    src/tokenizer.c
)

target_include_directories(ternary_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link math library for math functions (sqrt, exp, log)
if(NOT MSVC)
    target_link_libraries(ternary_engine m)
endif()

# Vulkan backend (optional)
if(ENGINE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_compile_definitions(ternary_engine PRIVATE ENGINE_VULKAN)
        target_sources(ternary_engine PRIVATE
            src/vulkan_backend.c
            src/vulkan_buffers.c
            src/vulkan_pipeline.c
            src/vulkan_dispatch.c)
        target_include_directories(ternary_engine PRIVATE ${Vulkan_INCLUDE_DIRS})
        target_link_libraries(ternary_engine ${Vulkan_LIBRARIES})

        find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
        if(GLSLC)
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_shaders.h
                COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_shaders.py
                    --shader-dir ${CMAKE_CURRENT_SOURCE_DIR}/shaders
                    --output ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_shaders.h
                    --glslc ${GLSLC}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp
                COMMENT "Compiling Vulkan compute shaders to SPIR-V"
            )
            add_custom_target(compile_shaders DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/vulkan_shaders.h)
            add_dependencies(ternary_engine compile_shaders)
        else()
            message(WARNING "glslc not found, using pre-compiled vulkan_shaders.h")
        endif()
        message(STATUS "Vulkan: enabled")
    else()
        message(STATUS "Vulkan: SDK not found (disabled)")
    endif()
else()
    message(STATUS "Vulkan: disabled")
endif()

# Tests
if(ENGINE_BUILD_TESTS)
    enable_testing()

    add_executable(test_build tests/test_build.c)
    target_link_libraries(test_build ternary_engine)
    add_test(NAME test_build COMMAND test_build)

    add_executable(test_gemv tests/test_gemv.c)
    target_link_libraries(test_gemv ternary_engine)
    add_test(NAME test_gemv COMMAND test_gemv)

    add_executable(test_popcount tests/test_popcount.c)
    target_link_libraries(test_popcount ternary_engine)
    target_include_directories(test_popcount PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME popcount_tests COMMAND test_popcount)

    add_executable(test_activations tests/test_activations.c)
    target_link_libraries(test_activations ternary_engine)
    target_include_directories(test_activations PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME activation_tests COMMAND test_activations)

    add_executable(test_ssm_scan tests/test_ssm_scan.c)
    target_link_libraries(test_ssm_scan ternary_engine)
    target_include_directories(test_ssm_scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME ssm_scan_tests COMMAND test_ssm_scan)

    add_executable(test_attention tests/test_attention.c)
    target_link_libraries(test_attention ternary_engine)
    target_include_directories(test_attention PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME attention_tests COMMAND test_attention)

    add_executable(test_model_io tests/test_model_io.c)
    target_link_libraries(test_model_io ternary_engine)
    target_include_directories(test_model_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME model_io_tests COMMAND test_model_io)

    add_executable(test_inference tests/test_inference.c)
    target_link_libraries(test_inference ternary_engine)
    target_include_directories(test_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME inference_tests COMMAND test_inference)

    add_executable(test_tokenizer tests/test_tokenizer.c)
    target_link_libraries(test_tokenizer PRIVATE ternary_engine)
    target_include_directories(test_tokenizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME tokenizer_tests COMMAND test_tokenizer)

    # --- Per-backend test variants (scalar forced) ---
    add_test(NAME test_gemv_scalar
             COMMAND ${CMAKE_COMMAND} -E env ENGINE_BACKEND=scalar $<TARGET_FILE:test_gemv>)
    add_test(NAME activation_tests_scalar
             COMMAND ${CMAKE_COMMAND} -E env ENGINE_BACKEND=scalar $<TARGET_FILE:test_activations>)
    add_test(NAME ssm_scan_tests_scalar
             COMMAND ${CMAKE_COMMAND} -E env ENGINE_BACKEND=scalar $<TARGET_FILE:test_ssm_scan>)
    add_test(NAME attention_tests_scalar
             COMMAND ${CMAKE_COMMAND} -E env ENGINE_BACKEND=scalar $<TARGET_FILE:test_attention>)
    add_test(NAME inference_tests_scalar
             COMMAND ${CMAKE_COMMAND} -E env ENGINE_BACKEND=scalar $<TARGET_FILE:test_inference>)

    message(STATUS "Tests: enabled")
endif()

# Benchmarks
option(ENGINE_BUILD_BENCHMARKS "Build benchmarks" ON)

if(ENGINE_BUILD_BENCHMARKS)
    add_executable(bench_gemv benchmarks/bench_gemv.c)
    target_link_libraries(bench_gemv ternary_engine)
    target_include_directories(bench_gemv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(bench_components benchmarks/bench_components.c)
    target_link_libraries(bench_components ternary_engine)
    target_include_directories(bench_components PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    message(STATUS "Benchmarks: enabled (bench_gemv, bench_components)")
endif()

# --- CLI Executable ---
add_executable(trnm_cli src/cli.c)
target_link_libraries(trnm_cli PRIVATE ternary_engine)
target_include_directories(trnm_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WIN32)
    target_compile_definitions(trnm_cli PRIVATE PLATFORM_WINDOWS)
endif()

# Summary
message(STATUS "")
message(STATUS "=== Ternary Engine Build Summary ===")
message(STATUS "  Architecture:  ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler:      ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  AVX2:          ${ENGINE_AVX2}")
message(STATUS "  AVX-512:       ${ENGINE_AVX512}")
message(STATUS "  NEON:          ${ENGINE_NEON}")
message(STATUS "  WASM SIMD:     ${ENGINE_WASM_SIMD}")
message(STATUS "  Vulkan:        ${ENGINE_VULKAN}")
message(STATUS "  Native:        ${ENGINE_NATIVE}")
message(STATUS "  Tests:         ${ENGINE_BUILD_TESTS}")
message(STATUS "  CLI:           enabled")
message(STATUS "=====================================")
