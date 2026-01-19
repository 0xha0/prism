cmake_minimum_required(VERSION 3.20)

# ============================================================================
# 本地配置（可选）
# ============================================================================
set(_LOCAL_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/local.cmake")
if(EXISTS "${_LOCAL_CMAKE}")
    message(STATUS "正在加载本地配置: ${_LOCAL_CMAKE}")
    include("${_LOCAL_CMAKE}")
else()
    message(STATUS "未找到本地配置 (cmake/local.cmake)")
    message(STATUS "  复制 cmake/local.cmake.example 到 cmake/local.cmake 以自定义配置")
endif()
unset(_LOCAL_CMAKE)

project(prism VERSION 0.1.0 LANGUAGES C CXX)

# ============================================================================
# PRISM - Parallel RF Instructions for Signal Manipulation
# ============================================================================
# 如需生成 Doxygen 文档，请执行 `doxygen docs/Doxyfile`（需 Graphviz 支持）

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# 构建选项
# ============================================================================
function(prism_define_bool_option NAME DEFAULT_VALUE DESCRIPTION)
    if(DEFINED CACHE{${NAME}})
        # Preserve cache value set via -D or previous configure.
    elseif(DEFINED ${NAME})
        set(${NAME} "${${NAME}}" CACHE BOOL "${DESCRIPTION}")
    else()
        set(${NAME} "${DEFAULT_VALUE}" CACHE BOOL "${DESCRIPTION}")
    endif()
endfunction()

prism_define_bool_option(PRISM_BUILD_TESTS ON "构建单元测试")
prism_define_bool_option(PRISM_BUILD_EXAMPLES ON "构建示例程序")
prism_define_bool_option(PRISM_BUILD_BENCHMARKS ON "构建基准程序")

set(_default_anderson_scripts_dir "")
if(EXISTS "${CMAKE_SOURCE_DIR}/misc/scripts/utils.sh")
    set(_default_anderson_scripts_dir "${CMAKE_SOURCE_DIR}/misc/scripts")
endif()
if(NOT DEFINED CACHE{PRISM_ANDERSON2021_SCRIPTS_DIR})
    set(PRISM_ANDERSON2021_SCRIPTS_DIR "${_default_anderson_scripts_dir}"
        CACHE PATH "Halide anderson2021 scripts 目录（包含 utils.sh）")
elseif(PRISM_ANDERSON2021_SCRIPTS_DIR STREQUAL "" AND
       NOT _default_anderson_scripts_dir STREQUAL "")
    set(PRISM_ANDERSON2021_SCRIPTS_DIR "${_default_anderson_scripts_dir}"
        CACHE PATH "Halide anderson2021 scripts 目录（包含 utils.sh）" FORCE)
endif()
unset(_default_anderson_scripts_dir)

find_program(PRISM_NVIDIA_SMI_BIN nvidia-smi)
if(PRISM_NVIDIA_SMI_BIN)
    set(PRISM_AUTOTUNE_FAKE_NVIDIA_SMI OFF)
else()
    set(PRISM_AUTOTUNE_FAKE_NVIDIA_SMI ON)
endif()


# 后端选项：支持 local.cmake 覆盖
if(NOT DEFINED PRISM_USE_VDSP)
    set(PRISM_USE_VDSP "AUTO" CACHE STRING "Apple vDSP 后端 (AUTO/ON/OFF)")
endif()
if(NOT DEFINED PRISM_USE_CUFFT)
    set(PRISM_USE_CUFFT "AUTO" CACHE STRING "NVIDIA cuFFT 后端 (AUTO/ON/OFF)")
endif()
if(NOT DEFINED PRISM_USE_HIPFFT)
    set(PRISM_USE_HIPFFT "AUTO" CACHE STRING "AMD hipFFT 后端 (AUTO/ON/OFF)")
endif()
if(NOT DEFINED PRISM_USE_VKFFT)
    set(PRISM_USE_VKFFT "AUTO" CACHE STRING "vkFFT 后端 (AUTO/ON/OFF)")
endif()

# ============================================================================
# 依赖项自动检测
# ============================================================================
macro(prism_detect_backend NAME CONDITION FOUND_VAR)
    if("${PRISM_USE_${NAME}}" STREQUAL "AUTO")
        if(${CONDITION})
            set(PRISM_HAS_${NAME} TRUE)
            set(PRISM_${NAME}_STATUS "✓ 已启用")
        else()
            set(PRISM_HAS_${NAME} FALSE)
            set(PRISM_${NAME}_STATUS "✗ 不可用")
        endif()
    elseif("${PRISM_USE_${NAME}}" STREQUAL "ON")
        set(PRISM_HAS_${NAME} TRUE)
        set(PRISM_${NAME}_STATUS "✓ 已启用 (强制)")
    else()
        set(PRISM_HAS_${NAME} FALSE)
        set(PRISM_${NAME}_STATUS "- 已禁用")
    endif()
endmacro()

if(APPLE)
    prism_detect_backend(VDSP TRUE PRISM_HAS_VDSP)
else()
    set(PRISM_HAS_VDSP FALSE)
    set(PRISM_VDSP_STATUS "- (非 macOS)")
endif()

find_package(CUDAToolkit QUIET)
prism_detect_backend(CUFFT CUDAToolkit_FOUND PRISM_HAS_CUFFT)
if(PRISM_HAS_CUFFT AND CUDAToolkit_FOUND)
    set(PRISM_CUFFT_STATUS "✓ 已启用 (CUDA ${CUDAToolkit_VERSION})")
elseif(PRISM_HAS_CUFFT AND NOT CUDAToolkit_FOUND)
    set(PRISM_HAS_CUFFT FALSE)
    set(PRISM_CUFFT_STATUS "✗ 未找到 CUDA Toolkit")
else()
    set(PRISM_CUFFT_STATUS "- 已禁用")
endif()

find_package(hipfft QUIET)
prism_detect_backend(HIPFFT hipfft_FOUND PRISM_HAS_HIPFFT)
if(PRISM_HAS_HIPFFT AND hipfft_FOUND)
    set(PRISM_HIPFFT_STATUS "✓ 已启用")
elseif(PRISM_HAS_HIPFFT AND NOT hipfft_FOUND)
    set(PRISM_HAS_HIPFFT FALSE)
    set(PRISM_HIPFFT_STATUS "✗ 未找到 hipfft")
else()
    set(PRISM_HIPFFT_STATUS "- 已禁用")
endif()

find_package(Halide REQUIRED)
set(PRISM_HALIDE_STATUS "✓ 已启用 (${Halide_VERSION})")
message(STATUS "[核心依赖] Halide: ${Halide_VERSION}")

# OpenCL 检测
find_package(OpenCL QUIET)

find_package(PNG QUIET)
find_package(JPEG QUIET)

set(PRISM_AUTOTUNE_PNG_CFLAGS "")
set(PRISM_AUTOTUNE_PNG_LIBS "")
if(PNG_FOUND)
    foreach(_dir IN LISTS PNG_INCLUDE_DIRS)
        string(APPEND PRISM_AUTOTUNE_PNG_CFLAGS " -I${_dir}")
    endforeach()
    string(STRIP "${PRISM_AUTOTUNE_PNG_CFLAGS}" PRISM_AUTOTUNE_PNG_CFLAGS)
    foreach(_lib IN LISTS PNG_LIBRARIES)
        string(APPEND PRISM_AUTOTUNE_PNG_LIBS " ${_lib}")
    endforeach()
    string(STRIP "${PRISM_AUTOTUNE_PNG_LIBS}" PRISM_AUTOTUNE_PNG_LIBS)
endif()

set(PRISM_AUTOTUNE_JPEG_CFLAGS "")
set(PRISM_AUTOTUNE_JPEG_LIBS "")
if(JPEG_FOUND)
    set(_jpeg_includes "")
    if(DEFINED JPEG_INCLUDE_DIR)
        list(APPEND _jpeg_includes "${JPEG_INCLUDE_DIR}")
    endif()
    if(DEFINED JPEG_INCLUDE_DIRS)
        list(APPEND _jpeg_includes ${JPEG_INCLUDE_DIRS})
    endif()
    foreach(_dir IN LISTS _jpeg_includes)
        string(APPEND PRISM_AUTOTUNE_JPEG_CFLAGS " -I${_dir}")
    endforeach()
    string(STRIP "${PRISM_AUTOTUNE_JPEG_CFLAGS}" PRISM_AUTOTUNE_JPEG_CFLAGS)
    if(DEFINED JPEG_LIBRARIES)
        foreach(_lib IN LISTS JPEG_LIBRARIES)
            string(APPEND PRISM_AUTOTUNE_JPEG_LIBS " ${_lib}")
        endforeach()
    elseif(DEFINED JPEG_LIBRARY)
        string(APPEND PRISM_AUTOTUNE_JPEG_LIBS " ${JPEG_LIBRARY}")
    endif()
    string(STRIP "${PRISM_AUTOTUNE_JPEG_LIBS}" PRISM_AUTOTUNE_JPEG_LIBS)
    unset(_jpeg_includes)
endif()

set(PRISM_AUTOTUNE_FRAMEWORKS "")
if(APPLE)
    set(PRISM_AUTOTUNE_FRAMEWORKS "-framework Metal -framework Foundation")
endif()

set(VKFFT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/vkfft/vkFFT")
set(VKFFT_METALCPP_DIR "${CMAKE_SOURCE_DIR}/external/vkfft/metal-cpp")

if(EXISTS "${VKFFT_INCLUDE_DIR}/vkFFT.h")
    if("${PRISM_USE_VKFFT}" STREQUAL "OFF")
        set(PRISM_HAS_VKFFT FALSE)
        set(PRISM_VKFFT_STATUS "- 已禁用")
    elseif(DEFINED PRISM_VKFFT_BACKEND)
        # 用户指定后端 (1=CUDA, 2=HIP, 3=OpenCL, 5=Metal)
        set(PRISM_HAS_VKFFT TRUE)
        set(VKFFT_BACKEND_VALUE ${PRISM_VKFFT_BACKEND})
        if(VKFFT_BACKEND_VALUE EQUAL 1)
            set(PRISM_VKFFT_STATUS "✓ 已启用 (CUDA - 强制)")
        elseif(VKFFT_BACKEND_VALUE EQUAL 2)
            set(PRISM_VKFFT_STATUS "✓ 已启用 (HIP - 强制)")
        elseif(VKFFT_BACKEND_VALUE EQUAL 3)
            set(PRISM_VKFFT_STATUS "✓ 已启用 (OpenCL - 强制)")
        elseif(VKFFT_BACKEND_VALUE EQUAL 5)
            set(PRISM_VKFFT_STATUS "✓ 已启用 (Metal - 强制)")
        else()
            message(FATAL_ERROR "无效的 PRISM_VKFFT_BACKEND 值: ${PRISM_VKFFT_BACKEND} (有效值: 1=CUDA, 2=HIP, 3=OpenCL, 5=Metal)")
        endif()
    elseif(APPLE)
        # macOS: 自动使用 Metal 后端
        set(PRISM_HAS_VKFFT TRUE)
        set(VKFFT_BACKEND_VALUE 5)
        set(PRISM_VKFFT_STATUS "✓ 已启用 (Metal)")
    elseif(PRISM_HAS_CUFFT)
        # CUDA 平台
        set(PRISM_HAS_VKFFT TRUE)
        set(VKFFT_BACKEND_VALUE 1)
        set(PRISM_VKFFT_STATUS "✓ 已启用 (CUDA)")
    elseif(PRISM_HAS_HIPFFT)
        # HIP/ROCm 平台
        set(PRISM_HAS_VKFFT TRUE)
        set(VKFFT_BACKEND_VALUE 2)
        set(PRISM_VKFFT_STATUS "✓ 已启用 (HIP)")
    elseif(OpenCL_FOUND)
        # OpenCL 后端
        set(PRISM_HAS_VKFFT TRUE)
        set(VKFFT_BACKEND_VALUE 3)
        set(PRISM_VKFFT_STATUS "✓ 已启用 (OpenCL)")
    else()
        set(PRISM_HAS_VKFFT FALSE)
        set(PRISM_VKFFT_STATUS "✗ 无可用 GPU API")
    endif()
else()
    set(PRISM_HAS_VKFFT FALSE)
    set(PRISM_VKFFT_STATUS "✗ 未找到 (git submodule update --init)")
endif()

# ============================================================================
# GPU 后端选择（用于 Halide 目标）
# 优先级：Metal > CUDA > HIP > OpenCL > CPU
# ============================================================================

if(APPLE)
    set(PRISM_GPU_BACKEND "Metal")
    set(PRISM_GPU_STATUS "✓ Metal")
elseif(CUDAToolkit_FOUND)
    set(PRISM_GPU_BACKEND "CUDA")
    set(PRISM_GPU_STATUS "✓ CUDA")
elseif(hipfft_FOUND)
    set(PRISM_GPU_BACKEND "HIP")
    set(PRISM_GPU_STATUS "✓ HIP/ROCm")
elseif(OpenCL_FOUND)
    set(PRISM_GPU_BACKEND "OpenCL")
    set(PRISM_GPU_STATUS "✓ OpenCL (回退)")
else()
    set(PRISM_GPU_BACKEND "CPU")
    set(PRISM_GPU_STATUS "- CPU (无 GPU)")
endif()

# ============================================================================
# 库目标
# ============================================================================

set(PRISM_SOURCES
    src/dsl/signal.cpp
    src/dsl/ops.cpp
    src/dsl/filter.cpp
    src/dsl/modem.cpp
    src/runtime/fft.cpp
    src/runtime/executor.cpp
    src/runtime/schedule_config.cpp
    src/runtime/arithmetic_handlers.cpp
    src/runtime/convolution_handlers.cpp
    src/runtime/filter_handlers.cpp
    src/runtime/modem_handlers.cpp
    src/backend/fft_backend.cpp
    src/simulation/rng.cpp
)

# 各 FFT 后端源文件（条件编译）
if(PRISM_HAS_VDSP)
    list(APPEND PRISM_SOURCES src/backend/fft_vdsp.cpp)
endif()

if(PRISM_HAS_CUFFT)
    list(APPEND PRISM_SOURCES src/backend/fft_cufft.cpp)
endif()

if(PRISM_HAS_HIPFFT)
    list(APPEND PRISM_SOURCES src/backend/fft_hipfft.cpp)
endif()

# vkFFT 后端（根据具体 GPU API 选择实现文件）
if(PRISM_HAS_VKFFT)
    if(VKFFT_BACKEND_VALUE EQUAL 1)
        list(APPEND PRISM_SOURCES src/backend/vkfft/vkfft_cuda.cpp)
    elseif(VKFFT_BACKEND_VALUE EQUAL 2)
        list(APPEND PRISM_SOURCES src/backend/vkfft/vkfft_hip.cpp)
    elseif(VKFFT_BACKEND_VALUE EQUAL 3)
        list(APPEND PRISM_SOURCES src/backend/vkfft/vkfft_opencl.cpp)
    elseif(VKFFT_BACKEND_VALUE EQUAL 5)
        list(APPEND PRISM_SOURCES src/backend/vkfft/vkfft_metal.mm)
    endif()
endif()

set(PRISM_HEADERS
    include/prism/prism.h
    include/prism/types.h
    include/prism/dsl/signal.h
    include/prism/dsl/ops.h
    include/prism/dsl/filter.h
    include/prism/dsl/modem.h
    include/prism/runtime/executor.h
    include/prism/runtime/fft.h
    include/prism/runtime/halide_builder.h
    include/prism/backend/fft_backend.h
    include/prism/backend/backend_registry.h
)

add_library(prism ${PRISM_SOURCES} ${PRISM_HEADERS})
target_include_directories(prism
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(prism PUBLIC Halide::Halide)
function(prism_define_autoscheduler_path target autoscheduler macro_name)
    if(NOT TARGET ${autoscheduler})
        return()
    endif()
    get_target_property(_loc ${autoscheduler} IMPORTED_LOCATION)
    if(NOT _loc)
        get_target_property(_loc ${autoscheduler} IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT _loc)
        get_target_property(_loc ${autoscheduler} IMPORTED_LOCATION_NOCONFIG)
    endif()
    if(_loc)
        target_compile_definitions(${target} PRIVATE ${macro_name}="${_loc}")
    endif()
endfunction()

function(prism_define_autoscheduler_path_by_name target basename macro_name)
    if(NOT TARGET Halide::Halide)
        return()
    endif()
    get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION)
    if(NOT _halide_loc)
        get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT _halide_loc)
        get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION_NOCONFIG)
    endif()
    if(NOT _halide_loc)
        return()
    endif()
    get_filename_component(_halide_dir "${_halide_loc}" DIRECTORY)
    set(_candidate "${_halide_dir}/libautoschedule_${basename}.so")
    if(EXISTS "${_candidate}")
        target_compile_definitions(${target} PRIVATE ${macro_name}="${_candidate}")
    endif()
endfunction()

function(prism_add_halide_tools_include target)
    if(NOT TARGET Halide::Halide)
        return()
    endif()
    get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION)
    if(NOT _halide_loc)
        get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT _halide_loc)
        get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION_NOCONFIG)
    endif()
    if(NOT _halide_loc)
        return()
    endif()
    get_filename_component(_halide_dir "${_halide_loc}" DIRECTORY)
    get_filename_component(_halide_prefix "${_halide_dir}" DIRECTORY)
    set(_tools_dir "${_halide_prefix}/share/tools")
    if(EXISTS "${_tools_dir}/halide_benchmark.h")
        target_include_directories(${target} PRIVATE "${_tools_dir}")
    endif()
endfunction()

function(prism_get_halide_prefix out_var)
    if(NOT TARGET Halide::Halide)
        return()
    endif()
    get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION)
    if(NOT _halide_loc)
        get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT _halide_loc)
        get_target_property(_halide_loc Halide::Halide IMPORTED_LOCATION_NOCONFIG)
    endif()
    if(NOT _halide_loc)
        return()
    endif()
    get_filename_component(_halide_dir "${_halide_loc}" DIRECTORY)
    get_filename_component(_halide_prefix "${_halide_dir}" DIRECTORY)
    set(${out_var} "${_halide_prefix}" PARENT_SCOPE)
endfunction()

prism_get_halide_prefix(PRISM_HALIDE_PREFIX)
if(PRISM_HALIDE_PREFIX)
    set(PRISM_AUTOTUNE_DIR "${CMAKE_BINARY_DIR}/autotune")
    add_custom_target(prism_autotune_setup
        COMMAND ${CMAKE_COMMAND}
            -DAUTOTUNE_DIR="${PRISM_AUTOTUNE_DIR}"
            -DHALIDE_PREFIX="${PRISM_HALIDE_PREFIX}"
            -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
            -DCREATE_FAKE_NVIDIA_SMI=${PRISM_AUTOTUNE_FAKE_NVIDIA_SMI}
            -DANDERSON2021_SCRIPTS_DIR="${PRISM_ANDERSON2021_SCRIPTS_DIR}"
            -DPRISM_AUTOTUNE_PNG_CFLAGS="${PRISM_AUTOTUNE_PNG_CFLAGS}"
            -DPRISM_AUTOTUNE_PNG_LIBS="${PRISM_AUTOTUNE_PNG_LIBS}"
            -DPRISM_AUTOTUNE_JPEG_CFLAGS="${PRISM_AUTOTUNE_JPEG_CFLAGS}"
            -DPRISM_AUTOTUNE_JPEG_LIBS="${PRISM_AUTOTUNE_JPEG_LIBS}"
            -DPRISM_AUTOTUNE_FRAMEWORKS="${PRISM_AUTOTUNE_FRAMEWORKS}"
            -P "${CMAKE_SOURCE_DIR}/cmake/autotune_setup.cmake"
        COMMENT "Prepare Halide autotune environment under ${PRISM_AUTOTUNE_DIR}"
    )
endif()

function(prism_link_halide_generator target)
    if(TARGET Halide::Generator)
        target_link_libraries(${target} PRIVATE Halide::Generator)
    else()
        target_link_libraries(${target} PRIVATE Halide::Halide)
    endif()
endfunction()

if(NOT APPLE)
    if(TARGET Halide::Mullapudi2016)
        target_link_libraries(prism PUBLIC Halide::Mullapudi2016)
    endif()
    if(TARGET Halide::Adams2019)
        target_link_libraries(prism PUBLIC Halide::Adams2019)
    endif()
    if(TARGET Halide::Anderson2021)
        target_link_libraries(prism PUBLIC Halide::Anderson2021)
    endif()
endif()

prism_define_autoscheduler_path(prism Halide::Adams2019
                                PRISM_AUTOSCHEDULER_ADAMS2019_PATH)
prism_define_autoscheduler_path(prism Halide::Anderson2021
                                PRISM_AUTOSCHEDULER_ANDERSON2021_PATH)
prism_define_autoscheduler_path_by_name(prism adams2019
                                        PRISM_AUTOSCHEDULER_ADAMS2019_PATH)
prism_define_autoscheduler_path_by_name(prism anderson2021
                                        PRISM_AUTOSCHEDULER_ANDERSON2021_PATH)
prism_define_autoscheduler_path_by_name(prism mullapudi2016
                                        PRISM_AUTOSCHEDULER_MULLAPUDI2016_PATH)

if(PRISM_HAS_VDSP)
    target_compile_definitions(prism PUBLIC PRISM_HAS_VDSP=1)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(prism PUBLIC ${ACCELERATE_FRAMEWORK})
endif()

if(PRISM_HAS_CUFFT)
    target_compile_definitions(prism PUBLIC PRISM_HAS_CUFFT=1 PRISM_BACKEND_CUDA=1)
    target_link_libraries(prism PUBLIC CUDA::cufft CUDA::cublas)
endif()

if(PRISM_HAS_HIPFFT)
    target_compile_definitions(prism PUBLIC PRISM_HAS_HIPFFT=1 PRISM_BACKEND_HIP=1)
    target_link_libraries(prism PUBLIC hip::hipfft)
endif()

if(PRISM_HAS_VKFFT)
    target_compile_definitions(prism PUBLIC PRISM_HAS_VKFFT=1)
    target_compile_definitions(prism PRIVATE VKFFT_BACKEND=${VKFFT_BACKEND_VALUE})
    target_include_directories(prism PRIVATE
        ${VKFFT_INCLUDE_DIR}
        ${VKFFT_METALCPP_DIR}
    )

    # Metal 后端需要 Metal 和 Foundation 框架
    if(VKFFT_BACKEND_VALUE EQUAL 5)
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        target_link_libraries(prism PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()

    # OpenCL 后端需要链接 OpenCL
    if(VKFFT_BACKEND_VALUE EQUAL 3)
        target_link_libraries(prism PUBLIC OpenCL::OpenCL)
    endif()
endif()

# GPU 后端宏定义（用于 Halide 目标选择）
target_compile_definitions(prism PUBLIC PRISM_GPU_BACKEND_${PRISM_GPU_BACKEND}=1)

# OpenCL 链接（如果使用）
if(PRISM_GPU_BACKEND STREQUAL "OpenCL")
    target_link_libraries(prism PUBLIC OpenCL::OpenCL)
endif()

# Metal 链接（如果使用）
if(PRISM_GPU_BACKEND STREQUAL "Metal")
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(prism PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# CUDA 运行时链接（如果使用）
if(PRISM_GPU_BACKEND STREQUAL "CUDA")
    target_link_libraries(prism PUBLIC CUDA::cudart)
endif()

# ============================================================================
# 测试
# ============================================================================

if(PRISM_BUILD_TESTS)
    enable_testing()

    add_executable(test_basic_ops tests/test_basic_ops.cpp)
    target_link_libraries(test_basic_ops PRIVATE prism Halide::Halide)
    add_test(NAME BasicOps COMMAND test_basic_ops)

    add_executable(test_advanced_ops tests/test_advanced_ops.cpp)
    target_link_libraries(test_advanced_ops PRIVATE prism Halide::Halide)
    add_test(NAME AdvancedOps COMMAND test_advanced_ops)

    add_executable(test_filter tests/test_filter.cpp)
    target_link_libraries(test_filter PRIVATE prism Halide::Halide)
    add_test(NAME Filter COMMAND test_filter)

    add_executable(test_modem tests/test_modem.cpp)
    target_link_libraries(test_modem PRIVATE prism Halide::Halide)
    add_test(NAME Modem COMMAND test_modem)

    add_executable(test_fft tests/test_fft.cpp)
    target_link_libraries(test_fft PRIVATE prism)
    add_test(NAME FFT COMMAND test_fft)

    add_executable(test_simulation tests/test_simulation.cpp)
    target_link_libraries(test_simulation PRIVATE prism)
    add_test(NAME Simulation COMMAND test_simulation)
endif()

# ============================================================================
# Examples
# ============================================================================

if(PRISM_BUILD_EXAMPLES)
    add_executable(example_apm_basic examples/apm_basic/main.cpp examples/example_helper.cpp)
    target_link_libraries(example_apm_basic PRIVATE prism Halide::Halide)
    target_include_directories(example_apm_basic PRIVATE
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/external/tomlplusplus/include
    )
    prism_add_halide_tools_include(example_apm_basic)

    add_executable(example_apm_basic_autotune
        examples/apm_basic/autotune.cpp
        examples/example_helper.cpp
    )
    target_link_libraries(example_apm_basic_autotune PRIVATE prism)
    prism_link_halide_generator(example_apm_basic_autotune)
    target_include_directories(example_apm_basic_autotune PRIVATE
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/external/tomlplusplus/include
    )
    prism_add_halide_tools_include(example_apm_basic_autotune)

    add_executable(example_apm_dsss examples/apm_dsss/main.cpp examples/example_helper.cpp)
    target_link_libraries(example_apm_dsss PRIVATE prism Halide::Halide)
    target_include_directories(example_apm_dsss PRIVATE
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/external/tomlplusplus/include
    )
    prism_add_halide_tools_include(example_apm_dsss)

    add_executable(example_apm_dsss_autotune
        examples/apm_dsss/autotune.cpp
        examples/example_helper.cpp
    )
    target_link_libraries(example_apm_dsss_autotune PRIVATE prism)
    prism_link_halide_generator(example_apm_dsss_autotune)
    target_include_directories(example_apm_dsss_autotune PRIVATE
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/external/tomlplusplus/include
    )
    prism_add_halide_tools_include(example_apm_dsss_autotune)

    add_executable(example_apm_dsss_eq examples/apm_dsss_eq/main.cpp examples/example_helper.cpp)
    target_link_libraries(example_apm_dsss_eq PRIVATE prism Halide::Halide)
    target_include_directories(example_apm_dsss_eq PRIVATE
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/external/tomlplusplus/include
    )
    prism_add_halide_tools_include(example_apm_dsss_eq)

    add_executable(example_apm_dsss_eq_autotune
        examples/apm_dsss_eq/autotune.cpp
        examples/example_helper.cpp
    )
    target_link_libraries(example_apm_dsss_eq_autotune PRIVATE prism)
    prism_link_halide_generator(example_apm_dsss_eq_autotune)
    target_include_directories(example_apm_dsss_eq_autotune PRIVATE
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/external/tomlplusplus/include
    )
    prism_add_halide_tools_include(example_apm_dsss_eq_autotune)
endif()

# ============================================================================
# Benchmark
# ============================================================================

if(PRISM_BUILD_BENCHMARKS)
    add_executable(bench_ops benchmark/bench_ops.cpp)
    target_link_libraries(bench_ops PRIVATE prism Halide::Halide)

    add_executable(bench_fft benchmark/bench_fft.cpp)
    target_link_libraries(bench_fft PRIVATE prism)

    add_executable(bench_filter benchmark/bench_filter.cpp)
    target_link_libraries(bench_filter PRIVATE prism Halide::Halide)

    add_executable(bench_adv_ops benchmark/bench_adv_ops.cpp)
    target_link_libraries(bench_adv_ops PRIVATE prism Halide::Halide)

    add_executable(bench_modem benchmark/bench_modem.cpp)
    target_link_libraries(bench_modem PRIVATE prism Halide::Halide)

    add_executable(bench_stress benchmark/bench_stress.cpp)
    target_link_libraries(bench_stress PRIVATE prism Halide::Halide)

endif()

# ============================================================================
# 文档生成
# ============================================================================

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/docs/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "生成 Doxygen 文档..."
        VERBATIM
    )

    message(STATUS "[文档] Doxygen: 已找到，可使用 'cmake --build build --target docs' 生成文档")
else()
    message(STATUS "[文档] Doxygen: 未找到，文档生成不可用")
endif()

# ============================================================================
# 安装
# ============================================================================

include(GNUInstallDirs)

install(TARGETS prism
    EXPORT prismTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/prism
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "PRISM 配置摘要")
message(STATUS "============================================================")
message(STATUS "")
message(STATUS "[基本信息]")
message(STATUS "  版本:              ${PROJECT_VERSION}")
message(STATUS "  构建类型:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ 标准:          ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "[编译器]")
message(STATUS "  C 编译器:          ${CMAKE_C_COMPILER}")
message(STATUS "  C 编译器版本:      ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++ 编译器:        ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ 编译器版本:    ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "[后端 - Vendor API]")
message(STATUS "  Apple vDSP:        ${PRISM_VDSP_STATUS}")
message(STATUS "  NVIDIA cuFFT:      ${PRISM_CUFFT_STATUS}")
message(STATUS "  AMD hipFFT:        ${PRISM_HIPFFT_STATUS}")
message(STATUS "")
message(STATUS "[核心依赖]")
message(STATUS "  Halide:            ${PRISM_HALIDE_STATUS}")
message(STATUS "")
message(STATUS "[FFT Fallback]")
message(STATUS "  vkFFT:             ${PRISM_VKFFT_STATUS}")
message(STATUS "")
message(STATUS "[构建选项]")
message(STATUS "  构建测试:          ${PRISM_BUILD_TESTS}")
message(STATUS "  构建示例:          ${PRISM_BUILD_EXAMPLES}")
message(STATUS "  构建基准:          ${PRISM_BUILD_BENCHMARKS}")
message(STATUS "")
message(STATUS "============================================================")
message(STATUS "  后端配置: AUTO=自动检测 | 在 cmake/local.cmake 中")
message(STATUS "            设置 ON/OFF 可强制启用/禁用特定后端")
message(STATUS "============================================================")
message(STATUS "")
