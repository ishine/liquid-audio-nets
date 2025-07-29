cmake_minimum_required(VERSION 3.16)
project(liquid-audio-nets VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(ENABLE_ARM_OPTIMIZATIONS "Enable ARM NEON/CMSIS optimizations" OFF)
option(ENABLE_EMBEDDED_BUILD "Build for embedded targets" OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -DNDEBUG)
    endif()
endif()

# ARM-specific optimizations
if(ENABLE_ARM_OPTIMIZATIONS)
    find_path(CMSIS_INCLUDE_DIR arm_math.h 
              HINTS /usr/include/CMSIS /opt/CMSIS/CMSIS/DSP/Include)
    if(CMSIS_INCLUDE_DIR)
        include_directories(${CMSIS_INCLUDE_DIR})
        add_definitions(-DARM_MATH_CM4 -D__FPU_PRESENT=1)
        message(STATUS "Found CMSIS-DSP at ${CMSIS_INCLUDE_DIR}")
    endif()
endif()

# Embedded build configuration
if(ENABLE_EMBEDDED_BUILD)
    set(CMAKE_SYSTEM_NAME Generic)
    set(CMAKE_SYSTEM_PROCESSOR arm)
    add_definitions(-DEMBEDDED_BUILD)
    # Disable standard library features for embedded
    add_compile_options(-fno-exceptions -fno-rtti -ffunction-sections -fdata-sections)
endif()

# Include directories
include_directories(include)
include_directories(src)

# Core library
add_library(liquid_audio_core STATIC
    src/core/lnn.cpp
    src/core/adaptive_timestep.cpp
    src/core/ode_solver.cpp
    src/core/feature_extractor.cpp
    src/core/power_manager.cpp
)

target_include_directories(liquid_audio_core PUBLIC include)

# ARM optimized kernels
if(ENABLE_ARM_OPTIMIZATIONS)
    add_library(liquid_audio_arm STATIC
        src/platform/arm/cmsis_kernels.c
        src/platform/arm/neon_optimizations.cpp
    )
    target_link_libraries(liquid_audio_core liquid_audio_arm)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(liquid_audio_py src/python/bindings.cpp)
    target_link_libraries(liquid_audio_py PRIVATE liquid_audio_core)
endif()

# Installation
install(TARGETS liquid_audio_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    liquid-audio-netsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    cmake/liquid-audio-netsConfig.cmake.in
    liquid-audio-netsConfig.cmake
    INSTALL_DESTINATION lib/cmake/liquid-audio-nets
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/liquid-audio-netsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/liquid-audio-netsConfigVersion.cmake
    DESTINATION lib/cmake/liquid-audio-nets
)