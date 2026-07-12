cmake_minimum_required(VERSION 3.15)

project(
    ClothOpt
    LANGUAGES C CXX
)

# ============================================================
# Basic configuration
# ============================================================

# Prevent in-source builds.
if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR
        "Source and build directories cannot be the same. "
        "Please create and use a separate build directory."
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type for single-config generators.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Choose the build type: Debug, Release, RelWithDebInfo, MinSizeRel"
        FORCE
    )
endif()

# Configure output locations.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ============================================================
# Conda environment
# ============================================================

if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" CONDA_PREFIX_CMAKE)

    message(STATUS "Using Conda environment: ${CONDA_PREFIX_CMAKE}")

    # Prefer packages installed inside the active Conda environment.
    list(PREPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX_CMAKE}")

    # Some conda-forge OpenVDB builds provide FindOpenVDB.cmake here
    # instead of, or in addition to, OpenVDBConfig.cmake.
    list(PREPEND CMAKE_MODULE_PATH
        "${CONDA_PREFIX_CMAKE}/lib/cmake/OpenVDB"
        "${CONDA_PREFIX_CMAKE}/share/cmake/OpenVDB"
    )

    # Help pkg-config locate Conda packages.
    set(
        ENV{PKG_CONFIG_PATH}
        "${CONDA_PREFIX_CMAKE}/lib/pkgconfig:${CONDA_PREFIX_CMAKE}/share/pkgconfig:$ENV{PKG_CONFIG_PATH}"
    )
else()
    message(WARNING
        "CONDA_PREFIX is not defined. Activate the Conda environment "
        "before running CMake if dependencies were installed with Conda."
    )
endif()

# ============================================================
# Compiler options
# ============================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    message(STATUS "Using Clang/GCC compiler flags")

    add_compile_options(
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-deprecated-declarations
        -Wno-missing-braces
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        message(STATUS "Setting Clang-specific options")

        add_compile_options(
            -ferror-limit=5
            -fcolor-diagnostics
        )

        add_compile_options(
            "$<$<CONFIG:Debug>:-fsanitize=address>"
            "$<$<CONFIG:Debug>:-fno-omit-frame-pointer>"
        )

        add_link_options(
            "$<$<CONFIG:Debug>:-fsanitize=address>"
        )

    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Setting GCC-specific options")

        add_compile_options(
            -fmax-errors=5
            -Wno-maybe-uninitialized
            -Wno-format-zero-length
            -Wno-unused-but-set-parameter
            -Wno-unused-but-set-variable
        )
    endif()

    add_compile_options(
        "$<$<CONFIG:Debug>:-g3>"
        "$<$<CONFIG:Release>:-O3>"
        "$<$<CONFIG:Release>:-march=native>"
        "$<$<CONFIG:Release>:-DNDEBUG>"
        "$<$<CONFIG:RelWithDebInfo>:-O2>"
        "$<$<CONFIG:RelWithDebInfo>:-g>"
        "$<$<CONFIG:MinSizeRel>:-Os>"
        "$<$<CONFIG:MinSizeRel>:-DNDEBUG>"
    )

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(STATUS "Using MSVC compiler flags")

    add_compile_options(
        /W4
        /MP
        /wd4267
        /wd4244
        /wd4305
    )

    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        _USE_MATH_DEFINES
    )

else()
    message(FATAL_ERROR
        "Unrecognized compiler: ${CMAKE_CXX_COMPILER_ID}"
    )
endif()

# ============================================================
# Dependencies
# ============================================================

# ------------------------------------------------------------
# Eigen
# ------------------------------------------------------------

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

message(STATUS "Found Eigen3: ${Eigen3_VERSION}")

# ------------------------------------------------------------
# Intel oneTBB
# ------------------------------------------------------------

set(HAVE_TBB FALSE)

find_package(TBB CONFIG QUIET)

if(TBB_FOUND AND TARGET TBB::tbb)
    message(STATUS "Found Intel oneTBB: ${TBB_VERSION}")
    set(HAVE_TBB TRUE)
else()
    find_package(PkgConfig QUIET)

    if(PkgConfig_FOUND)
        pkg_check_modules(TBB_PKG QUIET IMPORTED_TARGET tbb)

        if(TBB_PKG_FOUND)
            message(STATUS "Found TBB via pkg-config: ${TBB_PKG_VERSION}")
            set(HAVE_TBB TRUE)
            set(TBB_LINK_TARGET PkgConfig::TBB_PKG)
        endif()
    endif()

    if(NOT HAVE_TBB)
        find_path(
            TBB_INCLUDE_DIR
            NAMES tbb/tbb.h
            HINTS
                "${CONDA_PREFIX_CMAKE}/include"
            PATHS
                /usr/include
                /usr/local/include
                /opt/intel/oneapi/tbb/latest/include
        )

        find_library(
            TBB_LIBRARY
            NAMES tbb
            HINTS
                "${CONDA_PREFIX_CMAKE}/lib"
            PATHS
                /usr/lib
                /usr/lib/x86_64-linux-gnu
                /usr/local/lib
                /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8
        )

        if(TBB_INCLUDE_DIR AND TBB_LIBRARY)
            message(STATUS "Found TBB manually: ${TBB_LIBRARY}")

            add_library(TBB::tbb UNKNOWN IMPORTED)

            set_target_properties(
                TBB::tbb
                PROPERTIES
                    IMPORTED_LOCATION "${TBB_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}"
            )

            set(HAVE_TBB TRUE)
        else()
            message(WARNING
                "Intel oneTBB was not found. "
                "OpenMP will be used as a fallback if available."
            )
        endif()
    endif()
endif()

if(HAVE_TBB AND NOT DEFINED TBB_LINK_TARGET)
    set(TBB_LINK_TARGET TBB::tbb)
endif()

# ------------------------------------------------------------
# OpenMP fallback
# ------------------------------------------------------------

set(HAVE_OPENMP FALSE)

find_package(OpenMP QUIET)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
    set(HAVE_OPENMP TRUE)
else()
    message(WARNING
        "OpenMP was not found. Parallel processing may be limited."
    )
endif()

# ------------------------------------------------------------
# OpenVDB
# ------------------------------------------------------------

set(OpenVDB_FOUND FALSE)

# First try config mode, such as OpenVDBConfig.cmake.
find_package(OpenVDB CONFIG QUIET)

# Fall back to module mode, such as FindOpenVDB.cmake.
if(NOT OpenVDB_FOUND)
    find_package(OpenVDB MODULE REQUIRED)
endif()

# Different OpenVDB packages may export different target names.
if(TARGET OpenVDB::openvdb)
    set(OPENVDB_TARGET OpenVDB::openvdb)

elseif(TARGET openvdb)
    set(OPENVDB_TARGET openvdb)

else()
    message(FATAL_ERROR
        "OpenVDB was found, but neither OpenVDB::openvdb nor "
        "openvdb is available as a CMake target."
    )
endif()

if(DEFINED OpenVDB_VERSION)
    message(STATUS "Found OpenVDB: ${OpenVDB_VERSION}")
elseif(DEFINED OpenVDB_VERSION_STRING)
    message(STATUS "Found OpenVDB: ${OpenVDB_VERSION_STRING}")
else()
    message(STATUS "Found OpenVDB")
endif()

message(STATUS "OpenVDB target: ${OPENVDB_TARGET}")

# ------------------------------------------------------------
# LibXml2
# ------------------------------------------------------------

find_package(LibXml2 REQUIRED)

if(TARGET LibXml2::LibXml2)
    set(LIBXML2_TARGET LibXml2::LibXml2)
else()
    set(LIBXML2_TARGET "${LIBXML2_LIBRARIES}")
endif()

# ============================================================
# Bundled dependencies
# ============================================================

# ------------------------------------------------------------
# Polyscope
# ------------------------------------------------------------

message(STATUS "")
message(STATUS "Building bundled Polyscope")
message(STATUS "")

add_subdirectory("deps/polyscope")

# ------------------------------------------------------------
# Imath
# ------------------------------------------------------------

add_subdirectory("deps/Imath")

# ------------------------------------------------------------
# Alembic
# ------------------------------------------------------------

set(ALEMBIC_BUILD_TESTS OFF CACHE BOOL "Disable Alembic tests" FORCE)
set(ALEMBIC_LIB_USES_TESTS OFF CACHE BOOL "Disable Alembic tests" FORCE)
set(ALEMBIC_SHARED_LIBS OFF CACHE BOOL "Build Alembic statically" FORCE)
set(USE_HDF5 OFF CACHE BOOL "Disable HDF5 support" FORCE)
set(USE_MAYA OFF CACHE BOOL "Disable Maya support" FORCE)
set(USE_PYALEMBIC OFF CACHE BOOL "Disable Python Alembic bindings" FORCE)

add_subdirectory("deps/alembic")

# ============================================================
# Project sources
# ============================================================

file(
    GLOB_RECURSE
    SHARED_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# ============================================================
# Common executable helper
# ============================================================

function(add_main_executable target_name source_file)
    add_executable(
        ${target_name}
        "${source_file}"
        ${SHARED_SOURCES}
    )

    target_include_directories(
        ${target_name}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/fbx/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/alembic/lib"
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen"
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/json"
            "${CMAKE_CURRENT_SOURCE_DIR}/deps"
    )

    # Older FindLibXml2 modules expose include variables rather than
    # an imported target.
    if(LIBXML2_INCLUDE_DIRS)
        target_include_directories(
            ${target_name}
            PRIVATE
                ${LIBXML2_INCLUDE_DIRS}
        )
    elseif(LIBXML2_INCLUDE_DIR)
        target_include_directories(
            ${target_name}
            PRIVATE
                ${LIBXML2_INCLUDE_DIR}
        )
    endif()

    if(DEFINED OpenCASCADE_INCLUDE_DIR AND OpenCASCADE_INCLUDE_DIR)
        target_include_directories(
            ${target_name}
            PRIVATE
                ${OpenCASCADE_INCLUDE_DIR}
        )
    endif()

    target_link_libraries(
        ${target_name}
        PRIVATE
            polyscope
            Eigen3::Eigen
            Alembic
            ${OPENVDB_TARGET}
            ${LIBXML2_TARGET}
    )

    if(HAVE_TBB)
        target_link_libraries(
            ${target_name}
            PRIVATE
                ${TBB_LINK_TARGET}
        )

        target_compile_definitions(
            ${target_name}
            PRIVATE
                HAVE_TBB=1
                TBB_USE_THREADING_TOOLS
        )

        message(STATUS "Linking ${target_name} with Intel oneTBB")

    elseif(HAVE_OPENMP)
        target_link_libraries(
            ${target_name}
            PRIVATE
                OpenMP::OpenMP_CXX
        )

        target_compile_definitions(
            ${target_name}
            PRIVATE
                HAVE_OPENMP=1
        )

        message(STATUS "Linking ${target_name} with OpenMP")

    else()
        target_compile_definitions(
            ${target_name}
            PRIVATE
                NO_PARALLEL=1
        )

        message(STATUS
            "Building ${target_name} without parallel processing"
        )
    endif()

    target_compile_definitions(
        ${target_name}
        PRIVATE
            PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )
endfunction()

# ============================================================
# Main executables
# ============================================================

add_main_executable(Visualization "${CMAKE_CURRENT_SOURCE_DIR}/visual.cpp")
add_main_executable(Simulation "${CMAKE_CURRENT_SOURCE_DIR}/simulation.cpp")
add_main_executable(Optimization "${CMAKE_CURRENT_SOURCE_DIR}/optimization.cpp")

# ============================================================
# Configuration summary
# ============================================================

message(STATUS "")
message(STATUS "=== ClothOpt Configuration Summary ===")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(DEFINED CONDA_PREFIX_CMAKE)
    message(STATUS "Conda environment: ${CONDA_PREFIX_CMAKE}")
else()
    message(STATUS "Conda environment: not detected")
endif()

message(STATUS "OpenVDB target: ${OPENVDB_TARGET}")

if(HAVE_TBB)
    message(STATUS "Parallel processing: Intel oneTBB")
elseif(HAVE_OPENMP)
    message(STATUS "Parallel processing: OpenMP")
else()
    message(STATUS "Parallel processing: None")
endif()

message(STATUS "======================================")
message(STATUS "")