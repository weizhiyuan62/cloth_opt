cmake_minimum_required(VERSION 3.15)
project(ClothOptPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE
    bindings.cpp
    ../src/mesh.cpp
    ../src/cloth.cpp
    ../src/integrator.cpp
    ../src/controller.cpp
)

target_include_directories(_core PRIVATE
    ../include
    ../../deps/eigen
)

# integrator.cpp currently uses the TBB API unconditionally in collision code.
# Prefer a real TBB installation; otherwise provide an API-compatible serial
# backend for this Python extension without changing the simulation source.
find_package(TBB CONFIG QUIET)
if(TBB_FOUND AND TARGET TBB::tbb)
    target_link_libraries(_core PRIVATE TBB::tbb)
else()
    target_include_directories(_core PRIVATE compat)
endif()
target_compile_definitions(_core PRIVATE HAVE_TBB=1)

install(TARGETS _core LIBRARY DESTINATION cloth_opt)
