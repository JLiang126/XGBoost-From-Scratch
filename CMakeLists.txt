cmake_minimum_required(VERSION 3.15)
project(XGBoostFromScratch)

# 1. Use C++17 for modern features like unique_ptr and move semantics
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Tell CMake to find the pybind11 package installed via Homebrew
find_package(pybind11 REQUIRED)

# 3. List ALL your .cpp files so the compiler knows what to build
# We include the engine files and the bindings file.
set(SOURCES
    src/dataloader/DataMatrix.cpp
    src/engine/XGBoost.cpp
    src/objective/Loss.cpp
    src/XGTree/XGTree.cpp
    src/XGTree/XGNode.cpp
    python-packages/bindings.cpp
)

# 4. Create the Python module. 
# "my_xgboost" is the name you will use in 'import my_xgboost'
pybind11_add_module(my_xgboost ${SOURCES})

# 5. Tell the compiler where your .hpp files are
# This lets you write #include "Tree.hpp" instead of messy long paths.
target_include_directories(my_xgboost PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# 6. Final link: This connects the pybind11 logic to your binary
target_link_libraries(my_xgboost PRIVATE pybind11::module)