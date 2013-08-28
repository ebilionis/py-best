cmake_minimum_required(VERSION 2.6)

find_package(Boost REQUIRED COMPONENTS python)

include_directories("${Boost_INCLUDE_DIRS}")
include_directories("./")
include_directories("$ENV{PYTHON_INC_DIR}")
include_directories("$ENV{NUMPY_INC_DIR}")

add_library(_lhs SHARED _lhs.cpp)
target_link_libraries(_lhs ${Boost_LIBRARIES})
set_target_properties(_lhs PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "../best/core")

add_library(_ggsvd SHARED _ggsvd.cpp)
target_link_libraries(_ggsvd ${Boost_LIBRARIES} lapack)
set_target_properties(_ggsvd PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY "../best/core")