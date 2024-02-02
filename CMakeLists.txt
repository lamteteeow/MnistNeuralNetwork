cmake_minimum_required(VERSION 3.14)

project(AdvPT VERSION 1.0
        DESCRIPTION "AdvPT MNIST project"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(app)
add_subdirectory(src)
add_subdirectory(test)



