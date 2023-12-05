set(SRC_FILES)
file(GLOB SRC_FILES
        mnist
        tensor
        network)

add_library(mnistLib SHARED
        ${SRC_FILES})

set_target_properties(mnistLib PROPERTIES LINKER_LANGUAGE CXX)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mnistLib PUBLIC OpenMP::OpenMP_CXX)
endif()

target_include_directories(mnistLib PRIVATE .)

# fpe support for debug mode
if(CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(mnistLib PUBLIC -fnon-call-exceptions)
endif()
