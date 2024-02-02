file(GLOB filesToLink RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../data/mnist-datasets/* *.config)
foreach(f ${filesToLink})
    set(ff)
    get_filename_component(ff ${f} NAME)

    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_CURRENT_SOURCE_DIR}/${f}
            ${CMAKE_CURRENT_BINARY_DIR}/${ff})
endforeach()

set(targetName "mnist")
add_executable(${targetName} main.cpp)

# header and lib
target_include_directories(${targetName} PRIVATE ../src)
target_link_libraries(${targetName} mnistLib)

# compile and link flags
target_compile_features(${targetName} PRIVATE cxx_std_20)
target_compile_options(${targetName} PRIVATE -Wall -Wextra -pedantic -Werror -pg -g)
target_link_options(${targetName} PRIVATE -pg)