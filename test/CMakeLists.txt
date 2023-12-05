file(GLOB filesToLink RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ../data/mnist-datasets/* *.config)
foreach(f ${filesToLink})
    set(ff)
    get_filename_component(ff ${f} NAME)

    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_CURRENT_SOURCE_DIR}/${f}
            ${CMAKE_CURRENT_BINARY_DIR}/${ff})
endforeach()

# setup list of test targets and their c++ sources
set(testNames read_dataset_images read_dataset_labels test_fully_connected test_flatten_pixels
        test_optimized_fully_connected create_sub_dataset)
set(testSources read_dataset_images.cpp read_dataset_labels.cpp test_fully_connected.cpp test_flatten_pixels.cpp
        test_optimized_fully_connected.cpp create_sub_dataset.cpp)

list(LENGTH testNames len1)
math(EXPR len2 "${len1} - 1")

foreach(val RANGE ${len2})
    list(GET testNames ${val} targetName)
    list(GET testSources ${val} targetSource)

    add_executable(${targetName} ${targetSource})

    # header and lib
    target_include_directories(${targetName} PRIVATE ../src)
    target_link_libraries(${targetName} mnistLib)

    # compile and link flags
    target_compile_features(${targetName} PRIVATE cxx_std_20)
    target_compile_options(${targetName} PRIVATE -Wall -Wextra -pedantic -Werror -pg -g)
    target_link_options(${targetName} PRIVATE -pg)
endforeach()