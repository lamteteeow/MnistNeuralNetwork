#include "tensor.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <vector>

#define SPACE (" ")

/**
 * @author Junzhe Wang
 * @since 15.12.2024
 *
 * @brief Converts a 32-bit unsigned integer from big-endian format into little-endian format
 *
 * @param {big-endian} The 32-bit unsigned integer in big-endian format.
 *
 * @return The 32-bit unsigned integer in little-endian format.
 * */
uint32_t big_endian_to_lit_endian(uint32_t big_endianian) {

    uint32_t little_endian = 0;

    little_endian |= (big_endianian & 0xff'00'00'00) >> 24;
    little_endian |= (big_endianian & 0x00'ff'00'00) >> 8;
    little_endian |= (big_endianian & 0x00'00'ff'00) << 8;
    little_endian |= (big_endianian & 0x00'00'00'ff) << 24;

    return little_endian;

}

/**
 * @author Junzhe Wang
 * @since 15.12.2024
 *
 * @brief It reads the MNIST image data from a binary file and stores it in a vector of tensors.
 *
 * @param {image_file_name} The name of the binary file containing the image(s).
 * @param {images} A reference to the tensor of vectors.
 *
 * @return The image count.
 * */
uint32_t image_rd(std::string const& image_file_name, std::vector<Tensor<double>>& images) {

    std::ifstream input( image_file_name, std::ios::binary );
    if ( !input.is_open() ) {

        std::cerr 
            << "Error: Failed to open file [" << image_file_name << "]" 
            << std::endl;

        exit( EXIT_FAILURE );

    }

    uint32_t MAGIC;
    uint32_t IMAGE_COUNT;
    uint32_t ROWS;
    uint32_t COLS;

    input.read( reinterpret_cast<char*>(&MAGIC)      , 4 );
    input.read( reinterpret_cast<char*>(&IMAGE_COUNT), 4 );
    input.read( reinterpret_cast<char*>(&ROWS)       , 4 );
    input.read( reinterpret_cast<char*>(&COLS)       , 4 );

    MAGIC       = big_endian_to_lit_endian( MAGIC );
    IMAGE_COUNT = big_endian_to_lit_endian( IMAGE_COUNT );
    ROWS        = big_endian_to_lit_endian( ROWS );
    COLS        = big_endian_to_lit_endian( COLS );

    if (MAGIC != 0x0000'0803 ) {
    
        std::cerr 
            << "Error: Failed to read image. "
            << "File [" << image_file_name << "] is not in valid format!"
            << std::endl;

        exit( EXIT_FAILURE );

    }

    uint32_t const image_size = ROWS * COLS;
    images.reserve( image_size );

    for (uint32_t i = 0; i < IMAGE_COUNT; i++) {

        std::vector<uint8_t> raw_data( image_size );

        input.read( reinterpret_cast<char*>(raw_data.data()), image_size);

        Tensor<double> image( { ROWS, COLS } );
        for (uint32_t row = 0; row < ROWS; row++) {
            for (uint32_t col = 0; col < COLS; col++) {

                int32_t const index = row * COLS + col;
                image( { row, col } ) = raw_data[index] / 255.0;

            }
        }

        images.push_back( std::move( image ) );

    }

    input.close();

    return IMAGE_COUNT;

}

/**
 * @author Junzhe Wang
 * @since 15.12.2024
 *
 * @brief Displays the Tensor<double>
 *
 * @param {tensor} The tensor to be displayed.
 *
 * @return None
 *
 * */
void display_2d_tensor(Tensor<double> const& tensor) {

    auto const& shape = tensor.shape();
    bool const check = (shape.size() == 2) && (shape[0] == 28) && (shape[1] == 28);
    if ( !check ) {

        std::cerr << "Error: Tensor does NOT possess the expected 2D shape {28, 28}!";

        exit( EXIT_FAILURE );

    }

    for (uint32_t row = 0; row < shape[0]; row++) {
        for (uint32_t col = 0; col < shape[1]; col++) {

            std::cout << tensor({ row, col }) << " ";

        }

        std::cout << "\n";

    }

}

/**
 * @author Junzhe Wang
 * @since 15.12.2024
 *
 * @brief Entry point for the program {read_dataset_images.cpp}
 *
 * @param {argc} cmd-line argument count.
 * @param {argv} cmd-line argument values.
 *
 * @return The status code signifies the program return status.
 *
 * */
int main(int argc, const char * argv[]) {

    if ( argc != 2 ) {

        std::cerr 
            << "Usage:" 
            << SPACE
            << "./" << argv[0] 
            << SPACE
            << "<image-file>" << std::endl;

        return 1;

    }

    std::string const image_file_name = argv[1];

    std::vector<Tensor<double>> images;

    uint32_t const IMAGE_COUNT = image_rd( image_file_name , images );

    for (uint32_t i = 0; i < IMAGE_COUNT; i++) {

        display_2d_tensor( images[i] );
        std::cout << "\n\n";

    }

    return 0;

}

