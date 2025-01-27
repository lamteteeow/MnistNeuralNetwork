#pragma once

#include "Eigen/Dense"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Tensor = Eigen::MatrixXd;

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief A class to load MNIST dataset in Eigen::MatrixXd format.
 */

class EigenDataSetLoader
{
private:
  std::ifstream file;

  int32_t read_big_endian_int();
  std::vector<unsigned char> read_bytes(std::size_t size);
  void validate_file_open() const;
  Tensor normalize_image_data(const std::vector<unsigned char> &data, int rows, int cols) const;
  Tensor one_hot_encode_labels(const std::vector<unsigned char> &data, int numLabels) const;

public:
  explicit EigenDataSetLoader(const std::string &filename);
  ~EigenDataSetLoader();

  Tensor read_images();
  Tensor read_labels();
};

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Constructor for EigenDataSetLoader
 *
 * @param filename The name of the file to open
 *
 * @return None
 */

inline EigenDataSetLoader::EigenDataSetLoader(const std::string &filename)
{
  file.open(filename, std::ios::binary);
  if (!file.is_open())
  {
    throw std::runtime_error("Error: Unable to open file: " + filename);
  }
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Destructor for EigenDataSetLoader
 *
 * @return None
 */

inline EigenDataSetLoader::~EigenDataSetLoader()
{
  if (file.is_open())
  {
    file.close();
  }
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Utility function to read a big-endian integer from the file.
 *
 * @return A new string with leading and trailing whitespaces removed.
 */

inline int32_t EigenDataSetLoader::read_big_endian_int()
{
  int32_t value = 0;
  file.read(reinterpret_cast<char *>(&value), sizeof(int32_t));
  if (file.gcount() != sizeof(int32_t))
  {
    throw std::runtime_error("Error: Failed to read integer from file.");
  }
  return __builtin_bswap32(value);
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Utility function to read a specified number of bytes from the file.
 *
 * @param size The number of bytes to read.
 *
 * @return A vector of bytes read from the file.
 */

inline std::vector<unsigned char> EigenDataSetLoader::read_bytes(std::size_t size)
{
  std::vector<unsigned char> buffer(size);
  file.read(reinterpret_cast<char *>(buffer.data()), size);
  if (file.gcount() != static_cast<std::streamsize>(size))
  {
    throw std::runtime_error("Error: Unexpected end of file while reading bytes.");
  }
  return buffer;
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Utility function to validate that the file is open.
 *
 * @return None
 */

inline void EigenDataSetLoader::validate_file_open() const
{
  if (!file.is_open())
  {
    throw std::runtime_error("Error: File is not open.");
  }
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Normalize the image data to be in the range [0, 1]
 *
 * @param data The raw image data
 * @param rows The number of rows in the image
 * @param cols The number of columns in the image
 *
 * @return The normalized image data
 */

inline Tensor EigenDataSetLoader::normalize_image_data(const std::vector<unsigned char> &data, int rows, int cols) const
{
  Tensor images(1, rows * cols);
  for (int i = 0; i < rows * cols; ++i)
  {
    images(0, i) = static_cast<double>(data[i]) / 255.0;
  }
  return images;
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief One-hot encode the labels
 *
 * @param data The raw label data
 * @param numLabels The number of unique labels
 *
 * @return The one-hot encoded labels
 */

inline Tensor EigenDataSetLoader::one_hot_encode_labels(const std::vector<unsigned char> &data, int numLabels) const
{
  Tensor labels = Tensor::Zero(data.size(), numLabels);
  for (size_t i = 0; i < data.size(); ++i)
  {
    labels(i, data[i]) = 1.0;
  }
  return labels;
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Reads images from the dataset
 *
 * @return Tensor of images
 */

inline Tensor EigenDataSetLoader::read_images()
{
  validate_file_open();

  if (read_big_endian_int() != 2051)
  {
    throw std::runtime_error("Error: Invalid file type (not a MNIST image file).");
  }

  int numImages = read_big_endian_int();
  int rows = read_big_endian_int();
  int cols = read_big_endian_int();

  Tensor images(numImages, rows * cols);

  for (int i = 0; i < numImages; ++i)
  {
    auto rawData = read_bytes(rows * cols);
    images.row(i) = normalize_image_data(rawData, rows, cols);
  }

  return images;
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Reads labels from the dataset
 *
 * @return Tensor of one-hot encoded labels
 */

inline Tensor EigenDataSetLoader::read_labels()
{
  validate_file_open();

  if (read_big_endian_int() != 2049)
  {
    throw std::runtime_error("Error: Invalid file type (not a MNIST label file).");
  }

  int numLabels = read_big_endian_int();

  auto rawData = read_bytes(numLabels);

  return one_hot_encode_labels(rawData, 10);
}