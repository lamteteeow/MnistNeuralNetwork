#include "Eigen/Dense"
#include "EigenDataSetLoader.hpp"
#include "NeuralNetwork.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Tensor = Eigen::MatrixXd;

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Utility function to trim leading and trailing whitespace characters
 * from a string.
 *
 * @param str The input string that needs to be trimmed.
 * @return A new string with leading and trailing whitespaces removed.
 */
std::string trim(const std::string &str)
{
    auto start = str.find_first_not_of(" \t");
    auto end = str.find_last_not_of(" \t");
    return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Reads and parses key-value pairs from a configuration file and stores
 * them in a map. The configuration file should have lines in the format
 * `key=value`. Lines that are empty or start with `#` are ignored. Duplicate
 * keys will overwrite previous values.
 *
 * @param filename The path to the configuration file.
 * @return A map containing configuration key-value pairs. If the file is
 * invalid, an empty map is returned.
 */
std::map<std::string, std::string> read_network_configurations(const std::string &filename)
{
    std::map<std::string, std::string> configs;

    std::ifstream configFile(filename);
    if (!configFile.is_open())
    {
        std::cerr << "Error: Unable to open the config file: " << filename << std::endl;
        return configs;
    }

    std::string line;
    while (std::getline(configFile, line))
    {
        line = trim(line);

        if (line.empty() || line[0] == '#')
            continue;

        std::size_t equalPos = line.find('=');
        if (equalPos == std::string::npos)
        {
            std::cerr << "Warning: Skipping invalid line in config file: " << line << std::endl;
            continue;
        }

        std::string key = trim(line.substr(0, equalPos));
        std::string value = trim(line.substr(equalPos + 1));

        if (configs.find(key) != configs.end())
        {
            std::cerr << "Warning: Duplicate key found, overwriting previous value: " << key << std::endl;
        }

        configs[key] = value;
    }

    if (configs.empty())
    {
        std::cerr << "Warning: No valid configurations found in file: " << filename << std::endl;
    }

    return configs;
}

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Entry point for the program {neural_network.cpp}
 *
 * @param {argc} cmd-line argument count.
 * @param {argv} cmd-line argument values.
 *
 * @return The status code signifies the program return status.
 */
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    // Read configuration file
    std::string config_file(argv[1]);
    std::map<std::string, std::string> configs = read_network_configurations(config_file);
    if (!configs.size())
    {
        std::cerr << "Error: No valid configurations found in file: " << config_file << std::endl;
        return 1;
    }
    // configurations for the neural network
    unsigned int batch_size = static_cast<unsigned int>(std::stoul(configs["batch_size"]));
    unsigned int hidden_size = static_cast<unsigned int>(std::stoi(configs["hidden_size"]));
    double learning_rate = std::stod(configs["learning_rate"]);
    unsigned int num_epochs = static_cast<unsigned int>(std::stoi(configs["num_epochs"]));

    // configurations for the dataset
    std::string rel_path_train_images = configs["rel_path_train_images"];
    std::string rel_path_train_labels = configs["rel_path_train_labels"];
    std::string rel_path_test_images = configs["rel_path_test_images"];
    std::string rel_path_test_labels = configs["rel_path_test_labels"];

    // path to the log file
    std::string rel_path_log_file = configs["rel_path_log_file"];

    // Load MNIST dataset using EigenDataSetLoader
    EigenDataSetLoader read_training_images(rel_path_train_images);
    EigenDataSetLoader read_training_labels(rel_path_train_labels);
    EigenDataSetLoader read_test_images(rel_path_test_images);
    EigenDataSetLoader read_test_labels(rel_path_test_labels);

    Tensor train_images = read_training_images.read_images();
    Tensor train_labels = read_training_labels.read_labels();
    Tensor test_images = read_test_images.read_images();
    Tensor test_labels = read_test_labels.read_labels();

    std::cout << "Training images: " << train_images.rows() << ", Training labels: " << train_labels.rows() << std::endl;

    // Create and train the neural network
    NeuralNetwork nn(784, hidden_size, 10, learning_rate);

    std::cout << "Training the neural network..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    nn.fit(train_images, train_labels, num_epochs, batch_size);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Training completed " << std::endl;
    std::cout << "Training time: " << duration.count() << " seconds" << std::endl;

    double accuracy = nn.evaluate(test_images, test_labels, batch_size, rel_path_log_file);

    std::cout << "Testing completed: " << accuracy << "% >> Log File: " << rel_path_log_file << std::endl;

    return 0;
}
