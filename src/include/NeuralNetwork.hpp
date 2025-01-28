#pragma once

#include "BaseLayer.hpp"
#include "FullyConnected.hpp"
#include "Initializers.hpp"
#include "Loss.hpp"
#include "Optimizers.hpp"
#include "ReLU.hpp"
#include "SoftMax.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

/**
 * @author Hamiz Ali
 * @since 24.01.2025
 *
 * @brief Neural Network class
 */
class NeuralNetwork {
  private:
    FullyConnected *fc1;
    ReLU *relu;
    FullyConnected *fc2;
    SoftMax *softmax;
    CrossEntropyLoss *loss;
    // Optimizer *optimizer;
    Optimizer *optimizer1;
    Optimizer *optimizer2;
    Initializer *weights_initializer;
    Initializer *bias_initializer;

    unsigned int input_size;
    unsigned int hidden_size;
    unsigned int output_size;
    double learning_rate;

  public:
    /**
     * @author Hamiz Ali, Lam Tran
     * @since 29.01.2025
     *
     * @brief Construct a new Neural Network object
     *
     * @param input_size
     * @param hidden_size
     * @param output_size
     * @param learning_rate
     */
    NeuralNetwork(unsigned int input_size, unsigned int hidden_size, unsigned int output_size, double learning_rate)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
        // Initialize optimizer and initializers
        int seed = 123;
        // optimizer = new SGD(learning_rate);
        optimizer1 = new ADAM(learning_rate, 0.9, 0.999, 1e-8);
        // optimizer1 = new SGD(learning_rate);
        optimizer2 = new ADAM(learning_rate, 0.9, 0.999, 1e-8);
        // optimizer2 = new SGD(learning_rate);
        weights_initializer = new Xavier(seed);
        bias_initializer = new Xavier(seed);

        // Initialize layers
        fc1 = new FullyConnected(input_size, hidden_size, optimizer1);
        relu = new ReLU();
        fc2 = new FullyConnected(hidden_size, output_size, optimizer2);
        softmax = new SoftMax();
        loss = new CrossEntropyLoss();

        // Initialize weights and biases for layers
        // fc1->initialize(weights_initializer, weights_initializer);
        fc1->initialize(weights_initializer, bias_initializer);
        // fc2->initialize(weights_initializer, weights_initializer);
        fc2->initialize(weights_initializer, bias_initializer);
    }

    /**
     * @author Hamiz Ali
     * @since 24.01.2025
     *
     * @brief Forward pass through the neural network
     *
     * @param input_tensor
     * @return Tensor
     */
    Tensor forward(const Tensor &input_tensor) {
        Tensor output = fc1->forward(input_tensor);
        output = relu->forward(output);
        output = fc2->forward(output);
        return softmax->forward(output);
    }

    /**
     * @author Hamiz Ali
     * @since 24.01.2025
     *
     * @brief Training function for the neural network with forward and backward pass and loss computation
     *
     * @param input_tensor
     * @param label_tensor
     * @return double
     */
    double train(const Tensor &input_tensor, const Tensor &label_tensor) {
        // Forward pass
        Tensor predictions = forward(input_tensor);
        // Compute loss
        double loss_value = loss->computed_loss(predictions, label_tensor);
        // Backward pass
        Tensor error_tensor = loss->backward(label_tensor);
        error_tensor = softmax->backward(error_tensor);
        error_tensor = fc2->backward(error_tensor);
        error_tensor = relu->backward(error_tensor);
        fc1->backward(error_tensor);

        return loss_value;
    }

    /**
     * @author Hamiz Ali
     * @since 24.01.2025
     *
     * @brief Evaluate the model on the test set after training
     * @brief and write the results to a log file passed in the last argument
     *
     * @param test_images
     * @param test_labels
     * @param batch_size
     * @param log_file
     */
    double evaluate(const Tensor &test_images, const Tensor &test_labels, unsigned int batch_size,
                    const std::string &log_file) {
        std::ofstream log_stream(log_file);
        unsigned int correct_count = 0;
        for (int batch_start = 0; batch_start < test_images.rows(); batch_start += batch_size)

        {
            log_stream << "Current batch: " << batch_start / batch_size << std::endl;

            Tensor batch_images = test_images.middleRows(
                batch_start, std::min(batch_size, (unsigned int)test_images.rows() - batch_start));
            Tensor batch_labels = test_labels.middleRows(
                batch_start, std::min(batch_size, (unsigned int)test_labels.rows() - batch_start));
            Tensor predictions = forward(batch_images);

            for (int i = 0; i < predictions.rows(); i++) {
                int predicted_label = 0, actual_label = 0;
                predictions.row(i).maxCoeff(&predicted_label); // Get predicted class
                batch_labels.row(i).maxCoeff(&actual_label);   // Get actual class

                log_stream << " - image " << (batch_start + i) << ": Prediction=" << predicted_label
                           << ". Label=" << actual_label << std::endl;

                if (predicted_label == actual_label)
                    correct_count++;
            }
        }
        log_stream.close();
        return (static_cast<double>(correct_count) / test_images.rows()) * 100.0;
    }

    /**
     * @author Hamiz Ali
     * @since 24.01.2025
     *
     * @brief Train the model on the training set for a number of epochs
     * @brief and batch size
     *
     * @param train_images
     * @param train_labels
     * @param num_epochs
     * @param batch_size
     */
    void fit(const Tensor &train_images, const Tensor &train_labels, unsigned int num_epochs, unsigned int batch_size) {
        for (unsigned int epoch = 0; epoch < num_epochs; ++epoch) {
            int batch_num = 1;
            double batch_loss = 0.0;
            for (int i = 0; i < train_images.rows(); i += batch_size) {
                Tensor batch_images =
                    train_images.middleRows(i, std::min(batch_size, (unsigned int)train_images.rows() - i));
                Tensor batch_labels =
                    train_labels.middleRows(i, std::min(batch_size, (unsigned int)train_labels.rows() - i));
                batch_loss = (train(batch_images, batch_labels) / batch_images.rows());

                std::cout << "Current batch: " << batch_num << " " << "Batch Loss: " << batch_loss << std::endl;
                batch_num++;
            }

            std::cout << "Epoch: " << epoch << " done." << std::endl;
        }
    }

    ~NeuralNetwork() {
        delete fc1;
        delete relu;
        delete fc2;
        delete softmax;
        delete loss;
        // delete optimizer;
        delete optimizer1;
        delete optimizer2;
        delete weights_initializer;
        delete bias_initializer;
    }
};