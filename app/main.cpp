#include <iostream>
#include <cmath>
#include <filesystem>
#include <cfenv>

#include "mnist/config_parser.h"
#include "mnist/read_dataset.h"
#include "mnist/convert_dataset.h"
#include "mnist/flatten_pixel_tensors.h"
#include "mnist/evaluate_predictions.h"
#include "network/linear_layer.h"
#include "network/relu_layer.h"
#include "network/cross_entropy_loss_layer.h"
#include "network/softmax.h"
#include "network/linear_layer_optimized.h"

int main(int argc, char *argv[]) {

    /* PARSE COMMANDLINE ARGS */

    if (argc != 2) {
        std::cout
                << "Invalid arguments. Program must be run as: "
                   "./mnist <relative config file path (string)>"
                << std::endl;
        exit(EXIT_FAILURE);
    }

    Config config(argv[1]);

    auto num_epochs = config.getNumEpochs();
    auto batch_size = config.getBatchSize();
    auto hidden_size = config.getHiddenSize();
    auto learning_rate = config.getLearningRate();

    const auto &rel_train_image_path = config.getRelPathTrainImages();
    const auto &rel_train_label_path = config.getRelPathTrainLabels();

    if (rel_train_image_path.empty() || rel_train_label_path.empty()) {
        std::cout << "Empty path(s) to training dataset. Aborting..." << std::endl;
        return EXIT_FAILURE;
    }

    const auto &rel_test_image_path = config.getRelPathTestImages();
    const auto &rel_test_label_path = config.getRelPathTestLabels();

    if (rel_test_image_path.empty() || rel_test_label_path.empty()) {
        std::cout << "Empty path(s) to testing dataset. Aborting..." << std::endl;
        return EXIT_FAILURE;
    }

    const auto &rel_log_file_path = config.getRelPathLogFile();

    if (rel_log_file_path.empty()) {
        std::cout << "Empty log file path. Aborting..." << std::endl;
        return EXIT_FAILURE;
    }

#ifndef NDEBUG
    // check for floating point exceptions
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
#endif

    /* READ DATASET FOR TRAINING */

    size_t num_images_train, num_labels_train, image_size_train;
    auto images_train = read_mnist_images(rel_train_image_path, batch_size, num_images_train,
                                          image_size_train);
    auto labels_train = read_mnist_labels(rel_train_label_path, batch_size, num_labels_train);

    size_t num_images_test, num_labels_test, image_size_test;
    auto images_test = read_mnist_images(rel_test_image_path, batch_size, num_images_test,
                                         image_size_test);
    auto labels_test = read_mnist_labels(rel_test_label_path, batch_size, num_labels_test);

    assert(image_size_test == image_size_train);

    size_t image_size = image_size_train;
    int ncols = static_cast<int>(std::sqrt(image_size));
    int nrows = ncols;

    size_t labels_size = 10;

    // compute number of batches
    auto num_batches_train = num_images_train / batch_size;
    auto num_batches_test = num_images_test / batch_size;

    /* PRINT SIMULATION PARAMETERS */

    std::cout << "Network hyper-parameters: " << std::endl;
    std::cout << " - number of epochs: " << num_epochs << std::endl;
    std::cout << " - number of hidden layers: " << hidden_size << std::endl;
    std::cout << " - batch size: " << batch_size << std::endl;
    std::cout << " - learning rate: " << learning_rate << std::endl;

    std::cout << "General MNIST dataset infos: " << std::endl;
    std::cout << " - image size: " << image_size << std::endl;
    std::cout << " - image columns: " << ncols << std::endl;
    std::cout << " - image rows: " << nrows << std::endl;

    std::cout << "MNIST train dataset infos: " << std::endl;
    std::cout << " - num images (train): " << num_images_train << std::endl;
    std::cout << " - num labels (train): " << num_labels_train << std::endl;
    std::cout << " - num batches (train): " << num_batches_train << std::endl;

    std::cout << "MNIST test dataset infos: " << std::endl;
    std::cout << " - num images (test): " << num_images_test << std::endl;
    std::cout << " - num labels (test): " << num_labels_test << std::endl;
    std::cout << " - num batches (test): " << num_batches_test << std::endl;

    // convert images from uchar tensors to float tensors
    auto converted_images_train = encode_mnist_images(images_train, num_batches_train, batch_size, nrows, ncols);
    auto converted_images_test = encode_mnist_images(images_test, num_batches_test, batch_size, nrows, ncols);

    // one-hot-encoding of label data
    auto converted_labels_train = encode_mnist_labels(labels_train, num_batches_train, batch_size);
    auto converted_labels_test = encode_mnist_labels(labels_test, num_batches_test, batch_size);

    // linearize pixels to neurons
    auto flattened_image_tensors_train = flatten_pixel_tensors<real_t>(converted_images_train, num_batches_train,
                                                                       batch_size,
                                                                       nrows, ncols);
    auto flattened_image_tensors_test = flatten_pixel_tensors<real_t>(converted_images_test, num_batches_test,
                                                                      batch_size,
                                                                      nrows, ncols);

    // set up model
    LinearLayerOptimized linear_layer1(batch_size, hidden_size, image_size, learning_rate);
    ReLULayer relu_layer(batch_size, hidden_size);
    LinearLayerOptimized linear_layer2(batch_size, labels_size, hidden_size, learning_rate);
    SoftMaxLayer softmax_layer(batch_size, labels_size);

    CrossEntropyLossLayer loss_layer(batch_size, labels_size);

    /* training */
    for (size_t epoch = 0; epoch < (size_t) num_epochs; ++epoch) {
        for (size_t cur_batch = 0; cur_batch < num_batches_train; ++cur_batch) {
            auto image_tensor_per_batch = flattened_image_tensors_train[cur_batch];
            auto label_tensor_per_batch = converted_labels_train[cur_batch];

            // forward
            auto tensor1 = linear_layer1.forward(image_tensor_per_batch);
            auto tensor2 = relu_layer.forward(tensor1);
            auto tensor3 = linear_layer2.forward(tensor2);
            auto res = softmax_layer.forward(tensor3);

            // compute loss
            auto loss = loss_layer.forward(res, label_tensor_per_batch);

            // backward
            auto loss_err = loss_layer.backward(label_tensor_per_batch);
            auto err = softmax_layer.backward(loss_err);
            auto err2 = linear_layer2.backward(err);
            auto err3 = relu_layer.backward(err2);
            auto err4 = linear_layer1.backward(err3);

            // log loss
            if (cur_batch % 100 == 0) {
                std::cout << "Loss after batch " << cur_batch << " and epoch " << epoch << " = " << loss << std::endl;
            }
        }
    }

    /* testing */

    // open log file for evaluation
    std::ofstream outstream(rel_log_file_path);

    size_t correct = 0;
    for (size_t cur_batch = 0; cur_batch < num_batches_test; ++cur_batch) {
        auto image_tensor_per_batch = flattened_image_tensors_test[cur_batch];
        auto label_tensor_per_batch = converted_labels_test[cur_batch];

        // forward
        auto tensor1 = linear_layer1.forward(image_tensor_per_batch);
        auto tensor2 = relu_layer.forward(tensor1);
        auto tensor3 = linear_layer2.forward(tensor2);
        auto res = softmax_layer.forward(tensor3);

        // count correct predictions
        correct += count_correct_predictions(res, label_tensor_per_batch, batch_size, labels_size);

        // output predictions to log file
        log_predictions(res, label_tensor_per_batch, outstream, cur_batch, batch_size, labels_size);
    }

    // output final accuracy for test set
    std::cout << "Accuracy: " << ((real_t) correct / (real_t) num_labels_test) * 100. << "%" << std::endl;

    // close log file
    outstream.close();

    return EXIT_SUCCESS;
}