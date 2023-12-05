#pragma once

#include <iomanip>
#include <fstream>
#include "misc/typedefs.h"

class Config {

public:
    explicit Config(const std::string &rel_config_path) {
        std::ifstream ifs(rel_config_path);
        if (!ifs.good())
            throw std::runtime_error("Cannot open config file");

        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string key;

            // parse
            if (std::getline(iss, key, '=')) {
                std::string value;
                if (std::getline(iss, value)) {
                    // remove whitespaces
                    key.erase(remove_if(key.begin(), key.end(), isspace), key.end());
                    value.erase(remove_if(value.begin(), value.end(), isspace), value.end());

                    edit_config(key, value);
                }
            }
        }

        if (!ifs.eof())
            throw std::runtime_error("Error while reading config file");
    }

    [[nodiscard]] const std::string &getRelPathTrainImages() const { return rel_path_train_images_; }

    [[nodiscard]] const std::string &getRelPathTrainLabels() const { return rel_path_train_labels_; }

    [[nodiscard]] const std::string &getRelPathTestImages() const { return rel_path_test_images_; }

    [[nodiscard]] const std::string &getRelPathTestLabels() const { return rel_path_test_labels_; }

    [[nodiscard]] const std::string &getRelPathLogFile() const { return rel_path_log_file_; }

    [[nodiscard]] size_t getNumEpochs() const { return num_epochs_; }

    [[nodiscard]] size_t getBatchSize() const { return batch_size_; }

    [[nodiscard]] size_t getHiddenSize() const { return hidden_size_; }

    [[nodiscard]] real_t getLearningRate() const { return learning_rate_; }

private:

    void edit_config(const std::string &key, const std::string &value) {
        if (key == "rel_path_train_images") {
            rel_path_train_images_ = value;
        } else if (key == "rel_path_train_labels") {
            rel_path_train_labels_ = value;
        } else if (key == "rel_path_test_images") {
            rel_path_test_images_ = value;
        } else if (key == "rel_path_test_labels") {
            rel_path_test_labels_ = value;
        } else if (key == "rel_path_log_file") {
            rel_path_log_file_ = value;
        } else if (key == "num_epochs") {
            num_epochs_ = std::stoi(value);
        } else if (key == "batch_size") {
            batch_size_ = std::stoi(value);
        } else if (key == "hidden_size") {
            hidden_size_ = std::stoi(value);
        } else if (key == "learning_rate") {
            learning_rate_ = (real_t) std::stod(value);
        } else {
            throw std::runtime_error("Invalid key found in config file: " + key);
        }
    }

    std::string rel_path_train_images_;
    std::string rel_path_train_labels_;

    std::string rel_path_test_images_;
    std::string rel_path_test_labels_;

    std::string rel_path_log_file_;

    size_t num_epochs_{};
    size_t batch_size_{};
    size_t hidden_size_{};
    real_t learning_rate_{};
};
