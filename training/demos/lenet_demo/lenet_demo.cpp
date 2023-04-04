// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <malloc.h>
#include <cstring>
#include <vector>

#include "training.h"
#include "mnist_parser.hpp"

int main(int argc, char *argv[])
{
    const char *loss_type = "OpSoftmaxCrossEntropyLoss";
    float LEARNING_RATE = 0.001;
    if (argc > 1) {
        LEARNING_RATE = std::stof(std::string(argv[1]));
    }
    int batch_size = 512;
    if (argc > 2) {
        batch_size = std::stoi(std::string(argv[2]));
    }
    std::cout << "LEARNING_RATE: " << LEARNING_RATE << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    // data preparation
    std::vector<float> labels;
    read_Mnist_Label("train-labels-idx1-ubyte", labels);
    std::vector<std::vector<float>> one_hot_labels = one_hot_encoding<float, float>(labels, 10);
    std::vector<std::vector<float>> images;
    read_Mnist_Images("train-images-idx3-ubyte", images);
    // flatten labels
    int one_hot_labels_row = one_hot_labels.size();
    int one_hot_labels_column = one_hot_labels[0].size();
    float *labels_ptr = (float *)malloc(one_hot_labels_row * one_hot_labels_column * sizeof(float));
    for (int i = 0; i < one_hot_labels_row; i++) {
        memcpy(&(labels_ptr[i * one_hot_labels_column]), &(one_hot_labels[i][0]),
            sizeof(float) * one_hot_labels_column);
    }
    // flatten images
    int images_row = images.size();
    int images_column = images[0].size();
    float *images_ptr = (float *)malloc(images_row * images_column * sizeof(float));
    for (int i = 0; i < images_row; i++) {
        memcpy(&(images_ptr[i * images_column]), &(images[i][0]), sizeof(float) * images_column);
    }

    // First step: to load the model
    Graph_t *graph = NULL;
    const char *modelPath = "./lenet_sim_train.bolt";
    int target_size = 10;
    create_graph_from_bolt(
        modelPath, &graph, loss_type, batch_size, nullptr, 0, nullptr, target_size);

    // Second step: create optimizer
    // Current plan: create a simple sgd optimizer
    Optimizer_t *sgd_optimizer = NULL;
    create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE);

    // Third step: training single step + metric
    const char *loss_name = "loss_layer_output_tensor_name";
    float testLoss = 0;
    int iter_times = 10000 / batch_size;
    for (int i = 0; i < iter_times; i++) {
        // set input
        set_tensor(graph, "import/Placeholder:0", &images_ptr[i * batch_size * 1 * 28 * 28],
            (batch_size * 1 * 28 * 28));
        // set labels
        set_tensor(
            graph, "targets", &labels_ptr[i * batch_size * 10 * 1 * 1], (batch_size * 10 * 1 * 1));

        train_single_pass(graph, sgd_optimizer, &loss_name, 1, &testLoss);
        std::cout << "step: " << i << ", loss: " << testLoss << "\n\n";
    }

    // Fourth step: serialize the updated model
    save_graph(graph, modelPath, "./lenet_finetune.bolt");
    delete_graph(graph);
    delete_optimizer(sgd_optimizer);
    free(labels_ptr);
    free(images_ptr);
    return 0;
}
