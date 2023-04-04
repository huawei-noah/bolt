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
#include <fstream>

#include "training.h"

std::string testing_data_prefix = "./testing_dataset/testing_images_";
std::string testing_data_suffix = ".bin";
std::string testing_label_prefix = "./testing_labels/testing_label_";
std::string testing_label_suffix = ".bin";

void readVector(
    std::string path, std::vector<float> &myVector, int size, bool image_normalize = false)
{
    std::ifstream FILE(path, std::ios::in | std::ifstream::binary);
    myVector.clear();
    for (int k = 0; k < size; k++) {
        unsigned char tmp;
        FILE.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
        if (image_normalize) {
            myVector.push_back(((float)tmp / 255.0));
        } else {
            myVector.push_back((float)tmp);
        }
    }
}

// generate some shuffle data
void gen_batch_data_and_labels(std::vector<float> &batch_images,
    std::vector<float> &batch_labels,
    std::vector<int> file_indexes)
{
    for (int i = 0; i < (int)(file_indexes.size()); i++) {
        std::vector<float> tmp_image;
        std::vector<float> tmp_label;
        std::string cur_image_path =
            testing_data_prefix + std::to_string(file_indexes[i]) + testing_data_suffix;
        std::string cur_label_path =
            testing_label_prefix + std::to_string(file_indexes[i]) + testing_label_suffix;
        readVector(cur_image_path, tmp_image, 1 * 3 * 84 * 84, true);
        readVector(cur_label_path, tmp_label, 1 * 20);
        for (int j = 0; j < (int)(tmp_image.size()); j++) {
            batch_images.push_back(tmp_image[j]);
        }
        for (int j = 0; j < (int)(tmp_label.size()); j++) {
            batch_labels.push_back(tmp_label[j]);
        }
    }
}

int main()
{
    const float LEARNING_RATE = 0.1;

    std::cout << "LEARNING_RATE: " << LEARNING_RATE << std::endl;
    const char *loss_type = "OpCrossEntropyLoss";
    // First step: to load the model
    Graph_t *graph = NULL;
    const char *modelPath = "./mobilenet_v1_train.bolt";
    int batch_size = 20;
    const int target_size = 20;
    std::vector<size_t> input_size = {1, 3, 84, 84};
    char *modified_output = (char *)"fc7";
    create_graph_from_bolt(modelPath, &graph, loss_type, batch_size,
        input_size.data(), input_size.size(), modified_output, target_size);

    // Second step: create optimizer
    Optimizer_t *optimizer = NULL;
    create_adam_optimizer(&optimizer, 0.0001, 0.9, 0.999, 10e-8);

    // Add the interface for feeding the input data
    int iter_times = 50;
    int gap_size = 600;
    std::vector<std::vector<int>> file_indexes;
    for (int z = 0; z < iter_times; z++) {
        std::vector<int> tmp_vec;
        for (int i = 0; i < batch_size / target_size; i++) {
            for (int j = 0; j < target_size; j++) {
                tmp_vec.push_back(z * (batch_size / target_size) + i + j * gap_size);
            }
        }
        file_indexes.push_back(tmp_vec);
    }

    // Third step: training single step + metric
    const char *loss_name = "loss_layer_output_tensor_name";
    float testLoss = 0;
    for (int i = 0; i < iter_times; i++) {    // iter_times --> 1
        std::vector<float> images_ptr;
        std::vector<float> labels_ptr;
        gen_batch_data_and_labels(images_ptr, labels_ptr, file_indexes[i]);
        // set input
        set_tensor(graph, "data", &images_ptr[0], (batch_size * 3 * 84 * 84));
        // set labels
        set_tensor(graph, "targets", &labels_ptr[0], (batch_size * target_size * 1 * 1));

        train_single_pass(graph, optimizer, &loss_name, 1, &testLoss);
        std::cout << "step: " << i << ", loss: " << testLoss << "\n\n";
    }

    // Fourth step: serialize the updated model
    save_graph(graph, modelPath, "./mobilenet_v1_finetune.bolt");
    delete_optimizer(optimizer);
    delete_graph(graph);
    return 0;
}
