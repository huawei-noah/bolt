// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

int main()
{
    const char *loss_type = "OpSoftmaxCrossEntropyLoss";
    const float LEARNING_RATE = 0.05;
    // First step: to load the model
    Graph_t *graph = NULL;
    const char *modelPath = "./resnet18_v2_sim_train.bolt";
    int batch_size = 2;
    const int target_size = 1000;
    std::cout << "LEARNING_RATE: " << LEARNING_RATE << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    fflush(stdout);
    create_graph_from_bolt(modelPath, &graph, loss_type, batch_size, nullptr, 0, nullptr, target_size);

    // Second step: create optimizer
    Optimizer_t *sgd_optimizer = NULL;
    create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE);

    // Feeding model with fake data(full 1)
    const int fake_imgs_size = 100;
    std::vector<float> images_ptr(fake_imgs_size * (1 * 3 * 224 * 224), 1.0);
    std::vector<float> labels_ptr(fake_imgs_size * (1 * target_size));
    for (int i = 0; i < fake_imgs_size; i++) {
        std::vector<float> tmp_vec(target_size, 0.0);
        tmp_vec[i] = 1.0;
        memcpy(&(labels_ptr[i * target_size]), &(tmp_vec[0]), sizeof(float) * target_size);
    }

    // Third step: training single step + metric
    const char *loss_name = "loss_layer_output_tensor_name";
    float testLoss = 0;
    for (int i = 0; i < (fake_imgs_size / batch_size); i++) {
        // set input
        set_tensor(graph, "data", &images_ptr[i * batch_size * 3 * 224 * 224],
            (batch_size * 3 * 224 * 224));
        // set labels
        set_tensor(graph, "targets", &labels_ptr[i * batch_size * target_size * 1 * 1],
            (batch_size * target_size * 1 * 1));

        train_single_pass(graph, sgd_optimizer, &loss_name, 1, &testLoss);
        std::cout << "step: " << i << ", loss: " << testLoss << "\n\n";
    }

    // Fourth step: serialize the updated model
    save_graph(graph, modelPath, "./resnet18_v2_sim_finetune.bolt");
    delete_optimizer(sgd_optimizer);
    delete_graph(graph);
    return 0;
}
