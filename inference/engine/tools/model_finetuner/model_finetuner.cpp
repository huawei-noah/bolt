// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "model_tools.h"
#include "model_serialize_deserialize.h"
#include "model_optimizer.hpp"
#include "converter.h"
#include "model_print.h"
#include "image.h"
#include "training.h"
#include "profiling.h"

#define BATCH_SIZE 16
#define FILE_LEN 30730000
#define LEARNING_RATE 0.05

U8 *mmap_cifar10_dataset(std::string path)
{
    int fd = open(path.c_str(), O_RDONLY);
    CHECK_REQUIREMENT(-1 != fd);

    struct stat ss;
    CHECK_REQUIREMENT(fstat(fd, &ss) != -1);

    int fileLength = ss.st_size;
    CHECK_REQUIREMENT(fileLength == FILE_LEN);
    U8 *bytes = (U8 *)mmap(nullptr, fileLength, PROT_READ, MAP_SHARED, fd, 0);
    CHECK_REQUIREMENT(MAP_FAILED != bytes);
    close(fd);

    return bytes;
}

template <bool training, typename T>
void load_cifar10(U8 *dataset, U32 batchIdx, TensorDesc inDesc, float *pixels, T *labels)
{
    Tensor input = Tensor::alloc_sized<CPUMem>(tensor4df(DT_F32, DF_NCHW, 1, 3, 32, 32));
    F32 *inputPtr = (F32 *)((CpuMemory *)input.get_memory())->get_ptr();
    U32 byteCount = batchIdx * 3073;
    F32 factor = 1.0 / 255;
    ArchInfo archInfo;
    archInfo.arch = CPU_GENERAL;

    if (training) {
        memset(labels, 0, BATCH_SIZE * 10 * sizeof(float));
    }
    Tensor tmp, output;
    output.resize(inDesc);
    for (U32 i = 0; i < BATCH_SIZE; i++) {
        U32 label = dataset[byteCount];
        byteCount++;
        if (training) {
            labels[i * 10 + label] = 1.0;
        } else {
            labels[i] = label;
        }
        for (U32 j = 0; j < 3072; j++) {
            inputPtr[j] = factor * dataset[byteCount];
            byteCount++;
        }

        U8 *outputPtr = (U8 *)(pixels + i * tensorNumElements(inDesc));
        ((CpuMemory *)(output.get_memory()))
            ->set_shared_ptr(std::shared_ptr<U8>(outputPtr, [](U8 *ptr) {}));
        resize(input, tmp, output, &archInfo);
    }
}

int main(int argc, char *argv[])
{
    CHECK_REQUIREMENT(10000 % BATCH_SIZE == 0);
    // TODO: finalize command-line inputs
    CHECK_REQUIREMENT(argc >= 3);
    std::string dir = argv[1];
    std::string mfn = argv[2];

    int removePreprocessOpNum = 0;
    if (argc > 3) {
        removePreprocessOpNum = atoi(argv[3]);
    }

    ModelSpec ms;
    ModelSpec resultMs;
    CHECK_STATUS(mt_create_model(&ms));
    CHECK_STATUS(mt_create_model(&resultMs));
    CHECK_STATUS(onnx_converter(dir, mfn, removePreprocessOpNum, &ms));

    ModelSpecOptimizer msOptimizer;
    msOptimizer.suggest_for_training();
    msOptimizer.optimize(&ms);
    print_ms(ms);

    Graph_Description_t *desc = NULL;
    ASSERT_OK(create_graph_description(&desc));
    CHECK_REQUIREMENT(1 == ms.num_inputs);
    const char *data_layer_outputs[2];
    data_layer_outputs[0] = const_cast<const char *>(ms.input_names[0]);
    data_layer_outputs[1] = "labels";
    ASSERT_OK(
        add_data_layer_with_labels(desc, "data", data_layer_outputs, 2, ms.input_dims[0].dims[2],
            ms.input_dims[0].dims[1], ms.input_dims[0].dims[0], 10));  // 10 for Cifar10

    add_layers(&ms, desc);

    const char *lossInputs[2];
    lossInputs[0] = const_cast<const char *>(ms.output_names[0]);
    lossInputs[1] = "labels";
    const char *loss_name = "loss";
    ASSERT_OK(add_loss_layer(desc, "loss", lossInputs, loss_name, NLL_LOSS, 2, LOSS_REDUCTION_MEAN));

    Graph_t *graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));

    CHECK_REQUIREMENT(nullptr != graph);

    exchange_weight_with_ms(&ms, graph, FROM_MS);
    print_graph(graph);

    // Load CIFAR10 testing set
    U8 *testset = mmap_cifar10_dataset(dir + "/test_batch.bin");

    TensorDesc inDesc = ms.input_dims[0];
    inDesc.dims[3] = 1;
    size_t testLabels[BATCH_SIZE];
    F32 *pixels = (F32 *)malloc(BATCH_SIZE * tensorNumElements(inDesc) * sizeof(F32));

    F32 accuracy = 0;
    // Test one batch
    printf("Begin testing\n");
    for (int i = 0; i < 11; i++) {
        load_cifar10<false, size_t>(testset, i, inDesc, pixels, testLabels);
        ASSERT_OK(
            set_tensor(graph, ms.input_names[0], pixels, BATCH_SIZE * tensorNumElements(inDesc)));
        ASSERT_OK(test_network(graph, ms.output_names[0], testLabels, &accuracy));
        printf("Batch %d Test accuracy = %.2f\n", i, accuracy * 100);
    }
    // Finetune one epoch
    printf("Begin finetuning\n");
    Optimizer_t *sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    F32 trainLabels[BATCH_SIZE * 10];  // one-hot
    F32 averageLoss = 0;
    double epochStart = ut_time_ms();
    for (int i = 1; i <= 5; i++) {
        U8 *train = mmap_cifar10_dataset(dir + "/data_batch_" + std::to_string(i) + ".bin");
        for (int j = 0; j < 10000 / BATCH_SIZE; j++) {
            double start = ut_time_ms();
            load_cifar10<true, F32>(train, j, inDesc, pixels, trainLabels);
            double loadEnd = ut_time_ms();
            ASSERT_OK(set_tensor(
                graph, ms.input_names[0], pixels, BATCH_SIZE * tensorNumElements(inDesc)));
            ASSERT_OK(set_tensor(graph, "labels", trainLabels, BATCH_SIZE * 10));
            double setEnd = ut_time_ms();

            F32 testLoss;

            ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1, &testLoss));
            double end = ut_time_ms();
            averageLoss += testLoss;
            if (j % 5 == 0) {
                printf("iteration = %d of set %d, loss = %f\n", j, i, testLoss);
                printf("Load time: %f, Set time: %f\n", loadEnd - start, setEnd - loadEnd);
                printf("Batch training time: %f\n", end - setEnd);
                for (int k = 0; k < 9; k++) {
                    load_cifar10<false, size_t>(testset, k, inDesc, pixels, testLabels);
                    ASSERT_OK(set_tensor(
                        graph, ms.input_names[0], pixels, BATCH_SIZE * tensorNumElements(inDesc)));
                    ASSERT_OK(test_network(graph, ms.output_names[0], testLabels, &accuracy));
                    printf("Batch %d Test accuracy = %.2f\n", k, accuracy * 100);
                }
            }
        }
        munmap(train, FILE_LEN);
    }
    double epochEnd = ut_time_ms();
    printf("Average loss = %f\n", averageLoss * BATCH_SIZE / 50000);
    printf("Epoch Training taken = %.3fs \n", epochEnd - epochStart);

    // Check finetuned accuracy
    printf("Begin checking\n");
    for (int i = 0; i < 11; i++) {
        load_cifar10<false, size_t>(testset, i, inDesc, pixels, testLabels);
        ASSERT_OK(
            set_tensor(graph, ms.input_names[0], pixels, BATCH_SIZE * tensorNumElements(inDesc)));
        ASSERT_OK(test_network(graph, ms.output_names[0], testLabels, &accuracy));
        printf("Batch %d Test accuracy = %.2f\n", i, accuracy * 100);
    }

    // Get updated weights after finetuning
    exchange_weight_with_ms(&ms, graph, TO_MS);

    munmap(testset, FILE_LEN);
    free(pixels);
    ASSERT_OK(delete_graph(graph));

    // serialize ms to .bolt
    std::string modelStorePath =
        std::string(argv[1]) + "/" + std::string(argv[2]) + std::string("_f32.bolt");
    CHECK_STATUS(serialize_model_to_file(&ms, modelStorePath.c_str()));

    // deserialize .bolt to ms in memory
    CHECK_STATUS(deserialize_model_from_file(modelStorePath.c_str(), &resultMs));
    print_ms(resultMs);

    CHECK_STATUS(mt_destroy_model(&ms));
    CHECK_STATUS(mt_destroy_model(&resultMs));

    return 0;
}
