// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <algorithm>
#include <filesystem>
#include <tests/tools/TestTools.h>

#include <training/api/API.h>

#define ASSERT_OK(x) ASSERT_EQ(x, STATUS_OK)
#define EXPECT_OK(x) EXPECT_EQ(x, STATUS_OK)
#define ASSERT_ERROR(x) ASSERT_NE(x, STATUS_OK)

namespace
{

void load_resnet_weights(Graph_t* graph, const std::filesystem::path& weights_path, const std::string file_prefix = "init_", const bool extended = false, const bool shifted = false)
{
    using namespace std;
    std::map<raul::Name, std::pair<std::string, size_t>> pytorch_name_map{ { "input::conv1::Weights", { "conv1.weight", 9408 } },
                                                                           { "layer1::block0::conv1::Weights", { "layer1.0.conv1.weight", 36864 } },
                                                                           { "layer1::block0::conv2::Weights", { "layer1.0.conv2.weight", 36864 } },
                                                                           { "layer1::block1::conv1::Weights", { "layer1.1.conv1.weight", 36864 } },
                                                                           { "layer1::block1::conv2::Weights", { "layer1.1.conv2.weight", 36864 } },
                                                                           { "layer2::block0::conv1::Weights", { "layer2.0.conv1.weight", 73728 } },
                                                                           { "layer2::block0::conv2::Weights", { "layer2.0.conv2.weight", 147456 } },
                                                                           { "layer2::block0::downsample::conv1::Weights", { "layer2.0.downsample."s + (shifted ? "1" : "0") + ".weight", 8192 } },
                                                                           { "layer2::block1::conv1::Weights", { "layer2.1.conv1.weight", 147456 } },
                                                                           { "layer2::block1::conv2::Weights", { "layer2.1.conv2.weight", 147456 } },
                                                                           { "layer3::block0::conv1::Weights", { "layer3.0.conv1.weight", 294912 } },
                                                                           { "layer3::block0::conv2::Weights", { "layer3.0.conv2.weight", 589824 } },
                                                                           { "layer3::block0::downsample::conv1::Weights", { "layer3.0.downsample."s + (shifted ? "1" : "0") + ".weight", 32768 } },
                                                                           { "layer3::block1::conv1::Weights", { "layer3.1.conv1.weight", 589824 } },
                                                                           { "layer3::block1::conv2::Weights", { "layer3.1.conv2.weight", 589824 } },
                                                                           { "layer4::block0::conv1::Weights", { "layer4.0.conv1.weight", 1179648 } },
                                                                           { "layer4::block0::conv2::Weights", { "layer4.0.conv2.weight", 2359296 } },
                                                                           { "layer4::block0::downsample::conv1::Weights", { "layer4.0.downsample."s + (shifted ? "1" : "0") + ".weight", 131072 } },
                                                                           { "layer4::block1::conv1::Weights", { "layer4.1.conv1.weight", 2359296 } },
                                                                           { "layer4::block1::conv2::Weights", { "layer4.1.conv2.weight", 2359296 } },
                                                                           { "output::fc0::Biases", { "fc.bias", 10 } },
                                                                           { "output::fc0::Weights", { "fc.weight", 5120 } } };

    std::map<raul::Name, std::pair<std::string, size_t>> pytorch_name_map_extended{
        { "input::bn1::Biases", { "bn1.bias", 64 } },
        { "input::bn1::MeanEval", { "bn1.running_mean", 64 } },
        { "input::bn1::VarianceEval", { "bn1.running_var", 64 } },
        { "input::bn1::Weights", { "bn1.weight", 64 } },
        { "layer1::block0::bn1::Biases", { "layer1.0.bn1.bias", 64 } },
        { "layer1::block0::bn1::MeanEval", { "layer1.0.bn1.running_mean", 64 } },
        { "layer1::block0::bn1::VarianceEval", { "layer1.0.bn1.running_var", 64 } },
        { "layer1::block0::bn1::Weights", { "layer1.0.bn1.weight", 64 } },
        { "layer1::block0::bn2::Biases", { "layer1.0.bn2.bias", 64 } },
        { "layer1::block0::bn2::MeanEval", { "layer1.0.bn2.running_mean", 64 } },
        { "layer1::block0::bn2::VarianceEval", { "layer1.0.bn2.running_var", 64 } },
        { "layer1::block0::bn2::Weights", { "layer1.0.bn2.weight", 64 } },
        { "layer1::block1::bn1::Biases", { "layer1.1.bn1.bias", 64 } },
        { "layer1::block1::bn1::MeanEval", { "layer1.1.bn1.running_mean", 64 } },
        { "layer1::block1::bn1::VarianceEval", { "layer1.1.bn1.running_var", 64 } },
        { "layer1::block1::bn1::Weights", { "layer1.1.bn1.weight", 64 } },
        { "layer1::block1::bn2::Biases", { "layer1.1.bn2.bias", 64 } },
        { "layer1::block1::bn2::MeanEval", { "layer1.1.bn2.running_mean", 64 } },
        { "layer1::block1::bn2::VarianceEval", { "layer1.1.bn2.running_var", 64 } },
        { "layer1::block1::bn2::Weights", { "layer1.1.bn2.weight", 64 } },
        { "layer2::block0::bn1::Biases", { "layer2.0.bn1.bias", 128 } },
        { "layer2::block0::bn1::MeanEval", { "layer2.0.bn1.running_mean", 128 } },
        { "layer2::block0::bn1::VarianceEval", { "layer2.0.bn1.running_var", 128 } },
        { "layer2::block0::bn1::Weights", { "layer2.0.bn1.weight", 128 } },
        { "layer2::block0::bn2::Biases", { "layer2.0.bn2.bias", 128 } },
        { "layer2::block0::bn2::MeanEval", { "layer2.0.bn2.running_mean", 128 } },
        { "layer2::block0::bn2::VarianceEval", { "layer2.0.bn2.running_var", 128 } },
        { "layer2::block0::bn2::Weights", { "layer2.0.bn2.weight", 128 } },
        { "layer2::block0::downsample::bn1::Biases", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".bias", 128 } },
        { "layer2::block0::downsample::bn1::MeanEval", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".running_mean", 128 } },
        { "layer2::block0::downsample::bn1::VarianceEval", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".running_var", 128 } },
        { "layer2::block0::downsample::bn1::Weights", { "layer2.0.downsample."s + (shifted ? "2" : "1") + ".weight", 128 } },
        { "layer2::block1::bn1::Biases", { "layer2.1.bn1.bias", 128 } },
        { "layer2::block1::bn1::MeanEval", { "layer2.1.bn1.running_mean", 128 } },
        { "layer2::block1::bn1::VarianceEval", { "layer2.1.bn1.running_var", 128 } },
        { "layer2::block1::bn1::Weights", { "layer2.1.bn1.weight", 128 } },
        { "layer2::block1::bn2::Biases", { "layer2.1.bn2.bias", 128 } },
        { "layer2::block1::bn2::MeanEval", { "layer2.1.bn2.running_mean", 128 } },
        { "layer2::block1::bn2::VarianceEval", { "layer2.1.bn2.running_var", 128 } },
        { "layer2::block1::bn2::Weights", { "layer2.1.bn2.weight", 128 } },
        { "layer3::block0::bn1::Biases", { "layer3.0.bn1.bias", 256 } },
        { "layer3::block0::bn1::MeanEval", { "layer3.0.bn1.running_mean", 256 } },
        { "layer3::block0::bn1::VarianceEval", { "layer3.0.bn1.running_var", 256 } },
        { "layer3::block0::bn1::Weights", { "layer3.0.bn1.weight", 256 } },
        { "layer3::block0::bn2::Biases", { "layer3.0.bn2.bias", 256 } },
        { "layer3::block0::bn2::MeanEval", { "layer3.0.bn2.running_mean", 256 } },
        { "layer3::block0::bn2::VarianceEval", { "layer3.0.bn2.running_var", 256 } },
        { "layer3::block0::bn2::Weights", { "layer3.0.bn2.weight", 256 } },
        { "layer3::block0::downsample::bn1::Biases", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".bias", 256 } },
        { "layer3::block0::downsample::bn1::MeanEval", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".running_mean", 256 } },
        { "layer3::block0::downsample::bn1::VarianceEval", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".running_var", 256 } },
        { "layer3::block0::downsample::bn1::Weights", { "layer3.0.downsample."s + (shifted ? "2" : "1") + ".weight", 256 } },
        { "layer3::block1::bn1::Biases", { "layer3.1.bn1.bias", 256 } },
        { "layer3::block1::bn1::MeanEval", { "layer3.1.bn1.running_mean", 256 } },
        { "layer3::block1::bn1::VarianceEval", { "layer3.1.bn1.running_var", 256 } },
        { "layer3::block1::bn1::Weights", { "layer3.1.bn1.weight", 256 } },
        { "layer3::block1::bn2::Biases", { "layer3.1.bn2.bias", 256 } },
        { "layer3::block1::bn2::MeanEval", { "layer3.1.bn2.running_mean", 256 } },
        { "layer3::block1::bn2::VarianceEval", { "layer3.1.bn2.running_var", 256 } },
        { "layer3::block1::bn2::Weights", { "layer3.1.bn2.weight", 256 } },
        { "layer4::block0::bn1::Biases", { "layer4.0.bn1.bias", 512 } },
        { "layer4::block0::bn1::MeanEval", { "layer4.0.bn1.running_mean", 512 } },
        { "layer4::block0::bn1::VarianceEval", { "layer4.0.bn1.running_var", 512 } },
        { "layer4::block0::bn1::Weights", { "layer4.0.bn1.weight", 512 } },
        { "layer4::block0::bn2::Biases", { "layer4.0.bn2.bias", 512 } },
        { "layer4::block0::bn2::MeanEval", { "layer4.0.bn2.running_mean", 512 } },
        { "layer4::block0::bn2::VarianceEval", { "layer4.0.bn2.running_var", 512 } },
        { "layer4::block0::bn2::Weights", { "layer4.0.bn2.weight", 512 } },
        { "layer4::block0::downsample::bn1::Biases", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".bias", 512 } },
        { "layer4::block0::downsample::bn1::MeanEval", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".running_mean", 512 } },
        { "layer4::block0::downsample::bn1::VarianceEval", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".running_var", 512 } },
        { "layer4::block0::downsample::bn1::Weights", { "layer4.0.downsample."s + (shifted ? "2" : "1") + ".weight", 512 } },
        { "layer4::block1::bn1::Biases", { "layer4.1.bn1.bias", 512 } },
        { "layer4::block1::bn1::MeanEval", { "layer4.1.bn1.running_mean", 512 } },
        { "layer4::block1::bn1::VarianceEval", { "layer4.1.bn1.running_var", 512 } },
        { "layer4::block1::bn1::Weights", { "layer4.1.bn1.weight", 512 } },
        { "layer4::block1::bn2::Biases", { "layer4.1.bn2.bias", 512 } },
        { "layer4::block1::bn2::MeanEval", { "layer4.1.bn2.running_mean", 512 } },
        { "layer4::block1::bn2::VarianceEval", { "layer4.1.bn2.running_var", 512 } },
        { "layer4::block1::bn2::Weights", { "layer4.1.bn2.weight", 512 } }
    };

    raul::DataLoader dataLoader;
    std::cout << "Loading weights...";
    for (auto [tensor, value] : pytorch_name_map)
    {
        if (tensor == "output::fc0::Weights")
        {
            const auto [filename, size] = value;
            auto weight = dataLoader.loadData(weights_path / (file_prefix + filename + ".data"), size, 1, 1);
            raul::Tensor transposed(weight);
            raul::Common::transpose(transposed, 10U);
            ASSERT_OK(set_tensor(graph, tensor.c_str(), &transposed[0], transposed.size()));
        }
        else
        {
            const auto [filename, size] = value;
            auto weight = dataLoader.loadData(weights_path / (file_prefix + filename + ".data"), size, 1, 1);
            ASSERT_OK(set_tensor(graph, tensor.c_str(), weight.first, weight.second - weight.first));
        }
    }

    if (extended)
    {
        for (auto [tensor, value] : pytorch_name_map_extended)
        {
            const auto [filename, size] = value;
            auto weight = dataLoader.loadData(weights_path / (file_prefix + filename + ".data"), size, 1, 1);
            ASSERT_OK(set_tensor(graph, tensor.c_str(), weight.first, weight.second - weight.first));
        }
    }

    std::cout << "done" << std::endl;
}

void load_yannet_weights(Graph_t* graph, const std::filesystem::path& weights_path)
{
    using namespace std;
    std::map<raul::Name, std::pair<std::string, size_t>> pytorch_name_map{ { raul::Name("rnn") / "cell" / "linear_ih" / "Weights", { "rnn.weight_ih_l0", 3264 } },
                                                                           { raul::Name("rnn") / "cell" / "linear_hh" / "Weights", { "rnn.weight_hh_l0", 3072 } },
                                                                           { raul::Name("rnn") / "cell" / "linear_ih" / "Biases", { "rnn.bias_ih_l0", 96 } },
                                                                           { raul::Name("rnn") / "cell" / "linear_hh" / "Biases", { "rnn.bias_hh_l0", 96 } },
                                                                           { raul::Name("att") / "norm" / "Weights", { "att.norm.weight", 32 } },
                                                                           { raul::Name("att") / "norm" / "Biases", { "att.norm.bias", 32 } },
                                                                           { raul::Name("att") / "query_layer" / "Weights", { "att.query_layer.weight", 384 } },
                                                                           { raul::Name("att") / "query_layer" / "Biases", { "att.query_layer.bias", 32 } },
                                                                           { raul::Name("att") / "key_layer" / "Weights", { "att.key_layer.weight", 1024 } },
                                                                           { raul::Name("att") / "key_layer" / "Biases", { "att.key_layer.bias", 32 } },
                                                                           { raul::Name("att") / "attention_layer" / "Weights", { "att.attention_layer.weight", 32 } },
                                                                           { raul::Name("att") / "attention_layer" / "Biases", { "att.attention_layer.bias", 1 } },
                                                                           { raul::Name("fc") / "Weights", { "fc.weight", 32 } },
                                                                           { raul::Name("fc") / "Biases", { "fc.bias", 1 } } };

    raul::DataLoader dataLoader;
    std::cout << "Loading weights...";
    for (auto [tensor, value] : pytorch_name_map)
    {
        const auto [filename, size] = value;
        auto weight = dataLoader.loadData(weights_path / (filename + ".txt"), size, 1, 1);
        raul::Tensor transposed(weight);
        //raul::Common::transpose(transposed, 10U);
        ASSERT_OK(set_tensor(graph, tensor.c_str(), &transposed[0], transposed.size()));
    }

    std::cout << "done" << std::endl;
}

} // anonymous

namespace UT
{

TEST(TestAPI, EmptyGraphUnit)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;
    ASSERT_OK(create_graph_description_eager(&desc));

    Graph_t* graph = NULL;
    ASSERT_ERROR(create_graph(&desc, &graph, 0));

    ASSERT_EQ(graph, nullptr); // zero batch size

    ASSERT_OK(delete_graph(graph));
}

TEST(TestAPI, SimpleGraph1Unit)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;
    ASSERT_OK(create_graph_description_eager(&desc));

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, 1u));
    ASSERT_NE(graph, nullptr);
    ASSERT_OK(delete_graph(graph));
}

TEST(TestAPI, SimpleGraphDescriptionUnit)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;
    ASSERT_OK(create_graph_description_eager(&desc));
    ASSERT_OK(delete_graph_description(desc));
}

TEST(TestAPI, SimpleGraph2Unit)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;
    ASSERT_OK(create_graph_description_eager(&desc));

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, 1u));
    ASSERT_EQ(desc, nullptr);
    ASSERT_OK(delete_graph(graph));
    ASSERT_OK(delete_graph_description(desc));
}

TEST(TestAPI, ArangeUnit)
{
    PROFILE_TEST
    constexpr size_t size = 8;
    Graph_Description_t* desc = NULL;
    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data" };
        ASSERT_OK(add_data_layer(desc, "data", tensors, 1, 1, 2, 2));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, 2));

    arange(graph, "data", FLOAT_TYPE(1), FLOAT_TYPE(2));

    size_t output_size = 0;
    ASSERT_OK(get_tensor(graph, "data", nullptr, &output_size));
    ASSERT_EQ(output_size, size);
    FLOAT_TYPE data[size];
    ASSERT_OK(get_tensor(graph, "data", data, &output_size));
    FLOAT_TYPE real[size] = { 1.0_dt, 3.0_dt, 5.0_dt, 7.0_dt, 9.0_dt, 11.0_dt, 13.0_dt, 15.0_dt };

    for (size_t q = 0; q < output_size; ++q)
        EXPECT_EQ(data[q], real[q]);
}

TEST(TestAPI, PrintUnit)
{
    PROFILE_TEST
    const size_t BATCH_SIZE = 3;

    const size_t NUM_CLASSES = 2;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, 1, 1, NUM_CLASSES, NUM_CLASSES));
    }

    {
        const char* tensorsIn[] = { "data", "labels" };
        ASSERT_OK(add_loss_layer(desc, "loss", tensorsIn, "loss", CROSS_ENTROPY_LOSS, 2, LOSS_REDUCTION_MEAN));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));

    std::filesystem::path tmpPath{ tools::getTempDir() / "tmp.testapi.printunit.data" };
#if defined(_WIN32)
    std::filesystem::path goldenPath{ tools::getTestAssetsDir() / "c-api" / "golden.testapi.printunit.data" };
#else
    std::filesystem::path goldenPath{ tools::getTestAssetsDir() / "c-api" / "golden.testapi.printunit.linux.data" };
#endif

    FILE* f = fopen(tmpPath.string().c_str(), "w");
    EXPECT_OK(print_graph_to_file(graph, f));
    EXPECT_EQ(fclose(f), 0);

    EXPECT_TRUE(std::filesystem::exists(tmpPath));
    EXPECT_TRUE(std::filesystem::exists(goldenPath));

    std::string str = tools::read_file_to_string(tmpPath);
    std::string goldenStr = tools::read_file_to_string(goldenPath);

    EXPECT_TRUE(!str.empty());
    EXPECT_TRUE(!goldenStr.empty());

    EXPECT_EQ(str, goldenStr);

    remove(tmpPath.string().c_str());
}

TEST(TestAPI, PrintStringUnit)
{
    PROFILE_TEST
    const size_t BATCH_SIZE = 3;

    const size_t NUM_CLASSES = 2;

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, 1, 1, NUM_CLASSES, NUM_CLASSES));
    }

    {
        const char* tensorsIn[] = { "data", "labels" };
        ASSERT_OK(add_loss_layer(desc, "loss", tensorsIn, "loss", CROSS_ENTROPY_LOSS, 2, LOSS_REDUCTION_MEAN));
    }

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, BATCH_SIZE));

    {
        size_t size = 0;
        EXPECT_OK(print_graph_to_string(graph, NULL, &size));
        char* str = (char*)malloc((size + 1) * sizeof(char));
        EXPECT_OK(print_graph_to_string(graph, str, &size));

#if defined(_WIN32)
        std::filesystem::path goldenPath{ tools::getTestAssetsDir() / "c-api" / "golden.testapi.printunit.data" };
#else
        std::filesystem::path goldenPath{ tools::getTestAssetsDir() / "c-api" / "golden.testapi.printunit.linux.data" };
#endif
        std::string goldenStr = tools::read_file_to_string(goldenPath);

        EXPECT_EQ(std::string(str), goldenStr);

        free(str);
    }
}

TEST(TestAPI, LSTMLayerUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = 1e-6_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 5U;
    const auto batch_size = 2U;
    const auto lstmName = raul::Name("lstm");

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data" };
        const size_t tensor_amount = 1U;
        ASSERT_OK(add_data_layer(desc, "data", tensors, tensor_amount, sequence_length, 1, input_size));
    }
    ASSERT_OK(add_lstm_layer(desc, lstmName.c_str(), "data", "out", hidden_size, true));

    Graph_t* graph = NULL;
    ASSERT_OK(create_graph(&desc, &graph, batch_size));
    ASSERT_NE(graph, nullptr);

    const FLOAT_TYPE input_data[] = { -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                      7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                      9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                      2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                      -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };

    ASSERT_OK(set_tensor(graph, "data", input_data, batch_size * sequence_length * input_size));

    {
        /// @todo(ck): use tensor initializer
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_ih" / "Weights").c_str(), 1.0_dt));
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_hh" / "Weights").c_str(), 1.0_dt));
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_ih" / "Biases").c_str(), 1.0_dt));
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_hh" / "Biases").c_str(), 1.0_dt));
    }

    ASSERT_OK(network_forward(graph, true));

    FLOAT_TYPE* output_data;
    size_t output_size = 0;
    ASSERT_OK(get_tensor(graph, "out", nullptr, &output_size));

    ASSERT_EQ(output_size, batch_size * sequence_length * hidden_size);

    output_data = (FLOAT_TYPE*)malloc(output_size * sizeof(FLOAT_TYPE));

    ASSERT_OK(get_tensor(graph, "out", output_data, &output_size));

    const FLOAT_TYPE output_golden_data[] = { -1.037908e-02_dt, -1.037908e-02_dt, -1.037908e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, -7.253138e-02_dt, 2.026980e-01_dt, 2.026980e-01_dt,
                                              2.026980e-01_dt,  8.062391e-01_dt,  8.062391e-01_dt,  8.062391e-01_dt,  9.519626e-01_dt,  9.519626e-01_dt,  9.519626e-01_dt, 1.573658e-01_dt,
                                              1.573658e-01_dt,  1.573658e-01_dt,  7.829522e-01_dt,  7.829522e-01_dt,  7.829522e-01_dt,  9.537140e-01_dt,  9.537140e-01_dt, 9.537140e-01_dt,
                                              9.895445e-01_dt,  9.895445e-01_dt,  9.895445e-01_dt,  9.986963e-01_dt,  9.986963e-01_dt,  9.986963e-01_dt };

    for (size_t i = 0; i < output_size; ++i)
    {
        EXPECT_NEAR(output_data[i], output_golden_data[i], eps);
    }

    free(output_data);
}

TEST(TestAPI, LSTMLayerExtUnit)
{
    PROFILE_TEST
    // Test parameters
    const auto eps = 1e-6_dt;
    const auto input_size = 4U;
    const auto hidden_size = 3U;
    const auto sequence_length = 5U;
    const auto batch_size = 2U;
    const auto lstmName = raul::Name("lstm");

    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description_eager(&desc));

    {
        const char* tensors[] = { "data" };
        const size_t tensor_amount = 1U;
        ASSERT_OK(add_data_layer(desc, "data", tensors, tensor_amount, sequence_length, 1, input_size));
    }
    {
        const char* tensors[] = { "hidden_in", "cell_in" };
        const size_t tensor_amount = 2U;
        ASSERT_OK(add_data_layer(desc, "lstm_state_init", tensors, tensor_amount, 1, 1, hidden_size));
    }
    ASSERT_OK(add_lstm_layer_ext(desc, lstmName.c_str(), "data", "hidden_in", "cell_in", "out", "hidden_out", "cell_out", true));

    Graph_t* graph = NULL;
    // ASSERT_OK(create_graph(&desc, &graph, batch_size));
    create_graph(&desc, &graph, batch_size);
    std::cout << get_last_error() << std::endl;
    ASSERT_NE(graph, nullptr);

    const FLOAT_TYPE input_data[] = { -8.024929e-01_dt, -1.295186e+00_dt, -7.501815e-01_dt, -1.311966e+00_dt, -2.188337e-01_dt, -2.435065e+00_dt, -7.291476e-02_dt, -3.398641e-02_dt,
                                      7.968872e-01_dt,  -1.848416e-01_dt, -3.701473e-01_dt, -1.210281e+00_dt, -6.226985e-01_dt, -4.637222e-01_dt, 1.921782e+00_dt,  -4.025455e-01_dt,
                                      9.295023e-02_dt,  -6.660997e-01_dt, 6.080472e-01_dt,  -7.300199e-01_dt, -8.833758e-01_dt, -4.189135e-01_dt, -8.048265e-01_dt, 5.656096e-01_dt,
                                      2.885762e-01_dt,  3.865978e-01_dt,  -2.010639e-01_dt, -1.179270e-01_dt, -8.293669e-01_dt, -1.407257e+00_dt, 1.626847e+00_dt,  1.722732e-01_dt,
                                      -7.042940e-01_dt, 3.147210e-01_dt,  1.573929e-01_dt,  3.853627e-01_dt,  5.736546e-01_dt,  9.979313e-01_dt,  5.436094e-01_dt,  7.880439e-02_dt };

    const FLOAT_TYPE hidden_data[] = { -4.468389e-01_dt, 4.520225e-01_dt, -9.759244e-01_dt, 7.112372e-01_dt, -7.582265e-01_dt, -6.435831e-01_dt };
    const FLOAT_TYPE cell_data[] = { -6.461524e-01_dt, -1.590926e-01_dt, -1.778664e+00_dt, 8.476512e-01_dt, 2.459428e-01_dt, -1.311679e-01_dt };

    ASSERT_OK(set_tensor(graph, "data", input_data, batch_size * sequence_length * input_size));
    ASSERT_OK(set_tensor(graph, "hidden_in", hidden_data, batch_size * hidden_size));
    ASSERT_OK(set_tensor(graph, "cell_in", cell_data, batch_size * hidden_size));

    {
        /// @todo(ck): use tensor initializer
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_ih" / "Weights").c_str(), 1.0_dt));
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_hh" / "Weights").c_str(), 1.0_dt));
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_ih" / "Biases").c_str(), 1.0_dt));
        ASSERT_OK(fill_tensor(graph, (lstmName / "cell" / "linear_hh" / "Biases").c_str(), 1.0_dt));
    }

    ASSERT_OK(network_forward(graph, true));

    const auto verify_tensor = [&](const char* name, const FLOAT_TYPE* data, const size_t size) {
        FLOAT_TYPE* output_data;
        size_t output_size = 0;
        ASSERT_OK(get_tensor(graph, name, nullptr, &output_size));

        ASSERT_EQ(output_size, size);

        output_data = (FLOAT_TYPE*)malloc(output_size * sizeof(FLOAT_TYPE));
        ASSERT_OK(get_tensor(graph, name, output_data, &output_size));

        for (size_t i = 0; i < output_size; ++i)
        {
            EXPECT_NEAR(output_data[i], data[i], eps);
        }
        free(output_data);
    };

    const FLOAT_TYPE output_golden_data[] = { -2.873812e-03_dt, -2.023149e-03_dt, -4.841400e-03_dt, -7.045983e-02_dt, -6.851754e-02_dt, -7.495427e-02_dt, 2.086206e-01_dt, 2.114406e-01_dt,
                                              2.020342e-01_dt,  8.093282e-01_dt,  8.104742e-01_dt,  8.066313e-01_dt,  9.525990e-01_dt,  9.527962e-01_dt,  9.521345e-01_dt, 1.182437e-01_dt,
                                              3.509183e-03_dt,  -6.965960e-02_dt, 7.515067e-01_dt,  6.616085e-01_dt,  5.865334e-01_dt,  9.432809e-01_dt,  9.260048e-01_dt, 9.104356e-01_dt,
                                              9.885837e-01_dt,  9.860258e-01_dt,  9.836924e-01_dt,  9.986322e-01_dt,  9.982802e-01_dt,  9.979585e-01_dt };

    const FLOAT_TYPE hidden_golden_data[] = { 9.525990e-01_dt, 9.527962e-01_dt, 9.521345e-01_dt, 9.986322e-01_dt, 9.982802e-01_dt, 9.979585e-01_dt };
    const FLOAT_TYPE cell_golden_data[] = { 2.193398e+00_dt, 2.197572e+00_dt, 2.183694e+00_dt, 4.067711e+00_dt, 3.832196e+00_dt, 3.684592e+00_dt };

    verify_tensor("out", output_golden_data, batch_size * sequence_length * hidden_size);
    verify_tensor("hidden_out", hidden_golden_data, batch_size * hidden_size);
    verify_tensor("cell_out", cell_golden_data, batch_size * hidden_size);
}

TEST(TestAPI, NinTraining)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description(&desc));

    const raul::dtype LEARNING_RATE = TODTYPE(0.0001);
    const size_t BATCH_SIZE = 32;
    const raul::dtype EPSILON_ACCURACY = TODTYPE(1e-1);

    const size_t NUM_CLASSES = 10;
    const raul::dtype acc1 = TODTYPE(87.18f);
    const raul::dtype acc2 = TODTYPE(87.63f);
    const size_t IMAGE_SIZE = 32;
    const size_t IMAGE_CHANNELS = 3;

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES));
    }

    const char* loss_name = "loss";

    NIN_hyperparams_t hyperparams;

    hyperparams.conv1_filters = 192;
    hyperparams.conv1_kernel_size = 5;
    hyperparams.conv1_stride = 1;
    hyperparams.conv1_padding = 2;

    hyperparams.conv2_filters = 160;
    hyperparams.conv2_kernel_size = 1;

    hyperparams.conv3_filters = 96;
    hyperparams.conv3_kernel_size = 1;

    hyperparams.maxpool_kernel = 3;
    hyperparams.maxpool_stride = 2;
    hyperparams.maxpool_padding = 1;

    hyperparams.conv4_filters = 192;
    hyperparams.conv4_kernel_size = 5;
    hyperparams.conv4_stride = 1;
    hyperparams.conv4_padding = 2;

    hyperparams.conv5_filters = 192;
    hyperparams.conv5_kernel_size = 1;

    hyperparams.conv6_filters = 192;
    hyperparams.conv6_kernel_size = 1;

    hyperparams.avgpool1_kernel = 3;
    hyperparams.avgpool1_stride = 2;
    hyperparams.avgpool1_padding = 1;

    hyperparams.conv7_filters = 192;
    hyperparams.conv7_kernel_size = 3;
    hyperparams.conv7_stride = 1;
    hyperparams.conv7_padding = 1;

    hyperparams.conv8_filters = 192;
    hyperparams.conv8_kernel_size = 1;

    hyperparams.conv9_filters = 10;
    hyperparams.conv9_kernel_size = 1;

    hyperparams.avgpool2_kernel = 8;
    hyperparams.avgpool2_stride = 1;

    ASSERT_OK(add_nin_model(desc, "data", "labels", loss_name, &hyperparams));

    Graph_t* graph = NULL;
    create_graph(&desc, &graph, BATCH_SIZE);
    std::cout << get_last_error() << std::endl;
    ASSERT_NE(graph, nullptr);

    auto load_conv_params = [&graph](const char* name, int index, size_t count, size_t kernel_size, size_t depth) {
        std::string nameWeight = std::string(name) + "::Weights";
        std::string nameBias = std::string(name) + "::Biases";

        raul::DataLoader dataLoader;

        const std::string cifarDir = "cifarNINTrained_BS128_Canonical";
        auto weight = dataLoader.loadFilters(
            (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(index) + ".weight_")).string(), 0, ".data", kernel_size, kernel_size, depth, count);
        auto bias = dataLoader.loadData(tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_classifier." + Conversions::toString(index) + ".bias.data"), 1, count);

        ASSERT_OK(set_tensor(graph, nameWeight.c_str(), weight.first, weight.second - weight.first));
        ASSERT_OK(set_tensor(graph, nameBias.c_str(), bias.first, bias.second - bias.first));
    };

    load_conv_params("conv1", 0, hyperparams.conv1_filters, hyperparams.conv1_kernel_size, IMAGE_CHANNELS);
    load_conv_params("conv2", 2, hyperparams.conv2_filters, hyperparams.conv2_kernel_size, hyperparams.conv1_filters);
    load_conv_params("conv3", 4, hyperparams.conv3_filters, hyperparams.conv3_kernel_size, hyperparams.conv2_filters);
    load_conv_params("conv4", 8, hyperparams.conv4_filters, hyperparams.conv4_kernel_size, hyperparams.conv3_filters);
    load_conv_params("conv5", 10, hyperparams.conv5_filters, hyperparams.conv5_kernel_size, hyperparams.conv4_filters);
    load_conv_params("conv6", 12, hyperparams.conv6_filters, hyperparams.conv6_kernel_size, hyperparams.conv5_filters);
    load_conv_params("conv7", 16, hyperparams.conv7_filters, hyperparams.conv7_kernel_size, hyperparams.conv6_filters);
    load_conv_params("conv8", 18, hyperparams.conv8_filters, hyperparams.conv8_kernel_size, hyperparams.conv7_filters);
    load_conv_params("conv9", 20, hyperparams.conv9_filters, hyperparams.conv9_kernel_size, hyperparams.conv8_filters);

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    auto input_data = std::make_unique<raul::dtype[]>(IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE);

    raul::CIFAR10 cifar;
    ASSERT_EQ(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);

    auto testImages = cifar.getTestImages();
    auto testLabels = cifar.getTestLabels();
    const size_t stepsAmountTest = cifar.getTestImageAmount() / BATCH_SIZE;
    auto totalBatchSize = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE;
    size_t labels[BATCH_SIZE];
    raul::dtype testAcc = 0;
    raul::dtype testTime = 0;
    printf("Begin testing\n");
    for (size_t q = 0; q < stepsAmountTest; ++q)
    {
        raul::dtype accuracy = 0;
        std::transform(testImages.begin() + q * totalBatchSize, testImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
        std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
        ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
        auto timeStart = std::chrono::steady_clock::now();
        ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
        testTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
        testAcc += accuracy;
    }
    testAcc /= static_cast<raul::dtype>(stepsAmountTest);
    CHECK_NEAR(testAcc * 100_dt, acc1, EPSILON_ACCURACY);
    printf("Testing taken = %.3fs \n", testTime / 1000.f);
    printf("Test accuracy = %.2f\n", testAcc * 100);

    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / BATCH_SIZE;

    raul::DataLoader dataLoader;

    auto trainImages = cifar.getTrainImages();
    auto trainLabels = cifar.getTrainLabels();
    auto& encodedTrainLabels = dataLoader.buildOneHotVector(trainLabels, NUM_CLASSES);

    raul::Tensor& label_data = dataLoader.createTensor(NUM_CLASSES * BATCH_SIZE);

    raul::dtype totalTime = 0;
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        raul::dtype averageLoss = 0;
        raul::dtype epochTime = 0;
        for (size_t q = 0; q < stepsAmountTrain; ++q)
        {
            std::transform(trainImages.begin() + q * totalBatchSize, trainImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
            std::copy(encodedTrainLabels.begin() + q * BATCH_SIZE * NUM_CLASSES, encodedTrainLabels.begin() + (q + 1) * BATCH_SIZE * NUM_CLASSES, label_data.begin());

            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
            ASSERT_OK(set_tensor(graph, "labels", &label_data[0], NUM_CLASSES * BATCH_SIZE));

            raul::dtype testLoss = 0.f;
            auto timeStart = std::chrono::steady_clock::now();
            ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
            epochTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
            averageLoss += testLoss;
            if (q % (stepsAmountTrain / 3) == 0)
            {
                printf("iteration = %zu / %zu, loss = %f\n", q, stepsAmountTrain, testLoss);
            }
        }
        printf("Average loss = %f\n", averageLoss / static_cast<float>(stepsAmountTrain));
        printf("Epoch Training taken = %.3fs \n", epochTime / 1000.f);
        totalTime += epochTime;

        testAcc = 0;
        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            raul::dtype accuracy = 0;
            std::transform(testImages.begin() + q * totalBatchSize, testImages.begin() + (q + 1) * totalBatchSize, input_data.get(), [](auto _n) { return _n / 255.f; });
            std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSize));
            ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
            testAcc += accuracy;
        }
        testAcc /= static_cast<raul::dtype>(stepsAmountTest);
        printf("Test accuracy = %.2f\n", testAcc * 100);
        CHECK_NEAR(testAcc * 100.0_dt, acc2, EPSILON_ACCURACY);
    }

    printf("Time taken = %.3fs \n", totalTime / 1000.f);

    ASSERT_EQ(delete_optimizer(sgd_optimizer), STATUS_OK);
    ASSERT_EQ(delete_graph(graph), STATUS_OK);
}

TEST(TestAPI, MobileNetV2Training)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description(&desc));

    const raul::dtype LEARNING_RATE = 0.05_dt;
    const size_t BATCH_SIZE = 50;
    auto EPSILON_ACCURACY = 0.5_dt;

    const size_t NUM_CLASSES = 10;
    const auto acc1 = 83.36_dt;
    const auto acc2 = 83.41_dt;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES));
    }

    const char* loss_name = "loss";

    MobileNetV2_hyperparams_t hyperparams;
    hyperparams.bnMomentum = 0.1f;
    hyperparams.num_classes = NUM_CLASSES;

    size_t filterSizes[hyperparams.reproduceLayers][3] = { { 96, 96, 24 },    { 144, 144, 24 },  { 144, 144, 32 },  { 192, 192, 32 }, { 192, 192, 32 }, { 192, 192, 64 },
                                                           { 384, 384, 64 },  { 384, 384, 64 },  { 384, 384, 64 },  { 384, 384, 96 }, { 576, 576, 96 }, { 576, 576, 96 },
                                                           { 576, 576, 160 }, { 960, 960, 160 }, { 960, 960, 160 }, { 960, 960, 320 } };

    for (size_t q = 0; q < hyperparams.reproduceLayers; ++q)
    {
        std::copy(&filterSizes[q][0], &filterSizes[q][3], &hyperparams.filterSizes[q][0]);
    }

    const size_t strideSizes[hyperparams.reproduceLayers] = { 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1 };
    std::copy(&strideSizes[0], &strideSizes[hyperparams.reproduceLayers], &hyperparams.strideSizes[0]);

    const bool residual[hyperparams.reproduceLayers] = { 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0 };
    std::copy(&residual[0], &residual[hyperparams.reproduceLayers], &hyperparams.residual[0]);

    hyperparams.lastLayerSize = 1280;

    hyperparams.avgWidth = 7;

    ASSERT_OK(add_mobilenetv2_model(desc, "data", "labels", loss_name, &hyperparams));
    Graph_t* graph = NULL;
    create_graph(&desc, &graph, BATCH_SIZE);
    std::cout << get_last_error() << std::endl;
    ASSERT_NE(graph, nullptr);

    auto load_filters =
        [&graph](const std::string& weightName, const std::string& fileName, size_t fileIndexOffset, const std::string& pathPostfix, size_t width, size_t height, size_t depth, size_t filtersAmount) {
            raul::DataLoader dataLoader;

            auto weight = dataLoader.loadFilters(fileName, fileIndexOffset, pathPostfix, width, height, depth, filtersAmount);

            ASSERT_OK(set_tensor(graph, weightName.c_str(), weight.first, weight.second - weight.first));
        };

    auto load_weight = [&graph](const std::string& weightName, const std::filesystem::path& path, size_t width, size_t height, size_t depth = 1, bool transpose = false) {
        raul::DataLoader dataLoader;

        auto weight = dataLoader.loadData(path, width, height, depth);

        if (transpose)
        {
            raul::Tensor transposed(weight);
            raul::Common::transpose(transposed, width);
            ASSERT_OK(set_tensor(graph, weightName.c_str(), &transposed[0], transposed.size()));
        }
        else
        {
            ASSERT_OK(set_tensor(graph, weightName.c_str(), weight.first, weight.second - weight.first));
        }
    };

    const std::string cifarDir = "cifarMobilenetV2";

    load_filters("conv1::Weights", (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.0.weight_").string(), 0, ".data", 3, 3, IMAGE_CHANNELS, 32);
    load_weight("conv1::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.0.bias.data", 1, 32);
    load_weight("bn1::Weights", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.weight.data", 1, 32);
    load_weight("bn1::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.bias.data", 1, 32);
    load_weight("bn1::VarianceEval", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.running_var.data", 1, 32);
    load_weight("bn1::MeanEval", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.0.1.running_mean.data", 1, 32);

    // dw
    load_filters("conv2::Weights", (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.0.weight_").string(), 0, ".data", 3, 3, 1, 32);
    load_weight("conv2::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.0.bias.data", 1, 32);
    load_weight("bn2::Weights", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.weight.data", 1, 32);
    load_weight("bn2::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.bias.data", 1, 32);
    load_weight("bn2::VarianceEval", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.running_var.data", 1, 32);
    load_weight("bn2::MeanEval", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.1.running_mean.data", 1, 32);

    load_filters("conv3::Weights", (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.3.weight_").string(), 0, ".data", 1, 1, 32, 16);
    load_weight("conv3::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.3.bias.data", 1, 16);
    load_weight("bn3::Weights", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.weight.data", 1, 16);
    load_weight("bn3::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.bias.data", 1, 16);
    load_weight("bn3::VarianceEval", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.running_var.data", 1, 16);
    load_weight("bn3::MeanEval", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_features.1.conv.4.running_mean.data", 1, 16);

    const size_t firstFileIndex = 0;
    const size_t secondFileIndex = 3;
    const size_t thirdFileIndex = 6;

    size_t fileNames[hyperparams.reproduceLayers][3] = { { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex },
                                                         { firstFileIndex, secondFileIndex, thirdFileIndex }, { firstFileIndex, secondFileIndex, thirdFileIndex } };

    size_t layerIndexLoad = 4;

    for (size_t w = 0; w < hyperparams.reproduceLayers; ++w)
    {
        size_t prevFilterSizeFirst = 16;
        if (w != 0)
        {
            prevFilterSizeFirst = filterSizes[w - 1][2];
        }

        load_filters(
            "conv" + Conversions::toString(layerIndexLoad) + "::Weights",
            (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0]) + ".weight_")).string(),
            0,
            ".data",
            1,
            1,
            prevFilterSizeFirst,
            filterSizes[w][0]);
        load_weight("conv" + Conversions::toString(layerIndexLoad) + "::Biases",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0]) + ".bias.data"),
                    1,
                    filterSizes[w][0]);

        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Weights",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".weight.data"),
                    1,
                    filterSizes[w][0]);
        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Biases",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".bias.data"),
                    1,
                    filterSizes[w][0]);

        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir /
                        ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".running_var.data"),
                    1,
                    filterSizes[w][0]);
        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::MeanEval",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir /
                        ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][0] + 1) + ".running_mean.data"),
                    1,
                    filterSizes[w][0]);

        ++layerIndexLoad;

        // dw
        load_filters(
            "conv" + Conversions::toString(layerIndexLoad) + "::Weights",
            (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1]) + ".weight_")).string(),
            0,
            ".data",
            3,
            3,
            1,
            filterSizes[w][1]);
        load_weight("conv" + Conversions::toString(layerIndexLoad) + "::Biases",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1]) + ".bias.data"),
                    1,
                    filterSizes[w][1]);

        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Weights",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".weight.data"),
                    1,
                    filterSizes[w][1]);
        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Biases",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".bias.data"),
                    1,
                    filterSizes[w][1]);

        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir /
                        ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".running_var.data"),
                    1,
                    filterSizes[w][1]);
        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::MeanEval",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir /
                        ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][1] + 1) + ".running_mean.data"),
                    1,
                    filterSizes[w][1]);

        ++layerIndexLoad;

        load_filters(
            "conv" + Conversions::toString(layerIndexLoad) + "::Weights",
            (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2]) + ".weight_")).string(),
            0,
            ".data",
            1,
            1,
            filterSizes[w][1],
            filterSizes[w][2]);
        load_weight("conv" + Conversions::toString(layerIndexLoad) + "::Biases",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2]) + ".bias.data"),
                    1,
                    filterSizes[w][2]);

        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Weights",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".weight.data"),
                    1,
                    filterSizes[w][2]);
        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Biases",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".bias.data"),
                    1,
                    filterSizes[w][2]);

        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir /
                        ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".running_var.data"),
                    1,
                    filterSizes[w][2]);
        load_weight("bn" + Conversions::toString(layerIndexLoad) + "::MeanEval",
                    tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir /
                        ("0_features." + Conversions::toString(w + 2) + ".conv." + Conversions::toString(fileNames[w][2] + 1) + ".running_mean.data"),
                    1,
                    filterSizes[w][2]);

        ++layerIndexLoad;
    }

    load_filters("conv" + Conversions::toString(layerIndexLoad) + "::Weights",
                 (tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(hyperparams.reproduceLayers + 2) + ".0.weight_")).string(),
                 0,
                 ".data",
                 1,
                 1,
                 filterSizes[hyperparams.reproduceLayers - 1][2],
                 hyperparams.lastLayerSize);
    load_weight("conv" + Conversions::toString(layerIndexLoad) + "::Biases",
                tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(hyperparams.reproduceLayers + 2) + ".0.bias.data"),
                1,
                hyperparams.lastLayerSize);

    load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Weights",
                tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(hyperparams.reproduceLayers + 2) + ".1.weight.data"),
                1,
                hyperparams.lastLayerSize);
    load_weight("bn" + Conversions::toString(layerIndexLoad) + "::Biases",
                tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(hyperparams.reproduceLayers + 2) + ".1.bias.data"),
                1,
                hyperparams.lastLayerSize);

    load_weight("bn" + Conversions::toString(layerIndexLoad) + "::VarianceEval",
                tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(hyperparams.reproduceLayers + 2) + ".1.running_var.data"),
                1,
                hyperparams.lastLayerSize);
    load_weight("bn" + Conversions::toString(layerIndexLoad) + "::MeanEval",
                tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / ("0_features." + Conversions::toString(hyperparams.reproduceLayers + 2) + ".1.running_mean.data"),
                1,
                hyperparams.lastLayerSize);

    load_weight("fc::Weights", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.weight.data", NUM_CLASSES, hyperparams.lastLayerSize, 1, true);

    load_weight("fc::Biases", tools::getTestAssetsDir() / "test_cnn_layer" / cifarDir / "0_classifier.bias.data", 1, NUM_CLASSES);

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    auto input_data = std::make_unique<raul::dtype[]>(IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE);

    raul::CIFAR10 cifar;
    ASSERT_EQ(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);

    auto testImages = cifar.getTestImages();
    auto testLabels = cifar.getTestLabels();
    const size_t stepsAmountTest = cifar.getTestImageAmount() / BATCH_SIZE;
    auto totalBatchSizeOriginal = cifar.getImageSize() * cifar.getImageSize() * IMAGE_CHANNELS * BATCH_SIZE;
    auto totalBatchSizeRescaled = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE;
    size_t labels[BATCH_SIZE];
    raul::dtype testAcc = 0;
    raul::dtype testTime = 0;
    printf("Begin testing\n");
    for (size_t q = 0; q < stepsAmountTest; ++q)
    {
        raul::dtype accuracy = 0;
        raul::TensorU8 originalBatch(BATCH_SIZE, IMAGE_CHANNELS, cifar.getImageSize(), cifar.getImageSize());
        std::copy_n(testImages.begin() + q * totalBatchSizeOriginal, totalBatchSizeOriginal, originalBatch.begin());

        raul::TensorU8 rescaledBatch(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
        reshapeTensor(originalBatch, rescaledBatch);
        std::transform(rescaledBatch.begin(), rescaledBatch.end(), input_data.get(), [](auto _n) { return _n / 255.f; });

        std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
        ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSizeRescaled));
        auto timeStart = std::chrono::steady_clock::now();
        ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
        testTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
        testAcc += accuracy;
    }
    testAcc /= static_cast<raul::dtype>(stepsAmountTest);
    CHECK_NEAR(testAcc * 100_dt, acc1, EPSILON_ACCURACY);
    printf("Testing taken = %.3fs \n", testTime / 1000.f);
    printf("Test accuracy = %.2f\n", testAcc * 100);

    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / BATCH_SIZE;

    raul::DataLoader dataLoader;

    auto trainImages = cifar.getTrainImages();
    auto trainLabels = cifar.getTrainLabels();
    auto& encodedTrainLabels = dataLoader.buildOneHotVector(trainLabels, NUM_CLASSES);

    raul::Tensor& label_data = dataLoader.createTensor(NUM_CLASSES * BATCH_SIZE);

    raul::dtype totalTime = 0;
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        raul::dtype epochTime = 0;
        for (size_t q = 0; q < stepsAmountTrain; ++q)
        {
            raul::TensorU8 originalBatch(BATCH_SIZE, IMAGE_CHANNELS, cifar.getImageSize(), cifar.getImageSize());
            std::copy_n(trainImages.begin() + q * totalBatchSizeOriginal, totalBatchSizeOriginal, originalBatch.begin());

            raul::TensorU8 rescaledBatch(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
            reshapeTensor(originalBatch, rescaledBatch);
            std::transform(rescaledBatch.begin(), rescaledBatch.end(), input_data.get(), [](auto _n) { return _n / 255.f; });

            std::copy(encodedTrainLabels.begin() + q * BATCH_SIZE * NUM_CLASSES, encodedTrainLabels.begin() + (q + 1) * BATCH_SIZE * NUM_CLASSES, label_data.begin());

            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSizeRescaled));
            ASSERT_OK(set_tensor(graph, "labels", &label_data[0], NUM_CLASSES * BATCH_SIZE));

            raul::dtype testLoss = 0.f;
            auto timeStart = std::chrono::steady_clock::now();
            ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
            epochTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
            if (q % 100 == 0)
            {
                printf("iteration = %zu / %zu, loss = %f\n", q, stepsAmountTrain, testLoss);
            }
        }
        printf("Epoch Training taken = %.3fs \n", epochTime / 1000.f);
        totalTime += epochTime;

        testAcc = 0;
        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            raul::dtype accuracy = 0;
            raul::TensorU8 originalBatch(BATCH_SIZE, IMAGE_CHANNELS, cifar.getImageSize(), cifar.getImageSize());
            std::copy_n(testImages.begin() + q * totalBatchSizeOriginal, totalBatchSizeOriginal, originalBatch.begin());

            raul::TensorU8 rescaledBatch(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
            reshapeTensor(originalBatch, rescaledBatch);
            std::transform(rescaledBatch.begin(), rescaledBatch.end(), input_data.get(), [](auto _n) { return _n / 255.f; });

            std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSizeRescaled));
            ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
            testAcc += accuracy;
        }
        testAcc /= static_cast<raul::dtype>(stepsAmountTest);
        printf("Test accuracy = %.2f\n", testAcc * 100);
        CHECK_NEAR(testAcc * 100.0_dt, acc2, EPSILON_ACCURACY);
    }

    printf("Time taken = %.3fs \n", totalTime / 1000.f);

    ASSERT_EQ(delete_optimizer(sgd_optimizer), STATUS_OK);
    ASSERT_EQ(delete_graph(graph), STATUS_OK);
}

TEST(TestAPI, ResNet18Training)
{

    PROFILE_TEST
    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description(&desc));

    const raul::dtype LEARNING_RATE = 0.05_dt;
    const size_t BATCH_SIZE = 50;
    const auto EPSILON_ACCURACY = 0.1_dt;

    const auto acc1 = 83.24_dt;
    const auto acc2 = 85.24_dt;
    const size_t NUM_CLASSES = 10;
    const size_t IMAGE_SIZE = 224;
    const size_t IMAGE_CHANNELS = 3;

    {
        const char* tensors[] = { "data", "labels" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data", tensors, 2, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES));
    }

    const char* loss_name = "loss";

    ResNet18_hyperparams_t hyperparams;

    ASSERT_OK(add_resnet18_model(desc, "data", "labels", loss_name, &hyperparams));
    Graph_t* graph = NULL;
    create_graph(&desc, &graph, BATCH_SIZE);
    std::cout << get_last_error() << std::endl;
    ASSERT_NE(graph, nullptr);

    const auto weights_path = tools::getTestAssetsDir() / "resnet" / "18" / "seed_0_epoch10_acc_83.24";
    load_resnet_weights(graph, weights_path, "83.24_resnet.", true);

    Optimizer_t* sgd_optimizer = NULL;
    ASSERT_OK(create_sgd_optimizer(&sgd_optimizer, LEARNING_RATE));

    auto input_data = std::make_unique<raul::dtype[]>(IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE);

    raul::CIFAR10 cifar;
    ASSERT_EQ(cifar.loadingData(tools::getTestAssetsDir() / "CIFAR"), true);

    auto testImages = cifar.getTestImages();
    auto testLabels = cifar.getTestLabels();
    const size_t stepsAmountTest = cifar.getTestImageAmount() / BATCH_SIZE;
    auto totalBatchSizeOriginal = cifar.getImageSize() * cifar.getImageSize() * IMAGE_CHANNELS * BATCH_SIZE;
    auto totalBatchSizeRescaled = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS * BATCH_SIZE;
    size_t labels[BATCH_SIZE];
    raul::dtype testAcc = 0;
    raul::dtype testTime = 0;
    printf("Begin testing\n");
    for (size_t q = 0; q < stepsAmountTest; ++q)
    {
        raul::dtype accuracy = 0;
        raul::TensorU8 originalBatch(BATCH_SIZE, IMAGE_CHANNELS, cifar.getImageSize(), cifar.getImageSize());
        std::copy_n(testImages.begin() + q * totalBatchSizeOriginal, totalBatchSizeOriginal, originalBatch.begin());

        raul::TensorU8 rescaledBatch(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
        reshapeTensor(originalBatch, rescaledBatch);
        std::transform(rescaledBatch.begin(), rescaledBatch.end(), input_data.get(), [](auto _n) { return _n / 255.f; });

        std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
        ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSizeRescaled));
        auto timeStart = std::chrono::steady_clock::now();
        ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
        testTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
        testAcc += accuracy;
    }
    testAcc /= static_cast<raul::dtype>(stepsAmountTest);
    CHECK_NEAR(testAcc * 100_dt, acc1, EPSILON_ACCURACY);
    printf("Testing taken = %.3fs \n", testTime / 1000.f);
    printf("Test accuracy = %.2f\n", testAcc * 100);

    const size_t stepsAmountTrain = cifar.getTrainImageAmount() / BATCH_SIZE;

    raul::DataLoader dataLoader;

    auto trainImages = cifar.getTrainImages();
    auto trainLabels = cifar.getTrainLabels();
    auto& encodedTrainLabels = dataLoader.buildOneHotVector(trainLabels, NUM_CLASSES);

    raul::Tensor& label_data = dataLoader.createTensor(NUM_CLASSES * BATCH_SIZE);

    raul::dtype totalTime = 0;
    for (size_t epoch = 1; epoch <= 1; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        raul::dtype epochTime = 0;
        for (size_t q = 0; q < stepsAmountTrain; ++q)
        {
            raul::TensorU8 originalBatch(BATCH_SIZE, IMAGE_CHANNELS, cifar.getImageSize(), cifar.getImageSize());
            std::copy_n(trainImages.begin() + q * totalBatchSizeOriginal, totalBatchSizeOriginal, originalBatch.begin());

            raul::TensorU8 rescaledBatch(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
            reshapeTensor(originalBatch, rescaledBatch);
            std::transform(rescaledBatch.begin(), rescaledBatch.end(), input_data.get(), [](auto _n) { return _n / 255.f; });

            std::copy(encodedTrainLabels.begin() + q * BATCH_SIZE * NUM_CLASSES, encodedTrainLabels.begin() + (q + 1) * BATCH_SIZE * NUM_CLASSES, label_data.begin());

            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSizeRescaled));
            ASSERT_OK(set_tensor(graph, "labels", &label_data[0], NUM_CLASSES * BATCH_SIZE));

            raul::dtype testLoss = 0.f;
            auto timeStart = std::chrono::steady_clock::now();
            ASSERT_OK(train_single_pass(graph, sgd_optimizer, &loss_name, 1u, &testLoss));
            epochTime += static_cast<raul::dtype>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count());
            if (q % 100 == 0)
            {
                printf("iteration = %zu / %zu, loss = %f\n", q, stepsAmountTrain, testLoss);
            }
        }
        printf("Epoch Training taken = %.3fs \n", epochTime / 1000.f);
        totalTime += epochTime;

        testAcc = 0;
        for (size_t q = 0; q < stepsAmountTest; ++q)
        {
            raul::dtype accuracy = 0;
            raul::TensorU8 originalBatch(BATCH_SIZE, IMAGE_CHANNELS, cifar.getImageSize(), cifar.getImageSize());
            std::copy_n(testImages.begin() + q * totalBatchSizeOriginal, totalBatchSizeOriginal, originalBatch.begin());

            raul::TensorU8 rescaledBatch(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
            reshapeTensor(originalBatch, rescaledBatch);
            std::transform(rescaledBatch.begin(), rescaledBatch.end(), input_data.get(), [](auto _n) { return _n / 255.f; });

            std::copy_n(testLabels.begin() + q * BATCH_SIZE, BATCH_SIZE, labels);
            ASSERT_OK(set_tensor(graph, "data", input_data.get(), totalBatchSizeRescaled));
            ASSERT_OK(test_network(graph, "softmax", labels, &accuracy));
            testAcc += accuracy;
        }
        testAcc /= static_cast<raul::dtype>(stepsAmountTest);
        printf("Test accuracy = %.2f\n", testAcc * 100);
        CHECK_NEAR(testAcc * 100.0_dt, acc2, EPSILON_ACCURACY);
    }

    printf("Time taken = %.3fs \n", totalTime / 1000.f);

    ASSERT_EQ(delete_optimizer(sgd_optimizer), STATUS_OK);
    ASSERT_EQ(delete_graph(graph), STATUS_OK);
}

TEST(TestAPI, YannetTraining)
{
    PROFILE_TEST
    Graph_Description_t* desc = NULL;

    ASSERT_OK(create_graph_description(&desc));

    const size_t EPOCHS_TO_TRAIN = 1;
    const size_t EPOCHS_FOR_SCHEDULER = 50;
    const size_t STEPS_AMOUNT = 5000;

    const size_t BATCH_SIZE = 2;
    const size_t KEY_LENGTH = 48;
    const size_t QUERY_SIZE = 12;
    const size_t KEY_SIZE = 34;
    const size_t HIDDEN_SIZE = 32;

    // Cosine annealing params
    const raul::dtype MAX_A = 1.0_dt;
    const raul::dtype MIN_A = 0.0_dt;
    const raul::dtype WARMUP_PERCENTAGE = 0.1_dt;
    const raul::dtype WARMUP_POW = 1.0_dt;
    const raul::dtype ANNEALING_POW = 1.0_dt;

    // AdamW params
    const raul::dtype BASE_LEARNING_RATE = 0.01_dt;
    const raul::dtype BETA_1 = 0.9_dt;
    const raul::dtype BETA_2 = 0.999_dt;
    const raul::dtype EPSILON = 1e-8_dt;
    const raul::dtype WEIGHT_DECAY = 0.01_dt;

    const raul::dtype EPSILON_ACCURACY = 0.1_dt;

    // key and targets
    {
        const char* tensors[] = { "key", "targets" };
        ASSERT_OK(add_data_layer_with_labels(desc, "data1", tensors, 2, 1, KEY_LENGTH, KEY_SIZE, 1));
    }

    // query
    {
        const char* tensors[] = { "query" };
        ASSERT_OK(add_data_layer(desc, "data2", tensors, 1, 1, 1, QUERY_SIZE));
    }

    // initial hidden state
    {
        const char* tensors[] = { "initial_h" };
        ASSERT_OK(add_data_layer(desc, "data3", tensors, 1, 1, 1, HIDDEN_SIZE));
    }

    const char* loss_name = "loss";

    ASSERT_OK(add_yannet_model(desc, "query", "key", "initial_h", "targets", loss_name));

    Graph_t* graph = NULL;
    create_graph(&desc, &graph, BATCH_SIZE);
    std::cout << get_last_error() << std::endl;
    ASSERT_NE(graph, nullptr);

    // Load input data
    raul::DataLoader dataLoader;
    auto query = dataLoader.loadData(tools::getTestAssetsDir() / "wake_up" / "api_test" / "query.txt", QUERY_SIZE, 1, 1, BATCH_SIZE * STEPS_AMOUNT);
    auto key = dataLoader.loadData(tools::getTestAssetsDir() / "wake_up" / "api_test" / "key.txt", KEY_SIZE, KEY_LENGTH, 1, BATCH_SIZE * STEPS_AMOUNT);
    auto targets = dataLoader.loadData(tools::getTestAssetsDir() / "wake_up" / "api_test" / "targets.txt", 1, 1, 1, BATCH_SIZE * STEPS_AMOUNT);

    // Load ideal losses
    auto ideal_loss = dataLoader.loadData(tools::getTestAssetsDir() / "wake_up" / "api_test" / "ideal_losses_1_epoch.txt", 1, 1, 1, STEPS_AMOUNT);

    load_yannet_weights(graph, tools::getTestAssetsDir() / "wake_up" / "api_test" / "weights");

    LrScheduler_t* lr_scheduler = NULL;
    ASSERT_OK(create_cosine_annealing_adam_w_lr_scheduler(&lr_scheduler, EPOCHS_FOR_SCHEDULER * STEPS_AMOUNT, MAX_A, MIN_A, WARMUP_PERCENTAGE, WARMUP_POW, ANNEALING_POW, BASE_LEARNING_RATE, BETA_1, BETA_2, EPSILON, WEIGHT_DECAY));

    for (size_t epoch = 1; epoch <= EPOCHS_TO_TRAIN; ++epoch)
    {
        printf("Epoch = %zu\n", epoch);

        for (size_t q = 0; q < STEPS_AMOUNT; ++q)
        {
            // Set input data
            ASSERT_OK(set_tensor(graph, "query", query.first + q * BATCH_SIZE * QUERY_SIZE, BATCH_SIZE * QUERY_SIZE));
            ASSERT_OK(set_tensor(graph, "key", key.first + q * BATCH_SIZE * KEY_LENGTH * KEY_SIZE, BATCH_SIZE * KEY_LENGTH * KEY_SIZE));
            ASSERT_OK(set_tensor(graph, "targets", targets.first + q * BATCH_SIZE, BATCH_SIZE));

            // Calculate loss on currest step
            raul::dtype currentLoss = 0.f;
            ASSERT_OK(train_single_pass_with_scheduling(graph, lr_scheduler, &loss_name, 1u, &currentLoss));
            CHECK_NEAR(currentLoss, ideal_loss.first[q], EPSILON_ACCURACY);

            if (q % 500 == 0)
            {
                printf("Step = %zu, current loss = %f, ideal loss = %f\n", q, currentLoss, ideal_loss.first[q]);
            }
        }
    }
}

} // UT namespace
