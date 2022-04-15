// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/tools/TestTools.h>

#include <training/api/API.h>

#define ASSERT_OK(x) ASSERT_EQ(x, STATUS_OK)
#define EXPECT_OK(x) EXPECT_EQ(x, STATUS_OK)
#define ASSERT_ERROR(x) ASSERT_NE(x, STATUS_OK)

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
    ASSERT_OK(add_lstm_layer(desc, lstmName.c_str(), "data", "out", hidden_size, false, true));

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
    ASSERT_OK(add_lstm_layer_ext(desc, lstmName.c_str(), "data", "hidden_in", "cell_in", "out", "hidden_out", "cell_out", false, true));

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

} // UT namespace