// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <tests/GTestExtensions.h>
#include <tests/tools/TestTools.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/initializers/XavierInitializer.h>
#include <training/layers/activations/LeakyReLUActivation.h>
#include <training/layers/activations/SigmoidActivation.h>
#include <training/layers/activations/SoftMaxActivation.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/ElementWiseMulLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/ReduceSumLayer.h>
#include <training/layers/composite/rnn/GRULayer.h>
#include <training/layers/basic/TensorLayer.h>
#include <training/layers/basic/TransposeLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/basic/MatMulLayer.h>
#include <training/layers/basic/ReduceSumLayer.h>
#include <training/network/Workflow.h>
#include <training/optimizers/SGD.h>

namespace
{
    using namespace raul;
    
    void create_small_model(Workflow& work, const Names& inputs, const Name& out) 
    {
        size_t INPUT1_SIZE = 12;
        size_t INPUT2_LENGTH = 48;
        size_t INPUT2_SIZE = 34;

        size_t GRU_SIZE = 6;
        size_t LINEAR_SIZE = 6;
        size_t RELU_LINEAR_SIZE = 1;
        size_t FINAL_LINEAR_SIZE = 1;

        dtype RELU_ALPHA = 0.009999999776482582;

        auto [input1, input2, initial_h] = array<Name, 3>{inputs[0], inputs[1], inputs[2]};

        work.add<DataLayer>("input2", DataParams{ { input2 }, 1, INPUT2_LENGTH, INPUT2_SIZE });
        work.add<TensorLayer>("initial_h", TensorParams{ { "initial_h" }, { BS(), 1, 1, GRU_SIZE }, Workflow::Usage::ForwardAndBackward });
        //work.add<TransposeLayer>("Transpose_6", TransposingParams{ input2, "input2_t", "depth", "height" });
        GRULayer("gru", GRUParams{ { input2, "initial_h" }, { "gru_out", "new_hidden" } }, work.getNetworkParameters());
        work.add<LinearLayer>("linear2", LinearParams{ "gru_out", "linear2_out", LINEAR_SIZE });

        work.add<DataLayer>("input1", DataParams{ { input1 }, 1, 1, INPUT1_SIZE });
        work.add<LinearLayer>("linear1", LinearParams{ input1, "linear1_out", LINEAR_SIZE });

        work.add<ElementWiseSumLayer>("add", ElementWiseLayerParams{ {"linear2_out", "linear1_out"}, "sum" });
        work.add<LeakyReLUActivation>("relu", LeakyReLUParams{ "sum", "relu_out", RELU_ALPHA });
        work.add<LinearLayer>("linear3", LinearParams{ "relu_out", "linear3_out", RELU_LINEAR_SIZE });

        //work.add<TransposeLayer>("Transpose_18", TransposingParams{ "linear3_out", "linear3_out_t", "width", "height" });
        //work.add<SoftMaxActivation>("Softmax_19", BasicParamsWithDim{ { "linear3_out_t" }, { "softmax_t" }, "width" });
        //work.add<TransposeLayer>("Transpose_20", TransposingParams{ "softmax_t", "softmax", "width", "height" });
        work.add<SoftMaxActivation>("Softmax_19", BasicParamsWithDim{ { "linear3_out" }, { "softmax" }, "width" });

        work.add<ElementWiseMulLayer>("mul", ElementWiseLayerParams{ { "linear2_out", "softmax"}, "mul" });
        work.add<ReduceSumLayer>("reduce", BasicParamsWithDim{ { "mul" }, { "mul_reduced" }, "height" });
        work.add<LinearLayer>("Gemm_23", LinearParams{ "mul_reduced", "linear4_out", FINAL_LINEAR_SIZE });

        work.add<SigmoidActivation>("sigmoid", BasicParamsWithDim{ { "linear4_out" }, { out } });
    }
}

namespace UT
{

TEST(TestSSDWakeUp, SmallModelDimensionsUnit)
{
    PROFILE_TEST
    using namespace raul;

    size_t BATCH_SIZE = 1;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL);

    create_small_model(work, { "input1", "input2", "initial_h" }, "out");

    TENSORS_CREATE(BATCH_SIZE);

    work.printInfo(cout);
}

TEST(TestSSDWakeUp, SmallModelUnit)
{
    PROFILE_TEST
    using namespace raul;

    size_t BATCH_SIZE = UT::tools::getArg("batch", 1);
    cout << "Batch size: " << BATCH_SIZE << endl;
    size_t NITERS = 100;
    size_t seed = 0;
    auto LEARNING_RATE = 1e-3_dt;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL);

    create_small_model(work, { "input1", "input2", "initial_h" }, "out");

    TENSORS_CREATE(BATCH_SIZE);

    random::setGlobalSeed(seed);
    initializers::XavierNormInitializer initializer(seed);

    auto& memory_manager = work.getMemoryManager();
    memory_manager["input1"] = 1_dt;
    memory_manager["input2"] = 0.5_dt;
    memory_manager["initial_h"] = 0_dt;

    for (const auto& tp : work.getTrainableParameterNames())
    {
        initializer(memory_manager[tp]);
    }

    work.forwardPassTraining();
    work.backwardPassTraining();

    auto timeStart = chrono::steady_clock::now();

    auto optimizer = std::make_unique<optimizers::SGD>(LEARNING_RATE);

    auto trainableParams = work.getTrainableParameters();

    for (size_t iter = 0; iter < NITERS; ++iter)
    {
        work.forwardPassTraining();
        work.backwardPassTraining();

        for (auto& p : trainableParams)
        {
            optimizer->operator()(memory_manager, p.Param, p.Gradient);
        }
    }

    auto elapsed = TODTYPE(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeStart).count());

    cout << "Iteration time: " << elapsed / NITERS << "ms" << endl;
}


}
