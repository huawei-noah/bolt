// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TestTools.h"

#include <tests/tools/TestTools.h>
#include <training/api/API.h>
#include <training/layers/BasicLayer.h>
#include <training/network/Layers.h>

#ifndef RAUL_ASSETS
#define RAUL_ASSETS "assets"
#endif

namespace UT::tools
{

std::map<string, string> ARGS::ARGUMENTS;

using namespace std;
using namespace raul;

filesystem::path getTempDir()
{
    return filesystem::temp_directory_path();
}

template<>
void init_rand_tensor(const string& tensor_name, const random::dataRange<MemoryManagerGPU::type> random_range, MemoryManagerGPU& memory_manager)
{
    Tensor tensor = memory_manager[tensor_name];
    for (auto& val : tensor)
    {
        val = random::uniform::rand<MemoryManagerGPU::type>(random_range);
    }
    memory_manager[tensor_name] = TORANGE(tensor);
}

filesystem::path getTestAssetsDir()
{
    auto current_path = filesystem::current_path();
    const char* env_raul_test_assets_dir = getenv("RAUL_ASSETS");
    auto relative_path = (env_raul_test_assets_dir == nullptr) ? string(RAUL_ASSETS) : string(env_raul_test_assets_dir);
    current_path /= relative_path;

    return current_path.make_preferred();
}

void print_tensor(const Tensor& t, const filesystem::path& s)
{
    size_t ind = 0;
    ofstream f(s);
    f << "[" << t.getBatchSize() << " " << t.getDepth() << " " << t.getHeight() << " " << t.getWidth() << "]" << endl;
    f << "[";
    if (!t.empty())
    {

        for (size_t i = 0; i < t.getBatchSize(); ++i)
        {
            f << "[";
            for (size_t j = 0; j < t.getDepth(); ++j)
            {
                f << "[";
                for (size_t k = 0; k < t.getHeight(); ++k)
                {
                    f << "[";
                    for (size_t l = 0; l < t.getWidth(); ++l, ++ind)
                    {
                        f << t[ind] << " ";
                    }
                    f << "]" << endl;
                }
                f << "]" << endl;
            }
            f << "]" << endl;
        }
    }
    f << "]" << endl;
}

void print_tensor(const TensorGPUHelper& t, const filesystem::path& s)
{
    const Tensor tt = t;
    print_tensor(tt, s);
}

void checkTensors(const vector<pair<Name, Name>>& tensors, MemoryManagerGPU& m, dtype eps)
{
    for (const auto& it : tensors)
    {
        Tensor t = m[it.first];
        Tensor golden = m[it.second];
        EXPECT_EQ(t.getShape(), golden.getShape());
        for (size_t i = 0; i < golden.size(); ++i)
        {
            CHECK_NEAR(t[i], golden[i], eps);
        }
        cout << "Tensor '" << it.first << "' checked" << endl;
    }
}

void checkTensors(const vector<pair<Name, Name>>& tensors, MemoryManager& m, dtype eps)
{
    for (const auto& it : tensors)
    {
        Tensor& t = m[it.first];
        Tensor& golden = m[it.second];
        EXPECT_EQ(t.getShape(), golden.getShape());
        for (size_t i = 0; i < golden.size(); ++i)
        {
            CHECK_NEAR(t[i], golden[i], eps);
        }
        cout << "Tensor '" << it.first << "' checked" << endl;
    }
}

decltype(NetworkParameters::mCallbackGPU) createLayerTracerCallbackGPU(Workflow* work)
{
    return [work](BasicLayer* l, MemoryManagerGPU&, NetworkParameters::CallbackPlace p)
    {
        if (p == NetworkParameters::CallbackPlace::After_Forward)
        {
            Common::checkOpenCLStatus(work->getGpuCommandQueue().finish(), l->getName(), "forward failed");
            cout << l->getName() << " forward finished" << endl;
        }
        else if (p == NetworkParameters::CallbackPlace::After_Backward)
        {
            Common::checkOpenCLStatus(work->getGpuCommandQueue().finish(), l->getName(), "backward failed");
            cout << l->getName() << " backward finished" << endl;
        }
    };
}

size_t get_size_of_trainable_params(const Workflow& network)
{
    size_t amount_of_trainable_parameters = 0U;
    for (const auto& name : network.getTrainableParameterNames())
    {
        amount_of_trainable_parameters += network.getMemoryManager()[name].getShape().total_size();
    }
    return amount_of_trainable_parameters;
}

size_t get_size_of_trainable_params_mixed_precision(const Workflow& network)
{
    size_t amount_of_trainable_parameters = 0U;
    for (const auto& name : network.getTrainableParameterNames())
    {
        if (network.getMemoryManager().tensorExists(name))
            amount_of_trainable_parameters += network.getMemoryManager()[name].getShape().total_size();
        else if (network.getMemoryManager<MemoryManagerFP16>().tensorExists(name))
            amount_of_trainable_parameters += network.getMemoryManager<MemoryManagerFP16>()[name].getShape().total_size();
    }
    return amount_of_trainable_parameters;
}

string read_file_to_string(const filesystem::path& path)
{
    stringstream s;

    try
    {
        ifstream in(path);
        s << in.rdbuf();
    }
    catch (exception&)
    {
    }

    return s.str();
}

size_t countLines(const filesystem::path& p)
{
    size_t number_of_lines = 0;
    string line;
    ifstream myfile(p);

    while (getline(myfile, line))
    {
        ++number_of_lines;
    }

    return number_of_lines;
}

#if !defined(_WIN32)
long getPeakOfMemory()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

Timestamp getCPUTimestamp()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return { ru.ru_utime, ru.ru_stime };
}

std::pair<double, double> getElapsedTime(Timestamp begin, Timestamp end)
{
    const auto userDiff = (end.user.tv_sec - begin.user.tv_sec) + (end.user.tv_usec - begin.user.tv_usec) / 1e6;
    const auto systemDiff = end.system.tv_sec - begin.system.tv_sec + (end.system.tv_usec - begin.system.tv_usec) / 1e6;
    return make_pair(userDiff, systemDiff);
}

#endif

namespace toynet
{
unique_ptr<Workflow> buildUnique(const size_t width,
                                 const size_t height,
                                 const size_t channels,
                                 const size_t batch_size,
                                 const size_t classes,
                                 const size_t hidden_size,
                                 const Nonlinearity nl_type,
                                 const bool usePool,
                                 const optional_path& weights_path,
                                 const std::unordered_map<raul::Name, raul::ScalingStrategy>& scaling)
{
    auto network = make_unique<Workflow>(CompressionMode::NONE, CalculationMode::DETERMINISTIC, usePool ? AllocationMode::POOL : AllocationMode::STANDARD);

    network->add<DataLayer>("data", DataParams{ { "data", "labels" }, channels, height, width, classes });
    network->add<ReshapeLayer>("reshape", ViewParams{ "data", "datar", 1, 1, -1 });
    network->add<LinearLayer>("fc_in", LinearParams{ { "datar" }, { "fc_in" }, hidden_size });
    switch (nl_type)
    {
        case Nonlinearity::relu:
            network->add<ReLUActivation>("nl", BasicParams{ { "fc_in" }, { "nl" } });
            break;
        case Nonlinearity::sigmoid:
            network->add<SigmoidActivation>("nl", BasicParams{ { "fc_in" }, { "nl" } });
            break;
        case Nonlinearity::hsigmoid:
            network->add<HSigmoidActivation>("nl", HSigmoidActivationParams{ { "fc_in" }, { "nl" } });
            break;
        case Nonlinearity::swish:
            network->add<SwishActivation>("nl", BasicParams{ { "fc_in" }, { "nl" } });
            break;
        case Nonlinearity::hswish:
            network->add<HSwishActivation>("nl", HSwishActivationParams{ { "fc_in" }, { "nl" } });
            break;
            // default: Do nothing
    }
    network->add<LinearLayer>("fc_out", LinearParams{ { "nl" }, { "fc_out" }, classes });
    network->add<LogSoftMaxActivation>("softmax", BasicParamsWithDim{ { "fc_out" }, { "softmax" } });
    network->add<NLLLoss>("loss", LossParams{ { "softmax", "labels" }, { "loss" }, "batch_mean" });

    for (auto& [k, v] : scaling)
    {
        network->setScaling(k, v);
    }

    network->preparePipelines();
    network->setBatchSize(batch_size);
    network->prepareMemoryForTraining();

    if (weights_path)
    {
        DataLoader dataLoader;
        auto& memory_manager = network->getMemoryManager();
        memory_manager["fc_in::Weights"] = dataLoader.loadData(*weights_path / "init_fc1.weight.data", hidden_size, width * height);
        memory_manager["fc_in::Biases"] = dataLoader.loadData(*weights_path / "init_fc1.bias.data", 1, hidden_size);
        memory_manager["fc_out::Weights"] = dataLoader.loadData(*weights_path / "init_fc2.weight.data", classes, hidden_size);
        memory_manager["fc_out::Biases"] = dataLoader.loadData(*weights_path / "init_fc2.bias.data", 1, classes);

        Common::transpose(memory_manager["fc_in::Weights"], hidden_size);
        Common::transpose(memory_manager["fc_out::Weights"], classes);
    }

    return network;
}
} // toynet

} // UT::tools

namespace raul::helpers
{

using namespace std;

ProfilerGuard::ProfilerGuard(const string& testName)
    : mWasEnabled(false)
    , mTestName(testName)
{
    if (Profiler::getInstance().isDisabled())
    {
        return;
    }

    mWasEnabled = Profiler::getInstance().getState();
    if (Profiler::getInstance().useJsonFormat())
    {
        Profiler::getInstance().increasePrefix(mTestName);
    }
    else
    {
        *Profiler::getInstance().getOstream() << mTestName + "\n";
    }
    Profiler::getInstance().enableProfiler();
}

ProfilerGuard::~ProfilerGuard()
{
    if (!mWasEnabled && !Profiler::getInstance().isDisabled())
    {
        if (Profiler::getInstance().useJsonFormat())
        {
            Profiler::getInstance().clearPrefix();
        }
        Profiler::getInstance().disableProfiler();
    }
}

void ListenerHelper::addProcesser(const ListenerHelper::ProcessData& checkData)
{
    mProcessers.push_back(checkData);
}

void Checker::check(const Workflow& work) const
{
    const auto& memory_manager = work.getMemoryManager();

    if (mProcessers.empty())
    {
        THROW_NONAME("Checker", "empty processors");
    }

    for (const auto& checkData : mProcessers)
    {
        const auto& tensor = memory_manager[checkData.first];
        EXPECT_EQ(tensor.size(), checkData.second.size());

        for (size_t i = 0; i < tensor.size(); ++i)
        {
            ASSERT_TRUE(UT::tools::expect_near_relative(tensor[i], checkData.second[i], mEps)) << "at " << i << ", expected: " << checkData.second[i] << ", got: " << tensor[i];
        }
    }
}

void FillerBeforeBackward::BeforeBackward(Workflow& work)
{
    auto& memory_manager = work.getMemoryManager();

    if (mProcessers.empty())
    {
        THROW_NONAME("FillerBeforeBackward", "empty processors");
    }

    for (const auto& checkData : mProcessers)
    {
        auto& tensor = memory_manager[checkData.first];
        EXPECT_EQ(tensor.size(), checkData.second.size());

        tensor = TORANGE(checkData.second);
    }
}

} // namespace UT::tools