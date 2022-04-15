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

#include <training/common/Common.h>
#include <training/network/Workflow.h>
#include <training/network/Layers.h>

#include <training/optimizers/AdamW.h>
#include <training/optimizers/SGD.h>
#include <training/optimizers/schedulers/strategies/CosineAnnealing.h>

#include <fstream>
#include <filesystem>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace
{
    using namespace std;
    using namespace raul;

    filesystem::path getRaulAssetsDir()
    {
        auto current_path = filesystem::current_path();
        const char* env_raul_test_assets_dir = getenv("RAUL_ASSETS");
        auto relative_path = (env_raul_test_assets_dir == nullptr) ? string("assets") : string(env_raul_test_assets_dir);
        current_path /= relative_path;

        return current_path.make_preferred();
    }

    std::string printMemory(long value_kB)
    {
        std::stringstream os;
        os << value_kB << " kB";
        if (value_kB > 1024 * 1024)
        {
            os << " (" << value_kB / (1024 * 1024) << " GB)";
        }
        else if (value_kB > 1024)
        {
            os << " (" << value_kB / 1024 << " MB)";
        }
        return os.str();
    }

#if !defined(_WIN32)
    long getPeakOfMemory()
    {
        rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return ru.ru_maxrss;
    }
#endif

    template<typename T>
    void print_tensor(const string& fname, const T& t)
    {
        ofstream f(fname);
        f << "[" << t.getBatchSize() << " " << t.getDepth() << " " << t.getHeight() << " " << t.getWidth() << "]" << endl;
        f << "[";
        if (!t.empty())
        {
            size_t ind = 0;
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
                            f << toFloat32(t[ind]) << " ";
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

    template<typename MM>
    std::function<void(BasicLayer*, MM&, NetworkParameters::CallbackPlace)> createCallback(const string& path)
    {
        return [path](raul::BasicLayer* layer, MM& mm, NetworkParameters::CallbackPlace p) {
            /*if (p != NetworkParameters::CallbackPlace::Before_Backward)
            {
                return;
            }*/
            if (!filesystem::exists(path))
            {
                filesystem::create_directories(path);
            }

            if (layer->getName().str().find("unrolled") != string::npos)
            {
                return;
            }
            

            static size_t ii = 0;
            if (p == NetworkParameters::CallbackPlace::After_Forward)
            {
                for (const auto& tname : layer->getOutputs())
                {
                    if (tname.str().find("rnn") != string::npos)
                    {
                        continue;
                    }
                    if (tname.str().find("initial_h[") != string::npos)
                    {
                        continue;
                    }
                    string fname = path + "/" + to_string(ii) + "_" + tname + "(" + layer->getName() + ")" + ".txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname];
                    print_tensor(fname, t);
                    ++ii;
                }
            }
            if (p == NetworkParameters::CallbackPlace::After_Backward)
            {
                for (const auto& tname : layer->getInputs())
                {
                    if (tname.str().find("rnn") != string::npos)
                    {
                        continue;
                    }
                    if (tname.str().find("initial_h[") != string::npos)
                    {
                        continue;
                    }
                    if (!mm.tensorExists(tname.grad()))
                    {
                        continue;
                    }
                    string fname = path + "/" + to_string(ii) + "_" + tname + "_grad(" + layer->getName() + ")" + ".txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname.grad()];
                    print_tensor(fname, t);
                    ++ii;
                }
            }
            /*string nn = layer->getName();
            if (nn.find("attention") == string::npos)
            {
                return;
            }
            if (p == NetworkParameters::CallbackPlace::Before_Backward)
            {
                
                for (const auto& tname : layer->getOutputs())
                {
                    string fname = path + "/" + to_string(ii) + "_" + tname + "_grad.txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname.grad()];
                    print_tensor(fname, t);
                }
                for (const auto& tname : layer->getInputs())
                {
                    {
                        string fname = path + "/" + to_string(ii) + "_" + tname + ".txt";
                        Common::replaceAll(fname, "::", "_");
                        auto& t = mm[tname];
                        print_tensor(fname, t);
                    }
                    {
                        string fname = path + "/" + to_string(ii) + "_" + tname + "_grad.txt";
                        Common::replaceAll(fname, "::", "_");
                        auto& t = mm[tname.grad()];
                        print_tensor(fname, t);
                    }
                }
                {
                    string tname = layer->getName() / "Weights";
                    string fname = path + "/" + to_string(ii) + "_" + tname + ".txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname];
                    print_tensor(fname, t);
                }
                {
                    string tname = layer->getName() / "Biases";
                    string fname = path + "/" + to_string(ii) + "_" + tname + ".txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname];
                    print_tensor(fname, t);
                }
            }
            if (p == NetworkParameters::CallbackPlace::After_Backward)
            {
                for (const auto& tname : layer->getInputs())
                {
                    {
                        string fname = path + "/" + to_string(ii) + "_" + tname + "_grad_after.txt";
                        Common::replaceAll(fname, "::", "_");
                        auto& t = mm[tname.grad()];
                        print_tensor(fname, t);
                    }
                }
                {
                    auto tname = layer->getName() / "Weights";
                    string fname = path + "/" + to_string(ii) + "_" + tname + "_grad.txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname.grad()];
                    print_tensor(fname, t);
                }
                {
                    auto tname = layer->getName() / "Biases";
                    string fname = path + "/" + to_string(ii) + "_" + tname + "_grad.txt";
                    Common::replaceAll(fname, "::", "_");
                    auto& t = mm[tname.grad()];
                    print_tensor(fname, t);
                }
            }*/
            
        };
    }

    template<typename MM>
    void loadData(const filesystem::path& path, typename MM::tensor& tensor) 
    {
        if (!filesystem::exists(path))
        {
            throw;
        }
        ifstream f(path);
        istream_iterator<float> begin(f), end;
        std::transform(begin, end, tensor.begin(), 
            [](const auto& val_fp32) { return toFloat<typename MM::type>(val_fp32); });
    }

    void loadData(const filesystem::path& path, vector<float>& v) 
    {
        if (!filesystem::exists(path))
        {
            throw;
        }
        ifstream f(path);
        istream_iterator<float> begin(f), end;
        std::copy(begin, end, back_inserter(v));
    }

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
        work.add<TensorLayer>("initial_h", TensorParams{ { initial_h }, { BS(), 1, 1, GRU_SIZE }, Workflow::Usage::ForwardAndBackward });
        GRULayer("gru", GRUParams{ { input2, "initial_h" }, { "gru_out", "new_hidden" } }, work.getNetworkParameters());
        work.add<LinearLayer>("linear2", LinearParams{ "gru_out", "linear2_out", LINEAR_SIZE });

        work.add<DataLayer>("input1", DataParams{ { input1 }, 1, 1, INPUT1_SIZE });
        work.add<LinearLayer>("Gemm_10", LinearParams{ input1, "linear1_out", LINEAR_SIZE });

        work.add<ElementWiseSumLayer>("add", ElementWiseLayerParams{ {"linear2_out", "linear1_out"}, "sum" });
        work.add<LeakyReLUActivation>("relu", LeakyReLUParams{ "sum", "relu_out", RELU_ALPHA });
        work.add<LinearLayer>("linear3", LinearParams{ "relu_out", "linear3_out", RELU_LINEAR_SIZE });

        work.add<SoftMaxActivation>("Softmax_19", BasicParamsWithDim{ { "linear3_out" }, { "softmax" }, "width" });

        work.add<ElementWiseMulLayer>("mul", ElementWiseLayerParams{ { "linear2_out", "softmax" }, "mul"});
        work.add<ReduceSumLayer>("reduce", BasicParamsWithDim{ { "mul" }, { "mul_reduced" } });
        work.add<LinearLayer>("Gemm_23", LinearParams{ "mul_reduced", "linear4_out", FINAL_LINEAR_SIZE });

        work.add<SigmoidActivation>("sigmoid", BasicParamsWithDim{ { "linear4_out" }, { out } });
    }

    void create_yannet(Workflow& work, const Names& inputs, const Name& out, size_t hiddenSize, size_t query_size, size_t key_size, bool fusedGRU) 
    {
        size_t KEY_LENGTH = 48;

        auto [query, key, initial_h] = array<Name, 3>{inputs[0], inputs[1], inputs[2]};

        work.add<DataLayer>("key", DataParams{ { key }, 1, KEY_LENGTH, key_size });
        work.add<DataLayer>("query", DataParams{ { query }, 1, 1, query_size });
        work.add<TensorLayer>("initial_h", TensorParams{ { initial_h }, { BS(), 1, 1, hiddenSize }, 0, Workflow::Usage::ForwardAndBackward });
        GRULayer("rnn", GRUParams{ { key, "initial_h" }, { "gru_out", "new_hidden" }, true, false, fusedGRU }, work.getNetworkParameters());
        
        // Attention
        work.add<LinearLayer>(Name("att") / "key_layer", LinearParams{ "gru_out", "key_layer_out", hiddenSize });
        work.add<LinearLayer>(Name("att") / "query_layer", LinearParams{ query, "query_layer_out", hiddenSize });

        work.add<ElementWiseSumLayer>("add", ElementWiseLayerParams{ {"query_layer_out", "key_layer_out"}, "h" });
        // wait for final implementation of CastLayer
        //if (work.getExecutionTarget() == ExecutionTarget::CPUFP16)
        //{
            //work.add<CastLayer>("to_fp32", CastParams{ "h", "h_fp32", CastParams::Type::dtype });
            //work.add<LayerNorm2DLayer>(Name("att") / "norm", LayerNormParams{ "h_fp32", "h_norm_fp32", LayerExecutionTarget::CPU });
            //work.add<CastLayer>("to_half", CastParams{ "h_norm_fp32", "h_norm", CastParams::Type::half });
        //}
        //else
        //{
            work.add<LayerNorm2DLayer>(Name("att") / "norm", LayerNormParams{ "h", "h_norm" });
        //}
        work.add<LeakyReLUActivation>("relu", LeakyReLUParams{ "h_norm", "h_relu" });
        work.add<LinearLayer>(Name("att") / "attention_layer", LinearParams{ "h_relu", "attn_layer_out", 1 });
        work.add<SoftMaxActivation>("softmax", BasicParamsWithDim{ { "attn_layer_out" }, { "h_softmax" }, "height" });

        work.add<ElementWiseMulLayer>("mul", ElementWiseLayerParams{ { "h_softmax", "key_layer_out" }, "mul"});
        work.add<ReduceSumLayer>("reduce", BasicParamsWithDim{ { "mul" }, { "mul_reduced" }, "height" });

        work.add<LinearLayer>("fc", LinearParams{ "mul_reduced", "y", 1 });
        work.add<SigmoidActivation>("sigmoid", BasicParamsWithDim{ { "y" }, { out } });
    }
}

void simple_sample()
{
    using namespace raul;
    using namespace std;

    Workflow work(CompressionMode::NONE, CalculationMode::DETERMINISTIC, AllocationMode::POOL, ExecutionTarget::CPUFP16);
    auto& memory_manager = work.getMemoryManager<MemoryManagerFP16>();

    size_t batch_size = 2;
    size_t hidden_size = 6;

    create_small_model(work, { "input1", "input2", "initial_h" }, "out");

    map<string, vector<float>> values;
    values["GRU_7.W"] = vector<float>((3 * hidden_size) * 34, 1.f);
    values["GRU_7.R"] = vector<float>((3 * hidden_size) * hidden_size, 1.f);
    values["GRU_7.B"] = vector<float>((6 * hidden_size), 1.f);

    values["MatMul_11.B"] = vector<float>(6 * 6, 1.f);
    values["Add_12.B"] = vector<float>(6, 1.f);

    values["Gemm_10.B"] = vector<float>(6 * 12, 1.f);
    values["Gemm_10.C"] = vector<float>(6, 1.f);

    values["MatMul_16.B"] = vector<float>(6, 1.f);
    values["Add_17.B"] = vector<float>(1, 1.f);

    values["Gemm_23.B"] = vector<float>(6, 1.f);
    values["Gemm_23.C"] = vector<float>(1, 1.f);

    // GRU bias needs to be dealt separately: onnx stores input and hidden bias concatenated, we use two separate tensors
    values["GRU_7.B_1"] = vector<float>(values["GRU_7.B"].begin(), values["GRU_7.B"].begin() + 3 * hidden_size);
    values["GRU_7.B_2"] = vector<float>(values["GRU_7.B"].begin() + 3 * hidden_size, values["GRU_7.B"].end());

    map<string, Name> onnxToTrainingParams = {
        { "GRU_7.W", Name("gru") / "cell" / "linear_ih" / "Weights" }, 
        { "GRU_7.R", Name("gru") / "cell" / "linear_hh" / "Weights" }, 
        { "GRU_7.B_1", Name("gru") / "cell" / "linear_ih" / "Biases" }, 
        { "GRU_7.B_2", Name("gru") / "cell" / "linear_hh" / "Biases" }, 

        { "MatMul_11.B", Name("linear2") / "Weights" }, 
        { "Add_12.B", Name("linear2") / "Biases" },

        { "Gemm_10.B", Name("Gemm_10") / "Weights" },   
        { "Gemm_10.C", Name("Gemm_10") / "Biases" },

        { "MatMul_11.B", Name("linear3") / "Weights" }, 
        { "Add_12.B", Name("linear3") / "Biases" },

        { "Gemm_10.B", Name("Gemm_23") / "Weights" },   
        { "Gemm_10.C", Name("Gemm_23") / "Biases" },
    };

    work.preparePipelines();        
    work.setBatchSize(batch_size);  
    work.prepareMemoryForTraining();

    // Trainable parameters and shapes
    auto trainableNames = work.getTrainableParameterNames();
    for (const auto& t : trainableNames)
    {
        auto shape = memory_manager[t].getShape();
        cout << t << " [" << shape[1] << ", " << shape[2] << ", " << shape[3] << "]" << endl;
    }

    for (const auto& p : onnxToTrainingParams)
    {
        auto& v = values[p.first];
        auto tensorName = p.second;
        auto& tensor = memory_manager[tensorName];
        std::transform(v.begin(), v.end(), tensor.begin(), [](const auto& val_fp32) { return toFloat16(val_fp32); });
    }

}

template<typename MM>
struct Helper
{
    const raul::ExecutionTarget target = raul::ExecutionTarget::CPUFP16;
    const string outPath = "out_16";
};

template<>
struct Helper<raul::MemoryManager>
{
    const raul::ExecutionTarget target = raul::ExecutionTarget::CPU;
    const string outPath = "out_32";
};


template<typename MM>
vector<float> getReferenceData(const filesystem::path& path)
{
    vector<float> v;
    loadData(path / "ideal_loss.txt", v);
    return v;
}

template<>
vector<float> getReferenceData<MemoryManagerFP16>(const filesystem::path&)
{
    return vector<float>();
}

template<typename MM>
optimizers::Scheduler::LrScheduler createScheduler(size_t size, float warmupPercentage, float baseLR, float weightDecay)
{
    return optimizers::Scheduler::LrScheduler{ std::make_unique<optimizers::Scheduler::Strategies::CosineAnnealing>(size, 1.f, 0.f, warmupPercentage), std::make_unique<raul::optimizers::AdamW>(baseLR, 0.9f, 0.999f, 1e-8f, weightDecay) }; 
}

template<typename MM>
void trainYannet(bool fusedGru, CompressionMode comp, AllocationMode allocMode, size_t batchSize, const string& experimentName)
{
    size_t query_size = 12;
    size_t key_size = 34;
    size_t hidden_size = 32;

    using namespace raul;
    using namespace std;

    Workflow work(comp, CalculationMode::DETERMINISTIC, allocMode, Helper<MM>().target);
    auto& memory_manager = work.getMemoryManager<MM>();
    auto& memory_managerFP32 = work.getMemoryManager();

    size_t BATCH_SIZE = 2;
    size_t numEpoch = 50;
    size_t samplesCount = batchSize;
    float baseLearningRate = 0.01f;
    [[maybe_unused]]float weightDecay = 0.01f;
    float warmupPercentage = 0.1f;

    vector<float> idealLoss;
    if (batchSize == BATCH_SIZE)
    {
        idealLoss = getReferenceData<MM>(getRaulAssetsDir() / "wake_up" / experimentName);
    }
    
	create_yannet(work, { "query", "key", "initial_h" }, "out", hidden_size, query_size, key_size, fusedGru);
    work.add<DataLayer>("targets", DataParams{ { "targets" }, 1, 1, 1 });
    work.add<BinaryCrossEntropyLoss>("bce", LossParams{ { "out", "targets" }, {"loss"} });

#if !defined(_WIN32)
    cout << "Memory after workflow creation: " << printMemory(getPeakOfMemory()) << endl;
#endif

    work.preparePipelines();        
    work.setBatchSize(batchSize);  
    work.prepareMemoryForTraining();

#if !defined(_WIN32)
    cout << "Memory after memory preparation: " << printMemory(getPeakOfMemory()) << endl;
#endif

    auto trainableNames = work.getTrainableParameterNames();
    for (const auto& t : trainableNames)
    {
        if (memory_manager.tensorExists(t))
        {
            auto shape = memory_manager[t].getShape();
            cout << t << " [" << shape[1] << ", " << shape[2] << ", " << shape[3] << "]" << endl;
        }
        else 
        {
            auto shape = memory_managerFP32[t].getShape();
            cout << t << " [" << shape[1] << ", " << shape[2] << ", " << shape[3] << "]" << endl;
        }
    }

    map<string, string> pytorchOnnxMap = { 
        { "rnn.weight_ih_l0", "GRU_7.W" },
        { "rnn.weight_hh_l0", "GRU_7.R" },
        { "rnn.bias_ih_l0", "GRU_7.B_1_" },
        { "rnn.bias_hh_l0", "GRU_7.B_2_" },
        { "att.query_layer.weight", "Gemm_10.B" },
        { "att.query_layer.bias", "Gemm_10.C" },
        { "att.key_layer.weight", "MatMul_11.B" },
        { "att.key_layer.bias", "Add_12.B" },
        { "att.norm.weight", "Mul_25.B" },
        { "att.norm.bias", "Add_26.B" },
        { "att.attention_layer.weight", "MatMul_28.B" },
        { "att.attention_layer.bias", "Add_29.B" },
        { "fc.weight", "Gemm_35.B" },
        { "fc.bias", "Gemm_35.C" },
    };

    map<string, vector<float>> values;

    // load weights from files to vectors
    for (const auto& p : pytorchOnnxMap)
    {
        loadData(getRaulAssetsDir() / "wake_up" / experimentName / "weights" / (p.first + ".txt"), values[p.second]);
    }
    std::copy(values["GRU_7.B_1_"].begin(), values["GRU_7.B_1_"].end(), back_inserter(values["GRU_7.B"]));
    std::copy(values["GRU_7.B_2_"].begin(), values["GRU_7.B_2_"].end(), back_inserter(values["GRU_7.B"]));
    
    // GRU bias needs to be dealt separately: onnx stores input and hidden bias concatenated, we use two separate tensors
    values["GRU_7.B_1"] = vector<float>(values["GRU_7.B"].begin(), values["GRU_7.B"].begin() + 3 * hidden_size);
    values["GRU_7.B_2"] = vector<float>(values["GRU_7.B"].begin() + 3 * hidden_size, values["GRU_7.B"].end());

    map<string, Name> onnxToTrainingParams = {
        { "GRU_7.W", Name("rnn") / "cell" / "linear_ih" / "Weights" }, 
        { "GRU_7.R", Name("rnn") / "cell" / "linear_hh" / "Weights" }, 
        { "GRU_7.B_1", Name("rnn") / "cell" / "linear_ih" / "Biases" }, 
        { "GRU_7.B_2", Name("rnn") / "cell" / "linear_hh" / "Biases" }, 

        { "Mul_25.B", Name("att") / "norm" / "Weights" }, 
        { "Add_26.B", Name("att") / "norm" / "Biases" },

        { "Gemm_10.B", Name("att") / "query_layer" / "Weights" },   
        { "Gemm_10.C", Name("att") / "query_layer" / "Biases" },

        { "MatMul_11.B", Name("att") / "key_layer" / "Weights" }, 
        { "Add_12.B", Name("att") / "key_layer" / "Biases" },

        { "MatMul_28.B", Name("att") / "attention_layer" / "Weights" }, 
        { "Add_29.B", Name("att") / "attention_layer" / "Biases" },

        { "Gemm_35.B", Name("fc") / "Weights" },   
        { "Gemm_35.C", Name("fc") / "Biases" },
    };

    for (const auto& p : onnxToTrainingParams)
    {
        auto& v = values[p.first];
        auto tensorName = p.second;
        if (memory_manager.tensorExists(tensorName))
        {
            auto& tensor = memory_manager[tensorName];
            std::transform(v.begin(), v.end(), tensor.begin(), [](const auto& val_fp32) { return toFloat<typename MM::type>(val_fp32); });
            cout << "";
        }
        else
        {
            auto& tensor = memory_managerFP32[tensorName];
            std::transform(v.begin(), v.end(), tensor.begin(), [](const auto& val_fp32) { return toFloat<typename MM::type>(val_fp32); });
        }
    }

    // experiment1 contains data exported from pytorch, where key_layer is represented as LinearLayer (Gemm), so it means y = x * B_t + C
    // experiment2 - from onnx, where key_layer is represented as MatMul + Add, so it means y = x * B + C
    if (experimentName != "experiment1")
    {
        Common::transpose(memory_manager[Name("att") / "key_layer" / "Weights"], hidden_size);
    }

    size_t epochSize = samplesCount / batchSize;

    auto scheduler = createScheduler<MM>(numEpoch * epochSize, warmupPercentage, baseLearningRate, weightDecay);

    size_t iter = 0;

    for (size_t epoch = 0; epoch < 10; ++epoch)
    {
        cout << "Epoch " << epoch + 1 << endl;
        for (size_t batch = 0; batch < epochSize; ++batch, ++iter)
        {
            // load batch data - do not load from files each time in real scenario
            if (batchSize == BATCH_SIZE)
            {
                loadData<MM>(getRaulAssetsDir() / "wake_up" / "query.txt", memory_manager["query"]);
                loadData<MM>(getRaulAssetsDir() / "wake_up" / "key.txt", memory_manager["key"]);
                loadData<MM>(getRaulAssetsDir() / "wake_up" / "targets.txt", memory_manager["targets"]);
            }

            /*if (iter == 0)
            {
                work.getNetworkParameters().setCallback(createCallback<MM>("out/" + Helper<MM>().outPath + "/" + to_string(iter)));
            }
            else
            {
                work.getNetworkParameters().mCallback = nullptr;
                work.getNetworkParameters().mCallbackFP16 = nullptr;
            }*/

            work.forwardPassTraining();
            float lossValue = toFloat32(memory_manager["loss"][0]);
            cout << "Loss = " << lossValue;

            if (iter < idealLoss.size())
            {
                if (fabs(idealLoss[iter] - lossValue) < 1e-4)
                {
                    cout << " MATCH!";
                }
                else
                {
                    cout << " FAIL! (ideal value: " << idealLoss[iter] << ")";
                }
            }
            cout << endl;

            work.backwardPassTraining();

            scheduler.step();
            cout << "lr = " << scheduler.getLearningRate() << endl << endl;

            for (const auto& t : trainableNames)
            {
                //string nn = t.Param.getName();
                //Common::replaceAll(nn, "::", "_");
                /*if (iter)
                {
                    print_tensor("grad/" + nn + "_grad.txt", t.Gradient);
                }*/
                if (memory_manager.tensorExists(t))
                {
                    scheduler(memory_manager, memory_manager[t], memory_manager[t.grad()]);
                }
                else
                {
                    scheduler(memory_managerFP32, memory_managerFP32[t], memory_managerFP32[t.grad()]);
                }
                /*if (epoch == 0 && batch == 0)
                {
                    print_tensor("grad/" + nn + "_updated.txt", t.Param);
                }*/
            }
        }
    }
#if !defined(_WIN32)
    cout << "Memory peak during training: " << printMemory(getPeakOfMemory()) << endl;
#endif
}

int main(int argc, char *argv[])
{
    using namespace raul;
    using namespace std;

    cout << "SSD Wake-up Demo" << endl;

#if !defined(_WIN32)
    cout << "Memory at application start: " << printMemory(getPeakOfMemory()) << endl;
#endif

    bool fp16 = true;
    bool fusedGru = false;
    AllocationMode allocMode = AllocationMode::STANDARD;
    CompressionMode comp = CompressionMode::NONE;
    size_t batch = 2;

    string experimentName = "experiment1";

    for (int i = 1; i < argc; ++i)
    {
        string s = argv[i];

        if (s == "--fp32")
        {
            fp16 = false;
        }
        if (s == "--fused")
        {
            fusedGru = true;
        }
        if (s == "--c16")
        {
            comp = CompressionMode::FP16;
        }
        if (s == "--c8")
        {
            comp = CompressionMode::INT8;
        }
        if (s == "--pool")
        {
            allocMode = AllocationMode::STANDARD;
        }
        if (raul::Common::startsWith(s, "--batch"))
        {
            batch = stoull(s.substr(8));
        }
    }  

    cout << "Precision: " << (fp16 ? "FP16" : "FP32") << endl;
    cout << "GRU Fusion: " << (fusedGru ? "yes" : "no") << endl;
    cout << "Compression: " << (comp == CompressionMode::FP16 ? "FP16" : (comp == CompressionMode::INT8 ? "INT8" : "NONE")) << endl;
    cout << "Use pool: " << (allocMode == AllocationMode::STANDARD ? "no" : "yes") << endl;

    if (fp16)
    {
        trainYannet<MemoryManagerFP16>(fusedGru, comp, allocMode, batch, experimentName);
    }
    else
    {
        trainYannet<MemoryManager>(fusedGru, comp, allocMode, batch, experimentName);
    }

	return 0;
}
