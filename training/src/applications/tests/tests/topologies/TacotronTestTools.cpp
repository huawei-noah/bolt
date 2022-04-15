// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "TacotronTestTools.h"
#include <tests/tools/TestTools.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>

#include <tests/tools/TestTools.h>

#include <training/initializers/ConstantInitializer.h>
#include <training/layers/basic/DataLayer.h>
#include <training/layers/basic/ElementWiseSumLayer.h>
#include <training/layers/basic/SlicerLayer.h>
#include <training/layers/basic/SplitterLayer.h>
#include <training/layers/basic/trainable/LinearLayer.h>
#include <training/layers/composite/BahdanauMonotonicAttentionLayer.h>
#include <training/layers/composite/DynamicConvolutionAttentionLayer.h>
#include <training/layers/composite/TacotronModel.h>
#include <training/layers/composite/rnn/LSTMCellLayer.h>
#include <training/layers/composite/tacotron/MaskedCrossEntropy.h>
#include <training/layers/composite/tacotron/MaskedLoss.h>
#include <training/layers/composite/tacotron/PostNet.h>
#include <training/layers/composite/tacotron/TacotronDecoderCell.h>
#include <training/layers/composite/tacotron/TacotronDecoderRNN.h>
#include <training/layers/composite/tacotron/TacotronLoss.h>

#include <training/network/Layers.h>
#include <training/optimizers/Adam.h>
#include <training/optimizers/SGD.h>
#include <training/optimizers/schedulers/LrScheduler.h>
#include <training/optimizers/schedulers/strategies/ClipLower.h>
#include <training/optimizers/schedulers/strategies/ClipUpper.h>
#include <training/optimizers/schedulers/strategies/Exponential.h>
#include <training/optimizers/schedulers/strategies/StepOffset.h>
#include <training/optimizers/schedulers/strategies/WarmUp.h>
#include <training/postprocessing/GradientClipping.h>

#include <training/api/API.h>
#include <training/common/Train.h>
#include <training/tools/Dataset.h>

#include "TacotronTrainingParams.h"

namespace UT
{
using namespace raul;
using namespace std;

namespace
{
void saveTensor(const Tensor& t, ofstream& f)
{
    copy(t.begin(), t.end(), ostream_iterator<dtype>(f, "\n"));
}
}

dtype TensorDiff(const Tensor& a, const Tensor& b)
{
    if (a.size() != b.size() || a.empty() || b.empty())
    {
        THROW_NONAME("TacotronTestTools", "empty tensor");
    }
    dtype sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

dtype TensorNorm(const Tensor& a)
{
    if (a.empty())
    {
        THROW_NONAME("TacotronTestTools", "empty tensor");
    }
    dtype sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * a[i];
    }
    return std::sqrt(sum);
}

map<string, float> loadNamedValues(const filesystem::path& path, const string& name, const TacotronParams& params, bool spbModel)
{
    map<string, float> result;
    ifstream f(path);

    map<Name, string> mapTaco;

    tacotronParamNamesMaps(name, params, mapTaco, mapTaco, false, spbModel);
    map<string, string> mapTmp;
    for (auto& i : mapTaco)
    {
        mapTmp[i.second] = i.first.str();
    }
    mapTmp["inference.speaker_embedding"] = (Name(name) / "speaker_embedding").str();
    mapTmp["inference.fixed_spk_embedding"] = (Name(name) / "speaker_embedding").str();

    std::string s;
    while (f >> s)
    {
        float v;
        f >> v;

        Common::replaceAll(s, "/", ".");
        Common::replaceAll(s, ":0", "");
        Common::replaceAll(s, "Tacotron_model.", "");

        auto it = mapTmp.find(s);
        if (it != mapTmp.end())
        {
            result.emplace(it->second, v);
        }
    }
    return result;
}

void tacotronParamNamesMaps(const Name& name, const TacotronParams& params, map<Name, string>& oplain, map<Name, string>& otransposed, bool loadOptimizerParams, bool spbModel)
{
    map<Name, string> plain;
    map<Name, string> transposed;

    oplain.clear();
    otransposed.clear();

    // PostNet convolutions
    for (size_t i = 1; i <= params.postnetKernelSize.size(); ++i)
    {
        auto convName = name / "postnet_convolutions" / "conv1d[" + to_string(i) + "]";
        plain[convName / "conv1d" / "Weights"] = "inference.postnet_convolutions.conv_layer_" + to_string(i) + "_postnet_convolutions.conv1d.kernel";
        plain[convName / "conv1d" / "Biases"] = "inference.postnet_convolutions.conv_layer_" + to_string(i) + "_postnet_convolutions.conv1d.bias";
        plain[convName / "batch_norm" / "Weights"] = "inference.postnet_convolutions.conv_layer_" + to_string(i) + "_postnet_convolutions.batch_normalization.gamma";
        plain[convName / "batch_norm" / "Biases"] = "inference.postnet_convolutions.conv_layer_" + to_string(i) + "_postnet_convolutions.batch_normalization.beta";
    }

    // PostNet projection
    transposed[name / "postnet_projection" / "Weights"] = "inference.postnet_projection.projection_postnet_projection.kernel";
    plain[name / "postnet_projection" / "Biases"] = "inference.postnet_projection.projection_postnet_projection.bias";

    // PreNet
    for (size_t i = 1; i <= params.prenetLayers.size(); ++i)
    {
        auto denseName = name / "decoder" / "_cell" / "prenet" / "dense[" + to_string(i) + "]";
        transposed[denseName / "Weights"] = "inference.decoder.decoder_prenet.dense_" + to_string(i) + ".kernel";
        plain[denseName / "Biases"] = "inference.decoder.decoder_prenet.dense_" + to_string(i) + ".bias";
    }

    // Frame projection
    transposed[name / "decoder" / "_cell" / "frame_projection" / "Weights"] = "inference.decoder.linear_transform_projection.projection_linear_transform_projection.kernel";
    plain[name / "decoder" / "_cell" / "frame_projection" / "Biases"] = "inference.decoder.linear_transform_projection.projection_linear_transform_projection.bias";
    // Stop token projection
    if (!params.useDurationPrediction)
    {
        transposed[name / "decoder" / "_cell" / "stop_projection_dense" / "Weights"] = "inference.decoder.stop_token_projection.projection_stop_token_projection.kernel";
        plain[name / "decoder" / "_cell" / "stop_projection_dense" / "Biases"] = "inference.decoder.stop_token_projection.projection_stop_token_projection.bias";

        // Attention
        transposed[name / "decoder" / "_cell" / "attention_mechanism" / "memory_layer" / "Weights"] = "inference.memory_layer.kernel";

        if (params.attentionType == "StepwiseMonotonic")
        {
            // SMA
            transposed[name / "decoder" / "_cell" / "attention_mechanism" / "query_layer" / "Weights"] = "inference.decoder.bahdanau_monotonic_attention.query_layer.kernel";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "attention_v"] = "inference.decoder.bahdanau_monotonic_attention.attention_v";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "attention_b"] = "inference.decoder.bahdanau_monotonic_attention.attention_b";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "attention_g"] = "inference.decoder.bahdanau_monotonic_attention.attention_g";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "score_bias"] = "inference.decoder.bahdanau_monotonic_attention.attention_score_bias";
        }
        else if (params.attentionType == "DynamicConvolutional")
        {
            string dcaName = "GravesV2";
            if (spbModel)
            {
                dcaName = "DynamicConvolution";
            }
            // DCA
            transposed[name / "decoder" / "_cell" / "attention_mechanism" / "location_layer" / "Weights"] = "inference.decoder." + dcaName + ".location_features_layer.kernel";
            transposed[name / "decoder" / "_cell" / "attention_mechanism" / "dynamic_fc1" / "Weights"] = "inference.decoder." + dcaName + ".dynamic_fc1.kernel";
            transposed[name / "decoder" / "_cell" / "attention_mechanism" / "dynamic_fc2" / "Weights"] = "inference.decoder." + dcaName + ".dynamic_fc2.kernel";
            transposed[name / "decoder" / "_cell" / "attention_mechanism" / "dynamic_projection" / "Weights"] = "inference.decoder." + dcaName + ".dynamic_projection.kernel";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "location_convolution" / "Weights"] = "inference.decoder." + dcaName + ".location_features_convolution.kernel";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "location_convolution" / "Biases"] = "inference.decoder." + dcaName + ".location_features_convolution.bias";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "dynamic_fc1" / "Biases"] = "inference.decoder." + dcaName + ".dynamic_fc1.bias";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "attention_variable_projection"] = "inference.decoder." + dcaName + ".attention_variable_projection";
            plain[name / "decoder" / "_cell" / "attention_mechanism" / "attention_bias"] = "inference.decoder." + dcaName + ".attention_bias";
        }

        // Attention LSTM
        if (params.useAttentionRnn)
        {
            plain[name / "decoder" / "_cell" / "attention_LSTM" / "linear" / "Biases"] = "inference.decoder.attention_LSTM.lstm_cell.bias";
            plain[name / "decoder" / "_cell" / "attention_LSTM" / "linear" / "Weights"] = "inference.decoder.attention_LSTM.lstm_cell.kernel";
        }
    }
    else
    {
        // duration prediction
        transposed[name / "duration_predictor" / "range_predictor_projection" / "Weights"] = "inference.range_predictor_projection.projection_range_predictor_projection.kernel";
        plain[name / "duration_predictor" / "range_predictor_projection" / "Biases"] = "inference.range_predictor_projection.projection_range_predictor_projection.bias";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "1" / "direct" / "cell" / "linear" / "Biases"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.fw.range_predictor_fw_LSTM_0.bias";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "1" / "direct" / "cell" / "linear" / "Weights"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.fw.range_predictor_fw_LSTM_0.kernel";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "1" / "reversed" / "cell" / "linear" / "Biases"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.bw.range_predictor_bw_LSTM_0.bias";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "1" / "reversed" / "cell" / "linear" / "Weights"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.bw.range_predictor_bw_LSTM_0.kernel";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "2" / "direct" / "cell" / "linear" / "Biases"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.fw.range_predictor_fw_LSTM_1.bias";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "2" / "direct" / "cell" / "linear" / "Weights"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.fw.range_predictor_fw_LSTM_1.kernel";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "2" / "reversed" / "cell" / "linear" / "Biases"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.bw.range_predictor_bw_LSTM_1.bias";
        plain[name / "duration_predictor" / "range_predictor_LSTM" / "2" / "reversed" / "cell" / "linear" / "Weights"] =
            "inference.range_predictor_LSTM.bidirectional_rnn.bw.range_predictor_bw_LSTM_1.kernel";

        transposed[name / "duration_predictor" / "duration_predictor_projection" / "Weights"] = "inference.duration_predictor_projection.projection_duration_predictor_projection.kernel";
        plain[name / "duration_predictor" / "duration_predictor_projection" / "Biases"] = "inference.duration_predictor_projection.projection_duration_predictor_projection.bias";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "1" / "direct" / "cell" / "linear" / "Biases"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.fw.duration_predictor_fw_LSTM_0.bias";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "1" / "direct" / "cell" / "linear" / "Weights"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.fw.duration_predictor_fw_LSTM_0.kernel";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "1" / "reversed" / "cell" / "linear" / "Biases"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.bw.duration_predictor_bw_LSTM_0.bias";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "1" / "reversed" / "cell" / "linear" / "Weights"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.bw.duration_predictor_bw_LSTM_0.kernel";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "2" / "direct" / "cell" / "linear" / "Biases"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.fw.duration_predictor_fw_LSTM_1.bias";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "2" / "direct" / "cell" / "linear" / "Weights"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.fw.duration_predictor_fw_LSTM_1.kernel";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "2" / "reversed" / "cell" / "linear" / "Biases"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.bw.duration_predictor_bw_LSTM_1.bias";
        plain[name / "duration_predictor" / "duration_predictor_LSTM" / "2" / "reversed" / "cell" / "linear" / "Weights"] =
            "inference.duration_predictor_LSTM.bidirectional_rnn.bw.duration_predictor_bw_LSTM_1.kernel";
    }
    // Speaker embedding

    // Language embedding

    // LSTM
    for (size_t i = 0; i < params.decoderLstmUnits.size(); ++i)
    {
        plain[name / "decoder" / "_cell" / "decoder_LSTM" / "zoneout_lstm_cell" / to_string(i) / "linear" / "Biases"] =
            "inference.decoder.decoder_LSTM.multi_rnn_cell.cell_" + to_string(i) + ".decoder_LSTM_" + to_string(i + 1) + ".bias";
        plain[name / "decoder" / "_cell" / "decoder_LSTM" / "zoneout_lstm_cell" / to_string(i) / "linear" / "Weights"] =
            "inference.decoder.decoder_LSTM.multi_rnn_cell.cell_" + to_string(i) + ".decoder_LSTM_" + to_string(i + 1) + ".kernel";
    }

    if (spbModel)
    {
        plain[name / "speaker_embedding"] = "inference.fixed_spk_embedding";
    }

    oplain.insert(plain.begin(), plain.end());
    otransposed.insert(transposed.begin(), transposed.end());

    // Adam params
    if (loadOptimizerParams)
    {
        for (auto& p : plain)
        {
            oplain[Name("Adam") / p.first / "m"] = "Tacotron_model." + p.second + ".Adam";
            oplain[Name("Adam") / p.first / "v"] = "Tacotron_model." + p.second + ".Adam_1";
        }
        for (auto& p : transposed)
        {
            otransposed[Name("Adam") / p.first / "m"] = "Tacotron_model." + p.second + ".Adam";
            otransposed[Name("Adam") / p.first / "v"] = "Tacotron_model." + p.second + ".Adam_1";
        }
    }

    for (size_t i = 1; i <= params.postnetKernelSize.size(); ++i)
    {
        auto convName = name / "postnet_convolutions" / "conv1d[" + to_string(i) + "]";
        oplain[convName / "batch_norm" / "MeanEval"] = "inference.postnet_convolutions.conv_layer_" + to_string(i) + "_postnet_convolutions.batch_normalization.moving_mean";
        oplain[convName / "batch_norm" / "VarianceEval"] = "inference.postnet_convolutions.conv_layer_" + to_string(i) + "_postnet_convolutions.batch_normalization.moving_variance";
    }
}

template<typename MM>
void createAdamTensors(const Names& params, MM& m)
{
    for (const auto& p : params)
    {
        m.createTensor(Name("Adam") / p / "m", m[p].getShape());
        m.createTensor(Name("Adam") / p / "v", m[p].getShape());
    }
}

template void createAdamTensors<MemoryManager>(const Names& params, MemoryManager& m);
template void createAdamTensors<MemoryManagerFP16>(const Names& params, MemoryManagerFP16& m);

template<typename MM>
size_t loadTacotronParams(const string& pathPrefix, MM& m, const Name& name, const TacotronParams& params, bool allParamsShouldExist, bool loadOptimizerParams, bool spbModel)
{
    if (params.useDurationPrediction && spbModel)
    {
        THROW_NONAME("TacotronTestTools", "duration prediction not supported for SpB model");
    }

    map<Name, string> maps[2]; // plain and transposed

    tacotronParamNamesMaps(name, params, maps[0], maps[1], loadOptimizerParams, spbModel);

    size_t loaded = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        vector<Name> tensors;
        for (const auto& p : maps[i])
        {
            tensors.push_back(p.first);
        }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t j = 0; j < tensors.size(); ++j)
        {
            auto pname = tensors[j];
            auto file = pathPrefix + maps[i][tensors[j]] + ".0.data";

            if (!m.tensorExists(pname))
            {
                if (allParamsShouldExist)
                {
#if defined(_OPENMP)
#pragma omp critical
#endif
                    {
                        cout << "Tensor '" + pname + "' not found" << endl;
                    }
                }
                continue;
            }

            if (!filesystem::exists(file))
            {
#if defined(_OPENMP)
#pragma omp critical
#endif
                {
                    cout << "File '" + file + "' not found" << endl;
                }
                continue;
            }

#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                cout << "Loading '" + pname + "'" << endl << "    from '" << file << endl;
            }
            switch (i)
            {
                case 0:
                {
                    // simply load
                    DataLoader::loadData(file, m[pname]);
                    break;
                }
                case 1:
                    // load and transpose height <-> width
                    DataLoader::loadData(file, m[pname]);
                    Common::transpose(m[pname], m[pname].getHeight());
                    break;
                    // default: Do nothing
            }
#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                ++loaded;
            }
        }
    }

    return loaded;
}

template size_t
loadTacotronParams<MemoryManager>(const string& pathPrefix, MemoryManager& m, const Name& name, const TacotronParams& params, bool allParamsShouldExist, bool loadOptimizerParams, bool spbModel);
template size_t loadTacotronParams<MemoryManagerFP16>(const string& pathPrefix,
                                                      MemoryManagerFP16& m,
                                                      const Name& name,
                                                      const TacotronParams& params,
                                                      bool allParamsShouldExist,
                                                      bool loadOptimizerParams,
                                                      bool spbModel);

size_t loadTacotronParams(const string& pathPrefix, MemoryManagerGPU& m, const Name& name, const TacotronParams& params, bool allParamsShouldExist, bool loadOptimizerParams, bool spbModel)
{
    map<Name, string> maps[2]; // plain and transposed

    tacotronParamNamesMaps(name, params, maps[0], maps[1], loadOptimizerParams, spbModel);

    size_t loaded = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        vector<Name> tensors;
        for (const auto& p : maps[i])
        {
            tensors.push_back(p.first);
        }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t j = 0; j < tensors.size(); ++j)
        {
            auto pname = tensors[j];
            auto file = pathPrefix + maps[i][tensors[j]] + ".0.data";

            if (!m.tensorExists(pname))
            {
                if (allParamsShouldExist)
                {
#if defined(_OPENMP)
#pragma omp critical
#endif
                    {
                        cout << "Tensor '" + pname + "' not found" << endl;
                    }
                }
                continue;
            }

            if (!filesystem::exists(file))
            {
#if defined(_OPENMP)
#pragma omp critical
#endif
                {
                    cout << "File '" + file + "' not found" << endl;
                }
                continue;
            }

#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                cout << "Loading '" + pname + "'" << endl << "    from '" << file << endl;
            }
            switch (i)
            {
                case 0:
                {
                    // simply load
                    DataLoader::loadData(file, m, pname);
                    break;
                }
                case 1:
                    // load and transpose height <-> width
                    DataLoader::loadData(file, m, pname);
                    Common::transpose(m[pname], m(pname).getHeight());
                    break;
                    // default: Do nothing
            }
#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                ++loaded;
            }
        }
    }

    return loaded;
}

template<typename T>
bool loadTFData(const filesystem::path& p, const vector<T*>& tensors)
{
    if (!filesystem::exists(p))
    {
        cout << "File '" + p.string() + "' not found" << endl;
        return false;
    }

    size_t tensorsCount = tensors.size();

    ifstream f(p);

    std::stringstream buffer;
    buffer << f.rdbuf();
    auto lines = Common::split(buffer.str(), '\n');
    auto cnt = lines.size();

    if (cnt == 0 || cnt % tensorsCount != 0)
    {
        cout << "File '" + p.string() + "' has bad line count: unable to load " << tensorsCount << " tensors from " << cnt << " lines" << endl;
        return false;
    }

    size_t tSize = cnt / tensorsCount;

    string s;
    for (size_t k = 0; k < tensorsCount; ++k)
    {
        auto& t = *(tensors[k]);
        vector<typename T::type> vec;

        for (size_t i = 0; i < tSize - 1; ++i)
        {
            s = lines[k * tSize + i + 1];
            Common::replaceAll(s, "[", "");
            Common::replaceAll(s, "]", "");

            const char* begin = s.c_str();
            char* end = NULL;

            do
            {
                if (end != NULL)
                {
                    begin = end;
                }
                end = NULL;
                float v = strtof(begin, &end);
                if (end != begin)
                {
                    vec.push_back(static_cast<typename T::type>(v));
                }
            } while (end != begin);
        }

        if (t.size() != vec.size())
        {
            cout << t.getName() << " has inconsistent size with '" << p.string() << "' (" << t.size() << " vs " << vec.size() << ")" << endl;
            return false;
        }

        copy(vec.begin(), vec.end(), t.begin());
    }
    return true;
}

bool loadTFData(const filesystem::path& p, TensorGPUHelper tensor)
{
    Tensor t = tensor;
    bool ok = loadTFData(p, t);
    if (ok)
    {
        tensor = t;
    }
    return ok;
}

bool loadTFData(const filesystem::path& p, MemoryManagerGPU& m, const vector<TensorGPU*>& tensors)
{
    if (!filesystem::exists(p))
    {
        cout << "File '" + p.string() + "' not found" << endl;
        return false;
    }

    size_t tensorsCount = tensors.size();

    ifstream f(p);

    std::stringstream buffer;
    buffer << f.rdbuf();
    auto lines = Common::split(buffer.str(), '\n');
    auto cnt = lines.size();

    if (cnt == 0 || cnt % tensorsCount != 0)
    {
        cout << "File '" + p.string() + "' has bad line count: unable to load " << tensorsCount << " tensors from " << cnt << " lines" << endl;
        return false;
    }

    size_t tSize = cnt / tensorsCount;

    string s;
    for (size_t k = 0; k < tensorsCount; ++k)
    {
        auto& t = *(tensors[k]);
        getline(f, s);
        vector<float> vec;

        for (size_t i = 0; i < tSize - 1; ++i)
        {
            getline(f, s);
            Common::replaceAll(s, "[", "");
            Common::replaceAll(s, "]", "");
            stringstream str(s);
            float v = 1;
            while (str >> v)
            {
                vec.push_back(v);
            }
        }

        if (t.size() != vec.size())
        {
            cout << t.getName() << " has inconsistent size with '" << p.string() << "' (" << t.size() << " vs " << vec.size() << ")" << endl;
            return false;
        }

        TensorGPUHelper h(t, m.getKernelManager());
        h = raul::TensorImpl<float>::dt_range(&vec.front(), &vec.back() + 1);
    }
    return true;
}

template bool loadTFData<Tensor>(const filesystem::path& p, const vector<Tensor*>& tensors);
template bool loadTFData<TensorFP16>(const filesystem::path& p, const vector<TensorFP16*>& tensors);

bool loadTFData(const filesystem::path& p, MemoryManagerGPU& m, const Names& tensors)
{
    vector<TensorGPU*> tt;
    for (auto& name : tensors)
    {
        tt.push_back(&m(name));
    }

    return loadTFData(p, m, tt);
}

bool loadTFData(const filesystem::path& p, MemoryManager& m, const Names& tensors)
{
    vector<Tensor*> tt;
    for (auto& name : tensors)
    {
        tt.push_back(&m[name]);
    }

    return loadTFData<Tensor>(p, tt);
}

template<typename T>
bool loadTFData(const filesystem::path& p, T& tensor)
{
    return loadTFData<T>(p, { &tensor });
}

template bool loadTFData<Tensor>(const filesystem::path& p, Tensor& tensor);
template bool loadTFData<TensorFP16>(const filesystem::path& p, TensorFP16& tensor);

template<typename MM>
size_t saveTacotronParamsImpl(const filesystem::path& dir, const string& prefix, MM& m, const Name& name, const TacotronParams& params, bool withAdam, bool spbModel)
{
    map<Name, string> maps[2]; // plain and transposed

    string pathPrefix = (dir / prefix).string();
    filesystem::create_directories(dir);
    tacotronParamNamesMaps(name, params, maps[0], maps[1], true, spbModel);
    maps[0].emplace(name / "speaker_embedding", "speaker_embedding");
    size_t saved = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (const auto& p : maps[i])
        {
            auto pname = p.first;
            auto file = pathPrefix + p.second + ".0.data";

            if (!withAdam && Common::startsWith(pname, "Adam"))
            {
                continue;
            }

            if (!m.tensorExists(pname))
            {
                cout << "Tensor '" + pname + "' not found" << endl;
                continue;
            }
            const Tensor& tensorT = m[pname];
            Tensor tensor(tensorT.getShape());
            tensor = TORANGE(tensorT);

            ofstream f(file);
            auto shape = tensor.getShape();
            f << "Tacotron_model." << p.second << ".0" << endl;
            cout << "Tacotron_model." << p.second << ".0 ";
            switch (i)
            {
                case 0:
                    // simply load
                    if (pname.str().find("Weights") != string::npos)
                    {
                        if (pname.str().find("LSTM") != string::npos)
                        {
                            f << shape[3] << " " << shape[2] << endl;
                            cout << "(" << shape[3] << ", " << shape[2] << ")" << endl;
                        }
                        else if (pname.str().find("batch_norm") != string::npos)
                        {
                            f << shape.total_size() << endl;
                            cout << "(" << shape.total_size() << ",)" << endl;
                        }
                        else
                        {
                            f << shape[1] << " " << shape[2] << " " << shape[3] << endl;
                            cout << "(" << shape[1] << ", " << shape[2] << ", " << shape[3] << ")" << endl;
                        }
                    }
                    else
                    {
                        f << shape.total_size() << endl;
                        cout << "(" << shape.total_size() << ",)" << endl;
                    }
                    saveTensor(tensor, f);
                    break;
                case 1:
                    if (pname.str().find("Biases") != string::npos)
                    {
                        f << shape.total_size() << endl;
                        cout << "(" << shape.total_size() << ",)" << endl;
                    }
                    else
                    {
                        f << shape[3] << " " << shape[2] << endl;
                        cout << "(" << shape[3] << ", " << shape[2] << ")" << endl;
                    }
                    // load and transpose height <-> width
                    Common::transpose(tensor, tensor.getWidth());
                    saveTensor(m[pname], f);
                    break;
                    // default: Do nothing
            }
            ++saved;
        }
    }

    return saved;
}

size_t saveTacotronParams(const filesystem::path& dir, const string& prefix, MemoryManager& m, const Name& name, const TacotronParams& params, bool withAdam, bool spbModel)
{
    return saveTacotronParamsImpl<MemoryManager>(dir, prefix, m, name, params, withAdam, spbModel);
}

size_t saveTacotronParams(const filesystem::path& dir, const string& prefix, MemoryManagerGPU& m, const Name& name, const TacotronParams& params, bool withAdam, bool spbModel)
{
    return saveTacotronParamsImpl<MemoryManagerGPU>(dir, prefix, m, name, params, withAdam, spbModel);
}

pair<size_t, size_t> tacotronParamsCount(const TacotronParams& params)
{
    size_t cntTrainable = params.trainableSpeakerEmbedding ? 1 : 0; // speaker_embedding
    cntTrainable += params.prenetLayers.size() * 2;                 // prenet: bias, kernel
    if (params.useAttentionRnn)
    {
        cntTrainable += 2; // attention_rnn: bias, kernel
    }
    cntTrainable += 2 * params.decoderLstmUnits.size(); // decoder_LSTM: bias, kernel

    if (!params.useDurationPrediction)
    {
        cntTrainable += 2; // stop_token_projection: bias, kernel
    }
    cntTrainable += 2;                                   // frame_projection: bias, kernel
    cntTrainable += 4 * params.postnetKernelSize.size(); // postnet: bias, kernel, batch_norm.beta, batch_norm.gamma;
    cntTrainable += 2;                                   // postnet_projection: bias, kernel

    if (params.attentionType == "StepwiseMonotonic")
    {
        cntTrainable += 6; // sma: memory_layer.kernel, query_layer.kernel, score_bias, attention_v, attention_b, attention_g
    }
    else if (params.attentionType == "DynamicConvolutional")
    {
        cntTrainable += 3; // dca: memory_layer.kernel, attention_variable_projection, attention_bias
        cntTrainable += 4; // dynamic_fc1.bias, dynamic_fc1.kernel, dynamic_fc2.kernel, dynamic_projection.kernel
        cntTrainable += 3; // dca: location_features_layer.kernel, location_features_convolution.bias, location_features_convolution.kernel
    }
    else if (!params.attentionType.empty() && params.attentionType != "None")
    {
        THROW_NONAME("TacotronTestTools", "unsupported attention type '" + params.attentionType + "'");
    }
    if (params.useDurationPrediction)
    {
        cntTrainable += 2; // duration_prediction_projection: bias, kernel
        cntTrainable += 2; // range_prediction_projection: bias, kernel
        cntTrainable += 8; // range_predictor_LSTM: bias, kernel for forward and backward cell with 2 layers
        cntTrainable += 8; // range_predictor_LSTM: bias, kernel for forward and backward cell with 2 layers
    }

    size_t cntAll = cntTrainable + 2 * params.postnetKernelSize.size(); // postnet: batch_norm.mean, batch_norm.variance
    return make_pair(cntTrainable, cntAll);
}

unique_ptr<optimizers::Scheduler::LrScheduler> createOptimizer(const TacotronTrainingParams& trainParams, dtype ADAM_BETA1, dtype ADAM_BETA2, dtype ADAM_EPSILON)
{
    using namespace optimizers;
    auto strategyExpOffset = make_unique<Scheduler::Strategies::StepOffset>(-static_cast<long long>(trainParams.decayStartStep),
                                                                            make_unique<Scheduler::Strategies::Exponential>(trainParams.decayRate, trainParams.decaySteps));
    auto strategyClipped = make_unique<Scheduler::Strategies::ClipLower>(trainParams.finalLearningRate, make_unique<Scheduler::Strategies::ClipUpper>(move(strategyExpOffset)));
    // auto strategy = trainParams.warmupLearningRate ? make_unique<Scheduler::Strategies::WarmUp>(trainParams.warmupSteps, true, move(strategyClipped)) : std::move(strategyClipped);

    auto adam = make_unique<Adam>(trainParams.initialLearningRate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    if (trainParams.warmupLearningRate)
    {
        return make_unique<Scheduler::LrScheduler>(make_unique<Scheduler::Strategies::WarmUp>(trainParams.warmupSteps, true, move(strategyClipped)), move(adam));
    }
    return make_unique<Scheduler::LrScheduler>(move(strategyClipped), move(adam));
}

}
