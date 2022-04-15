// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "API.h"

#include <string.h>

#include <training/api/API.h>
#include <training/common/Common.h>
#include <training/initializers/ConstantInitializer.h>
#include <training/initializers/IInitializer.h>
#include <training/initializers/RandomNormInitializer.h>
#include <training/initializers/RandomUniformInitializer.h>
#include <training/initializers/XavierInitializer.h>
#include <training/layers/BasicLayer.h>
#include <training/layers/composite/BERT.h>
#include <training/layers/composite/rnn/LSTMLayer.h>
#include <training/layers/parameters/LayerParameters.h>

#include <training/network/Layers.h>
#include <training/network/Workflow.h>
#include <training/optimizers/Adadelta.h>
#include <training/optimizers/Adagrad.h>
#include <training/optimizers/Adam.h>
#include <training/optimizers/Adamax.h>
#include <training/optimizers/Momentum.h>
#include <training/optimizers/Nesterov.h>
#include <training/optimizers/Optimizer.h>
#include <training/optimizers/SGD.h>
#include <training/optimizers/AdamW.h>

#include <training/optimizers/schedulers/LrScheduler.h>
#include <training/optimizers/schedulers/strategies/CosineAnnealing.h>

#include "lowlevel/APIChecks.h"
#include "lowlevel/APIDefinitions.h"

namespace
{
thread_local std::string _lastError;

template<typename Layer, typename Param>
API_STATUS addLayer(Graph_Description_t* desc, const char* name, Param&& param)
{
    CHECK_NOT_NULL(desc);

    CHECK_STRING(name);

    try
    {
        desc->mDef->add<Layer>(name, std::forward<Param>(param));
    }
    catch (std::exception& e)
    {
        set_last_error(e.what());
        return STATUS_ERROR;
    }

    return STATUS_OK;
}

// resnet
using ResNetT = std::array<size_t, 4U>;
using downsampleT = std::function<void(raul::Workflow&, raul::Name&, const raul::Name&)>;

void add_resnet_basic_block(size_t& block_cnt,
                            raul::Workflow& netdef,
                            raul::Name& input,
                            const raul::Name& name_prefix,
                            const size_t planes,
                            const size_t stride = 1U,
                            const std::optional<downsampleT> downsample = std::nullopt,
                            const size_t dilation = 1U,
                            const float bnMomentum = 0.1f,
                            const bool bias = false,
                            const bool quantize = false)
{
    const auto kernel_size = 3U;
    const auto default_stride = 1U;
    const auto block_name = name_prefix / "block" + Conversions::toString(block_cnt);
    auto input_for_shortcut = input;

    ++block_cnt;

    if (quantize)
    {
        netdef.add<raul::FakeQuantLayer>(block_name + "::fq1", raul::FakeQuantParams{ { input }, { block_name + "::fq1" } });
        input = block_name + "::fq1";
    }

    netdef.add<raul::Convolution2DLayer>(block_name + "::conv1", raul::Convolution2DParams{ { input }, { block_name + "::conv1" }, kernel_size, planes, stride, dilation, bias, quantize });
    netdef.add<raul::BatchNormLayer>(block_name + "::bn1", raul::BatchnormParams{ { block_name + "::conv1" }, { block_name + "::bn1" }, bnMomentum });
    netdef.add<raul::ReLUActivation>(block_name + "::relu1", raul::BasicParams{ { block_name + "::bn1" }, { block_name + "::relu1" } });
    input = block_name + "::relu1";
    if (quantize)
    {
        netdef.add<raul::FakeQuantLayer>(block_name + "::fq2", raul::FakeQuantParams{ { input }, { block_name + "::fq2" } });
        input = block_name + "::fq2";
    }

    netdef.add<raul::Convolution2DLayer>(block_name + "::conv2", raul::Convolution2DParams{ { input }, { block_name + "::conv2" }, kernel_size, planes, default_stride, dilation, bias, quantize });
    netdef.add<raul::BatchNormLayer>(block_name + "::bn2", raul::BatchnormParams{ { block_name + "::conv2" }, { block_name + "::bn2" }, bnMomentum });
    input = block_name + "::bn2";

    if (downsample)
    {
        (*downsample)(netdef, input_for_shortcut, block_name);
    }

    netdef.add<raul::ElementWiseSumLayer>(block_name + "::sum", raul::ElementWiseLayerParams{ { input_for_shortcut, input }, { block_name + "::sum" } });
    netdef.add<raul::ReLUActivation>(block_name + "::relu2", raul::BasicParams{ { block_name + "::sum" }, { block_name + "::relu2" } });
    input = block_name + "::relu2";
}

void add_input_block(raul::Workflow& netdef, raul::Name& input, const float bnMomentum = 0.1f, const bool bias = false)
{
    const auto conv_in_planes = 64U;
    const auto conv_kernel_size = 7U;
    const auto conv_stride = 2U;
    const auto conv_padding = 3U;

    const auto max_pool_kernel_size = 3;
    const auto max_pool_stride = 2;
    const auto max_pool_padding = 1U;

    netdef.add<raul::Convolution2DLayer>("input::conv1", raul::Convolution2DParams{ { input }, { "input::conv1" }, conv_kernel_size, conv_in_planes, conv_stride, conv_padding, bias });
    netdef.add<raul::BatchNormLayer>("input::bn1", raul::BatchnormParams{ { "input::conv1" }, { "input::bn1" }, bnMomentum });
    netdef.add<raul::ReLUActivation>("input::relu", raul::BasicParams{ { "input::bn1" }, { "input::relu" } });
    netdef.add<raul::MaxPoolLayer2D>("input::maxpool", raul::Pool2DParams{ { "input::relu" }, { "input::maxpool" }, max_pool_kernel_size, max_pool_stride, max_pool_padding });
    input = "input::maxpool";
}

void add_output_block(raul::Workflow& netdef, raul::Name& input, const size_t num_classes = 10U)
{
    netdef.add<raul::GlobAveragePoolLayer>("output::avg", raul::BasicParams{ { input }, { "output::avg" } });
    netdef.add<raul::ReshapeLayer>("output::reshape", raul::ViewParams{ "output::avg", "output::avgr", 1, 1, -1 });
    netdef.add<raul::LinearLayer>("output::fc0", raul::LinearParams{ "output::avgr", "output::fc0", num_classes });
    input = "output::fc0";
}

void add_resnet_layer(size_t& layer_cnt,
                      raul::Workflow& netdef,
                      raul::Name& input,
                      size_t& inplanes,
                      size_t expansion,
                      const size_t planes,
                      const size_t blocks,
                      const size_t stride = 1U,
                      const size_t dilation = 1U,
                      const float bnMomentum = 0.1f,
                      const bool bias = false,
                      const bool quantize = false)
{
    const auto layer_name = "layer" + Conversions::toString(layer_cnt);
    size_t block_cnt = 0U;
    std::optional<downsampleT> downsample = std::nullopt;

    ++layer_cnt;

    if (stride != 1U || inplanes != planes * expansion)
    {
        downsample = [layer_name, bias, quantize, planes, expansion, stride, bnMomentum](raul::Workflow& netdef, raul::Name& input, const raul::Name& name_prefix)
        {
            const auto conv_kernel_size = 1U;
            const auto downsample_name = name_prefix + "::downsample";
            if (quantize)
            {
                netdef.add<raul::FakeQuantLayer>(downsample_name + "::fq1", raul::FakeQuantParams{ { input }, { downsample_name + "::fq1" } });
                input = downsample_name + "::fq1";
            }
            netdef.add<raul::Convolution2DLayer>(downsample_name + "::conv1",
                                                 raul::Convolution2DParams{ { input }, { downsample_name + "::conv1" }, conv_kernel_size, planes * expansion, stride, 0U, bias, quantize });
            netdef.add<raul::BatchNormLayer>(downsample_name + "::bn1", raul::BatchnormParams{ { downsample_name + "::conv1" }, { downsample_name + "::bn1" }, bnMomentum });
            input = downsample_name + "::bn1";
        };
    }
    add_resnet_basic_block(block_cnt, netdef, input, layer_name, planes, stride, downsample, dilation, bnMomentum, bias, quantize);
    inplanes = planes * expansion;
    downsample = std::nullopt;
    for (size_t i = 1U; i < blocks; ++i)
    {
        const auto default_stride = 1U;
        add_resnet_basic_block(block_cnt, netdef, input, layer_name, planes, default_stride, downsample, dilation, bnMomentum, bias, quantize);
    }
}

raul::Name build_resnet(raul::Workflow& netdef, const ResNetT layers, raul::Name dataTensorName, const float bnMomentum = 0.1f, const bool bias = false, const bool quantize = false)
{
    // This parameters are actual only for BasicBlock (see notebook for details)
    size_t inplanes = 64U;
    const size_t expansion = 1U;

    size_t layer_cnt = 1U;
    add_input_block(netdef, dataTensorName);
    add_resnet_layer(layer_cnt, netdef, dataTensorName, inplanes, expansion, 64U, layers[0], 1U, 1U, bnMomentum, bias, quantize);
    add_resnet_layer(layer_cnt, netdef, dataTensorName, inplanes, expansion, 128U, layers[1], 2U, 1U, bnMomentum, bias, quantize);
    add_resnet_layer(layer_cnt, netdef, dataTensorName, inplanes, expansion, 256U, layers[2], 2U, 1U, bnMomentum, bias, quantize);
    add_resnet_layer(layer_cnt, netdef, dataTensorName, inplanes, expansion, 512U, layers[3], 2U, 1U, bnMomentum, bias, quantize);
    add_output_block(netdef, dataTensorName);
    return dataTensorName;
}

raul::Name build_resnet18(raul::Workflow& netdef, const raul::Name& dataTensorName, const float bnMomentum = 0.1f, const bool bias = false, const bool quantize = false)
{
    const ResNetT layers{ 2U, 2U, 2U, 2U };
    return build_resnet(netdef, layers, dataTensorName, bnMomentum, bias, quantize);
}

} // anonymous

extern "C"
{

    struct Graph_t
    {
        Graph_t(Graph_Description_t** desc, size_t batch_size)
            : mGraph((*desc)->mDef)
        {
            delete *desc;
            *desc = nullptr;

            mGraph->preparePipelines();
            mGraph->setBatchSize(batch_size);
            mGraph->prepareMemoryForTraining();
        }

        Graph_t(Graph_Description_t** desc, size_t batch_size, bool keep_data_grads)
            : mGraph((*desc)->mDef)
        {
            (void)keep_data_grads;
            delete *desc;
            *desc = nullptr;

            mGraph->preparePipelines();
            mGraph->setBatchSize(batch_size);
            mGraph->prepareMemoryForTraining();
        }

        std::shared_ptr<raul::Workflow> mGraph;
    };

    struct Optimizer_t
    {
        Optimizer_t(std::shared_ptr<raul::optimizers::Optimizer> o)
            : mOptimizer(std::move(o))
        {
        }
        std::shared_ptr<raul::optimizers::Optimizer> mOptimizer;
    };

    struct Initializer_t
    {
        Initializer_t(std::shared_ptr<raul::initializers::IInitializer> i)
            : mInitializer(std::move(i))
        {
        }
        std::shared_ptr<raul::initializers::IInitializer> mInitializer;
    };

    struct LrScheduler_t
    {
        LrScheduler_t(std::shared_ptr<raul::optimizers::Scheduler::LrScheduler> l)
            : mScheduler(std::move(l))
        {
        }

        std::shared_ptr<raul::optimizers::Scheduler::LrScheduler> mScheduler;
    };

    API_EXPORT const char* get_last_error() { return _lastError.c_str(); }
    API_EXPORT void set_last_error(const char* err) { _lastError = err; }

    API_EXPORT API_STATUS create_graph_description(Graph_Description_t** descr)
    {
        CHECK_NOT_NULL(descr);
        try
        {
            *descr = new Graph_Description_t();
        }
        catch (std::exception& e)
        {
            *descr = NULL;
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_graph_description_eager(Graph_Description_t** descr)
    {
        CHECK_NOT_NULL(descr);
        try
        {
            *descr = new Graph_Description_t(true);
        }
        catch (std::exception& e)
        {
            *descr = NULL;
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS delete_graph_description(Graph_Description_t* descr)
    {
        try
        {
            delete descr;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_graph(Graph_Description_t** desc, Graph_t** graph, size_t batch_size)
    {
        CHECK_NOT_NULL(desc);
        CHECK_NOT_NULL(*desc);
        CHECK_NOT_NULL(graph);

        CHECK_PRECONDITION(batch_size > 0);

        try
        {
            *graph = new Graph_t(desc, batch_size);
        }
        catch (std::exception& e)
        {
            *graph = NULL;
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_graph_with_data_grads(Graph_Description_t** desc, Graph_t** graph, size_t batch_size)
    {
        CHECK_NOT_NULL(desc);
        CHECK_NOT_NULL(*desc);
        CHECK_NOT_NULL(graph);

        CHECK_PRECONDITION(batch_size > 0);

        try
        {
            *graph = new Graph_t(desc, batch_size, true);
        }
        catch (std::exception& e)
        {
            *graph = NULL;
            _lastError = e.what();
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS delete_graph(Graph_t* graph)
    {
        try
        {
            delete graph;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS get_tensor(Graph_t* graph, const char* tensor_name, FLOAT_TYPE* data, size_t* size)
    {
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);
        CHECK_PRECONDITION(size);

        API_STATUS status = STATUS_ERROR;
        try
        {
            const raul::Tensor& tensor = graph->mGraph->getMemoryManager()[tensor_name];
            if (!data)
            {
                *size = tensor.size();
                status = STATUS_OK;
            }
            else if (*size == tensor.size())
            {
                std::copy(tensor.begin(), tensor.end(), data);
                status = STATUS_OK;
            }
            else
                return STATUS_ERROR_BAD_SIZE;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }

        return status;
    }

    API_EXPORT API_STATUS get_model_parameters(Graph_t* graph, bool only_trainable, char** parameters, size_t* param_count, size_t* max_param_name_length)
    {
        CHECK_NOT_NULL(graph);
        CHECK_PRECONDITION(parameters != nullptr || (param_count != nullptr && max_param_name_length != nullptr));

        API_STATUS status = STATUS_ERROR;
        try
        {
            raul::Names params;
            if (only_trainable)
            {
                params = graph->mGraph->getTrainableParameterNames();
            }
            else
            {
                params = graph->mGraph->getTrainableParameterNames();
            }

            if (!parameters)
            {
                *param_count = params.size();
                *max_param_name_length = 0;
                for (const auto& p : params)
                {
                    if (*max_param_name_length < p.size())
                    {
                        *max_param_name_length = p.size();
                    }
                }
                *max_param_name_length += 1;
            }
            else
            {
                for (size_t i = 0; i < params.size(); ++i)
                {
                    strncpy(parameters[i], params[i].c_str(), params[i].size());
                    parameters[i][params[i].size()] = '\0';
                }
            }

            status = STATUS_OK;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }

        return status;
    }

    API_EXPORT API_STATUS set_tensor(Graph_t* graph, const char* tensor_name, const FLOAT_TYPE* data, size_t size)
    {
        CHECK_NOT_NULL(graph);
        CHECK_NOT_NULL(data);
        CHECK_STRING(tensor_name);

        API_STATUS status = STATUS_ERROR;
        try
        {
            raul::MemoryManager& memoryManager = graph->mGraph->getMemoryManager();
            memoryManager[std::string(tensor_name)] = raul::Tensor::dt_range(data, data + size);
            status = STATUS_OK;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }

        return status;
    }

    API_EXPORT API_STATUS create_tensor(Graph_t* graph, const char* tensor_name, size_t batchSize, size_t depth, size_t height, size_t width)
    {
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);

        API_STATUS status = STATUS_ERROR;
        try
        {
            raul::MemoryManager& memoryManager = graph->mGraph->getMemoryManager();
            memoryManager.createTensor(tensor_name, batchSize, depth, height, width);
            status = STATUS_OK;
        }
        catch (std::exception& e)
        {
            _lastError = e.what();
        }

        return status;
    }

    API_EXPORT API_STATUS arange(Graph_t* graph, const char* tensor_name, FLOAT_TYPE start, FLOAT_TYPE step)
    {
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);

        try
        {
            raul::MemoryManager& memoryManager = graph->mGraph->getMemoryManager();
            auto& tensor = memoryManager[std::string(tensor_name)];
            raul::Common::arange(tensor.begin(), tensor.end(), start, step);
            return STATUS_OK;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }

        return STATUS_ERROR;
    }

    API_EXPORT API_STATUS fill_tensor(Graph_t* graph, const char* tensor_name, const FLOAT_TYPE value)
    {
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);

        try
        {
            raul::MemoryManager& memoryManager = graph->mGraph->getMemoryManager();
            auto& tensor = memoryManager[std::string(tensor_name)];
            for (size_t i = 0; i < memoryManager[std::string(tensor_name)].size(); ++i)
            {
                tensor[i] = value;
            }
            return STATUS_OK;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }

        return STATUS_ERROR;
    }

    API_EXPORT API_STATUS create_adadelta_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate)
    {
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION(learning_rate > 0);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::Adadelta>(learning_rate));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_adagrad_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate)
    {
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION(learning_rate > 0);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::Adagrad>(learning_rate));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_adam_optimizer(Optimizer_t** optimizer, FLOAT_TYPE alfa, FLOAT_TYPE beta_1, FLOAT_TYPE beta_2, FLOAT_TYPE epsilon)
    {
        CHECK_NOT_NULL(optimizer);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::Adam>(alfa, beta_1, beta_2, epsilon));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_adamax_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate)
    {
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION(learning_rate > 0);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::Adamax>(learning_rate));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_momentum_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate, FLOAT_TYPE momentum)
    {
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION(learning_rate > 0);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::Momentum>(learning_rate, momentum));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_nesterov_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate, FLOAT_TYPE momentum)
    {
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION(learning_rate > 0);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::Nesterov>(learning_rate, momentum));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_sgd_optimizer(Optimizer_t** optimizer, FLOAT_TYPE learning_rate)
    {
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION(learning_rate > 0);

        try
        {
            *optimizer = new Optimizer_t(std::make_shared<raul::optimizers::SGD>(learning_rate));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS delete_optimizer(Optimizer_t* optimizer)
    {
        try
        {
            delete optimizer;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_constant_initializer(Initializer_t** initializer, FLOAT_TYPE value)
    {
        CHECK_NOT_NULL(initializer);

        try
        {
            *initializer = new Initializer_t(std::make_shared<raul::initializers::ConstantInitializer>(value));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_random_norm_initializer(Initializer_t** initializer, FLOAT_TYPE mean, FLOAT_TYPE stddev, size_t seed)
    {
        CHECK_NOT_NULL(initializer);

        try
        {
            *initializer = new Initializer_t(std::make_shared<raul::initializers::RandomNormInitializer>(seed, mean, stddev));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_random_uniform_initializer(Initializer_t** initializer, FLOAT_TYPE min_value, FLOAT_TYPE max_value, size_t seed)
    {
        CHECK_NOT_NULL(initializer);

        try
        {
            *initializer = new Initializer_t(std::make_shared<raul::initializers::RandomUniformInitializer>(seed, min_value, max_value));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_xavier_norm_initializer(Initializer_t** initializer, size_t seed)
    {
        CHECK_NOT_NULL(initializer);

        try
        {
            *initializer = new Initializer_t(std::make_shared<raul::initializers::XavierNormInitializer>(seed));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS create_xavier_uniform_initializer(Initializer_t** initializer, size_t seed)
    {
        CHECK_NOT_NULL(initializer);

        try
        {
            *initializer = new Initializer_t(std::make_shared<raul::initializers::XavierUniformInitializer>(seed));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS delete_initializer(Initializer_t* initializer)
    {
        try
        {
            delete initializer;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS initialize_tensor(Graph_t* graph, Initializer_t* initializer, const char* tensor_name)
    {
        CHECK_NOT_NULL(initializer);
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);

        API_STATUS status = STATUS_ERROR;
        try
        {
            raul::MemoryManager& memoryManager = graph->mGraph->getMemoryManager();
            CHECK_PRECONDITION_M(memoryManager.tensorExists(tensor_name), std::string("Tensor \"" + std::string(tensor_name) + "\" not found").c_str());
            auto& tensor = memoryManager[std::string(tensor_name)];

            initializer->mInitializer->operator()(tensor);

            status = STATUS_OK;
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }

        return status;
    }

    API_EXPORT API_STATUS create_cosine_annealing_adam_w_lr_scheduler(LrScheduler_t** scheduler, size_t size, FLOAT_TYPE max_a, FLOAT_TYPE min_a, FLOAT_TYPE warmup_percentage, FLOAT_TYPE warmup_pow, FLOAT_TYPE annealing_pow, FLOAT_TYPE base_lr, FLOAT_TYPE beta_1, FLOAT_TYPE beta_2, FLOAT_TYPE epsilon, FLOAT_TYPE weight_decay)
    {
        CHECK_NOT_NULL(scheduler);
        CHECK_PRECONDITION(base_lr > 0);
        CHECK_PRECONDITION(beta_1 >= 0 && beta_1 < 1);
        CHECK_PRECONDITION(beta_2 >= 0 && beta_2 < 1);

        try
        {
            *scheduler = new LrScheduler_t(std::make_shared<raul::optimizers::Scheduler::LrScheduler>(std::make_unique<raul::optimizers::Scheduler::Strategies::CosineAnnealing>(size, max_a, min_a, warmup_percentage, warmup_pow, annealing_pow), std::make_unique<raul::optimizers::AdamW>(base_lr, beta_1, beta_2, epsilon, weight_decay)));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS set_batch_size(Graph_t* graph, size_t batch_size)
    {
        CHECK_NOT_NULL(graph);
        CHECK_PRECONDITION(batch_size > 0);

        try
        {
            auto& network = graph->mGraph;
            network->setBatchSize(batch_size);
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS train_single_pass(Graph_t* graph, Optimizer_t* optimizer, const char** loss_names, size_t loss_count, FLOAT_TYPE* loss)
    {
        CHECK_NOT_NULL(graph);
        CHECK_NOT_NULL(optimizer);
        CHECK_PRECONDITION_M(optimizer->mOptimizer, "Optimizer not initialized");
        CHECK_PRECONDITION(loss_count == 0 || (loss_names != NULL && loss != NULL));

        try
        {
            auto& network = graph->mGraph;
            auto& opt = *optimizer->mOptimizer.get();
            auto& mm = network->getMemoryManager();
            network->forwardPassTraining();

            for (size_t i = 0; i < loss_count; ++i)
            {
                const raul::Tensor& loss_tensor = network->getMemoryManager()[loss_names[i]];
                raul::dtype totalLoss = loss_tensor[0];
                loss[i] = static_cast<FLOAT_TYPE>(totalLoss);
            }
            network->backwardPassTraining();
            auto params = network->getTrainableParameters();
            for (auto& p : params)
            {
                opt(mm, p.Param, p.Gradient);
            }
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS train_single_pass_with_scheduling(Graph_t* graph, LrScheduler_t* scheduler, const char** loss_names, size_t loss_count, FLOAT_TYPE* loss)
    {
        CHECK_NOT_NULL(graph);
        CHECK_NOT_NULL(scheduler);
        CHECK_PRECONDITION_M(scheduler->mScheduler, "Scheduler not initialized");
        CHECK_PRECONDITION(loss_count == 0 || (loss_names != NULL && loss != NULL));

        try
        {
            auto& network = graph->mGraph;
            auto& sch = *scheduler->mScheduler.get();
            auto& mm = network->getMemoryManager();
            network->forwardPassTraining();

            for (size_t i = 0; i < loss_count; ++i)
            {
                const raul::Tensor& loss_tensor = network->getMemoryManager()[loss_names[i]];
                raul::dtype totalLoss = loss_tensor[0];
                loss[i] = static_cast<FLOAT_TYPE>(totalLoss);
            }
            network->backwardPassTraining();

            sch.step();

            auto params = network->getTrainableParameters();
            for (auto& p : params)
            {
                sch(mm, p.Param, p.Gradient);
            }
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS test_network(Graph_t* graph, const char* prob_tensor_name, size_t* correctClasses, FLOAT_TYPE* accuracy)
    {
        CHECK_NOT_NULL(graph);
        CHECK_NOT_NULL(correctClasses);
        CHECK_NOT_NULL(accuracy);
        CHECK_STRING(prob_tensor_name);

        try
        {
            auto& network = graph->mGraph;

            CHECK_PRECONDITION_M(network->getMemoryManager().tensorExists(prob_tensor_name), "Tensor not found");

            auto batchSize = network->getBatchSize();

            network->forwardPassTesting();

            const raul::Tensor& softmax = network->getMemoryManager()[prob_tensor_name];

            size_t nClasses = softmax.size() / batchSize;
            size_t correctLabelsCounter = 0;

            for (size_t w = 0; w < network->getBatchSize(); ++w)
            {
                if (softmax.getMaxIndex(w * nClasses, (w + 1) * nClasses) == correctClasses[w]) ++correctLabelsCounter;
            }

            *accuracy = FLOAT_TYPE(correctLabelsCounter) / FLOAT_TYPE(batchSize);
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS network_forward(Graph_t* graph, bool is_test)
    {
        CHECK_NOT_NULL(graph);

        try
        {
            auto& network = graph->mGraph;
            if (is_test)
            {
                network->forwardPassTesting();
            }
            else
            {
                network->forwardPassTraining();
            }
        }
        catch (std::exception& e)
        {
            _lastError = e.what();
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS network_backward(Graph_t* graph)
    {
        CHECK_NOT_NULL(graph);

        try
        {
            auto& network = graph->mGraph;
            network->backwardPassTraining();
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS add_nin_model(Graph_Description_t* desc, const char* data_tensor_name, const char* labels_tensor_name, const char* loss_tensor_name, const NIN_hyperparams_t* hyperparams)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(data_tensor_name);
        CHECK_STRING(labels_tensor_name);
        CHECK_STRING(loss_tensor_name);
        CHECK_NOT_NULL(hyperparams);

        try
        {
            desc->mDef->add<raul::Convolution2DLayer>(
                "conv1",
                raul::Convolution2DParams{ { data_tensor_name }, { "conv1" }, hyperparams->conv1_kernel_size, hyperparams->conv1_filters, hyperparams->conv1_stride, hyperparams->conv1_padding });
            desc->mDef->add<raul::ReLUActivation>("relu1", raul::BasicParams{ { "conv1" }, { "relu1" } });
            desc->mDef->add<raul::Convolution2DLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, hyperparams->conv2_kernel_size, hyperparams->conv2_filters });
            desc->mDef->add<raul::ReLUActivation>("relu2", raul::BasicParams{ { "conv2" }, { "relu2" } });
            desc->mDef->add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, hyperparams->conv3_kernel_size, hyperparams->conv3_filters });
            desc->mDef->add<raul::ReLUActivation>("relu3", raul::BasicParams{ { "conv3" }, { "relu3" } });
            desc->mDef->add<raul::MaxPoolLayer2D>("mp", raul::Pool2DParams{ { "relu3" }, { "mp" }, hyperparams->maxpool_kernel, hyperparams->maxpool_stride, hyperparams->maxpool_padding });
            desc->mDef->add<raul::DropoutLayer>("drop1", raul::DropoutParams{ { "mp" }, { "drop1" }, 0.5f });

            desc->mDef->add<raul::Convolution2DLayer>(
                "conv4", raul::Convolution2DParams{ { "drop1" }, { "conv4" }, hyperparams->conv4_kernel_size, hyperparams->conv4_filters, hyperparams->conv4_stride, hyperparams->conv4_padding });
            desc->mDef->add<raul::ReLUActivation>("relu4", raul::BasicParams{ { "conv4" }, { "relu4" } });
            desc->mDef->add<raul::Convolution2DLayer>("conv5", raul::Convolution2DParams{ { "relu4" }, { "conv5" }, hyperparams->conv5_kernel_size, hyperparams->conv5_filters });
            desc->mDef->add<raul::ReLUActivation>("relu5", raul::BasicParams{ { "conv5" }, { "relu5" } });
            desc->mDef->add<raul::Convolution2DLayer>("conv6", raul::Convolution2DParams{ { "relu5" }, { "conv6" }, hyperparams->conv6_kernel_size, hyperparams->conv6_filters });
            desc->mDef->add<raul::ReLUActivation>("relu6", raul::BasicParams{ { "conv6" }, { "relu6" } });
            desc->mDef->add<raul::AveragePoolLayer>("avg1", raul::Pool2DParams{ { "relu6" }, { "avg1" }, hyperparams->avgpool1_kernel, hyperparams->avgpool1_stride, hyperparams->avgpool1_padding });
            desc->mDef->add<raul::DropoutLayer>("drop2", raul::DropoutParams{ { "avg1" }, { "drop2" }, 0.5f });

            desc->mDef->add<raul::Convolution2DLayer>(
                "conv7", raul::Convolution2DParams{ { "drop2" }, { "conv7" }, hyperparams->conv7_kernel_size, hyperparams->conv7_filters, hyperparams->conv7_stride, hyperparams->conv7_padding });
            desc->mDef->add<raul::ReLUActivation>("relu7", raul::BasicParams{ { "conv7" }, { "relu7" } });
            desc->mDef->add<raul::Convolution2DLayer>("conv8", raul::Convolution2DParams{ { "relu7" }, { "conv8" }, hyperparams->conv8_kernel_size, hyperparams->conv8_filters });
            desc->mDef->add<raul::ReLUActivation>("relu8", raul::BasicParams{ { "conv8" }, { "relu8" } });
            desc->mDef->add<raul::Convolution2DLayer>("conv9", raul::Convolution2DParams{ { "relu8" }, { "conv9" }, hyperparams->conv9_kernel_size, hyperparams->conv9_filters });
            desc->mDef->add<raul::ReLUActivation>("relu9", raul::BasicParams{ { "conv9" }, { "relu9" } });
            desc->mDef->add<raul::AveragePoolLayer>("avg2", raul::Pool2DParams{ { "relu9" }, { "avg2" }, hyperparams->avgpool2_kernel, hyperparams->avgpool2_stride });

            desc->mDef->add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "avg2" }, { "softmax" } });
            desc->mDef->add<raul::NLLLoss>(loss_tensor_name, raul::LossParams{ { "softmax", labels_tensor_name }, { loss_tensor_name }, "batch_mean" });
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS
    add_mobilenetv2_model(Graph_Description_t* desc, const char* data_tensor_name, const char* labels_tensor_name, const char* loss_tensor_name, const MobileNetV2_hyperparams_t* hyperparams)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(data_tensor_name);
        CHECK_STRING(labels_tensor_name);
        CHECK_STRING(loss_tensor_name);
        CHECK_NOT_NULL(hyperparams);

        try
        {

            const bool bias = true;

            // 0
            desc->mDef->add<raul::Convolution2DLayer>("conv1", raul::Convolution2DParams{ { data_tensor_name }, { "conv1" }, 3, 32, 2, 1, bias });
            desc->mDef->add<raul::BatchNormLayer>("bn1", raul::BatchnormParams{ { "conv1" }, { "bn1" }, hyperparams->bnMomentum, 1e-5f });
            desc->mDef->add<raul::ReLU6Activation>("relu1", raul::BasicParams{ { "bn1" }, { "relu1" } });

            // 1
            desc->mDef->add<raul::ConvolutionDepthwiseLayer>("conv2", raul::Convolution2DParams{ { "relu1" }, { "conv2" }, 3, 32, 1, 1, bias });
            desc->mDef->add<raul::BatchNormLayer>("bn2", raul::BatchnormParams{ { "conv2" }, { "bn2" }, hyperparams->bnMomentum, 1e-5f });
            desc->mDef->add<raul::ReLU6Activation>("relu2", raul::BasicParams{ { "bn2" }, { "relu2" } });

            desc->mDef->add<raul::Convolution2DLayer>("conv3", raul::Convolution2DParams{ { "relu2" }, { "conv3" }, 1, 16, 1, 0, bias });
            desc->mDef->add<raul::BatchNormLayer>("bn3", raul::BatchnormParams{ { "conv3" }, { "bn3" }, hyperparams->bnMomentum, 1e-5f });

            std::string inputName = "bn3";

            size_t layerIndex = 4;

            for (size_t w = 0; w < hyperparams->reproduceLayers; ++w)
            {
                desc->mDef->add<raul::Convolution2DLayer>("conv" + Conversions::toString(layerIndex),
                                                          raul::Convolution2DParams{ { inputName }, { "conv" + Conversions::toString(layerIndex) }, 1, hyperparams->filterSizes[w][0], 1, 0, bias });
                desc->mDef->add<raul::BatchNormLayer>(
                    "bn" + Conversions::toString(layerIndex),
                    raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, hyperparams->bnMomentum, 1e-5f });
                desc->mDef->add<raul::ReLU6Activation>("relu" + Conversions::toString(layerIndex),
                                                       raul::BasicParams{ { "bn" + Conversions::toString(layerIndex) }, { "relu" + Conversions::toString(layerIndex) } });

                ++layerIndex;

                desc->mDef->add<raul::ConvolutionDepthwiseLayer>(
                    "conv" + Conversions::toString(layerIndex),
                    raul::Convolution2DParams{
                        { "relu" + Conversions::toString(layerIndex - 1) }, { "conv" + Conversions::toString(layerIndex) }, 3, hyperparams->filterSizes[w][1], hyperparams->strideSizes[w], 1, bias });
                desc->mDef->add<raul::BatchNormLayer>(
                    "bn" + Conversions::toString(layerIndex),
                    raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, hyperparams->bnMomentum, 1e-5f });
                desc->mDef->add<raul::ReLU6Activation>("relu" + Conversions::toString(layerIndex),
                                                       raul::BasicParams{ { "bn" + Conversions::toString(layerIndex) }, { "relu" + Conversions::toString(layerIndex) } });

                ++layerIndex;

                desc->mDef->add<raul::Convolution2DLayer>(
                    "conv" + Conversions::toString(layerIndex),
                    raul::Convolution2DParams{ { "relu" + Conversions::toString(layerIndex - 1) }, { "conv" + Conversions::toString(layerIndex) }, 1, hyperparams->filterSizes[w][2], 1, 0, bias });
                desc->mDef->add<raul::BatchNormLayer>(
                    "bn" + Conversions::toString(layerIndex),
                    raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, hyperparams->bnMomentum, 1e-5f });

                if (hyperparams->residual[w])
                {
                    desc->mDef->add<raul::ElementWiseSumLayer>("sum" + Conversions::toString(layerIndex),
                                                               raul::ElementWiseLayerParams{ { "bn" + Conversions::toString(layerIndex), inputName }, { "sum" + Conversions::toString(layerIndex) } });
                    inputName = "sum" + Conversions::toString(layerIndex);
                }
                else
                    inputName = "bn" + Conversions::toString(layerIndex);

                ++layerIndex;
            }

            // 18
            desc->mDef->add<raul::Convolution2DLayer>("conv" + Conversions::toString(layerIndex),
                                                      raul::Convolution2DParams{ { inputName }, { "conv" + Conversions::toString(layerIndex) }, 1, hyperparams->lastLayerSize, 1, 0, bias });
            desc->mDef->add<raul::BatchNormLayer>(
                "bn" + Conversions::toString(layerIndex),
                raul::BatchnormParams{ { "conv" + Conversions::toString(layerIndex) }, { "bn" + Conversions::toString(layerIndex) }, hyperparams->bnMomentum, 1e-5f });
            desc->mDef->add<raul::ReLU6Activation>("relu" + Conversions::toString(layerIndex),
                                                   raul::BasicParams{ { "bn" + Conversions::toString(layerIndex) }, { "relu" + Conversions::toString(layerIndex) } });

            desc->mDef->add<raul::AveragePoolLayer>("avg", raul::Pool2DParams{ { "relu" + Conversions::toString(layerIndex) }, { "avg" }, hyperparams->avgWidth, 1 });
            desc->mDef->add<raul::ReshapeLayer>("reshape", raul::ViewParams{ "avg", "avgr", 1, 1, -1 });
            desc->mDef->add<raul::LinearLayer>("fc", raul::LinearParams{ "avgr", "fc", hyperparams->num_classes, bias });
            desc->mDef->add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "fc" }, { "softmax" } });
            desc->mDef->add<raul::NLLLoss>(loss_tensor_name, raul::LossParams{ { "softmax", labels_tensor_name }, { loss_tensor_name }, "batch_mean" });
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS
    add_resnet18_model(Graph_Description_t* desc, const char* data_tensor_name, const char* labels_tensor_name, const char* loss_tensor_name, const ResNet18_hyperparams_t* hyperparams)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(data_tensor_name);
        CHECK_STRING(labels_tensor_name);
        CHECK_STRING(loss_tensor_name);
        CHECK_NOT_NULL(hyperparams);

        try
        {
            const auto output_name = build_resnet18(*desc->mDef, data_tensor_name);
            desc->mDef->add<raul::LogSoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { output_name }, { "softmax" } });
            desc->mDef->add<raul::NLLLoss>(loss_tensor_name, raul::LossParams{ { "softmax", labels_tensor_name }, { loss_tensor_name }, "batch_mean" });
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }

    API_EXPORT API_STATUS add_bert_model(Graph_Description_t* desc,
                                         const char* name,
                                         const char* input_ids,
                                         const char* token_type_ids,
                                         const char* attention_mask,
                                         const char** output_names,
                                         size_t outputs_count,
                                         unsigned int vocab_size,
                                         unsigned int type_vocab_size,
                                         unsigned int num_hidden_layers,
                                         unsigned int hidden_size,
                                         unsigned int intermediate_size,
                                         unsigned int num_heads,
                                         unsigned int max_position_embeddings,
                                         const char* hidden_activation,
                                         float hidden_dropout,
                                         float attention_dropout)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_NOT_NULL(input_ids);
        CHECK_NOT_NULL(output_names);
        CHECK_PRECONDITION(outputs_count > 0);
        CHECK_STRING(hidden_activation);

        raul::BERTModel(name,
                        raul::BERTParams{ { input_ids, token_type_ids ? token_type_ids : "", attention_mask ? attention_mask : "" },
                                          raul::Names(output_names, output_names + outputs_count),
                                          vocab_size,
                                          type_vocab_size,
                                          num_hidden_layers,
                                          hidden_size,
                                          intermediate_size,
                                          num_heads,
                                          max_position_embeddings,
                                          hidden_activation,
                                          hidden_dropout,
                                          attention_dropout },
                        desc->mDef->getNetworkParameters());

        return STATUS_OK;
    }

    API_EXPORT API_STATUS
    add_data_layer_with_labels(Graph_Description_t* desc, const char* name, const char** output_names, size_t output_count, size_t depth, size_t height, size_t width, size_t labels_count)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_NOT_NULL(output_names);
        CHECK_PRECONDITION(output_count > 0);

        CHECK_PRECONDITION(width > 0);
        CHECK_PRECONDITION(height > 0);
        CHECK_PRECONDITION(depth > 0);
        CHECK_PRECONDITION(labels_count > 0);

        return addLayer<raul::DataLayer>(desc, name, raul::DataParams{ raul::Names(output_names, output_names + output_count), depth, height, width, labels_count });
    }

    API_EXPORT API_STATUS add_data_layer(Graph_Description_t* desc, const char* name, const char** output_names, size_t output_count, size_t depth, size_t height, size_t width)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_NOT_NULL(output_names);
        CHECK_PRECONDITION(output_count > 0);

        CHECK_PRECONDITION(width > 0);
        CHECK_PRECONDITION(height > 0);
        CHECK_PRECONDITION(depth > 0);

        return addLayer<raul::DataLayer>(desc, name, raul::DataParams{ raul::Names(output_names, output_names + output_count), depth, height, width });
    }

    API_EXPORT API_STATUS add_embedding_layer(Graph_Description_t* desc,
                                              const char* name,
                                              const char* input_name,
                                              const char* output_name,
                                              size_t dictionary_size,
                                              size_t embedding_size,
                                              int padding_idx,
                                              bool scale_by_size,
                                              bool scale_grad_by_frequency)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_STRING(input_name);
        CHECK_STRING(output_name);
        CHECK_PRECONDITION(dictionary_size > 0);
        CHECK_PRECONDITION(embedding_size > 0);
        CHECK_PRECONDITION(padding_idx >= -1);

        return addLayer<raul::Embedding>(
            desc, name, raul::EmbeddingParams{ raul::Name(input_name), raul::Name(output_name), dictionary_size, embedding_size, padding_idx, scale_by_size, scale_grad_by_frequency });
    }

    API_EXPORT API_STATUS add_labels(Graph_Description_t* desc, const char* name, const char** output_names, size_t output_count, size_t labels_count)
    {
        return add_data_layer(desc, name, output_names, output_count, 1u, 1u, labels_count);
    }

    API_EXPORT API_STATUS print_graph(Graph_t* graph)
    {
        CHECK_NOT_NULL(graph);

        try
        {
            graph->mGraph->printInfo(std::cout);
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS
    add_loss_layer(Graph_Description_t* desc, const char* name, const char** input_names, const char* loss_name, const char* loss_type, size_t inputs_count, LOSS_REDUCTION reduction)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(loss_type);
        CHECK_PRECONDITION(std::string_view(loss_type) == CROSS_ENTROPY_LOSS || std::string_view(loss_type) == NLL_LOSS || std::string_view(loss_type) == KL_DIV_LOSS ||
                           std::string_view(loss_type) == MSE_LOSS || std::string_view(loss_type) == L1_LOSS);
        CHECK_STRING(name);
        CHECK_PRECONDITION(input_names);
        CHECK_PRECONDITION(inputs_count > 0);

        raul::LossParams::Reduction red = raul::LossParams::Reduction::Mean;
        switch (reduction)
        {
            case LOSS_REDUCTION_MEAN:
                red = raul::LossParams::Reduction::Mean;
                break;
            case LOSS_REDUCTION_SUM:
                red = raul::LossParams::Reduction::Sum;
                break;
            case LOSS_REDUCTION_BATCH_MEAN:
                red = raul::LossParams::Reduction::Batch_Mean;
                break;
        }

        if (std::string_view(loss_type) == CROSS_ENTROPY_LOSS)
        {
            return addLayer<raul::CrossEntropyLoss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
        }
        else if (std::string_view(loss_type) == NLL_LOSS)
        {
            return addLayer<raul::NLLLoss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
        }
        else if (std::string_view(loss_type) == KL_DIV_LOSS)
        {
            return addLayer<raul::KLDivLoss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
        }
        else if (std::string_view(loss_type) == MSE_LOSS)
        {
            return addLayer<raul::MSELoss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
        }

        return addLayer<raul::L1Loss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
    }

    API_EXPORT API_STATUS add_lstm_layer(Graph_Description_t* desc, const char* name, const char* input_name, const char* output_name, size_t hidden_size, bool use_bias)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_STRING(input_name);
        CHECK_STRING(output_name);
        CHECK_PRECONDITION(hidden_size > 0);

        raul::LSTMLayer layer(name, raul::LSTMParams{ { input_name }, { output_name }, hidden_size, use_bias }, desc->mDef->getNetworkParameters());

        return STATUS_OK;
    }

    API_EXPORT API_STATUS add_lstm_layer_ext(Graph_Description_t* desc,
                                             const char* name,
                                             const char* input_name,
                                             const char* hidden_input_name,
                                             const char* cell_input_name,
                                             const char* output_name,
                                             const char* hidden_output_name,
                                             const char* cell_output_name,
                                             bool use_bias)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_STRING(input_name);
        CHECK_STRING(hidden_input_name);
        CHECK_STRING(cell_input_name);
        CHECK_STRING(output_name);
        CHECK_STRING(hidden_output_name);
        CHECK_STRING(cell_output_name);

        raul::LSTMLayer layer(
            name, raul::LSTMParams{ input_name, hidden_input_name, cell_input_name, output_name, hidden_output_name, cell_output_name, use_bias }, desc->mDef->getNetworkParameters());

        return STATUS_OK;
    }

    API_EXPORT API_STATUS print_graph_to_file(Graph_t* graph, FILE* file)
    {
        CHECK_NOT_NULL(graph);
        CHECK_NOT_NULL(file);

        try
        {
            std::stringstream s;
            graph->mGraph->printInfo(s);
            std::string str = s.str();
            fputs(str.c_str(), file);
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS print_graph_to_string(Graph_t* graph, char* string, size_t* length)
    {
        CHECK_NOT_NULL(graph);
        CHECK_PRECONDITION(string != nullptr || (string == nullptr && length != nullptr));

        try
        {
            std::stringstream s;
            graph->mGraph->printInfo(s);
            std::string str = s.str();

            if (!string)
            {
                *length = str.size();
            }
            else
            {
                strncpy(string, str.c_str(), str.size());
                string[str.size()] = '\0';
            }
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS add_yannet_model(Graph_Description_t* desc, const char* query_tensor_name, const char* key_tensor_name, const char* initial_hidden_tensor_name, const char* labels_tensor_name, const char* loss_tensor_name)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(query_tensor_name);
        CHECK_STRING(key_tensor_name);
        CHECK_STRING(initial_hidden_tensor_name);
        CHECK_STRING(labels_tensor_name);
        CHECK_STRING(loss_tensor_name);
        //CHECK_NOT_NULL(hyperparams);

        try
        {
            bool fusedGRU = true;
            size_t hiddenSize = 32;
            raul::GRULayer("rnn", raul::GRUParams{ { key_tensor_name, initial_hidden_tensor_name }, { "gru_out", "new_hidden" }, true, false, fusedGRU }, desc->mDef->getNetworkParameters());

            // Attention
            desc->mDef->add<raul::LinearLayer>(raul::Name("att") / "key_layer", raul::LinearParams{ "gru_out", "key_layer_out", hiddenSize });
            desc->mDef->add<raul::LinearLayer>(raul::Name("att") / "query_layer", raul::LinearParams{ query_tensor_name, "query_layer_out", hiddenSize });

            desc->mDef->add<raul::ElementWiseSumLayer>("add", raul::ElementWiseLayerParams{ {"query_layer_out", "key_layer_out"}, "h" });
            desc->mDef->add<raul::LayerNorm2DLayer>(raul::Name("att") / "norm", raul::LayerNormParams{ "h", "h_norm" });
            desc->mDef->add<raul::LeakyReLUActivation>("relu", raul::LeakyReLUParams{ "h_norm", "h_relu" });
            desc->mDef->add<raul::LinearLayer>(raul::Name("att") / "attention_layer", raul::LinearParams{ "h_relu", "attn_layer_out", 1 });
            desc->mDef->add<raul::SoftMaxActivation>("softmax", raul::BasicParamsWithDim{ { "attn_layer_out" }, { "h_softmax" }, "height" });

            desc->mDef->add<raul::ElementWiseMulLayer>("mul", raul::ElementWiseLayerParams{ { "h_softmax", "key_layer_out" }, "mul"});
            desc->mDef->add<raul::ReduceSumLayer>("reduce", raul::BasicParamsWithDim{ { "mul" }, { "mul_reduced" }, "height" });

            desc->mDef->add<raul::LinearLayer>("fc", raul::LinearParams{ "mul_reduced", "y", 1 });
            desc->mDef->add<raul::SigmoidActivation>("sigmoid", raul::BasicParamsWithDim{ { "y" }, { "out" } });
            desc->mDef->add<raul::BinaryCrossEntropyLoss>("bce", raul::LossParams{ { "out", labels_tensor_name }, { loss_tensor_name } });
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }
}
