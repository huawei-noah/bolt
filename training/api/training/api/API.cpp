// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "API.h"

#include <iostream>
#include <string.h>
#include <type_traits>

#include <training/api/API.h>
#include <training/base/common/Common.h>
#include <training/base/initializers/ConstantInitializer.h>
#include <training/base/initializers/IInitializer.h>
#include <training/base/initializers/RandomNormInitializer.h>
#include <training/base/initializers/RandomUniformInitializer.h>
#include <training/base/initializers/XavierInitializer.h>
#include <training/base/layers/BasicLayer.h>
#include <training/base/layers/composite/rnn/LSTMLayer.h>
#include <training/base/layers/parameters/LayerParameters.h>

#include <training/compiler/Layers.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/Adadelta.h>
#include <training/base/optimizers/Adagrad.h>
#include <training/base/optimizers/Adam.h>
#include <training/base/optimizers/Adamax.h>
#include <training/base/optimizers/Momentum.h>
#include <training/base/optimizers/Nesterov.h>
#include <training/base/optimizers/Optimizer.h>
#include <training/base/optimizers/SGD.h>
#include <training/base/optimizers/AdamW.h>
#include <training/compiler/Compiler.h>

#include <training/base/optimizers/schedulers/LrScheduler.h>
#include <training/base/optimizers/schedulers/strategies/CosineAnnealing.h>

#include "lowlevel/APIChecks.h"
#include "lowlevel/APIDefinitions.h"

namespace
{

using namespace raul;

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

} // anonymous

extern "C"
{
    using namespace raul;
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
    API_EXPORT void set_last_error(const char* err) 
    { 
        if (!err) 
        { 
            std::cout << "null exception" << std::endl;
            return; 
        }
        _lastError = err; 
        std::cout << err << std::endl; 
    }

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

    API_EXPORT API_STATUS create_graph_description_fp16(Graph_Description_t** descr)
    {
        CHECK_NOT_NULL(descr);
        try
        {
            *descr = new Graph_Description_t(raul::ExecutionTarget::CPUFP16);
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

    API_EXPORT API_STATUS create_graph_description_compiler(Graph_Description_t** descr)
    {
        CHECK_NOT_NULL(descr);
        try
        {
            *descr = new Graph_Description_t(raul::ExecutionTarget::CPU, true);
        }
        catch (std::exception& e)
        {
            *descr = NULL;
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }


    API_EXPORT API_STATUS create_graph_description_compiler_fp16(Graph_Description_t** descr)
    {
        CHECK_NOT_NULL(descr);
        try
        {
            *descr = new Graph_Description_t(raul::ExecutionTarget::CPUFP16, true);
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

    API_STATUS get_tensor_size(Graph_t* graph, const char* tensor_name, size_t* size)
    {
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);
        CHECK_PRECONDITION(size);

        API_STATUS status = STATUS_ERROR;
        try
        {
            if (graph->mGraph->getMemoryManager().tensorExists(tensor_name))
            {
                *size = graph->mGraph->getMemoryManager()[tensor_name].size();
                status = STATUS_OK;
            }
            else if (graph->mGraph->getMemoryManager<raul::MemoryManagerFP16>().tensorExists(tensor_name))
            {
                *size = graph->mGraph->getMemoryManager<raul::MemoryManagerFP16>()[tensor_name].size();
                status = STATUS_OK;
            }
            else
            {
                return STATUS_ERROR_BAD_NAME;
            }
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
        }
        return status;
    }

    API_EXPORT API_STATUS get_tensor(Graph_t* graph, const char* tensor_name, FLOAT_TYPE* data, size_t* size)
    {
        CHECK_NOT_NULL(graph);
        CHECK_STRING(tensor_name);
        CHECK_PRECONDITION(size);

        API_STATUS status = STATUS_ERROR;
        try
        {
            size_t sz = 0;
            auto status_get = get_tensor_size(graph, tensor_name, &sz);
            if (status_get != STATUS_OK)
            {
                return status_get;
            }
            
            if (!data)
            {
                *size = sz;
                return STATUS_OK;
            }
            else if (*size == sz)
            {
                if (graph->mGraph->getMemoryManager().tensorExists(tensor_name))
                {
                    const auto& tensor = graph->mGraph->getMemoryManager()[tensor_name];
                    std::copy(tensor.begin(), tensor.end(), data);
                }
                else if (graph->mGraph->getMemoryManager<raul::MemoryManagerFP16>().tensorExists(tensor_name))
                {
                    const auto& tensor = graph->mGraph->getMemoryManager<raul::MemoryManagerFP16>()[tensor_name];
                    std::transform(tensor.begin(), tensor.end(), data, [](const auto v16) { return raul::toFloat32(v16); });
                }
                status = STATUS_OK;
            }
            else
            {
                return STATUS_ERROR_BAD_SIZE;
            }
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
            if (graph->mGraph->getMemoryManager().tensorExists(tensor_name))
            {
                graph->mGraph->getMemoryManager()[std::string(tensor_name)] = raul::Tensor::dt_range(data, data + size);
            }
            else if (graph->mGraph->getMemoryManager<raul::MemoryManagerFP16>().tensorExists(tensor_name))
            {
                auto& tensor = graph->mGraph->getMemoryManager<raul::MemoryManagerFP16>()[tensor_name];
                std::transform(data, data + size, tensor.begin(), [](const auto v32) { return raul::toFloat16(v32); });
            }
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
            network->forwardPassTraining();

            for (size_t i = 0; i < loss_count; ++i)
            {
                float totalLoss = 0.f;
                size_t sz = 1;
                get_tensor(graph, loss_names[i], &totalLoss, &sz);
                loss[i] = static_cast<FLOAT_TYPE>(totalLoss);
            }
            network->backwardPassTraining();
            auto paramNames = network->getTrainableParameterNames();
            auto& mm = network->getMemoryManager();
            auto& mm16 = network->getMemoryManager<raul::MemoryManagerFP16>();
            for (auto& name : paramNames)
            {
                if (mm.tensorExists(name))
                {
                    opt(mm, mm[name], mm[name.grad()]);
                }
                else if (mm16.tensorExists(name))
                {
                    opt(mm16, mm16[name], mm16[name.grad()]);
                }
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

            network->forwardPassTraining();
            for (size_t i = 0; i < loss_count; ++i)
            {
                float totalLoss = 0.f;
                size_t sz = 1;
                FORWARD_ERROR(get_tensor(graph, loss_names[i], &totalLoss, &sz));
                loss[i] = static_cast<FLOAT_TYPE>(totalLoss);
            }
            network->backwardPassTraining();
            sch.step();
            auto paramNames = network->getTrainableParameterNames();
            auto& mm = network->getMemoryManager();
            auto& mm16 = network->getMemoryManager<raul::MemoryManagerFP16>();
            for (auto& name : paramNames)
            {
                if (mm.tensorExists(name))
                {
                    sch(mm, mm[name], mm[name.grad()]);
                }
                else if (mm16.tensorExists(name))
                {
                    sch(mm16, mm16[name], mm16[name.grad()]);
                }
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

    API_EXPORT API_STATUS add_transpose_layer(Graph_Description_t* desc,
                                              const char* name,
                                              const char* input_name,
                                              const char* output_name,
                                              DIM from,
                                              DIM to)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_STRING(input_name);
        CHECK_STRING(output_name);

        return addLayer<raul::TransposeLayer>(
            desc, name, raul::TransposingParams{ raul::Name(input_name), raul::Name(output_name), static_cast<raul::Dimension>(from), static_cast<raul::Dimension>(to) });
    }

    API_EXPORT API_STATUS
    add_reshape_layer(Graph_Description_t* desc, const char* name, const char* input_name, const char* output_name, int new_depth, int new_height, int new_width)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_STRING(input_name);
        CHECK_STRING(output_name);

        return addLayer<raul::ReshapeLayer>(
            desc, name, raul::ViewParams{ raul::Name(input_name), raul::Name(output_name), new_depth, new_height, new_width }); 
    }

    API_EXPORT API_STATUS
    add_loss_layer(Graph_Description_t* desc, const char* name, const char** input_names, const char* loss_name, const char* loss_type, size_t inputs_count, LOSS_REDUCTION reduction)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(loss_type);
        CHECK_PRECONDITION(std::string_view(loss_type) == CROSS_ENTROPY_LOSS || std::string_view(loss_type) == NLL_LOSS || std::string_view(loss_type) == KL_DIV_LOSS ||
                           std::string_view(loss_type) == MSE_LOSS || std::string_view(loss_type) == L1_LOSS || std::string_view(loss_type) == BINARY_CROSS_ENTROPY_LOSS ||
                           std::string_view(loss_type) == SOFTMAX_CROSS_ENTROPY_LOSS);
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
        else if (std::string_view(loss_type) == BINARY_CROSS_ENTROPY_LOSS)
        {
            return addLayer<raul::BinaryCrossEntropyLoss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
        }
        else if (std::string_view(loss_type) == SOFTMAX_CROSS_ENTROPY_LOSS)
        {
            return addLayer<raul::SoftmaxCrossEntropyLoss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
        }
        return addLayer<raul::L1Loss>(desc, name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red });
    }

    API_EXPORT API_STATUS
    add_loss_layer_with_compiler(Graph_Description_t* desc, const char* name, const char** input_names, const char* loss_name, const char* loss_type, size_t inputs_count, LOSS_REDUCTION reduction)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(loss_type);
        CHECK_PRECONDITION(std::string_view(loss_type) == CROSS_ENTROPY_LOSS || std::string_view(loss_type) == NLL_LOSS || std::string_view(loss_type) == KL_DIV_LOSS ||
                           std::string_view(loss_type) == MSE_LOSS || std::string_view(loss_type) == L1_LOSS || std::string_view(loss_type) == BINARY_CROSS_ENTROPY_LOSS);
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

        try
        {
            if (std::string_view(loss_type) == CROSS_ENTROPY_LOSS)
            {
                raul::LossWrapperFunction<raul::CrossEntropyLoss>(name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red }, *desc->mDef.get());
            }
            else if (std::string_view(loss_type) == NLL_LOSS)
            {
                raul::LossWrapperFunction<raul::NLLLoss>(name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red }, *desc->mDef.get());
            }
            else if (std::string_view(loss_type) == KL_DIV_LOSS)
            {
                raul::LossWrapperFunction<raul::KLDivLoss>(name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red }, *desc->mDef.get());
            }
            else if (std::string_view(loss_type) == MSE_LOSS)
            {
                raul::LossWrapperFunction<raul::MSELoss>(name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red }, *desc->mDef.get());
            }
            else if (std::string_view(loss_type) == BINARY_CROSS_ENTROPY_LOSS)
            {
                raul::LossWrapperFunction<raul::BinaryCrossEntropyLoss>(name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red }, *desc->mDef.get());
            }
            else
            {
                raul::LossWrapperFunction<raul::L1Loss>(name, raul::LossParams{ raul::Names(input_names, input_names + inputs_count), raul::Names(&loss_name, &loss_name + 1u), red }, *desc->mDef.get());
            }
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS add_lstm_layer(Graph_Description_t* desc, const char* name, const char* input_name, const char* output_name, size_t hidden_size, bool use_global_fusion, bool use_bias)
    {
        CHECK_NOT_NULL(desc);
        CHECK_STRING(name);
        CHECK_STRING(input_name);
        CHECK_STRING(output_name);
        CHECK_PRECONDITION(hidden_size > 0);

        raul::LSTMLayer layer(name, raul::LSTMParams{ { input_name }, { output_name }, hidden_size, use_global_fusion, use_bias }, desc->mDef->getNetworkParameters());

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
                                             bool use_global_fusion,
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
            name, raul::LSTMParams{ input_name, hidden_input_name, cell_input_name, output_name, hidden_output_name, cell_output_name, use_global_fusion, use_bias }, desc->mDef->getNetworkParameters());

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

    API_EXPORT API_STATUS set_constraint(Graph_Description_t* desc, const char* layer_name, CONSTRAINT_TYPE constraint)
    {
        CHECK_NOT_NULL(desc);
        CHECK_NOT_NULL(desc->mDef);
        CHECK_STRING(layer_name);

        try
        {
            desc->mDef->getCompiler().setConstraint(raul::Constraint(layer_name, static_cast<raul::ConstraintImpl>(constraint)));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS set_constraint_sequence(Graph_Description_t* desc, const char* layer_name_from, const char* layer_name_to, CONSTRAINT_TYPE constraint)
    {
        CHECK_NOT_NULL(desc);
        CHECK_NOT_NULL(desc->mDef);
        CHECK_STRING(layer_name_from);
        CHECK_STRING(layer_name_to);

        try
        {
            desc->mDef->getCompiler().setConstraint(raul::Constraint(layer_name_from, layer_name_to, static_cast<raul::ConstraintImpl>(constraint)));
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }

    API_EXPORT API_STATUS reset_layer_execution_target_override(Graph_Description_t* desc)
    {
        CHECK_NOT_NULL(desc);
        CHECK_NOT_NULL(desc->mDef);
        try
        {
            desc->mDef->resetLayerExecutionTargetOverride();
        }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }

        return STATUS_OK;
    }
    
    API_STATUS finetune_impl(Graph_t* graph, LrScheduler_t* lr_scheduler, const char* loss_name, [[maybe_unused]]size_t step)
    {
        raul::dtype loss = 0.0_dt;
	    return train_single_pass_with_scheduling(graph, lr_scheduler, &loss_name, 1, &loss);
    }
    
    API_EXPORT API_STATUS finetune([[maybe_unused]]Graph_t* graph, [[maybe_unused]]LrScheduler_t* lr_scheduler, const char** input_names, const FLOAT_TYPE** input_data, const int* input_sizes, const int input_num, [[maybe_unused]]const char* loss_name, [[maybe_unused]]size_t step)
    {
        try
	    {
            for (int i = 0; i < input_num; ++i)
	        {
                FORWARD_ERROR(set_tensor(graph, input_names[i], input_data[i], input_sizes[i]));
	        }
            FORWARD_ERROR(finetune_impl(graph, lr_scheduler, loss_name, step));
	    }
        catch (std::exception& e)
        {
            set_last_error(e.what());
            return STATUS_ERROR;
        }
        return STATUS_OK;
    }
}
