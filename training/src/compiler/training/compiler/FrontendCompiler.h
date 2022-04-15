// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef FRONTEND_COMPILER_H
#define FRONTEND_COMPILER_H

#include "Workflow.h"
#include "WorkflowBuilder.h"
#include <training/base/common/Common.h>
#include <training/base/layers/basic/DataLayer.h>
#include <training/base/loss/CrossEntropyLoss.h>
#include <training/frontend/Frontend.h>

#include <sstream>

namespace raul
{

struct FrontendCompiler
{
    enum class LossType
    {
        CrossEntropy
    };

    void setTopology(const frontend::Generator& init) { topology = init; }

    void addLoss(const LossType& lossType, const frontend::PortNames& inputs)
    {
        auto pair = std::make_pair(lossType, inputs);
        losses.push_back(pair);
    }

    auto& getWorkflow() { return work; }

    void compile(const std::initializer_list<size_t> shape = { 1, 1, 1, 1 })
    {
        std::unordered_map<Name, std::initializer_list<size_t>> init;

        for (const auto& input : topology.getInputs())
        {
            init[input] = shape;
        }
        compile(init);
    }

    void compile(std::unordered_map<Name, std::initializer_list<size_t>> shapeDict)
    {
        std::optional<size_t> batchSize;
        for (const auto& input : topology.getInputs())
        {
            const auto shape = shapeDict[input];
            if (std::distance(shape.begin(), shape.end()) != 4)
            {
                THROW_NONAME("FrontendCompiler", "core supports only 4 dimensional tensors")
            }

            if (!batchSize)
            {
                batchSize = *shape.begin();
            }
            work.add<DataLayer>(input, DataParams{ { input }, *(shape.begin() + 1), *(shape.begin() + 2), *(shape.begin() + 3) });
        }

        WorkflowBuilder builder{ work };
        topology.apply(builder);

        size_t lossIdx = 0;
        for (const auto& [lossType, inputs] : losses)
        {
            std::stringstream ss;
            ss << "loss" << lossIdx;
            auto realInputs = builder.getSourcePorts(std::nullopt, inputs);
            Names realInputsNames(realInputs.size());
            std::transform(realInputs.cbegin(), realInputs.cend(), realInputsNames.begin(), [](const frontend::Name& x) { return Name{ x }; });

            switch (lossType)
            {
                case LossType::CrossEntropy:
                    LossWrapperFunction<raul::CrossEntropyLoss>(ss.str(), raul::LossParams{ realInputsNames, Names{ ss.str() }, "batch_mean" }, work);
                    break;
            }
            ++lossIdx;
        }

        work.preparePipelines();
        work.setBatchSize(batchSize ? *batchSize : 1);
        work.prepareMemoryForTraining();
    }

  private:
    std::vector<std::pair<LossType, frontend::PortNames>> losses;
    frontend::Generator topology;
    Workflow work;
};

} // raul namespace

#endif // FRONTEND_COMPILER_H