// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WORKFLOW_EAGER_H
#define WORKFLOW_EAGER_H

#include "Workflow.h"

namespace raul
{

/**
 * @brief No memory optimizations performed, all tensors allocated once, external control of layers execution possible
 */
class WorkflowEager : public Workflow
{
  public:
    explicit WorkflowEager(CompressionMode compressionMode = CompressionMode::NONE,
                           CalculationMode calculationMode = CalculationMode::DETERMINISTIC,
                           AllocationMode allocationMode = AllocationMode::STANDARD,
                           ExecutionTarget executionTarget = ExecutionTarget::CPU,
                           bool useCompiler = false,
                           quantization::IQuantizer* quantizer = nullptr)
        : Workflow(compressionMode, calculationMode, allocationMode, executionTarget, useCompiler, quantizer)
    {
    }
    ~WorkflowEager(){}
    /**
     * @brief Create tensors and allocate memory for not optimized not batched tensors, create shapes for optimized not batched tensors
     */
    void prepareMemoryForTraining() override;

    /**
     * @brief Create pipelines for training process
     * execution - ignored in eager mode
     *
     * @note(ck): Default arguments on override methods are prohibited
     */
    void preparePipelines(Execution execution = Execution::Normal) override;

    /**
     * @brief Execute forward test pipeline
     */
    void forwardPassTesting() override;

    /**
     * @brief Execute forward train pipeline
     * performZero - ignored in eager mode
     *
     * @note(ck): Default arguments on override methods are prohibited
     */
    void forwardPassTraining(bool performZero = true) override;

    /**
     * @brief Execute backward train pipeline
     */
    void backwardPassTraining() override;

  private:
    void createAuxPipelines();

    void execTargetCreateAuxPipelines(const std::unordered_map<Name, size_t>& uniqueTensors);
};

// for test purposes
#define MANAGERS_DEFINE                                                                                                                                                                                \
    raul::WorkflowEager work;                                                                                                                                                                          \
    [[maybe_unused]] raul::MemoryManager& memory_manager = work.getMemoryManager();

} // namespace raul

#endif // WORKFLOW_EAGER_H
