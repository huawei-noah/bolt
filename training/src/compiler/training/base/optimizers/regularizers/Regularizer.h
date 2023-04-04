// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <iostream>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>
#include <training/compiler/Workflow.h>
#include <training/base/optimizers/IOptimizer.h>
#include <training/base/optimizers/regularizers/strategies/IRegularizerStrategy.h>

namespace raul::optimizers::Regularizers
{

/**
 * @brief The wrapper over Optimizer class allowing to use regularization during model training
 */
class Regularizer : public IOptimizer
{
  public:
    Regularizer(std::unique_ptr<raul::optimizers::Regularizers::Strategies::IRegularizerStrategy> regularizerStrategy, std::unique_ptr<raul::optimizers::IOptimizer> optimizer)
        : mRegularizerStrategy(std::move(regularizerStrategy))
        , mOptimizer(std::move(optimizer))
    {
    }

    raul::dtype getPenalty(Workflow& net) const;
    void operator()(MemoryManager& memory_manager, Tensor& param, Tensor& grad) override;
    void operator()(MemoryManagerFP16& memory_manager, TensorFP16& param, TensorFP16& grad) override;
    void setLearningRate(dtype lr) override;
    [[nodiscard]] dtype getLearningRate() final;
~ Regularizer(){}
  private:
    void addPenalty(Tensor& param, Tensor& grad);

    virtual std::ostream& as_ostream(std::ostream& out) const final;
    std::unique_ptr<raul::optimizers::Regularizers::Strategies::IRegularizerStrategy> mRegularizerStrategy;
    std::unique_ptr<raul::optimizers::IOptimizer> mOptimizer;
};

} // namespace raul::optimizers::Regularizers

#endif // REGULARIZER_H
