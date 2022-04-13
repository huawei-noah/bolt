// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>

#include <training/base/common/Common.h>
#include <training/base/common/MemoryManager.h>

#include "IOptimizer.h"

namespace raul::optimizers
{
/**
 * @brief Optimizer base class
 */
struct Optimizer : IOptimizer
{
    Optimizer() = default;
    Optimizer(const Optimizer& other) = delete;
    Optimizer(const Optimizer&& other) = delete;
    Optimizer& operator=(const Optimizer& other) = delete;
    Optimizer& operator=(const Optimizer&& other) = delete;

    /**
     * Call-method of an optimizer that modifies parameters with gradients.
     * @param memory_manager Memory manager reference
     * @param param Tensor of parameters.
     * @param grad Tensor of gradients.
     */
    void operator()(MemoryManager& memory_manager, Tensor& param, Tensor& grad) override;
    void operator()(MemoryManagerFP16& memory_manager, TensorFP16& param, TensorFP16& grad) override;
    void setLearningRate(dtype) override { THROW_NONAME("Optimizer", "Optimizer doesn't have configurable learning rate"); }
    [[nodiscard]] dtype getLearningRate() override { THROW_NONAME("Optimizer", "Optimizer doesn't have configurable learning rate"); }

    friend std::ostream& operator<<(std::ostream& out, const Optimizer& instance) { return instance.as_ostream(out); }

  private:
    virtual void optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad) = 0;
    virtual void optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad);
    virtual std::ostream& as_ostream(std::ostream& out) const = 0;

  private:
};

} // raul::optimizers

#endif // OPTIMIZER_H