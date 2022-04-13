// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Regularizer.h"

namespace raul::optimizers::Regularizers
{

void Regularizer::addPenalty(Tensor& param, Tensor& grad)
{
    for (auto paramIter = param.begin(), gradIter = grad.begin(); paramIter != param.end(); ++paramIter, ++gradIter)
    {
        *gradIter += mRegularizerStrategy->operator()(*paramIter);
    }
}

void Regularizer::operator()(MemoryManager& memory_manager, Tensor& param, Tensor& grad)
{
    addPenalty(param, grad);
    mOptimizer->operator()(memory_manager, param, grad);
}

void Regularizer::operator()(MemoryManagerFP16&, TensorFP16&, TensorFP16&)
{
    // d.polubotko: implement
    THROW_NONAME("Regularizer[operator()]", "not implemented");
    // addPenalty(param, grad);
    // mOptimizer->operator()(memory_manager, param, grad);
}

raul::dtype Regularizer::getPenalty(Workflow& net) const
{
    return mRegularizerStrategy->getPenalty(net);
}

void Regularizer::setLearningRate(dtype lr)
{
    mOptimizer->setLearningRate(lr);
}

[[nodiscard]] dtype Regularizer::getLearningRate()
{
    return mOptimizer->getLearningRate();
}

std::ostream& Regularizer::as_ostream(std::ostream& out) const
{
    return out << "Regularizer" << std::endl;
}

} // raul::optimizers::Regularizers