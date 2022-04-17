// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRADIENT_POSTPROCESSOR_H
#define GRADIENT_POSTPROCESSOR_H

#include <training/base/common/Common.h>
#include <training/base/common/Tensor.h>
#include <training/base/common/NetworkParameters.h>

namespace raul::postprocessing
{

/**
 * @brief Gradient Postprocessor interface
 */
struct GradientPostprocessor
{
    GradientPostprocessor() = default;
    virtual ~GradientPostprocessor() = default;

    GradientPostprocessor(const GradientPostprocessor& other) = delete;
    GradientPostprocessor(const GradientPostprocessor&& other) = delete;
    GradientPostprocessor& operator=(const GradientPostprocessor& other) = delete;
    GradientPostprocessor& operator=(const GradientPostprocessor&& other) = delete;

    /**
     * Call-method of an gradient postproccesor that modifies gradients.
     * @param trainableParams vector of tensors of parameters and gradients.
     */
    virtual void processGradients(std::vector<ParamAndGrad>& trainableParams, NetworkParameters& networkParameters) = 0;
    virtual void processGradients(std::vector<ParamAndGradImpl<TensorFP16>>& trainableParams, NetworkParameters& networkParameters) = 0;
    friend std::ostream& operator<<(std::ostream& out, const GradientPostprocessor& instance) { return instance.as_ostream(out); }

  private:
    virtual std::ostream& as_ostream(std::ostream& out) const = 0;
};

} // raul::postprocessing

#endif