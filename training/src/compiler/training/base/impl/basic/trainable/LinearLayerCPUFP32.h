// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef LINEAR_LAYER_CPU_FP32_H
#define LINEAR_LAYER_CPU_FP32_H

#include <training/base/layers/BasicImpl.h>

namespace raul
{
class LinearLayer;
template<typename Type>
class TensorImpl;
typedef TensorImpl<dtype> Tensor;

/**
 * @brief Linear layer CPU implementation transform
 */
class LinearLayerCPUFP32 : public BasicImpl
{
  public:
    LinearLayerCPUFP32(LinearLayer& layer)
        : mLayer(layer)
    {
    }

    LinearLayerCPUFP32(LinearLayerCPUFP32&&) = default;
    LinearLayerCPUFP32(const LinearLayerCPUFP32&) = delete;
    LinearLayerCPUFP32& operator=(const LinearLayerCPUFP32&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override;

  private:
    LinearLayer& mLayer;

    static std::shared_ptr<Tensor> mInput;
    static std::shared_ptr<Tensor> mOutput;
    static std::shared_ptr<Tensor> mDeltas;
    static std::shared_ptr<Tensor> mPrevLayerDeltas;
    static std::shared_ptr<Tensor> mWeights;
    static std::shared_ptr<Tensor> mGradWeights;
    static std::shared_ptr<Tensor> mBiases;
    static std::shared_ptr<Tensor> mGradBiases;
};
} // raul namespace

#endif