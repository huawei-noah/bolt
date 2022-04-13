// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef ELEMENT_WISE_COMPARE_LAYER_CPU_H
#define ELEMENT_WISE_COMPARE_LAYER_CPU_H

#include <training/base/layers/BasicImpl.h>

#include <unordered_map>

namespace
{
template<typename T>
std::unordered_map<std::string, std::function<bool(const T, const T, const T)>> comparators = {
    { "equal", [](const T x, const T y, const T tol) { return static_cast<T>(std::abs(TODTYPE(x - y))) <= tol; } },
    { "ne", [](const T x, const T y, const T tol) { return static_cast<T>(std::abs(TODTYPE(x - y))) > tol; } },
    { "less", [](const T x, const T y, const T tol) { return y - x > tol; } },
    { "greater", [](const T x, const T y, const T tol) { return x - y > tol; } },
    { "le", [](const T x, const T y, const T tol) { return y - x >= tol; } },
    { "ge", [](const T x, const T y, const T tol) { return x - y >= tol; } },
    { "exact_equal", [](const T x, const T y, const T) { return x == y; } },
    { "exact_ne", [](const T x, const T y, const T) { return x != y; } },
    { "exact_less", [](const T x, const T y, const T) { return x < y; } },
    { "exact_greater", [](const T x, const T y, const T) { return x > y; } },
    { "exact_le", [](const T x, const T y, const T) { return x <= y; } },
    { "exact_ge", [](const T x, const T y, const T) { return x >= y; } }
};
} // anonymous

namespace raul
{
class ElementWiseCompareLayer;

/**
 * @brief Element-wise Comparison Layer CPU implementation
 */
template<typename MM>
class ElementWiseCompareLayerCPU : public BasicImpl
{
  public:
    ElementWiseCompareLayerCPU(ElementWiseCompareLayer& layer)
        : mLayer(layer)
    {
    }

    ElementWiseCompareLayerCPU(ElementWiseCompareLayerCPU&&) = default;
    ElementWiseCompareLayerCPU(const ElementWiseCompareLayerCPU&) = delete;
    ElementWiseCompareLayerCPU& operator=(const ElementWiseCompareLayerCPU&) = delete;

    void forwardComputeImpl(NetworkMode mode) override;
    void backwardComputeImpl() override{};

  private:
    ElementWiseCompareLayer& mLayer;
};

} // raul namespace

#endif