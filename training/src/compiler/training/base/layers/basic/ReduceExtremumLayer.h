// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef REDUCE_EXTREMUM_LAYER_H
#define REDUCE_EXTREMUM_LAYER_H

#include <training/base/common/Common.h>
#include <training/base/layers/BasicLayer.h>

namespace raul
{

/**
 * @brief Reduce Extremum Layer
 *
 * Returns extremum values along chosen dimension.
 * If chosen dimension is raul::Dimension::Default
 * (or no dimension specified), layer returns global extremum.
 */
template<typename T>
struct Min
{
    static bool compare(const T x, const T y) { return y <= x; }
    static T getBound() { return std::numeric_limits<T>::infinity(); }
};

template<typename T>
struct Max
{
    static bool compare(const T x, const T y) { return y >= x; }
    static T getBound() { return -std::numeric_limits<T>::infinity(); }
};

template<template<typename> typename Comparator>
class ReduceExtremumLayer : public BasicLayer
{
  public:
    ReduceExtremumLayer(const Name& name, const BasicParamsWithDim& params, NetworkParameters& networkParameters);

    ReduceExtremumLayer(ReduceExtremumLayer&&) = default;
    ReduceExtremumLayer(const ReduceExtremumLayer&) = delete;
    ReduceExtremumLayer& operator=(const ReduceExtremumLayer&) = delete;

  private:
    Dimension mDim;
    // Need for backward path
    std::vector<size_t> mCountExtremums;

    template<template<typename> typename Comp, typename MM>
    friend class ReduceExtremumLayerCPU;
};

}

#endif