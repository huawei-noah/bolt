// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef IQUANTIZER_H
#define IQUANTIZER_H

#include <functional>
#include <optional>
#include <variant>

#include <training/base/common/Tensor.h>

#define DEFAUL_QUANTIZATION_BITSIZE 8U

/**
 * @brief Quantization namespace
 *
 * **Quantization** is a transformation of a machine learning model into one that uses
 * parameters and computations at a lower precision. It helps to reduce the model size
 * and, as a consequence, other characteristics like power consumption.
 * Quantization is a lossy process and can be considered in machine learning systems
 * as a source of the noise.
 *
 * **Quantization-aware training (QAT)** is a technique of including quantization noise
 * into the training process to improve model tolerance to precision reduction.
 * Basic idea behind QAT is emulation low-precision computation during inference
 * phase of the training process. It is reached by introducing special quantization
 * layer which changes the values of the tensors.
 *
 * From a mathematical point of view, quantization is a surjective function which can be defined using
 * different quantization strategies. This namespace contains the hierarchy of different quantization
 * algorithms represented by classes that implement interface IQuantizer.
 *
 * @see
 * - B. Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference” arXiv:1712.05877 [cs, stat], Dec. 2017, Accessed: Apr. 24, 2020. [Online].
 * - R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper” arXiv:1806.08342 [cs, stat], Jun. 2018, Accessed: May 05, 2020. [Online]. Available:
 *
 */
namespace raul::quantization
{

using RoundFuncT = std::function<dtype(dtype)>;
using RangeT = struct Range
{
    dtype min;
    dtype max;
};

/**
 * @brief Interface of quantizer algortithms
 *
 * Every quantizer must implement two methods:
 * - quantize,
 * - dequantize,
 * - backpropagate.
 *
 * it can help to reduce the amount of cycles
 */
struct IQuantizer
{
    using TensorItr = Tensor::iterator;
    using TensorConstItr = Tensor::const_iterator;

    IQuantizer() = default;
    virtual ~IQuantizer() = default;

    /**
     * @brief Quantize method
     *
     * This method maps floating-point number (dtype) to
     * quantized floating-point number with restricted dynamic range and precision
     *
     * @param begin Tensor iterator specifies the beginning of the range of elements to quantize
     * @param end Tensor iterator specifies the end of the range of elements to quantize
     */
    virtual void quantize(TensorItr begin, TensorItr end) = 0;

    void quantize(TensorFP16::iterator, TensorFP16::iterator) {}

    /**
     * @brief Dequantize method
     *
     * This method maps quantized floating-point (dtype) to
     * to floating-point number with original dynamic range but quantized precision
     *
     * @param begin Tensor iterator specifies the beginning of the range of elements to quantize
     * @param end Tensor iterator specifies the end of the range of elements to quantize
     */
    virtual void dequantize(TensorItr begin, TensorItr end) = 0;

    void dequantize(TensorFP16::iterator, TensorFP16::iterator) {}

    /**
     * @brief Backpropagate method
     *
     * @param begin Tensor iterator specifies the beginning of the range of elements to quantize
     * @param end Tensor iterator specifies the end of the range of elements to quantize
     * @param delta_begin Tensor iterator specifies the beginning of the range of elements of input deltas
     * @param grad_begin Tensor iterator specifies the beginning of the range of elements where gradient will be stored
     */
    virtual void backpropagate(TensorConstItr begin, TensorConstItr end, TensorConstItr delta_begin, TensorItr grad_begin) = 0;
};

} // raul::quantization

#endif // IQUANTIZER_H
