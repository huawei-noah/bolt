// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <algorithm>

#include "Tensor.h"

#include <training/system/TypeHalf.h>

namespace
{

/**
 * @brief This helper maps given offset and stride into indexes.
 * @param offset size_t offset in flat array
 * @param strides stride for 4d tensor
 * @return
 */
raul::shape offset_to_indexes(size_t offset, const raul::shape& strides)
{
    raul::shape indexes;
    size_t q = 0;
    while (q < strides.dimensions_num())
    {
        if (strides[q] != 0)
        {

            indexes[q] = offset / strides[q];
            offset %= strides[q];
        }
        else
        {
            indexes[q] = 0U;
        }
        ++q;
    }
    indexes[q - 1] += offset;
    return indexes;
}

} // anonymous namespace

namespace raul
{

template<>
template<>
Tensor& Tensor::operator=(TensorFP16::dt_range beginEnd)
{
    const size_t size = static_cast<size_t>(beginEnd.second - beginEnd.first);
    if (size != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(beginEnd.second - beginEnd.first) + ")");
    }

    auto ii = beginEnd.first;
    auto i = mTensorMem.begin();
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < size; ++q)
    {
        i[q] = toFloat32(ii[q]);
    }
    return *this;
}

template<>
template<>
TensorFP16& TensorFP16::operator=(Tensor::dt_range beginEnd)
{
    const auto size = static_cast<size_t>(beginEnd.second - beginEnd.first);
    if (size != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(beginEnd.second - beginEnd.first) + ")");
    }

    auto i = mTensorMem.begin();
    auto ii = beginEnd.first;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < size; ++q)
    {
        i[q] = toFloat16(ii[q]);
    }
    return *this;
}

template<>
Tensor& Tensor::operator+=(const Tensor& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::plus<dtype>());

    return *this;
}

template<>
Tensor& Tensor::operator+=(const Tensor::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::plus<dtype>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator+=(const TensorFP16& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::plus<half>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator+=(const TensorFP16::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::plus<half>());

    return *this;
}

template<>
Tensor& Tensor::operator-=(const Tensor& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::minus<dtype>());

    return *this;
}

template<>
Tensor& Tensor::operator-=(const Tensor::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::minus<dtype>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator-=(const TensorFP16& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::minus<half>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator-=(const TensorFP16::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::minus<half>());

    return *this;
}

template<>
Tensor& Tensor::operator*=(const Tensor& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::multiplies<dtype>());

    return *this;
}

template<>
Tensor& Tensor::operator*=(const Tensor::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::multiplies<dtype>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator*=(const TensorFP16& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::multiplies<half>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator*=(const TensorFP16::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::multiplies<half>());

    return *this;
}

template<>
Tensor& Tensor::operator/=(const Tensor& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::divides<dtype>());

    return *this;
}

template<>
Tensor& Tensor::operator/=(const Tensor::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::divides<dtype>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator/=(const TensorFP16& rhs)
{
    if (rhs.size() != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(rhs.size()) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::divides<half>());

    return *this;
}

template<>
TensorFP16& TensorFP16::operator/=(const TensorFP16::dt_range& rhs)
{
    auto sz = static_cast<size_t>(rhs.second - rhs.first);
    if (sz != this->size())
    {
        THROW("TensorFP16", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(sz) + ")");
    }
    std::transform(this->begin(), this->end(), rhs.first, this->begin(), std::divides<half>());

    return *this;
}

template<>
Tensor& Tensor::operator=(dtype rhs)
{
    std::fill(this->begin(), this->end(), rhs);

    return *this;
}

template<>
TensorFP16& TensorFP16::operator=(half rhs)
{
    std::fill(this->begin(), this->end(), rhs);

    return *this;
}

template<>
void Tensor::compress(CompressionMode mode)
{
    if (mTensorMem.empty())
    {
        THROW_NONAME("Tensor", "empty tensor");
    }

    if (mode == CompressionMode::NONE) return;

    if (mode == CompressionMode::FP16)
    {
        mCompressedDataFP16.resize(this->size());

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < this->size(); ++q)
        {
            mCompressedDataFP16[q] = toFloat16(mTensorMem[q]);
        }

        mTensorMem.clear();
        mTensorMem.shrink_to_fit();
    }

    if (mode == CompressionMode::INT8)
    {

#if defined(_OPENMP)
        mCompressInt8Min = std::numeric_limits<dtype>::max();
        mCompressInt8Max = std::numeric_limits<dtype>::lowest();

#pragma omp parallel for
        for (size_t q = 0; q < this->size(); ++q)
        {
            if (mTensorMem[q] > mCompressInt8Max)
            {
#pragma omp critical
                if (mTensorMem[q] > mCompressInt8Max) mCompressInt8Max = mTensorMem[q];
            }

            if (mTensorMem[q] < mCompressInt8Min)
            {
#pragma omp critical
                if (mTensorMem[q] < mCompressInt8Min) mCompressInt8Min = mTensorMem[q];
            }
        }
#else
        auto minMax = std::minmax_element(mTensorMem.begin(), mTensorMem.end());
        mCompressInt8Min = *minMax.first;
        mCompressInt8Max = *minMax.second;
#endif

        mCompressedDataInt8.resize(this->size());

        if (mCompressInt8Max != mCompressInt8Min)
        {
            dtype unit = TODTYPE(255.0f) / (mCompressInt8Max - mCompressInt8Min);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < this->size(); ++q)
            {
                mCompressedDataInt8[q] = static_cast<uint8_t>((mTensorMem[q] - mCompressInt8Min) * unit);
            }
        }
        else
        {
            std::fill(mCompressedDataInt8.begin(), mCompressedDataInt8.end(), static_cast<uint8_t>(0));
        }

        mTensorMem.clear();
        mTensorMem.shrink_to_fit();
    }
}

template<>
void TensorFP16::compress(CompressionMode mode)
{
    if (mTensorMem.empty())
    {
        throw std::runtime_error("Tensor[compress]: empty tensor");
    }

    if (mode == CompressionMode::NONE)
    {
        return;
    }

    if (mode == CompressionMode::FP16)
    {
        return;
    }

    if (mode == CompressionMode::INT8)
    {

#if defined(_OPENMP)
        mCompressInt8Min = std::numeric_limits<raul::half>::max();
        mCompressInt8Max = std::numeric_limits<raul::half>::lowest();

#pragma omp parallel for
        for (size_t q = 0; q < this->size(); ++q)
        {
            if (mTensorMem[q] > mCompressInt8Max)
            {
#pragma omp critical
                if (mTensorMem[q] > mCompressInt8Max) mCompressInt8Max = mTensorMem[q];
            }

            if (mTensorMem[q] < mCompressInt8Min)
            {
#pragma omp critical
                if (mTensorMem[q] < mCompressInt8Min) mCompressInt8Min = mTensorMem[q];
            }
        }
#else
        auto minMax = std::minmax_element(mTensorMem.begin(), mTensorMem.end());
        mCompressInt8Min = *minMax.first;
        mCompressInt8Max = *minMax.second;
#endif

        mCompressedDataInt8.resize(this->size());

        if (mCompressInt8Max != mCompressInt8Min)
        {
            raul::half unit = TOHTYPE(255.0f) / (mCompressInt8Max - mCompressInt8Min);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
            for (size_t q = 0; q < this->size(); ++q)
            {
                mCompressedDataInt8[q] = static_cast<uint8_t>((mTensorMem[q] - mCompressInt8Min) * unit);
            }
        }
        else
        {
            std::fill(mCompressedDataInt8.begin(), mCompressedDataInt8.end(), static_cast<uint8_t>(0));
        }

        mTensorMem.clear();
        mTensorMem.shrink_to_fit();
    }
}

template<>
void Tensor::decompress(CompressionMode mode)
{
    if (mode == CompressionMode::NONE) return;

    if (mode == CompressionMode::FP16)
    {
        mTensorMem.resize(mCompressedDataFP16.size(), nullptr);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < mCompressedDataFP16.size(); ++q)
        {
            mTensorMem[q] = toFloat32(mCompressedDataFP16[q]);
        }

        mCompressedDataFP16.clear();
        mCompressedDataFP16.shrink_to_fit();
    }

    if (mode == CompressionMode::INT8)
    {
        mTensorMem.resize(mCompressedDataInt8.size(), nullptr);

        dtype unit = (mCompressInt8Max - mCompressInt8Min) / TODTYPE(255.0f);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < mCompressedDataInt8.size(); ++q)
        {
            mTensorMem[q] = mCompressInt8Min + mCompressedDataInt8[q] * unit;
        }

        mCompressedDataInt8.clear();
        mCompressedDataInt8.shrink_to_fit();
    }
}

template<>
void TensorFP16::decompress(CompressionMode mode)
{
    if (mode == CompressionMode::NONE)
    {
        return;
    }

    if (mode == CompressionMode::FP16)
    {
        return;
    }

    if (mode == CompressionMode::INT8)
    {
        mTensorMem.resize(mCompressedDataInt8.size(), nullptr);

        raul::half unit = (mCompressInt8Max - mCompressInt8Min) / TOHTYPE(255.0f);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (size_t q = 0; q < mCompressedDataInt8.size(); ++q)
        {
            mTensorMem[q] = mCompressInt8Min + mCompressedDataInt8[q] * unit;
        }

        mCompressedDataInt8.clear();
        mCompressedDataInt8.shrink_to_fit();
    }
}

template<>
void Tensor::memClear()
{
    mTensorMem.clear();
    mTensorMem.shrink_to_fit();
}

template<>
void TensorFP16::memClear()
{
    mTensorMem.clear();
    mTensorMem.shrink_to_fit();
}

template<>
void Tensor::memAllocate(dtype* data)
{
    if (mTensorMem.empty())
    {
        mTensorMem.resize(mShape.total_size(), data);
    }
}

template<>
void TensorFP16::memAllocate(half* data)
{
    if (mTensorMem.empty())
    {
        mTensorMem.resize(mShape.total_size(), data);
    }
}

template<>
size_t Tensor::getMaxIndex() const
{
    return std::max_element(mTensorMem.begin(), mTensorMem.end()) - mTensorMem.begin();
}

template<>
size_t Tensor::getMaxIndex(size_t begin, size_t end) const
{
    return std::max_element(mTensorMem.begin() + begin, mTensorMem.begin() + end) - (mTensorMem.begin() + begin);
}

template<>
size_t TensorFP16::getMaxIndex(size_t begin, size_t end) const
{
    return std::max_element(mTensorMem.begin() + begin, mTensorMem.begin() + end) - (mTensorMem.begin() + begin);
}

template<>
size_t Tensor::broadcasted_viewer::get_offset(const size_t index) const
{
    const auto original_indexes = offset_to_indexes(index, mViewerStrides);
    const size_t offset = std::inner_product(mOriginalStrides.cbegin(), mOriginalStrides.cend(), original_indexes.cbegin(), static_cast<size_t>(0));
    return offset;
}

template<>
size_t TensorFP16::broadcasted_viewer::get_offset(const size_t index) const
{
    const auto original_indexes = offset_to_indexes(index, mViewerStrides);
    const size_t offset = std::inner_product(mOriginalStrides.cbegin(), mOriginalStrides.cend(), original_indexes.cbegin(), static_cast<size_t>(0));
    return offset;
}

template<>
dtype& Tensor::broadcasted_viewer::operator[](size_t index)
{
    const auto offset = get_offset(index);
    auto ptr = const_cast<Tensor*>(mTensor);
    return (*ptr)[offset];
}

template<>
half& TensorFP16::broadcasted_viewer::operator[](size_t index)
{
    const auto offset = get_offset(index);
    auto ptr = const_cast<TensorFP16*>(mTensor);
    return (*ptr)[offset];
}

template<>
const dtype& Tensor::broadcasted_viewer::operator[](size_t index) const
{
    const auto offset = get_offset(index);
    return (*mTensor)[offset];
}

template<>
const half& TensorFP16::broadcasted_viewer::operator[](size_t index) const
{
    const auto offset = get_offset(index);
    return (*mTensor)[offset];
}

template<>
Tensor::broadcasted_viewer Tensor::getBroadcastedViewer(const shape& viewer_shape) const
{
    const auto original_shape = getShape();
    const auto original_strides = Common::getStrides(original_shape);
    const auto viewer_strides = Common::getStrides(viewer_shape);
    size_t size = 1U;

    for (auto it = viewer_shape.cbegin(); it != viewer_shape.cend(); ++it)
    {
        size *= *it;
    }

    const auto broadcastable = isBroadcastableTo(viewer_shape);
    if (!broadcastable)
    {
        THROW("Tensor", mName, "tensor is not broadcastable [from " + seq2str(getShape()) + ", to " + seq2str(viewer_shape) + "]");
    }

    return broadcasted_viewer{ this, viewer_strides, original_strides, size };
}

template<>
TensorFP16::broadcasted_viewer TensorFP16::getBroadcastedViewer(const shape& viewer_shape) const
{
    const auto original_shape = getShape();
    const auto original_strides = Common::getStrides(original_shape);
    const auto viewer_strides = Common::getStrides(viewer_shape);
    size_t size = 1U;

    for (auto it = viewer_shape.cbegin(); it != viewer_shape.cend(); ++it)
    {
        size *= *it;
    }

    const auto broadcastable = isBroadcastableTo(viewer_shape);
    if (!broadcastable)
    {
        THROW("TensorFP16", mName, "tensor is not broadcastable [from " + seq2str(getShape()) + ", to " + seq2str(viewer_shape) + "]");
    }

    return broadcasted_viewer{ this, viewer_strides, original_strides, size };
}

template<>
std::string Tensor::getDescription() const
{
    return mName + " " + seq2str(getShape());
}

template<>
std::string TensorFP16::getDescription() const
{
    return mName + " " + seq2str(getShape());
}

} // namespace raul