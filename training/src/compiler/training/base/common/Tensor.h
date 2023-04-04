// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TENSOR_H
#define TENSOR_H

#include <string>

#include "TensorMem.h"
#include <training/system/Name.h>

#include <training/base/common/Common.h>
#include <training/base/common/Conversions.h>
#include <training/base/common/io/TensorStream.h>

namespace raul
{

/**
 * @brief Helper to convert sequences to str
 * @param data
 * @return str
 */
template<typename T>
std::string seq2str(const T& data, char separator = ',', std::pair<char, char> brackets = { '(', ')' }, bool compact = false)
{
    const size_t half_size = 5;
    std::string str;

    str += brackets.first;

    auto begin = data.cbegin();
    const auto end = data.cend();

    if (begin != end)
    {
        str += Conversions::toString(*begin);
        ++begin;
    }

    if (compact && data.size() > 2 * half_size)
    {
        const auto first_half_stop = begin + half_size - 1;
        for (; begin != first_half_stop; ++begin)
        {
            str += separator;
            str += Conversions::toString(*begin);
        }
        str += separator;
        str += "...";
        begin = end - half_size;
        for (; begin != end; ++begin)
        {
            str += separator;
            str += Conversions::toString(*begin);
        }
    }
    else
    {
        for (; begin != end; ++begin)
        {
            str += separator;
            str += Conversions::toString(*begin);
        }
    }

    str += brackets.second;

    return str;
}

template<typename dt>
class TensorImpl
{
  public:
    typedef std::pair<const dt*, const dt*> dt_range;
    typedef dt type;

    explicit TensorImpl() = delete;
    explicit TensorImpl(const TensorImpl&) = delete;
    explicit TensorImpl(TensorImpl&&) = delete;

    // no data
    TensorImpl(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, AllocationMode allocMode = AllocationMode::STANDARD, bool isAllocate = true)
        : mName(name)
        , mTensorMem(allocMode)
        , mCompressInt8Min(0)
        , mCompressInt8Max(0)
        , mShape(batchSize, depth, height, width)
    {
        if (isAllocate)
        {
            mTensorMem.resize(mShape.total_size(), nullptr);
        }
    }

    TensorImpl(size_t batchSize, size_t depth, size_t height, size_t width, bool isAllocate = true)
        : TensorImpl("", batchSize, depth, height, width, AllocationMode::STANDARD, isAllocate)
    {
    }

    TensorImpl(const Name& name, size_t size)
        : TensorImpl(name, 1u, 1u, 1u, size)
    {
    }

    explicit TensorImpl(size_t size)
        : TensorImpl("", size)
    {
    }

    TensorImpl(shape inShape)
        : TensorImpl("", inShape)
    {
    }

    TensorImpl(const Name& name, shape inShape, bool isAllocate = true)
        : TensorImpl(name, inShape[0], inShape[1], inShape[2], inShape[3], AllocationMode::STANDARD, isAllocate)
    {
    }

    // filler
    TensorImpl(const Name& name, shape inShape, dt filler)
        : mName(name)
        , mTensorMem(inShape.total_size(), filler)
        , mCompressInt8Min(0)
        , mCompressInt8Max(0)
        , mShape(inShape)
    {
    }

    TensorImpl(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, dt filler)
        : TensorImpl(name, shape(batchSize, depth, height, width), filler)
    {
    }

    TensorImpl(const Name& name, size_t size, dt filler)
        : TensorImpl(name, shape(1u, 1u, 1u, size), filler)
    {
    }

    TensorImpl(size_t size, dt filler)
        : TensorImpl("", size, filler)
    {
    }

    // initializer_list
    TensorImpl(const Name& name, shape inShape, std::initializer_list<dt> list)
        : mName(name)
        , mTensorMem(list)
        , mCompressInt8Min(0)
        , mCompressInt8Max(0)
        , mShape(inShape)
    {
        if (list.size() != mShape.total_size())
        {
            THROW("TensorImpl", name, "Bad initializer size");
        }
    }

    TensorImpl(const Name& name, std::initializer_list<dt> list)
        : TensorImpl(name, shape(1u, 1u, 1u, list.size()), list)
    {
    }

    TensorImpl(std::initializer_list<dt> list)
        : TensorImpl("", list)
    {
    }

    TensorImpl(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<dt> list)
        : TensorImpl(name, shape(batchSize, depth, height, width), list)
    {
    }

    TensorImpl(size_t batchSize, size_t depth, size_t height, size_t width, std::initializer_list<dt> list)
        : TensorImpl("", batchSize, depth, height, width, list)
    {
    }

    TensorImpl(shape inShape, std::initializer_list<dt> list)
        : TensorImpl("", inShape, list)
    {
    }

    // dt_range
    TensorImpl(const Name& name, shape inShape, dt_range beginEnd)
        : mName(name)
        , mTensorMem(beginEnd.first, beginEnd.second)
        , mCompressInt8Min(0)
        , mCompressInt8Max(0)
        , mShape(inShape)
    {
        if (this->size() != mShape.total_size())
        {
            THROW("TensorImpl", name, "Bad data size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(mShape.total_size()) + ")");
        }
    }

    TensorImpl(shape inShape, dt_range beginEnd)
        : TensorImpl("", inShape, beginEnd)
    {
    }

    TensorImpl(const Name& name, dt_range beginEnd)
        : TensorImpl(name, shape(1u, 1u, 1u, static_cast<size_t>(beginEnd.second - beginEnd.first)), beginEnd)
    {
    }

    TensorImpl(const Name& name, size_t batchSize, size_t depth, size_t height, size_t width, dt_range beginEnd)
        : TensorImpl(name, shape(batchSize, depth, height, width), beginEnd)
    {
    }

    TensorImpl(dt_range beginEnd)
        : TensorImpl("", beginEnd)
    {
    }

    INLINE dt& operator[](size_t index) noexcept { return mTensorMem[index]; }

    INLINE const dt& operator[](size_t index) const noexcept { return mTensorMem[index]; }

    INLINE dt* getBuffer() noexcept { return &mTensorMem[0]; }
    INLINE const dt* getBuffer() const noexcept { return &mTensorMem[0]; }

    /**
     * set mData with size check (copy)
     */
    template<typename T>
    TensorImpl& operator=(std::initializer_list<T> lst)
    {
        if (lst.size() != this->size())
        {
            THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(lst.size()) + ")");
        }
        std::transform(lst.begin(), lst.end(), mTensorMem.begin(), [](auto v) { return static_cast<dtype>(v); });
        return *this;
    }

    TensorImpl& operator=(dt_range beginEnd)
    {
        if (static_cast<size_t>(beginEnd.second - beginEnd.first) != this->size())
        {
            THROW("Tensor", mName, "wrong size (expected: " + Conversions::toString(this->size()) + ", got: " + Conversions::toString(beginEnd.second - beginEnd.first) + ")");
        }
        std::copy(beginEnd.first, beginEnd.second, mTensorMem.begin());
        return *this;
    }

    operator dt_range() const { return dt_range(&mTensorMem[0], &mTensorMem[0] + this->size()); }

    /**
     * conversions to/from TensorFP16
     */
    template<typename T>
    TensorImpl& operator=(T beginEnd);

    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;

    TensorImpl& operator+=(const TensorImpl& rhs);
    TensorImpl& operator-=(const TensorImpl& rhs);
    TensorImpl& operator*=(const TensorImpl& rhs);
    TensorImpl& operator/=(const TensorImpl& rhs);

    TensorImpl& operator+=(const dt_range& rhs);
    TensorImpl& operator-=(const dt_range& rhs);
    TensorImpl& operator*=(const dt_range& rhs);
    TensorImpl& operator/=(const dt_range& rhs);

    TensorImpl& operator=(dt rhs);

    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value || std::is_same<T, half>::value>::type>
    TensorImpl& operator+=(T rhs)
    {
        std::transform(this->begin(), this->end(), this->begin(), [&](const TensorImpl::type x) { return static_cast<TensorImpl::type>(x + rhs); });

        return *this;
    }

    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value || std::is_same<T, half>::value>::type>
    TensorImpl& operator-=(T rhs)
    {
        std::transform(this->begin(), this->end(), this->begin(), [&](const TensorImpl::type x) { return static_cast<TensorImpl::type>(x - rhs); });

        return *this;
    }

    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value || std::is_same<T, half>::value>::type>
    TensorImpl& operator*=(T rhs)
    {
        std::transform(this->begin(), this->end(), this->begin(), [&](const TensorImpl::type x) { return static_cast<TensorImpl::type>(x * rhs); });

        return *this;
    }

    template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value || std::is_same<T, half>::value>::type>
    TensorImpl& operator/=(T rhs)
    {
        std::transform(this->begin(), this->end(), this->begin(), [&](const TensorImpl::type x) { return static_cast<TensorImpl::type>(x / rhs); });

        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const TensorImpl& instance)
    {
        out << "Tensor";
        const auto name = instance.getName();
        if (!name.empty())
        {
            out << " '" << name << "'";
        }
        out << " " << seq2str(instance.getShape());

        if (io::tensor::isSetFlag(out, io::tensor::TensorView::size))
        {
            out << ", size: " << instance.size() * sizeof(type);
        }
        if (io::tensor::isSetFlag(out, io::tensor::TensorView::scale))
        {
            const auto scale = instance.getScale();
            if (scale)
            {
                out << ", scale: " << static_cast<dtype>(*scale);
            }
            else
            {
                out << ", scale: no";
            }
        }
        const bool compact = io::tensor::isSetFlag(out, io::tensor::TensorView::reduced);

        if (io::tensor::isSetFlag(out, io::tensor::TensorView::content))
        {
            out << std::endl << seq2str(instance, ',', { '[', ']' }, compact);
        }
        return out;
    }

    [[nodiscard]] size_t size() const { return mTensorMem.size(); }
    [[nodiscard]] bool empty() const { return mTensorMem.empty(); }

    void compress(CompressionMode mode);
    void decompress(CompressionMode mode);

    void resetScale(dtype scale) { mScale = scale; }

    void scale(dtype scale)
    {
        *this *= scale;
        if (mScale)
        {
            *mScale *= scale;
        }
        else
        {
            mScale = scale;
        }
    }

    void unscale()
    {
        if (mScale)
        {
            *this /= *mScale;
            mScale = std::nullopt;
        }
    }

    void memAllocate(dt* data); // d.polubotko: do not use directly in layers, should be used by actions only
    void memClear();            // d.polubotko: do not use directly in layers, should be used by actions only

    [[nodiscard]] size_t getMaxIndex() const;
    [[nodiscard]] size_t getMaxIndex(size_t begin, size_t end) const;

    template<typename pointer_type>
    class iterator_impl
    {
      public:
        typedef iterator_impl self_type;
        typedef pointer_type value_type;
        typedef value_type& reference;
        typedef value_type* pointer;
        typedef std::random_access_iterator_tag iterator_category;
        typedef std::ptrdiff_t difference_type;

        iterator_impl(pointer ptr)
            : mPtr(ptr)
        {
        }
	~iterator_impl(){}
        iterator_impl() = default;
        iterator_impl& operator=(const iterator_impl&) = default;

        self_type operator++()
        {
            mPtr++;
            return *this;
        }
        self_type operator++(int)
        {
            self_type i = *this;
            mPtr++;
            return i;
        }
        self_type operator+(const difference_type& n) const
        {
            self_type i = *this;
            i.mPtr += n;
            return i;
        }
        self_type operator-(const difference_type& n) const
        {
            self_type i = *this;
            i.mPtr -= n;
            return i;
        }
        difference_type operator-(const self_type& other) const { return mPtr - other.mPtr; }
        reference operator*() { return *mPtr; }
        reference operator*() const { return *mPtr; }
        pointer operator->() { return mPtr; }
        bool operator==(const self_type& rhs) { return mPtr == rhs.mPtr; }
        bool operator!=(const self_type& rhs) { return mPtr != rhs.mPtr; }

        operator iterator_impl<value_type const>() const { return iterator_impl<value_type const>(mPtr); }

      private:
        pointer mPtr;
    };

    typedef iterator_impl<dt> iterator;
    typedef iterator_impl<const dt> const_iterator;

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    const_iterator begin() const { return const_iterator(&mTensorMem[0]); }
    const_iterator end() const { return const_iterator(&mTensorMem[0] + this->size()); }
    iterator begin() { return iterator(&mTensorMem[0]); }
    iterator end() { return iterator(&mTensorMem[0] + this->size()); }

    dt* data() { return &mTensorMem[0]; }
    const dt* data() const { return &mTensorMem[0]; }

    template<size_t NewDimsNum>
    auto reshape(const yato::dimensionality<NewDimsNum, size_t>& extents) const
    {
        return mTensorMem.reshape(extents);
    }

    template<size_t NewDimsNum>
    auto reshape(const yato::dimensionality<NewDimsNum, size_t>& extents)
    {
        return mTensorMem.reshape(extents);
    }

    template<typename... Dims>
    auto reshape(Dims... extents) const
    {
        return mTensorMem.reshape(yato::dims(extents...));
    }

    template<typename... Dims>
    auto reshape(Dims... extents)
    {
        return mTensorMem.reshape(yato::dims(extents...));
    }

    auto get4DView() const { return reshape(getShape()); }
    auto get4DView() { return reshape(getShape()); }

    const std::string& getName() const { return mName; }
    [[nodiscard]] std::string getDescription() const;

    size_t getBatchSize() const { return mShape[0]; }
    size_t getDepth() const { return mShape[1]; }
    size_t getHeight() const { return mShape[2]; }
    size_t getWidth() const { return mShape[3]; }

    shape getShape() const { return mShape; }

    AllocationMode getAllocationMode() const { return mTensorMem.getAllocationMode(); }

    std::optional<dtype> getScale() const { return mScale; }

    /**
     * @brief Check if the tensor can be broadcasted to the provided shape
     * @param to_shape new tensor shape
     * @return true or false
     *
     */
    bool isBroadcastableTo(const shape& to_shape) const noexcept { return Common::shapeIsBroadcastable(getShape(), to_shape); }

    /**
     * @brief Helper class to emulate broadcasted tensor
     */
    class broadcasted_viewer
    {
        const TensorImpl* mTensor;
        const shape mViewerStrides;
        const shape mOriginalStrides;
        const size_t mSize;

        size_t get_offset(const size_t index) const;

      public:
        broadcasted_viewer(TensorImpl* tensor, const shape viewer_strides, const shape original_strides, const size_t size)
            : mTensor(tensor)
            , mViewerStrides(viewer_strides)
            , mOriginalStrides(original_strides)
            , mSize(size)
        {
        }
        broadcasted_viewer(const TensorImpl* tensor, const shape viewer_strides, const shape original_strides, const size_t size)
            : mTensor(tensor)
            , mViewerStrides(viewer_strides)
            , mOriginalStrides(original_strides)
            , mSize(size)
        {
        }
        ~broadcasted_viewer(){}
	/**
         * @warning Write-access to returned element is not thread-safe as multiple broadcasted indices correspond to single underlying tensor element
         */
        dt& operator[](size_t index);
        const dt& operator[](size_t index) const;
        size_t size() const { return mSize; }
    };

    /**
     * @brief The function returns BroadcastedViewer that performs tensor broadcasting
     * @param viewer_shape
     * @return Closure that returns element of broadcasted tenso
     *
     */
    broadcasted_viewer getBroadcastedViewer(const shape& viewer_shape) const;
    broadcasted_viewer getBroadcastedViewer(const shape& viewer_shape) { return const_cast<const TensorImpl*>(this)->getBroadcastedViewer(viewer_shape); }

  private:
    std::string mName;

    TensorMem<dt> mTensorMem;

    std::vector<half> mCompressedDataFP16;
    std::vector<uint8_t> mCompressedDataInt8;
    dt mCompressInt8Min;
    dt mCompressInt8Max;

    std::optional<dtype> mScale;

    shape mShape;
};

typedef TensorImpl<dtype> Tensor;
typedef TensorImpl<uint8_t> TensorU8;
typedef TensorImpl<half> TensorFP16;

template<typename T>
struct ParamAndGradImpl
{
    T& Param;
    T& Gradient;
};

typedef ParamAndGradImpl<Tensor> ParamAndGrad;

#define TORANGE(var) static_cast<raul::Tensor::dt_range>(var)
#define TORANGE_FP16(var) static_cast<raul::TensorFP16::dt_range>(var)
#define TORANGE_MM(var) static_cast<typename MM::tensor::dt_range>(var)

/**
 * @brief The function broadcast destination tensor to source tensor shape
 *        and applies binary operation: dst = op(src, dst)
 *
 */
template<typename TensorImpl, typename TOp>
void binaryOpBroadcastedDst(const TensorImpl& srcTensor, TensorImpl& dstTensor, TOp&& op)
{
    const auto& srcShape = srcTensor.getShape();
    const auto& dstShape = dstTensor.getShape();

    if (srcShape == dstShape)
    {
        std::transform(srcTensor.cbegin(), srcTensor.cend(), dstTensor.cbegin(), dstTensor.begin(), op);
        return;
    }

    const auto* src = srcTensor.data();
    auto* dst = dstTensor.data();

    size_t srcStride1 = srcShape[1] * srcShape[2] * srcShape[3];
    size_t srcStride2 = srcShape[2] * srcShape[3];
    size_t srcStride3 = srcShape[3];

    size_t dstStride1 = dstShape[1] * dstShape[2] * dstShape[3];
    size_t dstStride2 = dstShape[2] * dstShape[3];
    size_t dstStride3 = dstShape[3];

    for (size_t i1 = 0; i1 < srcShape[0]; ++i1)
    {
        size_t j1 = dstShape[0] == 1 ? 0 : i1;
        for (size_t i2 = 0; i2 < srcShape[1]; ++i2)
        {
            size_t j2 = dstShape[1] == 1 ? 0 : i2;
            for (size_t i3 = 0; i3 < srcShape[2]; ++i3)
            {
                size_t j3 = dstShape[2] == 1 ? 0 : i3;
                for (size_t i4 = 0; i4 < srcShape[3]; ++i4)
                {
                    size_t j4 = dstShape[3] == 1 ? 0 : i4;
                    size_t srcIdx = srcStride1 * i1 + srcStride2 * i2 + srcStride3 * i3 + i4;
                    size_t dstIdx = dstStride1 * j1 + dstStride2 * j2 + dstStride3 * j3 + j4;

                    dst[dstIdx] = op(src[srcIdx], dst[dstIdx]);
                }
            }
        }
    }
}

/**
 * @brief The function broadcast source tensor to destination tensor shape
 *        and applies binary operation: dst = op(src, dst)
 *
 */
template<typename TensorImpl, typename TOp>
void binaryOpBroadcastedSrc(const TensorImpl& srcTensor, TensorImpl& dstTensor, TOp&& op)
{
    const auto& srcShape = srcTensor.getShape();
    const auto& dstShape = dstTensor.getShape();

    if (srcShape == dstShape)
    {
        std::transform(srcTensor.cbegin(), srcTensor.cend(), dstTensor.cbegin(), dstTensor.begin(), op);
        return;
    }

    const auto* src = srcTensor.data();
    auto* dst = dstTensor.data();

    size_t srcStride1 = srcShape[1] * srcShape[2] * srcShape[3];
    size_t srcStride2 = srcShape[2] * srcShape[3];
    size_t srcStride3 = srcShape[3];

    size_t dstStride1 = dstShape[1] * dstShape[2] * dstShape[3];
    size_t dstStride2 = dstShape[2] * dstShape[3];
    size_t dstStride3 = dstShape[3];

    for (size_t i1 = 0; i1 < dstShape[0]; ++i1)
    {
        size_t j1 = srcShape[0] == 1 ? 0 : i1;
        for (size_t i2 = 0; i2 < dstShape[1]; ++i2)
        {
            size_t j2 = srcShape[1] == 1 ? 0 : i2;
            for (size_t i3 = 0; i3 < dstShape[2]; ++i3)
            {
                size_t j3 = srcShape[2] == 1 ? 0 : i3;
                for (size_t i4 = 0; i4 < dstShape[3]; ++i4)
                {
                    size_t j4 = srcShape[3] == 1 ? 0 : i4;

                    size_t srcIdx = srcStride1 * j1 + srcStride2 * j2 + srcStride3 * j3 + j4;
                    size_t dstIdx = dstStride1 * i1 + dstStride2 * i2 + dstStride3 * i3 + i4;

                    dst[dstIdx] = op(src[srcIdx], dst[dstIdx]);
                }
            }
        }
    }
}

} // raul namespace

#endif
