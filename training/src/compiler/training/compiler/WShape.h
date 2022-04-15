// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef WSHAPE_H
#define WSHAPE_H

namespace raul
{


template<bool B, class T, class F>
using if_t = typename std::conditional<B, T, F>::type;

/**
 * @brief Placeholder for batch size
 */
struct BS
{
    BS()
        : multiplier(1)
    {
    }
    explicit BS(size_t multi)
        : multiplier(multi)
    {
    }

    size_t multiplier;
};

class Workflow;

/**
 * @brief Shape of tensor with batch size placeholders and multiplier possible
 */
class WShape
{
  public:
    WShape();

    WShape(const shape& shapeVal);

    template<typename T, typename U, typename V, typename W>
    WShape(T a, U b, V c, W d)
    {
        struct CheckNone
        {
            static void fill(WShape& shape, size_t val, size_t index)
            {
                shape.mShape[index] = val;
                shape.mIsBS[index] = false;
                shape.mMultiplier[index] = 1u;
            }
        };

        struct CheckBS
        {
            static void fill(WShape& shape, const BS& val, size_t index)
            {
                shape.mShape[index] = 0u; // getShape will recalculate
                shape.mIsBS[index] = true;
                shape.mMultiplier[index] = val.multiplier;
            }
        };

        typedef if_t<std::is_same<T, BS>::value, CheckBS, CheckNone> TypeA;

        typedef if_t<std::is_same<U, BS>::value, CheckBS, CheckNone> TypeB;

        typedef if_t<std::is_same<V, BS>::value, CheckBS, CheckNone> TypeC;

        typedef if_t<std::is_same<W, BS>::value, CheckBS, CheckNone> TypeD;

        TypeA::fill(*this, a, 0);
        TypeB::fill(*this, b, 1);
        TypeC::fill(*this, c, 2);
        TypeD::fill(*this, d, 3);
    }

    [[nodiscard]] bool isBSDependent() const;

    /**
     * @brief Get shape of tensor. If placeholder used and no BS defined - exception
     */
    [[nodiscard]] shape getShape(const Workflow& work) const;

    void selectMaxShape(WShape& other)
    {
        for (size_t q = 0; q < shape::dimensions_number; ++q)
        {
            if (!mIsBS[q] && !other.mIsBS[q])
            {
                mMultiplier[q] = std::max(mMultiplier[q], other.mMultiplier[q]);
                mShape[q] = std::max(mShape[q], other.mShape[q]);
                other.mMultiplier[q] = mMultiplier[q];
                other.mShape[q] = mShape[q];
            }
            else if ((!mIsBS[q] && other.mIsBS[q]) || (mIsBS[q] && !other.mIsBS[q]))
            {
                THROW_NONAME("WShape", "not BS layout in shapes");
            }
        }
    }

    bool operator==(const WShape&) const;
    bool operator!=(const WShape& other) const { return !operator==(other); }

    [[nodiscard]] std::string toString() const;

    friend class Workflow;

    friend std::ostream& operator<<(std::ostream& out, const WShape& instance) { return out << instance.toString(); }

  private:
    shape mShape;
    bool mIsBS[shape::dimensions_number];
    size_t mMultiplier[shape::dimensions_number];
};

}


#endif // WSHAPE_H