// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Adam.h"
#include <iostream>
#include <stdexcept>

namespace
{
constexpr raul::dtype alpha_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_lower_boundary = 0.0_dt;
constexpr raul::dtype beta_upper_boundary = 1.0_dt;

bool abs_compare(raul::dtype a, raul::dtype b)
{
    return (std::abs(a) < std::abs(b));
}

}

namespace raul::optimizers
{

Adam::Adam(const dtype alpha, const dtype beta_1, const dtype beta_2, const dtype epsilon, bool use_simple_epsilon)
    : m_alpha(alpha)
    , m_beta_1(beta_1)
    , m_beta_2(beta_2)
    , m_epsilon(epsilon)
    , m_use_simple_epsilon(use_simple_epsilon)
{
    if (alpha <= alpha_lower_boundary)
    {
        THROW_NONAME("Adam", "reset alpha>" + Conversions::toString(alpha_lower_boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }

    if (beta_1 < beta_lower_boundary || beta_1 >= beta_upper_boundary)
    {
        THROW_NONAME("Adam",
                     "reset beta_1 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_1=" + Conversions::toString(beta_1) +
                         ")");
    }

    if (beta_2 < beta_lower_boundary || beta_2 >= beta_upper_boundary)
    {
        THROW_NONAME("Adam",
                     "reset beta_2 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_2=" + Conversions::toString(beta_2) +
                         ")");
    }
}

void Adam::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adam", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_1_t", 1, 1, 1, 1, this->m_beta_1);
    }
    b1tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_2_t", 1, 1, 1, 1, this->m_beta_2);
    }
    b2tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adam") / param.getName() / "m", param.getShape());
    }
    mp = &memory_manager.getTensor(Name("Adam") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("Adam") / param.getName() / "v", param.getShape());
    }
    vp = &memory_manager.getTensor(Name("Adam") / param.getName() / "v");

    Tensor& beta_1_t = *b1tp;
    Tensor& beta_2_t = *b2tp;
    Tensor& m = *mp;
    Tensor& v = *vp;

    const auto sqrt_beta_2_t_0 = std::sqrt(1.0_dt - beta_2_t[0]);
    const auto alpha_new = this->m_alpha * sqrt_beta_2_t_0 / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = m_use_simple_epsilon ? this->m_epsilon : this->m_epsilon * sqrt_beta_2_t_0;
    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        m[i] = this->m_beta_1 * m[i] + (1.0_dt - this->m_beta_1) * grad[i];
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        v[i] = this->m_beta_2 * v[i] + (1.0_dt - this->m_beta_2) * grad[i] * grad[i];
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new)
        param[i] = param[i] - alpha_new * m[i] / (std::sqrt(v[i]) + epsilon_new);
    }

    beta_1_t[0] *= this->m_beta_1;
    beta_2_t[0] *= this->m_beta_2;
}

void Adam::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("Adam", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    TensorFP16 *b1tp, *b2tp, *mp, *vp;
    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_1_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_1));
    }
    b1tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("Adam") / param.getName() / "beta_2_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_2));
    }
    b2tp = &memory_manager.getTensor(Name("Adam") / param.getName() / "beta_2_t");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "m"))
    {
        mp = memory_manager.createTensor(Name("Adam") / param.getName() / "m", param.getShape());
    }
    mp = &memory_manager.getTensor(Name("Adam") / param.getName() / "m");

    if (!memory_manager.tensorExists(Name("Adam") / param.getName() / "v"))
    {
        vp = memory_manager.createTensor(Name("Adam") / param.getName() / "v", param.getShape());
    }
    vp = &memory_manager.getTensor(Name("Adam") / param.getName() / "v");

    TensorFP16& beta_1_t = *b1tp;
    TensorFP16& beta_2_t = *b2tp;
    TensorFP16& m = *mp;
    TensorFP16& v = *vp;

    const auto sqrt_beta_2_t_0 = std::sqrt(1.0_dt - beta_2_t[0]);
    const auto alpha_new = this->m_alpha * sqrt_beta_2_t_0 / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = m_use_simple_epsilon ? this->m_epsilon : this->m_epsilon * sqrt_beta_2_t_0;
    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        const auto mTmp = this->m_beta_1 * TODTYPE(m[i]) + (1.0_dt - this->m_beta_1) * TODTYPE(grad[i]);
        m[i] = TOHTYPE(mTmp);
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        const auto vTmp = this->m_beta_2 * TODTYPE(v[i]) + (1.0_dt - this->m_beta_2) * TODTYPE(grad[i]) * TODTYPE(grad[i]);
        v[i] = TOHTYPE(vTmp);
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new)
        param[i] = TOHTYPE(TODTYPE(param[i]) - alpha_new * mTmp / (std::sqrt(vTmp) + epsilon_new));
    }

    beta_1_t[0] = TOHTYPE(TODTYPE(beta_1_t[0]) * this->m_beta_1);
    beta_2_t[0] = TOHTYPE(TODTYPE(beta_2_t[0]) * this->m_beta_2);
}

std::ostream& Adam::as_ostream(std::ostream& out) const
{
    out << "Adam(alpha=" << std::scientific << this->m_alpha << ", beta_1=" << this->m_beta_1 << ", beta_2=" << this->m_beta_2 << ", epsilon=" << this->m_epsilon << ")";
    return out;
}

AdamQuantized::AdamQuantized(size_t blockSize, const dtype alpha, const dtype beta_1, const dtype beta_2, const dtype epsilon, bool use_simple_epsilon)
    : m_alpha(alpha)
    , m_beta_1(beta_1)
    , m_beta_2(beta_2)
    , m_epsilon(epsilon)
    , m_use_simple_epsilon(use_simple_epsilon)
    , mBlockSize(blockSize)
{
    if (alpha <= alpha_lower_boundary)
    {
        THROW_NONAME("AdamQuantized", "reset alpha>" + Conversions::toString(alpha_lower_boundary) + " (current alpha=" + Conversions::toString(alpha) + ")");
    }

    if (beta_1 < beta_lower_boundary || beta_1 >= beta_upper_boundary)
    {
        THROW_NONAME("AdamQuantized",
                     "reset beta_1 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_1=" + Conversions::toString(beta_1) +
                         ")");
    }

    if (beta_2 < beta_lower_boundary || beta_2 >= beta_upper_boundary)
    {
        THROW_NONAME("AdamQuantized",
                     "reset beta_2 from [" + Conversions::toString(beta_lower_boundary) + ", " + Conversions::toString(beta_upper_boundary) + ") (current beta_2=" + Conversions::toString(beta_2) +
                         ")");
    }
}

void AdamQuantized::optimize(MemoryManager& memory_manager, Tensor& param, const Tensor& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("AdamQuantized", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    Tensor *b1tp, *b2tp;

    bool firstRunM = false;
    bool firstRunV = false;

    if (!memory_manager.tensorExists(Name("AdamQuantized") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("AdamQuantized") / param.getName() / "beta_1_t", 1, 1, 1, 1, this->m_beta_1);
    }
    b1tp = &memory_manager.getTensor(Name("AdamQuantized") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("AdamQuantized") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("AdamQuantized") / param.getName() / "beta_2_t", 1, 1, 1, 1, this->m_beta_2);
    }
    b2tp = &memory_manager.getTensor(Name("AdamQuantized") / param.getName() / "beta_2_t");

    if (!checkTensorExists(Name("AdamQuantized") / param.getName() / "m"))
    {
        Name t = Name("AdamQuantized") / param.getName() / "m";
        mTensors[t] = std::vector<dtype>(param.size(), 0_dt);
        firstRunM = true;
    }
    std::vector<dtype>& m = mTensors.find(Name("AdamQuantized") / param.getName() / "m")->second;

    if (!checkTensorExists(Name("AdamQuantized") / param.getName() / "v"))
    {
        Name t = Name("AdamQuantized") / param.getName() / "v";
        mTensors[t] = std::vector<dtype>(param.size(), 0_dt);
        firstRunV = true;
    }
    std::vector<dtype>& v = mTensors.find(Name("AdamQuantized") / param.getName() / "v")->second;

    Tensor& beta_1_t = *b1tp;
    Tensor& beta_2_t = *b2tp;

    //auto dynamicMapSigned = createDynamicMap(true);
    //dynamicMapSigned[0] = -1.0f;
    //auto dynamicMapUnsigned = createDynamicMap(false);

    auto dynamicMapSigned = createNormalQuantileMap(true);
    auto dynamicMapUnsigned = createNormalQuantileMap(false);

    auto nameM = Name("AdamQuantized") / param.getName() / "m";
    auto nameV = Name("AdamQuantized") / param.getName() / "v";

    //if(!firstRunM) decompress(nameM);
    //if(!firstRunV) decompress(nameV);

    if(!firstRunM) decompressDynamic(nameM, dynamicMapSigned);
    if(!firstRunV) decompressDynamic(nameV, dynamicMapUnsigned);

    //if(!firstRunM) decompressQuantile(nameM);
    //if(!firstRunV) decompressQuantile(nameV);

    const auto sqrt_beta_2_t_0 = std::sqrt(1.0_dt - beta_2_t[0]);
    const auto alpha_new = this->m_alpha * sqrt_beta_2_t_0 / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = m_use_simple_epsilon ? this->m_epsilon : this->m_epsilon * sqrt_beta_2_t_0;
    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        m[i] = this->m_beta_1 * m[i] + (1.0_dt - this->m_beta_1) * grad[i];
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        v[i] = this->m_beta_2 * v[i] + (1.0_dt - this->m_beta_2) * grad[i] * grad[i];
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new)
        param[i] = param[i] - alpha_new * m[i] / (std::sqrt(v[i]) + epsilon_new);
    }

    //compress(nameM, mBlockSize);
    //compress(nameV, mBlockSize);

    compressDynamic(nameM, dynamicMapSigned, mBlockSize);
    compressDynamic(nameV, dynamicMapUnsigned, mBlockSize);

    //compressQuantile(nameM);
    //compressQuantile(nameV);

    beta_1_t[0] *= this->m_beta_1;
    beta_2_t[0] *= this->m_beta_2;
}

void AdamQuantized::compress(const raul::Name& name, size_t bucketSize)
{
    auto it = mTensors.find(name);
    if(it == mTensors.end()) THROW_NONAME("AdamQuantized[compress]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensors.find(name);

    if(itc == mCompressedTensors.end())
    {
        mCompressedTensors[name] = std::vector<CompressBucket>();
        itc = mCompressedTensors.find(name);
    }
    std::vector<CompressBucket>& cBuckets = itc->second;

    std::vector<dtype>& data = (*it).second;

    size_t buckets = data.size() / bucketSize;
    
    if(!buckets)
    {
        buckets = 1;
        bucketSize = data.size();
    }

    for (size_t q = 0; q < buckets; ++q)
    {
        size_t offsetFirst = q * bucketSize;
        size_t offsetLast = (q + 1) * bucketSize;
        if(q == buckets - 1) offsetLast = data.size();

        CompressBucket newBuck;
        auto minMax = std::minmax_element(data.begin() + offsetFirst, data.begin() + offsetLast);
        newBuck.mCompressInt8Min = *minMax.first;
        newBuck.mCompressInt8Max = *minMax.second;
        newBuck.mCompressedDataInt8.resize(offsetLast - offsetFirst);

        dtype unit = TODTYPE(255.0f) / (newBuck.mCompressInt8Max - newBuck.mCompressInt8Min);

        for (size_t w = offsetFirst, w2 = 0; w < offsetLast; ++w, ++w2)
        {
            newBuck.mCompressedDataInt8[w2] = static_cast<uint8_t>((data[w] - newBuck.mCompressInt8Min) * unit);
        }

        cBuckets.emplace_back(newBuck);
    }

    data.clear();
    data.shrink_to_fit();
}

void AdamQuantized::decompress(const raul::Name& name)
{
    auto it = mTensors.find(name);
    if(it == mTensors.end()) THROW_NONAME("AdamQuantized[decompress]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensors.find(name);
    if(itc == mCompressedTensors.end()) THROW_NONAME("AdamQuantized[decompress]", "Tensor [" + name + "] not found in compressed");
    std::vector<CompressBucket>& cBuckets = itc->second;

    std::vector<dtype>& data = (*it).second;

    const size_t buckets = cBuckets.size();

    if(!buckets)
    {
        THROW_NONAME("AdamQuantized[decompress]", "Buckets are empty");
    }

    size_t totalSize = std::accumulate(cBuckets.begin(), cBuckets.end(), static_cast<size_t>(0), [](size_t sum, const CompressBucket& buck) { return buck.mCompressedDataInt8.size() + sum; });

    data.resize(totalSize);

    size_t offset = 0;

    for (size_t q = 0; q < buckets; ++q)
    {
        dtype unit = (cBuckets[q].mCompressInt8Max - cBuckets[q].mCompressInt8Min) / TODTYPE(255.0f);

        for (size_t w = 0; w < cBuckets[q].mCompressedDataInt8.size(); ++w)
        {
            data[offset + w] = cBuckets[q].mCompressInt8Min + cBuckets[q].mCompressedDataInt8[w] * unit;
        }

        offset += cBuckets[q].mCompressedDataInt8.size();
    }

    cBuckets.clear();
    cBuckets.shrink_to_fit();
}

void AdamQuantized::compressDynamic(const raul::Name& name, const std::vector<dtype>& map, size_t bucketSize)
{
    auto it = mTensors.find(name);
    if(it == mTensors.end()) THROW_NONAME("AdamQuantized[compressDynamic]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensors.find(name);

    if(itc == mCompressedTensors.end())
    {
        mCompressedTensors[name] = std::vector<CompressBucket>();
        itc = mCompressedTensors.find(name);
    }
    std::vector<CompressBucket>& cBuckets = itc->second;

    std::vector<dtype>& data = (*it).second;

    size_t buckets = data.size() / bucketSize;
    
    if(!buckets)
    {
        buckets = 1;
        bucketSize = data.size();
    }

    for (size_t q = 0; q < buckets; ++q)
    {
        size_t offsetFirst = q * bucketSize;
        size_t offsetLast = (q + 1) * bucketSize;
        if(q == buckets - 1) offsetLast = data.size();

        CompressBucket newBuck;
        newBuck.mCompressInt8Max = std::abs(*std::max_element(data.begin() + offsetFirst, data.begin() + offsetLast, abs_compare));
        newBuck.mCompressedDataInt8.resize(offsetLast - offsetFirst);

        for (size_t w = offsetFirst, w2 = 0; w < offsetLast; ++w, ++w2)
        {
            dtype normed = data[w] / newBuck.mCompressInt8Max;
            size_t index = std::lower_bound(map.begin(), map.end(), normed) - map.begin();
            if (index < 255)
            {
                float dist_left = fabs(normed - (map[index]));
                float dist_right = fabs(normed - (map[index + 1]));
                if(dist_right < dist_left) ++index;
            }
            newBuck.mCompressedDataInt8[w2] = static_cast<uint8_t>(index);
        }

        cBuckets.emplace_back(newBuck);
    }

    data.clear();
    data.shrink_to_fit();
}

void AdamQuantized::decompressDynamic(const raul::Name& name, const std::vector<dtype>& map)
{
    auto it = mTensors.find(name);
    if(it == mTensors.end()) THROW_NONAME("AdamQuantized[decompressDynamic]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensors.find(name);
    if(itc == mCompressedTensors.end()) THROW_NONAME("AdamQuantized[decompressDynamic]", "Tensor [" + name + "] not found in compressed");
    std::vector<CompressBucket>& cBuckets = itc->second;

    std::vector<dtype>& data = (*it).second;

    const size_t buckets = cBuckets.size();

    if(!buckets)
    {
        THROW_NONAME("AdamQuantized[decompressDynamic]", "Buckets are empty");
    }

    size_t totalSize = std::accumulate(cBuckets.begin(), cBuckets.end(), static_cast<size_t>(0), [](size_t sum, const CompressBucket& buck) { return buck.mCompressedDataInt8.size() + sum; });

    data.resize(totalSize);

    size_t offset = 0;

    for (size_t q = 0; q < buckets; ++q)
    {
        for (size_t w = 0; w < cBuckets[q].mCompressedDataInt8.size(); ++w)
        {
            data[offset + w] = map[cBuckets[q].mCompressedDataInt8[w]] * cBuckets[q].mCompressInt8Max;
        }

        offset += cBuckets[q].mCompressedDataInt8.size();
    }

    cBuckets.clear();
    cBuckets.shrink_to_fit();
}

void AdamQuantized::compressDynamicFP16(const raul::Name& name, const std::vector<half>& map, size_t bucketSize)
{
    auto it = mTensorsFP16.find(name);
    if(it == mTensorsFP16.end()) THROW_NONAME("AdamQuantized[compressDynamicFP16]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensors.find(name);

    if(itc == mCompressedTensors.end())
    {
        mCompressedTensors[name] = std::vector<CompressBucket>();
        itc = mCompressedTensors.find(name);
    }
    std::vector<CompressBucket>& cBuckets = itc->second;

    std::vector<half>& data = (*it).second;

    size_t buckets = data.size() / bucketSize;
    
    if(!buckets)
    {
        buckets = 1;
        bucketSize = data.size();
    }

    for (size_t q = 0; q < buckets; ++q)
    {
        size_t offsetFirst = q * bucketSize;
        size_t offsetLast = (q + 1) * bucketSize;
        if(q == buckets - 1) offsetLast = data.size();

        CompressBucket newBuck;
        newBuck.mCompressInt8Max = std::abs(TODTYPE(*std::max_element(data.begin() + offsetFirst, data.begin() + offsetLast, abs_compare)));
        newBuck.mCompressedDataInt8.resize(offsetLast - offsetFirst);

        for (size_t w = offsetFirst, w2 = 0; w < offsetLast; ++w, ++w2)
        {
            dtype normed = data[w] / newBuck.mCompressInt8Max;
            size_t index = std::lower_bound(map.begin(), map.end(), normed) - map.begin();
            if (index < 255)
            {
                float dist_left = fabs(normed - (map[index]));
                float dist_right = fabs(normed - (map[index + 1]));
                if(dist_right < dist_left) ++index;
            }
            newBuck.mCompressedDataInt8[w2] = static_cast<uint8_t>(index);
        }

        cBuckets.emplace_back(newBuck);
    }

    data.clear();
    data.shrink_to_fit();
}

void AdamQuantized::decompressDynamicFP16(const raul::Name& name, const std::vector<half>& map)
{
    auto it = mTensorsFP16.find(name);
    if(it == mTensorsFP16.end()) THROW_NONAME("AdamQuantized[decompressDynamicFP16]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensors.find(name);
    if(itc == mCompressedTensors.end()) THROW_NONAME("AdamQuantized[decompressDynamicFP16]", "Tensor [" + name + "] not found in compressed");
    std::vector<CompressBucket>& cBuckets = itc->second;

    std::vector<half>& data = (*it).second;

    const size_t buckets = cBuckets.size();

    if(!buckets)
    {
        THROW_NONAME("AdamQuantized[decompressDynamicFP16]", "Buckets are empty");
    }

    size_t totalSize = std::accumulate(cBuckets.begin(), cBuckets.end(), static_cast<size_t>(0), [](size_t sum, const CompressBucket& buck) { return buck.mCompressedDataInt8.size() + sum; });

    data.resize(totalSize);

    size_t offset = 0;

    for (size_t q = 0; q < buckets; ++q)
    {
        for (size_t w = 0; w < cBuckets[q].mCompressedDataInt8.size(); ++w)
        {
            data[offset + w] = map[cBuckets[q].mCompressedDataInt8[w]] * cBuckets[q].mCompressInt8Max;
        }

        offset += cBuckets[q].mCompressedDataInt8.size();
    }

    cBuckets.clear();
    cBuckets.shrink_to_fit();
}

void AdamQuantized::compressQuantile(const raul::Name& name)
{
    auto it = mTensors.find(name);
    if(it == mTensors.end()) THROW_NONAME("AdamQuantized[compressQuantile]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensorsQuantile.find(name);

    if(itc == mCompressedTensorsQuantile.end())
    {
        mCompressedTensorsQuantile[name] = CompressQuantile();
        itc = mCompressedTensorsQuantile.find(name);
    }
    CompressQuantile& cBucket = itc->second;

    std::vector<dtype>& data = (*it).second;

    cBucket.mCompressInt8Max = std::abs(*std::max_element(data.begin(), data.end(), abs_compare));
    dtype maxVal = cBucket.mCompressInt8Max;
    std::transform(data.begin(), data.end(), data.begin(), [maxVal](const dtype val) { return val / maxVal; });
    cBucket.mMap = createQuantileMap(data);
    cBucket.mCompressedDataInt8.resize(data.size());

    for (size_t w = 0; w < data.size(); ++w)
    {
        size_t index = std::lower_bound(cBucket.mMap.begin(), cBucket.mMap.end(), data[w]) - cBucket.mMap.begin();
        if (index < 255)
        {
            float dist_left = fabs(data[w] - (cBucket.mMap[index]));
            float dist_right = fabs(data[w] - (cBucket.mMap[index + 1]));
            if(dist_right < dist_left) ++index;
        }
        cBucket.mCompressedDataInt8[w] = static_cast<uint8_t>(index);
    }


    data.clear();
    data.shrink_to_fit();
}

void AdamQuantized::decompressQuantile(const raul::Name& name)
{
    auto it = mTensors.find(name);
    if(it == mTensors.end()) THROW_NONAME("AdamQuantized[decompressQuantile]", "Tensor [" + name + "] not found");

    auto itc = mCompressedTensorsQuantile.find(name);
    if(itc == mCompressedTensorsQuantile.end()) THROW_NONAME("AdamQuantized[decompressQuantile]", "Tensor [" + name + "] not found in compressed");
    CompressQuantile& cBucket = itc->second;

    std::vector<dtype>& data = (*it).second;

    data.resize(cBucket.mCompressedDataInt8.size());

    for (size_t w = 0; w < cBucket.mCompressedDataInt8.size(); ++w)
    {
        data[w] = cBucket.mMap[cBucket.mCompressedDataInt8[w]] * cBucket.mCompressInt8Max;
    }

    cBucket.mMap.clear();
    cBucket.mMap.shrink_to_fit();
    cBucket.mCompressedDataInt8.clear();
    cBucket.mCompressedDataInt8.shrink_to_fit();
}

std::vector<dtype> AdamQuantized::createQuantileMap(const std::vector<dtype>& param)
{
    std::vector<dtype> ret(256, 0.0f);

    std::vector<dtype> sorted(param.begin(), param.end());
    std::sort(sorted.begin(), sorted.end());
    //dtype maxVal = std::abs(*std::max_element(sorted.begin(), sorted.end(), abs_compare));
    //std::transform(sorted.begin(), sorted.end(), sorted.begin(), [maxVal](const dtype val) { return val / maxVal; });

    for (size_t q = 0; q < 256; ++q)
    {
        float qA = static_cast<float>(q) / 257.0f;
        float qB = static_cast<float>(q + 1) / 257.0f;
        size_t indA = static_cast<size_t>(std::floor(qA * static_cast<float>(sorted.size())));
        size_t indB = static_cast<size_t>(std::floor(qB * static_cast<float>(sorted.size())));
        ret[q] = (sorted[indA] + sorted[indB]) / 2.0f;
    }

    return ret;
}

std::vector<dtype> AdamQuantized::createNormalQuantileMap(bool isSigned)
{
    //bitsandbytes/functional.py optimal_normal, optimal_half_normal

    std::vector<dtype> ret;

    if (isSigned)
    {
        ret = { -1.0f, -0.8727636337280273f, -0.8097418546676636f, -0.7660024166107178f, -0.7318882346153259f, -0.6793879270553589f, -0.657649040222168f, -0.6385974884033203f, -0.6211113333702087f, -0.5901028513908386f, -0.5762918591499329f, -0.5630806684494019f, -0.5509274005889893f, -0.5394591689109802f, -0.5283197164535522f, -0.517780065536499f, -0.5074946284294128f, -0.4980469048023224f, -0.48867011070251465f, -0.48003149032592773f, -0.47125306725502014f, -0.4629971981048584f, -0.4547359049320221f, -0.446626216173172f, -0.43902668356895447f, -0.43158355355262756f, -0.4244747757911682f, -0.4173796474933624f, -0.41038978099823f, -0.4055633544921875f, -0.4035947024822235f, -0.39701032638549805f, -0.39057496190071106f, -0.38439232110977173f, -0.3782760500907898f, -0.3721940815448761f, -0.3661896586418152f, -0.3604033589363098f, -0.354605108499527f, -0.34892538189888f, -0.34320303797721863f, -0.3376772701740265f, -0.3323028087615967f, -0.3269782066345215f, -0.32166096568107605f, -0.316457599401474f, -0.3112771809101105f, -0.3061025142669678f, -0.30106794834136963f, -0.2961243987083435f, -0.2912728488445282f, -0.28644347190856934f, -0.28165507316589355f, -0.2769731283187866f, -0.2722635865211487f, -0.26779335737228394f, -0.26314786076545715f, -0.2586647868156433f, -0.2541804611682892f, -0.2496625930070877f, -0.24527113139629364f, -0.24097171425819397f, -0.23659978806972504f, -0.23218469321727753f, -0.22799566388130188f, -0.22380566596984863f, -0.21965542435646057f, -0.2154538631439209f, -0.2113603949546814f, -0.20735277235507965f, -0.20334717631340027f, -0.19932441413402557f, -0.19530178606510162f, -0.19136647880077362f, -0.18736697733402252f, -0.18337111175060272f, -0.17951400578022003f, -0.1757056713104248f, -0.17182783782482147f, -0.1680615097284317f, -0.16431649029254913f, -0.16053077578544617f, -0.15685945749282837f, -0.15298527479171753f, -0.1493264138698578f, -0.14566898345947266f, -0.14188314974308014f, -0.13819937407970428f, -0.1344561129808426f, -0.1306886374950409f, -0.1271020770072937f, -0.12346585839986801f, -0.11981867253780365f, -0.11614970862865448f, -0.11256207525730133f, -0.10889036953449249f, -0.10525048524141312f, -0.1016591489315033f, -0.09824034571647644f, -0.09469068050384521f, -0.0911419615149498f, -0.08773849159479141f, -0.08416644483804703f, -0.08071305602788925f, -0.07720902562141418f, -0.07371306419372559f, -0.07019119709730148f, -0.06673648208379745f, -0.06329209357500076f, -0.059800852090120316f, -0.0564190037548542f, -0.05296570807695389f, -0.049522045999765396f, -0.04609023034572601f, -0.04262964054942131f, -0.039246633648872375f, -0.03577171266078949f, -0.03236335143446922f, -0.028855687007308006f, -0.02542758360505104f, -0.022069433704018593f, -0.018754752352833748f, -0.015386369079351425f, -0.01194947212934494f, -0.008439815603196621f, -0.004995611496269703f, -0.0016682245768606663f, 0.0f, 0.0015510577941313386f, 0.005062474869191647f, 0.008417150937020779f, 0.011741090565919876f, 0.015184164978563786f, 0.018582714721560478f, 0.02204744517803192f, 0.025471193715929985f, 0.02889077737927437f, 0.0323684960603714f, 0.03579240292310715f, 0.039281025528907776f, 0.0427563451230526f, 0.04619763046503067f, 0.04968220740556717f, 0.05326594039797783f, 0.05679265409708023f, 0.060245808213949203f, 0.06372645497322083f, 0.06721872836351395f, 0.0706876739859581f, 0.0742349922657013f, 0.07774098962545395f, 0.08123527467250824f, 0.08468879014253616f, 0.08810535818338394f, 0.09155989438295364f, 0.09498448669910431f, 0.0985206812620163f, 0.10206405073404312f, 0.10563778132200241f, 0.10921968519687653f, 0.11284469068050385f, 0.11653254181146622f, 0.12008969485759735f, 0.12368203699588776f, 0.1272617131471634f, 0.13089501857757568f, 0.134552001953125f, 0.1382799744606018f, 0.14194637537002563f, 0.14563234150409698f, 0.14930322766304016f, 0.15303383767604828f, 0.1567956507205963f, 0.16050070524215698f, 0.16431072354316711f, 0.16813558340072632f, 0.17204202711582184f, 0.1758781224489212f, 0.17973239719867706f, 0.1836014688014984f, 0.18753431737422943f, 0.19138391315937042f, 0.19535475969314575f, 0.19931404292583466f, 0.20333819091320038f, 0.20738255977630615f, 0.21152682602405548f, 0.21568812429904938f, 0.21978361904621124f, 0.22393859922885895f, 0.22814159095287323f, 0.23241068422794342f, 0.23675410449504852f, 0.24123944342136383f, 0.24569889903068542f, 0.2500703036785126f, 0.25904011726379395f, 0.26349544525146484f, 0.2682226300239563f, 0.272907555103302f, 0.2774306833744049f, 0.28220856189727783f, 0.2869136929512024f, 0.2916390895843506f, 0.29649388790130615f, 0.30142995715141296f, 0.3065022826194763f, 0.3114383816719055f, 0.31648796796798706f, 0.3216581642627716f, 0.32700115442276f, 0.3322487473487854f, 0.33778008818626404f, 0.3431521952152252f, 0.3487405776977539f, 0.3543166518211365f, 0.3601346015930176f, 0.36605337262153625f, 0.37217751145362854f, 0.378179669380188f, 0.3843980133533478f, 0.3906566798686981f, 0.39714935421943665f, 0.40357843041419983f, 0.4104187488555908f, 0.4171563684940338f, 0.42418959736824036f, 0.43136918544769287f, 0.4389212429523468f, 0.44673123955726624f, 0.45457619428634644f, 0.4627031683921814f, 0.47130417823791504f, 0.4798591434955597f, 0.48897242546081543f, 0.4979848861694336f, 0.5f, 0.5076631307601929f, 0.5177803635597229f, 0.5282770991325378f, 0.5392990112304688f, 0.5506287813186646f, 0.5632893443107605f, 0.5764452815055847f, 0.5903191566467285f, 0.6051878333091736f, 0.6209936141967773f, 0.6382884979248047f, 0.6573970913887024f, 0.6795773506164551f, 0.7037051916122437f, 0.7327037453651428f, 0.7677436470985413f, 0.8111193776130676f, 0.875165581703186f, 1.0f };
    }
    else
    {
        ret = { 0.0f, 0.005811259150505066f, 0.00961565226316452f, 0.010822802782058716f, 0.013123787939548492f, 0.014242202043533325f, 0.0143156498670578f, 0.016469404101371765f, 0.017666727304458618f, 0.01773911714553833f, 0.0199756920337677f, 0.0210941880941391f, 0.021161124110221863f, 0.02451971173286438f, 0.024580076336860657f, 0.02685210108757019f, 0.028012827038764954f, 0.030198264867067337f, 0.0302925705909729f, 0.03136435151100159f, 0.03374280035495758f, 0.03487399220466614f, 0.035243816673755646f, 0.037192340940237045f, 0.03822284936904907f, 0.04164902865886688f, 0.04173608124256134f, 0.04401407018303871f, 0.04508155584335327f, 0.047482021152973175f, 0.04756556823849678f, 0.050963032990694046f, 0.05196474492549896f, 0.055417388677597046f, 0.05793146416544914f, 0.05799369141459465f, 0.05887940526008606f, 0.05895659327507019f, 0.062420234084129333f, 0.06493274495005608f, 0.06499008461833f, 0.06935599446296692f, 0.07197384163737297f, 0.07201516255736351f, 0.07276943325996399f, 0.07283210754394531f, 0.07550075277686119f, 0.07975354790687561f, 0.07980883121490479f, 0.08257630094885826f, 0.0867777168750763f, 0.08682405948638916f, 0.08967285975813866f, 0.09323835000395775f, 0.09386616945266724f, 0.09735457599163055f, 0.09739077091217041f, 0.10092401504516602f, 0.10444298386573792f, 0.10447832942008972f, 0.10770941898226738f, 0.10803905129432678f, 0.11161200702190399f, 0.1151546835899353f, 0.11520349979400635f, 0.11875157058238983f, 0.11879390478134155f, 0.1222602017223835f, 0.122351735830307f, 0.12240418791770935f, 0.12594850733876228f, 0.12597402930259705f, 0.12602100148797035f, 0.12960633635520935f, 0.1296597123146057f, 0.12966342642903328f, 0.13227657973766327f, 0.13325360417366028f, 0.1333133578300476f, 0.13691483438014984f, 0.1371927298605442f, 0.14066261053085327f, 0.14088113978505135f, 0.1447291411459446f, 0.14805573225021362f, 0.148526418954134f, 0.15170684456825256f, 0.15178103744983673f, 0.15225710347294807f, 0.1554398238658905f, 0.15609459951519966f, 0.15618794038891792f, 0.1592724472284317f, 0.1629735231399536f, 0.16382690146565437f, 0.16676269471645355f, 0.16873238794505596f, 0.17066434025764465f, 0.17068277299404144f, 0.1717144437134266f, 0.17558929696679115f, 0.17827065289020538f, 0.17835864424705505f, 0.18222273886203766f, 0.18353315070271492f, 0.18604370951652527f, 0.18611834943294525f, 0.1876586265861988f, 0.18996606767177582f, 0.19170701876282692f, 0.19398853182792664f, 0.19786442816257477f, 0.19795633852481842f, 0.20195159316062927f, 0.2058800607919693f, 0.2099103182554245f, 0.2122517265379429f, 0.21410366892814636f, 0.21819619834423065f, 0.22221362590789795f, 0.22233009338378906f, 0.22500130906701088f, 0.2251257635653019f, 0.22638091444969177f, 0.23067741096019745f, 0.23368822410702705f, 0.2348879873752594f, 0.2382080741226673f, 0.2390350103378296f, 0.2391497790813446f, 0.24253453686833382f, 0.24265171959996223f, 0.2470107562839985f, 0.24764248728752136f, 0.24777774512767792f, 0.2516774423420429f, 0.256104726344347f, 0.2564055472612381f, 0.2607169933617115f, 0.265461727976799f, 0.26985861361026764f, 0.2701106257736683f, 0.2702729292213917f, 0.274574413895607f, 0.2750340588390827f, 0.27919672429561615f, 0.283704474568367f, 0.28386808931827545f, 0.28953738883137703f, 0.2896753139793873f, 0.29320384562015533f, 0.29451676085591316f, 0.295327290892601f, 0.29802779853343964f, 0.29818175733089447f, 0.29972871020436287f, 0.30290623009204865f, 0.30305664241313934f, 0.30486901476979256f, 0.31299956142902374f, 0.31518544629216194f, 0.31790371239185333f, 0.3205283172428608f, 0.3230419009923935f, 0.32595496252179146f, 0.32612212374806404f, 0.3282426446676254f, 0.3283906430006027f, 0.33146094158291817f, 0.3316439874470234f, 0.33365286886692047f, 0.33723779395222664f, 0.3390095978975296f, 0.3427443392574787f, 0.34853987768292427f, 0.34869300201535225f, 0.35457711294293404f, 0.35537679493427277f, 0.3604113645851612f, 0.36124424636363983f, 0.3665340431034565f, 0.36667295172810555f, 0.3727492541074753f, 0.3729033060371876f, 0.37888188660144806f, 0.37907837703824043f, 0.3792510814964771f, 0.38557394221425056f, 0.38573457673192024f, 0.39108292758464813f, 0.39911722019314766f, 0.40589402988553047f, 0.40604450181126595f, 0.410498782992363f, 0.4106704741716385f, 0.4129834659397602f, 0.4131447561085224f, 0.4172855168581009f, 0.4202354736626148f, 0.4204071946442127f, 0.43538858368992805f, 0.4355536885559559f, 0.4432900734245777f, 0.44603554904460907f, 0.4461968094110489f, 0.451409537345171f, 0.4598204083740711f, 0.46002377942204475f, 0.46178819239139557f, 0.46868549659848213f, 0.46995367109775543f, 0.4868385046720505f, 0.48702501133084297f, 0.4958047419786453f, 0.4960057884454727f, 0.5051481872797012f, 0.506847757846117f, 0.5148334950208664f, 0.5150565356016159f, 0.5174009390175343f, 0.5249751061201096f, 0.5283288545906544f, 0.5355450958013535f, 0.539984006434679f, 0.5467876642942429f, 0.5522958822548389f, 0.5584012717008591f, 0.5706631988286972f, 0.5836620181798935f, 0.5836880058050156f, 0.5942088551819324f, 0.5975865572690964f, 0.6102624125778675f, 0.6124880760908127f, 0.6286389082670212f, 0.646102175116539f, 0.6471664495766163f, 0.665437325835228f, 0.6687244363129139f, 0.687017485499382f, 0.6932839937508106f, 0.7115348428487778f, 0.7218200154602528f, 0.7219699807465076f, 0.7747527211904526f, 0.7749756425619125f, 0.8192005604505539f, 0.8194110840559006f, 0.8830635994672775f, 0.9217727445065975f, 0.9245667457580566f, 0.947742685675621f, 0.9674464613199234f, 0.9890814647078514f, 0.9891453236341476f, 1.0f };
    }

    return ret;
}

std::vector<half> AdamQuantized::createNormalQuantileMapFP16(bool isSigned)
{
    //bitsandbytes/functional.py optimal_normal, optimal_half_normal

    std::vector<half> ret;

    if (isSigned)
    {
        ret = { -1.0_hf, -0.8727636337280273_hf, -0.8097418546676636_hf, -0.7660024166107178_hf, -0.7318882346153259_hf, -0.6793879270553589_hf, -0.657649040222168_hf, -0.6385974884033203_hf, -0.6211113333702087_hf, -0.5901028513908386_hf, -0.5762918591499329_hf, -0.5630806684494019_hf, -0.5509274005889893_hf, -0.5394591689109802_hf, -0.5283197164535522_hf, -0.517780065536499_hf, -0.5074946284294128_hf, -0.4980469048023224_hf, -0.48867011070251465_hf, -0.48003149032592773_hf, -0.47125306725502014_hf, -0.4629971981048584_hf, -0.4547359049320221_hf, -0.446626216173172_hf, -0.43902668356895447_hf, -0.43158355355262756_hf, -0.4244747757911682_hf, -0.4173796474933624_hf, -0.41038978099823_hf, -0.4055633544921875_hf, -0.4035947024822235_hf, -0.39701032638549805_hf, -0.39057496190071106_hf, -0.38439232110977173_hf, -0.3782760500907898_hf, -0.3721940815448761_hf, -0.3661896586418152_hf, -0.3604033589363098_hf, -0.354605108499527_hf, -0.34892538189888_hf, -0.34320303797721863_hf, -0.3376772701740265_hf, -0.3323028087615967_hf, -0.3269782066345215_hf, -0.32166096568107605_hf, -0.316457599401474_hf, -0.3112771809101105_hf, -0.3061025142669678_hf, -0.30106794834136963_hf, -0.2961243987083435_hf, -0.2912728488445282_hf, -0.28644347190856934_hf, -0.28165507316589355_hf, -0.2769731283187866_hf, -0.2722635865211487_hf, -0.26779335737228394_hf, -0.26314786076545715_hf, -0.2586647868156433_hf, -0.2541804611682892_hf, -0.2496625930070877_hf, -0.24527113139629364_hf, -0.24097171425819397_hf, -0.23659978806972504_hf, -0.23218469321727753_hf, -0.22799566388130188_hf, -0.22380566596984863_hf, -0.21965542435646057_hf, -0.2154538631439209_hf, -0.2113603949546814_hf, -0.20735277235507965_hf, -0.20334717631340027_hf, -0.19932441413402557_hf, -0.19530178606510162_hf, -0.19136647880077362_hf, -0.18736697733402252_hf, -0.18337111175060272_hf, -0.17951400578022003_hf, -0.1757056713104248_hf, -0.17182783782482147_hf, -0.1680615097284317_hf, -0.16431649029254913_hf, -0.16053077578544617_hf, -0.15685945749282837_hf, -0.15298527479171753_hf, -0.1493264138698578_hf, -0.14566898345947266_hf, -0.14188314974308014_hf, -0.13819937407970428_hf, -0.1344561129808426_hf, -0.1306886374950409_hf, -0.1271020770072937_hf, -0.12346585839986801_hf, -0.11981867253780365_hf, -0.11614970862865448_hf, -0.11256207525730133_hf, -0.10889036953449249_hf, -0.10525048524141312_hf, -0.1016591489315033_hf, -0.09824034571647644_hf, -0.09469068050384521_hf, -0.0911419615149498_hf, -0.08773849159479141_hf, -0.08416644483804703_hf, -0.08071305602788925_hf, -0.07720902562141418_hf, -0.07371306419372559_hf, -0.07019119709730148_hf, -0.06673648208379745_hf, -0.06329209357500076_hf, -0.059800852090120316_hf, -0.0564190037548542_hf, -0.05296570807695389_hf, -0.049522045999765396_hf, -0.04609023034572601_hf, -0.04262964054942131_hf, -0.039246633648872375_hf, -0.03577171266078949_hf, -0.03236335143446922_hf, -0.028855687007308006_hf, -0.02542758360505104_hf, -0.022069433704018593_hf, -0.018754752352833748_hf, -0.015386369079351425_hf, -0.01194947212934494_hf, -0.008439815603196621_hf, -0.004995611496269703_hf, -0.0016682245768606663_hf, 0.0_hf, 0.0015510577941313386_hf, 0.005062474869191647_hf, 0.008417150937020779_hf, 0.011741090565919876_hf, 0.015184164978563786_hf, 0.018582714721560478_hf, 0.02204744517803192_hf, 0.025471193715929985_hf, 0.02889077737927437_hf, 0.0323684960603714_hf, 0.03579240292310715_hf, 0.039281025528907776_hf, 0.0427563451230526_hf, 0.04619763046503067_hf, 0.04968220740556717_hf, 0.05326594039797783_hf, 0.05679265409708023_hf, 0.060245808213949203_hf, 0.06372645497322083_hf, 0.06721872836351395_hf, 0.0706876739859581_hf, 0.0742349922657013_hf, 0.07774098962545395_hf, 0.08123527467250824_hf, 0.08468879014253616_hf, 0.08810535818338394_hf, 0.09155989438295364_hf, 0.09498448669910431_hf, 0.0985206812620163_hf, 0.10206405073404312_hf, 0.10563778132200241_hf, 0.10921968519687653_hf, 0.11284469068050385_hf, 0.11653254181146622_hf, 0.12008969485759735_hf, 0.12368203699588776_hf, 0.1272617131471634_hf, 0.13089501857757568_hf, 0.134552001953125_hf, 0.1382799744606018_hf, 0.14194637537002563_hf, 0.14563234150409698_hf, 0.14930322766304016_hf, 0.15303383767604828_hf, 0.1567956507205963_hf, 0.16050070524215698_hf, 0.16431072354316711_hf, 0.16813558340072632_hf, 0.17204202711582184_hf, 0.1758781224489212_hf, 0.17973239719867706_hf, 0.1836014688014984_hf, 0.18753431737422943_hf, 0.19138391315937042_hf, 0.19535475969314575_hf, 0.19931404292583466_hf, 0.20333819091320038_hf, 0.20738255977630615_hf, 0.21152682602405548_hf, 0.21568812429904938_hf, 0.21978361904621124_hf, 0.22393859922885895_hf, 0.22814159095287323_hf, 0.23241068422794342_hf, 0.23675410449504852_hf, 0.24123944342136383_hf, 0.24569889903068542_hf, 0.2500703036785126_hf, 0.25904011726379395_hf, 0.26349544525146484_hf, 0.2682226300239563_hf, 0.272907555103302_hf, 0.2774306833744049_hf, 0.28220856189727783_hf, 0.2869136929512024_hf, 0.2916390895843506_hf, 0.29649388790130615_hf, 0.30142995715141296_hf, 0.3065022826194763_hf, 0.3114383816719055_hf, 0.31648796796798706_hf, 0.3216581642627716_hf, 0.32700115442276_hf, 0.3322487473487854_hf, 0.33778008818626404_hf, 0.3431521952152252_hf, 0.3487405776977539_hf, 0.3543166518211365_hf, 0.3601346015930176_hf, 0.36605337262153625_hf, 0.37217751145362854_hf, 0.378179669380188_hf, 0.3843980133533478_hf, 0.3906566798686981_hf, 0.39714935421943665_hf, 0.40357843041419983_hf, 0.4104187488555908_hf, 0.4171563684940338_hf, 0.42418959736824036_hf, 0.43136918544769287_hf, 0.4389212429523468_hf, 0.44673123955726624_hf, 0.45457619428634644_hf, 0.4627031683921814_hf, 0.47130417823791504_hf, 0.4798591434955597_hf, 0.48897242546081543_hf, 0.4979848861694336_hf, 0.5_hf, 0.5076631307601929_hf, 0.5177803635597229_hf, 0.5282770991325378_hf, 0.5392990112304688_hf, 0.5506287813186646_hf, 0.5632893443107605_hf, 0.5764452815055847_hf, 0.5903191566467285_hf, 0.6051878333091736_hf, 0.6209936141967773_hf, 0.6382884979248047_hf, 0.6573970913887024_hf, 0.6795773506164551_hf, 0.7037051916122437_hf, 0.7327037453651428_hf, 0.7677436470985413_hf, 0.8111193776130676_hf, 0.875165581703186_hf, 1.0_hf };
    }
    else
    {
        ret = { 0.0_hf, 0.005811259150505066_hf, 0.00961565226316452_hf, 0.010822802782058716_hf, 0.013123787939548492_hf, 0.014242202043533325_hf, 0.0143156498670578_hf, 0.016469404101371765_hf, 0.017666727304458618_hf, 0.01773911714553833_hf, 0.0199756920337677_hf, 0.0210941880941391_hf, 0.021161124110221863_hf, 0.02451971173286438_hf, 0.024580076336860657_hf, 0.02685210108757019_hf, 0.028012827038764954_hf, 0.030198264867067337_hf, 0.0302925705909729_hf, 0.03136435151100159_hf, 0.03374280035495758_hf, 0.03487399220466614_hf, 0.035243816673755646_hf, 0.037192340940237045_hf, 0.03822284936904907_hf, 0.04164902865886688_hf, 0.04173608124256134_hf, 0.04401407018303871_hf, 0.04508155584335327_hf, 0.047482021152973175_hf, 0.04756556823849678_hf, 0.050963032990694046_hf, 0.05196474492549896_hf, 0.055417388677597046_hf, 0.05793146416544914_hf, 0.05799369141459465_hf, 0.05887940526008606_hf, 0.05895659327507019_hf, 0.062420234084129333_hf, 0.06493274495005608_hf, 0.06499008461833_hf, 0.06935599446296692_hf, 0.07197384163737297_hf, 0.07201516255736351_hf, 0.07276943325996399_hf, 0.07283210754394531_hf, 0.07550075277686119_hf, 0.07975354790687561_hf, 0.07980883121490479_hf, 0.08257630094885826_hf, 0.0867777168750763_hf, 0.08682405948638916_hf, 0.08967285975813866_hf, 0.09323835000395775_hf, 0.09386616945266724_hf, 0.09735457599163055_hf, 0.09739077091217041_hf, 0.10092401504516602_hf, 0.10444298386573792_hf, 0.10447832942008972_hf, 0.10770941898226738_hf, 0.10803905129432678_hf, 0.11161200702190399_hf, 0.1151546835899353_hf, 0.11520349979400635_hf, 0.11875157058238983_hf, 0.11879390478134155_hf, 0.1222602017223835_hf, 0.122351735830307_hf, 0.12240418791770935_hf, 0.12594850733876228_hf, 0.12597402930259705_hf, 0.12602100148797035_hf, 0.12960633635520935_hf, 0.1296597123146057_hf, 0.12966342642903328_hf, 0.13227657973766327_hf, 0.13325360417366028_hf, 0.1333133578300476_hf, 0.13691483438014984_hf, 0.1371927298605442_hf, 0.14066261053085327_hf, 0.14088113978505135_hf, 0.1447291411459446_hf, 0.14805573225021362_hf, 0.148526418954134_hf, 0.15170684456825256_hf, 0.15178103744983673_hf, 0.15225710347294807_hf, 0.1554398238658905_hf, 0.15609459951519966_hf, 0.15618794038891792_hf, 0.1592724472284317_hf, 0.1629735231399536_hf, 0.16382690146565437_hf, 0.16676269471645355_hf, 0.16873238794505596_hf, 0.17066434025764465_hf, 0.17068277299404144_hf, 0.1717144437134266_hf, 0.17558929696679115_hf, 0.17827065289020538_hf, 0.17835864424705505_hf, 0.18222273886203766_hf, 0.18353315070271492_hf, 0.18604370951652527_hf, 0.18611834943294525_hf, 0.1876586265861988_hf, 0.18996606767177582_hf, 0.19170701876282692_hf, 0.19398853182792664_hf, 0.19786442816257477_hf, 0.19795633852481842_hf, 0.20195159316062927_hf, 0.2058800607919693_hf, 0.2099103182554245_hf, 0.2122517265379429_hf, 0.21410366892814636_hf, 0.21819619834423065_hf, 0.22221362590789795_hf, 0.22233009338378906_hf, 0.22500130906701088_hf, 0.2251257635653019_hf, 0.22638091444969177_hf, 0.23067741096019745_hf, 0.23368822410702705_hf, 0.2348879873752594_hf, 0.2382080741226673_hf, 0.2390350103378296_hf, 0.2391497790813446_hf, 0.24253453686833382_hf, 0.24265171959996223_hf, 0.2470107562839985_hf, 0.24764248728752136_hf, 0.24777774512767792_hf, 0.2516774423420429_hf, 0.256104726344347_hf, 0.2564055472612381_hf, 0.2607169933617115_hf, 0.265461727976799_hf, 0.26985861361026764_hf, 0.2701106257736683_hf, 0.2702729292213917_hf, 0.274574413895607_hf, 0.2750340588390827_hf, 0.27919672429561615_hf, 0.283704474568367_hf, 0.28386808931827545_hf, 0.28953738883137703_hf, 0.2896753139793873_hf, 0.29320384562015533_hf, 0.29451676085591316_hf, 0.295327290892601_hf, 0.29802779853343964_hf, 0.29818175733089447_hf, 0.29972871020436287_hf, 0.30290623009204865_hf, 0.30305664241313934_hf, 0.30486901476979256_hf, 0.31299956142902374_hf, 0.31518544629216194_hf, 0.31790371239185333_hf, 0.3205283172428608_hf, 0.3230419009923935_hf, 0.32595496252179146_hf, 0.32612212374806404_hf, 0.3282426446676254_hf, 0.3283906430006027_hf, 0.33146094158291817_hf, 0.3316439874470234_hf, 0.33365286886692047_hf, 0.33723779395222664_hf, 0.3390095978975296_hf, 0.3427443392574787_hf, 0.34853987768292427_hf, 0.34869300201535225_hf, 0.35457711294293404_hf, 0.35537679493427277_hf, 0.3604113645851612_hf, 0.36124424636363983_hf, 0.3665340431034565_hf, 0.36667295172810555_hf, 0.3727492541074753_hf, 0.3729033060371876_hf, 0.37888188660144806_hf, 0.37907837703824043_hf, 0.3792510814964771_hf, 0.38557394221425056_hf, 0.38573457673192024_hf, 0.39108292758464813_hf, 0.39911722019314766_hf, 0.40589402988553047_hf, 0.40604450181126595_hf, 0.410498782992363_hf, 0.4106704741716385_hf, 0.4129834659397602_hf, 0.4131447561085224_hf, 0.4172855168581009_hf, 0.4202354736626148_hf, 0.4204071946442127_hf, 0.43538858368992805_hf, 0.4355536885559559_hf, 0.4432900734245777_hf, 0.44603554904460907_hf, 0.4461968094110489_hf, 0.451409537345171_hf, 0.4598204083740711_hf, 0.46002377942204475_hf, 0.46178819239139557_hf, 0.46868549659848213_hf, 0.46995367109775543_hf, 0.4868385046720505_hf, 0.48702501133084297_hf, 0.4958047419786453_hf, 0.4960057884454727_hf, 0.5051481872797012_hf, 0.506847757846117_hf, 0.5148334950208664_hf, 0.5150565356016159_hf, 0.5174009390175343_hf, 0.5249751061201096_hf, 0.5283288545906544_hf, 0.5355450958013535_hf, 0.539984006434679_hf, 0.5467876642942429_hf, 0.5522958822548389_hf, 0.5584012717008591_hf, 0.5706631988286972_hf, 0.5836620181798935_hf, 0.5836880058050156_hf, 0.5942088551819324_hf, 0.5975865572690964_hf, 0.6102624125778675_hf, 0.6124880760908127_hf, 0.6286389082670212_hf, 0.646102175116539_hf, 0.6471664495766163_hf, 0.665437325835228_hf, 0.6687244363129139_hf, 0.687017485499382_hf, 0.6932839937508106_hf, 0.7115348428487778_hf, 0.7218200154602528_hf, 0.7219699807465076_hf, 0.7747527211904526_hf, 0.7749756425619125_hf, 0.8192005604505539_hf, 0.8194110840559006_hf, 0.8830635994672775_hf, 0.9217727445065975_hf, 0.9245667457580566_hf, 0.947742685675621_hf, 0.9674464613199234_hf, 0.9890814647078514_hf, 0.9891453236341476_hf, 1.0_hf };
    }

    return ret;
}

std::vector<dtype> AdamQuantized::createDynamicMap(bool isSigned, int n)
{
    //bitsandbytes/functional.py create_dynamic_map()
    std::vector<dtype> ret;

    int additionalItems = static_cast<int>(std::pow(2.0f, 7 - n) - 1);
    if(!isSigned) additionalItems *= 2;

    int q = 0;
    for (; q < n; ++q)
    {
        size_t fractionItems = static_cast<size_t>(std::pow(2.0f, q + 7 - n) + 1);
        if(!isSigned) fractionItems = static_cast<size_t>(std::pow(2.0f, q + 7 - n + 1) + 1);

        auto boundaries = linspace(0.1_dt, 1.0_dt, fractionItems);
        for (size_t w = 0; w < boundaries.size() - 1; ++w)
        {
            boundaries[w] += boundaries[w + 1];
            boundaries[w] /= 2.0_dt;
            ret.push_back(TODTYPE(std::pow(10.0f, -(n - 1) + q) * boundaries[w]));
            if(isSigned)
                ret.push_back(TODTYPE(-std::pow(10.0f, -(n - 1) + q) * boundaries[w]));
        }
    }

    --q;

    if (additionalItems > 0)
    {
        auto boundaries = linspace(0.1_dt, 1.0_dt, additionalItems + 1);
        for (size_t w = 0; w < boundaries.size() - 1; ++w)
        {
            boundaries[w] += boundaries[w + 1];
            boundaries[w] /= 2.0_dt;
            ret.push_back(TODTYPE(std::pow(10.0f, -(n - 1) + q) * boundaries[w]));
            if(isSigned)
                ret.push_back(TODTYPE(-std::pow(10.0f, -(n - 1) + q) * boundaries[w]));
        }
    }

    ret.push_back(0.0_dt);
    ret.push_back(1.0_dt);
    std::sort(ret.begin(), ret.end());

    return ret;
}

std::vector<dtype> AdamQuantized::linspace(dtype start_in, dtype end_in, size_t n)
{

    std::vector<dtype> linspaced(n);

    dtype delta = (end_in - start_in) / TODTYPE(n - 1);

    std::generate(linspaced.begin(), linspaced.end(), [delta, n = start_in]()mutable{auto result = n; n += delta; return result;});

    return linspaced;
}

void AdamQuantized::optimize(MemoryManagerFP16& memory_manager, TensorFP16& param, const TensorFP16& grad)
{
    if (param.size() != grad.size())
    {
        THROW_NONAME("AdamQuantized", "parameters and gradients must have the same size (" + Conversions::toString(param.size()) + " != " + Conversions::toString(grad.size()) + ")");
    }

    bool firstRunM = false;
    bool firstRunV = false;

    TensorFP16 *b1tp, *b2tp;
    if (!memory_manager.tensorExists(Name("AdamQuantized") / param.getName() / "beta_1_t"))
    {
        b1tp = memory_manager.createTensor(Name("AdamQuantized") / param.getName() / "beta_1_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_1));
    }
    b1tp = &memory_manager.getTensor(Name("AdamQuantized") / param.getName() / "beta_1_t");

    if (!memory_manager.tensorExists(Name("AdamQuantized") / param.getName() / "beta_2_t"))
    {
        b2tp = memory_manager.createTensor(Name("AdamQuantized") / param.getName() / "beta_2_t", 1, 1, 1, 1, TOHTYPE(this->m_beta_2));
    }
    b2tp = &memory_manager.getTensor(Name("AdamQuantized") / param.getName() / "beta_2_t");

    if (!checkTensorExistsFP16(Name("AdamQuantized") / param.getName() / "m"))
    {
        Name t = Name("AdamQuantized") / param.getName() / "m";
        mTensorsFP16[t] = std::vector<half>(param.size(), 0_hf);
        firstRunM = true;
    }
    std::vector<half>& m = mTensorsFP16.find(Name("AdamQuantized") / param.getName() / "m")->second;

    if (!checkTensorExistsFP16(Name("AdamQuantized") / param.getName() / "v"))
    {
        Name t = Name("AdamQuantized") / param.getName() / "v";
        mTensorsFP16[t] = std::vector<half>(param.size(), 0_hf);
        firstRunV = true;
    }
    std::vector<half>& v = mTensorsFP16.find(Name("AdamQuantized") / param.getName() / "v")->second;

    TensorFP16& beta_1_t = *b1tp;
    TensorFP16& beta_2_t = *b2tp;

    auto dynamicMapSigned = createNormalQuantileMapFP16(true);
    auto dynamicMapUnsigned = createNormalQuantileMapFP16(false);

    auto nameM = Name("AdamQuantized") / param.getName() / "m";
    auto nameV = Name("AdamQuantized") / param.getName() / "v";

    if(!firstRunM) decompressDynamicFP16(nameM, dynamicMapSigned);
    if(!firstRunV) decompressDynamicFP16(nameV, dynamicMapUnsigned);

    const auto sqrt_beta_2_t_0 = std::sqrt(1.0_dt - beta_2_t[0]);
    const auto alpha_new = this->m_alpha * sqrt_beta_2_t_0 / (1.0_dt - beta_1_t[0]);
    const auto epsilon_new = m_use_simple_epsilon ? this->m_epsilon : this->m_epsilon * sqrt_beta_2_t_0;
    const auto n = param.size();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < n; i++)
    {
        // m_new = beta_1*m + (1-beta_1)*grad
        const auto mTmp = this->m_beta_1 * TODTYPE(m[i]) + (1.0_dt - this->m_beta_1) * TODTYPE(grad[i]);
        m[i] = TOHTYPE(mTmp);
        // v_new = beta_2*v + (1-beta_2)*grad*grad
        const auto vTmp = this->m_beta_2 * TODTYPE(v[i]) + (1.0_dt - this->m_beta_2) * TODTYPE(grad[i]) * TODTYPE(grad[i]);
        v[i] = TOHTYPE(vTmp);
        // param_new = param - alpha_new*m_new/(sqrt(v_new) + epsilon_new)
        param[i] = TOHTYPE(TODTYPE(param[i]) - alpha_new * mTmp / (std::sqrt(vTmp) + epsilon_new));
    }

    compressDynamicFP16(nameM, dynamicMapSigned, mBlockSize);
    compressDynamicFP16(nameV, dynamicMapUnsigned, mBlockSize);

    beta_1_t[0] = TOHTYPE(TODTYPE(beta_1_t[0]) * this->m_beta_1);
    beta_2_t[0] = TOHTYPE(TODTYPE(beta_2_t[0]) * this->m_beta_2);
}

std::ostream& AdamQuantized::as_ostream(std::ostream& out) const
{
    out << "AdamQuantized(alpha=" << std::scientific << this->m_alpha << ", beta_1=" << this->m_beta_1 << ", beta_2=" << this->m_beta_2 << ", epsilon=" << this->m_epsilon << ")";
    return out;
}

} // raul::optimizers
