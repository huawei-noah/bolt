// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_FEATURE
#define _H_FEATURE

#include <vector>
#include <map>
#include <string>

#include "farmhash.h"

#include "tensor_desc.h"

typedef enum Combiner {
    Combiner_Mean,
    Combiner_Sum,
} Combiner;
typedef int TYPE;

inline uint64_t shift_mix(const uint64_t val)
{
    return val ^ (val >> 47);
}

inline uint64_t FingerprintCat64(const uint64_t &fp1, const uint64_t &fp2)
{
    static const uint64_t kMul = 0xc6a4a7935bd1e995ULL;
    uint64_t result = fp1 ^ kMul;
    result ^= shift_mix(fp2 * kMul) * kMul;
    result *= kMul;
    result = shift_mix(result) * kMul;
    result = shift_mix(result);
    return result;
}

static inline uint64_t tf_hash(const std::string &s)
{
    return ::util::Fingerprint64(s.data(), s.size());
}

static inline uint64_t tf_hash64(const uint64_t &s, const uint64_t &key)
{
    return FingerprintCat64(key, s);
}

static inline uint64_t tf_hash64(const std::string &s, const uint64_t &key)
{
    return FingerprintCat64(key, tf_hash(s));
}

std::vector<TYPE> categorical_column_with_vocabulary_list(const std::vector<std::string> &input,
    std::map<std::string, TYPE> &vocab,
    int default_value = -1,
    int num_oov_buckets = 0)
{
    int vocab_size = vocab.size();
    std::vector<TYPE> ret(input.size());
    for (uint32_t i = 0; i < input.size(); i++) {
        if (vocab.find(input[i]) != vocab.end()) {
            ret[i] = vocab[input[i]];
        } else {
            if (num_oov_buckets > 0) {
                ret[i] = tf_hash(input[i]) % num_oov_buckets + vocab_size;
            } else {
                ret[i] = default_value;
            }
        }
    }
    return ret;
}

std::vector<TYPE> categorical_column_with_hash_bucket(
    const std::vector<std::string> &input, int hash_bucket_size)
{
    std::vector<TYPE> ret(input.size());
    for (uint32_t i = 0; i < input.size(); i++) {
        ret[i] = tf_hash(input[i]) % hash_bucket_size;
    }
    return ret;
}

std::vector<TYPE> categorical_column_with_identity(
    const std::vector<TYPE> &input, int bucket_size, int default_value = 0)
{
    std::vector<TYPE> ret(input.size());
    for (uint32_t i = 0; i < input.size(); i++) {
        if (input[i] < bucket_size) {
            ret[i] = input[i];
        } else {
            ret[i] = default_value;
        }
    }
    return ret;
}

inline uint32_t quick_search(
    const std::vector<TYPE> &data, const TYPE &query, const uint32_t &left, const uint32_t &right)
{
#if 1
    for (int j = left; j < right; j++) {
        if (query < data[j]) {
            return j;
        }
    }
#else
    if (left >= right) {
        return left;
    }
    int mid = (left + right) / 2;
    if (query < data[mid]) {
        return quick_search(data, query, left, mid);
    } else {
        return quick_search(data, query, mid, right);
    }
#endif
}

std::vector<TYPE> bucketized_column(
    const std::vector<TYPE> &input, const std::vector<TYPE> &boundaries)
{
    std::vector<TYPE> ret(input.size());
    uint32_t size = boundaries.size();
    for (uint32_t i = 0; i < input.size(); i++) {
        ret[i] = quick_search(boundaries, input[i], 0, size);
        ;
    }
    return ret;
}

void indicator_column(const TensorDesc &input_desc,
    const TYPE *input,
    int categorical_num,
    TensorDesc *output_desc,
    TYPE *output,
    const TYPE *weight = nullptr)
{
    *output_desc = input_desc;
    output_desc->dims[0] = categorical_num;
    uint32_t count = 1;
    for (uint32_t i = 1; i < input_desc.nDims; i++) {
        count *= input_desc.dims[i];
    }

    memset(output, 0, count * categorical_num * sizeof(TYPE));
    if (weight != nullptr) {
        for (uint32_t i = 0, j = 0, n = 0; i < count; i++, j += categorical_num) {
            for (uint32_t k = 0; k < input_desc.dims[0]; k++, n++) {
                output[j + input[n]] += weight[n];
            }
        }
    } else {
        for (uint32_t i = 0, j = 0, n = 0; i < count; i++, j += categorical_num) {
            for (uint32_t k = 0; k < input_desc.dims[0]; k++, n++) {
                output[j + input[n]]++;
            }
        }
    }
}

template <typename TI0, typename TI1>
std::vector<TYPE> crossed_column(const std::vector<TI0> &input0,
    const std::vector<TI1> &input1,
    int hash_bucket_size,
    const uint64_t hash_key = 0xDECAFCAFFE)
{
    std::vector<TYPE> ret(input0.size());
    for (uint32_t i = 0; i < input0.size(); i++) {
        ret[i] = tf_hash64(input1[i], tf_hash64(input0[i], hash_key)) % hash_bucket_size;
    }
    return ret;
}

template <typename TI, typename TO, typename F>
std::vector<TO> numeric_column(const std::vector<TI> &input,
    F const &normalizer_fn = nullptr,
    int shape = 0,
    TO default_value = -1)
{
    if (shape > 0) {
        return std::vector<TO>(shape, default_value);
    }
    std::vector<TO> ret = std::vector<TO>(input.size());
    if (normalizer_fn == nullptr) {
        for (uint32_t i = 0; i < input.size(); i++) {
            ret[i] = input[i];
        }
    } else {
        for (uint32_t i = 0; i < input.size(); i++) {
            ret[i] = normalizer_fn(input[i]);
        }
    }
    return ret;
}

template <Combiner combiner, typename T>
inline void embedding_combine(const std::vector<T *> &input, const uint32_t &dimension, T *output)
{
    if (input.size() == 0) {
        memset(output, 0, sizeof(T) * dimension);
        return;
    }
    if (combiner == Combiner_Mean || combiner == Combiner_Sum) {
        memcpy(output, input[0], sizeof(T) * dimension);
        for (uint32_t i = 1; i < input.size(); i++) {
            for (uint32_t j = 0; j < dimension; j++) {
                output[j] += input[i][j];
            }
        }
        if (combiner == Combiner_Mean) {
            for (uint32_t j = 0; j < dimension; j++) {
                output[j] /= input.size();
            }
        }
    } else {
        printf("[ERROR] currently not support combine function %d.\n", combiner);
    }
}

template <Combiner combiner, typename T>
void embedding_column(const TensorDesc &input_desc,
    const TYPE *input,
    const T *vocab,
    const uint32_t &dimension,
    TensorDesc *output_desc,
    T *output)
{
    *output_desc = input_desc;
    output_desc->dims[0] = dimension;
    uint32_t count = 1;
    for (uint32_t i = 1; i < input_desc.nDims; i++) {
        count *= input_desc.dims[i];
    }

    std::vector<T *> vec(input_desc.dims[0]);
    for (uint32_t i = 0, j = 0; i < count; i++, output += dimension) {
        for (uint32_t k = 0; k < input_desc.dims[0]; k++, j++) {
            vec[k] = vocab + input[j] * dimension;
        }
        embedding_combine<combiner, T>(vec, dimension, output);
    }
}
#endif
