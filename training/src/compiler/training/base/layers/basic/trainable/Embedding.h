// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <training/base/layers/TrainableLayer.h>
#include <training/base/layers/parameters/trainable/EmbeddingParams.h>

#include <map>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Word embeddings using lookup table
 *
 * A simple lookup table that stores embeddings of a fixed dictionary and size.
 * This module is often used to store word embeddings and retrieve them using indices.
 * The input to the module is a list of indices (either in width or height dimension), and the output is the corresponding word embeddings.
 */
class Embedding : public TrainableLayer
{
  public:
    Embedding(const Name& name, const EmbeddingParams& params, NetworkParameters& networkParameters);

    Embedding(Embedding&&) = default;
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;

  protected:
    Name mInputName;
    Name mOutputName;
    std::string mLutTensorName;

    std::map<size_t, size_t> mIndices;

    size_t mDictionarySize;
    size_t mEmbeddingSize;
    size_t mPaddingIdx;
    raul::dtype mOutputScale = 1.0_dt;
    bool mScaleGradByFreq = false;

    template<typename MM>
    friend class EmbeddingCPU;
};

} // raul namespace
#endif