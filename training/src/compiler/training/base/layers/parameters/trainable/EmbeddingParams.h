// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef EMBEDDING_PARAMS_H
#define EMBEDDING_PARAMS_H

#include "TrainableParams.h"
#include <string>
#include <vector>

namespace raul
{

/**
 * @param input name of input tensor
 * @param output name of output tensor
 * @param dictSize width size in elements
 * @param embSize the size of each embedding vector
 * @param scaleByModelSize if given, this will multiply output by sqrt(embSize)
 * @param scaleGradByFreq if given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default: false
 */
struct EmbeddingParams : public TrainableParams
{
    EmbeddingParams() = delete;
    EmbeddingParams(const Name& input, const Name& output, size_t dictSize, size_t embSize, int paddingClassIdx = -1, bool scaleByModelSize = true, bool scaleGradByFreq = false)
        : TrainableParams(Names(1, input), Names(1, output))
        , dictionarySize(dictSize)
        , embeddingSize(embSize)
        , paddingClass(paddingClassIdx)
        , scaleOutput(scaleByModelSize)
        , scaleGradByFrequency(scaleGradByFreq)
    {
    }

    EmbeddingParams(const Name& input, const Name& output, bool paramFrozen, size_t dictSize, size_t embSize, int paddingClassIdx = -1, bool scaleByModelSize = true, bool scaleGradByFreq = false)
        : TrainableParams(Names(1, input), Names(1, output), paramFrozen)
        , dictionarySize(dictSize)
        , embeddingSize(embSize)
        , paddingClass(paddingClassIdx)
        , scaleOutput(scaleByModelSize)
        , scaleGradByFrequency(scaleGradByFreq)
    {
    }

    size_t dictionarySize;
    size_t embeddingSize;
    int paddingClass = -1;
    bool scaleOutput = true;
    bool scaleGradByFrequency = false;

    void print(std::ostream& stream) const override;
};

} // raul namespace

#endif // EMBEDDING_PARAMS_H