// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRANSFORMER_PARAMS_H
#define TRANSFORMER_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

/**
 * @param input names of input tensors (source, target, source_mask, target_mask)
 * @param output name of output tensor
 */
struct TransformerParams : public TrainableParams
{
    TransformerParams() = delete;
    TransformerParams(const Names& inputs,
                      const raul::Name& output,
                      unsigned int paramSrcVocabSize,
                      unsigned int paramTgtVocabSize,
                      unsigned int paramEncoderDecoderLength = 6,
                      unsigned int paramModelSize = 512,
                      unsigned int paramFeedForwardSize = 2048,
                      unsigned int paramHeadsCount = 8,
                      float paramDropout = 0.1f,
                      bool frozen = false);

    unsigned int srcVocabSize;
    unsigned int tgtVocabSize;
    unsigned int encoderDecoderLength;
    unsigned int modelSize;
    unsigned int feedForwardSize;
    unsigned int heads;
    float dropout;

    void print(std::ostream&) const override;
};

} // raul namespace
#endif // TRANSFORMER_PARAMS_H
