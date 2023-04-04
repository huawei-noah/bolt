// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <training/base/layers/BasicLayer.h>

#include <training/base/common/Common.h>

namespace raul
{

/**
 * @brief Range Positional Encoding Layer
 *
 * The layer encodes information about symbol position in sequence (e.g. word position in sentence) into its embedding vector.
 *
 */
class PositionalEncoding : public BasicLayer
{
  public:
    PositionalEncoding(const Name& name, const PositionalEncodingParams& params, NetworkParameters& networkParameters);

    PositionalEncoding(PositionalEncoding&&) = default;
    PositionalEncoding(const PositionalEncoding&) = delete;
    PositionalEncoding& operator=(const PositionalEncoding&) = delete;

  private:
    Name mInputName;
    Name mOutputName;

    size_t mModelSize = 2;
    size_t mMaxLength = 150;
    bool mDurationEncoding = false;
    size_t mMaxMelLength = 200;

    template<typename MM>
    friend class PositionalEncodingCPU;
};

} // raul namespace
#endif