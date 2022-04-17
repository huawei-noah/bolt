// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <training/base/common/Common.h>
#include <training/base/layers/parameters/trainable/TransformerParams.h>
#include <training/base/common/NetworkParameters.h>

namespace raul
{

/**
 * @brief Transformer Model
 *
 *
 * @see https://nlp.seas.harvard.edu/2018/04/03/attention.html
 */
class TransformerModel
{
  public:
    TransformerModel(const Name& name, const TransformerParams& params, NetworkParameters& networkParameters);

    TransformerModel(TransformerModel&&) = default;
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

  private:
};

void CreateGenerator(const Name& name, const BasicParams& params, size_t vocabularySize, NetworkParameters& networkParameters);

} // raul namespace
#endif // TRANSFORMER_MODEL_H