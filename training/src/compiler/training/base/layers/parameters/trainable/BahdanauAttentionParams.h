// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef BAHDANAU_ATTENTION_PARAMS_H
#define BAHDANAU_ATTENTION_PARAMS_H

#include "TrainableParams.h"

namespace raul
{

/** Parameters for Bahdanau Attention Layer
 * @param inputs vector of names of input tensors
 * @param outputs vector of names of output tensors
 * @param weights vector of names of weight tensors
 * @param mNumUnits - the depth of the query mechanism
 * @param normalize specifies the type of attention - normalized or not
 * @param mSigmoidNoise - standard deviation of pre-sigmoid noise
 * @param mScoreBiasInit - initial value for score bias scalar.
 * It is recommended to initialize this to a negative value when the length of the memory is large
 * @param mMode specifies how to compute the attention distribution.
 * Now only "parallel" mode is available
 * @param stepwise specifies whether to use simple monotonic or stepwise monotonic attention.
 */

struct BahdanauAttentionParams : public TrainableParams
{
    BahdanauAttentionParams() = delete;

    BahdanauAttentionParams(const BasicParams& params,
                            size_t numUnits,
                            bool normalize = false,
                            dtype noise = 0.0_dt,
                            dtype bias = 0.0_dt,
                            const std::string& mode = "parallel",
                            bool stepwise = false,
                            bool oldSMA = true,
                            bool frozen = false);

    BahdanauAttentionParams(const BasicParams& params,
                            const Name& sharedLayer,
                            size_t numUnits,
                            bool normalize = false,
                            dtype noise = 0.0_dt,
                            dtype bias = 0.0_dt,
                            const std::string& mode = "parallel",
                            bool stepwise = false,
                            bool oldSMA = true,
                            bool frozen = false);

    // Standard parameters
    size_t mNumUnits;
    bool mNormalize;

    // BahdanauMonotonicAttention parameters
    dtype mSigmoidNoise;
    dtype mScoreBiasInit;
    std::string mMode;

    // Stepwise or not
    bool mStepwise;
    bool mOldSMA;

    void print(std::ostream& stream) const override;
};

}

#endif
