// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "LabelSmoothingCPU.h"
#include "../LabelSmoothing.h"

namespace
{

const size_t NoPadding = std::numeric_limits<std::size_t>::max();

} // anonymous namespace

namespace raul
{
template<typename MM>
void LabelSmoothingCPU<MM>::forwardComputeImpl(NetworkMode mode)
{
    Workflow& work = mLayer.mNetworkParams.mWorkflow;

    auto& output = work.getMemoryManager<MM>()[mLayer.mOutputName];

    const auto& inputs = work.getMemoryManager<MM>()[mLayer.mInputName];

    if (mode == NetworkMode::Test)
    {
        std::copy(inputs.begin(), inputs.end(), output.begin());
        return;
    }

    size_t width = inputs.getWidth();
    size_t N = inputs.getBatchSize() * inputs.getDepth() * inputs.getHeight();
    auto confidence = (1.0_dt - mLayer.mSmoothing);

    const auto inData2D = inputs.reshape(yato::dims(N, width));
    auto outData2D = output.reshape(inData2D.dimensions());

    dtype bias = mLayer.mSmoothing / TODTYPE(width - 1);

    if (mLayer.mPaddingIdx != NoPadding)
    {
        bias = mLayer.mSmoothing / TODTYPE(width - 2);
    }

    std::fill(output.begin(), output.end(), TOMMTYPE(bias));

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t q = 0; q < N; ++q)
    {
        bool isZero = true;
        for (size_t k = 0; k < width; ++k)
        {
            if (isZero)
            {
                isZero = (inData2D[q][k] == TOMMTYPE(0.0_dt) || k == mLayer.mPaddingIdx);
            }
            if (k == mLayer.mPaddingIdx)
            {
                outData2D[q][k] = 0;
            }
            else if (inData2D[q][k] > 0)
            {
                outData2D[q][k] = inData2D[q][k] * TOMMTYPE(confidence);
            }
        }
        if (isZero)
        {
            std::fill(outData2D[q].begin(), outData2D[q].end(), TOMMTYPE(0.0_dt));
        }
    }
}

template class LabelSmoothingCPU<MemoryManager>;
template class LabelSmoothingCPU<MemoryManagerFP16>;
} // namespace raul