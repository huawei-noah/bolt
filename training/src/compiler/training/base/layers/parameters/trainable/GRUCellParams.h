// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef GRU_CELL_PARAMS_H
#define GRU_CELL_PARAMS_H

#include <string>
#include <vector>

#include "TrainableParams.h"

namespace raul
{

struct GRUCellParams : public TrainableParams
{
    GRUCellParams() = delete;
    /** Parameters for GRU Cell layer
     * @param inputs vector of names of input tensors
     * @param outputs vector of names of output tensors
     * @param useBiasForInput enable or disable bias usage for input
     * @param useBiasForHidden enable or disable bias usage for hidden
     */
    GRUCellParams(const BasicParams& params, bool useBiasForInput = true, bool useBiasForHidden = true, bool frozen = false, bool useFusion = false)
        : TrainableParams(params, frozen)
        , mUseBiasForInput(useBiasForInput)
        , mUseBiasForHidden(useBiasForHidden)
        , mUseFusion(useFusion)
    {
    }

    /** Parameters for GRU Cell layer
     * @param input name of input tensor
     * @param inputHidden name of input hidden state tensor
     * @param outputHidden name of output hidden state tensor
     * @param useBiasForInput enable or disable bias usage for input
     * @param useBiasForHidden enable or disable bias usage for hidden
     */
    GRUCellParams(const raul::Name& input,
                  const raul::Name& inputHidden,
                  const raul::Name& outputHidden,
                  const Names& weights,
                  bool useBiasForInput = true,
                  bool useBiasForHidden = true,
                  bool frozen = false,
                  bool useFusion = false)
        : TrainableParams({ input, inputHidden }, { outputHidden }, weights, frozen)
        , mUseBiasForInput(useBiasForInput)
        , mUseBiasForHidden(useBiasForHidden)
        , mUseFusion(useFusion)
    {
    }

    void print(std::ostream& stream) const override;

    bool mUseBiasForInput;
    bool mUseBiasForHidden;
    bool mUseFusion;
};

} // raul namespace

#endif