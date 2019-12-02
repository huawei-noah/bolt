// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _HPP_INFERENCE
#define _HPP_INFERENCE

#include "model_tools.h"
#include "cnn.hpp"


Vec<TensorDesc> extractInputDims(const ModelSpec *ms) {
    Vec<TensorDesc> inputDims;
    int inputNum = ms->num_inputs;
    for (int i=0; i< inputNum; i++) {
        TensorDesc tmpDim;
        memcpy(&tmpDim, &(ms->input_dims[i]), sizeof(TensorDesc));
        inputDims.push_back(tmpDim);
    }
    return inputDims;
}


template<Arch A>
std::shared_ptr<CNN<A> > createCNN(const ModelSpec* ms)
{
    auto cnn = new CNN<A>(ms->dt, ms->model_name);

    cnn->sort_operators_sequential(ms);
    //TODO this function is not tested
    //cnn ->sort_operators_tp(ms);

    // create ops
    cnn->initialize_ops(ms);

    Vec<TensorDesc> dims = extractInputDims(ms);
    // assign space for output, tmp, bias, and trans_weight
    cnn->ready(dims); 

    CHECK_STATUS(cnn->mark_input_output(ms));

    return std::shared_ptr<CNN<A>>(cnn);
}

#endif
