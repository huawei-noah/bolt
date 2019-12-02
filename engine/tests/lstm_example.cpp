// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <iostream>
#include <optional>

#include "sequential.hpp"
#include "factory.hpp"
#include "tensor.hpp"
#include "type.h"
#include "tensor_desc.h"


void print_help(char* argv[]) {
    printf("usage: %s", argv[0]);
}

int fake_input_initial(TensorDesc desc, Vec<Tensor> * input){
    (*input).clear();
    F16* arr = (F16*)operator new(tensorNumBytes(desc));
    for (U32 i = 0; i < tensorNumElements(desc); i++){
        arr[i] = 1;
    }
    U8* tmp_ptr = (U8*)arr;
    std::shared_ptr<U8> val(tmp_ptr);
    Tensor tmp_tensor(desc, val);
    (*input).push_back(tmp_tensor);
    return 0;
}

int main() {
    const Arch A = ARM_A76;
    DataType dt = DT_F16;
    auto model = Sequential<A>(dt, "lstm");

    auto op = Factory::createLstm<A>(dt, 1024);
    model.add(op);
    
    op = Factory::createLstm<A>(dt, 64);
    model.add(op);

    U32 inputDim = 15;
    TensorDesc input_desc = tensor3df(DT_F16, DF_MTK, 1, 1, inputDim);

    U32 weightNum = 4 * 1024 * (inputDim+1024) + 4 * 64 * (64 + 1024);
    auto weight = (F16*)operator new(weightNum*sizeof(F16));
    for (U32 i = 0; i < weightNum; i++) {
        weight[i] = 1;
    }
    U8 *w_ptr = (U8*)weight;
    std::shared_ptr<U8> modelPtr(w_ptr);
    model.ready({input_desc}, modelPtr);

    Vec<Tensor> inputs;
    fake_input_initial(input_desc, &inputs);
    model.set_input_tensors(inputs);
    model.run();
    auto outputs = model.get_output_tensors();
    for (auto output: outputs) {
        outputs[0].print<F16>();
    }
    return 0;
}

