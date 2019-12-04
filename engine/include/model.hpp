// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _MODEL_H
#define _MODEL_H
#include <math.h>
#include <cmath>
#include "operator.hpp"
#include "tensor_desc.h"

template<Arch A>
class Model {
public:
    /**
     * @param name
     */
    Model(DataType dt, std::string name) {
        this->dt = dt;
        this->name = name;
    }

    virtual EE infer_output_tensors_size(Vec<TensorDesc>) = 0;
    virtual void assign_output_tensor() = 0;
    virtual void infer_tmp_memory_size() = 0;
    virtual void assign_tmp_tensor() = 0;

    virtual void ready(Vec<TensorDesc> inDims) {
        infer_output_tensors_size(inDims);
        assign_output_tensor();

        infer_tmp_memory_size();
        assign_tmp_tensor();
    }

    virtual void run() {
#ifdef _DEBUG
        const char* funcStr = "[DEBUG] run()";
#endif
        for(auto op : ops) {
#ifdef _DEBUG
            std::cout << funcStr << " op: " << op->get_name() << "/"<< OperatorTypeName()[op->get_op_type()];
#endif
            op->run();

#ifdef _DEBUG
            // debug for nan
            Tensor outputTensor = op->get_output_tensors()[0];
            U32 elementNum = tensorNumElements(outputTensor.get_desc());
            F16* outputTensorPtr = (F16*)(outputTensor.get_val().get());
            for (U32 i = 0; i < elementNum; i++) {
                if (i < 32) {
                    if (i % 8 == 0)
                        std::cout << "\n    ";
                    std::cout << *outputTensorPtr << " ";
                }

                if (std::isinf(*outputTensorPtr)) {
                    std::cerr << "\n[ERROR] encounter inf" << std::endl;
                    exit(0);
                }
                if (std::isnan(*outputTensorPtr)) {
                    std::cerr << "\n[ERROR] encounter nan" << std::endl;
                    exit(0);
                }
                outputTensorPtr++;
            }
            std::cout << std::endl;
#endif
        }
    }

    virtual bool check() {
        for(auto op : this->ops) {
            if (! op->check()) return false;
        }
        return true;
    }

    std::string get_name() {return this->name;}

protected:
    Vec<std::shared_ptr<Operator<A> > > ops;
    DataType dt;

private:
    std::string name;
};
#endif
