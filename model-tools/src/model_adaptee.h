// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_MODELADAPTEE
#define _H_MODELADAPTEE

#include <string>
#include "model_tools.h"
#include "op_type.h"
#include "error.h"

#define UNIMPLEMENT() printf("UNIMPLEMENT %s CURRENTLY, ERROR OCCUR IN FILE: %s - LINE: %d \n", __PRETTY_FUNCTION__, __FILE__, __LINE__)

class ModelAdaptee {
public:
    virtual EE adapt(std::string dir, std::string mfn, ModelSpec* ms) {
	    EE ret;
	
        ret = parse_file(dir, mfn);

        ret = adapt_operators(ms);

        ret = adapt_weights(ms);

        return ret;
    }

    ModelAdaptee() {}
    virtual ~ModelAdaptee() {}

protected:
    virtual EE parse_file(std::string dir, std::string mfn) = 0;

    virtual EE adapt_operators(ModelSpec* ms) = 0;

    virtual EE adapt_weights(ModelSpec* ms) = 0;
    

    virtual EE adapt_operator(OperatorType type, ParameterSpec* ps) {
        if (type == OT_Input
         || type == OT_Softmax
         || type == OT_Relu
           )
        {
        } else if (type == OT_Conv) {
            *ps = adapt_Conv();
        } else if (type == OT_Deconvolution) {
	        *ps = adapt_Deconvolution();
	    } else if (type == OT_FC) {
            *ps = adapt_Fc();
        } else if (type == OT_Pooling) {
            *ps = adapt_Pooling();
        } else if (type == OT_Eltwise) {
            *ps = adapt_Eltwise();
        } else if (type == OT_BatchNorm) {
            *ps = adapt_BatchNorm();
        } else if (type == OT_Scale) {
            *ps = adapt_Scale();
        } else if (type == OT_Clip) {
            *ps = adapt_Clip();
        } else if (type == OT_LSTM) {
            *ps = adapt_LSTM();
        } else if (type == OT_Embedding) {
            *ps = adapt_Embedding();
        } else if (type == OT_Pad) {
            *ps = adapt_Pad();
        } else if (type == OT_LayerNorm) {
	        *ps = adapt_LayerNorm();
        } else if (type == OT_Multiply) {
            *ps = adapt_Multiply();
        } else if (type == OT_Reshape) {
            *ps = adapt_Reshape();
        } else if (type == OT_Slice) {
            *ps = adapt_Slice();
        } else if (type == OT_Transpose) {
            *ps = adapt_Transpose();
        } else if (type == OT_Attention) {
            *ps = adapt_Attention();
        } else if (type == OT_Upsample) {
            *ps = adapt_Upsample();
        } else if (type == OT_Cast) {
            *ps = adapt_Cast();
        } else if (type == OT_Gather) {
            *ps = adapt_Gather();
        } else if (type == OT_Unsqueeze) {
            *ps = adapt_Unsqueeze();
        } else if (type == OT_Squeeze) {
            *ps = adapt_Squeeze();
        } else if (type == OT_ArgMax) {
            *ps = adapt_ArgMax();
        } else if (type == OT_Repeat) {
            *ps = adapt_Repeat();
        } else if (type == OT_Check) {
            *ps = adapt_Check();
        } else if (type == OT_SharedWeight) {
            *ps = adapt_SharedWeight();
        } else if (type == OT_PreAllocatedMemory) {
            *ps = adapt_PreAllocatedMemory();
        } else if (type == OT_Copy) {
            *ps = adapt_Copy();
        } else if (type == OT_MatMul) {
            *ps = adapt_MatMul();
        } else if (type == OT_Interp) {
            *ps = adapt_Interp();
        } else if (type == OT_Flatten) {
            *ps = adapt_Flatten();
        }
        // Only these OPs have parameters
        return SUCCESS;
    }

    virtual ParameterSpec adapt_Interp()
    {
        UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Flatten()
    {
        UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Conv()
    {
        UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Deconvolution() 
    {
        UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Fc()
    {    
        UNIMPLEMENT();
	    ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Pooling()
    {
        UNIMPLEMENT();
	    ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Eltwise() {
	UNIMPLEMENT();
	ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_BatchNorm() {    
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Scale() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Clip() {
	UNIMPLEMENT();
	ParameterSpec curPs; 
	return curPs;   
    }

    virtual ParameterSpec adapt_LSTM() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Embedding() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Pad() { 
	UNIMPLEMENT();
	ParameterSpec curPs;  
	return curPs; 
    }

    virtual ParameterSpec adapt_LayerNorm() {
	UNIMPLEMENT();
	ParameterSpec curPs;   
	return curPs; 
    }

    virtual ParameterSpec adapt_Multiply() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Reshape() { 
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Slice() {
	UNIMPLEMENT();
	ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Transpose() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Attention() {    
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Upsample() { 
	UNIMPLEMENT();
	ParameterSpec curPs; 
	return curPs;
    }

    virtual ParameterSpec adapt_Cast() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Gather() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_Unsqueeze() {
	UNIMPLEMENT();
	ParameterSpec curPs;
	return curPs;
    }

    virtual ParameterSpec adapt_AxisMean() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Squeeze() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_ArgMax() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Repeat() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Check() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_PreAllocatedMemory() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_SharedWeight() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_Copy() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }

    virtual ParameterSpec adapt_MatMul() {
	UNIMPLEMENT();
        ParameterSpec curPs;
        return curPs;
    }
};

#undef UNIMPLEMENT
#endif
