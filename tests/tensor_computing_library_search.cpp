// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "ut_util.h"
#include "tensor_computing.h"
#include "tensor_computing_library_algorithm_search.h"

int convolutionCPUFloatAlgorithmSearch(DataType dt){
    TensorDesc inputDesc, filterDesc, outputDesc;
    ConvolutionPolicy policy = CONVOLUTION_TUNNING;
    ActivationMode activationMode = ACTIVATION_RELU;
    ConvolutionDesc convDesc;
    convDesc.dilatedRate_h = 1;
    convDesc.dilatedRate_w = 1;
    U32 in = 1;
    for (int ic = 8; ic < libraryAlgorithmParameters["convolution_ic_max"];
        ic+=libraryAlgorithmParameters["convolution_ic_step"]) {
        for (int ih = 8; ih < libraryAlgorithmParameters["convolution_ih_max"];
            ih+=libraryAlgorithmParameters["convolution_ih_step"]) {
            for (int fn = 8; fn < libraryAlgorithmParameters["convolution_fn_max"];
                fn+=libraryAlgorithmParameters["convolution_fn_step"]) {
                for (int fh = 8; fh < libraryAlgorithmParameters["convolution_fh_max"];
                    fh+=libraryAlgorithmParameters["convolution_fh_step"]) {
                    for (int sh = 1; sh < fh; sh++) {
                        for (int ph = 0; ph < fh; ph++) {
                            if (ic % 8 != 0) {
                                inputDesc = tensor4df(dt, DF_NCHW, in, ic, ih, ih);
                            } else {
                                inputDesc = tensor4df(dt, DF_NCHWC8, in, ic, ih, ih);
                            }
                            convDesc.stride_h = sh;
                            convDesc.stride_w = sh;
                            convDesc.padding_top = ph;
                            convDesc.padding_bottom = ph;
                            convDesc.padding_left = ph;
                            convDesc.padding_right = ph;
                            filterDesc = tensor4df(dt, DF_NCHW, fn, ic, fh, fh);
                            U32 outputBytes = 0;
                            CHECK_STATUS(convolution_infer_output_size(inputDesc,
                                filterDesc, convDesc, &outputDesc, dt, &outputBytes, UT_ARCH));
                            ConvolutionForwardAlgorithm algorithm = CONVOLUTION_ALGORITHM_NULL;
                            CHECK_STATUS(convolution_infer_forward_algorithm(inputDesc,
                                filterDesc, outputDesc, convDesc, policy, &algorithm, dt, activationMode, UT_ARCH));

                            std::string name = getConvolutionAlgorithmMapNameFromInput(inputDesc,
                                filterDesc, convDesc, dt);
                            libraryAlgorithmMap[name] = algorithm;
                        }
                    }
                }
            }
        }
    }
    return 0;
}


int main() {
#ifdef _USE_FP16
    convolutionCPUFloatAlgorithmSearch(DT_F16);
#endif
#ifdef _USE_FP32
    convolutionCPUFloatAlgorithmSearch(DT_F32);
#endif
    saveLibraryAlgorithmMapToTxt();
    return 0;
}
