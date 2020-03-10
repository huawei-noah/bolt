// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifdef _USE_FP16
#include <iostream>
#include "type.h"
#include "tensor_desc.h"
#include "sequential_ocl.hpp"
#include "factory.hpp"
#include "ocl/factory_ocl.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"


void print_help() {

    std::cout << "please set argvs:      "  <<std::endl;
    std::cout << "usage: argv[1]:  opName"  <<std::endl;
    std::cout << "usage: argv[2]:  in"      <<std::endl;
    std::cout << "usage: argv[3]:  ic"      <<std::endl;
    std::cout << "usage: argv[4]:  ih"      <<std::endl;
    std::cout << "usage: argv[5]:  iw"      <<std::endl;
    std::cout << "usage: argv[6]:  fn"      <<std::endl;
    std::cout << "usage: argv[7]:  fc"      <<std::endl;
    std::cout << "usage: argv[8]:  fh"      <<std::endl;
    std::cout << "usage: argv[9]:  fw"      <<std::endl;
    std::cout << "usage: argv[10]: sw"      <<std::endl;
    std::cout << "usage: argv[11]: sh"      <<std::endl;
    std::cout << "usage: argv[12]: pl"      <<std::endl;
    std::cout << "usage: argv[13]: pr"      <<std::endl;
    std::cout << "usage: argv[14]: pt"      <<std::endl;
    std::cout << "usage: argv[15]: pb"      <<std::endl;
    std::cout << "usage: argv[16]: inputNum"<<std::endl;
    std::cout << "usage: argv[17]: pm"      <<std::endl;
    std::cout << "usage: argv[18]: dt"      <<std::endl;
    std::cout << "supported op: OT_Pooling" <<std::endl;
    std::cout << "supported op: OT_Conv" <<std::endl;
    std::cout << "supported op: OT_Eltwise" <<std::endl;
    std::cout << "supported op: OT_Softmax" <<std::endl;
    std::cout << "supported op: OT_Relu" <<std::endl;
    std::cout << "supported op: OT_Relu6" <<std::endl;
    std::cout << "supported op: OT_HSwish" <<std::endl;
    std::cout << "supported op: OT_HSigmoid" <<std::endl;
    std::cout << "supported op: OT_HGelu" <<std::endl;
    std::cout << "supported op: OT_TanH" <<std::endl;
    std::cout << "supported op: OT_FC" <<std::endl;
    std::cout << "supported op: OT_Scale" <<std::endl;
    std::cout << "supported op: OT_Concat" <<std::endl;
}

template <typename T>
void buildInputTensor(DataType dt, DataFormat df, U32 n, U32 c, U32 h, U32 w, Vec<TensorDesc>* dims, Vec<Tensor>* inputTensors){
    TensorDesc inputDesc = tensor4df(dt, df, n, c, h, w);
    U32 inputNum  = tensorNumElements(inputDesc);
    U32 inputSize = tensorNumBytes(inputDesc); 
    U8* inputVal = (U8*) operator new (inputSize);
   
    T* data = (T*) inputVal;
    if(dt == DT_F16){
        for(U32 i = 0; i < inputNum; i++) data[i] = (T)(rand() & 255) / 256.0 - 0.5;
    }
    if(dt == DT_U8){
        for(U32 i = 0; i < inputNum; i++) {
            data[i] = (T)(i & 255);
        }
    }
    std::shared_ptr<Tensor> inputTensor = std::shared_ptr<Tensor>(new Tensor());
    inputTensor->set_desc(inputDesc);
    inputTensor->set_val(inputVal);

    dims->push_back(inputDesc);
    inputTensors->push_back(*inputTensor.get());
}

int main(int argc, char* argv[]) {

    if(argc != 16 && argc != 17 && argc != 18 &&argc != 19) {
        printf("%d\n", argc);
        print_help();
        return 0;
    }
  
    U32 inputNum = 1;
    std::string pm = "NULL";
    std::string DT_NAME = "F16";
    std::string opName = argv[1];


    U32 in     = atoi(argv[2]);
    U32 ic     = atoi(argv[3]);
    U32 ih     = atoi(argv[4]);
    U32 iw     = atoi(argv[5]);

    U32 fn     = atoi(argv[6]);
    U32 fc     = atoi(argv[7]);
    U32 fh     = atoi(argv[8]);
    U32 fw     = atoi(argv[9]);

    U32 sw     = atoi(argv[10]);
    U32 sh     = atoi(argv[11]);
    U32 pl     = atoi(argv[12]);
    U32 pr     = atoi(argv[13]);
    U32 pt     = atoi(argv[14]);
    U32 pb     = atoi(argv[15]);
    if(argc == 17){
        inputNum = atoi(argv[16]);
    }
    if(argc == 18){
        pm = argv[17];
    }
    if(argc == 19){
        DT_NAME = argv[18];
    }


    const Arch A = MALI;
    DataType dt = DT_F16;
    auto model = new SequentialOcl(A, dt, opName);
    std::shared_ptr<SequentialOcl> model_ptr = std::shared_ptr<SequentialOcl>(model);

    OperatorType OType;
    if(opName == "OT_Pooling")             OType = OT_Pooling;
    if(opName == "OT_Conv")                OType = OT_Conv;
    if(opName == "OT_Eltwise")             OType = OT_Eltwise;
    if(opName == "OT_Softmax")             OType = OT_Softmax;
    if(opName == "OT_Relu")                OType = OT_Relu;
    if(opName == "OT_Relu6")               OType = OT_Relu6;
    if(opName == "OT_HSwish")              OType = OT_HSwish;
    if(opName == "OT_HSigmoid")            OType = OT_HSigmoid;
    if(opName == "OT_Gelu")                OType = OT_Gelu;
    if(opName == "OT_TanH")                OType = OT_TanH;
    if(opName == "OT_Sigmoid")             OType = OT_Sigmoid;
    if(opName == "OT_FC")                  OType = OT_FC;
    if(opName == "OT_Scale")               OType = OT_Scale;
    if(opName == "OT_Concat")              OType = OT_Concat;
    Factory* factory_ocl = (Factory*)(new FactoryOCL());
    std::shared_ptr<Factory> factory;
    factory = std::shared_ptr<Factory>(factory_ocl);
    ConvolutionMode convMode;
    convMode = Convolution_Depthwise_Pointwise;
//    convMode = Convolution_Pointwise;

    switch(OType) {
        case OT_Pooling: {
            auto op = factory->createPooling(PoolingMode::POOLING_MAX, fh, fw, sh, sw, pt, pb, pl, pr, RoundMode::CEIL);
            model_ptr->add(op);
            break;
        }
        case OT_Eltwise: {
            auto op = factory->createEltwise(EltwiseMode::ELTWISE_SUM, 0, NULL);
            model_ptr->add(op);
            break;
        }
        case OT_Softmax: {
            auto op = factory->createSoftmax(dt);
            model_ptr->add(op);
            break;
        }
        case OT_Conv: {
            if(pm == "NULL") {
                //auto op = factory->createConvolution(dt, fn, fh, fw, sh, sw, pt, pb, pl, pr, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Pointwise, 1, 1, 1);
                //auto op = factory->createConvolution(dt, fn, fh, fw, sh, sw, pt, pb, pl, pr, ACTIVATION_NULL, ACTIVATION_NULL, Convolution_Depthwise, 1, 1, 1);
                auto op = factory->createConvolution(dt, fn, fh, fw, sh, sw, pt, pb, pl, pr, ACTIVATION_NULL, ACTIVATION_NULL, convMode, 1, 1, 1);
                model_ptr->add(op);
            }

            if(pm == "RELU") {
                //auto op = factory->createConvolution(dt, fn, fh, fw, sh, sw, pt, pb, pl, pr, ACTIVATION_NULL, ACTIVATION_RELU, Convolution_Pointwise, 1, 1, 1);
                auto op = factory->createConvolution(dt, fn, fh, fw, sh, sw, pt, pb, pl, pr, ACTIVATION_RELU, ACTIVATION_NULL, convMode, 1, 1, 1);
                model_ptr->add(op);
            }
            break;
        }
        case OT_Relu: {
            auto op = factory->createActivation(ACTIVATION_RELU);
            model_ptr->add(op);
            break;
        }
        case OT_Relu6: {
            auto op = factory->createActivation(ACTIVATION_RELU6);
            model_ptr->add(op);
            break;
        }
        case OT_HSwish: {
            auto op = factory->createActivation(ACTIVATION_H_SWISH);
            model_ptr->add(op);
            break;
        }
        case OT_HSigmoid: {
            auto op = factory->createActivation(ACTIVATION_H_SIGMOID);
            model_ptr->add(op);
            break;
        }
        case OT_Gelu: {
            auto op = factory->createActivation(ACTIVATION_GELU);
            model_ptr->add(op);
            break;
        }
        case OT_TanH: {
            auto op = factory->createActivation(ACTIVATION_TANH);
            model_ptr->add(op);
            break;
        }
        case OT_Sigmoid: {
            auto op = factory->createActivation(ACTIVATION_SIGMOID);
            model_ptr->add(op);
            break;
        }
        case OT_FC: {
            auto op = factory->createFullyConnectedEltwise(dt, ih * iw * ic, fn);
            model_ptr->add(op);
            break;
        }
        case OT_Scale: {
            auto op = factory->createScale(dt, fc, inputNum);
            model_ptr->add(op);
            break;
        }
        case OT_Concat: {
            auto op = factory->createConcat(1);
            model_ptr->add(op);
            break;
        }
        default: std::cout << "not support op" << std::endl;
    }

    Vec<TensorDesc> dims;
    Vec<Tensor> inputTensors;
    for(U32 i = 0; i < inputNum; i++){
        buildInputTensor<F16>(dt, DF_NCHW, in, ic, ih, iw, &dims, &inputTensors);
    }

    U8* weightVal = NULL;
    if(OType == OT_Conv){
        TensorDesc weightDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
        U32 weightNum  = tensorNumElements(weightDesc);
        U32 vectorNum  = fn;
        if(convMode == Convolution_Depthwise_Pointwise) vectorNum = fc + fn + fn * fc;
        U32 weightSize = tensorNumBytes(weightDesc) + vectorNum * bytesOf(dt);
        weightVal = (U8*) operator new (weightSize);
        F16* weight = (F16*) weightVal;
        for(U32 i = 0; i < weightNum + vectorNum; i++){
            weight[i] = (F16)(rand() & 255) / 256.0;
        }
    }

    if(OType == OT_FC){
        U32 weightNum = iw * ih * ic * fn;
        U32 biasNum   = fn;
        U32 weightSize = (weightNum + biasNum) * bytesOf(dt);
        weightVal = (U8*) operator new (weightSize);
        F16* weight = (F16*) weightVal;
        for(U32 i = 0; i < weightNum + biasNum; i++){
            weight[i] = (F16)(rand() & 255) / 256.0;
        }
    }

    if(OType == OT_Scale){
        U32 weightNum = fc;
        U32 biasNum   = fc;
        U32 weightSize = (weightNum + biasNum) * bytesOf(dt);
        weightVal = (U8*) operator new (weightSize);
        F16* weight = (F16*) weightVal;
        for(U32 i = 0; i < weightNum + biasNum; i++){
            weight[i] = (F16)(rand() & 255) / 256.0;
        }
    }

    if(weightVal){
        std::shared_ptr<U8> modelPtr(weightVal);
        model_ptr->ready(dims, modelPtr, 1);
    } else {
        model_ptr->ready(dims, NULL, 1);
    }
    model_ptr->mark_input_output();
    model_ptr->mali_prepare();
    model_ptr->set_input_tensors(inputTensors);
    model_ptr->run();
    
    auto output = model_ptr->get_output_tensors();
    output[0]->print();
/*    
    auto output = model_ptr->get_output_tensors_map();
    F16* val = (F16*) output[0];
    for(int i = 0; i < 64; i++) std::cout << val[i] << " ";
    std::cout << std::endl;
*/    
    return 0;
}
#endif
