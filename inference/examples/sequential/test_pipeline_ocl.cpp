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
#include "types.h"
#include "tensor_desc.h"
#include "sequential_ocl.hpp"
#include "factory.hpp"
#include "ocl/factory_ocl.hpp"
#include "tensor.hpp"
#include "data_loader.hpp"

void print_help()
{
    std::cout << "please set argvs:      " << std::endl;
    std::cout << "usage: argv[1]:  opName" << std::endl;
    std::cout << "usage: argv[2]:  in" << std::endl;
    std::cout << "usage: argv[3]:  ic" << std::endl;
    std::cout << "usage: argv[4]:  ih" << std::endl;
    std::cout << "usage: argv[5]:  iw" << std::endl;
    std::cout << "usage: argv[6]:  fn" << std::endl;
    std::cout << "usage: argv[7]:  fc" << std::endl;
    std::cout << "usage: argv[8]:  fh" << std::endl;
    std::cout << "usage: argv[9]:  fw" << std::endl;
    std::cout << "usage: argv[10]: sw" << std::endl;
    std::cout << "usage: argv[11]: sh" << std::endl;
    std::cout << "usage: argv[12]: pl" << std::endl;
    std::cout << "usage: argv[13]: pr" << std::endl;
    std::cout << "usage: argv[14]: pt" << std::endl;
    std::cout << "usage: argv[15]: pb" << std::endl;
    std::cout << "usage: argv[16]: inputNum" << std::endl;
    std::cout << "usage: argv[17]: pm" << std::endl;
    std::cout << "usage: argv[18]: dt" << std::endl;
    std::cout << "supported op: OT_Pooling" << std::endl;
    std::cout << "supported op: OT_Conv" << std::endl;
    std::cout << "supported op: OT_Eltwise" << std::endl;
    std::cout << "supported op: OT_Softmax" << std::endl;
    std::cout << "supported op: OT_Relu" << std::endl;
    std::cout << "supported op: OT_Relu6" << std::endl;
    std::cout << "supported op: OT_HSwish" << std::endl;
    std::cout << "supported op: OT_HSigmoid" << std::endl;
    std::cout << "supported op: OT_HGelu" << std::endl;
    std::cout << "supported op: OT_TanH" << std::endl;
    std::cout << "supported op: OT_FC" << std::endl;
    std::cout << "supported op: OT_Scale" << std::endl;
    std::cout << "supported op: OT_Concat" << std::endl;
    std::cout << "supported op: OT_Clip" << std::endl;
    std::cout << "supported op: OT_Squeeze" << std::endl;
    std::cout << "supported op: OT_Reshape" << std::endl;
    std::cout << "supported op: OT_Space2Depth" << std::endl;
    std::cout << "supported op: OT_Depth2Space" << std::endl;
}

template <typename T>
void buildInputTensor(DataType dt,
    DataFormat df,
    U32 n,
    U32 c,
    U32 h,
    U32 w,
    std::vector<TensorDesc> *dims,
    std::vector<Tensor> *inputTensors)
{
    TensorDesc inputDesc = tensor4df(dt, df, n, c, h, w);
    U32 inputNum = tensorNumElements(inputDesc);
    U32 inputSize = tensorNumBytes(inputDesc);
    U8 *inputVal = (U8 *)operator new(inputSize);

    T *data = (T *)inputVal;
    if (dt == DT_F16) {
        for (U32 i = 0; i < inputNum; i++) {
            data[i] = (T)(rand() & 255) / 256.0 - 0.5;
        }
        // for(U32 i = 0; i < inputNum; i++) data[i] = (T)(i & 255) / 255.0;
    }
    if (dt == DT_U8) {
        for (U32 i = 0; i < inputNum; i++) {
            data[i] = (T)(i & 255);
        }
    }
    std::shared_ptr<Tensor> inputTensor = std::shared_ptr<Tensor>(new Tensor());
    auto mem = (CpuMemory *)inputTensor->get_memory();
    mem->resize(inputDesc);
    mem->set_shared_ptr(std::shared_ptr<U8>(inputVal));

    dims->push_back(inputDesc);
    inputTensors->push_back(*inputTensor.get());
}

int main(int argc, char *argv[])
{
    if (argc != 16 && argc != 17 && argc != 18 && argc != 19) {
        printf("%d\n", argc);
        print_help();
        return 0;
    }

    U32 inputNum = 1;
    std::string pm = "NULL";
    std::string DT_NAME = "F16";
    std::string opName = argv[1];

    U32 in = atoi(argv[2]);
    U32 ic = atoi(argv[3]);
    U32 ih = atoi(argv[4]);
    U32 iw = atoi(argv[5]);

    U32 fn = atoi(argv[6]);
    U32 fc = atoi(argv[7]);
    U32 fh = atoi(argv[8]);
    U32 fw = atoi(argv[9]);

    U32 sw = atoi(argv[10]);
    U32 sh = atoi(argv[11]);
    U32 pl = atoi(argv[12]);
    U32 pr = atoi(argv[13]);
    U32 pt = atoi(argv[14]);
    U32 pb = atoi(argv[15]);
    if (argc == 17) {
        inputNum = atoi(argv[16]);
    }
    if (argc == 18) {
        pm = argv[17];
    }
    if (argc == 19) {
        DT_NAME = argv[18];
    }

    AffinityPolicy affinityPolicy = AFFINITY_GPU;
    DataType dt = DT_F16;
    auto model = new SequentialOcl(affinityPolicy, dt, opName);
    std::shared_ptr<SequentialOcl> model_ptr = std::shared_ptr<SequentialOcl>(model);

    OperatorType OType;
    if (opName == "OT_Pooling") {
        OType = OT_Pooling;
    }
    if (opName == "OT_Conv") {
        OType = OT_Conv;
    }
    if (opName == "OT_Eltwise") {
        OType = OT_Eltwise;
    }
    if (opName == "OT_Softmax") {
        OType = OT_Softmax;
    }
    if (opName == "OT_Relu") {
        OType = OT_Relu;
    }
    if (opName == "OT_Relu6") {
        OType = OT_Relu6;
    }
    if (opName == "OT_HSwish") {
        OType = OT_HSwish;
    }
    if (opName == "OT_HSigmoid") {
        OType = OT_HSigmoid;
    }
    if (opName == "OT_Gelu") {
        OType = OT_Gelu;
    }
    if (opName == "OT_TanH") {
        OType = OT_TanH;
    }
    if (opName == "OT_Sigmoid") {
        OType = OT_Sigmoid;
    }
    if (opName == "OT_FC") {
        OType = OT_FC;
    }
    if (opName == "OT_Scale") {
        OType = OT_Scale;
    }
    if (opName == "OT_Concat") {
        OType = OT_Concat;
    }
    if (opName == "OT_Clip") {
        OType = OT_Clip;
    }
    if (opName == "OT_Squeeze") {
        OType = OT_Squeeze;
    }
    if (opName == "OT_Reshape") {
        OType = OT_Reshape;
    }
    if (opName == "OT_Space2Depth") {
        OType = OT_Space2Depth;
    }
    if (opName == "OT_Depth2Space") {
        OType = OT_Depth2Space;
    }
    Factory *factory_ocl = (Factory *)(new FactoryOCL());
    std::shared_ptr<Factory> factory;
    factory = std::shared_ptr<Factory>(factory_ocl);
    ConvolutionMode convMode;
    //    convMode = Convolution_Depthwise_Pointwise;
    convMode = Convolution_Pointwise;

    switch (OType) {
        case OT_Pooling: {
            auto p = createPoolingParamSpec(
                PoolingMode::POOLING_MAX, fh, fw, sh, sw, pt, pb, pl, pr, RoundMode::CEIL);
            auto op = factory->createPooling(p);
            model_ptr->add(op);
            break;
        }
        case OT_Eltwise: {
            EltwiseParamSpec eltwiseDesc;
            eltwiseDesc.elt_mode = EltwiseMode::ELTWISE_SUM;
            eltwiseDesc.activation_type = ACTIVATION_NULL;
            auto op = factory->createEltwise(eltwiseDesc);
            model_ptr->add(op);
            break;
        }
        case OT_Softmax: {
            SoftmaxParamSpec p;
            p.axis = -1;
            auto op = factory->createSoftmax(dt, p);
            model_ptr->add(op);
            break;
        }
        case OT_Conv: {
            if (pm == "NULL") {
                ActivationParamSpec dwActivationParamSpec, pwActivationParamSpec;
                dwActivationParamSpec.mode = ACTIVATION_NULL;
                pwActivationParamSpec.mode = ACTIVATION_NULL;
                auto p = createConvolutionParamSpec(
                    1, fh, fw, sh, sw, pt, pb, pl, pr, 1, 1, fn, convMode);
                auto op =
                    factory->createConvolution(dt, p, dwActivationParamSpec, pwActivationParamSpec);
                model_ptr->add(op);
            }

            if (pm == "RELU") {
                ActivationParamSpec dwActivationParamSpec, pwActivationParamSpec;
                dwActivationParamSpec.mode = ACTIVATION_RELU;
                dwActivationParamSpec.value[0] = 0;
                pwActivationParamSpec.mode = ACTIVATION_NULL;
                auto p = createConvolutionParamSpec(
                    1, fh, fw, sh, sw, pt, pb, pl, pr, 1, 1, fn, convMode);
                auto op =
                    factory->createConvolution(dt, p, dwActivationParamSpec, pwActivationParamSpec);
                model_ptr->add(op);
            }
            break;
        }
        case OT_Relu: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_RELU;
            activationDesc.value[0] = 0;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_Relu6: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_RELU6;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_HSwish: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_H_SWISH;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_HSigmoid: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_H_SIGMOID;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_Gelu: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_GELU;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_TanH: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_TANH;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_Sigmoid: {
            ActivationParamSpec activationDesc;
            activationDesc.mode = ACTIVATION_SIGMOID;
            auto op = factory->createActivation(activationDesc);
            model_ptr->add(op);
            break;
        }
        case OT_FC: {
            auto p = createFullyConnectedParamSpec(fn, 1, nullptr);
            auto op = factory->createFullyConnected(dt, p, ih * iw * ic);
            model_ptr->add(op);
            break;
        }
        case OT_Scale: {
            ScaleParamSpec p;
            p.axis = 1;
            p.num_concat = inputNum;
            auto op = factory->createScale(dt, p, fc);
            model_ptr->add(op);
            break;
        }
        case OT_Concat: {
            ConcatParamSpec p;
            p.axis = 1;
            auto op = factory->createConcat(p);
            model_ptr->add(op);
            break;
        }
        case OT_Clip: {
            auto p = createClipParamSpec(0, 0.5);
            auto op = factory->createClip(dt, p);
            model_ptr->add(op);
            break;
        }
        case OT_Squeeze: {
            int dim[1] = {0};
            auto p = createSqueezeParamSpec(dim, 1);
            auto op = factory->createSqueeze(dt, p);
            model_ptr->add(op);
            break;
        }
        case OT_Reshape: {
            int dim[2] = {-1, 8};
            auto p = createReshapeParamSpec(dim, 2, 0, 0);
            auto op = factory->createReshape(dt, p);
            model_ptr->add(op);
            break;
        }
        case OT_Space2Depth: {
            auto op = factory->createSpace2Depth(dt);
            model_ptr->add(op);
            break;
        }
        case OT_Depth2Space: {
            Depth2SpaceParamSpec p;
            p.blockSize = 2;
            auto op = factory->createDepth2Space(dt, p);
            model_ptr->add(op);
            break;
        }
        default:
            std::cout << "not support op" << std::endl;
    }

    std::vector<TensorDesc> dims;
    std::vector<Tensor> inputTensors;
    if (OType == OT_Space2Depth) {
        for (U32 i = 0; i < inputNum; i++) {
            buildInputTensor<U8>(DT_U8, DF_NCHW, in, ic, ih, iw, &dims, &inputTensors);
        }
    } else {
        for (U32 i = 0; i < inputNum; i++) {
            buildInputTensor<F16>(DT_F16, DF_NCHW, in, ic, ih, iw, &dims, &inputTensors);
        }
    }

    U8 *weightVal = NULL;
    if (OType == OT_Conv) {
        TensorDesc weightDesc = tensor4df(dt, DF_NCHW, fn, fc, fh, fw);
        U32 weightNum = tensorNumElements(weightDesc);
        U32 vectorNum = fn;
        if (convMode == Convolution_Depthwise_Pointwise) {
            vectorNum = fc + fn + fn * fc;
        }
        U32 weightSize = tensorNumBytes(weightDesc) + vectorNum * bytesOf(dt);
        weightVal = (U8 *)operator new(weightSize);
        F16 *weight = (F16 *)weightVal;
        for (U32 i = 0; i < weightNum + vectorNum; i++) {
            weight[i] = (F16)(rand() & 255) / 256.0;
        }
    }

    if (OType == OT_FC) {
        U32 weightNum = iw * ih * ic * fn;
        U32 biasNum = fn;
        U32 weightSize = (weightNum + biasNum) * bytesOf(dt);
        weightVal = (U8 *)operator new(weightSize);
        F16 *weight = (F16 *)weightVal;
        for (U32 i = 0; i < weightNum + biasNum; i++) {
            weight[i] = (F16)(rand() & 255) / 256.0;
        }
    }

    if (OType == OT_Scale) {
        U32 weightNum = fc;
        U32 biasNum = fc;
        U32 weightSize = (weightNum + biasNum) * bytesOf(dt);
        weightVal = (U8 *)operator new(weightSize);
        F16 *weight = (F16 *)weightVal;
        for (U32 i = 0; i < weightNum + biasNum; i++) {
            weight[i] = (F16)(rand() & 255) / 256.0;
        }
    }

    if (weightVal) {
        std::shared_ptr<U8> modelPtr(weightVal);
        model_ptr->ready(dims, modelPtr, 1);
    } else {
        model_ptr->ready(dims, NULL, 1);
    }
    model_ptr->mark_input_output();
    model_ptr->set_input_tensors(inputTensors);
    model_ptr->run();

    auto output = model_ptr->get_output_tensors();
    auto oclMem = (OclMemory *)output[0]->get_memory();
    F16 *val = (F16 *)(oclMem->get_mapped_ptr());
    for (int i = 0; i < 64; i++) {
        std::cout << val[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
#endif
