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
#include <string>
#include <cstring>
#include "model_tools.h"
#include "model_serialize_deserialize.hpp"
#include "converter.h"

int main()
{
    ModelSpec fixedMs;
    
    std::string modelName = "leNet";
    DataType dt = DT_F32;
    str_copy(fixedMs.modelName, modelName.c_str(), NAME_LEN);
    fixedMs.dt = dt;
    std::cout << "group1 " << std::endl;

    int numInputs = 1;
    char* inputNames = "data";
    std::cout << "inputNames[0] " << inputNames[0] << std::endl;
    fixedMs.inputNames = {"data"};

    std::cout << "memcpy success " << std::endl;
    fixedMs.input_dims = (TensorDesc*)mt_new_storage(sizeof(TensorDesc) * numInputs);
    fixedMs.input_dims[0].dt = DT_F32;
    fixedMs.input_dims[0].df = DF_NCHW;
    fixedMs.input_dims[0].nDims = 4;
    fixedMs.input_dims[0].dims[0] = 64;
    fixedMs.input_dims[0].dims[1] = 1;
    fixedMs.input_dims[0].dims[2] = 28;
    fixedMs.input_dims[0].dims[3] = 28;
    fixedMs.numInputs = numInputs;
    std::cout << "group2 " << std::endl;

    int numOutputs = 1;
    fixedMs.output_names = {"prob"};
    fixedMs.numOutputs = numOutputs;
    std::cout << "group3 " << std::endl;

    int numOperatorSpecs = 7;
    OperatorSpec opsArr[7];

    // set each ops
    std::string opName0 = "conv1";
    OperatorType opType0 = OT_Conv;
    int opNumInputs0 = 1;
    char** opInputTensorsName0 = {"data"};
    int opNumOutputs0 = 1;
    char** opOutputTensorsName0 = {"conv1"};
    ConvolutionParamSpec convCps1;
    convCps1.num_kernels = 64;
    convCps1.kernel_size = 5;
    convCps1.stride = 1;
    convCps1.padding = 2;
    str_copy(opsArr[0].name, opName0.c_str(), opName0.length());
    opsArr[0].type = opType0;
    opsArr[0].numInputs = opNumInputs0;
    opsArr[0].input_tensors_name = opInputTensorsName0;
    opsArr[0].numOutputs = opNumOutputs0;
    opsArr[0].output_tensors_name = opOutputTensorsName0;
    opsArr[0].ps.conv_spec = convCps1;

    std::string opName1 = "pooling1";
    OperatorType opType1 = OT_Pooling;
    int opNumInputs1 = 1;
    char** opInputTensorsName1 = {"conv1"};
    int opNumOutputs1 = 1;
    char** opOutputTensorsName1 = {"pooling1"};
    PoolingParamSpec poolingPps1;
    poolingPps1.kernel_size = 2;
    poolingPps1.stride = 2;
    // poolingPps1.padding = 0;
    poolingPps1.mode = Max;
    str_copy(opsArr[1].name, opName1.c_str(), opName1.length());
    opsArr[1].type = opType1;
    opsArr[1].numInputs = opNumInputs1;
    opsArr[1].input_tensors_name = opInputTensorsName1;
    opsArr[1].numOutputs = opNumOutputs1;
    opsArr[1].output_tensors_name = opOutputTensorsName1;
    opsArr[1].ps.pooling_spec = poolingPps1;

    std::string opName2 = "conv2";
    OperatorType opType2 = OT_Conv;
    int opNumInputs2 = 1;
    char** opInputTensorsName2 = {"pooling1"};
    int opNumOutputs2 = 1;
    char** opOutputTensorsName2 = {"conv2"};
    ConvolutionParamSpec convCps2;
    convCps2.num_kernels = 32;
    convCps2.kernel_size = 5;
    convCps2.stride = 1;
    convCps2.padding = 2;
    str_copy(opsArr[2].name, opName2.c_str(), opName2.length());
    opsArr[2].type = opType2;
    opsArr[2].numInputs = opNumInputs2;
    opsArr[2].input_tensors_name = opInputTensorsName2;
    opsArr[2].numOutputs = opNumOutputs2;
    opsArr[2].output_tensors_name = opOutputTensorsName2;
    opsArr[2].ps.conv_spec = convCps2;

    std::string opName3 = "pooling2";
    OperatorType opType3 = OT_Pooling;
    int opNumInputs3 = 1;
    char** opInputTensorsName3 = {"conv2"};
    int opNumOutputs3 = 1;
    char **op_output_tensors_name_3 = {"pooling2"};
    PoolingParamSpec poolingPps2;
    poolingPps2.kernel_size = 2;
    poolingPps2.stride = 2;
    // poolingPps2.padding = 0;    // pooling no padding?
    poolingPps2.mode = Max;
    str_copy(opsArr[3].name, opName3.c_str(), opName3.length());
    opsArr[3].type = opType3;
    opsArr[3].numInputs = opNumInputs3;
    opsArr[3].input_tensors_name = opInputTensorsName3;
    opsArr[3].numOutputs = opNumOutputs3;
    opsArr[3].output_tensors_name = op_output_tensors_name_3;
    opsArr[3].ps.pooling_spec = poolingPps2;

    std::string opName4 = "fc1";
    OperatorType opType4 = OT_FC;
    int opNumInputs4 = 1;
    char** opInputTensorsName4 = {"pooling2"};
    int opNumOutputs4 = 1;
    char** opOutputTensorsName4 = {"fc1"};
    FullyConnectedParamSpec fcps1;
    fcps1.numOutputs = 100;
    str_copy(opsArr[4].name, opName4.c_str(), opName4.length());
    opsArr[4].type = opType4;
    opsArr[4].numInputs = opNumInputs4;
    opsArr[4].input_tensors_name = opInputTensorsName4;
    opsArr[4].numOutputs = opNumOutputs4;
    opsArr[4].output_tensors_name = opOutputTensorsName4;
    opsArr[4].ps.fc_spec = fcps1;

    std::string opName5 = "fc2";
    OperatorType opType5 = OT_FC;
    int opNumInputs5 = 1;
    char** opInputTensorsName5 = {"fc1"};
    int opNumOutputs5 = 1;
    char** opOutputTensorsName5 = {"fc2"};
    FullyConnectedParamSpec fcps2;
    fcps2.numOutputs = 50;
    str_copy(opsArr[5].name, opName5.c_str(), opName5.length());
    opsArr[5].type = opType5;
    opsArr[5].numInputs = opNumInputs5;
    opsArr[5].input_tensors_name = opInputTensorsName5;
    opsArr[5].numOutputs = opNumOutputs5;
    opsArr[5].output_tensors_name = opOutputTensorsName5;
    opsArr[5].ps.fc_spec = fcps2;

    std::string opName6 = "prob";
    OperatorType opType6 = OT_Softmax;
    int opNumInputs6 = 1;
    char** opInputTensorsName6 = {"fc2"};
    int opNumOutputs6 = 1;
    char** opOutputTensorsName6 = {"prob"};
    str_copy(opsArr[6].name, opName6.c_str(), opName6.length());
    opsArr[6].type = opType6;
    opsArr[6].numInputs = opNumInputs6;
    opsArr[6].input_tensors_name = opInputTensorsName6;
    opsArr[6].numOutputs = opNumOutputs6;
    opsArr[6].output_tensors_name = opOutputTensorsName6;

    fixedMs.numOperatorSpecs = numOperatorSpecs;
    fixedMs.ops = &opsArr[0];
    std::cout << "group4 " << std::endl;

    // weight op 信息
    I32 numWeightSpecs = 4;
    WeightSpec wsArr[4];
    F32 floatValue32 = 1.0;

    // set each ws
    std::string weigthOpNameConv1 = "conv1";
    DataType mdtConv1 = DT_F32;
    U32 bytesOfWeightConv1 = 1*5*5*20*bytesOf(mdtConv1);
    F32* conv1WeightPtr = (F32*)mt_new_storage(bytesOfWeightConv1);
    for (int i = 0; i < 1*5*5*20; i++) {
        conv1WeightPtr[i] = 1.0;
    }
    U32 bytesOfVecConv1 = 20*bytesOf(mdtConv1);
    F32* convVecPtr1 = (F32*)mt_new_storage(bytesOfVecConv1);
    for (int i = 0; i < 20; i++) {
        convVecPtr1[i] = 1.0;
    }
    str_copywsArr[0].op_name, weigthOpNameConv1.c_str(), weigthOpNameConv1.length());
    wsArr[0].mdt = mdtConv1;
    wsArr[0].bytes_of_weight = bytesOfWeightConv1;
    wsArr[0].weight = (U8*)conv1WeightPtr;
    wsArr[0].bytes_of_vec = bytesOfVecConv1;
    wsArr[0].vec = (U8*)convVecPtr1;

    std::string weightOpNameConv2 = "conv2";
    DataType mdtConv2 = DT_F32;
    U32 bytesOfWeightConv2 = 64*5*5*32*bytesOf(mdtConv2);
    F32* conv2WeightPtr = (F32*)mt_new_storage(bytesOfWeightConv2);
    for (int i = 0; i < 64*5*5*32; i++) {
        conv2WeightPtr[i] = 1.0;
    }
    U32 bytesOfVecConv2 = 32 * bytesOf(mdtConv2);
    F32* convVecPtr2 = (F32*)mt_new_storage(bytesOfVecConv2);
    for (int i=0; i<32; i++) {
        convVecPtr2[i] = 1.0;
    }
    str_copy(wsArr[1].op_name, weightOpNameConv2.c_str(), weightOpNameConv2.length());
    wsArr[1].mdt = mdtConv2;
    wsArr[1].bytes_of_weight = bytesOfWeightConv2;
    wsArr[1].weight = (U8*)conv2WeightPtr;
    wsArr[1].bytes_of_vec = bytesOfVecConv2;
    wsArr[1].vec = (U8*)convVecPtr2;

    std::string weightOpNameFc1 = "fc1";
    DataType mdtFc1 = DT_F32;
    U32 bytesOfWeightFc1 = 32*4*4*bytesOf(mdtFc1);
    F32* fcWeightPtr1 = (F32*)mt_new_storage(bytesOfWeightFc1);
    for (int i=0; i<32*4*4; i++) {
        fcWeightPtr1[i] = 1.0;
    }
    U32 bytesOfVecFc1 = 100*bytesOf(mdtFc1);
    F32* fcVecPtr1 = (F32*)mt_new_storage(bytesOfVecFc1);
    for (int i = 0; i < 100; i++) {
        fcVecPtr1[i] = 1.0;
    }
    str_copy(wsArr[2].op_name, weightOpNameFc1.c_str(), weightOpNameFc1.length());
    wsArr[2].mdt = mdtFc1;
    wsArr[2].bytes_of_weight = bytesOfWeightFc1;
    wsArr[2].weight = (U8*)fcWeightPtr1;
    wsArr[2].bytes_of_vec = bytesOfVecFc1;
    wsArr[2].vec = (U8*)fcVecPtr1;

    std::string weightOpNameFc2 = "fc2";
    DataType mdtFc2 = DT_F32;
    U32 bytesOfWeightFc2 = 100*10*bytesOf(mdtFc2);
    F32* fcWeightPtr2 = (F32*)mt_new_storage(bytesOfWeightFc2);
    for (int i = 0; i < 100*10; i++) {
        fcWeightPtr2[i] = 1.0;
    }
    U32 bytesOfVecFc2 = 10 * bytesOf(mdtFc2);
    F32* fcVecPtr2 = (F32*)mt_new_storage(bytesOfVecFc2);
    for (int i = 0; i < 10; i++) {
        fcVecPtr2[i] = 1.0;
    }
    str_copy(wsArr[3].op_name, weightOpNameFc2.c_str(), weightOpNameFc2.length());
    wsArr[3].mdt = mdtFc2;
    wsArr[3].bytes_of_weight = bytesOfWeightFc2;
    wsArr[3].weight = (U8*)fcWeightPtr2;
    wsArr[3].bytes_of_vec = bytesOfVecFc2;
    wsArr[3].vec = (U8*)fcVecPtr2;
    fixedMs.numWeightSpecs = numWeightSpecs;
    fixedMs.ws = &wsArr[0];

    int number = fixedMs.numWeightSpecs;
    for (int i=0; i < number; i++) {
        std::cout << "op name is : " << fixedMs.ws[i].op_name << std::endl;
        std::cout << "op mdt is: " << fixedMs.ws[i].mdt << std::endl;
        std::cout << "op bytes_of_weight is: " << fixedMs.ws[i].bytes_of_weight << std::endl;
        std::cout << "op bytes_of_vec is: " << fixedMs.ws[i].bytes_of_vec << std::endl;
        std::cout << "op weight address: " << (void*)fixedMs.ws[i].weight << std::endl;
        std::cout << "op bias address: " << (void*)fixedMs.ws[i].vec << std::endl;
        std::cout << "first weight value: " << ((F32*)fixedMs.ws[i].weight)[0] << std::endl;
        if(fixedMs.ws[i].bytes_of_vec > 0) {
            std::cout << "first bias value: " << ((F32*)fixedMs.ws[i].vec)[0] << std::endl;
        }
        std::cout << "\n\n\n";
    }

    std::string storePath = "./fixedMs.bolt";
    serialize_model_to_file(&fixedMs, storePath.c_str());
    
    return 0;
}
