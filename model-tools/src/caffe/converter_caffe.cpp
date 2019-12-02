// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"

#include "type.h"
#include "converter.h"
#include "model_serialize_deserialize.hpp"
#include "model_tools.h"


#ifdef _USE_CAFFE_MODEL

// read prototxt
EE read_from_prototxt(const char* path, google::protobuf::Message* message) {
    std::ifstream fs(path, std::ifstream::in);
    if (!fs.is_open()) {
        return NOT_FOUND;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool ret = google::protobuf::TextFormat::Parse(&input, message);
    fs.close();
    return (ret) ? SUCCESS : NOT_SUPPORTED;
}


// read caffemodel(bin)
EE read_from_caffemodel(const char* path, google::protobuf::Message* message) {
    std::ifstream fs(path, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open()) {
        return NOT_FOUND;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool ret = message -> ParseFromCodedStream(&codedstr);
    fs.close();

    return (ret) ? SUCCESS : NOT_SUPPORTED;
}


OperatorType convert_type(std::string inputType) {
    if (inputType == "Convolution") {
        return OT_Conv;
    }else if (inputType == "BatchNorm") {
        return OT_BatchNorm;
    }else if (inputType == "Scale") {
        return OT_Scale;
    }else if (inputType == "Eltwise") {
        return OT_Eltwise;
    }else if (inputType == "InnerProduct") {
        return OT_FC;
    }else if (inputType == "Pooling") {
        return OT_Pooling;
    }else if (inputType == "ReLU") {
        return OT_Relu;
    }else if (inputType == "ReLU6") {
        return OT_Relu6;
    }else if (inputType == "HSwish") {
        return OT_HSwish;
    }else if (inputType == "HSigmoid") {
        return OT_HSigmoid;
    }else if (inputType == "Softmax") {
        return OT_Softmax;
    }else if (inputType == "Concat") {
        return OT_Concat;
    }else if (inputType == "Embed") {  
        return OT_Embedding;
    }else if (inputType == "Gelu") {
        return OT_Gelu;
    }else if (inputType == "LayerNorm") {
        return OT_LayerNorm;
    }else if (inputType == "MatMul") {
        return OT_MatMul;
    }else if (inputType == "Multiply") {
        return OT_Multiply;
    }else if (inputType == "Reshape") {
        return OT_Reshape;
    }else if (inputType == "Slice") {
        return OT_Slice;
    }else if (inputType == "Transpose") {
        return OT_Transpose;
    }else if (inputType == "Attention") {
        return OT_Attention;
    }else if (inputType == "Input") {
        return OT_Input;
    }else if (inputType == "LSTM") {
        return OT_LSTM;
    }else if (inputType == "TanH") {
        return OT_TanH;
    }else if (inputType == "SoftmaxWithLoss") {
        return OT_SoftmaxWithLoss;
    }
    else {
        std::cerr << "[ERROR] encounter unsupported operator " << inputType << std::endl;
        return OT_None;
    }
}


inline int net_search_layerId(caffe::NetParameter& netParams, std::string& layerName) {
    int i = 0;
    if (netParams.layer_size() > 0) {
        for (i = 0; i < netParams.layer_size(); i++) {
            if (netParams.layer(i).name() == layerName) {
                return i;
            }
        }
    }
    else{
        for (i = 0; i < netParams.layers_size(); i++) {
            if (netParams.layers(i).name() == layerName) {
                return i;
            }
        }
    }
    return -1;
}


inline caffe::BlobProto net_get_blob(caffe::NetParameter& netParams, int layerId, int blobId) {
    if (netParams.layer_size() > 0) {
        return netParams.layer(layerId).blobs(blobId);
    }
    else{
        return netParams.layers(layerId).blobs(blobId);
    }
}


inline int net_get_blobs_size(caffe::NetParameter& netParams, int layerId) {
    if (netParams.layer_size() > 0) {
        return netParams.layer(layerId).blobs_size();
    }
    else{
        return netParams.layers(layerId).blobs_size();
    }
}


inline void net_copy_blob(WeightSpec* wsPtr, int weightIndex, caffe::NetParameter& netParams, int netLayerId, int blobNum, int mode) {
    wsPtr[weightIndex].mdt = DT_F32;
    wsPtr[weightIndex].bytes_of_weight = 0;
    wsPtr[weightIndex].weight = nullptr;
    wsPtr[weightIndex].bytes_of_vec = 0;
    wsPtr[weightIndex].vec = nullptr;
    
    if (mode == 0) {
        if (blobNum >= 1) {
            caffe::BlobProto blob0 = net_get_blob(netParams, netLayerId, 0);
            U32 elemSize = sizeof(*(blob0.data().data()));
            CHECK_REQUIREMENT(elemSize == bytesOf(wsPtr[weightIndex].mdt));
            wsPtr[weightIndex].bytes_of_weight = elemSize * blob0.data_size();
            wsPtr[weightIndex].weight = (U8*)mt_malloc(wsPtr[weightIndex].bytes_of_weight);
            memcpy(wsPtr[weightIndex].weight, (U8*)blob0.data().data(), wsPtr[weightIndex].bytes_of_weight);
        }
        if (blobNum >= 2) {
            caffe::BlobProto blob1 = net_get_blob(netParams, netLayerId, 1);
            CHECK_REQUIREMENT(sizeof(*(blob1.data().data())) == bytesOf(wsPtr[weightIndex].mdt));
            wsPtr[weightIndex].bytes_of_vec = sizeof(*(blob1.data().data())) * blob1.data_size();
            wsPtr[weightIndex].vec = (U8*)mt_malloc(wsPtr[weightIndex].bytes_of_vec);
            memcpy(wsPtr[weightIndex].vec, (U8*)blob1.data().data(), wsPtr[weightIndex].bytes_of_vec);
        }
    } else if (mode == 1) { // lstm
        if (blobNum == 2) {
            caffe::BlobProto blob0 = net_get_blob(netParams, netLayerId, 0);
            CHECK_REQUIREMENT(sizeof(*(blob0.data().data())) == bytesOf(wsPtr[weightIndex].mdt));
            
            caffe::BlobProto blob1 = net_get_blob(netParams, netLayerId, 1);
            CHECK_REQUIREMENT(sizeof(*(blob1.data().data())) == bytesOf(wsPtr[weightIndex].mdt));

            U32 lstmWeightNum0 = blob0.data_size();
            U32 lstmWeightNum1 = blob1.data_size();
     
            wsPtr[weightIndex].bytes_of_weight = sizeof(*(blob0.data().data())) * lstmWeightNum0 + sizeof(*(blob1.data().data())) * lstmWeightNum1;

            wsPtr[weightIndex].weight = (U8*)mt_malloc(wsPtr[weightIndex].bytes_of_weight);
            memcpy(wsPtr[weightIndex].weight, (U8*)blob0.data().data(), sizeof(*(blob0.data().data())) * lstmWeightNum0);
            memcpy(wsPtr[weightIndex].weight + sizeof(*(blob0.data().data())) * lstmWeightNum0 , (U8*)blob1.data().data(), sizeof(*(blob1.data().data())) * lstmWeightNum1);
        } else if (blobNum == 3) {
            caffe::BlobProto blob0 = net_get_blob(netParams, netLayerId, 0);
            CHECK_REQUIREMENT(sizeof(*(blob0.data().data())) == bytesOf(wsPtr[weightIndex].mdt));
            
            caffe::BlobProto blob2 = net_get_blob(netParams, netLayerId, 2);
            CHECK_REQUIREMENT(sizeof(*(blob2.data().data())) == bytesOf(wsPtr[weightIndex].mdt));

            U32 lstmWeightNum0 = blob0.data_size();
            U32 lstmWeightNum1 = blob2.data_size();
     
            wsPtr[weightIndex].bytes_of_weight = sizeof(*(blob0.data().data())) * lstmWeightNum0 + sizeof(*(blob2.data().data())) * lstmWeightNum1;

            wsPtr[weightIndex].weight = (U8*)mt_malloc(wsPtr[weightIndex].bytes_of_weight);
            memcpy(wsPtr[weightIndex].weight, (U8*)blob0.data().data(), sizeof(*(blob0.data().data())) * lstmWeightNum0);
            memcpy(wsPtr[weightIndex].weight + sizeof(*(blob0.data().data())) * lstmWeightNum0 , (U8*)blob2.data().data(), sizeof(*(blob2.data().data())) * lstmWeightNum1);


            caffe::BlobProto blob1 = net_get_blob(netParams, netLayerId, 1);
            CHECK_REQUIREMENT(sizeof(*(blob1.data().data())) == bytesOf(wsPtr[weightIndex].mdt));

            U32 lstmBiasNum = blob1.data_size();

            wsPtr[weightIndex].bytes_of_vec = sizeof(*(blob1.data().data())) * lstmBiasNum;

            wsPtr[weightIndex].vec = (U8*)mt_malloc(wsPtr[weightIndex].bytes_of_vec);
            memcpy(wsPtr[weightIndex].vec, (U8*)blob1.data().data(), wsPtr[weightIndex].bytes_of_vec);

        }
    }
}

EE caffe_model_info_check(CI8* dir, CI8* mfn) {
    std::string prototxtSuffix = ".prototxt";
    std::string caffeModelSuffix = ".caffemodel";
    std::string prototxtPath = dir + std::string(mfn) + prototxtSuffix;
    std::string caffeModelPath = dir + std::string(mfn) + caffeModelSuffix;
    std::map<std::string, int> layerTypeNum;

    caffe::NetParameter proto;
    caffe::NetParameter net;

    // load prototxt
    CHECK_STATUS_WITH_RETURN(read_from_prototxt(prototxtPath.c_str(), &proto));
    CHECK_REQUIREMENT(proto.layer_size() > 0);

    // load model bin
    CHECK_STATUS_WITH_RETURN(read_from_caffemodel(caffeModelPath.c_str(), &net));

    for (int i = 0; i < proto.layer_size(); i++) {
        const caffe::LayerParameter layer = proto.layer(i);
        std::string thisLayerType = layer.type();
        std::string thisLayerName = layer.name();
        std::cout << "[INFO] " << thisLayerName << " / " << thisLayerType << std::endl;
        if (layerTypeNum.find(thisLayerType) == layerTypeNum.end()) {
            layerTypeNum.insert(std::pair<std::string, int>(thisLayerType, 1));
        } else {
            layerTypeNum[thisLayerType] = layerTypeNum[thisLayerType] + 1;
        }

        // only transpose and multiply op need to newly extract
        if (thisLayerType == "Eltwise") {
            auto caffeEltwiseParam = layer.eltwise_param();
            auto op = caffeEltwiseParam.operation();
            switch (op)
            {
                case caffe::EltwiseParameter_EltwiseOp_PROD:
                    std::cout << "    eltwise mode: ELTWISE_PROD " << std::endl;
                    break;
                case caffe::EltwiseParameter_EltwiseOp_SUM:
                    std::cout << "    eltwise mode: ELTWISE_SUM " << std::endl;
                    break;
                case caffe::EltwiseParameter_EltwiseOp_MAX:
                    std::cout << "    eltwise mode: ELTWISE_MAX " << std::endl;
                    break;
                default:
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
            }
        } else if (thisLayerType == "Embed") {
            auto caffeEmbedParam = layer.embed_param();
            auto thisNumOutput = caffeEmbedParam.num_output();
            auto thisNumDim = caffeEmbedParam.input_dim();
            auto thisBiasTerm = caffeEmbedParam.bias_term();
            std::cout << "    num_output: " << thisNumOutput << std::endl;
            std::cout << "    input_dim: " << thisNumDim << std::endl;
            std::cout << "    bias_term: " << thisBiasTerm << std::endl;
        } else if (thisLayerType == "InnerProduct") {
            auto caffeInnerProductParam = layer.inner_product_param();
            auto thisNumOutput = caffeInnerProductParam.num_output();
            auto thisBiasTerm = caffeInnerProductParam.bias_term();
            std::cout << "    num_output: " << thisNumOutput << std::endl;
            std::cout << "    bias_term: " << thisBiasTerm << std::endl;
        } else if (thisLayerType == "Input") {
            auto caffeInputParam = layer.input_param();
            auto caffeInputOpShapeSize = caffeInputParam.shape_size();
            auto shapeDimsNum = caffeInputParam.shape(0).dim_size();
            std::cout << "    shape_size: " << caffeInputOpShapeSize << std::endl;
            std::cout << "    dim_size: " << shapeDimsNum << std::endl;
            for (int iter = 0; iter < shapeDimsNum; iter++) {
                std::cout << "    dim[0]: " << caffeInputParam.shape(0).dim(iter) << std::endl;
            }
        } else if (thisLayerType == "Multiply") {
            auto caffeMultiplyParam = layer.multiply_param();
            auto thisScale = caffeMultiplyParam.scale();
            std::cout << "    scale: " << thisScale << std::endl;
        } else if (thisLayerType == "Reshape") {
            auto caffeReshapeParam = layer.reshape_param();
            auto shapeDimsNum = caffeReshapeParam.shape().dim_size();
            std::cout << "    dim_size: " << shapeDimsNum << std::endl;
            for (int iter = 0; iter < shapeDimsNum; iter++) {
                std::cout << "    dim[0]: " << caffeReshapeParam.shape().dim(iter) << std::endl;
            }
            auto thisReshapeParamAxis = caffeReshapeParam.axis();
            std::cout << "    axis: " << thisReshapeParamAxis << std::endl;
            auto thisReshapeParamNumAxes = caffeReshapeParam.num_axes();
            std::cout << "    num_axes: " << thisReshapeParamNumAxes << std::endl;
        } else if (thisLayerType == "Slice") {
            auto caffeSliceParam = layer.slice_param();
            auto thisSliceParamSlicePoint = caffeSliceParam.slice_point(0);    // repeated
            auto thisSliceParamAxis = caffeSliceParam.axis();
            std::cout << "    slice_point[0]: " << thisSliceParamSlicePoint << std::endl;
            std::cout << "    axis: " << thisSliceParamAxis << std::endl;
        } else if (thisLayerType == "Transpose") {
            auto caffeTransposeParam = layer.transpose_param();
            auto dimDimsNum = caffeTransposeParam.dim().dim_size();
            std::cout << "    dim_size: " << dimDimsNum << std::endl;
            for (int iter = 0; iter < dimDimsNum; iter++) {
                std::cout << "    dim[0]: " << caffeTransposeParam.dim().dim(iter) << std::endl;
            }
        }
    }

    std::cout << "[INFO] summary:" << std::endl;
    for (const auto& it : layerTypeNum) {
        std::cout << "    " << it.first << " / " << it.second << std::endl;
    }

    return SUCCESS;
}


// Load caffe model 
//    dir: file_prototxt
//    mfn: file_caffemodel
//    ms:  the target object
EE mt_load_caffe(CI8* dir, CI8* mfn, ModelSpec* ms) {
    std::string prototxtSuffix = ".prototxt";
    std::string caffeModelSuffix = ".caffemodel";
    std::string prototxtPath = dir + std::string(mfn) + prototxtSuffix;
    std::string caffeModelPath = dir + std::string(mfn) + caffeModelSuffix;

    caffe::NetParameter proto;
    caffe::NetParameter net;

    // load prototxt
    CHECK_STATUS_WITH_RETURN(read_from_prototxt(prototxtPath.c_str(), &proto));
    CHECK_REQUIREMENT(proto.layer_size() > 0);

    // load model bin.
    CHECK_STATUS_WITH_RETURN(read_from_caffemodel(caffeModelPath.c_str(), &net));

    str_copy(ms->model_name, proto.name().c_str(), proto.name().length());
    ms->dt = DT_F32; // Do not forget to set default value
    
    ms->num_operator_specs = proto.layer_size();
    OperatorSpec* opsPtr = (OperatorSpec*)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
    ms->ops = opsPtr;

    int inputsNumber = 0;
    int weightNumber = 0;
    std::map<std::string, int> outputCounts;
    for (int i = 0; i < proto.input_size(); i++) {
        outputCounts[proto.input(i).c_str()] = 1;
    }
    for (int i = 0; i < proto.layer_size(); i++) {
        const caffe::LayerParameter layer = proto.layer(i);

        if (layer.type() == "Input") {
            inputsNumber++;
        }

        str_copy(opsPtr[i].name, layer.name().c_str(), layer.name().length());
        
        opsPtr[i].type = convert_type(layer.type());
        int bottomSize = layer.bottom_size();
        opsPtr[i].num_inputs = bottomSize;
        opsPtr[i].input_tensors_name = (I8 **)mt_malloc(bottomSize * sizeof(I8 *));
        for (int j = 0; j < bottomSize; j++) {
            opsPtr[i].input_tensors_name[j] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(opsPtr[i].input_tensors_name[j], layer.bottom(j).c_str(), layer.bottom(j).length());
            if (outputCounts.find(layer.bottom(j)) == outputCounts.end()) {
                std::cerr << "[ERROR] encounter no output as this operator's input" << std::endl;
                CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
            } else {
                outputCounts[layer.bottom(j)]--;
            }
        }
        int topSize = layer.top_size();
        opsPtr[i].num_outputs = topSize;
        opsPtr[i].output_tensors_name = (I8 **)mt_malloc(topSize * sizeof(I8 *));
        for (int j = 0; j < topSize; j++) {
            opsPtr[i].output_tensors_name[j] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(opsPtr[i].output_tensors_name[j], layer.top(j).c_str(), layer.top(j).length());
            if (outputCounts.find(layer.top(j)) == outputCounts.end()) {
                outputCounts[layer.top(j)] = 1;
            } else {
                outputCounts[layer.top(j)] ++;
            }
        }

        if (layer.type() == "Convolution") {
            weightNumber++;
            ConvolutionParamSpec cps;
            cps.num_kernels = layer.convolution_param().num_output();
            if (layer.convolution_param().has_kernel_w() && layer.convolution_param().has_kernel_h()) {
                int curKernelW = layer.convolution_param().kernel_w();
                int curKernelH = layer.convolution_param().kernel_h();
                if (curKernelW != curKernelH) {
                    std::cerr << "[ERROR] Convolution kernel_h != kernel_w" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                cps.kernel_size = curKernelW;
            } else {
                cps.kernel_size = layer.convolution_param().kernel_size(0);
            }
            
            cps.group = (layer.convolution_param().has_group()) ? layer.convolution_param().group() : 1;  // group[default=1]
            cps.dilation = (layer.convolution_param().dilation_size() != 0) ? layer.convolution_param().dilation(0) : 1;
            if (cps.group != cps.num_kernels) {
                std::cout << "[INFO] Convolution group != num_kernels" << std::endl;
                cps.group = 1;
            }
            else
                std::cout << "[INFO] Depthwise Convolution" << std::endl;
            if (cps.group == 1) {
                if (cps.dilation > 1) {
                    cps.convolution_type = Convolution_Dilation;
                }
                else{
                    cps.convolution_type = Convolution_Pointwise;
                }
            }
            else{
                cps.convolution_type = Convolution_Depthwise;
            }
            cps.dw_activation_type = ACTIVATION_NULL;
            cps.pw_activation_type = ACTIVATION_NULL;
            if (layer.convolution_param().has_stride_w() && layer.convolution_param().has_stride_h()) {
                int curStrideW = layer.convolution_param().stride_w();
                int curStrideH = layer.convolution_param().stride_h();
                if (curStrideW != curStrideH) {
                    std::cerr << "[ERROR] Convolution stride_h != stride_w" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                cps.stride = curStrideW;
            } else {
                cps.stride = (layer.convolution_param().stride_size() != 0) ? layer.convolution_param().stride(0) : 1; // stride[default=1]
            }
            if (layer.convolution_param().has_pad_w() && layer.convolution_param().has_pad_h()) {
                int curPadW = layer.convolution_param().pad_w();
                int curPadH = layer.convolution_param().pad_h();
                if (curPadW != curPadH) {
                    std::cerr << "[ERROR] Convolution padding_h != padding_w" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                cps.padding = curPadW;
            } else {
                cps.padding = (layer.convolution_param().pad_size() != 0) ? layer.convolution_param().pad(0) : 0;  // pad[default=0]
            }
            opsPtr[i].ps.conv_param_spec = cps;
        }else if (layer.type() == "Pooling") {
            PoolingParamSpec pps;
            if (layer.pooling_param().has_kernel_w() && layer.pooling_param().has_kernel_h()) {
                int curKernelW = layer.pooling_param().kernel_w();
                int curKernelH = layer.pooling_param().kernel_h();
                if (curKernelW != curKernelH) {
                    std::cerr << "[ERROR] Pooling kernel_h != kernel_w" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                pps.kernel_size = curKernelW;
            } else {
                pps.kernel_size = layer.pooling_param().kernel_size();
            }
            if (layer.pooling_param().has_stride_w() && layer.pooling_param().has_stride_h()) {
                int curStrideW = layer.pooling_param().stride_w();
                int curStrideH = layer.pooling_param().stride_h();
                if (curStrideW != curStrideH) {
                    std::cerr << "[ERROR] Pooling stride_h != stride_w" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                pps.stride = curStrideW;
            } else {
                pps.stride = layer.pooling_param().stride();
            }
            bool global_pooling = layer.pooling_param().global_pooling();
            if (global_pooling) {
                pps.kernel_size = 0;
                pps.stride = 1;
            }else {
                CHECK_REQUIREMENT(pps.kernel_size > 0);
            }
            if (layer.pooling_param().has_pad_w() && layer.pooling_param().has_pad_h()) {
                int curPadW = layer.pooling_param().pad_w();
                int curPadH = layer.pooling_param().pad_h();
                if (curPadW != curPadH) {
                    std::cerr << "[ERROR] Pooling padding_h != padding_w" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                }
                pps.padding = curPadW;
            } else {
                pps.padding = layer.pooling_param().has_pad() ? layer.pooling_param().pad() : 0;
            }
            
            if (layer.pooling_param().has_round_mode() && layer.pooling_param().round_mode() == 1) {
                pps.rm = FLOOR;
            }else{
                pps.rm = CEIL;
            }
            switch (layer.pooling_param().pool()) {
                case caffe::PoolingParameter_PoolMethod_MAX: {
                    pps.mode = Max;
                    break;
                }
                case caffe::PoolingParameter_PoolMethod_AVE: {
                    pps.mode = Mean;
                    break;
                }
                default: {
                    std::cerr << "[ERROR] encounter unsupported Pooling method " << layer.pooling_param().pool() << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                    break;
                }
            }
            opsPtr[i].ps.pooling_param_spec = pps;
        }else if (layer.type() == "Eltwise") {
            auto caffeEltwiseParam = layer.eltwise_param();
            auto op = caffeEltwiseParam.operation();
            switch (op)
            {
                case caffe::EltwiseParameter_EltwiseOp_PROD:
                    opsPtr[i].ps.eltwise_param_spec.elt_mode = ELTWISE_PROD;
                    break;
                case caffe::EltwiseParameter_EltwiseOp_SUM:
                    opsPtr[i].ps.eltwise_param_spec.elt_mode = ELTWISE_SUM;
                    break;
                case caffe::EltwiseParameter_EltwiseOp_MAX:
                    opsPtr[i].ps.eltwise_param_spec.elt_mode = ELTWISE_MAX;
                    break;
                default: {
                    std::cerr << "[ERROR] unknown eltwise mode" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                    break;
                }
            }
            U32 bytes = caffeEltwiseParam.coeff_size() * sizeof(F32);
            opsPtr[i].ps.eltwise_param_spec.elt_sum_spec.coeff_size = caffeEltwiseParam.coeff_size();
            opsPtr[i].ps.eltwise_param_spec.elt_sum_spec.coeff_values = (F32 *)mt_malloc(bytes);
            memcpy(opsPtr[i].ps.eltwise_param_spec.elt_sum_spec.coeff_values, caffeEltwiseParam.coeff().data(),bytes);
            for (int j = 0; j < caffeEltwiseParam.coeff_size(); j++) {
                CHECK_REQUIREMENT(opsPtr[i].ps.eltwise_param_spec.elt_sum_spec.coeff_values[j] == 1);
            }
        }else if (layer.type() == "InnerProduct") {
            weightNumber++;
            FullyConnectedParamSpec ips;
            ips.num_outputs = layer.inner_product_param().num_output();
            opsPtr[i].ps.ip_param_spec = ips;
        }else if (layer.type() == "BatchNorm") {
            weightNumber++;
            BatchNormParamSpec bnps;
            bnps.eps = layer.batch_norm_param().eps();
            opsPtr[i].ps.bn_param_spec = bnps;
        }else if (layer.type() == "Scale" || layer.type() == "LayerNorm") {
            weightNumber++;
        }else if (layer.type() == "Embed") {
            weightNumber++;
            EmbedParamSpec embedPs;
            auto caffeEmbedParam = layer.embed_param();
            embedPs.input_dim  = caffeEmbedParam.input_dim();
            embedPs.num_output = caffeEmbedParam.num_output();
            embedPs.bias_term  = caffeEmbedParam.bias_term() == 0 ? false: true;
            opsPtr[i].ps.embed_spec = embedPs;
        }else if (layer.type() == "LSTM") {
            weightNumber++;
            LstmParamSpec lstm_ps;
            auto caffeLstmParam = layer.recurrent_param();
            lstm_ps.num_output = caffeLstmParam.num_output();
            opsPtr[i].ps.lstm_spec = lstm_ps;
        }else if (layer.type() == "Multiply") {
            MultiplyParamSpec multiplyPs;
            auto caffeMultiplyParam = layer.multiply_param();
            float thisScale = caffeMultiplyParam.scale();
            multiplyPs.scale = thisScale;
            opsPtr[i].ps.multiply_spec = multiplyPs;
        }else if (layer.type() == "Reshape") {
            ReshapeParamSpec reshapePs;
            auto caffeReshapeParam = layer.reshape_param();
            reshapePs.shape_size = caffeReshapeParam.shape().dim_size();
            for (I32 iter = 0; iter < caffeReshapeParam.shape().dim_size(); iter++) {
                reshapePs.shape_dims[iter] = caffeReshapeParam.shape().dim(iter);
            }
            reshapePs.axis = caffeReshapeParam.axis();
            reshapePs.num_axes = caffeReshapeParam.num_axes();
            opsPtr[i].ps.reshape_spec = reshapePs;
        }else if (layer.type() == "Slice") {
            SliceParamSpec slicePs;
            auto caffeSliceParam = layer.slice_param();
            for (I32 iter = 0; iter < caffeSliceParam.slice_point().size(); iter++) {
                slicePs.slice_points[iter] = caffeSliceParam.slice_point(iter);
            }
            slicePs.slice_size = caffeSliceParam.slice_point().size();
            slicePs.axis = caffeSliceParam.axis();
            opsPtr[i].ps.slice_spec = slicePs;
        }else if (layer.type() == "Transpose") {
            TransposeParamSpec transPs;
            auto caffeTransposeParam = layer.transpose_param();
            for (I32 iter=0; iter < caffeTransposeParam.dim().dim_size(); iter++) {
                transPs.trans_dims[iter] = caffeTransposeParam.dim().dim(iter);
            }
            transPs.trans_size = caffeTransposeParam.dim().dim_size();
            opsPtr[i].ps.transpose_spec = transPs;
        }else if (layer.type() == "Attention") {
            AttentionParamSpec attentionPs;
            auto caffe_attention_param = layer.attention_param();
            attentionPs.num_attention  = caffe_attention_param.num_attention();
            opsPtr[i].ps.attention_spec = attentionPs;
        }
    }

    inputsNumber = (inputsNumber > proto.input_size()) ? inputsNumber : proto.input_size();
    ms->num_inputs  = inputsNumber;
    ms->input_names = (I8 **)mt_malloc(inputsNumber * sizeof(I8 *));
    ms->input_dims  = (TensorDesc*)mt_malloc(sizeof(TensorDesc) * inputsNumber);
    for (int i = 0; i < inputsNumber; i++) {
        ms->input_names[i] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
 
        if (proto.input_size() > 0) {
            str_copy(ms->input_names[i], proto.input(i).c_str(), proto.input(i).length());
            switch (proto.input_dim_size()) {
                case 2:
                    ms->input_dims[i] = tensor4df(DT_F32, DF_NORMAL, 1, 1,
                                                  proto.input_dim(0), 
                                                  proto.input_dim(1));
                    break;
                case 3:
                    ms->input_dims[i] = tensor4df(DT_F32, DF_NCHW, 1, 
                                                  proto.input_dim(0), 
                                                  proto.input_dim(1), 
                                                  proto.input_dim(2));
                    break;
                case 4:
                    ms->input_dims[i] = tensor4df(DT_F32, DF_NCHW, 
                                                  proto.input_dim(0),  
                                                  proto.input_dim(1),  
                                                  proto.input_dim(2),  
                                                  proto.input_dim(3));
                    break;
                default: {
                    std::cerr << "[ERROR] unsupport input dim" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                    break;
                }
            }
        }
        if (i < proto.input_shape_size()) {
            str_copy(ms->input_names[i], proto.input(i).c_str(), proto.input(i).length());
            switch (proto.input_shape(i).dim_size()) {
                case 3:
                    ms->input_dims[i] = tensor4df(DT_F32, DF_NCHW, 1,
                                                  proto.input_shape(i).dim(0),
                                                  proto.input_shape(i).dim(1),
                                                  proto.input_shape(i).dim(2));
                    break;
                case 4:
                    ms->input_dims[i] = tensor4df(DT_F32, DF_NCHW,
                                                  proto.input_shape(i).dim(0),
                                                  proto.input_shape(i).dim(1),
                                                  proto.input_shape(i).dim(2),
                                                  proto.input_shape(i).dim(3));
                    break;
                default: {
                    std::cerr << "[ERROR] unsupport input dim" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                    break;
                }
            }
        }
    }

    int outputsNumber = 0;
    for (auto iter: outputCounts) {
        if (iter.second > 0) {
            outputsNumber ++;
        }
    }
    ms->num_outputs = outputsNumber;
    ms->output_names = (I8 **)mt_malloc(outputsNumber * sizeof(I8 *));
    outputsNumber = 0;
    for (auto iter: outputCounts) {
        if (iter.second > 0) {
            ms->output_names[outputsNumber] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(ms->output_names[outputsNumber], iter.first.c_str(), iter.first.length());
            outputsNumber ++;
        }
    }
    ms->num_weight_specs = weightNumber;
    WeightSpec* wsPtr = (WeightSpec*)mt_malloc(sizeof(WeightSpec) * weightNumber);
    ms->ws = wsPtr;

    int inNamesIndex = 0;
    int weightIndex = 0;
    for (int i = 0; i < proto.layer_size(); i++) {
        const caffe::LayerParameter layer = proto.layer(i);
        std::string layerName = layer.name();
        std::string layerType = layer.type();

        if (layerType == "Input") {
            str_copy(ms->input_names[inNamesIndex], layerName.c_str(), layerName.length());
            switch (layer.input_param().shape(0).dim_size()) {
                case 2:
                    ms->input_dims[inNamesIndex] = tensor2df(DT_U32, DF_NORMAL,
                                                               layer.input_param().shape(0).dim(0),
                                                               layer.input_param().shape(0).dim(1));
                    break;
                case 3:
                    ms->input_dims[inNamesIndex] = tensor4df(DT_F32, DF_NCHW, 1,
                                                               layer.input_param().shape(0).dim(0),
                                                               layer.input_param().shape(0).dim(1),
                                                               layer.input_param().shape(0).dim(2));
                    break;
                case 4:
                    ms->input_dims[inNamesIndex] = tensor4df(DT_F32, DF_NCHW, layer.input_param().shape(0).dim(0),
                                                               layer.input_param().shape(0).dim(1),
                                                               layer.input_param().shape(0).dim(2),
                                                               layer.input_param().shape(0).dim(3));
                    break;
                default: {
                    std::cerr << "[ERROR] unsupport input dim" << std::endl;
                    CHECK_STATUS_WITH_RETURN(NOT_SUPPORTED);
                    break;
                }
            }
            inNamesIndex++;
        }else if (layerType == "Convolution" || layerType == "InnerProduct" || layerType == "BatchNorm" ||layerType == "Embed") {
            int netLayerId = net_search_layerId(net, layerName);
            CHECK_REQUIREMENT(netLayerId >= 0);
            str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
            U32 blobNum = net_get_blobs_size(net, netLayerId);
            net_copy_blob(wsPtr, weightIndex, net, netLayerId, blobNum, 0);
            weightIndex++;
        }else if (layerType == "Scale" || layerType == "LayerNorm") {
            int netLayerId = net_search_layerId(net, layerName);
            CHECK_REQUIREMENT(netLayerId >= 0);
            str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
            U32 blobNum = net_get_blobs_size(net, netLayerId);
            if (layer.bottom_size() == 1) {
                CHECK_REQUIREMENT(blobNum == 2);
            }
            else {
                CHECK_REQUIREMENT(blobNum == 0);
            }
            net_copy_blob(wsPtr, weightIndex, net, netLayerId, blobNum, 0);
            weightIndex++;
        }else if (layerType == "LSTM") {
            int netLayerId = net_search_layerId(net, layerName);
            CHECK_REQUIREMENT(netLayerId >= 0);
            str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
            U32 blobNum = net_get_blobs_size(net, netLayerId);
            net_copy_blob(wsPtr, weightIndex, net, netLayerId, blobNum, 1);
            weightIndex++;
        }
    }

    CHECK_REQUIREMENT(weightIndex == weightNumber);
    // relationship init null
    ms->num_op_tensor_entries = 0;
    ms->op_relationship_entries = nullptr;
    return SUCCESS;
}

extern "C" EE mt_store_caffe(CI8* dir, CI8* mfn, const ModelSpec* ms);
#endif
