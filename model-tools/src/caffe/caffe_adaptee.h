// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _H_CAFFEADAPTEE
#define _H_CAFFEADAPTEE

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
#include "model_adaptee.h"

class CaffeAdaptee: public ModelAdaptee {
public:
    CaffeAdaptee() {}
    ~CaffeAdaptee() {}

protected:
    
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

    OperatorType convert_caffe_type(std::string inputType)
    {
        if (inputType == "Convolution") {
            return OT_Conv;
        } else if (inputType == "Deconvolution") {
            return OT_Deconvolution;
        } else if (inputType == "BatchNorm") {
            return OT_BatchNorm;
        } else if (inputType == "Scale") {
            return OT_Scale;
        } else if (inputType == "Eltwise") {
            return OT_Eltwise;
        } else if (inputType == "InnerProduct") {
            return OT_FC;
        } else if (inputType == "Pooling") {
            return OT_Pooling;
        } else if (inputType == "ReLU") {
            return OT_Relu;
        } else if (inputType == "ReLU6") {
            return OT_Relu6;
        } else if (inputType == "HSwish") {
            return OT_HSwish;
        } else if (inputType == "Sigmoid") {
            return OT_Sigmoid;
        } else if (inputType == "HSigmoid") {
            return OT_HSigmoid;
        } else if (inputType == "Softmax") {
            return OT_Softmax;
        } else if (inputType == "Concat") {
            return OT_Concat;
        } else if (inputType == "Embed") {  
            return OT_Embedding;
        } else if (inputType == "Gelu") {
            return OT_Gelu;
        } else if (inputType == "LayerNorm") {
            return OT_LayerNorm;
        } else if (inputType == "MatMul") {
            return OT_MatMul;
        } else if (inputType == "Multiply") {
            return OT_Multiply;
        } else if (inputType == "Reshape") {
            return OT_Reshape;
        } else if (inputType == "Slice") {
            return OT_Slice;
        } else if (inputType == "Transpose") {
            return OT_Transpose;
        } else if (inputType == "Attention") {
            return OT_Attention;
        } else if (inputType == "Input") {
            return OT_Input;
        } else if (inputType == "LSTM") {
            return OT_LSTM;
        } else if (inputType == "TanH") {
            return OT_TanH;
        } else if (inputType == "SoftmaxWithLoss") {
            return OT_SoftmaxWithLoss;
        } else if (inputType == "Squeeze") {
            return OT_Squeeze;
        } else if (inputType == "Unsqueeze") {
            return OT_Unsqueeze;
        } else if (inputType == "AxisMean") {
            return OT_AxisMean;
        } else if (inputType == "ArgMax") {
            return OT_ArgMax;
        } else if (inputType == "PreAllocatedMemory") {
            return OT_PreAllocatedMemory;
        } else if (inputType == "SharedWeight") {
            return OT_SharedWeight;
        } else if (inputType == "Copy") {
            return OT_Copy;
        } else if (inputType == "Check") {
            return OT_Check;
        } else if (inputType == "Repeat") {
            return OT_Repeat;
        } else if (inputType == "Interp") {
            return OT_Interp;
        } else {
            std::cout << "[INFO] encounter unsupported operator " << inputType << std::endl;
            return OT_None;
        }
    }

    int net_search_layerId(caffe::NetParameter& netParams, std::string& layerName) {
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

    caffe::BlobProto net_get_blob(caffe::NetParameter& netParams, int layerId, int blobId) {
        if (netParams.layer_size() > 0) {
            return netParams.layer(layerId).blobs(blobId);
        }
        else{
            return netParams.layers(layerId).blobs(blobId);
        }
    }

    int net_get_blobs_size(caffe::NetParameter& netParams, int layerId) {
        if (netParams.layer_size() > 0) {
            return netParams.layer(layerId).blobs_size();
        }
        else{
            return netParams.layers(layerId).blobs_size();
        }
    }

    void net_copy_blob(WeightSpec* wsPtr, int weightIndex, caffe::NetParameter& netParams, int netLayerId, int blobNum)
    {
        wsPtr[weightIndex].mdt = DT_F32;
        wsPtr[weightIndex].bytes_of_weight = 0;
        wsPtr[weightIndex].weight = nullptr;
        wsPtr[weightIndex].bytes_of_vec = 0;
        wsPtr[weightIndex].vec = nullptr;
    
        if (blobNum >= 1) {
            caffe::BlobProto blob0 = net_get_blob(netParams, netLayerId, 0);
            U32 elemSize = sizeof(*(blob0.data().data()));
            CHECK_REQUIREMENT(elemSize == bytesOf(wsPtr[weightIndex].mdt));
            wsPtr[weightIndex].bytes_of_weight = elemSize * blob0.data_size();
            wsPtr[weightIndex].weight = (U8*)mt_new_storage(wsPtr[weightIndex].bytes_of_weight);
            memcpy(wsPtr[weightIndex].weight, (U8*)blob0.data().data(), wsPtr[weightIndex].bytes_of_weight);
        }
        if (blobNum >= 2) {
            caffe::BlobProto blob1 = net_get_blob(netParams, netLayerId, 1);
            CHECK_REQUIREMENT(sizeof(*(blob1.data().data())) == bytesOf(wsPtr[weightIndex].mdt));
            wsPtr[weightIndex].bytes_of_vec = sizeof(*(blob1.data().data())) * blob1.data_size();
            wsPtr[weightIndex].vec = (U8*)mt_new_storage(wsPtr[weightIndex].bytes_of_vec);
            memcpy(wsPtr[weightIndex].vec, (U8*)blob1.data().data(), wsPtr[weightIndex].bytes_of_vec);
        }
        // Batchnorm may have 3 blobs, but the third blob can be ignored
    }

    EE parse_file(std::string dir, std::string mfn) override {
        EE ret = SUCCESS;
        std::string prototxtSuffix = ".prototxt";
        std::string caffeModelSuffix = ".caffemodel";
        std::string prototxtPath = dir + "/" + mfn + prototxtSuffix;
        std::string caffeModelPath = dir + "/" + mfn + caffeModelSuffix;

        // load prototxt
        ret = read_from_prototxt(prototxtPath.c_str(), &proto);
        if (proto.layer_size() <= 0) {
            std::cerr << "[ERROR] null caffemodel " << caffeModelPath << std::endl;
            exit(-1);
        }

        // load model bin.
        ret = read_from_caffemodel(caffeModelPath.c_str(), &net);
        return ret;
    }

    // the first loop can specify the input info and output info
    EE adapt_operators(ModelSpec* ms) override
    {
	    EE ret = SUCCESS;
        // model_name
        str_copy(ms->model_name, proto.name().c_str(), proto.name().length());
        ms->dt = DT_F32; // set default value

        ms->num_operator_specs = proto.layer_size();
        OperatorSpec* opsPtr = (OperatorSpec*)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
        }

        int inputsNumber = 0;
        weightNumber = 0;    // set global variable initial value
        std::map<std::string, int> outputCounts;
        for (int i = 0; i < proto.input_size(); i++) {
            outputCounts[proto.input(i).c_str()] = 1;
        }
        for (int i = 0; i < proto.layer_size(); i++) {
            const caffe::LayerParameter curLayer = proto.layer(i);
            this->layer = curLayer; 

            if (layer.type() == "Input") {    // layer,the global variable
                inputsNumber++;
            }
            str_copy(opsPtr[i].name, layer.name().c_str(), layer.name().length());
            
            opsPtr[i].type = convert_caffe_type(layer.type());
            int bottomSize = layer.bottom_size();
            opsPtr[i].num_inputs = bottomSize;
            opsPtr[i].input_tensors_name = (I8**)mt_new_storage(bottomSize * sizeof(I8 *));
            for (int j = 0; j < bottomSize; j++) {
                opsPtr[i].input_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[i].input_tensors_name[j], layer.bottom(j).c_str(), layer.bottom(j).length());
                if (outputCounts.find(layer.bottom(j)) == outputCounts.end()) {
                    std::cerr << "[ERROR] encounter no output as this operator's input" << std::endl;
                    exit(1);
                } else {
                    outputCounts[layer.bottom(j)]--;
                }
            }
            int topSize = layer.top_size();
            opsPtr[i].num_outputs = topSize;
            opsPtr[i].output_tensors_name = (I8 **)mt_new_storage(topSize * sizeof(I8 *));
            for (int j = 0; j < topSize; j++) {
                opsPtr[i].output_tensors_name[j] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[i].output_tensors_name[j], layer.top(j).c_str(), layer.top(j).length());
                if (outputCounts.find(layer.top(j)) == outputCounts.end()) {
                    outputCounts[layer.top(j)] = 1;
                } else {
                    outputCounts[layer.top(j)]++;
                }
            }

            ParameterSpec curPs;
            OperatorType curType = convert_caffe_type(layer.type());     
            ret = adapt_operator(curType, &curPs);
	        ms->ops[i].ps = curPs;
        }

        inputsNumber = (inputsNumber > proto.input_size()) ? inputsNumber : proto.input_size();
        ms->num_inputs  = inputsNumber;
        ms->input_names = (I8**)mt_new_storage(inputsNumber * sizeof(I8 *));
        ms->input_dims  = (TensorDesc*)mt_new_storage(sizeof(TensorDesc) * inputsNumber);
        for (int i = 0; i < inputsNumber; i++) {
            ms->input_names[i] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
     
            if (proto.input_size() > 0) {
                str_copy(ms->input_names[i], proto.input(i).c_str(), proto.input(i).length());
                switch (proto.input_dim_size()) {
                    case 2:
                        ms->input_dims[i] = tensor2df(DT_U32, DF_NORMAL,
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
                        std::cerr << "[ERROR] unsupported input dim" << std::endl;
                        exit(-1);
                    }
                }
            }
            if (i < proto.input_shape_size()) {
                str_copy(ms->input_names[i], proto.input(i).c_str(), proto.input(i).length());
                switch (proto.input_shape(i).dim_size()) {
                    case 2:
                        ms->input_dims[i] = tensor2df(DT_U32, DF_NORMAL,
                                                      proto.input_shape(i).dim(0),
                                                      proto.input_shape(i).dim(1));
                        break;
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
                        exit(-1);
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
        ms->output_names = (I8**)mt_new_storage(outputsNumber * sizeof(I8*));
        outputsNumber = 0;
        for (auto iter: outputCounts) {
            if (iter.second > 0) {
                ms->output_names[outputsNumber] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(ms->output_names[outputsNumber], iter.first.c_str(), iter.first.length());
                outputsNumber ++;
            }
        }
        ms->num_weight_specs = this->weightNumber;    // use the global variable
        return ret;
    }

    EE adapt_weights(ModelSpec* ms) override {
	EE ret = SUCCESS;
        WeightSpec* wsPtr = (WeightSpec*)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
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
                        exit(-1);
                    }
                }
                inNamesIndex++;
            }else if (layerType == "Convolution"
                   || layerType == "InnerProduct"
                   || layerType == "BatchNorm"
                   || layerType == "Embed"
                   || layerType == "LSTM"
                   || layerType == "SharedWeight"
                   || layerType == "Deconvolution") {
                int netLayerId = net_search_layerId(net, layerName);
                CHECK_REQUIREMENT(netLayerId >= 0);
                str_copy(wsPtr[weightIndex].op_name, layerName.c_str(), layerName.length());
                U32 blobNum = net_get_blobs_size(net, netLayerId);
                net_copy_blob(wsPtr, weightIndex, net, netLayerId, blobNum);
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
                net_copy_blob(wsPtr, weightIndex, net, netLayerId, blobNum);
                weightIndex++;
            }
        }

        CHECK_REQUIREMENT(weightIndex == weightNumber);
        // relationship init null
        ms->num_op_tensor_entries = 0;
        ms->op_relationship_entries = nullptr;
	    return ret;
    }

    ParameterSpec adapt_Interp() override
    {
        ParameterSpec curPs;
        InterpParamSpec interpPs;
        auto caffeInterpParam = layer.interp_param();
        interpPs.height = caffeInterpParam.height();
        interpPs.width = caffeInterpParam.width();
        curPs.interp_spec = interpPs;
        return curPs;
    }

    ParameterSpec adapt_Conv() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        ConvolutionParamSpec cps;
        cps.num_kernels = layer.convolution_param().num_output();
        if (layer.convolution_param().has_kernel_w() && layer.convolution_param().has_kernel_h()) {
            cps.kernel_size_w = layer.convolution_param().kernel_w();
            cps.kernel_size_h = layer.convolution_param().kernel_h();
        } else {
            cps.kernel_size_h = layer.convolution_param().kernel_size(0);
            cps.kernel_size_w = cps.kernel_size_h;
        }
        
        cps.group = (layer.convolution_param().has_group()) ? layer.convolution_param().group() : 1;  // group[default=1]

        cps.dilatedRate_h = (layer.convolution_param().dilation_size() != 0) ? layer.convolution_param().dilation(0) : 1;
        cps.dilatedRate_w = cps.dilatedRate_h;

        if (cps.group != cps.num_kernels) {
            std::cout << "[INFO] Convolution group != num_kernels" << std::endl;
            cps.group = 1;
        } else {
            std::cout << "[INFO] Depthwise Convolution" << std::endl;
        }
        if (cps.group == 1) {
            if (cps.dilatedRate_h > 1) {
                cps.convolution_type = Convolution_Dilation;
            } else {
                cps.convolution_type = Convolution_Pointwise;
            }
        } else {
            cps.convolution_type = Convolution_Depthwise;
        }
        cps.dw_activation_type = ACTIVATION_NULL;
        cps.pw_activation_type = ACTIVATION_NULL;
        if (layer.convolution_param().has_stride_w() && layer.convolution_param().has_stride_h()) {
            cps.stride_w = layer.convolution_param().stride_w();
            cps.stride_h = layer.convolution_param().stride_h();
        } else {
            cps.stride_h = (layer.convolution_param().stride_size() != 0) ? layer.convolution_param().stride(0) : 1;  // stride[default=1]
            cps.stride_w = cps.stride_h;
        }
        if (layer.convolution_param().has_pad_w() && layer.convolution_param().has_pad_h()) {
            cps.padding_left = layer.convolution_param().pad_w();
            cps.padding_right = cps.padding_left;
            cps.padding_top = layer.convolution_param().pad_h();
            cps.padding_bottom = cps.padding_top;
        } else {
            cps.padding_top = (layer.convolution_param().pad_size() != 0) ? layer.convolution_param().pad(0) : 0;  // pad[default=0]
            cps.padding_bottom = cps.padding_top;
            cps.padding_left = cps.padding_top;
            cps.padding_right = cps.padding_top;
        }
        curPs.conv_spec = cps;
        return curPs;
    }

    ParameterSpec adapt_Deconvolution() override
    {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        ConvolutionParamSpec cps;
        cps.num_kernels = layer.convolution_param().num_output();
        if (layer.convolution_param().has_kernel_w() && layer.convolution_param().has_kernel_h()) {
            cps.kernel_size_w = layer.convolution_param().kernel_w();
            cps.kernel_size_h = layer.convolution_param().kernel_h();
        } else {
            cps.kernel_size_h = layer.convolution_param().kernel_size(0);
            cps.kernel_size_w = cps.kernel_size_h;
        }
        
        cps.group = (layer.convolution_param().has_group()) ? layer.convolution_param().group() : 1;
        if (1 != cps.group) {
            std::cerr << "[ERROR] Deconvolution group != 1" << std::endl;
	        std::cerr << "[ERROR]: UNSUPPORTED!" << std::endl;
            CHECK_STATUS(NOT_SUPPORTED);
        }
        cps.dilatedRate_h = 1;
        cps.dilatedRate_w = 1;
        cps.convolution_type = Convolution_Deconvolution;
        cps.dw_activation_type = ACTIVATION_NULL;
        cps.pw_activation_type = ACTIVATION_NULL;
        if (layer.convolution_param().has_stride_w() && layer.convolution_param().has_stride_h()) {
            cps.stride_w = layer.convolution_param().stride_w();
            cps.stride_h = layer.convolution_param().stride_h();
        } else {
            cps.stride_h = (layer.convolution_param().stride_size() != 0) ? layer.convolution_param().stride(0) : 1;  // stride[default=1]
            cps.stride_w = cps.stride_h;
        }
        if (layer.convolution_param().has_pad_w() && layer.convolution_param().has_pad_h()) {
            cps.padding_left = layer.convolution_param().pad_w();
            cps.padding_right = cps.padding_left;
            cps.padding_top = layer.convolution_param().pad_h();
            cps.padding_bottom = cps.padding_top;
        } else {
            cps.padding_top = (layer.convolution_param().pad_size() != 0) ? layer.convolution_param().pad(0) : 0;  // pad[default=0]
            cps.padding_bottom = cps.padding_top;
            cps.padding_left = cps.padding_top;
            cps.padding_right = cps.padding_top;
        }
        curPs.conv_spec = cps;
        return curPs;
    }

    ParameterSpec adapt_Pooling() override {
        ParameterSpec curPs;
        PoolingParamSpec pps;
        if (layer.pooling_param().has_kernel_w() && layer.pooling_param().has_kernel_h()) {
            pps.kernel_size_w = layer.pooling_param().kernel_w();
            pps.kernel_size_h = layer.pooling_param().kernel_h();
        } else {
            pps.kernel_size_h = layer.pooling_param().kernel_size();
            pps.kernel_size_w = pps.kernel_size_h;
        }
        if (layer.pooling_param().has_stride_w() && layer.pooling_param().has_stride_h()) {
            pps.stride_w = layer.pooling_param().stride_w();
            pps.stride_h = layer.pooling_param().stride_h();
        } else {
            pps.stride_h = layer.pooling_param().stride();
            pps.stride_w = pps.stride_h;
        }
        bool global_pooling = layer.pooling_param().global_pooling();
        if (global_pooling) {
            pps.kernel_size_h = 0;
            pps.kernel_size_w = 0;
            pps.stride_h = 1;
            pps.stride_w = 1;
        }else {
            CHECK_REQUIREMENT(pps.kernel_size_h > 0);
        }
        if (layer.pooling_param().has_pad_w() && layer.pooling_param().has_pad_h()) {
            pps.padding_left = layer.pooling_param().pad_w();
            pps.padding_right = pps.padding_left;
            pps.padding_top = layer.pooling_param().pad_h();
            pps.padding_bottom = pps.padding_top;
        } else {
            pps.padding_top = layer.pooling_param().has_pad() ? layer.pooling_param().pad() : 0;
            pps.padding_bottom = pps.padding_top;
            pps.padding_left = pps.padding_top;
            pps.padding_right = pps.padding_top;
        }
        
        if (layer.pooling_param().has_round_mode() && layer.pooling_param().round_mode() == 1) {
            pps.rm = FLOOR;
        }else{
            pps.rm = CEIL;
        }
        switch (layer.pooling_param().pool()) {
            case caffe::PoolingParameter_PoolMethod_MAX: {
                pps.mode = POOLING_MAX;
                break;
            }
            case caffe::PoolingParameter_PoolMethod_AVE: {
                pps.mode = POOLING_MEAN;
                break;
            }
            default: {
                std::cerr << "[ERROR] encounter unsupported Pooling method " << layer.pooling_param().pool() << std::endl;
                exit(1);
            }
        }
        curPs.pooling_spec = pps;
        return curPs;
    }

    ParameterSpec adapt_Fc() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1; 
        FullyConnectedParamSpec ips;
        ips.num_outputs = layer.inner_product_param().num_output();
        curPs.fc_spec = ips;
        return curPs;   
    }

    ParameterSpec adapt_BatchNorm() override {   
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        BatchNormParamSpec bnps;
        bnps.eps = layer.batch_norm_param().eps();
        curPs.bn_spec = bnps;
        return curPs; 
    }

    ParameterSpec adapt_LayerNorm() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        return curPs;
    }

    ParameterSpec adapt_Eltwise() override{
        ParameterSpec curPs;
        EltwiseParamSpec eps;
        EltwiseSumSpec ess;

        auto caffeEltwiseParam = layer.eltwise_param();
        auto op = caffeEltwiseParam.operation();
        switch (op)
        {
            case caffe::EltwiseParameter_EltwiseOp_PROD:
                eps.elt_mode = ELTWISE_PROD;
                break;
            case caffe::EltwiseParameter_EltwiseOp_SUM:
                eps.elt_mode = ELTWISE_SUM;
                break;
            case caffe::EltwiseParameter_EltwiseOp_MAX:
                eps.elt_mode = ELTWISE_MAX;
                break;
            default: {
                std::cerr << "[ERROR] unknown eltwise mode" << std::endl;
                exit(-1);
            }
        }
        U32 bytes = caffeEltwiseParam.coeff_size() * sizeof(F32);
        ess.coeff_size = caffeEltwiseParam.coeff_size();
        ess.coeff_values = (F32 *)mt_new_storage(bytes);
        memcpy(ess.coeff_values, caffeEltwiseParam.coeff().data(), bytes);
        for (int j = 0; j < caffeEltwiseParam.coeff_size(); j++) {
            CHECK_REQUIREMENT(ess.coeff_values[j] == 1);
        }
        eps.elt_sum_spec = ess;
        curPs.eltwise_spec = eps;
        return curPs;
    }

    ParameterSpec adapt_Embedding() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        EmbedParamSpec embedPs;
        auto caffeEmbedParam = layer.embed_param();
        embedPs.input_dim  = caffeEmbedParam.input_dim();
        embedPs.num_output = caffeEmbedParam.num_output();
        embedPs.bias_term  = caffeEmbedParam.bias_term() == 0 ? false: true;
        embedPs.transpose  = caffeEmbedParam.transpose() == 0 ? false : true;
        curPs.embed_spec = embedPs;
        return curPs;
    }

    ParameterSpec adapt_Multiply() override {
        ParameterSpec curPs;
        MultiplyParamSpec multiplyPs;
        auto caffeMultiplyParam = layer.multiply_param();
        multiplyPs.scale = caffeMultiplyParam.scale();
        multiplyPs.bias = caffeMultiplyParam.bias();
        curPs.multiply_spec = multiplyPs;
        return curPs;
    }

    ParameterSpec adapt_Reshape() override { 
        ParameterSpec curPs;
        ReshapeParamSpec reshapePs;
        auto caffeReshapeParam = layer.reshape_param();
        reshapePs.shape_size = caffeReshapeParam.shape().dim_size();
        for (I32 iter = 0; iter < caffeReshapeParam.shape().dim_size(); iter++) {
            reshapePs.shape_dims[iter] = caffeReshapeParam.shape().dim(iter);
        }
        reshapePs.axis = caffeReshapeParam.axis();
        reshapePs.num_axes = caffeReshapeParam.num_axes();
        curPs.reshape_spec = reshapePs;
        return curPs;
    }

    ParameterSpec adapt_Slice() override {
        ParameterSpec curPs;
        SliceParamSpec slicePs;
        auto caffeSliceParam = layer.slice_param();
        for (I32 iter = 0; iter < caffeSliceParam.slice_point().size(); iter++) {
            slicePs.slice_points[iter] = caffeSliceParam.slice_point(iter);
        }
        slicePs.slice_size = caffeSliceParam.slice_point().size();
        slicePs.axis = caffeSliceParam.axis();
        curPs.slice_spec = slicePs;
        return curPs;
    }

    ParameterSpec adapt_Transpose() override {
        ParameterSpec curPs;
        TransposeParamSpec transPs;
        auto caffeTransposeParam = layer.transpose_param();
        for (I32 iter=0; iter < caffeTransposeParam.dim().dim_size(); iter++) {
            transPs.trans_dims[iter] = caffeTransposeParam.dim().dim(iter);
        }
        transPs.trans_size = caffeTransposeParam.dim().dim_size();
        curPs.transpose_spec = transPs;
        return curPs;
    }

    ParameterSpec adapt_Attention() override {    
        ParameterSpec curPs;
        AttentionParamSpec attentionPs;
        auto caffe_attention_param = layer.attention_param();
        attentionPs.num_heads  = caffe_attention_param.num_heads();
        attentionPs.from_sequence_length  = caffe_attention_param.from_sequence_length();
        attentionPs.to_sequence_length  = caffe_attention_param.to_sequence_length();
        curPs.attention_spec = attentionPs;
        return curPs;
    }
    
    ParameterSpec adapt_LSTM() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        LSTMParamSpec lstmPs;
        auto caffeLSTMParam = layer.lstm_param();
        lstmPs.num_output = caffeLSTMParam.num_output();
        lstmPs.steps = caffeLSTMParam.steps();
        curPs.lstm_spec = lstmPs;
        return curPs;
    }

    ParameterSpec adapt_Scale() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        return curPs;
    }

    ParameterSpec adapt_AxisMean() override {
        ParameterSpec curPs;
        AxisMeanParamSpec axisMeanPs;
        auto caffeAxisMeanParam = layer.axis_mean_param();
        axisMeanPs.axis = caffeAxisMeanParam.axis();
        curPs.axis_mean_spec = axisMeanPs;
        return curPs;
    }

    ParameterSpec adapt_Squeeze() override {
        ParameterSpec curPs;
        SqueezeParamSpec squeezePs;
        auto caffeSqueezeParam = layer.squeeze_param();
        squeezePs.axis = caffeSqueezeParam.axis();
        squeezePs.axes_num = 0;
        curPs.squeeze_spec = squeezePs;
        return curPs;
    }

    ParameterSpec adapt_Unsqueeze() override {
        ParameterSpec curPs;
        UnsqueezeParamSpec unsqueezePs;
        auto caffeUnsqueezeParam = layer.unsqueeze_param();
        unsqueezePs.axis = caffeUnsqueezeParam.axis();
        unsqueezePs.axes_num = 0;
        curPs.unsqueeze_spec = unsqueezePs;
        return curPs;
    }

    ParameterSpec adapt_ArgMax() override {
        ParameterSpec curPs;
        ArgMaxParamSpec argmaxPs;
        auto caffeArgMaxParam = layer.argmax_param();
        argmaxPs.axis = caffeArgMaxParam.axis();
        curPs.argmax_spec = argmaxPs;
        return curPs;
    }

    ParameterSpec adapt_Repeat() override {
        ParameterSpec curPs;
        RepeatParamSpec repeatPs;
        auto caffeRepeatParam = layer.repeat_param();
        repeatPs.loops = caffeRepeatParam.loops();
        curPs.repeat_spec = repeatPs;
        return curPs;
    }

    ParameterSpec adapt_Check() override {
        ParameterSpec curPs;
        CheckParamSpec checkPs;
        auto caffeCheckParam = layer.check_param();
        auto op = caffeCheckParam.operation();
        switch (op)
        {
            case caffe::CheckParameter_CheckOp_EQUAL:
                checkPs.check_mode = CHECK_EQUAL;
                break;
            default: {
                std::cerr << "[ERROR] unknown check mode" << std::endl;
                exit(-1);
            }
        }
        curPs.check_spec = checkPs;
        return curPs;
    }

    ParameterSpec adapt_PreAllocatedMemory() override {
        ParameterSpec curPs;
        PreAllocatedMemoryParamSpec preAllocatedMemoryPs;
        auto caffePreAllocatedMemoryParam = layer.preallocated_memory_param();
        preAllocatedMemoryPs.desc.nDims = caffePreAllocatedMemoryParam.shape().dim_size();
        for (I32 iter=0; iter < caffePreAllocatedMemoryParam.shape().dim_size(); iter++) {
            preAllocatedMemoryPs.desc.dims[preAllocatedMemoryPs.desc.nDims - 1 - iter] = caffePreAllocatedMemoryParam.shape().dim(iter);
        }
        preAllocatedMemoryPs.desc.df = getTensorDefaultDataFormat(preAllocatedMemoryPs.desc.nDims);
        auto dt = caffePreAllocatedMemoryParam.data_type();
        switch (dt)
        {
            case caffe::PreAllocatedMemoryParameter_DataType_FLOAT32:
                preAllocatedMemoryPs.desc.dt = DT_F32;
                break;
            case caffe::PreAllocatedMemoryParameter_DataType_UINT32:
                preAllocatedMemoryPs.desc.dt = DT_U32;
                break;
            case caffe::PreAllocatedMemoryParameter_DataType_INT32:
                preAllocatedMemoryPs.desc.dt = DT_I32;
                break;
            default: {
                std::cerr << "[ERROR] unknown memory data type" << std::endl;
                exit(-1);
            }
        }
        curPs.preallocated_memory_spec = preAllocatedMemoryPs;
        return curPs;
    }

    ParameterSpec adapt_SharedWeight() override {
        ParameterSpec curPs;
        weightNumber = weightNumber + 1;
        SharedWeightParamSpec sharedWeightPs;
        auto caffeSharedWeightParam = layer.shared_weight_param();
        sharedWeightPs.desc.nDims = caffeSharedWeightParam.shape().dim_size();
        for (I32 iter=0; iter < caffeSharedWeightParam.shape().dim_size(); iter++) {
            sharedWeightPs.desc.dims[sharedWeightPs.desc.nDims - 1 - iter] = caffeSharedWeightParam.shape().dim(iter);
        }
        sharedWeightPs.desc.df = getTensorDefaultDataFormat(sharedWeightPs.desc.nDims);
        auto dt = caffeSharedWeightParam.data_type();
        switch (dt)
        {
            case caffe::SharedWeightParameter_DataType_FLOAT32:
                sharedWeightPs.desc.dt = DT_F32;
                break;
            case caffe::SharedWeightParameter_DataType_UINT32:
                sharedWeightPs.desc.dt = DT_U32;
                break;
            case caffe::SharedWeightParameter_DataType_INT32:
                sharedWeightPs.desc.dt = DT_I32;
                break;
            default: {
                std::cerr << "[ERROR] unknown weight data type" << std::endl;
                exit(-1);
            }
        }
        curPs.shared_weight_spec = sharedWeightPs;
        return curPs;
    }

    ParameterSpec adapt_Copy() override {
        ParameterSpec curPs;
        CopyParamSpec copyPs;
        auto caffeCopyParam = layer.copy_param();
        copyPs.src_dims[0] = caffeCopyParam.src_batch_stride();
        copyPs.src_dims[1] = caffeCopyParam.src_stride();
        copyPs.src_dims[2] = caffeCopyParam.src_offset();
        copyPs.dst_dims[0] = caffeCopyParam.dst_batch_stride();
        copyPs.dst_dims[1] = caffeCopyParam.dst_stride();
        copyPs.dst_dims[2] = caffeCopyParam.dst_offset();
        copyPs.length = caffeCopyParam.length();
        curPs.copy_spec = copyPs;
        return curPs;
    }

    ParameterSpec adapt_MatMul() override {
        ParameterSpec curPs;
        MatMulParamSpec matmulPs;
        auto caffeMatMulParam = layer.matmul_param();
        matmulPs.transpose_a = caffeMatMulParam.transpose_a();
        matmulPs.transpose_b = caffeMatMulParam.transpose_b();
        curPs.matmul_spec = matmulPs;
        return curPs;
    }

private:
    caffe::NetParameter proto;
    caffe::NetParameter net;
    caffe::LayerParameter layer;
    int weightNumber;
};
#endif
