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
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "onnx.pb.h"

#include "type.h"
#include "converter.h"
#include "model_serialize_deserialize.hpp"
#include "model_tools.h"
#include "model_adaptee.h"

class OnnxAdaptee: public ModelAdaptee {
public:
    OnnxAdaptee(int removePreprocessOpNum_outside, TensorDesc inputDesc_outside) {
        this->removePreprocessOpNum = removePreprocessOpNum_outside;
        this->inputDesc = inputDesc_outside;
    }
    ~OnnxAdaptee() {}

protected:   
    EE read_from_onnx_file(const char* path, google::protobuf::Message* message) {
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

    OperatorType convert_onnx_type(std::string inputType) {
        if (inputType == "Conv") {
            return OT_Conv;
        } else if (inputType == "BatchNormalization") {
            return OT_BatchNorm;
        } else if (inputType == "Sum" || inputType == "Add" || inputType == "Mul") {
            return OT_Eltwise;
        } else if (inputType == "Gemm") {
            return OT_FC;
        } else if (inputType == "AveragePool" || inputType == "MaxPool" || inputType == "ReduceMean" || inputType == "GlobalAveragePool") {
            return OT_Pooling;
        } else if (inputType == "Relu") {
            return OT_Relu;
        } else if (inputType == "Softmax") {
            return OT_Softmax;
        } else if (inputType == "Concat") {
            return OT_Concat;
        } else if (inputType == "Pad") {
            return OT_Pad;
        } else if (inputType == "Max" || inputType == "Min" || inputType == "Clip") {
            return OT_Clip;
        } else if (inputType == "Reshape") {
            return OT_Reshape;
        } else if (inputType == "Squeeze") {
            return OT_Squeeze;
        } else if (inputType == "Transpose") {
            return OT_Transpose;
        } else if (inputType == "Gather") {   
            return OT_Gather;
        } else if (inputType == "Unsqueeze") {
            return OT_Unsqueeze;
        } else if (inputType == "Upsample") {
            return OT_Upsample;
        } else if (inputType == "Cast") {
            return OT_Cast;
        } else if (inputType == "Constant") {
            return OT_Constant;
        } else if (inputType == "MatMul") {
            return OT_MatMul;
        } else if (inputType == "Flatten") {
            return OT_Flatten;
        } else {
            return OT_None;
        }
    }

    std::vector<int> get_node_vector_ints_attribute_by_name(const onnx::NodeProto& node, const char* key)
    {
        std::vector<int> result;
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto& attribute = node.attribute(i);
            if (attribute.name() == key) {
                result.resize(attribute.ints_size());
                for (int j = 0; j < attribute.ints_size(); j++) {
                    result[j] = attribute.ints(j);
                }
                break;
            }
        }
        return result;
    }

    std::vector<F32> get_node_vector_float_tensor_attribute_by_name(const onnx::NodeProto& node, const char* key)
    {
        std::vector<F32> result;
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto& attribute = node.attribute(i);
            if (attribute.name() == key) {
                CHECK_REQUIREMENT(4 == attribute.type());
                const onnx::TensorProto& tp = attribute.t();
                F32 *value;
                if (tp.has_raw_data()) {
                    const std::string& rawData = tp.raw_data();
                    value = (F32*)(rawData.data());
                } else if (tp.data_type() == 1) {
                    value = (F32*)(tp.float_data().data());
                } else {
                    std::cout << "[Warning] Constant not extracted\n";
                    return result;
                }

                result.resize(tp.dims(0));
                for (int j = 0; j < tp.dims(0); j++) {
                    result[j] = value[j];
                }
                break;
            }
        }
        return result;
    }

    int get_node_single_int_attribute_by_name(const onnx::NodeProto& node, const char* key, int defaultValue = 0) {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto& attribute = node.attribute(i);
            if (attribute.name() == key) {
                return attribute.i();
            }
        }
        return defaultValue;
    }

    std::string get_node_str_attribute_by_name(const onnx::NodeProto& node, const char* key,
        const std::string& defaultValue = std::string()) {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto& attribute = node.attribute(i);
            if (attribute.name() == key) {
                return attribute.s();
            }
        }
        return defaultValue;
    }

    float get_node_float_attribute_by_name(const onnx::NodeProto& node, const char* key, float defaultValue = 0.f) {
       for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto& attribute = node.attribute(i);
            if (attribute.name() == key) {
                return attribute.f();
            }
        }
        return defaultValue;
    }

    int get_data_size_from_tensor_proto(const onnx::TensorProto& tensorProto) {
        if (tensorProto.has_raw_data()) {
            const std::string& rawData = tensorProto.raw_data();   
            int size = (int)rawData.size() / sizeof(float);   
            return size;
        } else if (tensorProto.data_type() == 1) {
            return tensorProto.float_data_size();
        }
        return 0;
    }

    float* get_ptr_from_weight_obj(const onnx::TensorProto& tensorProto) {
        if (tensorProto.has_raw_data()) {
            const std::string& rawData = tensorProto.raw_data();
            float* paramPtr = (float*)(rawData.data());
            return paramPtr;
        } else if (tensorProto.data_type() == 1) {
            float* paramPtr = (float*)(tensorProto.float_data().data());
            return paramPtr;
        }
        return nullptr;
    }

    std::vector<int> get_reshapeInfo_from_tensorProto(const onnx::TensorProto& tp)
    {
        const int64_t* shapeData = 0;
        int size = 0;

        // int64
        if (tp.has_raw_data()) {
            shapeData = (const int64_t*)tp.raw_data().data();
            size = tp.raw_data().size() / 8;
        } else if (tp.data_type() == 7) {
            shapeData = tp.int64_data().data();
            size = tp.int64_data_size();
        }

        std::vector<int> shape;
        for (int j=0; j < size; j++) {
            shape.push_back(shapeData[j]);
        }

        return shape;
    }

    EE parse_file(std::string dir, std::string mfn) override
    {
        std::string onnxSuffix = ".onnx";
        std::string onnxPath = dir + "/" + mfn + onnxSuffix;
        
        this->modelName = mfn;

        EE ret = read_from_onnx_file(onnxPath.c_str(), &onnxModel);
        if (ret != SUCCESS) {
            std::cerr << "[ERROR] fail to load " << onnxPath;
            exit(1);
        }

        onnxGraph = onnxModel.graph();

        for (int i=0; i < onnxGraph.initializer_size(); i++) {
            const onnx::TensorProto& initializer = onnxGraph.initializer(i);
            weights[initializer.name()] = initializer;
        }
        return ret;
    }

    EE adapt_operators(ModelSpec* ms) override
    {
        EE ret = SUCCESS;
        str_copy(ms->model_name, modelName.c_str(), modelName.length());
        ms->model_name[NAME_LEN - 1] = '\0';
        ms->dt = DT_F32;
        
        int onnxNodeCount = onnxGraph.node_size();

        ms->num_inputs = 1;
        ms->input_names = (I8**)mt_new_storage(ms->num_inputs * sizeof(I8*));
        const onnx::NodeProto& theFirstNode = onnxGraph.node(removePreprocessOpNum);    // need to be flexible
        std::string modelInputName = theFirstNode.input(0);
        ms->input_names[0] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(ms->input_names[0], modelInputName.c_str(), modelInputName.length());
        ms->input_dims  = (TensorDesc*)mt_new_storage(sizeof(TensorDesc) * ms->num_inputs);
        ms->input_dims[0] = inputDesc;

        ms->num_outputs = 1;
        ms->output_names = (I8**)mt_new_storage(ms->num_outputs * sizeof(I8*));
        const onnx::NodeProto& the_last_node = onnxGraph.node(onnxNodeCount - 1);
        std::string modelOutputName = the_last_node.output(0);
        ms->output_names[0] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
        str_copy(ms->output_names[0], modelOutputName.c_str(), modelOutputName.length());

        int bnOpNum = 0;
        int constantOpNum = 0;
        for (int i=0; i < onnxNodeCount; i++) {
            const onnx::NodeProto& tmpNode = onnxGraph.node(i);
            if (tmpNode.op_type() == "BatchNormalization") {
                bnOpNum++;
            } else if (tmpNode.op_type() == "Constant") {
                if (i >= removePreprocessOpNum) {
                    constantOpNum++;
                }
            }
        }

        ms->num_operator_specs = onnxNodeCount + bnOpNum - constantOpNum - removePreprocessOpNum;   // appending space for scale op
        OperatorSpec* opsPtr = (OperatorSpec*)mt_new_storage(sizeof(OperatorSpec) * ms->num_operator_specs);
        ms->ops = opsPtr;
        for (I32 i = 0; i < ms->num_operator_specs; i++) {
            ms->ops[i].tensor_positions = nullptr;
        }

        // Some models transformed from TF store weight and bias as Constant OP
        int numUnseenConstants = 0;
        nodeIndex = 0;
        for (int i = 0; i < removePreprocessOpNum; i++) {
            this->node = onnxGraph.node(nodeIndex);
            this->op = node.op_type();
            if (op == "Constant") {
                handle_Constant();
                numUnseenConstants++;
            }
            nodeIndex++;
        }
        if (0 != numUnseenConstants) {
            std::cout << removePreprocessOpNum << " OPs are skipped, and " << numUnseenConstants << " of them are Constant OP.\n";
        }

        nodeIndex = removePreprocessOpNum;
        int opIndex = 0;
        for (int i = removePreprocessOpNum; i < onnxNodeCount; i++) {
            this->node = onnxGraph.node(nodeIndex);
            this->op = node.op_type();
            if (op == "Constant") {
                handle_Constant();
                nodeIndex++;
                continue;
            }
            std::string opName = node.name();
            if (opName.empty()) {
                opName = node.output(0);
            }
            int opInputNum = (int)node.input_size();    
            opFinalInputNum = opInputNum;
            std::vector<std::string> inputNames;
            std::vector<std::string> op_weight_objs;
            for (int j = 0; j < opInputNum; j++) {
                const std::string& input_name = node.input(j);
                if (weights.find(input_name) != weights.end()) {
                    opFinalInputNum--;
                    op_weight_objs.push_back(input_name);
                } else {
                    inputNames.push_back(input_name);
                    if (op == "Max" || op == "Min") {
                        opFinalInputNum = 1;
                        break;
                    }
                }
            }
            int opOutputNum = (int)node.output_size();   
            std::vector<std::string> outputNames;
            for (int j = 0; j < opOutputNum; j++) {
                const std::string& output_name = node.output(j);
                outputNames.push_back(output_name);
            }

            str_copy(opsPtr[opIndex].name, opName.c_str(), opName.length());
            OperatorType opType = convert_onnx_type(op);
            opsPtr[opIndex].type = opType;
            opsPtr[opIndex].num_inputs = opFinalInputNum;
            opsPtr[opIndex].input_tensors_name = (I8**)mt_new_storage(opsPtr[opIndex].num_inputs * sizeof(I8 *));
            for (U32 j = 0; j < opsPtr[opIndex].num_inputs; j++) {
                opsPtr[opIndex].input_tensors_name[j] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[opIndex].input_tensors_name[j], inputNames[j].c_str(), inputNames[j].length());
            }
            opsPtr[opIndex].num_outputs = opOutputNum;
            opsPtr[opIndex].output_tensors_name = (I8**)mt_new_storage(opsPtr[opIndex].num_outputs * sizeof(I8 *));
            for (U32 j = 0; j < opsPtr[opIndex].num_outputs; j++) {
                opsPtr[opIndex].output_tensors_name[j] = (I8*)mt_new_storage(NAME_LEN * sizeof(I8));
                str_copy(opsPtr[opIndex].output_tensors_name[j], outputNames[j].c_str(), outputNames[j].length());
            }

            if ((op == "Add" && opFinalInputNum == 1) || (op == "Mul" && opFinalInputNum == 1)) {
                weightOpIndexLists.push_back(nodeIndex);
                opsPtr[opIndex].type = OT_Scale;
            } else if (op == "Transpose" && opFinalInputNum == 0) {
                weightOpIndexLists.push_back(nodeIndex);
            } else {
                ParameterSpec curPs;
                ret = adapt_operator(opType, &curPs);
                CHECK_STATUS(ret);
                opsPtr[opIndex].ps = curPs;

                if (opType == OT_BatchNorm) {
                    std::string scaleInputName = outputNames[0];
                    std::string scaleOpName = "scale_" + opName;
                    opIndex++;
                    str_copy(opsPtr[opIndex].name, scaleOpName.c_str(), scaleOpName.length());
                    opsPtr[opIndex].type = OT_Scale;
                    opsPtr[opIndex].num_inputs = 1;
                    opsPtr[opIndex].input_tensors_name = (I8 **)mt_new_storage(sizeof(I8 *));
                    opsPtr[opIndex].input_tensors_name[0] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[opIndex].input_tensors_name[0], scaleInputName.c_str(), scaleInputName.length());
                    opsPtr[opIndex].num_outputs = 1;
                    opsPtr[opIndex].output_tensors_name = (I8 **)mt_new_storage(sizeof(I8 *));
                    opsPtr[opIndex].output_tensors_name[0] = (I8 *)mt_new_storage(NAME_LEN * sizeof(I8));
                    str_copy(opsPtr[opIndex].output_tensors_name[0], scaleInputName.c_str(), scaleInputName.length());
                }
            }

            nodeIndex++;
            opIndex++;
        }
        ms->num_weight_specs = weightOpIndexLists.size() + bnOpNum;
	    return ret;
    }

    EE adapt_weights(ModelSpec* ms) override
    {
	    EE ret = SUCCESS;
        WeightSpec* wsPtr = (WeightSpec*)mt_new_storage(sizeof(WeightSpec) * ms->num_weight_specs);
        ms->ws = wsPtr;
        int weightOpIndexIndeed = 0;
        for (U32 i = 0; i < (U32)ms->num_weight_specs; i++) {
            int weightOpIndex = weightOpIndexLists[weightOpIndexIndeed];
            const onnx::NodeProto& weightNode = onnxGraph.node(weightOpIndex);
            std::string weightOpName = weightNode.name();
            if (weightOpName.empty()) {
                weightOpName = weightNode.output(0);
            }
            const std::string& weightOpType = weightNode.op_type();

            if (weightOpType == "Conv") {
                // to check that if any op has bias
                int convInputNum = weightNode.input_size();    // if convInputNum == 3, means has bias , otherwise , do not have bias

                const onnx::TensorProto& convWeightTp = weights[weightNode.input(1)];
                
                int convWeightNum = get_data_size_from_tensor_proto(convWeightTp);
                float* convWeightParamPtr = get_ptr_from_weight_obj(convWeightTp);
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());

                // traverse weight elements to see whether it is bnn convolution
                U32 isDOREFA = 0;
                U32 isXNOR = 0;
                for (I32 i = 0; i < convWeightNum; i++) {
                    float cur = convWeightParamPtr[i];
                    if (cur!=1.0 && cur!=0 && cur!=-1.0) {
                        isDOREFA = 0;
                        isXNOR = 0;
                        break;
                    }
                    if (cur == 0) {
                        if (isXNOR == 1) {
                            isDOREFA = 0;
                            isXNOR = 0;
                            break;
                        } else if (isDOREFA == 0) {
                            isDOREFA = 1;
                        }
                    } else if (cur == -1.0) {
                        if (isDOREFA == 1) {
                            isDOREFA = 0;
                            isXNOR = 0;
                            break;
                        } else if (isXNOR == 0) {
                            isXNOR = 1;
                        }
                    }
                }
                if (isDOREFA == 1) {
                    wsPtr[i].mdt = DT_BIN01;
                } else if (isXNOR == 1) {
                    wsPtr[i].mdt = DT_BIN11;
                } else {
                    wsPtr[i].mdt = DT_F32; // Assume weights will not all be 1.0
                }
                //wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = convWeightNum * sizeof(float); // Please do not change to bytesOf(mdt)
                wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, convWeightParamPtr, wsPtr[i].bytes_of_weight);
                
                int convBiasNum = 0;
                float* convBiasParamPtr = nullptr;
                if (convInputNum == 3) {
                    const onnx::TensorProto& convBiasTp = weights[weightNode.input(2)];
                    convBiasNum = get_data_size_from_tensor_proto(convBiasTp);
                    convBiasParamPtr = get_ptr_from_weight_obj(convBiasTp);
                    wsPtr[i].bytes_of_vec = convBiasNum * sizeof(float);
                    if (isDOREFA || isXNOR) {
                        wsPtr[i].bytes_of_vec *= 2; // BNN conv must have a scale vector and a bias vector, so that it can fuse with BN
                    }
                    wsPtr[i].vec = (U8*)mt_new_storage(wsPtr[i].bytes_of_vec);
                    if (isDOREFA == 1 || isXNOR == 1) {
                        U32 vecBytes = convBiasNum * sizeof(float);
                        F32 *scale = (F32*)wsPtr[i].vec;
                        for (I32 j = 0; j < convBiasNum; j++) {
                            scale[j] = 1.0;
                        }
                        memcpy(wsPtr[i].vec + vecBytes, convBiasParamPtr, vecBytes); // Copy bias (if any) to the second half for BNN
                    } else {
                        memcpy(wsPtr[i].vec, convBiasParamPtr, wsPtr[i].bytes_of_vec);
                    }
                } else {
                    wsPtr[i].bytes_of_vec = 0;
                    wsPtr[i].vec = nullptr;
                }
            } else if (weightOpType == "Gemm") {
                // attention: fc op weight bias order is different from conv op
                const onnx::TensorProto& fcWeightTp = weights[weightNode.input(1)];
                const onnx::TensorProto& fcBiasTp = weights[weightNode.input(2)];
                int fcWeightNum = get_data_size_from_tensor_proto(fcWeightTp);
                int fcBiasNum = get_data_size_from_tensor_proto(fcBiasTp);
                float* fcWeightParamPtr = get_ptr_from_weight_obj(fcWeightTp);
                float* fcBiasParamPtr = get_ptr_from_weight_obj(fcBiasTp);
                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = fcWeightNum * sizeof(float);
                wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, fcWeightParamPtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].bytes_of_vec = fcBiasNum * sizeof(float);
                wsPtr[i].vec = (U8*)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, fcBiasParamPtr, wsPtr[i].bytes_of_vec);
            } else if(weightOpType == "BatchNormalization") {
                const onnx::TensorProto& scale = weights[weightNode.input(1)];
                const onnx::TensorProto& bias = weights[weightNode.input(2)];
                const onnx::TensorProto& mean = weights[weightNode.input(3)];
                const onnx::TensorProto& var = weights[weightNode.input(4)];

                float* meanPtr = get_ptr_from_weight_obj(mean);
                int bnMeanNum = get_data_size_from_tensor_proto(mean);
                float* varPtr = get_ptr_from_weight_obj(var);
                int bnVarNum = get_data_size_from_tensor_proto(var);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length()); 
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = bnMeanNum * sizeof(float);
                wsPtr[i].bytes_of_vec = bnVarNum * sizeof(float);

                wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, meanPtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = (U8*)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, varPtr, wsPtr[i].bytes_of_vec);

                // for scale
                std::string scaleWeightOpName = "scale_" + weightOpName;
                i = i + 1;
                float* scalePtr = get_ptr_from_weight_obj(scale);
                int scaleWeightNum = get_data_size_from_tensor_proto(scale);
                float* biasPtr = get_ptr_from_weight_obj(bias);
                int scaleBiasNum = get_data_size_from_tensor_proto(bias);

                str_copy(wsPtr[i].op_name, scaleWeightOpName.c_str(), scaleWeightOpName.length()); 
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = scaleWeightNum * sizeof(float);
                wsPtr[i].bytes_of_vec = scaleBiasNum * sizeof(float);
                
                wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, scalePtr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = (U8*)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, biasPtr, wsPtr[i].bytes_of_vec);
            } else if(weightOpType == "Add") {
                const onnx::TensorProto& bias = weights[weightNode.input(1)];
                float* bias_ptr = get_ptr_from_weight_obj(bias);
                int bias_num = get_data_size_from_tensor_proto(bias);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = 0;
                wsPtr[i].bytes_of_vec = bias_num * sizeof(float);
                wsPtr[i].weight = nullptr;
                wsPtr[i].vec = (U8*)mt_new_storage(wsPtr[i].bytes_of_vec);
                memcpy(wsPtr[i].vec, bias_ptr, wsPtr[i].bytes_of_vec);
            } else if(weightOpType == "Mul") {
                const onnx::TensorProto& weight = weights[weightNode.input(1)];
                float* weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
                wsPtr[i].bytes_of_vec = 0;
                wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, weight_ptr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = nullptr;
            } else if (weightOpType == "Transpose") {
                const onnx::TensorProto& weight = weights[weightNode.input(0)];
                float* weight_ptr = get_ptr_from_weight_obj(weight);
                int weight_num = get_data_size_from_tensor_proto(weight);

                str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
                wsPtr[i].mdt = DT_F32;
                wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
                // For the time being, use bytes_of_vec to record the horizontal length of weight
                wsPtr[i].bytes_of_vec = weight.dims(0);
                wsPtr[i].weight = (U8*)mt_new_storage(wsPtr[i].bytes_of_weight);
                memcpy(wsPtr[i].weight, weight_ptr, wsPtr[i].bytes_of_weight);
                wsPtr[i].vec = nullptr;
            }
            weightOpIndexIndeed++;
        }
	    return ret;
    }

    ParameterSpec adapt_Reshape() override
    {
        ParameterSpec curPs;
        ReshapeParamSpec reshapePs;
        std::vector<int> reshapeInfo;
        if (node.input_size() == 1) {
            reshapeInfo = get_node_vector_ints_attribute_by_name(node, "shape");
        } else {
            reshapeInfo = get_reshapeInfo_from_tensorProto(weights[node.input(1)]);    // tp:weights[node.input(1)]
        }
        reshapePs.shape_size = reshapeInfo.size();
        memcpy(reshapePs.shape_dims, reshapeInfo.data(), reshapePs.shape_size * sizeof(I32));
        curPs.reshape_spec = reshapePs;
        return curPs; 
    }

    ParameterSpec adapt_Upsample() override
    {
        ParameterSpec curPs;
        UpsampleParamSpec upsamplePs;
        std::string unsampleMode = get_node_str_attribute_by_name(node, "mode", "linear");
        str_copy(upsamplePs.upsample_mode, unsampleMode.c_str(), unsampleMode.length());

        // Get scales from Constant
        const onnx::TensorProto& scales = weights[node.input(1)];
        CHECK_REQUIREMENT(scales.dims(0) == 4);
        F32 *value = nullptr;
        if (scales.has_raw_data()) {
            const std::string& rawData = scales.raw_data();
            value = (F32*)(rawData.data());
        } else if (scales.data_type() == 1) {
            value = (F32*)(scales.float_data().data());
        } else {
            std::cerr << "[ERROR] Upsample cannot extract scales from Constant\n";
            CHECK_STATUS(NOT_SUPPORTED);
        }
        memcpy(upsamplePs.scale, value, 4 * bytesOf(DT_F32));
        curPs.upsample_spec = upsamplePs;
        return curPs;    
    }

    ParameterSpec adapt_Transpose() override
    {
        ParameterSpec curPs;
        TransposeParamSpec transposePs;
        std::vector<int> transpose_info = get_node_vector_ints_attribute_by_name(node, "perm");
        transposePs.trans_size = transpose_info.size();
        memcpy(transposePs.trans_dims, transpose_info.data(), transposePs.trans_size * sizeof(U32));
        curPs.transpose_spec = transposePs; 
        return curPs;     
    }

    ParameterSpec adapt_Clip() override
    {
        ParameterSpec curPs;
        ClipParamSpec clipParam;
        if (op == "Max") {
            clipParam.min = 0;
            clipParam.max = UNI_F16_MAX;
        } else if (op == "Min") {
            clipParam.min = -UNI_F16_MAX;
            clipParam.max = 1;
        } else {    // op == "Clip"
            clipParam.min = get_node_float_attribute_by_name(node, "min", -UNI_F16_MAX);;
            clipParam.max = get_node_float_attribute_by_name(node, "max", UNI_F16_MAX);;
        }
        curPs.clip_spec = clipParam;
        return curPs;
    }

    ParameterSpec adapt_Conv() override
    {
	    weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> dilations = get_node_vector_ints_attribute_by_name(node, "dilations");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
        int group = get_node_single_int_attribute_by_name(node, "group", 1);

        const onnx::TensorProto& bias = weights[node.input(1)];
        ConvolutionParamSpec cps;
        cps.num_kernels = bias.dims(0);

        if (kernelShape.size() == 2) {
            DEBUG_info("Conv kernel shape is " << kernelShape[0] << " x " << kernelShape[1]);
            cps.kernel_size_h = kernelShape[0];
            if (kernelShape[1] >= 1) {
                cps.kernel_size_w = kernelShape[1];
            } else {
                cps.kernel_size_w = cps.kernel_size_h;
            }
        } else {
            std::cerr << "[ERROR] convolution: kernel_size unknown" << std::endl;
            exit(1);
        }

        if (dilations.size() == 2) {
            cps.dilatedRate_h = dilations[0];
            cps.dilatedRate_w = dilations[1];
        }
        else {
            std::cerr << "[ERROR] convolution: dilation unknown" << std::endl;
            exit(1);
        }

        if (strides.size() == 2) {
            cps.stride_h = strides[0];
            cps.stride_w = strides[1];
        }
        else {
            std::cerr << "[ERROR] convolution: stride unknown" << std::endl;
            exit(1);
        }
        if (cps.kernel_size_h == cps.kernel_size_w) {
            // Ghostnet model transformed from TF is known to have padding errors
            cps.padding_top = (cps.kernel_size_h - 1) / 2;
            cps.padding_bottom = (cps.kernel_size_h - 1) / 2;
            cps.padding_left = (cps.kernel_size_w - 1) / 2;
            cps.padding_right = (cps.kernel_size_w - 1) / 2;
        } else {
            CHECK_REQUIREMENT(pads.size() == 4);
            cps.padding_top = pads[0];
            cps.padding_bottom = pads[2];
            cps.padding_left = pads[1];
            cps.padding_right = pads[3];
        }
        
        cps.group = group;
        if (cps.group == 1) {
            if (cps.dilatedRate_h > 1 || cps.dilatedRate_w > 1) {
                cps.convolution_type = Convolution_Dilation;
            } else {
                cps.convolution_type = Convolution_Pointwise;
            }
        } else {
            cps.convolution_type = Convolution_Depthwise;
        }

        cps.dw_activation_type = ACTIVATION_NULL;
        cps.pw_activation_type = ACTIVATION_NULL;
        curPs.conv_spec = cps;
        return curPs;    
    }

    ParameterSpec adapt_Pooling() override
    {
        ParameterSpec curPs;
        PoolingParamSpec pps;   
        std::string autoPad = get_node_str_attribute_by_name(node, "auto_pad"); // deprecated
        std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
        std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
        std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");

        if (op == "AveragePool" || op == "ReduceMean"  || op == "GlobalAveragePool") {
            pps.mode = POOLING_MEAN;
        } else {
            pps.mode = POOLING_MAX;
        }

        if (autoPad == "SAME_UPPER") {
            pps.rm = CEIL;
        } else {
            pps.rm = FLOOR;
        }

        if (kernelShape.size() == 2) {
            pps.kernel_size_h = kernelShape[0];
            pps.kernel_size_w = kernelShape[1];
        } else {
            pps.kernel_size_h = 0;
            pps.kernel_size_w = 0;
            std::cout << "[Info] pooling: kernel_size unknown. This could be global pooling." << std::endl;
        }

        if (strides.size() == 2) {
            pps.stride_h = strides[0];
            pps.stride_w = strides[1];
        } else {
            pps.stride_h = 0;
            pps.stride_w = 0;
            std::cout << "[Info] pooling: stride unknown. This could be global pooling." << std::endl;
        }

        if (pads.size() == 4) {
            pps.padding_top = pads[0];
            pps.padding_bottom = pads[2];
            pps.padding_left = pads[1];
            pps.padding_right = pads[3];
        } else {
            pps.padding_top = 0;
            pps.padding_bottom = 0;
            pps.padding_left = 0;
            pps.padding_right = 0;
        }  
        curPs.pooling_spec = pps;
        return curPs;  
    }

    ParameterSpec adapt_Flatten() override {
        ParameterSpec curPs;
        FlattenParamSpec flattenPs;
        flattenPs.axis = get_node_single_int_attribute_by_name(node, "axis", 1);
        curPs.flatten_spec = flattenPs;
        return curPs;
    }

    ParameterSpec adapt_MatMul() override
    {
        ParameterSpec curPs;
        MatMulParamSpec matmulPs;
        matmulPs.transpose_a = false;
        matmulPs.transpose_b = false;
        curPs.matmul_spec = matmulPs;
        return curPs;
    }

    ParameterSpec adapt_Fc() override
    {
	    weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        FullyConnectedParamSpec fcParamSpec;
        fcParamSpec.num_outputs = -1;
        float alpha = get_node_float_attribute_by_name(node, "alpha", 1.f);
        float beta = get_node_float_attribute_by_name(node, "beta", 1.f);
        int transA = get_node_single_int_attribute_by_name(node, "transA", 0);
        int transB = get_node_single_int_attribute_by_name(node, "transB", 0);

        if (alpha == 1.f && beta == 1.f) {
            if (transA ==0 && transB == 1) {
                const onnx::TensorProto& C = weights[node.input(2)];
                int num_output = get_data_size_from_tensor_proto(C);  
                fcParamSpec.num_outputs = num_output;
            }
        }else{
            std::cerr << "[ERROR] fc: num_output unknown" << std::endl;
            exit(1);
        }
        curPs.fc_spec = fcParamSpec;
        return curPs;
    }

    ParameterSpec adapt_BatchNorm() override
    {
	    weightOpIndexLists.push_back(nodeIndex);
        ParameterSpec curPs;
        float epsilon = get_node_float_attribute_by_name(node, "epsilon", 1e-5f);
        BatchNormParamSpec bnPs;
        bnPs.eps = epsilon; 
        curPs.bn_spec = bnPs;
        return curPs;     
    }

    ParameterSpec adapt_Eltwise() override
    {
        ParameterSpec curPs;
        EltwiseParamSpec eps;
        if (op == "Add") {
            eps.elt_mode = ELTWISE_SUM;
            EltwiseSumSpec elt_sum_spec;
            elt_sum_spec.coeff_size = 2;
            F32* f_ptr = (F32*)mt_new_storage(elt_sum_spec.coeff_size * sizeof(float));
            for (I32 j = 0; j < elt_sum_spec.coeff_size; j++) {
                f_ptr[j] = 1.0;
            }
            elt_sum_spec.coeff_values = f_ptr;
            eps.elt_sum_spec = elt_sum_spec;
        } else if (op == "Mul") {
            eps.elt_mode = ELTWISE_PROD;   
        }
        curPs.eltwise_spec = eps;
        return curPs;
    }

    void handle_Constant()
    {
        for (int i = 0; i < node.attribute_size(); i++) {
            const onnx::AttributeProto& attribute = node.attribute(i);
            if (attribute.name() == "value") {
                CHECK_REQUIREMENT(4 == attribute.type());
                const onnx::TensorProto& tp = attribute.t();
                weights[node.output(0)] = tp;
                break;
            }
        }           
    }

    ParameterSpec adapt_Pad() override
    {
        ParameterSpec curPs;
        PadParamSpec padPs;
        U32 top, bottom, left, right;
        PadMode padMode;
        std::string padModeStr = get_node_str_attribute_by_name(node, "mode");
        std::vector<int> padVec = get_node_vector_ints_attribute_by_name(node, "pads");
        F32 padValue = get_node_float_attribute_by_name(node, "value", 0.f);

        if (padModeStr == "constant") {
            padMode = Pad_Constant;
        } else if (padModeStr == "edge") {
            padMode = Pad_Edge;
        } else {    // padModeStr == "reflect"
            padMode = Pad_Reflect;
        }

        U32 padSize = padVec.size();
        if (padSize == 8) {    // NCHW
            top = padVec[2];
            bottom = padVec[6];
            left = padVec[3];
            right = padVec[7];
        } else if (padSize == 6) {      // CHW
            top = padVec[1];
            bottom = padVec[4];
            left = padVec[2];
            right = padVec[5];
        } else {
            top = padVec[0];
            bottom = padVec[2];
            left = padVec[1];
            right = padVec[3];
        }
        padPs.top = top;
        padPs.bottom = bottom;
        padPs.left = left;
        padPs.right = right;
        padPs.constant_value = padValue;
        padPs.pad_mode = padMode;
        curPs.pad_spec = padPs;
        return curPs;        
    }

    ParameterSpec adapt_Gather() override
    {
        ParameterSpec curPs;
        GatherParamSpec gps;
        int gatherAxis = get_node_single_int_attribute_by_name(node, "axis", 0);
        gps.gather_axis = gatherAxis;
        curPs.gather_spec = gps;    
        return curPs;  
    }

    ParameterSpec adapt_Squeeze() override {
        ParameterSpec curPs;
        SqueezeParamSpec squeezePs;
        std::vector<int> squeezeAxes = get_node_vector_ints_attribute_by_name(node, "axes");
        squeezePs.axes_num = squeezeAxes.size();
        squeezePs.axis = -1 * squeezeAxes.size();
        for (int squeeze_i = 0; squeeze_i < (int)squeezeAxes.size(); squeeze_i++) {
            squeezePs.squeeze_axes[squeeze_i] = squeezeAxes[squeeze_i];
        }
        curPs.squeeze_spec = squeezePs;
        return curPs;
    }

    ParameterSpec adapt_Unsqueeze() override {
        ParameterSpec curPs;
        UnsqueezeParamSpec unsqueezePs;
        std::vector<int> unsqueezeAxes = get_node_vector_ints_attribute_by_name(node, "axes");
        unsqueezePs.axes_num = unsqueezeAxes.size();
        unsqueezePs.axis = -1 * unsqueezeAxes.size();
        for (int unsqueeze_i = 0; unsqueeze_i < (int)unsqueezeAxes.size(); unsqueeze_i++) {
            unsqueezePs.unsqueeze_axes[unsqueeze_i] = unsqueezeAxes[unsqueeze_i];
        }
        curPs.unsqueeze_spec = unsqueezePs;
        return curPs;    
    }

    ParameterSpec adapt_Cast() override
    {
        ParameterSpec curPs;
        CastParamSpec castPs;
        int castTo = get_node_single_int_attribute_by_name(node, "to", 0);
        castPs.cast_to = castTo;
        curPs.cast_spec = castPs;
        return curPs;        
    }

private:
    std::string op;    // op type
    std::string modelName;
    int removePreprocessOpNum;
    TensorDesc inputDesc;
    onnx::ModelProto onnxModel;
    onnx::GraphProto onnxGraph;
    onnx::NodeProto node;
    std::map<std::string, onnx::TensorProto> weights;
    int nodeIndex;
    std::vector<int> weightOpIndexLists;
    int opFinalInputNum;
};
#endif
