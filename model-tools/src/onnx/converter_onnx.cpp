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
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "onnx.pb.h"

#include "type.h"
#include "converter.h"
#include "model_serialize_deserialize.hpp"
#include "model_tools.h"

#ifdef _USE_ONNX_MODEL

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

// map onnx op to bolt op type
OperatorType onnxType_to_boltType(std::string inputType) {
    if (inputType == "Conv") {
        return OT_Conv;
    } else if (inputType == "BatchNormalization") {
        return OT_BatchNorm;
    } else if (inputType == "Sum" || inputType == "Add" || inputType == "Mul") {
        return OT_Eltwise;
    } else if (inputType == "Gemm") {
        return OT_FC;
    } else if (inputType == "AveragePool" || inputType == "MaxPool" || inputType == "ReduceMean") {
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
    } else if (inputType == "Gather") {    // 20191119
        return OT_Gather;
    } else if (inputType == "Unsqueeze") {
        return OT_Unsqueeze;
    } else if (inputType == "Upsample") {
        return OT_Upsample;
    } else if (inputType == "Cast") {
        return OT_Cast;
    }
    else {
        return OT_None;
    }
}

std::vector<int> get_node_vector_ints_attribute_by_name(const onnx::NodeProto& node, const char* key) {
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

std::vector<int> get_reshapeInfo_from_tensorProto(const onnx::TensorProto& tp) {
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


EE mt_load_onnx(CI8* dir, CI8* mfn, ModelSpec* ms, int removePreprocessOpNum, TensorDesc inputDesc)
{
    const F16 F16_MAX = 65504;

    std::string onnxSuffix = ".onnx";
    std::string onnxPath = dir + std::string(mfn) + onnxSuffix;

    onnx::ModelProto onnxModel;

    std::string modelFileName = mfn;
    str_copy(ms->model_name, modelFileName.c_str(), modelFileName.length());
    ms->model_name[NAME_LEN - 1] = '\0';
    ms->dt = DT_F32;

    EE ret = read_from_onnx_file(onnxPath.c_str(), &onnxModel);
    if (ret != SUCCESS) {
        return UNKNOWN;
    }

    const onnx::GraphProto& onnxGraph = onnxModel.graph();
    int onnxNodeCount = onnxGraph.node_size();

    ms->num_inputs = 1;
    ms->input_names = (I8**)mt_malloc(ms->num_inputs * sizeof(I8*));
    const onnx::NodeProto& theFirstNode = onnxGraph.node(removePreprocessOpNum);    // need to be flexible
    std::string modelInputName = theFirstNode.input(0);
    ms->input_names[0] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
    str_copy(ms->input_names[0], modelInputName.c_str(), modelInputName.length());
    ms->input_dims  = (TensorDesc*)mt_malloc(sizeof(TensorDesc) * ms->num_inputs);
    ms->input_dims[0] = inputDesc;

    ms->num_outputs = 1;
    ms->output_names = (I8**)mt_malloc(ms->num_outputs * sizeof(I8*));
    const onnx::NodeProto& the_last_node = onnxGraph.node(onnxNodeCount - 1);
    std::string modelOutputName = the_last_node.output(0);
    ms->output_names[0] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
    str_copy(ms->output_names[0], modelOutputName.c_str(), modelOutputName.length());

    std::map<std::string, onnx::TensorProto> weights;
    for (int i=0; i < onnxGraph.initializer_size(); i++) {
        const onnx::TensorProto& initializer = onnxGraph.initializer(i);
        weights[initializer.name()] = initializer;
    }

    // special for bn
    int bnOpNum = 0;
    for (int i=0; i < onnxNodeCount; i++) {
        const onnx::NodeProto& node = onnxGraph.node(i);
        const std::string& op = node.op_type();
        if (op == "BatchNormalization") {
            bnOpNum++;
        }
    }

    ms->num_operator_specs = onnxNodeCount + bnOpNum - removePreprocessOpNum;   // appending space for scale op
    OperatorSpec* opsPtr = (OperatorSpec*)mt_malloc(sizeof(OperatorSpec) * ms->num_operator_specs);
    ms->ops = opsPtr;

    std::vector<int> weightOpIndexLists;    // conv+fc+bn+scale
    int nodeIndex = removePreprocessOpNum;
    for (int i = removePreprocessOpNum; i < onnxNodeCount + bnOpNum; i++) {
        int opIndex = i - removePreprocessOpNum;

        const onnx::NodeProto& node = onnxGraph.node(nodeIndex);
        const std::string& op = node.op_type();
        std::string opName = node.name();
        if (opName.empty()) {
            opName = node.output(0);
        }
        int opInputNum = (int)node.input_size();    
        int opFinalInputNum = opInputNum;
        std::vector<std::string> inputNames;
        std::vector<std::string> op_weight_objs;
        for (int j = 0; j < opInputNum; j++) {
            const std::string& input_name = node.input(j);
            if (weights.find(input_name) != weights.end()) {
                opFinalInputNum--;
                op_weight_objs.push_back(input_name);
            } else {
                inputNames.push_back(input_name);
            }
        }
        int opOutputNum = (int)node.output_size();   
        std::vector<std::string> outputNames;
        for (int j = 0; j < opOutputNum; j++) {
            const std::string& output_name = node.output(j);
            outputNames.push_back(output_name);
        }

        str_copy(opsPtr[opIndex].name, opName.c_str(), opName.length());
        OperatorType opType = onnxType_to_boltType(op);
        opsPtr[opIndex].type = opType;
        opsPtr[opIndex].num_inputs = opFinalInputNum;
        opsPtr[opIndex].input_tensors_name = (I8**)mt_malloc(opsPtr[opIndex].num_inputs * sizeof(I8 *));
        for (U32 j = 0; j < opsPtr[opIndex].num_inputs; j++) {
            opsPtr[opIndex].input_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(opsPtr[opIndex].input_tensors_name[j], inputNames[j].c_str(), inputNames[j].length());
        }
        opsPtr[opIndex].num_outputs = opOutputNum;
        opsPtr[opIndex].output_tensors_name = (I8**)mt_malloc(opsPtr[opIndex].num_outputs * sizeof(I8 *));
        for (U32 j = 0; j < opsPtr[opIndex].num_outputs; j++) {
            opsPtr[opIndex].output_tensors_name[j] = (I8*)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(opsPtr[opIndex].output_tensors_name[j], outputNames[j].c_str(), outputNames[j].length());
        }

        if (op == "Reshape") {
            ReshapeParamSpec reshapePs;
            std::vector<int> reshapeInfo;
            if (node.input_size() == 1) {
                reshapeInfo = get_node_vector_ints_attribute_by_name(node, "shape");
            }else{
                reshapeInfo = get_reshapeInfo_from_tensorProto(weights[node.input(1)]);
            }
            reshapePs.shape_size = reshapeInfo.size();
            memcpy(reshapePs.shape_dims, reshapeInfo.data(), reshapePs.shape_size * sizeof(I32));
            opsPtr[opIndex].ps.reshape_spec = reshapePs;
        } else if (op == "Transpose") {
            TransposeParamSpec transposePs;
            std::vector<int> transpose_info = get_node_vector_ints_attribute_by_name(node, "perm");
            transposePs.trans_size = transpose_info.size();
            memcpy(transposePs.trans_dims, transpose_info.data(), transposePs.trans_size * sizeof(U32));
            opsPtr[opIndex].ps.transpose_spec = transposePs;
        } else if (op == "Max" || op == "Min" || op == "Clip") {
            ClipParamSpec clipParam;
            if (op == "Max") {
                clipParam.min = 0;
                clipParam.max = F16_MAX;
            } else if (op == "Min") {
                clipParam.min = -F16_MAX;
                clipParam.max = 1;
            } else if (op == "Clip") {
                clipParam.min = get_node_float_attribute_by_name(node, "min", -F16_MAX);;
                clipParam.max = get_node_float_attribute_by_name(node, "max", F16_MAX);;
            }
            opsPtr[opIndex].ps.clip_spec = clipParam;
        } else if (op == "Conv") {
            weightOpIndexLists.push_back(nodeIndex);
            std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
            std::vector<int> dilations = get_node_vector_ints_attribute_by_name(node, "dilations");
            std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
            std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");
            int group = get_node_single_int_attribute_by_name(node, "group", 1);

            const onnx::TensorProto& bias = weights[node.input(1)];
            ConvolutionParamSpec cps;
            cps.num_kernels = bias.dims(0);

            if (kernelShape.size() > 0)
                cps.kernel_size = kernelShape[0];
            else {
                cps.kernel_size = 1;
                std::cerr << "[ERROR] convolution: kernel_size unknown" << std::endl;
            }

            if (dilations.size() > 0)
                cps.dilation = dilations[0];
            else {
                cps.dilation = 1;
                std::cerr << "[ERROR] convolution: dilation unknown" << std::endl;
            }

            if (strides.size() > 0)
                cps.stride = strides[0];
            else {
                cps.stride = 1;
                std::cerr << "[ERROR] convolution: stride unknown" << std::endl;
            }
            cps.padding = (cps.kernel_size - 1) / 2;
            cps.group = group;
            if (cps.group == 1) {
                if (cps.dilation > 1) {
                    cps.convolution_type = Convolution_Dilation;
                } else {
                    cps.convolution_type = Convolution_Pointwise;
                }
            } else {
                cps.convolution_type = Convolution_Depthwise;
            }

            cps.dw_activation_type = ACTIVATION_NULL;
            cps.pw_activation_type = ACTIVATION_NULL;
            opsPtr[opIndex].ps.conv_param_spec = cps;
        } else if (op == "AveragePool" || op == "MaxPool") {
            PoolingParamSpec pps;   
            std::string autoPad = get_node_str_attribute_by_name(node, "auto_pad"); // deprecated
            std::vector<int> kernelShape = get_node_vector_ints_attribute_by_name(node, "kernel_shape");
            std::vector<int> strides = get_node_vector_ints_attribute_by_name(node, "strides");
            std::vector<int> pads = get_node_vector_ints_attribute_by_name(node, "pads");

            if (op == "AveragePool") {
                pps.mode = Mean;
            } else {
                pps.mode = Max;
            }

            if (autoPad == "SAME_UPPER") {
                pps.rm = CEIL;
            } else {
                pps.rm = FLOOR;
            }

            if (kernelShape.size() > 0) {
                pps.kernel_size = kernelShape[0];
            } else {
                pps.kernel_size = 1;
                std::cerr << "[ERROR] pooling: kernel_size unknown" << std::endl;
            }

            if (strides.size() > 0) {
                pps.stride = strides[0];
            } else {
                pps.stride = 1;
                std::cerr << "[ERROR] pooling: stride unknown" << std::endl;
            }

            if (pads.size() > 0) {
                pps.padding = pads[0];
            } else {
                pps.padding = 0;
            }
            opsPtr[opIndex].ps.pooling_param_spec = pps;
        } else if (op == "ReduceMean") { 
            PoolingParamSpec pps; 
            pps.kernel_size = 0;
            pps.mode = Mean;
            opsPtr[opIndex].ps.pooling_param_spec = pps;
        } else if (op == "Gemm") { 
            weightOpIndexLists.push_back(nodeIndex);
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
            }
            opsPtr[opIndex].ps.ip_param_spec = fcParamSpec;
        } else if (op == "BatchNormalization") {
            weightOpIndexLists.push_back(nodeIndex);
            float epsilon = get_node_float_attribute_by_name(node, "epsilon", 1e-5f);
            BatchNormParamSpec bnPs;
            bnPs.eps = epsilon;
            opsPtr[opIndex].ps.bn_param_spec = bnPs;

            std::string scaleInputName = outputNames[0];
            std::string scaleOpName = "scale_" + opName;
            i++;
            opIndex = i - removePreprocessOpNum;
            str_copy(opsPtr[opIndex].name, scaleOpName.c_str(), scaleOpName.length());
            opsPtr[opIndex].type = OT_Scale;
            opsPtr[opIndex].num_inputs = 1;
            opsPtr[opIndex].input_tensors_name = (I8 **)mt_malloc(sizeof(I8 *));
            opsPtr[opIndex].input_tensors_name[0] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(opsPtr[opIndex].input_tensors_name[0], scaleInputName.c_str(), scaleInputName.length());
            opsPtr[opIndex].num_outputs = 1;
            opsPtr[opIndex].output_tensors_name = (I8 **)mt_malloc(sizeof(I8 *));
            opsPtr[opIndex].output_tensors_name[0] = (I8 *)mt_malloc(NAME_LEN * sizeof(I8));
            str_copy(opsPtr[opIndex].output_tensors_name[0], scaleInputName.c_str(), scaleInputName.length());
        } else if (op == "Add") {
            if (opFinalInputNum == 1) {
                weightOpIndexLists.push_back(nodeIndex);
                opsPtr[opIndex].type = OT_Scale;
            }
            else {
                EltwiseParamSpec eps;
                eps.elt_mode = ELTWISE_SUM;
                EltwiseSumSpec elt_sum_spec;
                elt_sum_spec.coeff_size = 2;
                F32* f_ptr = (F32*)malloc(elt_sum_spec.coeff_size * sizeof(float));
                for (I32 j = 0; j < elt_sum_spec.coeff_size; j++)
                    f_ptr[j] = 1.0;
                elt_sum_spec.coeff_values = f_ptr;
                eps.elt_sum_spec = elt_sum_spec;
                opsPtr[opIndex].ps.eltwise_param_spec = eps;
            }
        } else if (op == "Mul") {
            if (opFinalInputNum == 1) {
                weightOpIndexLists.push_back(nodeIndex);
                opsPtr[opIndex].type = OT_Scale;
            }
            else {
                EltwiseParamSpec eps;
                eps.elt_mode = ELTWISE_PROD;
                opsPtr[opIndex].ps.eltwise_param_spec = eps;
            }
        } else if (op == "Pad") {
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
            } else if (padModeStr == "reflect") {
                padMode = Pad_Reflect;
            } else {
                return NOT_SUPPORTED;
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

            opsPtr[opIndex].ps.pad_spec = padPs;
        } else if (op == "Gather") {    // 20191119
            GatherParamSpec gps;
            int gatherAxis = get_node_single_int_attribute_by_name(node, "axis", 0);
            gps.gather_axis = gatherAxis;
            opsPtr[opIndex].ps.gather_spec = gps;
        } else if (op == "Unsqueeze") {
            UnsqueezeParamSpec unsqueezePs;
            std::vector<int> unsqueezeAxes = get_node_vector_ints_attribute_by_name(node, "axes");
            unsqueezePs.axes_num = unsqueezeAxes.size();
            for (int unsqueeze_i = 0; unsqueeze_i < (int)unsqueezeAxes.size(); unsqueeze_i++) {
                unsqueezePs.unsqueeze_axes[unsqueeze_i] = unsqueezeAxes[unsqueeze_i];
            }
            opsPtr[opIndex].ps.unsqueeze_spec = unsqueezePs;
        } else if (op == "Upsample") {
            UpsampleParamSpec upsamplePs;
            std::string unsampleMode = get_node_str_attribute_by_name(node, "mode", "linear");
            str_copy(upsamplePs.upsample_mode, unsampleMode.c_str(), unsampleMode.length());
            opsPtr[opIndex].ps.upsample_spec = upsamplePs;
        } else if (op == "Cast") {
            CastParamSpec castPs;
            int castTo = get_node_single_int_attribute_by_name(node, "to", 0);
            castPs.cast_to = castTo;
            opsPtr[opIndex].ps.cast_spec = castPs;
        }

        nodeIndex++;
    }

    ms->num_weight_specs = weightOpIndexLists.size() + bnOpNum;
    WeightSpec* wsPtr = (WeightSpec*)mt_malloc(sizeof(WeightSpec) * ms->num_weight_specs);
    ms->ws = wsPtr;
    int weightOpIndexIndeed = 0;
    for (U32 i = 0; i < weightOpIndexLists.size() + bnOpNum; i++) {
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
                wsPtr[i].mdt = DT_DOREFA;
            } else if (isXNOR == 1) {
                wsPtr[i].mdt = DT_XNOR;
            } else {
                wsPtr[i].mdt = DT_F32; // Assume weights will not all be 1.0
            }
            //wsPtr[i].mdt = DT_F32;
            wsPtr[i].bytes_of_weight = convWeightNum * sizeof(float); // Please do not change to bytesOf(mdt)
            wsPtr[i].weight = (U8*)mt_malloc(wsPtr[i].bytes_of_weight);
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
                wsPtr[i].vec = (U8*)mt_malloc(wsPtr[i].bytes_of_vec);
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
            wsPtr[i].weight = (U8*)malloc(wsPtr[i].bytes_of_weight);
            memcpy(wsPtr[i].weight, fcWeightParamPtr, wsPtr[i].bytes_of_weight);
            wsPtr[i].bytes_of_vec = fcBiasNum * sizeof(float);
            wsPtr[i].vec = (U8*)malloc(wsPtr[i].bytes_of_vec);
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

            wsPtr[i].weight = (U8*)mt_malloc(wsPtr[i].bytes_of_weight);
            memcpy(wsPtr[i].weight, meanPtr, wsPtr[i].bytes_of_weight);
            wsPtr[i].vec = (U8*)mt_malloc(wsPtr[i].bytes_of_vec);
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
            
            wsPtr[i].weight = (U8*)mt_malloc(wsPtr[i].bytes_of_weight);
            memcpy(wsPtr[i].weight, scalePtr, wsPtr[i].bytes_of_weight);
            wsPtr[i].vec = (U8*)mt_malloc(wsPtr[i].bytes_of_vec);
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
            wsPtr[i].vec = (U8*)mt_malloc(wsPtr[i].bytes_of_vec);
            memcpy(wsPtr[i].vec, bias_ptr, wsPtr[i].bytes_of_vec);
        } else if(weightOpType == "Mul") {
            const onnx::TensorProto& weight = weights[weightNode.input(1)];
            float* weight_ptr = get_ptr_from_weight_obj(weight);
            int weight_num = get_data_size_from_tensor_proto(weight);

            str_copy(wsPtr[i].op_name, weightOpName.c_str(), weightOpName.length());
            wsPtr[i].mdt = DT_F32;
            wsPtr[i].bytes_of_weight = weight_num * sizeof(float);
            wsPtr[i].bytes_of_vec = 0;
            wsPtr[i].weight = (U8*)mt_malloc(wsPtr[i].bytes_of_weight);
            memcpy(wsPtr[i].weight, weight_ptr, wsPtr[i].bytes_of_weight);
            wsPtr[i].vec = nullptr;
        }
        weightOpIndexIndeed++;
    }

    return SUCCESS;
}
#endif
