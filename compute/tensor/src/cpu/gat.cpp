// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"
#ifdef _DEBUG
#include <set>
#endif

void preprocess(TensorDesc node_feature_desc,
    TensorDesc node_desc,
    void *node_features0,
    void *nodes0,
    void *node_features1,
    void *nodes1,
    void *edge_feature,
    ActivationParamSpec activationDesc,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    GatherParamSpec p;
    p.axis = 0;
    p.element_level = false;
    p.batch_dims = 0;

    void *out0 = tmp;
    CHECK_STATUS(gather_cpu(
        node_feature_desc, node_features0, node_desc, nodes0, p, nullptr, outputDesc, out0));

    void *out1 = (U8 *)out0 + tensorNumBytes(outputDesc);
    CHECK_STATUS(gather_cpu(
        node_feature_desc, node_features1, node_desc, nodes1, p, nullptr, outputDesc, out1));

    std::vector<TensorDesc> inputDescs = {outputDesc, outputDesc, outputDesc};
    std::vector<void *> inputs = {out0, out1, edge_feature};
    EltwiseParamSpec eltwiseDesc;
    eltwiseDesc.mode = ELTWISE_SUM;
    eltwiseDesc.activation_type = ACTIVATION_NULL;
    CHECK_STATUS(eltwise_cpu(inputDescs, inputs, eltwiseDesc, 0, nullptr, outputDesc, output, arch));

    CHECK_STATUS(
        activation_cpu(outputDesc, output, activationDesc, outputDesc, output, nullptr, arch));
}

template <typename T>
void neighborhood_aware_softmax_yun(TensorDesc inputDesc,
    T *input,
    const int *nodes0,
    const int *nodes1,
    void *tmp,
    int num_heads,
    int num_nodes,
    int num_edges,
    T *output,
    Arch arch)
{
    T *out0 = input;
    ActivationParamSpec activationDesc;
    activationDesc.mode = ACTIVATION_EXP;
    CHECK_STATUS(activation_cpu(inputDesc, input, activationDesc, inputDesc, out0, nullptr, arch));

#ifdef _DEBUG
    std::set<int> edge_set;
    for (int i = 0; i < num_nodes; i++) {
        int edge = nodes1[i] * num_nodes + nodes0[i];
        if (edge_set.find(edge) == edge_set.end()) {
            edge_set.insert(edge);
        } else {
            UNI_ERROR_LOG("there is duplicate edge(%d - %d)\n", nodes1[i], nodes0[i]);
        }
    }
#endif
    T *out1 = (T *)tmp;
    UNI_MEMSET(out1, 0, sizeof(T) * num_nodes * num_heads);
    for (int i = 0; i < num_edges; i++) {
        int node = nodes1[i];
        for (int j = 0; j < num_heads; j++) {
            out1[node * num_heads + j] += out0[i * num_heads + j];
        }
    }

    for (int i = 0; i < num_edges; i++) {
        int node = nodes1[i];
        UNI_MEMCPY(output + i * num_heads, out1 + node * num_heads, num_heads * sizeof(T));
    }

    std::vector<TensorDesc> inputDescs = {inputDesc, inputDesc};
    std::vector<void *> inputs = {out0, output};
    EltwiseParamSpec eltwiseDesc;
    eltwiseDesc.mode = ELTWISE_DIV;
    eltwiseDesc.activation_type = ACTIVATION_NULL;
    CHECK_STATUS(eltwise_cpu(inputDescs, inputs, eltwiseDesc, 0, nullptr, inputDesc, output, arch));
}

template <typename T>
void scatter_atten_score(const int *nodes0,
    const int *nodes1,
    const T *update,
    int num_heads,
    int num_nodes,
    int num_edges,
    T *out)
{
    UNI_MEMSET(out, 0, sizeof(T) * num_heads * num_nodes * num_nodes);
    for (int j = 0, k = 0; j < num_edges; j++) {
        int node0 = nodes0[j];
        int node1 = nodes1[j];
        for (int i = 0; i < num_heads; i++, k++) {
            int id = (i * num_nodes + node1) * num_nodes + node0;
            out[id] = update[k];
        }
    }
}

EE gat_cpu(TensorDesc node_feature_desc,
    TensorDesc node_desc,
    TensorDesc edge_feature_desc,
    void *node_features0,
    void *nodes0,
    void *node_features1,
    void *nodes1,
    void *edge_feature,
    GATParamSpec p,
    void *tmp,
    TensorDesc outputDesc,
    void *output,
    Arch arch)
{
    void *out0 = tmp;
    void *out1 = (U8 *)tmp + tensorNumBytes(edge_feature_desc);
    tmp = (U8 *)out1 + tensorNumBytes(edge_feature_desc);
    // tmpBytes = tensorNumBytes(edge_feature_desc) * 2
    preprocess(node_feature_desc, node_desc, node_features0, nodes0, node_features1, nodes1,
        edge_feature, p.activation_type, tmp, edge_feature_desc, out0, arch);

    int num_heads = p.num_heads;
    int num_nodes = node_feature_desc.dims[1];
    int num_edges = edge_feature_desc.dims[1];

    EE ret = SUCCESS;
    DataType dt = outputDesc.dt;
    switch (dt) {
#ifdef _USE_FP32
        case DT_F32: {
            // tmpBytes = sizeof(T) * num_nodes * num_heads
            neighborhood_aware_softmax_yun<F32>(edge_feature_desc, (F32 *)out0, (const int *)nodes0,
                (const int *)nodes1, tmp, num_heads, num_nodes, num_edges, (F32 *)out1, arch);
            scatter_atten_score<F32>((const int *)nodes0, (const int *)nodes1, (F32 *)out1,
                num_heads, num_nodes, num_edges, (F32 *)output);
            break;
        }
#endif
#ifdef _USE_FP16
        case DT_F16: {
            neighborhood_aware_softmax_yun<F16>(edge_feature_desc, (F16 *)out0, (const int *)nodes0,
                (const int *)nodes1, tmp, num_heads, num_nodes, num_edges, (F16 *)out1, arch);
            scatter_atten_score<F16>((const int *)nodes0, (const int *)nodes1, (F16 *)out1,
                num_heads, num_nodes, num_edges, (F16 *)output);
            break;
        }
#endif
        default:
            ret = NOT_SUPPORTED;
            break;
    }

    return ret;
}

EE gat_infer_output_size_cpu(TensorDesc node_feature_desc, GATParamSpec p, TensorDesc *outputDesc)
{
    int num_heads = p.num_heads;
    int num_nodes = node_feature_desc.dims[1];
    *outputDesc = tensor3df(node_feature_desc.dt, DF_NCHW, num_heads, num_nodes, num_nodes);
    return SUCCESS;
}

EE gat_infer_forward_tmp_bytes_cpu(
    TensorDesc node_feature_desc, TensorDesc edge_feature_desc, GATParamSpec p, U32 *bytes)
{
    if (bytes == nullptr) {
        CHECK_STATUS(NULL_POINTER);
    }
    int num_heads = p.num_heads;
    int num_nodes = node_feature_desc.dims[1];
    *bytes = tensorNumBytes(edge_feature_desc) * 4 +
        bytesOf(edge_feature_desc.dt) * num_nodes * num_heads;
    return SUCCESS;
}
