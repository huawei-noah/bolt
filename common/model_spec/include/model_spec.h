// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _H_MODEL_SPEC
#define _H_MODEL_SPEC

#include "parameter_spec.h"
#ifdef _WIN32
#include <windows.h>
#include <pthread.h>
#endif

static const I32 sg_boltVersion = 20220831;
static const I32 sg_magicNumber = 1141119;

#pragma pack(8)
typedef struct OperatorSpec {
    I8 name[NAME_LEN];
    OperatorType type = OT_None;
    U32 num_inputs = 0;
    I8 **input_tensors_name = NULL;
    U32 num_outputs = 0;
    I8 **output_tensors_name = NULL;
    I32 *tensor_positions = NULL;
    U32 num_quant_feature = 0;
    QuantSpec *feature_scale = NULL;
    ParameterSpec ps;
} OperatorSpec;

typedef struct WeightSpec {
    I8 op_name[NAME_LEN];
    DataType mdt = DT_U8;
    U32 bytes_of_weight = 0;
    U8 *weight = NULL;
    U32 bytes_of_vec = 0;
    U8 *vec = NULL;
    // Merged FC may have multiple weight scales
    U32 num_quant_scale = 0;
    QuantSpec *weight_scale = NULL;
} WeightSpec;

typedef struct OperatorRelationshipMapEntry {
    I8 op[NAME_LEN];
    U32 num_inputs = 0;
    I8 **input_op_names = NULL;
    U32 num_outputs = 0;
    I8 **output_op_names = NULL;
} OperatorRelationshipMapEntry;

typedef struct ModelFileDescriptor {
    U8 *content = NULL;
    size_t length = 0;
    bool stream_mode = false;
#ifdef _WIN32
    HANDLE file;
    HANDLE map;
    pthread_t thread;
#else
    I32 file;
#endif
} ModelFileDescriptor;

typedef struct ModelSpec{
    I32 version;
    I32 magic_number;

    I8 model_name[NAME_LEN];
    DataType dt;

    I32 num_inputs = 0;
    I8 **input_names = NULL;
    TensorDesc *input_dims = NULL;

    I32 num_outputs = 0;
    I8 **output_names = NULL;

    I32 num_operator_specs = 0;
    OperatorSpec *ops = NULL;

    I32 num_weight_specs = 0;
    WeightSpec *ws = NULL;

    I32 num_op_tensor_entries = 0;
    OperatorRelationshipMapEntry *op_relationship_entries = NULL;

    ModelFileDescriptor *file = NULL;
} ModelSpec;
#pragma pack()

EE mt_create_model(ModelSpec *spec);
EE serialize_model_to_file(const ModelSpec *spec, const char *filePath);
EE deserialize_model_from_file(
    const char *filePath, ModelSpec *spec, DataType targetDt, bool useFileStream = false);
EE mt_destroy_model(ModelSpec *spec);

#include "model_print.h"
#endif
