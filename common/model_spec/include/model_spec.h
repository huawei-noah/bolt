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

static const int sg_boltVersion = 20201120;
static const int sg_magicNumber = 1141119;

#pragma pack(8)
typedef struct {
    I8 name[NAME_LEN];
    OperatorType type;
    U32 num_inputs;
    I8 **input_tensors_name;
    U32 num_outputs;
    I8 **output_tensors_name;
    I32 *tensor_positions;
    U32 num_quant_feature;
    QuantSpec *feature_scale;
    ParameterSpec ps;
} OperatorSpec;

typedef struct WeightSpec {
    I8 op_name[NAME_LEN];
    DataType mdt = DT_U8;
    U32 bytes_of_weight = 0;
    U8 *weight;
    U32 bytes_of_vec = 0;
    U8 *vec;
    // Merged FC may have multiple weight scales
    U32 num_quant_scale;
    QuantSpec *weight_scale;
} WeightSpec;

typedef struct {
    I8 op[NAME_LEN];
    U32 num_inputs;
    I8 **input_op_names;
    U32 num_outputs;
    I8 **output_op_names;
} OperatorRelationshipMapEntry;

typedef struct {
    I32 fd;
    I8 *bytes;
    U32 fileLength;
    bool useFileStream;
} ModelFileDescriptor;

typedef struct {
    I32 version;
    I32 magic_number;

    I8 model_name[NAME_LEN];
    DataType dt;

    I32 num_inputs;
    I8 **input_names;
    TensorDesc *input_dims;

    I32 num_outputs;
    I8 **output_names;

    I32 num_operator_specs;
    OperatorSpec *ops;

    I32 num_weight_specs;
    WeightSpec *ws;

    I32 num_op_tensor_entries;
    OperatorRelationshipMapEntry *op_relationship_entries;

    ModelFileDescriptor *mfd;
} ModelSpec;
#pragma pack()

#define outOfFileMapRange(addr, mfd)                                  \
    ((mfd == nullptr) || (uintptr_t(addr) < uintptr_t(mfd->bytes)) || \
        (uintptr_t(addr) >= uintptr_t(mfd->bytes + mfd->fileLength)))

EE mt_create_model(ModelSpec *md);
EE serialize_model_to_file(const ModelSpec *spec, const char *fn);
EE deserialize_model_from_file(const char *fn, ModelSpec *spec, bool useFileStream = false);
EE mt_destroy_model(ModelSpec *md);

#include "model_print.h"
#endif
