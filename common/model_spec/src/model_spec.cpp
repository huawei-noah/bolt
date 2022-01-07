// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "model_spec.h"

EE mt_create_model(ModelSpec *ms)
{
    if (nullptr == ms) {
        return NULL_POINTER;
    }

    ms->version = sg_boltVersion;
    ms->magic_number = sg_magicNumber;
    ms->input_names = nullptr;
    ms->num_inputs = 0;
    ms->input_dims = nullptr;
    ms->num_outputs = 0;
    ms->output_names = nullptr;
    ms->num_operator_specs = 0;
    ms->ops = nullptr;
    ms->num_weight_specs = 0;
    ms->ws = nullptr;
    ms->num_op_tensor_entries = 0;
    ms->op_relationship_entries = nullptr;
    ms->mfd = nullptr;

    return SUCCESS;
}

EE mt_destroy_model(ModelSpec *ms)
{
    if (nullptr == ms) {
        return NULL_POINTER;
    }

    if (nullptr != ms->input_names) {
        for (int i = 0; i < ms->num_inputs; i++) {
            if (nullptr != ms->input_names[i]) {
                delete ms->input_names[i];
            }
            ms->input_names[i] = nullptr;
        }
        delete ms->input_names;
        ms->input_names = nullptr;
    }

    if (nullptr != ms->input_dims) {
        delete ms->input_dims;
        ms->input_dims = nullptr;
    }

    if (nullptr != ms->output_names) {
        for (int i = 0; i < ms->num_outputs; i++) {
            if (nullptr != ms->output_names[i]) {
                delete ms->output_names[i];
            }
            ms->output_names[i] = nullptr;
        }
        delete ms->output_names;
        ms->output_names = nullptr;
    }

    if (nullptr != ms->ops) {
        int op_num = ms->num_operator_specs;
        for (int i = 0; i < op_num; i++) {
            if (nullptr != ms->ops[i].input_tensors_name) {
                for (U32 j = 0; j < ms->ops[i].num_inputs; j++) {
                    if (nullptr != ms->ops[i].input_tensors_name[j]) {
                        delete ms->ops[i].input_tensors_name[j];
                    }
                    ms->ops[i].input_tensors_name[j] = nullptr;
                }
                delete ms->ops[i].input_tensors_name;
                ms->ops[i].input_tensors_name = nullptr;
            }
            if (nullptr != ms->ops[i].output_tensors_name) {
                for (U32 j = 0; j < ms->ops[i].num_outputs; j++) {
                    if (nullptr != ms->ops[i].output_tensors_name[j]) {
                        delete ms->ops[i].output_tensors_name[j];
                    }
                    ms->ops[i].output_tensors_name[j] = nullptr;
                }
                delete ms->ops[i].output_tensors_name;
                ms->ops[i].output_tensors_name = nullptr;
            }

            if (nullptr != ms->ops[i].tensor_positions) {
                delete ms->ops[i].tensor_positions;
            }

            if (0 != ms->ops[i].num_quant_feature && nullptr != ms->ops[i].feature_scale) {
                for (U32 j = 0; j < ms->ops[i].num_quant_feature; j++) {
                    if (0 != ms->ops[i].feature_scale[j].num_scale) {
                        if (nullptr != ms->ops[i].feature_scale[j].scale) {
                            delete ms->ops[i].feature_scale[j].scale;
                        }
                    }
                }
                delete ms->ops[i].feature_scale;
            }
        }
        delete ms->ops;
        ms->ops = nullptr;
    }

    if (nullptr != ms->ws) {
        int weightOpNum = ms->num_weight_specs;
        for (int i = 0; i < weightOpNum; i++) {
            if (nullptr != ms->ws[i].weight && outOfFileMapRange(ms->ws[i].weight, ms->mfd)) {
                delete ms->ws[i].weight;
            }
            ms->ws[i].weight = nullptr;
            if (nullptr != ms->ws[i].vec && outOfFileMapRange(ms->ws[i].vec, ms->mfd)) {
                delete ms->ws[i].vec;
            }
            ms->ws[i].vec = nullptr;
        }
        delete ms->ws;
        ms->ws = nullptr;
    }

    if (nullptr != ms->op_relationship_entries) {
        int numOpRelationPair = ms->num_op_tensor_entries;
        for (int i = 0; i < numOpRelationPair; i++) {
            if (nullptr != ms->op_relationship_entries[i].input_op_names) {
                for (U32 j = 0; j < ms->op_relationship_entries[i].num_inputs; j++) {
                    if (nullptr != ms->op_relationship_entries[i].input_op_names[j]) {
                        delete ms->op_relationship_entries[i].input_op_names[j];
                    }
                    ms->op_relationship_entries[i].input_op_names[j] = nullptr;
                }
                delete ms->op_relationship_entries[i].input_op_names;
                ms->op_relationship_entries[i].input_op_names = nullptr;
            }
            if (nullptr != ms->op_relationship_entries[i].output_op_names) {
                for (U32 j = 0; j < ms->op_relationship_entries[i].num_outputs; j++) {
                    if (nullptr != ms->op_relationship_entries[i].output_op_names[j]) {
                        delete ms->op_relationship_entries[i].output_op_names[j];
                    }
                    ms->op_relationship_entries[i].output_op_names[j] = nullptr;
                }
                delete ms->op_relationship_entries[i].output_op_names;
                ms->op_relationship_entries[i].output_op_names = nullptr;
            }
        }
        delete ms->op_relationship_entries;
        ms->op_relationship_entries = nullptr;
    }

    if (ms->mfd != nullptr && !ms->mfd->useFileStream && ms->mfd->bytes != nullptr) {
#ifdef _WIN32
        // use fread to read model file
        free(ms->mfd->bytes);
#else
        // use mmap to read model file
        munmap(ms->mfd->bytes, ms->mfd->fileLength);
        if (-1 != ms->mfd->fd) {
            close(ms->mfd->fd);
        }
#endif
    }

    delete ms->mfd;
    ms->mfd = nullptr;

    return SUCCESS;
}
