// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#if defined(__GLIBC__) || defined(__linux__)
#include <sys/mman.h>
#endif

#include "model_common.h"

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
    ms->file = nullptr;

    return SUCCESS;
}

EE mt_destroy_model(ModelSpec *ms)
{
    if (nullptr == ms) {
        return NULL_POINTER;
    }

    if (nullptr != ms->input_names) {
        for (int i = 0; i < ms->num_inputs; i++) {
            mt_free(ms->input_names[i]);
        }
        ms->num_inputs = 0;
        mt_free(ms->input_names);
    }

    if (nullptr != ms->input_dims) {
        mt_free(ms->input_dims);
    }

    if (nullptr != ms->output_names) {
        for (int i = 0; i < ms->num_outputs; i++) {
            mt_free(ms->output_names[i]);
        }
        ms->num_outputs = 0;
        mt_free(ms->output_names);
    }

    if (nullptr != ms->ops) {
        int op_num = ms->num_operator_specs;
        for (int i = 0; i < op_num; i++) {
            if (nullptr != ms->ops[i].input_tensors_name) {
                for (U32 j = 0; j < ms->ops[i].num_inputs; j++) {
                    mt_free(ms->ops[i].input_tensors_name[j]);
                }
                ms->ops[i].num_inputs = 0;
                mt_free(ms->ops[i].input_tensors_name);
            }
            if (nullptr != ms->ops[i].output_tensors_name) {
                for (U32 j = 0; j < ms->ops[i].num_outputs; j++) {
                    mt_free(ms->ops[i].output_tensors_name[j]);
                }
                ms->ops[i].num_outputs = 0;
                mt_free(ms->ops[i].output_tensors_name);
            }
            mt_free(ms->ops[i].tensor_positions);

            if (0 != ms->ops[i].num_quant_feature && nullptr != ms->ops[i].feature_scale) {
                for (U32 j = 0; j < ms->ops[i].num_quant_feature; j++) {
                    if (0 != ms->ops[i].feature_scale[j].num_scale) {
                        ms->ops[i].feature_scale[j].num_scale = 0;
                        mt_free(ms->ops[i].feature_scale[j].scale);
                    }
                }
                ms->ops[i].num_quant_feature = 0;
                mt_free(ms->ops[i].feature_scale);
            }
        }
        ms->num_operator_specs = 0;
        mt_free(ms->ops);
    }

    if (nullptr != ms->ws) {
        for (int i = 0; i < ms->num_weight_specs; i++) {
            ms->ws[i].bytes_of_weight = 0;
            mt_free(ms->ws[i].weight, ms);
            ms->ws[i].bytes_of_vec = 0;
            mt_free(ms->ws[i].vec, ms);
            for (U32 j = 0; j < ms->ws[i].num_quant_scale; j++) {
                if (0 != ms->ws[i].weight_scale[j].num_scale) {
                    ms->ws[i].weight_scale[j].num_scale = 0;
                    mt_free(ms->ws[i].weight_scale[j].scale);
                }
            }
            ms->ws[i].num_quant_scale = 0;
            mt_free(ms->ws[i].weight_scale);
        }
        ms->num_weight_specs = 0;
        mt_free(ms->ws);
    }

    if (nullptr != ms->op_relationship_entries) {
        for (int i = 0; i < ms->num_op_tensor_entries; i++) {
            if (nullptr != ms->op_relationship_entries[i].input_op_names) {
                for (U32 j = 0; j < ms->op_relationship_entries[i].num_inputs; j++) {
                    mt_free(ms->op_relationship_entries[i].input_op_names[j]);
                }
                ms->op_relationship_entries[i].num_inputs = 0;
                mt_free(ms->op_relationship_entries[i].input_op_names);
            }
            if (nullptr != ms->op_relationship_entries[i].output_op_names) {
                for (U32 j = 0; j < ms->op_relationship_entries[i].num_outputs; j++) {
                    mt_free(ms->op_relationship_entries[i].output_op_names[j]);
                }
                ms->op_relationship_entries[i].num_outputs = 0;
                mt_free(ms->op_relationship_entries[i].output_op_names);
            }
        }
        ms->num_op_tensor_entries = 0;
        mt_free(ms->op_relationship_entries);
    }

    if (ms->file != nullptr) {
        if (ms->file->stream_mode) {
            ms->file->content = nullptr;
            ms->file->length = 0;
        } else {
            if (ms->file->content != nullptr) {
#if defined(__GLIBC__) || defined(__linux__)
                munmap(ms->file->content, ms->file->length);
#elif defined(_WIN32)
                pthread_join(ms->file->thread, NULL);
                UnmapViewOfFile(ms->file->content);
#else
                UNI_FREE(ms->file->content);
#endif
            }
            ms->file->content = nullptr;
            ms->file->length = 0;
#if defined(__GLIBC__) || defined(__linux__)
            if (-1 != ms->file->file) {
                close(ms->file->file);
                ms->file->file = -1;
            }
#elif defined(_WIN32)
            if (ms->file->map != NULL) {
                CloseHandle(ms->file->map);
                ms->file->map = NULL;
            }
            if (ms->file->file != NULL) {
                CloseHandle(ms->file->file);
                ms->file->file = NULL;
            }
#endif
        }
        mt_free(ms->file);
    }
    return SUCCESS;
}
