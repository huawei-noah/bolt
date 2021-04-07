// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "tensor_computing.h"
#include "ut_util.h"

int detectionoutputTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 11);
    // in0 loc
    U32 ih0 = atoi(argv[1]);
    U32 iw0 = atoi(argv[2]);
    // in1 conf
    U32 ih1 = atoi(argv[3]);
    U32 iw1 = atoi(argv[4]);
    // in2 priorbox
    U32 in2 = atoi(argv[5]);
    U32 ic2 = atoi(argv[6]);
    U32 ilens2 = atoi(argv[7]);
    // output
    U32 oh = atoi(argv[8]);
    U32 ow = atoi(argv[9]);
    U32 num_class = atoi(argv[10]);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    DetectionOutputParamSpec detectionoutput_desc;
    detectionoutput_desc.num_class = num_class;
    detectionoutput_desc.nms_top_k = 400;
    detectionoutput_desc.nms_threshold = 0.449999988079;
    detectionoutput_desc.keep_top_k = 200;
    detectionoutput_desc.confidence_threshold = 0.00999999977648;

    std::vector<Tensor> inputTensors(3);
    std::vector<Tensor *> inputTensorPtrs(3);
    Tensor inputTensor_loc, inputTensor_conf, inputTensor_priorbox;
    TensorDesc inputDesc_loc = tensor2d(dt, ih0, iw0);
    TensorDesc inputDesc_conf = tensor2d(dt, ih1, iw1);
    TensorDesc inputDesc_priorbox = tensor3d(dt, in2, ic2, ilens2);
    inputTensor_loc.resize(inputDesc_loc);
    inputTensor_conf.resize(inputDesc_conf);
    inputTensor_priorbox.resize(inputDesc_priorbox);
    inputTensor_loc.alloc();
    inputTensor_conf.alloc();
    inputTensor_priorbox.alloc();
    U32 input_len_loc = tensorNumElements(inputDesc_loc);
    U32 input_len_conf = tensorNumElements(inputDesc_conf);
    U32 input_len_priorbox = tensorNumElements(inputDesc_priorbox);
    U8 *input_loc = ut_input_v(input_len_loc, dt, UT_INIT_RANDOM);
    U8 *input_conf = ut_input_v(input_len_conf, dt, UT_INIT_RANDOM);
    U8 *input_priorbox = ut_input_v(input_len_priorbox, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensor_loc, UT_ARCH), input_loc, tensorNumBytes(inputDesc_loc));
    memcpy(
        get_ptr_from_tensor(inputTensor_conf, UT_ARCH), input_conf, tensorNumBytes(inputDesc_conf));
    memcpy(get_ptr_from_tensor(inputTensor_priorbox, UT_ARCH), input_priorbox,
        tensorNumBytes(inputDesc_priorbox));
    inputTensors[0] = inputTensor_loc;
    inputTensors[1] = inputTensor_conf;
    inputTensors[2] = inputTensor_priorbox;
    inputTensorPtrs[0] = &inputTensors[0];
    inputTensorPtrs[1] = &inputTensors[1];
    inputTensorPtrs[2] = &inputTensors[2];
    // set output
    Tensor outputTensor, outputTensorRef;
    CHECK_STATUS(detectionoutput_infer_output_size(
        inputTensorPtrs, detectionoutput_desc, &outputTensor, &archInfo));
    outputTensor.alloc();
    TensorDesc outputDesc_ref = outputTensor.get_desc();
    outputTensorRef.resize(outputDesc_ref);
    outputTensorRef.alloc();
    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(input_len_loc == ih0 * iw0 && input_len_conf == ih1 * iw1 &&
        input_len_priorbox == in2 * ic2 * ilens2 && output_len == oh * ow);
    if (UT_CHECK) {
        CHECK_STATUS(detectionoutput(inputTensors, detectionoutput_desc, outputTensor, &archInfo));
        CHECK_STATUS(
            detectionoutput(inputTensors, detectionoutput_desc, outputTensorRef, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt, 0.05, __FILE__, __LINE__);
    }
    U32 num_detected_max = detectionoutput_desc.keep_top_k;
#ifdef _USE_FP16
    if (dt == DT_F16) {
        F16 *output_f16 = (F16 *)get_ptr_from_tensor(outputTensor, UT_ARCH);
        int idx = 0;
        for (U32 i = 0; i < 1 + num_detected_max; i++) {
            if (i >= 1 && output_f16[idx] == 0) {
                break;
            }
            for (int j = 0; j < 6; j++) {
                printf("%d:%f ", j + 1, output_f16[idx + j]);
            }
            printf("\n");
            idx = idx + 6;
        }
    }
#endif
    if (dt == DT_F32) {
        F32 *output_f32 = (F32 *)get_ptr_from_tensor(outputTensorRef, UT_ARCH);
        int idx = 0;
        for (U32 i = 0; i < 1 + num_detected_max; i++) {
            if (i >= 1 && output_f32[idx] == 0) {
                break;
            }
            for (int j = 0; j < 6; j++) {
                printf("%d:%f ", j + 1, output_f32[idx + j]);
            }
            printf("\n");
            idx = idx + 6;
        }
    }

    free(input_loc);
    free(input_conf);
    free(input_priorbox);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    printf("----- Testing FP16 Detectionoutput -----\n");
    detectionoutputTest(argc, argv, DT_F16);
    printf("----- Finished FP16 Detectionoutput -----\n");
#endif
#ifdef _USE_FP32
    printf("----- Testing FP32 Detectionoutput -----\n");
    detectionoutputTest(argc, argv, DT_F32);
    printf("----- Finished FP32 Detectionoutput -----\n");
#endif
    return 0;
}
