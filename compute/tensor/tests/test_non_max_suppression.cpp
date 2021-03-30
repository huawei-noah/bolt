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

int nonmaxsuppressionTest(int argc, char **argv, DataType dt)
{
    CHECK_REQUIREMENT(argc == 12);
    // in0 boxes
    U32 in0 = atoi(argv[1]);
    U32 ic0 = atoi(argv[2]);
    U32 ilens0 = atoi(argv[3]);
    // in1 scores
    U32 in1 = atoi(argv[4]);
    U32 ic1 = atoi(argv[5]);
    U32 ilens1 = atoi(argv[6]);
    // output
    U32 oh = atoi(argv[7]);
    U32 ow = atoi(argv[8]);
    // nonMaxSuppressionParamSpec
    U32 max_output_boxes_per_class = atoi(argv[9]);
    F32 iou_threshold = (F32)atof(argv[10]);
    F32 score_threshold = (F32)atof(argv[11]);
    ArchInfo archInfo;
    archInfo.arch = UT_ARCH;
    ArchInfo archInfo_org;
    archInfo_org.arch = CPU_GENERAL;

    NonMaxSuppressionParamSpec nonMaxSuppressionParamSpec;
    nonMaxSuppressionParamSpec.max_output_boxes_per_class = max_output_boxes_per_class;
    nonMaxSuppressionParamSpec.iou_threshold = iou_threshold;
    nonMaxSuppressionParamSpec.score_threshold = score_threshold;

    std::vector<Tensor> inputTensors(2);
    TensorDesc input_desc_boxes = tensor3d(dt, in0, ic0, ilens0);
    TensorDesc input_desc_scores = tensor3d(dt, in1, ic1, ilens1);
    inputTensors[0] = Tensor::alloc_sized<CPUMem>(input_desc_boxes);
    inputTensors[1] = Tensor::alloc_sized<CPUMem>(input_desc_scores);
    U32 input_len_boxes = tensorNumElements(input_desc_boxes);
    U8 *input_boxes = ut_input_v(input_len_boxes, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensors[0], UT_ARCH), input_boxes,
        tensorNumBytes(input_desc_boxes));
    U32 input_len_scores = tensorNumElements(input_desc_scores);
    U8 *input_scores = ut_input_v(input_len_scores, dt, UT_INIT_RANDOM);
    memcpy(get_ptr_from_tensor(inputTensors[1], UT_ARCH), input_scores,
        tensorNumBytes(input_desc_scores));
    std::vector<Tensor *> inputTensorsPtr(2);
    inputTensorsPtr[0] = &inputTensors[0];
    inputTensorsPtr[1] = &inputTensors[1];
    //set output
    Tensor outputTensor;
    CHECK_STATUS(non_max_suppression_infer_output_size(
        inputTensorsPtr, nonMaxSuppressionParamSpec, &outputTensor, &archInfo));
    outputTensor.alloc();
    Tensor outputTensorRef = Tensor::alloc_sized<CPUMem>(outputTensor.get_desc());
    U32 output_len = outputTensor.length();
    CHECK_REQUIREMENT(input_len_boxes == in0 * ic0 * ilens0 &&
        input_len_scores == in1 * ic1 * ilens1 && output_len == oh * ow);
    /*
       You can also change codes and use datas in the following example.
       Command: ./test_non_max_suppression 1 6 4 1 2 6 7 3 3 0.5 0
       example:
       input_box[24] = { 0.0, 0.0, 1.0, 1.0,
                           0.0, 0.1, 1.0, 1.1,
                           0.0, -0.1, 1.0, 0.9,
                           0.0, 10.0, 1.0, 11.0,
                           0.0, 10.1, 1.0, 11.1,
                           0.0, 100.0, 1.0, 101.0 };
       input_score[12] = { 0.75, 0.9, 0.6, 0.95, 0.5, 0.3, 0.75, 0.9, 0.6, 0.95, 0.5, 0.3 };
       output_expect:
                   { 6,  0,  0,
                       0,  0,  3,
                       0,  0,  1,
                       0,  0,  5,
                       0,  1,  3,
                       0,  1,  1,
                       0,  1,  5 };
     */
    if (UT_CHECK) {
        CHECK_STATUS(
            non_max_suppression(inputTensors, nonMaxSuppressionParamSpec, outputTensor, &archInfo));
        CHECK_STATUS(non_max_suppression(
            inputTensors, nonMaxSuppressionParamSpec, outputTensorRef, &archInfo_org));
        // check
        ut_check_v(get_ptr_from_tensor(outputTensor, UT_ARCH),
            get_ptr_from_tensor(outputTensorRef, UT_ARCH), output_len, dt, 0.05, __FILE__, __LINE__);
    }

    U32 num_detected_max = max_output_boxes_per_class * ic1;
    if (dt == DT_F32) {
        F32 *output_f32 = (F32 *)get_ptr_from_tensor(outputTensor, UT_ARCH);
        int idx = 0;
        for (U32 i = 0; i < 1 + num_detected_max; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%d:%f ", j, output_f32[idx + j]);
            }
            printf("\n");
            idx = idx + 3;
        }
    }
#ifdef _USE_FP16
    if (dt == DT_F16) {
        F16 *output_f16 = (F16 *)get_ptr_from_tensor(outputTensorRef, UT_ARCH);
        int idx = 0;
        for (U32 i = 0; i < 1 + num_detected_max; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%d:%f ", j + 1, output_f16[idx + j]);
            }
            printf("\n");
            idx = idx + 3;
        }
    }
#endif
    free(input_boxes);
    free(input_scores);
    return 0;
}

int main(int argc, char **argv)
{
#ifdef _USE_FP16
    printf("----- Testing FP16 Nonmaxsuppression -----\n");
    nonmaxsuppressionTest(argc, argv, DT_F16);
    printf("----- Finished FP16 Nonmaxsuppression -----\n");
#endif
#ifdef _USE_FP32
    printf("----- Testing FP32 Nonmaxsuppression -----\n");
    nonmaxsuppressionTest(argc, argv, DT_F32);
    printf("----- Finished FP32 Nonmaxsuppression -----\n");
#endif
    return 0;
}
