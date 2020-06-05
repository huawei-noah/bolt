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

int detectionoutputTest(int argc, char **argv, DataType dt){
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

    DetectionOutputDesc detectionoutput_desc;
    detectionoutput_desc.num_class = num_class;
    detectionoutput_desc.nms_top_k = 400;
    detectionoutput_desc.nms_threshold =  0.449999988079;
    detectionoutput_desc.keep_top_k = 200;
    detectionoutput_desc.confidence_threshold = 0.00999999977648;

    std::vector<TensorDesc> input_descs;
    TensorDesc output_desc;
    TensorDesc input_desc_loc = tensor2d(dt, ih0, iw0);
    TensorDesc input_desc_conf = tensor2d(dt, ih1, iw1);
    TensorDesc input_desc_priorbox = tensor3d(dt, in2, ic2, ilens2);
    input_descs.push_back(input_desc_loc);
    input_descs.push_back(input_desc_conf);
    input_descs.push_back(input_desc_priorbox);
    CHECK_STATUS(detectionoutput_infer_output_size(input_descs, detectionoutput_desc, &output_desc, UT_ARCH));
    U32 input_len_loc = tensorNumElements(input_descs[0]);
    U32 input_len_conf = tensorNumElements(input_descs[1]);
    U32 input_len_priorbox = tensorNumElements(input_descs[2]);
    U32 output_len = tensorNumElements(output_desc);
    CHECK_REQUIREMENT(input_len_loc == ih0 * iw0 && input_len_conf == ih1 * iw1 && input_len_priorbox == in2 * ic2 * ilens2 && output_len == oh * ow);

    std::vector<void*> input(3);
    U8* input_loc = ut_input_v(input_len_loc, dt, UT_INIT_RANDOM);
    U8* input_conf = ut_input_v(input_len_conf, dt, UT_INIT_RANDOM);
    U8* input_priorbox = ut_input_v(input_len_priorbox, dt, UT_INIT_RANDOM);
    input[0] = (void*)input_loc;
    input[1] = (void*)input_conf;
    input[2] = (void*)input_priorbox;

    U8* output = ut_input_v(output_len, dt, UT_INIT_ZERO);
    U8* output_ref = ut_input_v(output_len, dt, UT_INIT_ZERO);

    if (UT_CHECK) {

        CHECK_STATUS(detectionoutput(input_descs, input, detectionoutput_desc, output_desc, output, UT_ARCH));
        CHECK_STATUS(detectionoutput(input_descs, input, detectionoutput_desc, output_desc, output_ref, CPU_GENERAL));
        // check
        ut_check_v(output, output_ref, output_len, dt, 0.05, __FILE__, __LINE__);
    }
    
    U32 num_detected_max = detectionoutput_desc.keep_top_k;
#ifdef _USE_FP16
    if (dt == DT_F16) {
        F16* output_f16 = reinterpret_cast<F16 *>(output);
        int idx = 0;
        for (U32 i = 0 ; i < 1 + num_detected_max ; i++){
            if( i >= 1 && output_f16[idx] == 0) {
                break;
            }
            std::cout << " 1 : " << output_f16[idx] << " 2 : " << output_f16[idx+1] << " 3 : " << output_f16[idx+2]  << " 4 : " << output_f16[idx+3] << " 5 : " << output_f16[idx+4]  << " 6 : " << output_f16[idx+5] << std::endl; 
            idx = idx + 6;
        }
    }
#endif
    if (dt == DT_F32) {
        F32* output_f32 = reinterpret_cast<F32 *>(output_ref);
        int idx = 0;
        for (U32 i = 0 ; i < 1 + num_detected_max ; i++){
            if( i >= 1 && output_f32[idx] == 0) {
                break;
            }            
            std::cout << " 1 : " << output_f32[idx] << " 2 : " << output_f32[idx+1] << " 3 : " << output_f32[idx+2]  << " 4 : " << output_f32[idx+3] << " 5 : " << output_f32[idx+4]  << " 6 : " << output_f32[idx+5] << std::endl; 
            idx = idx + 6;
        }
    }

    free(input_loc);
    free(input_conf);
    free(input_priorbox);
    free(output);
    free(output_ref);
    return 0;
}

int main(int argc, char** argv){
#ifdef _USE_FP16
    std::cout << "----- Testing FP16 Detectionoutput -----" <<std::endl;
    detectionoutputTest(argc, argv, DT_F16);
    std::cout << "----- Finished FP16 Detectionoutput -----" <<std::endl;
#endif
#ifdef _USE_FP32
    std::cout << "----- Testing FP32 Detectionoutput -----" <<std::endl;
    detectionoutputTest(argc, argv, DT_F32);
    std::cout << "----- Finished FP32 Detectionoutput -----" <<std::endl;
#endif
    return 0;
}