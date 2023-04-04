// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <queue>
#include "tensor_computing.h"
#include "ut_util_ocl.h"

template <typename T>
inline T Clamp(T v, T lv, T hv)
{
    if (v < lv) {
        return lv;
    }
    if (v > hv) {
        return hv;
    }
    return v;
}

template <typename T>
void genAnchor(std::vector<I32> &size,
    std::vector<F32> &ratio,
    const I32 sw,
    const I32 sh,
    const I32 fw,
    const I32 fh,
    T *baseAnchor)
{
    int index = 0;
    for (auto sv : size) {  //build 4 * 15 base value
        F32 area = sv * sv;
        for (auto rv : ratio) {
            F32 w = sqrt(area / rv);
            F32 h = rv * w;
            baseAnchor[index] = -w * 0.5;
            baseAnchor[index + 1] = -h * 0.5;
            baseAnchor[index + 2] = w * 0.5;
            baseAnchor[index + 3] = h * 0.5;
            index += 4;
        }
    }
}

template <typename T>
void genProposals(
    I32 fw, I32 fh, I32 sw, I32 sh, I32 imgW, I32 imgH, I32 anchor_num, T *anchor, T *delta, T *proposal)
{
    for (I32 c = 0; c < anchor_num; c++) {
        F32 width = anchor[c * 4 + 2] - anchor[c * 4];
        F32 height = anchor[c * 4 + 3] - anchor[c * 4 + 1];
        for (I32 h = 0; h < fh; h++) {
            for (I32 w = 0; w < fw; w++) {
                F32 ctrX = w * sw;
                F32 ctrY = h * sh;  // h * sh ?
                F32 predCtrX = delta[(c * fh + h) * fw * 4 + w * 4];
                F32 predCtrY = delta[(c * fh + h) * fw * 4 + w * 4 + 1];
                F32 predW = delta[(c * fh + h) * fw * 4 + w * 4 + 2];
                F32 predH = delta[(c * fh + h) * fw * 4 + w * 4 + 3];
                predCtrX = predCtrX * width + ctrX;
                predCtrY = predCtrY * height + ctrY;
                predW = exp(predW) * width;
                predH = exp(predH) * height;
                proposal[(c * fh + h) * fw * 4 + w * 4] =
                    Clamp<T>(predCtrX - 0.5 * predW, 0, (T)imgW);
                proposal[(c * fh + h) * fw * 4 + w * 4 + 1] =
                    Clamp<T>(predCtrY - 0.5 * predH, 0, (T)imgH);
                proposal[(c * fh + h) * fw * 4 + w * 4 + 2] =
                    Clamp<T>(predCtrX + 0.5 * predW, 0, (T)imgW);
                proposal[(c * fh + h) * fw * 4 + w * 4 + 3] =
                    Clamp<T>(predCtrY + 0.5 * predH, 0, (T)imgH);
            }
        }
    }
}

template <typename T>
void sort(I32 k, I32 len, T *val, I32 *sortIndex)
{
    auto cmp = [val](int a, int b) { return val[a] < val[b]; };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> queue(cmp);
    for (I32 i = 0; i < len; i++) {
        queue.push(i);
    }
    for (I32 i = 0; i < k; i++) {
        sortIndex[i] = queue.top();
        queue.pop();
    }
}

template <typename T>
F32 iou(T *a, T *b)
{
    F32 area1 = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    F32 area2 = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    F32 x11 = (a[0] > b[0]) ? a[0] : b[0];
    F32 y11 = (a[1] > b[1]) ? a[1] : b[1];
    F32 x22 = (a[2] < b[2]) ? a[2] : b[2];
    F32 y22 = (a[3] < b[3]) ? a[3] : b[3];
    F32 w = x22 - x11 + 1;
    F32 h = y22 - y11 + 1;
    if (w < 0)
        w = 0;
    if (h < 0)
        h = 0;
    F32 I32er = w * h;
    return I32er / (area1 + area2 - I32er);
}

template <typename T>
int nms(I32 preTopK,
    I32 postTopK,
    F32 thresh,
    I32 *sortIndex,
    I32 *curIndex,
    const T *proposal,
    I32 *postTopIndex)
{
    int *indexA = sortIndex;
    int *indexB = curIndex;
    int vaildNum = postTopK;
    T a[4];
    T b[4];
    int indexANum = preTopK;
    int indexBNum = 0;
    for (I32 i = 0; i < postTopK; i++) {
        U32 ia = indexA[0];
        a[0] = proposal[ia * 4];
        a[1] = proposal[ia * 4 + 1];
        a[2] = proposal[ia * 4 + 2];
        a[3] = proposal[ia * 4 + 3];
        postTopIndex[i] = ia;
        for (I32 j = 1; j < indexANum; j++) {
            U32 ib = indexA[j];
            b[0] = proposal[ib * 4];
            b[1] = proposal[ib * 4 + 1];
            b[2] = proposal[ib * 4 + 2];
            b[3] = proposal[ib * 4 + 3];
            if (iou<T>(a, b) <= thresh) {
                indexB[indexBNum] = ib;
                indexBNum++;
            }
        }
        int *indexTmp = indexA;
        indexA = indexB;
        indexB = indexTmp;
        indexANum = indexBNum;
        indexBNum = 0;
        if (indexANum == 0) {
            vaildNum = i + 1;
            break;
        }
    }
    return vaildNum;
}

template <typename T>
void readTxtVal(U32 fw, U32 fh, U32 anchor_num, U32 points_num, const char *fileName, T *val)
{
    FILE *file = fopen(fileName, "r");
    if (!file || feof(file)) {
        printf("open %s falied.\n", fileName);
        return;
    }
    for (U32 c = 0; c < anchor_num; c++) {
        for (U32 j = 0; j < points_num; j++) {
            for (U32 h = 0; h < fh; h++) {
                for (U32 w = 0; w < fw; w++) {
                    float v;
                    fscanf(file, "%f", &v);
                    val[(c * fh + h) * fw * points_num + w * points_num + j] = (T)v;
                }
            }
        }
    }
    fclose(file);
}

template <typename T>
void generateProposalsCpu(I32 img_w,
    I32 img_h,
    I32 fmw,
    I32 fmh,
    I32 stride_w,
    I32 stride_h,
    I32 preNmsTopK,
    I32 postNmsTopK,
    I32 anchor_num,
    F32 thresh,
    DataType dt,
    T *delta,
    T *logit,
    T *anchor,
    T *output)
{
    U8 *sortIndex = ut_input_v(preNmsTopK, DT_I32, UT_INIT_ZERO);
    U8 *curIndex = ut_input_v(preNmsTopK, DT_I32, UT_INIT_ZERO);
    U8 *tmpIndex = ut_input_v(preNmsTopK, DT_I32, UT_INIT_ZERO);
    U8 *proposal = ut_input_v(anchor_num * fmh * fmw * 4, dt, UT_INIT_ZERO);

    genProposals<T>(fmw, fmh, stride_w, stride_h, img_w, img_h, anchor_num, (T *)anchor, (T *)delta,
        (T *)proposal);
    sort<T>(preNmsTopK, anchor_num * fmh * fmw, (T *)logit, (I32 *)sortIndex);
    nms<T>(preNmsTopK, postNmsTopK, thresh, (I32 *)sortIndex, (I32 *)curIndex, (T *)proposal,
        (I32 *)tmpIndex);
    for (int i = 0; i < postNmsTopK; i++) {
        int index = ((I32 *)(tmpIndex))[i];
        T *val = (T *)proposal;
        output[i * 4] = val[index * 4];
        output[i * 4 + 1] = val[index * 4 + 1];
        output[i * 4 + 2] = val[index * 4 + 2];
        output[i * 4 + 3] = val[index * 4 + 3];
    }
    free(sortIndex);
    free(curIndex);
    free(tmpIndex);
    free(proposal);
}

template <typename T>
void generateProposals(I32 argc, char *argv[], DataType dt)
{
    I32 img_w = 640;
    I32 img_h = 640;
    I32 fmw = 20;
    I32 fmh = 20;
    if (argc > 1) {
        img_w = atoi(argv[1]);  //640
        img_h = atoi(argv[2]);  //640
        fmw = atoi(argv[3]);    //20
        fmh = atoi(argv[4]);    //20
    }
    std::vector<I32> anchor_size = {32, 64, 128, 256, 512};
    std::vector<F32> anchor_ratio = {0.5, 1.0, 2.0};
    I32 preNmsTopK = 2000;
    I32 postNmsTopK = 36;
    I32 anchor_num = anchor_size.size() * anchor_ratio.size();
    F32 thresh = 0.7;
    I32 stride_w = img_w / fmw;
    I32 stride_h = img_h / fmh;

    U8 *deltaCpu = ut_input_v(anchor_num * fmh * fmw * 4, dt, UT_INIT_RANDOM);
    readTxtVal<T>(fmw, fmh, anchor_num, 4, "./pred_anchor_deltas.txt", (T *)deltaCpu);
    U8 *logitCpu = ut_input_v(anchor_num * fmh * fmw, dt, UT_INIT_RANDOM);
    readTxtVal<T>(fmw, fmh, anchor_num, 1, "./pred_objectness_logits.txt", (T *)logitCpu);
    U8 *anchorCpu = ut_input_v(anchor_num * 4, dt, UT_INIT_RANDOM);
    genAnchor<T>(anchor_size, anchor_ratio, stride_w, stride_h, fmw, fmh, (T *)anchorCpu);
    U8 *outputCpu = ut_input_v(postNmsTopK * 4, dt, UT_INIT_ZERO);
    generateProposalsCpu(img_w, img_h, fmw, fmh, stride_w, stride_h, preNmsTopK, postNmsTopK,
        anchor_num, thresh, dt, (T *)deltaCpu, (T *)logitCpu, (T *)anchorCpu, (T *)outputCpu);

    /***************************GPU**************************/
    ArchInfo archInfo;
    archInfo.arch = MALI;
    if (gcl_check_device_qualcomm(OCLContext::getInstance().handle.get())) {
        archInfo.arch = QUALCOMM;  //off qualcomm
    }

    std::shared_ptr<GCLHandle> handleSharedPtr = OCLContext::getInstance().handle;
    GCLHandle_t handle = handleSharedPtr.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    MaliPara maliPara;
    maliPara.handle = handle;
    archInfo.archPara = &maliPara;
    Tensor deltaTensor = Tensor(OCLMem);
    Tensor logitTensor = Tensor(OCLMem);
    Tensor imgInfoTensor = Tensor(OCLMem);
    Tensor anchorTensor = Tensor(OCLMem);
    Tensor tmpTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);

    TensorDesc deltaDesc =
        tensor4df(dt, DF_NCHWC4, 1, anchor_num * 4, fmh, fmw);  //data has trans in read txt
    TensorDesc logitDesc = tensor4df(dt, DF_NCHW, 1, anchor_num, fmh, fmw);
    TensorDesc imgInfoDesc = tensor1d(dt, 3);
    TensorDesc anchorDesc = tensor2df(dt, DF_NORMAL, anchor_num, 4);
    deltaTensor.resize(deltaDesc);
    logitTensor.resize(logitDesc);
    imgInfoTensor.resize(imgInfoDesc);
    anchorTensor.resize(anchorDesc);
    GenerateProposalsParamSpec generateProposalsParam;
    generateProposalsParam.angle_bound_hi = 180;
    generateProposalsParam.angle_bound_lo = -180;
    generateProposalsParam.angle_bound_on = 1;
    generateProposalsParam.clip_angle_thresh = 1;
    generateProposalsParam.legacy_plus_one = 0;
    generateProposalsParam.min_size = 0;
    generateProposalsParam.nms_thresh = 0.7;
    generateProposalsParam.post_nms_topN = 36;
    generateProposalsParam.pre_nms_topN = 2000;
    generateProposalsParam.spatial_scale = 0.3125;

    CHECK_STATUS(generate_proposals_infer_output_size(
        &deltaTensor, &logitTensor, generateProposalsParam, &outputTensor, &archInfo));
    GCLMem_t output = alloc(outputTensor);
    TensorDesc outputDesc = outputTensor.get_desc();

    U32 bytes[2] = {0, 0};
    CHECK_STATUS(generate_proposals_infer_forward_tmp_bytes(
        deltaTensor, logitTensor, generateProposalsParam, bytes, &archInfo));
    U32 maxGpuBytes = bytes[0];
    U32 maxCpuBytes = bytes[1];
    maxGpuBytes = (maxGpuBytes > tensorNumBytes(outputDesc)) ? maxGpuBytes
                                                             : tensorNumBytes(deltaDesc);

    T imgInfoCpu[3] = {(T)img_w, (T)img_h, 1};
    GCLMem_t tmpBuf = alloc_bytes(tmpTensor, maxGpuBytes);
    Tensor tmpTensorCpu = Tensor::alloc_sized<CPUMem>(tensor1d(DT_U8, maxCpuBytes));
    ;
    alloc_host_ptr(deltaTensor, deltaCpu);
    alloc_host_ptr(logitTensor, logitCpu);
    alloc_host_ptr(imgInfoTensor, (U8 *)imgInfoCpu);
    alloc_host_ptr(anchorTensor, anchorCpu);
    std::vector<Tensor> tmpTensors;
    tmpTensors.push_back(tmpTensor);
    tmpTensors.push_back(tmpTensorCpu);

    CHECK_STATUS(generate_proposals(deltaTensor, logitTensor, imgInfoTensor, anchorTensor,
        generateProposalsParam, tmpTensors, outputTensor, &archInfo));
    double time_start = ut_time_ms();
    for (int i = 0; i < UT_LOOPS; i++) {
        CHECK_STATUS(generate_proposals(deltaTensor, logitTensor, imgInfoTensor, anchorTensor,
            generateProposalsParam, tmpTensors, outputTensor, &archInfo));
    }
    CHECK_STATUS(gcl_finish(handle));
    double time_end = ut_time_ms();
    double time = (time_end - time_start) / UT_LOOPS;
    printf("generated proposals time cost  = %lf ms", time);

    U8 *outputGpu = ut_input_v(tensorNumElements(outputDesc), dt, UT_INIT_RANDOM);
    CHECK_STATUS(ocl_get_output(handle, output, outputDesc, outputGpu, tmpBuf, true));
    ut_check_v(outputGpu, outputCpu, tensorNumElements(outputDesc), dt, 0.3);

    free(deltaCpu);
    free(logitCpu);
    free(anchorCpu);
    free(outputCpu);
    free(outputGpu);
}

int main(I32 argc, char *argv[])
{
    generateProposals<F16>(argc, argv, DT_F16);
    return 0;
}
