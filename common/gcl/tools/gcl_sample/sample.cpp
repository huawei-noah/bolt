// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifdef _USE_FP16
#include "gcl.h"

#include "ocl_context.h"
#include "tensor.hpp"

int main()
{
    GCLHandle_t handle = OCLContext::getInstance().handle.get();
    std::vector<GCLKernelInfo> kernelVec;
    handle->kernelVec = &kernelVec;
    U32 iw, ih, it, ic, in;
    U32 fw, fh, ft, fc, fn;
    U32 pl, pr, pt, pb, pf, pd;
    U32 sw, sh, st;
    U32 ow, oh, ot, oc, on;

    iw = 3;
    ih = 3;
    it = 1;
    ic = 512;
    in = 30;

    fw = 3;
    fh = 3;
    ft = 1;
    fc = 512;
    fn = 512;

    pl = 1;
    pr = 1;
    pt = 1;
    pb = 1;
    pf = 0;
    pd = 0;

    sw = 1;
    sh = 1;
    st = 1;

    ow = (iw + pl + pr - fw) / sw + 1;
    oh = (ih + pt + pb - fh) / sh + 1;
    ot = (it + pf + pd - ft) / st + 1;
    oc = fn;
    on = in;

    TensorDesc inputDesc = tensor5df(DT_F16, DF_NCHW, in, ic, it, ih, iw);
    TensorDesc filterDesc = tensor5df(DT_F16, DF_NCHW, fn, fc, ft, fh, fw);
    TensorDesc biasDesc = tensor1d(DT_F16, fn);
    TensorDesc outputDesc = tensor5df(DT_F16, DF_NCHW, on, oc, ot, oh, ow);
    Tensor inputTensor = Tensor(OCLMem);
    Tensor filterTensor = Tensor(OCLMem);
    Tensor biasTensor = Tensor(OCLMem);
    Tensor outputTensor = Tensor(OCLMem);

    inputTensor.resize(inputDesc);
    filterTensor.resize(filterDesc);
    biasTensor.resize(biasDesc);
    outputTensor.resize(outputDesc);

    U32 stride[3] = {0, 0, 0};
    U32 offset[3] = {0, 0, 0};

    /*input*/
    GCLMemDesc inputGclDesc = gclmem_build_desc();
    U32 iw_align = ow;
    for (U32 i = 0; i < 8; i++) {
        U32 j = (ow + i - 1) / i * i;
        if (j > iw_align)
            iw_align = j;
    }
    iw_align = iw_align * sw;
    stride[0] = iw_align + pl + pr;
    stride[1] = ih + pt + pb;
    stride[2] = (ic + 3) / 4 * it * ((in + 3) / 4 * 4);
    offset[0] = pl;
    offset[1] = pt;
    offset[2] = 0;
    gclmem_set_desc_padding(
        &inputGclDesc, stride, offset, DT_F16, DF_NCWHC4, GCL_MEM_BUF, CL_MEM_READ_WRITE);
    auto gclmem = (OclMemory *)inputTensor.get_memory();
    gclmem->padding(inputGclDesc);
    gclmem->alloc();

    /*filter*/
    GCLMemDesc filterGclDesc = gclmem_build_desc();
    stride[0] = fw * fh * ft;
    stride[1] = fc;
    stride[2] = (fn + 3) / 4;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    gclmem_set_desc_padding(
        &filterGclDesc, stride, offset, DT_F16, DF_NCWHC4, GCL_MEM_BUF, CL_MEM_READ_WRITE);
    gclmem = (OclMemory *)filterTensor.get_memory();
    gclmem->padding(filterGclDesc);
    gclmem->alloc();

    /*bias*/
    GCLMemDesc biasGclDesc = gclmem_build_desc();
    stride[0] = (fn + 3) / 4;
    stride[1] = 1;
    stride[2] = 1;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    gclmem_set_desc_padding(
        &biasGclDesc, stride, offset, DT_F16, DF_NHWC, GCL_MEM_IMG_1D, CL_MEM_READ_WRITE);
    gclmem = (OclMemory *)biasTensor.get_memory();
    gclmem->padding(biasGclDesc);
    gclmem->alloc();

    GCLMemDesc outputGclDesc = gclmem_build_desc();
    stride[0] = oh;
    stride[1] = ow;
    stride[2] = (oc + 3) / 4 * ot * on;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    gclmem_set_desc_padding(
        &outputGclDesc, stride, offset, DT_F16, DF_NCWHC4, GCL_MEM_BUF, CL_MEM_READ_WRITE);
    gclmem = (OclMemory *)outputTensor.get_memory();
    gclmem->padding(outputGclDesc);
    gclmem->alloc();

    gclmem = (OclMemory *)inputTensor.get_memory();
    GCLMem_t input = (GCLMem_t)gclmem->get_ptr();
    gclmem = (OclMemory *)filterTensor.get_memory();
    GCLMem_t flt = (GCLMem_t)gclmem->get_ptr();
    gclmem = (OclMemory *)biasTensor.get_memory();
    GCLMem_t bias = (GCLMem_t)gclmem->get_ptr();
    gclmem = (OclMemory *)outputTensor.get_memory();
    GCLMem_t output = (GCLMem_t)gclmem->get_ptr();
    U32 iw_str, ih_str, iw_off, ih_off, iwh_str, ic_str;
    U32 ow_str, oh_str, oh_off, ow_off, owh_str, oc_str;
    U32 in_str, on_str;
    get_gclmem_dim(inputGclDesc, &iw_str, &ih_str, &ic_str, &iw_off, &ih_off);
    iwh_str = iw_str * ih_str;
    iw_off -= pl;
    ih_off -= pt;
    ic_str = ic_str / (it * ((in + 3) / 4 * 4));
    in_str = iwh_str * ic_str;
    get_gclmem_dim(outputGclDesc, &ow_str, &oh_str, &oc_str, &ow_off, &oh_off);
    owh_str = ow_str * oh_str;
    oc_str = oc_str / (ot * on);
    on_str = owh_str * oc_str;

    //    F16* input_val = (F16*)malloc(inputGclDesc.byteSize);
    //    F16* filter_val = (F16*)malloc(filterGclDesc.byteSize);
    //    F16* bias_val = (F16*)malloc(biasGclDesc.byteSize);
    //    for (U32 i = 0; i < inputGclDesc.num; i++) input_val[i] = (i % 16) * 0.1;
    //    for (U32 i = 0; i < filterGclDesc.num; i++) filter_val[i] = (i % 16) * 0.1;
    //    for (U32 i = 0; i < biasGclDesc.num * 4; i++) bias_val[i] = 1.0;
    //    U32 size[3] = {1, 1, 1};
    //    size[0] = inputGclDesc.byteSize;
    //    CHECK_STATUS(gcl_trans_memory(handle, input_val, input, size, HOST_TO_DEVICE_BUF, CL_TRUE));
    //    size[0] = filterGclDesc.byteSize;
    //    CHECK_STATUS(gcl_trans_memory(handle, filter_val, flt, size, HOST_TO_DEVICE_BUF, CL_TRUE));
    //    size[0] = biasGclDesc.num;
    //    CHECK_STATUS(gcl_trans_memory(handle, bias_val, bias, size, HOST_TO_DEVICE_IMG, CL_TRUE));
    //
    //    CHECK_STATUS(gcl_check_buf<F16>(handle, input->mem, inputGclDesc.byteSize, false, "input"));
    //    CHECK_STATUS(gcl_check_buf<F16>(handle, flt->mem, filterGclDesc.byteSize, false, "filter"));
    gcl_finish(handle);
    for (U32 item_bn = 2; item_bn <= 4; item_bn++) {
        for (U32 item_kn = 1; item_kn <= 2; item_kn = item_kn * 2) {
            for (U32 item_w = 1; item_w <= 3; item_w++) {
                if (item_kn == 2 && item_w > 1)
                    continue;
                if (item_bn > 2) {
                    if (item_w > 2)
                        continue;
                    if (item_kn > 2)
                        continue;
                    if (item_kn == 2 && item_w == 2)
                        continue;
                }

                Kernel kernel;
                char kernelName[1024];
                sprintf(kernelName, "conv_direct_multi_batch_s1_%d%d%d%d%d", fw, fh, item_w,
                    item_kn, item_bn);
                CHECK_STATUS(gcl_create_kernel(handle, kernelName, &kernel));
                if (oc_str % item_kn != 0) {
                    continue;
                }
                U32 gs[3] = {oh, (ow + item_w - 1) / item_w,
                    (oc + 3) / 4 / item_kn * ((on + item_bn - 1) / item_bn)};
                U32 ls[3] = {0, 0, 0};
                U32 dim = 3;
                CHECK_STATUS(gcl_set_kernelArgs(kernel, ih_str, iwh_str, ic_str, ih_off, iw_off,
                    oh_str, owh_str, oh_off, ow_off, ow, oc, on, sh, in_str, on_str, gs[0], gs[1],
                    input->mem, flt->mem, bias->mem, output->mem));
                gcl_set_kernelVec(handle, kernel, dim, gs, ls, kernelName);
                CHECK_STATUS(gcl_run_kernel_select_ls(handle, &kernelVec[0]));
#ifdef _DEBUG
                CHECK_STATUS(gcl_run_kernelVec_timing(handle, 0, handle->kernelVec->size()));
                double ops = (1.0 * on * oc * ot * oh * ow) * (2.0 * ic * ft * fh * fw + 1);
                double time = handle->t_execute;
                double gflops = 1e-3 * ops / time;
                UNI_INFO_LOG("gflops: %lf\n", gflops);
#else
                CHECK_STATUS(gcl_run_kernelVec(handle));
#endif
                //                CHECK_STATUS(gcl_check_buf<F16>(handle, output->mem, outputGclDesc.byteSize, false, "output"));
                //                CHECK_STATUS(gcl_fill_memory_zero(handle, output));
                CHECK_STATUS(gcl_clean_kernelVec(handle));
                gcl_finish(handle);
            }
        }
    }
}
#endif
