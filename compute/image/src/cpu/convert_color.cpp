// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/image_cpu.h"

static inline int mad24(int a, int b, int c)
{
    return a * b + c;
}

static const int HALF_MAX_NUM = 128;

template <typename IT, typename OT>
static void yuv_nv21_to_rgb(int src_step,
    int src_offset,
    int dst_step,
    int dst_offset,
    int rows,
    int cols,
    int batch,
    int C_NUM,
    int U_ID,
    int B_ID,
    float scale,
    const IT *_srcptr,
    OT *_dstptr)
{
    static float coeffs[5] = {
        1.163999557f, 2.017999649f, -0.390999794f, -0.812999725f, 1.5959997177f};
    int rgb_size = rows * dst_step;
    for (int b = 0; b < batch; b++) {
        const IT *srcptr = _srcptr + b * rgb_size / 2;
        OT *dstptr = _dstptr + b * rgb_size;
        for (int y = 0; y < rows / 2; y++) {
            for (int x = 0; x < cols / 2; x++) {
                const IT *ysrc = srcptr + mad24(y << 1, src_step, (x << 1) + src_offset);
                const IT *usrc = srcptr + mad24(rows + y, src_step, (x << 1) + src_offset);
                OT *dst1 = dstptr + mad24(y << 1, dst_step, mad24(x, C_NUM << 1, dst_offset));
                OT *dst2 = dst1 + dst_step;

                float Y1 = ysrc[0];
                float Y2 = ysrc[1];
                float Y3 = ysrc[src_step];
                float Y4 = ysrc[src_step + 1];

                float U = ((float)usrc[U_ID]) - HALF_MAX_NUM;
                float V = ((float)usrc[1 - U_ID]) - HALF_MAX_NUM;

                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                Y1 = UNI_MAX(0.f, Y1 - 16.f) * coeffs[0];
                dst1[2 - B_ID] = scale * (Y1 + ruv);
                dst1[1] = scale * (Y1 + guv);
                dst1[B_ID] = scale * (Y1 + buv);

                Y2 = UNI_MAX(0.f, Y2 - 16.f) * coeffs[0];
                dst1[C_NUM + 2 - B_ID] = scale * (Y2 + ruv);
                dst1[C_NUM + 1] = scale * (Y2 + guv);
                dst1[C_NUM + B_ID] = scale * (Y2 + buv);

                Y3 = UNI_MAX(0.f, Y3 - 16.f) * coeffs[0];
                dst2[2 - B_ID] = scale * (Y3 + ruv);
                dst2[1] = scale * (Y3 + guv);
                dst2[B_ID] = scale * (Y3 + buv);

                Y4 = UNI_MAX(0.f, Y4 - 16.f) * coeffs[0];
                dst2[C_NUM + 2 - B_ID] = scale * (Y4 + ruv);
                dst2[C_NUM + 1] = scale * (Y4 + guv);
                dst2[C_NUM + B_ID] = scale * (Y4 + buv);
            }
        }
    }
}

template <typename IT, typename OT>
static void rgb_to_yuv_nv21(int src_step,
    int src_offset,
    int dst_step,
    int dst_offset,
    int rows,
    int cols,
    int batch,
    int C_NUM,
    int U_ID,
    int B_ID,
    float scale,
    const IT *_srcptr,
    OT *_dstptr)
{
    float src_pix1[3], src_pix2[3], src_pix3[3], src_pix4[3], uv[2];
    static float coeffs[8] = {0.256999969f, 0.50399971f, 0.09799957f, -0.1479988098f,
        -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f};
    int yuv_size = rows * dst_step;
    for (int b = 0; b < batch; b++) {
        const IT *srcptr = _srcptr + b * yuv_size * 2;
        OT *dstptr = _dstptr + b * yuv_size;
        for (int y = 0; y < rows / 3; y++) {
            int y_rows = rows / 3 * 2;
            for (int x = 0; x < cols / 2; x++) {
                const IT *src1 = srcptr + mad24(y << 1, src_step, mad24(x << 1, C_NUM, src_offset));
                const IT *src2 = src1 + src_step;
                OT *ydst1 = dstptr + mad24(y << 1, dst_step, (x << 1) + dst_offset);
                OT *ydst2 = ydst1 + dst_step;
                OT *usrc = dstptr + mad24(y_rows + y, dst_step, (x << 1) + src_offset);

                for (int i = 0; i < 3; i++) {
                    src_pix1[i] = scale * (src1[i]);
                    src_pix2[i] = scale * (src1[i + C_NUM]);
                    src_pix3[i] = scale * (src2[i]);
                    src_pix4[i] = scale * (src2[i + C_NUM]);
                }

                ydst1[0] = fma(coeffs[0], src_pix1[2 - B_ID],
                    fma(coeffs[1], src_pix1[1], fma(coeffs[2], src_pix1[B_ID], 16.5f)));
                ydst1[1] = fma(coeffs[0], src_pix2[2 - B_ID],
                    fma(coeffs[1], src_pix2[1], fma(coeffs[2], src_pix2[B_ID], 16.5f)));
                ydst2[0] = fma(coeffs[0], src_pix3[2 - B_ID],
                    fma(coeffs[1], src_pix3[1], fma(coeffs[2], src_pix3[B_ID], 16.5f)));
                ydst2[1] = fma(coeffs[0], src_pix4[2 - B_ID],
                    fma(coeffs[1], src_pix4[1], fma(coeffs[2], src_pix4[B_ID], 16.5f)));

                uv[0] = fma(coeffs[5], src_pix1[2 - B_ID],
                    fma(coeffs[6], src_pix1[1], fma(coeffs[7], src_pix1[B_ID], 128.5f)));
                uv[1] = fma(coeffs[3], src_pix1[2 - B_ID],
                    fma(coeffs[4], src_pix1[1], fma(coeffs[5], src_pix1[B_ID], 128.5f)));
                usrc[U_ID] = uv[U_ID];
                usrc[1 - U_ID] = uv[1 - U_ID];
            }
        }
    }
}

EE convert_color_cpu(TensorDesc inputDesc,
    const void *input,
    ConvertColorParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    DataType idt, odt;
    DataFormat idf, odf;
    U32 in, ic, ih, iw;
    U32 on, oc, oh, ow;
    if (inputDesc.df == DF_NHWC) {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oc, &oh, &ow));
    } else {
        CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ih, &iw, &ic));
        CHECK_STATUS(tensor4dGet(outputDesc, &odt, &odf, &on, &oh, &ow, &oc));
    }
    if (in != on) {
        return NOT_MATCH;
    }
    EE ret = SUCCESS;
    int height, width;
    int src_step, src_offset = 0, dst_step, dst_offset = 0, rows, cols;
    int batch = in, c_num, u_id = 1, b_id;
    float scale = 1;
    if (p.src == YUV_NV21) {
        c_num = oc;
        height = oh;
        width = ow;
        src_step = width;
        dst_step = width * 3;
        rows = height;
        cols = width;
        if (p.dst == RGB_0_1 || p.dst == RGB_0_255 || p.dst == RGBA_0_1 || p.dst == RGBA_0_255) {
            b_id = 2;
        } else if (p.dst == BGR_0_1 || p.dst == BGR_0_255 || p.dst == BGRA_0_1 ||
            p.dst == BGRA_0_255) {
            b_id = 0;
        } else {
            return NOT_SUPPORTED;
        }
        if (p.dst == RGB_0_1 || p.dst == BGR_0_1 || p.dst == RGBA_0_1 || p.dst == BGRA_0_1) {
            scale = 1 / 255.0;
        }
        switch (odt) {
            case DT_F32:
                yuv_nv21_to_rgb<UINT8, F32>(src_step, src_offset, dst_step, dst_offset, rows, cols,
                    batch, c_num, u_id, b_id, scale, (const UINT8 *)input, (F32 *)output);
                break;
#ifdef _USE_FP16
            case DT_F16:
                yuv_nv21_to_rgb<UINT8, F16>(src_step, src_offset, dst_step, dst_offset, rows, cols,
                    batch, c_num, u_id, b_id, scale, (const UINT8 *)input, (F16 *)output);
                break;
#endif
            case DT_U8:
                yuv_nv21_to_rgb<UINT8, UINT8>(src_step, src_offset, dst_step, dst_offset, rows,
                    cols, batch, c_num, u_id, b_id, scale, (const UINT8 *)input, (UINT8 *)output);
                break;
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else if (p.src == RGB_0_255 || p.src == RGB_0_1 || p.src == BGR_0_255 || p.src == BGR_0_1 ||
        p.src == RGBA_0_255 || p.src == RGBA_0_1 || p.src == BGRA_0_255 || p.src == BGRA_0_1) {
        c_num = ic;
        height = ih;
        width = iw;
        src_step = width * 3;
        dst_step = width;
        rows = height / 2 * 3;
        cols = width;
        if (p.src == RGB_0_1 || p.src == RGB_0_255 || p.src == RGBA_0_1 || p.src == RGBA_0_255) {
            b_id = 2;
        } else if (p.src == BGR_0_1 || p.src == BGR_0_255 || p.src == BGRA_0_1 ||
            p.src == BGRA_0_255) {
            b_id = 0;
        } else {
            return NOT_SUPPORTED;
        }
        if (p.src == RGB_0_1 || p.src == BGR_0_1 || p.src == RGBA_0_1 || p.src == BGRA_0_1) {
            scale = 255.0;
        }
        switch (odt) {
            case DT_F32:
                rgb_to_yuv_nv21<F32, UINT8>(src_step, src_offset, dst_step, dst_offset, rows, cols,
                    batch, c_num, u_id, b_id, scale, (const F32 *)input, (UINT8 *)output);
                break;
#ifdef _USE_FP16
            case DT_F16:
                rgb_to_yuv_nv21<F16, UINT8>(src_step, src_offset, dst_step, dst_offset, rows, cols,
                    batch, c_num, u_id, b_id, scale, (const F16 *)input, (UINT8 *)output);
                break;
#endif
            case DT_U8:
                rgb_to_yuv_nv21<UINT8, UINT8>(src_step, src_offset, dst_step, dst_offset, rows,
                    cols, batch, c_num, u_id, b_id, scale, (const UINT8 *)input, (UINT8 *)output);
                break;
            default:
                ret = NOT_SUPPORTED;
                break;
        }
    } else {
        ret = NOT_SUPPORTED;
    }
    return ret;
}
