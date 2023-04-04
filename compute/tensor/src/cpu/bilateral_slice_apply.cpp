// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "cpu/tensor_computing_cpu.h"

template <typename T>
static inline T guide_cal(T r_, T g_, T b_)
{
    T a[4] = {r_, g_, b_, 1};

    static const T wx[4] = {
        0.9266905188560486, -0.07651382684707642, -0.11796596646308899, 0.03732128441333771};
    static const T wy[4] = {
        0.016965966671705246, 1.0332931280136108, 0.09558156877756119, 0.049296945333480835};
    static const T wz[4] = {
        -0.060142070055007935, -0.0184615608304739, 0.9641872048377991, 0.03588166460394859};
    T x = 0, y = 0, z = 0;
    for (int i = 0; i < 4; i++) {
        x += a[i] * wx[i];
        y += a[i] * wy[i];
        z += a[i] * wz[i];
    }

    static const T sx[16] = {-0.04031608998775482, 0.203898087143898, 0.21509018540382385,
        0.2156994342803955, 0.22189579904079437, 0.2710961699485779, 0.33060845732688904,
        0.3510134816169739, 0.3799624741077423, 0.4165642559528351, 0.5429311394691467,
        0.6519719958305359, 0.7579551339149475, 0.8117461800575256, 0.8115477561950684,
        0.811525821685791};
    static const T sy[16] = {-0.04493796080350876, 0.2501078248023987, 0.24961410462856293,
        0.24829524755477905, 0.25029096007347107, 0.25275537371635437, 0.2535839378833771,
        0.25915712118148804, 0.992545485496521, 0.869307279586792, 0.8143411874771118,
        0.8268355131149292, 0.849763810634613, 0.8641695380210876, 0.8749480843544006,
        0.9124495387077332};
    static const T sz[16] = {-0.0450710691511631, 0.17914339900016785, 0.20727036893367767,
        0.21128158271312714, 0.785589873790741, 0.40014126896858215, 0.39716723561286926,
        0.4003089666366577, 0.5749346613883972, 0.6277766227722168, 0.7884474992752075,
        0.788446307182312, 0.789533257484436, 0.7905913591384888, 0.7964500188827515,
        0.7964839339256287};
    T rx[16], ry[16], rz[16];
    for (int i = 0; i < 16; i++) {
        rx[i] = UNI_MAX(x - sx[i], 0);
        ry[i] = UNI_MAX(x - sy[i], 0);
        rz[i] = UNI_MAX(z - sz[i], 0);
    }

    static const T mx[16] = {0.9483454823493958, -0.02504969760775566, -0.0731356292963028,
        -0.08960649371147156, -0.0989985391497612, -0.0911787822842598, -0.07849951088428497,
        -0.07431424409151077, -0.05982533469796181, -0.027073463425040245, 0.09377846121788025,
        0.07562971860170364, -0.05076618492603302, 0.2615104913711548, 0.42631882429122925,
        0.6887183785438538};
    static const T my[16] = {0.9732255339622498, -0.03841959312558174, -0.07476486265659332,
        -0.08849595487117767, -0.10008298605680466, -0.10915014147758484, -0.1108635663986206,
        -0.09364574402570724, -0.04355158284306526, -0.015994733199477196, -0.025348246097564697,
        -0.051913388073444366, -0.07183714956045151, -0.0823502317070961, -0.09460879862308502,
        -0.13453315198421478};
    static const T mz[16] = {0.951180636882782, -0.014929438941180706, -0.022745108231902122,
        -0.042111292481422424, 0.061638616025447845, -0.04308458790183067, -0.050973013043403625,
        -0.045611534267663956, 0.037990815937519073, 0.04962018504738808, 0.15617141127586365,
        0.13662904500961304, 0.16109246015548706, 0.160025492310524, 0.12079561501741409,
        0.15001150965690613};
    x = 0;
    y = 0;
    z = 0;
    for (int i = 0; i < 16; i++) {
        x += rx[i] * mx[i];
        y += ry[i] * my[i];
        z += rz[i] * mz[i];
    }

    static const T w[4] = {
        0.28540247678756714, 0.31782254576683044, 0.28381019830703735, 0.06326253712177277};
    T ret = w[0] * x + w[1] * y + w[2] * z + w[3];
    ret = UNI_MIN(UNI_MAX(ret, 0), 1);
    return ret;
}

template <typename T1, typename T2, bool conv>
static void bilateral_slice_apply(const T1 *input,
    const T2 *guide,
    const T2 *grid,
    int batch,
    int height,
    int width,
    int grid_height,
    int grid_width,
    int grid_dc,
    T1 *output)
{
    const int chans = 12;
    T2 values[chans];
    int grid_depth = grid_dc / chans;
    int h_scale = grid_width * grid_depth * chans;
    int w_scale = grid_depth * chans;
    int d_scale = chans;
    for (int n = 0, i = 0; n < batch; n++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++, i++) {
                T2 r = input[3 * i + 0];
                T2 g = input[3 * i + 1];
                T2 b = input[3 * i + 2];
                if (sizeof(T1) == 1) {
                    r /= 255.;
                    g /= 255.;
                    b /= 255.;
                }

                float gh = (h + 0.5f) * grid_height / (1.0f * height) - 0.5f;
                float gw = (w + 0.5f) * grid_width / (1.0f * width) - 0.5f;
                float gd;
                if (conv) {
                    gd = guide_cal<float>(r, g, b);
                } else {
                    gd = guide[i];
                }
                gd = gd * grid_depth - 0.5f;

                int fh = floor(gh);
                int fw = floor(gw);
                int fd = floor(gd);

                UNI_MEMSET(values, 0, sizeof(T2) * chans);
                for (int hh = fh; hh < fh + 2; hh++) {
                    int h_idx = UNI_MAX(UNI_MIN(hh, grid_height - 1), 0);
                    float h_ratio_ = 1.0f - UNI_ABS(gh - hh);
                    float h_ratio = UNI_MAX(h_ratio_, 0.0f);
                    for (int ww = fw; ww < fw + 2; ww++) {
                        int w_idx = UNI_MAX(UNI_MIN(ww, grid_width - 1), 0);
                        float w_ratio_ = 1.0f - UNI_ABS(gw - ww);
                        float w_ratio = UNI_MAX(w_ratio_, 0.0f);
                        for (int dd = fd; dd < fd + 2; dd++) {
                            int d_idx = UNI_MAX(UNI_MIN(dd, grid_depth - 1), 0);
                            float d_ratio_ = 1.0f - UNI_ABS(gd - dd);
                            float d_ratio = UNI_MAX(d_ratio_, 0.0f);
                            for (int c = 0; c < chans; c++) {
                                int idx = h_idx * h_scale + w_idx * w_scale + d_idx * d_scale + c;
                                values[c] += grid[idx] * h_ratio * w_ratio * d_ratio;
                            }
                        }
                    }
                }

                T2 x = values[0] * r + values[1] * g + values[2] * b + values[3];
                T2 y = values[4] * r + values[5] * g + values[6] * b + values[7];
                T2 z = values[8] * r + values[9] * g + values[10] * b + values[11];
                if (sizeof(T1) == 1) {
                    x *= 255.;
                    y *= 255.;
                    z *= 255.;
                }
                output[3 * i + 0] = x;
                output[3 * i + 1] = y;
                output[3 * i + 2] = z;
            }
        }
    }
}

template <typename T1, typename T2>
static EE bilateral_slice_apply(const T1 *input,
    const T2 *guide,
    const T2 *grid,
    BilateralSliceApplyParamSpec p,
    int batch,
    int height,
    int width,
    int grid_height,
    int grid_width,
    int grid_dc,
    T1 *output)
{
    if (p.mode == BILATERAL_SLICE_APPLY_CONV) {
        bilateral_slice_apply<T1, T2, true>(
            input, guide, grid, batch, height, width, grid_height, grid_width, grid_dc, output);
    } else {
        bilateral_slice_apply<T1, T2, false>(
            input, guide, grid, batch, height, width, grid_height, grid_width, grid_dc, output);
    }

    return SUCCESS;
}

EE bilateral_slice_apply_cpu(TensorDesc inputDesc,
    const void *input,
    TensorDesc guideDesc,
    const void *guide,
    TensorDesc gridDesc,
    const void *grid,
    BilateralSliceApplyParamSpec p,
    TensorDesc outputDesc,
    void *output)
{
    DataType idt, gdt;
    DataFormat idf, gdf;
    U32 in, ic, ih, iw;
    U32 gn, gc, gh, gw;
    CHECK_STATUS(tensor4dGet(inputDesc, &idt, &idf, &in, &ic, &ih, &iw));
    CHECK_STATUS(tensor4dGet(gridDesc, &gdt, &gdf, &gn, &gc, &gh, &gw));
    if (in != gn) {
        return NOT_MATCH;
    }
    if (ic != 3 || gc % 12 != 0) {
        return NOT_MATCH;
    }
    EE ret = NOT_SUPPORTED;
    switch (idt) {
        case DT_F32: {
            ret = bilateral_slice_apply<F32, F32>((const F32 *)input, (const F32 *)guide,
                (const F32 *)grid, p, in, ih, iw, gh, gw, gc, (F32 *)output);
            break;
        }
#ifdef _USE_FP16
        case DT_F16: {
            ret = bilateral_slice_apply<F16, F16>((const F16 *)input, (const F16 *)guide,
                (const F16 *)grid, p, in, ih, iw, gh, gw, gc, (F16 *)output);
            break;
        }
#endif
        case DT_U8: {
            if (gdt == DT_F32) {
                ret = bilateral_slice_apply<UINT8, F32>((const UINT8 *)input, (const F32 *)guide,
                    (const F32 *)grid, p, in, ih, iw, gh, gw, gc, (UINT8 *)output);
#ifdef _USE_FP16
            } else if (gdt == DT_F16) {
                ret = bilateral_slice_apply<UINT8, F16>((const UINT8 *)input, (const F16 *)guide,
                    (const F16 *)grid, p, in, ih, iw, gh, gw, gc, (UINT8 *)output);
#endif
            }
            break;
        }
        default:
            break;
    }
    return ret;
}
