// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _KERNEL_DEF
#define _KERNEL_DEF

/*
 * READ IMAGE
 */
#if defined(USE_HALF)
#define READ_IMAGE(image, sampler, coord) read_imageh(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data) write_imageh(image, coord, data)
#else
#define READ_IMAGE(image, sampler, coord) read_imagef(image, sampler, coord)
#define WRITE_IMAGE(image, coord, data) write_imagef(image, coord, data)
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#if defined(USE_V1)
#define READ_BUF(v, off, buf) \
    {                         \
        v = buf[off];         \
    }
#elif defined(USE_V2)
#define READ_BUF(v, off, buf)     \
    {                             \
        v = vload2(0, buf + off); \
    }
#elif defined(USE_V3)
#define READ_BUF(v, off, buf)     \
    {                             \
        v = vload3(0, buf + off); \
    }
#elif defined(USE_V4)
#define READ_BUF(v, off, buf)     \
    {                             \
        v = vload4(0, buf + off); \
    }
#elif defined(USE_V8)
#define READ_BUF(v, off, buf)     \
    {                             \
        v = vload8(0, buf + off); \
    }
#elif defined(USE_V16)
#define READ_BUF(v, off, buf)      \
    {                              \
        v = vload16(0, buf + off); \
    }
#endif

/*
 * load data from buffer to reg array
 */
#define LOAD_BUF_ARRAY1(v, off, buf) \
    {                                \
        v[0] = buf[off];             \
    }

#define LOAD_BUF_ARRAY2(v, off, buf)   \
    {                                  \
        T2 tmp = vload2(0, buf + off); \
        v[0] = tmp.x;                  \
        v[1] = tmp.y;                  \
    }

#define LOAD_BUF_ARRAY3(v, off, buf)   \
    {                                  \
        T3 tmp = vload3(0, buf + off); \
        v[0] = tmp.x;                  \
        v[1] = tmp.y;                  \
        v[2] = tmp.z;                  \
    }

#define LOAD_BUF_ARRAY4(v, off, buf)   \
    {                                  \
        T4 tmp = vload4(0, buf + off); \
        v[0] = tmp.x;                  \
        v[1] = tmp.y;                  \
        v[2] = tmp.z;                  \
        v[3] = tmp.w;                  \
    }

#define LOAD_BUF_ARRAY5(v, off, buf)   \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] = tmp.s0;                 \
        v[1] = tmp.s1;                 \
        v[2] = tmp.s2;                 \
        v[3] = tmp.s3;                 \
        v[4] = tmp.s4;                 \
    }

#define LOAD_BUF_ARRAY6(v, off, buf)   \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] = tmp.s0;                 \
        v[1] = tmp.s1;                 \
        v[2] = tmp.s2;                 \
        v[3] = tmp.s3;                 \
        v[4] = tmp.s4;                 \
        v[5] = tmp.s5;                 \
    }

#define LOAD_BUF_ARRAY7(v, off, buf)   \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] = tmp.s0;                 \
        v[1] = tmp.s1;                 \
        v[2] = tmp.s2;                 \
        v[3] = tmp.s3;                 \
        v[4] = tmp.s4;                 \
        v[5] = tmp.s5;                 \
        v[6] = tmp.s6;                 \
    }

#define LOAD_BUF_ARRAY8(v, off, buf)   \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] = tmp.s0;                 \
        v[1] = tmp.s1;                 \
        v[2] = tmp.s2;                 \
        v[3] = tmp.s3;                 \
        v[4] = tmp.s4;                 \
        v[5] = tmp.s5;                 \
        v[6] = tmp.s6;                 \
        v[7] = tmp.s7;                 \
    }

#define ADD_BUF_ARRAY1(v, off, buf) \
    {                               \
        v[0] += buf[off];           \
    }

#define ADD_BUF_ARRAY2(v, off, buf)    \
    {                                  \
        T2 tmp = vload2(0, buf + off); \
        v[0] += tmp.x;                 \
        v[1] += tmp.y;                 \
    }

#define ADD_BUF_ARRAY3(v, off, buf)    \
    {                                  \
        T3 tmp = vload3(0, buf + off); \
        v[0] += tmp.x;                 \
        v[1] += tmp.y;                 \
        v[2] += tmp.z;                 \
    }

#define ADD_BUF_ARRAY4(v, off, buf)    \
    {                                  \
        T4 tmp = vload4(0, buf + off); \
        v[0] += tmp.x;                 \
        v[1] += tmp.y;                 \
        v[2] += tmp.z;                 \
        v[3] += tmp.w;                 \
    }

#define ADD_BUF_ARRAY5(v, off, buf)    \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] += tmp.s0;                \
        v[1] += tmp.s1;                \
        v[2] += tmp.s2;                \
        v[3] += tmp.s3;                \
        v[4] += tmp.s4;                \
    }

#define ADD_BUF_ARRAY6(v, off, buf)    \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] += tmp.s0;                \
        v[1] += tmp.s1;                \
        v[2] += tmp.s2;                \
        v[3] += tmp.s3;                \
        v[4] += tmp.s4;                \
        v[5] += tmp.s5;                \
    }

#define ADD_BUF_ARRAY7(v, off, buf)    \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] += tmp.s0;                \
        v[1] += tmp.s1;                \
        v[2] += tmp.s2;                \
        v[3] += tmp.s3;                \
        v[4] += tmp.s4;                \
        v[5] += tmp.s5;                \
        v[6] += tmp.s6;                \
    }

#define ADD_BUF_ARRAY8(v, off, buf)    \
    {                                  \
        T8 tmp = vload8(0, buf + off); \
        v[0] += tmp.s0;                \
        v[1] += tmp.s1;                \
        v[2] += tmp.s2;                \
        v[3] += tmp.s3;                \
        v[4] += tmp.s4;                \
        v[5] += tmp.s5;                \
        v[6] += tmp.s6;                \
        v[7] += tmp.s7;                \
    }
/*
 * set reg array to normal val
 */
#define SET_REG_ARRAY1(v, reg) \
    {                          \
        reg[0] = v;            \
    }

#define SET_REG_ARRAY2(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
    }

#define SET_REG_ARRAY3(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
        reg[2] = v;            \
    }

#define SET_REG_ARRAY4(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
        reg[2] = v;            \
        reg[3] = v;            \
    }
#define SET_REG_ARRAY5(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
        reg[2] = v;            \
        reg[3] = v;            \
        reg[4] = v;            \
    }

#define SET_REG_ARRAY6(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
        reg[2] = v;            \
        reg[3] = v;            \
        reg[4] = v;            \
        reg[5] = v;            \
    }

#define SET_REG_ARRAY7(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
        reg[2] = v;            \
        reg[3] = v;            \
        reg[4] = v;            \
        reg[5] = v;            \
        reg[6] = v;            \
    }

#define SET_REG_ARRAY8(v, reg) \
    {                          \
        reg[0] = v;            \
        reg[1] = v;            \
        reg[2] = v;            \
        reg[3] = v;            \
        reg[4] = v;            \
        reg[5] = v;            \
        reg[6] = v;            \
        reg[7] = v;            \
    }

#define MUL_REG_NORMAL_ARRAY1(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY2(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY3(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
        reg[2] = a * reg[2] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY4(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
        reg[2] = a * reg[2] + b;         \
        reg[3] = a * reg[3] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY5(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
        reg[2] = a * reg[2] + b;         \
        reg[3] = a * reg[3] + b;         \
        reg[4] = a * reg[4] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY6(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
        reg[2] = a * reg[2] + b;         \
        reg[3] = a * reg[3] + b;         \
        reg[4] = a * reg[4] + b;         \
        reg[5] = a * reg[5] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY7(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
        reg[2] = a * reg[2] + b;         \
        reg[3] = a * reg[3] + b;         \
        reg[4] = a * reg[4] + b;         \
        reg[5] = a * reg[5] + b;         \
        reg[6] = a * reg[6] + b;         \
    }

#define MUL_REG_NORMAL_ARRAY8(a, b, reg) \
    {                                    \
        reg[0] = a * reg[0] + b;         \
        reg[1] = a * reg[1] + b;         \
        reg[2] = a * reg[2] + b;         \
        reg[3] = a * reg[3] + b;         \
        reg[4] = a * reg[4] + b;         \
        reg[5] = a * reg[5] + b;         \
        reg[6] = a * reg[6] + b;         \
        reg[7] = a * reg[7] + b;         \
    }

#define ADD_REG_ARRAY4(reg0, reg1) \
    {                              \
        reg1[0] += reg0[0];        \
        reg1[1] += reg0[1];        \
        reg1[2] += reg0[2];        \
        reg1[3] += reg0[3];        \
    }

#define MINUS_REG_ARRAY4(reg0, reg1) \
    {                                \
        reg1[0] -= reg0[0];          \
        reg1[1] -= reg0[1];          \
        reg1[2] -= reg0[2];          \
        reg1[3] -= reg0[3];          \
    }

/*
 * DOT
 */
#define DOT_A4B16C4(a, b, c)                                        \
    {                                                               \
        c.x += (a.x * b.s0 + a.y * b.s1 + a.z * b.s2 + a.w * b.s3); \
        c.y += (a.x * b.s4 + a.y * b.s5 + a.z * b.s6 + a.w * b.s7); \
        c.z += (a.x * b.s8 + a.y * b.s9 + a.z * b.sa + a.w * b.sb); \
        c.w += (a.x * b.sc + a.y * b.sd + a.z * b.se + a.w * b.sf); \
    }

#define DOT_A4B4C1(a, b, c)                                       \
    {                                                             \
        c += (a.x * b.s0 + a.y * b.s1 + a.z * b.s2 + a.w * b.s3); \
    }

#define DOT_A4B4C4(a, b, c) \
    {                       \
        c.x += a.x * b.x;   \
        c.y += a.y * b.y;   \
        c.z += a.z * b.z;   \
        c.w += a.w * b.w;   \
    }

#define DOT_A2B2C1(a, b, c)               \
    {                                     \
        c += (a.s0 * b.s0 + a.s1 * b.s1); \
    }

#define DOT_A8B8C1(a, b, c)                                           \
    {                                                                 \
        c += (a.s0 * b.s0 + a.s1 * b.s1 + a.s2 * b.s2 + a.s3 * b.s3); \
        c += (a.s4 * b.s4 + a.s5 * b.s5 + a.s6 * b.s6 + a.s7 * b.s7); \
    }

#define DOT_A16B16C1(a, b, c)                                     \
    {                                                             \
        c += (a.x * b.s0 + a.y * b.s1 + a.z * b.s2 + a.w * b.s3); \
        c += (a.x * b.s4 + a.y * b.s5 + a.z * b.s6 + a.w * b.s7); \
        c += (a.x * b.s8 + a.y * b.s9 + a.z * b.sa + a.w * b.sb); \
        c += (a.x * b.sc + a.y * b.sd + a.z * b.se + a.w * b.sf); \
    }

#define DOT_A_NORMAL_B1C1_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
    }

#define DOT_A_NORMAL_B2C2_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
    }

#define DOT_A_NORMAL_B3C3_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
        c[2] += a * b[2];                \
    }

#define DOT_A_NORMAL_B4C4_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
        c[2] += a * b[2];                \
        c[3] += a * b[3];                \
    }

#define DOT_A_NORMAL_B5C5_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
        c[2] += a * b[2];                \
        c[3] += a * b[3];                \
        c[4] += a * b[4];                \
    }

#define DOT_A_NORMAL_B6C6_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
        c[2] += a * b[2];                \
        c[3] += a * b[3];                \
        c[4] += a * b[4];                \
        c[5] += a * b[5];                \
    }

#define DOT_A_NORMAL_B7C7_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
        c[2] += a * b[2];                \
        c[3] += a * b[3];                \
        c[4] += a * b[4];                \
        c[5] += a * b[5];                \
        c[6] += a * b[6];                \
    }

#define DOT_A_NORMAL_B8C8_ARRAY(a, b, c) \
    {                                    \
        c[0] += a * b[0];                \
        c[1] += a * b[1];                \
        c[2] += a * b[2];                \
        c[3] += a * b[3];                \
        c[4] += a * b[4];                \
        c[5] += a * b[5];                \
        c[6] += a * b[6];                \
        c[7] += a * b[7];                \
    }

#if defined(USE_V2)
#define DOT_VEC(a, b, c) DOT_A2B2C1(a, b, c)
#elif defined(USE_V4)
#define DOT_VEC(a, b, c) DOT_A4B4C1(a, b, c)
#elif defined(USE_V8)
#define DOT_VEC(a, b, c) DOT_A8B8C1(a, b, c)
#elif defined(USE_V16)
#define DOT_VEC(a, b, c) DOT_A16B16C1(a, b, c)
#else
#define DOT_A_VEC_B1C1_ARRAY(a, b, c) DOT_A_NORMAL_B1C1_ARRAY(a, b, c)
#define DOT_A_VEC_B2C2_ARRAY(a, b, c) DOT_A_NORMAL_B2C2_ARRAY(a, b, c)
#define DOT_A_VEC_B3C3_ARRAY(a, b, c) DOT_A_NORMAL_B3C3_ARRAY(a, b, c)
#define DOT_A_VEC_B4C4_ARRAY(a, b, c) DOT_A_NORMAL_B4C4_ARRAY(a, b, c)
#define DOT_A_VEC_B5C5_ARRAY(a, b, c) DOT_A_NORMAL_B5C5_ARRAY(a, b, c)
#define DOT_A_VEC_B6C6_ARRAY(a, b, c) DOT_A_NORMAL_B6C6_ARRAY(a, b, c)
#define DOT_A_VEC_B7C7_ARRAY(a, b, c) DOT_A_NORMAL_B7C7_ARRAY(a, b, c)
#define DOT_A_VEC_B8C8_ARRAY(a, b, c) DOT_A_NORMAL_B8C8_ARRAY(a, b, c)
#endif

#if defined(USE_V2) || defined(USE_V4) || defined(USE_V8) || defined(USE_V16)
#define DOT_A_VEC_B1C1_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
    }

#define DOT_A_VEC_B2C2_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
    }

#define DOT_A_VEC_B3C3_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
        DOT_VEC(a, b[2], c[2]);       \
    }

#define DOT_A_VEC_B4C4_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
        DOT_VEC(a, b[2], c[2]);       \
        DOT_VEC(a, b[3], c[3]);       \
    }

#define DOT_A_VEC_B5C5_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
        DOT_VEC(a, b[2], c[2]);       \
        DOT_VEC(a, b[3], c[3]);       \
        DOT_VEC(a, b[4], c[4]);       \
    }

#define DOT_A_VEC_B6C6_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
        DOT_VEC(a, b[2], c[2]);       \
        DOT_VEC(a, b[3], c[3]);       \
        DOT_VEC(a, b[4], c[4]);       \
        DOT_VEC(a, b[5], c[5]);       \
    }

#define DOT_A_VEC_B7C7_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
        DOT_VEC(a, b[2], c[2]);       \
        DOT_VEC(a, b[3], c[3]);       \
        DOT_VEC(a, b[4], c[4]);       \
        DOT_VEC(a, b[5], c[5]);       \
        DOT_VEC(a, b[6], c[6]);       \
    }

#define DOT_A_VEC_B8C8_ARRAY(a, b, c) \
    {                                 \
        DOT_VEC(a, b[0], c[0]);       \
        DOT_VEC(a, b[1], c[1]);       \
        DOT_VEC(a, b[2], c[2]);       \
        DOT_VEC(a, b[3], c[3]);       \
        DOT_VEC(a, b[4], c[4]);       \
        DOT_VEC(a, b[5], c[5]);       \
        DOT_VEC(a, b[6], c[6]);       \
        DOT_VEC(a, b[7], c[7]);       \
    }
#endif
/*
 * ACTIVATION
 */
#if defined(USE_RELU)
#define ACTIVATION_V4(v)    \
    {                       \
        v.x = fmax(v.x, 0); \
        v.y = fmax(v.y, 0); \
        v.z = fmax(v.z, 0); \
        v.w = fmax(v.w, 0); \
    }

#define ACTIVATION_V8(v)      \
    {                         \
        v.s0 = fmax(v.s0, 0); \
        v.s1 = fmax(v.s1, 0); \
        v.s2 = fmax(v.s2, 0); \
        v.s3 = fmax(v.s3, 0); \
        v.s4 = fmax(v.s4, 0); \
        v.s5 = fmax(v.s5, 0); \
        v.s6 = fmax(v.s6, 0); \
        v.s7 = fmax(v.s7, 0); \
    }

#define ACTIVATION_V16(v)     \
    {                         \
        v.s0 = fmax(v.s0, 0); \
        v.s1 = fmax(v.s1, 0); \
        v.s2 = fmax(v.s2, 0); \
        v.s3 = fmax(v.s3, 0); \
        v.s4 = fmax(v.s4, 0); \
        v.s5 = fmax(v.s5, 0); \
        v.s6 = fmax(v.s6, 0); \
        v.s7 = fmax(v.s7, 0); \
        v.s8 = fmax(v.s8, 0); \
        v.s9 = fmax(v.s9, 0); \
        v.sa = fmax(v.sa, 0); \
        v.sb = fmax(v.sb, 0); \
        v.sc = fmax(v.sc, 0); \
        v.sd = fmax(v.sd, 0); \
        v.se = fmax(v.se, 0); \
        v.sf = fmax(v.sf, 0); \
    }

#define ACTIVATION_V1(v) \
    {                    \
        v = fmax(v, 0);  \
    }

#define ACTIVATION_ARRAY1(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
    }

#define ACTIVATION_ARRAY2(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
    }

#define ACTIVATION_ARRAY3(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
        v[2] = fmax(v[2], 0); \
    }

#define ACTIVATION_ARRAY4(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
        v[2] = fmax(v[2], 0); \
        v[3] = fmax(v[3], 0); \
    }

#define ACTIVATION_ARRAY5(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
        v[2] = fmax(v[2], 0); \
        v[3] = fmax(v[3], 0); \
        v[4] = fmax(v[4], 0); \
    }

#define ACTIVATION_ARRAY6(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
        v[2] = fmax(v[2], 0); \
        v[3] = fmax(v[3], 0); \
        v[4] = fmax(v[4], 0); \
        v[5] = fmax(v[5], 0); \
    }

#define ACTIVATION_ARRAY7(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
        v[2] = fmax(v[2], 0); \
        v[3] = fmax(v[3], 0); \
        v[4] = fmax(v[4], 0); \
        v[5] = fmax(v[5], 0); \
        v[6] = fmax(v[6], 0); \
    }

#define ACTIVATION_ARRAY8(v)  \
    {                         \
        v[0] = fmax(v[0], 0); \
        v[1] = fmax(v[1], 0); \
        v[2] = fmax(v[2], 0); \
        v[3] = fmax(v[3], 0); \
        v[4] = fmax(v[4], 0); \
        v[5] = fmax(v[5], 0); \
        v[6] = fmax(v[6], 0); \
        v[7] = fmax(v[7], 0); \
    }
#elif defined(USE_RELU6)
#define ACTIVATION_V4(v)              \
    {                                 \
        v.x = clamp(v.x, (T)0, (T)6); \
        v.y = clamp(v.y, (T)0, (T)6); \
        v.z = clamp(v.z, (T)0, (T)6); \
        v.w = clamp(v.w, (T)0, (T)6); \
    }

#define ACTIVATION_V8(v)                \
    {                                   \
        v.s0 = clamp(v.s0, (T)0, (T)6); \
        v.s1 = clamp(v.s1, (T)0, (T)6); \
        v.s2 = clamp(v.s2, (T)0, (T)6); \
        v.s3 = clamp(v.s3, (T)0, (T)6); \
        v.s4 = clamp(v.s4, (T)0, (T)6); \
        v.s5 = clamp(v.s5, (T)0, (T)6); \
        v.s6 = clamp(v.s6, (T)0, (T)6); \
        v.s7 = clamp(v.s7, (T)0, (T)6); \
    }

#define ACTIVATION_V16(v)               \
    {                                   \
        v.s0 = clamp(v.s0, (T)0, (T)6); \
        v.s1 = clamp(v.s1, (T)0, (T)6); \
        v.s2 = clamp(v.s2, (T)0, (T)6); \
        v.s3 = clamp(v.s3, (T)0, (T)6); \
        v.s4 = clamp(v.s4, (T)0, (T)6); \
        v.s5 = clamp(v.s5, (T)0, (T)6); \
        v.s6 = clamp(v.s6, (T)0, (T)6); \
        v.s7 = clamp(v.s7, (T)0, (T)6); \
        v.s8 = clamp(v.s0, (T)0, (T)6); \
        v.s9 = clamp(v.s1, (T)0, (T)6); \
        v.sa = clamp(v.sa, (T)0, (T)6); \
        v.sb = clamp(v.sb, (T)0, (T)6); \
        v.sc = clamp(v.sc, (T)0, (T)6); \
        v.sd = clamp(v.sd, (T)0, (T)6); \
        v.se = clamp(v.se, (T)0, (T)6); \
        v.sf = clamp(v.sf, (T)0, (T)6); \
    }

#define ACTIVATION_V1(v)          \
    {                             \
        v = clamp(v, (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY1(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY2(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY3(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
        v[2] = clamp(v[2], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY4(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
        v[2] = clamp(v[2], (T)0, (T)6); \
        v[3] = clamp(v[3], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY5(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
        v[2] = clamp(v[2], (T)0, (T)6); \
        v[3] = clamp(v[3], (T)0, (T)6); \
        v[4] = clamp(v[4], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY6(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
        v[2] = clamp(v[2], (T)0, (T)6); \
        v[3] = clamp(v[3], (T)0, (T)6); \
        v[4] = clamp(v[4], (T)0, (T)6); \
        v[5] = clamp(v[5], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY7(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
        v[2] = clamp(v[2], (T)0, (T)6); \
        v[3] = clamp(v[3], (T)0, (T)6); \
        v[4] = clamp(v[4], (T)0, (T)6); \
        v[5] = clamp(v[5], (T)0, (T)6); \
        v[6] = clamp(v[6], (T)0, (T)6); \
    }

#define ACTIVATION_ARRAY8(v)            \
    {                                   \
        v[0] = clamp(v[0], (T)0, (T)6); \
        v[1] = clamp(v[1], (T)0, (T)6); \
        v[2] = clamp(v[2], (T)0, (T)6); \
        v[3] = clamp(v[3], (T)0, (T)6); \
        v[4] = clamp(v[4], (T)0, (T)6); \
        v[5] = clamp(v[5], (T)0, (T)6); \
        v[6] = clamp(v[6], (T)0, (T)6); \
        v[7] = clamp(v[7], (T)0, (T)6); \
    }
#elif defined(USE_GELU)
#define ACTIVATION_V4(v)                                     \
    {                                                        \
        T4 tmp = v;                                          \
        v.s0 = 0.797885 * (v.s0 + 0.044715 * pown(v.s0, 3)); \
        v.s1 = 0.797885 * (v.s1 + 0.044715 * pown(v.s1, 3)); \
        v.s2 = 0.797885 * (v.s2 + 0.044715 * pown(v.s2, 3)); \
        v.s3 = 0.797885 * (v.s3 + 0.044715 * pown(v.s3, 3)); \
        v.s0 = 1.0 - 2.0 / (exp(2.0 * v.s0) + 1.0);          \
        v.s1 = 1.0 - 2.0 / (exp(2.0 * v.s1) + 1.0);          \
        v.s2 = 1.0 - 2.0 / (exp(2.0 * v.s2) + 1.0);          \
        v.s3 = 1.0 - 2.0 / (exp(2.0 * v.s3) + 1.0);          \
        v.s0 = (v.s0 + (T)1.0) * (T)0.5;                     \
        v.s1 = (v.s1 + (T)1.0) * (T)0.5;                     \
        v.s2 = (v.s2 + (T)1.0) * (T)0.5;                     \
        v.s3 = (v.s3 + (T)1.0) * (T)0.5;                     \
        v.s0 = v.s0 * tmp.s0;                                \
        v.s1 = v.s1 * tmp.s1;                                \
        v.s2 = v.s2 * tmp.s2;                                \
        v.s3 = v.s3 * tmp.s3;                                \
    }

#define ACTIVATION_V8(v)                                     \
    {                                                        \
        T8 tmp = v;                                          \
        v.s0 = 0.797885 * (v.s0 + 0.044715 * pown(v.s0, 3)); \
        v.s1 = 0.797885 * (v.s1 + 0.044715 * pown(v.s1, 3)); \
        v.s2 = 0.797885 * (v.s2 + 0.044715 * pown(v.s2, 3)); \
        v.s3 = 0.797885 * (v.s3 + 0.044715 * pown(v.s3, 3)); \
        v.s4 = 0.797885 * (v.s4 + 0.044715 * pown(v.s4, 3)); \
        v.s5 = 0.797885 * (v.s5 + 0.044715 * pown(v.s5, 3)); \
        v.s6 = 0.797885 * (v.s6 + 0.044715 * pown(v.s6, 3)); \
        v.s7 = 0.797885 * (v.s7 + 0.044715 * pown(v.s7, 3)); \
        v.s0 = 1.0 - 2.0 / (exp(2.0 * v.s0) + 1.0);          \
        v.s1 = 1.0 - 2.0 / (exp(2.0 * v.s1) + 1.0);          \
        v.s2 = 1.0 - 2.0 / (exp(2.0 * v.s2) + 1.0);          \
        v.s3 = 1.0 - 2.0 / (exp(2.0 * v.s3) + 1.0);          \
        v.s4 = 1.0 - 2.0 / (exp(2.0 * v.s4) + 1.0);          \
        v.s5 = 1.0 - 2.0 / (exp(2.0 * v.s5) + 1.0);          \
        v.s6 = 1.0 - 2.0 / (exp(2.0 * v.s6) + 1.0);          \
        v.s7 = 1.0 - 2.0 / (exp(2.0 * v.s7) + 1.0);          \
        v.s0 = (v.s0 + (T)1.0) * (T)0.5;                     \
        v.s1 = (v.s1 + (T)1.0) * (T)0.5;                     \
        v.s2 = (v.s2 + (T)1.0) * (T)0.5;                     \
        v.s3 = (v.s3 + (T)1.0) * (T)0.5;                     \
        v.s4 = (v.s4 + (T)1.0) * (T)0.5;                     \
        v.s5 = (v.s5 + (T)1.0) * (T)0.5;                     \
        v.s6 = (v.s6 + (T)1.0) * (T)0.5;                     \
        v.s7 = (v.s7 + (T)1.0) * (T)0.5;                     \
        v.s0 = v.s0 * tmp.s0;                                \
        v.s1 = v.s1 * tmp.s1;                                \
        v.s2 = v.s2 * tmp.s2;                                \
        v.s3 = v.s3 * tmp.s3;                                \
        v.s4 = v.s4 * tmp.s4;                                \
        v.s5 = v.s5 * tmp.s5;                                \
        v.s6 = v.s6 * tmp.s6;                                \
        v.s7 = v.s7 * tmp.s7;                                \
    }

#define ACTIVATION_V16(v)                                    \
    {                                                        \
        T16 tmp = v;                                         \
        v.s0 = 0.797885 * (v.s0 + 0.044715 * pown(v.s0, 3)); \
        v.s1 = 0.797885 * (v.s1 + 0.044715 * pown(v.s1, 3)); \
        v.s2 = 0.797885 * (v.s2 + 0.044715 * pown(v.s2, 3)); \
        v.s3 = 0.797885 * (v.s3 + 0.044715 * pown(v.s3, 3)); \
        v.s4 = 0.797885 * (v.s4 + 0.044715 * pown(v.s4, 3)); \
        v.s5 = 0.797885 * (v.s5 + 0.044715 * pown(v.s5, 3)); \
        v.s6 = 0.797885 * (v.s6 + 0.044715 * pown(v.s6, 3)); \
        v.s7 = 0.797885 * (v.s7 + 0.044715 * pown(v.s7, 3)); \
        v.s8 = 0.797885 * (v.s8 + 0.044715 * pown(v.s8, 3)); \
        v.s9 = 0.797885 * (v.s9 + 0.044715 * pown(v.s9, 3)); \
        v.sa = 0.797885 * (v.sa + 0.044715 * pown(v.sa, 3)); \
        v.sb = 0.797885 * (v.sb + 0.044715 * pown(v.sb, 3)); \
        v.sc = 0.797885 * (v.sc + 0.044715 * pown(v.sc, 3)); \
        v.sd = 0.797885 * (v.sd + 0.044715 * pown(v.sd, 3)); \
        v.se = 0.797885 * (v.se + 0.044715 * pown(v.se, 3)); \
        v.sf = 0.797885 * (v.sf + 0.044715 * pown(v.sf, 3)); \
        v.s0 = 1.0 - 2.0 / (exp(2.0 * v.s0) + 1.0);          \
        v.s1 = 1.0 - 2.0 / (exp(2.0 * v.s1) + 1.0);          \
        v.s2 = 1.0 - 2.0 / (exp(2.0 * v.s2) + 1.0);          \
        v.s3 = 1.0 - 2.0 / (exp(2.0 * v.s3) + 1.0);          \
        v.s4 = 1.0 - 2.0 / (exp(2.0 * v.s4) + 1.0);          \
        v.s5 = 1.0 - 2.0 / (exp(2.0 * v.s5) + 1.0);          \
        v.s6 = 1.0 - 2.0 / (exp(2.0 * v.s6) + 1.0);          \
        v.s7 = 1.0 - 2.0 / (exp(2.0 * v.s7) + 1.0);          \
        v.s8 = 1.0 - 2.0 / (exp(2.0 * v.s8) + 1.0);          \
        v.s9 = 1.0 - 2.0 / (exp(2.0 * v.s9) + 1.0);          \
        v.sa = 1.0 - 2.0 / (exp(2.0 * v.sa) + 1.0);          \
        v.sb = 1.0 - 2.0 / (exp(2.0 * v.sb) + 1.0);          \
        v.sc = 1.0 - 2.0 / (exp(2.0 * v.sc) + 1.0);          \
        v.sd = 1.0 - 2.0 / (exp(2.0 * v.sd) + 1.0);          \
        v.se = 1.0 - 2.0 / (exp(2.0 * v.se) + 1.0);          \
        v.sf = 1.0 - 2.0 / (exp(2.0 * v.sf) + 1.0);          \
        v.s0 = (v.s0 + (T)1.0) * (T)0.5;                     \
        v.s1 = (v.s1 + (T)1.0) * (T)0.5;                     \
        v.s2 = (v.s2 + (T)1.0) * (T)0.5;                     \
        v.s3 = (v.s3 + (T)1.0) * (T)0.5;                     \
        v.s4 = (v.s4 + (T)1.0) * (T)0.5;                     \
        v.s5 = (v.s5 + (T)1.0) * (T)0.5;                     \
        v.s6 = (v.s6 + (T)1.0) * (T)0.5;                     \
        v.s7 = (v.s7 + (T)1.0) * (T)0.5;                     \
        v.s8 = (v.s8 + (T)1.0) * (T)0.5;                     \
        v.s9 = (v.s9 + (T)1.0) * (T)0.5;                     \
        v.sa = (v.sa + (T)1.0) * (T)0.5;                     \
        v.sb = (v.sb + (T)1.0) * (T)0.5;                     \
        v.sc = (v.sc + (T)1.0) * (T)0.5;                     \
        v.sd = (v.sd + (T)1.0) * (T)0.5;                     \
        v.se = (v.se + (T)1.0) * (T)0.5;                     \
        v.sf = (v.sf + (T)1.0) * (T)0.5;                     \
        v.s0 = v.s0 * tmp.s0;                                \
        v.s1 = v.s1 * tmp.s1;                                \
        v.s2 = v.s2 * tmp.s2;                                \
        v.s3 = v.s3 * tmp.s3;                                \
        v.s4 = v.s4 * tmp.s4;                                \
        v.s5 = v.s5 * tmp.s5;                                \
        v.s6 = v.s6 * tmp.s6;                                \
        v.s7 = v.s7 * tmp.s7;                                \
        v.s8 = v.s8 * tmp.s8;                                \
        v.s9 = v.s9 * tmp.s9;                                \
        v.sa = v.sa * tmp.sa;                                \
        v.sb = v.sb * tmp.sb;                                \
        v.sc = v.sc * tmp.sc;                                \
        v.sd = v.sd * tmp.sd;                                \
        v.se = v.se * tmp.se;                                \
        v.sf = v.sf * tmp.sf;                                \
    }

#define ACTIVATION_ARRAY1(v)                                 \
    {                                                        \
        T tmp = v[0];                                        \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp;                                   \
    }

#define ACTIVATION_ARRAY2(v)                                 \
    {                                                        \
        T tmp[2];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
    }

#define ACTIVATION_ARRAY3(v)                                 \
    {                                                        \
        T tmp[3];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        tmp[2] = v[2];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[2] = 0.797885 * (v[2] + 0.044715 * pown(v[2], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[2] = 1.0 - 2.0 / (exp(2.0 * v[2]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[2] = (v[2] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
        v[2] = v[2] * tmp[2];                                \
    }

#define ACTIVATION_ARRAY4(v)                                 \
    {                                                        \
        T tmp[4];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        tmp[2] = v[2];                                       \
        tmp[3] = v[3];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[2] = 0.797885 * (v[2] + 0.044715 * pown(v[2], 3)); \
        v[3] = 0.797885 * (v[3] + 0.044715 * pown(v[3], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[2] = 1.0 - 2.0 / (exp(2.0 * v[2]) + 1.0);          \
        v[3] = 1.0 - 2.0 / (exp(2.0 * v[3]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[2] = (v[2] + (T)1.0) * (T)0.5;                     \
        v[3] = (v[3] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
        v[2] = v[2] * tmp[2];                                \
        v[3] = v[3] * tmp[3];                                \
    }

#define ACTIVATION_ARRAY5(v)                                 \
    {                                                        \
        T tmp[5];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        tmp[2] = v[2];                                       \
        tmp[3] = v[3];                                       \
        tmp[4] = v[4];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[2] = 0.797885 * (v[2] + 0.044715 * pown(v[2], 3)); \
        v[3] = 0.797885 * (v[3] + 0.044715 * pown(v[3], 3)); \
        v[4] = 0.797885 * (v[4] + 0.044715 * pown(v[4], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[2] = 1.0 - 2.0 / (exp(2.0 * v[2]) + 1.0);          \
        v[3] = 1.0 - 2.0 / (exp(2.0 * v[3]) + 1.0);          \
        v[4] = 1.0 - 2.0 / (exp(2.0 * v[4]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[2] = (v[2] + (T)1.0) * (T)0.5;                     \
        v[3] = (v[3] + (T)1.0) * (T)0.5;                     \
        v[4] = (v[4] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
        v[2] = v[2] * tmp[2];                                \
        v[3] = v[3] * tmp[3];                                \
        v[4] = v[4] * tmp[4];                                \
    }

#define ACTIVATION_ARRAY6(v)                                 \
    {                                                        \
        T tmp[6];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        tmp[2] = v[2];                                       \
        tmp[3] = v[3];                                       \
        tmp[4] = v[4];                                       \
        tmp[5] = v[5];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[2] = 0.797885 * (v[2] + 0.044715 * pown(v[2], 3)); \
        v[3] = 0.797885 * (v[3] + 0.044715 * pown(v[3], 3)); \
        v[4] = 0.797885 * (v[4] + 0.044715 * pown(v[4], 3)); \
        v[5] = 0.797885 * (v[5] + 0.044715 * pown(v[5], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[2] = 1.0 - 2.0 / (exp(2.0 * v[2]) + 1.0);          \
        v[3] = 1.0 - 2.0 / (exp(2.0 * v[3]) + 1.0);          \
        v[4] = 1.0 - 2.0 / (exp(2.0 * v[4]) + 1.0);          \
        v[5] = 1.0 - 2.0 / (exp(2.0 * v[5]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[2] = (v[2] + (T)1.0) * (T)0.5;                     \
        v[3] = (v[3] + (T)1.0) * (T)0.5;                     \
        v[4] = (v[4] + (T)1.0) * (T)0.5;                     \
        v[5] = (v[5] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
        v[2] = v[2] * tmp[2];                                \
        v[3] = v[3] * tmp[3];                                \
        v[4] = v[4] * tmp[4];                                \
        v[5] = v[5] * tmp[5];                                \
    }

#define ACTIVATION_ARRAY7(v)                                 \
    {                                                        \
        T tmp[7];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        tmp[2] = v[2];                                       \
        tmp[3] = v[3];                                       \
        tmp[4] = v[4];                                       \
        tmp[5] = v[5];                                       \
        tmp[6] = v[6];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[2] = 0.797885 * (v[2] + 0.044715 * pown(v[2], 3)); \
        v[3] = 0.797885 * (v[3] + 0.044715 * pown(v[3], 3)); \
        v[4] = 0.797885 * (v[4] + 0.044715 * pown(v[4], 3)); \
        v[5] = 0.797885 * (v[5] + 0.044715 * pown(v[5], 3)); \
        v[6] = 0.797885 * (v[6] + 0.044715 * pown(v[6], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[2] = 1.0 - 2.0 / (exp(2.0 * v[2]) + 1.0);          \
        v[3] = 1.0 - 2.0 / (exp(2.0 * v[3]) + 1.0);          \
        v[4] = 1.0 - 2.0 / (exp(2.0 * v[4]) + 1.0);          \
        v[5] = 1.0 - 2.0 / (exp(2.0 * v[5]) + 1.0);          \
        v[6] = 1.0 - 2.0 / (exp(2.0 * v[6]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[2] = (v[2] + (T)1.0) * (T)0.5;                     \
        v[3] = (v[3] + (T)1.0) * (T)0.5;                     \
        v[4] = (v[4] + (T)1.0) * (T)0.5;                     \
        v[5] = (v[5] + (T)1.0) * (T)0.5;                     \
        v[6] = (v[6] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
        v[2] = v[2] * tmp[2];                                \
        v[3] = v[3] * tmp[3];                                \
        v[4] = v[4] * tmp[4];                                \
        v[5] = v[5] * tmp[5];                                \
        v[6] = v[6] * tmp[6];                                \
    }

#define ACTIVATION_ARRAY8(v)                                 \
    {                                                        \
        T tmp[8];                                            \
        tmp[0] = v[0];                                       \
        tmp[1] = v[1];                                       \
        tmp[2] = v[2];                                       \
        tmp[3] = v[3];                                       \
        tmp[4] = v[4];                                       \
        tmp[5] = v[5];                                       \
        tmp[6] = v[6];                                       \
        tmp[7] = v[7];                                       \
        v[0] = 0.797885 * (v[0] + 0.044715 * pown(v[0], 3)); \
        v[1] = 0.797885 * (v[1] + 0.044715 * pown(v[1], 3)); \
        v[2] = 0.797885 * (v[2] + 0.044715 * pown(v[2], 3)); \
        v[3] = 0.797885 * (v[3] + 0.044715 * pown(v[3], 3)); \
        v[4] = 0.797885 * (v[4] + 0.044715 * pown(v[4], 3)); \
        v[5] = 0.797885 * (v[5] + 0.044715 * pown(v[5], 3)); \
        v[6] = 0.797885 * (v[6] + 0.044715 * pown(v[6], 3)); \
        v[7] = 0.797885 * (v[7] + 0.044715 * pown(v[7], 3)); \
        v[0] = 1.0 - 2.0 / (exp(2.0 * v[0]) + 1.0);          \
        v[1] = 1.0 - 2.0 / (exp(2.0 * v[1]) + 1.0);          \
        v[2] = 1.0 - 2.0 / (exp(2.0 * v[2]) + 1.0);          \
        v[3] = 1.0 - 2.0 / (exp(2.0 * v[3]) + 1.0);          \
        v[4] = 1.0 - 2.0 / (exp(2.0 * v[4]) + 1.0);          \
        v[5] = 1.0 - 2.0 / (exp(2.0 * v[5]) + 1.0);          \
        v[6] = 1.0 - 2.0 / (exp(2.0 * v[6]) + 1.0);          \
        v[7] = 1.0 - 2.0 / (exp(2.0 * v[7]) + 1.0);          \
        v[0] = (v[0] + (T)1.0) * (T)0.5;                     \
        v[1] = (v[1] + (T)1.0) * (T)0.5;                     \
        v[2] = (v[2] + (T)1.0) * (T)0.5;                     \
        v[3] = (v[3] + (T)1.0) * (T)0.5;                     \
        v[4] = (v[4] + (T)1.0) * (T)0.5;                     \
        v[5] = (v[5] + (T)1.0) * (T)0.5;                     \
        v[6] = (v[6] + (T)1.0) * (T)0.5;                     \
        v[7] = (v[7] + (T)1.0) * (T)0.5;                     \
        v[0] = v[0] * tmp[0];                                \
        v[1] = v[1] * tmp[1];                                \
        v[2] = v[2] * tmp[2];                                \
        v[3] = v[3] * tmp[3];                                \
        v[4] = v[4] * tmp[4];                                \
        v[5] = v[5] * tmp[5];                                \
        v[6] = v[6] * tmp[6];                                \
        v[7] = v[7] * tmp[7];                                \
    }
#elif defined(USE_HSIGMOID)
#define ACTIVATION_V4(v)                  \
    {                                     \
        v.s0 = v.s0 + (T)3.0;             \
        v.s1 = v.s1 + (T)3.0;             \
        v.s2 = v.s2 + (T)3.0;             \
        v.s3 = v.s3 + (T)3.0;             \
        v.s0 = clamp(v.s0, (T)0, (T)6.0); \
        v.s1 = clamp(v.s1, (T)0, (T)6.0); \
        v.s2 = clamp(v.s2, (T)0, (T)6.0); \
        v.s3 = clamp(v.s3, (T)0, (T)6.0); \
        v.s0 = v.s0 * 0.166667;           \
        v.s1 = v.s1 * 0.166667;           \
        v.s2 = v.s2 * 0.166667;           \
        v.s3 = v.s3 * 0.166667;           \
    }
#elif defined(USE_HSWISH)
#define ACTIVATION_V4(v)                   \
    {                                      \
        T4 tmp = v;                        \
        v.s0 = v.s0 + (T)3.0;              \
        v.s1 = v.s1 + (T)3.0;              \
        v.s2 = v.s2 + (T)3.0;              \
        v.s3 = v.s3 + (T)3.0;              \
        v.s0 = clamp(v.s0, (T)0, (T)6.0);  \
        v.s1 = clamp(v.s1, (T)0, (T)6.0);  \
        v.s2 = clamp(v.s2, (T)0, (T)6.0);  \
        v.s3 = clamp(v.s3, (T)0, (T)6.0);  \
        v.s0 = tmp.s0 * (v.s0 * 0.166667); \
        v.s1 = tmp.s1 * (v.s1 * 0.166667); \
        v.s2 = tmp.s2 * (v.s2 * 0.166667); \
        v.s3 = tmp.s3 * (v.s3 * 0.166667); \
    }
#elif defined(USE_TANH)
#define ACTIVATION_V4(v)                            \
    {                                               \
        v.s0 = 1.0 - 2.0 / (exp(2.0 * v.s0) + 1.0); \
        v.s1 = 1.0 - 2.0 / (exp(2.0 * v.s1) + 1.0); \
        v.s2 = 1.0 - 2.0 / (exp(2.0 * v.s2) + 1.0); \
        v.s3 = 1.0 - 2.0 / (exp(2.0 * v.s3) + 1.0); \
    }
#elif defined(USE_SIGMOID)
#define ACTIVATION_V4(v)                       \
    {                                          \
        v.s0 = 1.0 / (1.0 + exp(-1.0 * v.s0)); \
        v.s1 = 1.0 / (1.0 + exp(-1.0 * v.s1)); \
        v.s2 = 1.0 / (1.0 + exp(-1.0 * v.s2)); \
        v.s3 = 1.0 / (1.0 + exp(-1.0 * v.s3)); \
    }
#elif defined(USE_ABS)
#define ACTIVATION_V4(v)                       \
    {                                          \
        v.s0 = fabs(v.s0); \
        v.s1 = fabs(v.s1); \
        v.s2 = fabs(v.s2); \
        v.s3 = fabs(v.s3); \
    }
#else
#define ACTIVATION_V1(v) \
    {}

#define ACTIVATION_V4(v) \
    {}

#define ACTIVATION_V8(v) \
    {}

#define ACTIVATION_V16(v) \
    {}

#define ACTIVATION_ARRAY1(v) \
    {}

#define ACTIVATION_ARRAY2(v) \
    {}

#define ACTIVATION_ARRAY3(v) \
    {}

#define ACTIVATION_ARRAY4(v) \
    {}

#define ACTIVATION_ARRAY5(v) \
    {}

#define ACTIVATION_ARRAY6(v) \
    {}

#define ACTIVATION_ARRAY7(v) \
    {}

#define ACTIVATION_ARRAY8(v) \
    {}
#endif
/*
* eltwise
*/
#if defined(USE_SUM)
#define ELTWISE_V4(v, res) \
    {                      \
        res.s0 += v.s0;    \
        res.s1 += v.s1;    \
        res.s2 += v.s2;    \
        res.s3 += v.s3;    \
    }
#elif defined(USE_SUB)
#define ELTWISE_V4(v, res) \
    {                      \
        res.s0 -= v.s0;    \
        res.s1 -= v.s1;    \
        res.s2 -= v.s2;    \
        res.s3 -= v.s3;    \
    }
#elif defined(USE_PROD)
#define ELTWISE_V4(v, res) \
    {                      \
        res.s0 *= v.s0;    \
        res.s1 *= v.s1;    \
        res.s2 *= v.s2;    \
        res.s3 *= v.s3;    \
    }
#elif defined(USE_DIV)
#define ELTWISE_V4(v, res) \
    {                      \
        res.s0 /= v.s0;    \
        res.s1 /= v.s1;    \
        res.s2 /= v.s2;    \
        res.s3 /= v.s3;    \
    }
#elif defined(USE_MAX)
#define ELTWISE_V4(v, res)  \
    {                       \
        res = fmax(res, v); \
    }
#elif defined(USE_MIN)
#define ELTWISE_V4(v, res)  \
    {                       \
        res = fmin(res, v); \
    }
#endif

/*
 * store data reg array to buffer
 */
#define STORE_BUF_ARRAY1(v, off, buf) \
    {                                 \
        ACTIVATION_ARRAY1(v);         \
        buf[off] = v[0];              \
    }

#define STORE_BUF_ARRAY2(v, off, buf)            \
    {                                            \
        ACTIVATION_ARRAY2(v);                    \
        vstore2((T2)(v[0], v[1]), 0, buf + off); \
    }

#define STORE_BUF_ARRAY3(v, off, buf)                  \
    {                                                  \
        ACTIVATION_ARRAY3(v);                          \
        vstore3((T3)(v[0], v[1], v[2]), 0, buf + off); \
    }

#define STORE_BUF_ARRAY4(v, off, buf)                        \
    {                                                        \
        ACTIVATION_ARRAY4(v);                                \
        vstore4((T4)(v[0], v[1], v[2], v[3]), 0, buf + off); \
    }

#define STORE_BUF_ARRAY5(v, off, buf)                        \
    {                                                        \
        ACTIVATION_ARRAY5(v);                                \
        vstore4((T4)(v[0], v[1], v[2], v[3]), 0, buf + off); \
        buf[off + 4] = v[4];                                 \
    }

#define STORE_BUF_ARRAY6(v, off, buf)                      \
    {                                                      \
        ACTIVATION_ARRAY6(v);                              \
        vstore3((T3)(v[0], v[1], v[2]), 0, buf + off);     \
        vstore3((T3)(v[3], v[4], v[5]), 0, buf + off + 3); \
    }

#define STORE_BUF_ARRAY7(v, off, buf)                        \
    {                                                        \
        ACTIVATION_ARRAY7(v);                                \
        vstore4((T4)(v[0], v[1], v[2], v[3]), 0, buf + off); \
        vstore3((T3)(v[4], v[5], v[6]), 0, buf + off + 4);   \
    }

#define STORE_BUF_ARRAY8(v, off, buf)                                                \
    {                                                                                \
        ACTIVATION_ARRAY8(v);                                                        \
        vstore8((T8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]), 0, buf + off); \
    }
/*
 * LOAD BIAS
 * Load bias from image 1D  based on out number
 * ON is out number
 */

#if (ON == 1)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
    }
#elif (ON == 2)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
    }
#elif (ON == 3)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
    }
#elif (ON == 4)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
    }
#elif (ON == 5)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
    }
#elif (ON == 6)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
    }
#elif (ON == 7)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
        V[6] = V[0];                         \
    }
#elif (ON == 8)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
        V[6] = V[0];                         \
        V[7] = V[0];                         \
    }
#elif (ON == 9)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
        V[6] = V[0];                         \
        V[7] = V[0];                         \
        V[8] = V[0];                         \
    }
#elif (ON == 10)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
        V[6] = V[0];                         \
        V[7] = V[0];                         \
        V[8] = V[0];                         \
        V[9] = V[0];                         \
    }
#elif (ON == 11)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
        V[6] = V[0];                         \
        V[7] = V[0];                         \
        V[8] = V[0];                         \
        V[9] = V[0];                         \
        V[10] = V[0];                        \
    }
#elif (ON == 12)
#define LOADBIAS_IMAGE_ARRAY_V4(V, id, img)  \
    {                                        \
        V[0] = READ_IMAGE(img, sampler, id); \
        V[1] = V[0];                         \
        V[2] = V[0];                         \
        V[3] = V[0];                         \
        V[4] = V[0];                         \
        V[5] = V[0];                         \
        V[6] = V[0];                         \
        V[7] = V[0];                         \
        V[8] = V[0];                         \
        V[9] = V[0];                         \
        V[10] = V[0];                        \
        V[11] = V[0];                        \
    }
#endif

/*
 * LOAD INPUT
 * load input from buffer based on len of array vector 4
 * len = N;
 * N is usually associated with number W
 *
 *
 * GEMM TN A x B = C
 * Matrix A has been transposed
 * Operator define for Matrix B and Matrix C
 */
#if (LN == 0)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {}
#elif (LN == 1)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY1(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {}

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY1(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY1(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B1C1_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY1(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf) \
    {                                   \
        STORE_BUF_ARRAY1(v, off, buf);  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf) \
    {                                    \
        READ_BUF(v[0], off, buf);        \
    }
#elif (LN == 2)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY2(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        reg[1] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY2(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY2(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B2C2_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY2(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 1) {                      \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf) \
    {                                    \
        READ_BUF(v[0], off, buf);        \
        READ_BUF(v[1], off + str, buf);  \
    }
#elif (LN == 3)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY3(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
        reg[2] = v[2];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        if (ex > 1)                     \
            reg[1] = 0;                 \
        reg[2] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY3(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY3(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B3C3_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY3(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 2) {                      \
            STORE_BUF_ARRAY3(v, off, buf); \
        } else if (ex > 1) {               \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
    }
#elif (LN == 4)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY4(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
        reg[2] = v[2];              \
        reg[3] = v[3];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        if (ex > 2)                     \
            reg[1] = 0;                 \
        if (ex > 1)                     \
            reg[2] = 0;                 \
        reg[3] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY4(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY4(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B4C4_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY4(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 3) {                      \
            STORE_BUF_ARRAY4(v, off, buf); \
        } else if (ex > 2) {               \
            STORE_BUF_ARRAY3(v, off, buf); \
        } else if (ex > 1) {               \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
    }
#elif (LN == 5)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY5(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
        reg[2] = v[2];              \
        reg[3] = v[3];              \
        reg[4] = v[4];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        if (ex > 3)                     \
            reg[1] = 0;                 \
        if (ex > 2)                     \
            reg[2] = 0;                 \
        if (ex > 1)                     \
            reg[3] = 0;                 \
        reg[4] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY5(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY5(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B5C5_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY5(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 4) {                      \
            STORE_BUF_ARRAY5(v, off, buf); \
        } else if (ex > 3) {               \
            STORE_BUF_ARRAY4(v, off, buf); \
        } else if (ex > 2) {               \
            STORE_BUF_ARRAY3(v, off, buf); \
        } else if (ex > 1) {               \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
    }
#elif (LN == 6)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY6(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
        reg[2] = v[2];              \
        reg[3] = v[3];              \
        reg[4] = v[4];              \
        reg[5] = v[5];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        if (ex > 4)                     \
            reg[1] = 0;                 \
        if (ex > 3)                     \
            reg[2] = 0;                 \
        if (ex > 2)                     \
            reg[3] = 0;                 \
        if (ex > 1)                     \
            reg[4] = 0;                 \
        reg[5] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY6(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY6(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B6C6_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY6(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 5) {                      \
            STORE_BUF_ARRAY6(v, off, buf); \
        } else if (ex > 4) {               \
            STORE_BUF_ARRAY5(v, off, buf); \
        } else if (ex > 3) {               \
            STORE_BUF_ARRAY4(v, off, buf); \
        } else if (ex > 2) {               \
            STORE_BUF_ARRAY3(v, off, buf); \
        } else if (ex > 1) {               \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
        READ_BUF(v[5], off + str * 5, buf); \
    }
#elif (LN == 7)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
        V[6] = vload4(off + str * 6, buf);        \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY7(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
        reg[2] = v[2];              \
        reg[3] = v[3];              \
        reg[4] = v[4];              \
        reg[5] = v[5];              \
        reg[6] = v[6];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        if (ex > 5)                     \
            reg[1] = 0;                 \
        if (ex > 4)                     \
            reg[2] = 0;                 \
        if (ex > 3)                     \
            reg[3] = 0;                 \
        if (ex > 2)                     \
            reg[4] = 0;                 \
        if (ex > 1)                     \
            reg[5] = 0;                 \
        reg[6] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY7(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY7(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B7C7_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY7(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 6) {                      \
            STORE_BUF_ARRAY7(v, off, buf); \
        } else if (ex > 5) {               \
            STORE_BUF_ARRAY6(v, off, buf); \
        } else if (ex > 4) {               \
            STORE_BUF_ARRAY5(v, off, buf); \
        } else if (ex > 3) {               \
            STORE_BUF_ARRAY4(v, off, buf); \
        } else if (ex > 2) {               \
            STORE_BUF_ARRAY3(v, off, buf); \
        } else if (ex > 1) {               \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
        READ_BUF(v[5], off + str * 5, buf); \
        READ_BUF(v[6], off + str * 6, buf); \
    }
#elif (LN == 8)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
        V[6] = vload4(off + str * 6, buf);        \
        V[7] = vload4(off + str * 7, buf);        \
    }

#define GEMM_SET_C_BIAS_A_X(v, reg) \
    {                               \
        SET_REG_ARRAY8(v, reg);     \
    }

#define GEMM_SET_C_BIAS_B_X(v, reg) \
    {                               \
        reg[0] = v[0];              \
        reg[1] = v[1];              \
        reg[2] = v[2];              \
        reg[3] = v[3];              \
        reg[4] = v[4];              \
        reg[5] = v[5];              \
        reg[6] = v[6];              \
        reg[7] = v[7];              \
    }

#define GEMM_SET_C_EDGE_ZERO_X(reg, ex) \
    {                                   \
        if (ex > 6)                     \
            reg[1] = 0;                 \
        if (ex > 5)                     \
            reg[2] = 0;                 \
        if (ex > 4)                     \
            reg[3] = 0;                 \
        if (ex > 3)                     \
            reg[4] = 0;                 \
        if (ex > 2)                     \
            reg[5] = 0;                 \
        if (ex > 1)                     \
            reg[6] = 0;                 \
        reg[7] = 0;                     \
    }

#define ADD_ELTWISE_NCHW_X(v, off, buf) \
    {                                   \
        ADD_BUF_ARRAY8(v, off, buf);    \
    }

#define GEMM_LOAD_B(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY8(v, off, buf); \
    }

#define GEMM_CALCORE_X(a, b, c)        \
    {                                  \
        DOT_A_VEC_B8C8_ARRAY(a, b, c); \
    }

#define GEMM_MUL_C_X(a, b, reg)           \
    {                                     \
        MUL_REG_NORMAL_ARRAY8(a, b, reg); \
    }

#define GEMM_STORE_C_X(v, off, ex, buf)    \
    {                                      \
        if (ex > 7) {                      \
            STORE_BUF_ARRAY8(v, off, buf); \
        } else if (ex > 6) {               \
            STORE_BUF_ARRAY7(v, off, buf); \
        } else if (ex > 5) {               \
            STORE_BUF_ARRAY6(v, off, buf); \
        } else if (ex > 4) {               \
            STORE_BUF_ARRAY5(v, off, buf); \
        } else if (ex > 3) {               \
            STORE_BUF_ARRAY4(v, off, buf); \
        } else if (ex > 2) {               \
            STORE_BUF_ARRAY3(v, off, buf); \
        } else if (ex > 1) {               \
            STORE_BUF_ARRAY2(v, off, buf); \
        } else {                           \
            STORE_BUF_ARRAY1(v, off, buf); \
        }                                  \
    }

#define GEMM_NT_LOAD_B(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
        READ_BUF(v[5], off + str * 5, buf); \
        READ_BUF(v[6], off + str * 6, buf); \
        READ_BUF(v[7], off + str * 7, buf); \
    }
#elif (LN == 9)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
        V[6] = vload4(off + str * 6, buf);        \
        V[7] = vload4(off + str * 7, buf);        \
        V[8] = vload4(off + str * 8, buf);        \
    }
#elif (LN == 10)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
        V[6] = vload4(off + str * 6, buf);        \
        V[7] = vload4(off + str * 7, buf);        \
        V[8] = vload4(off + str * 8, buf);        \
        V[9] = vload4(off + str * 9, buf);        \
    }
#elif (LN == 11)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
        V[6] = vload4(off + str * 6, buf);        \
        V[7] = vload4(off + str * 7, buf);        \
        V[8] = vload4(off + str * 8, buf);        \
        V[9] = vload4(off + str * 9, buf);        \
        V[10] = vload4(off + str * 10, buf);      \
    }
#elif (LN == 12)
#define LOAD_INPUT_BUF_ARRAY_V4(V, off, str, buf) \
    {                                             \
        V[0] = vload4(off, buf);                  \
        V[1] = vload4(off + str, buf);            \
        V[2] = vload4(off + str * 2, buf);        \
        V[3] = vload4(off + str * 3, buf);        \
        V[4] = vload4(off + str * 4, buf);        \
        V[5] = vload4(off + str * 5, buf);        \
        V[6] = vload4(off + str * 6, buf);        \
        V[7] = vload4(off + str * 7, buf);        \
        V[8] = vload4(off + str * 8, buf);        \
        V[9] = vload4(off + str * 9, buf);        \
        V[10] = vload4(off + str * 10, buf);      \
        V[11] = vload4(off + str * 11, buf);      \
    }
#endif

/*
 * GEMM A x B = C
 */
#if (LM == 1)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY1(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey) \
    {}

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf) \
    {                                          \
        GEMM_STORE_C_X(v[0], off, ex, buf);    \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf) \
    {                                    \
        READ_BUF(v[0], off, buf);        \
    }
#elif (LM == 2)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY2(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey) \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
    }

#define GEMM_MUL_C(a, b, reg)          \
    {                                  \
        GEMM_MUL_C_X(a, b, reg[0])     \
        {                              \
            GEMM_MUL_C_X(a, b, reg[1]) \
            {}

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)        \
    {                                                 \
        GEMM_STORE_C_X(v[0], off, ex, buf);           \
        if (ey > 1)                                   \
            GEMM_STORE_C_X(v[1], off + str, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf) \
    {                                    \
        READ_BUF(v[0], off, buf);        \
        READ_BUF(v[1], off + str, buf);  \
    }
#elif (LM == 3)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY3(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
        GEMM_SET_C_BIAS_A_X(v[2], reg[2]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
        GEMM_SET_C_BIAS_B_X(v, reg[2]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey)     \
    {                                       \
        if (ey > 1)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]);     \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[2], ex); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
        GEMM_CALCORE_X(a[2], b, c[2]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
        GEMM_MUL_C_X(a, b, reg[2]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)            \
    {                                                     \
        GEMM_STORE_C_X(v[0], off, ex, buf);               \
        if (ey > 1)                                       \
            GEMM_STORE_C_X(v[1], off + str, ex, buf);     \
        if (ey > 2)                                       \
            GEMM_STORE_C_X(v[2], off + str * 2, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
    }
#elif (LM == 4)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY4(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
        GEMM_SET_C_BIAS_A_X(v[2], reg[2]); \
        GEMM_SET_C_BIAS_A_X(v[3], reg[3]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
        GEMM_SET_C_BIAS_B_X(v, reg[2]); \
        GEMM_SET_C_BIAS_B_X(v, reg[3]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        GEMM_SET_C_BIAS_A_X(0, reg[3]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey)     \
    {                                       \
        if (ey > 2)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        if (ey > 1)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        GEMM_SET_C_BIAS_A_X(0, reg[3]);     \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[2], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[3], ex); \
    }

#define ADD_ELTWISE_NCHW(v, off, str, buf)            \
    {                                                 \
        ADD_ELTWISE_NCHW_X(v[0], off, buf);           \
        ADD_ELTWISE_NCHW_X(v[1], off + str, buf);     \
        ADD_ELTWISE_NCHW_X(v[2], off + str * 2, buf); \
        ADD_ELTWISE_NCHW_X(v[3], off + str * 3, buf); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
        GEMM_CALCORE_X(a[2], b, c[2]); \
        GEMM_CALCORE_X(a[3], b, c[3]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
        GEMM_MUL_C_X(a, b, reg[2]); \
        GEMM_MUL_C_X(a, b, reg[3]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)            \
    {                                                     \
        GEMM_STORE_C_X(v[0], off, ex, buf);               \
        if (ey > 1)                                       \
            GEMM_STORE_C_X(v[1], off + str, ex, buf);     \
        if (ey > 2)                                       \
            GEMM_STORE_C_X(v[2], off + str * 2, ex, buf); \
        if (ey > 3)                                       \
            GEMM_STORE_C_X(v[3], off + str * 3, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
    }
#elif (LM == 5)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY5(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
        GEMM_SET_C_BIAS_A_X(v[2], reg[2]); \
        GEMM_SET_C_BIAS_A_X(v[3], reg[3]); \
        GEMM_SET_C_BIAS_A_X(v[4], reg[4]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
        GEMM_SET_C_BIAS_B_X(v, reg[2]); \
        GEMM_SET_C_BIAS_B_X(v, reg[3]); \
        GEMM_SET_C_BIAS_B_X(v, reg[4]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        GEMM_SET_C_BIAS_A_X(0, reg[4]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey)     \
    {                                       \
        if (ey > 3)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        if (ey > 2)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        if (ey > 1)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        GEMM_SET_C_BIAS_A_X(0, reg[4]);     \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[2], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[3], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[4], ex); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
        GEMM_CALCORE_X(a[2], b, c[2]); \
        GEMM_CALCORE_X(a[3], b, c[3]); \
        GEMM_CALCORE_X(a[4], b, c[4]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
        GEMM_MUL_C_X(a, b, reg[2]); \
        GEMM_MUL_C_X(a, b, reg[3]); \
        GEMM_MUL_C_X(a, b, reg[4]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)            \
    {                                                     \
        GEMM_STORE_C_X(v[0], off, ex, buf);               \
        if (ey > 1)                                       \
            GEMM_STORE_C_X(v[1], off + str, ex, buf);     \
        if (ey > 2)                                       \
            GEMM_STORE_C_X(v[2], off + str * 2, ex, buf); \
        if (ey > 3)                                       \
            GEMM_STORE_C_X(v[3], off + str * 3, ex, buf); \
        if (ey > 4)                                       \
            GEMM_STORE_C_X(v[4], off + str * 4, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
    }
#elif (LM == 6)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY6(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
        GEMM_SET_C_BIAS_A_X(v[2], reg[2]); \
        GEMM_SET_C_BIAS_A_X(v[3], reg[3]); \
        GEMM_SET_C_BIAS_A_X(v[4], reg[4]); \
        GEMM_SET_C_BIAS_A_X(v[5], reg[5]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
        GEMM_SET_C_BIAS_B_X(v, reg[2]); \
        GEMM_SET_C_BIAS_B_X(v, reg[3]); \
        GEMM_SET_C_BIAS_B_X(v, reg[4]); \
        GEMM_SET_C_BIAS_B_X(v, reg[5]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        GEMM_SET_C_BIAS_A_X(0, reg[4]); \
        GEMM_SET_C_BIAS_A_X(0, reg[5]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey)     \
    {                                       \
        if (ey > 4)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        if (ey > 3)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        if (ey > 2)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        if (ey > 1)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[4]); \
        GEMM_SET_C_BIAS_A_X(0, reg[5]);     \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[2], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[3], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[4], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[5], ex); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
        GEMM_CALCORE_X(a[2], b, c[2]); \
        GEMM_CALCORE_X(a[3], b, c[3]); \
        GEMM_CALCORE_X(a[4], b, c[4]); \
        GEMM_CALCORE_X(a[5], b, c[5]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
        GEMM_MUL_C_X(a, b, reg[2]); \
        GEMM_MUL_C_X(a, b, reg[3]); \
        GEMM_MUL_C_X(a, b, reg[4]); \
        GEMM_MUL_C_X(a, b, reg[5]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)            \
    {                                                     \
        GEMM_STORE_C_X(v[0], off, ex, buf);               \
        if (ey > 1)                                       \
            GEMM_STORE_C_X(v[1], off + str, ex, buf);     \
        if (ey > 2)                                       \
            GEMM_STORE_C_X(v[2], off + str * 2, ex, buf); \
        if (ey > 3)                                       \
            GEMM_STORE_C_X(v[3], off + str * 3, ex, buf); \
        if (ey > 4)                                       \
            GEMM_STORE_C_X(v[4], off + str * 4, ex, buf); \
        if (ey > 5)                                       \
            GEMM_STORE_C_X(v[5], off + str * 5, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
        READ_BUF(v[5], off + str * 5, buf); \
    }
#elif (LM == 7)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY7(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
        GEMM_SET_C_BIAS_A_X(v[2], reg[2]); \
        GEMM_SET_C_BIAS_A_X(v[3], reg[3]); \
        GEMM_SET_C_BIAS_A_X(v[4], reg[4]); \
        GEMM_SET_C_BIAS_A_X(v[5], reg[5]); \
        GEMM_SET_C_BIAS_A_X(v[6], reg[6]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
        GEMM_SET_C_BIAS_B_X(v, reg[2]); \
        GEMM_SET_C_BIAS_B_X(v, reg[3]); \
        GEMM_SET_C_BIAS_B_X(v, reg[4]); \
        GEMM_SET_C_BIAS_B_X(v, reg[5]); \
        GEMM_SET_C_BIAS_B_X(v, reg[6]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        GEMM_SET_C_BIAS_A_X(0, reg[4]); \
        GEMM_SET_C_BIAS_A_X(0, reg[5]); \
        GEMM_SET_C_BIAS_A_X(0, reg[6]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey)     \
    {                                       \
        if (ey > 5)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        if (ey > 4)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        if (ey > 3)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        if (ey > 2)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[4]); \
        if (ey > 1)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[5]); \
        GEMM_SET_C_BIAS_A_X(0, reg[6]);     \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[2], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[3], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[4], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[5], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[6], ex); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
        GEMM_CALCORE_X(a[2], b, c[2]); \
        GEMM_CALCORE_X(a[3], b, c[3]); \
        GEMM_CALCORE_X(a[4], b, c[4]); \
        GEMM_CALCORE_X(a[5], b, c[5]); \
        GEMM_CALCORE_X(a[6], b, c[6]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
        GEMM_MUL_C_X(a, b, reg[2]); \
        GEMM_MUL_C_X(a, b, reg[3]); \
        GEMM_MUL_C_X(a, b, reg[4]); \
        GEMM_MUL_C_X(a, b, reg[5]); \
        GEMM_MUL_C_X(a, b, reg[6]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)            \
    {                                                     \
        GEMM_STORE_C_X(v[0], off, ex, buf);               \
        if (ey > 1)                                       \
            GEMM_STORE_C_X(v[1], off + str, ex, buf);     \
        if (ey > 2)                                       \
            GEMM_STORE_C_X(v[2], off + str * 2, ex, buf); \
        if (ey > 3)                                       \
            GEMM_STORE_C_X(v[3], off + str * 3, ex, buf); \
        if (ey > 4)                                       \
            GEMM_STORE_C_X(v[4], off + str * 4, ex, buf); \
        if (ey > 5)                                       \
            GEMM_STORE_C_X(v[5], off + str * 5, ex, buf); \
        if (ey > 6)                                       \
            GEMM_STORE_C_X(v[6], off + str * 6, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
        READ_BUF(v[5], off + str * 5, buf); \
        READ_BUF(v[6], off + str * 6, buf); \
    }
#elif (LM == 8)
#define GEMM_LOAD_A(v, off, buf)      \
    {                                 \
        LOAD_BUF_ARRAY8(v, off, buf); \
    }

#define GEMM_SET_C_BIAS_A(v, reg)          \
    {                                      \
        GEMM_SET_C_BIAS_A_X(v[0], reg[0]); \
        GEMM_SET_C_BIAS_A_X(v[1], reg[1]); \
        GEMM_SET_C_BIAS_A_X(v[2], reg[2]); \
        GEMM_SET_C_BIAS_A_X(v[3], reg[3]); \
        GEMM_SET_C_BIAS_A_X(v[4], reg[4]); \
        GEMM_SET_C_BIAS_A_X(v[5], reg[5]); \
        GEMM_SET_C_BIAS_A_X(v[6], reg[6]); \
        GEMM_SET_C_BIAS_A_X(v[7], reg[7]); \
    }

#define GEMM_SET_C_BIAS_B(v, reg)       \
    {                                   \
        GEMM_SET_C_BIAS_B_X(v, reg[0]); \
        GEMM_SET_C_BIAS_B_X(v, reg[1]); \
        GEMM_SET_C_BIAS_B_X(v, reg[2]); \
        GEMM_SET_C_BIAS_B_X(v, reg[3]); \
        GEMM_SET_C_BIAS_B_X(v, reg[4]); \
        GEMM_SET_C_BIAS_B_X(v, reg[5]); \
        GEMM_SET_C_BIAS_B_X(v, reg[6]); \
        GEMM_SET_C_BIAS_B_X(v, reg[7]); \
    }

#define GEMM_SET_C_ZERO(reg)            \
    {                                   \
        GEMM_SET_C_BIAS_A_X(0, reg[0]); \
        GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        GEMM_SET_C_BIAS_A_X(0, reg[4]); \
        GEMM_SET_C_BIAS_A_X(0, reg[5]); \
        GEMM_SET_C_BIAS_A_X(0, reg[6]); \
        GEMM_SET_C_BIAS_A_X(0, reg[7]); \
    }

#define GEMM_SET_C_EDGE_ZERO_H(reg, ey)     \
    {                                       \
        if (ey > 6)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[1]); \
        if (ey > 5)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[2]); \
        if (ey > 4)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[3]); \
        if (ey > 3)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[4]); \
        if (ey > 2)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[5]); \
        if (ey > 1)                         \
            GEMM_SET_C_BIAS_A_X(0, reg[6]); \
        GEMM_SET_C_BIAS_A_X(0, reg[7]);     \
    }

#define GEMM_SET_C_EDGE_ZERO_W(reg, ex)     \
    {                                       \
        GEMM_SET_C_EDGE_ZERO_X(reg[0], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[1], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[2], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[3], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[4], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[5], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[6], ex); \
        GEMM_SET_C_EDGE_ZERO_X(reg[7], ex); \
    }

#define ADD_ELTWISE_NCHW(v, off, str, buf)            \
    {                                                 \
        ADD_ELTWISE_NCHW_X(v[0], off, buf);           \
        ADD_ELTWISE_NCHW_X(v[1], off + str, buf);     \
        ADD_ELTWISE_NCHW_X(v[2], off + str * 2, buf); \
        ADD_ELTWISE_NCHW_X(v[3], off + str * 3, buf); \
        ADD_ELTWISE_NCHW_X(v[4], off + str * 4, buf); \
        ADD_ELTWISE_NCHW_X(v[5], off + str * 5, buf); \
        ADD_ELTWISE_NCHW_X(v[6], off + str * 6, buf); \
        ADD_ELTWISE_NCHW_X(v[7], off + str * 7, buf); \
    }

#define GEMM_CALCORE(a, b, c)          \
    {                                  \
        GEMM_CALCORE_X(a[0], b, c[0]); \
        GEMM_CALCORE_X(a[1], b, c[1]); \
        GEMM_CALCORE_X(a[2], b, c[2]); \
        GEMM_CALCORE_X(a[3], b, c[3]); \
        GEMM_CALCORE_X(a[4], b, c[4]); \
        GEMM_CALCORE_X(a[5], b, c[5]); \
        GEMM_CALCORE_X(a[6], b, c[6]); \
        GEMM_CALCORE_X(a[7], b, c[7]); \
    }

#define GEMM_MUL_C(a, b, reg)       \
    {                               \
        GEMM_MUL_C_X(a, b, reg[0]); \
        GEMM_MUL_C_X(a, b, reg[1]); \
        GEMM_MUL_C_X(a, b, reg[2]); \
        GEMM_MUL_C_X(a, b, reg[3]); \
        GEMM_MUL_C_X(a, b, reg[4]); \
        GEMM_MUL_C_X(a, b, reg[5]); \
        GEMM_MUL_C_X(a, b, reg[6]); \
        GEMM_MUL_C_X(a, b, reg[7]); \
    }

#define GEMM_STORE_C(v, off, str, ex, ey, buf)            \
    {                                                     \
        GEMM_STORE_C_X(v[0], off, ex, buf);               \
        if (ey > 1)                                       \
            GEMM_STORE_C_X(v[1], off + str, ex, buf);     \
        if (ey > 2)                                       \
            GEMM_STORE_C_X(v[2], off + str * 2, ex, buf); \
        if (ey > 3)                                       \
            GEMM_STORE_C_X(v[3], off + str * 3, ex, buf); \
        if (ey > 4)                                       \
            GEMM_STORE_C_X(v[4], off + str * 4, ex, buf); \
        if (ey > 5)                                       \
            GEMM_STORE_C_X(v[5], off + str * 5, ex, buf); \
        if (ey > 6)                                       \
            GEMM_STORE_C_X(v[6], off + str * 6, ex, buf); \
        if (ey > 7)                                       \
            GEMM_STORE_C_X(v[7], off + str * 7, ex, buf); \
    }

#define GEMM_NT_LOAD_A(v, off, str, buf)    \
    {                                       \
        READ_BUF(v[0], off, buf);           \
        READ_BUF(v[1], off + str, buf);     \
        READ_BUF(v[2], off + str * 2, buf); \
        READ_BUF(v[3], off + str * 3, buf); \
        READ_BUF(v[4], off + str * 4, buf); \
        READ_BUF(v[5], off + str * 5, buf); \
        READ_BUF(v[6], off + str * 6, buf); \
        READ_BUF(v[7], off + str * 7, buf); \
    }
#endif

/*
 * UPDATE VALUE OF REG
 */
#if (UN == 0)
#define UPDATE_REG(A) \
    {}
#elif (UN == 1)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
    }
#elif (UN == 2)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
    }
#elif (UN == 3)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
    }
#elif (UN == 4)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
    }
#elif (UN == 5)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
        A[4] = A[5];  \
    }
#elif (UN == 6)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
        A[4] = A[5];  \
        A[5] = A[6];  \
    }
#elif (UN == 7)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
        A[4] = A[5];  \
        A[5] = A[6];  \
        A[6] = A[7];  \
    }
#elif (UN == 8)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
        A[4] = A[5];  \
        A[5] = A[6];  \
        A[6] = A[7];  \
        A[7] = A[8];  \
    }
#elif (UN == 9)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
        A[4] = A[5];  \
        A[5] = A[6];  \
        A[6] = A[7];  \
        A[7] = A[8];  \
        A[8] = A[9];  \
    }
#elif (UN == 10)
#define UPDATE_REG(A) \
    {                 \
        A[0] = A[1];  \
        A[1] = A[2];  \
        A[2] = A[3];  \
        A[3] = A[4];  \
        A[4] = A[5];  \
        A[5] = A[6];  \
        A[6] = A[7];  \
        A[7] = A[8];  \
        A[8] = A[9];  \
        A[9] = A[10]; \
    }
#elif (UN == 11)
#define UPDATE_REG(A)  \
    {                  \
        A[0] = A[1];   \
        A[1] = A[2];   \
        A[2] = A[3];   \
        A[3] = A[4];   \
        A[4] = A[5];   \
        A[5] = A[6];   \
        A[6] = A[7];   \
        A[7] = A[8];   \
        A[8] = A[9];   \
        A[9] = A[10];  \
        A[10] = A[11]; \
    }
#elif (UN == 12)
#define UPDATE_REG(A)  \
    {                  \
        A[0] = A[1];   \
        A[1] = A[2];   \
        A[2] = A[3];   \
        A[3] = A[4];   \
        A[4] = A[5];   \
        A[5] = A[6];   \
        A[6] = A[7];   \
        A[7] = A[8];   \
        A[8] = A[9];   \
        A[9] = A[10];  \
        A[10] = A[11]; \
        A[11] = A[12]; \
    }
#endif

/*
 * Direct convolution calculate core
 * Depthwise calculate core
 */
#if (ON == 1)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
    }
#define DIRECT_CONV_CAL_CORE_S2(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
    }
#define DEPTHWISE_CAL_CORE_S2(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
    }
#elif (ON == 2)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
    }
#define DIRECT_CONV_CAL_CORE_S2(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[2], B, C[1]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
    }
#define DEPTHWISE_CAL_CORE_S2(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[2], B, C[1]);     \
    }
#elif (ON == 3)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
    }
#define DIRECT_CONV_CAL_CORE_S2(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[2], B, C[1]);      \
        DOT_A4B16C4(A[4], B, C[2]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
        DOT_A4B4C4(A[2], B, C[2]);     \
    }
#define DEPTHWISE_CAL_CORE_S2(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[2], B, C[1]);     \
        DOT_A4B4C4(A[4], B, C[2]);     \
    }
#elif (ON == 4)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
        DOT_A4B16C4(A[3], B, C[3]);      \
    }
#define DIRECT_CONV_CAL_CORE_S2(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
        DOT_A4B16C4(A[4], B, C[4]);      \
        DOT_A4B16C4(A[6], B, C[6]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
        DOT_A4B4C4(A[2], B, C[2]);     \
        DOT_A4B4C4(A[3], B, C[3]);     \
    }
#define DEPTHWISE_CAL_CORE_S2(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[2], B, C[1]);     \
        DOT_A4B4C4(A[4], B, C[2]);     \
        DOT_A4B4C4(A[6], B, C[3]);     \
    }
#elif (ON == 5)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
        DOT_A4B16C4(A[3], B, C[3]);      \
        DOT_A4B16C4(A[4], B, C[4]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
        DOT_A4B4C4(A[2], B, C[2]);     \
        DOT_A4B4C4(A[3], B, C[3]);     \
        DOT_A4B4C4(A[4], B, C[4]);     \
    }
#elif (ON == 6)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
        DOT_A4B16C4(A[3], B, C[3]);      \
        DOT_A4B16C4(A[4], B, C[4]);      \
        DOT_A4B16C4(A[5], B, C[5]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
        DOT_A4B4C4(A[2], B, C[2]);     \
        DOT_A4B4C4(A[3], B, C[3]);     \
        DOT_A4B4C4(A[4], B, C[4]);     \
        DOT_A4B4C4(A[5], B, C[5]);     \
    }
#elif (ON == 7)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
        DOT_A4B16C4(A[3], B, C[3]);      \
        DOT_A4B16C4(A[4], B, C[4]);      \
        DOT_A4B16C4(A[5], B, C[5]);      \
        DOT_A4B16C4(A[6], B, C[6]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
        DOT_A4B4C4(A[2], B, C[2]);     \
        DOT_A4B4C4(A[3], B, C[3]);     \
        DOT_A4B4C4(A[4], B, C[4]);     \
        DOT_A4B4C4(A[5], B, C[5]);     \
        DOT_A4B4C4(A[6], B, C[6]);     \
    }
#elif (ON == 8)
#define DIRECT_CONV_CAL_CORE_S1(A, B, C) \
    {                                    \
        DOT_A4B16C4(A[0], B, C[0]);      \
        DOT_A4B16C4(A[1], B, C[1]);      \
        DOT_A4B16C4(A[2], B, C[2]);      \
        DOT_A4B16C4(A[3], B, C[3]);      \
        DOT_A4B16C4(A[4], B, C[4]);      \
        DOT_A4B16C4(A[5], B, C[5]);      \
        DOT_A4B16C4(A[6], B, C[6]);      \
        DOT_A4B16C4(A[7], B, C[7]);      \
    }
#define DEPTHWISE_CAL_CORE_S1(A, B, C) \
    {                                  \
        DOT_A4B4C4(A[0], B, C[0]);     \
        DOT_A4B4C4(A[1], B, C[1]);     \
        DOT_A4B4C4(A[2], B, C[2]);     \
        DOT_A4B4C4(A[3], B, C[3]);     \
        DOT_A4B4C4(A[4], B, C[4]);     \
        DOT_A4B4C4(A[5], B, C[5]);     \
        DOT_A4B4C4(A[6], B, C[6]);     \
        DOT_A4B4C4(A[7], B, C[7]);     \
    }
#endif

/*
 * STORE_OUTPUT_BUF_ARRAY_V4 WITH ACTIVATION
 */
#if (ON == 1)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        vstore4(V[0], off, buf);                            \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
    }

#define STORE_OUTPUT_BUF_ARRAY_ALIGN(val, off, str, out) \
    {                                                    \
        out[off] = val[0];                               \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY1(v, reg); \
    }
#elif (ON == 2)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
    }

#define STORE_OUTPUT_BUF_ARRAY_ALIGN(val, off, str, out) \
    {                                                    \
        out[off] = val[0];                               \
        out[off + str] = val[1];                         \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY2(v, reg); \
    }
#elif (ON == 3)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        ACTIVATION_V4(V[2]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
        if (id + 2 < bd)                                    \
            vstore4(V[2], off + str * 2, buf);              \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
        V[2] += vload4(off + str * 2, buf);        \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        ACTIVATION_V4(V[2]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
        if (id + 2 < bd) {                                                 \
            buf[off + str_h * 2] = V[2].x;                                 \
            buf[off + str_h * 2 + str_hw] = V[2].y;                        \
            buf[off + str_h * 2 + str_hw * 2] = V[2].z;                    \
            buf[off + str_h * 2 + str_hw * 3] = V[2].w;                    \
        }                                                                  \
    }

#define STORE_OUTPUT_BUF_ARRAY_ALIGN(val, off, str, out) \
    {                                                    \
        out[off] = val[0];                               \
        out[off + str] = val[1];                         \
        out[off + str * 2] = val[2];                     \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY3(v, reg); \
    }
#elif (ON == 4)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        ACTIVATION_V4(V[2]);                                \
        ACTIVATION_V4(V[3]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
        if (id + 2 < bd)                                    \
            vstore4(V[2], off + str * 2, buf);              \
        if (id + 3 < bd)                                    \
            vstore4(V[3], off + str * 3, buf);              \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
        V[2] += vload4(off + str * 2, buf);        \
        V[3] += vload4(off + str * 3, buf);        \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        ACTIVATION_V4(V[2]);                                               \
        ACTIVATION_V4(V[3]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
        if (id + 2 < bd) {                                                 \
            buf[off + str_h * 2] = V[2].x;                                 \
            buf[off + str_h * 2 + str_hw] = V[2].y;                        \
            buf[off + str_h * 2 + str_hw * 2] = V[2].z;                    \
            buf[off + str_h * 2 + str_hw * 3] = V[2].w;                    \
        }                                                                  \
        if (id + 3 < bd) {                                                 \
            buf[off + str_h * 3] = V[3].x;                                 \
            buf[off + str_h * 3 + str_hw] = V[3].y;                        \
            buf[off + str_h * 3 + str_hw * 2] = V[3].z;                    \
            buf[off + str_h * 3 + str_hw * 3] = V[3].w;                    \
        }                                                                  \
    }

#define STORE_OUTPUT_BUF_ARRAY_ALIGN(val, off, str, out) \
    {                                                    \
        out[off] = val[0];                               \
        out[off + str] = val[1];                         \
        out[off + str * 2] = val[2];                     \
        out[off + str * 3] = val[3];                     \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY4(v, reg); \
    }
#elif (ON == 5)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        ACTIVATION_V4(V[2]);                                \
        ACTIVATION_V4(V[3]);                                \
        ACTIVATION_V4(V[4]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
        if (id + 2 < bd)                                    \
            vstore4(V[2], off + str * 2, buf);              \
        if (id + 3 < bd)                                    \
            vstore4(V[3], off + str * 3, buf);              \
        if (id + 4 < bd)                                    \
            vstore4(V[4], off + str * 4, buf);              \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
        V[2] += vload4(off + str * 2, buf);        \
        V[3] += vload4(off + str * 3, buf);        \
        V[4] += vload4(off + str * 4, buf);        \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        ACTIVATION_V4(V[2]);                                               \
        ACTIVATION_V4(V[3]);                                               \
        ACTIVATION_V4(V[4]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
        if (id + 2 < bd) {                                                 \
            buf[off + str_h * 2] = V[2].x;                                 \
            buf[off + str_h * 2 + str_hw] = V[2].y;                        \
            buf[off + str_h * 2 + str_hw * 2] = V[2].z;                    \
            buf[off + str_h * 2 + str_hw * 3] = V[2].w;                    \
        }                                                                  \
        if (id + 3 < bd) {                                                 \
            buf[off + str_h * 3] = V[3].x;                                 \
            buf[off + str_h * 3 + str_hw] = V[3].y;                        \
            buf[off + str_h * 3 + str_hw * 2] = V[3].z;                    \
            buf[off + str_h * 3 + str_hw * 3] = V[3].w;                    \
        }                                                                  \
        if (id + 4 < bd) {                                                 \
            buf[off + str_h * 4] = V[4].x;                                 \
            buf[off + str_h * 4 + str_hw] = V[4].y;                        \
            buf[off + str_h * 4 + str_hw * 2] = V[4].z;                    \
            buf[off + str_h * 4 + str_hw * 3] = V[4].w;                    \
        }                                                                  \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY5(v, reg); \
    }
#elif (ON == 6)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        ACTIVATION_V4(V[2]);                                \
        ACTIVATION_V4(V[3]);                                \
        ACTIVATION_V4(V[4]);                                \
        ACTIVATION_V4(V[5]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
        if (id + 2 < bd)                                    \
            vstore4(V[2], off + str * 2, buf);              \
        if (id + 3 < bd)                                    \
            vstore4(V[3], off + str * 3, buf);              \
        if (id + 4 < bd)                                    \
            vstore4(V[4], off + str * 4, buf);              \
        if (id + 5 < bd)                                    \
            vstore4(V[5], off + str * 5, buf);              \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
        V[2] += vload4(off + str * 2, buf);        \
        V[3] += vload4(off + str * 3, buf);        \
        V[4] += vload4(off + str * 4, buf);        \
        V[5] += vload4(off + str * 5, buf);        \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        ACTIVATION_V4(V[2]);                                               \
        ACTIVATION_V4(V[3]);                                               \
        ACTIVATION_V4(V[4]);                                               \
        ACTIVATION_V4(V[5]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
        if (id + 2 < bd) {                                                 \
            buf[off + str_h * 2] = V[2].x;                                 \
            buf[off + str_h * 2 + str_hw] = V[2].y;                        \
            buf[off + str_h * 2 + str_hw * 2] = V[2].z;                    \
            buf[off + str_h * 2 + str_hw * 3] = V[2].w;                    \
        }                                                                  \
        if (id + 3 < bd) {                                                 \
            buf[off + str_h * 3] = V[3].x;                                 \
            buf[off + str_h * 3 + str_hw] = V[3].y;                        \
            buf[off + str_h * 3 + str_hw * 2] = V[3].z;                    \
            buf[off + str_h * 3 + str_hw * 3] = V[3].w;                    \
        }                                                                  \
        if (id + 4 < bd) {                                                 \
            buf[off + str_h * 4] = V[4].x;                                 \
            buf[off + str_h * 4 + str_hw] = V[4].y;                        \
            buf[off + str_h * 4 + str_hw * 2] = V[4].z;                    \
            buf[off + str_h * 4 + str_hw * 3] = V[4].w;                    \
        }                                                                  \
        if (id + 5 < bd) {                                                 \
            buf[off + str_h * 5] = V[5].x;                                 \
            buf[off + str_h * 5 + str_hw] = V[5].y;                        \
            buf[off + str_h * 5 + str_hw * 2] = V[5].z;                    \
            buf[off + str_h * 5 + str_hw * 3] = V[5].w;                    \
        }                                                                  \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY6(v, reg); \
    }
#elif (ON == 7)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        ACTIVATION_V4(V[2]);                                \
        ACTIVATION_V4(V[3]);                                \
        ACTIVATION_V4(V[4]);                                \
        ACTIVATION_V4(V[5]);                                \
        ACTIVATION_V4(V[6]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
        if (id + 2 < bd)                                    \
            vstore4(V[2], off + str * 2, buf);              \
        if (id + 3 < bd)                                    \
            vstore4(V[3], off + str * 3, buf);              \
        if (id + 4 < bd)                                    \
            vstore4(V[4], off + str * 4, buf);              \
        if (id + 5 < bd)                                    \
            vstore4(V[5], off + str * 5, buf);              \
        if (id + 6 < bd)                                    \
            vstore4(V[6], off + str * 6, buf);              \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
        V[2] += vload4(off + str * 2, buf);        \
        V[3] += vload4(off + str * 3, buf);        \
        V[4] += vload4(off + str * 4, buf);        \
        V[5] += vload4(off + str * 5, buf);        \
        V[6] += vload4(off + str * 6, buf);        \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        ACTIVATION_V4(V[2]);                                               \
        ACTIVATION_V4(V[3]);                                               \
        ACTIVATION_V4(V[4]);                                               \
        ACTIVATION_V4(V[5]);                                               \
        ACTIVATION_V4(V[6]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
        if (id + 2 < bd) {                                                 \
            buf[off + str_h * 2] = V[2].x;                                 \
            buf[off + str_h * 2 + str_hw] = V[2].y;                        \
            buf[off + str_h * 2 + str_hw * 2] = V[2].z;                    \
            buf[off + str_h * 2 + str_hw * 3] = V[2].w;                    \
        }                                                                  \
        if (id + 3 < bd) {                                                 \
            buf[off + str_h * 3] = V[3].x;                                 \
            buf[off + str_h * 3 + str_hw] = V[3].y;                        \
            buf[off + str_h * 3 + str_hw * 2] = V[3].z;                    \
            buf[off + str_h * 3 + str_hw * 3] = V[3].w;                    \
        }                                                                  \
        if (id + 4 < bd) {                                                 \
            buf[off + str_h * 4] = V[4].x;                                 \
            buf[off + str_h * 4 + str_hw] = V[4].y;                        \
            buf[off + str_h * 4 + str_hw * 2] = V[4].z;                    \
            buf[off + str_h * 4 + str_hw * 3] = V[4].w;                    \
        }                                                                  \
        if (id + 5 < bd) {                                                 \
            buf[off + str_h * 5] = V[5].x;                                 \
            buf[off + str_h * 5 + str_hw] = V[5].y;                        \
            buf[off + str_h * 5 + str_hw * 2] = V[5].z;                    \
            buf[off + str_h * 5 + str_hw * 3] = V[5].w;                    \
        }                                                                  \
        if (id + 6 < bd) {                                                 \
            buf[off + str_h * 6] = V[6].x;                                 \
            buf[off + str_h * 6 + str_hw] = V[6].y;                        \
            buf[off + str_h * 6 + str_hw * 2] = V[6].z;                    \
            buf[off + str_h * 6 + str_hw * 3] = V[6].w;                    \
        }                                                                  \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY7(v, reg); \
    }
#elif (ON == 8)
#define STORE_OUTPUT_BUF_ARRAY_V4(V, off, str, id, bd, buf) \
    {                                                       \
        ACTIVATION_V4(V[0]);                                \
        ACTIVATION_V4(V[1]);                                \
        ACTIVATION_V4(V[2]);                                \
        ACTIVATION_V4(V[3]);                                \
        ACTIVATION_V4(V[4]);                                \
        ACTIVATION_V4(V[5]);                                \
        ACTIVATION_V4(V[6]);                                \
        ACTIVATION_V4(V[7]);                                \
        vstore4(V[0], off, buf);                            \
        if (id + 1 < bd)                                    \
            vstore4(V[1], off + str, buf);                  \
        if (id + 2 < bd)                                    \
            vstore4(V[2], off + str * 2, buf);              \
        if (id + 3 < bd)                                    \
            vstore4(V[3], off + str * 3, buf);              \
        if (id + 4 < bd)                                    \
            vstore4(V[4], off + str * 4, buf);              \
        if (id + 5 < bd)                                    \
            vstore4(V[5], off + str * 5, buf);              \
        if (id + 6 < bd)                                    \
            vstore4(V[6], off + str * 6, buf);              \
        if (id + 7 < bd)                                    \
            vstore4(V[7], off + str * 7, buf);              \
    }

#define ADD_ELTWISE_BUF_ARRAY_V4(V, off, str, buf) \
    {                                              \
        V[0] += vload4(off, buf);                  \
        V[1] += vload4(off + str, buf);            \
        V[2] += vload4(off + str * 2, buf);        \
        V[3] += vload4(off + str * 3, buf);        \
        V[4] += vload4(off + str * 4, buf);        \
        V[5] += vload4(off + str * 5, buf);        \
        V[6] += vload4(off + str * 6, buf);        \
        V[7] += vload4(off + str * 7, buf);        \
    }

#define STORE_OUTPUT_BUF_ARRAY_V4_NCWH(V, off, str_h, str_hw, id, bd, buf) \
    {                                                                      \
        ACTIVATION_V4(V[0]);                                               \
        ACTIVATION_V4(V[1]);                                               \
        ACTIVATION_V4(V[2]);                                               \
        ACTIVATION_V4(V[3]);                                               \
        ACTIVATION_V4(V[4]);                                               \
        ACTIVATION_V4(V[5]);                                               \
        ACTIVATION_V4(V[6]);                                               \
        ACTIVATION_V4(V[7]);                                               \
        buf[off] = V[0].x;                                                 \
        buf[off + str_hw] = V[0].y;                                        \
        buf[off + str_hw * 2] = V[0].z;                                    \
        buf[off + str_hw * 3] = V[0].w;                                    \
        if (id + 1 < bd) {                                                 \
            buf[off + str_h] = V[1].x;                                     \
            buf[off + str_h + str_hw] = V[1].y;                            \
            buf[off + str_h + str_hw * 2] = V[1].z;                        \
            buf[off + str_h + str_hw * 3] = V[1].w;                        \
        }                                                                  \
        if (id + 2 < bd) {                                                 \
            buf[off + str_h * 2] = V[2].x;                                 \
            buf[off + str_h * 2 + str_hw] = V[2].y;                        \
            buf[off + str_h * 2 + str_hw * 2] = V[2].z;                    \
            buf[off + str_h * 2 + str_hw * 3] = V[2].w;                    \
        }                                                                  \
        if (id + 3 < bd) {                                                 \
            buf[off + str_h * 3] = V[3].x;                                 \
            buf[off + str_h * 3 + str_hw] = V[3].y;                        \
            buf[off + str_h * 3 + str_hw * 2] = V[3].z;                    \
            buf[off + str_h * 3 + str_hw * 3] = V[3].w;                    \
        }                                                                  \
        if (id + 4 < bd) {                                                 \
            buf[off + str_h * 4] = V[4].x;                                 \
            buf[off + str_h * 4 + str_hw] = V[4].y;                        \
            buf[off + str_h * 4 + str_hw * 2] = V[4].z;                    \
            buf[off + str_h * 4 + str_hw * 3] = V[4].w;                    \
        }                                                                  \
        if (id + 5 < bd) {                                                 \
            buf[off + str_h * 5] = V[5].x;                                 \
            buf[off + str_h * 5 + str_hw] = V[5].y;                        \
            buf[off + str_h * 5 + str_hw * 2] = V[5].z;                    \
            buf[off + str_h * 5 + str_hw * 3] = V[5].w;                    \
        }                                                                  \
        if (id + 6 < bd) {                                                 \
            buf[off + str_h * 6] = V[6].x;                                 \
            buf[off + str_h * 6 + str_hw] = V[6].y;                        \
            buf[off + str_h * 6 + str_hw * 2] = V[6].z;                    \
            buf[off + str_h * 6 + str_hw * 3] = V[6].w;                    \
        }                                                                  \
        if (id + 7 < bd) {                                                 \
            buf[off + str_h * 7] = V[7].x;                                 \
            buf[off + str_h * 7 + str_hw] = V[7].y;                        \
            buf[off + str_h * 7 + str_hw * 2] = V[7].z;                    \
            buf[off + str_h * 7 + str_hw * 3] = V[7].w;                    \
        }                                                                  \
    }

#define SET_REG_ARRAY(v, reg)   \
    {                           \
        SET_REG_ARRAY8(v, reg); \
    }
#endif

#if defined(DILATION2)
#define LOAD_INPUT_EXCESS_DILATION2(reg, off, str, LN, buf) \
    {                                                       \
        reg[LN] = vload4(off + LN * str, buf);              \
        reg[LN + 1] = vload4(off + (LN + 1) * str, buf);    \
    }
#if (ON == 3)
#define UPDATE_REG_DILATION2(reg) \
    {                             \
        reg[0] = reg[2];          \
    }
#elif (ON == 4)
#define UPDATE_REG_DILATION2(reg) \
    {                             \
        reg[0] = reg[2];          \
        reg[1] = reg[3];          \
    }
#elif (ON == 5)
#define UPDATE_REG_DILATION2(reg) \
    {                             \
        reg[0] = reg[2];          \
        reg[1] = reg[3];          \
        reg[2] = reg[4];          \
    }
#elif (ON == 6)
#define UPDATE_REG_DILATION2(reg) \
    {                             \
        reg[0] = reg[2];          \
        reg[1] = reg[3];          \
        reg[2] = reg[4];          \
        reg[3] = reg[5];          \
    }
#elif (ON == 7)
#define UPDATE_REG_DILATION2(reg) \
    {                             \
        reg[0] = reg[2];          \
        reg[1] = reg[3];          \
        reg[2] = reg[4];          \
        reg[3] = reg[5];          \
        reg[4] = reg[6];          \
    }
#elif (ON == 8)
#define UPDATE_REG_DILATION2(reg) \
    {                             \
        reg[0] = reg[2];          \
        reg[1] = reg[3];          \
        reg[2] = reg[4];          \
        reg[3] = reg[5];          \
        reg[4] = reg[6];          \
        reg[5] = reg[7];          \
    }
#endif
#endif

#endif  //_KERNEL_DEF
