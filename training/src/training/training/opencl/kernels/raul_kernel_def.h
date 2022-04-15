R"(// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#define LOAD_VAL_T4(ew, idx, idy, idz, ih_str, iw_str, ih_off, iw_off, buf, val)   \
    T4 val = 0;                                                                     \
	{                                                                           \
        int off = (idz * ih_str + idy + ih_off) * iw_str + (idx << 2) + iw_off; \
        val = 0;                                                                \
        if (ew == 4) {                                                          \
            val = vload4(0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                val.x = buf[off];                                               \
            if (ew == 2) {                                                      \
                T2 tmp = vload2(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
            }                                                                   \
            if (ew == 3) {                                                      \
                T3 tmp = vload3(0, buf + off);                                  \
                val.x = tmp.x;                                                  \
                val.y = tmp.y;                                                  \
                val.z = tmp.z;                                                  \
            }                                                                   \
        }                                                                       \
    }
#define STORE_VAL_T4(ew, idx, idy, idz, oh_str, ow_str, oh_off, ow_off, buf, val)  \
    {                                                                           \
        int off = (idz * oh_str + idy + oh_off) * ow_str + (idx << 2) + ow_off; \
        if (ew == 4) {                                                          \
            vstore4(val, 0, buf + off);                                         \
        } else {                                                                \
            if (ew == 1)                                                        \
                buf[off] = val.x;                                               \
            if (ew == 2) {                                                      \
                vstore2((T2)(val.x, val.y), 0, buf + off);                      \
            }                                                                   \
            if (ew == 3) {                                                      \
                vstore3((T3)(val.x, val.y, val.z), 0, buf + off);               \
            }                                                                   \
        }                                                                       \
    }

#define ADD_STORE_BUF_ARRAY1(v, off, bet, buf) \
    {                                 \
        buf[off] = buf[off] * bet + v[0];              \
        ACTIVATION_V1(buf[off]); \
    }

#define ADD_STORE_BUF_ARRAY2(v, off, bet, buf)            \
    {                                            \
        T2 tmp = vload2(0, buf + off); \
		tmp *= bet; \
		tmp += (T2)(v[0], v[1]); \
		ACTIVATION_V2(tmp); \
		vstore2(tmp, 0, buf + off); \
    }

#define ADD_STORE_BUF_ARRAY3(v, off, bet, buf)                  \
    {                                                  \
        T3 tmp = vload3(0, buf + off); \
		tmp *= bet; \
		tmp += (T3)(v[0], v[1], v[2]); \
		ACTIVATION_V3(tmp); \
        vstore3(tmp, 0, buf + off); \
    }

#define ADD_STORE_BUF_ARRAY4(v, off, bet, buf)                        \
    {                                                        \
        T4 tmp = vload4(0, buf + off); \
		tmp *= bet; \
		tmp += (T4)(v[0], v[1], v[2], v[3]); \
		ACTIVATION_V4(tmp); \
        vstore4(tmp, 0, buf + off); \
    }

#define ADD_STORE_BUF_ARRAY5(v, off, bet, buf)                        \
    {                                                        \
		T4 tmp = vload4(0, buf + off); \
		tmp *= bet; \
		tmp += (T4)(v[0], v[1], v[2], v[3]); \
        ACTIVATION_V4(tmp); \
		vstore4(tmp, 0, buf + off); \
        buf[off + 4] = buf[off + 4] * bet + v[4]; \
		ACTIVATION_V1(buf[off + 4])	; \
    }

#define ADD_STORE_BUF_ARRAY6(v, off, bet, buf)                      \
    {                                                      \
        T3 tmp = vload3(0, buf + off); \
		tmp *= bet; \
		tmp += (T3)(v[0], v[1], v[2]); \
		ACTIVATION_V3(tmp); \
        vstore3(tmp, 0, buf + off);     \
		tmp = vload3(0, buf + off + 3); \
		tmp *= bet; \
		tmp += (T3)(v[3], v[4], v[5]); \
		ACTIVATION_V3(tmp); \
        vstore3(tmp, 0, buf + off + 3); \
    }

#define ADD_STORE_BUF_ARRAY7(v, off, bet, buf)                        \
    {                                                        \
        T4 tmp4 = vload4(0, buf + off); \
		tmp4 *= bet; \
		tmp4 += (T4)(v[0], v[1], v[2], v[3]); \
        ACTIVATION_V4(tmp4); \
		vstore4(tmp4, 0, buf + off); \
		T3 tmp3 = vload3(0, buf + off + 4); \
		tmp3 *= bet; \
		tmp3 += (T3)(v[4], v[5], v[6]); \
		ACTIVATION_V3(tmp3); \
        vstore3(tmp3, 0, buf + off + 4);   \
    }

#define ADD_STORE_BUF_ARRAY8(v, off, bet, buf)                                                \
    {                                                                                \
        T8 tmp = vload8(0, buf + off); \
		tmp *= bet; \
		tmp += (T8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); \
        ACTIVATION_V8(tmp); \
		vstore8(tmp, 0, buf + off); \
    }

#if (LN == 1)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
    }

#elif (LN == 2)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#elif (LN == 3)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 2) {\
            ADD_STORE_BUF_ARRAY3(v, off, bet, buf); \
        } else if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#elif (LN == 4)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 3) {\
            ADD_STORE_BUF_ARRAY4(v, off, bet, buf); \
        } else if (ex > 2) {\
            ADD_STORE_BUF_ARRAY3(v, off, bet, buf); \
        } else if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#elif (LN == 5)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 4) {\
            ADD_STORE_BUF_ARRAY5(v, off, bet, buf); \
        } else if (ex > 3) {\
            ADD_STORE_BUF_ARRAY4(v, off, bet, buf); \
        } else if (ex > 2) {\
            ADD_STORE_BUF_ARRAY3(v, off, bet, buf); \
        } else if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#elif (LN == 6)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 5) {\
            ADD_STORE_BUF_ARRAY6(v, off, bet, buf); \
        } else if (ex > 4) {\
            ADD_STORE_BUF_ARRAY5(v, off, bet, buf); \
        } else if (ex > 3) {\
            ADD_STORE_BUF_ARRAY4(v, off, bet, buf); \
        } else if (ex > 2) {\
            ADD_STORE_BUF_ARRAY3(v, off, bet, buf); \
        } else if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#elif (LN == 7)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 6) {\
            ADD_STORE_BUF_ARRAY7(v, off, bet, buf); \
        } else if (ex > 5) {\
            ADD_STORE_BUF_ARRAY6(v, off, bet, buf); \
        } else if (ex > 4) {\
            ADD_STORE_BUF_ARRAY5(v, off, bet, buf); \
        } else if (ex > 3) {\
            ADD_STORE_BUF_ARRAY4(v, off, bet, buf); \
        } else if (ex > 2) {\
            ADD_STORE_BUF_ARRAY3(v, off, bet, buf); \
        } else if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#elif (LN == 8)

#define GEMM_ADD_STORE_C_X(v, off, ex, bet, buf)    \
    {                                  \
        if (ex > 7) {\
            ADD_STORE_BUF_ARRAY8(v, off, bet, buf); \
        } else if (ex > 6) {\
            ADD_STORE_BUF_ARRAY7(v, off, bet, buf); \
        } else if (ex > 5) {\
            ADD_STORE_BUF_ARRAY6(v, off, bet, buf); \
        } else if (ex > 4) {\
            ADD_STORE_BUF_ARRAY5(v, off, bet, buf); \
        } else if (ex > 3) {\
            ADD_STORE_BUF_ARRAY4(v, off, bet, buf); \
        } else if (ex > 2) {\
            ADD_STORE_BUF_ARRAY3(v, off, bet, buf); \
        } else if (ex > 1) {\
            ADD_STORE_BUF_ARRAY2(v, off, bet, buf); \
        } else {\
            ADD_STORE_BUF_ARRAY1(v, off, bet, buf); \
        }\
    }

#endif

#if (LM == 1)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)  \
    {                                   \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf); \
    }

#elif (LM == 2)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)  \
    {                                         \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);       \
        if (ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf); \
    }

#elif (LM == 3)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)            \
    {                                             \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);           \
        if(ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf);     \
        if(ey > 2) GEMM_ADD_STORE_C_X(v[2], off + str * 2, ex, bet, buf); \
    }

#elif (LM == 4)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)            \
    {                                             \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);           \
        if(ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf);     \
        if(ey > 2) GEMM_ADD_STORE_C_X(v[2], off + str * 2, ex, bet, buf); \
		if(ey > 3) GEMM_ADD_STORE_C_X(v[3], off + str * 3, ex, bet, buf); \
    }

#elif (LM == 5)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)            \
    {                                             \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);           \
        if (ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf);     \
        if (ey > 2) GEMM_ADD_STORE_C_X(v[2], off + str * 2, ex, bet, buf); \
        if (ey > 3) GEMM_ADD_STORE_C_X(v[3], off + str * 3, ex, bet, buf); \
        if (ey > 4) GEMM_ADD_STORE_C_X(v[4], off + str * 4, ex, bet, buf); \
    }

#elif (LM == 6)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)            \
    {                                             \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);           \
        if (ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf);     \
        if (ey > 2) GEMM_ADD_STORE_C_X(v[2], off + str * 2, ex, bet, buf); \
        if (ey > 3) GEMM_ADD_STORE_C_X(v[3], off + str * 3, ex, bet, buf); \
        if (ey > 4) GEMM_ADD_STORE_C_X(v[4], off + str * 4, ex, bet, buf); \
        if (ey > 5) GEMM_ADD_STORE_C_X(v[5], off + str * 5, ex, bet, buf); \
    }

#elif (LM == 7)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)            \
    {                                             \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);           \
        if (ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf);     \
        if (ey > 2) GEMM_ADD_STORE_C_X(v[2], off + str * 2, ex, bet, buf); \
        if (ey > 3) GEMM_ADD_STORE_C_X(v[3], off + str * 3, ex, bet, buf); \
        if (ey > 4) GEMM_ADD_STORE_C_X(v[4], off + str * 4, ex, bet, buf); \
        if (ey > 5) GEMM_ADD_STORE_C_X(v[5], off + str * 5, ex, bet, buf); \
        if (ey > 6) GEMM_ADD_STORE_C_X(v[6], off + str * 6, ex, bet, buf); \
    }

#elif (LM == 8)

#define GEMM_ADD_STORE_C(v, off, str, ex, ey, bet, buf)            \
    {                                             \
        GEMM_ADD_STORE_C_X(v[0], off, ex, bet, buf);           \
        if (ey > 1) GEMM_ADD_STORE_C_X(v[1], off + str, ex, bet, buf);     \
        if (ey > 2) GEMM_ADD_STORE_C_X(v[2], off + str * 2, ex, bet, buf); \
        if (ey > 3) GEMM_ADD_STORE_C_X(v[3], off + str * 3, ex, bet, buf); \
        if (ey > 4) GEMM_ADD_STORE_C_X(v[4], off + str * 4, ex, bet, buf); \
        if (ey > 5) GEMM_ADD_STORE_C_X(v[5], off + str * 5, ex, bet, buf); \
        if (ey > 6) GEMM_ADD_STORE_C_X(v[6], off + str * 6, ex, bet, buf); \
        if (ey > 7) GEMM_ADD_STORE_C_X(v[7], off + str * 7, ex, bet, buf); \
    }

#endif
)"
