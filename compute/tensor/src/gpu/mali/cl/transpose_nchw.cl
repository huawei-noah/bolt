// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "kernel_def.h"

#if (DN <= 2)
#define SWAP_ID_CORE(id, dim, d0, n0) \
    {                                 \
        if (dim == n0) {              \
            id = d0;                  \
        }                             \
    }
#define SWAP_STR()            \
    {                         \
        if (dim0 == 1) {      \
            out_str = ow_str; \
        }                     \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1);
#define SWAP_ID_1 SWAP_ID_CORE(od1, dim1, id0 << 2, 0);
#elif (DN == 3)
#define SWAP_ID_CORE(id, dim, d0, n0, d1, n1) \
    {                                         \
        if (dim == n0) {                      \
            id = d0;                          \
        } else if (dim == n1) {               \
            id = d1;                          \
        }                                     \
    }
#define SWAP_STR()                     \
    {                                  \
        if (dim1 == 0) {               \
            out_str = ow_str;          \
        } else if (dim2 == 0) {        \
            out_str = ow_str * oh_str; \
        }                              \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1, id2, 2);
#define SWAP_ID_1 SWAP_ID_CORE(od1, dim1, id0 << 2, 0, id2, 2);
#define SWAP_ID_2 SWAP_ID_CORE(od2, dim2, id0 << 2, 0, id1, 1);
#elif (DN == 4)
#define SWAP_ID_CORE(id, dim, d0, n0, d1, n1, d2, n2) \
    {                                                 \
        if (dim == n0) {                              \
            id = d0;                                  \
        } else if (dim == n1) {                       \
            id = d1;                                  \
        } else if (dim == n2) {                       \
            id = d2;                                  \
        }                                             \
    }
#define SWAP_STR()                                \
    {                                             \
        if (dim1 == 0) {                          \
            out_str = ow_str;                     \
        } else if (dim2 == 0) {                   \
            out_str = ow_str * oh_str;            \
        } else if (dim3 == 0) {                   \
            out_str = ow_str * oh_str * oDimLen2; \
        }                                         \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1, id2, 2, id3, 3);
#define SWAP_ID_1 SWAP_ID_CORE(od1, dim1, id0 << 2, 0, id2, 2, id3, 3);
#define SWAP_ID_2 SWAP_ID_CORE(od2, dim2, id0 << 2, 0, id1, 1, id3, 3);
#define SWAP_ID_3 SWAP_ID_CORE(od3, dim3, id0 << 2, 0, id1, 1, id2, 2);
#elif (DN == 5)
#define SWAP_ID_CORE(id, dim, d0, n0, d1, n1, d2, n2, d3, n3) \
    {                                                         \
        if (dim == n0) {                                      \
            id = d0;                                          \
        } else if (dim == n1) {                               \
            id = d1;                                          \
        } else if (dim == n2) {                               \
            id = d2;                                          \
        } else if (dim == n3) {                               \
            id = d3;                                          \
        }                                                     \
    }
#define SWAP_STR()                                           \
    {                                                        \
        if (dim1 == 0) {                                     \
            out_str = ow_str;                                \
        } else if (dim2 == 0) {                              \
            out_str = ow_str * oh_str;                       \
        } else if (dim3 == 0) {                              \
            out_str = ow_str * oh_str * oDimLen2;            \
        } else if (dim4 == 0) {                              \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3; \
        }                                                    \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1, id2, 2, id3, 3, id4, 4);
#define SWAP_ID_1 SWAP_ID_CORE(od1, dim1, id0 << 2, 0, id2, 2, id3, 3, id4, 4);
#define SWAP_ID_2 SWAP_ID_CORE(od2, dim2, id0 << 2, 0, id1, 1, id3, 3, id4, 4);
#define SWAP_ID_3 SWAP_ID_CORE(od3, dim3, id0 << 2, 0, id1, 1, id2, 2, id4, 4);
#define SWAP_ID_4 SWAP_ID_CORE(od4, dim4, id0 << 2, 0, id1, 1, id2, 2, id3, 3);
#elif (DN == 6)
#define SWAP_ID_CORE(id, dim, d0, n0, d1, n1, d2, n2, d3, n3, d4, n4) \
    {                                                                 \
        if (dim == n0) {                                              \
            id = d0;                                                  \
        } else if (dim == n1) {                                       \
            id = d1;                                                  \
        } else if (dim == n2) {                                       \
            id = d2;                                                  \
        } else if (dim == n3) {                                       \
            id = d3;                                                  \
        } else if (dim == n4) {                                       \
            id = d4;                                                  \
        }                                                             \
    }
#define SWAP_STR()                                                      \
    {                                                                   \
        if (dim1 == 0) {                                                \
            out_str = ow_str;                                           \
        } else if (dim2 == 0) {                                         \
            out_str = ow_str * oh_str;                                  \
        } else if (dim3 == 0) {                                         \
            out_str = ow_str * oh_str * oDimLen2;                       \
        } else if (dim4 == 0) {                                         \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3;            \
        } else if (dim5 == 0) {                                         \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3 * oDimLen4; \
        }                                                               \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1, id2, 2, id3, 3, id4, 4, id5, 5);
#define SWAP_ID_1 SWAP_ID_CORE(od1, dim1, id0 << 2, 0, id2, 2, id3, 3, id4, 4, id5, 5);
#define SWAP_ID_2 SWAP_ID_CORE(od2, dim2, id0 << 2, 0, id1, 1, id3, 3, id4, 4, id5, 5);
#define SWAP_ID_3 SWAP_ID_CORE(od3, dim3, id0 << 2, 0, id1, 1, id2, 2, id4, 4, id5, 5);
#define SWAP_ID_4 SWAP_ID_CORE(od4, dim4, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id5, 5);
#define SWAP_ID_5 SWAP_ID_CORE(od5, dim5, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id4, 4);
#elif (DN == 7)
#define SWAP_ID_CORE(id, dim, d0, n0, d1, n1, d2, n2, d3, n3, d4, n4, d5, n5) \
    {                                                                         \
        if (dim == n0) {                                                      \
            id = d0;                                                          \
        } else if (dim == n1) {                                               \
            id = d1;                                                          \
        } else if (dim == n2) {                                               \
            id = d2;                                                          \
        } else if (dim == n3) {                                               \
            id = d3;                                                          \
        } else if (dim == n4) {                                               \
            id = d4;                                                          \
        } else if (dim == n5) {                                               \
            id = d5;                                                          \
        }                                                                     \
    }
#define SWAP_STR()                                                                 \
    {                                                                              \
        if (dim1 == 0) {                                                           \
            out_str = ow_str;                                                      \
        } else if (dim2 == 0) {                                                    \
            out_str = ow_str * oh_str;                                             \
        } else if (dim3 == 0) {                                                    \
            out_str = ow_str * oh_str * oDimLen2;                                  \
        } else if (dim4 == 0) {                                                    \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3;                       \
        } else if (dim5 == 0) {                                                    \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3 * oDimLen4;            \
        } else if (dim6 == 0) {                                                    \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3 * oDimLen4 * oDimLen5; \
        }                                                                          \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1, id2, 2, id3, 3, id4, 4, id5, 5, id6, 6);
#define SWAP_ID_1 SWAP_ID_CORE(od1, dim1, id0 << 2, 0, id2, 2, id3, 3, id4, 4, id5, 5, id6, 6);
#define SWAP_ID_2 SWAP_ID_CORE(od2, dim2, id0 << 2, 0, id1, 1, id3, 3, id4, 4, id5, 5, id6, 6);
#define SWAP_ID_3 SWAP_ID_CORE(od3, dim3, id0 << 2, 0, id1, 1, id2, 2, id4, 4, id5, 5, id6, 6);
#define SWAP_ID_4 SWAP_ID_CORE(od4, dim4, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id5, 5, id6, 6);
#define SWAP_ID_5 SWAP_ID_CORE(od5, dim5, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id4, 4, id6, 6);
#define SWAP_ID_6 SWAP_ID_CORE(od5, dim5, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id4, 4, id5, 5);
#elif (DN == 8)
#define SWAP_ID_CORE(id, dim, d0, n0, d1, n1, d2, n2, d3, n3, d4, n4, d5, n5, d6, n6) \
    {                                                                                 \
        if (dim == n0) {                                                              \
            id = d0;                                                                  \
        } else if (dim == n1) {                                                       \
            id = d1;                                                                  \
        } else if (dim == n2) {                                                       \
            id = d2;                                                                  \
        } else if (dim == n3) {                                                       \
            id = d3;                                                                  \
        } else if (dim == n4) {                                                       \
            id = d4;                                                                  \
        } else if (dim == n5) {                                                       \
            id = d5;                                                                  \
        } else if (dim == n6) {                                                       \
            id = d6;                                                                  \
        }                                                                             \
    }
#define SWAP_STR()                                                                            \
    {                                                                                         \
        if (dim1 == 0) {                                                                      \
            out_str = ow_str;                                                                 \
        } else if (dim2 == 0) {                                                               \
            out_str = ow_str * oh_str;                                                        \
        } else if (dim3 == 0) {                                                               \
            out_str = ow_str * oh_str * oDimLen2;                                             \
        } else if (dim4 == 0) {                                                               \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3;                                  \
        } else if (dim5 == 0) {                                                               \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3 * oDimLen4;                       \
        } else if (dim6 == 0) {                                                               \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3 * oDimLen4 * oDimLen5;            \
        } else if (dim7 == 0) {                                                               \
            out_str = ow_str * oh_str * oDimLen2 * oDimLen3 * oDimLen4 * oDimLen5 * oDimLen6; \
        }                                                                                     \
    }
#define SWAP_ID_0 SWAP_ID_CORE(od0, dim0, id1, 1, id2, 2, id3, 3, id4, 4, id5, 5, id6, 6, id7, 7);
#define SWAP_ID_1 \
    SWAP_ID_CORE(od1, dim1, id0 << 2, 0, id2, 2, id3, 3, id4, 4, id5, 5, id6, 6, id7, 7);
#define SWAP_ID_2 \
    SWAP_ID_CORE(od2, dim2, id0 << 2, 0, id1, 1, id3, 3, id4, 4, id5, 5, id6, 6, id7, 7);
#define SWAP_ID_3 \
    SWAP_ID_CORE(od3, dim3, id0 << 2, 0, id1, 1, id2, 2, id4, 4, id5, 5, id6, 6, id7, 7);
#define SWAP_ID_4 \
    SWAP_ID_CORE(od4, dim4, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id5, 5, id6, 6, id7, 7);
#define SWAP_ID_5 \
    SWAP_ID_CORE(od5, dim5, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id4, 4, id6, 6, id7, 7);
#define SWAP_ID_6 \
    SWAP_ID_CORE(od5, dim5, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id4, 4, id5, 5, id7, 7);
#define SWAP_ID_7 \
    SWAP_ID_CORE(od5, dim5, id0 << 2, 0, id1, 1, id2, 2, id3, 3, id4, 4, id5, 5, id6, 6);
#endif

__kernel void KERNEL_NAME(const int iw_str,
    const int ih_str,
    const int ow_str,
    const int oh_str,
    const int i_off,
    const int o_off,
    const int dim0,
    const int dim1,
    const int dim2,
#if (DN > 3)
    const int dim3,
    const int iDimLen2,
    const int oDimLen2,
#endif
#if (DN > 4)
    const int dim4,
    const int iDimLen3,
    const int oDimLen3,
#endif
#if (DN > 5)
    const int dim5,
    const int iDimLen4,
    const int oDimLen4,
#endif
#if (DN > 6)
    const int dim6,
    const int iDimLen5,
    const int oDimLen5,
#endif
#if (DN > 7)
    const int dim7,
    const int iDimLen6,
    const int oDimLen6,
#endif
    const int iw,
    const int bx,
    const int by,
    READ_ONLY_KERNEL_MEM in,
    KERNEL_MEM out)
{
    int id0 = get_global_id(0);
    int id1 = get_global_id(1);
    int id2 = get_global_id(2);
    if (id0 >= bx || id1 >= by) {
        return;
    }
    int idz = id2;
    int out_str = 1;
    int od0 = id0 << 2;
    int od1 = id1;
#if (DN > 3)
    int tmpId = id2;
    id2 = tmpId % iDimLen2;
    int id3 = tmpId / iDimLen2;
#endif
#if (DN > 4)
    tmpId = id3;
    id3 = tmpId % iDimLen3;
    int id4 = tmpId / iDimLen3;
#endif
#if (DN > 5)
    tmpId = id4;
    id4 = tmpId % iDimLen4;
    int id5 = tmpId / iDimLen4;
#endif
#if (DN > 6)
    tmpId = id5;
    id5 = tmpId % iDimLen5;
    int id6 = tmpId / iDimLen5;
#endif
#if (DN > 7)
    tmpId = id6;
    id6 = tmpId % iDimLen6;
    int id7 = tmpId / iDimLen6;
#endif
    int od2 = id2;
#if (DN > 3)
    int od3 = id3;
#endif
#if (DN > 4)
    int od4 = id4;
#endif
#if (DN > 5)
    int od5 = id5;
#endif
#if (DN > 6)
    int od6 = id6;
#endif
#if (DN > 7)
    int od7 = id7;
#endif

#if defined(USE_INPUT_IMG)
    int4 in_off = (int4)(id0, id1, idz, 0);
#else
    const int in_off = (idz * ih_str + id1) * iw_str + (id0 << 2) + i_off;
#endif
    T4 val = 0;
    char ew = (((id0 << 2) + 4) <= iw) ? 4 : (iw & 3);
    LOAD_MEM_V4_C1(val, in_off, ew, in);

    int z_off = 0;
    SWAP_ID_0;
    SWAP_ID_1;
#if (DN > 2)
    SWAP_ID_2;
    z_off += od2;
#endif
#if (DN > 3)
    SWAP_ID_3;
    z_off += od3 * oDimLen2;
#endif
#if (DN > 4)
    SWAP_ID_4;
    z_off += od4 * oDimLen3 * oDimLen2;
#endif
#if (DN > 5)
    SWAP_ID_5;
    z_off += od5 * oDimLen4 * oDimLen3 * oDimLen2;
#endif
#if (DN > 6)
    SWAP_ID_6;
    z_off += od6 * oDimLen5 * oDimLen4 * oDimLen3 * oDimLen2;
#endif
#if (DN > 7)
    SWAP_ID_7;
    z_off += od7 * oDimLen6 * oDimLen5 * oDimLen4 * oDimLen3 * oDimLen2;
#endif
    SWAP_STR();
    int out_off = (z_off * oh_str + od1) * ow_str + od0 + o_off;
    out[out_off] = val.x;
    if (ew > 1) {
        out[out_off + out_str] = val.y;
    }
    if (ew > 2) {
        out[out_off + out_str * 2] = val.z;
    }
    if (ew > 3) {
        out[out_off + out_str * 3] = val.w;
    }
}
