// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include <string.h>
#include <math.h>

#include "cpu/general/tensor_computing_general.h"

EE lstm_general(TensorDesc inputDesc, const void* input, TensorDesc filterDesc, const void* filter,
    LSTMDesc lstmDesc, TensorDesc biasDesc, const void* bias, U32 tmpBytes, void* tmp,
    TensorDesc outputDesc, void* output)
{
    UNUSED(tmpBytes);

    F16 *inArray = (F16*)input;
    F16 *filterArray = (F16*)filter;
    F16 *biasArray = (F16*)bias;
    F16 *outArray = (F16*)output;

    DataType idt, fdt, bdt, odt;
    DataFormat idf, fdf, odf;
    U32 in_b, in_t, in_x;
    U32 out_b, out_t, out_h;
    U32 fk, fn;
    U32 bh;
    CHECK_STATUS_WITH_RETURN(tensor3dGet(inputDesc, &idt, &idf, &in_b, &in_t, &in_x));
    CHECK_STATUS_WITH_RETURN(tensor2dfGet(filterDesc, &fdt, &fdf, &fn, &fk));
    CHECK_STATUS_WITH_RETURN(tensor1dGet(biasDesc, &bdt, &bh));
    CHECK_STATUS_WITH_RETURN(tensor3dGet(outputDesc, &odt, &odf, &out_b, &out_t, &out_h));

    U32 h_dim = lstmDesc.num_output;
    U32 x_dim = in_x;
    U32 step = in_t;
    U32 batch = in_b;

    if (!(idt == DT_F16 && fdt == DT_F16 && bdt == DT_F16 && odt == DT_F16)) {
        return NOT_MATCH;
    }

    if (!(h_dim == out_h && 4*h_dim == fn && bh == fn && (in_x+out_h) == fk && in_b == out_b && in_t == out_t)) {
        return NOT_MATCH;
    }

    F16 *cell_state = (F16*)tmp;
    F16 *in_hx = cell_state + batch*h_dim;
    F16 *out_4h = in_hx + batch*(h_dim+x_dim);
    // initialize c_t, h_t
    memset(cell_state, 0, batch*h_dim);
    memset(in_hx, 0, batch*h_dim);

    for (U32 m = 0; m < batch; m++) {
        for (U32 t = 0; t < step; t++) {
            memcpy(in_hx + m*(h_dim+x_dim), inArray + (m*step+t)*x_dim, x_dim*sizeof(F16));
            // MVM
            F16 *out = out_4h + m*fn;
            F16 *in0 = in_hx + m*fk;

            for (U32 n = 0; n < fn; n++) {
                F16 *in = in0;
                for (U32 k = 0; k < fk; k++) {
                    out[n] += in[k] * filterArray[n*fk + k];
                }
                out[n] += biasArray[n];
            }
            F16 *out_i = out_4h + m*fn;
            F16 *out_f = out_i + h_dim;
            F16 *out_o = out_i + h_dim*2;
            F16 *out_g = out_i + h_dim*3;
            F16 *cell = cell_state + m*h_dim;
            F16 *out_hidden = in_hx + m*h_dim;

            for (U32 h = 0; h < h_dim; h++) {
                F16 I = 1.0f / (1.0f + exp(-out_i[h]));
                F16 F = 1.0f / (1.0f + exp(-out_f[h]));
                F16 O = 1.0f / (1.0f + exp(-out_o[h]));
                F16 G = tanh(out_g[h]);
                cell[h] = cell[h]*F + I*G;
                out_hidden[h] = O*tanh(cell[h]);
                outArray[t*batch*h_dim + m*h_dim + h] = out_hidden[h];
            }

        }
    }
    return SUCCESS;
}
