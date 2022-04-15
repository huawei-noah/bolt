R"(// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__kernel void adam(const int bx,
    const int size,
    const T alpha,
    const T beta1,
    const T beta2,
    const T epsilon,
    const int use_simple_epsilon,
    __global const T *grad,
    __global const T *betat1,
    __global const T *betat2,
    __global T *m,
    __global T *v,
    __global T *param)
{
    const int idx = get_global_id(0);
    if (idx >= bx) {
        return;
    }

    const T sqrt_beta_2_t_0 = sqrt(1.0f - betat2[0]);
    const T alpha_new = alpha * sqrt_beta_2_t_0 / (1.0f - betat1[0]);
    const T epsilon_new = use_simple_epsilon ? epsilon : epsilon * sqrt_beta_2_t_0;
    
    const int off = idx << 2;
    char ew = ((off + 4) <= size) ? 4 : (size & 3);
    T4 m_tmp;
    T4 v_tmp;
    T4 g;
    T4 p;
    if (ew == 4) {
        m_tmp = vload4(0, m + off);
        v_tmp = vload4(0, v + off);
        g = vload4(0, grad + off);
        p = vload4(0, param + off);
    }
    else {
        if (ew == 1) {
            m_tmp.x = m[off];
            v_tmp.x = v[off];
            g.x = grad[off];
            p.x = param[off];
        } if (ew == 2) {
            m_tmp.xy = vload2(0, m + off);
            v_tmp.xy = vload2(0, v + off);
            g.xy = vload2(0, grad + off);
            p.xy = vload2(0, param + off);
        } if (ew == 3) {
            m_tmp.xyz = vload3(0, m + off);
            v_tmp.xyz = vload3(0, v + off);
            g.xyz = vload3(0, grad + off);
            p.xyz = vload3(0, param + off);
        }
    }

    m_tmp.x = beta1 * m_tmp.x + (1.0f - beta1) * g.x;
    m_tmp.y = beta1 * m_tmp.y + (1.0f - beta1) * g.y;
    m_tmp.z = beta1 * m_tmp.z + (1.0f - beta1) * g.z;
    m_tmp.w = beta1 * m_tmp.w + (1.0f - beta1) * g.w;

    v_tmp.x = beta2 * v_tmp.x + (1.0f - beta2) * g.x * g.x;
    v_tmp.y = beta2 * v_tmp.y + (1.0f - beta2) * g.y * g.y;
    v_tmp.z = beta2 * v_tmp.z + (1.0f - beta2) * g.z * g.z;
    v_tmp.w = beta2 * v_tmp.w + (1.0f - beta2) * g.w * g.w;

    p.x = p.x - alpha_new * m_tmp.x / (sqrt(v_tmp.x) + epsilon_new);
    p.y = p.y - alpha_new * m_tmp.y / (sqrt(v_tmp.y) + epsilon_new);
    p.z = p.z - alpha_new * m_tmp.z / (sqrt(v_tmp.z) + epsilon_new);
    p.w = p.w - alpha_new * m_tmp.w / (sqrt(v_tmp.w) + epsilon_new);

    if (ew == 4) {
        vstore4(m_tmp, 0, m + off);
        vstore4(v_tmp, 0, v + off);
        vstore4(p, 0, param + off);
    }
    else {
        if (ew == 1) {
            m[off] = m_tmp.x;
            v[off] = v_tmp.x;
            param[off] = p.x;
        } if (ew == 2) {
            vstore2((T2)(m_tmp.x, m_tmp.y), 0, m + off);
            vstore2((T2)(v_tmp.x, v_tmp.y), 0, v + off);
            vstore2((T2)(p.x, p.y), 0, param + off);
        } if (ew == 3) {
            vstore3((T3)(m_tmp.x, m_tmp.y, m_tmp.z), 0, m + off);
            vstore3((T3)(v_tmp.x, v_tmp.y, v_tmp.z), 0, v + off);
            vstore3((T3)(p.x, p.y, p.z), 0, param + off);
        }
    }
}
)"