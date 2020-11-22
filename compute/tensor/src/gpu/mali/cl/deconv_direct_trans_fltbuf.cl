#define loadFltval(off, str, flt, val) \
    {                                  \
        val.x = flt[off];              \
        val.y = flt[off + str];        \
        val.z = flt[off + str * 2];    \
        val.w = flt[off + str * 3];    \
    }
#define loadFltvalEdge(off, str, flt, val, edge) \
    {                                            \
        val.x = flt[off];                        \
        if (edge > 1)                            \
            val.y = flt[off + str];              \
        if (edge > 2)                            \
            val.z = flt[off + str * 2];          \
    }

//  conv filter gs[3] = {fwh, (fc+3)/4, (fn+3)/4*4};
// deconv filter gs[3] = {fwh, (fc+3)/4*4, (fn+3)/4};
// iohw -> nchwn4c4

__kernel void deconv_direct_trans_fltbuf(
    const int fwh, const int fc, const int fn, __global const T *fltdata, __global T *fltbuf)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    short ec = ((idy + 1) * 4 <= fc) ? 4 : (fc % 4);

    int flt_off = (idz * fc + idy * 4) * fwh + idx;

    T4 val = 0;

    int str = fwh;
    if (idz < fn) {
        if (ec == 4) {
            loadFltval(flt_off, str, fltdata, val);
        } else {
            loadFltvalEdge(flt_off, str, fltdata, val, ec);
        }
    }
    int bc = (fn + 4 - 1) / 4;
    int out_off;
    out_off = (idy * bc + idz / 4) * fwh * 4 + idx * 4 + (idz % 4);
    //    out_off = (idy / 4 * bc + idz) * fwh * 4 + idx * 4 + (idy % 4);

    vstore4(val, out_off, fltbuf);
}
