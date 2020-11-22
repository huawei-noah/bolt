__kernel void space2depth(const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int oh_str,
    const int ohw_str,
    const int ow_off,
    const int oh_off,
    const int bx,
    const int by,
    __global const uchar *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= bx || idy >= by) {
        return;
    }

    const int in_off = ((idx << 2) + ih_off) * iw_str + (idy << 2) + iw_off;
    uchar4 tmp0 = vload4(0, in + in_off);
    uchar4 tmp1 = vload4(0, in + in_off + iw_str);
    uchar4 tmp2 = vload4(0, in + in_off + (iw_str << 1));
    uchar4 tmp3 = vload4(0, in + in_off + iw_str * 3);
    T4 val0, val1, val2, val3;
    val0.x = tmp0.x / (T)(255);
    val0.y = tmp0.y / (T)(255);
    val0.z = tmp0.z / (T)(255);
    val0.w = tmp0.w / (T)(255);
    val1.x = tmp1.x / (T)(255);
    val1.y = tmp1.y / (T)(255);
    val1.z = tmp1.z / (T)(255);
    val1.w = tmp1.w / (T)(255);
    val2.x = tmp2.x / (T)(255);
    val2.y = tmp2.y / (T)(255);
    val2.z = tmp2.z / (T)(255);
    val2.w = tmp2.w / (T)(255);
    val3.x = tmp3.x / (T)(255);
    val3.y = tmp3.y / (T)(255);
    val3.z = tmp3.z / (T)(255);
    val3.w = tmp3.w / (T)(255);

    const int out_off = (idy + ow_off) * oh_str + idx + oh_off;
    vstore4(val0, out_off, out);
    vstore4(val1, out_off + ohw_str, out);
    vstore4(val2, out_off + ohw_str * 2, out);
    vstore4(val3, out_off + ohw_str * 3, out);
}
