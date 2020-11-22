__kernel void depth2space(const int iw,
    const int ih,
    const int iw_str,
    const int ih_str,
    const int iw_off,
    const int ih_off,
    const int ow_str,
    const int oh_str,
    const int ow_off,
    const int oh_off,
    __global const T *in,
    __global uchar *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= ih || idy >= (iw << 2)) {
        return;
    }
    const int ix = idx;
    const int iy = idy % iw;
    const int iz = idy / iw;

    const int in_off = (iz * iw_str + iy + iw_off) * ih_str + ix + ih_off;
    T4 tmp = vload4(in_off, in);
    uchar4 val;
    val.x = tmp.x * 255.0;
    val.y = tmp.y * 255.0;
    val.z = tmp.z * 255.0;
    val.w = tmp.w * 255.0;

    const int out_off = ((ix << 2) + iz + oh_off) * ow_str + (iy << 2) + ow_off;
    vstore4(val, 0, out + out_off);
}
