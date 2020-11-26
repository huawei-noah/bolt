__kernel void depth2space_nchw(const int blockSize,
    const int iw_str,
    const int iwh_str,
    const int iw_off,
    const int ih_off,
    const int oh_str,
    const int ohw_str,
    const int oh_off,
    const int ow_off,
    const int iw,
    const int ih,
    const int ic,
    __global const T *in,
    __global T *out)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    if (idx >= iw || idy >= ih) {
        return;
    }
    const int idz = get_global_id(2);
    const int bs2 = blockSize * blockSize;
    const int z_group = idz / bs2;
    const int z_group_lane = idz % bs2;
    const int z_group_lane_x = z_group_lane % blockSize;
    const int z_group_lane_y = z_group_lane / blockSize;

    const int z_off = z_group * (bs2 << 2) + z_group_lane;
    int in_off = z_off * iwh_str + (idy + ih_off) * iw_str + idx + iw_off;
    T4 val = 0;
    val.x = in[in_off];
    if (z_off + bs2 < ic) {
        val.y = in[in_off + bs2 * iwh_str];
    }
    if (z_off + bs2 * 2 < ic) {
        val.z = in[in_off + bs2 * 2 * iwh_str];
    }
    if (z_off + bs2 * 3 < ic) {
        val.w = in[in_off + bs2 * 3 * iwh_str];
    }

    int out_off = idy * blockSize + z_group_lane_y + oh_off;
    out_off += (idx * blockSize + z_group_lane_x + ow_off) * oh_str;
    out_off += z_group * ohw_str;
    vstore4(val, out_off, out);
}
