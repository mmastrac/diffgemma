// out[t, d] = decode(table[ids[t]], d) * embed_scale. Mirrors embed_gather.metal.

extern "C" __global__ void embed_gather(
    const unsigned *blob,
    const unsigned *ids,
    float *out,
    unsigned hidden,
    unsigned num_tokens,
    unsigned w_off_lo,
    unsigned w_off_hi,
    float embed_scale,
    unsigned vocab,
    unsigned raw
) {
    const unsigned long long w_off = ((unsigned long long)w_off_hi << 32) | (unsigned long long)w_off_lo;
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total = hidden * num_tokens;
    if (gid >= total) {
        return;
    }
    const unsigned tok = gid / hidden;
    const unsigned d = gid % hidden;
    const unsigned id = ids[tok];
    if (id >= vocab) {
        return;
    }
    float v;
    if (raw != 0u) {
        const unsigned short *row =
            (const unsigned short *)((const char *)blob + w_off) + (unsigned long long)id * hidden;
        v = __uint_as_float((unsigned)row[d] << 16);
    } else {
        const unsigned char *row =
            (const unsigned char *)blob + w_off + (unsigned long long)id * (hidden + 2u);
        float scale = __uint_as_float((unsigned)(row[0] | ((unsigned)row[1] << 8)) << 16);
        v = scale * (float)((int)(signed char)row[2 + d]);
    }
    out[(unsigned long long)tok * hidden + d] = v * embed_scale;
}
