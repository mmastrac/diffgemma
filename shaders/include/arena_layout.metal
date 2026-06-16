#ifndef DGQ_ARENA_LAYOUT_METAL
#define DGQ_ARENA_LAYOUT_METAL

/// Arena scratch plane byte offsets (host-built; bind as b8). All planes 16B-aligned.
struct ArenaLayout {
    uint a_hidden;
    uint a_resid;
    uint a_attnq;
    uint a_attnk;
    uint a_attnv;
    uint a_attno;
    uint a_ffg;
    uint a_ffu;
    uint a_moein;
    uint a_dense;
    uint a_moeout;
    uint a_soft;
    uint a_stream;
    uint a_tmp;
    uint a_rs_sc;
    uint a_rs_samp;
    uint arena_bytes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

#endif
