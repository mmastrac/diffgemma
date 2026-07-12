// GPU-side invariant reporting (P3.7). Included by subkernels; off unless K_DEBUG_FAST / K_DEBUG_DEEP.
#ifndef DGQ_DEBUG_STATUS_METAL
#define DGQ_DEBUG_STATUS_METAL

#include "fc_axes.metal"
#include "dgq_kernels.metal"

enum DbgErrorCode : uint {
    DbgErrNone = 0u,
    DbgErrIndexOob = 1u,
    DbgErrRouteTokenOob = 2u,
    DbgErrSoftmaxNormZero = 3u,
    DbgErrNonFinite = 4u,
    DbgErrEntropyTooHigh = 5u,
    DbgErrQuantUnsupported = 6u,
    DbgErrArenaOob = 7u,
    DbgErrBadDim = 8u,
    DbgErrNegativeEntropy = 9u,
};

struct DebugStatus {
    atomic_uint code;
    atomic_uint kernel_id;
    atomic_uint tg_id; // threadgroup id: tg.x | (tg.y << 16)
    atomic_uint value;
};

inline bool dgq_debug_fast_enabled() {
    return K_SHAPE_ASSERT || K_DEBUG_FAST;
}

inline bool dgq_debug_deep_enabled() {
    return K_DEBUG_DEEP;
}

/// First writer wins — later errors must not clobber root cause.
inline void debug_set_error(device DebugStatus *st, uint code, uint kernel_id,
                            uint tg, uint value) {
    if (st == nullptr) {
        return;
    }
    if (!dgq_debug_fast_enabled() && code != DbgErrNonFinite) {
        return;
    }
    uint expected = 0u;
    atomic_compare_exchange_weak_explicit(
        &st->code, &expected, code, memory_order_relaxed, memory_order_relaxed);
    if (expected == 0u) {
        atomic_store_explicit(&st->kernel_id, kernel_id, memory_order_relaxed);
        atomic_store_explicit(&st->tg_id, tg, memory_order_relaxed);
        atomic_store_explicit(&st->value, value, memory_order_relaxed);
    }
}

inline void dgq_assert_fast(bool cond, device DebugStatus *st, uint code, uint kernel_id,
                          uint value) {
    if (!dgq_debug_fast_enabled() || cond) {
        return;
    }
    const uint tg = 0u;
    debug_set_error(st, code, kernel_id, tg, value);
}

inline void dgq_assert_dims_nonzero(device DebugStatus *st, uint kernel_id, uint a, uint b) {
    if (!dgq_debug_fast_enabled()) {
        return;
    }
    if (a == 0u || b == 0u) {
        debug_set_error(st, DbgErrBadDim, kernel_id, 0u, (a << 16u) | b);
    }
}

inline void dgq_assert_index(device DebugStatus *st, uint kernel_id, uint idx, uint limit) {
    if (!dgq_debug_fast_enabled()) {
        return;
    }
    if (idx >= limit) {
        debug_set_error(st, DbgErrIndexOob, kernel_id, 0u, (limit << 16u) | idx);
    }
}

inline bool dgq_is_finite_f32(float x) {
    return !(isnan(x) || isinf(x));
}

inline void dgq_assert_finite_f32(device DebugStatus *st, uint kernel_id, float x, uint tag) {
    if (!dgq_debug_deep_enabled()) {
        return;
    }
    if (!dgq_is_finite_f32(x)) {
        debug_set_error(st, DbgErrNonFinite, kernel_id, 0u, tag);
    }
}

inline void dgq_assert_positive_f32(device DebugStatus *st, uint kernel_id, float x, uint tag) {
    if (!dgq_debug_fast_enabled()) {
        return;
    }
    if (!(x > 0.f) || !dgq_is_finite_f32(x)) {
        debug_set_error(st, DbgErrSoftmaxNormZero, kernel_id, 0u, tag);
    }
}

inline void dgq_assert_token_id(device DebugStatus *st, uint kernel_id, uint id, uint vocab) {
    if (!dgq_debug_fast_enabled()) {
        return;
    }
    if (id >= vocab) {
        debug_set_error(st, DbgErrIndexOob, kernel_id, 0u, (vocab << 16u) | id);
    }
}

inline void dgq_assert_entropy_nonneg(device DebugStatus *st, uint kernel_id, float ent, uint row) {
    if (!dgq_debug_fast_enabled()) {
        return;
    }
    if (!dgq_is_finite_f32(ent) || ent < -1e-4f) {
        debug_set_error(st, DbgErrNegativeEntropy, kernel_id, row, as_type<uint>(ent));
    }
}

#endif
