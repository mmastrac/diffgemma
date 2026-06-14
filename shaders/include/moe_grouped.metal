#ifndef DGQ_INCLUDE_MOE_GROUPED_METAL
#define DGQ_INCLUDE_MOE_GROUPED_METAL

constant uint MOE_MAX_FF = 704u;
constant uint MOE_MAX_HIDDEN = 2816u;

struct MoeGroupedDims {
    uint canvas;
    uint hidden;
    uint moe_ff;
    uint n_experts;
};

// device atomic_float fetch_add is unreliable on MPS for moe_out scatter; CAS on uint bits.
inline void atomic_add_f32(device atomic_uint* bits, float val) {
    uint old = atomic_load_explicit(bits, memory_order_relaxed);
    for (;;) {
        float new_f = as_type<float>(old) + val;
        uint new_bits = as_type<uint>(new_f);
        if (atomic_compare_exchange_weak_explicit(
                bits, &old, new_bits, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

#endif
