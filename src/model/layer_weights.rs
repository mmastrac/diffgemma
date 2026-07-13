use crate::config::{LayerType, TextConfig};
use crate::safetensors::Error;
use crate::tensor::TensorView;
use crate::weights::WeightStore;

/// Safetensors key names for one decoder layer.
#[derive(Debug, Clone)]
pub struct DecoderLayerKeys {
    pub layer: usize,
    pub input_layernorm: String,
    pub q_norm: String,
    pub k_norm: String,
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    pub post_attention_layernorm: String,
    pub pre_feedforward_layernorm: String,
    pub post_feedforward_layernorm: String,
    pub post_feedforward_layernorm_1: String,
    pub post_feedforward_layernorm_2: String,
    pub pre_feedforward_layernorm_2: String,
    pub mlp_gate: String,
    pub mlp_up: String,
    pub mlp_down: String,
    pub router_proj: String,
    pub router_scale: String,
    pub router_per_expert_scale: String,
    pub experts_gate_up: String,
    pub experts_down: String,
    pub layer_scalar: String,
}

impl DecoderLayerKeys {
    pub fn new(layer: usize) -> Self {
        let p = format!("model.decoder.layers.{layer}");
        Self {
            layer,
            input_layernorm: format!("{p}.input_layernorm.weight"),
            q_norm: format!("{p}.self_attn.q_norm.weight"),
            k_norm: format!("{p}.self_attn.k_norm.weight"),
            q_proj: format!("{p}.self_attn.q_proj.weight"),
            k_proj: format!("{p}.self_attn.k_proj.weight"),
            v_proj: format!("{p}.self_attn.v_proj.weight"),
            o_proj: format!("{p}.self_attn.o_proj.weight"),
            post_attention_layernorm: format!("{p}.post_attention_layernorm.weight"),
            pre_feedforward_layernorm: format!("{p}.pre_feedforward_layernorm.weight"),
            post_feedforward_layernorm: format!("{p}.post_feedforward_layernorm.weight"),
            post_feedforward_layernorm_1: format!("{p}.post_feedforward_layernorm_1.weight"),
            post_feedforward_layernorm_2: format!("{p}.post_feedforward_layernorm_2.weight"),
            pre_feedforward_layernorm_2: format!("{p}.pre_feedforward_layernorm_2.weight"),
            mlp_gate: format!("{p}.mlp.gate_proj.weight"),
            mlp_up: format!("{p}.mlp.up_proj.weight"),
            mlp_down: format!("{p}.mlp.down_proj.weight"),
            router_proj: format!("{p}.router.proj.weight"),
            router_scale: format!("{p}.router.scale"),
            router_per_expert_scale: format!("{p}.router.per_expert_scale"),
            experts_gate_up: format!("{p}.experts.gate_up_proj"),
            experts_down: format!("{p}.experts.down_proj"),
            layer_scalar: format!("{p}.layer_scalar"),
        }
    }
}

/// Expected weight shapes derived from `TextConfig` (PyTorch linear: `[out_features, in_features]`).
pub struct DecoderLayerShapes {
    pub q_proj: [i64; 2],
    pub k_proj: [i64; 2],
    pub v_proj: Option<[i64; 2]>,
    pub o_proj: [i64; 2],
    pub head_norm: [i64; 1],
    pub norm_1d: [i64; 1],
    pub mlp_gate_up: [i64; 2],
    pub mlp_down: [i64; 2],
    pub router_proj: [i64; 2],
    pub experts_gate_up: [i64; 3],
    pub experts_down: [i64; 3],
}

impl DecoderLayerShapes {
    pub fn for_layer(cfg: &TextConfig, layer: usize) -> Result<Self, Error> {
        let layer_type = cfg
            .layer_types
            .get(layer)
            .ok_or(Error::Runtime("invalid layer index"))?;
        let h = cfg.hidden_size as i64;
        let (head_dim, n_kv) = match layer_type {
            LayerType::SlidingAttention => (cfg.head_dim, cfg.num_key_value_heads),
            LayerType::FullAttention => (cfg.global_head_dim, cfg.num_global_key_value_heads),
        };
        let q = (cfg.num_attention_heads * head_dim) as i64;
        let kv = (n_kv * head_dim) as i64;
        let moe = cfg.moe_intermediate_size as i64;
        let shared = cfg.intermediate_size as i64;
        let experts = cfg.num_experts as i64;
        Ok(Self {
            q_proj: [q, h],
            k_proj: [kv, h],
            v_proj: match layer_type {
                LayerType::SlidingAttention => Some([kv, h]),
                LayerType::FullAttention => None,
            },
            o_proj: [h, q],
            head_norm: [head_dim as i64],
            norm_1d: [h],
            mlp_gate_up: [shared, h],
            mlp_down: [h, shared],
            router_proj: [experts, h],
            experts_gate_up: [experts, moe * 2, h],
            experts_down: [experts, h, moe],
        })
    }
}

#[allow(dead_code)] // fields used starting in phase 3 (decoder forward)
pub struct DecoderLayerWeights<'a> {
    pub keys: DecoderLayerKeys,
    pub input_layernorm: TensorView<'a>,
    pub q_norm: TensorView<'a>,
    pub k_norm: TensorView<'a>,
    pub q_proj: TensorView<'a>,
    pub k_proj: TensorView<'a>,
    pub v_proj: Option<TensorView<'a>>,
    pub o_proj: TensorView<'a>,
    pub post_attention_layernorm: TensorView<'a>,
    pub pre_feedforward_layernorm: TensorView<'a>,
    pub post_feedforward_layernorm: TensorView<'a>,
    pub post_feedforward_layernorm_1: TensorView<'a>,
    pub post_feedforward_layernorm_2: TensorView<'a>,
    pub pre_feedforward_layernorm_2: TensorView<'a>,
    pub mlp_gate: TensorView<'a>,
    pub mlp_up: TensorView<'a>,
    pub mlp_down: TensorView<'a>,
    pub router_proj: TensorView<'a>,
    pub router_scale: TensorView<'a>,
    pub router_per_expert_scale: TensorView<'a>,
    pub experts_gate_up: TensorView<'a>,
    pub experts_down: TensorView<'a>,
    pub layer_scalar: TensorView<'a>,
}

impl<'a> DecoderLayerWeights<'a> {
    pub fn load(store: &'a WeightStore, layer: usize, cfg: &TextConfig) -> Result<Self, Error> {
        let keys = DecoderLayerKeys::new(layer);
        let shapes = DecoderLayerShapes::for_layer(cfg, layer)?;

        let input_layernorm = store.tensor(&keys.input_layernorm)?;
        input_layernorm.expect_shape(&shapes.norm_1d)?;

        let q_norm = store.tensor(&keys.q_norm)?;
        q_norm.expect_shape(&shapes.head_norm)?;

        let k_norm = store.tensor(&keys.k_norm)?;
        k_norm.expect_shape(&shapes.head_norm)?;

        let q_proj = store.tensor(&keys.q_proj)?;
        q_proj.expect_shape(&shapes.q_proj)?;

        let k_proj = store.tensor(&keys.k_proj)?;
        k_proj.expect_shape(&shapes.k_proj)?;

        let v_proj = match shapes.v_proj {
            Some(shape) => {
                let v = store.tensor(&keys.v_proj)?;
                v.expect_shape(&shape)?;
                Some(v)
            }
            None => None,
        };

        let o_proj = store.tensor(&keys.o_proj)?;
        o_proj.expect_shape(&shapes.o_proj)?;

        let post_attention_layernorm = store.tensor(&keys.post_attention_layernorm)?;
        post_attention_layernorm.expect_shape(&shapes.norm_1d)?;

        let pre_feedforward_layernorm = store.tensor(&keys.pre_feedforward_layernorm)?;
        pre_feedforward_layernorm.expect_shape(&shapes.norm_1d)?;

        let post_feedforward_layernorm = store.tensor(&keys.post_feedforward_layernorm)?;
        post_feedforward_layernorm.expect_shape(&shapes.norm_1d)?;

        let post_feedforward_layernorm_1 = store.tensor(&keys.post_feedforward_layernorm_1)?;
        post_feedforward_layernorm_1.expect_shape(&shapes.norm_1d)?;

        let post_feedforward_layernorm_2 = store.tensor(&keys.post_feedforward_layernorm_2)?;
        post_feedforward_layernorm_2.expect_shape(&shapes.norm_1d)?;

        let pre_feedforward_layernorm_2 = store.tensor(&keys.pre_feedforward_layernorm_2)?;
        pre_feedforward_layernorm_2.expect_shape(&shapes.norm_1d)?;

        let mlp_gate = store.tensor(&keys.mlp_gate)?;
        mlp_gate.expect_shape(&shapes.mlp_gate_up)?;

        let mlp_up = store.tensor(&keys.mlp_up)?;
        mlp_up.expect_shape(&shapes.mlp_gate_up)?;

        let mlp_down = store.tensor(&keys.mlp_down)?;
        mlp_down.expect_shape(&shapes.mlp_down)?;

        let router_proj = store.tensor(&keys.router_proj)?;
        router_proj.expect_shape(&shapes.router_proj)?;

        let router_scale = store.tensor(&keys.router_scale)?;
        let router_per_expert_scale = store.tensor(&keys.router_per_expert_scale)?;

        let experts_gate_up = store.tensor(&keys.experts_gate_up)?;
        experts_gate_up.expect_shape(&shapes.experts_gate_up)?;

        let experts_down = store.tensor(&keys.experts_down)?;
        experts_down.expect_shape(&shapes.experts_down)?;

        let layer_scalar = store.tensor(&keys.layer_scalar)?;

        Ok(Self {
            keys,
            input_layernorm,
            q_norm,
            k_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
            mlp_gate,
            mlp_up,
            mlp_down,
            router_proj,
            router_scale,
            router_per_expert_scale,
            experts_gate_up,
            experts_down,
            layer_scalar,
        })
    }

    #[allow(dead_code)]
    pub fn print_summary(&self) {
        println!("decoder layer {} weights (validated)", self.keys.layer);
        self.q_proj.print_info();
    }
}
