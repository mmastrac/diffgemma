> This is the conceptual overview. The precise implemented contract — sampler
> semantics (accept rule, argmax commit, no-freeze/unfreezing, early stop),
> prefill selection, SC details, and every deliberate divergence from the
> MLX/HF reference — lives in SPEC.md.

## The core trade: memory bandwidth → compute

The fundamental motivation is the AR decode bottleneck. Standard autoregressive models are memory-bandwidth-bound — each forward pass produces exactly one token, and the GPU sweeps all model weights through HBM just to advance one vocabulary lookup. Tensor cores sit largely idle because the arithmetic intensity of per-token decode is too low to saturate them.

DiffusionGemma bypasses this by shifting the bottleneck from memory bandwidth to compute, generating and refining a 256-token canvas in parallel. By providing the GPU with a large parallel workload, it utilizes tensor cores that would otherwise sit idle during local serving. Each denoising forward pass operates on 256 tokens simultaneously, which is dense enough to actually saturate the compute pipeline. On a single H100 with FP8 in low-batch settings, it exceeds 1,100 tokens/second.

One immediate corollary: this only helps on compute-bound hardware. Unified-memory architectures like Apple Silicon — which are memory-bandwidth-bound rather than compute-bound during inference — may not see the same acceleration.

## Backbone

The backbone is the 26B A4B Gemma 4 MoE architecture. The "A4B" names the effective inference cost:

25.2B total parameters, 3.8B active per forward pass, 30 layers, 8 active experts out of 128 total (plus 1 shared), 262K-token vocabulary, up to 256K context, a ~550M-parameter vision encoder.

The sparse MoE means the effective per-forward-pass cost is closer to a dense ~4B model — which is how it fits in 18 GB VRAM when quantized.

## Discrete diffusion, not continuous

Image diffusion adds Gaussian noise to continuous pixel values. Text diffusion operates over a discrete vocabulary, so "noise" means something different: noised positions are replaced with uniformly random token IDs from the 262K vocabulary. A fully noised canvas is 256 random tokens.

Rather than predicting tokens sequentially, DiffusionGemma starts with a canvas of random placeholder tokens and iteratively refines them in parallel. Over multiple denoising passes, highly confident tokens help resolve adjacent positions, causing the entire sequence to snap into focus.

This is also distinct from masked LMs (which use a single fixed `[MASK]` token) — uncertain positions get fresh random samples at every denoising step, not a stable placeholder.

## The two attention phases

DiffusionGemma uses the same transformer weights in two distinct modes:

**Causal prefill (encoder role).** The prompt is processed with standard causal attention, exactly as in autoregressive Gemma 4. This builds the KV cache. It runs once at the start, and again once per completed canvas block to extend the cache.

**Bidirectional denoising (decoder role).** The denoising step uses bidirectional attention — query tokens at any position on the canvas can attend to all other canvas tokens as well as the KV cache. This lets the model process context bidirectionally within the block. Within the canvas, there's no causal mask. Canvas tokens attend causally to the KV cache (past context), but fully and symmetrically to each other.

The encoder reads the clean prompt and response sequence with causal attention; the decoder denoises the canvas — the response region — with bidirectional (block-causal) attention, predicting the clean token at every canvas position simultaneously.

## The entropy-bound sampler

This is the critical inference-time mechanism. After each denoising step, the sampler measures per-token prediction entropy across all 256 canvas positions:

Every denoising step re-samples all canvas positions, but only the positions the model is confident about are kept; the rest are discarded and replaced with fresh random tokens for the next step. Confidence is measured by the entropy of each position's predicted distribution — low entropy means the model has largely made up its mind.

The recommended production config (from the model card) runs with entropy bound = 0.1 nats for token selection (natural-log entropy, per the reference code), max 48 steps, and a linear temperature decay from 0.8 → 0.4. Early stopping requires both: (a) average canvas entropy below 0.005, AND (b) argmax predictions identical across two consecutive steps. Both conditions must hold simultaneously. Simpler tasks (structured code, constrained output) converge faster; more open-ended generation uses more steps.

## Block-autoregressive chaining

For sequences longer than 256 tokens, once a canvas is fully denoised, the model runs a causal prefill over those committed tokens to extend the KV cache, then initializes a fresh 256-token canvas conditioned on the now-extended history.

This is strictly left-to-right at the block level — each new canvas attends to all prior committed tokens, but prior blocks are immutable once committed. Variable-length generation is just more blocks. The block boundary is a genuine hard commit: there's no cross-block revision.

## Quality tradeoffs

DiffusionGemma's overall output quality is lower than standard Gemma 4. On MMLU Pro it scores 77.6% vs. 82.6% for standard Gemma 4 26B A4B; on LiveCodeBench v6, 69.1% vs. 77.1%. The gap is real across most benchmarks.

The exception is constraint-propagation tasks. Bidirectional attention within the canvas lets the model enforce constraints symmetrically — a token in position 1 can "know about" position 200 during the same denoising step. AR models fundamentally can't do this, which is why the Sudoku fine-tuning result (0% → 80% correctness) is notable — the architecture itself is better suited for the problem, independent of speed.

Here's the inference loop as a diagram (click nodes to go deeper):The dashed boxes in the loop are clickable to drill into each component.

A few things worth noting from a systems standpoint. The left-side loop arrow (entropy → denoising) skips the "initialize canvas" node intentionally — re-noising uncertain positions isn't a canvas re-initialization, it's a partial in-place replacement using fresh random token IDs. The KV cache isn't rebuilt during the denoising loop; it's stable and read-only until a block commits. And the 15–20 tokens actually committed per forward pass (not all 256) is the practical throughput number — the model generates 256 logit distributions per step but the sampler only locks in the most confident fraction.

The vLLM integration required a new abstraction (`ModelState`) specifically because dLLMs don't fit the standard autoregressive serving path — they need bidirectional attention, per-request canvas state, iterative sampling behavior, and a custom early-stopping hook that's completely absent from AR models.