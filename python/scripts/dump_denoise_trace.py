#!/usr/bin/env python3
"""
Run HuggingFace DiffusionGemma and dump a compact denoise trace for P1.6 parity.

Requires optional deps: `uv sync --extra model` (torch + transformers).

Example:
  uv sync --extra model
  uv run python/scripts/dump_denoise_trace.py \\
    --model google/diffusiongemma-26B-A4B-it \\
    -p Hello --seed 42 --steps 48 --max-new-tokens 256 \\
    -o /tmp/hf_hello_trace.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffgemma_parity.denoise_stats import (  # noqa: E402
    ENTROPY_BOUND,
    step_stats_from_entropies,
    trace_step_record,
)

SCHEMA_VERSION = 1


class TracingSampler:
    """Wrap HF EntropyBoundSampler and record per-step stats."""

    def __init__(self, inner, block: int, max_denoise_steps: int, records: list[dict[str, Any]]):
        self._inner = inner
        self.block = block
        self.max_denoise_steps = max_denoise_steps
        self.records = records
        self.step_index = 0
        self.last_finished = False

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def initialize_canvas(self, batch_size: int, device):
        return self._inner.initialize_canvas(batch_size, device)

    def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
        import torch
        from torch.distributions import Categorical

        cs = int(cur_step.item()) if torch.is_tensor(cur_step) else int(cur_step)
        dist = Categorical(logits=logits)
        ent = dist.entropy()[0].float().cpu().tolist()
        out = self._inner.accept_canvas(current_canvas, denoiser_canvas, logits, cur_step)
        mask = self._inner.accepted_token_mask[0].bool().cpu().tolist()
        stats = step_stats_from_entropies(ent, mask, self._inner.entropy_bound)
        self.step_index += 1
        argmax = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        self.records.append(
            trace_step_record(
                block=self.block,
                step_index=self.step_index,
                max_denoise_steps=self.max_denoise_steps,
                stats=stats,
                argmax_prefix=argmax,
                early_stop=False,
            )
        )
        return out

    def renoise_canvas(self, accepted_canvas, cur_step):
        return self._inner.renoise_canvas(accepted_canvas, cur_step)


def patch_prepare_sampler(model, tracing: TracingSampler):
    orig = model._prepare_sampler

    def wrapped(generation_config):
        inner = orig(generation_config)
        tracing._inner = inner
        tracing.max_denoise_steps = int(generation_config.max_denoising_steps)
        return tracing

    model._prepare_sampler = wrapped  # type: ignore[method-assign]


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = inputs["input_ids"][0].tolist()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    step_records: list[dict[str, Any]] = []
    tracing = TracingSampler(None, block=1, max_denoise_steps=args.steps, records=step_records)
    patch_prepare_sampler(model, tracing)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "max_denoising_steps": args.steps,
        "diffusion_entropy_bound": args.entropy_bound,
        "diffusion_confidence_threshold": args.confidence_threshold,
        "do_sample": True,
        "temperature": args.temperature,
    }
    if args.no_early_stop:
        gen_kwargs["diffusion_confidence_threshold"] = 1e9

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    output_ids = output.sequences[0].tolist()
    if step_records:
        step_records[-1]["early_stop"] = len(step_records) < args.steps

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "hf-python",
        "prompt": args.prompt,
        "prompt_token_ids": prompt_ids,
        "seed": args.seed,
        "max_denoise_steps": args.steps,
        "layers": int(getattr(model.config.text_config, "num_hidden_layers", 30)),
        "max_new_tokens": args.max_new_tokens,
        "weights_profile": "hf-bf16",
        "entropy_bound": args.entropy_bound,
        "step_traces": step_records,
        "denoise_steps_run": len(step_records),
        "blocks_committed": 1 if len(output_ids) > len(prompt_ids) else 0,
        "output_token_ids": output_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("-p", "--prompt", default="Hello")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--entropy-bound", type=float, default=ENTROPY_BOUND)
    parser.add_argument("--confidence-threshold", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output JSON path",
    )
    args = parser.parse_args()

    try:
        trace = run_trace(args)
    except ImportError as exc:
        print(
            "error: torch/transformers not installed. Run: uv sync --extra model",
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    print(
        f"wrote {args.output} ({len(trace['step_traces'])} steps, "
        f"{len(trace['output_token_ids'])} tokens)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
