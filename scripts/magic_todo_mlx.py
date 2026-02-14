#!/usr/bin/env python3
"""
Magic ToDo (terminal) using local MLX-LM models.

Runs entirely locally (no network needed at runtime) assuming the model is
already present in the HuggingFace cache.
"""

from __future__ import annotations

# Strip PYTHONPATH early to prevent nix/flox system packages from shadowing
# the venv's own site-packages (e.g. an old huggingface_hub leaking in).
import os
import sys

_pp = os.environ.pop("PYTHONPATH", None)
if _pp:
    # Remove any PYTHONPATH-injected dirs from sys.path that aren't ours
    _venv = sys.prefix
    sys.path[:] = [p for p in sys.path if p.startswith(_venv) or p == "" or "site-packages" not in p or p.startswith(sys.base_prefix)]

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HF_CACHE_DEFAULT = Path.home() / ".cache" / "huggingface" / "hub"


def _guess_cached_mlx_community_models(cache_root: Path) -> list[str]:
    if not cache_root.exists():
        return []

    out: list[str] = []
    for p in cache_root.glob("models--mlx-community--*"):
        name = p.name.removeprefix("models--").replace("--", "/", 1)
        # name is now "mlx-community/<repo>"
        out.append(name)
    return sorted(out)


def _spice_to_guidance(spice: int) -> str:
    """Return prompt guidance calibrated to the spiciness level.

    Spice reflects how hard/stressful the user finds the task:
    1 = straightforward, just list the obvious steps
    5 = overwhelming, break it into the smallest possible actions
    """
    return {
        1: (
            "This is a straightforward task. Give 3-5 concrete steps.\n"
            "Keep it simple — only the essential actions.\n"
        ),
        2: (
            "This task needs some thought. Give 5-8 concrete steps.\n"
            "Think about what might be missed and include preparation steps.\n"
        ),
        3: (
            "This task feels moderately hard. Give 8-12 concrete steps.\n"
            "Think carefully about ordering, dependencies, and potential blockers.\n"
            "Break any ambiguous step into its real sub-actions.\n"
        ),
        4: (
            "This task feels hard and stressful. Give 12-18 concrete steps.\n"
            "Think deeply about what makes this overwhelming and break those parts down.\n"
            "Include sub-steps for anything that isn't immediately obvious how to do.\n"
            "Consider edge cases, preparation, and follow-up actions.\n"
        ),
        5: (
            "This task feels extremely overwhelming. Give 18-28 concrete steps.\n"
            "Break everything into the smallest possible atomic actions.\n"
            "Assume the user needs every implicit step spelled out.\n"
            "Include sub-steps liberally. Think about what could go wrong at each stage.\n"
            "Consider research, preparation, execution, verification, and cleanup phases.\n"
        ),
    }[spice]


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extractor: find the first {...} JSON object in model output.
    Tries to repair common model output issues.
    """
    # Preferred: sentinel terminator.
    if "ENDJSON" in text:
        before = text.split("ENDJSON", 1)[0]
        start = before.find("{")
        end = before.rfind("}")
        if start >= 0 and end > start:
            return before[start : end + 1].strip()

    # Prefer fenced JSON blocks if present.
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Otherwise, scan for balanced braces.
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    # Unbalanced — try closing open braces/brackets.
    fragment = text[start:]
    fragment = _repair_json(fragment)
    # Verify it at least starts with { after repair
    if fragment.startswith("{"):
        return fragment

    raise ValueError("Unbalanced JSON braces in model output.")


def _repair_json(text: str) -> str:
    """Try to fix common JSON issues from LLM output."""
    # Strip trailing commas before closing delimiters and at end
    text = re.sub(r",\s*$", "", text.rstrip())
    # Fix missing commas between key-value pairs: "value" "key" -> "value", "key"
    text = re.sub(r'"\s*\n?\s*"(?=[a-z_])', '", "', text)
    # Fix missing commas between objects: } { -> }, {
    text = re.sub(r"}\s*\n?\s*{", "}, {", text)
    # Count unmatched openers
    stack = []
    in_string = False
    escape = False
    for c in text:
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in ("{", "["):
            stack.append(c)
        elif c == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif c == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    # Close in reverse order
    for opener in reversed(stack):
        text += "]" if opener == "[" else "}"
    return text


def _validate_plan(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Plan JSON must be an object.")
    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Plan JSON must contain non-empty 'steps' array.")
    for s in steps:
        if not isinstance(s, dict) or not isinstance(s.get("text"), str) or not s["text"].strip():
            raise ValueError("Each step must be an object with non-empty 'text'.")
    return obj


@dataclass(frozen=True)
class GenCfg:
    model: str
    spice: int
    max_tokens: int
    temp: float
    top_p: float
    seed: int | None
    context: str | None = None


def _build_messages(task: str, cfg: GenCfg) -> list[dict[str, str]]:
    guidance = _spice_to_guidance(cfg.spice)
    system = (
        "You are Magic ToDo, an ADHD-friendly task breakdown tool.\n"
        "You MUST respond with ONLY a JSON object — no prose, no markdown, no lists.\n"
        "Schema: {\"title\": \"...\", \"steps\": [{\"text\": \"...\", \"substeps\": null}]}\n"
        "After the JSON, output ENDJSON on its own line.\n\n"
        "Voice rules:\n"
        "- Be CONCISE. Every step is max ~10 words. Walls of text cause shutdown.\n"
        "- Assume the user does NOT write code by hand — prefer CLI tools, generators,\n"
        "  templates, and automation over manual editing.\n"
        "- Still break things down thoroughly — many small steps, not few big ones.\n"
        "- Start each step with a verb. No filler words.\n"
        "- If a good tool exists, name it with one alternative (e.g. 'using X or Y').\n"
        "- NEVER: 'consider', 'ensure', 'review', 'plan', 'finalize', 'research options'."
    )
    user_parts = [
        f"{guidance}\n"
    ]
    if cfg.context:
        user_parts.append(
            "\nExisting breakdowns (match this style):\n"
            f"{cfg.context.strip()}\n"
        )
    user_parts.append(
        "\nExample output:\n"
        '{"title": "Deploy app", "steps": ['
        '{"text": "Build image with docker build or podman", "substeps": null}, '
        '{"text": "Push to registry with docker push or crane", "substeps": null}, '
        '{"text": "Apply k8s manifest with kubectl apply", "substeps": null}'
        "]}\nENDJSON\n\n"
        f"Task: {task.strip()}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "".join(user_parts)},
    ]


def _generate_plan(task: str, cfg: GenCfg) -> dict[str, Any]:
    from mlx_lm import load, stream_generate  # type: ignore
    from mlx_lm.sample_utils import make_sampler  # type: ignore

    model, tokenizer = load(cfg.model)

    def _mk_prompt(*, prefill: str | None = None) -> str:
        messages = _build_messages(task, cfg)
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        if hasattr(tokenizer, "apply_chat_template"):
            result = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=not prefill,
                tokenize=False,
            )
            if isinstance(result, str):
                return result
            return tokenizer.decode(result)
        # Fallback: concat all message contents
        text = "\n\n".join(m["content"] for m in messages)
        if prefill:
            text += "\n" + prefill
        return text

    def _mk_sampler(temp: float, top_p: float):
        return make_sampler(temp=temp, top_p=top_p)

    def _run_once(*, temp: float, top_p: float, prefill: str | None = None) -> str:
        prompt = _mk_prompt(prefill=prefill)
        sampler = _mk_sampler(temp, top_p)

        gen_kwargs: dict[str, Any] = {"sampler": sampler}
        if cfg.seed is not None:
            gen_kwargs["seed"] = cfg.seed

        # Stream so we can stop early when the model emits ENDJSON, but only
        # after we've actually started a JSON object (avoid premature "ENDJSON").
        parts: list[str] = []
        if prefill:
            parts.append(prefill)
        seen_lbrace = bool(prefill and "{" in prefill)
        for r in stream_generate(model, tokenizer, prompt, max_tokens=cfg.max_tokens, **gen_kwargs):
            parts.append(r.text)
            if not seen_lbrace and "{" in r.text:
                seen_lbrace = True
            tail = "".join(parts[-12:])
            if seen_lbrace and ("ENDJSON" in r.text or "\nENDJSON" in tail):
                break
        return "".join(parts)

    def _parse(text: str) -> dict[str, Any]:
        json_str = _extract_first_json_object(text)
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            # Repair: trailing commas, missing commas, unclosed braces
            fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
            fixed = _repair_json(fixed)
            obj = json.loads(fixed)
        return _validate_plan(obj)

    text = _run_once(temp=cfg.temp, top_p=cfg.top_p)
    try:
        return _parse(text)
    except Exception as e1:
        # Retry with stricter sampling and an assistant prefill to force JSON.
        text2 = _run_once(temp=0.0, top_p=1.0, prefill='{"title":')
        try:
            return _parse(text2)
        except Exception as e2:
            sys.stderr.write("Failed to parse model output as JSON.\n")
            sys.stderr.write(f"First error: {type(e1).__name__}: {e1}\n")
            sys.stderr.write("Raw model output (attempt 1):\n")
            sys.stderr.write(text.strip() + "\n")
            sys.stderr.write(f"Second error: {type(e2).__name__}: {e2}\n")
            sys.stderr.write("Raw model output (attempt 2):\n")
            sys.stderr.write(text2.strip() + "\n")
            raise


def _iter_steps(plan: dict[str, Any]) -> Iterable[tuple[int, str, list[str] | None]]:
    steps = plan["steps"]
    for i, s in enumerate(steps, start=1):
        text = str(s.get("text", "")).strip()
        sub = s.get("substeps")
        if isinstance(sub, list):
            substeps = []
            for ss in sub:
                if isinstance(ss, dict) and isinstance(ss.get("text"), str):
                    t = ss["text"].strip()
                    if t:
                        substeps.append(t)
            substeps_out = substeps or None
        else:
            substeps_out = None
        yield (i, text, substeps_out)


def _print_plan(plan: dict[str, Any], fmt: str) -> None:
    title = plan.get("title")
    if isinstance(title, str) and title.strip():
        if fmt == "md":
            print(f"# {title.strip()}\n")
        else:
            print(f"{title.strip()}\n")

    for i, text, substeps in _iter_steps(plan):
        if fmt == "md":
            print(f"- [ ] {text}")
            if substeps:
                for ss in substeps:
                    print(f"  - [ ] {ss}")
        else:
            print(f"{i}. {text}")
            if substeps:
                for j, ss in enumerate(substeps, start=1):
                    print(f"   {i}.{j} {ss}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="magic_todo_mlx",
        description="Magic ToDo clone (terminal) powered by local MLX-LM models.",
    )
    ap.add_argument("task", nargs="*", help="Task text. If omitted, reads stdin.")
    ap.add_argument(
        "--model",
        default=os.environ.get("MAGIC_TODO_MODEL", "mlx-community/Qwen3-8B-4bit"),
        help="MLX model repo id (default: mlx-community/Qwen3-8B-4bit).",
    )
    ap.add_argument("--spice", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Granularity 1-5.")
    ap.add_argument("--max-tokens", type=int, default=900, help="Max tokens to generate.")
    ap.add_argument("--temp", type=float, default=0.2, help="Sampling temperature.")
    ap.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top_p.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    ap.add_argument("--context", type=str, default=None, help="Existing breakdowns for context.")
    ap.add_argument("--format", choices=["text", "md", "json"], default="text", help="Output format.")
    ap.add_argument("--list-models", action="store_true", help="List cached mlx-community models and exit.")
    ap.add_argument(
        "--hf-cache",
        default=str(HF_CACHE_DEFAULT),
        help=f"HuggingFace cache root (default: {HF_CACHE_DEFAULT}).",
    )

    args = ap.parse_args(argv)

    cache_root = Path(args.hf_cache).expanduser()
    if args.list_models:
        for m in _guess_cached_mlx_community_models(cache_root):
            print(m)
        return 0

    task = " ".join(args.task).strip()
    if not task:
        task = sys.stdin.read().strip()
    if not task:
        ap.error("Provide a task as arguments or via stdin.")

    cfg = GenCfg(
        model=args.model,
        spice=args.spice,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        seed=args.seed,
        context=args.context,
    )

    plan = _generate_plan(task, cfg)

    if args.format == "json":
        print(json.dumps(plan, indent=2, ensure_ascii=True))
    else:
        _print_plan(plan, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
