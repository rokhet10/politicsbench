#!/usr/bin/env python3
"""
Generate surface-wording variants of scenario prompts via OpenAI Chat Completions.

- One API call per prompt line (Prompt1..4).
- Optional: several wording variants per base scenario (--num-variants + --variant-suffixes),
  e.g. from data/scenario_prompts.txt produce 1-wa, 1-wb, … with parallel API calls.
- Logs every request + response to JSONL; optional consolidated JSON and formatted .txt.

Environment (same as rest of repo):
  OPENAI_API_KEY (required)
  OPENAI_API_URL (optional, default https://api.openai.com/v1/chat/completions)

Example (standard 20 scenarios → 2 paraphrases each, parallel):
  cd repo-root
  python3 paraphrase_robustness/scripts/generate_wordings.py \\
    --prompts-file data/scenario_prompts.txt \\
    --num-variants 2 --variant-suffixes wa,wb \\
    --workers 12 \\
    --out-txt paraphrase_robustness/archive/pre_unified_prompts/scenario_prompts_pilot.txt \\
    --log paraphrase_robustness/results/wording_generate_log.jsonl \\
    --out-json paraphrase_robustness/results/wording_generated_bundle.json

Example (single pass over an existing pilot file):
  python3 paraphrase_robustness/scripts/generate_wordings.py \\
    --prompts-file paraphrase_robustness/archive/wording_1/scenario_prompts_pilot.txt \\
    --log paraphrase_robustness/results/wording_generate_log.jsonl \\
    --skip-controls
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DEFAULT_INSTRUCTION = """Rewrite the scenario preserving:
- same actors, order, and relationships
- identical facts and stakes
- same emotional tone

Only vary surface wording (synonyms, sentence structure).
Do NOT change framing or emphasis.

Output ONLY the rewritten text. Do not use markdown fences or labels."""


def _load_parser():
    vp_path = Path(__file__).resolve().parent / "validate_prompts.py"
    spec = importlib.util.spec_from_file_location("validate_prompts", vp_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.parse_scenario_prompts


parse_scenario_prompts = _load_parser()


def parse_scenario_header_tails(file_path: Path) -> Dict[str, str]:
    """Map base scenario_id (first token on ######## line) -> rest of title (e.g. '| A | B')."""
    tails: Dict[str, str] = {}
    header_re = re.compile(r"^########\s*(\S+)\s*(.*)$")
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            m = header_re.match(raw.rstrip())
            if m:
                sid, rest = m.group(1), m.group(2).strip()
                tails[sid] = rest
    return tails


def natural_sort_scenario_ids(ids: List[str]) -> List[str]:
    def key(x: str) -> Tuple[int, Any]:
        if str(x).isdigit():
            return (0, int(x))
        return (1, str(x))

    return sorted(ids, key=key)


def sort_full_scenario_ids(ids: List[str]) -> List[str]:
    """Order keys like 1-wa, 1-wb, 2-wa, …, 10-wa (not lexicographic on the full string)."""

    def key(x: str) -> Tuple[Any, ...]:
        m = re.match(r"^(\d+)-(\S+)$", str(x))
        if m:
            return (0, int(m.group(1)), m.group(2))
        if str(x).isdigit():
            return (0, int(x), "")
        return (1, str(x))

    return sorted(ids, key=key)


def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _strip_variant_meta_lines(text: str) -> str:
    """Remove leaked internal variant-guidance lines from model output."""
    lines = text.splitlines()
    cleaned = [
        ln
        for ln in lines
        if not ln.strip().startswith("This is paraphrase variant ")
    ]
    return "\n".join(cleaned).strip()


def _load_done_keys(log_path: Path) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    if not log_path.exists():
        return done
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = row.get("scenario_id")
                pi = row.get("prompt_index")
                if sid is not None and pi is not None and row.get("error") is None:
                    done.add((str(sid), int(pi)))
            except json.JSONDecodeError:
                continue
    return done


def _chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
) -> Tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return content, data


def _write_out_txt(
    path: Path,
    *,
    header_tails: Dict[str, str],
    scenario_meta: Dict[str, Dict[str, Any]],
    ordered_full_ids: List[str],
    prompts_by_sid: Dict[str, List[str]],
) -> None:
    blocks: List[str] = []
    for full_sid in ordered_full_ids:
        prompts = prompts_by_sid.get(full_sid)
        if not prompts or len(prompts) != 4 or not all(prompts):
            continue
        meta = scenario_meta.get(full_sid, {})
        base = meta.get("base_id", full_sid)
        tail = header_tails.get(base, "").strip()
        letter = meta.get("variant_letter")
        head = f"######## {full_sid}"
        if tail:
            head += f" {tail}"
        if letter:
            head += f" (wording variant {letter})"
        lines: List[str] = [head]
        for i, p in enumerate(prompts):
            lines.append(f"####### Prompt{i + 1}")
            lines.append(p)
        blocks.append("\n".join(lines))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GPT surface paraphrases for scenario prompts (parallel-capable; optional multi-suffix variants)."
    )
    ap.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Input scenario prompts .txt (######## / ####### PromptN format).",
    )
    ap.add_argument(
        "--instruction",
        type=str,
        default=_DEFAULT_INSTRUCTION,
        help="Paraphrase instructions (same text sent on every call).",
    )
    ap.add_argument("--model", default="gpt-4.1-mini", help="Chat model id.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--variant-min-temperature",
        type=float,
        default=1.0,
        help="When --num-variants>1, enforce at least this temperature per call.",
    )
    ap.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("paraphrase_robustness/results/wording_generate_log.jsonl"),
        help="Append-only JSONL: one object per API call (input, output, metadata).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write consolidated JSON {scenario_id: [4 prompts]} when run completes.",
    )
    ap.add_argument(
        "--out-txt",
        type=Path,
        default=None,
        help="Write formatted scenario prompts .txt (complete 4-prompt blocks only).",
    )
    ap.add_argument(
        "--num-variants",
        type=int,
        default=1,
        metavar="N",
        help="If >1, emit N paraphrases per base scenario id using --variant-suffixes (e.g. 1→1-wa,1-wb). Default 1 = ids unchanged.",
    )
    ap.add_argument(
        "--variant-suffixes",
        type=str,
        default="wa,wb",
        help="Comma-separated suffixes; must have length num-variants when num-variants>1.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent API calls (thread pool). Ignored for dry-run.",
    )
    ap.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario_ids (base ids when using num-variants>1).",
    )
    ap.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip scenario_ids whose id starts with ctrl-",
    )
    ap.add_argument(
        "--scenario-suffix",
        type=str,
        default=None,
        help="Only process scenario_ids that end with this (e.g. -wa). Applies to full ids.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip (scenario_id, prompt_index) pairs already present in --log without errors.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds after each API call (rate limiting). Use 0 with --workers for throughput.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned calls only; no API requests.",
    )
    args = ap.parse_args()

    if args.num_variants < 1:
        print("ERROR: --num-variants must be >= 1.", file=sys.stderr)
        return 2

    suffixes = [x.strip() for x in args.variant_suffixes.split(",") if x.strip()]
    if args.num_variants > 1 and len(suffixes) != args.num_variants:
        print(
            f"ERROR: --variant-suffixes must list exactly {args.num_variants} entries (got {len(suffixes)}).",
            file=sys.stderr,
        )
        return 2

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
    )
    if not args.dry_run and not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    scenarios = parse_scenario_prompts(str(args.prompts_file))
    if not scenarios:
        print("ERROR: No scenarios parsed.", file=sys.stderr)
        return 1

    header_tails = parse_scenario_header_tails(args.prompts_file)

    filter_ids: Optional[set] = None
    if args.scenarios:
        filter_ids = {x.strip() for x in args.scenarios.split(",") if x.strip()}

    ordered_bases = natural_sort_scenario_ids(list(scenarios.keys()))

    # full_sid -> { base_id, variant_letter?, suffix? }
    scenario_meta: Dict[str, Dict[str, Any]] = {}
    rows: List[Tuple[str, str, List[str]]] = []
    for base_sid in ordered_bases:
        if filter_ids is not None and base_sid not in filter_ids:
            continue
        if args.skip_controls and str(base_sid).startswith("ctrl-"):
            continue
        prompts = scenarios[base_sid]
        if len(prompts) != 4:
            print(
                f"WARN: {base_sid} has {len(prompts)} prompts (expected 4); skipping.",
                file=sys.stderr,
            )
            continue

        if args.num_variants <= 1:
            full_sid = str(base_sid)
            if args.scenario_suffix and not full_sid.endswith(args.scenario_suffix):
                continue
            scenario_meta[full_sid] = {"base_id": str(base_sid)}
            rows.append((full_sid, base_sid, prompts))
        else:
            for vi, suf in enumerate(suffixes):
                full_sid = f"{base_sid}-{suf}"
                if args.scenario_suffix and not full_sid.endswith(args.scenario_suffix):
                    continue
                letter = chr(ord("A") + vi) if vi < 26 else str(vi + 1)
                scenario_meta[full_sid] = {
                    "base_id": str(base_sid),
                    "suffix": suf,
                    "variant_letter": letter,
                }
                rows.append((full_sid, base_sid, prompts))

    done_keys: Set[Tuple[str, int]] = set()
    if args.resume:
        done_keys = _load_done_keys(args.log)

    system_msg = (
        "You rewrite scenario prompts for a political-values benchmark. "
        "Follow the user's constraints exactly. "
        "Reply with only the rewritten scenario text—no preamble or markdown."
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    workers = max(1, int(args.workers))
    use_throttle_sleep = args.sleep > 0 and workers == 1

    def run_call(
        full_sid: str,
        base_sid: str,
        pi: int,
        raw_text: str,
        variant_ix: Optional[int],
    ) -> Tuple[str, int, Dict[str, Any]]:
        user_body = (
            args.instruction.strip()
            + "\n\n---\n\nTEXT TO REWRITE:\n\n"
            + raw_text.strip()
        )
        call_temperature = args.temperature
        if args.num_variants > 1:
            call_temperature = max(float(args.temperature), float(args.variant_min_temperature))
        if args.num_variants > 1 and variant_ix is not None:
            letter = chr(ord("A") + variant_ix)
            user_body += (
                f"\n\nThis is paraphrase variant {letter} of {args.num_variants} "
                f"for the same scenario spine (base id {base_sid}); use noticeably "
                "different surface wording from the other variants."
            )

        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "scenario_id": full_sid,
            "base_scenario_id": base_sid,
            "prompt_index": pi,
            "model": args.model,
            "temperature": call_temperature,
            "top_p": args.top_p,
            "instruction": args.instruction,
            "input_prompt": raw_text,
            "messages_preview": {"system": system_msg[:200], "user_chars": len(user_body)},
        }

        try:
            content, raw_api = _chat(
                api_key=api_key,
                base_url=base_url,
                model=args.model,
                system=system_msg,
                user=user_body,
                temperature=call_temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            out = _strip_variant_meta_lines(_strip_markdown_fence(content))
            record["output"] = out
            record["output_chars"] = len(out)
            record["raw_response_meta"] = {
                "id": raw_api.get("id"),
                "model": raw_api.get("model"),
                "usage": raw_api.get("usage"),
            }
        except Exception as e:
            record["error"] = str(e)
            record["output"] = None
            print(f"ERROR {full_sid} Prompt{pi + 1}: {e}", file=sys.stderr)

        with log_lock:
            with open(args.log, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(record, ensure_ascii=False) + "\n")

        if use_throttle_sleep:
            time.sleep(args.sleep)
        return full_sid, pi, record

    pending: List[
        Tuple[str, str, int, str, Optional[int]]
    ] = []  # full_sid, base_sid, pi, raw_text, variant_ix
    for full_sid, base_sid, prompts in rows:
        var_ix: Optional[int] = None
        if args.num_variants > 1:
            suf = scenario_meta[full_sid].get("suffix")
            if suf is not None:
                var_ix = suffixes.index(suf)
        for pi, raw_text in enumerate(prompts):
            key = (full_sid, pi)
            if args.resume and key in done_keys:
                continue
            pending.append((full_sid, base_sid, pi, raw_text, var_ix))

    if args.dry_run:
        print(f"Dry run: {len(pending)} API calls planned ({len(rows)} scenario rows after filters).")
        for full_sid, _b, pi, _t, _vx in pending[:50]:
            print(f"  [dry-run] {full_sid} Prompt{pi + 1}")
        if len(pending) > 50:
            print(f"  ... and {len(pending) - 50} more")
        return 0

    n_skip = 0
    if args.resume:
        for full_sid, _b, _prompts in rows:
            for pi in range(4):
                if (full_sid, pi) in done_keys:
                    n_skip += 1

    n_calls = 0
    if workers == 1:
        for tup in pending:
            full_sid, base_sid, pi, raw_text, var_ix = tup
            _fsid, _pi, rec = run_call(full_sid, base_sid, pi, raw_text, var_ix)
            if rec.get("error") is None and rec.get("output") is not None:
                n_calls += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(run_call, full_sid, base_sid, pi, raw_text, var_ix)
                for full_sid, base_sid, pi, raw_text, var_ix in pending
            ]
            for fut in as_completed(futs):
                _fsid, _pi, rec = fut.result()
                if rec.get("error") is None and rec.get("output") is not None:
                    n_calls += 1

    print(
        f"API calls attempted: {len(pending)}, successful: {n_calls}, "
        f"skipped (resume): {n_skip}, log: {args.log}"
    )

    latest_by_sid: Dict[str, List[Optional[str]]] = {}
    if args.log.exists():
        with open(args.log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("error") or row.get("output") is None:
                    continue
                sid = row["scenario_id"]
                pi = int(row["prompt_index"])
                latest_by_sid.setdefault(sid, [None, None, None, None])
                latest_by_sid[sid][pi] = row["output"]

    if args.out_json:
        bundle = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_prompts_file": str(args.prompts_file.resolve()),
            "model": args.model,
            "settings": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_variants": args.num_variants,
                "variant_suffixes": suffixes,
            },
            "instruction": args.instruction,
            "results": {
                k: latest_by_sid[k]
                for k in sort_full_scenario_ids(
                    [
                        kk
                        for kk, vv in latest_by_sid.items()
                        if all(x is not None for x in vv)
                    ]
                )
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote bundle (complete 4-prompt scenarios only): {args.out_json}")

    if args.out_txt:
        ordered_full = [r[0] for r in rows]
        _write_out_txt(
            args.out_txt,
            header_tails=header_tails,
            scenario_meta=scenario_meta,
            ordered_full_ids=ordered_full,
            prompts_by_sid={
                k: v  # type: ignore
                for k, v in latest_by_sid.items()
                if all(x is not None for x in v)
            },
        )
        n_written = sum(
            1
            for k in ordered_full
            if k in latest_by_sid and all(x is not None for x in latest_by_sid[k])
        )
        print(f"Wrote {n_written} complete scenarios to {args.out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
