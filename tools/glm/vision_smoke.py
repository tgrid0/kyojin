#!/usr/bin/env python3
"""GPU smoke test for GLM-5.3-Flash image input. Run on node-strix (Strix Halo), not on the editing PC.

Direct mode (the go/no-go gate, no server involved):
    python tools/glm/vision_smoke.py --model DIR --image examples/media/cat.png [--question TEXT] [--ctx N] [--no-spec]

Server mode (quality and prefix-reuse checks against a running tools/glm/serve.py):
    python tools/glm/vision_smoke.py --server http://127.0.0.1:8000 --image A.png [--image B.png ...] [--question TEXT]
    python tools/glm/vision_smoke.py --server http://127.0.0.1:8000 --image A.png --reuse
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return 0.0


def data_url(path: str) -> str:
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def direct(args: argparse.Namespace) -> int:
    """Load text model, MTP model and vision tower the way ResidentEngine does and answer one question."""
    sys.path.insert(0, str(HERE))
    import serve                                   # SPEED_ENV only; nothing of the HTTP layer is used
    for key, value in serve.SPEED_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
    import torch
    from PIL import Image
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.generator.sampler import GreedySampler

    torch.set_grad_enabled(False)
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=args.ctx + 4096, max_history=args.num_draft)
    model.load(device="cuda:0", progressbar=False)
    draft = draft_cache = None
    if not args.no_spec:
        draft = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft, max_num_tokens=args.ctx + 4096, max_history=args.num_draft)
        draft.load(device="cuda:0", progressbar=False)
    before = mem_available_gib()
    t0 = time.perf_counter()
    config.vision_pp.max_image_tokens = min(config.vision_pp.max_image_tokens, args.max_image_tokens)
    vision = Model.from_config(config, component="vision")
    vision.load(device="cuda:0", progressbar=False)
    print(f"vision tower: load {time.perf_counter() - t0:.1f} s, host MemAvailable {before - mem_available_gib():+.2f} GiB, "
          f"torch allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    kw = {"draft_model": draft, "draft_cache": draft_cache, "num_draft_tokens": args.num_draft} if draft else {}
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, **kw)

    emb = vision.get_image_embeddings(tokenizer=tokenizer, image=Image.open(args.image[0]).convert("RGB"))
    print(f"image {args.image[0]}: {emb.mm_length} embedding rows, original {emb.metadata['original_size']}, "
          f"canvas {emb.metadata['preprocessed_size']}")
    prompt = "[gMASK]<sop><|user|>\n" + emb.text_alias + "\n" + args.question + "<|assistant|>\n<think></think>"
    ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True, embeddings=[emb])
    job = Job(input_ids=ids, max_new_tokens=args.max_new, sampler=GreedySampler(), embeddings=[emb],
              stop_conditions=config.eos_token_id_list)
    generator.enqueue(job)
    out = ""
    t0 = time.perf_counter()
    while generator.num_remaining_jobs():
        for ev in generator.iterate():
            out += ev.get("text", "")
    print(f"prompt {int(ids.numel())} tokens, {time.perf_counter() - t0:.1f} s, speculative {'off' if args.no_spec else 'on'}")
    print("ANSWER:", out.strip())
    return 0


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.reason}


def ask(url: str, messages: list, max_tokens: int = 160, **extra) -> tuple[int, dict, float]:
    t0 = time.perf_counter()
    status, body = post(url, {"messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra})
    return status, body, time.perf_counter() - t0


def user(question: str, paths: list[str]) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": question}] +
            [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in paths]}


def show(label: str, status: int, body: dict, secs: float) -> str:
    if status != 200:
        print(f"{label}: HTTP {status} {body}")
        return ""
    msg = body["choices"][0]["message"]
    t = body.get("timings", {})
    print(f"{label}: {secs:.1f} s, prompt {body['usage']['prompt_tokens']} tok (cached {t.get('cache_n', '?')}), "
          f"{t.get('predicted_per_second', 0):.1f} tok/s decode")
    print("   ", (msg.get("content") or "").strip().replace("\n", " ")[:300])
    return msg.get("content") or ""


def server(args: argparse.Namespace) -> int:
    if args.reuse:
        return reuse(args)
    for path in args.image:
        show(path, *ask(args.server, [user(args.question, [path])]))
    return 0


def reuse(args: argparse.Namespace) -> int:
    """Two-turn check of prefix reuse: the second turn repeats the first (same image) and adds a follow-up.

    The cold answer is recomputed by erasing the slot cache (/slots/0?action=erase) and compared token for token."""
    first = [user(args.question, args.image[:1])]
    s1, b1, t1 = ask(args.server, first, 120)
    a1 = show("turn 1 (cold)", s1, b1, t1)
    if s1 != 200:
        return 1
    follow = first + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Now describe it in five words."}]
    s2, b2, t2 = ask(args.server, follow, 60)
    warm = show("turn 2 (same image, prefix reused)", s2, b2, t2)
    cached = b2.get("timings", {}).get("cache_n", 0)
    prompt = b2["usage"]["prompt_tokens"]
    print(f"reuse: {cached}/{prompt} prompt tokens served from cache")
    req = urllib.request.Request(args.server + "/slots/0?action=erase", b"{}", {"content-type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=60).read()
    s3, b3, t3 = ask(args.server, follow, 60)
    cold = show("turn 2 (cold, cache erased)", s3, b3, t3)
    print("warm == cold:", warm == cold)
    return 0 if warm == cold and s3 == 200 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/llm/models/glm-5.3-flash/yamz"))
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--question", default="What is in this picture? Answer in one or two sentences.")
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--num-draft", type=int, default=2)
    parser.add_argument("--no-spec", action="store_true", help="direct mode without the MTP draft model")
    parser.add_argument("--max-image-tokens", type=int, default=1280)
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--server", help="URL of a running serve.py: switches to server mode")
    parser.add_argument("--reuse", action="store_true", help="server mode: two-turn prefix reuse check")
    args = parser.parse_args()
    return server(args) if args.server else direct(args)


if __name__ == "__main__":
    sys.exit(main())
