# GLM-5.3-Flash Vision Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tools/glm/serve.py` accepts OpenAI `image_url` content parts and answers about them, using the vision tower that already ships in the Yamz GLM-5.3-Flash EXL3 pack.

**Architecture:** The engine side is mostly done: `Glm5NextVisionModel` (`exllamav3/architecture/glm5_next.py:377`), the GLM5.3 canvas resize (`mm_processing/glm5_next.py`) and the config reader already exist, and the pack holds 1007 `model.visual.*` tensors in shards 11-12. What is missing is the server: load the tower, turn images into `MMEmbedding`s, get the placeholder into the prompt (the stock chat template discards images), and pass `embeddings=` to the `Job`. `tools/qwen/serve.py` already does exactly this and is the pattern to copy.

**Tech Stack:** Python, aiohttp, PIL, exllamav3 `Model.from_config(component="vision")`, `MMEmbedding`, `Job(embeddings=...)`.

**Spec:** none; this is derived from the feasibility review of 2026-10-07 (this conversation).

## Feasibility verdict

**Feasible, small-to-medium. Roughly 150-250 lines of server code plus tests.** No new kernels, no new quantization.

Evidence found:
- Pack (`walker@node-strix:/home/walker/llm/models/glm-5.3-flash/yamz`): `config.json` has `vision_config` (24 layers, hidden 1024, `glm5_next_vision`), `image_token_id 154854`, begin/end image ids 154830/154831; `processor_config.json` is present; `quantization_config` has `vision_bits`; tower tensors are EXL3-quantized (about 1 GB expected).
- Engine: `Glm5NextConfig` already registers `"vision": Glm5NextVisionModel` and parses `vision_pp`. NoPE attention means no mRoPE position work (unlike Qwen). `Embedding` and the MTP input layer already take `indexed_embeddings`; `Job` already forwards them to the MTP prefill (`job.py:1526`).
- Server (`tools/glm/serve.py`): nothing vision-related. `SERVE.md:81` says images are never sent to the tower.

Risks (each has a task or gate below):
1. **The model may not have been trained for these image tokens.** The shipped `chat_template.jinja` replaces every image with `<reminder>You are unable to process this image...</reminder>`, i.e. the vendor template is text-only. The checkpoint still ships the tower, so it very likely works, but only a real run shows it. Task 1 is a go/no-go gate.
2. **Hybrid KDA state and prefix cache.** `job.py:1470,1703` skip some recurrent-state checkpoint paths when a job has embeddings, and every new `MMEmbedding` gets fresh token ids, so an image prompt may re-prefill in full on every turn (about 580 tok/s). Task 7 measures this and fixes what it can.
3. **Memory.** The host runs at about 118 of 124 GiB with `-c 524288`. Tower weights plus activations for up to 8000 image tokens add roughly 1-2 GB. Default the server cap well below the processor's 8000 tokens.
4. **Speculative decoding with embeddings.** `generator.py:712` asserts no embeddings for the classic draft-model path. GLM uses the MTP path (`mtp_draft`), which is allowed, but this is untested for GLM. Task 1 and Task 4 test it; fallback is `speculative: false` for image requests.

## Global Constraints

- Inference and model files live on `walker@node-strix`; this PC is for editing and git only. Run anything that needs the GPU there.
- Text-only behaviour and the existing `tools/glm/test_serve.py` tests must not change; `--no-vision` must restore today's behaviour exactly.
- Match the style of `tools/glm/serve.py` and `tools/qwen/serve.py`; reuse names (`extract_images`, `load_image`, `BadRequest`, `MAX_IMAGE_BYTES`) rather than inventing new ones.

## Review Focus

- Image sent while the server runs with `--no-vision`: clean HTTP 400, not a crash.
- Undecodable, oversize, or non-http(s)/non-data URL image: 400 with a message, and the worker queue keeps serving.
- Image in a non-user message, or more images than placeholders: 400 (as in Qwen).
- Prompt plus image tokens exceeds the context: 400 `context length exceeded` (the chat handler counts an image as one token; the real count is known only after embedding).
- Two images in one message, and an image in a later turn after a text-only first turn.
- Text-only request after an image request: no stale embeddings, no MTP state leak.
- Huge image (e.g. 8000x8000): resize cap holds, no OOM.

## File Structure

- Modify `tools/glm/serve.py`: image extraction and loading, placeholder rewriting, vision tower load in `ResidentEngine`, `embeddings=` in `generate`, `--no-vision` and `--max-image-tokens` flags, `/health` flag if one exists.
- Modify `tools/glm/test_serve.py`: model-free tests with the existing `FakeEngine`.
- Create `tools/glm/vision_smoke.py`: GPU smoke test run on node-strix (Task 1 gate and final check).
- Modify `tools/glm/SERVE.md`, `README.md`: document the feature and its limits.

---

### Task 1: Go/no-go gate (offline, on node-strix, no server changes)

**Files:**
- Create: `tools/glm/vision_smoke.py`

**Interfaces:**
- Produces: `vision_smoke.py --model DIR --image PATH [--question TEXT] [--ctx N] [--no-spec]` printing the model's answer. Reused by Task 5.

- [ ] **Step 1: Write the script.** Follow `examples/multimodal.py` and `ResidentEngine.__init__`: load `Config`, text model, MTP model, then `Model.from_config(config, component="vision")`; load all three (use a small `ctx`, e.g. 16384, since the production server may be using the memory; stop it first or `llama-swap` will contend).

```python
emb = vision.get_image_embeddings(tokenizer=tokenizer, image=Image.open(path).convert("RGB"))
prompt = ("[gMASK]<sop><|user|>\n" + emb.text_alias + "\n" + question + "<|assistant|>\n<think></think>")
ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True, embeddings=[emb])
job = Job(input_ids=ids, max_new_tokens=200, sampler=GreedySampler(), embeddings=[emb],
          stop_conditions=config.eos_token_id_list)
```

  Use `examples/media/cat.png` and a screenshot with text; run once with MTP on, once with `--no-spec` (no draft model).

- [ ] **Step 2: Run on node-strix.** `ssh walker@node-strix`, activate the engine env (`source tools/strix_halo/env.sh`), run the script.
  Expected pass: the description matches the picture (cat; readable text) and MTP-on and MTP-off agree.
  Fail modes and action: load error on `model.visual.*` means a tensor-name/quant gap in `Glm4VVisionModel` (fix there); garbage or unrelated answer means check prompt format, then compare against the HF reference (`transformers` 5.16 `Glm5NextForConditionalGeneration` on a tiny sample) before concluding the model cannot see; assertion at `generator.py:712` means the MTP path is not taken, investigate `caps["mtp_draft"]`.
  **Stop here and report if the gate fails.**

- [ ] **Step 3: Record** the token count per image, vision-tower load time, and peak GPU/host memory delta in the commit message.

- [ ] **Step 4: Commit** `tools/glm/vision_smoke.py`.

### Task 2: Image plumbing in the server (model-free, TDD)

**Files:**
- Modify: `tools/glm/serve.py` (near `render_prompt`, line 123)
- Test: `tools/glm/test_serve.py`

**Interfaces:**
- Produces: `IMAGE_MARK = "<|exl3_image|>"`; `extract_images(messages) -> list[str]` (copy of `tools/qwen/serve.py:269`); `load_image(url)` (copy of `tools/qwen/serve.py:293`, with `BadRequest` and `MAX_IMAGE_BYTES = 20 * 1024**2`); `mark_images(messages) -> list[dict]` returning copies in which each image part becomes `{"type": "text", "text": IMAGE_MARK}` and raising `BadRequest` for images outside `user` messages.
- Consumes: nothing new. `BadRequest` must exist in `tools/glm/serve.py`; if the GLM server has no such class, define `class BadRequest(Exception)` and map it to HTTP 400 where the handler already maps template errors.

- [ ] **Step 1: Failing tests** in `test_serve.py`:

```python
def test_mark_images_replaces_parts_in_order(self):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]}]
    out = serve.mark_images(msgs)
    self.assertEqual([p["text"] for p in out[0]["content"]], ["a", serve.IMAGE_MARK, serve.IMAGE_MARK])
    self.assertEqual(len(serve.extract_images(msgs)), 2)

def test_image_in_assistant_message_rejected(self):
    msgs = [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    with self.assertRaises(serve.BadRequest):
        serve.mark_images(msgs)

def test_load_image_rejects_non_data_non_http(self):
    with self.assertRaises(serve.BadRequest):
        serve.load_image("file:///etc/passwd")
```

- [ ] **Step 2: Run** `python tools/glm/test_serve.py -k image` — expect FAIL (names missing).
- [ ] **Step 3: Implement** the four names above. `load_image` must keep the Qwen behaviour: data URI or http(s), size-capped, `Image.open(...).convert("RGB")`, decode errors become `BadRequest`.
- [ ] **Step 4: Run** the full `test_serve.py` — expect PASS, including the old tests.
- [ ] **Step 5: Commit** `feat(glm): image part extraction and placeholder marking`.

### Task 3: Template handling and placeholder substitution

**Files:**
- Modify: `tools/glm/serve.py` (chat handler, around where `render_prompt` is called, and `apply_template`)
- Test: `tools/glm/test_serve.py`

**Interfaces:**
- Consumes: `mark_images`, `extract_images`, `IMAGE_MARK`.
- Produces: in the chat handler, `images = extract_images(messages)`; `prompt = render_prompt(template, mark_images(messages), ...)`; validation `len(images) == prompt.count(IMAGE_MARK)` else `BadRequest`; job dict carries `"images": images`.

Why this works: the vendor template's `visible_text` macro passes `text` parts through unchanged, so a marker in a text part survives rendering; the real image parts would be replaced by the "unable to process" reminder, which is why they must be rewritten first.

- [ ] **Step 1: Failing test:** with the test template, a request holding one image part renders a prompt containing `IMAGE_MARK` exactly once and never the original data URI; FakeEngine receives `images=[url]` in `generate` kwargs. Extend `FakeEngine.generate` to accept `**kwargs` (it already does) and record `kwargs`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** in the chat handler and in `worker()` (pass `images=body.get("_images")` to `engine.generate`). Reject with 400 `"this server runs without the vision tower (--no-vision)"` when images are present and `getattr(engine, "supports_vision", False)` is false. `apply_template` should also use `mark_images` so conformance output matches real requests.
- [ ] **Step 4: Run all tests, expect PASS.**
- [ ] **Step 5: Commit** `feat(glm): route image parts through placeholder markers`.

### Task 4: Engine side: load tower, embed, pass to Job

**Files:**
- Modify: `tools/glm/serve.py` (`ResidentEngine.__init__` line 244, `generate` line 280, `main` line 769)
- Test: `tools/glm/test_serve.py` (CPU-only checks) and `tools/glm/vision_smoke.py` (GPU)

**Interfaces:**
- Produces: `ResidentEngine(..., vision: bool = True, max_image_tokens: int = 1280)`, attribute `supports_vision: bool`, `generate(..., images: list[str] | None = None)`.
- CLI: `--no-vision`, `--max-image-tokens N` (default 1280).

- [ ] **Step 1: Load the tower** after the text and MTP models, modelled on `tools/qwen/serve.py:641-650`:

```python
self.vision = None
if vision:
    try:
        self.config.vision_pp.max_image_tokens = min(self.config.vision_pp.max_image_tokens, max_image_tokens)
        self.vision = Model.from_config(self.config, component="vision")
        self.vision.load(device="cuda:0", progressbar=False)
    except Exception as exc:
        self.vision = None
        print(f"serve: vision tower not loaded ({exc!r}); image input disabled", flush=True)
self.supports_vision = self.vision is not None
```

  Check `Config.from_directory` / `Model.load` signatures against `ResidentEngine`'s existing calls and mirror them.

- [ ] **Step 2: Embed in `generate`.** Before `tokenizer.encode`:

```python
embs = []
for url in images or []:
    embs.append(self.vision.get_image_embeddings(tokenizer=self.tokenizer, image=load_image(url)))
for emb in embs:
    prompt = prompt.replace(IMAGE_MARK, emb.text_alias, 1)
ids = self.tokenizer.encode(prompt, encode_special_tokens=True, **({"embeddings": embs} if embs else {}))
```

  Run the embedding in `run_in_executor` (it is blocking GPU work; the Qwen server does the same at `serve.py:779`). Pass `embeddings=embs or None` to `Job`. Image decode errors must surface as `BadRequest` (HTTP 400) and leave the worker alive.

- [ ] **Step 3: Context check.** After encoding, if `ids.numel()` plus the reply budget exceeds the context, raise `BadRequest("context length exceeded: ...")` using the existing `reply_room` helper (line 218).
- [ ] **Step 4: Flags.** Add `--no-vision` and `--max-image-tokens` to `main()`; print a one-line startup note saying whether vision is on.
- [ ] **Step 5: GPU check on node-strix.** Start the server with `-c 32768 --num-draft 2`, then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":[{"type":"text","text":"What is in this picture?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -w0 examples/media/cat.png)"'"}}]}],"max_tokens":120,"temperature":0}'
```

  Expected: a correct description. Then send a text-only request and confirm normal output and speed. Then a two-image request, then an image request with a corrupt base64 payload (expect 400, server still answers the next request).

- [ ] **Step 6: Commit** `feat(glm): vision tower loading and image embeddings in the resident engine`.

### Task 5: Quality and regression checks

**Files:**
- Modify: `tools/glm/vision_smoke.py` (add a `--server URL` mode)

- [ ] **Step 1:** Run five fixed prompts over the server (cat, line drawing `examples/media/strawberry.png`, a text-heavy screenshot, a chart, a 3000x2000 photo). Record answers and token counts per image. Expected: sensible answers, large image stays under `--max-image-tokens`.
- [ ] **Step 2: Text regression.** Run the existing speed protocol (`tools/glm/serve_smoke.sh`) with and without `--no-vision`; decode and prefill speed must stay within the documented run-to-run noise (about 10 %). Note the resident-memory delta with the tower loaded.
- [ ] **Step 3: Long context.** Start at the production context (`-c 524288`) and check that the server still loads with the tower; if memory is too tight, document a lower recommended `-c` for vision or `--no-vision`.
- [ ] **Step 4: Commit** the results into the plan's notes or `tools/glm/SERVE.md`.

### Task 6: Documentation

**Files:**
- Modify: `tools/glm/SERVE.md` (the "Ignored" list at line 81 and the flags table), `README.md`, `README.upstream.md:225` (mark vision as supported for GLM 5.3-Flash if verified)

- [ ] **Step 1:** Remove "multimodal content parts" from the ignored list; add an "Images" section: accepted formats (data URI, http(s)), 20 MB cap, `--max-image-tokens`, images in user messages only, no video, image prompts do not reuse the recurrent-state prefix cache (full re-prefill each turn), `--no-vision`.
- [ ] **Step 2:** State what was and was not measured (Task 5), including the unknown that the vendor chat template says the model cannot process images.
- [ ] **Step 3: Commit** `docs(glm): vision input`.

### Task 7: Prefix reuse for prompts with images (extension)

**Problem.** `MMEmbedding.__init__` takes fresh token ids from a process-wide allocator for every embedding, so the same picture re-sent next turn gets different ids, different KV page hashes, and nothing is reused from the first image onwards. Separately `job.py` skips parts of the recurrent-state checkpointing when `self.embeddings` is set (lines 1470, 1703, 1725), which for the hybrid KDA stack may block reuse even with equal ids. Measure first, then fix only what the measurement shows.

**Files:** `tools/glm/serve.py` (`EmbeddingCache`, `ResidentEngine._embed`), `tools/glm/test_serve.py`, `tools/glm/vision_smoke.py` (`--reuse` mode), possibly `exllamav3/generator/job.py`.

- [ ] **Step 1: Failing model-free tests** for `EmbeddingCache(capacity)`: same image content returns the same object (embed function called once); different content embeds again; LRU eviction beyond capacity; key covers pixel content and size, not the URL (the same bytes under a different URL hit).
- [ ] **Step 2: Implement** `EmbeddingCache` (key = sha256 of mode, size, raw pixels) and use it in the engine so repeated images across requests keep their token ids.
- [ ] **Step 3: Measure on node-strix** with `vision_smoke.py --reuse`: turn 1 = image + question, turn 2 = same conversation plus the reply plus a follow-up question, same image. Record `cached_tokens` of turn 2 against the prompt length, with MTP on, for an image early in a long text prompt and for text after the image. Expected: cached_tokens covers at least the image and everything before the last recurrent checkpoint.
- [ ] **Step 4: If cached_tokens is still 0 or stops before the image end,** find why in `job.py` (recurrent checkpoint gating with embeddings, page hashing of multimodal spans) and lift the restriction with a test of the hybrid state (greedy output of the cached run must equal the uncached run for the same prompt). If a restriction is real (state that cannot be checkpointed across a non-causal span), keep it and document exactly what is and is not reused.
- [ ] **Step 5: Correctness guard.** Output of a reused-prefix turn must equal a cold run (`cache_prompt` off or after `/slots` erase) token for token at temperature 0.
- [ ] **Step 6:** Update `SERVE.md` (remove "full re-prefill each turn" if no longer true), commit `feat(glm): prefix reuse for image prompts`.

## Out of scope (YAGNI)

Video input (processor exists, but needs frame sampling and timestamps), a patched chat template inside the model pack, and image support through `/v1/completions`. (Prefix caching for image prompts was out of scope in the first draft; it is now Task 7.)

## Self-Review

- Coverage: tower load (T4), placeholder (T2/T3), embeddings to Job and MTP (T1/T4), flags and memory (T4/T5), docs (T6). Review Focus items map to T2 (assistant-message image, bad URL), T3 (`--no-vision`, mismatch count), T4 (corrupt image, context overflow, two images), T5 (huge image, text after image).
- Names are consistent across tasks: `IMAGE_MARK`, `extract_images`, `mark_images`, `load_image`, `BadRequest`, `supports_vision`, `images=` kwarg.
- Unverified assumptions, to settle in Task 1 and Task 4: exact `ResidentEngine`/`Model.load` call signatures, whether `BadRequest` already exists in `tools/glm/serve.py`, and the prompt format for the model (`<|begin_of_image|>` `<|image|>` tokens come from `MMEmbedding.token_string`; if Task 1 shows the model wants text between them, adjust there).
