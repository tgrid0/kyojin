# glm-serve — GLM-5.3 EXL3 OpenAI-compatible server

`tools/glm/serve.py` is a single-flight aiohttp server around one resident EXL3 model
(GLM-5.3 `Glm5NextForConditionalGeneration` + its MTP head). It exposes the two
OpenAI endpoints a client needs and nothing else.

## Start

```bash
source tools/strix_halo/env.sh   # from the repository root; sets PYTHONPATH
export TMPDIR="$PWD/scratch/glm-serve"; mkdir -p "$TMPDIR"
export EXL3_MOE_UNION_V2=1 EXL3_MOE_CFG=2 EXL3_HIP_PREFILL_MIN_ROWS=2 \
       EXL3_BLOCK_GRAPH=1 EXL3_BLOCK_GRAPH_MLA=2
python tools/glm/serve.py \
    --model ~/models/glm53-exl3-td205 --port 18080
```

`tools/glm/serve_smoke.sh` does exactly this (honouring `GLM_SRV_ROOT`, `GLM_SRV_PORT`,
`GLM_SRV_MODEL`, default port **18080**), waits for `/v1/models`, then exercises a plain
call, a streamed call, a tool call, and two 400-token benchmark calls.

| flag | default | meaning |
|---|---|---|
| `--model` | `~/models/glm53-exl3-td205` (hardcoded `DEFAULT_MODEL`) | EXL3 pack dir; also supplies `chat_template.jinja` |
| `--model-id` | `glm-5.3-exl3` | id reported by `/v1/models` and echoed in every response |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8000` | port (smoke script uses 18080) |
| `--chat-template` | the model's `chat_template.jinja` | Jinja template file; `tools/lanes/assets/glm53-template-medium.jinja` is the one the lane uses |
| `--default-temperature` / `--default-top-p` | `0.0` / `1.0` | used when a request omits the field (explicit null too). The lane passes `1.0` / `0.95`: greedy defaults loop on agent clients |
| `--default-reasoning-effort` | none | template `reasoning_effort` when a request omits it (the lane passes `medium`) |
| `--no-vision` | off | do not load the vision tower (about 0.3 GB of GPU memory); image requests then get a 400 |
| `--max-image-tokens` | `1280` | cap on the embedding rows of one image (the processor allows 8000) |
| `--max-history` | `1` | recurrent history slots; must equal draft tokens (1 for MTP). Each extra slot costs ~2.15 GiB |

Env vars the server sets for itself via `os.environ.setdefault` (so an external value
wins): `EXL3_MOE_UNION_V2=1`, `EXL3_MOE_CFG=2`, `EXL3_HIP_PREFILL_MIN_ROWS=2`,
`EXL3_BLOCK_GRAPH=1`, `EXL3_BLOCK_GRAPH_MLA=2`, plus the module-level
`Generator.MTP_FUSE_CATCHUP = 2`. There is no `--temperature`/seed handling beyond the
request body.

### MTP sidecar (higher draft acceptance)

The pack stores the MTP `eh_proj` at 2 bits. The pack also ships an unquantized copy,
`mtp_eh_proj.st` (BF16, 64 MiB), and the server loads it automatically: no flag, no variable.
It is a normal safetensors file with a `.st` extension, so no weight loader indexes it as
model weights. It only changes the draft; target token ids stay equal. Measured on the card
protocol (3 prompts, 400 tokens, greedy, Ryzen AI Max+ 395, MTP 2): 26.7 -> 31.6 tok/s.

Source order, one log line says which one was used:

1. `EXL3_MTP_EH_FP16=<path>` (a path that does not exist stops the load with an error; `0` turns the sidecar off)
2. `mtp_eh_proj.st` in the folder given by `--model`
3. `~/models/glm53-mtp-eh-proj-bf16.safetensors` (older location)
4. none: the server runs without it, with lower draft acceptance (one log line)

To build it yourself from the official GLM-5.3 checkpoint:

```bash
python tools/glm/mtp_eh_sidecar.py <official GLM-5.3 checkpoint dir> <model folder>/mtp_eh_proj.st
```

## Endpoints

### `GET /v1/models`
One entry: `{"id": "<model-id>", "object": "model", "created": 0, "owned_by": "local"}`.

### `POST /v1/chat/completions`

Honored:
- `messages` (required, non-empty array) — rendered through the model's own
  `chat_template.jinja` in a Jinja sandbox, with `add_generation_prompt=True`.
  Inbound `tool_calls[].function.arguments` are `json.loads`-ed when they are strings,
  because the GLM template iterates them as a dict.
- `tools` — passed to the template verbatim (no server-side validation).
- `stream` (bool) — SSE or a single JSON body.
- `max_completion_tokens` or `max_tokens` (the first wins; default **4096**), `temperature` (default **0.0** → greedy),
  `top_p` (default 1.0), `stop` (string or array; also fed to EXL3 as stop conditions).
- `model` — must equal `--model-id` if present, else 400.
- `clear_thinking` — forwarded to the template as a Jinja variable.

Ignored (accepted, no effect): `n`, `seed`, `logprobs`, `top_k`, `presence_penalty`,
`frequency_penalty`, `response_format`, `stream_options`, `user`, `parallel_tool_calls`,
`tool_choice`. Image parts are not ignored, see [Images](#images).

### Streaming format
`Content-Type: text/event-stream`. Every frame is a bare `data: {...}` line (no `event:`
name is emitted); frames are:

1. `{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`
2. incremental `delta.reasoning_content` while inside the think block, then
   `delta.content` for prose;
3. one frame with `delta.tool_calls` — the full parsed call list, each with `index`,
   emitted **only at the end** (the parser needs complete XML, so everything from
   `<\u200btool_call>` on is held back);
4. a final frame with an empty delta, `finish_reason` (`tool_calls` or `stop`) and
   `usage: {prompt_tokens, completion_tokens, total_tokens}`;
5. `data: [DONE]`.

If the engine fails mid-stream, the server sends a frame `{"error": {"message": ...}}`, a final frame with `finish_reason: "error"`, then `data: [DONE]`.

Every frame carries `id`, `object: chat.completion.chunk`, `created`, `model`.

Non-streaming returns `chat.completion` with `choices[0].message` holding `content`,
optional `reasoning_content`, optional `tool_calls`, plus `usage`.

### Images

`image_url` content parts (`{"type":"image_url","image_url":{"url":...}}`) are answered with the vision tower
that ships in the pack.

- URL: a `data:` URI or an `http(s)` URL; at most 20 MB. Anything else, an image that does not decode, or a URL that
  cannot be fetched is a 400 (also when `stream` is true), and the server keeps serving.
- Images are allowed in `user` messages only (400 otherwise). No video. `/completion` and `/v1/completions` take no
  images.
- Size: the picture is scaled to at most `--max-image-tokens` embedding rows (default 1280, about a megapixel
  of picture); an 8000x8000 image stays at that cap. Measured: a 640x800 photo is 667 rows, a 3000x2000 photo about 1300
  prompt tokens. `usage.prompt_tokens` counts the real rows.
- `--no-vision`, or a pack without the tower, answers image requests with 400 `this server runs without the vision tower`.
- The pack's chat template says the model cannot process images and replaces image parts by a reminder. The server
  therefore swaps every image for a marker before rendering and puts the embedding in afterwards. The marker text
  `<|exl3_image|>` must not appear in user text (that request gets a 400).
- Speculative decoding (MTP) stays on for image requests.

**Prefix reuse.** Pictures are cached by pixel content (8 most recent), so the same picture in the next turn keeps
its token ids and the KV pages and recurrent checkpoints of the earlier turn still match. Measured (6.4K-token prompt
with a picture, second turn adds a follow-up): 6400 of 6455 prompt tokens came from cache, 2.3 s against 25 s cold;
without the cache only the text ahead of the picture was reused. Limits: reuse needs the same picture and everything
before it unchanged; the same picture twice in one message is embedded twice and that request is not reused by the
next; a prompt shorter than the recurrent checkpoint spacing reuses little (a 700-token prompt reused 512 tokens).
A reused turn gives the same text as a cold one wherever the cold run itself repeats (MTP decoding is not bit-stable
between runs: 1 of 6 follow-ups differed for the image prompt, 2 of 6 for a text-only control).

**What was measured** (one Strix Halo machine, MTP 2, `-c 32768` and `-c 524288`): a cat photo, a line drawing, an
invoice text image and a bar chart were described correctly (the chart values were read); text-only prefill (655
tok/s) and decode (30 to 33 tok/s) are the same with and without the tower; at `-c 524288` the server loads with the
tower and runs at 117 of 124 GiB. Not measured: more than one picture-heavy conversation at long context, photos
with fine print, video. The vendor template says the model cannot see images; the checkpoint nevertheless answers
about them correctly in these tests, but it was not trained or evaluated for it by the vendor as far as is known.

### Thinking
GLM's generation prompt ends with `<think>`, so the model *starts inside* the block.
`opens_thinking()` detects that (`prompt.rstrip().endswith("<think>")`) and the server
prepends a synthetic `<think>` to the text it parses. That prefix is stripped again
before `usage` is computed. If a custom template does not end in `<think>`, reasoning is
simply absent. GLM tool calls are emitted by the model as `<\u200btool_call>name
<arg_key>k</arg_key><arg_value>v</arg_value></\u200btool_call>` — the zero-width space is
deliberate so plain prose never triggers the parser; arg values are JSON-parsed when they
parse, otherwise kept as strings.

## Known limits

- **One model, one flight.** The `asyncio.Queue` worker serializes every request; a
  second concurrent request waits, it is not batched. `n > 1` is ignored.
- **Client disconnect leaks work.** If the HTTP client goes away mid-stream, the worker
  keeps generating into the orphaned queue and every other request stalls until that
  generation ends.
- **Context length** is min(max_position_embeddings, 131072) = 128k (cache sized 128k + 4096).
- **Speed.** ~23.8 t/s end-to-end on the lighthouse bench through this server vs ~30.8 t/s
  in the standalone MTP bench. The gap is content, not the server: mtp_round on the same prose prompt
  gives 24.6 t/s (draft acceptance 0.57) vs 28.2 on chat (0.79); server overhead is ~3%.
- No auth, no CORS, no `/v1/completions`, no embeddings endpoint. Bind to 127.0.0.1.

## Open issues (found while documenting, NOT fixed)

1. **(FIXED) Context length was silently 32768, not 128k.** `ResidentEngine.__init__` does
   `self.config.config_dict.get("max_position_embeddings") or 32768`. For this pack
   `max_position_embeddings` (1048576) lives under `text_config`, and `config_dict` is
   the raw JSON — the top-level lookup misses and the fallback wins. The `Cache` is
   therefore sized `32768 + 4096` instead of `min(max_position, 131072) + 4096`, so
   requests past ~32k tokens will hit cache exhaustion. Fix: read
   `text_config->max_position_embeddings` (as `Glm5NextConfig` does for every other key).
2. **`main()` imports `os` a second time** inside the function (line 324) while the module
   already imports it at top level. Harmless, dead code.
3. **`usage.completion_tokens` is re-tokenized from the parsed text**, so it counts the
   *rendered* string (reasoning + prose + XML tool call) rather than the tokens actually
   generated. Close, but not the same number as EXL3's own count.

## Tests

```bash
python -m pytest -q tools/glm/test_serve.py
```

Model-free: a `FakeEngine` emits deterministic GLM XML in 3-character chunks so tags are
split mid-token, covering template rendering, think/tool parsing, stop strings, SSE
framing, `/v1/models`, usage, and the streamed open-`<think>` path. No GPU needed.

## Lane launcher

`tools/lanes/serve_glm.sh` runs this server with the exact flags and speed envs of our serving lane (see `tools/lanes/README.md`). The runtime uncensor hook is off unless `GLM_ABLIT` points at an edit spec.
