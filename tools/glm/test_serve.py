#!/usr/bin/env python3
"""Model-free glm-serve tests. Fake engines emit deterministic GLM XML."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve


class FakeEngine:
    def __init__(self, output: str):
        self.output = output
        self.prompts = []
        self.calls = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.calls.append(kwargs)
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


class ServeTests(unittest.TestCase):
    template = "{% for m in messages %}<|im_start|>{{ m.role }}: {{ m.content }}<|im_end|>{% endfor %}{% if tools %}TOOLS={{ tools|length }}{% endif %}"

    def test_eh_sidecar_source_order(self):
        import os, tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as home:
            pack = Path(home) / "pack"
            pack.mkdir()
            with mock.patch.dict(os.environ, {"HOME": home}):
                os.environ.pop("EXL3_MTP_EH_FP16", None)
                # nothing anywhere: variable stays unset, no FileNotFoundError at load
                self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "absent")
                self.assertNotIn("EXL3_MTP_EH_FP16", os.environ)
                # old default path only
                old = Path(home) / "models" / "glm53-mtp-eh-proj-bf16.safetensors"
                old.parent.mkdir()
                old.write_bytes(b"x")
                self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "default")
                self.assertEqual(os.environ["EXL3_MTP_EH_FP16"], str(old))
                # file in the model folder wins over the old path
                os.environ.pop("EXL3_MTP_EH_FP16")
                (pack / serve.EH_SIDECAR_IN_PACK).write_bytes(b"x")
                self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "pack")
                self.assertEqual(os.environ["EXL3_MTP_EH_FP16"], str(pack / "mtp_eh_proj.st"))
                # explicit value (even a missing file, even "0") is never touched
                for value in ("/nope/x.safetensors", "0"):
                    os.environ["EXL3_MTP_EH_FP16"] = value
                    self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "explicit")
                    self.assertEqual(os.environ["EXL3_MTP_EH_FP16"], value)
        self.assertNotIn("EXL3_MTP_EH_FP16", serve.SPEED_ENV)
        self.assertFalse(serve.EH_SIDECAR_IN_PACK.endswith(".safetensors"))

    def test_stream_engine_error_still_ends_with_finish_reason(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Boom(FakeEngine):
            async def generate(self, prompt: str, **kwargs):
                yield "partial "
                raise RuntimeError("kaput")

        async def check():
            client = TestClient(TestServer(serve.create_app(Boom("x"), "m", self.template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await r.read()).decode()
            await client.close()
            return raw

        raw = run(check())
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        chunks = [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: {")]
        self.assertTrue(any("kaput" in c.get("error", {}).get("message", "") for c in chunks))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "error")

    def test_max_completion_tokens_is_honored(self):
        self.assertEqual(serve.token_limit({"max_completion_tokens": 7, "max_tokens": 9}), 7)
        self.assertEqual(serve.token_limit({"max_tokens": 9}), 9)
        self.assertEqual(serve.token_limit({"max_tokens": None}), 4096)
        from aiohttp.test_utils import TestClient, TestServer

        class Spy(FakeEngine):
            async def generate(self, prompt: str, **kwargs):
                self.kwargs = kwargs
                async for piece in super().generate(prompt, **kwargs):
                    yield piece

        engine = Spy("one two three four")

        async def check():
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "max_completion_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
            body = await r.json()
            await client.close()
            return body

        body = run(check())
        self.assertEqual(engine.kwargs["max_tokens"], 4)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_reply_budget_is_clamped_to_the_context_room(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Spy(FakeEngine):
            ctx = 20
            num_draft = 1

            async def generate(self, prompt: str, **kwargs):
                self.kwargs = kwargs
                async for piece in super().generate(prompt, **kwargs):
                    yield piece

        engine = Spy("one two")

        async def post(content, **extra):
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": content}], **extra})
            status = r.status
            await r.read()
            await client.close()
            return status

        # room = ctx - prompt - 1 slot - 1 draft token
        self.assertEqual(run(post("a b c d", max_completion_tokens=32768)), 200)
        used = engine.count_tokens(engine.prompts[-1])
        self.assertEqual(engine.kwargs["max_tokens"], 20 - used - 1 - 1)
        self.assertEqual(run(post("a b c d", max_tokens=3)), 200)
        self.assertEqual(engine.kwargs["max_tokens"], 3)
        engine.kwargs = None
        self.assertEqual(run(post(" ".join("w" * 30))), 400)
        self.assertIsNone(engine.kwargs)

    def test_mark_images_replaces_parts_in_order(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "a"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]}]
        out = serve.mark_images(msgs)
        self.assertEqual([p["text"] for p in out[0]["content"]], ["a", serve.IMAGE_MARK, serve.IMAGE_MARK])
        self.assertEqual(len(serve.extract_images(msgs)), 2)
        self.assertEqual(msgs[0]["content"][1]["type"], "image_url")   # the input is not modified

    def test_image_in_assistant_message_rejected(self):
        msgs = [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        with self.assertRaises(serve.BadRequest):
            serve.mark_images(msgs)

    def test_load_image_rejects_non_data_non_http(self):
        with self.assertRaises(serve.BadRequest):
            serve.load_image("file:///etc/passwd")

    def test_load_image_rejects_undecodable_and_oversize(self):
        for url in ("data:image/png;base64,!!!notbase64", "data:image/png;base64,AAAA", "data:image/png", ""):
            with self.assertRaises(serve.BadRequest, msg=url):
                serve.load_image(url)
        big = "data:image/png;base64," + "A" * (serve.MAX_IMAGE_BYTES * 4 // 3 + 16)
        with self.assertRaises(serve.BadRequest):
            serve.load_image(big)

    # Renders list content like the vendor template's visible_text macro: text parts pass through, anything else is dropped.
    image_template = ("{% for m in messages %}<|{{ m.role }}|>"
                      "{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}"
                      "{% if p.type == 'text' %}{{ p.text }}{% else %}[DROPPED]{% endif %}{% endfor %}{% endif %}"
                      "{% endfor %}")

    @staticmethod
    def png_url(size=(8, 6), color=(200, 30, 30)) -> str:
        import base64, io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def chat(self, engine, messages, template=None, **extra):
        from aiohttp.test_utils import TestClient, TestServer

        async def go():
            client = TestClient(TestServer(serve.create_app(engine, "m", template or self.image_template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": messages, **extra})
            status, text = r.status, await r.text()
            r2 = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "still alive?"}]})
            await r2.read()
            await client.close()
            return status, text, r2.status
        return run(go())

    def user_with_images(self, *urls, text="what is this?"):
        return [{"role": "user", "content": [{"type": "text", "text": text}] +
                 [{"type": "image_url", "image_url": {"url": u}} for u in urls]}]

    def test_image_request_routes_marker_and_images_to_the_engine(self):
        engine = FakeEngine("a cat")
        engine.supports_vision = True
        status, text, after = self.chat(engine, self.user_with_images(self.png_url(), self.png_url((4, 4))))
        self.assertEqual((status, after), (200, 200))
        self.assertEqual(engine.prompts[0].count(serve.IMAGE_MARK), 2)
        self.assertNotIn("data:image", engine.prompts[0])
        self.assertNotIn("DROPPED", engine.prompts[0])
        self.assertEqual([im.size for im in engine.calls[0]["images"]], [(8, 6), (4, 4)])
        self.assertFalse(engine.calls[1].get("images"))     # a text request after an image request carries none

    def test_image_without_vision_tower_is_400(self):
        engine = FakeEngine("x")
        engine.supports_vision = False
        status, text, after = self.chat(engine, self.user_with_images(self.png_url()))
        self.assertEqual((status, after), (400, 200))
        self.assertIn("--no-vision", text)
        self.assertEqual(len(engine.prompts), 1)              # only the follow-up text request ran

    def test_bad_images_are_400_and_the_server_keeps_serving(self):
        engine = FakeEngine("x")
        engine.supports_vision = True
        for url in ("data:image/png;base64,AAAA", "file:///etc/passwd", "data:image/png;base64,!!!"):
            status, text, after = self.chat(engine, self.user_with_images(url))
            self.assertEqual((status, after), (400, 200), url)
        self.assertEqual(len(engine.prompts), 3)              # the three follow-ups only

    def test_image_outside_a_user_message_is_400(self):
        engine = FakeEngine("x")
        engine.supports_vision = True
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": self.png_url()}}]},
                {"role": "user", "content": "and?"}]
        status, text, after = self.chat(engine, msgs)
        self.assertEqual((status, after), (400, 200))

    def test_typed_marker_cannot_forge_an_image_slot(self):
        engine = FakeEngine("x")
        engine.supports_vision = True
        status, text, after = self.chat(engine, [{"role": "user", "content": "look " + serve.IMAGE_MARK}])
        self.assertEqual((status, after), (400, 200))

    def test_streaming_image_request_with_a_bad_image_is_400_not_a_stream_error(self):
        engine = FakeEngine("x")
        engine.supports_vision = True
        status, text, after = self.chat(engine, self.user_with_images("data:image/png;base64,AAAA"), stream=True)
        self.assertEqual((status, after), (400, 200))

    def test_apply_template_marks_images_like_a_real_request(self):
        from aiohttp.test_utils import TestClient, TestServer

        async def go():
            client = TestClient(TestServer(serve.create_app(FakeEngine("x"), "m", self.image_template)))
            await client.start_server()
            r = await client.post("/apply-template", json={"messages": self.user_with_images(self.png_url())})
            body = await r.json()
            await client.close()
            return body["prompt"]
        prompt = go_prompt = run(go())
        self.assertEqual(prompt.count(serve.IMAGE_MARK), 1)
        self.assertNotIn("DROPPED", prompt)

    def test_template_rendering(self):
        out = serve.render_prompt(self.template, [{"role": "user", "content": "hi"}], [{"type": "function"}])
        self.assertEqual(out, "<|im_start|>user: hi<|im_end|>TOOLS=1")

    def test_parse_thinking_and_tool_calls(self):
        text = ('<think>reasoning with\nlines</think>Visible '
                '<tool_call>weather<arg_key>city</arg_key><arg_value>{"name": "Paris",\n"x": 1}</arg_value></tool_call>'
                '<tool_call>time<arg_key>tz</arg_key><arg_value>UTC</arg_value></tool_call>')
        result = serve.parse_completion(text)
        self.assertEqual(result["reasoning_content"], "reasoning with\nlines")
        self.assertEqual(result["content"], "Visible ")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(json.loads(result["tool_calls"][0]["function"]["arguments"]), {"city": {"name": "Paris", "x": 1}})
        self.assertEqual(result["tool_calls"][1]["function"]["name"], "time")

    def test_warmup_runs_one_long_prompt(self):
        engine = FakeEngine("ok")
        elapsed = serve.warmup(engine, self.template, prompt_tokens=300, decode_tokens=4)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(len(engine.prompts), 1)
        self.assertGreaterEqual(engine.count_tokens(engine.prompts[0]), 290)

    def test_dense_tune_path_follows_the_cpp_tuner(self):
        from unittest import mock
        with mock.patch.dict("os.environ", {"EXL3_DENSE_GEMM_TUNE_FILE": "/x/t.txt"}):
            self.assertEqual(serve.dense_tune_path(), Path("/x/t.txt"))
        with mock.patch.dict("os.environ", {"HOME": "/h"}, clear=True):
            self.assertEqual(serve.dense_tune_path(), Path("/h/.cache/exllamav3/dense_gemm_tune.txt"))
        with mock.patch.dict("os.environ", {"EXL3_DENSE_GEMM_TUNE_FILE": "/nonexistent/t.txt"}):
            self.assertEqual(serve.prime_dense_tune(10.0), (0, 0.0))

    def test_stop_strings(self):
        self.assertEqual(serve.stop_text("hello STOP world", ["STOP"]), ("hello ", "STOP"))
        self.assertEqual(serve.stop_text("hello", ["STOP"]), ("hello", None))

    def test_sse_framing(self):
        frame = serve.sse({"hello": "world"})
        self.assertEqual(frame, b'data: {"hello":"world"}\n\n')

    def test_endpoints_and_usage_without_model(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('<think>why</think>hello')
        app = serve.create_app(engine, "test-model", self.template)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            models = await (await client.get("/v1/models")).json()
            self.assertEqual(models["data"][0]["id"], "test-model")
            response = await client.post("/v1/chat/completions", json={
                "model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
            body = await response.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "hello")
            self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "why")
            self.assertIn("usage", body)
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
            cut = await (await client.post("/v1/chat/completions", json={
                "model": "test-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})).json()
            self.assertEqual(cut["choices"][0]["finish_reason"], "length")
            await client.close()

        run(check())

    def test_stream_parallel_tool_calls_one_message_each(self):
        from aiohttp.test_utils import TestClient, TestServer
        call = '<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>'
        app = serve.create_app(FakeEngine("plan</think>Sure." + call * 3), "m", self.template + "<think>")

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await response.read()).decode()
            await client.close()
            return raw
        chunks = [json.loads(l[6:]) for l in run(check()).split("\n") if l.startswith("data: {")]
        msgs = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
        self.assertEqual([len(m) for m in msgs], [1, 1, 1])
        self.assertEqual([m[0]["index"] for m in msgs], [0, 1, 2])

    def test_stream_open_think_and_tool_call(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('plan it</think>Sure.<tool_call>get_weather<arg_key>city</arg_key>'
                            '<arg_value>Paris</arg_value></tool_call>')
        app = serve.create_app(engine, "m", self.template + "<think>")

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await response.read()).decode()
            await client.close()
            return raw

        raw = run(check())
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        chunks = [json.loads(line[6:]) for line in raw.split("\n") if line.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertEqual("".join(d.get("reasoning_content", "") for d in deltas), "plan it")
        self.assertEqual("".join(d.get("content", "") for d in deltas), "Sure.")
        self.assertGreater(sum("reasoning_content" in d for d in deltas), 1)  # incremental
        calls = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        self.assertEqual(calls[0][0]["function"]["name"], "get_weather")
        self.assertEqual(calls[0][0]["index"], 0)
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks][-1], "tool_calls")
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))
        self.assertIn("usage", chunks[-1])


    def test_client_disconnect_cancels_the_job(self):
        """Stream and non-stream drops stop generation within a few steps, a job cancelled while queued
        never runs, and the request after them is served."""
        from aiohttp.test_utils import TestClient, TestServer

        class SlowEngine(FakeEngine):
            steps = 0
            stopped = None

            async def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                cancel = kwargs.get("cancel")
                if "slow" not in prompt:
                    yield "ok"
                    return
                self.stopped = None
                for _ in range(1000):
                    if cancel is not None and cancel.is_set():
                        self.stopped = "cancelled"
                        return
                    self.steps += 1
                    await asyncio.sleep(0.01)
                    yield "t "
                self.stopped = "finished"

        def body(text, stream):
            return {"model": "m", "stream": stream, "messages": [{"role": "user", "content": text}]}

        async def settle(engine, want_steps_at_most):
            for _ in range(200):
                if engine.stopped is not None:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(engine.stopped, "cancelled")
            self.assertLess(engine.steps, want_steps_at_most)

        async def check():
            engine = SlowEngine("x")
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            # 1. streaming drop mid-stream
            response = await client.post("/v1/chat/completions", json=body("slow a", True))
            await response.content.readany()
            await asyncio.sleep(0.15)
            response.close()
            await settle(engine, 100)
            # 2. a following request is served at once
            ok = await asyncio.wait_for(client.post("/v1/chat/completions", json=body("hi", False)), 5)
            self.assertEqual(ok.status, 200)
            self.assertEqual((await ok.json())["choices"][0]["message"]["content"], "ok")
            # 3. non-streaming drop
            engine.steps = 0
            try:
                await asyncio.wait_for(client.post("/v1/chat/completions", json=body("slow b", False)), 0.3)
            except asyncio.TimeoutError:
                pass
            await settle(engine, 100)
            # 4. a job cancelled while queued never reaches the engine
            engine.steps, engine.prompts = 0, []
            first = await client.post("/v1/chat/completions", json=body("slow c", True))
            await first.content.readany()
            try:
                await asyncio.wait_for(client.post("/v1/chat/completions", json=body("slow queued", False)), 0.3)
            except asyncio.TimeoutError:
                pass
            first.close()
            ok = await asyncio.wait_for(client.post("/v1/chat/completions", json=body("hi again", False)), 5)
            self.assertEqual(ok.status, 200)
            self.assertFalse(any("slow queued" in p for p in engine.prompts))
            await client.close()

        run(check())


if __name__ == "__main__":
    unittest.main()
