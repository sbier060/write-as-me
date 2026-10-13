"""Serve the Voiceify writer: one HTTPS endpoint that turns a writer prompt
(src/lib/writer.ts) into N candidate messages.

    modal deploy scripts/writer/serve_modal.py        # prints the URL
    WRITER_BASE=Qwen/Qwen2.5-7B-Instruct modal deploy scripts/writer/serve_modal.py

INTERNAL ONLY while the adapter is the Enron-trained one: that data is a
proof of concept and nothing trained on it ships to users. The app only calls
this for admin accounts (WRITER_URL + ADMIN_EMAILS).

The GPU sleeps after 15 idle minutes (no fixed cost); the first request after
that waits for the model to load (~40 s for 1.5B). Requests must carry the
shared token in the x-writer-token header (Modal secret voiceify-writer-token,
key WRITER_TOKEN).
"""
import os

import modal

# 7B is what the size comparison picked (2026-10-05, docs/writer + CLAUDE.md):
# quality 4.71 vs 4.50 for real mail, preferred over the current method 86%,
# Apache-2.0. 1.5B and 3B adapters are on the volume too; WRITER_BASE swaps.
BASE = os.environ.get("WRITER_BASE", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("WRITER_GPU", "A10G")
SLUG = BASE.split("/")[-1].lower()

app = modal.App("voiceify-writer-serve")
vol = modal.Volume.from_name("voiceify-writer")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1", "sentencepiece", "fastapi[standard]"
).env({"WRITER_BASE": BASE, "HF_HOME": "/vol/hf"})  # weights cached on the volume: a cold start loads 15 GB from disk, not from the internet


@app.cls(image=image, gpu=GPU, volumes={"/vol": vol}, scaledown_window=900, secrets=[modal.Secret.from_name("voiceify-writer-token")])
class Writer:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model = PeftModel.from_pretrained(base, f"/vol/adapters/{SLUG}/shared").eval()
        vol.commit()  # keeps the downloaded base weights for the next cold start

    @modal.fastapi_endpoint(method="POST")
    def write(self, body: dict, x_writer_token: str = ""):
        import torch
        from fastapi import Header, HTTPException  # noqa: F401

        if body.get("token") != os.environ["WRITER_TOKEN"]:
            raise HTTPException(status_code=401, detail="bad token")
        n = max(1, min(8, int(body.get("n", 4))))
        # "adapter": a voice trained into its own adapter (train_modal.py::train_users
        # writes user-<name>), e.g. "pg-essay". Absent = the shared writer, as before.
        # Loaded on first use and kept; one request runs at a time, so switching is safe.
        name = str(body.get("adapter") or "default")
        if name != "default":
            import re
            path = f"/vol/adapters/{SLUG}/user-{name}"
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", name):
                raise HTTPException(status_code=422, detail="bad adapter name")
            if name not in self.model.peft_config:
                if not os.path.exists(path):
                    vol.reload()  # trained after this container started
                if not os.path.exists(path):
                    raise HTTPException(status_code=404, detail=f"no adapter {name} for {SLUG}")
                self.model.load_adapter(path, adapter_name=name)
        self.model.set_adapter(name)
        prompt = self.tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True)
        self.tok.truncation_side = "left"
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=1536).to("cuda")
        with torch.no_grad():
            gen = self.model.generate(**ids, max_new_tokens=int(body.get("max_new_tokens", 300)), do_sample=True, temperature=float(body.get("temperature", 0.7)), top_p=0.9, repetition_penalty=1.05, num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
        texts = [self.tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        return {"texts": texts, "model": BASE, "adapter": name}
