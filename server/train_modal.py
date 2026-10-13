"""Train and run the MyModel writer on a rented GPU (Modal).

The writer is a small open-weight model with LoRA adapters:
  shared   one adapter trained on many real people's mail — learns the task
           "plan + this person's messages -> the message as they'd write it".
  per user the shared adapter trained a little further on ONE person's own
           mail. That copy is their adapter: their voice in weights, not in a
           prompt.

    modal run scripts/writer/train_modal.py::upload --data-dir <outdir of dataset.mts>
    modal run scripts/writer/train_modal.py::train_shared
    modal run scripts/writer/train_modal.py::train_users
    modal run scripts/writer/train_modal.py::exam --out exam-out.jsonl

exam writes, for every hidden message, what the writer produced three ways:
  shared     shared adapter + the person's messages in the prompt
  personal   that person's own adapter (unseen authors only)
  base       the untrained base model, same prompt — the control
Nothing here replaces the product's current method; the outputs are scored
against it by scripts/writer/ab.mts.

UNTESTED until a Modal account exists: written 2026-10-05 without a GPU to
run it on. Expect to fix small things on first run.
"""
import json
import os

import modal

# The base model is a parameter, not a decision. 1.5B is the FIRST BASELINE
# (Apache-2.0, commercially usable), not the chosen production model. To try
# a bigger writer change nothing but the environment:
#   WRITER_BASE=Qwen/Qwen2.5-7B-Instruct WRITER_GPU=L40S WRITER_BATCH=4 modal run ...
# Adapters and exam outputs are stored per base model (see _slug), LoRA
# targets every linear layer whatever the architecture calls them, and the
# prompt comes from the tokenizer's own chat template — so nothing below is
# specific to Qwen or to a size. (Qwen2.5-3B is research-licence only; Llama
# 3.2 3B is gated; Phi-4-mini 3.8B and Qwen2.5-7B are the next candidates.)
BASE = os.environ.get("WRITER_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
GPU = os.environ.get("WRITER_GPU", "A10G")
# 4 x 2 accumulation = an effective batch of 8. A real batch of 8 at 1,536
# tokens ran a 24 GB A10G out of memory on the 1.5B model (first smoke run).
BATCH = int(os.environ.get("WRITER_BATCH", "4"))
ACCUM = int(os.environ.get("WRITER_ACCUM", "2"))
MAX_TOKENS = int(os.environ.get("WRITER_MAX_TOKENS", "1536"))
# Which dataset directory on the volume (dataset.mts output). "data" is the
# Enron set; a second set (e.g. one essayist) uploads beside it.
DATA = os.environ.get("WRITER_DATA", "data")
# Candidates per exam row. With K > 1 the exam also returns "<arm>_k": every
# sample, so a meaning checker can choose among them (scripts/writer/verify.mts).
K = int(os.environ.get("WRITER_K", "1"))


app = modal.App("voiceify-writer")
vol = modal.Volume.from_name("voiceify-writer", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "datasets==3.1.0", "accelerate==1.1.1", "sentencepiece"
).env({"WRITER_BASE": BASE, "WRITER_BATCH": str(BATCH), "WRITER_ACCUM": str(ACCUM), "WRITER_MAX_TOKENS": str(MAX_TOKENS), "WRITER_DATA": DATA, "WRITER_K": str(K), "WRITER_SKIP_BASE": os.environ.get("WRITER_SKIP_BASE", "0"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # the container trains what this shell asked for

def _slug():
    return BASE.split("/")[-1].lower()

V = "/vol"


def _rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load(adapter=None, trainable=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules="all-linear"))
    return tok, model


def _train(rows, out, init=None, epochs=2, lr=2e-4):
    """Standard causal-LM fine-tune; the loss counts only the assistant turn
    (the person's real message), never the prompt."""
    import torch
    from torch.utils.data import DataLoader

    tok, model = _load(init, trainable=True)
    model.print_trainable_parameters()
    # LoRA on every linear layer keeps ~1 GB of activations per layer at
    # 4 x 1,500 tokens; checkpointing recomputes them instead (fix found by the
    # "Claude Chrome extension modal" session, commit 92a1014: peak 4.5 GB).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    def encode(r):
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(r["messages"][-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + a)[-MAX_TOKENS:]
        labels = ([-100] * len(p) + a)[-MAX_TOKENS:]
        return {"input_ids": ids, "labels": labels}

    data = [encode(r) for r in rows]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: xs + [v] * (n - len(xs))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }

    loader = DataLoader(data, batch_size=BATCH, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = max(1, epochs * len(loader) // ACCUM)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 1, pct_start=0.05)
    model.train()
    lm = model.get_base_model()  # the causal LM with the LoRA layers in place
    step, seen, running = 0, 0, 0.0
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # Logits only where there is something to learn. The stock forward
            # builds vocabulary-sized logits (150k wide) for every prompt
            # token too — 4 x 1,536 x 150k floats is what ran a 24 GB card
            # out of memory, and none of it reaches the loss. Works for any
            # HF causal LM: body via .base_model, head via get_output_embeddings().
            hidden = lm.base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            want = batch["labels"][:, 1:] != -100
            logits = lm.get_output_embeddings()(hidden[:, :-1][want])
            loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"][:, 1:][want])
            (loss / ACCUM).backward()
            running += loss.item(); seen += 1
            if seen % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 25 == 0:
                    print(f"step {step}/{steps} loss {running / (25 * ACCUM):.3f}", flush=True)
                    running = 0.0
        # Saved every epoch: a timeout then costs one epoch, not the run.
        model.save_pretrained(out)
        vol.commit()
        print(f"saved {out} · peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def _generate(tok, model, rows, max_new=260, batch=16, k=1):
    """Batched sampling (left-padded): one at a time made the exam take hours."""
    import torch

    out = []
    model.eval()
    tok.padding_side = "left"
    # If a prompt is too long, lose the START (oldest example), never the end:
    # the notes are last, and a prompt cut on the right has no notes, so the
    # model just continues the person's examples (seen on a Paul Graham row).
    tok.truncation_side = "left"
    for i in range(0, len(rows), batch):
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows[i:i + batch]]
        ids = tok(prompts, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=MAX_TOKENS).to("cuda")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.05, pad_token_id=tok.pad_token_id, num_return_sequences=k)
        texts = [tok.decode(g[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        out += texts if k == 1 else [texts[j:j + k] for j in range(0, len(texts), k)]
        print(f"generated {len(out)}/{len(rows)}", flush=True)
    return out


@app.local_entrypoint()
def upload(data_dir: str):
    with vol.batch_upload(force=True) as b:
        b.put_directory(data_dir, f"/{DATA}")
    print("uploaded", data_dir)


def _check_manifest():
    """Refuse to train on a dataset that did not pass dataset.mts's safeguards."""
    m = json.load(open(f"{V}/{DATA}/manifest.json"))
    s = m["safeguards"]
    assert s["unseenAuthorsInSharedTraining"] == 0 and s["examMessagesInTraining"] == 0, s
    print("dataset:", m["dataset"])
    print("unseen authors (never in shared training):", ", ".join(m["authors"]["unseen"]))
    print("rows:", m["rows"], "| plans excluded for leaking:", s["plansExcludedForLeakingWording"], flush=True)


@app.function(image=image, gpu=GPU, volumes={V: vol}, timeout=8 * 3600)
def train_shared(epochs: int = 2, limit: int = 0):
    """limit > 0 trains on the first N rows only: a smoke test of the plumbing."""
    _check_manifest()
    rows = _rows(f"{V}/{DATA}/train.jsonl")
    _train(rows[:limit] if limit else rows, f"{V}/adapters/{_slug()}/shared", epochs=epochs)


@app.function(image=image, gpu=GPU, volumes={V: vol}, timeout=8 * 3600)
def train_users(epochs: int = 3, limit: int = 0):
    """One adapter per unseen author, each starting from the shared writer."""
    d = f"{V}/{DATA}/adapters"
    for name in sorted(os.listdir(d))[: limit or None]:
        author = name.removesuffix(".jsonl")
        print("== adapter for", author, flush=True)
        _train(_rows(f"{d}/{name}"), f"{V}/adapters/{_slug()}/user-{author}", init=f"{V}/adapters/{_slug()}/shared", epochs=epochs, lr=1e-4)


@app.function(image=image, gpu=GPU, volumes={V: vol}, timeout=8 * 3600)
def run_exam(limit: int = 0):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import torch

    rows = _rows(f"{V}/{DATA}/exam.jsonl")
    if limit:  # smoke test: a few unseen-author rows and a few seen
        rows = [r for r in rows if r["unseen"]][:limit] + [r for r in rows if not r["unseen"]][:limit]
    tok, base = _load()
    results = [{"id": r["id"], "author": r["author"], "unseen": r["unseen"], "real": r["real"], "plan": r.get("plan")} for r in rows]
    if os.environ.get("WRITER_SKIP_BASE") != "1":  # the untrained control is slow; skip it on re-runs
        for res, text in zip(results, _generate(tok, base, rows)):
            res["base"] = text
    shared = PeftModel.from_pretrained(base, f"{V}/adapters/{_slug()}/shared")
    for res, text in zip(results, _generate(tok, shared, rows, k=K, batch=16 if K == 1 else 8)):
        res["shared"] = text if K == 1 else text[0]
        if K > 1:
            res["shared_k"] = text
    del shared, base
    torch.cuda.empty_cache()
    for author in sorted({r["author"] for r in rows if r["unseen"]}):
        path = f"{V}/adapters/{_slug()}/user-{author}"
        if not os.path.exists(path):
            continue
        m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda"), path)
        idx = [i for i, r in enumerate(rows) if r["author"] == author]
        for i, text in zip(idx, _generate(tok, m, [rows[i] for i in idx])):
            results[i]["personal"] = text
        del m
        torch.cuda.empty_cache()
    for r in results:
        r["base_model"] = BASE
    return results


@app.local_entrypoint()
def exam(out: str = "", limit: int = 0):
    out = out or f"exam-out-{_slug()}.jsonl"
    with open(out, "w") as f:
        for r in run_exam.remote(limit):
            f.write(json.dumps(r) + "\n")
    print("wrote", out)
