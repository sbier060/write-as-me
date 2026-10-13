# write-as-me

A Claude Code skill (it also works in Cursor) that stops the assistant writing
in its own voice. The assistant decides what a message has to say; a small
fine-tuned model writes it the way one specific person writes, from that
person's real writing; the assistant checks the meaning and hands the text
back unedited.

```
you: "write Sam a note that the launch moved to Thursday, as me"
  → assistant: a plan (points and facts, no phrasing)
  → writer model: 6 candidates in your voice, from the plan + your real writing
  → assistant: checks each against the plan, returns one as written
```

## Install

Claude Code:

```
/plugin marketplace add sbier060/write-as-me
/plugin install write-as-me@write-as-me
```

Cursor: copy `skills/write-as-me/` into `~/.cursor/skills/`.

Needs Node 18 or newer. No npm install.

## Set up

1. **An endpoint.** Put this in `~/.config/mymodel/writer.env`:

   ```
   WRITER_URL=https://...
   WRITER_TOKEN=...
   ```

   Either someone who runs an endpoint gives you both, or you run your own
   (below).

2. **Your writing.** The voice comes from things you really wrote: at least
   3 samples, 30 or more is better. Sent emails and posts work best.

   ```bash
   node skills/write-as-me/write.mjs add my-emails.txt
   ```

   Separate samples with a line of `---`. They stay on your machine; the six
   closest to each request are sent to the endpoint with it.

3. Ask for something "as me", or "in my voice".

## Other voices

```bash
node skills/write-as-me/write.mjs fetch paul-graham
```

downloads Paul Graham's essays from paulgraham.com to your machine and cuts
them into paragraphs. Then "write this like Paul Graham" uses them. For anyone
else, collect what they wrote and `add <file> --voice <name>`.

This borrows a style. Do not sign as the other person or present the result as
something they wrote.

## Run your own endpoint

`server/serve_modal.py` serves the writer on [Modal](https://modal.com): a
Qwen2.5-7B-Instruct base with a LoRA adapter, on a GPU that sleeps when idle.
`server/train_modal.py` trains the adapter. The reference setup:

- **Shared adapter**: LoRA r=16, alpha=32, all linear layers, 2 epochs at
  2e-4, on about 6,000 (plan → real message) rows from 76 authors. Each row
  shows up to six of the same person's other messages, then notes on what the
  new message must say; the target is what they really wrote. It learns the
  task, not a person.
- **Per-voice adapter** (optional): the shared adapter trained further on one
  writer. Requested with `{"adapter": "<name>"}`; a voice folder's
  `voice.json` sets it.

Both scripts expect a Modal volume named `voiceify-writer` and a secret named
`voiceify-writer-token` with the key `WRITER_TOKEN`. The training rows are
chat-format JSONL built with the exact prompt in `write.mjs`; change one and
you must change the other.

## Honest limits

- It writes short pieces: about 20 to 150 words per request. Longer text is
  written a paragraph at a time.
- It invents sometimes. The skill has the assistant check every candidate
  against the plan and retry once; read what it gives you.
- The reference shared adapter was trained on the Enron email corpus as a
  proof of concept. That is public research data from people who did not
  consent to this use: fine for an experiment, not for a product.
- Nothing here is tuned against AI detectors, and it should not be described
  as undetectable.
