---
name: write-as-me
description: >-
  Write prose in the user's own voice with a small fine-tuned writer model
  instead of writing it yourself. Use whenever you are about to produce text
  that goes out under their name: an email, reply, DM, Slack or text message,
  social post, bio, intro, announcement, outreach note, or when they say
  "write this as me", "in my voice", "make it sound like me", or ask for
  another installed voice ("write this like Paul Graham"). Not for code, code
  comments, commit messages, technical docs, or copy in a brand's voice.
---

# Write as me

A small fine-tuned model (Qwen2.5-7B + LoRA, hosted on Modal) owns the words.
Your job is to decide WHAT the text says and to check the result. You do not
write or polish the prose: every word you add puts a generic assistant's voice
back in, which is the thing this skill exists to remove.

The script is `write.mjs` in this skill's folder. It needs Node 18+ and
`~/.config/mymodel/writer.env` holding `WRITER_URL` and `WRITER_TOKEN`. If a
command says there is no endpoint, stop and tell the user: the README explains
how to get one. Do not write the text yourself instead.

## Steps

1. **Plan.** Write a plan as JSON to a temp file. Content only, no phrasing:

   ```json
   {
     "purpose": "one line: what this message is for",
     "to": "who it is to and the relationship, or omit",
     "recipientName": "first name for the greeting, or omit",
     "senderName": "their first name, only if it should be signed",
     "speaker": "I",
     "items": [
       { "kind": "inform", "note": "plain statement of the point", "facts": ["Thursday", "$4,000"], "tone": "neutral" }
     ]
   }
   ```

   - `kind`: ask | inform | commit | thanks | apology | decline | offer.
     `tone` (optional): polite | neutral | firm.
   - `facts`: every number, date, time, price, name, URL, exactly as it must
     appear. The writer copies these verbatim.
   - Notes are flat and short ("can't make Thursday, offer Friday morning").
     Do not write nice sentences in the notes; the writer will copy them.
   - Only what they told you or what the source material says. Add nothing: no
     pleasantries, no reasons, no offers they did not ask for.
   - One plan is one short piece: about 20 to 150 words, 2 to 6 items. The
     model was trained on short messages and stops near 200 words. For
     anything longer, make one plan per paragraph or section, run each, and
     join the results in order.

2. **Write.** `node <skill dir>/write.mjs --plan <plan.json> --n 6`
   It prints JSON with `candidates`, each with `text`, `words`, and
   `missingFacts`. The first call after a quiet spell takes up to two
   minutes while the GPU wakes; later calls take a few seconds.

3. **Check, do not rewrite.** For each candidate, against the plan:
   - unsupported: says something the plan does not
   - missing: a plan point or fact is absent (`missingFacts` must be empty)
   - changed: a point's meaning shifted (watch near-synonyms: "postponed"
     for "moved", a different day, a softened commitment)

   Greetings and sign-offs are style, not content.

4. **Retry once if none pass.** Take the closest candidate, write
   `{"unsupported": [...], "missing": [...], "changed": [...]}` to a file,
   and run again with `--fix <file>`.

5. **Deliver** the passing candidate that reads most like the samples,
   exactly as written: their casing, punctuation, and length included. Do not
   fix grammar or tidy it. If nothing passed after the retry, give the
   closest one and say in one line what is wrong with it. Never fall back to
   writing it yourself without saying so.

## Their writing samples

The voice comes from the user's real writing in `~/.config/mymodel/voice/` (`.txt`
or `.md`, samples separated by a line of `---`); the six closest to the plan
go into each request. More and better samples means a closer voice.

- `node write.mjs samples` shows how many there are. Fewer than 3 and the
  writer refuses: ask for a few things they wrote (sent emails, posts) first.
- `node write.mjs add <file>` (or `add -` from stdin) adds more. When they
  paste or point at something they wrote themselves, offer to add it. Never add
  text that a model wrote.

## Other voices

`--voice <name>` on any command swaps whose writing is shown to the writer:
`~/.config/mymodel/voices/<name>/`, same file format. Use it when they ask
to write "as" or "like" that person.

- `node write.mjs fetch paul-graham` downloads his essays from his own site
  (about two minutes, once) and cuts them into paragraphs. Run it the first
  time that voice is asked for. The text stays on this machine.
- A voice folder with `voice.json` (`{"adapter": "pg-essay"}`) also uses an
  adapter trained on that writer, when the endpoint has it. For a new voice, collect 30+ passages of 40 to 160
words that the person really wrote and `add` them with `--voice <name>`.

- Leave `senderName` out and never sign as the other person: this borrows a
  style, it does not put words in their mouth. Do not present the result as
  something they wrote or said.
- Essay voices write paragraphs, not messages: plan one paragraph at a time
  (`kind: inform`, 2 to 4 items) and expect the argument to need checking.
  In testing the Paul Graham voice matched his length and rhythm and
  sometimes garbled the logic.

## Limits to be honest about

- The shared adapter in the reference setup was trained on the Enron corpus
  as a proof of concept. Personal use only: do not ship its output as a
  product.
- Do not describe the output as undetectable or as passing AI detectors.
