#!/usr/bin/env node
// write-as-me: hand a plan to the trained MyModel writer (Qwen2.5-7B + LoRA on
// Modal) and get the message back in the user's voice. Zero dependencies.
//
//   node write.mjs --plan plan.json [--n 6] [--fix fix.json]   candidates as JSON
//   node write.mjs add <file|->                                 add real writing (samples split on a line of ---)
//   node write.mjs fetch <paul-graham|mark-twain|...>           download a public writer's work as a voice (stays on this machine; no name lists them)
//   node write.mjs samples                                      how much writing it has
//   --voice <name> on any of these: someone else's voice, from ~/.config/mymodel/voices/<name>/
//   node write.mjs warm                                         wake the GPU (it sleeps after 15 idle minutes)
//
// The prompt below is the training contract (WRITER_FORMAT 1, the same text
// server/train_modal.py's dataset is built with). The adapter was trained on exactly
// this text: change a word here and the output gets worse, silently.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const CONF = process.env.WRITE_AS_ME_HOME || path.join(os.homedir(), ".config", "mymodel");
const env = { ...readEnv(path.join(CONF, "writer.env")), ...process.env };
const URL_ = env.WRITER_URL;
const TOKEN = env.WRITER_TOKEN;

const SYSTEM = "You write a message exactly the way one specific person writes. You are shown messages they really wrote, then notes on what a new message must say. Write the new message as they would. Every fact in the notes appears exactly as given. Nothing that is not in the notes is added.";

function readEnv(file) {
  if (!fs.existsSync(file)) return {};
  return Object.fromEntries(fs.readFileSync(file, "utf8").split("\n").map((l) => /^\s*(?:export\s+)?([A-Z_]+)=(.*)$/.exec(l)).filter(Boolean).map((m) => [m[1], m[2].trim().replace(/^["']|["']$/g, "")]));
}

function planNotes(plan) {
  return [
    `What it's for: ${plan.purpose}`,
    plan.to && plan.to !== "unknown" ? `To: ${plan.to}` : "",
    plan.recipientName ? `Recipient's name (from the greeting): ${plan.recipientName}` : "Recipient's name: not given",
    plan.senderName ? `Your name (from the sign-off): ${plan.senderName}` : "",
    `You are writing as: ${plan.speaker ?? "I"}`,
    "Points:",
    ...plan.items.map((it, i) => `${i + 1}. [${it.kind}${it.tone ? ` · ${it.tone}` : ""}] ${it.note}${it.facts?.length ? ` — facts: ${it.facts.map((f) => `"${f}"`).join(", ")}` : ""}`),
  ].filter(Boolean).join("\n");
}
const clip = (t, words = 140) => { const w = t.trim().split(/\s+/); return w.length > words ? `${w.slice(0, words).join(" ")} …` : t.trim(); };
const writerPrompt = (plan, examples) => `Messages this person really wrote:\n${examples.slice(0, 6).map((e) => `<message>\n${clip(e)}\n</message>`).join("\n")}\n\nNotes for the new message:\n${planNotes(plan)}\n\nWrite the message.`;

function samples() {
  if (!fs.existsSync(VOICE)) return [];
  return fs.readdirSync(VOICE).filter((f) => /\.(txt|md)$/.test(f)).sort()
    .flatMap((f) => fs.readFileSync(path.join(VOICE, f), "utf8").split(/^\s*---\s*$/m))
    .map((s) => s.trim()).filter((s) => s.split(/\s+/).length >= 8);
}

// Closest in subject to the notes, which is what the writer saw in training.
const STOP = new Set("the a an and or but of to in on for with at by from is are was were be been it this that i you we they he she my your our as not so if do have has had will can just me".split(" "));
const bag = (t) => { const m = new Map(); for (const w of t.toLowerCase().match(/[a-z']{2,}/g) ?? []) if (!STOP.has(w)) m.set(w, (m.get(w) ?? 0) + 1); return m; };
const cosine = (a, b) => { let d = 0, na = 0, nb = 0; for (const [k, v] of a) { na += v * v; if (b.has(k)) d += v * b.get(k); } for (const v of b.values()) nb += v * v; return na && nb ? d / Math.sqrt(na * nb) : 0; };
function pick(plan, all, k = 6) {
  const q = bag(planNotes(plan));
  return all.map((t, i) => ({ t, s: cosine(q, bag(t)) + (all.length - i) * 1e-9 })).sort((a, b) => b.s - a.s).slice(0, k).map((x) => x.t);
}

async function call(body, wait = 180_000) {
  if (!URL_ || !TOKEN) fail(`No writer endpoint. Put WRITER_URL=... and WRITER_TOKEN=... in ${path.join(CONF, "writer.env")} (see the README: deploy server/serve_modal.py, or ask whoever runs one for a token).`);
  const until = Date.now() + wait;
  for (let last = "";;) {
    try {
      const r = await fetch(URL_, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ token: TOKEN, ...body }), signal: AbortSignal.timeout(120_000) });
      if (r.ok) return await r.json();
      last = `writer ${r.status}: ${(await r.text()).slice(0, 200)}`;
      if (r.status === 404 && body.adapter) { delete body.adapter; continue; } // this endpoint has no such adapter: the samples alone carry the voice
      if (r.status === 401 || r.status === 422) fail(last);
    } catch (e) { last = String(e); }
    // Asleep or still loading the model: it wakes in about a minute.
    if (Date.now() > until) fail(`The writer did not answer (${last}). The GPU may still be waking: try again in a minute.`);
    await new Promise((r) => setTimeout(r, 5000));
  }
}
function fail(msg) { console.error(msg); process.exit(1); }
// Whose voice: the user's own by default; --voice <name> reads ~/.config/mymodel/voices/<name>/.
const arg = (name) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : undefined; };
const VOICE = arg("voice") ? path.join(CONF, "voices", arg("voice").toLowerCase().replace(/[^a-z0-9]+/g, "-")) : path.join(CONF, "voice");
// A voice folder may hold voice.json {"adapter": "<name>"}: that voice was also
// trained into its own adapter on the endpoint, which is then used with the samples.
const ADAPTER = (() => { try { return JSON.parse(fs.readFileSync(path.join(VOICE, "voice.json"), "utf8")).adapter; } catch { return undefined; } })();
const readJson = (f) => JSON.parse(f === "-" ? fs.readFileSync(0, "utf8") : fs.readFileSync(f, "utf8"));

// Public writers whose posts can be fetched as a voice. The text is downloaded
// from the writer's own site to this machine and is never redistributed.
//
// The `texts` voices are public domain in the US: published before 1931, by
// writers dead long enough that no estate licenses the name. Books come from
// Project Gutenberg's own mirror (gutenberg.pglaf.org: Gutenberg asks scripts
// to use it, not the main site), which only hosts texts out of US copyright. Translations are the
// old ones on purpose: a modern translation has its own copyright, so never
// swap in a newer edition. Hopkins is not on Gutenberg; his is the library scan
// of the 1927 first edition. Anyone added here must pass the same test.
//
// `from` and `to` cut a book down to the writer's own words: from the last line
// starting with `from` (past any table of contents) up to the first line
// starting with `to` (an editor's appendix, an index). Keep editors'
// introductions and dialogues with speaker tags out: they teach the wrong voice.
const book = (id, from, to) => ({ url: `https://gutenberg.pglaf.org/cache/epub/${id}/pg${id}.txt`, from, to });
const SOURCES = {
  "paul-graham": { index: "https://paulgraham.com/articles.html", base: "https://paulgraham.com/", skip: ["index.html", "articles.html", "rss.html", "bio.html", "faq.html"], adapter: "pg-essay" },
  "claude-hopkins": { texts: [{ url: "https://archive.org/download/mylifeinadverti00unkngoog/mylifeinadverti00unkngoog_djvu.txt", from: "THIS book is not written" }] }, // My Life in Advertising, 1927
  "pt-barnum": { texts: [book(8581)] }, // The Art of Money Getting, 1880
  "benjamin-franklin": { texts: [book(148, "Dear Son:")] }, // Autobiography
  "mark-twain": { texts: [book(3176), book(2895)] }, // The Innocents Abroad; Following the Equator
  "ralph-waldo-emerson": { texts: [book(2944), book(2945)] }, // Essays, First and Second Series
  "henry-david-thoreau": { texts: [book(205)] }, // Walden; Civil Disobedience
  "oscar-wilde": { texts: [book(1017), book(921)] }, // The Soul of Man under Socialism; De Profundis
  "abraham-lincoln": { texts: [book(14721, "LINCOLN'S SPEECHES AND LETTERS", "ANECDOTES")] }, // Speeches & Letters, 1832-1865
  "theodore-roosevelt": { texts: [book(58821, "THE STRENUOUS LIFE")] }, // The Strenuous Life
  "marcus-aurelius": { texts: [book(15877, "THE THOUGHTS", "INDEXES.")] }, // George Long's 1862 translation
  "seneca": { texts: [book(64576), book(3794)] }, // Minor Dialogues; On Benefits; Aubrey Stewart's 1887-1889 translations
};
const decode = (s) => s.replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(+n)).replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&nbsp;/g, " ").replace(/&rsquo;|&lsquo;/g, "'").replace(/&rdquo;|&ldquo;/g, '"').replace(/&mdash;/g, "—");
function essayText(html) {
  const body = html.replace(/<script[\s\S]*?<\/script>/gi, "").replace(/<br\s*\/?>\s*<br\s*\/?>/gi, "\n\n").replace(/<br\s*\/?>/gi, "\n").replace(/<p[^>]*>/gi, "\n\n");
  let text = decode(body.replace(/<[^>]+>/g, "")).replace(/\r/g, "").replace(/[ \t]+/g, " ").replace(/\n[ \t]+/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  const date = text.match(/^(January|February|March|April|May|June|July|August|September|October|November|December) \d{4}/m)?.[0] ?? "";
  if (date) text = text.slice(text.indexOf(date) + date.length).trim();
  // Notes, thanks and translation lists are not the writer's voice.
  text = text.split(/\n\s*(Notes|Thanks to|Related:|Translations?:|Japanese Translation)\b/)[0].replace(/\[\d+\]/g, "").trim();
  return text.replace(/([^\n])\n(?!\n)/g, "$1 ").replace(/ {2,}/g, " ");
}
// A plain-text book cut into paragraphs: Gutenberg's licence header and footer
// off, OCR page furniture out, words split across lines and paragraphs split
// across scanned pages joined back up.
function bookParas(raw, { from, to } = {}) {
  let text = raw.replace(/\r/g, "");
  const start = text.search(/^\*\*\* ?START OF/m), end = text.search(/^\*\*\* ?END OF/m);
  if (start >= 0) text = text.slice(text.indexOf("\n", start) + 1, end > start ? end : undefined);
  const lines = text.split("\n");
  const first = from ? lines.findLastIndex((l) => l.startsWith(from)) : -1;
  if (from && first < 0) fail(`"${from}" not found: the edition may have changed.`);
  const last = to ? lines.findIndex((l, i) => i > first && l.startsWith(to)) : -1;
  text = lines.slice(Math.max(first, 0), last > first ? last : undefined).join("\n");
  text = text.replace(/^.*Digitized by.*$|^\s*Google\s*$/gm, "").replace(/(\w)- ?\n\s*(\w)/g, "$1$2");
  const chunks = [];
  for (const c of text.split(/\n\s*\n/)) {
    const lns = c.split("\n").filter((l) => l.trim());
    if (!lns.length) continue;
    // Verse (epigraphs, quoted poems) is set in short lines; it is not the voice.
    if (lns.length >= 3 && lns.reduce((n, l) => n + l.trim().length, 0) / lns.length < 50) continue;
    const p = c.replace(/\s+/g, " ").trim();
    if (/^[a-z]/.test(p) && chunks.length) chunks[chunks.length - 1] += " " + p; // a page break mid-sentence
    else chunks.push(p);
  }
  // Section numbers and footnote marks off; headings, chapter synopses, tables
  // of contents, licence text and scan noise are not the voice.
  return chunks.map((p) => /^\[/.test(p) ? "" : p.replace(/^([IVXLC]+|\d+)\. /, "").replace(/\[[A-Z0-9]{1,3}\]/g, "")).filter((p) => {
    const n = p.split(" ").length, letters = p.replace(/[^a-z]/gi, "").length;
    return n >= 40 && n <= 160 && letters / p.length > 0.7 && !/^chapter\b|gutenberg|google/i.test(p) && p !== p.toUpperCase();
  });
}
async function fetchVoice(name) {
  const src = SOURCES[name];
  if (!src) fail(`No built-in source for "${name}". Known: ${Object.keys(SOURCES).join(", ")}. For anyone else, collect what they wrote and use: add <file> --voice ${name}`);
  const dir = path.join(CONF, "voices", name), ua = "write-as-me/1.0 (personal style samples)";
  if (src.texts) {
    const paras = [];
    for (const t of src.texts) {
      const r = await fetch(t.url, { headers: { "user-agent": ua } }).catch(() => undefined);
      if (!r?.ok) fail(`Could not download ${t.url} (${r?.status ?? "network error"}).`);
      paras.push(...bookParas(await r.text(), t));
    }
    return saveVoice(name, dir, src, paras);
  }
  const index = await (await fetch(src.index, { headers: { "user-agent": ua } })).text();
  const links = [...new Set([...index.matchAll(/href="([a-z0-9_-]+\.html)"/gi)].map((m) => m[1]))].filter((l) => !src.skip.includes(l));
  const paras = [];
  for (const [i, link] of links.entries()) {
    const r = await fetch(src.base + link, { headers: { "user-agent": ua } }).catch(() => undefined);
    if (r?.ok) {
      const text = essayText(await r.text());
      if (text.split(/\s+/).length >= 300) for (const p of text.split(/\n\s*\n/).map((x) => x.replace(/\s+/g, " ").trim())) { const n = p.split(" ").length; if (n >= 40 && n <= 160) paras.push(p); }
    }
    if (i % 25 === 24) console.error(`${i + 1}/${links.length} pages · ${paras.length} paragraphs`);
    await new Promise((res) => setTimeout(res, 400)); // one request at a time, politely
  }
  saveVoice(name, dir, src, paras);
}
function saveVoice(name, dir, src, paras) {
  if (paras.length < 30) fail(`Only ${paras.length} paragraphs found: the source's layout may have changed.`);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "posts.txt"), paras.join("\n---\n") + "\n");
  if (src.adapter && !fs.existsSync(path.join(dir, "voice.json"))) fs.writeFileSync(path.join(dir, "voice.json"), JSON.stringify({ adapter: src.adapter }) + "\n");
  console.log(`${name}: ${paras.length} paragraphs in ${dir}`);
}

const cmd = process.argv[2];
if (cmd === "samples") {
  const s = samples();
  console.log(JSON.stringify({ dir: VOICE, samples: s.length, words: s.join(" ").split(/\s+/).filter(Boolean).length }));
} else if (cmd === "fetch") {
  await fetchVoice((process.argv[3] ?? "").toLowerCase());
} else if (cmd === "warm") {
  await call({ n: 1, max_new_tokens: 1, messages: [{ role: "user", content: "hi" }] });
  console.log("warm");
} else if (cmd === "add") {
  const src = process.argv[3] && !process.argv[3].startsWith("--") ? process.argv[3] : "-";
  const text = (src === "-" ? fs.readFileSync(0, "utf8") : fs.readFileSync(src, "utf8")).trim();
  if (text.split(/\s+/).length < 8) fail("Too short to be a sample.");
  fs.mkdirSync(VOICE, { recursive: true });
  const file = path.join(VOICE, `added-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`);
  fs.writeFileSync(file, text + "\n");
  console.log(`added ${file} · now ${samples().length} samples`);
} else {
  const planFile = arg("plan");
  if (!planFile) fail("usage: node write.mjs --plan plan.json [--n 6] [--fix fix.json]");
  const plan = readJson(planFile);
  if (!plan.purpose || !Array.isArray(plan.items) || !plan.items.length) fail("The plan needs a purpose and at least one item.");
  const all = samples();
  if (all.length < 3) fail(`Only ${all.length} samples of the person's own writing in ${VOICE}. The voice comes from these: add at least 3 (node write.mjs add <file>).`);
  const examples = pick(plan, all);
  let user = writerPrompt(plan, examples);
  const fix = arg("fix") ? readJson(arg("fix")) : undefined;
  if (fix && [fix.unsupported, fix.missing, fix.changed].some((x) => x?.length)) {
    const lines = [
      ...(fix.unsupported ?? []).slice(0, 6).map((x) => `- Do not say: "${x}" (it is not in the notes)`),
      ...(fix.missing ?? []).slice(0, 6).map((x) => `- Must include: ${x}`),
      ...(fix.changed ?? []).slice(0, 6).map((x) => `- This changed the meaning, say what the notes say instead: "${x}"`),
    ];
    user = user.replace(/\n\nWrite the message\.$/, `\n\nAn earlier attempt got these wrong:\n${lines.join("\n")}\n\nWrite the message.`);
  }
  const n = Math.max(1, Math.min(8, Number(arg("n") ?? 6)));
  const j = await call({ n, ...(ADAPTER && !process.argv.includes("--no-adapter") ? { adapter: ADAPTER } : {}), messages: [{ role: "system", content: SYSTEM }, { role: "user", content: user }] });
  const facts = plan.items.flatMap((it) => it.facts ?? []);
  const candidates = [...new Set((j.texts ?? []).map((t) => t.trim()).filter(Boolean))].map((text) => ({
    text, words: text.split(/\s+/).length,
    // The cheap gate: every fact the plan lists must be in the text, exactly.
    missingFacts: facts.filter((f) => !text.toLowerCase().includes(f.toLowerCase())),
  })).sort((a, b) => a.missingFacts.length - b.missingFacts.length);
  console.log(JSON.stringify({ model: j.model, adapter: j.adapter ?? "default", samplesAvailable: all.length, examplesUsed: examples.length, candidates }, null, 2));
}
