// End-to-end check of the main screen in mock mode: every recorded run, plus a synthetic refusal and an unknown
// question, in headless Chromium at 1920x1080 against a running app.
//
//   NEXT_PUBLIC_MOCK_SPEED=10 npm run build && NEXT_PUBLIC_MOCK_SPEED=10 npm start   # in one shell
//   NEXT_PUBLIC_MOCK_SPEED=10 npm run e2e                                            # in another
//
// Env: E2E_BASE_URL (default http://localhost:3000), NEXT_PUBLIC_MOCK_SPEED (must match the server; default 1),
// E2E_SHOTS=<dir> to save screenshots of finished answers.

import { mkdirSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const FIXTURES = path.join(ROOT, "fixtures");
const BASE = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const SPEED = Number(process.env.NEXT_PUBLIC_MOCK_SPEED) > 0 ? Number(process.env.NEXT_PUBLIC_MOCK_SPEED) : 1;
const SHOTS = process.env.E2E_SHOTS;
const VIEWPORT = { width: 1920, height: 1080 };

const readJson = (...p) => JSON.parse(readFileSync(path.join(FIXTURES, ...p), "utf8"));
const readRun = (id) =>
  readFileSync(path.join(FIXTURES, "ask", `${id}.jsonl`), "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
const fmt = (s) => `${s < 0.1 ? s.toFixed(2) : s.toFixed(1)} s`;
const SOURCE_LABEL = {
  slack: "Slack",
  gmail: "Gmail",
  google_drive: "Google Drive",
  confluence: "Confluence",
  jira: "Jira",
  linear: "Linear",
  github: "GitHub",
  hubspot: "HubSpot",
  fireflies: "Fireflies",
};

class Check {
  constructor(name) {
    this.name = name;
    this.failures = [];
    this.passed = 0;
  }
  ok(cond, what) {
    if (cond) this.passed += 1;
    else this.failures.push(what);
  }
  eq(actual, expected, what) {
    const a = JSON.stringify(actual);
    const e = JSON.stringify(expected);
    this.ok(a === e, `${what}: expected ${e}, got ${a}`);
  }
}

async function newPage(browser) {
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(BASE);
  await page.locator("[data-question]").first().waitFor();
  return { page, errors, close: () => context.close() };
}

async function checkRecorded(browser, q) {
  const c = new Check(q.question_id);
  const events = readRun(q.question_id);
  const D = Object.fromEntries(events.map((e) => [e.event, e.data]));
  const { page, errors, close } = await newPage(browser);
  const lastT = events.at(-1).t;

  const t0 = Date.now();
  await page.locator(`[data-question="${q.question_id}"]`).click();

  // While waiting: the answer skeleton shows, and stages fill in as their events arrive.
  await page.locator('[data-role="answer-skeleton"]').waitFor({ timeout: 5000 });
  c.ok(true, "answer skeleton shown");
  if (SPEED <= 2) {
    const rerank = page.locator('[data-stage="rerank"][data-status="running"]');
    await rerank.waitFor({ timeout: 30000 });
    const t1 = parseFloat(await rerank.locator('[data-role="stage-time"]').innerText());
    await page.waitForTimeout(1200);
    const t2 = parseFloat(await rerank.locator('[data-role="stage-time"]').innerText());
    c.ok(t2 > t1 + 0.8, `rerank timer runs (${t1} -> ${t2})`);
    for (const s of ["route", "retrieve", "fuse"]) {
      c.eq(await page.locator(`[data-stage="${s}"]`).getAttribute("data-status"), "done", `${s} done while reranking`);
    }
    for (const s of ["generate", "verify"]) {
      c.eq(await page.locator(`[data-stage="${s}"]`).getAttribute("data-status"), "pending", `${s} pending while reranking`);
    }
    c.ok(await page.locator('[data-role="fused-preview"]').isVisible(), "fused candidates shown while reranking");
  }

  // The cost renders only after `done` (the live timer alone can pass through the recorded total).
  await page.locator('[data-role="cost"]').waitFor({ timeout: (lastT / SPEED) * 1000 + 20000 });
  c.ok((await page.locator('[data-role="total"]').innerText()).startsWith(fmt(D.done.seconds.total)), "total time");
  const wall = (Date.now() - t0) / 1000;
  c.ok(wall >= (lastT / SPEED) * 0.9 && wall <= lastT / SPEED + 4, `replay took ${wall.toFixed(1)} s for ${lastT} s at ${SPEED}x`);

  // Timeline: one card per start.stages entry, all done with the event's own seconds; route shows its sources.
  const stages = await page.locator("[data-stage]").evaluateAll((els) => els.map((e) => e.dataset.stage));
  c.eq(stages, D.start.stages, "timeline stages");
  for (const s of D.start.stages) {
    const card = page.locator(`[data-stage="${s}"]`);
    c.eq(await card.getAttribute("data-status"), "done", `${s} status`);
    c.eq(await card.locator('[data-role="stage-time"]').innerText(), fmt(D[s].seconds), `${s} seconds`);
  }
  const badges = await page.locator('[data-role="routed"] [data-source]').evaluateAll((els) => els.map((e) => e.dataset.source));
  c.eq(badges, D.route.sources, "route badges");
  c.ok((await page.locator('[data-role="cost"]').innerText()).includes(`$${D.done.cost_usd.toFixed(4)}`), "total cost");

  // Answer: every marker became a chip, the chips cite exactly generate.citations, banners match the flags.
  const answer = page.locator('[data-role="answer"]');
  const chips = await page
    .locator('[data-role="answer"] [data-cite], [data-role="not-covered"] [data-cite]')
    .evaluateAll((els) => els.map((e) => Number(e.dataset.cite)));
  c.eq([...new Set(chips)].sort((a, b) => a - b), D.generate.citations.map((x) => x.n).sort((a, b) => a - b), "answer chips");
  const answerText = await answer.innerText();
  c.ok(!/[[【]\d/.test(answerText), "no raw citation markers left");
  const opening = D.generate.answer.replace(/[*`]/g, "").split(/\s+/).slice(0, 5).join(" ");
  c.ok(answerText.replace(/\s+/g, " ").includes(opening), `answer text shown ("${opening}")`);
  c.eq(await page.locator('[data-banner="partial"]').count(), D.generate.partial ? 1 : 0, "partial banner");
  c.eq(await page.locator('[data-role="not-covered"]').count() > 0, D.generate.partial, "not-covered sentence marked");
  c.eq(await page.locator('[data-banner="abstained"]').count(), 0, "no refusal banner");
  c.eq(await page.locator('[data-role="version-pair"]').count(), D.generate.version_pairs.length, "version pair notes");

  // Confidence.
  const conf = D.verify.confidence;
  if (conf.flagged) {
    c.eq(await page.locator('[data-banner="flagged"]').count(), 1, "flagged banner");
    const doubtful = conf.claims.filter((x) => x.verdict !== "supported").length;
    c.eq(await page.locator('[data-role="doubtful-claims"] > li').count(), doubtful, "doubtful claims listed");
    c.ok((await page.locator('[data-role="reasoning"]').innerText()) === conf.reasoning, "verifier reasoning shown");
  } else {
    c.eq(await page.locator('[data-banner="verified"]').count(), 1, "verified line");
  }
  c.eq(await page.locator('[data-role="low-retrieval"]').count(), conf.low_retrieval_score ? 1 : 0, "low retrieval note");

  // Evidence: the 10 hits in order, cited ones marked.
  const hits = await page.locator("[data-rank]").evaluateAll((els) =>
    els.map((e) => ({ rank: Number(e.dataset.rank), doc: e.dataset.doc, cited: e.dataset.cited === "true" })),
  );
  c.eq(hits.map((h) => h.doc), D.rerank.hits.map((h) => h.doc_id), "evidence order");
  c.eq(
    hits.filter((h) => h.cited).map((h) => h.rank).sort((a, b) => a - b),
    D.generate.citations.map((x) => x.n).sort((a, b) => a - b),
    "cited marks",
  );
  const first = D.rerank.hits[0];
  const card1 = await page.locator('[data-rank="1"]').evaluate((e) => e.textContent);
  c.ok(card1.includes(first.title) && card1.includes(SOURCE_LABEL[first.source]), "hit title and source");
  c.ok(card1.includes(`was #${first.fused_rank} before reranking`) && card1.includes(first.rerank_score.toFixed(3)), "hit score and fused rank");

  // A chip highlights its document and scrolls it into the evidence column's view.
  const n = D.generate.citations.at(-1).n;
  await answer.locator(`[data-cite="${n}"]`).first().click();
  await page.waitForTimeout(800);
  c.eq(await page.locator(`[data-rank="${n}"]`).getAttribute("data-highlighted"), "true", `chip ${n} highlights doc ${n}`);
  const inView = await page.evaluate((rank) => {
    const col = document.querySelector('[data-role="evidence-column"]').getBoundingClientRect();
    const card = document.querySelector(`[data-rank="${rank}"]`).getBoundingClientRect();
    return card.top >= col.top - 1 && card.bottom <= col.bottom + 1;
  }, n);
  c.ok(inView, `doc ${n} scrolled into view`);

  if (SHOTS) {
    // Frame the confidence panel for flagged answers whose banner is below the fold.
    await page.evaluate(() => {
      document.querySelector('[data-banner="flagged"]')?.scrollIntoView({ block: "center" });
    });
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(SHOTS, `${q.question_id}.png`) });
  }

  // The drawer shows the full document.
  const doc = readJson("documents", `${D.rerank.hits[n - 1].doc_id}.json`);
  await page.locator(`[data-rank="${n}"] button`).click();
  const content = page.locator('[data-role="document-content"]');
  await content.waitFor({ timeout: 5000 });
  const shown = await content.evaluate((e) => e.textContent);
  c.eq(shown.length, doc.content.length, "drawer shows the full content");
  c.ok((await page.locator('[data-role="drawer"] h2').innerText()) === doc.title, "drawer title");
  await page.keyboard.press("Escape");
  c.eq(await page.locator('[data-role="drawer"]').count(), 0, "drawer closes on Escape");

  c.eq(errors, [], "no page errors");
  await close();
  return c;
}

async function checkRefusal(browser) {
  // No recorded run is a refusal: build one from the contract on top of the info_not_found run's earlier stages.
  const c = new Check("synthetic refusal");
  const events = readRun("qst_0484");
  const D = Object.fromEntries(events.map((e) => [e.event, e.data]));
  const frame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const body =
    events
      .filter((e) => ["start", "route", "retrieve", "fuse", "rerank"].includes(e.event))
      .map((e) => frame(e.event, e.data))
      .join("") +
    frame("generate", {
      answer: "The provided documents do not contain the information needed to answer this question.",
      abstained: true,
      partial: false,
      citations: [],
      context: D.generate.context,
      version_pairs: [],
      seconds: 3.1,
      cost_usd: 0.0029,
    }) +
    frame("verify", { confidence: null, skipped: "abstained", seconds: 0, cost_usd: 0 }) +
    frame("done", { seconds: { route: 1.8, retrieve: 4.3, fetch: 0.1, rerank: 33.8, generate: 3.1, total: 43.1 }, cost_usd: 0.0029 });

  const { page, errors, close } = await newPage(browser);
  await page.route("**/mock-api/ask?*", (route) =>
    route.fulfill({ status: 200, headers: { "Content-Type": "text/event-stream" }, body }),
  );
  await page.locator('[data-question="qst_0484"]').click();
  await page.locator('[data-banner="abstained"]').waitFor({ timeout: 10000 });
  c.ok(true, "refusal banner");
  c.ok((await page.getByText("Not verified: refusals are not checked.").count()) === 1, "verify skipped note");
  c.eq(await page.locator('[data-stage="verify"]').getAttribute("data-status"), "done", "verify stage done");
  c.eq(await page.locator('[data-rank][data-cited="true"]').count(), 0, "nothing cited");
  c.eq(await page.locator('[data-banner="flagged"]').count(), 0, "no flag");
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, "refusal.png") });
  c.eq(errors, [], "no page errors");
  await close();
  return c;
}

async function checkUnknown(browser) {
  const c = new Check("unknown question");
  const { page, errors, close } = await newPage(browser);
  await page.getByRole("textbox", { name: "Question" }).fill("Who won the company chess tournament?");
  await page.getByRole("textbox", { name: "Question" }).press("Enter");
  const banner = page.locator('[data-banner="failure"]');
  await banner.waitFor({ timeout: 10000 });
  c.ok((await banner.innerText()).includes("Only the demo questions work offline"), "offline message");
  c.eq(await page.locator('[data-stage="route"]').getAttribute("data-status"), "pending", "no stage ran");
  c.eq(errors, [], "no page errors");
  await close();
  return c;
}

const questions = readJson("questions.json");
if (readdirSync(path.join(FIXTURES, "ask")).length !== questions.length) throw new Error("fixture count mismatch");
if (SHOTS) mkdirSync(SHOTS, { recursive: true });

const browser = await chromium.launch(
  process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {},
);
const started = Date.now();
const results = await Promise.all([
  ...questions.map((q) => checkRecorded(browser, q)),
  checkRefusal(browser),
  checkUnknown(browser),
]);
await browser.close();

let failed = 0;
for (const r of results) {
  console.log(`${r.failures.length ? "FAIL" : "ok  "} ${r.name}: ${r.passed} checks passed${r.failures.length ? `, ${r.failures.length} failed` : ""}`);
  for (const f of r.failures) console.log(`       - ${f}`);
  failed += r.failures.length;
}
console.log(`\nspeed ${SPEED}x, ${results.length} scenarios, ${((Date.now() - started) / 1000).toFixed(1)} s, ${failed} failures`);
process.exit(failed ? 1 : 0);
