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

import AxeBuilder from "@axe-core/playwright";
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
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
const fmt = (s) => `${s < 0.1 ? s.toFixed(2) : s.toFixed(1)} s`;
const frame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const sse = (body) => ({ status: 200, headers: { "Content-Type": "text/event-stream" }, body });
const LISTS = ["bm25", "dense", "bm25_routed", "dense_routed"];
const foundBy = (r) => {
  const b = r.bm25 !== null || r.bm25_routed !== null;
  const d = r.dense !== null || r.dense_routed !== null;
  return b && d ? "both" : b ? "bm25" : "dense";
};
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

/** Expands the pipeline steps of the latest turn (the setting then holds for later turns). */
async function expandSteps(page) {
  const toggle = page.locator('[data-role="steps-toggle"]').last();
  await toggle.waitFor({ timeout: 10000 });
  if ((await toggle.getAttribute("aria-expanded")) !== "true") await toggle.click();
}

const PROGRESS = {
  route: "Choosing which sources to search",
  rerank: "Reranking 100 candidates",
};

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
  // Collapsed, the steps are one line naming the step in progress; it changes as each step starts.
  const stepsLabel = page.locator('[data-role="steps-label"]');
  if (SPEED <= 2) {
    await page.waitForFunction((t) => document.querySelector('[data-role="steps-label"]')?.textContent === t, PROGRESS.rerank, { timeout: 30000 });
    c.ok(true, "collapsed steps line moves on to the running step");
    c.eq(await page.locator("[data-stage]").count(), 0, "collapsed: no step rows");
  }
  await expandSteps(page);
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
      c.eq(await page.locator(`[data-stage="${s}"]`).count(), 0, `${s} not shown before it starts`);
    }
    c.eq(await stepsLabel.innerText(), PROGRESS.rerank, "expanded: header still names the running step");
    c.ok(await page.locator('[data-role="fused-preview"]').isVisible(), "fused candidates shown while reranking");
    c.eq(await page.locator("[data-fused-rank]").count(), 10, "fused top 10 shown before rerank");

    // When rerank arrives the fused order holds, each card marked kept or dropped, then cards move to reranked slots.
    await page.locator('[data-phase="judging"]').waitFor({ timeout: 60000 });
    const top10 = D.fuse.candidates.slice(0, 10).map((x) => x.doc_id);
    const kept = top10.filter((id) => D.rerank.hits.some((h) => h.doc_id === id)).length;
    c.eq(await page.locator('[data-verdict="kept"]').count(), kept, "fused cards marked kept");
    c.eq(await page.locator('[data-verdict="dropped"]').count(), 10 - kept, "fused cards marked dropped");
    await page.locator('[data-phase="reranked"]').waitFor({ timeout: 5000 });
    c.ok(true, "reorder settles on the reranked order");
  }

  // The cost renders only after `done` (the live timer alone can pass through the recorded total).
  await page.locator('[data-role="cost"]').waitFor({ timeout: (lastT / SPEED) * 1000 + 20000 });
  c.ok((await page.locator('[data-role="total"]').innerText()).startsWith(fmt(D.done.seconds.total)), "total time");
  const wall = (Date.now() - t0) / 1000;
  c.ok(wall >= (lastT / SPEED) * 0.9 && wall <= lastT / SPEED + 4, `replay took ${wall.toFixed(1)} s for ${lastT} s at ${SPEED}x`);

  await page.locator('[data-role="hits"]').waitFor({ timeout: 5000 });

  // Timeline: one card per start.stages entry, all done with the event's own seconds; route shows its sources.
  const stages = await page.locator("[data-stage]").evaluateAll((els) => els.map((e) => e.dataset.stage));
  c.eq(stages, D.start.stages, "timeline stages");
  for (const s of D.start.stages) {
    const card = page.locator(`[data-stage="${s}"]`);
    c.eq(await card.getAttribute("data-status"), "done", `${s} status`);
    c.eq(await card.locator('[data-role="stage-time"]').innerText(), fmt(D[s].seconds), `${s} seconds`);
  }
  c.eq(await stepsLabel.innerText(), `Ran ${D.start.stages.length} steps`, "finished steps header");
  // The total sits on the header row, right-aligned, on one line.
  // Measured in one pass: the thread may still be scrolling to the bottom.
  const box = await page.evaluate(() => {
    const r = (sel) => document.querySelector(sel).getBoundingClientRect();
    const [head, total, row] = [r('[data-role="steps-toggle"]'), r('[data-role="total"]'), r('[data-role="steps"]')];
    return { dy: Math.abs(head.top + head.height / 2 - (total.top + total.height / 2)), h: total.height, dx: Math.abs(total.right - row.right) };
  });
  c.ok(box.dy < 2 && box.h < 24, `total on the steps header row, one line (${JSON.stringify(box)})`);
  c.ok(box.dx < 2, "total right-aligned in the answer column");
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

  // Retrieval tab (R): one row per fused candidate, ranks per list, RRF, reranked position, who found it.
  await page.keyboard.press("r");
  await page.locator('[data-role="retrieval"]').waitFor({ timeout: 3000 });
  c.eq(await page.locator("#tab-retrieval").getAttribute("aria-selected"), "true", "R opens the Retrieval tab");
  const shown = await page.locator("tr[data-fused-rank]").evaluateAll((trs) =>
    trs.map((tr) => {
      const td = [...tr.querySelectorAll("td")].map((x) => x.textContent.trim());
      return { doc: tr.dataset.doc, foundBy: tr.dataset.foundBy, ranks: td.slice(2, 6), rrf: td[6], top: td[7] };
    }),
  );
  const top = new Map(D.rerank.order.slice(0, 10).map((o, i) => [o.doc_id, String(i + 1)]));
  const expected = D.fuse.candidates.map((x) => ({
    doc: x.doc_id,
    foundBy: foundBy(x.ranks),
    ranks: LISTS.map((l) => (x.ranks[l] === null ? "" : String(x.ranks[l]))),
    rrf: x.rrf_score.toFixed(4),
    top: top.get(x.doc_id) ?? "",
  }));
  c.eq(shown, expected, "retrieval rows: ranks, RRF, top 10 and found-by for all 100");
  const counts = { both: 0, bm25: 0, dense: 0 };
  for (const e of expected) counts[e.foundBy] += 1;
  for (const k of Object.keys(counts)) {
    c.eq(await page.locator(`[data-count="${k}"]`).innerText(), String(counts[k]), `${k} count`);
  }
  if (SHOTS && q.question_id === "qst_0147") await page.screenshot({ path: path.join(SHOTS, "retrieval-qst_0147.png") });
  await page.keyboard.press("r");
  await page.locator('[data-role="hits"]').waitFor({ timeout: 3000 });

  // Gold answer toggle: the benchmark's gold answer and where each gold document landed.
  await page.getByRole("button", { name: "Show gold answer" }).click();
  c.eq(await page.locator('[data-role="gold-answer"]').innerText(), q.gold_answer.trim(), "gold answer text");
  const gold = await page.locator("[data-gold]").evaluateAll((els) => els.map((e) => [e.dataset.gold, e.dataset.status]));
  c.eq(
    gold,
    q.expected_doc_ids.map((id) => [id, D.rerank.hits.some((h) => h.doc_id === id) ? "top10" : "missed"]),
    "gold documents placed",
  );
  for (const id of q.expected_doc_ids) {
    const hit = D.rerank.hits.find((h) => h.doc_id === id);
    const text = await page.locator(`[data-gold="${id}"]`).innerText();
    if (hit) c.ok(text.includes(String(hit.rank)) && text.includes(hit.title), `gold ${id} rank ${hit.rank}`);
    else {
      const i = D.fuse.candidates.findIndex((x) => x.doc_id === id);
      c.ok(text.includes(i === -1 ? "not among the 100" : `fused #${i + 1} of 100`), `gold ${id} missed detail`);
    }
  }
  if (SHOTS && q.question_id === "qst_0037") {
    await page.locator('[data-role="gold-panel"]').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(SHOTS, "gold-qst_0037.png") });
  }
  await page.getByRole("button", { name: "Hide gold answer" }).click();

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
  const full = await content.evaluate((e) => e.textContent);
  c.eq(full.length, doc.content.length, "drawer shows the full content");
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
  await expandSteps(page);
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
  c.eq(await page.locator('[data-role="steps"]').count(), 0, "no steps shown when no stage ran");
  c.eq(errors, [], "no page errors");
  await close();
  return c;
}

async function waitDone(page, timeoutMs) {
  await page.locator('[data-role="cost"]').last().waitFor({ timeout: timeoutMs });
}
const runMs = (id) => (readRun(id).at(-1).t / SPEED) * 1000 + 20000;

async function checkKeyboard(browser, questions) {
  const c = new Check("keyboard");
  const { page, errors, close } = await newPage(browser);
  await page.keyboard.press("/");
  c.eq(await page.evaluate(() => document.activeElement?.getAttribute("aria-label")), "Question", "/ focuses the input");
  await page.keyboard.press("Escape");
  await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());
  await page.keyboard.press("1");
  await page.locator("[data-turn]").first().waitFor({ timeout: 3000 });
  c.ok((await page.locator("[data-turn]").first().innerText()).includes(questions[0].question), "1 asks the first demo question");
  await waitDone(page, runMs(questions[0].question_id));
  await page.keyboard.press("R");
  c.eq(await page.locator('#tab-retrieval').getAttribute("aria-selected"), "true", "R toggles to Retrieval");
  await page.keyboard.press("r");
  c.eq(await page.locator('#tab-evidence').getAttribute("aria-selected"), "true", "R toggles back to Evidence");
  // Typing in the input does not trigger shortcuts.
  await page.keyboard.press("/");
  await page.keyboard.type("2 r");
  c.eq(await page.locator("[data-turn]").count(), 1, "digits typed in the input do not ask a question");
  c.eq(await page.locator('#tab-evidence').getAttribute("aria-selected"), "true", "r typed in the input does not switch tabs");
  // Visible focus ring on keyboard focus.
  await page.keyboard.press("Escape");
  await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur());
  await page.keyboard.press("Tab");
  const outline = await page.evaluate(() => getComputedStyle(document.activeElement).outlineStyle);
  c.ok(outline !== "none", `focus ring visible (outline ${outline})`);
  c.eq(errors, [], "no page errors");
  await close();
  return c;
}

/** Each failure shows a clear message and a Retry that recovers once the backend is fine again. */
async function checkErrors(browser, q) {
  const c = new Check("error states");
  const D = Object.fromEntries(readRun(q.question_id).map((e) => [e.event, e.data]));
  const ready = readJson("health.json");
  const cases = [
    ["busy", "**/mock-api/health", (route) => route.fulfill({ json: { ...ready, busy: true } }), "Another question is running"],
    ["error", "**/mock-api/ask?*", (route) => route.fulfill(sse(frame("start", D.start) + frame("error", { message: "RuntimeError: CUDA out of memory" }))), "CUDA out of memory"],
    ["disconnected", "**/mock-api/ask?*", (route) => route.fulfill(sse(frame("start", D.start) + frame("route", D.route))), "The connection dropped"],
  ];
  for (const [kind, url, handler, text] of cases) {
    const { page, errors, close } = await newPage(browser);
    await page.route(url, handler);
    await page.locator(`[data-question="${q.question_id}"]`).click();
    const banner = page.locator(`[data-failure="${kind}"]`);
    await banner.waitFor({ timeout: 10000 });
    c.ok((await banner.innerText()).includes(text), `${kind}: message`);
    await page.unroute(url);
    await banner.getByRole("button", { name: /Retry/ }).click();
    await waitDone(page, runMs(q.question_id));
    c.eq(await page.locator("[data-turn]").count(), 1, `${kind}: retry replaces the failed turn and finishes`);
    c.eq(errors.filter((e) => !e.includes("Failed to load resource")), [], `${kind}: no page errors`);
    await close();
  }

  // Loading: /health says loading for the first few polls, then ready; the question reruns on its own.
  const { page, errors, close } = await newPage(browser);
  let calls = 0;
  await page.route("**/mock-api/health", (route) => {
    calls += 1;
    return route.fulfill({ json: calls <= 3 ? { ...ready, status: "loading" } : ready });
  });
  await page.reload();
  await page.locator('[data-banner="health"]').waitFor({ timeout: 5000 });
  c.ok((await page.locator('[data-banner="health"]').innerText()).includes("Loading indexes"), "loading: banner on start");
  await page.locator(`[data-question="${q.question_id}"]`).click();
  await page.locator('[data-failure="loading"]').waitFor({ timeout: 5000 });
  c.ok((await page.locator('[data-failure="loading"]').innerText()).includes("Loading indexes"), "loading: message");
  await waitDone(page, runMs(q.question_id) + 10000);
  c.ok(calls >= 4, `loading: polled /health (${calls} calls) and reran when ready`);
  c.eq(await page.locator('[data-banner="health"]').count(), 0, "loading: banner clears when ready");
  c.eq(errors, [], "loading: no page errors");
  await close();
  return c;
}

async function checkRecordMode(browser, questions) {
  const c = new Check("record mode");
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await context.newPage();
  await page.goto(`${BASE}/?record=1`);
  await page.locator("[data-question]").first().waitFor();
  const visibleDev = await page.locator(".dev-only").evaluateAll((els) => els.filter((e) => e.offsetParent !== null).length);
  c.eq(visibleDev, 0, "dev-only UI hidden");
  c.eq(await page.evaluate(() => getComputedStyle(document.body).zoom), "1.125", "type one step larger");
  const box = await page.locator('textarea[aria-label="Question"]').boundingBox();
  c.ok(box && box.y + box.height <= VIEWPORT.height, "input still on screen");
  await page.mouse.move(400, 400);
  c.eq(await page.evaluate(() => "idle" in document.documentElement.dataset), false, "cursor visible after moving");
  await page.waitForTimeout(2300);
  c.eq(await page.evaluate(() => "idle" in document.documentElement.dataset), true, "cursor hidden after 2 s idle");
  c.eq(await page.evaluate(() => getComputedStyle(document.body).cursor), "none", "cursor: none applied");
  await page.mouse.move(500, 500);
  c.eq(await page.evaluate(() => "idle" in document.documentElement.dataset), false, "cursor back on movement");

  // No page scroll in either direction: real Chrome shows scrollbars that headless hides, and a vertical one would
  // also push the layout sideways. Checked with an answered question (steps expanded), on Evidence and Retrieval.
  const q = questions.reduce((a, b) => (readRun(a.question_id).at(-1).t <= readRun(b.question_id).at(-1).t ? a : b));
  await page.locator(`[data-question="${q.question_id}"]`).click();
  await expandSteps(page);
  await waitDone(page, runMs(q.question_id));
  await page.locator('[data-role="hits"]').waitFor();
  const fits = () =>
    page.evaluate(() => {
      const d = document.documentElement;
      return { sh: d.scrollHeight, ch: d.clientHeight, sw: d.scrollWidth, cw: d.clientWidth };
    });
  for (const view of ["evidence", "retrieval"]) {
    if (view === "retrieval") await page.keyboard.press("r");
    for (const size of [VIEWPORT, { width: 1600, height: 900 }]) {
      await page.setViewportSize(size);
      await page.waitForTimeout(400);
      const m = await fits();
      const at = `${size.width}x${size.height} ${view}`;
      c.ok(m.sh <= m.ch, `${at}: scrollHeight ${m.sh} <= clientHeight ${m.ch}`);
      c.ok(m.sw <= m.cw, `${at}: scrollWidth ${m.sw} <= clientWidth ${m.cw}`);
    }
  }
  await context.close();
  return c;
}

async function checkBenchmark(browser) {
  const c = new Check("benchmark page");
  const { page, errors, close } = await newPage(browser);
  await page.goto(`${BASE}/benchmark`);
  const readme = readFileSync(path.join(ROOT, "..", "README.md"), "utf8").split(/\r?\n/);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((x) => x.trim());
  const tables = await page.locator("[data-table]").evaluateAll((els) =>
    els.map((el) => ({
      id: el.dataset.table,
      head: [...el.querySelectorAll("th")].map((th) => th.textContent.trim()),
      rows: [...el.querySelectorAll("tbody tr")].map((tr) => [...tr.querySelectorAll("[data-cell]")].map((x) => x.textContent)),
    })),
  );
  c.eq(tables.map((t) => t.id), ["heldOut", "recallByType", "devRetrieval", "devDense", "devEndToEnd", "latency", "confidenceFlag"], "all 7 README tables");
  for (const t of tables) {
    const at = readme.findIndex((l) => l.startsWith("|") && cells(l).join("\u0000") === t.head.join("\u0000"));
    const rows = [];
    for (let i = at + 2; readme[i]?.startsWith("|"); i++) rows.push(cells(readme[i]));
    c.ok(at > 0, `${t.id}: header found in README`);
    c.eq(t.rows, rows, `${t.id}: every rendered cell equals README`);
  }
  c.eq(await page.locator("[data-headline]").count(), 4, "headline rows marked");
  c.eq(errors, [], "no page errors");
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, "benchmark.png"), fullPage: true });
  await close();
  return c;
}

/** WCAG 2 A/AA rules (contrast included) on a finished, flagged answer with the gold panel open, in both themes. */
async function checkA11y(browser, q) {
  const c = new Check("accessibility");
  for (const theme of ["light", "dark"]) {
    const { page, close } = await newPage(browser);
    await page.goto(`${BASE}/?theme=${theme}`);
    await page.locator(`[data-question="${q.question_id}"]`).click();
    await expandSteps(page);
    await waitDone(page, runMs(q.question_id));
    await page.locator('[data-role="hits"]').waitFor();
    await page.getByRole("button", { name: "Show gold answer" }).click();
    await page.waitForTimeout(600);
    for (const view of ["evidence", "retrieval"]) {
      if (view === "retrieval") await page.keyboard.press("r");
      await page.waitForTimeout(400);
      const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
      const v = result.violations.map((x) => `${x.id} (${x.nodes.length}): ${x.nodes[0]?.target.join(" ")}`);
      c.eq(v, [], `${theme} ${view}: no WCAG A/AA violations`);
    }
    if (SHOTS && theme === "dark") await page.screenshot({ path: path.join(SHOTS, "dark-retrieval.png") });
    await page.goto(`${BASE}/benchmark?theme=${theme}`);
    const bench = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
    c.eq(bench.violations.map((x) => `${x.id} (${x.nodes.length}): ${x.nodes[0]?.target.join(" ")}`), [], `${theme} benchmark: no WCAG A/AA violations`);
    await close();
  }
  return c;
}

const questions = readJson("questions.json");
if (readdirSync(path.join(FIXTURES, "ask")).length !== questions.length) throw new Error("fixture count mismatch");
if (SHOTS) mkdirSync(SHOTS, { recursive: true });

const browser = await chromium.launch(
  process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {},
);
const started = Date.now();
/** A scenario that throws (a timeout, a missing element) counts as one failure instead of aborting the run. */
const safe = (name, run) =>
  run().catch((e) => {
    const c = new Check(name);
    c.ok(false, `threw: ${String(e).split("\n")[0]}`);
    return c;
  });
const pick = (id) => questions.find((x) => x.question_id === id) ?? questions[0];
const results = await Promise.all([
  ...questions.map((q) => safe(q.question_id, () => checkRecorded(browser, q))),
  safe("synthetic refusal", () => checkRefusal(browser)),
  safe("unknown question", () => checkUnknown(browser)),
  safe("keyboard", () => checkKeyboard(browser, questions)),
  safe("error states", () => checkErrors(browser, pick("qst_0459"))),
  safe("record mode", () => checkRecordMode(browser, questions)),
  safe("benchmark page", () => checkBenchmark(browser)),
  safe("accessibility", () => checkA11y(browser, pick("qst_0037"))),
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
