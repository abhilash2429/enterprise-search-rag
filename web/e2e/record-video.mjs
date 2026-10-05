// Records one full mock-mode run in recording mode (?record=1) as a 1920x1080 Playwright video, driven by keyboard
// shortcuts the way a presenter would: pick a demo question, watch the pipeline, open a citation, show Retrieval.
//
//   NEXT_PUBLIC_MOCK_SPEED=1 npm run build && NEXT_PUBLIC_MOCK_SPEED=1 npm start   # in one shell
//   E2E_VIDEO=out npm run e2e:video                                               # writes out/<question>.webm
//
// Env: E2E_BASE_URL (default http://localhost:3000), E2E_VIDEO (output dir, default e2e-video), E2E_QUESTION (default
// qst_0421), E2E_THEME (light or dark, default light).

import { mkdirSync, readFileSync, renameSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const BASE = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const OUT = path.resolve(process.env.E2E_VIDEO ?? "e2e-video");
const ID = process.env.E2E_QUESTION ?? "qst_0421";
const THEME = process.env.E2E_THEME === "dark" ? "dark" : "light";
const SIZE = { width: 1920, height: 1080 };

const questions = JSON.parse(readFileSync(path.join(ROOT, "fixtures", "questions.json"), "utf8"));
const index = questions.findIndex((q) => q.question_id === ID);
if (index < 0 || index > 7) throw new Error(`${ID} is not one of the first 8 demo questions`);
const lastT = JSON.parse(readFileSync(path.join(ROOT, "fixtures", "ask", `${ID}.jsonl`), "utf8").trim().split("\n").at(-1)).t;

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch(
  process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {},
);
const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: OUT, size: SIZE } });
const page = await context.newPage();
await page.goto(`${BASE}/?record=1&theme=${THEME}`);
await page.locator("[data-question]").first().waitFor();
await page.waitForTimeout(1500);

const started = Date.now();
await page.keyboard.press(String(index + 1));
await page.locator('[data-role="cost"]').waitFor({ timeout: lastT * 1000 + 30000 });
console.log(`run finished in ${((Date.now() - started) / 1000).toFixed(1)} s (recorded total ${lastT} s)`);
await page.waitForTimeout(2500);

// Click a citation: the evidence rail highlights and scrolls to it; then open the document and close it with Esc.
const chip = page.locator('[data-role="answer"] [data-cite]').last();
await page.mouse.move(700, 500, { steps: 12 });
await chip.hover();
await chip.click();
await page.waitForTimeout(1800);
const n = await chip.getAttribute("data-cite");
await page.locator(`[data-rank="${n}"] button`).click();
await page.locator('[data-role="document-content"]').waitFor();
await page.waitForTimeout(2500);
await page.keyboard.press("Escape");
await page.waitForTimeout(1000);

// R: the Retrieval tab with the four lists, RRF and the reranked top 10.
await page.keyboard.press("r");
await page.waitForTimeout(3500);
await page.keyboard.press("r");
await page.waitForTimeout(1200);

const video = page.video();
await context.close();
await browser.close();
const file = path.join(OUT, `${ID}${THEME === "dark" ? "-dark" : ""}.webm`);
renameSync(await video.path(), file);
console.log(`video: ${file}`);
