// Captures the README screenshots from a mock-mode run in recording mode (?record=1).
//
//   NEXT_PUBLIC_MOCK_SPEED=20 npm run dev       # in one shell
//   npm run e2e:screenshots                     # writes ../docs/images/*.png
//
// Env: E2E_BASE_URL (default http://localhost:3000), E2E_QUESTION (default qst_0421), PLAYWRIGHT_CHROMIUM_PATH.

import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const BASE = process.env.E2E_BASE_URL ?? "http://localhost:3000";
const OUT = path.join(ROOT, "..", "docs", "images");
const ID = process.env.E2E_QUESTION ?? "qst_0421";

const questions = JSON.parse(readFileSync(path.join(ROOT, "fixtures", "questions.json"), "utf8"));
const index = questions.findIndex((q) => q.question_id === ID);
if (index < 0 || index > 7) throw new Error(`${ID} is not one of the first 8 demo questions`);

mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch(
  process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {},
);
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
// The dev server's floating Next.js badge is not part of the app.
const shot = async (name) => {
  await page.addStyleTag({ content: "nextjs-portal { display: none !important; }" });
  await page.screenshot({ path: path.join(OUT, `${name}.png`) });
};

await page.goto(`${BASE}/?record=1&theme=light`);
await page.locator("[data-question]").first().waitFor();
await page.keyboard.press(String(index + 1));
await page.locator('[data-role="steps-toggle"]').click();
await page.locator('[data-role="cost"]').waitFor({ timeout: 120000 });
await page.waitForTimeout(1500);

// A citation chip highlights its document in the evidence rail.
await page.locator('[data-role="answer"] [data-cite]').first().click();
await page.waitForTimeout(1500);
await page.mouse.move(0, 0);
await shot("answer");

await page.keyboard.press("r");
await page.waitForTimeout(2000);
await shot("retrieval");

await page.goto(`${BASE}/benchmark?record=1&theme=light`);
await page.waitForLoadState("networkidle");
await page.waitForTimeout(1000);
await shot("benchmark");

await browser.close();
console.log(`screenshots: ${OUT}`);
