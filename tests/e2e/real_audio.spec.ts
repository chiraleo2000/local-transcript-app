import { expect, test, type Page } from "@playwright/test";

import fs from "fs";
import path from "path";

const REPO_ROOT = path.resolve(__dirname, "../..");
const SHORT_AUDIO =
  process.env.E2E_SHORT_AUDIO ?? path.join(REPO_ROOT, "tests/Recording 1274.wav");
const LONG_AUDIO =
  process.env.E2E_LONG_AUDIO ?? path.join(REPO_ROOT, "tests/sudar-0001.m4a");

const MODEL_READY_MS = Number(process.env.E2E_MODEL_READY_MS ?? 20 * 60 * 1000);
const SHORT_TRANSCRIBE_MS = Number(process.env.E2E_SHORT_TRANSCRIBE_MS ?? 15 * 60 * 1000);
const LONG_TRANSCRIBE_MS = Number(process.env.E2E_LONG_TRANSCRIBE_MS ?? 4 * 60 * 60 * 1000);
const LONG_UI_STABLE_MS = Number(process.env.E2E_LONG_UI_STABLE_MS ?? 2 * 60 * 1000);

const SPEAKER_TS_RE =
  /^\[\d{2}:\d{2}:\d{2} → \d{2}:\d{2}:\d{2}\] \[SPEAKER_\d+\]:/;

async function assertAppReachable(page: Page): Promise<void> {
  const response = await page.goto("/", { waitUntil: "domcontentloaded" });
  expect(response?.ok(), "Gradio UI HTTP response should be 2xx").toBeTruthy();
  await expect(page.getByRole("heading", { name: "Local Transcript App" })).toBeVisible({
    timeout: 60_000,
  });
}

async function waitForModelsReady(page: Page): Promise<void> {
  const transcribe = page.locator("#transcribe-btn");
  await expect(transcribe, "Transcribe button should become enabled after model preload").toBeEnabled({
    timeout: MODEL_READY_MS,
  });
}

async function uploadAudio(page: Page, audioPath: string): Promise<void> {
  expect(fs.existsSync(audioPath), `audio missing: ${audioPath}`).toBeTruthy();
  const fileInput = page.locator("#media-input input[type='file']");
  await expect(fileInput).toHaveCount(1);
  await fileInput.setInputFiles(audioPath);
  await expect(page.getByText(path.basename(audioPath), { exact: false })).toBeVisible({
    timeout: 120_000,
  });
}

async function ensureDiarizationOn(page: Page): Promise<void> {
  const diarize = page.locator("#diarization-checkbox input[type='checkbox']");
  if (!(await diarize.isChecked())) {
    await diarize.check();
  }
}

async function assertPreviewDisabledMessage(page: Page): Promise<void> {
  await expect(page.locator("body")).toContainText(/Browser preview disabled/i, { timeout: 120_000 });
}

async function clickTranscribe(page: Page): Promise<void> {
  const transcribe = page.locator("#transcribe-btn");
  await expect(transcribe).toBeEnabled();
  await transcribe.click();
}

function statusBanner(page: Page) {
  return page.locator("#live-status .live-status");
}

function elapsedTimer(page: Page) {
  return page.locator("#elapsed-timer");
}

async function assertPageAlive(page: Page): Promise<void> {
  expect(page.isClosed(), "browser tab should remain open").toBeFalsy();
  expect(page.context().pages().length).toBe(1);
  const title = await page.title();
  expect(title).not.toMatch(/Aw, Snap|Out of Memory/i);
}

async function waitForJobStarted(page: Page, timeoutMs: number): Promise<void> {
  const status = statusBanner(page);
  await expect(status).toHaveClass(/running|done/, { timeout: timeoutMs });
}

async function waitForTranscriptionDone(page: Page, timeoutMs: number): Promise<void> {
  const status = statusBanner(page);
  await expect(status).toBeVisible();
  await expect(status).toHaveClass(/done/, { timeout: timeoutMs });
  await expect(status).not.toHaveClass(/error/);
}

async function assertStopwatchTicks(page: Page, stableMs: number): Promise<void> {
  const timer = elapsedTimer(page);
  await expect(timer).toBeVisible({ timeout: 60_000 });
  const first = (await timer.textContent()) ?? "";
  expect(first.length).toBeGreaterThan(0);
  await page.waitForTimeout(Math.min(stableMs, 5000));
  const second = (await timer.textContent()) ?? "";
  expect(second).not.toBe(first);
}

async function readTranscript(page: Page): Promise<string> {
  const tab = page.getByRole("tab", { name: "Output" });
  if (await tab.isVisible()) {
    await tab.click();
  }
  const transcript = page.locator("#output-transcript textarea");
  await expect(transcript).toBeVisible();
  return (await transcript.inputValue()).trim();
}

function assertDiarizationTimestamps(text: string): void {
  expect(text.length, "transcript should not be empty").toBeGreaterThan(0);
  expect(text.startsWith("ERROR"), "transcript should not be an error message").toBeFalsy();
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  const speakerLines = lines.filter((line) => line.includes("[SPEAKER_"));
  expect(speakerLines.length, "expected speaker-labelled lines").toBeGreaterThan(0);
  for (const line of speakerLines) {
    expect(line, `missing timestamp prefix: ${line.slice(0, 80)}`).toMatch(SPEAKER_TS_RE);
    expect(line).not.toMatch(/^\[SPEAKER_/);
  }
}

test.beforeAll(() => {
  if (!fs.existsSync(SHORT_AUDIO)) {
    throw new Error(`Missing short audio fixture: ${SHORT_AUDIO}`);
  }
});

test.describe("Real audio — Local Transcript App", () => {
  test("Recording 1274.wav: full happy path with diarization timestamps", async ({ page }) => {
    test.setTimeout(SHORT_TRANSCRIBE_MS + MODEL_READY_MS);

    await assertAppReachable(page);
    await waitForModelsReady(page);
    await assertPageAlive(page);

    await uploadAudio(page, SHORT_AUDIO);
    await assertPageAlive(page);
    await ensureDiarizationOn(page);
    await clickTranscribe(page);

    await waitForJobStarted(page, 60_000);
    await assertStopwatchTicks(page, 3000);
    await waitForTranscriptionDone(page, SHORT_TRANSCRIBE_MS);

    const transcript = await readTranscript(page);
    assertDiarizationTimestamps(transcript);

    await expect(page.getByRole("button", { name: /Download/i }).first()).toBeVisible();
    await expect(page.getByRole("tab", { name: "Output" })).toBeVisible();
    await assertPageAlive(page);
  });

  test("sudar-0001.m4a: long job OOM-safe UI and back-to-back smoke", async ({ page }) => {
    test.skip(!fs.existsSync(LONG_AUDIO), `long audio missing: ${LONG_AUDIO}`);
    test.setTimeout(LONG_TRANSCRIBE_MS + MODEL_READY_MS + SHORT_TRANSCRIBE_MS);

    await assertAppReachable(page);
    await waitForModelsReady(page);

    await uploadAudio(page, LONG_AUDIO);
    await assertPageAlive(page);
    await assertPreviewDisabledMessage(page);
    await ensureDiarizationOn(page);

    await clickTranscribe(page);
    await waitForJobStarted(page, 120_000);
    await assertStopwatchTicks(page, LONG_UI_STABLE_MS);
    await assertPageAlive(page);

    await waitForTranscriptionDone(page, LONG_TRANSCRIBE_MS);

    const transcript = await readTranscript(page);
    assertDiarizationTimestamps(transcript);
    expect(
      transcript.includes("Displaying last") ||
        transcript.includes("characters omitted") ||
        transcript.length <= 120_000,
    ).toBeTruthy();

    await uploadAudio(page, SHORT_AUDIO);
    await ensureDiarizationOn(page);
    await clickTranscribe(page);
    await waitForTranscriptionDone(page, SHORT_TRANSCRIBE_MS);
    const second = await readTranscript(page);
    assertDiarizationTimestamps(second);
    await assertPageAlive(page);
  });
});
