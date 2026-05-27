import { expect, test, type Page, type APIRequestContext } from "@playwright/test";

import fs from "fs";

import path from "path";



const FIXTURES_DIR = path.join(__dirname, "fixtures");

const SMALL_AUDIO = process.env.E2E_SMALL_AUDIO ?? path.join(FIXTURES_DIR, "small.wav");

const LARGE_AUDIO = process.env.E2E_LARGE_AUDIO ?? path.join(FIXTURES_DIR, "large.wav");



const MODEL_READY_MS = Number(process.env.E2E_MODEL_READY_MS ?? 20 * 60 * 1000);

const SMALL_TRANSCRIBE_MS = Number(process.env.E2E_SMALL_TRANSCRIBE_MS ?? 15 * 60 * 1000);

const LARGE_TRANSCRIBE_MS = Number(process.env.E2E_LARGE_TRANSCRIBE_MS ?? 35 * 60 * 1000);



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



async function configureFastPath(page: Page): Promise<void> {

  const enhance = page.locator("#enhance-checkbox input[type='checkbox']");

  if (await enhance.isChecked()) {

    await enhance.uncheck();

  }

  const diarize = page.locator("#diarization-checkbox input[type='checkbox']");

  if (await diarize.isChecked()) {

    await diarize.uncheck();

  }

}



async function uploadAudio(page: Page, audioPath: string): Promise<void> {

  expect(fs.existsSync(audioPath), `fixture missing: ${audioPath}`).toBeTruthy();

  const fileInput = page.locator("#media-input input[type='file']");

  await expect(fileInput).toHaveCount(1);

  await fileInput.setInputFiles(audioPath);

  await expect(page.getByText(path.basename(audioPath), { exact: false })).toBeVisible({

    timeout: 120_000,

  });

}



async function clickTranscribe(page: Page): Promise<void> {

  const transcribe = page.locator("#transcribe-btn");

  await expect(transcribe).toBeEnabled();

  await transcribe.click();

}



async function waitForJobStarted(page: Page, timeoutMs: number): Promise<void> {

  const status = statusBanner(page);

  await expect(status).toHaveClass(/running|done/, { timeout: timeoutMs });

}



function statusBanner(page: Page) {

  return page.locator("#live-status .live-status");

}



function progressPanel(page: Page) {

  return page.locator("#job-progress .job-progress-panel");

}



function progressFill(page: Page) {

  return page.locator("#job-progress .progress-fill");

}



async function readProgressApi(request: APIRequestContext): Promise<{

  active: boolean;

  percent: number;

  elapsed_s: number;

  remaining_s: number | null;

  phase: string;

}> {

  const response = await request.get("/job/progress");

  expect(response.ok(), "/job/progress should return 2xx").toBeTruthy();

  return response.json();

}



async function assertProgressDuringJob(page: Page): Promise<void> {

  const panel = progressPanel(page);

  await expect(panel).toBeVisible({ timeout: 60_000 });



  const elapsed = page.locator("#job-progress .progress-elapsed strong");

  const remaining = page.locator("#job-progress .progress-remaining strong");

  await expect(elapsed).toBeVisible();

  await expect(remaining).toBeVisible();



  const fill = progressFill(page);

  await expect(fill).toBeVisible();



  let firstPct = -1;

  let lastPct = -1;

  let lastElapsed = -1;



  for (let i = 0; i < 20; i += 1) {

    const api = await readProgressApi(page.request);

    const attr = await fill.getAttribute("data-progress");

    const pct = attr ? Number.parseFloat(attr) : api.percent;

    const elapsedVal = api.elapsed_s;



    if (firstPct < 0 && pct > 0) {

      firstPct = pct;

    }

    lastPct = Math.max(lastPct, pct);

    if (elapsedVal >= lastElapsed) {

      lastElapsed = elapsedVal;

    }



    if (api.phase === "done" || pct >= 100) {

      break;

    }

    await page.waitForTimeout(400);

  }



  expect(firstPct, "progress bar should move above 0% during job").toBeGreaterThan(0);

  expect(lastElapsed, "elapsed timer should increment").toBeGreaterThanOrEqual(0);

  expect(lastPct, "progress should advance during job").toBeGreaterThan(0);

}



async function waitForTranscriptionDone(page: Page, timeoutMs: number): Promise<void> {

  const status = statusBanner(page);

  await expect(status).toBeVisible();

  await expect(status).toHaveClass(/done/, { timeout: timeoutMs });

  await expect(status).not.toHaveClass(/error/);



  const panel = progressPanel(page);

  await expect(panel).toHaveAttribute("data-phase", "done", { timeout: 30_000 });

  await expect(progressFill(page)).toHaveAttribute("data-progress", /100(\.0)?/);

}



async function readTranscript(page: Page): Promise<string> {

  const tab = page.getByRole("tab", { name: "Typhoon Whisper" });

  if (await tab.isVisible()) {

    await tab.click();

  }

  const transcript = page.locator("#typhoon-transcript textarea");

  await expect(transcript).toBeVisible();

  return (await transcript.inputValue()).trim();

}



function assertTranscriptValid(text: string): void {

  expect(text.length, "transcript should not be empty").toBeGreaterThan(0);

  const lower = text.toLowerCase();

  expect(lower, "transcript should not be a placeholder").not.toMatch(

    /^\((no speech detected|failed|not selected|cancelled)\)/,

  );

  expect(text.startsWith("ERROR"), "transcript should not be an error message").toBeFalsy();

  expect(lower).not.toContain("cancelled");

}



test.beforeAll(() => {

  if (!fs.existsSync(SMALL_AUDIO) || !fs.existsSync(LARGE_AUDIO)) {

    throw new Error(

      `Missing fixtures. Run: cd tests/e2e && python generate_fixtures.py\n` +

        `  expected: ${SMALL_AUDIO}, ${LARGE_AUDIO}`,

    );

  }

});



test.describe("Local Transcript App — container UI", () => {

  test("UI loads and models become ready", async ({ page }) => {

    await assertAppReachable(page);

    await waitForModelsReady(page);

    await expect(statusBanner(page)).toHaveClass(/idle|done/);

    await expect(progressPanel(page)).toHaveAttribute("data-phase", "idle");



    const idleApi = await readProgressApi(page.request);

    expect(idleApi.phase).toBe("idle");

    expect(idleApi.percent).toBe(0);

  });



  test("small audio: upload, transcribe, progress bar and timer", async ({ page }) => {

    test.setTimeout(SMALL_TRANSCRIBE_MS + MODEL_READY_MS);



    await assertAppReachable(page);

    await waitForModelsReady(page);

    await configureFastPath(page);



    const started = Date.now();

    await uploadAudio(page, SMALL_AUDIO);

    await clickTranscribe(page);



    await waitForJobStarted(page, 60_000);

    await assertProgressDuringJob(page);

    await waitForTranscriptionDone(page, SMALL_TRANSCRIBE_MS);



    const transcript = await readTranscript(page);

    assertTranscriptValid(transcript);



    const elapsed = page.locator("#typhoon-elapsed textarea, #typhoon-elapsed input");

    await expect(elapsed.first()).toBeVisible();

    expect((await elapsed.first().inputValue()).length).toBeGreaterThan(0);



    const jobInfo = page.locator("#job-info textarea, #job-info input");

    expect(await jobInfo.first().inputValue()).toContain("Job ID:");



    console.log(`[small] transcript_chars=${transcript.length} wall_ms=${Date.now() - started}`);

  });



  test("large audio: upload, transcribe, sustained progress UI", async ({ page }) => {

    test.setTimeout(LARGE_TRANSCRIBE_MS + MODEL_READY_MS);



    await assertAppReachable(page);

    await waitForModelsReady(page);

    await configureFastPath(page);



    const largeStat = fs.statSync(LARGE_AUDIO);

    console.log(`[large] file_mb=${(largeStat.size / 1024 / 1024).toFixed(2)}`);



    const started = Date.now();

    await uploadAudio(page, LARGE_AUDIO);

    await clickTranscribe(page);



    await waitForJobStarted(page, 120_000);

    await assertProgressDuringJob(page);

    await waitForTranscriptionDone(page, LARGE_TRANSCRIBE_MS);



    const transcript = await readTranscript(page);

    assertTranscriptValid(transcript);



    await expect(page.getByRole("button", { name: "Transcribe" })).toBeEnabled();

    await expect(page.getByRole("heading", { name: "Local Transcript App" })).toBeVisible();



    const doneApi = await readProgressApi(page.request);

    expect(doneApi.phase).toBe("done");

    expect(doneApi.percent).toBe(100);



    console.log(`[large] transcript_chars=${transcript.length} wall_ms=${Date.now() - started}`);

  });

});


