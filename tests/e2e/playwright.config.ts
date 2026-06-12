import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:7987";

export default defineConfig({
  testDir: ".",
  testMatch: ["transcription.spec.ts", "real_audio.spec.ts"],
  globalSetup: "./global-setup.ts",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  timeout: 45 * 60 * 1000,
  expect: { timeout: 30_000 },
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    launchOptions: {
      args: ["--use-gl=egl", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
    },
    ...devices["Desktop Chrome"],
  },
  projects: [{
    name: "chromium",
    use: {
      ...devices["Desktop Chrome"],
      launchOptions: {
        args: ["--use-gl=egl", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
      },
    },
  }],
});
