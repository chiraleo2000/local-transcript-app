import { request } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://localhost:7896";

export default async function globalSetup(): Promise<void> {
  const client = await request.newContext();
  try {
    const response = await client.get(baseURL, { timeout: 15_000 });
    if (!response.ok()) {
      throw new Error(`E2E target returned HTTP ${response.status()} at ${baseURL}`);
    }
    console.log(`E2E global-setup: ${baseURL} is reachable (HTTP ${response.status()})`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Cannot reach ${baseURL}. Start the app first:\n` +
        `  docker compose -f docker-compose.gpu.yml up -d\n` +
        `Original error: ${message}`,
    );
  } finally {
    await client.dispose();
  }
}
