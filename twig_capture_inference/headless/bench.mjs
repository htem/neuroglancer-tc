// Drive the Twig Capture in-browser bench headlessly: click each requested
// button and wait for its run to finish, then print the page log.
import { chromium } from "playwright-core";
const url = process.env.BENCH_URL || "https://radagast.hms.harvard.edu/twigcapture_dev/bench.html";
const which = (process.argv[2] || "webgpu:fp16pad").split(",");
const browser = await chromium.launch({
  executablePath: process.env.HOME + "/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome",
  headless: true,
  args: ["--headless=new", "--enable-unsafe-webgpu", "--enable-features=Vulkan",
         "--use-vulkan=native", "--use-angle=vulkan", "--disable-vulkan-surface",
         "--ignore-gpu-blocklist", "--no-sandbox"],
});
const page = await browser.newPage();
page.on("console", m => { const t = m.text(); if (!t.startsWith("[bench]")) console.log("[console]", m.type(), t.slice(0, 400)); });
page.on("pageerror", e => console.log("[pageerror]", e.message));
await page.goto(url, { waitUntil: "load" });
console.log("navigator.gpu:", await page.evaluate(() => !!navigator.gpu));
console.log("adapter:", await page.evaluate(async () => {
  const a = navigator.gpu && await navigator.gpu.requestAdapter();
  return a ? JSON.stringify({ vendor: a.info?.vendor, arch: a.info?.architecture, desc: a.info?.description, fallback: a.info?.isFallbackAdapter }) : "none";
}));
for (const w of which) {
  const [ep, m] = w.split(":");
  const before = await page.$eval("#log", e => e.textContent.length);
  await page.click(`button[data-ep="${ep}"][data-m="${m}"]`);
  await page.waitForFunction(([b, stop]) => { const t = document.getElementById("log").textContent.slice(b);
    return new RegExp(stop).test(t); }, [before, process.env.STOP || "FAILED|batch 8:"], { timeout: 20 * 60e3, polling: 1000 });
  console.log((await page.$eval("#log", e => e.textContent)).slice(before));
}
await browser.close();
