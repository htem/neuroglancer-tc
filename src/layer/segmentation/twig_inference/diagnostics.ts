/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Bug reports for in-browser inference.
 *
 * Reviewers run this on their own machines (Mac, Windows, Linux; Apple,
 * NVIDIA, AMD, Intel GPUs, or none), so a failure we cannot reproduce is the
 * normal case. Everything needed to diagnose one is gathered here into a
 * plain-text block the reviewer copies and sends: what the machine is, what
 * the browser offers (WebGPU adapter, f16, limits, threads), what was run
 * (dataset, root, settings, backend) and what happened (the error with its
 * stack, and a short log of recent run events). No tokens, no URL hash.
 */

import { ORT_VERSION } from "#src/layer/segmentation/twig_inference/ort_version.js";

export interface LogEntry {
  t: string;
  level: "info" | "warn" | "error";
  msg: string;
}

const MAX_LOG = 200;
const log: LogEntry[] = [];

/** Record a run event (also echoed to the console). */
export function logEvent(level: LogEntry["level"], msg: string, err?: unknown) {
  const text = err === undefined ? msg : `${msg}: ${errorText(err)}`;
  log.push({ t: new Date().toISOString().slice(11, 23), level, msg: text });
  if (log.length > MAX_LOG) log.splice(0, log.length - MAX_LOG);
  const c =
    level === "error"
      ? console.error
      : level === "warn"
        ? console.warn
        : console.log;
  c(`Twig Selector: ${msg}`, ...(err === undefined ? [] : [err]));
}

export function recentLog(n = 60): LogEntry[] {
  return log.slice(-n);
}

/** Message plus stack (first lines), for any thrown value. */
export function errorText(e: unknown): string {
  if (e instanceof Error) {
    const stack = (e.stack ?? "").split("\n").slice(0, 8).join("\n");
    return stack.includes(e.message)
      ? stack
      : `${e.name}: ${e.message}\n${stack}`;
  }
  try {
    return typeof e === "string" ? e : JSON.stringify(e);
  } catch {
    return String(e);
  }
}

/** What the machine and browser offer, gathered fresh each time. */
export async function environmentInfo(): Promise<Record<string, unknown>> {
  const nav = navigator as any;
  const env: Record<string, unknown> = {
    userAgent: navigator.userAgent,
    platform: nav.userAgentData?.platform ?? navigator.platform,
    browser: nav.userAgentData?.brands
      ?.map((b: any) => `${b.brand} ${b.version}`)
      .join(", "),
    cpuThreads: navigator.hardwareConcurrency,
    deviceMemoryGB: nav.deviceMemory,
    crossOriginIsolated: self.crossOriginIsolated,
    ortVersion: ORT_VERSION,
    ortLoaded: (window as any).ort?.env?.versions?.web ?? !!(window as any).ort,
    screen: `${screen.width}x${screen.height} @${devicePixelRatio}`,
    page: location.origin + location.pathname,
  };
  try {
    const gpu = nav.gpu;
    if (!gpu) {
      env.webgpu = "not available (navigator.gpu missing)";
    } else {
      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        env.webgpu = "navigator.gpu present but no adapter";
      } else {
        const info =
          adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
        env.webgpu = {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
          shaderF16: adapter.features?.has?.("shader-f16"),
          maxBufferSize: adapter.limits?.maxBufferSize,
          maxStorageBufferBindingSize:
            adapter.limits?.maxStorageBufferBindingSize,
          isFallbackAdapter:
            adapter.isFallbackAdapter ?? info.isFallbackAdapter,
        };
      }
    }
  } catch (e) {
    env.webgpu = `error probing: ${errorText(e)}`;
  }
  return env;
}

/**
 * A plain-text report for pasting into a message or issue: a heading, the
 * error (if any), what was run, the environment and the recent log.
 */
export async function buildReport(opts: {
  title: string;
  error?: string;
  run?: Record<string, unknown>;
}): Promise<string> {
  const env = await environmentInfo();
  const lines: string[] = [];
  lines.push("```");
  lines.push(`Twig Selector report: ${opts.title}`);
  lines.push(`time: ${new Date().toISOString()}`);
  if (opts.error) {
    lines.push("", "error:", opts.error);
  }
  if (opts.run) {
    lines.push("", "run:", JSON.stringify(opts.run, null, 2));
  }
  lines.push("", "environment:", JSON.stringify(env, null, 2));
  const recent = recentLog();
  if (recent.length) {
    lines.push("", "recent log:");
    for (const e of recent)
      lines.push(`${e.t} ${e.level.toUpperCase()} ${e.msg}`);
  }
  lines.push("```");
  return lines.join("\n");
}

/**
 * Put text on the clipboard. Where the browser refuses (no permission, not a
 * secure context), show it in a selected text box instead so it can be copied
 * by hand -- the report must always be reachable.
 */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const box = document.createElement("div");
    box.style.cssText =
      "position:fixed;inset:10% 10%;z-index:100000;background:#222;color:#eee;" +
      "padding:8px;border:1px solid #888;display:flex;flex-direction:column;gap:6px";
    const label = document.createElement("div");
    label.textContent =
      "Copy this report (Ctrl/Cmd+C) and send it to the Twig Capture team:";
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "flex:1;font-family:monospace;font-size:11px";
    const close = document.createElement("button");
    close.textContent = "Close";
    close.addEventListener("click", () => box.remove());
    box.append(label, ta, close);
    document.body.appendChild(box);
    ta.focus();
    ta.select();
    return false;
  }
}
