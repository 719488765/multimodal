// API base: empty = same-origin. Set VITE_API_BASE_URL for explicit backend URL.
const ENV_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const DEPLOY_MODE = String(import.meta.env.VITE_DEPLOY_MODE || "dev").toLowerCase();
const SERVER_MAX_MULTIPART_BYTES = 5 * 1024 * 1024;
/** Cursor/SSH 端口转发：单次 multipart 上限（音频+视频合计） */
export const TUNNEL_MAX_MULTIPART_BYTES = 384 * 1024;

const CHUNK_SIZE = 16384;
const CHUNK_RETRIES = 3;
const CHUNKS_PER_REQUEST = 5;
/** 超过此次数时前端应改用 JPEG，避免 Cursor 端口转发下大量请求失败 */
export const MAX_SAFE_UPLOAD_CHUNKS = 8;

export const CAPTURE_MAX_SEC_SERVER = 30;
/** Cloudflare/Cursor 隧道 multipart 约 384KB，约可容纳 10s 16kHz mono WAV + 小图 */
export const CAPTURE_MAX_SEC_TUNNEL = 10;

/** 采集上传最长秒数：服务器 30s；隧道 10s（受 384KB 限制） */
export function getCaptureMaxSec() {
  if (typeof window === "undefined") return CAPTURE_MAX_SEC_SERVER;
  if (isCloudflareQuickTunnel() || isCursorTunnelMode()) return CAPTURE_MAX_SEC_TUNNEL;
  if (isServerDeployMode()) return CAPTURE_MAX_SEC_SERVER;
  return CAPTURE_MAX_SEC_TUNNEL;
}

/** Cloudflare Quick Tunnel 对单次 HTTP 约 100s 上限；ASR+推理需控制在此以内 */
export function isCloudflareQuickTunnel() {
  if (typeof window === "undefined") return false;
  return window.location.hostname.endsWith(".trycloudflare.com");
}

export function estimateUploadChunks(blobSize) {
  if (!blobSize) return 0;
  return Math.max(1, Math.ceil(blobSize / CHUNK_SIZE));
}

/** Cursor 转发 127.0.0.1:8000：多次 POST 易断，仅允许小包单次 multipart */
export function isCursorTunnelMode() {
  if (typeof window === "undefined") return false;
  const { hostname, port } = window.location;
  const h = (hostname || "").toLowerCase();
  return (h === "127.0.0.1" || h === "localhost") && port === "8000";
}

/** 浏览器直连服务器（nginx HTTPS 或内网 IP），非 Cursor/Cloudflare 隧道 */
export function isServerDeployMode() {
  if (isCursorTunnelMode() || isCloudflareQuickTunnel()) return false;
  if (DEPLOY_MODE === "server") return true;
  if (typeof window === "undefined") return false;
  const { protocol, hostname } = window.location;
  return (
    protocol === "https:" &&
    hostname !== "localhost" &&
    hostname !== "127.0.0.1" &&
    !hostname.endsWith(".local")
  );
}

export function getUploadProfile() {
  if (isCloudflareQuickTunnel()) {
    return {
      mode: "cloudflare_tunnel",
      label: "Cloudflare 隧道（同源 multipart，禁止分块；建议≤10s）",
      maxMultipartBytes: TUNNEL_MAX_MULTIPART_BYTES,
    };
  }
  if (isCursorTunnelMode()) {
    return {
      mode: "cursor_tunnel",
      label: "Cursor 隧道（单次小包 multipart，禁止分块）",
      maxMultipartBytes: TUNNEL_MAX_MULTIPART_BYTES,
    };
  }
  if (isServerDeployMode()) {
    return {
      mode: "server",
      label: "服务器直连（整包 webm multipart，≤5MB）",
      maxMultipartBytes: SERVER_MAX_MULTIPART_BYTES,
    };
  }
  return {
    mode: "dev",
    label: "开发模式（小包或分块）",
    maxMultipartBytes: TUNNEL_MAX_MULTIPART_BYTES,
  };
}

async function uploadSingleChunk(sessionId, field, chunkIndex, totalChunks, slice) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("field", field);
  form.append("chunk_index", String(chunkIndex));
  form.append("total_chunks", String(totalChunks));
  form.append("chunk", slice, `${field}.part${chunkIndex}`);
  const resp = await requestForm("/api/v1/emotion/upload-chunk", form, 90000, true);
  if (!resp?.accepted) {
    throw new Error(resp?.error || "分块被拒绝");
  }
}

async function uploadChunkBatch(sessionId, field, startIndex, totalChunks, slices) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("field", field);
  form.append("start_index", String(startIndex));
  form.append("total_chunks", String(totalChunks));
  slices.forEach((slice, idx) => {
    form.append("chunks", slice, `${field}.part${startIndex + idx}`);
  });
  const resp = await requestForm("/api/v1/emotion/upload-chunks-batch", form, 120000, true);
  if (!resp?.accepted) {
    throw new Error(resp?.error || "批量分块被拒绝");
  }
}

let resolvedBase = ENV_BASE;
let resolvePromise = null;

function isSamePortBackend() {
  if (typeof window === "undefined") return false;
  return window.location.port === "8000";
}

/** 页面与 backend 同源（单端口 8000 / Cloudflare 隧道 / nginx 反代），API 应走相对路径 */
function isApiSameOrigin() {
  return isSamePortBackend() || isCloudflareQuickTunnel() || isServerDeployMode();
}

function directBackendCandidates() {
  if (typeof window === "undefined") return [];
  if (isApiSameOrigin()) {
    return [];
  }
  const host = window.location.hostname || "127.0.0.1";
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  const list = [];
  if (host === "127.0.0.1" || host === "localhost") {
    list.push(`${protocol}//${host}:8000`);
  }
  if (host !== "127.0.0.1") list.push("http://127.0.0.1:8000");
  if (host !== "localhost") list.push("http://localhost:8000");
  return [...new Set(list)];
}

async function probeHealth(baseUrl, timeoutMs = 4000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/api/v1/health`, { signal: controller.signal });
    if (!res.ok) return false;
    const data = await res.json();
    return Boolean(data?.ok);
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** Pick a working API base. Call once on app mount. */
export async function initApiBase() {
  if (ENV_BASE) {
    resolvedBase = ENV_BASE;
    return resolvedBase;
  }
  if (isApiSameOrigin()) {
    resolvedBase = "";
    return resolvedBase;
  }
  if (resolvePromise) return resolvePromise;

  resolvePromise = (async () => {
    if (typeof window !== "undefined" && (await probeHealth(window.location.origin))) {
      resolvedBase = "";
      return resolvedBase;
    }
    for (const direct of directBackendCandidates()) {
      if (await probeHealth(direct)) {
        resolvedBase = direct;
        return resolvedBase;
      }
    }
    if (await probeHealth("")) {
      resolvedBase = "";
      return resolvedBase;
    }
    resolvedBase = "";
    return resolvedBase;
  })();

  return resolvePromise;
}

export function getApiBase() {
  return resolvedBase;
}

export const apiBaseUrl = ENV_BASE;

export function getApiDisplayUrl() {
  if (isCloudflareQuickTunnel()) {
    return `${window.location.origin}（Cloudflare 隧道，同源 API）`;
  }
  if (isSamePortBackend()) {
    return `${window.location.origin}（单端口 8000，同源 API）`;
  }
  const base = resolvedBase || ENV_BASE;
  if (base) {
    return `${base}（直连后端）`;
  }
  if (typeof window !== "undefined" && window.location?.origin) {
    return `${window.location.origin}（Vite 代理 → :8000）`;
  }
  return "未配置";
}

function networkErrorHint(tried) {
  let extra;
  if (isCloudflareQuickTunnel()) {
    extra =
      "请通过 Cloudflare 隧道地址打开页面（不要用 :8000 端口）。若仍失败请刷新页面；录制请控制在约 10 秒内。";
  } else if (isCursorTunnelMode()) {
    extra =
      "Cursor 隧道对大文件/多次请求不稳定；已自动压缩上传。若仍失败请刷新重试，或使用 cloudflared/服务器直连 HTTPS。";
  } else if (isServerDeployMode()) {
    extra = "请确认 nginx 与 backend(8000) 已启动，且 client_max_body_size 足够。";
  } else if (isSamePortBackend() || resolvedBase === "") {
    extra = "请确认 backend 在 8000 运行，并在 Cursor Ports 面板转发 8000；刷新页面后重试。";
  } else {
    extra =
      "请确认 backend 在 8000 运行，并在 Cursor 中转发 8000 端口；" +
      "推荐直接打开 http://127.0.0.1:8000（单端口），或设置 VITE_API_BASE_URL=http://127.0.0.1:8000 后重启 npm run dev";
  }
  return `无法连接情绪推理 API（已尝试: ${tried.join(" → ")}）。${extra}`;
}

function parseResponseBody(text) {
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function formatHttpError(data, res) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `HTTP ${res.status} ${res.statusText}`;
}

function basesForRequest({ forLargeUpload = false } = {}) {
  if (isApiSameOrigin()) {
    return [""];
  }
  const list = [];
  if (typeof window !== "undefined") {
    const origin = window.location.origin;
    if (origin && !list.includes(origin)) {
      list.push("");
    }
  }
  const primary = resolvedBase || ENV_BASE;
  if (primary && !list.includes(primary)) list.push(primary);
  for (const d of directBackendCandidates()) {
    if (!list.includes(d)) list.push(d);
  }
  if (!forLargeUpload && !list.includes("")) {
    list.push("");
  }
  return list;
}

async function fetchWithFallback(path, init, { timeoutMs = 120000, forLargeUpload = false } = {}) {
  const bases = basesForRequest({ forLargeUpload });
  const tried = [];
  let lastErr = null;

  for (const base of bases) {
    const label =
      base ||
      (typeof window !== "undefined" && isApiSameOrigin()
        ? window.location.origin
        : "vite-proxy");
    tried.push(label);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${base}${path}`, {
        ...init,
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (base && base !== (resolvedBase || ENV_BASE)) {
        resolvedBase = base;
      }
      return res;
    } catch (e) {
      clearTimeout(timer);
      lastErr = e;
      if (e?.name === "AbortError") {
        throw new Error(`请求超时（>${Math.round(timeoutMs / 1000)}s）`);
      }
    }
  }

  if (lastErr instanceof TypeError && /fetch/i.test(String(lastErr.message))) {
    throw new Error(`${networkErrorHint(tried)}（${lastErr.message}）`);
  }
  throw lastErr || new Error(networkErrorHint(tried));
}

async function requestJson(path, { method = "GET", body, timeoutMs = 120000, forLargeUpload = false } = {}) {
  const res = await fetchWithFallback(
    path,
    {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    },
    { timeoutMs, forLargeUpload },
  );
  const text = await res.text();
  const data = parseResponseBody(text);
  if (!res.ok) {
    throw new Error(formatHttpError(data, res));
  }
  return data;
}

async function requestForm(path, formData, timeoutMs = 300000, forLargeUpload = false) {
  const res = await fetchWithFallback(
    path,
    { method: "POST", body: formData },
    { timeoutMs, forLargeUpload },
  );
  const text = await res.text();
  const data = parseResponseBody(text);
  if (!res.ok) {
    throw new Error(formatHttpError(data, res));
  }
  return data;
}

async function uploadBlobInChunks(sessionId, field, blob, onProgress) {
  const total = estimateUploadChunks(blob.size);
  let i = 0;
  while (i < total) {
    const batchSlices = [];
    const batchStart = i;
    while (batchSlices.length < CHUNKS_PER_REQUEST && i < total) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(blob.size, start + CHUNK_SIZE);
      batchSlices.push(blob.slice(start, end));
      i += 1;
    }
    let lastErr = null;
    for (let attempt = 1; attempt <= CHUNK_RETRIES; attempt += 1) {
      try {
        if (batchSlices.length === 1) {
          await uploadSingleChunk(sessionId, field, batchStart, total, batchSlices[0]);
        } else {
          await uploadChunkBatch(sessionId, field, batchStart, total, batchSlices);
        }
        lastErr = null;
        break;
      } catch (e) {
        lastErr = e;
        if (attempt < CHUNK_RETRIES) {
          await new Promise((r) => setTimeout(r, 500 * attempt));
        }
      }
    }
    if (lastErr) {
      const endIdx = batchStart + batchSlices.length;
      throw new Error(
        `${field} 分块 ${batchStart + 1}-${endIdx}/${total} 上传失败: ${lastErr.message}`,
      );
    }
    if (onProgress) {
      onProgress({ field, index: i, total, bytes: blob.size });
    }
    await new Promise((r) => setTimeout(r, 80));
  }
}

async function inferEmotionUploadMultipart({
  sessionId,
  text,
  audioBlob,
  videoBlob,
  videoName,
  mergedMeta,
  totalBytes,
  onUploadProgress,
  timeoutMs = 300000,
}) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("text", text || "");
  if (audioBlob) form.append("audio", audioBlob, audioBlob.name || "capture.wav");
  if (videoBlob && videoBlob.size > 0) {
    form.append("video", videoBlob, videoName);
  }
  form.append("metadata", JSON.stringify(mergedMeta));
  if (onUploadProgress) {
    onUploadProgress({ field: "multipart", index: 1, total: 1, bytes: totalBytes });
  }
  return requestForm("/api/v1/emotion/infer-upload", form, timeoutMs, true);
}

async function inferEmotionUploadChunked({
  sessionId,
  text,
  audioBlob,
  videoBlob,
  metadata,
  onUploadProgress,
  timeoutMs = 300000,
}) {
  if (audioBlob) {
    await uploadBlobInChunks(sessionId, "audio", audioBlob, onUploadProgress);
  }
  if (videoBlob && videoBlob.size > 0) {
    await uploadBlobInChunks(sessionId, "video", videoBlob, onUploadProgress);
  }
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("text", text || "");
  if (metadata) {
    form.append("metadata", JSON.stringify(metadata));
  }
  return requestForm("/api/v1/emotion/infer-from-upload", form, timeoutMs, true);
}

export async function checkHealth() {
  await initApiBase();
  return requestJson("/api/v1/health", { timeoutMs: 15000 });
}

export async function ingestChunk(payload) {
  return requestJson("/api/v1/ingest/chunk", {
    method: "POST",
    body: payload,
    timeoutMs: 120000,
  });
}

export async function inferEmotion(payload) {
  return requestJson("/api/v1/emotion/infer", {
    method: "POST",
    body: payload,
    timeoutMs: 300000,
  });
}

export async function inferEmotionUpload({
  sessionId,
  text,
  audioBlob,
  videoBlob,
  videoMime = "",
  videoFilename = "",
  metadata = {},
  onUploadProgress,
}) {
  const mergedMeta = {
    ...metadata,
    video_mime: videoMime || metadata.video_mime || videoBlob?.type || "",
    video_filename: videoFilename || metadata.video_filename || videoBlob?.name || "capture.webm",
  };
  const videoName =
    mergedMeta.video_filename ||
    (mergedMeta.video_mime?.includes("webm") ? "capture.webm" : "capture.jpg");
  const totalBytes = (audioBlob?.size || 0) + (videoBlob?.size || 0);
  const tunnel = isCursorTunnelMode();
  const cloudflare = isCloudflareQuickTunnel();
  const server = isServerDeployMode();
  const maxMultipart =
    cloudflare || tunnel
      ? TUNNEL_MAX_MULTIPART_BYTES
      : server
        ? SERVER_MAX_MULTIPART_BYTES
        : TUNNEL_MAX_MULTIPART_BYTES;

  if ((tunnel || cloudflare) && totalBytes > maxMultipart) {
    throw new Error(
      `采集数据 ${Math.round(totalBytes / 1024)}KB 超过隧道单次上限 ${Math.round(maxMultipart / 1024)}KB，请缩短录制（Cloudflare 建议≤10s）或使用 http://127.0.0.1:8000 直连`,
    );
  }

  const useMultipartOnly = tunnel || cloudflare || server || isApiSameOrigin();
  const useMultipartFirst =
    useMultipartOnly ||
    (totalBytes <= maxMultipart && (isApiSameOrigin() || !resolvedBase));

  const inferTimeoutMs = cloudflare ? 120000 : 300000;

  if (useMultipartFirst && totalBytes <= maxMultipart) {
    let lastMultipartErr = null;
    const retries = tunnel || cloudflare ? 3 : 1;
    for (let attempt = 1; attempt <= retries; attempt += 1) {
      try {
        return await inferEmotionUploadMultipart({
          sessionId,
          text,
          audioBlob,
          videoBlob,
          videoName,
          mergedMeta,
          totalBytes,
          onUploadProgress,
          timeoutMs: inferTimeoutMs,
        });
      } catch (multipartErr) {
        lastMultipartErr = multipartErr;
        if (attempt < retries) {
          await new Promise((r) => setTimeout(r, 800 * attempt));
        }
      }
    }
    if (tunnel || cloudflare) {
      const via = cloudflare ? "Cloudflare 隧道" : "Cursor 隧道";
      throw new Error(
        `${via}单次上传失败（已重试 ${retries} 次）: ${lastMultipartErr?.message || "unknown"}。请刷新页面后重试，勿使用带 :8000 的地址。`,
      );
    }
  }

  if (tunnel || cloudflare) {
    throw new Error("隧道模式不支持分块上传，请缩短录制后使用整包 multipart");
  }

  let lastMultipartErr = null;
  try {
    return await inferEmotionUploadChunked({
      sessionId,
      text,
      audioBlob,
      videoBlob,
      metadata: mergedMeta,
      onUploadProgress,
      timeoutMs: inferTimeoutMs,
    });
  } catch (chunkErr) {
    if (totalBytes < 120000 && !isServerDeployMode()) {
      try {
        return await inferEmotionUploadMultipart({
          sessionId,
          text,
          audioBlob,
          videoBlob,
          videoName,
          mergedMeta,
          totalBytes,
          onUploadProgress,
          timeoutMs: inferTimeoutMs,
        });
      } catch {
        // fall through
      }
    }
    throw chunkErr;
  }
}

export async function respondAgent(payload) {
  return requestJson("/api/v1/agent/respond", {
    method: "POST",
    body: payload,
    timeoutMs: 120000,
  });
}

export function buildSessionWs(sessionId) {
  if (isApiSameOrigin()) {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return new WebSocket(`${wsProtocol}//${window.location.host}/api/v1/session/${sessionId}/stream`);
  }
  const base = resolvedBase || ENV_BASE;
  if (base) {
    const wsUrl = base.replace("http://", "ws://").replace("https://", "wss://");
    return new WebSocket(`${wsUrl}/api/v1/session/${sessionId}/stream`);
  }
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return new WebSocket(`${wsProtocol}//${window.location.host}/api/v1/session/${sessionId}/stream`);
}
