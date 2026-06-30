import { useEffect, useMemo, useRef, useState } from "react";
import {
  buildSessionWs,
  checkHealth,
  estimateUploadChunks,
  getApiDisplayUrl,
  getUploadProfile,
  initApiBase,
  getCaptureMaxSec,
  inferEmotionUpload,
  isCloudflareQuickTunnel,
  isCursorTunnelMode,
  isServerDeployMode,
  MAX_SAFE_UPLOAD_CHUNKS,
  respondAgent,
  TUNNEL_MAX_MULTIPART_BYTES,
} from "./api";

function randomSessionId() {
  return `sess_${Math.random().toString(36).slice(2, 10)}`;
}

function randomRequestId() {
  return `req_${Math.random().toString(36).slice(2, 10)}`;
}

/** Strip `data:*;base64,` prefix so backend receives raw base64. */
function stripDataUrl(value) {
  if (!value) return "";
  const text = String(value);
  if (text.startsWith("data:")) {
    const comma = text.indexOf(",");
    if (comma !== -1) return text.slice(comma + 1);
  }
  return text;
}

/** @deprecated 使用 stripDataUrl；保留别名避免旧缓存脚本引用报错 */
function stripBase64Payload(value) {
  return stripDataUrl(value);
}

async function dataUrlToBlob(dataUrl) {
  const res = await fetch(dataUrl);
  return res.blob();
}

function trimAudioSamples(samples, sampleRate, maxSec = getCaptureMaxSec(), fromEnd = false) {
  const maxLen = Math.floor(sampleRate * maxSec);
  if (samples.length <= maxLen) return samples;
  // 长时推理：默认保留从开头起的 maxSec 秒（后端按 3s 窗切分）；fromEnd 时仅取尾部
  return fromEnd ? samples.subarray(samples.length - maxLen) : samples.subarray(0, maxLen);
}

function EmotionTimeline({ windows, summary }) {
  if (!windows?.length) return null;
  const maxConf = Math.max(...windows.map((w) => w.confidence || 0), 0.01);
  return (
    <div className="emotion-timeline">
      <h4>情绪时间线（每 3 秒一窗）</h4>
      {summary ? (
        <p className="hint">
          共 {summary.num_windows} 窗 · {summary.total_duration_sec}s · 聚合：近端加权 ·
          {summary.emotion_shift_detected ? " 检测到情绪变化" : " 情绪较稳定"}
        </p>
      ) : null}
      <ul className="timeline-bars">
        {windows.map((w) => (
          <li key={w.index} className="timeline-item" title={`${w.start_sec}s–${w.end_sec}s`}>
            <span className="timeline-label">
              {w.start_sec}s–{w.end_sec}s
            </span>
            <span className="timeline-emotion">{w.emotion_label_cn || w.emotion_label}</span>
            <div className="timeline-bar-track">
              <div
                className="timeline-bar-fill"
                style={{ width: `${Math.round(((w.confidence || 0) / maxConf) * 100)}%` }}
              />
            </div>
            <span className="timeline-conf">{(w.confidence || 0).toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

const EMOTION_CN = {
  happy: "开心",
  sad: "难过",
  angry: "生气",
  fear: "害怕",
  neutral: "平静",
  anxious: "焦虑",
  other: "其他",
};

function formatLogTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString("zh-CN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function EmotionProbBars({ items, activeLabel }) {
  if (!items?.length) return <p className="hint">暂无分类概率</p>;
  const sorted = [...items].sort((a, b) => (b.prob || 0) - (a.prob || 0));
  return (
    <div className="prob-bars">
      {sorted.map((item) => {
        const pct = Math.max(0, Math.min(100, Number(item.prob || 0) * 100));
        const isActive = item.label === activeLabel;
        return (
          <div key={item.label} className={`prob-row ${isActive ? "prob-row-active" : ""}`}>
            <span className="prob-label">
              {item.label_cn || EMOTION_CN[item.label] || item.label}
              <small>{item.label}</small>
            </span>
            <div className="prob-track">
              <div className="prob-fill" style={{ width: `${pct}%` }} />
            </div>
            <span className="prob-value">{Number(item.prob || 0).toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}

function isTrainedCheckpoint(source) {
  const s = String(source || "");
  return s === "checkpoint" || s.startsWith("checkpoint_");
}

function formatPipelineStep(step) {
  if (!step) return [];
  const lines = [];
  if (step.stage === "1_ingest") {
    lines.push(`上传: 视频 ${step.video_bytes ?? 0} B · 音频 ${step.audio_bytes ?? 0} B`);
    if (step.detail) lines.push(step.detail);
  } else if (step.stage === "2_asr") {
    lines.push(`引擎: ${step.provider || "-"} · 置信度 ${Number(step.confidence ?? 0).toFixed(2)}`);
    lines.push(`文本预览: ${step.text_preview || "（空）"}`);
    if (step.error) lines.push(`错误: ${step.error}`);
  } else if (step.stage === "5_asr_calibration") {
    lines.push(
      `校正: ${step.status} · ${step.label_before || "-"} → ${step.label_after || "-"}`,
    );
    lines.push(`原因: ${step.reason || "-"}`);
  } else if (step.stage === "6_arbitration") {
    lines.push(
      `仲裁: ${step.status} · 模型=${step.model_label || "-"} → 最终=${step.final_label || "-"}`,
    );
    lines.push(`来源: ${step.source || "-"} · ${step.reason || "-"}`);
  } else if (step.stage === "3_text_merge") {
    lines.push(`来源: ${step.source || "-"} · 长度 ${step.merged_len ?? 0}`);
    lines.push(`合并预览: ${step.merged_preview || "（空）"}`);
  } else if (step.stage === "4_emotion_model") {
    lines.push(
      `模式: ${step.mode === "temporal" ? "长时多窗" : "单窗"} · 来源 ${step.inference_source || "-"}`,
    );
    if (step.num_windows) lines.push(`时间窗: ${step.num_windows} 个`);
    lines.push(
      `结果: ${step.label || "-"} · 置信度 ${Number(step.confidence ?? 0).toFixed(3)} · ${Number(step.inference_ms ?? 0).toFixed(0)} ms`,
    );
  }
  return lines;
}

function ModalityCards({ trace }) {
  const mods = trace?.modalities || {};
  const temporal = trace?.temporal || {};
  const cards = [
    {
      key: "video",
      title: "视频模态",
      ok: mods.video?.preprocessed,
      lines: [
        `接收: ${mods.video?.received_bytes ?? 0} bytes`,
        `预处理: ${mods.video?.preprocessed ? "成功" : "失败/缺失"}`,
        `解码: ${mods.video?.decode_mode || trace?.video_decode || "-"}`,
        `抽帧: ${mods.video?.frames_extracted ?? "-"} / ${mods.video?.num_frames ?? "-"}`,
        mods.video?.temporal_windows
          ? `时间窗: ${mods.video.temporal_windows} × ${temporal.window_sec ?? 3}s`
          : null,
        mods.video?.tensor_shape ? `批次张量: ${mods.video.tensor_shape.join("×")}` : null,
      ].filter(Boolean),
    },
    {
      key: "audio",
      title: "音频模态",
      ok: mods.audio?.preprocessed,
      lines: [
        `接收: ${mods.audio?.received_bytes ?? 0} bytes`,
        `预处理: ${mods.audio?.preprocessed ? "成功" : "失败/缺失"}`,
        mods.audio?.tensor_shape ? `批次张量: ${mods.audio.tensor_shape.join("×")}` : "张量: -",
        `采样率: ${mods.audio?.sample_rate ?? "-"}Hz · 全长: ${mods.audio?.duration_sec ?? "-"}s`,
        mods.audio?.temporal_windows
          ? `切窗: ${mods.audio.temporal_windows} × ${mods.audio?.window_sec ?? temporal.window_sec ?? 3}s`
          : null,
      ].filter(Boolean),
    },
    {
      key: "text",
      title: "文本模态",
      ok: mods.text?.preprocessed && (mods.text?.char_len || 0) > 0,
      lines: [
        `来源: ${mods.text?.source || trace?.text_merge?.source || "-"}`,
        `内容: ${mods.text?.merged_text || mods.text?.content_preview || trace?.text_merge?.merged_text || "（空）"}`,
        `Tokenizer: ${mods.text?.tokenizer || "-"}`,
        `Token 数: ${mods.text?.token_count ?? "-"}`,
      ],
    },
  ];
  return (
    <div className="modality-grid">
      {cards.map((c) => (
        <div key={c.key} className={`modality-card ${c.ok ? "modality-ok" : "modality-warn"}`}>
          <div className="modality-head">
            <strong>{c.title}</strong>
            <span className={`chip ${c.ok ? "chip-ok" : "chip-warn"}`}>{c.ok ? "已送入模型" : "未就绪"}</span>
          </div>
          {c.lines.map((line) => (
            <p key={line} className="hint modality-line">
              {line}
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}

function summarizeEvent(evt) {
  if (!evt || typeof evt !== "object") return "-";
  const t = evt.event || "";
  if (t === "connected") return `会话 ${evt.session_id || "-"} 已连接`;
  if (t === "ingest") return `已接收窗口数：${evt.windows ?? "-"}`;
  if (t === "emotion") {
    const emotion = evt.emotion?.emotion_label || "-";
    const conf = evt.emotion?.confidence;
    const confTxt = typeof conf === "number" ? conf.toFixed(2) : "-";
    const asrLen = (evt.asr?.text || "").length;
    return `情绪=${emotion}（${confTxt}），识别文本长度=${asrLen}`;
  }
  if (t === "agent") {
    const provider = evt.response?.llm_provider || "-";
    const text = evt.response?.reply_text || "";
    return `回复来源=${provider}，内容：${text.slice(0, 48)}${text.length > 48 ? "..." : ""}`;
  }
  return JSON.stringify(evt);
}

export default function App() {
  const [sessionId] = useState(randomSessionId);
  const [status, setStatus] = useState("idle");
  const [emotion, setEmotion] = useState(null);
  const [agentReply, setAgentReply] = useState(null);
  const [events, setEvents] = useState([]);
  const [text, setText] = useState("");
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [asrText, setAsrText] = useState("");
  const [asrConfidence, setAsrConfidence] = useState(0);
  const [asrProvider, setAsrProvider] = useState("");
  const [asrOk, setAsrOk] = useState(null);
  const [llmOk, setLlmOk] = useState(null);
  const [apiReady, setApiReady] = useState(false);
  const [capturedImage, setCapturedImage] = useState("");
  const [audioInfo, setAudioInfo] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [captureAbortReason, setCaptureAbortReason] = useState("");
  const [micLevel, setMicLevel] = useState(0);
  const [actionCount, setActionCount] = useState(0);
  const [selfCheckRunning, setSelfCheckRunning] = useState(false);
  const [selfCheckResult, setSelfCheckResult] = useState(null);
  const [selfCheckAudioUrl, setSelfCheckAudioUrl] = useState("");
  const [audioInputDevices, setAudioInputDevices] = useState([]);
  const [selectedAudioInputId, setSelectedAudioInputId] = useState("");
  const [error, setError] = useState("");
  const [backendOk, setBackendOk] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [showRawResult, setShowRawResult] = useState(false);
  const [showRawDebug, setShowRawDebug] = useState(false);
  const [runtimeLogs, setRuntimeLogs] = useState([]);
  const mediaStreamRef = useRef(null);
  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioCtxRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const audioSourceRef = useRef(null);
  const pcmRef = useRef([]);
  const pcmSampleRateRef = useRef(16000);
  const videoRecorderRef = useRef(null);
  const videoChunksRef = useRef([]);
  const videoMimeRef = useRef("video/webm");
  const frameSnapshotsRef = useRef([]);
  const frameIntervalRef = useRef(null);
  const [checkpointPreset, setCheckpointPreset] = useState("meld_only");

  const shortEvents = useMemo(() => events.slice(-10).reverse(), [events]);
  const statusText =
    {
      idle: "待命",
      capture_ready: "设备就绪",
      capturing: "采集中",
      running: "正在推理",
      agent_pending: "情绪已出，等待助手回复",
      done: "已完成",
      error: "异常",
    }[status] || status;
  const micLevelState = micLevel < 20 ? "偏低" : micLevel < 60 ? "正常" : "偏高";
  const isAnalyzing = status === "running" || status === "agent_pending";
  const emotionLabelText =
    {
      happy: "开心",
      sad: "难过",
      angry: "生气",
      fear: "害怕",
      neutral: "平静",
      anxious: "焦虑",
      other: "其他",
    }[emotion?.emotion_label || ""] || "暂无";
  const emotionToneClass =
    {
      happy: "emotion-positive",
      neutral: "emotion-neutral",
      anxious: "emotion-negative",
      sad: "emotion-negative",
      angry: "emotion-negative",
      fear: "emotion-negative",
      other: "emotion-neutral",
    }[emotion?.emotion_label || ""] || "emotion-neutral";

  function appendRuntimeLog(level, message, data = null) {
    setRuntimeLogs((prev) => [
      ...prev.slice(-49),
      { ts: Date.now(), level, message, data },
    ]);
  }

  useEffect(() => {
    let cancelled = false;
    initApiBase().then(() => {
      if (!cancelled) setApiReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!apiReady) return;
    let disposed = false;
    const ws = buildSessionWs(sessionId);
    wsRef.current = ws;
    ws.onopen = () => {
      if (disposed) return;
      setWsConnected(true);
    };
    ws.onmessage = (ev) => {
      if (disposed) return;
      const payload = JSON.parse(ev.data);
      setEvents((prev) => [...prev, payload]);
      if (payload?.event === "emotion") {
        appendRuntimeLog("info", "WebSocket: 收到情绪推理结果", {
          label: payload.emotion?.emotion_label,
          source: payload.emotion?.inference_source,
        });
      }
    };
    ws.onerror = () => {
      if (disposed) return;
      setWsConnected(false);
    };
    ws.onclose = () => {
      if (disposed) return;
      setWsConnected(false);
    };
    return () => {
      disposed = true;
      ws.close();
    };
  }, [sessionId, apiReady]);

  async function refreshAudioDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter((d) => d.kind === "audioinput");
      setAudioInputDevices(mics);
      if (!selectedAudioInputId && mics.length > 0) {
        setSelectedAudioInputId(mics[0].deviceId);
      }
    } catch (e) {
      // ignore device enumeration errors here; capture flow will show detailed error later.
    }
  }

  useEffect(() => {
    refreshAudioDevices();
  }, []);

  useEffect(() => {
    if (!apiReady) return;
    let cancelled = false;
    (async () => {
      try {
        const health = await checkHealth();
        if (cancelled) return;
        setBackendOk(Boolean(health?.ok));
        setAsrOk(health?.asr_ok ?? null);
        setLlmOk(health?.llm_ok ?? null);
        const msgs = [];
        if (health?.asr_ok === false) {
          msgs.push(
            `ASR 未就绪：${health?.asr?.message || "请启动 asr-local"}（./emotion-agent/asr-local/start_server.sh）`,
          );
        }
        if (health?.llm_ok === false) {
          msgs.push(
            `LLM 未就绪：${health?.llm?.message || ""} ${health?.llm?.hint || "请运行 emotion-agent/scripts/start_ollama.sh"}`,
          );
        }
        if (health?.using_trained_checkpoint === false) {
          msgs.push("情绪模型未加载训练 checkpoint，请检查 MODEL_PROVIDER=current");
        }
        appendRuntimeLog("info", "健康检查完成", {
          backend: health?.ok,
          asr: health?.asr_ok,
          llm: health?.llm_ok,
          checkpoint: health?.using_trained_checkpoint,
          provider: health?.model?.provider,
          preset: health?.model?.preset,
        });
        const preset = health?.model?.preset || health?.model?.checkpoint_preset;
        if (preset) {
          setCheckpointPreset(preset);
        }
        setError(msgs.join(" | "));
      } catch (e) {
        if (cancelled) return;
        setBackendOk(false);
        setError(`后端未就绪: ${e.message}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiReady]);

  async function ensureCaptureDevice() {
    if (mediaStreamRef.current) {
      return mediaStreamRef.current;
    }
    try {
      const audioConstraint = selectedAudioInputId
        ? { deviceId: { exact: selectedAudioInputId } }
        : true;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: audioConstraint,
      });
      mediaStreamRef.current = stream;
      setCaptureEnabled(true);
      setStatus("capture_ready");
      await refreshAudioDevices();
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      return stream;
    } catch (e) {
      setError(`采集授权失败: ${e.message}`);
      throw e;
    }
  }

  function pickVideoMimeType() {
    const candidates = ["video/webm;codecs=vp8", "video/webm", "video/mp4"];
    for (const mime of candidates) {
      if (window.MediaRecorder?.isTypeSupported?.(mime)) {
        return mime;
      }
    }
    return "";
  }

  function startVideoRecorder(stream) {
    if (!window.MediaRecorder) {
      return;
    }
    const mimeType = pickVideoMimeType();
    if (!mimeType) {
      appendRuntimeLog("warn", "浏览器不支持 MediaRecorder 视频编码，将回退单帧 JPEG");
      return;
    }
    videoMimeRef.current = mimeType.split(";")[0];
    videoChunksRef.current = [];
    const tunnel = isCursorTunnelMode();
    const recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: tunnel ? 80000 : 150000,
      audioBitsPerSecond: tunnel ? 32000 : 48000,
    });
    recorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) {
        videoChunksRef.current.push(ev.data);
      }
    };
    recorder.start(250);
    videoRecorderRef.current = recorder;
    appendRuntimeLog("info", `开始录制视频 clip (${videoMimeRef.current})`);
  }

  function stopVideoRecorder() {
    const recorder = videoRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      recorder.onstop = () => {
        const blob = new Blob(videoChunksRef.current, {
          type: videoMimeRef.current || "video/webm",
        });
        videoRecorderRef.current = null;
        resolve(blob.size > 0 ? blob : null);
      };
      try {
        recorder.stop();
      } catch {
        resolve(null);
      }
    });
  }

  function snapshotFrame() {
    const videoEl = videoRef.current;
    if (!videoEl || videoEl.videoWidth === 0 || videoEl.videoHeight === 0) {
      return "";
    }
    const maxW = 160;
    let w = videoEl.videoWidth;
    let h = videoEl.videoHeight;
    if (w > maxW) {
      h = Math.round((h * maxW) / w);
      w = maxW;
    }
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return "";
    }
    ctx.drawImage(videoEl, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", 0.8);
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve((reader.result || "").toString());
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Encode mono float32 PCM [-1,1] into WAV (16-bit PCM)
  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: "audio/wav" });
  }

  function concatFloat32(chunks) {
    let total = 0;
    for (const c of chunks) total += c.length;
    const out = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  }

  // Linear resample float32 array from srcRate to dstRate
  function resampleLinear(input, srcRate, dstRate) {
    if (srcRate === dstRate) return input;
    const ratio = srcRate / dstRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const t = i * ratio;
      const i0 = Math.floor(t);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = t - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  function calcRmsAndPeak(samples) {
    if (!samples || samples.length === 0) return { rms: 0, peak: 0 };
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const v = Math.abs(samples[i]);
      sum += v * v;
      if (v > peak) peak = v;
    }
    return { rms: Math.sqrt(sum / samples.length), peak };
  }

  function calcRms(samples) {
    if (!samples || samples.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / samples.length);
  }

  async function startCapture() {
    setError("");
    setCaptureAbortReason("");
    setMicLevel(0);
    setActionCount((v) => v + 1);
    setDebugInfo({
      phase: "start_clicked",
      at: new Date().toISOString(),
      session_id: sessionId,
      capturing,
      capture_ready: captureEnabled,
      ws_connected: wsConnected,
      end_button_enabled: capturing,
      action_count: actionCount + 1,
    });
    try {
      const stream = await ensureCaptureDevice();
      setCapturedImage("");
      setAudioInfo(null);
      setAsrText("");
      setAsrConfidence(0);
      setCapturing(true);
      setStatus("capturing");
      setDebugInfo((prev) => ({
        ...(prev || {}),
        phase: "capturing",
        at: new Date().toISOString(),
        capture_ready: true,
        capturing: true,
        end_button_enabled: true,
      }));

      // Prefer WebAudio -> WAV to avoid ffmpeg dependency on server.
      pcmRef.current = [];
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        setError("当前浏览器不支持 AudioContext，无法采集音频。");
        return;
      }
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;
      pcmSampleRateRef.current = audioCtx.sampleRate || 48000;
      const source = audioCtx.createMediaStreamSource(stream);
      audioSourceRef.current = source;
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      audioProcessorRef.current = processor;
      processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        pcmRef.current.push(new Float32Array(input));
        const rms = calcRms(input);
        // Scale RMS into 0-100 for visual level bar.
        const level = Math.max(0, Math.min(100, Math.round(rms * 800)));
        setMicLevel(level);
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      startVideoRecorder(stream);
      frameSnapshotsRef.current = [];
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      frameIntervalRef.current = setInterval(() => {
        const url = snapshotFrame();
        if (!url) return;
        const b64 = stripDataUrl(url);
        const arr = frameSnapshotsRef.current;
        if (arr.length >= 4) {
          arr.shift();
        }
        arr.push(b64);
      }, 750);
    } catch (e) {
      setError(`开始采集失败: ${e.message}`);
    }
  }

  function stopFrameSampling() {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  }

  async function runMicSelfCheck() {
    if (capturing || selfCheckRunning) return;
    setSelfCheckRunning(true);
    setError("");
    setCaptureAbortReason("");
    setMicLevel(0);
    setSelfCheckResult(null);
    if (selfCheckAudioUrl) {
      URL.revokeObjectURL(selfCheckAudioUrl);
      setSelfCheckAudioUrl("");
    }
    try {
      const stream = await ensureCaptureDevice();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        throw new Error("浏览器不支持 AudioContext");
      }
      const audioCtx = new AudioCtx();
      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      const chunks = [];
      processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(input));
        const rms = calcRms(input);
        const level = Math.max(0, Math.min(100, Math.round(rms * 800)));
        setMicLevel(level);
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);

      await new Promise((resolve) => setTimeout(resolve, 2000));

      processor.disconnect();
      source.disconnect();
      await audioCtx.close();
      setMicLevel(0);

      const pcm = concatFloat32(chunks);
      const resampled = resampleLinear(pcm, audioCtx.sampleRate || 48000, 16000);
      const { rms, peak } = calcRmsAndPeak(resampled);
      const durationSec = resampled.length / 16000;
      const wavBlob = encodeWav(resampled, 16000);
      const url = URL.createObjectURL(wavBlob);
      setSelfCheckAudioUrl(url);
      setSelfCheckResult({
        durationSec: Number(durationSec.toFixed(2)),
        rms: Number(rms.toFixed(5)),
        peak: Number(peak.toFixed(5)),
        bytes: wavBlob.size,
        verdict: rms >= 0.002 || peak >= 0.02 ? "麦克风输入正常" : "输入接近静音，请更换设备或检查系统权限",
      });
    } catch (e) {
      setError(`麦克风自检失败: ${e.message}`);
    } finally {
      setSelfCheckRunning(false);
    }
  }

  async function stopCaptureAndInfer() {
    setActionCount((v) => v + 1);
    appendRuntimeLog("info", "用户点击「结束并推理」");
    setDebugInfo((prev) => ({
      ...(prev || {}),
      phase: "stop_clicked",
      at: new Date().toISOString(),
      session_id: sessionId,
      capturing_before_stop: capturing,
      capture_ready_before_stop: captureEnabled,
      ws_connected: wsConnected,
      end_button_enabled_before_stop: capturing,
      action_count: actionCount + 1,
    }));
    if (!capturing) {
      return;
    }
    if (!captureEnabled) {
      setStatus("error");
      const reason = "采集未就绪，请先允许摄像头和麦克风权限后再开始。";
      setError(reason);
      setCaptureAbortReason(reason);
      setMicLevel(0);
      setDebugInfo({
        phase: "precheck",
        reason,
        session_id: sessionId,
        capture_ready: captureEnabled,
        ws_connected: wsConnected,
      });
      setCapturing(false);
      return;
    }
    setCapturing(false);
    stopFrameSampling();
    setStatus("running");
    try {
      const imageDataUrl = snapshotFrame();
      if (imageDataUrl) {
        const b64 = stripDataUrl(imageDataUrl);
        const arr = frameSnapshotsRef.current;
        if (!arr.length || arr[arr.length - 1] !== b64) {
          if (arr.length >= 4) arr.shift();
          arr.push(b64);
        }
      }
      setCapturedImage(imageDataUrl);
      if (!imageDataUrl) {
        setStatus("error");
        const reason = "未获取到有效视频帧，请确认摄像头画面正常后重试。";
        setError(reason);
        setCaptureAbortReason(reason);
        setDebugInfo({
          phase: "video_guard",
          reason,
          session_id: sessionId,
          capture_ready: captureEnabled,
          ws_connected: wsConnected,
        });
        return;
      }

      let wavBlob = null;
      let currentAudioInfo = null;
      // Stop WebAudio capture and encode WAV (16k mono).
      const audioCtx = audioCtxRef.current;
      const source = audioSourceRef.current;
      const processor = audioProcessorRef.current;
      if (processor) {
        try {
          processor.disconnect();
        } catch {}
      }
      if (source) {
        try {
          source.disconnect();
        } catch {}
      }
      if (audioCtx) {
        try {
          await audioCtx.close();
        } catch {}
      }
      audioCtxRef.current = null;
      audioSourceRef.current = null;
      audioProcessorRef.current = null;
      setMicLevel(0);

      if (pcmRef.current.length > 0) {
        const pcm = concatFloat32(pcmRef.current);
        let resampled = resampleLinear(pcm, pcmSampleRateRef.current, 16000);
        resampled = trimAudioSamples(resampled, 16000, getCaptureMaxSec(), false);
        const durationSec = resampled.length / 16000;
        const { rms, peak } = calcRmsAndPeak(resampled);
        if (durationSec < 0.8) {
          setStatus("error");
          const reason = "采集音频时长过短，请至少说话 1 秒后再结束。";
          setError(reason);
          setCaptureAbortReason(reason);
          setMicLevel(0);
          setAudioInfo({
            mimeType: "audio/wav",
            bytes: 0,
            chunks: pcmRef.current.length,
            sampleRate: 16000,
            durationSec: Number(durationSec.toFixed(2)),
            rms: Number(rms.toFixed(5)),
            peak: Number(peak.toFixed(5)),
          });
          setDebugInfo({
            phase: "audio_guard",
            reason,
            session_id: sessionId,
            duration_sec: Number(durationSec.toFixed(2)),
            rms: Number(rms.toFixed(5)),
            peak: Number(peak.toFixed(5)),
            capture_ready: captureEnabled,
            ws_connected: wsConnected,
          });
          return;
        }
        if (rms < 0.002 && peak < 0.02) {
          setStatus("error");
          const reason = "检测到音频接近静音，请检查麦克风权限/设备占用后重试。";
          setError(reason);
          setCaptureAbortReason(reason);
          setMicLevel(0);
          setAudioInfo({
            mimeType: "audio/wav",
            bytes: 0,
            chunks: pcmRef.current.length,
            sampleRate: 16000,
            durationSec: Number(durationSec.toFixed(2)),
            rms: Number(rms.toFixed(5)),
            peak: Number(peak.toFixed(5)),
          });
          setDebugInfo({
            phase: "audio_guard",
            reason,
            session_id: sessionId,
            duration_sec: Number(durationSec.toFixed(2)),
            rms: Number(rms.toFixed(5)),
            peak: Number(peak.toFixed(5)),
            capture_ready: captureEnabled,
            ws_connected: wsConnected,
          });
          return;
        }
        wavBlob = encodeWav(resampled, 16000);
        currentAudioInfo = {
          mimeType: wavBlob.type || "audio/wav",
          bytes: wavBlob.size,
          chunks: pcmRef.current.length,
          sampleRate: 16000,
          durationSec: Number(durationSec.toFixed(2)),
          rms: Number(rms.toFixed(5)),
          peak: Number(peak.toFixed(5)),
        };
        setAudioInfo(currentAudioInfo);
      } else {
        setStatus("error");
        const reason = "未采集到音频帧，请确认麦克风权限后重新开始。";
        setError(reason);
        setCaptureAbortReason(reason);
        setMicLevel(0);
        setAudioInfo({ mimeType: "-", bytes: 0, chunks: 0, sampleRate: 0, durationSec: 0, rms: 0, peak: 0 });
        setDebugInfo({
          phase: "audio_guard",
          reason,
          session_id: sessionId,
          capture_ready: captureEnabled,
          ws_connected: wsConnected,
        });
        return;
      }

      if (!wavBlob) {
        setStatus("error");
        setError("未生成有效音频，请重试。");
        return;
      }

      let videoBlob = await stopVideoRecorder();
      let videoMime = videoBlob?.type || videoMimeRef.current || "video/webm";
      let videoFilename = "capture.webm";
      const tunnelMode = isCursorTunnelMode();
      const serverMode = isServerDeployMode();
      const uploadProfile = getUploadProfile();
      const MAX_VIDEO_BYTES = serverMode ? 3000000 : 96000;
      const videoChunkCount = videoBlob ? estimateUploadChunks(videoBlob.size) : 0;
      const totalUploadBytes = (wavBlob?.size || 0) + (videoBlob?.size || 0);

      appendRuntimeLog("info", `上传策略: ${uploadProfile.label}`);
      if (isCloudflareQuickTunnel()) {
        appendRuntimeLog(
          "warn",
          "Cloudflare Quick Tunnel 单次请求约 100s 上限；ASR 超时将自动跳过，优先保证情绪结果返回",
        );
      }

      if (!videoBlob || videoBlob.size === 0) {
        appendRuntimeLog("warn", "视频 clip 为空，回退单帧 JPEG");
        videoBlob = await dataUrlToBlob(imageDataUrl);
        videoMime = "image/jpeg";
        videoFilename = "capture.jpg";
      } else if (
        tunnelMode &&
        (totalUploadBytes > TUNNEL_MAX_MULTIPART_BYTES ||
          (videoMime.startsWith("video/") && videoBlob.size > MAX_VIDEO_BYTES))
      ) {
        appendRuntimeLog(
          "warn",
          `Cursor 隧道：合计 ${Math.round(totalUploadBytes / 1024)}KB 超限，改用 JPEG 单次上传（≤${Math.round(TUNNEL_MAX_MULTIPART_BYTES / 1024)}KB）`,
        );
        videoBlob = await dataUrlToBlob(imageDataUrl);
        videoMime = "image/jpeg";
        videoFilename = "capture.jpg";
      } else if (
        !serverMode &&
        !tunnelMode &&
        videoMime.startsWith("video/") &&
        (videoBlob.size > MAX_VIDEO_BYTES || videoChunkCount > MAX_SAFE_UPLOAD_CHUNKS)
      ) {
        appendRuntimeLog(
          "warn",
          `视频需 ${videoChunkCount} 次分块，改用 JPEG`,
        );
        videoBlob = await dataUrlToBlob(imageDataUrl);
        videoMime = "image/jpeg";
        videoFilename = "capture.jpg";
      } else if (serverMode && videoMime.startsWith("video/")) {
        appendRuntimeLog(
          "info",
          `服务器直连：保留 webm (${Math.round(videoBlob.size / 1024)}KB，整包 multipart)`,
        );
      } else if (tunnelMode && videoMime.startsWith("video/")) {
        appendRuntimeLog(
          "info",
          `Cursor 隧道：webm ${Math.round(videoBlob.size / 1024)}KB + 音频，单次 multipart 上传`,
        );
      } else if (videoMime.includes("mp4")) {
        videoFilename = "capture.mp4";
      }

      const requestId = randomRequestId();
      appendRuntimeLog("info", "采集完成，准备上传推理", {
        audio_bytes: wavBlob.size,
        video_bytes: videoBlob.size,
        video_mime: videoMime,
        audio_duration_sec: currentAudioInfo?.durationSec,
      });
      setCaptureAbortReason("");
      setDebugInfo({
        request_id: requestId,
        phase: "infer_request",
        upload: "chunked-or-multipart",
        at: new Date().toISOString(),
        session_id: sessionId,
        has_audio: Boolean(wavBlob),
        has_video: Boolean(videoBlob),
        audio_bytes: wavBlob.size,
        video_bytes: videoBlob.size,
        capture_ready: captureEnabled,
        ws_connected: wsConnected,
        audio_info: currentAudioInfo,
        end_button_enabled: false,
        action_count: actionCount + 1,
      });

      let emo;
      try {
        emo = await inferEmotionUpload({
          sessionId,
          text,
          audioBlob: wavBlob,
          videoBlob,
          videoMime,
          videoFilename,
          metadata: {
            source: "browser_capture_clip",
            request_id: requestId,
            video_mime: videoMime,
            video_filename: videoFilename,
            cloudflare_tunnel: isCloudflareQuickTunnel(),
            checkpoint_preset: checkpointPreset,
            capture_frames_b64: [...frameSnapshotsRef.current],
            temporal_inference: {
              max_windows: isCloudflareQuickTunnel() || isCursorTunnelMode() ? 4 : 10,
              stride_sec: 1.5,
            },
          },
          onUploadProgress: (p) => {
            appendRuntimeLog("info", `上传 ${p.field} ${p.index}/${p.total}`);
          },
        });
      } catch (e) {
        throw new Error(`情绪推理失败: ${e.message}`);
      }
      setEmotion(emo);
      setAsrText(emo.asr_text || "");
      setAsrConfidence(emo.asr_confidence || 0);
      setAsrProvider(emo.asr_provider || "");
      setStatus("agent_pending");
      appendRuntimeLog(
        String(emo.inference_source || "").startsWith("checkpoint") ? "info" : "warn",
        `情绪模型返回: ${emo.emotion_label} (source=${emo.inference_source}, conf=${Number(emo.confidence || 0).toFixed(3)})`,
        {
          is_mock: String(emo.inference_source || "").startsWith("mock"),
          inference_ms: emo.inference_ms,
          top3: emo.top_emotions,
          pipeline_steps: emo.pipeline_trace?.steps,
        },
      );
      if (emo.pipeline_trace?.notes?.length) {
        emo.pipeline_trace.notes.forEach((note) => appendRuntimeLog("info", note));
      }
      if (emo.temporal_summary?.num_windows) {
        appendRuntimeLog(
          "info",
          `长时推理 ${emo.temporal_summary.num_windows} 窗 / ${emo.temporal_summary.total_duration_sec}s · 聚合 ${emo.temporal_summary.aggregation || "recency_weighted"}`,
          { windows: emo.temporal_windows?.length },
        );
      }
      if (emo.asr_provider === "mock") {
        setError("ASR 仍为 mock 模式，识别文本不可信。请配置 whisper_api 并启动 asr-local。");
      } else if (emo.asr_error && !(emo.asr_text || "").trim()) {
        setError(`ASR 转写失败: ${emo.asr_error}`);
      }
      setDebugInfo((prev) => ({
        ...(prev || {}),
        phase: "infer_response",
        asr_text_len: (emo.asr_text || "").length,
        asr_confidence: emo.asr_confidence || 0,
        asr_error: emo.asr_error || "",
        asr_provider: emo.asr_provider || "",
        model_provider: emo.model_provider || "",
        inference_source: emo.inference_source || "",
      }));

      let reply;
      try {
        reply = await respondAgent({
          session_id: sessionId,
          emotion_label: emo.emotion_label,
          confidence: emo.confidence,
          context_text: emo.asr_text || text,
          valence: emo.valence,
          arousal: emo.arousal,
          all_probs: emo.all_probs || [],
          all_probs_labeled: emo.all_probs_labeled || [],
          top_emotions: emo.top_emotions || [],
        });
      } catch (e) {
        throw new Error(`助手回复失败: ${e.message}`);
      }
      setAgentReply(reply);
      appendRuntimeLog("info", `LLM 回复完成 (provider=${reply?.llm_provider})`, {
        note: "LLM 以情绪模型输出为主，ASR 为辅",
        model_label: emo.emotion_label,
        model_conf: emo.confidence,
      });
      if (reply?.llm_provider === "template" && reply?.llm_error) {
        setError((prev) =>
          [prev, `LLM 使用模板兜底: ${reply.llm_error}`].filter(Boolean).join(" | "),
        );
      }
      setDebugInfo((prev) => ({
        ...(prev || {}),
        phase: "agent_response",
        llm_provider: reply?.llm_provider || "",
        llm_error: reply?.llm_error || "",
        llm_model: reply?.llm_model || "",
      }));
      setStatus("done");
      setMicLevel(0);
    } catch (e) {
      setStatus("error");
      appendRuntimeLog("error", `推理失败: ${e.message}`);
      setError(`结束采集并推理失败: ${e.message}`);
      setMicLevel(0);
      setDebugInfo((prev) => ({
        ...(prev || {}),
        phase: "infer_failed",
        error: String(e?.message || e),
      }));
    }
  }

  return (
    <main className="container">
      <header className="hero">
        <h1>情绪助手控制台</h1>
        <p className="subtitle">服务地址：{getApiDisplayUrl()}</p>
      </header>
      {error ? <p className="error">{error}</p> : null}

      <section className="card">
        <h2>会话状态</h2>
        <div className="kv-list">
          <p>
            会话编号：<code>{sessionId}</code>
          </p>
          <p>当前状态：{statusText}</p>
          <p>实时通道：{wsConnected ? "已连接" : "未连接"}</p>
          <p className="preset-row">
            推理权重：
            <select
              value={checkpointPreset}
              onChange={(e) => setCheckpointPreset(e.target.value)}
              disabled={capturing || isAnalyzing}
            >
              <option value="meld_only">meld_only（MELD 单域，推荐 Agent）</option>
              <option value="ap2_m1">ap2_m1（三混合 F1≈0.56）</option>
              <option value="mosei_only">mosei_only（MOSEI 单域，实验）</option>
              <option value="agent_chinese">agent_chinese（中文 BERT 微调）</option>
              <option value="ap4_w005">ap4_w005（DA 预训练）</option>
            </select>
          </p>
        </div>
        <div className="status-chips">
          <span className={`chip ${backendOk ? "ok" : backendOk === false ? "bad" : "warn"}`}>
            {backendOk ? "后端 API 正常" : backendOk === false ? "后端 API 不可达" : "正在检测后端…"}
          </span>
          <span className={`chip ${asrOk ? "ok" : asrOk === false ? "bad" : "warn"}`}>
            {asrOk ? "ASR 服务正常" : asrOk === false ? "ASR 未就绪" : "检测 ASR…"}
          </span>
          <span className={`chip ${llmOk ? "ok" : llmOk === false ? "bad" : "warn"}`}>
            {llmOk ? "LLM 已连接" : llmOk === false ? "LLM 未连接" : "检测 LLM…"}
          </span>
          <span className={`chip ${wsConnected ? "ok" : "bad"}`}>{wsConnected ? "实时通道正常" : "实时通道断开"}</span>
          <span className={`chip ${captureEnabled ? "ok" : "warn"}`}>{captureEnabled ? "采集设备就绪" : "等待设备授权"}</span>
          <span className={`chip ${status === "error" ? "bad" : "info"}`}>流程状态：{statusText}</span>
        </div>
      </section>

      <section className="grid three-col">
        <section className="card preview-card">
          <h2>摄像头预览</h2>
          <div className="media-grid">
            <div>
              <p className="media-title">实时画面</p>
              <video ref={videoRef} autoPlay muted playsInline className="preview-video" />
            </div>
            <div>
              <p className="media-title">采样截图</p>
              {capturedImage ? (
                <img src={capturedImage} alt="captured-frame" className="preview-image" />
              ) : (
                <div className="empty-image">完成一次推理后显示截图</div>
              )}
            </div>
          </div>
          <div className="mic-meter-wrap">
            <span>麦克风电平</span>
            <div className="mic-meter">
              <div
                className={`mic-meter-fill ${micLevel < 20 ? "lv-low" : micLevel < 60 ? "lv-mid" : "lv-high"}`}
                style={{ width: `${micLevel}%` }}
              />
            </div>
            <span>
              {micLevel}（{micLevelState}）
            </span>
          </div>
          {capturing ? (
            <div className="waveform" aria-label="recording-waveform">
              <span style={{ animationDelay: "0ms" }} />
              <span style={{ animationDelay: "120ms" }} />
              <span style={{ animationDelay: "240ms" }} />
              <span style={{ animationDelay: "360ms" }} />
              <span style={{ animationDelay: "480ms" }} />
            </div>
          ) : null}
          {audioInfo ? (
            <p className="hint">
              音频信息：{audioInfo.bytes} bytes，{audioInfo.chunks} chunks，{audioInfo.mimeType}
              {audioInfo.sampleRate ? `, ${audioInfo.sampleRate}Hz` : ""}{" "}
              {audioInfo.durationSec ? `, ${audioInfo.durationSec}s` : ""}{" "}
              {audioInfo.rms !== undefined ? `, rms=${audioInfo.rms}` : ""}{" "}
              {audioInfo.peak !== undefined ? `, peak=${audioInfo.peak}` : ""}
            </p>
          ) : null}
          <p className="hint">设备状态：{captureEnabled ? "就绪" : "未就绪"}</p>
        </section>

        <section className="card control-card">
          <h2>功能操作</h2>
          <div className="actions">
            <label className="field">
              麦克风设备：
              <select
                value={selectedAudioInputId}
                onChange={(e) => setSelectedAudioInputId(e.target.value)}
                disabled={capturing || isAnalyzing}
                style={{ marginLeft: 8 }}
              >
                {audioInputDevices.length === 0 ? (
                  <option value="">未检测到设备（先允许权限）</option>
                ) : null}
                {audioInputDevices.map((d, idx) => (
                  <option key={d.deviceId || idx} value={d.deviceId}>
                    {d.label || `麦克风 ${idx + 1}`}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={refreshAudioDevices} disabled={capturing || isAnalyzing}>
              刷新设备
            </button>
          </div>
          <div className="actions">
            <button onClick={startCapture} disabled={capturing || isAnalyzing} className={capturing ? "btn-active" : ""}>
              {capturing ? "录制中..." : "开始录制"}
            </button>
            <button onClick={stopCaptureAndInfer} disabled={!capturing || isAnalyzing} className={isAnalyzing ? "btn-active" : ""}>
              {isAnalyzing ? (
                <span className="inline-loading">
                  <span className="spinner" /> 分析中...
                </span>
              ) : (
                "结束并推理"
              )}
            </button>
            <button onClick={runMicSelfCheck} disabled={capturing || selfCheckRunning || isAnalyzing}>
              {selfCheckRunning ? "正在自检..." : "麦克风自检（2秒）"}
            </button>
          </div>
          <p className="hint">
            操作说明：点击「开始录制」→ 连续说话（建议 5–{getCaptureMaxSec()} 秒）→ 点击「结束并推理」。
            系统将完整采集（最长 <strong>{getCaptureMaxSec()} 秒</strong>）按 <strong>3 秒</strong>一窗送入模型，并以近端加权聚合最终情绪。
            {isCloudflareQuickTunnel() ? (
              <span className="hint-warn"> Cloudflare 隧道下最长约 10 秒以控制请求超时。</span>
            ) : null}
          </p>
          {captureAbortReason ? <p className="error">采集终止原因：{captureAbortReason}</p> : null}
          <section className="card soft-card">
            <h3>补充说明（可选）</h3>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              placeholder="可输入额外上下文。若识别到语音文本，将优先使用语音内容。"
            />
          </section>
          {selfCheckResult ? (
            <p className="hint">
              自检结果：{selfCheckResult.verdict}，{selfCheckResult.durationSec}s，rms={selfCheckResult.rms}，peak=
              {selfCheckResult.peak}，{selfCheckResult.bytes} bytes
            </p>
          ) : null}
          {selfCheckAudioUrl ? <audio controls src={selfCheckAudioUrl} style={{ marginTop: 8, width: "100%" }} /> : null}
        </section>

        <section className="card result-card">
          <h2>情感结果展示</h2>
          <div className={`emotion-pill ${emotionToneClass}`}>{emotion ? `${emotionLabelText}` : "暂无结果"}</div>
          <div className="metric-row">
            <div className="metric-box">
              <span>置信度</span>
              <strong>{emotion ? Number(emotion.confidence || 0).toFixed(2) : "-"}</strong>
            </div>
            <div className="metric-box">
              <span>Valence</span>
              <strong>{emotion ? Number(emotion.valence || 0).toFixed(2) : "-"}</strong>
            </div>
            <div className="metric-box">
              <span>Arousal</span>
              <strong>{emotion ? Number(emotion.arousal || 0).toFixed(2) : "-"}</strong>
            </div>
          </div>
          <p className="hint">识别文本：{asrText || "暂无识别文本"}</p>
          <p className="hint">
            识别置信度：{Number(asrConfidence || 0).toFixed(2)}
            {asrProvider ? ` · 引擎：${asrProvider}` : ""}
          </p>
          {asrProvider === "mock" ? (
            <p className="error">当前为 mock ASR，显示内容并非真实语音转写。</p>
          ) : null}
          <div className="chat-wrap">
            <div className="chat-bubble">
              {agentReply?.reply_text || "等待助手回复..."}
            </div>
          </div>
          <p className="hint">
            回复来源：{agentReply?.llm_provider || "-"}
            {agentReply?.llm_model ? ` · ${agentReply.llm_model}` : ""}
          </p>
          {emotion?.asr_calibration_applied ? (
            <p className="hint-warn">
              ASR 校正：{emotion.asr_calibration_reason || "已根据中文 ASR 调整"}
              {emotion.model_emotion_label
                ? `（模型原输出 ${emotion.model_emotion_label}）`
                : ""}
            </p>
          ) : null}
          {emotion?.arbitration_source && emotion.arbitration_source !== "model" ? (
            <p className="hint-warn">
              多模态仲裁 [{emotion.arbitration_source}]：{emotion.arbitration_reason || ""}
            </p>
          ) : null}
          {emotion?.is_flat_distribution ? (
            <p className="hint">模型概率分布较扁平（不确定），已结合语音语义综合决策。</p>
          ) : null}
          {emotion?.video_decode_mode ? (
            <p className="hint">视频解码：{emotion.video_decode_mode}</p>
          ) : null}
          {emotion?.emotion_label ? (
            <p className="hint">
              回复依据（情绪模型）：{emotionLabelText} · 置信度 {Number(emotion.confidence || 0).toFixed(2)}
              {emotion?.top_emotions?.length
                ? ` · Top3: ${emotion.top_emotions.map((x) => `${EMOTION_CN[x.label] || x.label}:${Number(x.prob).toFixed(2)}`).join(" / ")}`
                : ""}
            </p>
          ) : null}
          <EmotionTimeline windows={emotion?.temporal_windows} summary={emotion?.temporal_summary} />
          {agentReply?.llm_provider === "template" ? (
            <p className="error">
              当前为模板回复（非生成式 LLM）。{agentReply?.llm_error || "请启动 Ollama 并拉取模型。"}
            </p>
          ) : null}
          {String(emotion?.inference_source || "").startsWith("checkpoint") && asrText && emotion?.emotion_label !== "happy" && /开心|高兴|快乐/.test(asrText) ? (
            <p className="hint pipeline-note">
              说明：ASR 识别到积极语义，但情绪模型输出「{emotionLabelText}」。LLM 回复基于 ASR 文本，与模型分类结果可能不同——这表示模型已真实推理，但在线中文+单帧视频与训练分布存在偏差。
            </p>
          ) : null}
        </section>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>调试信息</h2>
          <button className="ghost-btn" onClick={() => setShowRawDebug((v) => !v)}>
            {showRawDebug ? "收起原始数据" : "查看原始数据"}
          </button>
        </div>
        <div className="metric-row">
          <div className="metric-box">
            <span>当前阶段</span>
            <strong>{debugInfo?.phase || "待命"}</strong>
          </div>
          <div className="metric-box">
            <span>请求编号</span>
            <strong>{debugInfo?.request_id || "-"}</strong>
          </div>
          <div className="metric-box">
            <span>动作次数</span>
            <strong>{debugInfo?.action_count ?? actionCount}</strong>
          </div>
        </div>
        {showRawDebug ? (
          <pre>
            {JSON.stringify(
              debugInfo || {
                tip: "完成一次“开始录制 -> 结束并推理”后，这里会显示调试信息。",
              },
              null,
              2
            )}
          </pre>
        ) : null}
      </section>

      <section className="grid">
        <div className="card">
          <h3>情绪推理输出</h3>
          <div className="result-panel">
            <p className="result-title">{emotion ? `当前情绪：${emotionLabelText}` : "当前情绪：暂无"}</p>
            <div className="metric-row">
              <div className="metric-box">
                <span>置信度</span>
                <strong>{emotion ? Number(emotion.confidence || 0).toFixed(2) : "-"}</strong>
              </div>
              <div className="metric-box">
                <span>Valence</span>
                <strong>{emotion ? Number(emotion.valence || 0).toFixed(2) : "-"}</strong>
              </div>
              <div className="metric-box">
                <span>Arousal</span>
                <strong>{emotion ? Number(emotion.arousal || 0).toFixed(2) : "-"}</strong>
              </div>
            </div>
            <p className="hint">
              推理引擎：{emotion?.inference_source || "-"} / 提供方 {emotion?.model_provider || "-"}
              {isTrainedCheckpoint(emotion?.inference_source) ? "（训练权重）" : ""}
              {emotion?.checkpoint_preset ? ` · 预设 ${emotion.checkpoint_preset}` : ""}
              {emotion?.fusion_strategy ? ` · ${emotion.fusion_strategy}` : ""}
              {emotion?.inference_ms ? ` · ${Number(emotion.inference_ms).toFixed(0)}ms` : ""}
            </p>
            {emotion?.top_emotions?.length ? (
              <p className="hint">
                Top3 概率：
                {emotion.top_emotions
                  .map((x) => `${EMOTION_CN[x.label] || x.label}:${Number(x.prob).toFixed(2)}`)
                  .join(" · ")}
              </p>
            ) : null}
            <EmotionProbBars
              items={emotion?.all_probs_labeled || emotion?.top_emotions?.map((x) => ({ ...x, label_cn: EMOTION_CN[x.label] }))}
              activeLabel={emotion?.emotion_label}
            />
            {isTrainedCheckpoint(emotion?.inference_source) && Number(emotion?.confidence || 0) < 0.45 ? (
              <p className="hint">
                模型对当前片段把握较低（非 mock）；在线摄像头+中文与训练集分布有差异，可多录几秒或填写补充说明。
              </p>
            ) : null}
          </div>
        </div>
        <div className="card">
          <h3>助手回复</h3>
          <div className="result-panel">
            <p className="result-text">{agentReply?.reply_text || "暂无回复内容"}</p>
            <div className="metric-row">
              <div className="metric-box">
                <span>语气</span>
                <strong>{agentReply?.tone || "-"}</strong>
              </div>
              <div className="metric-box">
                <span>安全模式</span>
                <strong>{agentReply?.safe_mode ? "是" : "否"}</strong>
              </div>
              <div className="metric-box">
                <span>回复来源</span>
                <strong>{agentReply?.llm_provider || "-"}</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>原始结果数据</h2>
          <button className="ghost-btn" onClick={() => setShowRawResult((v) => !v)}>
            {showRawResult ? "收起原始 JSON" : "展开原始 JSON"}
          </button>
        </div>
        {showRawResult ? (
          <div className="grid">
            <pre>{JSON.stringify(emotion, null, 2)}</pre>
            <pre>{JSON.stringify(agentReply, null, 2)}</pre>
          </div>
        ) : (
          <p className="hint">默认展示可读结果，若需排查细节可展开原始 JSON。</p>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>流水线监控（数据流转）</h2>
          <span className={`chip ${isTrainedCheckpoint(emotion?.inference_source) ? "chip-ok" : "chip-warn"}`}>
            {isTrainedCheckpoint(emotion?.inference_source)
              ? `真实 checkpoint 推理 (${emotion.inference_source})`
              : emotion?.inference_source || "待推理"}
          </span>
        </div>
        {emotion?.pipeline_trace?.temporal?.enabled ? (
          <p className="hint">
            长时模式：{emotion.pipeline_trace.temporal.num_windows} 窗 ×{" "}
            {emotion.pipeline_trace.temporal.window_sec}s，全长{" "}
            {emotion.pipeline_trace.temporal.total_duration_sec}s，聚合{" "}
            {emotion.pipeline_trace.temporal.aggregation || "recency_weighted"}
          </p>
        ) : null}
        {emotion?.pipeline_trace?.steps?.length ? (
          <ol className="pipeline-steps">
            {emotion.pipeline_trace.steps.map((step) => (
              <li key={step.stage} className={`pipeline-step step-${step.status}`}>
                <div className="pipeline-step-head">
                  <strong>{step.stage}</strong>
                  <span className={`chip chip-${step.status === "ok" ? "ok" : "warn"}`}>{step.status}</span>
                </div>
                {formatPipelineStep(step).map((line) => (
                  <p key={line} className="hint pipeline-step-line">
                    {line}
                  </p>
                ))}
              </li>
            ))}
          </ol>
        ) : (
          <p className="hint">完成一次推理后，此处展示 ingest → ASR → 文本合并 → 情绪模型 四步 trace。</p>
        )}
        <ModalityCards trace={emotion?.pipeline_trace} />
        {emotion?.pipeline_trace?.model ? (
          <div className="model-call-box">
            <p>
              <strong>模型调用：</strong>
              {emotion.pipeline_trace.model.called ? "已调用 MultimodalEmotionModel.forward()" : "未调用"}
              {" · "}
              mock={String(emotion.pipeline_trace.model.is_mock ?? true)}
              {" · "}
              device={emotion.pipeline_trace.model.device || "-"}
              {" · "}
              {emotion.pipeline_trace.model.inference_ms ?? emotion.inference_ms}ms
            </p>
            <p className="hint">
              checkpoint: {emotion.checkpoint_file || emotion.pipeline_trace.model.checkpoint_file || "-"}
              {" · "}
              fusion: {emotion.fusion_strategy || "-"}
            </p>
          </div>
        ) : null}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>运行日志</h2>
          <button className="ghost-btn" type="button" onClick={() => setRuntimeLogs([])}>
            清空
          </button>
        </div>
        <div className="runtime-log-panel">
          {runtimeLogs.length === 0 ? (
            <p className="hint">系统运行日志将在此实时显示（健康检查、采集、上传、模型推理、LLM 回复）。</p>
          ) : (
            runtimeLogs
              .slice()
              .reverse()
              .map((log, idx) => (
                <div key={`${log.ts}-${idx}`} className={`runtime-log-item log-${log.level}`}>
                  <span className="runtime-log-time">{formatLogTime(log.ts)}</span>
                  <span className="runtime-log-msg">{log.message}</span>
                  {log.data ? (
                    <details>
                      <summary>详情</summary>
                      <pre>{JSON.stringify(log.data, null, 2)}</pre>
                    </details>
                  ) : null}
                </div>
              ))
          )}
        </div>
      </section>

      <section className="card">
        <h2>最近事件（实时通道）</h2>
        <ul className="timeline">
          {shortEvents.map((evt, idx) => (
            <li key={idx} className="timeline-item">
              <span className="timeline-dot" />
              <div className="timeline-body">
                <div className="timeline-head">
                  <strong>{evt?.event || "event"}</strong>
                  <span className="timeline-index">#{shortEvents.length - idx}</span>
                </div>
                <p>{summarizeEvent(evt)}</p>
                <details>
                  <summary>查看原始事件</summary>
                  <code>{JSON.stringify(evt)}</code>
                </details>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
