const AUDIO_SOCKET_URL = 'ws://127.0.0.1:8766/audio';
const WEBUI_PORT_START = 7860;
const WEBUI_PORT_ATTEMPTS = 40;
const WEBUI_DISCOVERY_INTERVAL_MS = 2400;
const CAPTURE_TAKEN_OVER_CLOSE_CODE = 4001;

let mediaStream = null;
let audioContext = null;
let workletNode = null;
let audioSocket = null;
let reconnectTimer = null;
let reconnectDelayMs = 1000;
let captureTabId = null;
let captureSessionStartedAt = 0;
let webuiPollTimer = null;
let webuiAvailable = false;
let backendConnected = false;
let runtimeState = 'unavailable';
let lastPublishedState = '';
let webuiFailureCount = 0;
let webuiPollCount = 0;
let webuiPort = null;
let webuiPollInFlight = false;
let webuiLastDiscoveryAt = 0;

async function fetchWebuiStatus(port, timeoutMs = 500) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(
      `http://127.0.0.1:${port}/api/status?client=extension`,
      {
        cache: 'no-store',
        signal: controller.signal,
      },
    );
    if (!response.ok) {
      return null;
    }
    const payload = await response.json();
    return payload?.service === 'realtime-translator' ? payload : null;
  } catch (_) {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function discoverWebuiStatus() {
  if (Number.isInteger(webuiPort)) {
    const payload = await fetchWebuiStatus(webuiPort);
    if (payload) {
      return payload;
    }
    webuiPort = null;
  }

  const now = Date.now();
  if (now - webuiLastDiscoveryAt < WEBUI_DISCOVERY_INTERVAL_MS) {
    return null;
  }
  webuiLastDiscoveryAt = now;

  const candidates = Array.from(
    { length: WEBUI_PORT_ATTEMPTS },
    (_, index) => WEBUI_PORT_START + index,
  );
  const results = await Promise.all(
    candidates.map(port => fetchWebuiStatus(port)),
  );
  const index = results.findIndex(Boolean);
  if (index < 0) {
    return null;
  }
  webuiPort = candidates[index];
  return results[index];
}

function publishServiceState(force = false) {
  if (!Number.isInteger(captureTabId)) {
    return;
  }
  const signature = JSON.stringify({
    tabId: captureTabId,
    webuiAvailable,
    backendConnected,
    runtimeState,
  });
  if (!force && signature === lastPublishedState) {
    return;
  }
  lastPublishedState = signature;
  chrome.runtime.sendMessage({
    type: 'SERVICE_STATE',
    tabId: captureTabId,
    webuiAvailable,
    backendConnected,
    runtimeState,
    heartbeat: force,
  }).catch(() => {});
}

async function pollWebuiStatus() {
  if (!mediaStream || webuiPollInFlight) {
    return;
  }
  webuiPollInFlight = true;
  try {
    const payload = await discoverWebuiStatus();
    if (!payload) {
      throw new Error('Realtime Translator WebUI is unavailable');
    }
    webuiFailureCount = 0;
    webuiAvailable = true;
    runtimeState = String(payload?.runtime?.state || 'idle');
  } catch (_) {
    webuiFailureCount += 1;
    if (!webuiAvailable || webuiFailureCount >= 2) {
      webuiAvailable = false;
      runtimeState = 'unavailable';
    }
  } finally {
    webuiPollInFlight = false;
    webuiPollCount += 1;
    publishServiceState(webuiPollCount % 4 === 0);
  }
}

function startWebuiPolling() {
  clearInterval(webuiPollTimer);
  pollWebuiStatus().catch(() => {});
  webuiPollTimer = setInterval(() => {
    pollWebuiStatus().catch(() => {});
  }, 1200);
}

function sendStreamMetadata() {
  if (audioSocket?.readyState !== WebSocket.OPEN || !audioContext) {
    return;
  }
  audioSocket.send(JSON.stringify({
    type: 'stream_start',
    format: 'f32le',
    sample_rate: audioContext.sampleRate,
    channels: 1,
  }));
}

function connectAudioSocket() {
  if (!mediaStream || audioSocket?.readyState === WebSocket.OPEN
      || audioSocket?.readyState === WebSocket.CONNECTING) {
    return;
  }
  clearTimeout(reconnectTimer);
  const socketUrl = `${AUDIO_SOCKET_URL}?session_started_at=${captureSessionStartedAt}`;
  audioSocket = new WebSocket(socketUrl);
  audioSocket.binaryType = 'arraybuffer';
  audioSocket.onopen = () => {
    reconnectDelayMs = 1000;
    backendConnected = true;
    sendStreamMetadata();
    publishServiceState();
  };
  audioSocket.onclose = event => {
    audioSocket = null;
    backendConnected = false;
    publishServiceState();
    if (event.code === CAPTURE_TAKEN_OVER_CLOSE_CODE) {
      stopCapture(true).catch(() => {});
      return;
    }
    if (mediaStream) {
      reconnectTimer = setTimeout(
        connectAudioSocket,
        reconnectDelayMs,
      );
      reconnectDelayMs = Math.min(10000, reconnectDelayMs * 2);
    }
  };
  audioSocket.onerror = () => {
    audioSocket?.close();
  };
}

function sendPCM(sequence, samplesBuffer) {
  if (audioSocket?.readyState !== WebSocket.OPEN) {
    return;
  }
  const frame = new ArrayBuffer(4 + samplesBuffer.byteLength);
  new DataView(frame).setUint32(0, sequence, true);
  new Uint8Array(frame, 4).set(new Uint8Array(samplesBuffer));
  audioSocket.send(frame);
}

async function stopCapture(notify = false) {
  clearTimeout(reconnectTimer);
  clearInterval(webuiPollTimer);
  reconnectTimer = null;
  webuiPollTimer = null;
  reconnectDelayMs = 1000;
  workletNode?.disconnect();
  workletNode = null;
  for (const track of mediaStream?.getTracks() || []) {
    track.stop();
  }
  mediaStream = null;
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
  audioSocket?.close(1000, 'Capture stopped');
  audioSocket = null;
  const stoppedTabId = captureTabId;
  captureTabId = null;
  captureSessionStartedAt = 0;
  webuiAvailable = false;
  webuiFailureCount = 0;
  webuiPollCount = 0;
  webuiPort = null;
  webuiPollInFlight = false;
  webuiLastDiscoveryAt = 0;
  backendConnected = false;
  runtimeState = 'unavailable';
  lastPublishedState = '';
  if (notify && stoppedTabId !== null) {
    chrome.runtime.sendMessage({
      type: 'CAPTURE_STOPPED',
      tabId: stoppedTabId,
    }).catch(() => {});
  }
}

async function startCapture(streamId, tabId) {
  await stopCapture(false);
  captureTabId = tabId;
  captureSessionStartedAt = Date.now();
  publishServiceState(true);
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });
  mediaStream.getAudioTracks()[0]?.addEventListener('ended', () => {
    stopCapture(true).catch(() => {});
  }, { once: true });

  audioContext = new AudioContext({
    sampleRate: 48000,
    latencyHint: 'interactive',
  });
  await audioContext.audioWorklet.addModule('audio-worklet.js');
  const source = audioContext.createMediaStreamSource(mediaStream);

  // tabCapture suppresses normal tab playback, so route it back to speakers.
  source.connect(audioContext.destination);

  workletNode = new AudioWorkletNode(
    audioContext,
    'pcm-chunk-processor',
    {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { chunkMs: 40 },
    },
  );
  const silentOutput = audioContext.createGain();
  silentOutput.gain.value = 0;
  source.connect(workletNode);
  workletNode.connect(silentOutput);
  silentOutput.connect(audioContext.destination);
  workletNode.port.onmessage = event => {
    if (event.data?.type === 'pcm') {
      sendPCM(event.data.sequence, event.data.samples);
    }
  };
  await audioContext.resume();
  startWebuiPolling();
  connectAudioSocket();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== 'offscreen') {
    return false;
  }
  if (message.type === 'START_CAPTURE') {
    startCapture(message.streamId, message.tabId)
      .then(() => sendResponse({ ok: true }))
      .catch(error => {
        stopCapture(true).catch(() => {});
        sendResponse({
          ok: false,
          error: String(error?.message || error),
        });
      });
    return true;
  }
  if (message.type === 'STOP_CAPTURE') {
    stopCapture(false)
      .then(() => sendResponse({ ok: true }))
      .catch(error => sendResponse({
        ok: false,
        error: String(error?.message || error),
      }));
    return true;
  }
  return false;
});
