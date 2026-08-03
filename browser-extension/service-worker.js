const OFFSCREEN_DOCUMENT = 'offscreen.html';
const SUPPORTED_HOSTS = new Set([
  'youtube.com',
  'www.youtube.com',
  'm.youtube.com',
  'music.youtube.com',
  'youtu.be',
  'twitch.tv',
  'www.twitch.tv',
]);
const UI_STATES = {
  inactive: {
    icon: 'gray',
    title: 'Enable realtime translation for this tab',
  },
  webuiOffline: {
    icon: 'orange',
    title: 'Tab selected; Realtime Translator WebUI is offline',
  },
  translatorIdle: {
    icon: 'blue',
    title: 'Tab selected; translator is not running',
  },
  translating: {
    icon: 'green',
    title: 'Realtime translation is active',
  },
};

let captureTabId = null;
let creatingOffscreen = null;
let uiUpdateChain = Promise.resolve();
let uiStateRevision = 0;
let serviceState = {
  webuiAvailable: false,
  backendConnected: false,
  runtimeState: 'unavailable',
};

const stateReady = Promise.all([
  chrome.storage.session.get('captureTabId'),
  chrome.tabCapture.getCapturedTabs(),
]).then(async ([stored, capturedTabs]) => {
  const storedTabId = Number.isInteger(stored.captureTabId)
    ? stored.captureTabId
    : null;
  const captureIsAlive = storedTabId !== null && capturedTabs.some(info =>
    info.tabId === storedTabId
      && (info.status === 'active' || info.status === 'pending')
  );
  captureTabId = captureIsAlive ? storedTabId : null;
  if (!captureIsAlive) {
    await chrome.storage.session.remove('captureTabId');
  }
});

function isSupportedTab(tab) {
  if (!tab?.url) {
    return false;
  }
  try {
    return SUPPORTED_HOSTS.has(new URL(tab.url).hostname.toLowerCase());
  } catch (_) {
    return false;
  }
}

function currentUiState(tabId) {
  if (tabId !== captureTabId) {
    return 'inactive';
  }
  if (!serviceState.webuiAvailable) {
    return 'webuiOffline';
  }
  if (serviceState.runtimeState !== 'running'
      || !serviceState.backendConnected) {
    return 'translatorIdle';
  }
  return 'translating';
}

async function updateActionState(tabId) {
  if (!Number.isInteger(tabId)) {
    return;
  }
  const revision = uiStateRevision;
  const stateName = currentUiState(tabId);
  const state = UI_STATES[stateName];
  await Promise.all([
    chrome.action.setIcon({
      tabId,
      path: {
        16: `icons/power-${state.icon}-16.png`,
        32: `icons/power-${state.icon}-32.png`,
      },
    }),
    chrome.action.setBadgeText({ tabId, text: '' }),
    chrome.action.setTitle({ tabId, title: state.title }),
  ]);
  if (revision !== uiStateRevision) {
    await updateActionState(tabId);
  }
}

function scheduleCapturedUiUpdate(tabId, notifyPage = true) {
  uiUpdateChain = uiUpdateChain
    .catch(() => {})
    .then(async () => {
      if (tabId !== captureTabId) {
        return;
      }
      await updateActionState(tabId);
      if (notifyPage) {
        await notifyCaptureState(tabId, true);
      }
    });
  return uiUpdateChain;
}

async function notifyCaptureState(tabId, active, error = '') {
  if (!Number.isInteger(tabId)) {
    return;
  }
  try {
    await chrome.tabs.sendMessage(tabId, {
      type: 'CAPTURE_STATE',
      active,
      error,
      ...serviceState,
    });
  } catch (_) {}
}

async function showActionError(tabId, error) {
  if (!Number.isInteger(tabId)) {
    return;
  }
  await chrome.action.setBadgeText({ tabId, text: '!' });
  await chrome.action.setBadgeBackgroundColor({
    tabId,
    color: '#b94a55',
  });
  await chrome.action.setTitle({
    tabId,
    title: `Realtime Translator: ${error}`,
  });
  await notifyCaptureState(tabId, false, error);
}

async function ensureOffscreenDocument() {
  const url = chrome.runtime.getURL(OFFSCREEN_DOCUMENT);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [url],
  });
  if (contexts.length) {
    return;
  }
  if (!creatingOffscreen) {
    creatingOffscreen = chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT,
      reasons: ['USER_MEDIA'],
      justification: 'Keep the selected tab media stream available for the local translator.',
    }).finally(() => {
      creatingOffscreen = null;
    });
  }
  await creatingOffscreen;
}

async function stopCapture() {
  await stateReady;
  const previousTabId = captureTabId;
  captureTabId = null;
  serviceState = {
    webuiAvailable: false,
    backendConnected: false,
    runtimeState: 'unavailable',
  };
  uiStateRevision += 1;
  await chrome.storage.session.remove('captureTabId');
  try {
    await chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'STOP_CAPTURE',
    });
  } catch (_) {}
  if (Number.isInteger(previousTabId)) {
    await updateActionState(previousTabId);
    await notifyCaptureState(previousTabId, false);
  }
}

async function startCapture(tab, streamId) {
  await stateReady;
  if (!isSupportedTab(tab)) {
    throw new Error('Open a YouTube or Twitch tab first.');
  }
  if (!streamId) {
    throw new Error('Chromium did not grant tab audio capture.');
  }
  if (captureTabId !== null) {
    await stopCapture();
  }

  await ensureOffscreenDocument();
  const response = await chrome.runtime.sendMessage({
    target: 'offscreen',
    type: 'START_CAPTURE',
    streamId,
    tabId: tab.id,
  });
  if (!response?.ok) {
    throw new Error(response?.error || 'Failed to start tab capture.');
  }
  captureTabId = tab.id;
  serviceState = {
    webuiAvailable: false,
    backendConnected: false,
    runtimeState: 'unavailable',
  };
  uiStateRevision += 1;
  await chrome.storage.session.set({ captureTabId });
  await updateActionState(tab.id);
  await notifyCaptureState(tab.id, true);
  return true;
}

function pageCaptureError(error) {
  const message = String(error?.message || error);
  if (/not been invoked|activeTabPermission|activeTab permission/i.test(message)) {
    return (
      'Click the Realtime Translator extension button in the browser toolbar '
      + 'while this tab is active. You only need to do this once per tab.'
    );
  }
  return message;
}

async function toggleCaptureFromPage(tab) {
  await stateReady;
  if (!Number.isInteger(tab?.id)) {
    throw new Error('No active browser tab was found.');
  }
  if (!isSupportedTab(tab)) {
    throw new Error('Open a YouTube or Twitch tab first.');
  }
  if (captureTabId === tab.id) {
    await stopCapture();
    return false;
  }

  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
  });
  return startCapture(tab, streamId);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.target === 'offscreen') {
    return false;
  }
  if (message?.type === 'GET_CAPTURE_STATE') {
    stateReady.then(() => {
      const requestedTabId = Number.isInteger(message.tabId)
        ? message.tabId
        : sender.tab?.id;
      sendResponse({
        active: Number.isInteger(requestedTabId)
          && requestedTabId === captureTabId,
        captureTabId,
        ...serviceState,
      });
    });
    return true;
  }
  if (message?.type === 'TOGGLE_CAPTURE' && Number.isInteger(sender.tab?.id)) {
    toggleCaptureFromPage(sender.tab)
      .then(active => sendResponse({ ok: true, active }))
      .catch(error => sendResponse({
        ok: false,
        error: pageCaptureError(error),
      }));
    return true;
  }
  if (message?.type === 'START_CAPTURE_WITH_STREAM'
      && Number.isInteger(message.tabId)) {
    chrome.tabs.get(message.tabId)
      .then(tab => startCapture(tab, message.streamId))
      .then(active => sendResponse({ ok: true, active }))
      .catch(async error => {
        const text = String(error?.message || error);
        await showActionError(message.tabId, text);
        sendResponse({ ok: false, active: false, error: text });
      });
    return true;
  }
  if (message?.type === 'STOP_CAPTURE_REQUEST') {
    stopCapture()
      .then(() => sendResponse({ ok: true, active: false }))
      .catch(error => sendResponse({
        ok: false,
        error: String(error?.message || error),
      }));
    return true;
  }
  if (message?.type === 'POPUP_CAPTURE_ERROR') {
    const tabId = Number.isInteger(message.tabId)
      ? message.tabId
      : captureTabId;
    showActionError(tabId, String(message.error || 'Capture failed'))
      .catch(() => {});
  }
  if (message?.type === 'CAPTURE_STOPPED') {
    stopCapture().catch(() => {});
  }
  if (message?.type === 'SERVICE_STATE'
      && Number.isInteger(message.tabId)
      && message.tabId === captureTabId) {
    const nextServiceState = {
      webuiAvailable: Boolean(message.webuiAvailable),
      backendConnected: Boolean(message.backendConnected),
      runtimeState: String(message.runtimeState || 'unavailable'),
    };
    const stateChanged = (
      nextServiceState.webuiAvailable !== serviceState.webuiAvailable
      || nextServiceState.backendConnected !== serviceState.backendConnected
      || nextServiceState.runtimeState !== serviceState.runtimeState
    );
    serviceState = nextServiceState;
    if (stateChanged) {
      uiStateRevision += 1;
    }
    scheduleCapturedUiUpdate(
      message.tabId,
      stateChanged || !message.heartbeat,
    ).catch(() => {});
  }
  return false;
});

chrome.tabs.onActivated.addListener(({ tabId }) => {
  stateReady.then(() => updateActionState(tabId)).catch(() => {});
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'complete') {
    stateReady.then(() => {
      updateActionState(tabId);
      if (tabId === captureTabId) {
        notifyCaptureState(tabId, true);
      }
    }).catch(() => {});
  }
});

chrome.tabs.onRemoved.addListener(tabId => {
  if (tabId === captureTabId) {
    stopCapture().catch(() => {});
  }
});

chrome.tabCapture.onStatusChanged.addListener(info => {
  if (info.tabId === captureTabId
      && (info.status === 'stopped' || info.status === 'error')) {
    stopCapture().catch(() => {});
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.tabs.query({}).then(tabs => Promise.all(
    tabs.filter(tab => Number.isInteger(tab.id))
      .map(tab => updateActionState(tab.id))
  )).catch(() => {});
});

chrome.runtime.onStartup.addListener(async () => {
  await stateReady;
  if (captureTabId !== null) {
    await stopCapture();
  }
});
