let targetTabId = null;

function friendlyCaptureError(error) {
  const message = String(error?.message || error);
  if (/not been invoked|activeTabPermission|activeTab permission/i.test(message)) {
    return (
      'Click the Realtime Translator extension button in the browser toolbar '
      + 'while this tab is active. You only need to do this once per tab.'
    );
  }
  return message;
}

async function resolveTargetTab() {
  const stored = await chrome.storage.session.get('pendingPopupTarget');
  const pending = stored.pendingPopupTarget;
  await chrome.storage.session.remove('pendingPopupTarget');

  if (Number.isInteger(pending?.tabId)
      && Date.now() - Number(pending.createdAt || 0) < 5000) {
    targetTabId = pending.tabId;
    return chrome.tabs.get(targetTabId);
  }

  const [activeTab] = await chrome.tabs.query({
    active: true,
    currentWindow: true,
  });
  targetTabId = activeTab?.id ?? null;
  return activeTab;
}

async function toggleCapture() {
  const tab = await resolveTargetTab();
  if (!Number.isInteger(tab?.id)) {
    throw new Error('No active browser tab was found.');
  }

  const state = await chrome.runtime.sendMessage({
    type: 'GET_CAPTURE_STATE',
    tabId: tab.id,
  });
  if (state?.active) {
    return chrome.runtime.sendMessage({
      type: 'STOP_CAPTURE_REQUEST',
      tabId: tab.id,
    });
  }
  if (Number.isInteger(state?.captureTabId)) {
    await chrome.runtime.sendMessage({
      type: 'STOP_CAPTURE_REQUEST',
      tabId: state.captureTabId,
    });
  }

  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
  });
  return chrome.runtime.sendMessage({
    type: 'START_CAPTURE_WITH_STREAM',
    tabId: tab.id,
    streamId,
  });
}

toggleCapture()
  .catch(async error => {
    const message = friendlyCaptureError(error);
    await chrome.runtime.sendMessage({
      type: 'POPUP_CAPTURE_ERROR',
      tabId: targetTabId,
      error: message,
    }).catch(() => {});
  })
  .finally(() => window.close());
