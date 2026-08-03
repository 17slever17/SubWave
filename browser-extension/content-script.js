(function () {
    'use strict';

    if (window.top !== window.self) {
        return;
    }

    const BRIDGE_BASE = 'http://127.0.0.1:8765';
    const STATUS_URL = `${BRIDGE_BASE}/overlay_status`;
    const SUBTITLES_URL = `${BRIDGE_BASE}/overlay_subtitles`;
    const TAB_ID_KEY = `realtime_translator_tab_id:${window.location.hostname}`;

    let isCapturing = false;
    let webuiAvailable = false;
    let backendConnected = false;
    let runtimeState = 'unavailable';
    let tabId = sessionStorage.getItem(TAB_ID_KEY);
    let lastSubtitleSeq = -1;
    let lastMode = 'desktop';

    if (!tabId) {
        tabId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        sessionStorage.setItem(TAB_ID_KEY, tabId);
    }

    const audioButton = document.createElement('button');
    audioButton.id = 'audio-redirect-indicator';
    audioButton.type = 'button';
    audioButton.setAttribute('aria-label', 'Toggle realtime tab translation');

    const powerIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    powerIcon.setAttribute('width', '24');
    powerIcon.setAttribute('height', '24');
    powerIcon.setAttribute('viewBox', '0 0 24 24');
    powerIcon.setAttribute('fill', 'none');
    powerIcon.setAttribute('stroke', 'currentColor');
    powerIcon.setAttribute('stroke-width', '2');
    powerIcon.setAttribute('stroke-linecap', 'round');
    powerIcon.setAttribute('stroke-linejoin', 'round');
    powerIcon.setAttribute('aria-hidden', 'true');
    for (const [tag, attributes] of [
        ['circle', { cx: '12', cy: '12', r: '10' }],
        ['path', { d: 'M12 7v4' }],
        ['path', { d: 'M7.998 9.003a5 5 0 1 0 8-.005' }]
    ]) {
        const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [name, value] of Object.entries(attributes)) {
            element.setAttribute(name, value);
        }
        powerIcon.appendChild(element);
    }
    audioButton.appendChild(powerIcon);

    let overlayHost = null;
    let overlayRoot = null;
    let overlayBubble = null;
    let overlayPrev = null;
    let overlayCurr = null;
    let dragState = null;
    let lastRenderedPrevious = '';
    let lastRenderedTranslation = '';
    let lastRenderedShowPrev = false;
    let latestSubtitlePayload = null;
    let captureNotice = null;
    const DRAG_EDGE_PX = 12;
    const PLAYER_EDGE_MARGIN_PX = 14;

    function showCaptureNotice(message, isError = false) {
        if (!captureNotice) {
            captureNotice = document.createElement('div');
            captureNotice.id = 'rt-capture-notice';
            captureNotice.style.cssText = `
                position: fixed;
                top: 72px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 2147483647;
                max-width: min(520px, calc(100vw - 32px));
                padding: 10px 16px;
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 12px;
                background: rgba(28, 24, 32, 0.94);
                color: #f4edf5;
                box-shadow: 0 12px 34px rgba(0, 0, 0, 0.38);
                font: 600 13px/1.4 "Segoe UI", sans-serif;
                text-align: center;
                pointer-events: none;
                opacity: 0;
                transition: opacity 160ms ease;
            `;
        }
        if (!captureNotice.isConnected) {
            document.documentElement.appendChild(captureNotice);
        }
        captureNotice.textContent = message;
        captureNotice.style.borderColor = isError
            ? 'rgba(224, 112, 126, 0.6)'
            : 'rgba(62, 166, 255, 0.45)';
        captureNotice.style.opacity = '1';
        clearTimeout(showCaptureNotice.hideTimer);
        showCaptureNotice.hideTimer = setTimeout(() => {
            if (captureNotice) {
                captureNotice.style.opacity = '0';
            }
        }, 5000);
    }

    audioButton.style.cssText = `
        all: unset;
        box-sizing: border-box;
        width: 40px;
        height: 40px;
        flex: 0 0 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        align-self: center;
        vertical-align: middle;
        border-radius: 50%;
        cursor: pointer;
        color: #8f8f8f;
        background: transparent;
        transition: color 180ms ease, filter 180ms ease, background-color 180ms ease, transform 120ms ease;
    `;
    powerIcon.style.cssText = 'display: block; width: 24px; height: 24px; pointer-events: none;';

    const styleAudioButton = () => {
        let color = '#8f8f8f';
        let filter = 'none';
        let title = 'Enable realtime translation for this tab';

        if (isCapturing && !webuiAvailable) {
            color = '#f06443';
            filter = 'drop-shadow(0 0 3px rgba(240, 100, 67, 0.95)) drop-shadow(0 0 8px rgba(240, 100, 67, 0.60))';
            title = 'Tab selected; Realtime Translator WebUI is offline';
        } else if (isCapturing
                   && (runtimeState !== 'running' || !backendConnected)) {
            color = '#3ea6ff';
            filter = 'drop-shadow(0 0 3px rgba(62, 166, 255, 0.95)) drop-shadow(0 0 8px rgba(62, 166, 255, 0.65))';
            title = 'Tab selected; translator is not running';
        } else if (isCapturing) {
            color = '#46c878';
            filter = 'drop-shadow(0 0 3px rgba(70, 200, 120, 0.95)) drop-shadow(0 0 8px rgba(70, 200, 120, 0.65))';
            title = 'Realtime translation is active';
        }

        audioButton.setAttribute('aria-pressed', isCapturing ? 'true' : 'false');
        audioButton.dataset.realtimeState = !isCapturing
            ? 'inactive'
            : !webuiAvailable
                ? 'webui-offline'
                : runtimeState !== 'running' || !backendConnected
                    ? 'translator-idle'
                    : 'translating';
        audioButton.title = title;
        audioButton.style.color = color;
        audioButton.style.filter = filter;
    };

    function placeAudioButton(parent, beforeElement = null) {
        const isAlreadyPlaced = audioButton.parentElement === parent
            && (beforeElement ? audioButton.nextSibling === beforeElement : audioButton === parent.lastElementChild);
        if (!isAlreadyPlaced) {
            parent.insertBefore(audioButton, beforeElement);
        }
        audioButton.style.position = 'relative';
        audioButton.style.inset = 'auto';
        audioButton.style.zIndex = 'auto';
        audioButton.style.margin = '0 4px';
    }

    function mountAudioButton() {
        const hostname = window.location.hostname;
        if (hostname.includes('youtube') || hostname === 'youtu.be') {
            const buttons = document.querySelector('ytd-masthead #buttons, #masthead-container #buttons');
            if (buttons?.parentElement) {
                placeAudioButton(buttons.parentElement, buttons);
                return;
            }
            const container = document.querySelector('ytd-masthead #container, #masthead-container #container');
            if (container) {
                placeAudioButton(container);
                return;
            }
        }

        if (hostname.includes('twitch.tv')) {
            const topNavigation = document.querySelector('.top-nav__menu');
            const exactTarget = topNavigation?.querySelector(':scope > .bZYcrx');
            const stableTarget = exactTarget || Array.from(topNavigation?.children || []).find(
                element => element.querySelector?.('[data-a-target="user-menu-toggle"]')
            );
            if (stableTarget?.parentElement) {
                placeAudioButton(stableTarget.parentElement, stableTarget);
                return;
            }
            if (topNavigation) {
                placeAudioButton(topNavigation);
                return;
            }
        }

        if (audioButton.parentElement !== document.documentElement) {
            document.documentElement.appendChild(audioButton);
        }
        audioButton.style.position = 'fixed';
        audioButton.style.top = '8px';
        audioButton.style.right = '12px';
        audioButton.style.zIndex = '2147483647';
        audioButton.style.margin = '0';
    }

    function ensurePageOverlay() {
        if (overlayHost && overlayHost.isConnected) {
            return;
        }

        overlayHost = document.createElement('div');
        overlayHost.id = 'rt-browser-overlay-host';
        overlayHost.style.cssText = `
            position: absolute;
            inset: 0;
            z-index: 1000;
            pointer-events: none;
            display: none;
            overflow: visible;
        `;

        overlayRoot = overlayHost.attachShadow({ mode: 'open' });
        const style = document.createElement('style');
        style.textContent = `
            :host {
                position: absolute;
                inset: 0;
                overflow: visible;
            }
            .dock {
                position: absolute;
                left: 50%;
                bottom: var(--rt-bottom, 78px);
                transform: translateX(-50%);
                pointer-events: none;
            }
            .bubble {
                max-width: min(72vw, 1000px);
                min-width: 280px;
                padding: 6px 18px 8px;
                border-radius: 18px;
                background: rgba(17, 22, 31, 0.8);
                color: #f5f5f5;
                backdrop-filter: blur(5px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.32);
                font-family: "Segoe UI", sans-serif;
                line-height: 1.25;
                text-align: center;
                pointer-events: auto;
                user-select: text;
                -webkit-user-select: text;
                cursor: default;
                touch-action: none;
                transform: translate(var(--rt-x, 0px), var(--rt-y, 0px));
            }
            .prev {
                color: #b8b8b8;
                font-size: 15px;
                margin-bottom: 2px;
                white-space: pre-wrap;
                word-break: break-word;
                user-select: text;
            }
            .curr {
                color: #f5f5f5;
                font-size: 23px;
                font-weight: 600;
                white-space: pre-wrap;
                word-break: break-word;
                user-select: text;
            }
            .hidden {
                display: none;
            }
        `;
        overlayRoot.appendChild(style);

        const dock = document.createElement('div');
        dock.className = 'dock';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        const prev = document.createElement('div');
        prev.className = 'prev hidden';
        const curr = document.createElement('div');
        curr.className = 'curr';

        bubble.appendChild(prev);
        bubble.appendChild(curr);
        dock.appendChild(bubble);
        overlayRoot.appendChild(dock);

        overlayBubble = bubble;
        overlayPrev = prev;
        overlayCurr = curr;
        // The player can briefly disappear during navigation or layout changes.
        // A replacement overlay must not inherit render caches from the old DOM.
        lastRenderedPrevious = '';
        lastRenderedTranslation = '';
        lastRenderedShowPrev = false;
        installOverlayDrag();
        attachOverlayToPlayer();
    }

    function getPlayerContainer() {
        const host = window.location.hostname;
        if (host.includes('youtube') || host === 'youtu.be') {
            const moviePlayer = document.querySelector('#movie_player');
            const fullscreenElement = document.fullscreenElement;
            if (
                moviePlayer instanceof HTMLElement
                && fullscreenElement instanceof HTMLElement
                && (
                    fullscreenElement === moviePlayer
                    || fullscreenElement.contains(moviePlayer)
                )
            ) {
                return moviePlayer;
            }

            const youtubeSelectors = [
                'ytd-player#ytd-player',
                '#player-container-inner > #player > ytd-player',
                '#shorts-player',
                '#movie_player',
                '.html5-video-container:not(#inline-player *)',
            ];
            for (const selector of youtubeSelectors) {
                const node = document.querySelector(selector);
                if (node instanceof HTMLElement) {
                    return node;
                }
            }
        }

        const selectors = host.includes('twitch.tv')
            ? ['.video-ref', 'main > div > section > div > div > div', '[data-a-target="video-player"]']
            : ['.player-container'];

        for (const selector of selectors) {
            const node = document.querySelector(selector);
            if (node instanceof HTMLElement) {
                return node;
            }
        }

        const video = document.querySelector('video');
        if (video instanceof HTMLVideoElement) {
            return video.parentElement;
        }
        return null;
    }

    function attachOverlayToPlayer() {
        ensurePageOverlayBaseState();
        const mount = getPlayerContainer();
        if (!mount) {
            if (overlayHost?.isConnected) {
                overlayHost.remove();
            }
            return;
        }
        const computed = window.getComputedStyle(mount);
        if (computed.position === 'static') {
            mount.style.position = 'relative';
        }
        if (overlayHost.parentElement !== mount) {
            mount.appendChild(overlayHost);
        }
        updateDockBottom(mount);
        clampOverlayPosition(mount);
    }

    function ensurePageOverlayBaseState() {
        if (!overlayHost) {
            return;
        }
        if (!overlayBubble) {
            overlayBubble = overlayRoot.querySelector('.bubble');
        }
    }

    function installOverlayDrag() {
        if (!overlayBubble) {
            return;
        }
        const isDragZone = (event) => {
            if (!overlayBubble) {
                return false;
            }
            const rect = overlayBubble.getBoundingClientRect();
            return (
                (event.clientY - rect.top) <= DRAG_EDGE_PX ||
                (rect.bottom - event.clientY) <= DRAG_EDGE_PX ||
                (event.clientX - rect.left) <= DRAG_EDGE_PX ||
                (rect.right - event.clientX) <= DRAG_EDGE_PX
            );
        };

        overlayBubble.addEventListener('pointerdown', (event) => {
            if (event.button !== 0) {
                return;
            }
            if (!isDragZone(event)) {
                return;
            }
            dragState = {
                startX: event.clientX,
                startY: event.clientY,
                x: Number(overlayBubble?.dataset.rtX || 0),
                y: Number(overlayBubble?.dataset.rtY || 0),
            };
            overlayBubble.setPointerCapture?.(event.pointerId);
            overlayBubble.style.cursor = 'grabbing';
            event.preventDefault();
        });
        overlayBubble.addEventListener('pointermove', (event) => {
            if (!overlayBubble) {
                return;
            }
            const inDragZone = isDragZone(event);
            overlayBubble.style.cursor = dragState || inDragZone ? 'grab' : 'text';
            if (!dragState || !overlayBubble) {
                return;
            }
            const dx = event.clientX - dragState.startX;
            const dy = event.clientY - dragState.startY;
            const mount = getPlayerContainer();
            const next = clampOverlayPosition(
                mount,
                dragState.x + dx,
                dragState.y + dy
            );
            const x = next.x;
            const y = next.y;
            overlayBubble.dataset.rtX = String(x);
            overlayBubble.dataset.rtY = String(y);
            overlayBubble.style.setProperty('--rt-x', `${x}px`);
            overlayBubble.style.setProperty('--rt-y', `${y}px`);
            overlayBubble.style.cursor = 'grabbing';
        });

        const releaseDrag = (event) => {
            if (overlayBubble) {
                if (event && typeof event.pointerId === 'number') {
                    try {
                        overlayBubble.releasePointerCapture?.(event.pointerId);
                    } catch (_) {}
                }
                overlayBubble.style.cursor = 'text';
            }
            dragState = null;
        };

        overlayBubble.addEventListener('pointerup', releaseDrag);
        overlayBubble.addEventListener('pointercancel', releaseDrag);
    }

    function getPlayerBottomInset(mount) {
        if (!(mount instanceof HTMLElement)) {
            return 78;
        }
        const style = window.getComputedStyle(mount);
        const candidates = [
            style.getPropertyValue('--yt-delhi-bottom-controls-height'),
            style.getPropertyValue('--yt-delhi-big-mode-bottom-controls-height'),
            style.getPropertyValue('--ytp-chrome-controls-height'),
        ];
        for (const raw of candidates) {
            const value = Number.parseFloat(String(raw || '').replace('px', '').trim());
            if (Number.isFinite(value) && value > 0) {
                return Math.round(value + 18);
            }
        }
        return 78;
    }

    function updateDockBottom(mount) {
        if (!overlayRoot) {
            return;
        }
        const dock = overlayRoot.querySelector('.dock');
        if (!(dock instanceof HTMLElement)) {
            return;
        }
        dock.style.setProperty('--rt-bottom', `${getPlayerBottomInset(mount)}px`);
    }

    function clampOverlayPosition(mount, desiredX = null, desiredY = null) {
        if (!(mount instanceof HTMLElement) || !overlayBubble) {
            return {
                x: Number(overlayBubble?.dataset.rtX || 0),
                y: Number(overlayBubble?.dataset.rtY || 0),
            };
        }

        const bubbleRect = overlayBubble.getBoundingClientRect();
        const mountRect = mount.getBoundingClientRect();
        const bubbleWidth = bubbleRect.width || overlayBubble.offsetWidth || 0;
        const bubbleHeight = bubbleRect.height || overlayBubble.offsetHeight || 0;
        const bottomInset = getPlayerBottomInset(mount);
        const maxX = Math.max(0, (mountRect.width - bubbleWidth) / 2 - PLAYER_EDGE_MARGIN_PX);
        const minY = -(mountRect.height - bubbleHeight - bottomInset - PLAYER_EDGE_MARGIN_PX);
        // Allow the subtitle to slide below the seek bar / controls area
        // instead of clamping it strictly above the player chrome.
        const maxY = Math.max(DRAG_EDGE_PX, bottomInset - PLAYER_EDGE_MARGIN_PX);
        const rawX = desiredX ?? Number(overlayBubble.dataset.rtX || 0);
        const rawY = desiredY ?? Number(overlayBubble.dataset.rtY || 0);
        const x = Math.min(maxX, Math.max(-maxX, rawX));
        const y = Math.min(maxY, Math.max(minY, rawY));

        overlayBubble.dataset.rtX = String(x);
        overlayBubble.dataset.rtY = String(y);
        overlayBubble.style.setProperty('--rt-x', `${x}px`);
        overlayBubble.style.setProperty('--rt-y', `${y}px`);
        return { x, y };
    }

    function setPageOverlayVisible(visible) {
        ensurePageOverlay();
        attachOverlayToPlayer();
        overlayHost.style.display = visible ? 'block' : 'none';
    }

    function renderPageOverlay(payload) {
        ensurePageOverlay();
        if (!payload || payload.mode !== 'browser' || !isCapturing || !payload.translation) {
            setPageOverlayVisible(false);
            return;
        }

        latestSubtitlePayload = payload;

        if (document.hidden) {
            setPageOverlayVisible(false);
            return;
        }

        overlayHost.style.display = 'block';
        attachOverlayToPlayer();
        const alpha = Math.min(1, Math.max(0, Number(payload.opacity ?? 0.82)));
        const fontSize = Math.min(96, Math.max(10, Number(payload.font_size ?? 23)));
        const sourceFontSize = Math.min(72, Math.max(8, Number(payload.source_font_size ?? 15)));
        overlayRoot.querySelector('.bubble').style.background = `rgba(18, 20, 22, ${alpha})`;
        overlayCurr.style.fontSize = `${fontSize}px`;
        overlayPrev.style.fontSize = `${sourceFontSize}px`;

        const showPrev = Boolean(payload.show_source && payload.previous_translation);
        const nextPrev = payload.previous_translation || '';
        const nextCurr = payload.translation || '';
        if (nextPrev !== lastRenderedPrevious || overlayPrev.textContent !== nextPrev) {
            overlayPrev.textContent = nextPrev;
            lastRenderedPrevious = nextPrev;
        }
        // Reassert visibility even when the cached value did not change: the
        // player or page scripts may have recreated or modified this element.
        overlayPrev.classList.toggle('hidden', !showPrev);
        lastRenderedShowPrev = showPrev;
        if (nextCurr !== lastRenderedTranslation || overlayCurr.textContent !== nextCurr) {
            overlayCurr.textContent = nextCurr;
            lastRenderedTranslation = nextCurr;
        }
        clampOverlayPosition(getPlayerContainer());
    }

    async function reportStatus() {
        if (!isCapturing || !backendConnected) {
            return;
        }
        const payload = {
            tab_id: tabId,
            host: window.location.hostname,
            url: window.location.href,
            redirected: isCapturing,
            actual_redirected: isCapturing,
            visible: !document.hidden,
            focused: document.hasFocus(),
            ts: Date.now()
        };

        try {
            await fetch(STATUS_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                keepalive: true
            });
        } catch (_) {}
    }

    async function pullSubtitles(force = false) {
        if (!isCapturing || !backendConnected) {
            return;
        }
        try {
            const url = force ? `${SUBTITLES_URL}?ts=${Date.now()}` : SUBTITLES_URL;
            const response = await fetch(url, { cache: 'no-store' });
            if (!response.ok) {
                return;
            }
            const payload = await response.json();
            lastMode = payload.mode || 'desktop';
            if (lastMode !== 'browser') {
                setPageOverlayVisible(false);
                return;
            }
            latestSubtitlePayload = payload;
            if (payload.sequence === lastSubtitleSeq && !document.hidden && isCapturing) {
                renderPageOverlay(latestSubtitlePayload);
                setPageOverlayVisible(true);
                attachOverlayToPlayer();
                return;
            }
            lastSubtitleSeq = payload.sequence;
            renderPageOverlay(payload);
        } catch (_) {}
    }

    async function toggleAudio() {
        let response;
        try {
            response = await chrome.runtime.sendMessage({
                type: 'TOGGLE_CAPTURE',
            });
        } catch (error) {
            const message = String(error?.message || error);
            if (message.includes('Extension context invalidated')) {
                const refreshMessage = 'Extension was updated. Refresh this tab once to reconnect it.';
                audioButton.title = refreshMessage;
                showCaptureNotice(refreshMessage, true);
                return;
            }
            audioButton.title = message;
            console.error('[Realtime Translator] Capture failed:', message);
            showCaptureNotice(message, true);
            return;
        }
        if (!response?.ok) {
            const error = response?.error || 'Tab capture permission was not granted';
            audioButton.title = error;
            showCaptureNotice(
                error,
                true,
            );
            return;
        }
        // CAPTURE_STATE is also broadcast by the service worker after the switch.
    }

    audioButton.onclick = () => {
        toggleAudio().catch(error => {
            const message = String(error?.message || error);
            if (!message.includes('Extension context invalidated')) {
                console.error('[Realtime Translator] Capture failed:', message);
            }
        });
    };
    audioButton.onmouseenter = () => {
        audioButton.style.backgroundColor = 'rgba(255, 255, 255, 0.10)';
    };
    audioButton.onmouseleave = () => {
        audioButton.style.backgroundColor = 'transparent';
    };
    audioButton.onpointerdown = () => { audioButton.style.transform = 'scale(0.90)'; };
    audioButton.onpointerup = () => { audioButton.style.transform = 'scale(1)'; };
    audioButton.onpointercancel = () => { audioButton.style.transform = 'scale(1)'; };

    function init() {
        mountAudioButton();
        styleAudioButton();
        ensurePageOverlay();
        attachOverlayToPlayer();
        if (isCapturing && backendConnected) {
            reportStatus().catch(() => {});
            pullSubtitles().catch(() => {});
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }

    document.addEventListener('visibilitychange', () => {
        if (isCapturing && backendConnected) {
            reportStatus().catch(() => {});
        }
        if (document.hidden) {
            setPageOverlayVisible(false);
        } else if (isCapturing && backendConnected) {
            if (latestSubtitlePayload) {
                renderPageOverlay(latestSubtitlePayload);
            }
            pullSubtitles(true).catch(() => {});
            setTimeout(() => {
                pullSubtitles(true).catch(() => {});
            }, 180);
        }
    });

    window.addEventListener('beforeunload', () => {
        if (isCapturing && backendConnected) {
            reportStatus().catch(() => {});
        }
    });

    setInterval(() => {
        if (isCapturing && backendConnected) {
            reportStatus().catch(() => {});
        }
    }, 700);

    setInterval(() => {
        if (isCapturing && backendConnected) {
            pullSubtitles(false).catch(() => {});
        }
    }, 250);

    const observer = new MutationObserver(() => {
        mountAudioButton();
        ensurePageOverlay();
        attachOverlayToPlayer();
    });

    observer.observe(document.documentElement, { childList: true, subtree: true });

    document.addEventListener('fullscreenchange', () => {
        attachOverlayToPlayer();
        if (latestSubtitlePayload) {
            renderPageOverlay(latestSubtitlePayload);
        }
    });

    chrome.runtime.onMessage.addListener(message => {
        if (message?.type !== 'CAPTURE_STATE') {
            if (message?.type === 'BACKEND_STATE') {
                backendConnected = Boolean(message.connected);
                styleAudioButton();
                if (isCapturing && backendConnected) {
                    reportStatus().catch(() => {});
                    pullSubtitles(true).catch(() => {});
                } else if (!backendConnected) {
                    setPageOverlayVisible(false);
                }
            }
            return;
        }
        isCapturing = Boolean(message.active);
        webuiAvailable = Boolean(message.webuiAvailable);
        backendConnected = Boolean(message.backendConnected);
        runtimeState = String(message.runtimeState || 'unavailable');
        styleAudioButton();
        if (isCapturing && backendConnected) {
            reportStatus().catch(() => {});
        }
        if (!isCapturing) {
            setPageOverlayVisible(false);
        } else {
            pullSubtitles(true).catch(() => {});
        }
        if (message.error) {
            audioButton.title = message.error;
            console.error('[Realtime Translator] Capture failed:', message.error);
            showCaptureNotice(message.error, true);
        }
    });

    chrome.runtime.sendMessage({ type: 'GET_CAPTURE_STATE' })
        .then(response => {
            isCapturing = Boolean(response?.active);
            webuiAvailable = Boolean(response?.webuiAvailable);
            backendConnected = Boolean(response?.backendConnected);
            runtimeState = String(response?.runtimeState || 'unavailable');
            styleAudioButton();
            if (isCapturing && backendConnected) {
                reportStatus().catch(() => {});
            }
        })
        .catch(() => {});
})();
