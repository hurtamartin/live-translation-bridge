'use strict';

/**
 * Live Translation Bridge - Frontend Application
 * Adapted to work with the existing FastAPI + SeamlessM4T backend.
 * Backend sends plain text via WebSocket, language change via {"type":"set_lang","lang":"ces"}.
 */

/* ========== Safe localStorage ========== */

function safeGetItem(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

function safeSetItem(key, value) {
  try { safeSetItem(key, value); } catch (e) { /* ignore */ }
}

/* ========== Configuration ========== */

const LANGUAGES = [
  { code: 'ces', label: 'CZ', flag: '\u{1F1E8}\u{1F1FF}', name: '\u010Ce\u0161tina',   desc: 'Czech' },
  { code: 'eng', label: 'EN', flag: '\u{1F1EC}\u{1F1E7}', name: 'English',  desc: 'English' },
  { code: 'spa', label: 'ES', flag: '\u{1F1EA}\u{1F1F8}', name: 'Español', desc: 'Spanish' },
  { code: 'ukr', label: 'UA', flag: '\u{1F1FA}\u{1F1E6}', name: '\u0423\u043A\u0440\u0430\u0457\u043D\u0441\u044C\u043A\u0430', desc: 'Ukrainian' },
  { code: 'deu', label: 'DE', flag: '\u{1F1E9}\u{1F1EA}', name: 'Deutsch',  desc: 'German' },
  { code: 'pol', label: 'PL', flag: '\u{1F1F5}\u{1F1F1}', name: 'Polski',   desc: 'Polish' },
];

const TRANSLATIONS = {
  ces: {
    waitingForTranslation: '\u010Cek\u00e1m na p\u0159eklad...',
    switchingLanguage: 'P\u0159ep\u00edn\u00e1m jazyk...',
    connecting: 'P\u0159ipojov\u00e1n\u00ed...',
    live: 'LIVE',
    disconnected: 'Odpojeno',
    disconnectedTimer: 'Odpojeno ({0}s)',
    connectionError: 'Spojen\u00ed ztraceno. Obnoven\u00ed za {0}s...',
    selectLanguage: 'Vyberte jazyk',
    close: 'Zav\u0159\u00edt',
    settings: 'Nastaven\u00ed',
    fontSize: 'Velikost p\u00edsma',
    showTimestamps: 'Zobrazit \u010das',
    subtitleCount: 'Po\u010det titulk\u016f',

    showQrCode: 'Zobrazit QR k\u00f3d',
    qrPageTitle: 'Live P\u0159eklad',
    qrInstruction: 'P\u0159ipojte se na Wi-Fi a naskenujte k\u00f3d',
    pageTitle: 'Live P\u0159eklad',
    offline: 'Offline re\u017eim',
    updateAvailable: 'Nov\u00e1 verze k dispozici',
    updateReload: 'Aktualizovat',
  },
  eng: {
    waitingForTranslation: 'Waiting for translation...',
    switchingLanguage: 'Switching language...',
    connecting: 'Connecting...',
    live: 'LIVE',
    disconnected: 'Disconnected',
    disconnectedTimer: 'Disconnected ({0}s)',
    connectionError: 'Connection lost. Reconnecting in {0}s...',
    selectLanguage: 'Select language',
    close: 'Close',
    settings: 'Settings',
    fontSize: 'Font size',
    showTimestamps: 'Show timestamps',
    subtitleCount: 'Subtitle count',

    showQrCode: 'Show QR code',
    qrPageTitle: 'Live Translation',
    qrInstruction: 'Connect to Wi-Fi and scan the code',
    pageTitle: 'Live Translation',
    offline: 'Offline mode',
    updateAvailable: 'New version available',
    updateReload: 'Reload',
  },
  spa: {
    waitingForTranslation: 'Esperando traducción...',
    switchingLanguage: 'Cambiando idioma...',
    connecting: 'Conectando...',
    live: 'LIVE',
    disconnected: 'Desconectado',
    disconnectedTimer: 'Desconectado ({0}s)',
    connectionError: 'Conexión perdida. Reconectando en {0}s...',
    selectLanguage: 'Seleccionar idioma',
    close: 'Cerrar',
    settings: 'Configuración',
    fontSize: 'Tamaño de fuente',
    showTimestamps: 'Mostrar hora',
    subtitleCount: 'Cantidad de subtítulos',

    showQrCode: 'Mostrar código QR',
    qrPageTitle: 'Traducción en vivo',
    qrInstruction: 'Conéctese al Wi-Fi y escanee el código',
    pageTitle: 'Traducción en vivo',
    offline: 'Sin conexión',
    updateAvailable: 'Nueva versión disponible',
    updateReload: 'Recargar',
  },
  ukr: {
    waitingForTranslation: '\u041E\u0447\u0456\u043A\u0443\u0432\u0430\u043D\u043D\u044F \u043F\u0435\u0440\u0435\u043A\u043B\u0430\u0434\u0443...',
    switchingLanguage: '\u0417\u043C\u0456\u043D\u0430 \u043C\u043E\u0432\u0438...',
    connecting: '\u041F\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043D\u044F...',
    live: 'LIVE',
    disconnected: '\u0412\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043E',
    disconnectedTimer: '\u0412\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043E ({0}\u0441)',
    connectionError: '\u0417\'\u0454\u0434\u043D\u0430\u043D\u043D\u044F \u0432\u0442\u0440\u0430\u0447\u0435\u043D\u043E. \u041F\u043E\u0432\u0442\u043E\u0440\u043D\u0435 \u043F\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043D\u044F \u0447\u0435\u0440\u0435\u0437 {0}\u0441...',
    selectLanguage: '\u041E\u0431\u0435\u0440\u0456\u0442\u044C \u043C\u043E\u0432\u0443',
    close: '\u0417\u0430\u043A\u0440\u0438\u0442\u0438',
    settings: '\u041D\u0430\u043B\u0430\u0448\u0442\u0443\u0432\u0430\u043D\u043D\u044F',
    fontSize: '\u0420\u043E\u0437\u043C\u0456\u0440 \u0448\u0440\u0438\u0444\u0442\u0443',
    showTimestamps: '\u041F\u043E\u043A\u0430\u0437\u0430\u0442\u0438 \u0447\u0430\u0441',
    subtitleCount: '\u041A\u0456\u043B\u044C\u043A\u0456\u0441\u0442\u044C \u0441\u0443\u0431\u0442\u0438\u0442\u0440\u0456\u0432',

    showQrCode: '\u041F\u043E\u043A\u0430\u0437\u0430\u0442\u0438 QR \u043A\u043E\u0434',
    qrPageTitle: 'Live \u041F\u0435\u0440\u0435\u043A\u043B\u0430\u0434',
    qrInstruction: '\u041F\u0456\u0434\u043A\u043B\u044E\u0447\u0456\u0442\u044C\u0441\u044F \u0434\u043E Wi-Fi \u0442\u0430 \u0432\u0456\u0434\u0441\u043A\u0430\u043D\u0443\u0439\u0442\u0435 \u043A\u043E\u0434',
    pageTitle: 'Live \u041F\u0435\u0440\u0435\u043A\u043B\u0430\u0434',
    offline: '\u041E\u0444\u043B\u0430\u0439\u043D',
    updateAvailable: '\u0414\u043E\u0441\u0442\u0443\u043F\u043D\u0430 \u043D\u043E\u0432\u0430 \u0432\u0435\u0440\u0441\u0456\u044F',
    updateReload: '\u041E\u043D\u043E\u0432\u0438\u0442\u0438',
  },
  deu: {
    waitingForTranslation: 'Warte auf \u00dcbersetzung...',
    switchingLanguage: 'Sprache wird gewechselt...',
    connecting: 'Verbindung wird hergestellt...',
    live: 'LIVE',
    disconnected: 'Getrennt',
    disconnectedTimer: 'Getrennt ({0}s)',
    connectionError: 'Verbindung verloren. Wiederverbindung in {0}s...',
    selectLanguage: 'Sprache w\u00e4hlen',
    close: 'Schlie\u00dfen',
    settings: 'Einstellungen',
    fontSize: 'Schriftgr\u00f6\u00dfe',
    showTimestamps: 'Zeitstempel anzeigen',
    subtitleCount: 'Untertitelanzahl',

    showQrCode: 'QR-Code anzeigen',
    qrPageTitle: 'Live \u00dcbersetzung',
    qrInstruction: 'Verbinden Sie sich mit dem WLAN und scannen Sie den Code',
    pageTitle: 'Live \u00dcbersetzung',
    offline: 'Offline-Modus',
    updateAvailable: 'Neue Version verf\u00fcgbar',
    updateReload: 'Aktualisieren',
  },
  pol: {
    waitingForTranslation: 'Oczekiwanie na t\u0142umaczenie...',
    switchingLanguage: 'Zmiana j\u0119zyka...',
    connecting: '\u0141\u0105czenie...',
    live: 'LIVE',
    disconnected: 'Roz\u0142\u0105czono',
    disconnectedTimer: 'Roz\u0142\u0105czono ({0}s)',
    connectionError: 'Utracono po\u0142\u0105czenie. Ponowne po\u0142\u0105czenie za {0}s...',
    selectLanguage: 'Wybierz j\u0119zyk',
    close: 'Zamknij',
    settings: 'Ustawienia',
    fontSize: 'Rozmiar czcionki',
    showTimestamps: 'Poka\u017c czas',
    subtitleCount: 'Liczba napisy',

    showQrCode: 'Poka\u017c kod QR',
    qrPageTitle: 'Live T\u0142umaczenie',
    qrInstruction: 'Po\u0142\u0105cz si\u0119 z Wi-Fi i zeskanuj kod',
    pageTitle: 'Live T\u0142umaczenie',
    offline: 'Tryb offline',
    updateAvailable: 'Dost\u0119pna nowa wersja',
    updateReload: 'Od\u015bwie\u017c',
  },
};

const CONFIG = {
  RECONNECT_BASE: 1000,
  RECONNECT_MAX: 30000,
  MAX_SUBTITLE_LENGTH: 500,
  DEFAULT_LANG: 'ces',
  DEFAULT_MAX_SUBTITLES: 10,
  DEFAULT_FONT_SIZE: 28,
  PING_INTERVAL: 30000,
  PING_TIMEOUT: 5000,
};

/* ========== State ========== */

const state = {
  ws: null,
  language: safeGetItem('lang') || CONFIG.DEFAULT_LANG,
  theme: safeGetItem('theme') || 'auto',
  fontSize: parseInt(safeGetItem('fontSize') || CONFIG.DEFAULT_FONT_SIZE, 10),
  showTimestamps: safeGetItem('showTimestamps') === 'true',
  maxSubtitles: parseInt(safeGetItem('maxSubtitles') || CONFIG.DEFAULT_MAX_SUBTITLES, 10),
  reconnectDelay: CONFIG.RECONNECT_BASE,
  reconnectTimer: null,
  reconnectCountdownTimer: null,
  connected: false,
  wakeLock: null,
  previousFocus: null,
  autoHideTimer: null,
  switchingLanguage: false,
  switchingTimeout: null,
  pingInterval: null,
  pingTimeout: null,
};

/* ========== DOM References ========== */

const dom = {
  subtitlesContainer: document.getElementById('subtitlesContainer'),
  emptyState: document.getElementById('emptyState'),
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),
  themeToggle: document.getElementById('themeToggle'),
  themeIcon: document.getElementById('themeIcon'),
  settingsToggle: document.getElementById('settingsToggle'),
  langFab: document.getElementById('langFab'),
  fabLabel: document.getElementById('fabLabel'),
  langModal: document.getElementById('langModal'),
  langModalClose: document.getElementById('langModalClose'),
  langOptions: document.getElementById('langOptions'),
  settingsModal: document.getElementById('settingsModal'),
  settingsModalClose: document.getElementById('settingsModalClose'),
  fontSizeSlider: document.getElementById('fontSizeSlider'),
  fontSizeValue: document.getElementById('fontSizeValue'),
  showTimestamps: document.getElementById('showTimestamps'),
  maxSubtitles: document.getElementById('maxSubtitles'),
  showQrLink: document.getElementById('showQrLink'),
  qrOverlay: document.getElementById('qrOverlay'),
  qrClose: document.getElementById('qrClose'),
  header: document.getElementById('header'),
};

/* ========== Helpers ========== */

function getLangInfo(code) {
  return LANGUAGES.find(function(l) { return l.code === code; }) || { code: code, label: code.toUpperCase(), flag: '', name: code, desc: '' };
}

function t(key) {
  var lang = state.language;
  if (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) {
    return TRANSLATIONS[lang][key];
  }
  // Fallback to Czech
  if (TRANSLATIONS.ces && TRANSLATIONS.ces[key]) {
    return TRANSLATIONS.ces[key];
  }
  return key;
}

/* ========== i18n ========== */

function updateUILanguage() {
  // Update all elements with data-i18n attribute
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    // Skip empty state text while switching language
    if (key === 'waitingForTranslation' && state.switchingLanguage) return;
    el.textContent = t(key);
  });

  // Update page title
  document.title = t('pageTitle');

  // Update status text based on current connection state
  if (state.connected) {
    setStatus('live');
  } else if (state.reconnectTimer) {
    // Keep current reconnect text
  } else {
    setStatus('connecting');
  }
}

/* ========== Theme ========== */

function applyTheme() {
  var effectiveTheme = state.theme;
  if (effectiveTheme === 'auto') {
    effectiveTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  document.documentElement.setAttribute('data-theme', effectiveTheme);
  dom.themeIcon.textContent = effectiveTheme === 'dark' ? '\u263E' : '\u2600';
}

function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme');
  state.theme = current === 'dark' ? 'light' : 'dark';
  safeSetItem('theme', state.theme);
  applyTheme();
}

/* ========== Connection Status ========== */

function setStatus(status, text) {
  dom.statusDot.className = 'status__dot';
  switch (status) {
    case 'live':
      dom.statusDot.classList.add('status__dot--live');
      dom.statusText.textContent = text || t('live');
      break;
    case 'error':
      dom.statusDot.classList.add('status__dot--error');
      dom.statusText.textContent = text || t('disconnected');
      break;
    default:
      dom.statusText.textContent = text || t('connecting');
  }
}

/* ========== WebSocket ========== */

function connect() {
  if (state.ws && (state.ws.readyState === WebSocket.CONNECTING || state.ws.readyState === WebSocket.OPEN)) {
    return;
  }

  setStatus('connecting');
  var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  var url = protocol + '//' + window.location.host + '/ws';

  try {
    state.ws = new WebSocket(url);
  } catch (err) {
    console.error('[WS] Connection error:', err);
    scheduleReconnect();
    return;
  }

  state.ws.onopen = function() {
    console.log('[WS] Connected');
    state.connected = true;
    state.reconnectDelay = CONFIG.RECONNECT_BASE;
    setStatus('live');
    acquireWakeLock();

    // Send current language preference to backend
    state.ws.send(JSON.stringify({ type: 'set_lang', lang: state.language }));

    // Start heartbeat ping
    clearInterval(state.pingInterval);
    state.pingInterval = setInterval(function() {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: 'ping' }));
        // If no pong within timeout, force reconnect
        state.pingTimeout = setTimeout(function() {
          console.warn('[WS] Ping timeout — reconnecting');
          if (state.ws) state.ws.close();
        }, CONFIG.PING_TIMEOUT);
      }
    }, CONFIG.PING_INTERVAL);
  };

  state.ws.onmessage = function(event) {
    // Backend sends JSON {"type":"subtitle","text":"..."} or {"type":"pong"}
    var text = '';
    try {
      var data = JSON.parse(event.data);
      if (data.type === 'pong') {
        clearTimeout(state.pingTimeout);
        return;
      }
      if (data.type === 'subtitle' && typeof data.text === 'string') {
        text = data.text;
      }
    } catch (e) {
      console.warn('[WS] Invalid message format:', e.message);
      return;
    }
    if (!text) return;
    // Sanitize: limit length, strip RTL/LTR override characters
    text = text.substring(0, CONFIG.MAX_SUBTITLE_LENGTH)
               .replace(/[\u202A-\u202E\u200F\u200E\u061C\u2066-\u2069]/g, '')
               .trim();
    if (text) {
      addSubtitle(text);
    }
  };

  state.ws.onclose = function() {
    console.log('[WS] Disconnected');
    state.connected = false;
    clearInterval(state.pingInterval);
    clearTimeout(state.pingTimeout);
    setStatus('error');
    releaseWakeLock();
    scheduleReconnect();
  };

  state.ws.onerror = function(err) {
    console.error('[WS] Error:', err);
    state.connected = false;
  };
}

function scheduleReconnect() {
  if (state.reconnectTimer) return;

  // Add random jitter ±25% to prevent thundering herd
  var baseDelay = state.reconnectDelay;
  var jitter = baseDelay * (0.5 * Math.random() - 0.25);
  var delay = Math.max(baseDelay + jitter, 500);
  var remaining = Math.round(delay / 1000);
  setStatus('error', t('disconnectedTimer').replace('{0}', remaining));

  // Live countdown every second in status indicator
  state.reconnectCountdownTimer = setInterval(function() {
    remaining--;
    if (remaining > 0) {
      setStatus('error', t('disconnectedTimer').replace('{0}', remaining));
    }
  }, 1000);

  state.reconnectTimer = setTimeout(function() {
    state.reconnectTimer = null;
    if (state.reconnectCountdownTimer) {
      clearInterval(state.reconnectCountdownTimer);
      state.reconnectCountdownTimer = null;
    }
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, CONFIG.RECONNECT_MAX);
    connect();
  }, delay);
}

/* ========== Language ========== */

function changeLanguage(langCode) {
  if (state.switchingLanguage) return;
  state.language = langCode;
  safeSetItem('lang', langCode);

  var info = getLangInfo(langCode);
  dom.fabLabel.textContent = info.label;

  // Update active state in language modal
  document.querySelectorAll('.lang-option').forEach(function(btn) {
    var isActive = btn.dataset.lang === langCode;
    btn.classList.toggle('lang-option--active', isActive);
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
  });

  // Clear current subtitles and show "switching" state
  state.switchingLanguage = true;
  var subtitles = dom.subtitlesContainer.querySelectorAll('.subtitle');
  subtitles.forEach(function(el) { el.remove(); });
  if (dom.emptyState) {
    dom.emptyState.style.display = '';
    var emptyText = dom.emptyState.querySelector('.empty__text');
    if (emptyText) {
      emptyText.textContent = t('switchingLanguage');
    }
  }

  // Disable language buttons during switch
  var langButtons = document.querySelectorAll('.lang-option');
  langButtons.forEach(function(b) { b.disabled = true; });
  dom.langFab.disabled = true;

  // Timeout: reset switching state after 10s if no subtitle arrives
  clearTimeout(state.switchingTimeout);
  state.switchingTimeout = setTimeout(function() {
    if (state.switchingLanguage) {
      state.switchingLanguage = false;
      langButtons.forEach(function(b) { b.disabled = false; });
      dom.langFab.disabled = false;
      if (dom.emptyState) {
        var txt = dom.emptyState.querySelector('.empty__text');
        if (txt) txt.textContent = t('waitingForTranslation');
      }
    }
  }, 10000);

  // Send language change to backend
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'set_lang', lang: langCode }));
  }

  // Update UI language (but not the empty text we just set)
  updateUILanguage();
}

/* ========== Subtitles ========== */

function addSubtitle(text) {
  // Hide empty state and reset switching flag
  if (state.switchingLanguage) {
    state.switchingLanguage = false;
    clearTimeout(state.switchingTimeout);
    // Re-enable language buttons
    document.querySelectorAll('.lang-option').forEach(function(b) { b.disabled = false; });
    dom.langFab.disabled = false;
  }
  if (dom.emptyState) {
    dom.emptyState.style.display = 'none';
  }

  var el = document.createElement('div');
  el.className = 'subtitle';

  // Translated text (font-size driven by --user-font-size CSS variable)
  var translated = document.createElement('div');
  translated.className = 'subtitle__translated';
  translated.textContent = text;
  el.appendChild(translated);

  // Meta line
  var meta = document.createElement('div');
  meta.className = 'subtitle__meta';

  var timeEl = document.createElement('span');
  timeEl.className = 'subtitle__time' + (state.showTimestamps ? ' subtitle__time--visible' : '');
  timeEl.textContent = new Date().toLocaleTimeString();
  meta.appendChild(timeEl);

  el.appendChild(meta);

  dom.subtitlesContainer.appendChild(el);

  // Trim excess subtitles
  trimSubtitles();

  // Auto-scroll
  requestAnimationFrame(function() {
    dom.subtitlesContainer.scrollTop = dom.subtitlesContainer.scrollHeight;
  });
}

function trimSubtitles() {
  var subtitles = dom.subtitlesContainer.querySelectorAll('.subtitle');
  var excess = subtitles.length - state.maxSubtitles;

  for (var i = 0; i < excess; i++) {
    var el = subtitles[i];
    el.classList.add('subtitle--exiting');
    el.addEventListener('animationend', function() { this.remove(); }, { once: true });
  }
}

function clearSubtitles() {
  var subtitles = dom.subtitlesContainer.querySelectorAll('.subtitle');
  subtitles.forEach(function(el) { el.remove(); });

  if (dom.emptyState) {
    dom.emptyState.style.display = '';
  }
}

/* ========== Settings ========== */

function applySettings() {
  dom.fontSizeSlider.value = state.fontSize;
  dom.fontSizeValue.textContent = state.fontSize + 'px';
  document.documentElement.style.setProperty('--user-font-size', state.fontSize + 'px');

  dom.showTimestamps.checked = state.showTimestamps;
  document.querySelectorAll('.subtitle__time').forEach(function(el) {
    el.classList.toggle('subtitle__time--visible', state.showTimestamps);
  });

  dom.maxSubtitles.value = state.maxSubtitles;

  var info = getLangInfo(state.language);
  dom.fabLabel.textContent = info.label;
}

/* ========== Dynamic Language Options ========== */

function buildLanguageOptions() {
  dom.langOptions.innerHTML = '';
  dom.langOptions.setAttribute('role', 'listbox');
  dom.langOptions.setAttribute('aria-label', t('selectLanguage'));
  LANGUAGES.forEach(function(lang) {
    var btn = document.createElement('button');
    btn.className = 'lang-option';
    if (lang.code === state.language) {
      btn.className += ' lang-option--active';
    }
    btn.dataset.lang = lang.code;
    btn.setAttribute('tabindex', '0');
    btn.setAttribute('role', 'option');
    btn.setAttribute('aria-selected', lang.code === state.language ? 'true' : 'false');
    btn.setAttribute('aria-label', lang.name + ' (' + lang.desc + ')');

    var flagSpan = document.createElement('span');
    flagSpan.className = 'lang-option__flag';
    flagSpan.setAttribute('aria-hidden', 'true');
    flagSpan.textContent = lang.flag;

    var nameSpan = document.createElement('span');
    nameSpan.className = 'lang-option__name';
    nameSpan.textContent = lang.name;

    var descSpan = document.createElement('span');
    descSpan.className = 'lang-option__desc';
    descSpan.textContent = lang.desc;

    btn.appendChild(flagSpan);
    btn.appendChild(nameSpan);
    btn.appendChild(descSpan);

    dom.langOptions.appendChild(btn);
  });
}

/* ========== Modal Helpers ========== */

function openModal(overlay) {
  state.previousFocus = document.activeElement;
  overlay.classList.add('modal-overlay--open');
  // Focus first focusable element in the modal
  var focusable = overlay.querySelector('button, [tabindex="0"], input, select, a');
  if (focusable) {
    requestAnimationFrame(function() { focusable.focus(); });
  }
  // Install focus trap
  overlay._focusTrap = function(e) {
    if (e.key !== 'Tab') return;
    var focusableEls = overlay.querySelectorAll('button, [tabindex="0"], input, select, a, [tabindex]:not([tabindex="-1"])');
    if (focusableEls.length === 0) return;
    var first = focusableEls[0];
    var last = focusableEls[focusableEls.length - 1];
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };
  overlay.addEventListener('keydown', overlay._focusTrap);
}

function closeModal(overlay) {
  overlay.classList.remove('modal-overlay--open');
  // Remove focus trap
  if (overlay._focusTrap) {
    overlay.removeEventListener('keydown', overlay._focusTrap);
    overlay._focusTrap = null;
  }
  if (state.previousFocus) {
    state.previousFocus.focus();
    state.previousFocus = null;
  }
}

function getOpenOverlay() {
  if (dom.langModal.classList.contains('modal-overlay--open')) return dom.langModal;
  if (dom.settingsModal.classList.contains('modal-overlay--open')) return dom.settingsModal;
  if (dom.qrOverlay.classList.contains('qr-overlay--open')) return dom.qrOverlay;
  return null;
}

/* ========== Event Listeners ========== */

function initEventListeners() {
  dom.themeToggle.addEventListener('click', toggleTheme);

  dom.langFab.addEventListener('click', function() {
    document.querySelectorAll('.lang-option').forEach(function(btn) {
      btn.classList.toggle('lang-option--active', btn.dataset.lang === state.language);
    });
    openModal(dom.langModal);
  });

  dom.langOptions.addEventListener('click', function(e) {
    var btn = e.target.closest('.lang-option');
    if (btn && btn.dataset.lang && !btn.disabled) {
      changeLanguage(btn.dataset.lang);
      closeModal(dom.langModal);
      // Debounce: disable all lang buttons for 500ms
      var buttons = document.querySelectorAll('.lang-option');
      buttons.forEach(function(b) { b.disabled = true; });
      setTimeout(function() {
        buttons.forEach(function(b) { b.disabled = false; });
      }, 500);
    }
  });

  dom.langModalClose.addEventListener('click', function() { closeModal(dom.langModal); });
  dom.langModal.addEventListener('click', function(e) {
    if (e.target === dom.langModal) closeModal(dom.langModal);
  });

  dom.settingsToggle.addEventListener('click', function() { openModal(dom.settingsModal); });
  dom.settingsModalClose.addEventListener('click', function() { closeModal(dom.settingsModal); });
  dom.settingsModal.addEventListener('click', function(e) {
    if (e.target === dom.settingsModal) closeModal(dom.settingsModal);
  });

  dom.fontSizeSlider.addEventListener('input', function(e) {
    state.fontSize = parseInt(e.target.value, 10);
    safeSetItem('fontSize', state.fontSize);
    dom.fontSizeValue.textContent = state.fontSize + 'px';
    document.documentElement.style.setProperty('--user-font-size', state.fontSize + 'px');
  });

  dom.showTimestamps.addEventListener('change', function(e) {
    state.showTimestamps = e.target.checked;
    safeSetItem('showTimestamps', state.showTimestamps);
    document.querySelectorAll('.subtitle__time').forEach(function(el) {
      el.classList.toggle('subtitle__time--visible', state.showTimestamps);
    });
  });

  dom.maxSubtitles.addEventListener('change', function(e) {
    state.maxSubtitles = parseInt(e.target.value, 10);
    safeSetItem('maxSubtitles', state.maxSubtitles);
    trimSubtitles();
  });

  dom.showQrLink.addEventListener('click', function(e) {
    e.preventDefault();
    closeModal(dom.settingsModal);
    dom.qrOverlay.classList.add('qr-overlay--open');
  });

  dom.qrClose.addEventListener('click', function() {
    dom.qrOverlay.classList.remove('qr-overlay--open');
  });

  if (window.location.hash === '#qr') {
    dom.qrOverlay.classList.add('qr-overlay--open');
  }

  window.addEventListener('hashchange', function() {
    if (window.location.hash === '#qr') {
      dom.qrOverlay.classList.add('qr-overlay--open');
    }
  });

  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function() {
    if (state.theme === 'auto') {
      applyTheme();
    }
  });

  // Escape key closes open modal/overlay
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      var overlay = getOpenOverlay();
      if (overlay) {
        if (overlay === dom.qrOverlay) {
          dom.qrOverlay.classList.remove('qr-overlay--open');
        } else {
          closeModal(overlay);
        }
      }
    }
  });

  // Re-acquire wake lock on visibility change
  document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible' && state.connected) {
      acquireWakeLock();
    }
  });
}

/* ========== Wake Lock ========== */

function acquireWakeLock() {
  if (!('wakeLock' in navigator)) return;
  navigator.wakeLock.request('screen').then(function(lock) {
    state.wakeLock = lock;
    lock.addEventListener('release', function() {
      state.wakeLock = null;
    });
  }).catch(function() {
    // Silent fail on unsupported or denied
  });
}

function releaseWakeLock() {
  if (state.wakeLock) {
    state.wakeLock.release().catch(function() {});
    state.wakeLock = null;
  }
}

/* ========== Auto-hide Header & FAB ========== */

function setupAutoHide() {
  var HIDE_DELAY = 5000;
  var lastShowTime = 0;

  function showUI() {
    var now = Date.now();
    if (now - lastShowTime < 100) return;
    lastShowTime = now;
    dom.header.classList.remove('header--hidden');
    dom.langFab.classList.remove('fab--hidden');
    clearTimeout(state.autoHideTimer);
    state.autoHideTimer = setTimeout(hideUI, HIDE_DELAY);
  }

  function hideUI() {
    // Don't hide if a modal is open
    if (getOpenOverlay()) return;
    dom.header.classList.add('header--hidden');
    dom.langFab.classList.add('fab--hidden');
  }

  ['mousemove', 'touchstart', 'keydown'].forEach(function(evt) {
    document.addEventListener(evt, showUI, { passive: true });
  });

  // Start the initial timer
  state.autoHideTimer = setTimeout(hideUI, HIDE_DELAY);
}

/* ========== PWA / Service Worker ========== */

function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').then(function(registration) {
      console.log('[SW] Registered');
      // Listen for updates
      registration.addEventListener('updatefound', function() {
        var newWorker = registration.installing;
        if (!newWorker) return;
        newWorker.addEventListener('statechange', function() {
          if (newWorker.state === 'activated' && navigator.serviceWorker.controller) {
            // Deduplicate: only one update banner at a time
            if (document.querySelector('.update-banner')) return;
            // New version available — show update banner
            var banner = document.createElement('div');
            banner.className = 'update-banner';
            var span = document.createElement('span');
            span.textContent = t('updateAvailable');
            banner.appendChild(span);
            var btn = document.createElement('button');
            btn.className = 'update-banner__btn';
            btn.textContent = t('updateReload');
            btn.addEventListener('click', function() { window.location.reload(); });
            banner.appendChild(btn);
            document.body.appendChild(banner);
          }
        });
      });
    }).catch(function(err) {
      console.error('[SW] Registration failed:', err);
    });
  }
}

/* ========== Offline Detection ========== */

function setupOfflineDetection() {
  // Create offline banner (inserted at top of body)
  var banner = document.createElement('div');
  banner.id = 'offlineBanner';
  banner.className = 'offline-banner';
  banner.style.display = 'none';
  document.body.appendChild(banner);

  function updateBannerContent() {
    banner.textContent = '';
    var textSpan = document.createElement('span');
    textSpan.textContent = t('offline');
    banner.appendChild(textSpan);
    var dismissBtn = document.createElement('button');
    dismissBtn.className = 'offline-banner__dismiss';
    dismissBtn.textContent = '\u00D7';
    dismissBtn.setAttribute('aria-label', t('close'));
    dismissBtn.addEventListener('click', function() {
      banner.style.display = 'none';
    });
    banner.appendChild(dismissBtn);
  }

  window.addEventListener('offline', function() {
    updateBannerContent();
    banner.style.display = '';
  });
  window.addEventListener('online', function() {
    banner.style.display = 'none';
  });
  // Show immediately if already offline
  if (!navigator.onLine) {
    updateBannerContent();
    banner.style.display = '';
  }
}

/* ========== Init ========== */

function init() {
  buildLanguageOptions();
  applyTheme();
  applySettings();
  updateUILanguage();
  initEventListeners();
  setupAutoHide();
  setupOfflineDetection();
  registerServiceWorker();
  connect();

  // QR image fallback (moved from inline onerror for CSP compliance)
  var qrImg = document.getElementById('qrImage');
  if (qrImg) {
    qrImg.addEventListener('error', function() {
      this.src = '/static/assets/qr-placeholder.svg';
    }, { once: true });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
