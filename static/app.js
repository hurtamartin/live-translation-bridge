'use strict';

/**
 * Live Translation Bridge - Frontend Application
 * Adapted to work with the existing FastAPI + SeamlessM4T backend.
 * Backend sends plain text via WebSocket, language change via {"type":"set_lang","lang":"ces"}.
 */

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
    connecting: 'P\u0159ipojov\u00e1n\u00ed...',
    live: 'LIVE',
    disconnected: 'Odpojeno',
    disconnectedTimer: 'Odpojeno ({0}s)',
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
  },
  eng: {
    waitingForTranslation: 'Waiting for translation...',
    connecting: 'Connecting...',
    live: 'LIVE',
    disconnected: 'Disconnected',
    disconnectedTimer: 'Disconnected ({0}s)',
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
  },
  spa: {
    waitingForTranslation: 'Esperando traducción...',
    connecting: 'Conectando...',
    live: 'LIVE',
    disconnected: 'Desconectado',
    disconnectedTimer: 'Desconectado ({0}s)',
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
  },
  ukr: {
    waitingForTranslation: '\u041E\u0447\u0456\u043A\u0443\u0432\u0430\u043D\u043D\u044F \u043F\u0435\u0440\u0435\u043A\u043B\u0430\u0434\u0443...',
    connecting: '\u041F\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043D\u044F...',
    live: 'LIVE',
    disconnected: '\u0412\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043E',
    disconnectedTimer: '\u0412\u0456\u0434\u043A\u043B\u044E\u0447\u0435\u043D\u043E ({0}\u0441)',
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
  },
  deu: {
    waitingForTranslation: 'Warte auf \u00dcbersetzung...',
    connecting: 'Verbindung wird hergestellt...',
    live: 'LIVE',
    disconnected: 'Getrennt',
    disconnectedTimer: 'Getrennt ({0}s)',
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
  },
  pol: {
    waitingForTranslation: 'Oczekiwanie na t\u0142umaczenie...',
    connecting: '\u0141\u0105czenie...',
    live: 'LIVE',
    disconnected: 'Roz\u0142\u0105czono',
    disconnectedTimer: 'Roz\u0142\u0105czono ({0}s)',
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
  },
};

const CONFIG = {
  RECONNECT_BASE: 1000,
  RECONNECT_MAX: 30000,
  DEFAULT_LANG: 'ces',
  DEFAULT_MAX_SUBTITLES: 10,
  DEFAULT_FONT_SIZE: 28,
};

/* ========== State ========== */

const state = {
  ws: null,
  language: localStorage.getItem('lang') || CONFIG.DEFAULT_LANG,
  theme: localStorage.getItem('theme') || 'auto',
  fontSize: parseInt(localStorage.getItem('fontSize') || CONFIG.DEFAULT_FONT_SIZE, 10),
  showTimestamps: localStorage.getItem('showTimestamps') === 'true',
  maxSubtitles: parseInt(localStorage.getItem('maxSubtitles') || CONFIG.DEFAULT_MAX_SUBTITLES, 10),
  reconnectDelay: CONFIG.RECONNECT_BASE,
  reconnectTimer: null,
  connected: false,
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
  localStorage.setItem('theme', state.theme);
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

    // Send current language preference to backend
    state.ws.send(JSON.stringify({ type: 'set_lang', lang: state.language }));
  };

  state.ws.onmessage = function(event) {
    // Backend sends JSON {"type":"subtitle","text":"..."} or plain text (fallback)
    var text = '';
    try {
      var data = JSON.parse(event.data);
      if (data.type === 'subtitle' && data.text) {
        text = data.text;
      }
    } catch (e) {
      // Fallback: plain text from older backend
      text = event.data;
    }
    if (text && text.trim()) {
      addSubtitle(text.trim());
    }
  };

  state.ws.onclose = function() {
    console.log('[WS] Disconnected');
    state.connected = false;
    setStatus('error');
    scheduleReconnect();
  };

  state.ws.onerror = function(err) {
    console.error('[WS] Error:', err);
    state.connected = false;
  };
}

function scheduleReconnect() {
  if (state.reconnectTimer) return;

  var delay = state.reconnectDelay;
  var delaySec = Math.round(delay / 1000);
  setStatus('error', t('disconnectedTimer').replace('{0}', delaySec));

  state.reconnectTimer = setTimeout(function() {
    state.reconnectTimer = null;
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, CONFIG.RECONNECT_MAX);
    connect();
  }, delay);
}

/* ========== Language ========== */

function changeLanguage(langCode) {
  state.language = langCode;
  localStorage.setItem('lang', langCode);

  var info = getLangInfo(langCode);
  dom.fabLabel.textContent = info.label;

  // Update active state in language modal
  document.querySelectorAll('.lang-option').forEach(function(btn) {
    btn.classList.toggle('lang-option--active', btn.dataset.lang === langCode);
  });

  // Clear current subtitles
  clearSubtitles();

  // Send language change to backend
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: 'set_lang', lang: langCode }));
  }

  // Update UI language
  updateUILanguage();
}

/* ========== Subtitles ========== */

function addSubtitle(text) {
  // Hide empty state
  if (dom.emptyState) {
    dom.emptyState.style.display = 'none';
  }

  var el = document.createElement('div');
  el.className = 'subtitle';

  // Translated text
  var translated = document.createElement('div');
  translated.className = 'subtitle__translated';
  translated.style.fontSize = state.fontSize + 'px';
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
  document.querySelectorAll('.subtitle__translated').forEach(function(el) {
    el.style.fontSize = state.fontSize + 'px';
  });

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
  LANGUAGES.forEach(function(lang) {
    var btn = document.createElement('button');
    btn.className = 'lang-option';
    if (lang.code === state.language) {
      btn.className += ' lang-option--active';
    }
    btn.dataset.lang = lang.code;

    btn.innerHTML =
      '<span class="lang-option__flag">' + lang.flag + '</span>' +
      '<span class="lang-option__name">' + lang.name + '</span>' +
      '<span class="lang-option__desc">' + lang.desc + '</span>';

    dom.langOptions.appendChild(btn);
  });
}

/* ========== Modal Helpers ========== */

function openModal(overlay) {
  overlay.classList.add('modal-overlay--open');
}

function closeModal(overlay) {
  overlay.classList.remove('modal-overlay--open');
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
    if (btn && btn.dataset.lang) {
      changeLanguage(btn.dataset.lang);
      closeModal(dom.langModal);
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
    localStorage.setItem('fontSize', state.fontSize);
    dom.fontSizeValue.textContent = state.fontSize + 'px';
    document.querySelectorAll('.subtitle__translated').forEach(function(el) {
      el.style.fontSize = state.fontSize + 'px';
    });
  });

  dom.showTimestamps.addEventListener('change', function(e) {
    state.showTimestamps = e.target.checked;
    localStorage.setItem('showTimestamps', state.showTimestamps);
    document.querySelectorAll('.subtitle__time').forEach(function(el) {
      el.classList.toggle('subtitle__time--visible', state.showTimestamps);
    });
  });

  dom.maxSubtitles.addEventListener('change', function(e) {
    state.maxSubtitles = parseInt(e.target.value, 10);
    localStorage.setItem('maxSubtitles', state.maxSubtitles);
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
}

/* ========== PWA / Service Worker ========== */

function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').then(function() {
      console.log('[SW] Registered');
    }).catch(function(err) {
      console.error('[SW] Registration failed:', err);
    });
  }
}

/* ========== Init ========== */

function init() {
  buildLanguageOptions();
  applyTheme();
  applySettings();
  updateUILanguage();
  initEventListeners();
  registerServiceWorker();
  connect();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
