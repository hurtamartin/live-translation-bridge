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
  { code: 'rus', label: 'RU', flag: '\u{1F1F7}\u{1F1FA}', name: '\u0420\u0443\u0441\u0441\u043A\u0438\u0439', desc: 'Russian' },
  { code: 'ukr', label: 'UA', flag: '\u{1F1FA}\u{1F1E6}', name: '\u0423\u043A\u0440\u0430\u0457\u043D\u0441\u044C\u043A\u0430', desc: 'Ukrainian' },
  { code: 'deu', label: 'DE', flag: '\u{1F1E9}\u{1F1EA}', name: 'Deutsch',  desc: 'German' },
  { code: 'pol', label: 'PL', flag: '\u{1F1F5}\u{1F1F1}', name: 'Polski',   desc: 'Polish' },
];

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
      dom.statusText.textContent = text || 'LIVE';
      break;
    case 'error':
      dom.statusDot.classList.add('status__dot--error');
      dom.statusText.textContent = text || 'Odpojeno';
      break;
    default:
      dom.statusText.textContent = text || 'P\u0159ipojov\u00e1n\u00ed...';
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
    // Backend sends plain text (translated sentence)
    var text = event.data;
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
  setStatus('error', 'Odpojeno (' + delaySec + 's)');

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

  var langTag = document.createElement('span');
  langTag.className = 'subtitle__lang-tag';
  var info = getLangInfo(state.language);
  langTag.textContent = info.label;
  meta.appendChild(langTag);

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
  initEventListeners();
  registerServiceWorker();
  connect();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
