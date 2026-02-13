'use strict';

/* ========== i18n Translations ========== */

var ADMIN_TRANSLATIONS = {
  cs: {
    adminTitle: 'Admin Panel',
    mainPage: 'Hlavni stranka',
    statusTitle: 'Status',
    clients: 'Klienti',
    languages: 'Jazyky',
    hardware: 'Hardware',
    uptime: 'Uptime',
    audioDeviceTitle: 'Audio zarizeni',
    device: 'Zarizeni',
    channel: 'Kanal',
    apply: 'Pouzit',
    loading: 'Nacitam...',
    systemDefault: '-- Systemovy vychozi --',
    inputSignalTitle: 'Vstupni signal',
    translationParamsTitle: 'Parametry prekladu',
    silenceDuration: 'Silence Duration',
    minChunkDuration: 'Min Chunk Duration',
    maxChunkDuration: 'Max Chunk Duration',
    contextOverlap: 'Context Overlap',
    defaultTargetLang: 'Default Target Language',
    defaults: 'Vychozi',
    export: 'Export',
    import: 'Import',
    preprocessingTitle: 'Audio preprocessing',
    preprocessingDesc: 'Zpracovani vstupniho signalu pred odeslanim do modelu. Zapnete jen to, co potrebujete.',
    noiseGate: 'Noise Gate',
    noiseGateHint: 'Ztisi audio pod prahem hlasitosti',
    threshold: 'Threshold',
    volumeNormalize: 'Normalizace hlasitosti',
    volumeNormalizeHint: 'Vyrovnava uroven hlasitosti',
    targetLevel: 'Target level',
    highpassFilter: 'High-pass filtr',
    highpassHint: 'Odrizne hluboke frekvence pod reci',
    cutoff: 'Cutoff',
    autoLangDetect: 'Automaticka detekce zdrojoveho jazyka',
    autoLangHint: 'Model se pokusi detekovat jazyk reci',
    performanceTitle: 'Vykon',
    totalTranslations: 'Celkem prekladu',
    encoderAvg: 'Encoder (avg)',
    decoderAvg: 'Decoder (avg)',
    gpuMemory: 'GPU pamet',
    connectedClientsTitle: 'Pripojeni klienti',
    thId: 'ID',
    thLang: 'Jazyk',
    thIp: 'IP',
    thConnected: 'Pripojeno',
    noClients: 'Zadni klienti',
    translationHistoryTitle: 'Historie prekladu',
    refresh: 'Obnovit',
    noTranslations: 'Zatim zadne preklady',
    logTitle: 'Log',
    autoScroll: 'Auto-scroll',
    clearLog: 'Vymazat',
    confirmCancel: 'Zrusit',
    confirmReset: 'Ano, resetovat',
    // Toast / dynamic messages
    saving: 'Ukladam...',
    saved: 'Ulozeno',
    connectionError: 'Chyba spojeni',
    serverConnectionError: 'Chyba spojeni se serverem',
    saveError: 'Chyba ukladani',
    defaultsRestored: 'Vychozi nastaveni obnovena',
    configExported: 'Konfigurace exportovana',
    exportError: 'Chyba exportu',
    configImported: 'Konfigurace importovana',
    importError: 'Chyba importu',
    invalidJson: 'Neplatny JSON soubor',
    fileTooLarge: 'Soubor je prilis velky (max 100 KB)',
    deviceChanged: 'Audio zarizeni zmeneno',
    deviceLoadError: 'Chyba nacitani zarizeni',
    unknownError: 'Neznama chyba',
    error: 'Chyba',
    ok: 'OK!',
    confirmResetText: 'Opravdu chcete resetovat vsechna nastaveni na vychozi hodnoty? Tato akce je nevratna.',
    stopped: 'Stopped',
    pending: 'Pending',
    statusParseError: 'Chyba zpracovani statusu',
    configLoadError: 'Chyba nacteni konfigurace',
    logParseError: 'Chyba zpracovani logu',
    metricsLoadError: 'Chyba nacteni metrik',
    sessionsLoadError: 'Chyba nacteni sessions',
    translationsLoadError: 'Chyba nacteni prekladu',
    audioHistoryError: 'Chyba nacteni audio historie',
    keys: 'klicu',
    mono: 'Mono',
  },
  en: {
    adminTitle: 'Admin Panel',
    mainPage: 'Main Page',
    statusTitle: 'Status',
    clients: 'Clients',
    languages: 'Languages',
    hardware: 'Hardware',
    uptime: 'Uptime',
    audioDeviceTitle: 'Audio Device',
    device: 'Device',
    channel: 'Channel',
    apply: 'Apply',
    loading: 'Loading...',
    systemDefault: '-- System default --',
    inputSignalTitle: 'Input Signal',
    translationParamsTitle: 'Translation Parameters',
    silenceDuration: 'Silence Duration',
    minChunkDuration: 'Min Chunk Duration',
    maxChunkDuration: 'Max Chunk Duration',
    contextOverlap: 'Context Overlap',
    defaultTargetLang: 'Default Target Language',
    defaults: 'Defaults',
    export: 'Export',
    import: 'Import',
    preprocessingTitle: 'Audio Preprocessing',
    preprocessingDesc: 'Processing of input signal before sending to the model. Enable only what you need.',
    noiseGate: 'Noise Gate',
    noiseGateHint: 'Silences audio below volume threshold',
    threshold: 'Threshold',
    volumeNormalize: 'Volume Normalization',
    volumeNormalizeHint: 'Equalizes volume level',
    targetLevel: 'Target level',
    highpassFilter: 'High-pass Filter',
    highpassHint: 'Cuts low frequencies below speech range',
    cutoff: 'Cutoff',
    autoLangDetect: 'Auto source language detection',
    autoLangHint: 'Model will try to detect speech language',
    performanceTitle: 'Performance',
    totalTranslations: 'Total translations',
    encoderAvg: 'Encoder (avg)',
    decoderAvg: 'Decoder (avg)',
    gpuMemory: 'GPU Memory',
    connectedClientsTitle: 'Connected Clients',
    thId: 'ID',
    thLang: 'Language',
    thIp: 'IP',
    thConnected: 'Connected',
    noClients: 'No clients',
    translationHistoryTitle: 'Translation History',
    refresh: 'Refresh',
    noTranslations: 'No translations yet',
    logTitle: 'Log',
    autoScroll: 'Auto-scroll',
    clearLog: 'Clear',
    confirmCancel: 'Cancel',
    confirmReset: 'Yes, reset',
    // Toast / dynamic messages
    saving: 'Saving...',
    saved: 'Saved',
    connectionError: 'Connection error',
    serverConnectionError: 'Server connection error',
    saveError: 'Save error',
    defaultsRestored: 'Default settings restored',
    configExported: 'Configuration exported',
    exportError: 'Export error',
    configImported: 'Configuration imported',
    importError: 'Import error',
    invalidJson: 'Invalid JSON file',
    fileTooLarge: 'File too large (max 100 KB)',
    deviceChanged: 'Audio device changed',
    deviceLoadError: 'Error loading devices',
    unknownError: 'Unknown error',
    error: 'Error',
    ok: 'OK!',
    confirmResetText: 'Are you sure you want to reset all settings to defaults? This action cannot be undone.',
    stopped: 'Stopped',
    pending: 'Pending',
    statusParseError: 'Status parse error',
    configLoadError: 'Config load error',
    logParseError: 'Log parse error',
    metricsLoadError: 'Error loading metrics',
    sessionsLoadError: 'Error loading sessions',
    translationsLoadError: 'Error loading translations',
    audioHistoryError: 'Error loading audio history',
    keys: 'keys',
    mono: 'Mono',
  }
};

function adminT(key) {
  var lang = adminState.lang;
  if (ADMIN_TRANSLATIONS[lang] && ADMIN_TRANSLATIONS[lang][key]) {
    return ADMIN_TRANSLATIONS[lang][key];
  }
  if (ADMIN_TRANSLATIONS.cs && ADMIN_TRANSLATIONS.cs[key]) {
    return ADMIN_TRANSLATIONS.cs[key];
  }
  return key;
}

function updateAdminUI() {
  document.querySelectorAll('[data-i18n]').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    el.textContent = adminT(key);
  });
}

/* ========== State ========== */

var adminState = {
  logWs: null,
  statusWs: null,
  autoScroll: true,
  devicesCache: [],
  audioHistoryData: [],
  confirmCallback: null,
  saveDebounceTimer: null,
  ppSaveTimer: null,
  lang: localStorage.getItem('adminLang') || 'cs',
};

var MAX_LOG_ENTRIES = 200;

function getAdminWsUrl(path) {
  var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  var base = protocol + '//' + window.location.host + path;
  // Browser WebSocket API cannot send custom headers, so we pass the
  // auth token as a query parameter. The token is injected by the server
  // into admin.html via a <meta> tag when the page is rendered.
  var meta = document.querySelector('meta[name="ws-token"]');
  var token = meta ? meta.getAttribute('content') : '';
  return token ? base + '?token=' + encodeURIComponent(token) : base;
}

/* ========== DOM ========== */

var dom = {
  themeToggle: document.getElementById('themeToggle'),
  themeIcon: document.getElementById('themeIcon'),
  adminLangSelect: document.getElementById('adminLangSelect'),
  // Status
  dotModel: document.getElementById('dotModel'),
  dotVad: document.getElementById('dotVad'),
  dotAudio: document.getElementById('dotAudio'),
  dotInference: document.getElementById('dotInference'),
  detailModel: document.getElementById('detailModel'),
  detailVad: document.getElementById('detailVad'),
  detailAudio: document.getElementById('detailAudio'),
  detailInference: document.getElementById('detailInference'),
  infoClients: document.getElementById('infoClients'),
  infoLanguages: document.getElementById('infoLanguages'),
  infoDevice: document.getElementById('infoDevice'),
  infoUptime: document.getElementById('infoUptime'),
  // Devices
  deviceSelect: document.getElementById('deviceSelect'),
  channelSelect: document.getElementById('channelSelect'),
  applyDevice: document.getElementById('applyDevice'),
  // Config
  silenceDuration: document.getElementById('silenceDuration'),
  silenceDurationVal: document.getElementById('silenceDurationVal'),
  minChunkDuration: document.getElementById('minChunkDuration'),
  minChunkDurationVal: document.getElementById('minChunkDurationVal'),
  maxChunkDuration: document.getElementById('maxChunkDuration'),
  maxChunkDurationVal: document.getElementById('maxChunkDurationVal'),
  contextOverlap: document.getElementById('contextOverlap'),
  contextOverlapVal: document.getElementById('contextOverlapVal'),
  defaultTargetLang: document.getElementById('defaultTargetLang'),
  resetConfig: document.getElementById('resetConfig'),
  exportConfig: document.getElementById('exportConfig'),
  importConfig: document.getElementById('importConfig'),
  importConfigFile: document.getElementById('importConfigFile'),
  configStatus: document.getElementById('configStatus'),
  // Preprocessing
  ppNoiseGate: document.getElementById('ppNoiseGate'),
  ppNoiseGateSettings: document.getElementById('ppNoiseGateSettings'),
  ppNoiseGateThreshold: document.getElementById('ppNoiseGateThreshold'),
  ppNoiseGateThresholdVal: document.getElementById('ppNoiseGateThresholdVal'),
  ppNormalize: document.getElementById('ppNormalize'),
  ppNormalizeSettings: document.getElementById('ppNormalizeSettings'),
  ppNormalizeTarget: document.getElementById('ppNormalizeTarget'),
  ppNormalizeTargetVal: document.getElementById('ppNormalizeTargetVal'),
  ppHighpass: document.getElementById('ppHighpass'),
  ppHighpassSettings: document.getElementById('ppHighpassSettings'),
  ppHighpassCutoff: document.getElementById('ppHighpassCutoff'),
  ppHighpassCutoffVal: document.getElementById('ppHighpassCutoffVal'),
  ppAutoLang: document.getElementById('ppAutoLang'),
  preprocessStatus: document.getElementById('preprocessStatus'),
  // VU Meter
  audioMeterBar: document.getElementById('audioMeterBar'),
  audioMeterPeak: document.getElementById('audioMeterPeak'),
  audioMeterValue: document.getElementById('audioMeterValue'),
  audioHistoryCanvas: document.getElementById('audioHistoryCanvas'),
  // Metrics
  metricTranslations: document.getElementById('metricTranslations'),
  metricEncoder: document.getElementById('metricEncoder'),
  metricDecoder: document.getElementById('metricDecoder'),
  metricGpu: document.getElementById('metricGpu'),
  // Sessions
  sessionsBody: document.getElementById('sessionsBody'),
  // Translation history
  translationHistory: document.getElementById('translationHistory'),
  refreshTranslations: document.getElementById('refreshTranslations'),
  // Log
  logContainer: document.getElementById('logContainer'),
  logAutoScroll: document.getElementById('logAutoScroll'),
  clearLog: document.getElementById('clearLog'),
  // Confirm
  confirmOverlay: document.getElementById('confirmOverlay'),
  confirmText: document.getElementById('confirmText'),
  confirmOk: document.getElementById('confirmOk'),
  confirmCancel: document.getElementById('confirmCancel'),
  // Toast
  toastContainer: document.getElementById('toastContainer'),
};

/* ========== Theme ========== */

function applyTheme() {
  var theme = localStorage.getItem('theme') || 'auto';
  var effective = theme;
  if (effective === 'auto') {
    effective = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  document.documentElement.setAttribute('data-theme', effective);
  dom.themeIcon.textContent = effective === 'dark' ? '\u263E' : '\u2600';
}

function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme');
  var next = current === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', next);
  applyTheme();
}

/* ========== Toast Notifications ========== */

function showToast(message, type) {
  var toast = document.createElement('div');
  toast.className = 'toast toast--' + (type || 'info');
  toast.textContent = message;
  dom.toastContainer.appendChild(toast);
  toast.offsetHeight;
  toast.classList.add('toast--visible');
  setTimeout(function() {
    toast.classList.remove('toast--visible');
    toast.addEventListener('transitionend', function() { toast.remove(); });
  }, 3000);
}

/* ========== Confirm Dialog ========== */

function showConfirm(text, callback) {
  dom.confirmText.textContent = text;
  adminState.confirmCallback = callback;
  dom.confirmOverlay.classList.add('confirm-overlay--open');
}

function hideConfirm() {
  dom.confirmOverlay.classList.remove('confirm-overlay--open');
  adminState.confirmCallback = null;
}

/* ========== Auto-save helpers ========== */

function showSaveStatus(el, text, cssClass) {
  if (!el) return;
  el.textContent = text;
  el.className = 'save-status ' + (cssClass || '');
  clearTimeout(el._timer);
  el._timer = setTimeout(function() {
    el.textContent = '';
    el.className = 'save-status';
  }, 2000);
}

function postConfig(configObj, statusEl) {
  showSaveStatus(statusEl, adminT('saving'), 'save-status--saving');
  return fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(configObj),
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.errors) {
        showSaveStatus(statusEl, adminT('error') + ': ' + Object.values(data.errors).join(', '), 'save-status--error');
        showToast(adminT('saveError') + ': ' + Object.values(data.errors).join(', '), 'error');
      } else {
        showSaveStatus(statusEl, adminT('saved'), 'save-status--ok');
      }
      return data;
    })
    .catch(function(err) {
      console.error('Config save error:', err);
      showSaveStatus(statusEl, adminT('connectionError'), 'save-status--error');
      showToast(adminT('serverConnectionError'), 'error');
    });
}

/* ========== Status via WebSocket ========== */

function formatUptime(seconds) {
  var h = Math.floor(seconds / 3600);
  var m = Math.floor((seconds % 3600) / 60);
  var s = seconds % 60;
  if (h > 0) return h + 'h ' + m + 'm';
  if (m > 0) return m + 'm ' + s + 's';
  return s + 's';
}

function setDot(el, status) {
  el.className = 'status-item__dot';
  if (status === 'running') {
    el.classList.add('status-item__dot--running');
  } else {
    el.classList.add('status-item__dot--stopped');
  }
}

function handleStatusData(data) {
  var c = data.components || {};

  if (c.model) {
    setDot(dom.dotModel, c.model.status);
    var modelDetail = c.model.device.toUpperCase();
    if (c.model.gpu_name) modelDetail += ' (' + c.model.gpu_name + ')';
    dom.detailModel.textContent = modelDetail;
  }
  if (c.vad) {
    setDot(dom.dotVad, c.vad.status);
    dom.detailVad.textContent = c.vad.type;
  }
  if (c.audio_stream) {
    setDot(dom.dotAudio, c.audio_stream.status);
    if (c.audio_stream.status === 'running') {
      dom.detailAudio.textContent = c.audio_stream.device_name + ' ch' + c.audio_stream.channel;
    } else {
      dom.detailAudio.textContent = adminT('stopped');
    }
  }
  if (c.inference_executor) {
    setDot(dom.dotInference, c.inference_executor.status);
    dom.detailInference.textContent = adminT('pending') + ': ' + c.inference_executor.pending_tasks;
  }

  dom.infoClients.textContent = adminT('clients') + ': ' + data.clients;
  dom.infoLanguages.textContent = adminT('languages') + ': ' + (data.active_languages.length > 0 ? data.active_languages.join(', ') : '--');
  dom.infoDevice.textContent = adminT('hardware') + ': ' + data.device.toUpperCase();
  dom.infoUptime.textContent = adminT('uptime') + ': ' + formatUptime(data.uptime);

  // VU Meter with peak hold
  if (data.audio_level_db !== undefined) {
    var db = data.audio_level_db;
    var pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
    dom.audioMeterBar.style.width = pct + '%';
    dom.audioMeterValue.textContent = db.toFixed(1) + ' dB';

    if (data.audio_level_peak !== undefined) {
      var peakPct = Math.max(0, Math.min(100, ((data.audio_level_peak + 60) / 60) * 100));
      dom.audioMeterPeak.style.left = peakPct + '%';
    }
  }
}

function connectStatusWs() {
  adminState.statusWs = new WebSocket(getAdminWsUrl('/api/status/ws'));

  adminState.statusWs.onmessage = function(event) {
    try {
      var data = JSON.parse(event.data);
      handleStatusData(data);
    } catch (e) {
      console.error('Status parse error:', e);
      showToast(adminT('statusParseError'), 'error');
    }
  };

  adminState.statusWs.onclose = function() {
    setTimeout(connectStatusWs, 3000);
  };

  adminState.statusWs.onerror = function(err) {
    console.error('Status WS error:', err);
  };
}

/* ========== Config UI sync ========== */

function syncConfigUI(config) {
  dom.silenceDuration.value = config.silence_duration;
  dom.silenceDurationVal.textContent = config.silence_duration.toFixed(1) + 's';
  dom.minChunkDuration.value = config.min_chunk_duration;
  dom.minChunkDurationVal.textContent = config.min_chunk_duration.toFixed(1) + 's';
  dom.maxChunkDuration.value = config.max_chunk_duration;
  dom.maxChunkDurationVal.textContent = config.max_chunk_duration.toFixed(1) + 's';
  dom.contextOverlap.value = config.context_overlap;
  dom.contextOverlapVal.textContent = config.context_overlap.toFixed(1) + 's';
  dom.defaultTargetLang.value = config.default_target_lang;

  if (config.preprocess_noise_gate !== undefined) {
    syncPreprocessUI(config);
  }
}

function syncPreprocessUI(config) {
  dom.ppNoiseGate.checked = config.preprocess_noise_gate;
  togglePreprocessSettings(dom.ppNoiseGate, dom.ppNoiseGateSettings);
  dom.ppNoiseGateThreshold.value = config.preprocess_noise_gate_threshold;
  dom.ppNoiseGateThresholdVal.textContent = config.preprocess_noise_gate_threshold + ' dB';

  dom.ppNormalize.checked = config.preprocess_normalize;
  togglePreprocessSettings(dom.ppNormalize, dom.ppNormalizeSettings);
  dom.ppNormalizeTarget.value = config.preprocess_normalize_target;
  dom.ppNormalizeTargetVal.textContent = config.preprocess_normalize_target + ' dB';

  dom.ppHighpass.checked = config.preprocess_highpass;
  togglePreprocessSettings(dom.ppHighpass, dom.ppHighpassSettings);
  dom.ppHighpassCutoff.value = config.preprocess_highpass_cutoff;
  dom.ppHighpassCutoffVal.textContent = config.preprocess_highpass_cutoff + ' Hz';

  dom.ppAutoLang.checked = config.preprocess_auto_language;
}

function togglePreprocessSettings(checkbox, settingsEl) {
  if (!settingsEl) return;
  if (checkbox.checked) {
    settingsEl.classList.add('preprocess-item__settings--visible');
  } else {
    settingsEl.classList.remove('preprocess-item__settings--visible');
  }
}

function loadConfig() {
  fetch('/api/config')
    .then(function(r) { return r.json(); })
    .then(function(config) { syncConfigUI(config); })
    .catch(function(err) {
      console.error('Config load error:', err);
      showToast(adminT('configLoadError'), 'error');
    });
}

/* ========== Auto-save: Translation Parameters ========== */

function autoSaveConfig() {
  var config = {
    silence_duration: parseFloat(dom.silenceDuration.value),
    min_chunk_duration: parseFloat(dom.minChunkDuration.value),
    max_chunk_duration: parseFloat(dom.maxChunkDuration.value),
    context_overlap: parseFloat(dom.contextOverlap.value),
    default_target_lang: dom.defaultTargetLang.value,
  };
  postConfig(config, dom.configStatus);
}

/* ========== Auto-save: Preprocessing ========== */

function autoSavePreprocess() {
  var config = {
    preprocess_noise_gate: dom.ppNoiseGate.checked,
    preprocess_noise_gate_threshold: parseFloat(dom.ppNoiseGateThreshold.value),
    preprocess_normalize: dom.ppNormalize.checked,
    preprocess_normalize_target: parseFloat(dom.ppNormalizeTarget.value),
    preprocess_highpass: dom.ppHighpass.checked,
    preprocess_highpass_cutoff: parseInt(dom.ppHighpassCutoff.value),
    preprocess_auto_language: dom.ppAutoLang.checked,
  };
  postConfig(config, dom.preprocessStatus);
}

/* ========== Reset with Confirm ========== */

function resetConfig() {
  showConfirm(adminT('confirmResetText'), function() {
    var defaults = {
      silence_duration: 0.8,
      min_chunk_duration: 1.5,
      max_chunk_duration: 12.0,
      context_overlap: 0.5,
      default_target_lang: 'ces',
      preprocess_noise_gate: false,
      preprocess_noise_gate_threshold: -40.0,
      preprocess_normalize: false,
      preprocess_normalize_target: -3.0,
      preprocess_highpass: false,
      preprocess_highpass_cutoff: 80,
      preprocess_auto_language: false,
    };

    fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(defaults),
    })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (!data.errors) {
          syncConfigUI(data);
          showToast(adminT('defaultsRestored'), 'success');
        }
      })
      .catch(function(err) {
        console.error('Config reset error:', err);
        showToast(adminT('error') + ': ' + err.message, 'error');
      });
  });
}

/* ========== Config Export/Import ========== */

function exportConfig() {
  fetch('/api/config/export')
    .then(function(r) { return r.json(); })
    .then(function(config) {
      var blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'config.json';
      a.click();
      URL.revokeObjectURL(url);
      showToast(adminT('configExported'), 'success');
    })
    .catch(function(err) {
      console.error('Config export error:', err);
      showToast(adminT('exportError'), 'error');
    });
}

function importConfig() {
  dom.importConfigFile.click();
}

function handleImportFile(e) {
  var file = e.target.files[0];
  if (!file) return;
  if (file.size > 102400) {
    showToast(adminT('fileTooLarge'), 'error');
    e.target.value = '';
    return;
  }
  var reader = new FileReader();
  reader.onload = function(ev) {
    try {
      var config = JSON.parse(ev.target.result);
      fetch('/api/config/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
        .then(function(r) { return r.json(); })
        .then(function(data) {
          if (data.ok) {
            syncConfigUI(data.config);
            var msg = adminT('configImported') + ' (' + data.imported + ' ' + adminT('keys') + ')';
            if (data.errors) {
              msg += ', ' + Object.keys(data.errors).length + ' rejected';
              console.warn('Config import rejected keys:', data.errors);
            }
            showToast(msg, data.errors ? 'warning' : 'success');
          } else {
            showToast(adminT('importError') + ': ' + (data.error || ''), 'error');
          }
        })
        .catch(function(err) {
          console.error('Config import error:', err);
          showToast(adminT('connectionError'), 'error');
        });
    } catch (err) {
      showToast(adminT('invalidJson'), 'error');
    }
  };
  reader.readAsText(file);
  e.target.value = '';
}

/* ========== Devices ========== */

function fetchDevices() {
  fetch('/api/devices')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      adminState.devicesCache = data.devices || [];
      dom.deviceSelect.innerHTML = '';

      var defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.textContent = adminT('systemDefault');
      dom.deviceSelect.appendChild(defaultOpt);

      adminState.devicesCache.forEach(function(dev) {
        var opt = document.createElement('option');
        opt.value = dev.index;
        opt.textContent = dev.name + ' (' + dev.max_input_channels + ' ch)';
        if (dev.is_default) opt.textContent += ' [default]';
        dom.deviceSelect.appendChild(opt);
      });

      if (data.current_device_index !== null) {
        dom.deviceSelect.value = data.current_device_index;
      } else {
        dom.deviceSelect.value = '';
      }

      updateChannelOptions(data.current_device_index, data.current_channel);
    })
    .catch(function(err) {
      console.error('Devices load error:', err);
      showToast(adminT('deviceLoadError'), 'error');
    });
}

function updateChannelOptions(deviceIndex, currentChannel) {
  dom.channelSelect.innerHTML = '';
  var maxChannels = 1;

  if (deviceIndex !== null && deviceIndex !== '') {
    var dev = adminState.devicesCache.find(function(d) { return d.index === parseInt(deviceIndex); });
    if (dev) maxChannels = dev.max_input_channels;
  }

  for (var i = 0; i < maxChannels; i++) {
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = (i + 1) + (maxChannels === 1 ? ' (' + adminT('mono') + ')' : '');
    dom.channelSelect.appendChild(opt);
  }

  if (currentChannel !== undefined && currentChannel < maxChannels) {
    dom.channelSelect.value = currentChannel;
  }
}

function applyDeviceSelection() {
  var deviceIndex = dom.deviceSelect.value === '' ? null : parseInt(dom.deviceSelect.value);
  var channel = parseInt(dom.channelSelect.value);

  dom.applyDevice.disabled = true;
  dom.applyDevice.textContent = adminT('loading');

  fetch('/api/devices/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device_index: deviceIndex, channel: channel }),
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) {
        dom.applyDevice.textContent = adminT('ok');
        showToast(adminT('deviceChanged'), 'success');
      } else {
        dom.applyDevice.textContent = adminT('error');
        showToast(adminT('error') + ': ' + (data.error || adminT('unknownError')), 'error');
      }
    })
    .catch(function(err) {
      console.error('Device select error:', err);
      dom.applyDevice.textContent = adminT('error');
      showToast(adminT('error') + ': ' + err.message, 'error');
    })
    .finally(function() {
      setTimeout(function() {
        dom.applyDevice.disabled = false;
        dom.applyDevice.textContent = adminT('apply');
      }, 1500);
    });
}

/* ========== Audio History Graph ========== */

function fetchAudioHistory() {
  fetch('/api/audio-history')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      adminState.audioHistoryData = data;
      drawAudioHistory();
    })
    .catch(function(err) {
      console.error('Audio history error:', err);
      showToast(adminT('audioHistoryError'), 'error');
    });
}

function drawAudioHistory() {
  var canvas = dom.audioHistoryCanvas;
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var w = canvas.width;
  var h = canvas.height;
  var isDark = document.documentElement.getAttribute('data-theme') !== 'light';

  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = isDark ? '#0d0d12' : '#f0f0f5';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = isDark ? '#2a2a3a' : '#e2e4ea';
  ctx.lineWidth = 1;
  [-40, -20, 0].forEach(function(db) {
    var y = h - ((db + 60) / 60) * h;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  });

  if (adminState.audioHistoryData.length < 2) return;

  var data = adminState.audioHistoryData;
  var step = w / 60;

  ctx.strokeStyle = '#22c55e';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (var i = 0; i < data.length; i++) {
    var x = w - (data.length - 1 - i) * step;
    var y = h - ((data[i].db + 60) / 60) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (var i = 0; i < data.length; i++) {
    var x = w - (data.length - 1 - i) * step;
    var y = h - ((data[i].peak + 60) / 60) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

/* ========== Metrics ========== */

function fetchMetrics() {
  fetch('/api/metrics')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      dom.metricTranslations.textContent = data.total_translations;
      dom.metricEncoder.textContent = data.avg_encoder_ms + ' ms';
      dom.metricDecoder.textContent = data.avg_decoder_ms + ' ms';
      if (data.gpu_memory_used_mb !== null) {
        dom.metricGpu.textContent = data.gpu_memory_used_mb + ' / ' + data.gpu_memory_total_mb + ' MB';
      } else {
        dom.metricGpu.textContent = 'N/A (CPU)';
      }
    })
    .catch(function(err) {
      console.error('Metrics load error:', err);
      showToast(adminT('metricsLoadError'), 'error');
    });
}

/* ========== Sessions ========== */

function fetchSessions() {
  fetch('/api/sessions')
    .then(function(r) { return r.json(); })
    .then(function(sessions) {
      if (sessions.length === 0) {
        dom.sessionsBody.innerHTML = '';
        var emptyTr = document.createElement('tr');
        var emptyTd = document.createElement('td');
        emptyTd.colSpan = 4;
        emptyTd.className = 'sessions-table__empty';
        emptyTd.textContent = adminT('noClients');
        emptyTr.appendChild(emptyTd);
        dom.sessionsBody.appendChild(emptyTr);
        return;
      }
      dom.sessionsBody.innerHTML = '';
      sessions.forEach(function(s) {
        var tr = document.createElement('tr');
        var cells = [s.id, s.lang, s.ip, formatUptime(s.connected_for)];
        cells.forEach(function(text) {
          var td = document.createElement('td');
          td.textContent = text;
          tr.appendChild(td);
        });
        dom.sessionsBody.appendChild(tr);
      });
    })
    .catch(function(err) {
      console.error('Sessions load error:', err);
      showToast(adminT('sessionsLoadError'), 'error');
    });
}

/* ========== Translation History ========== */

function fetchTranslations() {
  fetch('/api/translations')
    .then(function(r) { return r.json(); })
    .then(function(entries) {
      if (entries.length === 0) {
        dom.translationHistory.innerHTML = '';
        var emptyDiv = document.createElement('div');
        emptyDiv.className = 'translation-history__empty';
        emptyDiv.textContent = adminT('noTranslations');
        dom.translationHistory.appendChild(emptyDiv);
        return;
      }
      dom.translationHistory.innerHTML = '';
      entries.reverse().forEach(function(entry) {
        var el = document.createElement('div');
        el.className = 'translation-entry';
        var timeSpan = document.createElement('span');
        timeSpan.className = 'translation-entry__time';
        timeSpan.textContent = entry.time;
        el.appendChild(timeSpan);
        Object.keys(entry.translations).forEach(function(lang) {
          var langSpan = document.createElement('span');
          langSpan.className = 'translation-entry__lang';
          langSpan.textContent = lang;
          el.appendChild(langSpan);
          el.appendChild(document.createTextNode(' ' + entry.translations[lang]));
          el.appendChild(document.createElement('br'));
        });
        dom.translationHistory.appendChild(el);
      });
    })
    .catch(function(err) {
      console.error('Translations load error:', err);
      showToast(adminT('translationsLoadError'), 'error');
    });
}

/* ========== Log WebSocket ========== */

function addLogEntry(entry) {
  var el = document.createElement('div');
  el.className = 'log-entry log-entry--' + entry.level;
  el.textContent = '[' + entry.time + '] ' + entry.level + ' ' + entry.message;
  dom.logContainer.appendChild(el);

  while (dom.logContainer.children.length > MAX_LOG_ENTRIES) {
    dom.logContainer.removeChild(dom.logContainer.firstChild);
  }

  if (adminState.autoScroll) {
    dom.logContainer.scrollTop = dom.logContainer.scrollHeight;
  }
}

function connectLogWs() {
  adminState.logWs = new WebSocket(getAdminWsUrl('/api/logs'));

  adminState.logWs.onmessage = function(event) {
    try {
      var data = JSON.parse(event.data);
      if (data.type === 'history' && data.entries) {
        data.entries.forEach(addLogEntry);
      } else if (data.type === 'log' && data.entry) {
        addLogEntry(data.entry);
      }
    } catch (e) {
      console.error('Log parse error:', e);
      showToast(adminT('logParseError'), 'error');
    }
  };

  adminState.logWs.onclose = function() {
    setTimeout(connectLogWs, 3000);
  };

  adminState.logWs.onerror = function(err) {
    console.error('Log WS error:', err);
  };
}

/* ========== Event Listeners ========== */

function initEvents() {
  dom.themeToggle.addEventListener('click', toggleTheme);

  // Language selector
  if (dom.adminLangSelect) {
    dom.adminLangSelect.value = adminState.lang;
    dom.adminLangSelect.addEventListener('change', function() {
      adminState.lang = dom.adminLangSelect.value;
      localStorage.setItem('adminLang', adminState.lang);
      updateAdminUI();
    });
  }

  // Device selection
  dom.deviceSelect.addEventListener('change', function() {
    var val = dom.deviceSelect.value;
    updateChannelOptions(val === '' ? null : parseInt(val), 0);
  });
  dom.applyDevice.addEventListener('click', applyDeviceSelection);

  // Translation parameters — auto-save
  function bindAutoSlider(slider, display, suffix, saveFn) {
    slider.addEventListener('input', function() {
      display.textContent = parseFloat(slider.value).toFixed(1) + suffix;
    });
    slider.addEventListener('change', saveFn);
  }

  bindAutoSlider(dom.silenceDuration, dom.silenceDurationVal, 's', autoSaveConfig);
  bindAutoSlider(dom.minChunkDuration, dom.minChunkDurationVal, 's', autoSaveConfig);
  bindAutoSlider(dom.maxChunkDuration, dom.maxChunkDurationVal, 's', autoSaveConfig);
  bindAutoSlider(dom.contextOverlap, dom.contextOverlapVal, 's', autoSaveConfig);

  dom.defaultTargetLang.addEventListener('change', autoSaveConfig);
  dom.resetConfig.addEventListener('click', resetConfig);

  // Export/Import
  dom.exportConfig.addEventListener('click', exportConfig);
  dom.importConfig.addEventListener('click', importConfig);
  dom.importConfigFile.addEventListener('change', handleImportFile);

  // Preprocessing — auto-save
  dom.ppNoiseGate.addEventListener('change', function() {
    togglePreprocessSettings(dom.ppNoiseGate, dom.ppNoiseGateSettings);
    autoSavePreprocess();
  });
  dom.ppNormalize.addEventListener('change', function() {
    togglePreprocessSettings(dom.ppNormalize, dom.ppNormalizeSettings);
    autoSavePreprocess();
  });
  dom.ppHighpass.addEventListener('change', function() {
    togglePreprocessSettings(dom.ppHighpass, dom.ppHighpassSettings);
    autoSavePreprocess();
  });
  dom.ppAutoLang.addEventListener('change', autoSavePreprocess);

  dom.ppNoiseGateThreshold.addEventListener('input', function() {
    dom.ppNoiseGateThresholdVal.textContent = parseInt(dom.ppNoiseGateThreshold.value) + ' dB';
  });
  dom.ppNoiseGateThreshold.addEventListener('change', autoSavePreprocess);

  dom.ppNormalizeTarget.addEventListener('input', function() {
    dom.ppNormalizeTargetVal.textContent = parseInt(dom.ppNormalizeTarget.value) + ' dB';
  });
  dom.ppNormalizeTarget.addEventListener('change', autoSavePreprocess);

  dom.ppHighpassCutoff.addEventListener('input', function() {
    dom.ppHighpassCutoffVal.textContent = parseInt(dom.ppHighpassCutoff.value) + ' Hz';
  });
  dom.ppHighpassCutoff.addEventListener('change', autoSavePreprocess);

  // Log
  dom.logAutoScroll.addEventListener('change', function() {
    adminState.autoScroll = dom.logAutoScroll.checked;
  });
  dom.clearLog.addEventListener('click', function() {
    dom.logContainer.innerHTML = '';
  });

  // Confirm dialog
  dom.confirmOk.addEventListener('click', function() {
    if (adminState.confirmCallback) adminState.confirmCallback();
    hideConfirm();
  });
  dom.confirmCancel.addEventListener('click', hideConfirm);
  dom.confirmOverlay.addEventListener('click', function(e) {
    if (e.target === dom.confirmOverlay) hideConfirm();
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && dom.confirmOverlay.classList.contains('confirm-overlay--open')) {
      hideConfirm();
    }
  });

  // Translation history refresh
  dom.refreshTranslations.addEventListener('click', fetchTranslations);
}

/* ========== Periodic Data Fetch ========== */

function fetchPeriodicData() {
  fetchMetrics();
  fetchSessions();
  fetchAudioHistory();
}

/* ========== Init ========== */

function init() {
  applyTheme();
  updateAdminUI();
  initEvents();
  loadConfig();
  fetchDevices();
  connectStatusWs();
  connectLogWs();
  fetchTranslations();
  fetchPeriodicData();

  setInterval(fetchPeriodicData, 5000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
