'use strict';

/* ========== State ========== */

var logWs = null;
var statusWs = null;
var autoScroll = true;
var devicesCache = [];
var saveDebounceTimer = null;
var confirmCallback = null;

/* ========== DOM ========== */

var dom = {
  themeToggle: document.getElementById('themeToggle'),
  themeIcon: document.getElementById('themeIcon'),
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

/* ========== Toast Notifications (D) ========== */

function showToast(message, type) {
  var toast = document.createElement('div');
  toast.className = 'toast toast--' + (type || 'info');
  toast.textContent = message;
  dom.toastContainer.appendChild(toast);
  // Trigger reflow for animation
  toast.offsetHeight;
  toast.classList.add('toast--visible');
  setTimeout(function() {
    toast.classList.remove('toast--visible');
    toast.addEventListener('transitionend', function() { toast.remove(); });
  }, 3000);
}

/* ========== Confirm Dialog (C) ========== */

function showConfirm(text, callback) {
  dom.confirmText.textContent = text;
  confirmCallback = callback;
  dom.confirmOverlay.classList.add('confirm-overlay--open');
}

function hideConfirm() {
  dom.confirmOverlay.classList.remove('confirm-overlay--open');
  confirmCallback = null;
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
  showSaveStatus(statusEl, 'Ukladam...', 'save-status--saving');
  return fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(configObj),
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.errors) {
        showSaveStatus(statusEl, 'Chyba: ' + Object.values(data.errors).join(', '), 'save-status--error');
        showToast('Chyba ukladani: ' + Object.values(data.errors).join(', '), 'error');
      } else {
        showSaveStatus(statusEl, 'Ulozeno', 'save-status--ok');
      }
      return data;
    })
    .catch(function() {
      showSaveStatus(statusEl, 'Chyba spojeni', 'save-status--error');
      showToast('Chyba spojeni se serverem', 'error');
    });
}

/* ========== Status via WebSocket (E) ========== */

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
      dom.detailAudio.textContent = 'Stopped';
    }
  }
  if (c.inference_executor) {
    setDot(dom.dotInference, c.inference_executor.status);
    dom.detailInference.textContent = 'Pending: ' + c.inference_executor.pending_tasks;
  }

  dom.infoClients.textContent = 'Klienti: ' + data.clients;
  dom.infoLanguages.textContent = 'Jazyky: ' + (data.active_languages.length > 0 ? data.active_languages.join(', ') : '--');
  dom.infoDevice.textContent = 'Hardware: ' + data.device.toUpperCase();
  dom.infoUptime.textContent = 'Uptime: ' + formatUptime(data.uptime);

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
  var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  var url = protocol + '//' + window.location.host + '/api/status/ws';

  statusWs = new WebSocket(url);

  statusWs.onmessage = function(event) {
    try {
      var data = JSON.parse(event.data);
      handleStatusData(data);
    } catch (e) {
      console.error('Status parse error:', e);
    }
  };

  statusWs.onclose = function() {
    setTimeout(connectStatusWs, 3000);
  };

  statusWs.onerror = function() {};
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
    .catch(function(err) { console.error('Config load error:', err); });
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

var ppSaveTimer = null;

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

/* ========== Reset with Confirm (C) ========== */

function resetConfig() {
  showConfirm('Opravdu chcete resetovat vsechna nastaveni na vychozi hodnoty? Tato akce je nevratna.', function() {
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
          showToast('Vychozi nastaveni obnovena', 'success');
        }
      })
      .catch(function(err) {
        showToast('Chyba: ' + err.message, 'error');
      });
  });
}

/* ========== Config Export/Import (B) ========== */

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
      showToast('Konfigurace exportovana', 'success');
    })
    .catch(function() { showToast('Chyba exportu', 'error'); });
}

function importConfig() {
  dom.importConfigFile.click();
}

function handleImportFile(e) {
  var file = e.target.files[0];
  if (!file) return;
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
            showToast('Konfigurace importovana (' + data.imported + ' klicu)', 'success');
          } else {
            showToast('Chyba importu: ' + (data.error || ''), 'error');
          }
        })
        .catch(function() { showToast('Chyba spojeni', 'error'); });
    } catch (err) {
      showToast('Neplatny JSON soubor', 'error');
    }
  };
  reader.readAsText(file);
  // Reset file input
  e.target.value = '';
}

/* ========== Devices ========== */

function fetchDevices() {
  fetch('/api/devices')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      devicesCache = data.devices || [];
      dom.deviceSelect.innerHTML = '';

      var defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.textContent = '-- Systemovy vychozi --';
      dom.deviceSelect.appendChild(defaultOpt);

      devicesCache.forEach(function(dev) {
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
    .catch(function() { showToast('Chyba nacitani zarizeni', 'error'); });
}

function updateChannelOptions(deviceIndex, currentChannel) {
  dom.channelSelect.innerHTML = '';
  var maxChannels = 1;

  if (deviceIndex !== null && deviceIndex !== '') {
    var dev = devicesCache.find(function(d) { return d.index === parseInt(deviceIndex); });
    if (dev) maxChannels = dev.max_input_channels;
  }

  for (var i = 0; i < maxChannels; i++) {
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = (i + 1) + (maxChannels === 1 ? ' (Mono)' : '');
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
  dom.applyDevice.textContent = 'Nacitam...';

  fetch('/api/devices/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device_index: deviceIndex, channel: channel }),
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) {
        dom.applyDevice.textContent = 'OK!';
        showToast('Audio zarizeni zmeneno', 'success');
      } else {
        dom.applyDevice.textContent = 'Chyba';
        showToast('Chyba: ' + (data.error || 'Neznama chyba'), 'error');
      }
    })
    .catch(function(err) {
      dom.applyDevice.textContent = 'Chyba';
      showToast('Chyba: ' + err.message, 'error');
    })
    .finally(function() {
      setTimeout(function() {
        dom.applyDevice.disabled = false;
        dom.applyDevice.textContent = 'Pouzit';
      }, 1500);
    });
}

/* ========== Audio History Graph (F) ========== */

var audioHistoryData = [];

function fetchAudioHistory() {
  fetch('/api/audio-history')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      audioHistoryData = data;
      drawAudioHistory();
    })
    .catch(function() {});
}

function drawAudioHistory() {
  var canvas = dom.audioHistoryCanvas;
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var w = canvas.width;
  var h = canvas.height;
  var isDark = document.documentElement.getAttribute('data-theme') !== 'light';

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = isDark ? '#0d0d12' : '#f0f0f5';
  ctx.fillRect(0, 0, w, h);

  // Grid lines at -40, -20, 0 dB
  ctx.strokeStyle = isDark ? '#2a2a3a' : '#e2e4ea';
  ctx.lineWidth = 1;
  [-40, -20, 0].forEach(function(db) {
    var y = h - ((db + 60) / 60) * h;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  });

  if (audioHistoryData.length < 2) return;

  var data = audioHistoryData;
  var step = w / 60;

  // Draw RMS level
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

  // Draw peak
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

/* ========== Metrics (G) ========== */

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
    .catch(function() {});
}

/* ========== Sessions (G) ========== */

function fetchSessions() {
  fetch('/api/sessions')
    .then(function(r) { return r.json(); })
    .then(function(sessions) {
      if (sessions.length === 0) {
        dom.sessionsBody.innerHTML = '<tr><td colspan="4" class="sessions-table__empty">Zadni klienti</td></tr>';
        return;
      }
      dom.sessionsBody.innerHTML = '';
      sessions.forEach(function(s) {
        var tr = document.createElement('tr');
        tr.innerHTML = '<td>' + s.id + '</td><td>' + s.lang + '</td><td>' + s.ip + '</td><td>' + formatUptime(s.connected_for) + '</td>';
        dom.sessionsBody.appendChild(tr);
      });
    })
    .catch(function() {});
}

/* ========== Translation History (G) ========== */

function fetchTranslations() {
  fetch('/api/translations')
    .then(function(r) { return r.json(); })
    .then(function(entries) {
      if (entries.length === 0) {
        dom.translationHistory.innerHTML = '<div class="translation-history__empty">Zatim zadne preklady</div>';
        return;
      }
      dom.translationHistory.innerHTML = '';
      // Show newest first
      entries.reverse().forEach(function(entry) {
        var el = document.createElement('div');
        el.className = 'translation-entry';
        var langs = Object.keys(entry.translations).map(function(lang) {
          return '<span class="translation-entry__lang">' + lang + '</span> ' + entry.translations[lang];
        }).join('<br>');
        el.innerHTML = '<span class="translation-entry__time">' + entry.time + '</span>' + langs;
        dom.translationHistory.appendChild(el);
      });
    })
    .catch(function() {});
}

/* ========== Log WebSocket ========== */

var MAX_LOG_ENTRIES = 200;

function addLogEntry(entry) {
  var el = document.createElement('div');
  el.className = 'log-entry log-entry--' + entry.level;
  el.textContent = '[' + entry.time + '] ' + entry.level + ' ' + entry.message;
  dom.logContainer.appendChild(el);

  while (dom.logContainer.children.length > MAX_LOG_ENTRIES) {
    dom.logContainer.removeChild(dom.logContainer.firstChild);
  }

  if (autoScroll) {
    dom.logContainer.scrollTop = dom.logContainer.scrollHeight;
  }
}

function connectLogWs() {
  var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  var url = protocol + '//' + window.location.host + '/api/logs';

  logWs = new WebSocket(url);

  logWs.onmessage = function(event) {
    try {
      var data = JSON.parse(event.data);
      if (data.type === 'history' && data.entries) {
        data.entries.forEach(addLogEntry);
      } else if (data.type === 'log' && data.entry) {
        addLogEntry(data.entry);
      }
    } catch (e) {
      console.error('Log parse error:', e);
    }
  };

  logWs.onclose = function() {
    setTimeout(connectLogWs, 3000);
  };

  logWs.onerror = function() {};
}

/* ========== Event Listeners ========== */

function initEvents() {
  dom.themeToggle.addEventListener('click', toggleTheme);

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
    autoScroll = dom.logAutoScroll.checked;
  });
  dom.clearLog.addEventListener('click', function() {
    dom.logContainer.innerHTML = '';
  });

  // Confirm dialog
  dom.confirmOk.addEventListener('click', function() {
    if (confirmCallback) confirmCallback();
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
  initEvents();
  loadConfig();
  fetchDevices();
  connectStatusWs();
  connectLogWs();
  fetchTranslations();
  fetchPeriodicData();

  // Refresh metrics, sessions, audio history every 5s
  setInterval(fetchPeriodicData, 5000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
