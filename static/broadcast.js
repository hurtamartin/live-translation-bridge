// Browser microphone broadcaster for AUDIO_SOURCE=network.
// Captures the local mic, resamples to 16 kHz mono, and streams Int16 PCM frames
// to /api/audio/ingest over WebSocket. CSP-safe: no inline script, token read from
// a data attribute on #broadcast.
(function () {
    "use strict";

    var root = document.getElementById("broadcast");
    var TOKEN = root.dataset.wsToken || "";
    var AUDIO_SOURCE = (root.dataset.audioSource || "device").toLowerCase();

    var btn = document.getElementById("toggle");
    var dot = document.getElementById("dot");
    var statusText = document.getElementById("statusText");
    var meterBar = document.getElementById("meterBar");
    var logEl = document.getElementById("log");
    var modeWarning = document.getElementById("modeWarning");

    var TARGET_RATE = 16000;
    var FRAME_SAMPLES = 512;          // 512 Int16 = 1024 bytes, under default WS_MAX_SIZE (2048)
    var MAX_BUFFERED = 1 << 20;       // 1 MB WS backpressure cap

    var ws = null, audioCtx = null, stream = null, source = null;
    var processor = null, mute = null, resample = null;
    var pcmCarry = new Float32Array(0);
    var running = false;

    if (AUDIO_SOURCE !== "network") {
        modeWarning.hidden = false;
    }

    function log(msg) {
        var t = new Date().toLocaleTimeString();
        logEl.textContent = "[" + t + "] " + msg + "\n" + logEl.textContent;
    }

    function setStatus(text, cls) {
        statusText.textContent = text;
        dot.className = "dot" + (cls ? " " + cls : "");
    }

    // Stateful linear resampler from inRate -> TARGET_RATE, preserving phase across blocks.
    function makeResampler(inRate) {
        var ratio = inRate / TARGET_RATE; // input samples consumed per output sample
        var pos = 0;
        var prevLast = 0;
        var havePrev = false;
        return function (input) {
            var n = input.length;
            if (n === 0) return new Float32Array(0);
            var out = [];
            while (pos < n) {
                var i = Math.floor(pos);
                var frac = pos - i;
                var s0 = (i < 0) ? (havePrev ? prevLast : input[0]) : input[i];
                var i1 = i + 1;
                var s1;
                if (i1 < 0) s1 = havePrev ? prevLast : input[0];
                else if (i1 < n) s1 = input[i1];
                else s1 = input[n - 1];
                out.push(s0 + (s1 - s0) * frac);
                pos += ratio;
            }
            pos -= n;
            prevLast = input[n - 1];
            havePrev = true;
            return Float32Array.from(out);
        };
    }

    function floatToInt16(f32) {
        var out = new Int16Array(f32.length);
        for (var i = 0; i < f32.length; i++) {
            var s = Math.max(-1, Math.min(1, f32[i]));
            out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return out;
    }

    function onAudio(e) {
        var input = e.inputBuffer.getChannelData(0);

        // VU meter (RMS -> dB -> 0..100%)
        var sum = 0;
        for (var k = 0; k < input.length; k++) sum += input[k] * input[k];
        var rms = Math.sqrt(sum / input.length);
        var db = rms > 1e-7 ? 20 * Math.log10(rms) : -60;
        var pct = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
        meterBar.style.width = pct.toFixed(0) + "%";

        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        var res = resample(input);
        // Concatenate carry + new resampled samples
        var merged = new Float32Array(pcmCarry.length + res.length);
        merged.set(pcmCarry, 0);
        merged.set(res, pcmCarry.length);

        var offset = 0;
        while (merged.length - offset >= FRAME_SAMPLES) {
            if (ws.bufferedAmount > MAX_BUFFERED) break; // drop on backpressure
            var int16 = floatToInt16(merged.subarray(offset, offset + FRAME_SAMPLES));
            ws.send(int16.buffer);
            offset += FRAME_SAMPLES;
        }
        pcmCarry = merged.slice(offset);
    }

    function wsUrl() {
        var proto = location.protocol === "https:" ? "wss:" : "ws:";
        return proto + "//" + location.host + "/api/audio/ingest?token=" + encodeURIComponent(TOKEN);
    }

    async function start() {
        try {
            setStatus("Requesting microphone…");
            stream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
                video: false
            });
        } catch (err) {
            setStatus("Microphone denied", "err");
            log("getUserMedia failed: " + err.message);
            return;
        }

        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (err) {
            setStatus("AudioContext failed", "err");
            log(err.message);
            stopTracks();
            return;
        }
        resample = makeResampler(audioCtx.sampleRate);
        pcmCarry = new Float32Array(0);
        log("Capturing at " + audioCtx.sampleRate + " Hz, resampling to " + TARGET_RATE + " Hz");

        ws = new WebSocket(wsUrl());
        ws.binaryType = "arraybuffer";

        ws.onopen = function () {
            running = true;
            btn.textContent = "Stop broadcasting";
            btn.classList.add("live");
            setStatus("Live", "on");
            log("Connected to /api/audio/ingest");

            source = audioCtx.createMediaStreamSource(stream);
            processor = audioCtx.createScriptProcessor(4096, 1, 1);
            mute = audioCtx.createGain();
            mute.gain.value = 0; // route to destination silently so onaudioprocess fires (no feedback)
            processor.onaudioprocess = onAudio;
            source.connect(processor);
            processor.connect(mute);
            mute.connect(audioCtx.destination);
        };

        ws.onclose = function (ev) {
            if (ev.code === 4401) log("Auth failed (token expired?) — reload the page");
            else if (ev.code === 4409) log("Another broadcaster is already streaming");
            else if (ev.code === 4400) log("Server is not in network mode (AUDIO_SOURCE=network required)");
            else if (running) log("Connection closed (code " + ev.code + ")");
            stop();
        };

        ws.onerror = function () { log("WebSocket error"); };
    }

    function stopTracks() {
        if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; }
    }

    function stop() {
        running = false;
        btn.textContent = "Start broadcasting";
        btn.classList.remove("live");
        setStatus("Idle");
        meterBar.style.width = "0%";
        try { if (processor) { processor.disconnect(); processor.onaudioprocess = null; } } catch (e) {}
        try { if (source) source.disconnect(); } catch (e) {}
        try { if (mute) mute.disconnect(); } catch (e) {}
        try { if (audioCtx) audioCtx.close(); } catch (e) {}
        processor = source = mute = audioCtx = null;
        stopTracks();
        if (ws) {
            var w = ws; ws = null;
            try { if (w.readyState === WebSocket.OPEN || w.readyState === WebSocket.CONNECTING) w.close(); } catch (e) {}
        }
    }

    btn.addEventListener("click", function () {
        if (running || (ws && ws.readyState === WebSocket.CONNECTING)) stop();
        else start();
    });

    window.addEventListener("beforeunload", stop);
})();
