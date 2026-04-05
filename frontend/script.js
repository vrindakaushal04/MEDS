const LS_API       = "meds_api_base";
const LS_HISTORY   = "meds_mood_history";   // persisted 7-day log
const HISTORY_DAYS = 7;                      // keep last N days

const emotionEmojiMap = {
    high_distress: "🆘",
    distressed: "😔",
    emotional_mismatch: "🫥",
    low_mood: "😕",
    agitated: "😤",
    neutral: "🙂",
    happy: "😊",
    sad: "💧",
    anxious: "🌧️",
    angry: "🔥",
};

let mediaRecorder;
let audioChunks = [];
let voiceEnabled = true;
let moodChart;
let audioCtx;
let analyser;
let waveRaf;
let mediaStream;

// ── Persistent 7-day history ─────────────────────────────────────────────────
function loadHistory() {
    try {
        const raw = localStorage.getItem(LS_HISTORY);
        const all = raw ? JSON.parse(raw) : [];
        const cutoff = Date.now() - HISTORY_DAYS * 24 * 60 * 60 * 1000;
        return all.filter(p => p.ts >= cutoff);
    } catch { return []; }
}

function saveHistory(points) {
    try {
        localStorage.setItem(LS_HISTORY, JSON.stringify(points));
    } catch {}
}

function addHistoryPoint(emotion, score) {
    const points = loadHistory();
    points.push({ ts: Date.now(), emotion, score });
    saveHistory(points);
    return points;
}

function clearHistory() {
    localStorage.removeItem(LS_HISTORY);
    rebuildChart([]);
}

function formatLabel(ts) {
    const d = new Date(ts);
    const days  = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
    const hh    = String(d.getHours()).padStart(2, "0");
    const mm    = String(d.getMinutes()).padStart(2, "0");
    return `${days[d.getDay()]} ${hh}:${mm}`;
}

function rebuildChart(points) {
    if (!moodChart) return;
    moodChart.data.labels   = points.map(p => formatLabel(p.ts));
    moodChart.data.datasets[0].data = points.map(p => p.score);
    // Colour each point by emotion
    moodChart.data.datasets[0].pointBackgroundColor = points.map(p => emotionColor(p.emotion));
    moodChart.update();
}

const EMOTION_COLORS = {
    happy:              "#34d399",
    neutral:            "#5eead4",
    low_mood:           "#fbbf24",
    anxious:            "#f97316",
    sad:                "#818cf8",
    angry:              "#f87171",
    agitated:           "#fb923c",
    emotional_mismatch: "#a78bfa",
    distressed:         "#ef4444",
    high_distress:      "#dc2626",
};
function emotionColor(e) { return EMOTION_COLORS[e] || "#5eead4"; }

const statusText = document.getElementById("status");
const statusDot = document.getElementById("statusIndicator");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const voiceToggleBtn = document.getElementById("voiceToggleBtn");
const voiceLabel = document.getElementById("voiceLabel");
const analyzeTextBtn = document.getElementById("analyzeTextBtn");
const demoText = document.getElementById("demoText");
const apiBaseInput = document.getElementById("apiBase");
const backendPill = document.getElementById("backendPill");
const loadingBadge = document.getElementById("loadingBadge");
const waveCanvas = document.getElementById("waveCanvas");
const waveCtx = waveCanvas.getContext("2d");

function getApiBase() {
    const raw = (apiBaseInput.value || "").trim().replace(/\/$/, "");
    return raw || "https://meds-08ez.onrender.com/";
}

function apiUrl(path) {
    return `${getApiBase()}${path}`;
}

function loadStoredApi() {
    const s = localStorage.getItem(LS_API);
    if (s) apiBaseInput.value = s;
    else apiBaseInput.value = "https://meds-08ez.onrender.com/";
}

function persistApi() {
    localStorage.setItem(LS_API, getApiBase());
}

function setLoading(on) {
    loadingBadge.classList.toggle("hidden", !on);
    statusDot.classList.toggle("busy", on);
}

function updateStatus(message, type = "accent") {
    statusText.textContent = message;
    statusDot.classList.remove("error", "busy");
    if (type === "error") statusDot.classList.add("error");
}

function drawWavePlaceholder() {
    const w = waveCanvas.width;
    const h = waveCanvas.height;
    waveCtx.clearRect(0, 0, w, h);
    waveCtx.strokeStyle = "rgba(148,163,184,0.25)";
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(0, h / 2);
    waveCtx.lineTo(w, h / 2);
    waveCtx.stroke();
}

function startWaveform(stream) {
    stopWaveform();
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);

    const loop = () => {
        waveRaf = requestAnimationFrame(loop);
        analyser.getByteTimeDomainData(buf);
        const w = waveCanvas.width;
        const h = waveCanvas.height;
        waveCtx.fillStyle = "rgba(0,0,0,0.15)";
        waveCtx.fillRect(0, 0, w, h);
        waveCtx.strokeStyle = "rgba(94,234,212,0.85)";
        waveCtx.lineWidth = 2;
        waveCtx.beginPath();
        const step = w / buf.length;
        for (let i = 0; i < buf.length; i++) {
            const v = buf[i] / 128 - 1;
            const y = h / 2 + v * (h * 0.38);
            if (i === 0) waveCtx.moveTo(0, y);
            else waveCtx.lineTo(i * step, y);
        }
        waveCtx.stroke();
    };
    loop();
}

function stopWaveform() {
    if (waveRaf) cancelAnimationFrame(waveRaf);
    waveRaf = null;
    if (audioCtx) audioCtx.close().catch(() => {});
    audioCtx = null;
    analyser = null;
    drawWavePlaceholder();
}

async function pingBackend() {
    backendPill.textContent = "Backend: …";
    try {
        const res = await fetch(apiUrl("/health"), { method: "GET" });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error();
        backendPill.textContent = "Backend: online";
        backendPill.style.color = "var(--ok)";
    } catch {
        backendPill.textContent = "Backend: offline";
        backendPill.style.color = "var(--danger)";
    }
}

function initMoodChart() {
    const ctx = document.getElementById("moodChart").getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, 0, 260);
    gradient.addColorStop(0, "rgba(94, 234, 212, 0.25)");
    gradient.addColorStop(1, "rgba(94, 234, 212, 0)");

    const history = loadHistory();

    moodChart = new Chart(ctx, {
        type: "line",
        data: {
            labels:   history.map(p => formatLabel(p.ts)),
            datasets: [
                {
                    label: "Wellness Score",
                    data:  history.map(p => p.score),
                    borderColor: "#5eead4",
                    borderWidth: 2.5,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: history.map(p => emotionColor(p.emotion)),
                    tension: 0.38,
                    fill: true,
                    backgroundColor: gradient,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => items[0].label,
                        label: (item) => {
                            const pts = loadHistory();
                            const pt  = pts[item.dataIndex];
                            const em  = pt ? humanizeEmotion(pt.emotion) : "";
                            return ` Score: ${item.raw}  |  ${em}`;
                        },
                    },
                    backgroundColor: "rgba(15,23,42,0.92)",
                    titleColor: "#94a3b8",
                    bodyColor: "#e2e8f0",
                    borderColor: "rgba(94,234,212,0.3)",
                    borderWidth: 1,
                    padding: 10,
                },
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(255,255,255,0.06)" },
                    ticks: { color: "#94a3b8" },
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", maxRotation: 30, font: { size: 11 } },
                },
            },
        },
    });
}

function riskToCalmness(finalEmotion, combinedRisk) {
    const numericRisk = Number(combinedRisk);
    const risk = Number.isFinite(numericRisk) ? numericRisk : 0;

    // Curved scale: maps risk 0→85, 0.5→55, 1.0→20
    // This keeps scores mid-range (20–85) regardless of raw risk value
    let score = Math.round(85 - risk * 65);

    // Emotion modifiers — small nudges to reflect emotional state
    const modifiers = {
        happy:              +8,
        neutral:            +5,
        low_mood:           -5,
        anxious:            -6,
        sad:                -7,
        angry:              -8,
        agitated:           -9,
        emotional_mismatch: -6,
        distressed:         -12,
        high_distress:      -18,
    };
    score += modifiers[finalEmotion] || 0;

    // Floor at 12, ceiling at 92 — never show extreme numbers
    return Math.max(12, Math.min(92, score));
}

function pushMoodPoint(finalEmotion, combinedRisk) {
    const score = riskToCalmness(finalEmotion, combinedRisk);
    const points = addHistoryPoint(finalEmotion, score);
    rebuildChart(points);
}

function speakResponse(text) {
    if (!voiceEnabled || !("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    u.pitch = 1;
    window.speechSynthesis.speak(u);
}

voiceToggleBtn.addEventListener("click", () => {
    voiceEnabled = !voiceEnabled;
    voiceLabel.textContent = voiceEnabled ? "Voice on" : "Voice off";
    if (!voiceEnabled) window.speechSynthesis.cancel();
});

recordBtn.addEventListener("click", async () => {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(mediaStream);
        mediaRecorder.start();
        startWaveform(mediaStream);

        recordBtn.classList.add("recording");
        recordBtn.querySelector("span:last-child").textContent = "Recording…";
        recordBtn.disabled = true;
        stopBtn.disabled = false;

        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        updateStatus("Listening…");
    } catch {
        updateStatus("Microphone blocked", "error");
    }
});

stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.classList.remove("recording");
        recordBtn.querySelector("span:last-child").textContent = "Record";
        recordBtn.disabled = false;
        stopBtn.disabled = true;

        mediaRecorder.onstop = async () => {
            stopWaveform();
            if (mediaStream) {
                mediaStream.getTracks().forEach((t) => t.stop());
                mediaStream = null;
            }
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("audio", blob, "recording.webm");
            await processAnalysis(apiUrl("/analyze"), formData, true);
        };
    }
});

analyzeTextBtn.addEventListener("click", async () => {
    const text = demoText.value.trim();
    if (!text) {
        updateStatus("Add some text first", "error");
        return;
    }
    await processAnalysis(apiUrl("/analyze-text"), JSON.stringify({ text }), false);
});

async function processAnalysis(url, body, isFormData) {
    persistApi();
    setLoading(true);
    updateStatus("Fusing voice + text…");
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: isFormData ? {} : { "Content-Type": "application/json" },
            body,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Server error");
        renderAnalysis(data);
        updateStatus("Done");
    } catch (e) {
        updateStatus(e.message || "Request failed", "error");
    } finally {
        setLoading(false);
    }
}

function humanizeEmotion(key) {
    return (key || "neutral").replaceAll("_", " ");
}

function renderAnalysis(data) {
    const analysis = data.analysis || {};
    const finalKey = data.final_emotion || "neutral";
    const riskRaw = analysis.combined_risk;
    const calm = riskToCalmness(finalKey, riskRaw);

    document.getElementById("text").textContent = data.text || "—";
    document.getElementById("finalEmotion").textContent = humanizeEmotion(finalKey);
    document.getElementById("riskScore").textContent =
        Number.isFinite(Number(riskRaw)) ? String(calm) : "—";
    document.getElementById("riskFill").style.width = `${calm}%`;
    document.getElementById("reason").textContent =
        analysis.reason || "Fusion completed.";
    document.getElementById("response").textContent =
        data.response || "I'm here for you.";
    document.getElementById("emoji").textContent =
        emotionEmojiMap[finalKey] || "🙂";

    document.getElementById("audioLayer").textContent = humanizeEmotion(
        analysis.audio_label || data.audio_emotion?.label
    );
    document.getElementById("textLayer").textContent = humanizeEmotion(
        analysis.text_label || data.text_emotion?.label
    );

    pushMoodPoint(finalKey, riskRaw);
    speakResponse(data.response);
}

apiBaseInput.addEventListener("change", () => {
    persistApi();
    pingBackend();
});

window.addEventListener("load", () => {
    loadStoredApi();
    drawWavePlaceholder();
    initMoodChart();
    pingBackend();
});
