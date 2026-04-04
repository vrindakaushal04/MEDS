const LS_API = "meds_api_base";

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
let recordingCount = 0;
let moodChart;
const moodLabels = [];
const moodScores = [];
let audioCtx;
let analyser;
let waveRaf;
let mediaStream;

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
    return raw || "http://127.0.0.1:5000";
}

function apiUrl(path) {
    return `${getApiBase()}${path}`;
}

function loadStoredApi() {
    const s = localStorage.getItem(LS_API);
    if (s) apiBaseInput.value = s;
    else apiBaseInput.value = "http://127.0.0.1:5000";
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
    gradient.addColorStop(0, "rgba(94, 234, 212, 0.35)");
    gradient.addColorStop(1, "rgba(94, 234, 212, 0)");

    moodChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: moodLabels,
            datasets: [
                {
                    label: "Calmness",
                    data: moodScores,
                    borderColor: "#5eead4",
                    borderWidth: 2.5,
                    pointRadius: 3,
                    tension: 0.35,
                    fill: true,
                    backgroundColor: gradient,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(255,255,255,0.06)" },
                    ticks: { color: "#94a3b8" },
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8", maxRotation: 0 },
                },
            },
        },
    });
}

function riskToCalmness(finalEmotion, combinedRisk) {
    const numericRisk = Number(combinedRisk);
    const risk = Number.isFinite(numericRisk) ? numericRisk : 0;
    let score = Math.round((1 - risk) * 100);
    const penalties = {
        high_distress: 18,
        distressed: 12,
        agitated: 8,
        low_mood: 6,
        emotional_mismatch: 5,
        sad: 4,
        anxious: 4,
        angry: 5,
    };
    score -= penalties[finalEmotion] || 0;
    return Math.max(0, Math.min(100, score));
}

function pushMoodPoint(finalEmotion, combinedRisk) {
    recordingCount += 1;
    moodLabels.push(`#${recordingCount}`);
    moodScores.push(riskToCalmness(finalEmotion, combinedRisk));
    if (moodLabels.length > 12) {
        moodLabels.shift();
        moodScores.shift();
    }
    moodChart.update();
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
