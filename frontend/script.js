let mediaRecorder;
let audioChunks = [];
let voiceEnabled = true;
let recordingCount = 0;
let moodChart;
const moodLabels = [];
const moodScores = [];

const statusText = document.getElementById("status");
const statusDot = document.getElementById("statusIndicator");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const voiceToggleBtn = document.getElementById("voiceToggleBtn");
const analyzeTextBtn = document.getElementById("analyzeTextBtn");
const demoText = document.getElementById("demoText");

const emotionEmojiMap = {
    high_distress: "😟",
    distressed: "😔",
    emotional_mismatch: "🫤",
    low_mood: "😕",
    agitated: "😤",
    neutral: "🙂",
    happy: "😊"
};

function updateStatus(message, type = 'accent') {
    statusText.innerText = `Status: ${message}`;
    statusDot.style.backgroundColor = `var(--${type === 'error' ? 'error' : 'accent'})`;
}

function initMoodChart() {
    const ctx = document.getElementById("moodChart").getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 173, 181, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 173, 181, 0)');

    moodChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: moodLabels,
            datasets: [{
                label: "Emotional Well-being",
                data: moodScores,
                borderColor: "#00adb5",
                borderWidth: 3,
                pointBackgroundColor: "#00adb5",
                pointRadius: 4,
                tension: 0.4,
                fill: true,
                backgroundColor: gradient
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });
}

function riskToMoodScore(finalEmotion, combinedRisk) {
    const numericRisk = Number(combinedRisk) || 0;
    let baseScore = Math.round((1 - numericRisk) * 100);
    const penalties = { high_distress: 12, distressed: 8, agitated: 6, low_mood: 5, emotional_mismatch: 4 };
    baseScore -= (penalties[finalEmotion] || 0);
    return Math.max(0, Math.min(100, baseScore));
}

function pushMoodPoint(finalEmotion, combinedRisk) {
    recordingCount += 1;
    moodLabels.push(`Entry ${recordingCount}`);
    moodScores.push(riskToMoodScore(finalEmotion, combinedRisk));
    if (moodLabels.length > 10) { moodLabels.shift(); moodScores.shift(); }
    moodChart.update();
}

function speakResponse(text) {
    if (!voiceEnabled || !("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}

voiceToggleBtn.onclick = () => {
    voiceEnabled = !voiceEnabled;
    voiceToggleBtn.querySelector('span').innerText = voiceEnabled ? "Voice: On" : "Voice: Off";
    voiceToggleBtn.querySelector('i').className = voiceEnabled ? "fas fa-volume-up" : "fas fa-volume-mute";
    if (!voiceEnabled) window.speechSynthesis.cancel();
};

recordBtn.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        
        recordBtn.classList.add('recording-active');
        recordBtn.querySelector('span').innerText = "Recording...";
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        updateStatus("Listening...", "success");
    } catch (error) {
        updateStatus("Mic Access Denied", "error");
    }
};

stopBtn.onclick = () => {
    if (mediaRecorder?.state === "recording") {
        mediaRecorder.stop();
        recordBtn.classList.remove('recording-active');
        recordBtn.querySelector('span').innerText = "Start Recording";
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        
        mediaRecorder.onstop = async () => {
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("audio", blob, "recording.webm");
            processAnalysis("http://127.0.0.1:5000/analyze", formData, true);
        };
    }
};

analyzeTextBtn.onclick = () => {
    const text = demoText.value.trim();
    if (!text) return updateStatus("Please enter text", "error");
    processAnalysis("http://127.0.0.1:5000/analyze-text", JSON.stringify({ text }), false);
};

async function processAnalysis(url, body, isFormData) {
    updateStatus("Analyzing resonance...", "accent");
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: isFormData ? {} : { "Content-Type": "application/json" },
            body: body
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Server error");
        
        renderAnalysis(data);
        updateStatus("Analysis Complete", "success");
    } catch (error) {
        updateStatus(error.message, "error");
    }
}

function renderAnalysis(data) {
    const analysis = data.analysis || {};
    document.getElementById("text").innerText = data.text || "No speech detected";
    document.getElementById("finalEmotion").innerText = (data.final_emotion || "neutral").replace('_', ' ');
    document.getElementById("riskScore").innerText = analysis.combined_risk ?? "0.0";
    document.getElementById("reason").innerText = analysis.reason || "Context successfully processed.";
    document.getElementById("response").innerText = data.response || "I'm here for you.";
    document.getElementById("emoji").innerText = emotionEmojiMap[data.final_emotion] || "🙂";
    
    pushMoodPoint(data.final_emotion, analysis.combined_risk);
    speakResponse(data.response);
}

// Initialize on Load
window.onload = initMoodChart;
