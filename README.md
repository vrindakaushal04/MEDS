# MEDS
Multimodal Emotion Detection System - Voice based Emotion detection model  
Bridging the "Emotion Gap" between spoken words and true feelings.

Developed by Team pENTEX for the Eclipse 6.0 SLM Voice Agents Challenge

## The Problem
Most AI systems rely almost exclusively on lexical data. This creates an "Emotional Gap" which is a failure to detect linguistic incongruence. Without the ability to interpret vocal nuance, human subtext is often missed.

## Our Solution
Our solution is an Emotion AI interface that analyzes user input to detect emotional tone, identify potential emotional inconsistencies, and provide supportive feedback.

The system uses sentiment analysis logic to classify emotions, assigns a risk score, and visualizes emotional trends over time using an interactive dashboard.

This helps bridge the emotional gap by making AI responses more empathetic and context-aware.

## Tech Stack
* **Frontend:** HTML, CSS, JS dashboard with live waveform, calmness trend chart, configurable API URL.
* **Backend:** Python Flask API — STT, librosa audio features, lexical + fusion logic, **Oumi SLM** via OpenAI-compatible HTTP.
* **The Brain:** Fine-tune with **[Oumi](https://github.com/oumi-ai/oumi)**.

![WhatsApp Image 2026-04-05 at 05 02 29](https://github.com/user-attachments/assets/cb4318b9-de9d-43d9-9e29-fd5825edfaa8)

---

## How to run (start here)

You run **two pieces**: the **Python backend** (API) and the **web frontend** (dashboard). The backend must be running before the dashboard can analyze anything.

### What you need

- **Python 3.10+** installed and available as `python` in the terminal.
- A normal **web browser** (Chrome, Edge, Firefox).

### Step 1 — Start the backend (API)

1. Open **PowerShell** (or Terminal).
2. Go to the **backend** folder (the folder that contains `app.py`):

   ```powershell
   cd path\to\MEDS\backend
   ```

   Example if the project is on your Desktop:

   ```powershell
   cd $HOME\Desktop\hack-with-oumi\MEDS\backend
   ```

3. **(First time only)** Create a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   copy .env.example .env
   ```

4. **(Every time)** Activate the venv and start the server:

   ```powershell
   cd path\to\MEDS\backend
   .\.venv\Scripts\Activate.ps1
   python app.py
   ```

5. You should see Flask start on **port 5000**. Leave this window **open**.

6. **Quick check:** in the browser, open:

   **http://127.0.0.1:5000/health**

   You want JSON with `"status": "ok"`. If that fails, the backend is not running or something else is using port 5000.

### Step 2 — Open the frontend (dashboard)

**Option A — Simple (double-click)**

1. In File Explorer, go to the **`MEDS\frontend`** folder.
2. Double-click **`index.html`** so it opens in your browser.

**Option B — Local web server (sometimes more reliable)**

From the **MEDS** folder (parent of `frontend`):

```powershell
cd path\to\MEDS\frontend
npx --yes serve .
```

Open the URL it prints (often **http://localhost:3000**).

### Step 3 — Point the UI at your API

At the top of the dashboard there is an **API base URL** field. Set it exactly to:

**http://127.0.0.1:5000**

Press Enter or change focus so it saves. The pill **Backend: online** should turn green after a successful `/health` check.

### Step 4 — Try it

- Type how you feel and click **Analyze text**, or  
- Click **Record**, speak, then **Stop** (the browser will ask for microphone permission).

### Optional — Oumi / local language model (for smarter replies)

This is **not required** to run the demo. Without a model server, the backend still returns **fallback** supportive text.

When you are ready:

1. Serve your fine-tuned model with any **OpenAI-compatible** API (e.g. vLLM, LM Studio, MLX stack — see [Oumi](https://github.com/oumi-ai/oumi) and [hack-with-oumi](https://github.com/oumi-ai/hack-with-oumi)).
2. Edit **`backend/.env`**:
   - `OUMI_BASE_URL` — e.g. `http://127.0.0.1:1234/v1`
   - `OUMI_MODEL` — the model name your server expects
   - `USE_LLM=true`
3. Restart `python app.py`.

Example training config in this repo: `oumi/configs/meds_empathy_sft.yaml` (adjust per [Oumi training docs](https://oumi.ai/docs/en/latest/user_guides/train/configuration.html)).

### If something goes wrong

| Symptom | What to do |
|--------|------------|
| `Backend: offline` in the UI | Confirm Step 1 is running and API base is `http://127.0.0.1:5000` (no trailing slash). |
| `/health` does not load | Backend not started, or port 5000 blocked / used by another app. |
| `Activate.ps1` cannot be loaded | Run PowerShell as Administrator once, or run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Mic does not work | Use **https** or **localhost** if the browser blocks mic on `file://`; use Option B (serve) for the frontend. |
| `pip install` errors on Windows | Close other Python/IDE processes using the same environment; retry `pip install -r requirements.txt`. |

---



## The Team: pENTEX
* **Mannat Sharma (Team Leader):** Project Architecture, Pitch & Documentation.
* **Chaitali Mahajan:** Frontend Lead.
* **Gurshant Singh Mohal (Backend A):** AI pipeline and integration.
* **Soham Sahu (Backend B):** Infrastructure and server routing.
* **Vrinda Kaushal:** DevOps and Git repository management.
