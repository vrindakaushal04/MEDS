from flask import Flask, request, jsonify
from flask_cors import CORS

from config import OUMI_BASE_URL, USE_LLM

from services import stt, audio_emotion, text_emotion, fusion, response

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "message": "Emotion AI backend running",
        "routes": {
            "analyze": "POST /analyze",
            "analyze_text": "POST /analyze-text",
            "health": "GET /health",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "use_llm": USE_LLM,
        "oumi_base_url": OUMI_BASE_URL,
        "brain": "Oumi SLM (OpenAI-compatible); set OUMI_BASE_URL / OUMI_MODEL",
    })



def safe_pipeline(text, audio_emotion_input=None):
    try:
        audio_em = audio_emotion_input or {
            "label": "neutral",
            "intensity": "low",
            "energy": 0.0,
            "pitch": 0.0,
            "stress_score": 0.0,
        }

        try:
            text_em = text_emotion.detect(text)
        except Exception as e:
            print(f"[text_emotion ERROR] {e}")
            import traceback; traceback.print_exc()
            text_em = {"label": "neutral"}

        try:
            fused = fusion.combine(audio_em, text_em)
        except Exception as e:
            print(f"[fusion ERROR] {e}")
            import traceback; traceback.print_exc()
            fused = {"final_emotion": "neutral"}

        try:
            reply = response.generate(text, fused)
        except Exception as e:
            print(f"[response ERROR] {e}")
            reply = "I'm here to help."

        return {
            "success": True,
            "text": text,
            "audio_emotion": audio_em,
            "text_emotion": text_em,
            "analysis": fused,
            "final_emotion": fused.get("final_emotion", "neutral"),
            "response": reply,
        }

    except Exception as e:
        print(f"[pipeline ERROR] {e}")
        import traceback; traceback.print_exc()
        return {
            "success": False,
            "error": "Pipeline failed",
            "final_emotion": "neutral"
        }



@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "Missing audio file"}), 400

        file = request.files["audio"]

        if file.filename == "":
            return jsonify({"success": False, "error": "Invalid file name"}), 400

        audio_bytes = file.read()

        if not audio_bytes:
            return jsonify({"success": False, "error": "Empty audio file"}), 400

        try:
            text = stt.convert(audio_bytes, file.filename)
        except Exception:
            text = "Unable to transcribe"

        try:
            audio_em = audio_emotion.detect(audio_bytes, file.filename)
        except Exception:
            audio_em = None

        result = safe_pipeline(text, audio_em)

        return jsonify(result), 200

    except Exception:
        return jsonify({"success": False, "error": "Server error"}), 500


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    try:
        body = request.get_json(silent=True) or {}

        # ✅ FIXED (no bug)
        text = (body.get("text") or "").strip()

        if not text:
            return jsonify({"success": False, "error": "Missing text"}), 400

        result = safe_pipeline(text)

        return jsonify(result), 200

    except Exception:
        return jsonify({"success": False, "error": "Server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
