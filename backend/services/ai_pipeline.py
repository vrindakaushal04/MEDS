import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class EmotionPipeline:
    def __init__(self):
        print("[System] Initializing Full AI Pipeline...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "gpt-3.5-turbo" 
        
        self.system_prompt = """
        You are 'EmotionMate', an empathetic voice assistant. 
        Analyze the user's text, identify their primary emotion, and generate a short, supportive reply (under 20 words).
        
        You MUST respond strictly in valid JSON format matching this schema:
        {
            "emotion": "[Insert primary emotion like Anxious, Happy, Angry, Sad]",
            "reply": "[Insert your short, comforting response]"
        }
        """

    def transcribe_audio(self, audio_file_path):
        
        print(f"[STT] Transcribing audio file: {audio_file_path}")
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            return transcript.text
        except Exception as e:
            print(f"[Error] Transcription failed: {e}")
            return "Error transcribing audio."

    def analyze_text(self, transcribed_text):
       
        print(f"[SLM] Analyzing text: '{transcribed_text}'")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": transcribed_text}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.6 
            )
            raw_output = response.choices[0].message.content
            parsed_data = json.loads(raw_output)
            
            return parsed_data.get("emotion", "Neutral"), parsed_data.get("reply", "I am here for you.")
        except Exception as e:
            print(f"[Error] SLM Connection failed: {e}")
            return "Error", "Connection failed."

    def generate_audio(self, reply_text):
       
        print(f"[TTS] Generating voice for: '{reply_text}'")
        output_file = "response_audio.mp3"
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="nova", 
                input=reply_text
            )
            response.stream_to_file(output_file)
            return output_file
        except Exception as e:
            print(f"[Error] Voice generation failed: {e}")
            return None

    def process_full_interaction(self, audio_file_path):
       
        user_text = self.transcribe_audio(audio_file_path)
        
       
        emotion, reply_text = self.analyze_text(user_text)
        
       
        audio_response_path = self.generate_audio(reply_text)
        
        return {
            "status": "success",
            "user_said": user_text,
            "detected_emotion": emotion,
            "ai_reply": reply_text,
            "audio_file": audio_response_path
        }
