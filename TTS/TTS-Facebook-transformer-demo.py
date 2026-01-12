import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from bottle import Bottle, request, response, run
import io
import numpy as np

model_name = "facebook/mms-tts-tha"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model and tokenizer...")
model = VitsModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded successfully!")

samplerate = model.config.sampling_rate

app = Bottle()

# Enable CORS globally
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'

app.add_hook('after_request', enable_cors)

# Handle preflight OPTIONS requests
@app.route('/', method='OPTIONS')
def root_options():
    response.status = 204
    return ''

@app.route('/tts', method='OPTIONS')
def tts_options():
    response.status = 204
    return ''

@app.post('/tts')
@app.post('/tts')
def synthesize():
    try:
        data = request.json
        if not data or 'text' not in data:
            response.status = 400
            return {"error": "JSON body with 'text' field is required"}

        text = data['text'].strip()
        if not text:
            response.status = 400
            return {"error": "Text cannot be empty"}
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        print(f"Generating speech for: {text}")
        with torch.no_grad():
            waveform = model(input_ids).waveform
        audio_data = waveform.squeeze(0).cpu().numpy()
        volume = float(data.get('volume', 1.0))
        audio_data = audio_data * volume
        audio_data = np.clip(audio_data, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, audio_data, samplerate, format='WAV')
        buf.seek(0)
        response.content_type = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename="tts_output.wav"'
        return buf.read()

    except Exception as e:
        response.status = 500
        return {"error": str(e)}

@app.get('/')
def index():
    return {
        "message": "Thai TTS API (MMS-TTS) is running",
        "usage": "POST /tts with JSON body {'text': 'ข้อความภาษาไทยที่นี่'}",
        "example_curl": 'curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d \'{"text": "สวัสดีครับ"}\' --output output.wav'
    }

if __name__ == '__main__':
    print("\nThai TTS API started on http://localhost:8080")
    print("POST /tts with JSON {'text': 'your Thai text'} to get WAV audio")
    run(app, host='0.0.0.0', port=8080, debug=True)
