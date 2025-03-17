from fastapi import FastAPI, File, UploadFile
import whisper
import os

LOW_MEMORY = os.environ.get("LOW_MEMORY", "false").lower() == "true"
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)

app = FastAPI()

whisper_model = whisper.load_model("tiny")

if not LOW_MEMORY and HF_AUTH_TOKEN:
    from pyannote.audio import Pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_AUTH_TOKEN
    )
else:
    diarization_pipeline = None

from transformers import MarianTokenizer, MarianMTModel
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_filename = "temp_audio_file"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    result = whisper_model.transcribe(temp_filename, fp16=False)
    os.remove(temp_filename)
    return {"transcript": result["text"]}

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    if diarization_pipeline is None:
        return {"error": "Diarization disabled or missing HF_AUTH_TOKEN."}

    temp_filename = "temp_audio_file"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    diarization_result = diarization_pipeline(temp_filename)
    os.remove(temp_filename)

    speaker_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return {"segments": speaker_segments}

@app.post("/translate")
async def translate_text(text: str):
    inputs = tokenizer([text], return_tensors="pt")
    translated_tokens = translation_model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return {
        "original_text": text,
        "translated_text": translated_text
    }
