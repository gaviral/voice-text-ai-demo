from fastapi import FastAPI, File, UploadFile, Body
import whisper
import os
from transformers import pipeline
from pydantic import BaseModel

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

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Define request models
class TextRequest(BaseModel):
    text: str

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
async def translate_text(request: TextRequest):
    inputs = tokenizer([request.text], return_tensors="pt")
    translated_tokens = translation_model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return {
        "original_text": request.text,
        "translated_text": translated_text
    }

@app.post("/analyze-sentiment")
async def analyze_sentiment(request: TextRequest):
    """
    Analyze the sentiment of the provided text.
    Returns sentiment classification (POSITIVE/NEGATIVE/NEUTRAL) and confidence score.
    """
    sentiment_result = sentiment_analyzer(request.text)
    
    return {
        "text": request.text,
        "sentiment": sentiment_result[0]["label"],
        "score": sentiment_result[0]["score"],
        "explanation": f"The text has been classified as {sentiment_result[0]['label']} with a confidence of {sentiment_result[0]['score']:.2f}"
    }

@app.post("/transcribe-with-sentiment")
async def transcribe_with_sentiment(file: UploadFile = File(...)):
    """
    Transcribe audio and analyze the sentiment of the transcribed text in one step.
    """
    temp_filename = "temp_audio_file"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    result = whisper_model.transcribe(temp_filename, fp16=False)
    transcript = result["text"]
    os.remove(temp_filename)
    
    # Analyze sentiment
    sentiment_result = sentiment_analyzer(transcript)
    
    return {
        "transcript": transcript,
        "sentiment": sentiment_result[0]["label"],
        "score": sentiment_result[0]["score"],
        "explanation": f"The transcript has been classified as {sentiment_result[0]['label']} with a confidence of {sentiment_result[0]['score']:.2f}"
    }
