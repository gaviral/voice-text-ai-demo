from fastapi import FastAPI, File, UploadFile
import whisper
import os
from pyannote.audio import Pipeline

app = FastAPI()

whisper_model = whisper.load_model("tiny")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

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
