from fastapi import FastAPI, File, UploadFile
import whisper
import os

app = FastAPI()

model = whisper.load_model("tiny")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_filename = "temp_audio_file"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    result = model.transcribe(temp_filename, fp16=False)

    os.remove(temp_filename)

    return {"transcript": result["text"]}
