from fastapi import FastAPI, File, UploadFile, Body
import whisper
import os
from transformers import pipeline
from pydantic import BaseModel
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util
import torch

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

# Initialize sentence transformer model for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define compliance keywords with categories
COMPLIANCE_KEYWORDS = {
    "insider_trading": [
        "insider information", 
        "non-public information", 
        "before announcement", 
        "confidential data",
        "trading window",
        "blackout period"
    ],
    "market_manipulation": [
        "pump and dump", 
        "artificially inflate", 
        "manipulate price", 
        "spread rumors",
        "false information"
    ],
    "confidentiality": [
        "strictly confidential", 
        "not for distribution", 
        "internal only", 
        "do not share",
        "between us only"
    ],
    "suspicious_activity": [
        "off the record", 
        "don't tell anyone", 
        "delete this message", 
        "not through email",
        "call me instead",
        "avoid documentation"
    ]
}

# Precompute embeddings for all keywords
KEYWORD_EMBEDDINGS = {}
for category, keywords in COMPLIANCE_KEYWORDS.items():
    KEYWORD_EMBEDDINGS[category] = sentence_model.encode(keywords)

# Define request models
class TextRequest(BaseModel):
    text: str
    
class KeywordDetectionRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.75
    max_results: Optional[int] = 5

class KeywordMatch(BaseModel):
    keyword: str
    category: str
    similarity_score: float
    
class KeywordDetectionResponse(BaseModel):
    text: str
    matches: List[KeywordMatch]
    risk_level: str
    explanation: str

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

@app.post("/detect-compliance-keywords")
async def detect_compliance_keywords(request: KeywordDetectionRequest):
    """
    Detect potentially problematic phrases in financial communications.
    Uses semantic similarity to identify phrases similar to known compliance risk keywords.
    """
    # Encode the input text
    text_embedding = sentence_model.encode(request.text)
    
    # Find matches across all categories
    matches = []
    
    for category, keyword_embeddings in KEYWORD_EMBEDDINGS.items():
        # Calculate cosine similarity between text and all keywords in this category
        similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]
        
        # Get the keywords for this category
        keywords = COMPLIANCE_KEYWORDS[category]
        
        # Find matches above threshold
        for i, similarity in enumerate(similarities):
            if similarity >= request.threshold:
                matches.append(KeywordMatch(
                    keyword=keywords[i],
                    category=category,
                    similarity_score=float(similarity)
                ))
    
    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    
    # Limit to max_results
    matches = matches[:request.max_results]
    
    # Determine risk level based on number and strength of matches
    risk_level = "LOW"
    if matches:
        avg_score = sum(match.similarity_score for match in matches) / len(matches)
        if len(matches) >= 3 or avg_score > 0.9:
            risk_level = "HIGH"
        elif len(matches) >= 1 or avg_score > 0.8:
            risk_level = "MEDIUM"
    
    # Create explanation
    if not matches:
        explanation = "No compliance risk keywords detected in the text."
    else:
        categories = set(match.category for match in matches)
        explanation = f"Detected {len(matches)} potential compliance issues in categories: {', '.join(categories)}."
        if risk_level == "HIGH":
            explanation += " This communication requires immediate review."
        elif risk_level == "MEDIUM":
            explanation += " This communication should be reviewed."
    
    return KeywordDetectionResponse(
        text=request.text,
        matches=matches,
        risk_level=risk_level,
        explanation=explanation
    )

@app.post("/transcribe-with-compliance-check")
async def transcribe_with_compliance_check(file: UploadFile = File(...), threshold: float = 0.75, max_results: int = 5):
    """
    Transcribe audio and check for compliance keywords in one step.
    """
    temp_filename = "temp_audio_file"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    result = whisper_model.transcribe(temp_filename, fp16=False)
    transcript = result["text"]
    os.remove(temp_filename)
    
    # Check for compliance keywords
    text_embedding = sentence_model.encode(transcript)
    
    # Find matches across all categories
    matches = []
    
    for category, keyword_embeddings in KEYWORD_EMBEDDINGS.items():
        # Calculate cosine similarity between text and all keywords in this category
        similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]
        
        # Get the keywords for this category
        keywords = COMPLIANCE_KEYWORDS[category]
        
        # Find matches above threshold
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                matches.append({
                    "keyword": keywords[i],
                    "category": category,
                    "similarity_score": float(similarity)
                })
    
    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Limit to max_results
    matches = matches[:max_results]
    
    # Determine risk level based on number and strength of matches
    risk_level = "LOW"
    if matches:
        avg_score = sum(match["similarity_score"] for match in matches) / len(matches)
        if len(matches) >= 3 or avg_score > 0.9:
            risk_level = "HIGH"
        elif len(matches) >= 1 or avg_score > 0.8:
            risk_level = "MEDIUM"
    
    # Create explanation
    if not matches:
        explanation = "No compliance risk keywords detected in the transcript."
    else:
        categories = set(match["category"] for match in matches)
        explanation = f"Detected {len(matches)} potential compliance issues in categories: {', '.join(categories)}."
        if risk_level == "HIGH":
            explanation += " This communication requires immediate review."
        elif risk_level == "MEDIUM":
            explanation += " This communication should be reviewed."
    
    return {
        "transcript": transcript,
        "compliance_check": {
            "matches": matches,
            "risk_level": risk_level,
            "explanation": explanation
        }
    }
