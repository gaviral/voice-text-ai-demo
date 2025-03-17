# Voice Text AI Demo

A microservice for financial compliance audio processing.

## Core Technologies

- **FastAPI + Docker**: Containerized API service
- **Whisper (OpenAI)**: Speech recognition model
- **PyAnnote**: Speaker diarization framework
- **MarianMT (Helsinki-NLP)**: Translation model
- **Hugging Face Transformers**: NLP pipelines
- **Sentence Transformers**: Semantic similarity detection

## Key Features & API Endpoints

### 1. Speech Transcription
Convert audio to text using **Whisper model**.
- **Financial Use Cases**: Regulatory record-keeping, call monitoring, searchable voice archives
- **Endpoint**: `POST /transcribe`
  ```bash
  curl -X POST "http://localhost:8000/transcribe" -F "file=@sample.wav"
  ```

### 2. Speaker Diarization
Identify speakers in recordings using **PyAnnote**.
- **Financial Use Cases**: Meeting attribution, unauthorized speaker detection, conversation analysis
- **Endpoint**: `POST /diarize`
  ```bash
  curl -X POST "http://localhost:8000/diarize" -F "file=@sample.wav"
  ```

### 3. Text Translation
English-to-French translation with **MarianMT**.
- **Financial Use Cases**: Cross-border compliance, multi-jurisdiction regulation
- **Endpoint**: `POST /translate`
  ```bash
  curl -X POST "http://localhost:8000/translate" -d '{"text": "Hello, how are you?"}'
  ```

### 4. Sentiment Analysis
Classify text as **POSITIVE/NEGATIVE/NEUTRAL**.
- **Financial Use Cases**: Risk signaling, emotional pattern detection, satisfaction monitoring
- **Endpoint**: `POST /analyze-sentiment`
  ```bash
  curl -X POST "http://localhost:8000/analyze-sentiment" -d '{"text": "I am extremely happy with the service provided."}'
  ```

### 5. Compliance Keyword Detection
Flag compliance issues through **semantic similarity**.
- **Financial Use Cases**: Insider trading prevention, confidentiality monitoring, regulatory violation detection
- **Endpoint**: `POST /detect-compliance-keywords`
  ```bash
  curl -X POST "http://localhost:8000/detect-compliance-keywords" -d '{"text": "Let me share some insider information about the merger."}'
  ```

### 6. Combined Pipelines

#### Transcribe with Sentiment Analysis
One-step speech-to-sentiment analysis.
- **Endpoint**: `POST /transcribe-with-sentiment`
  ```bash
  curl -X POST "http://localhost:8000/transcribe-with-sentiment" -F "file=@sample.wav"
  ```

#### Transcribe with Compliance Check
Combined transcription and compliance analysis.
- **Endpoint**: `POST /transcribe-with-compliance-check`
  ```bash
  curl -X POST "http://localhost:8000/transcribe-with-compliance-check" -F "file=@sample.wav"
  ```

## Setup Instructions

### Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

### Docker Deployment
```bash
docker build -t voice-text-ai-demo .
docker run -p 8000:8000 -e HF_AUTH_TOKEN=your_token voice-text-ai-demo
```

### Environment Variables
- `LOW_MEMORY`: Set "true" to disable memory-intensive features
- `HF_AUTH_TOKEN`: HuggingFace token for diarization