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
  curl -X POST "http://localhost:8000/translate" -H "Content-Type: application/json" -d '{"text": "Hello, how are you?"}'
  ```

### 4. Sentiment Analysis
Classify text as **POSITIVE/NEGATIVE/NEUTRAL**.
- **Financial Use Cases**: Risk signaling, emotional pattern detection, satisfaction monitoring
- **Endpoint**: `POST /analyze-sentiment`
  ```bash
  curl -X POST "http://localhost:8000/analyze-sentiment" -H "Content-Type: application/json" -d '{"text": "I am extremely happy with the service provided."}'
  ```

### 5. Compliance Keyword Detection
Flag compliance issues through **semantic similarity**.
- **Financial Use Cases**: Insider trading prevention, confidentiality monitoring, regulatory violation detection
- **Endpoint**: `POST /detect-compliance-keywords`
  ```bash
  curl -X POST "http://localhost:8000/detect-compliance-keywords" -H "Content-Type: application/json" -d '{"text": "This is strictly confidential information that is not for distribution.", "threshold": 0.5}'
  ```
  
  Optional parameters:
  - `threshold`: Similarity threshold (default: 0.5). Lower values increase detection sensitivity.
  - `max_results`: Maximum number of matches to return (default: 5).
  
  Example with parameters:
  ```bash
  curl -X POST "http://localhost:8000/detect-compliance-keywords" -H "Content-Type: application/json" -d '{"text": "This is strictly confidential information.", "threshold": 0.4}'
  ```

  **Note**: The compliance detection uses semantic similarity rather than exact matching, so adjust the threshold based on your sensitivity requirements. Values between 0.4-0.6 provide good detection with minimal false positives.

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
  
  Optional parameters:
  - `threshold`: Similarity threshold (default: 0.5). Lower values increase detection sensitivity.
  - `max_results`: Maximum number of matches to return (default: 5).
  
  Example with parameters:
  ```bash
  curl -X POST "http://localhost:8000/transcribe-with-compliance-check" -F "file=@sample.wav" -F "threshold=0.4"
  ```
  
  Example response (assuming the audio contains "This is strictly confidential information"):
  ```json
  {
    "transcript": "This is strictly confidential information.",
    "compliance_check": {
      "matches": [
        {
          "keyword": "strictly confidential",
          "category": "confidentiality",
          "similarity_score": 0.6313067674636841
        },
        {
          "keyword": "confidential data",
          "category": "insider_trading",
          "similarity_score": 0.6141104698181152
        }
      ],
      "risk_level": "MEDIUM",
      "explanation": "Detected 2 potential compliance issues in categories: confidentiality, insider_trading. This communication should be reviewed."
    }
  }
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

#### Lightweight Mode vs. Full-Featured Mode
By default, the Docker deployment runs in lightweight mode (`LOW_MEMORY=true`), which disables memory-intensive features to ensure stability in resource-constrained environments. 

To run the Docker container in full-featured mode:
```bash
docker run -p 8000:8000 -e LOW_MEMORY=false -e HF_AUTH_TOKEN=your_token voice-text-ai-demo
```

**Warning**: Full-featured mode requires significantly more memory (at least 4GB RAM recommended) and may cause container crashes in resource-constrained environments.

### Environment Variables
- `LOW_MEMORY`: Set "true" to disable memory-intensive features
  - When set to "true", the following features are disabled:
    - Speech transcription
    - Speaker diarization
    - Text translation
    - Sentiment analysis
    - Compliance keyword detection
    - Combined pipelines
  - Default: "false" for local development, "true" for Docker deployment
- `HF_AUTH_TOKEN`: HuggingFace token for diarization

## Testing the Endpoints

### Sentiment Analysis Example
```bash
curl -X POST "http://localhost:8000/analyze-sentiment" -H "Content-Type: application/json" -d '{"text": "I am extremely happy with the service provided."}'
```

Example response:
```json
{
  "text": "I am extremely happy with the service provided.",
  "sentiment": "POSITIVE",
  "score": 0.9998642206192017,
  "explanation": "The text has been classified as POSITIVE with a confidence of 1.00"
}
```

### Compliance Keyword Detection Example
```bash
curl -X POST "http://localhost:8000/detect-compliance-keywords" -H "Content-Type: application/json" -d '{"text": "This is strictly confidential information that is not for distribution.", "threshold": 0.5}'
```

Example response:
```json
{
  "text": "This is strictly confidential information that is not for distribution.",
  "matches": [
    {
      "keyword": "strictly confidential",
      "category": "confidentiality",
      "similarity_score": 0.6313067674636841
    },
    {
      "keyword": "confidential data",
      "category": "insider_trading",
      "similarity_score": 0.6141104698181152
    },
    {
      "keyword": "not for distribution",
      "category": "confidentiality",
      "similarity_score": 0.5479345917701721
    },
    {
      "keyword": "non-public information",
      "category": "insider_trading",
      "similarity_score": 0.5427697896957397
    }
  ],
  "risk_level": "HIGH",
  "explanation": "Detected 4 potential compliance issues in categories: confidentiality, insider_trading. This communication requires immediate review."
}
```

**Note**: The compliance detection sensitivity can be adjusted by modifying the `threshold` parameter. Lower values (e.g., 0.4-0.5) will catch more potential issues but may include false positives, while higher values (e.g., 0.7-0.8) will be more precise but might miss some relevant matches.