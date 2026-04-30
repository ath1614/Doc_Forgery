# Quick Start Guide

Get the document forgery detection system running in 5 minutes.

## 1. Install Dependencies (One-time)

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3.10-venv tesseract-ocr libzbar0

# macOS
brew install python@3.10 tesseract zbar
```

## 2. Setup Project

```bash
# Clone
git clone https://github.com/ath1614/Doc_Forgery.git
cd Doc_Forgery

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## 3. Start Server

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn api.app:app --host 0.0.0.0 --port 8001
```

Server starts at: http://localhost:8001

## 4. Test It

Open another terminal:

```bash
# Health check
curl http://localhost:8001/health

# Test with sample document
curl -X POST http://localhost:8001/api/v1/detect/document \
  -F "file=@data/samples/aadhaar_0001_genuine.jpg"
```

## 5. Try Your Own Documents

```bash
curl -X POST http://localhost:8001/api/v1/detect/document \
  -F "file=@/path/to/your/document.jpg"
```

## Common Issues

**"ModuleNotFoundError"**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**"TesseractNotFoundError"**
```bash
sudo apt install tesseract-ocr  # Ubuntu
brew install tesseract          # macOS
```

**"libzbar not found"**
```bash
sudo apt install libzbar0  # Ubuntu
brew install zbar          # macOS
```

## What's Next?

- Read full documentation: [README.md](README.md)
- Explore sample data: `data/samples/`
- Adjust thresholds: `core/scoring.py`
- Add new validators: `core/validators.py`

## API Response Format

```json
{
  "status": "success",
  "result": {
    "is_forged": false,
    "forgery_probability": 0.23,
    "verdict": "APPROVED",
    "document_type": "AADHAAR",
    "scores": {
      "ela": 0.18,
      "copy_move": 0.12,
      "noise": 0.25,
      "validator": 0.10,
      "moire": 0.05,
      "liveness": 0.08,
      "velocity": 0.0
    },
    "flags": []
  }
}
```

**Verdicts:**
- `APPROVED` — Genuine document (< 35% forgery probability)
- `REVIEW` — Borderline, needs manual check (35-42%)
- `REJECTED` — Forged document (> 42%)

## Need Help?

- Check [README.md](README.md) for detailed docs
- Open an issue on GitHub
- Contact: [your-email]
