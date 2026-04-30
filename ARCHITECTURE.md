# Technical Architecture

Deep dive into the document forgery detection system architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI REST API                        │
│                    (api/app.py:8001)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Document Detector                           │
│                  (core/detector.py)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Preprocessor → Resize, normalize, denoise        │  │
│  │  2. Forensics → ELA, copy-move, noise analysis       │  │
│  │  3. OCR → Extract text fields                        │  │
│  │  4. Validators → Document-specific checks            │  │
│  │  5. Scoring → Weighted ensemble                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. API Layer (`api/app.py`)

**Responsibilities:**
- HTTP request handling
- File upload validation
- Response formatting
- Error handling

**Key Endpoints:**
```python
@app.post("/api/v1/detect/document")
async def detect_document(file: UploadFile):
    # 1. Validate file type (JPG, PNG, PDF)
    # 2. Convert to PIL Image
    # 3. Call detector
    # 4. Format response
    return {"status": "success", "result": {...}}
```

**Technologies:**
- FastAPI (async web framework)
- Pydantic (request/response validation)
- Uvicorn (ASGI server)

---

### 2. Detector (`core/detector.py`)

**Main orchestrator** that coordinates all detection layers.

**Flow:**
```python
def detect(image_path: str) -> dict:
    # 1. Preprocess
    image = preprocessor.load_and_normalize(image_path)
    
    # 2. Run forensics
    ela_score = forensics.compute_ela(image)
    copy_move_score = forensics.detect_copy_move(image)
    noise_score = forensics.analyze_noise(image)
    
    # 3. OCR
    text_fields = ocr.extract_fields(image)
    
    # 4. Validate
    doc_type = identify_document_type(text_fields)
    validator_score = validators.validate(doc_type, text_fields, image)
    
    # 5. Score
    final_score = scoring.compute_weighted_score({
        'ela': ela_score,
        'copy_move': copy_move_score,
        'noise': noise_score,
        'validator': validator_score,
        ...
    }, doc_type)
    
    # 6. Verdict
    verdict = 'REJECTED' if final_score > 0.42 else 'APPROVED'
    
    return {
        'is_forged': final_score > 0.42,
        'forgery_probability': final_score,
        'verdict': verdict,
        'scores': {...}
    }
```

---

### 3. Preprocessor (`core/preprocessor.py`)

**Responsibilities:**
- Load images (JPG, PNG, PDF)
- Resize to standard dimensions (1024x1024 max)
- Normalize brightness/contrast
- Denoise (optional)

**Key Functions:**
```python
def load_and_normalize(path: str) -> np.ndarray:
    # Load
    if path.endswith('.pdf'):
        image = pdf_to_image(path)
    else:
        image = cv2.imread(path)
    
    # Resize if too large
    if max(image.shape) > 1024:
        image = resize_keep_aspect(image, 1024)
    
    # Normalize
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    return image
```

---

### 4. Forensics Engine (`core/forensics.py`)

**Implements 6 forensic techniques:**

#### 4.1 Error Level Analysis (ELA)

Detects edited regions by analyzing JPEG compression artifacts.

```python
def compute_ela(image: np.ndarray) -> float:
    # Save at quality 90
    cv2.imwrite('/tmp/temp.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Reload
    recompressed = cv2.imread('/tmp/temp.jpg')
    
    # Compute difference
    ela = cv2.absdiff(image, recompressed)
    
    # Convert to grayscale
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find high-error regions
    _, thresh = cv2.threshold(ela_gray, 30, 255, cv2.THRESH_BINARY)
    
    # Score = percentage of high-error pixels
    score = np.sum(thresh > 0) / thresh.size
    
    return score
```

**Why it works:** Edited regions compress differently than original regions.

#### 4.2 Copy-Move Detection

Finds duplicated regions using ORB feature matching.

```python
def detect_copy_move(image: np.ndarray) -> float:
    # Extract ORB features
    orb = cv2.ORB_create(nfeatures=1000)
    kp, desc = orb.detectAndCompute(image, None)
    
    # Match features to themselves
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc, desc, k=2)
    
    # Filter for good matches (same feature, different location)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            # Check if keypoints are far apart (not the same point)
            pt1 = kp[m.queryIdx].pt
            pt2 = kp[m.trainIdx].pt
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist > 50:  # At least 50 pixels apart
                good_matches.append(m)
    
    # Score = number of suspicious matches
    score = min(len(good_matches) / 100, 1.0)
    
    return score
```

**Why it works:** Copy-pasted regions have identical feature descriptors.

#### 4.3 Noise Pattern Analysis

Genuine documents have uniform noise; edited regions have different noise.

```python
def analyze_noise(image: np.ndarray) -> float:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute noise using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise = laplacian.var()
    
    # Split into 8x8 blocks
    h, w = gray.shape
    block_size = 64
    blocks = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            block_noise = cv2.Laplacian(block, cv2.CV_64F).var()
            blocks.append(block_noise)
    
    # Compute variance of block noises
    noise_variance = np.var(blocks)
    
    # High variance → inconsistent noise → likely edited
    score = min(noise_variance / 1000, 1.0)
    
    return score
```

#### 4.4 Moire Pattern Detection

Detects if document was photographed from a screen.

```python
def detect_moire(image: np.ndarray) -> float:
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Moire patterns show up as rainbow-like color variations
    saturation = hsv[:, :, 1]
    
    # Compute FFT to find periodic patterns
    f = np.fft.fft2(saturation)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Look for high-frequency peaks (moire signature)
    h, w = magnitude.shape
    center_region = magnitude[h//4:3*h//4, w//4:3*w//4]
    outer_region = magnitude - center_region
    
    # Score = ratio of outer to center energy
    score = np.sum(outer_region) / np.sum(center_region)
    
    return min(score / 10, 1.0)
```

#### 4.5 Liveness Check

Detects photo-of-photo (document photographed from another photo).

```python
def check_liveness(image: np.ndarray) -> float:
    # Look for reflections, screen edges, shadows
    
    # 1. Edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # 2. Find straight lines (screen edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # 3. Check for rectangular boundary (photo frame)
    if lines is not None and len(lines) > 4:
        # Likely a photo of a photo
        return 0.8
    
    # 4. Check for specular highlights (flash reflection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    bright_ratio = np.sum(bright > 0) / bright.size
    
    if bright_ratio > 0.05:  # More than 5% very bright pixels
        return 0.6
    
    return 0.1
```

#### 4.6 Metadata Analysis

Checks EXIF data for editing software signatures.

```python
from PIL import Image
from PIL.ExifTags import TAGS

def analyze_metadata(image_path: str) -> float:
    image = Image.open(image_path)
    exif = image._getexif()
    
    if not exif:
        return 0.3  # No metadata (suspicious)
    
    # Check for editing software
    editing_software = ['Photoshop', 'GIMP', 'Picsart', 'Snapseed']
    
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == 'Software':
            for software in editing_software:
                if software.lower() in str(value).lower():
                    return 0.9  # Edited in photo editor
    
    return 0.1  # Clean metadata
```

---

### 5. OCR Engine (`core/ocr.py`)

**Extracts text fields** using Tesseract OCR.

```python
import pytesseract

def extract_fields(image: np.ndarray) -> dict:
    # Preprocess for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Run OCR
    text = pytesseract.image_to_string(thresh, lang='eng+hin')
    
    # Parse fields using regex
    fields = {
        'uid': extract_uid(text),
        'name': extract_name(text),
        'dob': extract_dob(text),
        'address': extract_address(text),
        'epic': extract_epic(text),
        'pan': extract_pan(text)
    }
    
    return fields
```

---

### 6. Validators (`core/validators.py`)

**Document-specific validation rules.**

#### 6.1 Aadhaar Validator

```python
class AadhaarValidator:
    def validate(self, fields: dict, image: np.ndarray) -> float:
        score = 0.0
        
        # 1. UID format check (12 digits)
        uid = fields.get('uid', '')
        if not re.match(r'^\d{12}$', uid):
            score += 0.3
        
        # 2. QR code verification
        qr_data = decode_qr(image)
        if qr_data:
            qr_uid = parse_aadhaar_qr(qr_data)
            if qr_uid != uid:
                score += 0.5  # QR mismatch = high suspicion
        
        # 3. Checksum validation (Verhoeff algorithm)
        if not verify_aadhaar_checksum(uid):
            score += 0.4
        
        return min(score, 1.0)
```

#### 6.2 PAN Validator

```python
class PANValidator:
    def validate(self, fields: dict, image: np.ndarray) -> float:
        score = 0.0
        
        # PAN format: ABCDE1234F
        pan = fields.get('pan', '')
        
        # 1. Format check
        if not re.match(r'^[A-Z]{5}\d{4}[A-Z]$', pan):
            score += 0.4
        
        # 2. Fourth character should be 'P' for individual
        if len(pan) >= 4 and pan[3] not in ['P', 'C', 'H', 'F', 'A', 'T', 'B', 'L', 'J', 'G']:
            score += 0.3
        
        return min(score, 1.0)
```

#### 6.3 Voter ID Validator

```python
class VoterIDValidator:
    _VALID_STATE_CODES = ['BLA', 'CHH', 'DL', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 'RJ', 'SK', 'TN', 'TR', 'UP', 'UK', 'WB']
    
    def validate(self, fields: dict, image: np.ndarray) -> float:
        score = 0.0
        
        # EPIC format: ABC1234567
        epic = fields.get('epic', '')
        
        # 1. Format check (3 letters + 7 digits)
        if not re.match(r'^[A-Z]{3}\d{7}$', epic):
            score += 0.3
        
        # 2. State code check
        if len(epic) >= 3:
            state_code = epic[:3]
            if state_code not in self._VALID_STATE_CODES:
                score += 0.4
        
        return min(score, 1.0)
```

---

### 7. Scoring Engine (`core/scoring.py`)

**Combines all scores** using weighted ensemble.

```python
# Scoring profiles per document type
SCORING_PROFILES = {
    'AADHAAR': {
        'ela': 0.25,
        'copy_move': 0.20,
        'noise': 0.15,
        'validator': 0.25,
        'moire': 0.05,
        'liveness': 0.05,
        'velocity': 0.05
    },
    'PAN': {
        'ela': 0.20,
        'copy_move': 0.25,
        'noise': 0.10,
        'validator': 0.20,
        'moire': 0.15,
        'liveness': 0.10,
        'velocity': 0.01
    },
    'VOTER_ID': {
        'ela': 0.25,
        'copy_move': 0.20,
        'noise': 0.15,
        'validator': 0.25,
        'moire': 0.05,
        'liveness': 0.05,
        'velocity': 0.05
    }
}

def compute_weighted_score(scores: dict, doc_type: str) -> float:
    profile = SCORING_PROFILES.get(doc_type, SCORING_PROFILES['AADHAAR'])
    
    final_score = sum(
        scores.get(key, 0) * weight
        for key, weight in profile.items()
    )
    
    return final_score
```

**Thresholds:**
```python
REVIEW_THRESHOLD = 0.35
REJECT_THRESHOLD = 0.42

def get_verdict(score: float) -> str:
    if score < REVIEW_THRESHOLD:
        return 'APPROVED'
    elif score < REJECT_THRESHOLD:
        return 'REVIEW'
    else:
        return 'REJECTED'
```

---

## Data Flow Example

**Input:** Aadhaar card image (potentially forged)

**Step 1: Preprocessing**
```
Original: 2048x1536 JPG
↓
Resized: 1024x768
↓
Normalized: brightness/contrast adjusted
```

**Step 2: Forensics**
```
ELA: 0.68 (high error in UID region)
Copy-Move: 0.12 (no duplicates found)
Noise: 0.45 (inconsistent noise patterns)
Moire: 0.08 (no screen artifacts)
Liveness: 0.15 (no photo-of-photo signs)
```

**Step 3: OCR**
```
Extracted:
  UID: 1234 5678 9012
  Name: John Doe
  DOB: 01/01/1990
  Address: 123 Main St
```

**Step 4: Validation**
```
UID format: ✓ (12 digits)
QR decode: ✓
QR UID: 1234 5678 9013 ← MISMATCH!
Validator score: 0.85
```

**Step 5: Scoring**
```
Weighted sum:
  0.68 × 0.25 (ELA)
+ 0.12 × 0.20 (copy-move)
+ 0.45 × 0.15 (noise)
+ 0.85 × 0.25 (validator)
+ 0.08 × 0.05 (moire)
+ 0.15 × 0.05 (liveness)
+ 0.00 × 0.05 (velocity)
= 0.52
```

**Step 6: Verdict**
```
0.52 > 0.42 → REJECTED
```

**Output:**
```json
{
  "is_forged": true,
  "forgery_probability": 0.52,
  "verdict": "REJECTED",
  "document_type": "AADHAAR",
  "scores": {
    "ela": 0.68,
    "copy_move": 0.12,
    "noise": 0.45,
    "validator": 0.85,
    "moire": 0.08,
    "liveness": 0.15,
    "velocity": 0.0
  },
  "flags": [
    "QR code UID mismatch",
    "High ELA in UID region",
    "Inconsistent noise patterns"
  ]
}
```

---

## Performance Characteristics

**Processing Time:**
- Preprocessing: ~50ms
- ELA: ~200ms
- Copy-move: ~300ms
- Noise analysis: ~100ms
- OCR: ~500ms
- Validation: ~50ms
- **Total: ~1.2 seconds**

**Memory Usage:**
- Image buffer: ~10MB (1024x768 RGB)
- ORB features: ~5MB
- OCR: ~50MB (Tesseract models)
- **Total: ~65MB per request**

**Scalability:**
- Single process: ~50 requests/minute
- With 4 workers: ~200 requests/minute
- GPU acceleration (face detection): 2x speedup

---

## Future Improvements

1. **GPU acceleration** for ELA and copy-move
2. **Parallel processing** of forensic layers
3. **Model-based detection** (CNN for forgery classification)
4. **Hologram detection** (UV light simulation)
5. **Multi-language OCR** (Hindi, Tamil, Telugu)
6. **Real-time streaming** (video KYC)

---

## References

- **ELA**: Krawetz, N. (2007). "A Picture's Worth"
- **Copy-Move**: Fridrich, J. (2003). "Detection of Copy-Move Forgery"
- **ORB**: Rublee, E. (2011). "ORB: An efficient alternative to SIFT or SURF"
- **Aadhaar QR**: UIDAI Technical Specifications
