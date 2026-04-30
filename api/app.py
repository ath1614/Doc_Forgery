"""
KAVACH — Document Forgery Detection API v3
Port 8001
"""

import sys
sys.path.insert(0, '/home/tech/kavach_document')

import time, uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import structlog

from pipeline.pipeline import get_detector

logger = structlog.get_logger()

app = FastAPI(
    title='KAVACH — Document Forgery Detection v3',
    version='3.0.0',
    description="""
RBI-grade document fraud detection.

**Detects:**
- Pixel-level tampering (Multi-scale ELA + DCT)
- Copy-move forgery (ORB feature matching)
- Noise inconsistency (splicing detection)
- Photo substitution (Face region ELA)
- Aadhaar QR vs OCR mismatch
- PAN field validation + cross-check
- Bank statement math fraud
- PDF metadata editing fingerprint
- Document reuse / velocity attack

**Supports:** Aadhaar, PAN, Driving Licence, Voter ID, Ration Card, Bank Statement, Passport

**OCR:** PaddleOCR (primary) + EasyOCR (fallback) — Hindi + English
    """,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

MAX_MB = 20


class EvidenceItem(BaseModel):
    type:        str
    description: str
    confidence:  float
    class Config: extra = 'allow'

class DetectionResult(BaseModel):
    document_type:       str
    is_forged:           bool
    forgery_probability: float
    risk_level:          str
    decision:            str
    forgery_type:        str
    evidence:            List[EvidenceItem]
    extracted_fields:    Dict[str, Any]
    scores:              Dict[str, float]
    signal_breakdown:    Optional[Dict[str, float]] = None
    top_signals:         Optional[List]             = None
    velocity_info:       Optional[Dict[str, Any]]   = None
    ocr_engine:          Optional[str]              = None
    ocr_confidence:      Optional[float]            = None
    face_match:          Optional[Dict[str, Any]]   = None
    qr_verification:     Optional[Dict[str, Any]]   = None
    pdf_metadata:        Optional[Dict[str, Any]]   = None
    class Config: extra = 'allow'

class DetectionResponse(BaseModel):
    success:            bool
    request_id:         str
    timestamp:          datetime
    processing_time_ms: float
    data_region:        str = 'india-asia-south1'
    result:             Optional[DetectionResult] = None
    error:              Optional[str]             = None


@app.on_event('startup')
async def startup():
    get_detector()
    logger.info('kavach_document_v3_ready', port=8001)


@app.post('/api/v1/detect/document', response_model=DetectionResponse)
async def detect_document(
    file:   UploadFile = File(..., description='Document image (JPG/PNG) or PDF'),
    selfie: Optional[UploadFile] = File(None, description='Selfie for face matching (optional)'),
):
    start      = time.perf_counter()
    request_id = str(uuid.uuid4())

    ct       = (file.content_type or '').lower()
    filename = file.filename or ''

    if not any(t in ct for t in ['image', 'pdf']):
        raise HTTPException(415, f'Unsupported type: {ct}. Use JPG/PNG/PDF')

    file_bytes = await file.read()
    if len(file_bytes) / 1024 / 1024 > MAX_MB:
        raise HTTPException(413, f'File too large. Max {MAX_MB}MB')
    if not file_bytes:
        raise HTTPException(400, 'Empty file')

    selfie_bytes = await selfie.read() if selfie else None

    logger.info('document_request', request_id=request_id,
                filename=filename, size_kb=round(len(file_bytes)/1024, 1))

    try:
        raw = await get_detector().analyze_async(file_bytes, filename, selfie_bytes)
    except Exception as e:
        logger.error('detection_error', error=str(e))
        raise HTTPException(500, f'Detection failed: {e}')

    elapsed = round((time.perf_counter() - start) * 1000, 2)

    if not raw.get('success'):
        return DetectionResponse(
            success=False, request_id=request_id,
            timestamp=datetime.utcnow(), processing_time_ms=elapsed,
            error=raw.get('error'),
        )

    result = DetectionResult(
        document_type       = raw['document_type'],
        is_forged           = raw['is_forged'],
        forgery_probability = raw['forgery_probability'],
        risk_level          = raw['risk_level'],
        decision            = raw['decision'],
        forgery_type        = raw['forgery_type'],
        evidence            = [EvidenceItem(**e) for e in raw.get('evidence', [])],
        extracted_fields    = raw.get('extracted_fields', {}),
        scores              = {k: float(v) for k, v in raw.get('scores', {}).items()},
        signal_breakdown    = raw.get('signal_breakdown'),
        top_signals         = raw.get('top_signals'),
        velocity_info       = raw.get('velocity_info'),
        ocr_engine          = raw.get('ocr_engine'),
        ocr_confidence      = raw.get('ocr_confidence'),
        face_match          = raw.get('face_match'),
        qr_verification     = raw.get('qr_verification'),
        pdf_metadata        = raw.get('pdf_metadata'),
    )

    logger.info('document_result', request_id=request_id,
                doc_type=result.document_type, decision=result.decision,
                prob=result.forgery_probability, ms=elapsed)

    return DetectionResponse(
        success=True, request_id=request_id,
        timestamp=datetime.utcnow(), processing_time_ms=elapsed,
        result=result,
    )


@app.get('/api/v1/health')
async def health():
    return {
        'status':     'operational',
        'module':     'document_forgery_detection',
        'version':    '3.0.0',
        'detectors':  [
            'Multi-scale ELA + DCT',
            'Copy-Move (ORB)',
            'Noise Inconsistency',
            'Face Region ELA',
            'InsightFace Matching',
            'Aadhaar QR Verification',
            'PAN Validation',
            'Bank Statement Math',
            'PDF Metadata',
            'Velocity Check',
        ],
        'ocr':        'PaddleOCR (primary) + EasyOCR (fallback)',
        'supports':   ['AADHAAR', 'PAN', 'DRIVING_LICENCE', 'VOTER_ID',
                       'RATION_CARD', 'BANK_STATEMENT', 'PASSPORT',
                       'CASTE_CERTIFICATE', 'INCOME_CERTIFICATE'],
        'data_region': 'india-asia-south1',
        'timestamp':   datetime.utcnow().isoformat(),
    }

@app.get('/')
async def root():
    return {
        'service':  'KAVACH Document Forgery Detection v3',
        'endpoint': 'POST /api/v1/detect/document',
        'docs':     '/docs',
        'health':   '/api/v1/health',
    }
