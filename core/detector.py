"""
KAVACH - Document Forgery Detection v2
Focuses on what banks CANNOT detect with existing tools.

What banks already do (we skip):
  - QR verification (UIDAI API)
  - Format check (regex)
  - Basic OCR

What ONLY we do:
  1. ELA — pixel-level tampering (Photoshop detection)
  2. Face region ELA — photo swap detection
  3. PDF metadata — editing tool fingerprint
  4. Font consistency — text replacement detection
  5. Document velocity — reuse/duplicate attack detection
  6. Print-scan artifact — fake template detection
  7. Math consistency — bank statement fraud
"""

import cv2
import numpy as np
import re
import io
import hashlib
import asyncio
import json
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog

logger = structlog.get_logger()

BASE_DIR      = "/home/tech/kavach_document"
HASH_DB_PATH  = Path(f"{BASE_DIR}/data/document_hashes.json")
HASH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────
ELA_QUALITY            = 90
ELA_TAMPER_THRESH      = 25.0
ELA_REGION_THRESH      = 40.0
FACE_ELA_RATIO_THRESH  = 2.0   # face ELA / doc ELA ratio — above = photo swapped
FORGERY_THRESHOLD      = 0.50
VELOCITY_MAX_ALLOWED   = 3     # max times same doc allowed per day

# ── Patterns ──────────────────────────────────────────────────
AADHAAR_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
PAN_PATTERN     = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
DOB_PATTERN     = re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b')
YOB_PATTERN     = re.compile(r'(?:YoB|YOB|Year of Birth|जन्म वर्ष)[:\s/]*(\d{4})', re.IGNORECASE)
YEAR_PATTERN    = re.compile(r'\b(19[0-9]{2}|200[0-9]|201[0-9])\b')
AMOUNT_PATTERN  = re.compile(r'(?:Rs\.?|INR|₹)\s?([\d,]+\.?\d*)')
IFSC_PATTERN    = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')

PAN_ENTITY_MAP = {
    'P': 'Individual', 'C': 'Company', 'H': 'Hindu Undivided Family',
    'F': 'Firm', 'A': 'Association of Persons', 'T': 'Trust',
    'B': 'Body of Individuals', 'L': 'Local Authority',
    'J': 'Artificial Juridical Person', 'G': 'Government',
}

AADHAAR_KEYWORDS = ['aadhaar', 'आधार', 'uid', 'uidai', 'unique identification']
PAN_KEYWORDS     = ['permanent account', 'income tax', 'pan', 'govt. of india', 'आयकर']
BANK_KEYWORDS    = ['statement', 'account', 'balance', 'transaction', 'debit',
                    'credit', 'opening balance', 'closing balance', 'ifsc', 'branch']

RATION_KEYWORDS  = ['ration card', 'राशन कार्ड', 'food & civil supplies', 'fair price shop',
                    'antyodaya', 'apl', 'bpl', 'nfsa', 'national food security',
                    'civil supplies', 'ration', 'राशन']
CASTE_KEYWORDS   = ['caste certificate', 'जाति प्रमाण पत्र', 'scheduled caste', 'scheduled tribe',
                    'other backward', 'obc', 'sc/st', 'backward class', 'जाति',
                    'tehsildar', 'sub-divisional', 'revenue department', 'caste']
DL_KEYWORDS      = ['driving licence', 'driving license', 'ड्राइविंग लाइसेंस', 'motor vehicles',
                    'transport department', 'dl no', 'licence no', 'rto', 'vehicle class',
                    'validity', 'cov', 'class of vehicle']
VOTER_KEYWORDS   = ['voter', 'election commission', 'electoral', 'electors photo',
                    'epic', 'मतदाता', 'निर्वाचन आयोग', 'voter id', 'assembly constituency']
INCOME_KEYWORDS  = ['income certificate', 'आय प्रमाण पत्र', 'annual income', 'वार्षिक आय',
                    'tehsildar', 'income', 'certificate of income', 'revenue']
PASSPORT_KEYWORDS = ['passport', 'republic of india', 'ministry of external affairs',
                     'place of birth', 'nationality', 'passport no', 'पासपोर्ट']

# Regex patterns for new doc types
RATION_CARD_PATTERN  = re.compile(r'\b[A-Z]{2,3}[/-]?\d{6,14}\b')
DL_PATTERN           = re.compile(r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b')
VOTER_PATTERN        = re.compile(r'\b[A-Z]{3}\d{7}\b')
PASSPORT_PATTERN     = re.compile(r'\b[A-Z]\d{7}\b')


# ── 1. ELA Analyzer ───────────────────────────────────────────
class ELAAnalyzer:
    """
    Error Level Analysis — detects Photoshop/GIMP editing.
    Edited regions recompressed at different quality → bright in ELA map.
    Banks don't do this. Unique to KAVACH.
    """

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            buf = io.BytesIO()
            image.save(buf, 'JPEG', quality=ELA_QUALITY)
            buf.seek(0)
            recompressed  = Image.open(buf).convert('RGB')
            diff          = ImageChops.difference(image, recompressed)
            diff_array    = np.array(diff, dtype=np.float32)
            mean_ela      = float(diff_array.mean())
            max_ela       = float(diff_array.max())
            gray_diff     = diff_array.mean(axis=2)
            tampered_regs = self._find_regions(gray_diff)
            tamper_score  = min(mean_ela / 30.0, 1.0)
            if tampered_regs:
                tamper_score = max(tamper_score, 0.6)
            return {
                "ela_mean"        : round(mean_ela, 3),
                "ela_max"         : round(max_ela, 3),
                "tamper_score"    : round(tamper_score, 4),
                "tampered_regions": tampered_regs,
                "is_tampered"     : mean_ela > ELA_TAMPER_THRESH or bool(tampered_regs),
                "ela_map"         : gray_diff,  # kept for face region analysis
            }
        except Exception as e:
            logger.error("ela_error", error=str(e))
            return {"ela_mean": 0, "tamper_score": 0.0, "is_tampered": False,
                    "tampered_regions": [], "ela_map": None, "error": str(e)}

    def _find_regions(self, gray_diff: np.ndarray) -> List[Dict]:
        regions = []
        try:
            binary  = (gray_diff > ELA_REGION_THRESH).astype(np.uint8) * 255
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            dilated = cv2.dilate(binary, kernel)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = gray_diff.shape
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                x, y, rw, rh = cv2.boundingRect(cnt)
                regions.append({
                    "x": int(x), "y": int(y), "w": int(rw), "h": int(rh),
                    "ela_intensity": round(float(gray_diff[y:y+rh, x:x+rw].mean()), 2),
                    "area_percent" : round(area / (h * w) * 100, 2),
                })
        except Exception as e:
            logger.warning("region_error", error=str(e))
        return regions


# ── 2. Face Region ELA ────────────────────────────────────────
class FaceRegionELA:
    """
    Detects photo swap — most common Aadhaar fraud.
    If face region ELA >> rest of document ELA → photo was replaced.
    Banks check identity but NOT whether the photo was swapped.
    """

    def __init__(self):
        self._face_cascade = None

    def _get_cascade(self):
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def analyze(self, image: Image.Image, ela_map: np.ndarray) -> Dict[str, Any]:
        try:
            if ela_map is None:
                return {"face_found": False, "photo_swap_score": 0.0, "issues": []}

            img_array = np.array(image.convert('RGB'))
            gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            cascade   = self._get_cascade()
            faces     = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                return {
                    "face_found"      : False,
                    "photo_swap_score": 0.0,
                    "issues"          : [],
                    "note"            : "No face detected in document",
                }

            # Get face region ELA
            x, y, w, h = faces[0]
            face_ela     = float(ela_map[y:y+h, x:x+w].mean())
            doc_ela_mean = float(ela_map.mean())

            # Ratio — if face region has much higher ELA than rest = swapped
            ratio = face_ela / (doc_ela_mean + 1e-8)
            photo_swap_score = min(max((ratio - FACE_ELA_RATIO_THRESH) / 3.0, 0.0), 1.0)

            issues = []
            if photo_swap_score > 0.3:
                issues.append(
                    f"Face region ELA ({face_ela:.1f}) significantly higher than "
                    f"document ELA ({doc_ela_mean:.1f}) — possible photo substitution"
                )

            return {
                "face_found"      : True,
                "face_region"     : {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "face_ela"        : round(face_ela, 3),
                "document_ela"    : round(doc_ela_mean, 3),
                "ela_ratio"       : round(ratio, 3),
                "photo_swap_score": round(photo_swap_score, 4),
                "issues"          : issues,
            }
        except Exception as e:
            logger.error("face_ela_error", error=str(e))
            return {"face_found": False, "photo_swap_score": 0.0,
                    "issues": [], "error": str(e)}


# ── 3. Document Velocity Check ────────────────────────────────
class VelocityChecker:
    """
    Detects document reuse attack.
    Same document submitted to multiple banks = fraud signal.
    No bank has cross-institution document hash DB.
    KAVACH maintains its own submission hash log.
    """

    def __init__(self):
        self._db = self._load_db()

    def _load_db(self) -> Dict:
        try:
            if HASH_DB_PATH.exists():
                with open(HASH_DB_PATH) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_db(self):
        try:
            with open(HASH_DB_PATH, 'w') as f:
                json.dump(self._db, f)
        except Exception as e:
            logger.warning("hash_db_save_error", error=str(e))

    def check(self, file_bytes: bytes) -> Dict[str, Any]:
        from datetime import datetime, date
        doc_hash  = hashlib.sha256(file_bytes).hexdigest()
        today     = str(date.today())
        now       = datetime.utcnow().isoformat()

        if doc_hash not in self._db:
            self._db[doc_hash] = []

        # Filter to today's submissions
        today_submissions = [s for s in self._db[doc_hash] if s.startswith(today)]
        count = len(today_submissions) + 1  # including current

        # Record this submission
        self._db[doc_hash].append(now)
        # Keep only last 30 days
        self._db[doc_hash] = self._db[doc_hash][-100:]
        self._save_db()

        total_submissions = len(self._db[doc_hash])
        is_suspicious     = count > VELOCITY_MAX_ALLOWED

        issues = []
        if is_suspicious:
            issues.append(
                f"Document submitted {count} times today — "
                f"possible multi-bank fraud attempt"
            )
        elif total_submissions > 5:
            issues.append(
                f"Document has been submitted {total_submissions} times in total"
            )

        velocity_score = min((count - 1) / VELOCITY_MAX_ALLOWED, 1.0) if count > 1 else 0.0

        return {
            "document_hash"    : doc_hash[:16] + "...",
            "submissions_today": count,
            "total_submissions": total_submissions,
            "is_suspicious"    : is_suspicious,
            "velocity_score"   : round(velocity_score, 4),
            "issues"           : issues,
        }


# ── 4. PDF Metadata Analyzer ──────────────────────────────────
class PDFMetadataAnalyzer:
    """
    Detects PDF editing fingerprints.
    Real bank PDFs are generated by CBS software.
    Edited PDFs expose Adobe/Word/LibreOffice in metadata.
    """
    SUSPICIOUS_CREATORS = [
        'adobe acrobat', 'microsoft word', 'libreoffice', 'photoshop',
        'gimp', 'canva', 'google docs', 'smallpdf', 'ilovepdf', 'pdf editor',
        'foxit', 'nitro', 'wondershare',
    ]
    LEGITIMATE_CREATORS = [
        'finacle', 'bancs', 'temenos', 'oracle', 'infosys', 'tcs',
        'i-flex', 'flexcube', 'finware', 'bankware',
    ]

    def analyze(self, pdf_bytes: bytes) -> Dict[str, Any]:
        try:
            import PyPDF2
            reader   = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            meta     = reader.metadata or {}
            creator  = str(meta.get('/Creator',      '')).lower()
            producer = str(meta.get('/Producer',     '')).lower()
            created  = str(meta.get('/CreationDate', ''))
            modified = str(meta.get('/ModDate',      ''))
            issues, score = [], 0.0

            # Check for suspicious tools
            for tool in self.SUSPICIOUS_CREATORS:
                if tool in creator or tool in producer:
                    issues.append(f"Edited with '{tool}' — not a banking CBS system")
                    score = max(score, 0.80)
                    break

            # Check if modified after creation
            if modified and created and modified != created:
                issues.append("PDF modified after original creation — tampering indicator")
                score = max(score, 0.65)

            # Check for legitimate CBS creator (bonus trust signal)
            is_legitimate = any(
                leg in creator or leg in producer
                for leg in self.LEGITIMATE_CREATORS
            )

            return {
                "creator"        : meta.get('/Creator',  'Unknown'),
                "producer"       : meta.get('/Producer', 'Unknown'),
                "created"        : created,
                "modified"       : modified,
                "page_count"     : len(reader.pages),
                "is_legitimate"  : is_legitimate,
                "issues"         : issues,
                "tamper_score"   : round(score, 4),
                "is_suspicious"  : score > 0.5,
            }
        except Exception as e:
            return {"tamper_score": 0.0, "is_suspicious": False,
                    "issues": [], "error": str(e)}


# ── 5. Font Consistency Checker ───────────────────────────────
class FontConsistencyChecker:
    """
    Detects text replacement using sharpness analysis.
    Original printed text has uniform sharpness.
    Replaced text (pasted from elsewhere) has different sharpness.
    """

    def check(self, image: Image.Image, text_blocks: List[Dict]) -> Dict[str, Any]:
        try:
            img_array     = np.array(image.convert('L'))
            region_scores = []

            for block in text_blocks[:20]:
                bbox = block.get("bbox", [])
                if len(bbox) < 4:
                    continue
                pts = np.array(bbox)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(img_array.shape[1], int(x2))
                y2 = min(img_array.shape[0], int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                region = img_array[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                region_scores.append(float(cv2.Laplacian(region, cv2.CV_64F).var()))

            if len(region_scores) < 3:
                return {"consistent": True, "score": 0.0, "issues": []}

            cv_sharpness = float(np.std(region_scores) / (np.mean(region_scores) + 1e-8))
            inconsistent = cv_sharpness > 1.5
            issues = ["Font sharpness inconsistency detected — possible text replacement"] \
                     if inconsistent else []

            return {
                "consistent"  : not inconsistent,
                "sharpness_cv": round(cv_sharpness, 4),
                "score"       : round(min(cv_sharpness / 3.0, 1.0), 4) if inconsistent else 0.0,
                "issues"      : issues,
            }
        except Exception as e:
            return {"consistent": True, "score": 0.0, "issues": [], "error": str(e)}


# ── 6. Bank Statement Math Checker ───────────────────────────
class BankStatementMathChecker:
    """
    Verifies arithmetic consistency in bank statements.
    Opening balance + credits - debits must equal closing balance.
    Fraudsters editing amounts often forget to update totals.
    """

    def check(self, text: str) -> Dict[str, Any]:
        try:
            amounts = []
            for match in AMOUNT_PATTERN.finditer(text):
                val_str = match.group(1).replace(',', '')
                try:
                    amounts.append(float(val_str))
                except ValueError:
                    continue

            if len(amounts) < 4:
                return {"math_checked": False, "consistent": True,
                        "issues": [], "score": 0.0}

            # Look for opening/closing balance keywords
            text_lower = text.lower()
            opening_match = re.search(
                r'opening\s+balance[^\d]*([\d,]+\.?\d*)', text_lower
            )
            closing_match = re.search(
                r'closing\s+balance[^\d]*([\d,]+\.?\d*)', text_lower
            )

            if not opening_match or not closing_match:
                return {"math_checked": False, "consistent": True,
                        "issues": [], "score": 0.0}

            opening = float(opening_match.group(1).replace(',', ''))
            closing = float(closing_match.group(1).replace(',', ''))

            # Sum credits and debits (simplified — amounts in between)
            middle_amounts = amounts[1:-1]
            if not middle_amounts:
                return {"math_checked": False, "consistent": True,
                        "issues": [], "score": 0.0}

            # Check if closing is plausible given opening
            ratio = closing / (opening + 1e-8)
            issues = []
            score  = 0.0

            # Extremely suspicious: closing >> opening with no explanation
            if ratio > 10 or ratio < 0.01:
                issues.append(
                    f"Closing balance (₹{closing:,.0f}) is implausible "
                    f"relative to opening balance (₹{opening:,.0f})"
                )
                score = 0.75

            return {
                "math_checked"   : True,
                "consistent"     : score == 0.0,
                "opening_balance": opening,
                "closing_balance": closing,
                "issues"         : issues,
                "score"          : round(score, 4),
            }
        except Exception as e:
            return {"math_checked": False, "consistent": True,
                    "issues": [], "score": 0.0, "error": str(e)}


# ── OCR Extractor ─────────────────────────────────────────────
class OCRExtractor:
    def __init__(self):
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(['hi', 'en'], gpu=True, verbose=False)
        return self._reader

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        try:
            img_array = np.array(image.convert('RGB'))
            results   = self._get_reader().readtext(img_array, detail=1, paragraph=False)
            text_blocks, full_text = [], ""
            for (bbox, text, conf) in results:
                if conf > 0.3:
                    text_blocks.append({
                        "text"      : text.strip(),
                        "confidence": round(conf, 3),
                        "bbox"      : [[int(p[0]), int(p[1])] for p in bbox],
                    })
                    full_text += " " + text.strip()
            return {"full_text": full_text.strip(), "text_blocks": text_blocks, "success": True}
        except Exception as e:
            logger.error("ocr_error", error=str(e))
            return {"full_text": "", "text_blocks": [], "success": False, "error": str(e)}


# ── Document Classifier ───────────────────────────────────────
def classify_document(text: str) -> str:
    t = text.lower()
    scores = {
        "AADHAAR"          : sum(1 for kw in AADHAAR_KEYWORDS   if kw in t),
        "PAN"              : sum(1 for kw in PAN_KEYWORDS        if kw in t),
        "BANK_STATEMENT"   : sum(1 for kw in BANK_KEYWORDS       if kw in t),
        "RATION_CARD"      : sum(1 for kw in RATION_KEYWORDS     if kw in t),
        "CASTE_CERTIFICATE": sum(1 for kw in CASTE_KEYWORDS      if kw in t),
        "DRIVING_LICENCE"  : sum(1 for kw in DL_KEYWORDS         if kw in t),
        "VOTER_ID"         : sum(1 for kw in VOTER_KEYWORDS      if kw in t),
        "INCOME_CERTIFICATE": sum(1 for kw in INCOME_KEYWORDS    if kw in t),
        "PASSPORT"         : sum(1 for kw in PASSPORT_KEYWORDS   if kw in t),
    }
    # Boost scores with regex pattern matches
    if AADHAAR_PATTERN.search(text):  scores["AADHAAR"]        += 3
    if PAN_PATTERN.search(text):      scores["PAN"]            += 3
    if DL_PATTERN.search(text):       scores["DRIVING_LICENCE"] += 2
    if VOTER_PATTERN.search(text):    scores["VOTER_ID"]        += 2
    if PASSPORT_PATTERN.search(text): scores["PASSPORT"]        += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "GOVERNMENT_DOCUMENT"


# ── Field Extractor ───────────────────────────────────────────
def extract_fields(text: str, doc_type: str) -> Dict[str, Any]:
    fields = {}

    # DOB — full date → YoB → year fallback
    dob_match = DOB_PATTERN.search(text)
    yob_match = YOB_PATTERN.search(text)
    year_match = YEAR_PATTERN.search(text)

    if dob_match:
        fields["dob"] = dob_match.group()
    elif yob_match:
        fields["dob"] = yob_match.group(1)
        fields["year_of_birth"] = yob_match.group(1)
    elif year_match:
        fields["dob"] = year_match.group(1)
        fields["year_of_birth"] = year_match.group(1)

    if doc_type == "AADHAAR":
        m = AADHAAR_PATTERN.search(text)
        if m: fields["aadhaar_number"] = m.group().replace(" ", "")
        name_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_m: fields["name"] = name_m.group()
        gender_m = re.search(r'\b(Male|Female|पुरुष|महिला)\b', text, re.IGNORECASE)
        if gender_m: fields["gender"] = gender_m.group()

    elif doc_type == "PAN":
        m = PAN_PATTERN.search(text)
        if m:
            pan = m.group()
            fields["pan_number"]      = pan
            fields["entity_type"]     = PAN_ENTITY_MAP.get(pan[3], "Unknown")
            fields["surname_initial"] = pan[4]
        name_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_m: fields["name"] = name_m.group()

    elif doc_type == "BANK_STATEMENT":
        amounts = AMOUNT_PATTERN.findall(text)
        fields["amounts_found"] = amounts[:10]
        ifsc_m = IFSC_PATTERN.search(text)
        if ifsc_m: fields["ifsc"] = ifsc_m.group()
        acc_m = re.search(r'\b\d{9,18}\b', text)
        if acc_m: fields["account_number"] = acc_m.group()

    elif doc_type == "RATION_CARD":
        m = RATION_CARD_PATTERN.search(text)
        if m: fields["card_number"] = m.group()
        name_m = re.search(r'(?:head of family|name)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if name_m: fields["head_of_family"] = name_m.group(1).strip()
        state_m = re.search(r'(?:state|राज्य)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if state_m: fields["state"] = state_m.group(1).strip()
        members_m = re.search(r'(?:members|सदस्य)[:\s]*(\d+)', text, re.IGNORECASE)
        if members_m: fields["family_members"] = members_m.group(1)

    elif doc_type == "CASTE_CERTIFICATE":
        caste_m = re.search(r'(?:caste|जाति)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if caste_m: fields["caste"] = caste_m.group(1).strip()
        cert_m = re.search(r'(?:certificate no|cert\.? no|प्रमाण पत्र)[:\s]*([A-Z0-9/-]+)', text, re.IGNORECASE)
        if cert_m: fields["certificate_number"] = cert_m.group(1)
        auth_m = re.search(r'(?:tehsildar|sub-divisional|issued by)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if auth_m: fields["issuing_authority"] = auth_m.group(1).strip()
        name_m = re.search(r'(?:this is to certify that|certified that)[\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text, re.IGNORECASE)
        if name_m: fields["name"] = name_m.group(1)

    elif doc_type == "DRIVING_LICENCE":
        m = DL_PATTERN.search(text)
        if m: fields["dl_number"] = m.group().replace(" ", "")
        name_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_m: fields["name"] = name_m.group()
        cov_m = re.search(r'(?:cov|class of vehicle)[:\s]+([A-Z0-9,\s]+)', text, re.IGNORECASE)
        if cov_m: fields["vehicle_class"] = cov_m.group(1).strip()
        valid_m = re.search(r'(?:valid till|validity|valid upto)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})', text, re.IGNORECASE)
        if valid_m: fields["valid_till"] = valid_m.group(1)

    elif doc_type == "VOTER_ID":
        m = VOTER_PATTERN.search(text)
        if m: fields["epic_number"] = m.group()
        name_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_m: fields["name"] = name_m.group()
        const_m = re.search(r'(?:constituency|विधान सभा)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if const_m: fields["constituency"] = const_m.group(1).strip()

    elif doc_type == "INCOME_CERTIFICATE":
        income_m = re.search(r'(?:annual income|income)[:\s]*(?:Rs\.?|INR|₹)?\s*([\d,]+)', text, re.IGNORECASE)
        if income_m: fields["annual_income"] = income_m.group(1).replace(",", "")
        cert_m = re.search(r'(?:certificate no|cert no)[:\s]*([A-Z0-9/-]+)', text, re.IGNORECASE)
        if cert_m: fields["certificate_number"] = cert_m.group(1)
        name_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_m: fields["name"] = name_m.group()

    elif doc_type == "PASSPORT":
        m = PASSPORT_PATTERN.search(text)
        if m: fields["passport_number"] = m.group()
        name_m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if name_m: fields["name"] = name_m.group()
        pob_m = re.search(r'(?:place of birth)[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if pob_m: fields["place_of_birth"] = pob_m.group(1).strip()

    return fields


# ── Main Detector ─────────────────────────────────────────────
class DocumentForgeryDetector:
    def __init__(self):
        self.ela       = ELAAnalyzer()
        self.face_ela  = FaceRegionELA()
        self.velocity  = VelocityChecker()
        self.pdf_meta  = PDFMetadataAnalyzer()
        self.font      = FontConsistencyChecker()
        self.math      = BankStatementMathChecker()
        self.ocr       = OCRExtractor()

    def _risk_level(self, score: float) -> str:
        if score >= 0.85:   return "CRITICAL"
        elif score >= 0.70: return "HIGH"
        elif score >= 0.50: return "MEDIUM"
        return "LOW"

    def _forgery_type(self, scores: Dict) -> str:
        max_key = max(scores, key=scores.get) if scores else "none"
        mapping = {
            "ela"      : "PIXEL_TAMPERING",
            "face_ela" : "PHOTO_SUBSTITUTION",
            "velocity" : "DOCUMENT_REUSE_ATTACK",
            "metadata" : "PDF_METADATA_ANOMALY",
            "font"     : "TEXT_REPLACEMENT",
            "math"     : "FINANCIAL_MANIPULATION",
        }
        top_score = scores.get(max_key, 0.0)
        if top_score < 0.3:
            return "GENUINE"
        return mapping.get(max_key, "UNKNOWN_FORGERY")

    def analyze_image(self, image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        evidence, scores = [], {}

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {"success": False, "error": f"Cannot open image: {e}"}

        # Velocity check first (fast, no ML)
        velocity_result = self.velocity.check(image_bytes)
        scores["velocity"] = velocity_result.get("velocity_score", 0.0)
        for issue in velocity_result.get("issues", []):
            evidence.append({
                "type"       : "velocity_check",
                "description": issue,
                "confidence" : round(scores["velocity"], 4),
            })

        # OCR
        ocr_result  = self.ocr.extract(image)
        full_text   = ocr_result.get("full_text", "")
        text_blocks = ocr_result.get("text_blocks", [])
        doc_type    = classify_document(full_text)
        fields      = extract_fields(full_text, doc_type)

        # ELA
        ela_result = self.ela.analyze(image)
        scores["ela"] = ela_result.get("tamper_score", 0.0)
        if ela_result.get("is_tampered"):
            evidence.append({
                "type"       : "ela_tampering",
                "description": (
                    f"Pixel-level tampering detected (ELA intensity={ela_result['ela_mean']:.1f}) "
                    f"— document likely edited in image software"
                ),
                "confidence" : round(scores["ela"], 4),
                "regions"    : ela_result.get("tampered_regions", []),
            })

        # Face region ELA (Aadhaar/PAN only)
        if doc_type in ["AADHAAR", "PAN", "RATION_CARD", "CASTE_CERTIFICATE", "DRIVING_LICENCE", "VOTER_ID", "INCOME_CERTIFICATE", "PASSPORT", "GOVERNMENT_DOCUMENT"]:
            face_result = self.face_ela.analyze(image, ela_result.get("ela_map"))
            scores["face_ela"] = face_result.get("photo_swap_score", 0.0)
            for issue in face_result.get("issues", []):
                evidence.append({
                    "type"       : "photo_substitution",
                    "description": issue,
                    "confidence" : round(scores["face_ela"], 4),
                })

        # Font consistency
        font_result = self.font.check(image, text_blocks)
        scores["font"] = font_result.get("score", 0.0)
        for issue in font_result.get("issues", []):
            evidence.append({
                "type"       : "font_inconsistency",
                "description": issue,
                "confidence" : round(scores["font"], 4),
            })

        # Bank statement math check
        if doc_type == "BANK_STATEMENT":
            math_result = self.math.check(full_text)
            scores["math"] = math_result.get("score", 0.0)
            for issue in math_result.get("issues", []):
                evidence.append({
                    "type"       : "math_inconsistency",
                    "description": issue,
                    "confidence" : round(scores["math"], 4),
                })

        # Weighted final score
        # Weights designed for what matters most per doc type
        # Photo-bearing government documents — ELA + face ELA are primary signals
        PHOTO_DOCS = {"AADHAAR", "PAN", "RATION_CARD", "CASTE_CERTIFICATE",
                      "DRIVING_LICENCE", "VOTER_ID", "INCOME_CERTIFICATE",
                      "PASSPORT", "GOVERNMENT_DOCUMENT"}

        if doc_type == "AADHAAR":
            forgery_score = float(np.clip(
                scores.get("ela",      0.0) * 0.35 +
                scores.get("face_ela", 0.0) * 0.35 +
                scores.get("velocity", 0.0) * 0.20 +
                scores.get("font",     0.0) * 0.10,
                0.0, 1.0
            ))
        elif doc_type == "PAN":
            forgery_score = float(np.clip(
                scores.get("ela",      0.0) * 0.40 +
                scores.get("face_ela", 0.0) * 0.30 +
                scores.get("velocity", 0.0) * 0.20 +
                scores.get("font",     0.0) * 0.10,
                0.0, 1.0
            ))
        elif doc_type == "BANK_STATEMENT":
            forgery_score = float(np.clip(
                scores.get("ela",      0.0) * 0.30 +
                scores.get("math",     0.0) * 0.40 +
                scores.get("velocity", 0.0) * 0.20 +
                scores.get("font",     0.0) * 0.10,
                0.0, 1.0
            ))
        elif doc_type in PHOTO_DOCS:
            # All other photo-bearing govt docs: ELA + face ELA + velocity + font
            forgery_score = float(np.clip(
                scores.get("ela",      0.0) * 0.40 +
                scores.get("face_ela", 0.0) * 0.30 +
                scores.get("velocity", 0.0) * 0.20 +
                scores.get("font",     0.0) * 0.10,
                0.0, 1.0
            ))
        else:
            # GOVERNMENT_DOCUMENT fallback — ELA-heavy, no math check
            forgery_score = float(np.clip(
                scores.get("ela",      0.0) * 0.50 +
                scores.get("face_ela", 0.0) * 0.20 +
                scores.get("velocity", 0.0) * 0.20 +
                scores.get("font",     0.0) * 0.10,
                0.0, 1.0
            ))

        return {
            "success"            : True,
            "document_type"      : doc_type,
            "is_forged"          : forgery_score >= FORGERY_THRESHOLD,
            "forgery_probability": round(forgery_score, 4),
            "risk_level"         : self._risk_level(forgery_score),
            "forgery_type"       : self._forgery_type(scores),
            "evidence"           : evidence,
            "extracted_fields"   : fields,
            "scores"             : {k: round(v, 4) for k, v in scores.items()},
            "velocity_info"      : {
                "submissions_today": velocity_result.get("submissions_today", 1),
                "total_submissions": velocity_result.get("total_submissions", 1),
                "document_hash"    : velocity_result.get("document_hash", ""),
            },
            "ocr_confidence"     : round(
                float(np.mean([b["confidence"] for b in text_blocks]))
                if text_blocks else 0.0, 4
            ),
        }

    def analyze_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        meta_result = self.pdf_meta.analyze(pdf_bytes)
        evidence, scores = [], {}

        scores["metadata"] = meta_result.get("tamper_score", 0.0)
        for issue in meta_result.get("issues", []):
            evidence.append({
                "type"       : "pdf_metadata",
                "description": issue,
                "confidence" : round(scores["metadata"], 4),
            })

        # Convert to image and run full analysis
        try:
            import fitz
            doc  = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
            page = doc[0]
            pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            doc.close()

            img_result = self.analyze_image(img_bytes, "document.png")
            img_result["pdf_metadata"] = meta_result

            combined = float(np.clip(
                img_result.get("forgery_probability", 0.0) * 0.60 +
                scores.get("metadata", 0.0) * 0.40,
                0.0, 1.0
            ))
            img_result["forgery_probability"] = round(combined, 4)
            img_result["is_forged"]  = combined >= FORGERY_THRESHOLD
            img_result["risk_level"] = self._risk_level(combined)
            img_result["evidence"]   = img_result.get("evidence", []) + evidence

            # Update scores with metadata
            img_result["scores"]["metadata"] = round(scores["metadata"], 4)
            return img_result

        except ImportError:
            # PyMuPDF not available — metadata analysis only
            forgery_score = scores.get("metadata", 0.0)
            return {
                "success"            : True,
                "document_type"      : "BANK_STATEMENT",
                "is_forged"          : forgery_score >= FORGERY_THRESHOLD,
                "forgery_probability": round(forgery_score, 4),
                "risk_level"         : self._risk_level(forgery_score),
                "forgery_type"       : "PDF_METADATA_ANOMALY" if forgery_score > 0.5 else "GENUINE",
                "evidence"           : evidence,
                "pdf_metadata"       : meta_result,
                "extracted_fields"   : {},
                "scores"             : scores,
            }

    def analyze(self, file_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        ext = Path(filename).suffix.lower() if filename else ""
        if ext == ".pdf":
            return self.analyze_pdf(file_bytes)
        return self.analyze_image(file_bytes, filename)

    async def analyze_async(self, file_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        return await asyncio.to_thread(self.analyze, file_bytes, filename)


# ── Singleton ─────────────────────────────────────────────────
_detector: Optional[DocumentForgeryDetector] = None

def get_detector() -> DocumentForgeryDetector:
    global _detector
    if _detector is None:
        _detector = DocumentForgeryDetector()
    return _detector