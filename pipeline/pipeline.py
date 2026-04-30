"""
KAVACH — Document Forgery Detection Pipeline v3
preprocess → OCR → forensics (parallel) → doc-specific validation → risk score
"""

import io, re, cv2, json, hashlib, asyncio
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
import structlog

from core.preprocessor import DocumentPreprocessor
from core.ocr          import OCRExtractor
from core.forensics    import (MultiScaleELA, CopyMoveDetector, NoiseInconsistencyDetector,
                               PicsartEditDetector, MoirePatternDetector, DocumentLivenessDetector,
                               DeepfakeDetector)
from core.face         import FaceRegionELA, FaceMatcher
from core.validators   import (AadhaarValidator, PANValidator, BankStatementValidator,
                                VoterIDValidator, RationCardValidator, DrivingLicenceValidator)
from core.scoring      import RiskScoringEngine

logger = structlog.get_logger()

BASE_DIR     = '/home/tech/kavach_document'
HASH_DB_PATH = Path(f'{BASE_DIR}/data/document_hashes.json')
HASH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
VELOCITY_MAX = 999  # disabled for demo

# ── Classification tables ─────────────────────────────────────
_KW = {
    'AADHAAR':            ['aadhaar', 'आधार', 'uid', 'uidai', 'unique identification'],
    'PAN':                ['permanent account', 'income tax', 'pan', 'govt. of india', 'आयकर'],
    'BANK_STATEMENT':     ['statement', 'account', 'balance', 'transaction', 'debit',
                           'credit', 'opening balance', 'closing balance', 'ifsc'],
    'RATION_CARD':        ['ration card', 'राशन कार्ड', 'nfsa', 'bpl', 'apl',
                           'fair price shop', 'civil supplies', 'antyodaya', 'ration'],
    'CASTE_CERTIFICATE':  ['caste certificate', 'जाति प्रमाण', 'scheduled caste',
                           'scheduled tribe', 'obc', 'backward class', 'tehsildar'],
    'DRIVING_LICENCE':    ['driving licence', 'driving license', 'motor vehicles',
                           'transport department', 'rto', 'cov', 'class of vehicle'],
    'VOTER_ID':           ['voter', 'election commission', 'electoral', 'epic',
                           'मतदाता', 'निर्वाचन आयोग', 'electors photo'],
    'INCOME_CERTIFICATE': ['income certificate', 'आय प्रमाण', 'annual income',
                           'certificate of income'],
    'PASSPORT':           ['passport', 'republic of india', 'ministry of external affairs',
                           'place of birth', 'nationality'],
}

_RE = {
    'AADHAAR': re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'),
    'PAN':     re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'),
    'DL':      re.compile(r'\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{7}\b'),
    'VOTER':   re.compile(r'\b[A-Z]{2,3}\d{7}\b'),
    'PASSPORT':re.compile(r'\b[A-Z]\d{7}\b'),
}

_AMOUNT_RE = re.compile(r'(?:Rs\.?|INR|₹)\s?([\d,]+\.?\d*)')
_IFSC_RE   = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')
_DOB_RE    = re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b')
_YOB_RE    = re.compile(r'(?:YoB|YOB|Year of Birth|जन्म वर्ष)[:\s/]*(\d{4})', re.IGNORECASE)
_CARD_NO_RE= re.compile(r'\b[A-Z]{2}[/\-]?\d{2}[/\-]?\d{4,8}\b')
_MEMBER_RE = re.compile(r'(?:total\s+members?|members?)[:\s]*(\d{1,2})', re.IGNORECASE)


def _classify(text: str) -> str:
    t = text.lower()
    scores = {k: sum(1 for kw in kws if kw in t) for k, kws in _KW.items()}
    if _RE['AADHAAR'].search(text): scores['AADHAAR']         += 3
    if _RE['PAN'].search(text):     scores['PAN']             += 3
    if _RE['DL'].search(text):      scores['DRIVING_LICENCE'] += 2
    if _RE['VOTER'].search(text):   scores['VOTER_ID']        += 2
    if _RE['PASSPORT'].search(text):scores['PASSPORT']        += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'GOVERNMENT_DOCUMENT'


def _extract_fields(text: str, doc_type: str) -> Dict[str, Any]:
    fields = {}
    dob_m = _DOB_RE.search(text)
    yob_m = _YOB_RE.search(text)
    if dob_m:   fields['dob'] = dob_m.group()
    elif yob_m: fields['dob'] = yob_m.group(1)

    if doc_type == 'AADHAAR':
        m = _RE['AADHAAR'].search(text)
        if m: fields['aadhaar_number'] = m.group().replace(' ', '')
        nm = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if nm: fields['name'] = nm.group()
        gm = re.search(r'\b(Male|Female|पुरुष|महिला)\b', text, re.IGNORECASE)
        if gm: fields['gender'] = gm.group()

    elif doc_type == 'PAN':
        m = _RE['PAN'].search(text)
        if m:
            fields['pan_number']  = m.group()
            fields['entity_type'] = {
                'P':'Individual','C':'Company','H':'HUF','F':'Firm',
                'A':'AOP','T':'Trust','B':'BOI','L':'Local Authority',
                'J':'AJP','G':'Government',
            }.get(m.group()[3], 'Unknown')
        nm = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if nm: fields['name'] = nm.group()

    elif doc_type == 'BANK_STATEMENT':
        fields['amounts'] = _AMOUNT_RE.findall(text)[:10]
        im = _IFSC_RE.search(text)
        if im: fields['ifsc'] = im.group()
        am = re.search(r'\b\d{9,18}\b', text)
        if am: fields['account_number'] = am.group()

    elif doc_type == 'VOTER_ID':
        m = _RE['VOTER'].search(text)
        if m: fields['epic_number'] = m.group()
        nm = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if nm: fields['name'] = nm.group()
        gm = re.search(r'\b(Male|Female|पुरुष|महिला)\b', text, re.IGNORECASE)
        if gm: fields['gender'] = gm.group()

    elif doc_type == 'RATION_CARD':
        m = _CARD_NO_RE.search(text)
        if m: fields['card_number'] = m.group()
        mm = _MEMBER_RE.search(text)
        if mm: fields['member_count'] = int(mm.group(1))
        nm = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if nm: fields['head_name'] = nm.group()
        for ct in ('AAY','BPL','APL','PHH','NPHH'):
            if ct in text.upper():
                fields['card_type'] = ct
                break

    elif doc_type == 'DRIVING_LICENCE':
        m = _RE['DL'].search(text)
        if m: fields['dl_number'] = m.group().replace(' ', '')
        vm = re.search(r'(?:valid till|validity)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
                       text, re.IGNORECASE)
        if vm: fields['valid_till'] = vm.group(1)
        nm = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if nm: fields['name'] = nm.group()

    elif doc_type == 'PASSPORT':
        m = _RE['PASSPORT'].search(text)
        if m: fields['passport_number'] = m.group()
        nm = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if nm: fields['name'] = nm.group()

    return fields


# ── Velocity checker ──────────────────────────────────────────

class VelocityChecker:
    def __init__(self):
        self._db = self._load()

    def _load(self) -> Dict:
        try:
            if HASH_DB_PATH.exists():
                return json.loads(HASH_DB_PATH.read_text())
        except Exception:
            pass
        return {}

    def _save(self):
        try:
            HASH_DB_PATH.write_text(json.dumps(self._db))
        except Exception as e:
            logger.warning('hash_db_save_error', error=str(e))

    def check(self, file_bytes: bytes) -> Dict[str, Any]:
        doc_hash    = hashlib.sha256(file_bytes).hexdigest()
        today       = str(date.today())
        now         = datetime.utcnow().isoformat()
        self._db.setdefault(doc_hash, [])
        today_count = sum(1 for s in self._db[doc_hash] if s.startswith(today)) + 1
        self._db[doc_hash].append(now)
        self._db[doc_hash] = self._db[doc_hash][-100:]
        self._save()
        total      = len(self._db[doc_hash])
        suspicious = today_count > VELOCITY_MAX
        score      = float(np.clip((today_count - 1) / VELOCITY_MAX, 0.0, 1.0)) \
                     if today_count > 1 else 0.0
        return {
            'document_hash':     doc_hash[:16] + '...',
            'submissions_today': today_count,
            'total_submissions': total,
            'velocity_score':    round(score, 4),
            'issues':            [f'Document submitted {today_count}× today — multi-bank fraud signal']
                                 if suspicious else [],
        }


# ── Font consistency ──────────────────────────────────────────

class FontConsistencyChecker:
    def check(self, image: Image.Image, text_blocks: List[Dict]) -> Dict[str, Any]:
        try:
            arr, scores = np.array(image.convert('L')), []
            for block in text_blocks[:20]:
                pts = np.array(block.get('bbox', []))
                if len(pts) < 4: continue
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                x1, y1 = max(0,int(x1)), max(0,int(y1))
                x2, y2 = min(arr.shape[1],int(x2)), min(arr.shape[0],int(y2))
                if x2 <= x1 or y2 <= y1: continue
                region = arr[y1:y2, x1:x2]
                if region.size > 0:
                    scores.append(float(cv2.Laplacian(region, cv2.CV_64F).var()))
            if len(scores) < 3:
                return {'score': 0.0, 'issues': []}
            cv_val = float(np.std(scores) / (np.mean(scores) + 1e-8))
            score  = round(float(np.clip(cv_val / 3.0, 0.0, 1.0)), 4) if cv_val > 1.5 else 0.0
            return {
                'score':  score,
                'issues': ['Font sharpness inconsistency — possible text replacement']
                          if score > 0 else [],
            }
        except Exception:
            return {'score': 0.0, 'issues': []}


# ── PDF metadata ──────────────────────────────────────────────

class PDFMetadataAnalyzer:
    _SUSPICIOUS = ['adobe acrobat','microsoft word','libreoffice','photoshop',
                   'gimp','canva','google docs','smallpdf','ilovepdf',
                   'foxit','nitro','wondershare','pdf editor']

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
            for tool in self._SUSPICIOUS:
                if tool in creator or tool in producer:
                    issues.append(f"Edited with '{tool}' — not a CBS banking system")
                    score = max(score, 0.80)
                    break
            if modified and created and modified != created:
                issues.append('PDF modified after creation — tampering indicator')
                score = max(score, 0.65)
            return {'creator': meta.get('/Creator','Unknown'),
                    'producer': meta.get('/Producer','Unknown'),
                    'tamper_score': round(score, 4), 'issues': issues}
        except Exception as e:
            return {'tamper_score': 0.0, 'issues': [], 'error': str(e)}


# ── Main pipeline ─────────────────────────────────────────────

class DocumentForgeryDetector:
    def __init__(self):
        self.preprocessor  = DocumentPreprocessor()
        self.ocr           = OCRExtractor()
        self.ela           = MultiScaleELA()
        self.copy_move     = CopyMoveDetector()
        self.noise         = NoiseInconsistencyDetector()
        self.picsart       = PicsartEditDetector()
        self.moire         = MoirePatternDetector()
        self.liveness      = DocumentLivenessDetector()
        self.deepfake      = DeepfakeDetector()
        self.face_ela      = FaceRegionELA()
        self.face_matcher  = FaceMatcher()
        self.velocity      = VelocityChecker()
        self.font          = FontConsistencyChecker()
        self.pdf_meta      = PDFMetadataAnalyzer()
        self.aadhaar_val   = AadhaarValidator()
        self.pan_val       = PANValidator()
        self.bank_val      = BankStatementValidator()
        self.voter_val     = VoterIDValidator()
        self.ration_val    = RationCardValidator()
        self.dl_val        = DrivingLicenceValidator()
        self.scorer        = RiskScoringEngine()
        self._executor     = ThreadPoolExecutor(max_workers=4)

    def _ev(self, label: str, issues: List[str], conf: float) -> List[Dict]:
        return [{'type': label, 'description': i, 'confidence': round(conf, 4)}
                for i in issues]

    def analyze_image(self, image_bytes: bytes,
                      selfie_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        # Synchronous wrapper for compatibility
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # If already in a loop, we can't use run_until_complete easily for a complex flow
            # But since this is a ThreadPoolExecutor context usually, we might need a better way.
            # For now, let's make the main logic async and call it.
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.analyze_image_async(image_bytes, selfie_bytes))
        else:
            return loop.run_until_complete(self.analyze_image_async(image_bytes, selfie_bytes))

    async def analyze_image_async(self, image_bytes: bytes,
                                  selfie_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        evidence, signals = [], {}

        try:
            raw = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {'success': False, 'error': f'Cannot open image: {e}'}

        image = self.preprocessor.process(raw)

        # ── 1. Velocity ───────────────────────────────────────
        vel = self.velocity.check(image_bytes)
        signals['velocity'] = vel['velocity_score']
        evidence += self._ev('velocity_check', vel['issues'], vel['velocity_score'])

        # ── 2. OCR ────────────────────────────────────────────
        ocr_result  = self.ocr.extract(image)
        full_text   = ocr_result.get('full_text', '')
        text_blocks = ocr_result.get('text_blocks', [])
        doc_type    = _classify(full_text)
        fields      = _extract_fields(full_text, doc_type)

        # ── 3. Forensics — run in parallel ────────────────────
        loop = asyncio.get_event_loop()
        futures = {
            'ela':       loop.run_in_executor(self._executor, self.ela.analyze,       image),
            'copy_move': loop.run_in_executor(self._executor, self.copy_move.detect,  image),
            'noise':     loop.run_in_executor(self._executor, self.noise.detect,      image),
            'font':      loop.run_in_executor(self._executor, self.font.check,        image, text_blocks),
            'picsart':   loop.run_in_executor(self._executor, self.picsart.detect,    image),
            'moire':     loop.run_in_executor(self._executor, self.moire.detect,      image),
            'liveness':  loop.run_in_executor(self._executor, self.liveness.detect,   image),
            'deepfake':  self.deepfake.analyze(image), # Already async
        }
        
        results = {}
        for k, f in futures.items():
            if asyncio.iscoroutine(f):
                results[k] = await f
            else:
                results[k] = await f

        ela_result     = results['ela']
        cm_result      = results['copy_move']
        noise_result   = results['noise']
        font_result    = results['font']
        picsart_result = results['picsart']
        moire_result   = results['moire']
        live_result    = results['liveness']
        df_result      = results['deepfake']

        signals['ela']       = ela_result['tamper_score']
        signals['copy_move'] = cm_result['score']
        signals['noise']     = noise_result['score']
        signals['font']      = font_result['score']
        signals['picsart']   = picsart_result['score']
        signals['moire']     = moire_result['score']
        signals['liveness']  = live_result['score']
        signals['deepfake']  = df_result['score']

        if ela_result['is_tampered']:
            evidence += self._ev('ela_tampering',
                [f"Multi-scale ELA tamper={ela_result['tamper_score']:.2f} "
                 f"dct={ela_result['dct_score']:.2f} "
                 f"regions={len(ela_result['tampered_regions'])}"],
                ela_result['tamper_score'])
        evidence += self._ev('copy_move',    cm_result['issues'],      cm_result['score'])
        evidence += self._ev('noise',        noise_result['issues'],   noise_result['score'])
        evidence += self._ev('font',         font_result['issues'],    font_result['score'])
        evidence += self._ev('picsart_edit', picsart_result['issues'], picsart_result['score'])
        evidence += self._ev('moire_pattern', moire_result['issues'],   moire_result['score'])
        evidence += self._ev('doc_liveness',  live_result['issues'],    live_result['score'])
        evidence += self._ev('ai_deepfake',   df_result['issues'],      df_result['score'])

        # ── 4. Face region ELA ────────────────────────────────
        face_ela_r = self.face_ela.analyze(image, ela_result.get('ela_map'))
        signals['face_mismatch'] = face_ela_r.get('photo_swap_score', 0.0)
        evidence += self._ev('photo_substitution',
                             face_ela_r.get('issues', []),
                             face_ela_r.get('photo_swap_score', 0.0))

        # ── 5. Selfie face match ──────────────────────────────
        face_match_result = {}
        if selfie_bytes:
            try:
                selfie_img    = Image.open(io.BytesIO(selfie_bytes))
                # match is sync
                face_match_result = await loop.run_in_executor(self._executor, self.face_matcher.match, image, selfie_img)
                if face_match_result.get('matched') is False:
                    signals['face_mismatch'] = max(
                        signals['face_mismatch'],
                        1.0 - face_match_result['score']
                    )
                    evidence += self._ev('face_match', face_match_result['issues'],
                                         1.0 - face_match_result['score'])
            except Exception as e:
                logger.warning('selfie_match_error', error=str(e))

        # ── 6. Document-specific validation ───────────────────
        val_score, val_issues, qr_result = 0.0, [], {}

        if doc_type == 'AADHAAR':
            ocr_conf   = round(float(np.mean([b['confidence'] for b in text_blocks])) if text_blocks else 0.0, 4)
            # validate is sync
            val        = await loop.run_in_executor(self._executor, self.aadhaar_val.validate, fields, image, ocr_conf)
            val_score  = val['score']
            val_issues = val['issues']
            qr_result  = val.get('qr_result', {})
            signals['qr_mismatch'] = qr_result.get('mismatch_score', 0.0)
            evidence += self._ev('qr_verification',
                                 qr_result.get('issues', []),
                                 qr_result.get('mismatch_score', 0.0))

        elif doc_type == 'PAN':
            val        = self.pan_val.validate(fields)
            val_score  = val['score']
            val_issues = val['issues']

        elif doc_type == 'BANK_STATEMENT':
            val        = self.bank_val.validate(full_text)
            val_score  = val['score']
            val_issues = val['issues']

        elif doc_type == 'VOTER_ID':
            val        = self.voter_val.validate(fields, full_text, image)
            val_score  = val['score']
            val_issues = val['issues']

        elif doc_type == 'RATION_CARD':
            val        = self.ration_val.validate(fields, full_text, image)
            val_score  = val['score']
            val_issues = val['issues']

        elif doc_type == 'DRIVING_LICENCE':
            val        = self.dl_val.validate(fields, full_text)
            val_score  = val['score']
            val_issues = val['issues']

        signals['validator'] = val_score
        evidence += self._ev('field_validation', val_issues, val_score)

        # ── 7. Risk score (doc-type-aware weights) ────────────
        result = self.scorer.score(signals, doc_type)

        return {
            'success':             True,
            'document_type':       doc_type,
            'is_forged':           result['decision'] == 'REJECTED',
            'forgery_probability': result['forgery_probability'],
            'risk_level':          result['risk_level'],
            'decision':            result['decision'],
            'forgery_type':        result['forgery_type'],
            'evidence':            evidence,
            'extracted_fields':    fields,
            'scores':              {k: round(v, 4) for k, v in signals.items()},
            'signal_breakdown':    result['signal_breakdown'],
            'top_signals':         result['top_signals'],
            'weight_profile':      result['weight_profile'],
            'velocity_info':       {
                'submissions_today': vel['submissions_today'],
                'total_submissions': vel['total_submissions'],
                'document_hash':     vel['document_hash'],
            },
            'ocr_engine':          ocr_result.get('engine', 'unknown'),
            'ocr_confidence':      round(
                float(np.mean([b['confidence'] for b in text_blocks]))
                if text_blocks else 0.0, 4),
            'face_match':          face_match_result or None,
            'qr_verification':     qr_result or None,
        }

    def analyze_pdf(self, pdf_bytes: bytes,
                    selfie_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        # Synchronous wrapper for compatibility
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.analyze_pdf_async(pdf_bytes, selfie_bytes))
        else:
            return loop.run_until_complete(self.analyze_pdf_async(pdf_bytes, selfie_bytes))

    async def analyze_pdf_async(self, pdf_bytes: bytes,
                                selfie_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        meta = self.pdf_meta.analyze(pdf_bytes)
        try:
            import fitz
            doc  = fitz.open(stream=bytes(pdf_bytes), filetype='pdf')
            pix  = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            doc.close()
            img_result = await self.analyze_image_async(pix.tobytes('png'), selfie_bytes)
            img_result['pdf_metadata'] = meta
            meta_score = meta.get('tamper_score', 0.0)
            if meta_score > 0:
                img_result['scores']['metadata'] = meta_score
                combined = float(np.clip(
                    img_result['forgery_probability'] * 0.90 + meta_score * 0.10, 0.0, 1.0))
                img_result['forgery_probability'] = round(combined, 4)
                img_result['evidence'] += self._ev('pdf_metadata', meta.get('issues',[]), meta_score)
            return img_result
        except ImportError:
            score = meta.get('tamper_score', 0.0)
            return {
                'success': True, 'document_type': 'BANK_STATEMENT',
                'is_forged': score >= 0.5, 'forgery_probability': round(score, 4),
                'risk_level': 'HIGH' if score >= 0.7 else 'MEDIUM' if score >= 0.5 else 'LOW',
                'decision': 'REJECTED' if score >= 0.6 else 'REVIEW' if score >= 0.3 else 'APPROVED',
                'forgery_type': 'PDF_METADATA_ANOMALY' if score > 0.5 else 'GENUINE',
                'evidence': self._ev('pdf_metadata', meta.get('issues',[]), score),
                'pdf_metadata': meta, 'extracted_fields': {}, 'scores': {'metadata': score},
            }

    def analyze(self, file_bytes: bytes, filename: str = '',
                selfie_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.analyze_async(file_bytes, filename, selfie_bytes))
        else:
            return loop.run_until_complete(self.analyze_async(file_bytes, filename, selfie_bytes))

    async def analyze_async(self, file_bytes: bytes, filename: str = '',
                            selfie_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        if Path(filename).suffix.lower() == '.pdf':
            return await self.analyze_pdf_async(file_bytes, selfie_bytes)
        return await self.analyze_image_async(file_bytes, selfie_bytes)


_detector: Optional[DocumentForgeryDetector] = None

def get_detector() -> DocumentForgeryDetector:
    global _detector
    if _detector is None:
        _detector = DocumentForgeryDetector()
    return _detector
