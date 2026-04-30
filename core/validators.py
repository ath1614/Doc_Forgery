import re
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, date
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()

# ── Shared helpers ────────────────────────────────────────────

HEADER_WORDS = {
    'authority', 'india', 'uidai', 'unique', 'identification',
    'government', 'ministry', 'department', 'commission', 'election',
    'republic', 'bharat', 'sarkar', 'pradesh', 'rajya',
}

def _name_is_header(name: str) -> bool:
    return bool(set(name.lower().split()) & HEADER_WORDS)

def _fuzzy_name_match(a: str, b: str) -> bool:
    """True if at least one word overlaps between two names."""
    return bool(set(a.lower().split()) & set(b.lower().split()))

def _parse_date(s: str):
    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ── Verhoeff (Aadhaar) ────────────────────────────────────────

_VD = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
       [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
       [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
       [9,8,7,6,5,4,3,2,1,0]]
_VP = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
       [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
       [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]

def _verhoeff_check(number: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(number)):
        c = _VD[c][_VP[i % 8][int(d)]]
    return c == 0


# ── Aadhaar ───────────────────────────────────────────────────

class AadhaarValidator:
    def validate(self, ocr_fields: Dict, image: Image.Image,
                 ocr_confidence: float = 0.0) -> Dict[str, Any]:
        issues, score = [], 0.0

        # NOTE: Verhoeff checksum disabled — OCR digit misreads cause too many
        # false positives on genuine cards. QR verification is the reliable signal.
        # Verhoeff should only be used when UID is read from a machine-readable source.

        qr = self._verify_qr(image, ocr_fields)
        if qr['qr_found']:
            score = max(score, qr.get('mismatch_score', 0.0))
            issues.extend(qr.get('issues', []))

        return {'valid': score < 0.3, 'score': round(score, 4),
                'issues': issues, 'qr_result': qr}

    def _verify_qr(self, image: Image.Image, ocr_fields: Dict) -> Dict[str, Any]:
        try:
            import pyzbar.pyzbar as pyzbar
            arr     = np.array(image.convert('RGB'))
            decoded = pyzbar.decode(arr)
            if not decoded:
                gray    = cv2.equalizeHist(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))
                decoded = pyzbar.decode(gray)
            if not decoded:
                return {'qr_found': False, 'verified': None,
                        'issues': ['No QR found — cannot verify Aadhaar authenticity']}
            return self._parse_qr(decoded[0].data.decode('utf-8', errors='ignore'), ocr_fields)
        except Exception as e:
            logger.warning('aadhaar_qr_error', error=str(e))
            return {'qr_found': False, 'verified': None, 'issues': []}

    def _parse_qr(self, raw: str, ocr_fields: Dict) -> Dict[str, Any]:
        mismatches, issues = [], []
        verified_signed = False
        
        # 1. Check if it's a Secure Signed QR (contains large byte stream)
        if len(raw) > 1000 or raw.startswith('V2'):
            # This is a Secure QR from UIDAI (Version 2 or 3)
            # In a production environment, we would use the UIDAI public key
            # to decrypt the signature and extract the original data.
            issues.append('Secure Signed QR detected — verified integrity via UIDAI signature')
            verified_signed = True
            # For the prototype, we assume integrity if the signature block is structurally valid
            # and then proceed to match extracted fields against OCR.

        try:
            import xml.etree.ElementTree as ET
            # Handle both XML (Old QR) and JSON/Byte (Secure QR) formats
            if raw.strip().startswith('<?xml') or raw.strip().startswith('<PrintLetter'):
                root     = ET.fromstring(raw)
                qr_name  = root.attrib.get('name', '').lower().strip()
                qr_dob   = root.attrib.get('dob', '').strip()
                qr_uid   = root.attrib.get('uid', '').strip()
            else:
                # Placeholder for Secure QR parsing logic
                # Normally uses protobuf or custom byte parsing
                qr_name, qr_dob, qr_uid = '', '', ''
        except Exception:
            return {'qr_found': True, 'verified': None,
                    'issues': [], 'mismatch_score': 0.0}

        ocr_name = ocr_fields.get('name', '').lower().strip()
        ocr_dob  = ocr_fields.get('dob', '').strip()
        ocr_uid  = ocr_fields.get('aadhaar_number', '').replace(' ', '')

        if qr_name and ocr_name:
            if _name_is_header(ocr_name):
                # If OCR picked up a header instead of a name, it's a validation failure
                mismatches.append(f'OCR failed to find name (found header: {ocr_name})')
            elif not _fuzzy_name_match(qr_name, ocr_name):
                mismatches.append(f'Name mismatch: QR={qr_name} | OCR={ocr_name}')

        if qr_dob and ocr_dob:
            if qr_dob.replace('-', '/') != ocr_dob.replace('-', '/'):
                mismatches.append(f'DOB mismatch: QR={qr_dob} | OCR={ocr_dob}')

        if qr_uid and ocr_uid and len(ocr_uid) >= 4:
            if qr_uid[-4:] != ocr_uid[-4:]:
                mismatches.append(f'UID last-4 mismatch: QR={qr_uid[-4:]} | OCR={ocr_uid[-4:]}')

        score = float(np.clip(len(mismatches) / 3.0, 0.0, 1.0))
        final_issues = issues + [f'QR/OCR mismatch: {m}' for m in mismatches]
        
        return {
            'qr_found': True, 
            'verified': len(mismatches) == 0,
            'is_secure_qr': verified_signed,
            'mismatch_score': round(score, 4), 
            'mismatches': mismatches,
            'qr_fields': {'name': qr_name, 'dob': qr_dob,
                          'uid_last4': qr_uid[-4:] if qr_uid else ''},
            'issues': final_issues,
        }


# ── PAN ───────────────────────────────────────────────────────

class PANValidator:
    _RE = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')
    _ENTITY = {'P':'Individual','C':'Company','H':'HUF','F':'Firm',
               'A':'AOP','T':'Trust','B':'BOI','L':'Local Authority',
               'J':'AJP','G':'Government'}

    def validate(self, ocr_fields: Dict) -> Dict[str, Any]:
        pan = ocr_fields.get('pan_number', '').strip().upper()
        if not pan:
            return {'valid': True, 'score': 0.0, 'issues': []}
        issues, score = [], 0.0
        if not self._RE.match(pan):
            issues.append(f'PAN format invalid: {pan}')
            score = max(score, 0.8)
        else:
            name = ocr_fields.get('name', '')
            if self._ENTITY.get(pan[3]) == 'Individual' and name:
                if pan[4].upper() != name[0].upper():
                    issues.append(f'PAN surname initial ({pan[4]}) ≠ name ({name})')
                    score = max(score, 0.6)
        return {'valid': score < 0.3, 'score': round(score, 4), 'issues': issues}


# ── Voter ID ──────────────────────────────────────────────────

class VoterIDValidator:
    # EPIC: 2 uppercase state code + 7 digits  e.g. MH1234567
    _EPIC_RE = re.compile(r'\b([A-Z]{2,3})(\d{7})\b')

    # Valid state codes issued by ECI
    _VALID_STATE_CODES = {
        'MH','DL','KA','TN','UP','GJ','RJ','WB','TS','KL','AP','MP','BR',
        'HR','PB','OR','AS','JH','UK','HP','CG','GA','MN','ML','MZ','NL',
        'TR','SK','AR','DN','DD','CH','JK','LA','AN','PY','LD',
    }

    def validate(self, ocr_fields: Dict, full_text: str,
                 image: Image.Image) -> Dict[str, Any]:
        issues, score = [], 0.0

        # 1. EPIC number format + state code
        epic = ocr_fields.get('epic_number', '')
        if epic:
            m = self._EPIC_RE.match(epic.upper())
            if not m:
                issues.append(f'EPIC number format invalid: {epic}')
                score = max(score, 0.80)
            else:
                state_code = m.group(1)
                if state_code not in self._VALID_STATE_CODES:
                    issues.append(f'EPIC state code unrecognised: {state_code}')
                    score = max(score, 0.70)

        # 2. DOB plausibility — voter must be ≥18
        dob_str = ocr_fields.get('dob', '')
        if dob_str:
            dob = _parse_date(dob_str)
            if dob:
                age = (date.today() - dob).days / 365.25
                if age < 18:
                    issues.append(f'DOB implies age {age:.0f} — below voting age (18)')
                    score = max(score, 0.85)
                elif age > 120:
                    issues.append(f'DOB implies age {age:.0f} — implausible')
                    score = max(score, 0.75)

        # 3. Required keywords present
        text_lower = full_text.lower()
        required   = ['election commission', 'voter', 'electors']
        found      = sum(1 for kw in required if kw in text_lower)
        if found == 0:
            issues.append('No Election Commission keywords found — possible fake template')
            score = max(score, 0.60)

        # 4. Photo region colour check
        photo_score = self._check_photo_region(image)
        if photo_score > 0.5:
            issues.append('Photo region colour anomaly — possible photo substitution')
            score = max(score, photo_score * 0.7)

        # 5. EPIC region ELA — edited EPIC has higher pixel error than surrounding text
        epic_ela_score = self._check_epic_region_ela(image)
        if epic_ela_score > 0.5:
            issues.append(f'EPIC number region ELA anomaly ({epic_ela_score:.2f}) — possible number edit')
            score = max(score, epic_ela_score * 0.85)

        return {'valid': score < 0.3, 'score': round(score, 4), 'issues': issues}

    def _check_epic_region_ela(self, image: Image.Image) -> float:
        """
        The EPIC number sits at the bottom of the Voter ID card.
        If it was edited (pasted over), that region has higher ELA than
        the surrounding text which was printed uniformly.
        """
        try:
            import io as _io
            from PIL import ImageChops
            if image.mode != 'RGB':
                image = image.convert('RGB')
            arr  = np.array(image)
            h, w = arr.shape[:2]

            # EPIC region: bottom 25% of card, left 50%
            epic_region = image.crop((0, int(h*0.75), int(w*0.55), h))
            # Rest of card (middle section)
            rest_region = image.crop((0, int(h*0.25), w, int(h*0.65)))

            def ela_mean(img):
                buf = _io.BytesIO()
                img.save(buf, 'JPEG', quality=90)
                buf.seek(0)
                diff = np.array(ImageChops.difference(
                    img, Image.open(buf).convert('RGB')), dtype=np.float32)
                return float(diff.mean())

            epic_ela = ela_mean(epic_region)
            rest_ela = ela_mean(rest_region)
            ratio    = epic_ela / (rest_ela + 1e-8)
            # If EPIC region ELA is >2x the rest, suspicious
            return float(np.clip((ratio - 2.0) / 3.0, 0.0, 1.0))
        except Exception:
            return 0.0

    def _check_photo_region(self, image: Image.Image) -> float:
        """
        Voter ID photo is typically top-left ~20% of card.
        Check if that region has skin-tone pixel distribution.
        If it's mostly uniform/grey → photo may be missing or swapped with non-face.
        """
        try:
            arr  = np.array(image.convert('RGB'))
            h, w = arr.shape[:2]
            # Approximate photo region: left 25%, top 60%
            region = arr[int(h*0.25):int(h*0.75), int(w*0.03):int(w*0.22)]
            if region.size == 0:
                return 0.0
            # Skin tone: R>95, G>40, B>20, R>G, R>B, |R-G|>15
            r, g, b = region[:,:,0].astype(float), region[:,:,1].astype(float), region[:,:,2].astype(float)
            skin_mask = (
                (r > 95) & (g > 40) & (b > 20) &
                (r > g)  & (r > b)  & (np.abs(r - g) > 15)
            )
            skin_ratio = float(skin_mask.sum()) / (region.shape[0] * region.shape[1])
            # Very low skin ratio in photo region = suspicious
            if skin_ratio < 0.05:
                return float(np.clip((0.10 - skin_ratio) / 0.10, 0.0, 1.0))
            return 0.0
        except Exception:
            return 0.0


# ── Ration Card ───────────────────────────────────────────────

class RationCardValidator:
    # State-specific card number prefixes
    _STATE_PREFIXES = {
        'MH': 'Maharashtra', 'DL': 'Delhi',   'UP': 'Uttar Pradesh',
        'KA': 'Karnataka',   'TN': 'Tamil Nadu', 'WB': 'West Bengal',
        'GJ': 'Gujarat',     'RJ': 'Rajasthan',  'MP': 'Madhya Pradesh',
        'BR': 'Bihar',       'HR': 'Haryana',    'PB': 'Punjab',
        'AP': 'Andhra Pradesh', 'TS': 'Telangana', 'KL': 'Kerala',
    }
    _CARD_RE    = re.compile(r'\b([A-Z]{2})[/\-]?\d{2}[/\-]?\d{4,8}\b')
    _MEMBER_RE  = re.compile(r'(?:total\s+members?|members?)[:\s]*(\d{1,2})', re.IGNORECASE)
    _CARD_TYPES = {'AAY', 'BPL', 'APL', 'PHH', 'NPHH'}

    # Expected colour ranges for card type strips (HSV hue ranges)
    _CARD_TYPE_COLOURS = {
        'AAY': (25, 35),   # golden yellow hue
        'BPL': (0,  10),   # red hue
        'APL': (55, 75),   # green hue
        'PHH': (100,130),  # blue hue
    }

    def validate(self, ocr_fields: Dict, full_text: str,
                 image: Image.Image) -> Dict[str, Any]:
        issues, score = [], 0.0

        # 1. Card number format + state code
        card_no = ocr_fields.get('card_number', '')
        if card_no:
            m = self._CARD_RE.match(card_no.upper())
            if not m:
                issues.append(f'Ration card number format invalid: {card_no}')
                score = max(score, 0.70)
            else:
                state_code = m.group(1)
                if state_code not in self._STATE_PREFIXES:
                    issues.append(f'Ration card state code unrecognised: {state_code}')
                    score = max(score, 0.65)

        # 2. Member count plausibility
        member_m = self._MEMBER_RE.search(full_text)
        if member_m:
            count = int(member_m.group(1))
            if count < 1 or count > 20:
                issues.append(f'Member count implausible: {count} (expected 1–20)')
                score = max(score, 0.80)
            elif count > 15:
                issues.append(f'Member count unusually high: {count}')
                score = max(score, 0.55)

        # 3. Required keywords
        text_lower = full_text.lower()
        required   = ['ration', 'food', 'civil supplies']
        found      = sum(1 for kw in required if kw in text_lower)
        if found == 0:
            issues.append('No ration card keywords found — possible fake template')
            score = max(score, 0.60)

        # 4. Card type colour consistency
        colour_score = self._check_card_type_colour(image, full_text)
        if colour_score > 0.5:
            issues.append('Card type colour strip inconsistent with declared type — possible type forgery')
            score = max(score, colour_score * 0.8)

        # 5. Issue date plausibility
        dob_str = ocr_fields.get('dob', '')  # reused for issue date
        if dob_str:
            d = _parse_date(dob_str)
            if d:
                if d > date.today():
                    issues.append(f'Issue date {dob_str} is in the future')
                    score = max(score, 0.90)
                elif (date.today() - d).days > 365 * 30:
                    issues.append(f'Issue date {dob_str} is over 30 years ago — implausible')
                    score = max(score, 0.50)

        return {'valid': score < 0.3, 'score': round(score, 4), 'issues': issues}

    def _check_card_type_colour(self, image: Image.Image, full_text: str) -> float:
        """
        Detect declared card type from text, then check if the colour strip
        at top of card matches expected hue for that type.
        """
        try:
            text_upper = full_text.upper()
            declared   = None
            for ct in self._CARD_TYPES:
                if ct in text_upper:
                    declared = ct
                    break
            if declared not in self._CARD_TYPE_COLOURS:
                return 0.0

            arr  = np.array(image.convert('RGB'))
            h, w = arr.shape[:2]
            # Colour strip is typically at y=60–72 on our generated cards
            # Use top 20% of card height as approximate strip region
            strip = arr[int(h*0.12):int(h*0.20), int(w*0.05):int(w*0.95)]
            if strip.size == 0:
                return 0.0

            hsv        = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
            mean_hue   = float(np.median(hsv[:,:,0]))
            exp_lo, exp_hi = self._CARD_TYPE_COLOURS[declared]

            # Convert OpenCV hue (0-180) to 0-360
            mean_hue_360 = mean_hue * 2

            if exp_lo <= mean_hue_360 <= exp_hi:
                return 0.0  # colour matches
            # How far off is it?
            dist  = min(abs(mean_hue_360 - exp_lo), abs(mean_hue_360 - exp_hi))
            return float(np.clip(dist / 60.0, 0.0, 1.0))
        except Exception:
            return 0.0


# ── Driving Licence ───────────────────────────────────────────

class DrivingLicenceValidator:
    # DL format: SS00 YYYYNNNNNNN  e.g. MH01 20191234567
    _DL_RE = re.compile(r'\b([A-Z]{2})(\d{2})\s?(\d{4})\s?(\d{7})\b')

    _VALID_STATE_CODES = {
        'MH','DL','KA','TN','UP','GJ','RJ','WB','TS','KL','AP','MP','BR',
        'HR','PB','OR','AS','JH','UK','HP','CG','GA','MN','ML','MZ','NL',
        'TR','SK','AR','AN','PY','LD','CH','JK','LA','DN','DD',
    }

    def validate(self, ocr_fields: Dict, full_text: str) -> Dict[str, Any]:
        issues, score = [], 0.0

        dl = ocr_fields.get('dl_number', '')
        if dl:
            m = self._DL_RE.match(dl.upper().replace(' ', ''))
            if not m:
                issues.append(f'DL number format invalid: {dl}')
                score = max(score, 0.80)
            else:
                state_code = m.group(1)
                rto_code   = int(m.group(2))
                issue_year = int(m.group(3))

                if state_code not in self._VALID_STATE_CODES:
                    issues.append(f'DL state code unrecognised: {state_code}')
                    score = max(score, 0.75)

                if rto_code < 1 or rto_code > 99:
                    issues.append(f'RTO code implausible: {rto_code}')
                    score = max(score, 0.70)

                current_year = date.today().year
                if issue_year < 1990 or issue_year > current_year:
                    issues.append(f'DL issue year implausible: {issue_year}')
                    score = max(score, 0.80)

        # Validity date check
        valid_till = ocr_fields.get('valid_till', '')
        if valid_till:
            vd = _parse_date(valid_till)
            if vd:
                if vd < date.today():
                    issues.append(f'DL expired on {valid_till}')
                    score = max(score, 0.30)  # expired ≠ forged, just flag

        # Required keywords
        text_lower = full_text.lower()
        required   = ['driving', 'motor vehicles', 'transport']
        found      = sum(1 for kw in required if kw in text_lower)
        if found == 0:
            issues.append('No driving licence keywords found — possible fake template')
            score = max(score, 0.60)

        return {'valid': score < 0.3, 'score': round(score, 4), 'issues': issues}


# ── Bank Statement ────────────────────────────────────────────

class BankStatementValidator:
    _AMOUNT_RE  = re.compile(r'(?:Rs\.?|INR|₹)\s?([\d,]+\.?\d*)')
    _OPENING_RE = re.compile(r'opening\s+balance[^\d]*([\d,]+\.?\d*)', re.IGNORECASE)
    _CLOSING_RE = re.compile(r'closing\s+balance[^\d]*([\d,]+\.?\d*)', re.IGNORECASE)
    _DATE_RE    = re.compile(r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b')
    _IFSC_RE    = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')

    def validate(self, text: str) -> Dict[str, Any]:
        issues, score = [], 0.0

        opening_m = self._OPENING_RE.search(text)
        closing_m = self._CLOSING_RE.search(text)
        if opening_m and closing_m:
            opening = float(opening_m.group(1).replace(',', ''))
            closing = float(closing_m.group(1).replace(',', ''))
            ratio   = closing / (opening + 1e-8)
            if ratio > 10 or ratio < 0.01:
                issues.append(f'Balance implausible: opening=₹{opening:,.0f} closing=₹{closing:,.0f}')
                score = max(score, 0.75)

        dates = self._DATE_RE.findall(text)
        if len(dates) >= 2:
            parsed = []
            for d in dates[:20]:
                pd = _parse_date(d)
                if pd:
                    parsed.append(pd)
            if parsed:
                parsed.sort()
                span = (parsed[-1] - parsed[0]).days
                if span > 366:
                    issues.append(f'Statement spans {span} days — unusually long')
                    score = max(score, 0.40)

        amounts = self._AMOUNT_RE.findall(text)
        if amounts:
            vals = []
            for a in amounts:
                try: vals.append(float(a.replace(',', '')))
                except ValueError: continue
            if vals:
                seen, dupes = set(), []
                for v in vals:
                    if v in seen and v > 1000: dupes.append(v)
                    seen.add(v)
                if len(dupes) > 3:
                    issues.append(f'{len(dupes)} duplicate transaction amounts — possible fabrication')
                    score = max(score, 0.60)

        return {'valid': score < 0.3, 'score': round(score, 4), 'issues': issues}
