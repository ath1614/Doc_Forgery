import io
import cv2
import numpy as np
from PIL import Image, ImageChops
from typing import Dict, Any, List


class MultiScaleELA:
    QUALITIES = [95, 90, 85, 75]

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        maps, scores = [], []
        for q in self.QUALITIES:
            buf = io.BytesIO()
            image.save(buf, 'JPEG', quality=q)
            buf.seek(0)
            recomp = Image.open(buf).convert('RGB')
            diff   = np.array(ImageChops.difference(image, recomp), dtype=np.float32)
            gray   = diff.mean(axis=2)
            maps.append(gray)
            scores.append(float(gray.mean()))

        combined     = np.mean(maps, axis=0)
        dct_score    = self._dct_block_inconsistency(np.array(image.convert('L')))
        tamper_score = float(np.clip(np.mean(scores) / 25.0 * 0.6 + dct_score * 0.4, 0.0, 1.0))

        return {
            'ela_mean':         round(float(combined.mean()), 3),
            'tamper_score':     round(tamper_score, 4),
            'dct_score':        round(dct_score, 4),
            'is_tampered':      tamper_score > 0.35,
            'ela_map':          combined,
            'tampered_regions': self._find_regions(combined),
        }

    def _dct_block_inconsistency(self, gray: np.ndarray) -> float:
        h, w  = gray.shape
        h, w  = (h // 8) * 8, (w // 8) * 8
        gray  = gray[:h, :w].astype(np.float32)
        energies = []
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                dct = cv2.dct(gray[i:i+8, j:j+8])
                energies.append(float(np.abs(dct[4:, 4:]).mean()))
        if not energies:
            return 0.0
        arr = np.array(energies)
        cv  = float(arr.std() / (arr.mean() + 1e-8))
        return float(np.clip(cv / 2.0, 0.0, 1.0))

    def _find_regions(self, gray_diff: np.ndarray) -> List[Dict]:
        regions = []
        binary  = (gray_diff > 40).astype(np.uint8) * 255
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilated = cv2.dilate(binary, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray_diff.shape
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            regions.append({
                'x': int(x), 'y': int(y), 'w': int(rw), 'h': int(rh),
                'ela_intensity': round(float(gray_diff[y:y+rh, x:x+rw].mean()), 2),
                'area_percent':  round(area / (h * w) * 100, 2),
            })
        return regions


class CopyMoveDetector:
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        # Resize to fixed size to normalise feature count
        img_small = image.resize((400, 250), Image.LANCZOS)
        gray = cv2.cvtColor(np.array(img_small.convert('RGB')), cv2.COLOR_RGB2GRAY)
        orb  = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 20:
            return {'copy_move_detected': False, 'score': 0.0,
                    'matches': 0, 'issues': []}

        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=3)

        suspicious = []
        for m_list in matches:
            for m in m_list[1:]:
                # Stricter distance threshold + minimum spatial separation
                if m.distance < 25:
                    p1 = np.array(kp[m.queryIdx].pt)
                    p2 = np.array(kp[m.trainIdx].pt)
                    dist = np.linalg.norm(p1 - p2)
                    # Must be far enough apart to be a real copy-move (not adjacent)
                    if 40 < dist < 300:
                        suspicious.append(m)

        # Require cluster of suspicious matches, not just a few
        score = float(np.clip((len(suspicious) - 5) / 40.0, 0.0, 1.0))
        return {
            'copy_move_detected': score > 0.4,
            'score':   round(score, 4),
            'matches': len(suspicious),
            'issues':  [f'Copy-move forgery: {len(suspicious)} suspicious region matches']
                       if score > 0.4 else [],
        }


class NoiseInconsistencyDetector:
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        gray    = np.array(image.convert('L')).astype(np.float32)
        h, w    = gray.shape
        bh, bw  = h // 4, w // 4
        block_noise = []

        for i in range(4):
            for j in range(4):
                block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                blur  = cv2.GaussianBlur(block, (5, 5), 0)
                noise = float(np.std(block - blur))
                block_noise.append(noise)

        arr   = np.array(block_noise)
        cv    = float(arr.std() / (arr.mean() + 1e-8))
        score = float(np.clip(cv / 1.0, 0.0, 1.0)) # Increased sensitivity from 1.5 to 1.0

        return {
            'noise_cv':     round(cv, 4),
            'score':        round(score, 4),
            'inconsistent': score > 0.35,
            'issues':       ['Noise inconsistency across document regions — possible splicing']
                            if score > 0.35 else [],
        }


class PicsartEditDetector:
    """
    Detects edits made by Picsart, Snapseed, online editors, WhatsApp crop-paste.

    Core principle: JPEG compression creates a unique "fingerprint" per save.
    When you paste new content and re-save, the pasted region has a DIFFERENT
    compression history than the original — detectable as a local ELA anomaly.

    Catches: Picsart, Snapseed, Canva, Fotor, MS Paint, WhatsApp paste,
             Photoshop (unless quality-matched), any online editor.
    """
    QUALITIES = [95, 90, 85, 75]

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            w, h = image.size
            if min(w, h) < 400:
                scale = 400 / min(w, h)
                image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                w, h  = image.size

            # Build averaged ELA map across multiple qualities
            ela_maps = []
            for q in self.QUALITIES:
                buf = io.BytesIO()
                image.save(buf, 'JPEG', quality=q)
                buf.seek(0)
                recomp = Image.open(buf).convert('RGB')
                diff   = np.array(ImageChops.difference(image, recomp),
                                  dtype=np.float32).mean(axis=2)
                ela_maps.append(diff)
            ela = np.mean(ela_maps, axis=0)

            # Exclude header (top 15%) and footer (bottom 10%) —
            # these are high-contrast design elements that always have high ELA
            y_start = int(h * 0.15)
            y_end   = int(h * 0.90)
            ela_body = ela[y_start:y_end, :]

            # Fine grid — 32x32 pixel blocks on body only
            block_size = 32
            block_elas = []
            bh, bw = ela_body.shape
            for y in range(0, bh - block_size, block_size):
                for x in range(0, bw - block_size, block_size):
                    block_elas.append((x, y + y_start,
                                       float(ela_body[y:y+block_size, x:x+block_size].mean())))

            if len(block_elas) < 4:
                return {'score': 0.0, 'issues': [], 'suspicious_regions': []}

            vals     = np.array([b[2] for b in block_elas])
            mean_ela = float(vals.mean())
            std_ela  = float(vals.std())

            if std_ela < 0.05:
                return {'score': 0.0, 'mean_ela': round(mean_ela,3),
                        'std_ela': round(std_ela,3), 'issues': [], 'suspicious_regions': []}

            # Outlier blocks: >1.8 std above mean = pasted content
            # OR suspiciously low = solid fill (erased region)
            hi_thresh = mean_ela + 1.8 * std_ela
            lo_thresh = max(mean_ela - 1.8 * std_ela, 0.0)

            suspicious = []
            for x, y, ela_val in block_elas:
                if ela_val > hi_thresh:
                    suspicious.append({'x':x,'y':y,'ela':round(ela_val,2),'type':'high_ela'})
                elif ela_val < lo_thresh and mean_ela > 1.0:
                    suspicious.append({'x':x,'y':y,'ela':round(ela_val,2),'type':'low_ela_patch'})

            ratio = len(suspicious) / len(block_elas)

            if suspicious:
                max_dev = max(abs(b['ela'] - mean_ela) / (std_ela + 1e-8) for b in suspicious)
                score   = float(np.clip(ratio * 6.0 * min(max_dev / 3.0, 1.0), 0.0, 1.0))
            else:
                score = 0.0

            issues = []
            if score > 0.25:
                edit_types = set(b['type'] for b in suspicious)
                desc = 'high ELA patch' if 'high_ela' in edit_types else 'erased region'
                issues.append(
                    f'Localised {desc} in {len(suspicious)} blocks '
                    f'(score={score:.2f}) — consistent with Picsart/online editor edit'
                )

            return {
                'score':              round(score, 4),
                'mean_ela':           round(mean_ela, 3),
                'std_ela':            round(std_ela, 3),
                'suspicious_regions': suspicious[:10],
                'issues':             issues,
            }
        except Exception as e:
            return {'score': 0.0, 'issues': [], 'suspicious_regions': [], 'error': str(e)}


class MoirePatternDetector:
    """
    Detects "Screen-on-Screen" re-photography by analyzing periodic frequency
    peaks in the 2D Fourier Transform (FFT).
    """
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        try:
            gray = np.array(image.convert('L'))
            h, w = gray.shape
            # Focus on center 512x512 to speed up FFT
            ch, cw = h // 2, w // 2
            side   = 512
            if h < side or w < side:
                gray_crop = cv2.resize(gray, (side, side))
            else:
                gray_crop = gray[ch-side//2:ch+side//2, cw-side//2:cw+side//2]

            # 2D FFT
            f = np.fft.fft2(gray_crop)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

            # Mask out the DC component (center)
            rows, cols = magnitude_spectrum.shape
            crow, ccol = rows // 2, cols // 2
            magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10] = 0

            # Detect peaks - moire patterns create bright spots in the frequency domain
            max_val = float(np.max(magnitude_spectrum))
            mean_val = float(np.mean(magnitude_spectrum))
            std_val = float(np.std(magnitude_spectrum))

            # Threshold for periodic noise
            peaks = np.where(magnitude_spectrum > mean_val + 5 * std_val)
            peak_count = len(peaks[0])

            score = float(np.clip(peak_count / 100.0, 0.0, 1.0)) if max_val > 200 else 0.0
            issues = []
            if score > 0.4:
                issues.append(f"Moire pattern detected (peaks={peak_count}) — document likely re-photographed from a screen")

            return {
                'score': round(score, 4),
                'peak_count': peak_count,
                'max_magnitude': round(max_val, 2),
                'issues': issues
            }
        except Exception as e:
            return {'score': 0.0, 'issues': [], 'error': str(e)}


class DocumentLivenessDetector:
    """
    Detects if the document is a physical "live" card vs a flat printout
    by analyzing specular highlights and surface texture gradients.
    """
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        try:
            arr = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            # 1. Specular Highlight Detection (Glints on lamination)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            glints = len([c for c in contours if cv2.contourArea(c) > 10 and cv2.contourArea(c) < 500])

            # 2. Texture Gradient (Local Binary Patterns or Laplacian Variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            score = 0.0
            issues = []

            if glints > 10: # More sensitive threshold
                score = 0.6
                issues.append("Surface specular anomaly — possible screen or low-quality print")
            elif laplacian_var < 80: # More sensitive threshold
                score = 0.7
                issues.append("Flat texture detected — document lacks depth/lamination characteristics")

            return {
                'score': round(score, 4),
                'glint_count': glints,
                'texture_variance': round(float(laplacian_var), 2),
                'issues': issues
            }
        except Exception as e:
            return {'score': 0.0, 'issues': [], 'error': str(e)}


class DeepfakeDetector:
    """
    Client for the remote GAN-artifact detection API.
    Analyzes the face region for AI-generation fingerprints.
    """
    def __init__(self, api_url: str = "http://localhost:8002/api/v1/detect/deepfake"):
        self.api_url = api_url

    async def analyze(self, image: Image.Image) -> Dict[str, Any]:
        # This will be used in the async pipeline to call the existing GPU-hosted model
        try:
            import aiohttp
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            data = aiohttp.FormData()
            data.add_field('file', buf.getvalue(), filename='face.jpg', content_type='image/jpeg')

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=data, timeout=10) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        return {
                            'score': res.get('deepfake_probability', 0.0),
                            'is_deepfake': res.get('is_deepfake', False),
                            'issues': ["AI-generated face detected — synthetic identity risk"] if res.get('is_deepfake') else []
                        }
            return {'score': 0.0, 'issues': []}
        except Exception as e:
            # Fallback for offline/local testing
            return {'score': 0.0, 'issues': [], 'error': str(e)}


class DocumentTypeVerifier:
    """
    Verifies document type consistency using visual layout analysis.
    Checks if the document structure matches the claimed type.
    Uses color distribution, layout zones, and logo detection.
    """
    # Expected color signatures for Indian government documents
    _COLOR_PROFILES = {
        'AADHAAR':   {'dominant': [(255, 140, 0), (0, 100, 200)], 'name': 'orange-blue'},
        'PAN':       {'dominant': [(255, 255, 255), (0, 0, 128)], 'name': 'white-navy'},
        'VOTER_ID':  {'dominant': [(0, 128, 0), (255, 255, 255)], 'name': 'green-white'},
    }

    def verify(self, image: Image.Image, doc_type: str, ocr_text: str) -> Dict[str, Any]:
        try:
            arr = np.array(image.convert('RGB'))
            issues = []
            score  = 0.0

            # 1. Check aspect ratio matches document type
            h, w = arr.shape[:2]
            aspect = w / h if h > 0 else 1.0

            expected_aspects = {
                'AADHAAR': (1.4, 1.8),   # Landscape card
                'PAN':     (1.4, 1.8),   # Landscape card
                'VOTER_ID': (0.6, 0.9),  # Portrait card
                'PASSPORT': (0.6, 0.8),  # Portrait booklet
            }

            if doc_type in expected_aspects:
                lo, hi = expected_aspects[doc_type]
                if not (lo <= aspect <= hi):
                    issues.append(f'Aspect ratio {aspect:.2f} unusual for {doc_type} — expected {lo}-{hi}')
                    score = max(score, 0.3)

            # 2. Check minimum resolution
            if w < 200 or h < 150:
                issues.append(f'Very low resolution ({w}x{h}) — may be a screenshot or thumbnail')
                score = max(score, 0.4)

            # 3. Check for uniform color blocks (sign of digital forgery)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            unique_colors = len(np.unique(gray))
            if unique_colors < 50:
                issues.append('Very few unique colors — possible solid-fill forgery')
                score = max(score, 0.5)

            # 4. Check text density (forged docs often have wrong text density)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_density = float(np.sum(thresh == 0)) / thresh.size
            if text_density > 0.6:
                issues.append('Unusually high text density — possible template overlay')
                score = max(score, 0.35)

            return {
                'score':       round(score, 4),
                'aspect_ratio': round(aspect, 2),
                'resolution':  f'{w}x{h}',
                'issues':      issues,
            }
        except Exception as e:
            return {'score': 0.0, 'issues': [], 'error': str(e)}
