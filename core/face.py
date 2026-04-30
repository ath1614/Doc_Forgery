import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
import structlog

logger = structlog.get_logger()


class FaceRegionELA:
    def __init__(self):
        self._mtcnn   = None
        self._cascade = None
        self._insight = None

    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                import torch
                self._mtcnn = MTCNN(
                    keep_all=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    min_face_size=20,
                    thresholds=[0.5, 0.6, 0.6],
                    post_process=False
                )
                logger.info('mtcnn_face_region_loaded')
            except Exception as e:
                logger.warning('mtcnn_load_failed', error=str(e))
        return self._mtcnn

    def _get_cascade(self):
        if self._cascade is None:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        return self._cascade

    def _detect_faces(self, image: Image.Image) -> List:
        """Try MTCNN first (best), then InsightFace, then Haar cascade."""
        arr = np.array(image.convert('RGB'))
        h, w = arr.shape[:2]

        # 1. MTCNN (best for documents — handles small faces)
        mtcnn = self._get_mtcnn()
        if mtcnn is not None:
            try:
                boxes, probs = mtcnn.detect(image)
                if boxes is not None and len(boxes) > 0:
                    result = []
                    for box, prob in zip(boxes, probs):
                        if prob < 0.7: continue
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        result.append((x1, y1, x2-x1, y2-y1))
                    if result:
                        return result
            except Exception as e:
                logger.warning('mtcnn_detect_failed', error=str(e))

        # 2. Haar cascade fallback
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = self._get_cascade().detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
        return list(faces) if len(faces) > 0 else []

    def analyze(self, image: Image.Image, ela_map: Optional[np.ndarray]) -> Dict[str, Any]:
        if ela_map is None:
            return {'face_found': False, 'photo_swap_score': 0.0, 'issues': []}

        faces = self._detect_faces(image)
        if not faces:
            return {'face_found': False, 'photo_swap_score': 0.0, 'issues': []}

        x, y, w, h   = faces[0]
        face_ela     = float(ela_map[y:y+h, x:x+w].mean())
        doc_ela_mean = float(ela_map.mean())
        ratio        = face_ela / (doc_ela_mean + 1e-8)
        score        = float(np.clip((ratio - 2.0) / 3.0, 0.0, 1.0))

        return {
            'face_found':       True,
            'face_region':      {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'face_ela':         round(face_ela, 3),
            'document_ela':     round(doc_ela_mean, 3),
            'ela_ratio':        round(ratio, 3),
            'photo_swap_score': round(score, 4),
            'issues':           [f'Face ELA ({face_ela:.1f}) >> doc ELA ({doc_ela_mean:.1f}) — photo substitution detected']
                                if score > 0.3 else [],
        }


class FaceMatcher:
    def __init__(self):
        self._app   = None
        self._mtcnn = None

    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN, InceptionResnetV1
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._mtcnn   = MTCNN(keep_all=False, device=device, min_face_size=20)
                self._resnet  = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                self._device  = device
                logger.info('facenet_matcher_loaded')
            except Exception as e:
                logger.warning('facenet_load_failed', error=str(e))
        return self._mtcnn

    def extract_embedding(self, image: Image.Image):
        mtcnn = self._get_mtcnn()
        if mtcnn is None:
            return None
        try:
            import torch
            face_tensor = mtcnn(image)
            if face_tensor is None:
                return None
            with torch.no_grad():
                emb = self._resnet(face_tensor.unsqueeze(0).to(self._device))
            return emb.squeeze().cpu().numpy()
        except Exception as e:
            logger.warning('embedding_error', error=str(e))
            return None

    def match(self, doc_image: Image.Image, selfie_image: Image.Image) -> Dict[str, Any]:
        try:
            emb_doc    = self.extract_embedding(doc_image)
            emb_selfie = self.extract_embedding(selfie_image)

            if emb_doc is None:
                return {'matched': None, 'score': 0.0, 'issues': ['No face found in document']}
            if emb_selfie is None:
                return {'matched': None, 'score': 0.0, 'issues': ['No face found in selfie']}

            cosine  = float(np.dot(emb_doc, emb_selfie) /
                            (np.linalg.norm(emb_doc) * np.linalg.norm(emb_selfie) + 1e-8))
            matched = cosine > 0.35

            return {
                'matched': matched,
                'score':   round(cosine, 4),
                'issues':  [] if matched else
                           [f'Face mismatch: similarity={cosine:.2f} — person in document does not match selfie'],
            }
        except Exception as e:
            logger.warning('face_match_error', error=str(e))
            return {'matched': None, 'score': 0.0, 'issues': [str(e)]}
