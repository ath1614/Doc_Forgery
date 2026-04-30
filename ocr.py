import io
import numpy as np
from PIL import Image
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()


class OCRExtractor:
    def __init__(self):
        self._paddle = None
        self._easy   = None

    def _get_paddle(self):
        if self._paddle is None:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,   # PaddlePaddle 2.6 needs cuDNN — use CPU (still fast)
                show_log=False,
            )
            logger.info('paddleocr_loaded')
        return self._paddle

    def _get_easy(self):
        if self._easy is None:
            import easyocr
            self._easy = easyocr.Reader(['hi', 'en'], gpu=True, verbose=False)
            logger.info('easyocr_loaded')
        return self._easy

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        img_array = np.array(image.convert('RGB'))

        try:
            result = self._get_paddle().ocr(img_array, cls=True)
            blocks, full_text = [], ''
            for line in (result[0] or []):
                bbox, (text, conf) = line
                if conf > 0.4:
                    blocks.append({
                        'text':       text.strip(),
                        'confidence': round(conf, 3),
                        'bbox':       [[int(p[0]), int(p[1])] for p in bbox],
                    })
                    full_text += ' ' + text.strip()
            if len(blocks) >= 2:
                return {'full_text': full_text.strip(), 'text_blocks': blocks,
                        'success': True, 'engine': 'paddle'}
        except Exception as e:
            logger.warning('paddle_ocr_failed', error=str(e))

        try:
            results = self._get_easy().readtext(img_array, detail=1, paragraph=False)
            blocks, full_text = [], ''
            for (bbox, text, conf) in results:
                if conf > 0.3:
                    blocks.append({
                        'text':       text.strip(),
                        'confidence': round(conf, 3),
                        'bbox':       [[int(p[0]), int(p[1])] for p in bbox],
                    })
                    full_text += ' ' + text.strip()
            return {'full_text': full_text.strip(), 'text_blocks': blocks,
                    'success': True, 'engine': 'easyocr'}
        except Exception as e:
            logger.error('easyocr_failed', error=str(e))
            return {'full_text': '', 'text_blocks': [], 'success': False, 'error': str(e)}
