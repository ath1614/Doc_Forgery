import numpy as np
from typing import Dict, Any


# Per-document-type weight profiles.
# Documents WITH QR (Aadhaar): QR mismatch is the dominant signal.
# Documents WITHOUT QR (Voter ID, Ration Card, DL): redistribute to
# validator + ELA + face_mismatch + copy_move.

_PROFILES: Dict[str, Dict[str, float]] = {
    'AADHAAR': {
        'qr_mismatch':   0.45,
        'validator':     0.25,
        'moire':         0.30,
        'liveness':      0.25,
        'deepfake':      0.25,
        'ela':           0.20,
        'face_mismatch': 0.20,
        'picsart':       0.15,
        'copy_move':     0.10,
        'noise':         0.10,
        'font':          0.05,
        'velocity':      0.05,
    },
    'PAN': {
        'validator':     0.20,
        'moire':         0.15,
        'liveness':      0.10,
        'deepfake':      0.25,
        'ela':           0.25,
        'picsart':       0.20,
        'face_mismatch': 0.20,
        'copy_move':     0.15,
        'noise':         0.10,
        'font':          0.10,
        'velocity':      0.01,
    },
    'VOTER_ID': {
        'validator':     0.40,
        'moire':         0.30,
        'liveness':      0.25,
        'deepfake':      0.25,
        'face_mismatch': 0.25,
        'picsart':       0.20,
        'ela':           0.20,
        'copy_move':     0.15,
        'noise':         0.15,
        'font':          0.10,
        'velocity':      0.05,
    },
    'RATION_CARD': {
        'validator':     0.40,
        'moire':         0.30,
        'liveness':      0.25,
        'picsart':       0.25,
        'ela':           0.25,
        'copy_move':     0.20,
        'noise':         0.20,
        'face_mismatch': 0.20,
        'font':          0.15,
        'velocity':      0.05,
        'deepfake':      0.05,
    },
}

# Fallback for unknown doc types
_DEFAULT_PROFILE: Dict[str, float] = {
    'validator':     0.35,
    'moire':         0.30,
    'liveness':      0.25,
    'deepfake':      0.25,
    'picsart':       0.20,
    'ela':           0.20,
    'face_mismatch': 0.20,
    'copy_move':     0.15,
    'noise':         0.15,
    'font':          0.10,
    'velocity':      0.05,
}

_FORGERY_TYPE_MAP = {
    'qr_mismatch':   'QR_DATA_MISMATCH',
    'face_mismatch': 'PHOTO_SUBSTITUTION',
    'picsart':       'ONLINE_EDITOR_TAMPERING',
    'ela':           'PIXEL_TAMPERING',
    'copy_move':     'COPY_MOVE_FORGERY',
    'noise':         'SPLICING_DETECTED',
    'validator':     'FIELD_VALIDATION_FAILURE',
    'font':          'TEXT_REPLACEMENT',
    'velocity':      'DOCUMENT_REUSE_ATTACK',
    'moire':         'SCREEN_REPHOTO_DETECTED',
    'liveness':      'FLAT_PRINTOUT_DETECTED',
    'deepfake':      'SYNTHETIC_IDENTITY_AI',
}

# Decision thresholds — tuned for higher sensitivity
_THRESHOLDS = {'APPROVED': 0.15, 'REVIEW': 0.42}


class RiskScoringEngine:
    def score(self, signals: Dict[str, float],
              doc_type: str = 'UNKNOWN') -> Dict[str, Any]:

        weights  = _PROFILES.get(doc_type, _DEFAULT_PROFILE)
        weighted = {k: signals.get(k, 0.0) * w for k, w in weights.items()}
        total    = float(np.clip(sum(weighted.values()), 0.0, 1.0))

        if total < _THRESHOLDS['APPROVED']:
            decision = 'APPROVED'
        elif total < _THRESHOLDS['REVIEW']:
            decision = 'REVIEW'
        else:
            decision = 'REJECTED'

        top = sorted(weighted.items(), key=lambda x: -x[1])
        forgery_type = 'GENUINE'
        for sig, val in top:
            if val > 0.04:
                forgery_type = _FORGERY_TYPE_MAP.get(sig, 'UNKNOWN_FORGERY')
                break

        return {
            'forgery_probability': round(total, 4),
            'decision':            decision,
            'risk_level':          ('CRITICAL' if total >= 0.85 else
                                    'HIGH'     if total >= 0.70 else
                                    'MEDIUM'   if total >= 0.50 else 'LOW'),
            'forgery_type':        forgery_type,
            'signal_breakdown':    {k: round(v, 4) for k, v in weighted.items()},
            'top_signals':         [(k, round(v, 4)) for k, v in top[:3] if v > 0],
            'weight_profile':      doc_type,
        }
