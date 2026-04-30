"""
KAVACH — Synthetic Aadhaar Generator
Generates realistic Aadhaar card images for testing document fraud detection.

Produces:
  - GENUINE  : valid number + QR matches OCR fields
  - FORGED   : various tampering types with ground truth labels

Usage:
  python generate_aadhaar.py --count 20 --out /home/tech/kavach_document/data/samples
"""

import os
import re
import io
import json
import random
import argparse
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, timedelta
from typing import Tuple, Dict, Any

import qrcode
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ── Font paths ────────────────────────────────────────────────
FONT_EN_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_EN_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_HI_REG  = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
FONT_HI_BOLD = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf'

# ── Aadhaar colours ───────────────────────────────────────────
BG_WHITE     = (255, 255, 255)
UIDAI_BLUE   = (0,  83, 156)
UIDAI_ORANGE = (255, 102, 0)
TEXT_DARK    = (30,  30,  30)
TEXT_GREY    = (100, 100, 100)
BORDER_BLUE  = (0,  83, 156)

# ── Indian name data ──────────────────────────────────────────
FIRST_NAMES_M  = ['Rahul','Amit','Suresh','Rajesh','Vikram','Arun','Deepak','Sanjay',
                   'Manoj','Pradeep','Ravi','Ajay','Vijay','Nitin','Rohit','Sachin',
                   'Ankit','Gaurav','Harsh','Karan']
FIRST_NAMES_F  = ['Priya','Sunita','Kavita','Rekha','Anita','Pooja','Neha','Divya',
                   'Meena','Geeta','Sonal','Ritu','Nisha','Swati','Anjali','Shweta',
                   'Pallavi','Sneha','Komal','Aarti']
LAST_NAMES     = ['Sharma','Verma','Singh','Kumar','Gupta','Patel','Joshi','Mishra',
                   'Yadav','Tiwari','Pandey','Chauhan','Rao','Nair','Reddy','Iyer',
                   'Mehta','Shah','Desai','Pillai']
STATES         = ['Maharashtra','Delhi','Karnataka','Tamil Nadu','Uttar Pradesh',
                   'Gujarat','Rajasthan','West Bengal','Telangana','Kerala',
                   'Madhya Pradesh','Bihar','Punjab','Haryana','Odisha']
CITIES         = ['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Ahmedabad',
                   'Pune','Kolkata','Jaipur','Lucknow','Surat','Kanpur','Nagpur','Indore']
STREETS        = ['MG Road','Gandhi Nagar','Nehru Street','Patel Colony','Shivaji Nagar',
                   'Laxmi Nagar','Rajiv Gandhi Road','Ambedkar Colony','Subhash Nagar']

# ── Verhoeff tables ───────────────────────────────────────────
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
      [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
      [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
      [9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
      [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
      [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
_INV = [0,4,3,2,1,5,6,7,8,9]

def _verhoeff_generate(number: str) -> str:
    c = 0
    arr = [int(d) for d in reversed('0' + number)]
    for i, d in enumerate(arr):
        c = _D[c][_P[i % 8][d]]
    return str(_INV[c])

def make_aadhaar_number() -> str:
    base = ''.join([str(random.randint(0,9)) for _ in range(11)])
    while base[0] == '0':
        base = str(random.randint(1,9)) + base[1:]
    check = _verhoeff_generate(base)
    full  = base + check
    return f'{full[:4]} {full[4:8]} {full[8:]}'

def make_dob(min_age=18, max_age=75) -> str:
    days = random.randint(min_age * 365, max_age * 365)
    d    = date.today() - timedelta(days=days)
    return d.strftime('%d/%m/%Y')

def make_address() -> Tuple[str, str, str]:
    house  = f'{random.randint(1,999)}, {random.choice(STREETS)}'
    city   = random.choice(CITIES)
    state  = random.choice(STATES)
    return house, city, state

def make_person(gender: str = None) -> Dict[str, Any]:
    if gender is None:
        gender = random.choice(['Male', 'Female'])
    first = random.choice(FIRST_NAMES_M if gender == 'Male' else FIRST_NAMES_F)
    last  = random.choice(LAST_NAMES)
    house, city, state = make_address()
    return {
        'name':    f'{first} {last}',
        'gender':  gender,
        'dob':     make_dob(),
        'uid':     make_aadhaar_number(),
        'address': f'{house}, {city}, {state}',
        'city':    city,
        'state':   state,
        'pincode': str(random.randint(100000, 999999)),
    }

def make_qr_xml(person: Dict) -> str:
    uid_clean = person['uid'].replace(' ', '')
    root = ET.Element('PrintLetterBarcodeData')
    root.set('uid',    uid_clean[-4:])   # masked
    root.set('name',   person['name'])
    root.set('gender', person['gender'][0])
    root.set('dob',    person['dob'].replace('/', '-'))
    root.set('co',     f"S/O: {random.choice(LAST_NAMES)}")
    root.set('house',  person['address'].split(',')[0])
    root.set('loc',    person['city'])
    root.set('vtc',    person['city'])
    root.set('dist',   person['city'])
    root.set('state',  person['state'])
    root.set('pc',     person['pincode'])
    return ET.tostring(root, encoding='unicode')

def make_qr_image(xml_data: str, size: int = 120) -> Image.Image:
    qr = qrcode.QRCode(version=4, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=3, border=2)
    qr.add_data(xml_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    return img.resize((size, size), Image.LANCZOS)

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def make_face_photo(gender: str, size: Tuple[int,int] = (90, 110)) -> Image.Image:
    w, h   = size
    img    = Image.new('RGB', (w, h), (220, 180, 140))
    draw   = ImageDraw.Draw(img)
    # Simple placeholder face silhouette
    cx, cy = w // 2, h // 2
    skin   = (210, 170, 130) if gender == 'Male' else (225, 185, 150)
    draw.ellipse([cx-28, cy-38, cx+28, cy+18], fill=skin)
    draw.rectangle([cx-22, cy+10, cx+22, cy+45], fill=skin)
    hair = (60, 40, 20) if random.random() > 0.3 else (20, 20, 20)
    draw.ellipse([cx-28, cy-50, cx+28, cy-20], fill=hair)
    draw.ellipse([cx-10, cy-28, cx-2,  cy-20], fill=(50, 35, 20))
    draw.ellipse([cx+2,  cy-28, cx+10, cy-20], fill=(50, 35, 20))
    draw.arc([cx-8, cy-10, cx+8, cy+2], start=0, end=180, fill=(150, 80, 80), width=2)
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    border = Image.new('RGB', (w+4, h+4), UIDAI_BLUE)
    border.paste(img, (2, 2))
    return border

def draw_aadhaar(person: Dict, qr_img: Image.Image,
                 face_img: Image.Image, card_type: str = 'front') -> Image.Image:
    W, H = 638, 400
    img  = Image.new('RGB', (W, H), BG_WHITE)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([0, 0, W-1, H-1], outline=BORDER_BLUE, width=3)

    # Top header bar
    draw.rectangle([0, 0, W, 52], fill=UIDAI_BLUE)

    # Header text
    f_bold_18 = load_font(FONT_EN_BOLD, 18)
    f_bold_14 = load_font(FONT_EN_BOLD, 14)
    f_reg_13  = load_font(FONT_EN_REG,  13)
    f_reg_11  = load_font(FONT_EN_REG,  11)
    f_hi_bold = load_font(FONT_HI_BOLD, 14)
    f_hi_reg  = load_font(FONT_HI_REG,  12)
    f_uid     = load_font(FONT_EN_BOLD, 22)
    f_small   = load_font(FONT_EN_REG,   9)

    draw.text((12, 8),  'भारत सरकार / Government of India', font=f_hi_bold, fill=BG_WHITE)
    draw.text((12, 30), 'Unique Identification Authority of India',
              font=f_reg_11, fill=(200, 220, 255))

    # UIDAI logo placeholder (orange circle)
    draw.ellipse([W-55, 6, W-10, 46], fill=UIDAI_ORANGE)
    draw.text((W-46, 14), 'UIDAI', font=load_font(FONT_EN_BOLD, 9), fill=BG_WHITE)

    # Orange accent line
    draw.rectangle([0, 52, W, 56], fill=UIDAI_ORANGE)

    # Face photo
    face_x, face_y = 18, 68
    img.paste(face_img, (face_x, face_y))
    fw, fh = face_img.size

    # Aadhaar title
    draw.text((fw + 30, 65), 'आधार', font=load_font(FONT_HI_BOLD, 28),
              fill=UIDAI_BLUE)
    draw.text((fw + 30, 98), 'AADHAAR', font=load_font(FONT_EN_BOLD, 20),
              fill=UIDAI_BLUE)

    # Name
    draw.text((fw + 30, 132), person['name'],
              font=load_font(FONT_EN_BOLD, 16), fill=TEXT_DARK)

    # DOB
    draw.text((fw + 30, 158), 'Date of Birth / जन्म तिथि',
              font=f_small, fill=TEXT_GREY)
    draw.text((fw + 30, 170), person['dob'],
              font=load_font(FONT_EN_BOLD, 14), fill=TEXT_DARK)

    # Gender
    draw.text((fw + 180, 158), 'Gender / लिंग',
              font=f_small, fill=TEXT_GREY)
    draw.text((fw + 180, 170), person['gender'],
              font=load_font(FONT_EN_BOLD, 14), fill=TEXT_DARK)

    # Address
    draw.text((fw + 30, 196), 'Address / पता',
              font=f_small, fill=TEXT_GREY)
    addr_lines = textwrap.wrap(person['address'], width=42)
    for i, line in enumerate(addr_lines[:3]):
        draw.text((fw + 30, 208 + i * 14), line, font=f_reg_11, fill=TEXT_DARK)

    # UID number — large, centred at bottom
    uid_display = person['uid']
    draw.rectangle([0, H-80, W, H-80], fill=(245, 245, 245))
    draw.line([0, H-80, W, H-80], fill=(200, 200, 200), width=1)

    uid_bbox = draw.textbbox((0, 0), uid_display, font=f_uid)
    uid_w    = uid_bbox[2] - uid_bbox[0]
    draw.text(((W - uid_w) // 2, H-72), uid_display, font=f_uid, fill=UIDAI_BLUE)

    draw.text((W//2 - 60, H-46), 'मेरा आधार, मेरी पहचान',
              font=f_hi_reg, fill=TEXT_GREY)
    draw.text((W//2 - 55, H-30), 'My Aadhaar, My Identity',
              font=f_reg_11, fill=TEXT_GREY)

    # QR code — bottom right
    qr_x = W - qr_img.width - 15
    qr_y = H - qr_img.height - 15
    img.paste(qr_img, (qr_x, qr_y))
    draw.text((qr_x, qr_y - 12), 'Scan QR', font=f_small, fill=TEXT_GREY)

    # Subtle background pattern
    for i in range(0, W, 40):
        draw.line([i, 56, i, H-82], fill=(245, 248, 255), width=1)

    return img


def apply_print_scan_effect(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 4, arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)
    result = result.filter(ImageFilter.GaussianBlur(0.6))
    enhancer = ImageEnhance.Contrast(result)
    result   = enhancer.enhance(0.92)
    enhancer = ImageEnhance.Brightness(result)
    result   = enhancer.enhance(0.96)
    return result

def apply_jpeg_compression(img: Image.Image, quality: int = 75) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def apply_photo_swap(img: Image.Image) -> Image.Image:
    result     = img.copy()
    draw       = ImageDraw.Draw(result)
    new_face   = make_face_photo(random.choice(['Male', 'Female']))
    result.paste(new_face, (18, 68))
    return result

def apply_text_edit(img: Image.Image, person: Dict) -> Tuple[Image.Image, str]:
    result = img.copy()
    draw   = ImageDraw.Draw(result)
    f_bold = load_font(FONT_EN_BOLD, 16)
    # Paste white rectangle over name and write different name
    fw = make_face_photo('Male').size[0]
    draw.rectangle([fw+28, 130, fw+280, 152], fill=BG_WHITE)
    fake_name = f"{random.choice(FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
    draw.text((fw+30, 132), fake_name, font=f_bold, fill=TEXT_DARK)
    return result, fake_name

def apply_uid_edit(img: Image.Image) -> Image.Image:
    result = img.copy()
    draw   = ImageDraw.Draw(result)
    W, H   = img.size
    f_uid  = load_font(FONT_EN_BOLD, 22)
    draw.rectangle([W//2 - 160, H-78, W//2 + 160, H-56], fill=(245, 245, 245))
    fake_uid = make_aadhaar_number()
    uid_bbox = draw.textbbox((0,0), fake_uid, font=f_uid)
    uid_w    = uid_bbox[2] - uid_bbox[0]
    draw.text(((W - uid_w) // 2, H-72), fake_uid, font=f_uid, fill=UIDAI_BLUE)
    return result


def generate_sample(out_dir: Path, idx: int) -> Dict[str, Any]:
    person   = make_person()
    qr_xml   = make_qr_xml(person)
    qr_img   = make_qr_image(qr_xml)
    face_img = make_face_photo(person['gender'])

    # Decide sample type
    sample_type = random.choices(
        ['genuine', 'photo_swap', 'text_edit', 'uid_edit', 'qr_mismatch', 'compressed_fake'],
        weights=[40, 15, 15, 10, 10, 10],
        k=1
    )[0]

    base_img = draw_aadhaar(person, qr_img, face_img)
    label    = {'type': sample_type, 'is_forged': sample_type != 'genuine',
                'person': person, 'qr_xml': qr_xml}

    if sample_type == 'genuine':
        final = apply_print_scan_effect(apply_jpeg_compression(base_img, 85))
        label['forgery_details'] = 'Clean genuine document'

    elif sample_type == 'photo_swap':
        final = apply_print_scan_effect(apply_photo_swap(base_img))
        label['forgery_details'] = 'Face photo replaced with different person'

    elif sample_type == 'text_edit':
        edited, fake_name = apply_text_edit(base_img, person)
        final = apply_print_scan_effect(edited)
        label['forgery_details'] = f'Name changed to: {fake_name}'
        label['original_name']   = person['name']

    elif sample_type == 'uid_edit':
        final = apply_print_scan_effect(apply_uid_edit(base_img))
        label['forgery_details'] = 'UID number replaced with invalid number'

    elif sample_type == 'qr_mismatch':
        # Different person's QR on this card
        other   = make_person()
        bad_qr  = make_qr_image(make_qr_xml(other))
        final   = apply_print_scan_effect(draw_aadhaar(person, bad_qr, face_img))
        label['forgery_details'] = 'QR contains different person data'
        label['qr_person']       = other['name']

    elif sample_type == 'compressed_fake':
        # Heavy compression to simulate WhatsApp-forwarded fake
        final = apply_jpeg_compression(apply_print_scan_effect(base_img), quality=45)
        label['forgery_details'] = 'Heavily compressed (WhatsApp-style) genuine doc'
        label['is_forged']       = False  # still genuine, just degraded

    fname = f'aadhaar_{idx:04d}_{sample_type}.jpg'
    fpath = out_dir / fname
    final.save(str(fpath), 'JPEG', quality=88)

    label['filename'] = fname
    label['filepath'] = str(fpath)
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--out',   type=str,
                        default='/home/tech/kavach_document/data/samples')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for i in range(args.count):
        label = generate_sample(out_dir, i)
        labels.append(label)
        status = 'FORGED' if label['is_forged'] else 'GENUINE'
        print(f'[{i+1:3d}/{args.count}] {status:7s}  {label["type"]:20s}  {label["filename"]}')

    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(labels, f, indent=2, default=str)

    genuine_count = sum(1 for l in labels if not l['is_forged'])
    forged_count  = sum(1 for l in labels if l['is_forged'])
    print(f'\nGenerated {args.count} samples → {out_dir}')
    print(f'  Genuine : {genuine_count}')
    print(f'  Forged  : {forged_count}')
    print(f'  Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
