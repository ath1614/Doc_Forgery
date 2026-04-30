"""
KAVACH — Synthetic Voter ID Generator
Multilingual: English + Hindi + Tamil + Telugu + Bengali + Marathi
Generates genuine and forged samples with ground truth labels.

Usage:
  python generate_voter_id.py --count 20 --out /home/tech/kavach_document/data/samples
"""

import os, io, re, json, random, argparse, textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, timedelta
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

FONT_EN_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_EN_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_HI_REG  = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
FONT_HI_BOLD = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf'

# ECI colour scheme
ECI_BLUE    = (0, 56, 147)
ECI_ORANGE  = (255, 103, 31)
ECI_GREEN   = (19, 136, 8)
BG_WHITE    = (255, 255, 255)
TEXT_DARK   = (20, 20, 20)
TEXT_GREY   = (90, 90, 90)

FIRST_NAMES_M = ['Ramesh','Suresh','Mahesh','Rajesh','Dinesh','Ganesh','Naresh',
                  'Umesh','Viresh','Lokesh','Anil','Sunil','Kapil','Sahil','Nikhil']
FIRST_NAMES_F = ['Sunita','Kavita','Savita','Lalita','Mamta','Rekha','Usha',
                  'Asha','Nisha','Misha','Priya','Divya','Neha','Sneha','Geeta']
LAST_NAMES    = ['Sharma','Verma','Singh','Kumar','Gupta','Patel','Yadav','Tiwari',
                 'Pandey','Mishra','Chauhan','Joshi','Nair','Reddy','Iyer','Pillai',
                 'Mehta','Shah','Desai','Rao']
STATES        = ['Maharashtra','Delhi','Karnataka','Tamil Nadu','Uttar Pradesh',
                 'Gujarat','Rajasthan','West Bengal','Telangana','Kerala']
CONSTITUENCIES = ['Andheri East','Borivali West','Kurla','Dharavi','Bandra East',
                   'Malad West','Goregaon','Vikhroli','Ghatkopar','Mulund',
                   'Lajpat Nagar','Karol Bagh','Rohini','Dwarka','Janakpuri']
RELATIONS     = ['S/O','D/O','W/O','H/O']

# Multilingual labels
LANG_LABELS = {
    'hindi':   {'voter':'मतदाता पहचान पत्र','name':'नाम','father':'पिता/पति का नाम',
                'dob':'जन्म तिथि','gender':'लिंग','epic':'क्रमांक',
                'male':'पुरुष','female':'महिला','constituency':'विधान सभा'},
    'marathi': {'voter':'मतदार ओळखपत्र','name':'नाव','father':'वडिलांचे/पतीचे नाव',
                'dob':'जन्म दिनांक','gender':'लिंग','epic':'अनुक्रमांक',
                'male':'पुरुष','female':'स्त्री','constituency':'विधानसभा'},
    'bengali': {'voter':'ভোটার পরিচয়পত্র','name':'নাম','father':'পিতা/স্বামীর নাম',
                'dob':'জন্ম তারিখ','gender':'লিঙ্গ','epic':'ক্রমিক নং',
                'male':'পুরুষ','female':'মহিলা','constituency':'বিধানসভা'},
}

def load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def make_epic() -> str:
    state_codes = ['MH','DL','KA','TN','UP','GJ','RJ','WB','TS','KL']
    return random.choice(state_codes) + ''.join([str(random.randint(0,9)) for _ in range(7)])

def make_dob(min_age=18, max_age=80) -> str:
    days = random.randint(min_age*365, max_age*365)
    d    = date.today() - timedelta(days=days)
    return d.strftime('%d/%m/%Y')

def make_person(gender=None) -> Dict:
    if gender is None: gender = random.choice(['Male','Female'])
    first = random.choice(FIRST_NAMES_M if gender=='Male' else FIRST_NAMES_F)
    last  = random.choice(LAST_NAMES)
    rel   = random.choice(RELATIONS)
    father_first = random.choice(FIRST_NAMES_M)
    father_last  = last
    return {
        'name':         f'{first} {last}',
        'relation':     rel,
        'father_name':  f'{father_first} {father_last}',
        'gender':       gender,
        'dob':          make_dob(),
        'epic':         make_epic(),
        'constituency': random.choice(CONSTITUENCIES),
        'state':        random.choice(STATES),
        'part_no':      str(random.randint(1,300)),
        'serial_no':    str(random.randint(1,1000)),
    }

def make_face_photo(gender, size=(80,100)):
    w, h = size
    img  = Image.new('RGB',(w,h),(215,175,135))
    draw = ImageDraw.Draw(img)
    cx,cy = w//2, h//2
    skin = (205,165,125) if gender=='Male' else (220,180,145)
    draw.ellipse([cx-25,cy-35,cx+25,cy+15], fill=skin)
    draw.rectangle([cx-20,cy+8,cx+20,cy+42], fill=skin)
    hair = (40,25,15) if random.random()>0.2 else (80,50,30)
    draw.ellipse([cx-25,cy-47,cx+25,cy-18], fill=hair)
    draw.ellipse([cx-8,cy-26,cx-1,cy-18], fill=(45,30,15))
    draw.ellipse([cx+1,cy-26,cx+8,cy-18], fill=(45,30,15))
    draw.arc([cx-7,cy-8,cx+7,cy+2], start=0, end=180, fill=(140,70,70), width=2)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    border = Image.new('RGB',(w+4,h+4),ECI_BLUE)
    border.paste(img,(2,2))
    return border

def draw_voter_id(person: Dict, face_img: Image.Image,
                  lang: str = 'hindi') -> Image.Image:
    W, H = 638, 400
    img  = Image.new('RGB',(W,H),BG_WHITE)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([0,0,W-1,H-1], outline=ECI_BLUE, width=3)

    # Top bar
    draw.rectangle([0,0,W,55], fill=ECI_BLUE)

    # Tricolour strip
    draw.rectangle([0,55,W,62], fill=(255,153,51))   # saffron
    draw.rectangle([0,62,W,69], fill=BG_WHITE)
    draw.rectangle([0,69,W,76], fill=(19,136,8))     # green

    # Header text
    f_hi_bold_16 = load_font(FONT_HI_BOLD, 16)
    f_en_bold_14 = load_font(FONT_EN_BOLD, 14)
    f_en_bold_12 = load_font(FONT_EN_BOLD, 12)
    f_en_reg_11  = load_font(FONT_EN_REG,  11)
    f_en_reg_10  = load_font(FONT_EN_REG,  10)
    f_en_bold_18 = load_font(FONT_EN_BOLD, 18)
    f_hi_reg_12  = load_font(FONT_HI_REG,  12)
    f_small      = load_font(FONT_EN_REG,   9)

    lbl = LANG_LABELS.get(lang, LANG_LABELS['hindi'])

    draw.text((12,8),  'भारत निर्वाचन आयोग', font=f_hi_bold_16, fill=BG_WHITE)
    draw.text((12,30), 'Election Commission of India', font=f_en_reg_11, fill=(200,220,255))

    # ECI emblem placeholder
    draw.ellipse([W-52,5,W-8,50], fill=ECI_ORANGE)
    draw.text((W-46,16), 'ECI', font=load_font(FONT_EN_BOLD,11), fill=BG_WHITE)

    # EPIC title
    draw.text((W//2-90, 80), 'ELECTORS PHOTO IDENTITY CARD',
              font=f_en_bold_12, fill=ECI_BLUE)
    draw.text((W//2-70, 96), lbl['voter'],
              font=f_hi_reg_12, fill=ECI_BLUE)

    # Face photo
    fx, fy = 18, 112
    img.paste(face_img, (fx, fy))
    fw, fh = face_img.size

    # Fields
    x0 = fw + 30
    y  = 115

    def field(label_en, label_hi, value, bold=False):
        nonlocal y
        draw.text((x0, y), f'{label_en} / {label_hi}', font=f_small, fill=TEXT_GREY)
        y += 12
        fn = load_font(FONT_EN_BOLD if bold else FONT_EN_REG, 13)
        draw.text((x0, y), value, font=fn, fill=TEXT_DARK)
        y += 18

    field('Name', lbl['name'], person['name'], bold=True)
    field(person['relation'], lbl['father'], person['father_name'])
    field('Date of Birth', lbl['dob'], person['dob'])
    field('Gender', lbl['gender'],
          person['gender'] + ' / ' + (lbl['male'] if person['gender']=='Male' else lbl['female']))
    field('Constituency', lbl['constituency'], person['constituency'])

    # EPIC number — prominent
    draw.rectangle([0, H-80, W, H-80], fill=(245,247,255))
    draw.line([0,H-80,W,H-80], fill=(200,210,230), width=1)

    draw.text((18, H-72), 'EPIC No. / ' + lbl['epic'],
              font=f_small, fill=TEXT_GREY)
    draw.text((18, H-58), person['epic'],
              font=load_font(FONT_EN_BOLD, 22), fill=ECI_BLUE)

    draw.text((W-200, H-72), f"Part No: {person['part_no']}",
              font=f_en_reg_10, fill=TEXT_GREY)
    draw.text((W-200, H-58), f"Serial No: {person['serial_no']}",
              font=f_en_reg_10, fill=TEXT_GREY)

    draw.text((W//2-80, H-30), person['state'] + ' / ' + person['state'],
              font=f_en_reg_10, fill=TEXT_GREY)

    # Subtle grid lines
    for i in range(0, W, 50):
        draw.line([i,76,i,H-82], fill=(248,250,255), width=1)

    return img

def apply_effects(img, quality=85, noise_level=3):
    arr   = np.array(img).astype(np.float32)
    arr   = np.clip(arr + np.random.normal(0, noise_level, arr.shape), 0, 255).astype(np.uint8)
    result = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.5))
    buf   = io.BytesIO()
    result.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def apply_photo_swap(img):
    result = img.copy()
    new_face = make_face_photo(random.choice(['Male','Female']))
    result.paste(new_face, (18, 112))
    return result

def apply_epic_edit(img, person):
    result = img.copy()
    draw   = ImageDraw.Draw(result)
    W, H   = img.size
    draw.rectangle([16, H-62, 300, H-42], fill=(245,247,255))
    fake_epic = make_epic()
    draw.text((18, H-58), fake_epic,
              font=load_font(FONT_EN_BOLD, 22), fill=ECI_BLUE)
    return result, fake_epic

def apply_name_edit(img, person):
    result = img.copy()
    draw   = ImageDraw.Draw(result)
    fw     = make_face_photo('Male').size[0]
    x0     = fw + 30
    draw.rectangle([x0, 127, x0+260, 143], fill=BG_WHITE)
    fake_name = f"{random.choice(FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
    draw.text((x0, 127), fake_name,
              font=load_font(FONT_EN_BOLD, 13), fill=TEXT_DARK)
    return result, fake_name

def generate_sample(out_dir: Path, idx: int) -> Dict:
    lang   = random.choice(['hindi','marathi','hindi','hindi'])  # hindi weighted
    person = make_person()
    face   = make_face_photo(person['gender'])

    sample_type = random.choices(
        ['genuine','photo_swap','epic_edit','name_edit','compressed'],
        weights=[40, 20, 15, 15, 10], k=1
    )[0]

    base  = draw_voter_id(person, face, lang)
    label = {'type': sample_type, 'is_forged': sample_type not in ('genuine','compressed'),
             'person': person, 'lang': lang}

    if sample_type == 'genuine':
        final = apply_effects(base, quality=88, noise_level=3)
        label['forgery_details'] = 'Clean genuine Voter ID'

    elif sample_type == 'photo_swap':
        final = apply_effects(apply_photo_swap(base), quality=85)
        label['forgery_details'] = 'Face photo replaced'

    elif sample_type == 'epic_edit':
        edited, fake_epic = apply_epic_edit(base, person)
        final = apply_effects(edited, quality=85)
        label['forgery_details'] = f'EPIC changed to {fake_epic}'
        label['original_epic']   = person['epic']

    elif sample_type == 'name_edit':
        edited, fake_name = apply_name_edit(base, person)
        final = apply_effects(edited, quality=85)
        label['forgery_details'] = f'Name changed to {fake_name}'
        label['original_name']   = person['name']

    elif sample_type == 'compressed':
        final = apply_effects(base, quality=45, noise_level=6)
        label['forgery_details'] = 'Heavily compressed (WhatsApp-style)'
        label['is_forged']       = False

    fname = f'voter_{idx:04d}_{sample_type}_{lang[:2]}.jpg'
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
        print(f'[{i+1:3d}/{args.count}] {status:7s}  {label["type"]:15s}  {label["lang"]:8s}  {label["filename"]}')

    manifest_path = out_dir / 'voter_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(labels, f, indent=2, default=str)

    genuine = sum(1 for l in labels if not l['is_forged'])
    forged  = sum(1 for l in labels if l['is_forged'])
    print(f'\nGenerated {args.count} Voter ID samples → {out_dir}')
    print(f'  Genuine: {genuine}  Forged: {forged}')
    print(f'  Manifest: {manifest_path}')

if __name__ == '__main__':
    main()
