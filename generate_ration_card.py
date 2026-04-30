"""
KAVACH — Synthetic Ration Card Generator
Multilingual: English + Hindi + Marathi + Tamil + Telugu + Bengali
State-specific formats: Maharashtra, Delhi, UP, Karnataka, Tamil Nadu, West Bengal

Usage:
  python generate_ration_card.py --count 20 --out /home/tech/kavach_document/data/samples
"""

import io, json, random, argparse, textwrap
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_EN_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_EN_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_HI_REG  = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
FONT_HI_BOLD = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf'

BG_WHITE   = (255, 255, 255)
TEXT_DARK  = (20,  20,  20)
TEXT_GREY  = (90,  90,  90)

# State-specific colour schemes
STATE_THEMES = {
    'Maharashtra': {'primary': (0,  84, 166),  'accent': (255, 153, 0),   'label': 'महाराष्ट्र शासन'},
    'Delhi':       {'primary': (0,  112, 60),   'accent': (255, 200, 0),   'label': 'दिल्ली सरकार'},
    'UP':          {'primary': (0,  56, 147),   'accent': (255, 103, 31),  'label': 'उत्तर प्रदेश सरकार'},
    'Karnataka':   {'primary': (220, 20, 60),   'accent': (255, 215, 0),   'label': 'ಕರ್ನಾಟಕ ಸರ್ಕಾರ'},
    'Tamil Nadu':  {'primary': (0,  0,  128),   'accent': (255, 165, 0),   'label': 'தமிழ்நாடு அரசு'},
    'West Bengal': {'primary': (0,  128, 0),    'accent': (255, 255, 255), 'label': 'পশ্চিমবঙ্গ সরকার'},
}

CARD_TYPES = {
    'AAY':  {'color': (255, 215, 0),   'label': 'Antyodaya Anna Yojana (AAY)', 'hi': 'अंत्योदय अन्न योजना'},
    'BPL':  {'color': (255, 100, 100), 'label': 'Below Poverty Line (BPL)',    'hi': 'गरीबी रेखा से नीचे'},
    'APL':  {'color': (100, 200, 100), 'label': 'Above Poverty Line (APL)',    'hi': 'गरीबी रेखा से ऊपर'},
    'PHH':  {'color': (100, 150, 255), 'label': 'Priority Household (PHH)',    'hi': 'प्राथमिकता परिवार'},
}

FIRST_NAMES_M = ['Ramesh','Suresh','Mahesh','Rajesh','Dinesh','Ganesh','Anil',
                  'Sunil','Kapil','Vijay','Ravi','Ajay','Sanjay','Manoj','Deepak']
FIRST_NAMES_F = ['Sunita','Kavita','Savita','Rekha','Usha','Asha','Nisha',
                  'Geeta','Meena','Seema','Anita','Mamta','Lalita','Pushpa','Kamla']
LAST_NAMES    = ['Sharma','Verma','Singh','Kumar','Gupta','Patel','Yadav','Tiwari',
                 'Pandey','Mishra','Chauhan','Joshi','Nair','Reddy','Mehta','Shah']
STREETS       = ['Gandhi Nagar','Nehru Colony','Ambedkar Nagar','Shivaji Nagar',
                 'Rajiv Gandhi Road','Laxmi Nagar','Subhash Nagar','Patel Colony']
CITIES        = {'Maharashtra':['Mumbai','Pune','Nagpur','Nashik','Aurangabad'],
                 'Delhi':['New Delhi','North Delhi','South Delhi','East Delhi','West Delhi'],
                 'UP':['Lucknow','Kanpur','Agra','Varanasi','Allahabad'],
                 'Karnataka':['Bangalore','Mysore','Hubli','Mangalore','Belgaum'],
                 'Tamil Nadu':['Chennai','Coimbatore','Madurai','Salem','Trichy'],
                 'West Bengal':['Kolkata','Howrah','Durgapur','Asansol','Siliguri']}

def load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def make_card_number(state: str) -> str:
    codes = {'Maharashtra':'MH','Delhi':'DL','UP':'UP',
             'Karnataka':'KA','Tamil Nadu':'TN','West Bengal':'WB'}
    code = codes.get(state, 'XX')
    return f"{code}/{random.randint(10,99)}/{random.randint(100000,999999)}"

def make_person(gender=None) -> Dict:
    if gender is None: gender = random.choice(['Male','Female'])
    first = random.choice(FIRST_NAMES_M if gender=='Male' else FIRST_NAMES_F)
    last  = random.choice(LAST_NAMES)
    return {'name': f'{first} {last}', 'gender': gender}

def make_family(size: int) -> List[Dict]:
    head_gender = random.choice(['Male','Female'])
    head = make_person(head_gender)
    head['relation'] = 'Self'
    head['age']      = random.randint(25, 60)
    members = [head]
    for i in range(size - 1):
        m = make_person()
        m['relation'] = random.choice(['Spouse','Son','Daughter','Father','Mother'])
        m['age']      = random.randint(5, 70)
        members.append(m)
    return members

def draw_ration_card(card_number: str, family: List[Dict],
                     state: str, card_type: str,
                     issue_date: str, address: str) -> Image.Image:
    W, H  = 680, 480
    img   = Image.new('RGB', (W, H), BG_WHITE)
    draw  = ImageDraw.Draw(img)

    theme = STATE_THEMES[state]
    ctype = CARD_TYPES[card_type]
    primary = theme['primary']
    accent  = theme['accent']

    # Border
    draw.rectangle([0,0,W-1,H-1], outline=primary, width=4)
    draw.rectangle([4,4,W-5,H-5], outline=accent,  width=1)

    # Header bar
    draw.rectangle([0,0,W,60], fill=primary)

    # Card type colour strip
    draw.rectangle([0,60,W,72], fill=ctype['color'])

    # Fonts
    f_hi_bold_16 = load_font(FONT_HI_BOLD, 16)
    f_hi_bold_14 = load_font(FONT_HI_BOLD, 14)
    f_hi_reg_12  = load_font(FONT_HI_REG,  12)
    f_en_bold_14 = load_font(FONT_EN_BOLD, 14)
    f_en_bold_12 = load_font(FONT_EN_BOLD, 12)
    f_en_reg_11  = load_font(FONT_EN_REG,  11)
    f_en_reg_10  = load_font(FONT_EN_REG,  10)
    f_small      = load_font(FONT_EN_REG,   9)

    # Header text
    draw.text((12, 8),  theme['label'], font=f_hi_bold_16, fill=BG_WHITE)
    draw.text((12, 30), 'Food, Civil Supplies & Consumer Protection Dept.',
              font=load_font(FONT_EN_REG, 10), fill=(200,220,255))

    # Govt emblem placeholder
    draw.ellipse([W-55,5,W-8,52], fill=accent)
    draw.text((W-48,18), 'GOVT', font=load_font(FONT_EN_BOLD,9), fill=primary)

    # Card type label
    draw.text((W//2 - 120, 62), ctype['label'], font=f_en_bold_12, fill=TEXT_DARK)

    # Title
    draw.text((W//2 - 80, 80), 'राशन कार्ड / RATION CARD',
              font=f_hi_bold_14, fill=primary)

    # Card number
    draw.text((18, 105), 'Card No. / कार्ड क्रमांक:', font=f_small, fill=TEXT_GREY)
    draw.text((18, 117), card_number,
              font=load_font(FONT_EN_BOLD, 18), fill=primary)

    # Issue date
    draw.text((W-200, 105), 'Issue Date / जारी दिनांक:', font=f_small, fill=TEXT_GREY)
    draw.text((W-200, 117), issue_date, font=f_en_reg_11, fill=TEXT_DARK)

    # Address
    draw.text((18, 145), 'Address / पता:', font=f_small, fill=TEXT_GREY)
    addr_lines = textwrap.wrap(address, width=55)
    for i, line in enumerate(addr_lines[:2]):
        draw.text((18, 157 + i*13), line, font=f_en_reg_10, fill=TEXT_DARK)

    # Divider
    draw.line([0, 188, W, 188], fill=(220,225,235), width=1)

    # Family members table header
    draw.rectangle([0, 190, W, 210], fill=primary)
    cols = [18, 200, 340, 430, 530]
    headers = ['Name / नाम', 'Relation / संबंध', 'Age / आयु', 'Gender / लिंग', 'Aadhaar (Last 4)']
    for i, (x, h) in enumerate(zip(cols, headers)):
        draw.text((x, 196), h, font=f_small, fill=BG_WHITE)

    # Family rows
    row_colors = [(255,255,255),(245,248,255)]
    for idx, member in enumerate(family[:6]):
        y   = 212 + idx * 22
        bg  = row_colors[idx % 2]
        draw.rectangle([0, y, W, y+22], fill=bg)
        draw.text((cols[0], y+4), member['name'][:22],       font=f_en_reg_10, fill=TEXT_DARK)
        draw.text((cols[1], y+4), member['relation'],         font=f_en_reg_10, fill=TEXT_DARK)
        draw.text((cols[2], y+4), str(member['age']),         font=f_en_reg_10, fill=TEXT_DARK)
        draw.text((cols[3], y+4), member['gender'],           font=f_en_reg_10, fill=TEXT_DARK)
        draw.text((cols[4], y+4), f'XXXX XXXX {random.randint(1000,9999)}',
                  font=f_en_reg_10, fill=TEXT_GREY)

    # Bottom section
    bottom_y = 212 + min(len(family), 6) * 22 + 8
    draw.line([0, bottom_y, W, bottom_y], fill=(220,225,235), width=1)

    draw.text((18, bottom_y+6),
              f'Total Members: {len(family)}  |  Card Type: {card_type}  |  State: {state}',
              font=f_en_reg_10, fill=TEXT_GREY)

    # FPS shop
    fps_no = f'FPS/{random.randint(100,999)}/{random.randint(1000,9999)}'
    draw.text((18, bottom_y+20), f'Fair Price Shop No: {fps_no}',
              font=f_en_reg_10, fill=TEXT_GREY)

    # Signature line
    draw.line([W-180, H-40, W-20, H-40], fill=TEXT_DARK, width=1)
    draw.text((W-160, H-30), 'Authorised Signatory', font=f_small, fill=TEXT_GREY)

    # Subtle background
    for i in range(0, W, 45):
        draw.line([i, 72, i, bottom_y], fill=(248,250,255), width=1)

    return img

def apply_effects(img, quality=85, noise=3):
    arr    = np.array(img).astype(np.float32)
    arr    = np.clip(arr + np.random.normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
    result = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.4))
    buf    = io.BytesIO()
    result.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()

def apply_member_count_edit(img, family):
    result = img.copy()
    draw   = ImageDraw.Draw(result)
    W, H   = img.size
    bottom_y = 212 + min(len(family), 6) * 22 + 8
    draw.rectangle([16, bottom_y+4, 300, bottom_y+18], fill=BG_WHITE)
    fake_count = len(family) + random.randint(2, 5)
    draw.text((18, bottom_y+6),
              f'Total Members: {fake_count}  |  Card Type: {family[0].get("card_type","BPL")}',
              font=load_font(FONT_EN_REG, 10), fill=TEXT_GREY)
    return result, fake_count

def apply_card_type_edit(img, original_type):
    result    = img.copy()
    draw      = ImageDraw.Draw(result)
    W, H      = img.size
    new_type  = random.choice([t for t in CARD_TYPES if t != original_type])
    ctype     = CARD_TYPES[new_type]
    draw.rectangle([0, 60, W, 72], fill=ctype['color'])
    draw.rectangle([16, 78, W-16, 98], fill=BG_WHITE)
    draw.text((W//2 - 80, 80), 'राशन कार्ड / RATION CARD',
              font=load_font(FONT_HI_BOLD, 14), fill=(0,56,147))
    return result, new_type

def generate_sample(out_dir: Path, idx: int) -> Dict:
    state     = random.choice(list(STATE_THEMES.keys()))
    card_type = random.choice(list(CARD_TYPES.keys()))
    fam_size  = random.randint(2, 6)
    family    = make_family(fam_size)
    for m in family: m['card_type'] = card_type

    days_ago  = random.randint(30, 3650)
    issue_dt  = (date.today() - timedelta(days=days_ago)).strftime('%d/%m/%Y')
    city      = random.choice(CITIES[state])
    address   = f'{random.randint(1,999)}, {random.choice(STREETS)}, {city}, {state}'
    card_no   = make_card_number(state)

    sample_type = random.choices(
        ['genuine','member_count_edit','card_type_edit','compressed','name_edit'],
        weights=[40, 20, 15, 15, 10], k=1
    )[0]

    base  = draw_ration_card(card_no, family, state, card_type, issue_dt, address)
    label = {'type': sample_type,
             'is_forged': sample_type not in ('genuine','compressed'),
             'state': state, 'card_type': card_type,
             'family_size': fam_size, 'head_name': family[0]['name']}

    if sample_type == 'genuine':
        final = apply_effects(base, quality=88, noise=3)
        label['forgery_details'] = 'Clean genuine Ration Card'

    elif sample_type == 'member_count_edit':
        edited, fake_count = apply_member_count_edit(base, family)
        final = apply_effects(edited, quality=85)
        label['forgery_details'] = f'Member count inflated to {fake_count} (actual {fam_size})'

    elif sample_type == 'card_type_edit':
        edited, new_type = apply_card_type_edit(base, card_type)
        final = apply_effects(edited, quality=85)
        label['forgery_details'] = f'Card type changed from {card_type} to {new_type}'
        label['original_type']   = card_type

    elif sample_type == 'compressed':
        final = apply_effects(base, quality=45, noise=7)
        label['forgery_details'] = 'Heavily compressed'
        label['is_forged']       = False

    elif sample_type == 'name_edit':
        result = base.copy()
        draw   = ImageDraw.Draw(result)
        draw.rectangle([18, 213, 195, 233], fill=BG_WHITE)
        fake_name = f"{random.choice(FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
        draw.text((18, 216), fake_name[:22],
                  font=load_font(FONT_EN_REG, 10), fill=TEXT_DARK)
        final = apply_effects(result, quality=85)
        label['forgery_details'] = f'Head name changed to {fake_name}'
        label['original_name']   = family[0]['name']

    state_code = state.replace(' ','')[:4].lower()
    fname = f'ration_{idx:04d}_{sample_type}_{state_code}.jpg'
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
        print(f'[{i+1:3d}/{args.count}] {status:7s}  {label["type"]:20s}  '
              f'{label["state"]:12s}  {label["card_type"]}  {label["filename"]}')

    manifest_path = out_dir / 'ration_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(labels, f, indent=2, default=str)

    genuine = sum(1 for l in labels if not l['is_forged'])
    forged  = sum(1 for l in labels if l['is_forged'])
    print(f'\nGenerated {args.count} Ration Card samples → {out_dir}')
    print(f'  Genuine: {genuine}  Forged: {forged}')
    print(f'  Manifest: {manifest_path}')

if __name__ == '__main__':
    main()
