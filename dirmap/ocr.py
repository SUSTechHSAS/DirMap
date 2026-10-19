"""Optional Tesseract number recognition; uncertain labels stay reviewable."""

import csv
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from .symbols import course_ink_mask


def label_controls(purple, controls, text_ink=None):
    executable = os.environ.get('DIRMAP_TESSERACT') or shutil.which('tesseract')
    if not executable or not controls:
        return {'available': bool(executable), 'recognized': 0, 'method': 'tesseract_global_and_digit_crops'}
    eligible = {c['id'] for c in controls if c.get('role', 'control') == 'control' and c.get('state') != 'accepted'
                and c.get('labelSource') in ('unread', 'tesseract')}
    # A fresh OCR pass must not count labels left over from an earlier run.
    # Explicit teacher labels and accepted targets are kept separately.
    for index, control in enumerate(controls):
        if control['id'] in eligible:
            control.update(label=f'待核{index+1}', labelSource='unread')
            control.pop('labelConfidence', None)
            control.pop('labelPixelBox', None)
            control.pop('labelAmbiguous', None)
    height, width = purple.shape
    # Saturated course text separates bold numbers from pale restriction
    # hatching. Keep the broad purple mask for terrain and circle detection.
    text_ink = purple if text_ink is None else text_ink
    clean = text_ink.copy()
    broad_clean = purple.copy()
    strokes_removed = cv2.bitwise_and(text_ink, cv2.bitwise_not(course_ink_mask(purple, controls)))
    broad_strokes_removed = cv2.bitwise_and(purple, cv2.bitwise_not(course_ink_mask(purple, controls)))
    for c in controls:
        cv2.circle(clean, tuple(np.rint(c['pixel']).astype(int)), round(c['radiusPx']*1.2), 0, -1)
        cv2.circle(broad_clean, tuple(np.rint(c['pixel']).astype(int)), round(c['radiusPx']*1.2), 0, -1)
    with tempfile.TemporaryDirectory(prefix='dirmap-ocr-') as temporary:
        output = Path(temporary)/'numbers.png'

        def recognize(mask, offset=(0, 0), scale=1, psm=11):
            border = 16
            crop = cv2.copyMakeBorder(255-mask, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
            if scale != 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(output), crop)
            try:
                process = subprocess.run([executable, str(output), 'stdout', '--psm', str(psm),
                                           '-c', 'tessedit_char_whitelist=0123456789-', 'tsv'],
                                          capture_output=True, text=True, timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                return []
            words = []
            for row in csv.DictReader(io.StringIO(process.stdout), delimiter='\t'):
                text = row.get('text', '').strip()
                confidence = float(row.get('conf', -1))
                label = re.fullmatch(r'(?:[1-9]-)?([1-9]\d{1,2})', text)
                if not label or confidence < 65:
                    continue
                center = np.array([offset[0]-border+(int(row['left'])+int(row['width'])/2)/scale,
                                   offset[1]-border+(int(row['top'])+int(row['height'])/2)/scale])
                box = [offset[0]-border+int(row['left'])/scale, offset[1]-border+int(row['top'])/scale,
                       offset[0]-border+(int(row['left'])+int(row['width']))/scale,
                       offset[1]-border+(int(row['top'])+int(row['height']))/scale]
                words.append({'label': label.group(1), 'confidence': confidence, 'center': center, 'box': box})
            return words

        def glyph_rows(mask, point, radius):
            x0, y0 = np.maximum(0, np.floor(point-radius*5).astype(int))
            x1, y1 = np.minimum([width, height], np.ceil(point+radius*5).astype(int))
            crop = mask[y0:y1, x0:x1]
            _, _, stats, _ = cv2.connectedComponentsWithStats((crop > 0).astype(np.uint8), connectivity=8)
            glyphs = [s[:4] for s in stats[1:] if radius*.25 <= s[3] <= radius*1.8
                      and radius*.06 <= s[2] <= radius*2 and s[4] >= max(4, radius*radius*.035)]
            rows, used = [], set()
            for i, box in sorted(enumerate(glyphs), key=lambda item: item[1][0]):
                if i in used:
                    continue
                x, y, w, h = box
                row = [box]; used.add(i)
                right = x+w
                for j, other in sorted(enumerate(glyphs), key=lambda item: item[1][0]):
                    xx, yy, ww, hh = other
                    if j in used or not 0 <= xx-right <= max(h, hh)*.7:
                        continue
                    if abs(yy+hh/2-y-h/2) > max(h, hh)*.35 or not .55 <= hh/h <= 1.8:
                        continue
                    row.append(other); used.add(j); right = xx+ww
                left, top = np.min(np.array(row)[:, :2], axis=0)
                right, bottom = np.max(np.array(row)[:, :2]+np.array(row)[:, 2:], axis=0)
                if .45 <= (right-left)/max(1, bottom-top) <= 4:
                    rows.append((int(left+x0), int(top+y0), int(right+x0), int(bottom+y0)))
            rows.sort(key=lambda b: np.linalg.norm(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])-point))
            return rows[:8]

        global_words = recognize(clean)
        if not np.array_equal(clean, broad_clean):
            global_words += recognize(broad_clean)
        for control in controls:
            if control['id'] not in eligible:
                continue
            point, radius = np.asarray(control['pixel']), control['radiusPx']
            words = global_words.copy()
            nearby = [word for word in words if np.linalg.norm(word['center']-point) < radius*4.8]
            if not any(word['confidence'] > 90 for word in nearby):
                for factor in (3, 4.5):
                    x0, y0 = np.maximum(0, np.floor(point-radius*factor).astype(int))
                    x1, y1 = np.minimum([width, height], np.ceil(point+radius*factor).astype(int))
                    words += recognize(clean[y0:y1, x0:x1], (x0, y0), scale=2)
                x0, y0 = np.maximum(0, np.floor(point+[radius*.35, -radius*3.6]).astype(int))
                x1, y1 = np.minimum([width, height], np.ceil(point+[radius*3.2, -radius*.35]).astype(int))
                if x1 > x0 and y1 > y0:
                    words += recognize(clean[y0:y1, x0:x1], (x0, y0), scale=3, psm=7)
                # A numeral often touches a circle. Erasing the whole disk
                # destroys that numeral, so retry actual glyph rows from the
                # original ink and a stroke-only mask, keeping full geometry.
                for mask in (text_ink, strokes_removed, purple, broad_strokes_removed):
                    if x1 > x0 and y1 > y0:
                        words += recognize(mask[y0:y1, x0:x1], (x0, y0), scale=4, psm=7)
                        # An upper crop can contain a second row of unrelated
                        # hatching. Sparse-text mode plus a small opening
                        # separates the digits from circle remnants/JPEG ink.
                        opened = cv2.morphologyEx(mask[y0:y1, x0:x1], cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                        words += recognize(opened, (x0, y0), scale=4, psm=11)
                    for left, top, right, bottom in glyph_rows(mask, point, radius):
                        words += recognize(mask[top:bottom, left:right], (left, top), scale=max(2, 48/(bottom-top)), psm=7)
            options = []
            for word in words:
                dist = np.linalg.norm(word['center']-point)
                if dist > radius*4.8:
                    continue
                closest = min((c for c in controls if c.get('role', 'control') == 'control'),
                              key=lambda c: np.linalg.norm(word['center']-c['pixel'])/c['radiusPx'])
                if closest['id'] != control['id']:
                    continue
                score = (100-word['confidence'])/8+dist/max(1, radius)
                options.append((score, word['label'], word['confidence'], word['box']))
            if options:
                _, label, confidence, box = min(options)
                control.update(label=label, labelSource='tesseract', labelConfidence=round(confidence/100, 3), labelPixelBox=box)
    counts = {}
    for c in controls:
        counts[c['label']] = counts.get(c['label'], 0)+1
    for c in controls:
        if counts[c['label']] > 1:
            c['labelAmbiguous'] = True
    return {'available': True, 'recognized': sum(c.get('labelSource') == 'tesseract' and c['id'] in eligible for c in controls),
            'preserved': sum(c.get('role', 'control') == 'control' and c['id'] not in eligible for c in controls),
            'method': 'tesseract_global_and_digit_crops'}
