import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from dirmap import store
from dirmap.ocr import label_controls
from dirmap.vision import analyze, read_image


def test_empty_ocr_output_does_not_count_old_numbers_or_overwrite_teacher_labels(monkeypatch):
    monkeypatch.setenv('DIRMAP_TESSERACT', 'test-no-digits')
    monkeypatch.setattr('dirmap.ocr.subprocess.run', lambda *a, **k: SimpleNamespace(stdout='level\ttext\tconf\n'))
    controls = [
        {'id': 'auto', 'pixel': [20, 20], 'radiusPx': 6, 'label': '31', 'labelSource': 'tesseract', 'labelConfidence': .98, 'state': 'candidate'},
        {'id': 'manual', 'pixel': [50, 20], 'radiusPx': 6, 'label': '老师的编号', 'labelSource': 'manual', 'state': 'candidate'},
        {'id': 'accepted', 'pixel': [50, 50], 'radiusPx': 6, 'label': '40', 'labelSource': 'tesseract', 'state': 'accepted'},
    ]
    result = label_controls(np.zeros((80, 80), np.uint8), controls)
    assert result['recognized'] == 0 and result['preserved'] == 2
    assert controls[0]['labelSource'] == 'unread' and 'labelConfidence' not in controls[0]
    assert controls[1]['label'] == '老师的编号' and controls[2]['label'] == '40'


@pytest.mark.skipif(not (os.environ.get('DIRMAP_TESSERACT') or shutil.which('tesseract')), reason='Actual low-resolution OCR requires optional Tesseract')
def test_low_resolution_sheet_numbers_are_recovered_from_fresh_detections():
    image = read_image(store.get_asset('szu-2')['path'])
    result = analyze(image)
    assert all(c['label'].startswith('待核') for c in result['controls'] if c['role'] == 'control')
    ocr = label_controls(result['masks']['purple'], result['controls'], result['masks']['course_text'])
    assert ocr['recognized'] == 6 and ocr['preserved'] == 0
    assert sorted(c['label'] for c in result['controls'] if c['role'] == 'control') == [str(i) for i in range(31, 37)]
