import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

base = Path(__file__).resolve().parent
plan = json.loads((base / 'deck-plan.json').read_text(encoding='utf-8'))
deck = base.parent / 'output' / 'QWOP运动控制与策略优化研究_课堂汇报_v3.pptx'
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(deck) as archive:
    slides = [n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
    notes = [n for n in archive.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', n)]
    notes_text = [''.join(ET.fromstring(archive.read(n)).itertext()) for n in notes]
    assert len(slides) == len(notes) == 12
    for item in plan['slides']:
        assert len(item['bullets']) <= 5
        assert any(item['speaker_note'] in text for text in notes_text), item['number']
    assert sum(s['duration_seconds'] for s in plan['slides']) == 600
    assert archive.testzip() is None
    embedded = [n for n in archive.namelist() if n.endswith('.xlsx')]
    assert len(embedded) == 4

def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()

same_visuals = all(digest(base/'final_previews_v2'/f'slide-{i:02d}.png') == digest(base/'final_previews_v3'/f'slide-{i:02d}.png') for i in range(1,13))
assert same_visuals, 'Notes-only revision unexpectedly changed slide rendering'
result = {'passed': True, 'slides': len(slides), 'nonempty_matching_speaker_notes': len(notes), 'planned_seconds':600, 'speaker_note_character_count':sum(len(s['speaker_note']) for s in plan['slides']), 'max_bullets':max(len(s['bullets']) for s in plan['slides']), 'editable_chart_workbooks':len(embedded), 'final_visuals_identical_to_reviewed_v2':same_visuals, 'sha256':digest(deck)}
(base/'content_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
