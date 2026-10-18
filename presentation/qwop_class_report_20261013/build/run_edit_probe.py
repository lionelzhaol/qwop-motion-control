import runpy
import sys
import tempfile
from pathlib import Path

build = Path(__file__).resolve().parent
temp_root = (build / 'edit_probe_temp').resolve()
assert temp_root.is_relative_to(build)
temp_root.mkdir(exist_ok=True)
tempfile.tempdir = str(temp_root)
sys.argv = ['probe_pptx_editability.py', str(build.parent/'output'/'QWOP运动控制与策略优化研究_课堂汇报_v3.pptx'), '--search', '系统架构与训练闭环', '--replacement', '系统架构与训练闭环（校验副本）', '--json-output', str(build/'editability_probe.json')]
runpy.run_path('C:/Users/Administrator/.codex/skills/knowledge-cat-ppt-skill/scripts/probe_pptx_editability.py', run_name='__main__')
