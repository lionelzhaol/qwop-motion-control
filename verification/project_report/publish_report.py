from pathlib import Path
import json,re,shutil,hashlib
BASE=Path(__file__).resolve().parent
target=Path(r'E:\LZ_vault\LioneZhao\未命名.md')
assert target.parent.resolve()==Path(r'E:\LZ_vault\LioneZhao').resolve()
source=BASE/'项目研究记录.md';body=source.read_text(encoding='utf-8')
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body)
assert len(images)==55
attachments=target.parent/'QWOP项目图表'
attachments.mkdir(exist_ok=True)
for relative in images:
    destination=target.parent/relative
    assert destination.parent.resolve()==attachments.resolve()
    image=BASE/'figures'/destination.name
    assert image.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    shutil.copy2(image,destination)
    assert hashlib.sha256(image.read_bytes()).digest()==hashlib.sha256(destination.read_bytes()).digest()
current=target.read_text(encoding='utf-8-sig') if target.exists() else ''
(BASE/'目标文档写入前备份.md').write_text(current,encoding='utf-8')
if current.strip():
    # Preserve any text the user may have added since the initial empty-file read.
    body=current.rstrip()+'\n\n---\n\n'+body
target.write_text(body,encoding='utf-8')
saved=target.read_text(encoding='utf-8')
assert saved==body and all((target.parent/p).exists() for p in images)
assert saved.count('```')%2==0
report=dict(document=str(target),bytes=target.stat().st_size,characters=len(saved),lines=len(saved.splitlines()),
    figure_count=len(images),attachments=str(attachments),preserved_preexisting_text=bool(current.strip()),
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),image_links_verified=True,
    png_copies_verified=True,training_code_modified=False)
(BASE/'publish_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
