import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='E:/Work/强化学习小项目/presentation/qwop_5slides_template_20261015';
await fs.mkdir(path.join(root,'build/template-previews'),{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load('E:/PPT模板-先电.pptx'));
const ins=await p.inspect({kind:'slide,layout,textbox,shape,image',maxChars:50000});
await fs.writeFile(path.join(root,'build/template-inspect.ndjson'),ins.ndjson);
console.log('COUNT',p.slides.items.length,'MASTERS',p.masters.items.length);
for(const [i,s] of p.slides.items.entries()){
  const png=await p.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(root,`build/template-previews/slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
  const lay=await s.export({format:'layout'});
  await fs.writeFile(path.join(root,`build/template-previews/slide-${i+1}.json`),await lay.text());
  console.log('RENDERED',i+1,s.id);
}
