import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);
const WORK = 'E:/Work/强化学习小项目';
const SKILL = 'C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const PYTHON = 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;
const { finalizePresentation, applyPresentationChartFont, resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT = resolvePresentationFont({fontFamily:'Microsoft YaHei'});
const plan = JSON.parse(await fs.readFile(path.join(HERE,'deck-plan.json'),'utf8'));
const events = JSON.parse(await fs.readFile(path.join(WORK,'verification/project_report/events_data.json'),'utf8'));
const p = Presentation.create({slideSize:{width:1280,height:720}});
const C = {bg:'#FAFBFC',ink:'#16243A',blue:'#225B85',teal:'#078578',muted:'#556578',line:'#D9E1E8',light:'#EEF3F7',amber:'#9A5C12',white:'#FFFFFF'};
const previews = path.join(HERE,'previews');
await fs.mkdir(previews,{recursive:true});
const revision = process.env.QWOP_DECK_REVISION || 'v3';
const finalPath = path.join(ROOT,'output',`QWOP运动控制与策略优化研究_课堂汇报_${revision}.pptx`);

function txt(s,text,x,y,w,h,size=28,opt={}) {
  const q=s.shapes.add({geometry:'textbox',name:opt.name||`text-${s.shapes.items.length+1}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  q.text=text;
  q.text.style={fontSize:size,typeface:FONT,color:opt.color||C.ink,bold:opt.bold||false,alignment:opt.align||'left',verticalAlignment:'top',wrap:'square',autoFit:'none',insets:0,lineSpacing:1.12};
  return q;
}
function newSlide(i) {
  const s=p.slides.add();s.background.fill=C.bg;
  if(i!==1)txt(s,plan.slides[i-1].action_title,72,54,1136,88,42,{bold:true,name:'slide-title'});
  txt(s,String(i).padStart(2,'0'),1185,671,45,26,18,{color:C.muted,align:'right',name:'page-number'});
  const item=plan.slides[i-1];
  s.speakerNotes.text=`建议讲述时间：${item.duration_seconds}秒\n\n${item.speaker_note}\n\n来源（不朗读）：\n${item.source_refs.join('\n')}`;
  return s;
}
function list(s,items,x,y,w,size=28,gap=98) {
  items.forEach((item,i)=>{
    txt(s,String(i+1).padStart(2,'0'),x,y+i*gap,44,34,20,{color:C.teal,bold:true});
    txt(s,item,x+58,y+i*gap-2,w-58,gap-12,size);
  });
}
function chart(s,type,cfg) {
  const ch=s.charts.add(type,{
    chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    hasLegend:false,
    xAxis:{textStyle:{fill:C.muted,fontSize:19},line:{fill:C.line,width:1},majorGridlines:null},
    yAxis:{min:0,textStyle:{fill:C.muted,fontSize:18},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},
    ...cfg,
    ...(cfg.dataLabels?{dataLabels:{...cfg.dataLabels,visible:true}}:{})
  });
  applyPresentationChartFont(ch,{fontFamily:FONT});
  return ch;
}
function table(s,values,x,y,w,h,widths,font=25) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
  t.borders.assign({fill:C.line,width:1,style:'solid'});
  t.styleOptions={headerRow:false,bandedRows:false};
  for(let r=0;r<values.length;r++){
    t.rows[r].height=r===0?54:(h-54)/(values.length-1);
    for(let c=0;c<values[0].length;c++) {
      const cell=t.getCell(r,c);cell.fill=r===0?C.blue:C.white;
      cell.text.style={fontSize:r===0?font-2:font,typeface:FONT,color:r===0?C.white:C.ink,bold:r===0,verticalAlignment:'middle',alignment:c===0?'left':'center',wrap:'square',autoFit:'none',insets:12};
    }
  }
  return t;
}
async function embed(s,file,x,y,w,h,alt) {
  const bytes=await fs.readFile(file);
  s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:file.endsWith('.jpg')?'image/jpeg':'image/png',position:{left:x,top:y,width:w,height:h},fit:'contain',alt});
}

// 1: cover
{
 const s=newSlide(1);
 txt(s,'基于深度强化学习的',72,152,1136,75,58,{bold:true,color:C.blue});
 txt(s,'QWOP 运动控制',72,247,1136,91,68,{bold:true});
 txt(s,'与策略优化研究',72,344,1136,91,68,{bold:true});
 txt(s,'百米提速与跳远迁移',75,475,1040,54,30,{color:C.muted});
 txt(s,'Lionel_Zhao  ·  课堂项目汇报',75,596,950,42,26,{color:C.blue});
}
// 2: task and illustration, original image preserved
{
 const s=newSlide(2);
 list(s,plan.slides[1].bullets,72,200,470,26,103);
 await embed(s,path.join(ROOT,'assets/qwop-original.jpg'),560,185,648,405,'原版QWOP游戏示意，非本项目训练结果');
 txt(s,'原版 QWOP 画面示意（非本项目成绩截图）',562,610,646,33,19,{color:C.muted});
}
// 3: text based workflow, no decorative diagram assets
{
 const s=newSlide(3);
 const stages=[['01','状态处理','固定缩放\n历史堆帧'],['02','环境交互','并行C++环境\n原生物理与计分'],['03','策略更新','rollout 与 GAE\nPPO 小批量更新'],['04','记录验证','日志与模型保存\n独立评测与回放']];
 stages.forEach((a,i)=>{let x=72+i*285;txt(s,a[0],x,209,220,62,42,{color:C.teal,bold:true});txt(s,a[1],x,294,245,51,32,{bold:true});txt(s,a[2],x,365,250,106,27,{color:C.muted});});
 txt(s,'Python / PyTorch 负责学习，C++ 环境与物理 DLL 提供交互',72,530,1136,54,30,{color:C.blue});
 txt(s,'入口 train_ppo.py   ·   环境 training_env.cpp   ·   奖励 reward_profiles.cpp',72,603,1136,40,20,{color:C.muted});
}
// 4: method
{
 const s=newSlide(4);
 txt(s,'Actor',72,193,490,65,44,{bold:true,color:C.blue});
 txt(s,'身体状态决定按键概率\n16类离散动作分布',72,278,490,108,29);
 txt(s,'Critic',690,193,490,65,44,{bold:true,color:C.teal});
 txt(s,'预测当前状态的未来回报\n辅助GAE估计动作优势',690,278,490,108,29);
 txt(s,'PPO 裁剪目标',72,427,1040,45,30,{bold:true});
 txt(s,'Lclip = E[min(ρₜ Âₜ, clip(ρₜ, 1−ε, 1+ε) Âₜ)]',72,493,1136,61,34,{color:C.blue});
 txt(s,'概率比 ρₜ 衡量新旧策略变化；优势 Âₜ 表示动作相对预期的好坏',72,578,1136,53,24,{color:C.muted});
}
// 5: original baseline and reward curve from one real run
{
 const s=newSlide(5);
 txt(s,'0 / 64',72,181,400,100,76,{bold:true,color:C.blue});
 txt(s,'原奖励随机评测通关数',74,286,405,50,25,{color:C.muted});
 txt(s,'低姿态前进、输入尺度失效\n促使我们调整奖励与观测',72,386,419,91,27);
 txt(s,'v3：38 / 64 通关\n确定性用时 75.27 秒',72,520,419,90,28,{bold:true,color:C.teal});
 const run=events.find(x=>x.name==='ppo_foot_running_v3');
 const rewards=run.series['charts/episode_return_mean'];
 // Six decimals retain far more detail than the projected curve can show and
 // provide exactly representable literal values for the chart workbook.
 const smooth=rewards.map((q,i)=>({x:(q.step-6144)/32768,y:Number((rewards.slice(Math.max(0,i-19),i+1).reduce((a,b)=>a+b.value,0)/(Math.min(20,i+1))).toFixed(6))}));
 const sampled=smooth.filter((_,i)=>i%16===0||i===smooth.length-1);
 txt(s,'v3 单版本回合回报曲线',533,175,663,43,27,{bold:true});
 chart(s,'scatter',{position:{left:520,top:242,width:692,height:350},series:[{name:'20点滚动均值',xValues:sampled.map(x=>x.x),values:sampled.map(x=>x.y),line:{fill:C.teal,width:3},marker:{symbol:'none'}}],scatterOptions:{style:'line'},xAxis:{min:0,max:512,majorUnit:128,position:'bottom',tickLabelPosition:'low',title:{text:'训练更新',textStyle:{fontSize:19,fill:C.muted}},textStyle:{fontSize:18,fill:C.muted},majorGridlines:null,line:{fill:C.line,width:1}},yAxis:{min:-100,max:200,majorUnit:100,crossesAt:-100,textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.line,width:1},line:{fill:'none',width:0}}});
 txt(s,'同一奖励版本，20点滚动均值；跨版本回报不可直接比较',533,623,675,32,18,{color:C.muted});
 await fs.writeFile(path.join(HERE,'reward_curve_data.json'),JSON.stringify({run:run.name,smoothing:20,sampling:'every 16 recorded points, and last point',points:sampled},null,2));
}
// 6: verified pure PPO candidates
{
 const s=newSlide(6);
 txt(s,'纯 PPO 成功候选用时（秒，越低越好）',72,169,890,45,27,{bold:true});
 chart(s,'bar',{position:{left:60,top:246,width:823,height:350},categories:['低姿态优化','姿态与速度','50秒阶段','起步与探索'],series:[{name:'原生百米用时',values:[75.27,65.63,50.17,45.40],fill:C.blue,points:[{idx:3,fill:C.teal}]}],barOptions:{direction:'column',grouping:'clustered',gapWidth:105},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:24,fill:C.ink,bold:true}},yAxis:{min:0,max:85,majorUnit:20,textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.line,width:1},line:{fill:'none',width:0}}});
 txt(s,'45.40 秒',929,248,282,79,48,{bold:true,color:C.teal});
 txt(s,'纯PPO最好已验证候选',931,337,280,70,24,{color:C.muted});
 txt(s,'44.93 秒',929,450,280,79,48,{bold:true,color:C.blue});
 txt(s,'Actor偏置局部搜索后',931,539,275,70,24,{color:C.muted});
 txt(s,'阶段改进：姿态与速度反馈、32帧历史、起步及越栏反馈；43秒目标未达成',72,632,1136,29,18,{color:C.muted});
}
// 7: audit ranking within each reward version
{
 const s=newSlide(7);
 txt(s,'起步与越栏恢复',72,195,345,48,31,{bold:true,color:C.blue});
 txt(s,'10米约需6.9秒\n越栏后存在速度损失',72,267,334,91,27);
 txt(s,'姿态规则是研究假设',72,402,345,48,29,{bold:true,color:C.teal});
 txt(s,'5°前倾、步频和平稳度\n未必对应更快的成功策略',72,472,340,100,26);
 const defs=[['旧奖励',[1450.95,1877.42],435],['时间导向任务奖励',[2300.67,2055.80],838]];
 for(const [title,values,x] of defs){
  txt(s,title,x,174,367,42,26,{bold:true});
  chart(s,'bar',{position:{left:x-8,top:238,width:376,height:361},categories:['33.03秒示范','44.93秒策略'],series:[{name:'累计回报',values,fill:C.blue,points:[{idx:0,fill:C.teal}]}],barOptions:{direction:'column',gapWidth:82},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0',textStyle:{fontSize:21,fill:C.ink,bold:true}},xAxis:{textStyle:{fontSize:18,fill:C.muted},majorGridlines:null,line:{fill:C.line,width:1}},yAxis:{min:0,max:2600,majorUnit:1000,textStyle:{fontSize:16,fill:C.muted},majorGridlines:{fill:C.line,width:1},line:{fill:'none',width:0}}});
 }
 txt(s,'只比较每种奖励内部的排序；审计对象为两条成功轨迹',435,623,780,38,20,{color:C.muted});
}
// 8: matched RND evidence
{
 const s=newSlide(8);
 txt(s,'新颖性来自预测误差',72,193,398,55,32,{bold:true,color:C.teal});
 txt(s,'随机固定目标网络\n预测器学习拟合\n误差提供内在奖励',72,278,377,161,28);
 txt(s,'两组同起点、同训练量\n各约1678万条新转移',72,507,377,87,25,{color:C.muted});
 table(s,[['指标','任务奖励','任务奖励 + RND'],['随机通关数','57 / 64','61 / 64'],['成功平均用时','49.26 秒','48.84 秒'],['最快保存点用时','46.80 秒','47.23 秒']],482,204,727,333,[279,215,233],24);
 txt(s,'部分成功率改善，未刷新旧44.93秒纪录；只有一个训练种子',482,578,727,73,25,{bold:true,color:C.blue});
}
// 9: BC has closed-loop failure under tiny deviation
{
 const s=newSlide(9);
 txt(s,'33.03 秒',72,179,487,105,78,{bold:true,color:C.teal});
 txt(s,'本地网络，固定初态闭环执行',76,296,482,57,27,{color:C.muted});
 txt(s,'公开冠军XYF的991帧示范\n状态动作监督，交叉熵训练Actor',72,394,465,103,27);
 txt(s,'确定性16 / 16成功\n原始随机采样0 / 64成功',72,547,467,95,29,{bold:true,color:C.blue});
 table(s,[['训练轮次','动作准确率','闭环结果'],['900','99.899%','52米失败'],['1000','100%','33.03秒完成']],562,211,646,246,[170,205,271],25);
 txt(s,'极小动作偏差也会累积\n偏离轨迹后的恢复仍是瓶颈',564,519,633,115,30,{bold:true,color:C.ink});
}
// 10: transfer and officially scored distances
{
 const s=newSlide(10);
 txt(s,'校准位置字段原点',72,203,422,55,33,{bold:true,color:C.blue});
 txt(s,'25米、334帧助跑动作一致\n同为68维，语义仍需核对',72,288,419,110,27);
 txt(s,'冻结助跑，局部学习起跳',72,465,442,54,31,{bold:true,color:C.teal});
 txt(s,'起跳区域调整层 + 独立Critic',72,549,426,76,26);
 table(s,[['跳远路线','确定性','随机最佳'],['旧PPO迁移','2.60米','2.80米\n1024次搜索'],['保留33秒助跑','2.59米','3.16米\n2048次搜索']],548,201,660,304,[244,171,245],25);
 txt(s,'新模型独立64次：51次成功，成功平均1.51米\n搜索规模不同，最佳值不能代表平均能力',550,553,660,102,23,{color:C.muted});
}
// 11: differentiated outcome matrix
{
 const s=newSlide(11);
 table(s,[['方法路线','已验证成绩','评测与结论边界'],['纯 PPO','45.40 秒','自主训练的成功候选'],['PPO + 偏置搜索','44.93 秒','包含局部权重搜索'],['公开示范 BC','33.03 秒','固定初态复现，原始随机0/64'],['跳远迁移','2.59 / 3.16 米','确定性 / 2048次随机最佳']],72,191,1136,372,[334,241,561],26);
 txt(s,'原生成绩、成功率与随机均值分开报告；固定初态复现尚不证明泛化',72,610,1136,46,25,{color:C.blue,bold:true});
}
// 12: research plan, future items explicitly labeled
{
 const s=newSlide(12);
 list(s,plan.slides[11].bullets,72,186,1136,30,97);
 txt(s,'奖励对齐目标，原生评测检验进步',72,620,1136,47,33,{bold:true,color:C.blue});
}

const md=[`# ${plan.deck.title}`,`\n课堂汇报：12页，计划600秒（约10分钟）。每页不超过5个要点。成果口径依据截至2026-10-11的本地记录。`, '\n## 整体结构\n', '| 页 | 内容 | 时间 |','|---|---|---|',...plan.slides.map(s=>`| ${s.number} | ${s.action_title.replaceAll('\n','')} | ${s.duration_seconds}秒 |`),'\n## 分页内容与演讲备注\n'];
for(const s of plan.slides) {
 md.push(`### 第${s.number}页：${s.action_title.replaceAll('\n','')}`,`\n建议讲述：${s.duration_seconds}秒。\n`,...s.bullets.map(x=>`- ${x}`),`\n**视觉建议：** ${s.visual}\n`,`**演讲备注：**\n\n${s.speaker_note}\n`,'**来源：**\n',...s.source_refs.map(x=>`- ${x}`),'');
}
md.push('\n## 成绩口径\n','- 45.40秒为已核实的纯PPO快照成绩，44.93秒包含Actor偏置参数搜索。','- 33.03秒来自本地模型模仿公开冠军XYF示范，不能表述为自主PPO探索出的纪录。','- 3.16米来自2048次随机运行的最佳回放，与旧1024次搜索规模不同。','- 单版本奖励曲线与不同版本的回报不可直接比较。','- 固定初态确定性复跑验证一致性，不能替代扰动与多初态泛化实验。');
await fs.writeFile(path.join(ROOT,'output','QWOP课堂汇报_分页内容与演讲备注.md'),md.join('\n'),'utf8');
await fs.writeFile(path.join(HERE,'presentation.proto.json'),JSON.stringify(p.toProto()),'utf8');
await fs.writeFile(path.join(HERE,'inspection.ndjson'),(await p.inspect({kind:'slide,textbox,chart,table,notes',maxChars:100000})).ndjson,'utf8');
const draftPath=path.join(HERE,`draft_${revision}.pptx`);
await(await PresentationFile.exportPptx(p)).save(draftPath);
for(const [i,s] of p.slides.items.entries()){
 const blob=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(previews,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
 const l=await s.export({format:'layout'});await fs.writeFile(path.join(previews,`slide-${String(i+1).padStart(2,'0')}.layout.json`),await l.text());
}
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:draftPath,finalPath,explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[8,9,10,11],requiredNativeChartOwnerSlides:[5,6,7],materializeLiteralChartWorkbooks:true,pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[8,9,10,11].flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(HERE,`validation_${revision}.json`)});
console.log(JSON.stringify({finalPath,slideCount:12,bytes:result.byteCount,sha256:result.finalSha256,package:result.packageIntegrity.status,layout:result.presentationLayout.findingCount,nativeCharts:result.nativeChartValidation.passed}));
// Final-file renders check the exported package, including chart workbooks.
const imported=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const finalPreviews=path.join(HERE,`final_previews_${revision}`);await fs.mkdir(finalPreviews,{recursive:true});
for(const [i,s] of imported.slides.items.entries()){
 const blob=await imported.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(finalPreviews,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(JSON.stringify({renderedFinalSlides:imported.slides.items.length,finalPreviews}));
