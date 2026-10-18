import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';

const WORK='E:/Work/强化学习小项目';
const ROOT=path.join(WORK,'presentation/qwop_5slides_template_20261015');
const BUILD=path.join(ROOT,'build');
const SKILL='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const PYTHON='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const TEMPLATE='E:/PPT模板-先电.pptx';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT='微软雅黑';
const C={blue:'#064A8D',ink:'#1D2A38',muted:'#586778',line:'#D6E0EA',teal:'#11796E',white:'#FFFFFF'};
const revision=process.env.QWOP_PPT_REVISION||'v1';
const FINAL=path.join(ROOT,'output',`QWOP百米运动控制研究_先电模板_5页_${revision}.pptx`);
await fs.mkdir(path.join(ROOT,'output'),{recursive:true});
await fs.mkdir(path.join(BUILD,`previews_${revision}`),{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(TEMPLATE));
p.slides.keep([2]);
for(let i=0;i<4;i++)p.slides.items[0].duplicate();

const notes=[
`建议讲述时间：60秒。\n\n本次汇报介绍基于深度强化学习的 QWOP 百米运动控制研究。QWOP 通过 Q、W、O、P 四个按键控制腿部，动作需要连续协调，单帧的失误可能改变之后的身体状态并导致摔倒。我们研究的目标是在原生规则下稳定完成百米，同时缩短正式完赛用时。模型读取物理环境中的身体和任务状态，并通过按键组合控制运动。图片是原版 QWOP 的画面示意，便于解释玩法，图中的距离不是本项目训练成绩。这个游戏适合作为强化学习课程项目，因为它同时涉及平衡、探索以及较长时间跨度的动作协调。接下来先说明开源项目已经提供的基础，再说明本地 PPO 训练和专家模仿的具体工作。\n\n来源：\nhttps://github.com/LYiHub/QWOP_rl_training_public\nhttps://thecreativeindependent.com/people/bennett-foddy-on-the-pitfalls-of-perfectionism/\n游戏图片原始来源：https://s3.amazonaws.com/tci-assets/uploads/bennett-pic.jpg\n本地图片：presentation/qwop_class_report_20261013/assets/qwop-original.jpg`,
`建议讲述时间：75秒。\n\n本项目以 LYiHub 的 QWOP_rl_training_public 为基础。作者提供了 C++ 训练封装、原生物理后端、Python 接口以及多个任务，已有 PPO、Actor/Critic、GAE、并行采样、训练日志、checkpoint 和回放导出。因此本地工作重点是复用这些基础并开展可验证的实验。首先在 Windows 11 配置并验证 PyTorch/CUDA 训练环境，检查最小训练、reset 和模型保存加载。然后调整任务奖励、增加历史状态和角速度信息，尝试探索与局部策略参数搜索。最后根据公开专家动作回放实现行为克隆，并继续进行 PPO 微调。原仓库还提供策略蒸馏，但它需要教师模型和 logits；公开影子只有动作序列，所以新增了从回放采集状态动作样本的训练入口，复用了原来的 ActorCritic、帧堆叠、日志和 checkpoint。我们没有将作者已有的仿真或 PPO 实现作为本地原创。\n\n来源：\nhttps://github.com/LYiHub/QWOP_rl_training_public\nQWOP_rl_training_public/README.md\nQWOP_rl_training_public/scripts/train_ppo.py\nQWOP_rl_training_public/docs/shadow-imitation-pilot.md`,
`建议讲述时间：135秒。\n\n这一页说明 PPO 如何训练。第一步，环境返回 68 维身体与任务观测，后期策略堆叠 32 帧，形成 2176 维输入，使网络能够利用运动历史。主要 Actor 网络包含两层各 256 个神经元，并有角速度特征分支。Actor 输出 16 种按键组合的概率，训练时按概率采样，确定性评测时选最大概率动作。第二步，原生环境执行动作，得到新的状态、奖励和终止标志。第三步，多个并行环境采集一段 rollout，记录观测、动作概率、奖励及价值估计。这是 PPO 的在策略数据，不是经验回放池。第四步，Critic 估计长期回报，GAE 结合奖励和价值变化计算优势，告诉 Actor 哪些动作比预期更好。第五步，用 PPO 的裁剪目标限制策略更新幅度，更新网络后继续采样。网络结构和超参数在实验阶段有所变化，本页展示后期主要策略，而不是宣称每轮都采用相同配置。独立 Critic 用于后续保护已学步态的试验，基础闭环仍沿用作者架构。\n\n来源：\nQWOP_rl_training_public/scripts/train_ppo.py\nQWOP_rl_training_public/configs\nQWOP_rl_training_public/docs/bottleneck-followup-training.md\nQWOP_rl_training_public/docs/shadow-imitation-pilot.md`,
`建议讲述时间：150秒。\n\n左侧是 v33 起步探索阶段一次真实 PPO 续训的平均回合回报，来自 TensorBoard 记录。横轴为这一轮的 PPO 更新编号，纵轴是回合回报的 20 个记录点滚动均值。曲线仍有波动，不能据此宣称训练完全收敛。不同奖励版本的回报不能直接比较，因此这里只画同一次固定奖励配置的曲线。\n\n本地自主路线先处理低姿态运动，再尝试姿态、速度、历史状态、起步及越栏反馈和探索。没有原生膝盖触地标志，低姿态采用躯干高度代理，也没有直接测量身体重心。固定前倾五度、压低上下运动等偏好属于待检验假设，专家轨迹审计表明它们可能与最快完赛目标冲突。RND 使用预测误差提供新颖性奖励，但已完成的匹配实验没有刷新最快成绩。\n\n成绩必须准确区分：经过原生验证的纯 PPO 候选最快是 v33 的 45.400 秒。44.933334 秒则来自已有 44.966667 秒模型第二隐藏层偏置的小范围参数搜索，重新加载后可复现。它属于 PPO 训练加局部参数搜索的自主优化结果，不能标成纯 PPO 单独获得。两项均为模拟游戏时间，不是电脑运行耗时。我们保留原物理和正式计分规则。\n\n曲线来源：verification/project_report/events_data.json，run=ppo_phase_v33_effective_explore，series=charts/episode_return_mean。每次更新128环境×256步，起始global_step=348143616。\n成绩来源：QWOP_rl_training_public/docs/bottleneck-followup-training.md\nRND来源：QWOP_rl_training_public/docs/task-reward-rnd-experiment.md`,
`建议讲述时间：150秒。\n\n自主优化遇到瓶颈后，我们引入公开的 XYF 专家影子。影子文件包含 991 帧按键动作。先在原生环境中重放并验证，再在每次动作执行前采集当前观测，将其与专家动作配对，得到 991 条状态动作样本，每条输入为32×68=2176维。确认物理等价后再合并冗余按键标签。以原有跑步模型初始化 Actor，使用交叉熵行为克隆，Adam 学习率1e-4，batch256，训练1000轮，动作监督阶段冻结独立 Critic。推理阶段模型读取实时观测并选择动作，没有按时间查找专家动作表。\n\n闭环验证是关键。900轮时，专家状态上的动作准确率已达99.899%，但实际执行仍在52米失败。1000轮时准确率达到100%，策略才完成原生百米，正式时间33.033333秒。这说明分类准确率不能替代环境中的闭环测试。图中44.933与33.033均为固定初态下的确定性完赛时间，前者包含参数搜索。原始 BC 模型固定初态确定性重复16次均成功，但直接按其原始概率采样为0/64成功，说明轨迹外恢复能力不足。后续 PPO 微调、Actor 解冻与时间奖励试验仍未获得快于33.033秒的有效成绩。因此这里的成果是本地神经网络复现外部专家策略，并非独立发现或打破专家纪录。\n\n来源：QWOP_rl_training_public/docs/shadow-imitation-pilot.md\nverification/shadow_audit/imitation_pilot.json\nverification/shadow_audit/student_deterministic16.json\nverification/shadow_audit/student_stochastic64.json\nQWOP_rl_training_public/docs/classic-speed-signal-20261015.md\n公开示范来源：https://run.lyihub.com/，下载时署名XYF，不声明其为当前实时纪录。`
];
const titles=['QWOP 游戏简介与研究任务','开源项目提供的训练基础','PPO 强化学习训练流程','自主训练与策略优化结果','专家模仿的实现与结果'];
function txt(s,text,x,y,w,h,size=28,o={}){
  const q=s.shapes.add({geometry:'textbox',name:o.name||`content-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  q.text=text;
  q.text.style={fontSize:size,typeface:FONT,color:o.color||C.ink,bold:o.bold||false,alignment:o.align||'left',verticalAlignment:o.middle?'middle':'top',wrap:'square',autoFit:'none',insets:0,lineSpacing:1.12};
  return q;
}
function box(s,text,x,y,w,h,size=27){
  const a=s.shapes.add({geometry:'rect',name:`process-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:C.white,line:{fill:C.blue,width:1.6}});
  a.text=text;a.text.style={fontSize:size,typeface:FONT,color:C.blue,bold:true,alignment:'center',verticalAlignment:'middle',insets:10,wrap:'square',autoFit:'none'};return a;
}
function connect(s,a,b,from='right',to='left'){
  s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:C.blue,width:2},tail:{type:'triangle',width:'sm',length:'sm'}});
}
function chart(s,type,cfg){
 const c=s.charts.add(type,{chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},hasLegend:false,xAxis:{textStyle:{fill:C.muted,fontSize:20},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{textStyle:{fill:C.muted,fontSize:20},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}},...cfg});
 applyPresentationChartFont(c,{fontFamily:FONT});return c;
}
async function embed(s,file,x,y,w,h,alt){const b=await fs.readFile(file);s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/jpeg',position:{left:x,top:y,width:w,height:h},fit:'contain',alt});}
for(const [i,s] of p.slides.items.entries()){
 const title=s.shapes.items.find(q=>q.text?.toString().includes('XXX'))||s.placeholders.getItem('title');
 title.text=titles[i]; title.position={left:40,top:16,width:1085,height:57};
 title.text.style={fontSize:37.33,typeface:FONT,color:C.white,bold:true,alignment:'left',verticalAlignment:'middle',insets:0,wrap:'none',autoFit:'none'};
 s.speakerNotes.text=notes[i];
}

// 1: first page includes the game introduction, as requested; no extra cover.
{
 const s=p.slides.items[0];
 txt(s,'基于深度强化学习的运动控制与策略优化',50,115,1165,51,32,{bold:true,color:C.blue});
 await embed(s,path.join(WORK,'presentation/qwop_class_report_20261013/assets/qwop-original.jpg'),50,199,655,409,'原版QWOP游戏画面示意，非本项目训练成绩');
 const rows=[['四键控制双腿','Q / W / O / P 组成 16 类动作'],['目标：稳定完成百米','在原生计分规则下缩短完赛用时'],['难点：全身连续协调','平衡、推进与越栏需要连续配合']];
 rows.forEach(([a,b],i)=>{txt(s,a,755,214+i*139,455,47,30,{bold:true,color:C.blue});txt(s,b,755,271+i*139,455,67,25);});
 txt(s,'原版 QWOP 游戏示意，图中数值不代表本项目成绩',50,619,690,30,19,{color:C.muted});
 txt(s,'Lionel_Zhao    课堂项目汇报',50,669,900,30,21,{color:C.muted});
}
// 2: native comparison table.
{
 const s=p.slides.items[1];
 txt(s,'复用作者架构，在任务奖励与策略学习上开展实验',50,117,1170,48,30,{bold:true,color:C.blue});
 const values=[['工作层次','开源项目已提供','本地完成的工作'],['仿真环境','原生物理后端、Python 接口\n百米、跨栏与跳远任务','Windows 11 / PyTorch / CUDA\n跑通训练、重置和模型加载'],['训练算法','PPO、Actor / Critic、GAE\n并行采样与策略蒸馏','奖励迭代、历史状态与探索\nPPO 微调及局部参数搜索'],['实验工具','训练日志与 checkpoint\n评估、模型加载及回放导出','公开回放采集与行为克隆\n原生闭环验证及结果归档']];
 const t=s.tables.add({rows:4,columns:3,left:50,top:202,width:1170,height:365,columnWidths:[190,475,505],values});
 t.borders.assign({fill:C.line,width:1});t.styleOptions={headerRow:false,bandedRows:false};
 for(let r=0;r<4;r++){t.rows[r].height=r===0?65:100;for(let c=0;c<3;c++){const a=t.getCell(r,c);a.fill=r===0?C.blue:C.white;a.text.style={fontSize:r===0?27:25,typeface:FONT,color:r===0?C.white:C.ink,bold:r===0||c===0,verticalAlignment:'middle',alignment:'left',wrap:'square',autoFit:'none',insets:16};}}
 txt(s,'项目来源：LYiHub / QWOP_rl_training_public',50,603,1170,40,24,{color:C.blue});
 txt(s,'保留作者的 PPO、采样、日志与模型保存流程',50,656,1170,37,23,{color:C.muted});
}
// 3: requested editable training-loop diagram.
{
 const s=p.slides.items[2];
 txt(s,'通过环境交互采样，再用价值估计更新动作策略',50,117,1170,48,30,{bold:true,color:C.blue});
 const obs=box(s,'身体状态\n68 维 × 32 帧',65,219,255,112,27);
 const act=box(s,'Actor 策略网络\n两层 256，输出 16 类',430,219,352,112,27);
 const env=box(s,'原生环境执行\n奖励、新状态、终止',891,219,317,112,27);
 const roll=box(s,'Rollout 数据采集\n并行环境的连续交互',891,427,317,112,27);
 const upd=box(s,'Critic 与 GAE 估计优势\nPPO 裁剪目标更新策略',430,427,352,112,27);
 connect(s,obs,act);connect(s,act,env);connect(s,env,roll,'bottom','top');connect(s,roll,upd,'left','right');connect(s,upd,act,'top','bottom');
 txt(s,'按键动作',793,179,90,36,20,{color:C.muted});
 txt(s,'更新策略',656,359,150,35,21,{color:C.muted});
 txt(s,'Actor 决定怎样动作，Critic 评估长期回报',65,589,1130,43,29,{bold:true,color:C.blue});
 txt(s,'GAE 分配动作优势，PPO 限制单次策略变化，训练后保存 checkpoint',65,646,1140,42,24,{color:C.muted});
}
// 4: actual fixed-reward PPO continuation, native editable scatter chart.
const events=JSON.parse(await fs.readFile(path.join(WORK,'verification/project_report/events_data.json'),'utf8'));
const run=events.find(x=>x.name==='ppo_phase_v33_effective_explore');
const r=run.series['charts/episode_return_mean'];
const rollout=Number(run.config.num_envs)*Number(run.config.rollout_steps);
const base=r[0].step-rollout;
const points=r.map((q,i)=>({x:(q.step-base)/rollout,y:Number((r.slice(Math.max(0,i-19),i+1).reduce((a,b)=>a+b.value,0)/Math.min(i+1,20)).toFixed(6))})).filter((q,i)=>i%16===0||i===r.length-1);
await fs.writeFile(path.join(BUILD,'ppo_curve_data.json'),JSON.stringify({run:run.name,source:'verification/project_report/events_data.json',base_global_step:base,rollout,smoothing:20,points},null,2));
{
 const s=p.slides.items[3];
 txt(s,'起步探索阶段：平均回合回报',50,126,760,45,29,{bold:true,color:C.blue});
 chart(s,'scatter',{position:{left:43,top:190,width:745,height:347},series:[{name:'回合回报，20点滚动均值',xValues:points.map(x=>x.x),values:points.map(x=>x.y),line:{fill:C.blue,width:3},marker:{symbol:'none'}}],scatterOptions:{style:'line'},xAxis:{min:0,max:1024,majorUnit:256,tickLabelPosition:'low',position:'bottom',title:{text:'本轮 PPO 更新数',textStyle:{fontSize:21,fill:C.muted}},numberFormatCode:'0',textStyle:{fontSize:20,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min:-100,max:900,majorUnit:250,tickLabelPosition:'nextTo',numberFormatCode:'0',textStyle:{fontSize:20,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.line,width:1}}});
 txt(s,'纯 PPO 最好候选',850,143,350,43,27,{color:C.muted});
 txt(s,'45.400 秒',846,199,365,83,57,{bold:true,color:C.blue});
 txt(s,'局部参数搜索后',850,333,350,43,27,{color:C.muted});
 txt(s,'44.933 秒',846,389,365,83,57,{bold:true,color:C.teal});
 txt(s,'均通过原生环境完赛验证',850,494,365,42,22,{color:C.muted});
 txt(s,'真实 v33 单次续训，20 个记录点滚动均值',50,543,755,32,19,{color:C.muted});
 txt(s,'调整进度、低姿态与完赛反馈，结合历史状态和起步探索',50,600,1170,39,27,{bold:true,color:C.blue});
 txt(s,'RND 对照未刷新最快成绩，固定姿态偏好仍需实验检验',50,654,1170,38,24,{color:C.muted});
}
// 5: demonstration-to-policy mechanism and attributed result.
{
 const s=p.slides.items[4];
 const steps=[['公开 XYF 回放','991 帧按键动作'],['原生重放采样','状态与动作对齐'],['行为克隆训练','交叉熵更新 Actor'],['闭环验证','实时观测决定动作']];
 const a=steps.map(([x,y],i)=>box(s,`${x}\n${y}`,50+i*300,141,270,110,25));
 for(let i=0;i<3;i++)connect(s,a[i],a[i+1]);
 txt(s,'固定初态下的确定性百米用时',50,300,665,43,28,{bold:true,color:C.blue});
 chart(s,'bar',{position:{left:42,top:364,width:673,height:244},categories:['自主优化','专家模仿'],series:[{name:'百米用时（秒）',values:[44.933334,33.033333],fill:C.blue,points:[{idx:1,fill:C.teal}]}],barOptions:{direction:'column',gapWidth:140},xAxis:{textStyle:{fontSize:24,fill:C.ink},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min:0,max:55,majorUnit:20,numberFormatCode:'0',textStyle:{fontSize:20,fill:C.muted},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:25,fill:C.ink,bold:true}}});
 txt(s,'33.033 秒',780,333,432,91,62,{bold:true,color:C.teal});
 txt(s,'1000 轮行为克隆后完成百米',784,439,425,42,25,{color:C.ink});
 txt(s,'固定初态确定性复测：16 / 16 成功',784,500,425,69,24,{color:C.ink});
 txt(s,'自主优化含 PPO 后的局部参数搜索',50,613,681,32,19,{color:C.muted});
 txt(s,'公开专家示范的本地复现，后续 PPO 微调尚未突破该时间',50,661,1175,37,24,{bold:true,color:C.blue});
}

const sourceHash=createHash('sha256').update(await fs.readFile(TEMPLATE)).digest('hex');
const draft=path.join(BUILD,`draft_${revision}.pptx`);
await(await PresentationFile.exportPptx(p)).save(draft);
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:draft,finalPath:FINAL,pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:5,requiredNativeChartOwnerSlides:[4,5],requiredNativeTableOwnerSlides:[2],materializeLiteralChartWorkbooks:true,layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2'],fontPolicy:{basis:'reference',families:['微软雅黑','等线'],referencePath:TEMPLATE,referenceSha256:sourceHash},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,`validation_${revision}.json`)});
const final=await PresentationFile.importPptx(await FileBlob.load(FINAL));
for(const [i,s] of final.slides.items.entries()){
 const img=await final.export({slide:s,format:'png',scale:1.5});
 await fs.writeFile(path.join(BUILD,`previews_${revision}/slide-${i+1}.png`),new Uint8Array(await img.arrayBuffer()));
 const lay=await s.export({format:'layout'});
 await fs.writeFile(path.join(BUILD,`previews_${revision}/slide-${i+1}.json`),await lay.text());
}
const montage=await final.export({format:'png',montage:{columns:2,slideWidth:640,gap:12,padding:12}});
await fs.writeFile(path.join(ROOT,'output',`五页预览_${revision}.png`),new Uint8Array(await montage.arrayBuffer()));
await fs.writeFile(path.join(ROOT,'output','逐页演讲备注.md'),titles.map((t,i)=>`## 第 ${i+1} 页：${t}\n\n${notes[i].split('\n\n来源：')[0].split('\n\n曲线来源：')[0]}\n`).join('\n'));
console.log('FINAL',FINAL,'SLIDES',final.slides.items.length,'RESULT',JSON.stringify(result).slice(0,400));
