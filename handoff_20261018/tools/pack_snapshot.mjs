import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {Writable} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {createZstdCompress,constants} from 'node:zlib';
const ROOT='E:/Work/强化学习小项目/handoff_20261018';
const PYTHON='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const PART=1024*1024*1024;
await fs.mkdir(path.join(ROOT,'archive'),{recursive:true});
class Splitter extends Writable{
  constructor(){super();this.part=0;this.bytes=0;this.total=0;this.files=[];this.handle=null;}
  async open(){this.part++;this.name=`qwop-workspace-20261018.tar.zst.part${String(this.part).padStart(3,'0')}`;this.handle=await fs.open(path.join(ROOT,'archive',this.name),'wx');this.bytes=0;this.hash=createHash('sha256');}
  async close(){if(!this.handle)return;await this.handle.close();this.files.push({name:this.name,bytes:this.bytes,sha256:this.hash.digest('hex')});console.log('PART',this.name,this.bytes);this.handle=null;await fs.writeFile(path.join(ROOT,'pack_progress.json'),JSON.stringify({parts:this.files,total_compressed_bytes:this.total},null,2));}
  _write(chunk,encoding,done){this.writeChunk(chunk).then(()=>done(),done);}
  async writeChunk(chunk){let i=0;while(i<chunk.length){if(!this.handle)await this.open();const take=Math.min(PART-this.bytes,chunk.length-i);const slice=chunk.subarray(i,i+take);let wrote=0;while(wrote<slice.length){const r=await this.handle.write(slice,wrote,slice.length-wrote);wrote+=r.bytesWritten;}this.hash.update(slice);this.bytes+=take;this.total+=take;i+=take;if(this.bytes===PART)await this.close();}}
  _final(done){this.close().then(()=>done(),done);}
}
const py=spawn(PYTHON,[path.join(ROOT,'tools/prepare_snapshot.py'),'tar'],{stdio:['ignore','pipe','inherit'],env:{...process.env,PYTHONIOENCODING:'utf-8'}});
const exited=new Promise((resolve,reject)=>{py.on('error',reject);py.on('exit',code=>code===0?resolve():reject(new Error('Tar producer exit '+code)))});
const z=createZstdCompress({params:{[constants.ZSTD_c_compressionLevel]:5,[constants.ZSTD_c_windowLog]:27,[constants.ZSTD_c_nbWorkers]:0,[constants.ZSTD_c_enableLongDistanceMatching]:1}});
const split=new Splitter();
await Promise.all([pipeline(py.stdout,z,split),exited]);
const sums=split.files.map(x=>`${x.sha256}  ${x.name}`).join('\n')+'\n';
await fs.writeFile(path.join(ROOT,'SHA256SUMS.txt'),sums);
console.log('PACK_COMPLETE',JSON.stringify({parts:split.files.length,compressed_bytes:split.total}));
