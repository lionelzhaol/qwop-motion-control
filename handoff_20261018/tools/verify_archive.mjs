import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import {Readable} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {createZstdDecompress} from 'node:zlib';
import {spawn} from 'node:child_process';
const ROOT='E:/Work/强化学习小项目/handoff_20261018';
const parts=(await fsp.readdir(path.join(ROOT,'archive'))).filter(x=>/^qwop-workspace-20261018\.tar\.zst\.part\d+$/.test(x)).sort();
async function* input(){for(const p of parts){for await(const b of fs.createReadStream(path.join(ROOT,'archive',p)))yield b;}}
const py=spawn('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',[path.join(ROOT,'tools/verify_archive.py')],{stdio:['pipe','inherit','inherit']});
const done=new Promise((resolve,reject)=>{py.on('error',reject);py.on('exit',x=>x===0?resolve():reject(new Error('verification failed '+x)))});
await Promise.all([pipeline(Readable.from(input()),createZstdDecompress(),py.stdin),done]);
