import assert from 'node:assert/strict';
import fs from 'node:fs';
import {calculate,defaults,storage,GiB} from '../../dist/supernode/engine.mjs';
const m=JSON.parse(fs.readFileSync(new URL('../../dist/supernode/models.json',import.meta.url)))[0];
const calc=(a={},model=m)=>calculate(model,{...defaults,...a});
const close=(a,b)=>assert.ok(Math.abs(a-b)<=Math.max(1e-8,Math.abs(b)*1e-10),`${a} != ${b}`);
assert.equal(m.total,2779931837184);assert.equal(m.routed,2722740830208);assert.equal(m.active,104175425536);
close(storage(32,'MXFP4'),17);close(storage(16,'NVFP4',0,1),13);close(storage(128,'INT4'),68);close(storage(128,'INT8'),132);close(storage(64,'NF4'),34);
// Single rank baseline: Adam BF16 + grad2 + master4 + moments8 = 16 bytes/parameter.
let r=calc({phase:'pretrain',tp:1,dp:1,ep:1,etp:1,sp:1,imbalance:1,zero:0,format:'BF16'});
assert.deepEqual(r.errors,[]);close((r.weightsGiB+r.gradientGiB+r.masterGiB+r.optimizerGiB)*GiB,m.total*16);
assert.equal(r.moeBytes,0);assert.equal(r.tpBytes,0);assert.equal(r.dpBytes,0);assert.equal(r.kvGiB,0);assert.equal(r.kdaGiB,0);
// Independent ZeRO3 expectation: all model states uniquely partition over 512 ranks.
r=calc({phase:'fullft',tp:8,dp:64,ep:64,etp:1,sp:8,zero:3,imbalance:1});
close((r.weightsGiB+r.gradientGiB+r.masterGiB+r.optimizerGiB)*GiB,m.total*16/512);
let s=calc({phase:'fullft',tp:8,dp:64,ep:64,etp:1,sp:8,zero:3,imbalance:1,format:'BF16'});
close(r.totalGiB,s.totalGiB); // Low-bit selector does NOT erase full-training states.
for(const key of ['tp','pp','dp','ep','etp','domain','cacheShards'])assert.ok(calc({[key]:0}).errors.length,key);
assert.ok(calc({tp:5}).errors.length);assert.ok(calc({tp:8,dp:8,ep:128}).errors.length);assert.ok(calc({phase:'qlora',format:'FP8'}).errors.length);
assert.ok(calc({}, {...m,latentKvWidth:0.5}).errors.some(e=>e.includes('缓存布局宽度')));
assert.ok(calc({reserve:1}).errors.length);assert.ok(calc({intraEff:0}).errors.length);assert.ok(calc({}, {...m,kdaLayers:null}).errors.length);assert.ok(calc({etp:7,dp:56}).errors.some(e=>e.includes('专家中间维度')));
assert.ok(calc({hbmGiB:null}).errors.length);assert.equal(calc({interGBps:null,epCross:.5}).modeledFloor,null);
for(const key of ['tflops','hbmGBps','intraGBps','interGBps'])for(const value of [Infinity,-Infinity,NaN,-1])assert.ok(calc({[key]:value}).errors.length,`${key}=${value}`);
for(const key of ['tflops','hbmGBps','intraGBps','interGBps'])assert.ok(calc({[key]:null}).errors.length===0,`${key}=null may mean unknown`);
// Cache independently recomputed: HF expanded K/V vs fixed vLLM NVIDIA latent layout.
r=calc();close(r.kvGiB*GiB,32769*24*30720*2*1.02);close(r.kdaGiB*GiB,69*96*128*128*4);assert.equal(r.kvWidthUsed,30720);assert.equal(r.maxMlaLayers,24);assert.equal(r.maxKdaLayers,69);
const pp3=calc({pp:3});assert.deepEqual(pp3.cacheStageCounts.map(v=>[v.mla,v.layers-v.mla]),[[7,24],[8,23],[9,22]]);assert.equal(pp3.maxMlaLayers,9);assert.equal(pp3.maxKdaLayers,24);const pp93=calc({pp:93});assert.equal(pp93.maxMlaLayers,1);assert.equal(pp93.maxKdaLayers,1);const changedMix=calc({pp:3},{...m,mlaLayers:20,kdaLayers:73});assert.equal(changedMix.maxMlaLayers,20);assert.equal(changedMix.maxKdaLayers,31);assert.equal(changedMix.cacheStageCounts,null);
const special=calc({cacheLayout:'vllm-fp8-ds-mla'});close(special.kvGiB*GiB,32769*24*656*1.02);assert.equal(special.kvWidthUsed,'656 B special layout');
const latent=calc({cacheLayout:'vllm-latent'});close(latent.kvGiB*GiB,32769*24*576*2*1.02);assert.equal(latent.kvWidthUsed,576);
s=calc({context:65536});close(s.kvGiB,r.kvGiB*65537/32769);close(s.kdaGiB,r.kdaGiB);
const emptyPrefill=calc({phase:'prefill',context:0,tokens:2048});close(emptyPrefill.peakCacheTokens,2048);close(emptyPrefill.effectiveNewTokens,2048);const emptyDecode=calc({phase:'decode',context:0,tokens:2048});close(emptyDecode.peakCacheTokens,1);close(emptyDecode.effectiveNewTokens,1);
const zeroEach=calc({phase:'pretrain',zero:3,accum:4,gradSync:'per_micro'}),zeroDeferred=calc({phase:'pretrain',zero:3,accum:4,gradSync:'deferred'});assert.ok(zeroDeferred.gradientGiB>zeroEach.gradientGiB);close(zeroEach.dpBytes,4*calc({phase:'pretrain',zero:3,accum:1}).dpBytes);close(zeroDeferred.dpBytes,calc({phase:'pretrain',zero:3,accum:1,gradSync:'deferred'}).dpBytes);
const f8=calc({actFormat:'FP8-E4M3',pp:2}),save8=calc({savedActBytes:1}),comm8=calc({commBytes:1,pp:2});close(f8.activationGiB,calc({pp:2}).activationGiB);close(f8.ppBytes,calc({pp:2}).ppBytes);close(save8.activationGiB*2,calc().activationGiB);close(comm8.ppBytes*2,calc({pp:2}).ppBytes);
const w4=calc({format:'MXFP4'}),w8=calc({format:'FP8-E4M3'}),w16=calc({format:'BF16'});assert.ok(w4.weightsGiB<w8.weightsGiB&&w8.weightsGiB<w16.weightsGiB);
const l=calc({phase:'lora',format:'BF16',adapterB:0}),q=calc({phase:'qlora',format:'INT4',adapterB:0});assert.equal(l.optimizerGiB,0);assert.equal(q.gradientGiB,0);assert.ok(q.weightsGiB<l.weightsGiB);
const a=calc({phase:'pretrain',zero:3,accum:1}),b=calc({phase:'pretrain',zero:3,accum:4});close(b.gatherBytes,4*a.gatherBytes);close(b.dpBytes,4*a.dpBytes);
assert.ok(calc({interGBps:25,epCross:1}).a2aSeconds>calc({interGBps:50,epCross:1}).a2aSeconds);
const ppTrain=calc({phase:'pretrain',pp:2,ppCross:0,tokens:2048,batch:1});close(ppTrain.ppBytes,2*2048*7168*2);
const ppDecode=calc({phase:'decode',pp:2,ppCross:0,tokens:2048,batch:1});close(ppDecode.ppBytes,1*1*7168*2);
console.log('Supernode checks passed: independent state/cache/format calculations, sharding, accumulation and invalid inputs. No hardware tests.');
