// Units: parameter counts in elements; capacities in GiB (2**30); bandwidth GB/s (1e9).
export const GiB=2**30;
export const defaults={phase:'decode',format:'MXFP4',actFormat:'BF16',gradSync:'per_micro',cacheLayout:'hf-expanded',tp:8,pp:1,dp:8,ep:64,etp:1,zero:0,
 domain:8,hbmGiB:128,reserve:.15,intraGBps:200,interGBps:50,intraEff:.65,interEff:.65,
 hbmGBps:1600,hbmEff:.6,tflops:500,computeEff:.35,latencyUs:5,tpCross:0,epCross:.5,dpCross:1,ppCross:1,
 batch:1,tokens:2048,context:32768,accum:8,liveMicro:1,actCoeff:12,savedActBytes:2,commBytes:2,sp:1,
 kvBytes:2,kvMeta:.02,cacheShards:1,stateBytes:4,convBytes:4,workspaceGiB:16,gatherGiB:16,
 imbalance:1.1,packing:.02,adapterB:1,adapterExpertFraction:0,gradBytes:2,masterBytes:4,optimBytes:8,
 computeCopies:0,recompute:.5,extraFlops:.2,weightReadFraction:1,extraMs:0};
export const formats={BF16:{bits:16,group:1,scale:0,zero:0,tensor:0},'FP8-E4M3':{bits:8,group:128,scale:4,zero:0,tensor:0},'FP8-E5M2':{bits:8,group:128,scale:4,zero:0,tensor:0},MXFP8:{bits:8,group:32,scale:1,zero:0,tensor:0},
 INT8:{bits:8,group:128,scale:2,zero:2,tensor:0},MXFP4:{bits:4,group:32,scale:1,zero:0,tensor:0},
 NVFP4:{bits:4,group:16,scale:1,zero:0,tensor:4},INT4:{bits:4,group:128,scale:2,zero:2,tensor:0},NF4:{bits:4,group:64,scale:2,zero:0,tensor:0}};
export function storage(p,format,packing=0,tensors=0){const f=formats[format];return (p*f.bits/8+Math.ceil(p/f.group)*(f.scale+f.zero)+f.tensor*tensors)*(1+packing)}
export function calculate(m,x){
 const errors=[],warnings=[];
 const positive=['tp','pp','dp','ep','etp','domain','batch','tokens','accum','liveMicro','sp','cacheShards'];
 for(const key of positive)if(!Number.isInteger(x[key])||x[key]<1)errors.push(key+' 必须为正整数');
 if(!Number.isInteger(x.context)||x.context<0)errors.push('历史缓存长度必须为非负整数');
 if(!['per_micro','deferred'].includes(x.gradSync))errors.push('梯度同步策略必须为per_micro/deferred');
 if(!['hf-expanded','vllm-latent','vllm-fp8-ds-mla'].includes(x.cacheLayout))errors.push('MLA缓存布局须选择明确的HF展开、vLLM latent或fp8_ds_mla专用布局');
 if(!['BF16','FP8-E4M3','FP8-E5M2','MXFP8','INT8'].includes(x.actFormat))errors.push('激活格式须明确选择BF16、FP8 E4M3、FP8 E5M2、MXFP8或INT8');
 for(const key of ['reserve','intraEff','interEff','hbmEff','computeEff'])if(!Number.isFinite(x[key])||x[key]<0||x[key]>=1&&(key==='reserve')||x[key]>1||key!=='reserve'&&x[key]===0)errors.push(key+' 比例不合法');
 for(const key of ['tpCross','epCross','dpCross','ppCross','adapterExpertFraction','weightReadFraction'])if(!Number.isFinite(x[key])||x[key]<0||x[key]>1)errors.push(key+' 须在0至1');
 for(const key of ['hbmGiB','savedActBytes','commBytes','kvBytes','stateBytes','convBytes','gradBytes'])if(!Number.isFinite(x[key])||x[key]<=0)errors.push(key+' 须大于0');
 for(const key of ['tflops','hbmGBps','intraGBps','interGBps'])if(x[key]!==null&&x[key]!==undefined&&(!Number.isFinite(x[key])||x[key]<=0))errors.push(key+' 仅接受null（未知）或有限正数');
 for(const key of ['actCoeff','workspaceGiB','gatherGiB','packing','adapterB','masterBytes','optimBytes','computeCopies','recompute','extraFlops','kvMeta','extraMs','latencyUs'])if(!Number.isFinite(x[key])||x[key]<0)errors.push(key+' 须非负');
 if(!Number.isFinite(x.imbalance)||x.imbalance<1)errors.push('负载/内存不均衡系数至少1');
 if(![0,1,2,3].includes(x.zero))errors.push('ZeRO stage为0..3');
 if(!formats[x.format])errors.push('未知权重格式');
 if(!['pretrain','continue','fullft','lora','qlora','prefill','decode'].includes(x.phase))errors.push('未知阶段');
 if(!m||![m.total,m.routed,m.active,m.layers,m.hidden,m.moeLayers,m.experts,m.topK,m.dispatchWidth,m.expertIntermediate,m.mlaLayers,m.kdaLayers,m.heads,m.headDim,m.kvWidth,m.latentKvWidth,m.convWidth].every(Number.isFinite))errors.push('模型参数不足。需补模型与本工具支持的MLA/KDA缓存形状；其他MHA/GQA/SSM需自有缓存公式，不自动代用K3');
 if(errors.length)return {errors,warnings};
 if(m.total<=0||m.routed<0||m.routed>m.total||m.active<=0||m.active>m.total)errors.push('模型参数关系不合法');
 for(const key of ['layers','hidden','experts','topK','dispatchWidth','expertIntermediate','heads','headDim','kvWidth','convWidth'])if(!Number.isInteger(m[key])||m[key]<1)errors.push('模型 '+key+' 须为正整数');
 for(const key of ['moeLayers','mlaLayers','kdaLayers'])if(!Number.isInteger(m[key])||m[key]<0||m[key]>m.layers)errors.push('模型 '+key+' 层数不合法');
 if(m.topK>m.experts||m.mlaLayers+m.kdaLayers>m.layers)errors.push('topK或缓存层数超过模型范围');
 if(m.kvWidth<=0||!Number.isInteger(m.latentKvWidth)||m.latentKvWidth<=0)errors.push('模型缓存布局宽度须为正整数');
 const n=x.tp*x.pp*x.dp,edp=x.tp*x.dp/(x.etp*x.ep);
 if(!Number.isInteger(edp)||edp<1)errors.push('TP×DP 必须能被 ETP×EP 整除（专家数据复制组）');
 if(m.experts%x.ep)errors.push('专家数须能被EP整除（此预算不支持专家padding分配）');
 if(m.heads%x.tp)errors.push('注意力head数须能被TP整除（本方案不含head复制）');
 if(m.hidden%x.tp||m.dispatchWidth%x.etp||m.expertIntermediate%x.etp)errors.push('hidden/TP、专家通信宽度/ETP、专家中间维度/ETP必须整除');
 if(x.pp>m.layers)errors.push('PP超过语言层数');
 if(x.sp!==1&&x.sp!==x.tp)errors.push('SP仅支持1或TP；SP不是额外乘卡数的维度');
 if(x.cacheShards>x.tp||x.tp%x.cacheShards)errors.push('cache分片须为TP的约数，且需要后端依据');
 if(x.phase==='qlora'&&formats[x.format].bits!==4)errors.push('QLoRA/低位PEFT必须选择4bit冻结底座存储情景');
 if(x.phase==='lora'&&x.format!=='BF16')errors.push('本LoRA基线冻结BF16底座；低位底座请选QLoRA/低位PEFT');
 if(m.layers%x.pp)warnings.push('层数不可均分PP；激活使用ceil(L/PP)比例，缓存按逐stage层数计并取对应分支峰值；权重另乘不均衡系数，仍需实际最重stage分层核验');
 if(x.ep>1&&x.tp>1&&x.sp===1)warnings.push('训练TP+EP通常需SP；当前为容量假设，Megatron等后端要求SP时本配置不可直接启动');
 if(errors.length)return {errors,warnings};
 const full=['pretrain','continue','fullft'].includes(x.phase),peft=['lora','qlora'].includes(x.phase),training=full||peft;
 const dense=m.total-m.routed,ad=x.adapterB*1e9,adE=ad*x.adapterExpertFraction,adD=ad-adE;
 const z=x.zero,dshare=x.tp*x.pp,eshare=x.etp*x.ep*x.pp;
 const state=(pd,pe,bytes,stage)=>bytes*(pd/(dshare*(z>=stage?x.dp:1))+pe/(eshare*(z>=stage?edp:1)));
 const gradStage=x.gradSync==='per_micro'&&z>=2?2:99;
 let weights,gradient=0,optimizer=0,master=0,copies=0;
 if(full){weights=state(dense,m.routed,2,3);gradient=state(dense,m.routed,x.gradBytes,gradStage);optimizer=state(dense,m.routed,x.optimBytes,1);master=state(dense,m.routed,x.masterBytes,1);
  copies=x.computeCopies*(storage(dense,x.format,x.packing)/dshare+storage(m.routed,x.format,x.packing)/eshare);
 }else{
  // Conservative model: non-routed tensors stay BF16. Quantize routed experts only.
  // No implicit FSDP sharding of frozen PEFT base; record this conservative choice.
  weights=2*dense/dshare+storage(m.routed,x.format,x.format==='BF16'?0:x.packing)/eshare;
  if(peft){weights+=state(adD,adE,2,3);gradient=state(adD,adE,x.gradBytes,gradStage);optimizer=state(adD,adE,x.optimBytes,1);master=state(adD,adE,x.masterBytes,1)}
 }
 const boundaries=Array.from({length:x.pp+1},(_,i)=>Math.floor(i*m.layers/x.pp));
 const exactPattern=Array.isArray(m.mlaLayerIndices)&&m.layers===m.referenceLayers&&m.mlaLayers===m.mlaLayerIndices.length&&m.kdaLayers===m.layers-m.mlaLayers&&new Set(m.mlaLayerIndices).size===m.mlaLayerIndices.length&&m.mlaLayerIndices.every(v=>Number.isInteger(v)&&v>=1&&v<=m.layers);
 const stageCounts=exactPattern?Array.from({length:x.pp},(_,i)=>({mla:m.mlaLayerIndices.filter(layer=>layer>boundaries[i]&&layer<=boundaries[i+1]).length,layers:boundaries[i+1]-boundaries[i]})):null;
 const maxStageSize=Math.ceil(m.layers/x.pp);
 const maxMlaLayers=stageCounts?Math.max(...stageCounts.map(v=>v.mla)):Math.min(m.mlaLayers,maxStageSize);
 const maxKdaLayers=stageCounts?Math.max(...stageCounts.map(v=>v.layers-v.mla)):Math.min(m.kdaLayers,maxStageSize);
 const localLayers=Math.ceil(m.layers/x.pp),layerFraction=localLayers/m.layers,T=x.phase==='decode'?1:x.tokens;
 const activation=x.batch*T*m.hidden*localLayers*x.savedActBytes*x.actCoeff*x.liveMicro/x.sp;
 const peakCacheTokens=x.context+T;
 const kvWidth=x.cacheLayout==='vllm-latent'?m.latentKvWidth:m.kvWidth;
 const mla=training?0:x.batch*peakCacheTokens*maxMlaLayers*(x.cacheLayout==='vllm-fp8-ds-mla'?656:kvWidth*x.kvBytes)*(1+x.kvMeta)/x.cacheShards;
 const kda=training?0:x.batch*maxKdaLayers*m.heads*m.headDim*m.headDim*x.stateBytes/x.cacheShards;
 const conv=training?0:x.batch*maxKdaLayers*3*m.heads*m.headDim*m.convWidth*x.convBytes/x.cacheShards;
 const workspace=x.workspaceGiB*GiB, gather=training&&z===3?x.gatherGiB*GiB:0;
 const statics=(weights+gradient+optimizer+master+copies)*x.imbalance;
 const total=statics+activation+mla+kda+conv+workspace+gather,usable=x.hbmGiB*GiB*(1-x.reserve);
 const unique=full?m.total*(2+x.gradBytes+x.optimBytes+x.masterBytes):2*dense+storage(m.routed,x.format,x.format==='BF16'?0:x.packing)+(peft?ad*(2+x.gradBytes+x.optimBytes+x.masterBytes):0);
 const stateOnlyLowerCards=Math.ceil(unique/(x.hbmGiB*GiB*(1-x.reserve)));
 // Communication proxies: bytes SENT per rank, logical payload, no sum of send+receive.
 const tokenLocal=x.batch*T,pass=training?2:1;
 const moeBytes=2*pass*m.moeLayers*layerFraction*(tokenLocal/x.tp)*m.topK*m.dispatchWidth*x.commBytes*(1-1/x.ep)*x.imbalance;
 const tpCalls=2*pass*localLayers, tpBytes=tpCalls*2*(x.tp-1)/x.tp*tokenLocal*m.hidden*x.commBytes;
 const dpPayload=training?(full?state(dense,m.routed,x.gradBytes,gradStage):state(adD,adE,x.gradBytes,gradStage)):0;
 const denseGrad=training?(full?dense:adD)*x.gradBytes/dshare:0,expertGrad=training?(full?m.routed:adE)*x.gradBytes/eshare:0;
 const syncCount=x.gradSync==='per_micro'?x.accum:1;
 const dpBytes=training?syncCount*2*((x.dp-1)/x.dp*denseGrad+(edp-1)/edp*expertGrad):0;
 const gatherBytes=training&&z===3?2*x.accum*((x.dp-1)/x.dp*(full?dense:adD)*2/dshare+(edp-1)/edp*(full?m.routed:adE)*2/eshare):0;
 // Count outgoing bytes only: activation in forward, gradient in backward. A rank's received payload is not another send.
 const ppBytes=x.pp>1?pass*tokenLocal*m.hidden*x.commBytes:0;
 const comm=(bytes,cross,calls)=>{
  if(bytes===0)return 0;
  if((cross<1&&!(x.intraGBps>0))||(cross>0&&!(x.interGBps>0)))return null;
  return bytes*((1-cross)/(x.intraGBps*x.intraEff*1e9||1)+cross/(x.interGBps*x.interEff*1e9||1))+calls*x.latencyUs/1e6;
 };
 const a2aSeconds=comm(moeBytes,x.epCross,2*pass*localLayers),tpSeconds=comm(tpBytes,x.tpCross,tpCalls),ppSeconds=comm(ppBytes,x.ppCross,2*pass);
 const dpSeconds=comm(dpBytes+gatherBytes,x.dpCross,training?2*syncCount+(z===3?2*localLayers*x.accum:0):0);
 const ops=full?(6+2*x.recompute)*m.active:(peft?(4+2*x.recompute)*m.active+6*ad:2*m.active);
 const flops=ops*tokenLocal*x.dp*(1+x.extraFlops);
 const computeSeconds=x.tflops>0?flops/(n*x.tflops*1e12*x.computeEff):null;
 const traffic=weights*x.imbalance*x.weightReadFraction+(training?0:mla+kda*2+conv*2);
 const memorySeconds=x.hbmGBps>0?traffic/(x.hbmGBps*1e9*x.hbmEff):null;
 // Training window: gradient sync defaults to every microbatch. Deferred sync retains full local gradients.
 const micro=[computeSeconds,memorySeconds,a2aSeconds,tpSeconds,ppSeconds];
 let modeledFloor=null,modeledSerial=null;
 if(micro.every(v=>v!==null)&&dpSeconds!==null){const mult=training?x.accum:1;modeledFloor=Math.max(...micro)*mult+dpSeconds+x.extraMs/1000;modeledSerial=micro.reduce((a,b)=>a+b,0)*mult+dpSeconds+x.extraMs/1000}
 const globalTokens=tokenLocal*x.dp*(training?x.accum:1);
 const pipeline=training?(x.accum+x.pp-1)/x.accum:x.pp;
 if(modeledFloor!==null){modeledFloor*=pipeline;modeledSerial*=pipeline}
 if(stageCounts)warnings.push('PP缓存按连续层号逐stage计算；MLA/KDA/conv分支各取最大层数并独立相加为保守上界。各stage MLA/KDA='+JSON.stringify(stageCounts.map(v=>[v.mla,v.layers-v.mla])));else warnings.push('模型层放置未知；MLA/KDA各自取min(分支层数,ceil(总层数/PP))作为连续均衡stage的最大分支上界，再独立相加，可能高估；若stage不连续或不均衡需自行增大。');
 warnings.push('只验证代数与容量预算；backend可执行性、精度收敛和实际吞吐尚未设备验证');
 if(full&&x.format!=='BF16')warnings.push('低精度为矩阵计算情景；仍保留BF16参数、梯度/master/optimizer独立状态；不能按4bit×P替代训练显存');
 if(peft)warnings.push('冻结底座保守按TP/EP/PP分片、未再用ZeRO分片；adapter大小与分布是可改假设，需核对真实目标模块');
 if(training&&x.gradSync==='deferred')warnings.push('延迟同步须后端支持coalesced/no_sync；本预算保留未分片本地梯度峰值，不能与每微批分片梯度同时取最优；DeepSpeed/FSDP具体限制另验');
 if(x.format==='NVFP4')warnings.push('NVFP4存储公式是1D block16代理（0.5625 B/参数，另加全局scale近似）；训练recipe通常对权重采用16×16二维scale、对激活/梯度采用1D block16，转置副本、padding、RHT/随机舍入与高精度状态未完整建模');
 if(x.format==='FP8-E4M3'||x.format==='FP8-E5M2'||x.format==='MXFP8')warnings.push('FP8编码与scale粒度依格式/recipe而异；E4M3、E5M2、MXFP8不是同一种格式。此处scale仅为存储代理，不证明算子或训练recipe可用');
 warnings.push('通信按均匀路由近似且无去重，跨域比例由输入指定；PP时延采用简化气泡/串行系数，不是调度仿真');
 return {errors,warnings,n,edp,effectiveNewTokens:T,historicalCacheTokens:x.context,peakCacheTokens,maxMlaLayers,maxKdaLayers,cacheStageCounts:stageCounts,cacheLayout:x.cacheLayout,kvWidthUsed:x.cacheLayout==='vllm-fp8-ds-mla'?'656 B special layout':kvWidth,domains:Math.ceil(n/x.domain),weightsGiB:weights/GiB,gradientGiB:gradient/GiB,optimizerGiB:optimizer/GiB,masterGiB:master/GiB,copiesGiB:copies/GiB,activationGiB:activation/GiB,kvGiB:mla/GiB,kdaGiB:kda/GiB,convGiB:conv/GiB,workspaceGiB:(workspace+gather)/GiB,imbalanceExtraGiB:(weights+gradient+optimizer+master+copies)*(x.imbalance-1)/GiB,totalGiB:total/GiB,usableGiB:usable/GiB,capacityFits:total<=usable,stateOnlyLowerCards,stateOnlyLowerDomains:Math.ceil(stateOnlyLowerCards/x.domain),moeBytes,tpBytes,dpBytes,gatherBytes,ppBytes,dpPayload,flops,computeSeconds,memorySeconds,a2aSeconds,tpSeconds,dpSeconds,ppSeconds,modeledFloor,modeledSerial,globalTokens,theoreticalTokensPerSecond:modeledFloor>0?globalTokens/modeledFloor:null};
}
