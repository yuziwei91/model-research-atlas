"""Curated source register and research tables; no changes to audited model assets."""
from pathlib import Path
import json, csv

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/supernode'
OUT.mkdir(exist_ok=True)
DATE='2026-10-01'
SOURCES=[
 ('k3-config','K3 固定官方配置','https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json','固定 revision；93 层，24 MLA、69 KDA，896 专家、top16，MXFP4 group32；量化排除 attention/shared/vision 等'),
 ('k3-audit','已审 K3 权重文件头汇总','https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/data/families/kimi/k3-layers.json','已独立审查的 96 分片文件头；逻辑参数与物理存储分开。不是完整权重加载'),
 ('k3-impl','K3 官方参考实现','https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py','KDA/MLA/latent MoE；缓存公式属于条件推导，实际 backend 可改变布局'),
 ('vllm-k3-mla','vLLM K3 NVIDIA MLA cache implementation','https://github.com/vllm-project/vllm/blob/36229e1337ae0e44e60075a11ebfde210c11d706/vllm/models/kimi_k3/nvidia/mla.py','固定源码快照；吸收kv_b_proj并有512+64=576 latent元素布局；fp8_ds_mla分支另用656 B/token/layer专用cache布局。仅此backend路径，不代表Ascend或HF cache布局'),
 ('nv-gb300','NVIDIA GB300 NVL72 产品规格','https://www.nvidia.com/en-us/data-center/gb300-nvl72/','72 GPU 域；800 Gb/s/GPU 外网；Tensor Core 表默认稀疏，不直接作为稠密训练峰值'),
 ('nv-components','NVIDIA GB300 系统组件','https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html','每 GPU 288 GB，NVLink5；系统内存不等于单 GPU 可用内存'),
 ('nv-rubin','NVIDIA Vera Rubin NVL72 当前规格','https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/','当前表：72 GPU，288 GB/GPU，19.2 TB/s HBM，3 TB/s NVLink/GPU；与早期博客 3.6 TB/s 不同，不混用。规格不是 K3 验收'),
 ('nv-te','Transformer Engine 精度支持矩阵','https://nvidia.github.io/TransformerEngine/support_matrix.html','访问版本 2.21.0.dev0：Hopper FP8，Blackwell DC MXFP8/NVFP4；CUDA及compute capability约束'),
 ('nv-fp4','Transformer Engine NVFP4 recipe','https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html','block16 E4M3 scale 与全张量 FP32 scale；低精度训练 recipe 不等于所有状态为4bit'),
 ('amd-mi355','AMD MI355X 产品规格','https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html','288 GB，8 TB/s，CDNA4，MXFP4；link带宽与聚合带宽不同'),
 ('amd-platform','AMD MI355X 八卡平台','https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf','8 OAM全连接；每对153.6 GB/s双向，不是每卡单向153.6'),
 ('amd-cdna','AMD CDNA 代际与 Helios','https://www.amd.com/en/technologies/cdna.html','CDNA3/4/5数据类型分列；MI455X 72 GPU UALoE，432 GB/GPU，23.3 TB/s HBM，3.6 TB/s双向scale-up；实际供货/软件组合另验'),
 ('asc-cm','Huawei/SiliconFlow CloudMatrix384 论文','https://arxiv.org/html/2506.12708v2','384封装/768die；910C 每die64 GB、1.6 TB/s、UB196 GB/s单向、RDMA200 Gb/s单向。论文厂商测试不是本项目实测'),
 ('asc-950','华为 Atlas950 实机展示（2026-07）','https://www.huawei.com/cn/news/2026/7/atlas-950-superpod','1024卡展示、256TB全局编址、FP8/FP4；全局编址不能直接除卡数当HBM。单卡本研究不填猜测值'),
 ('asc-roadmap','华为 Atlas950 最大互联范围','https://www.huawei.com/cn/news/2026/3/mwc-superpod-computing','正文日期2026-02-28（MWC26）；8192卡最大互联口径，与7月1024卡实机展示范围分列'),
 ('asc-amct','CANN9.1 AMCT 量化工作流及类型约束','https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/amct/atlasamct_16_0049.html','已取得HTML全文。文中同时列出HiF8/FP8/MXFP8/MXFP4/FP4仅950PR/DT、当前版本仅INT8/INT4；属于文档内范围/版本冲突。Linear表又给出条件格式与shape/scale限制；不得据此宣称K3或全参训练支持'),
 ('asc-k3','vLLM-Ascend K3 当前教程','https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html','main教程基于vLLM0.27.1；Eco-Tech W4A8、4台A3/每台16逻辑NPU、DP4 TP16 EP64；服务配方非训练证明'),
 ('asc-pinned','vLLM-Ascend 已核验实现快照','https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py','SiTU、latent MoE、AttnRes、视觉分支；与当前教程不自动视为兼容软件栈'),
 ('asc-quant','Ascend W4A8 实现快照','https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/quantization/methods/w4a8/w4a8.py','量化方法与加载布局；存在代码不能证明所有设备/形状均可用'),
 ('asc-kda','Ascend KDA 实现快照','https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_kda.py','prefill chunk与decode recurrent分支；核查数值与状态误差'),
 ('zero','DeepSpeed ZeRO3 文档','https://deepspeed.readthedocs.io/en/latest/zero3.html','stage1优化器，stage2梯度，stage3权重；gather缓冲/层峰值和offload单独处理'),
 ('zero-paper','ZeRO 论文','https://arxiv.org/abs/1910.02054','混合精度Adam模型状态与分片；不是具体K3执行证明'),
 ('megatron','Megatron 并行策略','https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html','TP/PP/DP/CP/SP/EP组合；软件的实际rank映射需单独核对'),
 ('megatron-moe','Megatron MoE 并行','https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html','EP与TP及SP约束、expert tensor parallel、router负载/通信；不能把所有并行度直接相乘'),
 ('lora','LoRA 原始论文','https://arxiv.org/abs/2106.09685','冻结底座，训练低秩增量；P_adapter=sum r(d_in+d_out)，需明确目标层'),
 ('qlora','QLoRA 原始论文','https://arxiv.org/abs/2305.14314','冻结4bit底座+高精度计算/adapter；NF4、double quantization、paged optimizer。不是4bit全参训练'),
]
sources=[dict(id=i,title=t,url=u,checked=DATE,evidence=e) for i,t,u,e in SOURCES]
hardware=[
 dict(id='gb300',vendor='NVIDIA',name='GB300 NVL72',unit='GPU',domain=72,hbmGB=288,hbmGBps=8000,intraGBps=900,interGBps=100,topology='NVLink5交换域；外部IB/Ethernet',basis='1.8 TB/s双向/2作单向上限；800Gb/s/8；实效需另乘效率',status='官方产品规格；未实测',refs='nv-gb300;nv-components'),
 dict(id='mi355',vendor='AMD',name='MI355X 八卡域',unit='GPU/OAM',domain=8,hbmGB=288,hbmGBps=8000,intraGBps=537.6,interGBps=None,topology='8卡Infinity Fabric全连接；跨节点网络由部署填写',basis='7×153.6/2 GB/s单向注入上限；单peer仅76.8，bisection与拥塞另限',status='官方产品规格；未实测',refs='amd-mi355;amd-platform'),
 dict(id='cm384',vendor='Ascend',name='CloudMatrix384',unit='910C die / 逻辑NPU',domain=768,hbmGB=64,hbmGBps=1600,intraGBps=196,interGBps=25,topology='384封装×2die；UB域内，RoCE域间',basis='论文明确每die单向；不将768die误写768张卡。实际rank暴露核对npu-smi',status='厂商论文规格；未实测',refs='asc-cm'),
 dict(id='rubin',vendor='NVIDIA',name='Vera Rubin NVL72',unit='GPU',domain=72,hbmGB=288,hbmGBps=19200,intraGBps=None,interGBps=225,topology='NVLink6；3TB/s/GPU栏未在此推断方向',basis='外网0.45TB/s双向/2；较早3.6TB/s公告与当前表冲突，域内留空待选定规格',status='当前官方规格，落地软件/供货另验',refs='nv-rubin'),
 dict(id='helios',vendor='AMD',name='Helios MI455X',unit='GPU',domain=72,hbmGB=432,hbmGBps=23300,intraGBps=1800,interGBps=None,topology='72GPU UALoE单跳全互联',basis='3.6TB/s双向/2；外网配置未填',status='官方架构规格，落地软件/供货另验',refs='amd-cdna'),
 dict(id='atlas950',vendor='Ascend',name='Atlas950 SuperPoD',unit='卡（具体PR/DT待指定）',domain=1024,hbmGB=None,hbmGBps=None,intraGBps=None,interGBps=None,topology='灵衢/UB；1024卡实机与8192卡最大范围分列',basis='不把256TB全局编址当HBM；未明确单卡/单双向值均留空',status='官方展示/范围声明；容量输入待具体SKU',refs='asc-950;asc-roadmap;asc-amct')
]
precision=[
 ('BF16','全部三平台参考','W16A16；梯度/主权重/优化器可更高精度','2 B/参数；无量化scale','预训练/持续训练/全参微调基线；K3训练软件与原始高精度checkpoint另验','nv-te;amd-cdna;asc-cm'),
 ('FP8 E4M3','Hopper/Blackwell；CDNA3/4/5；950按算子及版本','IEEE-like E4M3浮点编码；与E5M2、MXFP8分列','1 B/值；per-tensor或block scale按recipe另计，此处group128/4B只是代理','TE delayed scaling与特定训练recipe有格式/前后向约束；BF16权重/FP32 Adam状态不自动消失','nv-te;amd-cdna;asc-amct'),
 ('FP8 E5M2','按硬件、算子与训练recipe核验','E5M2浮点编码；不等于E4M3或MXFP8','1 B/值；scale另计，此处group128/4B只是代理','不能因FP8统称推断所有算子/前后向均支持E5M2','nv-te;amd-cdna'),
 ('MXFP8','Blackwell TE recipe、CDNA代际、Ascend950PR/DT条件','MX block-scaled FP8；E4M3FN、block32 E8M0 scale；不等于延迟缩放FP8','1+1/32=1.03125 B/值（一维代理）；矩阵布局/padding另计','硬件/版本/shape条件适用；AMCT页面对950与910版本有内部范围冲突，作为条件候选而非K3支持证明','nv-te;amd-cdna;asc-amct'),
 ('INT8','三平台具体GEMM/量化后端条件','W8A8或W8A16，独立选择','1 B + per-channel/group scale/zero','主要给推理条件路径；不默认有INT8全参训练；激活校准、异常值处理影响质量','nv-gb300;amd-cdna;asc-cm'),
 ('MXFP4','Blackwell对应服务内核；CDNA4/5；950条件','E2M1 block32 + E8M0 scale；W4A16/W4A8要另核实','0.5+1/32=0.53125 B/量化参数；按每张量padding；K3只量化特定专家','原K3检查点格式；不能当作Ascend910C INT4 W4A8直接兼容，需转换配方','k3-config;amd-cdna;asc-amct;asc-k3'),
 ('NVFP4','Blackwell DC TE完整recipe；Rubin规格；其他平台不默认二进制兼容','E2M1 block16 + E4M3局部scale + FP32全局scale','0.5+1/16 B/参数 + 4 B/张量（双scale/转置另计）','存在4bit矩阵训练recipe，仍是混合精度；不是所有optimizer/grad/master均4bit，K3收敛未验证','nv-fp4;nv-te;nv-rubin'),
 ('INT4','按INT4解包/反量化/GEMM后端；不以FP4能力替代','W4A16冻结低位权重/高精度激活；W4A8再量化激活','0.5 B + scale/zero/packing；本情景group128、BF16 scale与2B zero仅是假设','910C K3使用转换W4A8已有服务教程；不能推出完整训练支持','asc-quant;asc-k3'),
 ('NF4 / QLoRA','具体框架与设备kernel另验','4bit冻结底座，adapter BF16，计算反量化','0.5 B + codebook/scales/double-quant metadata；不是MXFP4','本计算器将0.5B/值+2B/group64 scale作为预算假设；codebook、实际scale dtype及double-quant元数据不完整，不等于完整QLoRA格式','qlora;lora'),
 ('KV / KDA','与权重精度独立','KV BF16/FP8/INT8按engine；KDA矩阵状态通常更高精度','KV scale/page padding另列；KDA FP32默认假设','W4不代表KV4；训练不分配推理历史cache；GQA不可套K3混合缓存','k3-impl;vllm-k3-mla;asc-k3'),
]
precision=[dict(format=a,hardware=b,arithmetic=c,storage=d,limit=e,refs=f) for a,b,c,d,e,f in precision]
tasks=[
 ('pretrain','预训练','全部模型权重','BF16参数+梯度+FP32 master/Adam；低位GEMM副本另外计','训练序列、全局batch、数据及学习率；原发布MXFP4不能还原原BF16训练起点'),
 ('continue','持续训练','通常全部权重','同全参状态；恢复optimizer/RNG/dataloader状态','加载原量化权重再训练会改变起点，需注明恢复/反量化策略'),
 ('fullft','全参微调','明确需训练的全参集合','与预训练同状态结构，数据量少不自动减少模型状态','冻结视觉/部分模块需重算参数集合；本K3容量参考含全模型'),
 ('lora','LoRA','仅目标adapter','冻结BF16底座 + adapter16 B参考状态；仍有底座前向及反传激活','rank与目标层决定adapter量；本默认1B仅压力假设，不是K3已训练adapter'),
 ('qlora','QLoRA/低位底座PEFT','仅目标adapter','冻结4bit底座 + 高精度adapter与反量化工作区','NF4与MXFP4/INT4不是同格式；本模型存储预算不能证明后端可执行'),
 ('prefill','推理 Prefill','不训练','权重+新增KV/KDA状态+本批prefill激活/临时区','TTFT包括队列、tokenize、视觉、prefill、首decode、PD迁移；模型只算组件'),
 ('decode','推理 Decode','不训练','权重+历史KV+KDA固定状态+调度工作区','TPOT与吞吐需结合并发、路由稀疏性、通信启动与质量，不能由峰值乘卡数保证')
]
tasks=[dict(id=a,name=b,trainable=c,memory=d,limit=e) for a,b,c,d,e in tasks]
optimizations=[
 ('P0','全阶段','固定可运行基线','Ascend设备代际、CANN/PyTorch-NPU/vLLM或训练框架、checkpoint与转换版本同时固定','先跑BF16可对照层；核对SiTU/NoPE/KDA/AttnRes/视觉及条件分派','加载成功率、logits/状态误差、端到端质量、失败shape清单','不以950精度能力外推910C，不把K3推理教程当训练支持','asc-pinned;asc-k3;asc-amct'),
 ('P0','全参训练 BF16/FP8/FP4','训练图及精度可用性','需要backward、optimizer、分布式保存/恢复和实际K3训练代码','逐模块前后向验证；矩阵低位时保留高精度累计/主权重，记录scale更新','梯度误差、loss曲线、溢出率、resume一致性','未取得K3在Ascend全参FP8/FP4收敛证据，不声明已支持','k3-impl;zero;nv-fp4;asc-amct'),
 ('P0','W8A8/W4A8 推理','量化转换与质量','固定vLLM-Ascend W4A8实现只接受per-channel权重，显式拒绝group_size>0；与本预算group128格式不可直接等同。910C使用经转换的配方；950另验原生格式','先验证校准集与长上下文质量，再比较quant/dequant/scale/layout总成本','精度评测、权重峰值、加载时间、GEMM与量化各自耗时','不能把MXFP4检查点直接改标签成INT4；相同比特不等价','asc-k3;asc-quant;asc-cm'),
 ('P0','训练 / 推理 EP','拓扑与rank分组','EP all-to-all与TP group尽量位于可提供所需带宽的域；DP同步可以跨域但需实测','导出rank到die/封装/节点/域映射；实测dispatch/combine及allreduce','每rank字节、有效单向GB/s、p50/p99 collective时间、跨域比例','不能以总域带宽除卡数代替单连接/双截带宽；910C die与封装分开','asc-cm;megatron-moe'),
 ('P1','MoE 训练 / Prefill','dispatch/combine与Grouped GEMM重叠','足够token/专家负载与合法缓冲生命周期；训练额外反向路由','在profile显示关键路径时，尝试MC2/融合通信及分桶，不直接增加EP','路由偏斜、专家负载p99、padding/drop率、通信暴露时间、梯度正确性','大EP小token会降低矩阵利用率；重叠缓冲增加HBM','asc-cm;asc-pinned;megatron-moe'),
 ('P1','MoE Decode 4/8bit','路由与小矩阵尾部','低batch、小专家M；latent宽3584不同于残差7168','按M/N/K及dtype分桶，核对SiTU/量化/归一化/投影融合分支','token间隔p99、算子启动数、dequant占比、HBM实际字节','已有融合不等于全部shape命中；不编造加速比例','asc-pinned;asc-quant'),
 ('P1','训练','FSDP/ZeRO与PP峰值','有DP/EDP复制组；ZeRO3需要临时gather最大层，EP分片不能重复算','先统计最大rank状态，再比较重计算、stage2/3和PP层划分','峰值HBM、allgather/RS字节、step time、气泡占比','K3层类型交替且93不可任意等分；PP按最重stage而非平均','zero;megatron'),
 ('P1','LoRA / QLoRA','冻结底座与adapter覆盖','明确r和attention/shared/routed/vision目标模块','检查冻结参数无optimizer state；量化反传与adapter分布跨专家单独验','可训练参数计数、底座意外梯度、HBM、loss/质量','paged optimizer将HBM压力转移到CPU/链路；仍有底座激活与反传','lora;qlora;zero'),
 ('P1','Prefill / Decode','KV与KDA状态','缓存布局/精度要有后端支持；混合模型同时有线性与历史项','按实际PP分层取最重stage；分开计HF展开MLA30720、vLLM latent576或fp8_ds_mla专用656B布局、KDA矩阵及conv、分页余量；量化cache前测误差','每序列增量HBM、长上下文误差、KV命中/迁移量、OOM边界','KDA不是全模型常数缓存；权重4bit不意味着KV或KDA状态4bit','k3-config;k3-impl;asc-kda'),
 ('P1','Prefill','chunk KDA/MLA与视觉峰值','按文本tokens/视觉patch数/并发分桶','chunked prefill与视觉encoder独立profile，避免瞬时工作区压垮decode','TTFT分解、峰值HBM、chunk边界状态误差','减小chunk可能增加启动次数和TTFT，需连同TPOT评估','asc-kda;asc-k3'),
 ('P2','Decode','图执行与CPU调度','稳定shape覆盖，支持相应graph模式','比较eager/图回放，记录重新编译、capture内存与队列等待','host gaps、recompile数、p99 TPOT、capture失败率','动态batch/缓存/多模态可能触发回退；不能只看kernel时间','asc-k3'),
 ('P2','PD分离 / 跨超节点','迁移与链路成本','实际PD协议支持KDA状态及KV；带宽与拥塞可测','增加KV+KDA迁移模型，按SLO选择共置或分离，不默认分离更快','迁移字节/时延、排队、端到端TTFT、网络尾延迟','序列变长时迁移可能抵消计算收益；量化迁移含scale与转换','asc-cm;asc-k3'),
 ('P2','训练 / 大模型加载','检查点、CPU/NVMe与冗余','磁盘/CPU内存/网络及复本另有预算','记录sharded save、恢复时间、优化器写出和故障恢复副本','checkpoint秒数、step暂停、CPU峰值、存储吞吐','本HBM情景未含CPU/NVMe硬件容量，不作为完整采购单','zero')
]
optimizations=[dict(priority=a,phase=b,target=c,condition=d,action=e,metrics=f,limit=g,refs=h) for a,b,c,d,e,f,g,h in optimizations]

def write_csv(name, rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build():
 for name,rows in [('sources',sources),('hardware',hardware),('precision-support',precision),('workloads',tasks),('ascend-priorities',optimizations)]:
  (OUT/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
  write_csv(name+'.csv',rows)
 d=json.loads((ROOT/'data/families/kimi/k3-layers.json').read_text())
 route=sum(g.get('routed_parameters',0) for g in d['groups'])
 active=sum(g.get('active_linear_parameters',0) for g in d['groups'] if g['group'] in ['decoder','output_head'])
 k3=dict(id='kimi-kimi-k3',name='Kimi-K3',kind='固定文件头/配置 + 运行假设',total=d['logical_parameters'],routed=route,active=active,layers=93,referenceLayers=93,mlaLayerIndices=[*range(4,93,4),93],moeLayers=92,experts=896,topK=16,hidden=7168,dispatchWidth=3584,expertIntermediate=3072,mlaLayers=24,kdaLayers=69,heads=96,headDim=128,kvWidth=30720,latentKvWidth=576,cacheLayout='hf-expanded',cacheBasis='HF固定参考代码 past_key_values.update 前已扩展K/V：96 heads × (128 NoPE key + 64额外投影维度 + 128 value)=30,720 elements/layer/token；不表示64维key已旋转',latentCacheBasis='vLLM NVIDIA K3 MLA固定源码吸收kv_b_proj；cache latent=512+64=576 elements/layer/token；K3 NoPE路径不表示64维key已旋转；仅该backend路径',convWidth=4,checkpointBytes=d['payload_bytes'],source='k3-config;k3-audit;k3-impl;vllm-k3-mla',baseEvidence='k3-config;k3-impl;vllm-k3-mla',userModified=False,cacheStatus='仅未修改HF参考投影形状有来源；运行时allocator/page布局未实测，用户编辑后来源不再验证自定义情景')
 models=[k3]
 for scale,total,activep in [('5t',5e12,2e11),('10t',1e13,4e11)]:
  models.append(dict(k3,id='scenario-'+scale,name=scale.upper()+' 参数化情景（非已发布模型）',kind='纯压力情景：结构/缓存布局沿用K3假设且均可修改，不代表可实现的具体网络',total=total,routed=total*.98,active=activep,checkpointBytes=None,cacheLayout='assumed-hf-expanded',baseEvidence=None,userModified=False,cacheBasis='沿用K3的假设缓存几何；不是模型来源或运行验证',latentCacheBasis='沿用K3的假设latent几何；不是模型来源或运行验证',cacheStatus='假设K3缓存几何；非已发布或运行实测模型',source='用户可配置情景，无模型发布来源'))
 manifest=json.loads((ROOT/'dist/downloads/model-documents/manifest.json').read_text())
 coverage=[]
 for m in manifest['models']:
  coverage.append(dict(id=m['id'],name=m['name'],family=m['family'],scope='完整K3情景入口' if m['id']==k3['id'] else '待补超节点运行参数；不套K3缓存或训练图',total_parameters=k3['total'] if m['id']==k3['id'] else None,active_linear_proxy=k3['active'] if m['id']==k3['id'] else None,config_source=m.get('config_source',''),required_inputs='总参数/激活参数/专家分解/训练参数集合/缓存结构/后端支持/设备拓扑',analysis_url='supernode/index.html?model='+m['id'],runtime_verified=False))
 (OUT/'models.json').write_text(json.dumps(models,ensure_ascii=False,indent=2));(OUT/'model-coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2));write_csv('model-coverage.csv',coverage)
 print('Research tables:',len(sources),'sources;',len(hardware),'hardware records;',len(coverage),'model scope records')

if __name__=='__main__':build()
