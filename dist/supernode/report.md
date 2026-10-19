# K3 与更大模型的超节点训练、量化和部署

研究日期：2026-10-01。独立早审已提出需修订项；本版本正在逐项修订，尚待最终复核。原有模型数据审查结论不自动覆盖本专题。此处没有进行模型加载、GPU/NPU运行、数值收敛或吞吐测试。

[交互计算器](index.html) · [情景矩阵 CSV](scenario-matrix.csv) · [公式 CSV](formulas.csv) · [硬件规格 CSV](hardware.csv) · [精度支持 CSV](precision-support.csv) · [Ascend 优化清单](ascend-priorities.csv) · [来源登记](sources.csv) · [全部资料 ZIP](research-bundle.zip)

## 研究结论与使用方法

K3 的已审检查点包含 2,779,931,837,184 个逻辑参数，实际张量载荷 1,560,860,324,864 字节。全参数量决定驻留权重及全参训练状态；不能用每 token 的激活参数替代它。检查点将部分路由专家存成 MXFP4，其他模块有 BF16/FP32 张量，实际存储不是“总参数×0.5”。这里继续使用固定 revision 与已审文件头，不把它当作已通过整模型运行。[固定 K3 配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)、[已审文件头汇总](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/data/families/kimi/k3-layers.json)。

K3 路由专家参数为 2,722,740,830,208，其他参数为 57,191,006,976。语言 decoder 加输出头的激活线性参数代理为 104,175,425,536。后者只用于线性矩阵计算近似；它不是官方宣称的完整“激活参数”，也未包含 attention 二次项、KDA状态更新、路由、视觉、归一化等全部运算。总参数容量保守包含视觉与全局组件，文本计算代理不包含视觉计算。多模态作业必须另外输入视觉激活及耗时。

5T/10T 条目是参数压力情景，分别假设 200B/400B 激活线性参数和 98% 路由参数。缓存与层数暂沿用 K3 结构以便观察容量变化。这些参数并未构成经验证可实现的网络，更不是虚构已发布模型。可修改所有输入或另建匹配实际网络的情景。

先选工作阶段，再填实际设备、逻辑 rank 和并行分组，最后填缓存、激活、workspace与效率。容量筛查只回答“这些假设下是否超出每设备预算”。真实可运行还要求最大层/专家可放入、布局与 kernel 可用、并行策略受软件支持；高效还需要负载均衡、通信与计算重叠、调度以及精度质量通过。唯一状态容量下限只是理想完全分片下的必要条件，不是最小可部署卡数或采购建议。

## 各工作阶段不能共用一张 decode 表

|阶段|可训练状态|主要额外内存与验证要求|
|---|---|---|
|预训练|全部指定训练参数|梯度、optimizer、master、保存激活、通信/gather缓冲。需要训练代码、数据、初始化与收敛；发布的低位checkpoint不提供原高精度起点|
|持续训练|通常全部权重|与全参训练相同的状态结构；恢复优化器、随机数与数据位置。低位反量化后的起点必须另记|
|全参微调|指定全部权重，可另行冻结视觉|训练数据少不会按比例减少模型状态；参数集合改变要重算|
|LoRA|sum r×(d_in+d_out)，按真实目标矩阵相加|冻结BF16底座、adapter状态与仍需保留的激活；adapter分布到专家/非专家组分别计|
|QLoRA / 低位底座 PEFT|只训练adapter|冻结量化底座、反量化计算和workspace。仅路由专家权重按选定4bit格式预算，非专家权重仍按BF16；NF4、MXFP4、INT4不同。本计算器将NF4的2B/group64 scale作为存储预算假设；codebook、实际scale dtype与QLoRA double-quant元数据未完整计入，不等同于完整QLoRA double quant实现，也不宣称K3 NF4训练已适配|
|Prefill|无梯度/optimizer|新token激活、MLA缓存、KDA状态、视觉与长序列临时区；首token时间还包含首decode与排队|
|Decode|无梯度/optimizer|历史KV、固定KDA状态、读权重与细粒度通信；低batch常受启动/带宽限制，不能只看峰值算力|

LoRA目标集合、rank和bias/embedding是否训练必须写入实验记录。默认 adapter=1B 是压力假设，不是 K3 的实测 adapter 参数数。[LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)。

## 硬件范围与计量口径

本研究同时对照 NVIDIA、AMD、Ascend，优化行动重点是 Ascend。三者使用相同内存公式和输入单位，不用不同稀疏性、精度、batch或口径拼出性能排名。

|设备/域|官方事实与计算口径|不能据此推定的内容|
|---|---|---|
|GB300 NVL72|72 GPU NVLink域；每GPU288GB；外网800Gb/s。示例将每GPU1.8TB/s双向NVLink换为900GB/s单向注入上限|整域130TB/s不是单卡速率。厂商Tensor Core规格默认稀疏，不能直接代入稠密训练|
|MI355X|每GPU288GB、8TB/s HBM；8 OAM全互联；每对153.6GB/s双向。单向聚合上限7×76.8=537.6GB/s|任意单peer不能达到聚合注入上限；跨节点NIC/Bisection由部署补齐，不能把128GPU机架都当8卡同速域|
|CloudMatrix384|384颗910C封装、768die。每die64GB HBM、1.6TB/s、UB196GB/s单向、外域RDMA200Gb/s=25GB/s|每封装128GB不能给每die都算128GB；实际rank暴露需核对。论文测试对象是其指定DeepSeek负载，不能移植为K3成绩|
|Vera Rubin NVL72|当前产品表72GPU、每GPU288GB与19.2TB/s HBM；网络0.45TB/s双向。当前NVLink栏3TB/s，与早期3.6TB/s公告不同|本工具不默选冲突的域内带宽；用户需固定具体规格和方向。当前规格不等于K3训练后端验证|
|Helios MI455X|官方CDNA5页列72GPU UALoE、432GB/GPU、23.3TB/s HBM、3.6TB/s双向scale-up|供货状态、实际NIC、后端与量化格式组合仍须逐部署确认|
|Atlas950|7月实机展示1024卡，MWC26（正文日期2026-02-28）公告最大互联8192卡；FP8/FP4与256TB全局编址属于官方展示范围|不将全局编址内存除卡数冒充HBM。单卡PR/DT规格、方向与完整软件组合尚未固定，数值留空|

量化流程也要区分：PTQ是在既有权重上校准/转换；fake-quant QAT在训练中模拟量化误差，算子可仍以BF16/FP32执行；低位GEMM训练实际用低位操作数，但通常保留更高精度累计与optimizer/master状态；QLoRA冻结量化底座，只更新adapter。四者的显存、计算和质量不可由同一个4bit开关代表。资料：[GB300](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)、[GB300组件](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html)、[MI355X平台](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf)、[CloudMatrix384 §3.3](https://arxiv.org/html/2506.12708v2)、[Rubin当前规格](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)、[CDNA5/Helios](https://www.amd.com/en/technologies/cdna.html)、[Atlas950实机](https://www.huawei.com/cn/news/2026/7/atlas-950-superpod)、[Atlas950最大范围](https://www.huawei.com/cn/news/2026/3/mwc-superpod-computing)。

## 8bit、4bit 与训练精度

BF16参考下，混合精度Adam的示例常驻状态按每可训练参数2B参数、2B梯度、8B一二阶矩、4B master合计16B计算。各字节数独立可改；有的系统使用FP32梯度、没有单独master、或不同optimizer，不应机械套16B。激活、参数gather、通信和workspace在此之外。[ZeRO文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)。

FP8 E4M3、FP8 E5M2、MXFP8、INT8以及FP4/MXFP4/NVFP4/INT4不能只按位宽合并。W8A8说明权重与激活位宽，并未说明浮点还是整型；W4A16通常需要反量化或融合解包，W4A8另有激活量化，累计精度也要记录。权重精度不自动决定梯度、optimizer、master、KV或KDA状态精度。

对某量化张量，存储预算为 `ceil(P*b/8) + ceil(P/group)*(scaleBytes+zeroBytes) + tensorScaleBytes`，再加逐行/逐块packing与alignment开销。聚合计算器使用连续近似和可调padding比例，不能替代逐张量载荷审计。MXFP4 block32的1B scale令每参数均摊0.53125B；泛FP8/E4M3/E5M2的group128、4B scale只是一种可改存储代理，不代表scale recipe；MXFP8 block32的E8M0 scale一维代理约1.03125B/值；NVFP4 block16的一维聚合存储代理约0.5625B加每张量开销，Transformer Engine训练默认权重可用16×16二维scale，激活/梯度则是一维block16；转置副本、行列scale、padding、RHT与梯度随机舍入可能另增存储/算子成本；INT4 group128、2B scale与2B zero只是本情景选定的格式，不是全部INT4标准。[K3量化配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)、[NVFP4说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)。

NVIDIA Transformer Engine 已有低精度训练recipe：Hopper支持延迟缩放FP8，E4M3/E5M2用途与反向约束不同；Blackwell数据中心架构支持独立的MXFP8 block-scaled recipe与NVFP4 recipe。低位GEMM仍配合高精度状态、scale和适用层选择；不能说“4bit只能推理”，也不能反过来说“全参训练所有状态都4bit”。[TE支持矩阵](https://nvidia.github.io/TransformerEngine/support_matrix.html)。

AMD CDNA3/4/5格式支持不同，CDNA4/5列MXFP4并不证明任意NVFP4 checkpoint和K3训练路径直接可用。[AMD代际表](https://www.amd.com/en/technologies/cdna.html)。

Ascend910C论文依据是BF16/FP16和INT8。当前K3教程使用转换后的Eco-Tech W4A8检查点，是推理配方。固定vLLM-Ascend W4A8源码只接受per-channel权重，并对group_size>0显式报错；本预算的group128 INT4不能直接冒充该加载路径。AMCT全文一方面说HiF8/FP8/MXFP8/MXFP4/FP4仅950PR/DT，另一方面说当前版本仅INT8/INT4；Linear表还有不同的条件组合和shape/scale限制，文档内部存在版本/范围冲突。因此950格式只列条件候选，不能作K3或全参训练支持证据，也不能外推910C。完整K3全参FP8/FP4、NF4 QLoRA收敛和完整反向适配在本研究中仍未核验。[Ascend K3教程](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html)、[CANN类型限制](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/amct/atlasamct_16_0049.html)。

## 并行策略与公式

计算器使用一套明确的rank预算：`N=TP×PP×DP`；注意力有DP个副本，专家另有 `ETP×EP×EDP=TP×DP`，其中 `EDP=TP×DP/(ETP×EP)` 必须是正整数。EP不是再乘一次总卡数。TP与ETP分开，分别限制注意力与专家矩阵切分。SP取1或TP，不增加卡数。梯度同步默认每微批同步、梯度按ZeRO状态分片累积；可选延迟同步会保留本地全梯度并只在累积末同步一次，需要后端支持。真实框架支持的folding、CP、并行分组及expert loader仍需要独立核验。[Megatron策略](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[MoE并行](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)。

Ascend K3教程给出的TP16、DP4、EP64四节点推理例子，在本预算中对应ETP1、EDP1，共64逻辑NPU；这是并行计数解释，不能把本计算器的其他参数视为该教程的实测配置。该教程版本、旧源码快照和当前模型权重也不自动组成经过兼容性测试的组合。

令`Pd=Ptotal-Prouted`、`Pe=Prouted`、`Sd=TP×PP`、`Se=ETP×EP×PP`。按状态种类x的每参数字节bx，每rank常驻量为 `Pd*bx/(Sd*Zd) + Pe*bx/(Se*Ze)`。未分片时Zd=Ze=1；ZeRO1只令optimizer与master的Zd=DP、Ze=EDP；stage2再分梯度；stage3再分参数。梯度同步模式与该驻留量必须配套：默认“每微批同步”按DP/EDP分片累积并每个microbatch同步；“延迟同步”仅在后端支持no_sync/coalesced reduction时选用，保留未沿DP/EDP切分的本地全梯度峰值，整次累积后只同步一次。不能同时取延迟同步通信次数和分片梯度显存的好处。ZeRO不是把独有专家沿EP再除一次。FSDP full-shard与ZeRO3在这一常驻预算层面类似，但prefetch、reshard、通信与实现不同；需要另外加最大层gather峰值。F16同步代理按模式取syncCount：每微批模式为accum，延迟模式为1；本地梯度驻留随之采用分片或未分片状态，不能混搭。

LoRA/低位PEFT冻结底座在计算器中保守只按TP/EP/PP切分，不额外应用FSDP。adapter按其专家/非专家比例进入训练状态公式。支持冻结权重分片的真实实现可在专门方案中重新估算，不默认为此处已支持。

保存激活预算为 `B*T_effective*H*ceil(L/PP)*savedActBytes*保存系数*同时存活微批/SP`。savedActBytes与计算矩阵格式、通信payload字节分开；FP8/INT8算术选择不会自动把保存激活或TP/EP/PP通信缩成1字节。默认保存和通信payload均按BF16=2B；如后端实际格式不同，须分别修改并记录。这是可校准系数模型；不是从旧decode表复制的训练激活。FlashAttention、选择性/全量重计算、AttnRes、视觉、loss/vocab logits、pipeline调度都会改变系数和峰值。重计算会减少保存激活，但增加计算，需同时修改系数与重计算倍数；工具不假装自动推导。

K3 MLA缓存必须区分参考实现和部署后端。缓存峰值长度L_KV=history_context+effective_new_tokens；prefill的effective_new_tokens=T_configured、允许history_context=0；decode新增1 token，不能把界面保留的T_configured误当作decode实际新增长度。固定HF Transformers参考实现先将 `kv_b_proj` 展开后写入 `past_key_values`；每头保存key的128维NoPE分量+64维额外投影分量、value 128维；这里描述缓存形状，不宣称64维key已施加RoPE，故每个MLA层每token为 `96*(128+64+128)=30,720`元素。固定vLLM NVIDIA K3 MLA源码吸收 `kv_b_proj`，采用512维latent+64维key分量，即576元素；固定K3 NoPE路径不应被表述为已对64维key施加RoPE。该实现另有fp8_ds_mla专用656 B/token/layer布局（512×1B FP8值+4×4B FP32 scale+64×2B BF16值）。它是特定分支的物理字节布局，不能把576元素按FP8位宽换算为等价存储；用户可调的kvMeta在此代表额外页/allocator余量，不重复计算这组内置scale。三种是互斥backend布局，界面逐项选择；656B也不含allocator/page和实际分片保证。结构元素布局与656B专用cache布局是不同预算表达；allocator/page、量化scale和KV sharing仍需按实际部署核验。来源：[HF固定配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)、[HF参考实现](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)、[固定vLLM源码快照](https://github.com/vllm-project/vllm/blob/36229e1337ae0e44e60075a11ebfde210c11d706/vllm/models/kimi_k3/nvidia/mla.py)。5T/10T只把该几何当作可修改压力假设，不标为已验证。修改K3参数后，来源仍是基线依据，但不会再验证当前自定义参数、层放置或缓存结果；界面将它标成用户假设。其他模型默认不提供MLA/KDA参数，因此本工具不输出缓存容量；MHA/GQA/滑窗/SSM等需采用各自缓存公式，填入K3兼容字段也只代表人为假设。K3按连续PP stage及一基层号显式分配：MLA位于4、8、…、92、93层，其余69层为KDA；PP3的两类层数分别为7/8/9与24/23/22，PP93也按各rank真实的一层类型计，绝不按总层数均分。缓存预算取MLA、KDA和卷积各自最重stage层数，再独立相加；当各分支峰值stage不重合时这是保守上界，不是精确同rank峰值。用户改了K3结构层数时，因层放置未知，使用 `min(分支层数,ceil(总层数/PP))` 作为均衡连续stage可达到的最大分支层数。专家TP还要求3072维expert intermediate能被ETP整除；仅检查3584 dispatch width不够。KDA矩阵 `B*69*96*128*128*stateBytes`；短卷积 `B*69*3*96*128*convLength*convBytes`。默认矩阵/卷积FP32是预算假设。每项使用各自最重stage层数并除经后端证明的cache分片数，MLA另加scale/页余量。cache分片默认1（TP内复制）；不能默认KV随TP线性缩小。训练时这三项为0，历史推理cache不应作为训练缓存，训练中间状态由激活/workspace预算覆盖。其他模型的MHA/GQA/滑窗/SSM必须换自己的缓存模型，不能照套K3。

## 通信与性能计算的限度

MoE dispatch+combine每微批平均出站字节代理为 `2*passes*Lmoe_stage*(B*T/TP)*topK*dispatchWidth*commBytes*(1-1/EP)*imbalance`。该2分别计dispatch和combine；推理passes=1，训练passes=2以计前向与反向路由。只累加rank出站，不把接收重复计作发送。这里假设token均分到发送rank、无目的地去重。K3默认latent通信宽3584，实际backend若在投影前后不同位置通信必须修改。专家路由热点、drop/padding、TP复制/去重策略可改变流量，实际trace优先。

TP的ring allreduce每次发送 `2*(p-1)/p*M`；此预算前向每层2次、训练再加反向。SP下可改成allgather/reduce-scatter，其总字节相近但时间不能机械等同。DP梯度采用保守allreduce/Reduce-Scatter等效出站代理，逐微批模式同步次数=accum，延迟模式=1；其梯度驻留按相应模式计算。真实backend collective和分片语义须用trace核验，不能将代理值当测量。ZeRO3加前后向参数gather代理；PP边界发送按 `passes*B*T*H*commBytes` 计每rank发出的字节，不重复计接收；训练每个microbatch计前向激活和反向梯度两次，推理只计前向一次，PP=1时为0。梯度累积放大计算与TP/EP/PP流量；DP同步字节按显式模式乘accum或一次，延迟同步的更大本地梯度峰值同步计入驻留。这里的ZeRO3 gather按每微批前后向各一次，真实reshard/prefetch及重计算可显著改变次数，部署前必须以trace替换。

按每类通信的跨域比例c，时间代理为 `bytes*((1-c)/(BW_in*eta_in)+c/(BW_out*eta_out)) + calls*latency`。所有带宽都是每设备单向GB/s，效率无默认实测依据。真实ring/tree/rail、oversubscription、单peer带宽、bisection、共享链路和协议开销可能进一步限制。域大小只算域数，不自动替你推导rank映射或跨域比例。

算术粗预算：推理`2*Pactive*T`，全参训练`(6+2*recompute)*Pactive*T`，LoRA/QLoRA约`(4+2*recompute)*Pactive*T + 6*Padapter*T`；另乘用户指定额外FLOP比例。HBM仅按权重读比例、KV和KDA状态读写代理估计，未完整包含optimizer/gradient/activation流量。将组件max视为理想重叠组合、sum视为未重叠组合，再加简化PP系数和额外开销；二者不是严格上下界，也不是置信区间。未建模开销可以使实际耗时超过两者，默认效率不能用来排名三平台。

训练 tokens/s 应记录global batch、sequence length、有效非padding tokens和每optimizer step耗时。TTFT应分解queue、tokenize、视觉、prefill、首decode及PD迁移；TPOT应记录服务并发、输出长度、p50/p95/p99与SLO。计算器不生成虚构benchmark。prefill/decode分离还要算KV和KDA状态迁移、量化metadata、传输转换和网络竞争；容量可行不意味着分离优于共置。

## Ascend 优化执行顺序

P0先固定设备代际与版本，验证K3特有模块和前后向、量化与精度。P0同时检查910C die/rank映射与域内/域间通信能力。只有在基线正确且profile显示瓶颈后，才按P1处理MoE dispatch/combine与Grouped GEMM重叠、量化布局、最重stage、KV/KDA状态以及LoRA冻结范围。P2再比较图执行、CPU调度、PD分离与checkpoint/offload。完整条件、具体行动和质量/性能指标见[13项清单](ascend-priorities.csv)。

不能直接复制某平台融合kernel当作另一平台的性能解法。Ascend已有SiTU、KDA、AttnRes和条件量化路径，先确认是否命中、shape是否覆盖以及工作区代价；“存在fallback”不等于当前运行一定落在fallback。收益判断采用同模型revision、同精度质量阈值、同拓扑和同SLO下的端到端变化，不事先给加速百分比。[固定Ascend K3实现](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py)。

## 交付范围与保留缺口

专题提供K3、5T、10T的情景矩阵，三平台参考规格、精度支持、公式和Ascend优化CSV，以及156个原模型条目的独立超节点补充CSV和入口。只有K3在此拥有已导入的完整参数/混合缓存基线；其他模型补充文件明确列出尚缺的运行字段，不以原有配置层展开宣称完成超节点训练分析。旧156份XLSX及其数据CSV不修改，避免让新增情景改变已审数据含义。

仍缺实际超节点型号/板卡暴露方式、网络rank映射、K3训练后端完整支持/收敛、各量化格式数值质量、实际gather/通信/activation峰值、checkpoint恢复与故障冗余配置。用户指定设备后可以替换情景输入，但不能在设备测试前把状态改成“性能已验证”。旧站的34个配置缺口、K3原模板1,932项待核验与HerculesBench证据不足仍保留。
