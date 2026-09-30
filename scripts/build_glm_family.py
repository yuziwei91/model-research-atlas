"""Author the GLM family research data from the pinned CANN recipe snapshot.

All facts are transcribed from cann-recipes-infer at commit
96e5813f3a62686c9f1f9abba7d07dcd94687971. Because that repository is Huawei's inference
recipe rather than the model vendor's own model card, every transcribed fact carries
evidence "derived" and points at the pinned blob it was read from. Nothing here is a
weight-header audit and nothing here is a reproduced benchmark.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/families/glm'
COMMIT = '96e5813f3a62686c9f1f9abba7d07dcd94687971'
BLOB = f'https://gitcode.com/cann/cann-recipes-infer/blob/{COMMIT}/'

CFG53 = BLOB + 'models/glm_5_3/models/configuration_glm53.py'
README53 = BLOB + 'models/glm_5_3/README.md'
CFG5 = BLOB + 'models/glm_5/models/configuration_glm.py'
CFG52 = BLOB + 'models/glm_5_2/models/configuration_glm.py'
README5 = BLOB + 'models/glm_5/README.md'
README52 = BLOB + 'models/glm_5_2/README.md'


def f(value, source, evidence='derived', note=None):
    d = dict(value=value, evidence=evidence, source=source)
    if note:
        d['note'] = note
    return d


def unknown(source, note=None):
    return f(None, source, 'unknown', note)


NO_VISION_53 = '发布 config.json 为多模态嵌套结构，本配方丢弃 vision_config 并在加载时跳过 model.visual.*，因此视觉通路未进入本次研究。'

MODELS = [
    dict(
        id='glm-5',
        name='GLM-5',
        branch='主线',
        summary='78 层 DSA（MLA + Lightning Indexer）+ MoE + MTP；配方自述结构与 DeepSeek-V3.2-Exp 基本一致，在 Atlas A3 上运行。',
        tags=['长文档', '通用对话'],
        source='https://huggingface.co/zai-org/GLM-5',
        summaryEvidence='derived',
        sections=['s2', 's3'],
        facts=dict(
            layers=f(78, CFG5), hidden=f(6144, CFG5), heads=f(64, CFG5),
            attention=f('78 层 DSA：MLA + Lightning Indexer', README5),
            experts=f(256, CFG5), topK=f(8, CFG5), shared=f(1, CFG5),
            expertInput=f(6144, CFG5), expertWidth=f(2048, CFG5), denseWidth=f(12288, CFG5),
            context=f(202752, CFG5, note='配置上限，不等于实测质量或硬件运行上限'),
            vocab=f(154880, CFG5),
            visionLayers=unknown(CFG5), visionHidden=unknown(CFG5),
            quantization=f('发布 BF16 与 FP8 两种权重；配方转换为 W8A8（int8）或 MXFP8', README5),
            parameters=unknown(CFG5, '本站未做权重审计'),
            payload=unknown(CFG5, '本站未做权重审计'),
        ),
        auditPath=None, configPath=None, architecturePath=None, revision=None,
    ),
    dict(
        id='glm-5.2',
        name='GLM-5.2',
        branch='主线',
        summary='沿用 GLM-5 的 78 层 DSA + MoE + MTP 主干；核心新增是 IndexShare —— 部分层复用上一个 full 层的 top-k，不再持有自己的 indexer 权重。',
        tags=['长文档', '通用对话'],
        source='https://huggingface.co/zai-org/GLM-5.2',
        summaryEvidence='derived',
        sections=['s2', 's3', 's4'],
        facts=dict(
            layers=f(78, CFG52), hidden=f(6144, CFG52), heads=f(64, CFG52),
            attention=f('78 层 DSA：MLA + Lightning Indexer，按 indexer_types 分 full / shared（IndexShare）', README52),
            experts=f(256, CFG52), topK=f(8, CFG52), shared=f(1, CFG52),
            expertInput=f(6144, CFG52), expertWidth=f(2048, CFG52), denseWidth=f(12288, CFG52),
            context=f(202752, CFG52, note='配置上限，不等于实测质量或硬件运行上限'),
            vocab=f(154880, CFG52),
            visionLayers=unknown(CFG52), visionHidden=unknown(CFG52),
            quantization=f('发布 BF16 权重；配方转换为 W8A8（int8）', README52),
            parameters=unknown(CFG52, '本站未做权重审计'),
            payload=unknown(CFG52, '本站未做权重审计'),
        ),
        auditPath=None, configPath=None, architecturePath=None, revision=None,
    ),
    dict(
        id='glm-5.3-flash',
        name='GLM-5.3-Flash',
        branch='主线',
        summary='45 层混合注意力：34 层 KDA 线性注意力 + 11 层 DSA（全程 NoPE 的吸收式 MLA + k-pool 压缩 indexer），叠加 4 流 mHC 超连接与 288 选 8 MoE。层数与残差宽度都比 GLM-5.2 小，上下文配置上限反而提升到 1,048,576。',
        tags=['长文档', '通用对话', '高吞吐部署'],
        source='https://huggingface.co/zai-org/GLM-5.3-Flash',
        summaryEvidence='derived',
        sections=['s2', 's3', 's4', 's5', 's6'],
        facts=dict(
            layers=f(45, CFG53), hidden=f(4096, CFG53), heads=f(64, CFG53),
            attention=f('34 KDA + 11 DSA（MLA 全程 NoPE + k-pool 压缩 indexer）', CFG53,
                        note='排布来自配置默认值 layer_types：i%4==3 为 DSA'),
            experts=f(288, CFG53), topK=f(8, CFG53), shared=f(1, CFG53),
            expertInput=f(4096, CFG53), expertWidth=f(2048, CFG53), denseWidth=f(12288, CFG53),
            context=f(1048576, CFG53, note='配置上限，不等于实测质量或硬件运行上限'),
            vocab=f(154880, CFG53),
            visionLayers=unknown(CFG53, NO_VISION_53),
            visionHidden=unknown(CFG53, NO_VISION_53),
            quantization=f('发布 FP8；本配方部署为 Hybrid HiF8-MXFP8-MXFP4', README53),
            parameters=unknown(README53,
                               'README 自述约 306B。本站未做权重审计；按配方配置默认值逐矩阵推导为 313,326,811,966，'
                               '两者相差约一层 MoE，详见结构页与来源页的口径说明。'),
            payload=unknown(README53, '本站未做权重审计'),
        ),
        auditPath=None,
        configPath=None,
        architecturePath='data/families/glm/glm-5.3-flash-architecture.json',
        revision=None,
        implementationLinks=[
            dict(topic='hybrid-attention', path='implementation/all/glm-5.3-flash'),
            dict(topic='mhc', path='implementation/all/glm-5.3-flash'),
            dict(topic='hybrid-quant', path='implementation/all/glm-5.3-flash'),
        ],
    ),
]

FAMILY = dict(
    id='glm',
    name='GLM',
    publisher='智谱 Zhipu AI',
    updated='2026-09-30',
    description='从 GLM-5 / GLM-5.2 的 DSA + MoE 主干，到 GLM-5.3-Flash 的 KDA 混合注意力、mHC 超连接与昇腾 NPU 上的 Hybrid HiF8-MXFP8-MXFP4 部署。',
    scope=(
        '3 个名称条目，全部通过 CANN 推理配方 cann-recipes-infer 的固定快照 96e5813 研究，'
        '不是智谱全部公开模型，也不是厂商模型卡的逐项复核。'
        '只有 GLM-5.3-Flash 有逐矩阵结构；GLM-5 与 GLM-5.2 仅记录配置级规格。'
        '本站未加载权重、未做文件头审计，也未运行任何 NPU 性能实验。'
    ),
    models=MODELS,
    figures=[],
    downloads=[
        dict(name='GLM-implementation-hardware-ascend.md',
             path='assets/glm/GLM-implementation-hardware-ascend.md', bytes=0, sha256=''),
        dict(name='GLM-hardware-sources.csv',
             path='assets/glm/GLM-hardware-sources.csv', bytes=0, sha256=''),
    ],
    reportPath='data/families/glm/report.json',
    officialDirectory='https://huggingface.co/zai-org',
    scenarios=[
        dict(
            title='长上下文阅读与检索',
            text='GLM-5.3-Flash 的配置上下文上限为 1,048,576，是 GLM-5 / GLM-5.2 的 202,752 的约 5 倍；34 层 KDA 用固定大小的递推状态代替随长度增长的 KV，11 层 DSA 再用 k-pool 压缩 indexer 做 top-k 稀疏选择。',
            models=['glm-5.3-flash', 'glm-5.2', 'glm-5'],
            boundary='配置上限不等于实测可用长度。混合注意力的长程质量需要固定数据集与评测协议单独验证，本站没有做。',
        ),
        dict(
            title='高吞吐在线服务',
            text='配方给出昇腾 950DT 八卡、8K 序列下的 decode 每步耗时与 TPS，并默认开启共享专家多流与 npugraph_ex 图模式。',
            models=['glm-5.3-flash'],
            boundary='这些数字来自配方 README 自述，本站未复现。比较吞吐必须同时固定卡数、序列长度、批大小、量化格式与图模式。',
        ),
        dict(
            title='昇腾平台迁移',
            text='同一套配方里 GLM-5 与 GLM-5.2 面向 Atlas A3、走 W8A8/MXFP8；GLM-5.3-Flash 面向 950DT、走 Hybrid HiF8-MXFP8-MXFP4，并依赖 mHC 与 causal_conv1d 两个自定义算子包。',
            models=['glm-5', 'glm-5.2', 'glm-5.3-flash'],
            boundary='配方存在不等于本站完成部署。算子包、CANN 版本与固件驱动版本必须逐项对齐后才能比较。',
        ),
    ],
    overview=dict(
        title='主干缩小，注意力混合，精度分层',
        conclusion='GLM-5 与 GLM-5.2 共用 78 层 DSA + MoE 主干，5.2 的增量是 IndexShare。GLM-5.3-Flash 改动更大：层数降到 45、残差宽度降到 4096，但把 34 层换成 KDA 线性注意力、加入 4 流 mHC 超连接，并在昇腾侧按模块分配 HiF8 / MXFP8 / MXFP4 三种精度。',
        highlights=[
            dict(id='glm-5', label='GLM-5 · DSA 主干'),
            dict(id='glm-5.2', label='GLM-5.2 · IndexShare'),
            dict(id='glm-5.3-flash', label='GLM-5.3-Flash · 混合注意力'),
        ],
    ),
    historyNote='顺序按配方仓库中样例的适配关系整理：GLM-5.2 样例自述基于 GLM-5 样例迁移，GLM-5.3 是独立的新样例。这是推理配方的适配顺序，不是权重继承证明。',
    auditNote='本次研究没有加载任何权重，也没有做文件头审计。GLM-5.3-Flash 的逐矩阵形状来自实现源码中参数的声明形状，逐项相乘得到逻辑参数；GLM-5 与 GLM-5.2 只记录配置类默认值。',
    limitations='结构排布取自配置类默认值，发布 config.json 可能覆盖。按默认值推导的 GLM-5.3-Flash 参数总数为 313,326,811,966，与 README 自述的约 306B 相差约一层 MoE，本站未消解该差异。配方 README 的吞吐数字未经本站复现。',
    technology=[
        dict(id='evolution', title='演进', section='s2',
             summary='GLM-5 → GLM-5.2 的增量是 IndexShare；GLM-5.3-Flash 换了注意力组成、残差结构与部署精度。'),
        dict(id='spec', title='规格', section='s3',
             summary='三个版本的层数、宽度、专家数与上下文上限并列；规模比较必须区分总参数与激活参数。'),
        dict(id='attention', title='注意力', section='s4',
             summary='从全 DSA 到 34 KDA + 11 DSA 的混合排布，以及 k-pool 压缩 indexer 与全程 NoPE 的吸收式 MLA。'),
        dict(id='mhc', title='mHC 超连接', section='s5',
             summary='层间隐藏态是 4 条并行流；每个子层前后各用一组 Sinkhorn 归一化的混合矩阵折叠与展开。'),
        dict(id='quant', title='混合精度与 NPU 优化', section='s6',
             summary='按模块分配 HiF8 / MXFP8 / MXFP4，并保留一批必须留在 BF16/FP32 的张量；配套自定义算子与多流重叠。'),
    ],
    hardwarePath='data/families/glm/hardware.json',
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'family.json').write_text(
        json.dumps(FAMILY, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'GLM family: {len(MODELS)} model entries, '
          f'{len(FAMILY["technology"])} technology topics, '
          f'{len(FAMILY["scenarios"])} scenarios.')


if __name__ == '__main__':
    main()
