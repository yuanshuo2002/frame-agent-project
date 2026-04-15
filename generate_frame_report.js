/**
 * FRAME 论文复现项目 - 给导师的汇报 PPT (简洁版)
 * 生成命令: node generate_frame_report.js
 *
 * 简化原则:
 * 1. 每页聚焦1-2个要点，留足呼吸空间
 * 2. 减少复杂图形，改用清晰的列表/表格布局
 * 3. 增大元素间距，杜绝重叠
 * 4. 总页数控制在10页左右
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

// ============================================================
// 配色方案: 学术简洁风
// ============================================================
const C = {
    primary: "1E2761",       // 深海军蓝 (标题)
    accent: "0EA5E9",        // 亮蓝 (强调)
    dark: "0F172A",          // 近黑 (封面背景)
    white: "FFFFFF",
    lightBg: "F8FAFC",       // 浅灰白 (内容页背景)
    text: "334155",          // 深灰 (正文)
    muted: "64748B",         // 灰色 (次要文字)
    success: "059669",       // 绿色
    warning: "D97706",       // 橙色
    error: "DC2626",         // 红色
    border: "E2E8F0",        // 边框灰
};

// ============================================================
// 创建演示文稿
// ============================================================
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "FRAME 复现团队";
pres.title = "FRAME 论文复现项目进展报告";

const FONT = "Arial";

// ============================================================
// 辅助函数
// ============================================================

/** 封面页 */
function coverSlide(title, subtitle) {
    const s = pres.addSlide();
    s.background = { color: C.dark };

    // 装饰线
    s.addShape(pres.shapes.RECTANGLE, {
        x: 2.5, y: 2.35, w: 5, h: 0.03,
        fill: { color: C.accent },
    });

    s.addText(title, {
        x: 0.5, y: 1.5, w: 9, h: 0.85,
        fontSize: 36, fontFace: FONT, bold: true,
        color: C.white, align: "center",
    });

    if (subtitle) {
        s.addText(subtitle, {
            x: 0.5, y: 2.55, w: 9, h: 0.5,
            fontSize: 18, fontFace: FONT,
            color: C.secondary || "CADCFC", align: "center",
        });
    }

    return s;
}

/** 内容页模板 — 统一标题栏 */
function contentSlide(title) {
    const s = pres.addSlide();
    s.background = { color: C.lightBg };

    // 顶部色条
    s.addShape(pres.shapes.RECTANGLE, {
        x: 0, y: 0, w: 10, h: 0.85,
        fill: { color: C.primary },
    });

    s.addText(title, {
        x: 0.4, y: 0.2, w: 9.2, h: 0.5,
        fontSize: 24, fontFace: FONT, bold: true,
        color: C.white,
    });

    return s;
}

/** 在内容区添加一个简洁的信息块（无阴影、无嵌套） */
function infoBlock(s, x, y, w, h, title, lines, opts) {
    var titleColor = (opts && opts.titleColor) ? opts.titleColor : C.primary;

    // 外框
    s.addShape(pres.shapes.RECTANGLE, {
        x: x, y: y, w: w, h: h,
        fill: { color: C.white },
        line: { color: C.border, width: 0.7 },
    });

    // 标题栏
    if (title) {
        s.addShape(pres.shapes.RECTANGLE, {
            x: x, y: y, w: w, h: 0.36,
            fill: { color: titleColor },
        });
        s.addText(title, {
            x: x + 0.12, y: y + 0.05, w: w - 0.24, h: 0.28,
            fontSize: 11.5, fontFace: FONT, bold: true,
            color: C.white,
        });
    }

    // 内容文本
    if (lines && lines.length > 0) {
        var textY = title ? y + 0.42 : y + 0.12;
        var textH = title ? h - 0.52 : h - 0.18;
        s.addText(lines, {
            x: x + 0.15, y: textY, w: w - 0.3, h: textH,
            fontSize: 11, fontFace: FONT,
            color: C.text, valign: "top",
            paraSpaceAfter: 5,
        });
    }
}


// ============================================================
// Slide 1: 封面
// ============================================================
var s1 = coverSlide(
    "FRAME 论文复现项目进展报告",
    "Feedback-Refined Agent Methodology for Medical Research Insights"
);

s1.addText("汇报人：[你的名字]          日期：2026年4月          导师：[导师姓名]", {
    x: 0.5, y: 3.5, w: 9, h: 0.5,
    fontSize: 14, fontFace: FONT, color: "94A3B8", align: "center",
});


// ============================================================
// Slide 2: 目录
// ============================================================
var s2 = contentSlide("目录");

var toc = [
    "01   研究背景与 FRAME 核心方法",
    "02   系统架构与技术栈",
    "03   Phase 2-3：数据集构建与训练循环",
    "04   Phase 4-5：推理与评估",
    "05   实验环境与部署配置",
    "06   当前进度与成果",
    "07   遇到的问题与解决方案",
    "08   下一步计划与总结",
];

for (var ti = 0; ti < toc.length; ti++) {
    var row = Math.floor(ti / 2);
    var col = ti % 2;
    s2.addText(toc[ti], {
        x: 0.6 + col * 4.6, y: 1.15 + row * 0.95, w: 4.3, h: 0.45,
        fontSize: 15, fontFace: FONT,
        color: C.text,
    });
}


// ============================================================
// Slide 3: 研究背景 + FRAME 方法 (合并原 slide 3+4)
// ============================================================
var s3 = contentSlide("研究背景与 FRAME 核心方法");

// --- 左侧: 背景 ---
infoBlock(s3, 0.35, 1.05, 4.55, 2.05, "研究背景", [
    { text: "LLM 论文生成质量参差不齐，缺乏反馈闭环", options: { bullet: true, breakLine: false } },
    { text: "传统 RAG 仅做向量检索，无迭代优化能力", options: { bullet: true, breakLine: false } },
    { text: "医学论文对准确性、完整性要求极高", options: { bullet: true } },
], { titleColor: C.primary });

// --- 右侧: G-E-R 流程 ---
s3.addText("G → E → R 三 Agent 闭环训练", {
    x: 5.1, y: 1.05, w: 4.4, h: 0.32,
    fontSize: 13, fontFace: FONT, bold: true, color: C.primary,
});

var gerBoxes = [
    { label: "Generator", desc: "生成论文章节", x: 5.1, c: "3B82F6" },
    { label: "Evaluator", desc: "多维度评分(1-5)", x: 6.65, c: "F59E0B" },
    { label: "Reflector", desc: "改进建议报告", x: 8.2, c: "10B981" },
];
for (var gi = 0; gi < gerBoxes.length; gi++) {
    var g = gerBoxes[gi];
    s3.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: g.x, y: 1.48, w: 1.42, h: 0.78,
        fill: { color: g.c }, rectRadius: 0.06,
    });
    s3.addText(g.label, {
        x: g.x, y: 1.51, w: 1.42, h: 0.38,
        fontSize: 12, fontFace: FONT, bold: true, color: C.white, align: "center",
    });
    s3.addText(g.desc, {
        x: g.x, y: 1.9, w: 1.42, h: 0.3,
        fontSize: 9, fontFace: FONT, color: "E2E8F0", align: "center",
    });
    if (gi < 2) {
        s3.addShape(pres.shapes.LINE, {
            x: g.x + 1.44, y: 1.87, w: 0.17, h: 0,
            line: { color: C.muted, width: 1.5, endArrowType: 'triangle' },
        });
    }
}
s3.addText("N轮迭代 → Reflection Reports 存入 FAISS 向量库", {
    x: 5.1, y: 2.35, w: 4.4, h: 0.28,
    fontSize: 10, fontFace: FONT, color: C.success, bold: true,
});

// --- 推理流水线 ---
s3.addText("推理阶段：Topic → RAG检索 → Filter筛选 → Integrator合并 → Generator生成", {
    x: 5.1, y: 2.75, w: 4.4, h: 0.35,
    fontSize: 10.5, fontFace: FONT, color: C.text,
});

// --- 底部: 对比数据 ---
infoBlock(s3, 0.35, 3.25, 9.05, 1.95, "实验效果对比（原论文数据）", [
    { text: "No-RAG (纯 LLM)：77.86 分     RAG (基线)：85.18 分 (+9.5%)     Ours (FRAME)：90.19 分 (+15.9%)", options: {} },
    { text: "评估体系：Soft Precision / Recall 统计指标  +  LLM 多维度评分 (6章节 × 6维度)", options: {} },
], { titleColor: C.accent });


// ============================================================
// Slide 4: 系统架构与技术栈
// ============================================================
var s4 = contentSlide("系统架构与技术栈");

// 架构组件 — 两列布局
var archLeft = [
    { t: "数据源", d: "medRxiv / Europe PMC 医学论文" },
    { t: "Phase 2", d: "Extractor → Checker 迭代提取" },
    { t: "Phase 3", d: "G-E-R 训练循环 (N轮)" },
    { t: "Phase 4", d: "RAG + Filter + Integrate 推理" },
    { t: "Phase 5", d: "统计指标 + LLM 评分评估" },
];

for (var li = 0; li < archLeft.length; li++) {
    var ly = 1.1 + li * 0.82;
    s4.addShape(pres.shapes.RECTANGLE, {
        x: 0.35, y: ly, w: 0.06, h: 0.68,
        fill: { color: C.primary },
    });
    s4.addText(archLeft[li].t, {
        x: 0.55, y: ly, w: 1.5, h: 0.3,
        fontSize: 13, fontFace: FONT, bold: true, color: C.primary,
    });
    s4.addText(archLeft[li].d, {
        x: 0.55, y: ly + 0.32, w: 4.0, h: 0.32,
        fontSize: 11, fontFace: FONT, color: C.text,
    });
}

// 右侧技术栈
infoBlock(s4, 5.0, 1.1, 4.45, 2.25, "技术栈", [
    { text: "模型: Qwen2.5-7B-Instruct (vLLM 加速推理)", options: { bullet: true, breakLine: false } },
    { text: "Embedding: 阿里云 text-embedding-v1 (1536维)", options: { bullet: true, breakLine: false } },
    { text: "向量库: FAISS IndexFlatIP 内积索引", options: { bullet: true, breakLine: false } },
    { text: "框架: Python + PyTorch + sentence-transformers", options: { bullet: true } },
], { titleColor: C.accent });

// 项目结构
s4.addText("项目结构", {
    x: 0.35, y: 3.5, w: 3, h: 0.3,
    fontSize: 13, fontFace: FONT, bold: true, color: C.primary,
});
s4.addText([
    { text: "src/agents/      — Generator / Evaluator / Reflector\n", options: {} },
    { text: "src/inference/   — Retriever / Filter / Integrator\n", options: {} },
    { text: "src/dataset/     — Extractor / Checker / Builder\n", options: {} },
    { text: "experiments/    — run_dataset_build / training / inference / evaluation", options: {} },
], {
    x: 0.35, y: 3.85, w: 9.05, h: 1.35,
    fontSize: 11, fontFace: "Consolas", color: C.text,
});


// ============================================================
// Slide 5: Phase 2 数据集构建
// ============================================================
var s5 = contentSlide("Phase 2: 数据集构建流程");

var steps = [
    { n: "1", t: "数据采集", d: "从 medRxiv / Europe PMC 下载医学论文元数据和全文" },
    { n: "2", t: "Stage 1 过滤", d: "基础质量筛选（最小字数、格式检查）" },
    { n: "3", t: "Extractor → Checker", d: "N 轮迭代提取 6 个章节 (topic/background/methodology 等)" },
    { n: "4", t: "Stage 2/3 过滤 + 划分", d: "结构完整性检查 → 训练集/测试集划分 (92:8)" },
];

for (var si = 0; si < steps.length; si++) {
    var st = steps[si];
    var sy = 1.15 + si * 0.92;

    // 编号圆圈
    s5.addShape(pres.shapes.OVAL, {
        x: 0.4, y: sy, w: 0.46, h: 0.46,
        fill: { color: C.primary },
    });
    s5.addText(st.n, {
        x: 0.4, y: sy + 0.06, w: 0.46, h: 0.36,
        fontSize: 17, fontFace: FONT, bold: true,
        color: C.white, align: "center",
    });

    // 步骤名称 + 描述
    s5.addText(st.t, {
        x: 1.02, y: sy, w: 2.3, h: 0.34,
        fontSize: 14, fontFace: FONT, bold: true, color: C.primary,
    });
    s5.addText(st.d, {
        x: 1.02, y: sy + 0.36, w: 8.2, h: 0.38,
        fontSize: 11.5, fontFace: FONT, color: C.text,
    });

    // 连接线
    if (si < steps.length - 1) {
        s5.addShape(pres.shapes.LINE, {
            x: 0.63, y: sy + 0.5, w: 0, h: 0.38,
            line: { color: C.border, width: 1.5, dashType: 'dash' },
        });
    }
}

// 规模信息
s5.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 4.88, w: 9.05, h: 0.58,
    fill: { color: "EFF6FF" },
    line: { color: C.accent, width: 0.7 },
});
s5.addText([
    { text: "当前规模: ", options: { bold: true, breakLine: false } },
    { text: "20 篇 × 6 章节 = 120 个样本   |   方向: DL医学影像 / 联邦学习 / NLP临床笔记   |   入口: run_dataset_build.py --raw_dir data/raw/train_only/", options: { color: C.primary } },
], {
    x: 0.5, y: 4.98, w: 8.8, h: 0.38,
    fontSize: 11, fontFace: FONT, valign: "middle",
});


// ============================================================
// Slide 6: Phase 3 训练循环 (核心)
// ============================================================
var s6 = contentSlide("Phase 3: G→E→R 训练循环（核心）");

// Round 1 & Round 2 并排
var rounds = [
    { label: "Round 1", yBase: 1.1, color: C.primary },
    { label: "Round 2", yBase: 2.65, color: C.accent },
];

for (var ri = 0; ri < rounds.length; ri++) {
    var rnd = rounds[ri];
    s6.addText(rnd.label, {
        x: 0.35, y: rnd.yBase, w: 1.0, h: 0.28,
        fontSize: 13, fontFace: FONT, bold: true, color: rnd.color,
    });

    var agents = [
        { abbr: "G", name: "Generator", action: ri === 0 ? "基于原文生成章节初稿" : "结合反馈生成改进版", c: "3B82F6" },
        { abbr: "E", name: "Evaluator", action: "6维度×1-5分评分", c: "F59E0B" },
        { abbr: "R", name: "Reflector", action: "生成 Reflection Report", c: "10B981" },
    ];

    for (var ai = 0; ai < agents.length; ai++) {
        var ag = agents[ai];
        var ax = 1.35 + ai * 2.75;

        s6.addShape(pres.shapes.ROUNDED_RECTANGLE, {
            x: ax, y: rnd.yBase, w: 2.55, h: 0.95,
            fill: { color: ag.c }, rectRadius: 0.06,
        });
        s6.addText(ag.abbr, {
            x: ax + 0.08, y: rnd.yBase + 0.08, w: 0.45, h: 0.45,
            fontSize: 22, fontFace: FONT, bold: true, color: C.white,
        });
        s6.addText(ag.name, {
            x: ax + 0.55, y: rnd.yBase + 0.1, w: 1.9, h: 0.28,
            fontSize: 12, fontFace: FONT, bold: true, color: C.white,
        });
        s6.addText(ag.action, {
            x: ax + 0.55, y: rnd.yBase + 0.42, w: 1.9, h: 0.43,
            fontSize: 10, fontFace: FONT, color: "E2E8F0",
        });

        if (ai < 2) {
            s6.addShape(pres.shapes.LINE, {
                x: ax + 2.57, y: rnd.yBase + 0.47, w: 0.16, h: 0,
                line: { color: C.white, width: 1.5, endArrowType: 'triangle' },
            });
        }
    }
}

// Round 间箭头
s6.addShape(pres.shapes.LINE, {
    x: 6.5, y: 2.1, w: 0, h: 0.45,
    line: { color: C.muted, width: 1.5, dashType: 'dash', endArrowType: 'triangle' },
});
s6.addText("反馈传入下一轮", {
    x: 6.65, y: 2.26, w: 1.4, h: 0.22,
    fontSize: 9, fontFace: FONT, color: C.muted,
});

// 右侧规模面板
infoBlock(s6, 5.0, 3.72, 4.45, 1.68, "训练规模", [
    { text: "20 篇论文 × 6 章 = 120 个任务", options: { breakLine: false } },
    { text: "每任务 G-E-R = 3 次 LLM 调用", options: { breakLine: false } },
    { text: "总计 ~360 次 API 调用 → Reflection Reports → FAISS", options: {} },
], { titleColor: C.success });

s6.addText("> python experiments/run_training.py --rounds 1", {
    x: 0.35, y: 5.15, w: 5, h: 0.28,
    fontSize: 10, fontFace: "Consolas", color: C.primary,
});


// ============================================================
// Slide 7: Phase 4-5 推理与评估 (合并)
// ============================================================
var s7 = contentSlide("Phase 4-5: 推理流水线与评估体系");

// 左侧: 推理流水线
s7.addText("推理流水线 (5步)", {
    x: 0.35, y: 1.08, w: 3.5, h: 0.28,
    fontSize: 13, fontFace: FONT, bold: true, color: C.primary,
});

var pipeSteps = [
    ["① 输入研究主题", "e.g. Deep Learning for CT-based Medical Image Analysis"],
    ["② FAISS 向量检索", "Top-K=5 相似度检索 Reflection Reports"],
    ["③ Filter 二次筛选", "LLM 判断相关性 (阈值≥0.6)，去除噪声"],
    ["④ Integrator 合并", "多份报告合并为统一上下文"],
    ["⑤ Generator 生成", "基于合并上下文生成最终章节内容"],
];

for (var psi = 0; psi < pipeSteps.length; psi++) {
    var ps = pipeSteps[psi];
    var py = 1.42 + psi * 0.62;
    s7.addText(ps[0], {
        x: 0.35, y: py, w: 2.2, h: 0.26,
        fontSize: 11.5, fontFace: FONT, bold: true, color: C.accent,
    });
    s7.addText(ps[1], {
        x: 0.35, y: py + 0.26, w: 4.8, h: 0.28,
        fontSize: 10, fontFace: FONT, color: C.text,
    });
}

// 右侧: 对比实验
infoBlock(s7, 5.2, 1.08, 4.3, 2.55, "三组对比实验", [
    { text: "No-RAG: 直接 LLM 生成（无外部知识）— 基线", options: { bullet: true, breakLine: false } },
    { text: "RAG: FAISS 检索后拼接第一份报告 — 传统基线", options: { bullet: true, breakLine: false } },
    { text: "Ours: RAG + Filter(相关性筛选) + Integrator(多报告合并) — 完整 FRAME", options: { bullet: true } },
], { titleColor: C.warning });

// 评估体系
infoBlock(s7, 0.35, 3.78, 4.55, 1.58, "评估体系 (双轨)", [
    { text: "统计指标: Soft Precision / Recall (Embedding 相似度矩阵, 阈值0.7)", options: { bullet: true, breakLine: false } },
    { text: "LLM 评分: 复用 Evaluator Agent, 6章节×6维度, 1-5分 (0.1步进)", options: { bullet: true } },
], { titleColor: "8B5CF6" });

// 命令提示
s7.addText("> python experiments/run_inference.py --mode comparison --demo", {
    x: 0.35, y: 5.15, w: 4.5, h: 0.25,
    fontSize: 10, fontFace: "Consolas", color: C.primary,
});


// ============================================================
// Slide 8: 实验环境与部署
// ============================================================
var s8 = contentSlide("实验环境与部署配置");

// 两列卡片
infoBlock(s8, 0.35, 1.08, 4.55, 1.75, "硬件环境", [
    { text: "GPU: NVIDIA RTX 3090 × 1 (24GB VRAM)", options: { bullet: true, breakLine: false } },
    { text: "平台: AutoDL 云GPU + SSH 隧道 (本地10088→远端8800)", options: { bullet: true, breakLine: false } },
    { text: "本地: Windows PowerShell + Python 3.11", options: { bullet: true } },
], { titleColor: C.primary });

infoBlock(s8, 5.1, 1.08, 4.45, 1.75, "vLLM 配置", [
    { text: "模型: Qwen/Qwen2.5-7B-Instruct", options: { bullet: true, breakLine: false } },
    { text: "max-model-len: 16384 | gpu-memory-util: 0.92", options: { bullet: true, breakLine: false } },
    { text: "max-num-seqs: 4 | port: 8800 | api-key: dummy", options: { bullet: true } },
], { titleColor: C.accent });

// 启动命令
s8.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 2.97, w: 9.05, h: 1.22,
    fill: { color: "1E293B" },
});
s8.addText("# AutoDL 启动 vLLM 服务命令:", {
    x: 0.5, y: 3.07, w: 8.8, h: 0.25,
    fontSize: 10.5, fontFace: FONT, bold: true, color: "93C5FD",
});
s8.addText(
    "python -m vllm.entrypoints.openai.api_server \\\n" +
    "    --model Qwen/Qwen2.5-7B-Instruct \\\n" +
    "    --tensor-parallel-size 1 --dtype float16 \\\n" +
    "    --max-model-len 16384 --gpu-memory-utilization 0.92 \\\n" +
    "    --max-num-seqs 4 --port 8800 --api-key dummy"
, {
    x: 0.5, y: 3.35, w: 8.8, h: 0.76,
    fontSize: 10, fontFace: "Consolas", color: "A5B4FC",
});

// 关键技术决策
infoBlock(s8, 0.35, 4.32, 9.05, 1.15, "关键技术决策", [
    { text: "LLM Client: 用 requests 替代 OpenAI SDK → 解决 vLLM 502 兼容性问题", options: { bullet: true, breakLine: false } },
    { text: "Embedding: 阿里云百炼 text-embedding-v1 (1536维) → 无需本地 GPU 资源", options: { bullet: true, breakLine: false } },
    { text: "FAISS: IndexFlatIP 内积索引 → 小数据量足够，归一化后等价余弦相似度", options: { bullet: true } },
], { titleColor: C.success });


// ============================================================
// Slide 9: 当前进度 + 问题解决 (合并)
// ============================================================
var s9 = contentSlide("当前进展与问题解决");

// 进度时间线
var phases = [
    { p: "Phase 1", n: "数据准备", st: "已完成", d: "收集 3 方向医学论文", sc: C.success },
    { p: "Phase 2", n: "数据集构建", st: "已完成", d: "20篇×6章=120样本, dataset_full.json 已生成", sc: C.success },
    { p: "Phase 3", n: "G-E-R 训练", st: "进行中", d: "vLLM ~9-17s/轮, 预计~1.3h完成全部", sc: C.warning },
    { p: "Phase 4", n: "推理生成", st: "待执行", d: "三组对比实验 No-RAG vs RAG vs Ours", sc: C.muted },
    { p: "Phase 5", n: "评估分析", st: "待执行", d: "统计指标 + LLM评分, 生成完整报告", sc: C.muted },
];

for (var phi = 0; phi < phases.length; phi++) {
    var ph = phases[phi];
    var phy = 1.08 + phi * 0.73;

    // 圆点
    s9.addShape(pres.shapes.OVAL, {
        x: 0.38, y: phy + 0.08, w: 0.24, h: 0.24,
        fill: { color: ph.sc },
    });
    if (phi < phases.length - 1) {
        s9.addShape(pres.shapes.LINE, {
            x: 0.5, y: phy + 0.32, w: 0, h: 0.39,
            line: { color: C.border, width: 1.5 },
        });
    }

    // 文字
    s9.addText(ph.p, {
        x: 0.72, y: phy, w: 0.8, h: 0.26,
        fontSize: 11.5, fontFace: FONT, bold: true, color: ph.sc,
    });
    s9.addText(ph.n, {
        x: 1.55, y: phy, w: 1.4, h: 0.26,
        fontSize: 12, fontFace: FONT, bold: true, color: C.text,
    });
    // 状态标签
    s9.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 8.55, y: phy + 0.01, w: 0.78, h: 0.24,
        fill: { color: ph.sc }, rectRadius: 0.06,
    });
    s9.addText(ph.st, {
        x: 8.55, y: phy + 0.01, w: 0.78, h: 0.24,
        fontSize: 8.5, fontFace: FONT, bold: true,
        color: C.white, align: "center", valign: "middle",
    });
    s9.addText(ph.d, {
        x: 0.72, y: phy + 0.28, w: 7.6, h: 0.32,
        fontSize: 10, fontFace: FONT, color: C.muted,
    });
}

// 关键问题解决
infoBlock(s9, 0.35, 4.82, 9.05, 0.68, "关键问题解决", [
    { text: "① 502 Bad Gateway → 用 requests 替代 OpenAI SDK (httpx 兼容问题)     ② 404 Not Found → 补充 model_config.yaml 的 model 字段", options: { breakLine: false } },
    { text: "③ PowerShell curl 转义困难 → 改用 Python requests 调试     ④ 推理慢(~30s) → 部署 vLLM 实现 ~2.3x 加速", options: {} },
], { titleColor: C.error });


// ============================================================
// Slide 10: 下一步计划 + 总结 + 结束 (合并)
// ============================================================
var s10 = contentSlide("下一步计划与总结");

// 下一步
infoBlock(s10, 0.35, 1.08, 4.55, 2.0, "下一步计划", [
    { text: "[近期 1-2周] 完成 Phase 3 训练 → Phase 4/5 推理与评估", options: { bullet: true, breakLine: false } },
    { text: "[中期 2-4周] 扩展至 50-100 篇论文, 增加 G-E-R 至 2-3 轮", options: { bullet: true, breakLine: false } },
    { text: "[长期 1-2月] 消融实验验证 Filter/Integrator 贡献, 撰写复现报告", options: { bullet: true } },
], { titleColor: C.primary });

// 总结
infoBlock(s10, 5.1, 1.08, 4.45, 2.0, "项目总结", [
    { text: "✅ 搭建完成 FRAME 全链路 5 阶段流水线代码", options: { bullet: true, breakLine: false } },
    { text: "✅ 攻克 vLLM 部署难题: requests替代SDK, ~2.3x加速", options: { bullet: true, breakLine: false } },
    { text: "🔄 Phase 1-2 完成, Phase 3 训练进行中", options: { bullet: true, breakLine: false } },
    { text: "✅ 模块化设计, 配置驱动, 可扩展性强", options: { bullet: true } },
], { titleColor: C.success });

// 致谢区域
s10.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 3.25, w: 9.05, h: 2.2,
    fill: { color: "EFF6FF" },
    line: { color: C.accent, width: 0.7 },
});

s10.addText("感谢聆听 · Questions & Discussion", {
    x: 0.35, y: 3.45, w: 9.05, h: 0.5,
    fontSize: 22, fontFace: FONT, bold: true,
    color: C.primary, align: "center",
});

s10.addText([
    { text: "项目仓库: frame_reproduction/\n", options: { fontSize: 13 } },
    { text: "技术栈: Python + PyTorch + FAISS + vLLM + Qwen2.5-7B\n", options: { fontSize: 12 } },
    { text: "平台: AutoDL RTX 3090  |  已产出: dataset_full.json | training_results.json | comparison_results.json", options: { fontSize: 11 } },
], {
    x: 0.5, y: 4.05, w: 8.8, h: 1.2,
    fontFace: FONT, color: C.text, align: "center",
});


// ============================================================
// 保存文件
// ============================================================
var outputPath = path.join(__dirname, "FRAME_Project_Report.pptx");
console.log("正在生成 PPT (简洁版): " + outputPath);

pres.writeFile({ fileName: outputPath })
    .then(function() {
        console.log("\nPPT 生成成功!");
        console.log("文件路径: " + outputPath);
        console.log("共 " + pres.slides.length + " 页幻灯片");
    })
    .catch(function(err) {
        console.error("生成失败:", err.message);
        process.exit(1);
    });
