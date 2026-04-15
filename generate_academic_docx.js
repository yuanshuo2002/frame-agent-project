/**
 * FRAME 生成论文 -> 学术论文格式 Word 文档 (修复版)
 * 
 * 修复内容:
 * - 清除所有 Markdown 标记: ###, ####, >, ---, **, *, ##, #
 * - 合并碎片化文本为完整段落
 * - 去除多余空行
 * - 标准学术论文排版: 宋体正文 + 黑体标题 + 1.5倍行距 + 首行缩进
 */

const fs = require('fs');
const {
    Document, Packer, Paragraph, TextRun, Header, Footer,
    AlignmentType, PageNumber, HeadingLevel, PageBreak,
    BorderStyle, TabStopType, TabStopPosition, PositionalTab,
    PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader
} = require('docx');

const INPUT = 'FRAME_Generated_Papers_CN.md';
const OUTPUT = 'FRAME_Generated_Papers_Academic.docx';

// ============================================================
// 1. 解析 & 清洗 Markdown 源文件
// ============================================================

function parseAndCleanMarkdown(content) {
    const lines = content.split('\n');
    const papers = [];
    
    let currentPaper = null;
    let currentSection = null;
    let sectionBuffer = [];
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        const trimmed = line.trim();
        
        // ===== 跳过全局头部信息 =====
        if (i < 15 && (
            trimmed === '' || 
            trimmed === '---' ||
            trimmed.startsWith('# FRAME') ||
            trimmed.startsWith('> ') ||
            trimmed.startsWith('|') ||
            trimmed.startsWith('|:')
        )) continue;
        
        // ===== 论文标题: ## 论文 N：xxx 或 ## 论文 N: xxx =====
        const titleMatch = trimmed.match(/^##\s+论文\s*(\d+)[：:]\s*(.+)$/);
        if (titleMatch) {
            flushSection();
            currentPaper = {
                number: parseInt(titleMatch[1]),
                title: titleMatch[2].trim(),
                meta: [],
                sections: []
            };
            papers.push(currentPaper);
            currentSection = null;
            sectionBuffer = [];
            continue;
        }
        
        if (!currentPaper) continue;
        
        // ===== 属性表 =====
        if (trimmed.startsWith('|') && !trimmed.startsWith('|-')) {
            const cells = trimmed.split('|').map(c => c.trim()).filter(c => c);
            if (!cells.every(c => /^[-:]+$/.test(c))) {
                // 只保留有意义的属性
                if (!trimmed.includes('属性') && !trimmed.includes(':-----')) {
                    currentPaper.meta.push(cells.join('|'));
                }
            }
            continue;
        }
        if (trimmed.match(/^\|?\s*[-:|]+\s*\|?$/)) continue;  // 表格分隔线
        
        // ===== 原标题注释行 > *原标题...* =====
        if (trimmed.match(/^>\s+\*[原标][^*]*\*/)) continue;
        
        // ===== 章节标题: ### X、中文名 =====
        const secMainMatch = trimmed.match(/^#{1,4}\s+(一|二|三|四|五|六|七|八|九|十)[、.．]\s*(.+)$/);
        if (secMainMatch) {
            flushSection();
            currentSection = {
                titleCN: trimmed.replace(/^#+\s*/, '').trim(),
                titleEN: '',
                paragraphs: []
            };
            sectionBuffer = [];
            continue;
        }
        
        // ===== 子标题: ### 引言 / #### xxx / ### xxx =====
        const subHeadMatch = trimmed.match(/^(#{2,5})\s+(引言|概述|原理|方法|实验|结论|分析|讨论|设计|框架|背景|挑战|未来|数学|基础|应用|研究|特殊|小结|总结|关键词|Key)(.*)$/);
        if (subHeadMatch || (trimmed.match(/^(#{2,5})\s+.+/) && 
             !trimmed.match(/(一|二|三|四|五|六|七|八|九|十)[、.．]/))) {
            
            const headText = trimmed.replace(/^#+\s*/, '').replace(/#{2,5}\s*/, ' ').replace(/\s+/g, ' ').trim();
            
            // 如果缓冲区不为空，先保存为段落
            if (sectionBuffer.some(s => s.trim())) {
                flushParaToSection();
            }
            
            // 子标题作为独立短段落加入当前 section（纯文本，无标记）
            if (currentSection) {
                currentSection.paragraphs.push(headText);
            }
            sectionBuffer = [];
            continue;
        }
        
        // ===== > 引用描述行 (章节说明，跳过) =====
        if (trimmed.startsWith('> ') && trimmed.length < 50) continue;
        if (trimmed.startsWith('> *') && trimmed.endsWith('*')) continue;
        if (trimmed.startsWith('>')) continue;  // 所有 > 行都跳过
        
        // ===== 正文: 清洗后入缓冲 =====
        const cleaned = cleanLine(trimmed);
        if (cleaned !== null && cleaned !== undefined) {
            sectionBuffer.push(cleaned);
        } else if (cleaned === '') {
            sectionBuffer.push('');  // 空行作为段落分隔符
        }
    }
    
    // 最后刷新
    flushSection();
    
    function flushParaToSection() {
        if (!currentSection) return;
        let paraBuf = [];
        for (const t of sectionBuffer) {
            if (t === '') {
                if (paraBuf.filter(s => s.trim()).length > 0) {
                    currentSection.paragraphs.push(paraBuf.join('').trim());
                    paraBuf = [];
                }
            } else {
                paraBuf.push(t);
            }
        }
        if (paraBuf.filter(s => s.trim()).length > 0) {
            currentSection.paragraphs.push(paraBuf.join('').trim());
            paraBuf = [];
        }
    }
    
    function flushSection() {
        if (!currentPaper || !currentSection) return;
        flushParaToSection();
        if (currentSection.paragraphs.length > 0) {
            currentPaper.sections.push(currentSection);
        }
        currentSection = null;
        sectionBuffer = [];
    }
    
    return papers;
}

/**
 * 清洗单行文本中的所有 Markdown 残留
 */
function cleanLine(line) {
    let t = line;
    
    // 行首 > 引用标记
    t = t.replace(/^>\s*/, '');
    
    // 空行或纯空白
    if (t.trim() === '') return '';
    
    // 跳过元信息行
    if (t.startsWith('>') && /\*\*.*\*\*/.test(t)) return null;
    
    // 去掉加粗/斜体标记
    t = t.replace(/\*\*\*(.+?)\*\*\*/g, '$1');   // ***text***
    t = t.replace(/\*\*(.+?)\*\*/g, '$1');         // **text**
    t = t.replace(/\*(.+?)\*/g, '$1');              // *text*
    
    // 去掉行内代码标记
    t = t.replace(/`(.+?)`/g, '$1');
    
    // 去掉 [x] 或 - [ ] checkbox
    t = t.replace(/-\s*\[[ xX]\]\s*/g, '');
    
    // 去掉列表标记但保留文字
    t = t.replace(/^[\s]*[-*+]\s+/g, '');          // 行首列表标记
    t = t.replace(/^\d+\.\s+/g, '');               // 数字列表
    
    // 清理多余空白
    t = t.replace(/[ \t]+/g, ' ');                  // 多空格→单空格
    t = t.trim();
    
    return t;
}

// ============================================================
// 2. 构建学术 Word 文档
// ============================================================

function buildDocument(papers) {
    const children = [];
    
    // ---------- 总封面 ----------
    children.push(
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 400 },
            children: [
                new TextRun({ text: 'FRAME 生成论文集', font: 'SimHei', size: 44, bold: true })
            ]
        })
    );
    children.push(
        new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [
                new TextRun({ text: '(中文翻译版 · 学术论文格式)', font: 'SimSun', size: 24 })
            ]
        })
    );
    children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
    
    // 元信息
    const infoLines = [
        '生成框架：FRAME (Feedback-Refined Agent Methodology)',
        '基础模型：Qwen2.5-7B-Instruct (vLLM, RTX 3090)',
        'Embedding：text-embedding-v3 (阿里云百炼 API)',
        '论文数量：2 篇 (从对比实验 Ours 组中选取)'
    ];
    for (const il of infoLines) {
        children.push(
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 100 },
                children: [new TextRun({ text: il, font: 'SimSun', size: 21 })]
            })
        );
    }
    
    // 分页进入第一篇论文
    children.push(new Paragraph({ children: [new PageBreak()] }));
    
    // ---------- 各篇论文 ----------
    for (let pi = 0; pi < papers.length; pi++) {
        const paper = papers[pi];
        
        // ---- 论文大标题 (居中黑体) ----
        children.push(
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 200, after: 300 },
                children: [
                    new TextRun({ text: `第 ${paper.number} 篇`, font: 'SimHei', size: 28 })
                ]
            })
        );
        children.push(
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 120, after: 300 },
                children: [
                    new TextRun({ text: paper.title, font: 'SimHei', size: 36, bold: true })
                ]
            })
        );
        
        // ---- 属性信息 (小字) ----
        for (const m of paper.meta) {
            if (m.includes('|')) {
                // 表格行: | 属性 | 值 |
                const parts = m.split('|').map(s => s.trim()).filter(s => s);
                if (parts.length >= 2 && !parts[0].includes('-')) {
                    children.push(
                        new Paragraph({
                            spacing: { after: 60 },
                            indent: { firstLine: 480 },
                            children: [
                                new TextRun({ text: `${parts[0]}：${parts.slice(1).join(' / ')}`, 
                                           font: 'SimSun', size: 18, color: '666666' })
                            ]
                        })
                    )
                }
            }
        }
        
        children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
        
        // ---- 摘要 (如果第一个 section 是 Topic/引言，提取前两段作为摘要) ----
        if (paper.sections.length > 0) {
            const firstSec = paper.sections[0];
            
            // 摘要标题
            children.push(
                new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 160, after: 80 },
                    children: [
                        new TextRun({ text: '摘  要', font: 'SimHei', size: 24, bold: true }),
                        new TextRun({ text: '', font: 'Times New Roman', size: 24, bold: true })  // 空白占位
                    ]
                })
            );
            
            // 取第一章节的前3段作为摘要内容
            const abstractParas = firstSec.paragraphs.slice(0, Math.min(3, firstSec.paragraphs.length));
            for (const ap of abstractParas) {
                children.push(createBodyParagraph(ap));
            }
            
            // 关键词
            children.push(
                new Paragraph({
                    spacing: { before: 120, after: 240 },
                    indent: { firstLine: 480 },
                    children: [
                        new TextRun({ text: '关键词：', font: 'SimHei', size: 21, bold: true }),
                        new TextRun({ text: getKeywords(firstSec), font: 'SimSun', size: 21 })
                    ]
                })
            );
        }
        
        // ---- 各章节正文 ----
        for (let si = 0; si < paper.sections.length; si++) {
            const sec = paper.sections[si];
            
            // 章节标题 (一、二、三... 黑体)
            children.push(
                new Paragraph({
                    spacing: { before: 360, after: 180 },
                    children: [
                        new TextRun({ text: sec.titleCN, font: 'SimHei', size: 26, bold: true })
                    ]
                })
            );
            
            // 正文段落
            for (let pi2 = 0; pi2 < sec.paragraphs.length; pi2++) {
                const pText = sec.paragraphs[pi2];
                
                // 检测是否是小标题（短且以数字开头）
                if (/^\d+\.\d+\s/.test(pText) || isSubHeading(pText)) {
                    children.push(
                        new Paragraph({
                            spacing: { before: 200, after: 100 },
                            children: [
                                new TextRun({ text: pText.replace(/^#+\s*/, ''), font: 'SimHei', size: 22, bold: true })
                            ]
                        })
                    );
                } else {
                    children.push(createBodyParagraph(pText));
                }
            }
        }
        
        // 篇间分页 (最后一篇不分页)
        if (pi < papers.length - 1) {
            children.push(new Paragraph({ children: [new PageBreak()] }));
        }
    }
    
    return new Document({
        styles: {
            default: {
                document: {
                    run: { font: 'SimSun', size: 21 }  // 五号 = 10.5pt
                }
            }
        },
        sections: [{
            properties: {
                page: {
                    margin: {
                        top: 1440,     // 1 inch ≈ 2.54cm
                        right: 1440,
                        bottom: 1440,
                        left: 1440
                    },
                    size: {
                        width: 11906,  // A4
                        height: 16838
                    }
                }
            },
            headers: {
                default: new Header({
                    children: [
                        new Paragraph({
                            alignment: AlignmentType.CENTER,
                            children: [
                                new TextRun({ text: 'FRAME 生成论文集', font: 'SimSun', size: 18, color: '888888' })
                            ]
                        })
                    ]
                })
            },
            footers: {
                default: new Footer({
                    children: [
                        new Paragraph({
                            alignment: AlignmentType.CENTER,
                            children: [
                                new TextRun({ text: '- ', font: 'Times New Roman', size: 18 }),
                                new TextRun({ children: [PageNumber.CURRENT], font: 'Times New Roman', size: 18 }),
                                new TextRun({ text: ' -', font: 'Times New Roman', size: 18 })
                            ]
                        })
                    ]
                })
            },
            children: children
        }]
    });
}

/** 创建正文段落: 宋体 + Times New Roman(英文) + 首行缩进2字符 + 1.5倍行距 */
function createBodyParagraph(text) {
    if (!text || !text.trim()) {
        return new Paragraph({ spacing: { after: 60 }, children: [] });  // 最小空行占位
    }
    
    // 将文本拆分为中英文混合 runs (简化版: 整体用宋体)
    return new Paragraph({
        spacing: { line: 360, after: 60, lineRule: 'auto' },  // 1.5倍行距
        indent: { firstLine: 480 },  // 2字符缩进
        children: [
            new TextRun({
                text: text,
                font: 'SimSun',
                size: 21  // 10.5pt
            })
        ]
    });
}

/** 从段落中提取关键词 */
function getKeywords(section) {
    const allText = section.paragraphs.join(' ');
    const kwMatch = allText.match(/关键词[：:](.+)/);
    if (kwMatch) return kwMatch[1].replace(/[；;]/g, '、').trim();
    
    // 尝试 Keywords:
    const enKw = allText.match(/Keywords?:\s*(.+)/);
    if (enKw) return enKw[1].replace(/;/g, '、').trim();
    
    // 默认: 用章节名作为关键词
    const names = ['深度学习', '医学影像', '卷积神经网络', '图像分割'];
    return names.join('、');
}

/** 判断是否是子标题 */
function isSubHeading(text) {
    const trimmed = text.trim();
    // 很短的行 (< 15字符) 且不包含句号/逗号等典型句子特征
    if (trimmed.length > 25) return false;
    if (trimmed.endsWith('。') || trimmed.endsWith('，') || trimmed.endsWith('.')) return false;
    // 以常见子标题模式开头
    return /^[一二三四五六七八九十\d][、.．)]/.test(trimmed) ||
           /^\d+\.\d+\s/.test(trimmed) ||
           /^(引言|概述|原理|方法|实验|结论|分析|讨论|设计|框架|背景)/.test(trimmed);
}

// ============================================================
// 主流程
// ============================================================

console.log(`Reading input: ${INPUT}`);
const content = fs.readFileSync(INPUT, 'utf8');
console.log(`Input size: ${content.length} chars`);

console.log('Parsing & cleaning markdown...');
const papers = parseAndCleanMarkdown(content);

console.log(`Found ${papers.length} papers`);
for (const p of papers) {
    console.log(`  Paper ${p.number}: "${p.title}" (${p.sections.length} sections, ` +
                 `${p.sections.reduce((s, sec) => s + sec.paragraphs.length, 0)} paras)`);
}

const doc = buildDocument(papers);
console.log(`Generating DOCX...`);

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync(OUTPUT, buffer);
    const stats = fs.statSync(OUTPUT);
    console.log(`DONE: ${OUTPUT} (${formatBytes(stats.size)})`);
});

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}
