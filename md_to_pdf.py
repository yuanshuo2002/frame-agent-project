#!/usr/bin/env python3
"""将 README.md 转换为排版的 PDF 报告"""

import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import markdown
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
pt = 1  # reportlab point = 1 unit
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================================================
# 字体注册（中文支持）
# ============================================================
FONT_DIR = r"C:\Windows\Fonts"

# 注册中文字体
try:
    pdfmetrics.registerFont(TTFont("SimSun", f"{FONT_DIR}\\simsun.ttc"))
    pdfmetrics.registerFont(TTFont("SimHei", f"{FONT_DIR}\\simhei.ttf"))
    pdfmetrics.registerFont(TTFont("Times", f"{FONT_DIR}\\times.ttf"))
    CN_FONT = "SimSun"
    H_FONT = "SimHei"
    EN_FONT = "Times"
except Exception as e:
    print(f"[WARN] Font register failed: {e}")
    CN_FONT = "Helvetica"
    H_FONT = "Helvetica-Bold"
    EN_FONT = "Times-Roman"


# ============================================================
# 样式定义
# ============================================================
styles = getSampleStyleSheet()

# 封面大标题
styles.add(ParagraphStyle(
    name="CoverTitle",
    fontName=H_FONT,
    fontSize=26,
    leading=36,
    alignment=TA_CENTER,
    spaceAfter=12 * mm,
    textColor=HexColor("#1a365d"),
))

# 封面副标题
styles.add(ParagraphStyle(
    name="CoverSub",
    fontName=CN_FONT,
    fontSize=14,
    leading=22,
    alignment=TA_CENTER,
    spaceAfter=6 * mm,
    textColor=HexColor("#2d3748"),
))

# 一级标题 (##)
styles.add(ParagraphStyle(
    name="H1",
    fontName=H_FONT,
    fontSize=18,
    leading=28,
    spaceBefore=16 * pt,
    spaceAfter=10 * pt,
    textColor=HexColor("#1a365d"),
    borderPadding=(0, 0, 3, 0),
))

# 二级标题 (###)
styles.add(ParagraphStyle(
    name="H2",
    fontName=H_FONT,
    fontSize=14,
    leading=22,
    spaceBefore=12 * pt,
    spaceAfter=6 * pt,
    textColor=HexColor("#2c5282"),
))

# 三级标题 (####)
styles.add(ParagraphStyle(
    name="H3",
    fontName=H_FONT,
    fontSize=11,
    leading=18,
    spaceBefore=8 * pt,
    spaceAfter=4 * pt,
    textColor=HexColor("#2b6cb0"),
))

# 正文
styles.add(ParagraphStyle(
    name="Body",
    fontName=CN_FONT,
    fontSize=10,
    leading=18,
    alignment=TA_JUSTIFY,
    spaceBefore=2 * pt,
    spaceAfter=4 * pt,
    firstLineIndent=21,  # 首行缩进2字符
))

# 正文无缩进（列表等）
styles.add(ParagraphStyle(
    name="BodyNoIndent",
    fontName=CN_FONT,
    fontSize=10,
    leading=17,
    alignment=TA_JUSTIFY,
    spaceBefore=1 * pt,
    spaceAfter=3 * pt,
))

# 表格内容
styles.add(ParagraphStyle(
    name="TableCell",
    fontName=CN_FONT,
    fontSize=9,
    leading=14,
    alignment=TA_CENTER,
))

# 表格内容左对齐
styles.add(ParagraphStyle(
    name="TableCellLeft",
    fontName=CN_FONT,
    fontSize=9,
    leading=14,
    alignment=TA_LEFT,
))

# 代码块
styles.add(ParagraphStyle(
    name="CodeBlock",
    fontName="Courier",
    fontSize=7.5,
    leading=11,
    leftIndent=10,
    rightIndent=10,
    spaceBefore=4 * pt,
    spaceAfter=4 * pt,
    backColor=HexColor("#f7fafc"),
    borderColor=HexColor("#e2e8f0"),
    borderWidth=0.5,
    borderPadding=4,
))


# ============================================================
# Markdown 解析器
# ============================================================

def parse_markdown(md_text: str) -> list:
    """将 Markdown 文本解析为 reportlab Flowable 列表"""
    
    elements = []
    lines = md_text.split("\n")
    i = 0
    
    # 先检测是否是表格行
    def is_table_separator(line):
        return bool(re.match(r'^[\s|:-]+$', line))
    
    def is_table_row(line):
        return line.startswith("|") and "|" in line[1:]
    
    def parse_table_row(line):
        """解析表格行为单元格"""
        cells = [c.strip() for c in line.split("|")[1:-1]]
        return cells
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        # 跳过空行
        if not line:
            i += 1
            continue
        
        # ===== 封面检测 =====
        if line.startswith("# ") and not any("项目复现报告" in el_text for el_text in [str(e) for e in elements]):
            title_text = line[2:].strip()
            # 创建封面
            elements.append(Spacer(1, 40 * mm))
            
            # 英文副标题处理
            en_sub = ""
            if "(" in title_text and ")" in title_text:
                en_sub = title_text[title_text.find("("):title_text.rfind(")")+1]
                cn_title = title_text[:title_text.find("(")].strip()
                elements.append(Paragraph(cn_title, styles["CoverTitle"]))
                elements.append(Paragraph(en_sub, styles["CoverSub"]))
            else:
                elements.append(Paragraph(title_text, styles["CoverTitle"]))
            
            elements.append(Spacer(1, 15 * mm))
            elements.append(HRFlowable(width="60%", thickness=1, color=HexColor("#cbd5e0"), spaceAfter=15*mm))
            
            i += 1
            continue
        
        # ===== 一级标题 =====
        if line.startswith("## ") and not line.startswith("### "):
            text = line[3:].strip()
            # 清理特殊标记
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            elements.append(Spacer(1, 6 * pt))
            elements.append(Paragraph(text, styles["H1"]))
            i += 1
            continue
        
        # ===== 二级标题 =====
        if line.startswith("### ") and not line.startswith("#### "):
            text = line[4:].strip()
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            elements.append(Paragraph(text, styles["H2"]))
            i += 1
            continue
        
        # ===== 三级标题 =====
        if line.startswith("#### "):
            text = line[5:].strip()
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            elements.append(Paragraph(text, styles["H3"]))
            i += 1
            continue
        
        # ===== 分隔线 =====
        if set(line.replace(" ", "")) <= {"-", "_", "=", "*"} and len(line) >= 3:
            elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cbd5e0")))
            i += 1
            continue
        
        # ===== 表格检测 =====
        if is_table_row(line) and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            # 收集所有表格行
            table_rows = []
            header_cells = parse_table_row(line)  # header
            table_rows.append(header_cells)
            i += 2  # skip separator
            
            while i < len(lines) and is_table_row(lines[i]):
                table_rows.append(parse_table_row(lines[i]))
                i += 1
            
            # 构建表格
            if len(table_rows) > 0:
                # 确定列数
                ncols = max(len(row) for row in table_rows)
                
                # 标准化每行长度
                for row in table_rows:
                    while len(row) < ncols:
                        row.append("")
                
                # 构建单元格数据
                cell_data = []
                for ri, row in enumerate(table_rows):
                    cells = []
                    for ci, cell in enumerate(row):
                        # 清理 markdown 格式
                        clean = cell.strip()
                        clean = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', clean)
                        clean = clean.replace("🏆", "[BEST]")
                        clean = clean.replace("🥇", "")
                        clean = clean.replace("🥈", "")
                        clean = clean.replace("🥉", "")
                        clean = clean.replace("📈", "")
                        clean = clean.replace("📉", "")
                        clean = clean.replace("✅", "[OK]")
                        clean = clean.replace("⚠️", "[!]")
                        clean = clean.replace("🔴", "")
                        clean = clean.replace("🟢", "")
                        clean = clean.replace("🔍", "")
                        clean = clean.replace("📌", "")
                        style_name = "TableCell" if ci > 0 else "TableCellLeft"
                        cells.append(Paragraph(clean, styles[style_name]))
                    cell_data.append(cells)
                
                # 计算列宽比例
                col_widths = []
                total_width = 170 * mm
                for ci in range(ncols):
                    max_len = max(len(table_rows[ri][ci]) for ri in range(len(table_rows))) if table_rows else 1
                    if ci == 0:
                        col_widths.append(total_width * 0.32)  # 第一列较宽
                    else:
                        col_widths.append((total_width - total_width*0.32) / max(ncols - 1, 1))
                
                tbl = Table(cell_data, colWidths=col_widths, repeatRows=1)
                
                tbl_style = [
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, 0), H_FONT),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#edf2f7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#1a365d")),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ("FONTNAME", (0, 1), (-1, -1), CN_FONT),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                    ("TOPPADDING", (0, 1), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f7fafc")]),
                ]
                tbl.setStyle(TableStyle(tbl_style))
                elements.append(Spacer(1, 6 * pt))
                elements.append(tbl)
                elements.append(Spacer(1, 6 * pt))
            continue
        
        # ===== 引用块 (> ) =====
        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:].strip())
                i += 1
            quote_text = "<br/>".join(quote_lines)
            # 清理格式
            quote_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', quote_text)
            pstyle = ParagraphStyle(
                "Quote",
                parent=styles["BodyNoIndent"],
                leftIndent=15,
                rightIndent=10,
                textColor=HexColor("#4a5568"),
                backColor=HexColor("#f7fafc"),
                borderPadding=(6, 6, 6, 6),
            )
            elements.append(Paragraph(quote_text, pstyle))
            continue
        
        # ===== 无序列表 =====
        if re.match(r'^[\s]*[-*] ', line):
            items = []
            while i < len(lines) and re.match(r'^[\s]*[-*] ', lines[i]):
                item_text = re.sub(r'^[\s]*[-*]\s+', "", lines[i]).strip()
                item_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', item_text)
                item_text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="8">\1</font>', item_text)
                items.append(item_text)
                i += 1
            for item in items:
                elements.append(Paragraph(f"• {item}", styles["BodyNoIndent"]))
            continue
        
        # ===== 有序列表 =====
        if re.match(r'^[\s]*\d+[.\)]\s', line):
            items = []
            while i < len(lines) and re.match(r'^[\s]*\d+[.\)]\s', lines[i]):
                item_text = re.sub(r'^[\s]*\d+[.\)]\s+', "", lines[i]).strip()
                item_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', item_text)
                items.append(item_text)
                i += 1
            for idx, item in enumerate(items, 1):
                elements.append(Paragraph(f"{idx}. {item}", styles["BodyNoIndent"]))
            continue
        
        # ===== 代码块 (``` 或 缩进) =====
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = "\n".join(code_lines)
            # 转义 XML 特殊字符
            code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            elements.append(Paragraph(code_text, styles["CodeBlock"]))
            continue
        
        # ===== 普通段落 =====
        para_lines = []
        while i < len(lines) and lines[i].strip() and \
              not lines[i].startswith("#") and \
              not lines[i].startswith(">") and \
              not is_table_row(lines[i]) and \
              not is_table_separator(lines[i]) and \
              not re.match(r'^[\s]*[-*] ', lines[i]) and \
              not re.match(r'^[\s]*\d+[.\)]\s', lines[i]) and \
              not lines[i].startswith("```") and \
              not set(lines[i].replace(" ", "")) <= {"-", "_", "=", "*"}:
            para_lines.append(lines[i].strip())
            i += 1
        
        if para_lines:
            para_text = " ".join(para_lines)
            # Markdown → HTML 格式转换
            para_text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', para_text)
            para_text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', para_text)
            para_text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="8">\1</font>', para_text)
            para_text = para_text.replace("→", "-&gt;")
            para_text = para_text.replace("←", "&lt;-")
            para_text = para_text.replace("↑", "^")
            para_text = para_text.replace("↓", "v")
            para_text = para_text.replace("✅", "[OK]")
            para_text = para_text.replace("⚠️", "[!]")
            para_text = para_text.replace("🏆", "")
            para_text = para_text.replace("🥇", "")
            para_text = para_text.replace("🥈", "")
            para_text = para_text.replace("🥉", "")
            para_text = para_text.replace("📈", "")
            para_text = para_text.replace("📉", "")
            para_text = para_text.replace("🔴", "")
            para_text = para_text.replace("🟢", "")
            para_text = para_text.replace("🔍", "")
            para_text = para_text.replace("📌", "")
            para_text = para_text.replace("█", "#")
            elements.append(Paragraph(para_text, styles["Body"]))
        else:
            i += 1
    
    return elements


# ============================================================
# 页码函数
# ============================================================
def add_page_number(canvas, doc):
    canvas.saveState()
    page_num = canvas.getPageNumber()
    if page_num > 1:  # 封面不显示页码
        canvas.setFont(CN_FONT, 8)
        canvas.setFillColor(HexColor("#718096"))
        canvas.drawCentredString(A4[0]/2, 15*mm, f"- {page_num - 1} -")
    canvas.restoreState()


# ============================================================
# 主程序
# ============================================================
def main():
    input_file = "README.md"
    output_file = "FRAME_Project_Report.pdf"
    
    with open(input_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    print(f"Reading: {input_file} ({len(md_content)} chars)")
    
    # 解析 Markdown 为元素
    elements = parse_markdown(md_content)
    print(f"Parsed: {len(elements)} flowable elements")
    
    # 构建 PDF
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        leftMargin=2.54*cm,
        rightMargin=2.54*cm,
        topMargin=2.54*cm,
        bottomMargin=2*cm,
    )
    
    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    
    print(f"DONE: {output_file}")


if __name__ == "__main__":
    main()
