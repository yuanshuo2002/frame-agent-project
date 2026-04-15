"""检查 docx XML 中的格式问题"""
import zipfile, re, os, io, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.makedirs('unpacked_docx', exist_ok=True)
with zipfile.ZipFile('FRAME_Generated_Papers_Academic.docx', 'r') as z:
    z.extractall('unpacked_docx')

xml = open('unpacked_docx/word/document.xml', 'r', encoding='utf-8').read()
texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', xml)

print(f"总文本节点: {len(texts)}")

issues = []

# 空段落
empty_count = sum(1 for t in texts if not t.strip())
if empty_count > 0:
    issues.append(f"[空段落] {empty_count} 个完全空的文本节点")

# ### 残留
hash_texts = [t for t in texts if '###' in t]
if hash_texts:
    issues.append(f"[###符号残留] {len(hash_texts)} 个:")
    for ht in hash_texts[:10]:
        issues.append(f"   -> {ht[:80]}")

# Markdown 残留 (# > - * ---)
for marker, name in [('# ', '标题标记'), ('> ', '引用标记'), ('- ', '列表标记'),
                     ('* ', '星号列表'), ('---', '分隔线')]:
    found = [t for t in texts if t.strip().startswith(marker) or (marker == '---' and marker in t)]
    if found:
        issues.append(f"[{name}残留] {len(found)} 个:")
        for f in found[:10]:
            issues.append(f"   -> {f[:80]}")

# 连续空行检查
consecutive_empty = 0
max_consecutive = 0
for t in texts:
    if not t.strip():
        consecutive_empty += 1
        max_consecutive = max(max_consecutive, consecutive_empty)
    else:
        consecutive_empty = 0
if max_consecutive >= 3:
    issues.append(f"[连续空行] 最大连续: {max_consecutive}")

# 内嵌换行符
newlines = [(i,t) for i,t in enumerate(texts) if '\n' in t or '\r' in t]
if newlines:
    issues.append(f"[内嵌换行] {len(newlines)} 个段落含 \\n/\\r:")
    for idx, nt in newlines[:5]:
        issues.append(f"   [{idx}] len={len(nt)} \"{nt[:60]}\"")

# 超短碎片
skip = {'A','B','C','D','E','I','II','III','IV','V','①','②','③','%','$','#','+','-','*'}
shorts = [(i,t) for i,t in enumerate(texts) if 0 < len(t.strip()) < 5 and t.strip() not in skip]
if shorts:
    issues.append(f"[超短片段] {len(shorts)} 个 (<5字符):")
    for idx, st in shorts[:15]:
        issues.append(f"   [{idx}] \"{st}\"")

# 打印前50个非空段落的预览
print(f"\n{'='*60}")
print(f"前 50 个非空文本节点预览:")
print(f"{'='*60}")
count = 0
for i, t in enumerate(texts):
    if t.strip():
        preview = t.replace('\n', '\\n').replace('\r', '\\r')[:90]
        print(f"  [{i:3d}] ({len(t):4d}ch) {preview}")
        count += 1
        if count >= 50:
            break

total_chars = sum(len(t) for t in texts)
nonempty = [t for t in texts if t.strip()]
print(f"\n统计: 总{len(texts)} | 非空{len(nonempty)} | 字符{total_chars}")

print(f"\n{'='*60}")
if issues:
    print("发现问题:")
    print(f"{'='*60}")
    for issue in issues:
        print(issue)
else:
    print("未发现明显格式问题")
