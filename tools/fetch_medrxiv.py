#!/usr/bin/env python3
"""
medRxiv 批量论文下载工具
支持：
  1. 按关键词搜索元数据（标题/摘要）
  2. 下载 PDF 并用 pdfplumber 提取正文
  3. 输出标准 JSON 格式供 run_dataset_build.py 使用

用法:
    python tools/fetch_medrxiv.py --query "deep learning medical imaging" --n 20
    python tools/fetch_medrxiv.py --query "nlp clinical notes" --n 15 --output data/raw/nlp_papers.json

依赖:
    pip install requests pdfplumber tqdm
"""
import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

try:
    import requests
except ImportError:
    print("请先安装: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# medRxiv/bioRxiv API endpoint (官方 API，无需 key)
BIORXIV_API = "https://api.biorxiv.org/details/medrxiv"
SEARCH_API  = "https://api.biorxiv.org/search/"   # 搜索端点

# 备选：Europe PMC（支持更丰富的关键词搜索，有完整开放获取全文）
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_PDF    = "https://europepmc.org/backend/ptpmcrender.fcgi"


def search_medrxiv_by_date(
    start: str = "2023-01-01",
    end: str   = "2024-12-31",
    server: str = "medrxiv",
    cursor: int = 0,
    page_size: int = 100,
) -> List[Dict]:
    """
    按时间段获取 medRxiv 论文列表（官方 API）
    https://api.biorxiv.org/details/medrxiv/YYYY-MM-DD/YYYY-MM-DD/cursor
    """
    url = f"{BIORXIV_API}/{start}/{end}/{cursor}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("collection", [])
    except Exception as e:
        print(f"[ERROR] medRxiv API 请求失败: {e}")
        return []


def search_europepmc(
    query: str,
    n: int = 50,
    has_pdf: bool = True,
) -> List[Dict]:
    """
    通过 Europe PMC 搜索医学论文（支持关键词，有开放全文）
    """
    results = []
    page = 1
    page_size = min(n, 25)

    while len(results) < n:
        params = {
            "query": query + (" AND OPEN_ACCESS:Y" if has_pdf else ""),
            "format": "json",
            "pageSize": page_size,
            "page": page,
            "resultType": "core",
            "sort": "CITED desc",  # 按引用排序，质量更高
        }
        try:
            resp = requests.get(EUROPEPMC_SEARCH, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("resultList", {}).get("result", [])
            if not items:
                break
            results.extend(items)
            page += 1
            time.sleep(0.5)  # 礼貌性限速
        except Exception as e:
            print(f"[ERROR] Europe PMC 搜索失败 (page {page}): {e}")
            break

    return results[:n]


def download_pdf(url: str, save_path: Path, timeout: int = 60) -> bool:
    """下载 PDF 文件"""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[WARN] PDF 下载失败 {url}: {e}")
        return False


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 30) -> str:
    """用 pdfplumber 从 PDF 提取正文"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()
    except ImportError:
        print("[WARN] pdfplumber 未安装，跳过 PDF 提取。运行: pip install pdfplumber")
        return ""
    except Exception as e:
        print(f"[WARN] PDF 解析失败 {pdf_path}: {e}")
        return ""


def europepmc_to_paper(item: Dict) -> Optional[Dict]:
    """
    将 Europe PMC 结果转换为 FRAME 标准格式
    {"id", "title", "text", "paper_title", "abstract", "authors", "year", "source"}
    """
    pmid = item.get("pmid") or item.get("id", "")
    pmcid = item.get("pmcid", "")
    title = item.get("title", "").strip()
    abstract = item.get("abstractText", "") or item.get("abstract", "")

    if not title or not abstract:
        return None

    # 构造正文 text（标题 + 摘要，后续可扩展为全文）
    text = f"Title: {title}\n\nAbstract:\n{abstract}"

    # 如果有全文 URL 则附上（供下载 PDF 用）
    fulltext_url = None
    if pmcid:
        fulltext_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

    paper_id = pmid or pmcid or hashlib.md5(title.encode()).hexdigest()[:8]

    return {
        "id": f"real_{paper_id}",
        "paper_title": title,
        "title": title,
        "abstract": abstract,
        "text": text,
        "authors": item.get("authorString", ""),
        "year": item.get("pubYear", ""),
        "journal": item.get("journalTitle", ""),
        "pmid": pmid,
        "pmcid": pmcid,
        "fulltext_url": fulltext_url,
        "source": "europepmc",
    }


def fetch_fulltext_from_pmc(pmcid: str) -> str:
    """
    从 PubMed Central 获取全文（XML → 纯文本）
    PMC 提供免费全文 API
    """
    if not pmcid:
        return ""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid.replace("PMC", ""),
        "rettype": "xml",
        "retmode": "xml",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        # 简单从 XML 提取文本
        import re
        xml = resp.text
        # 去掉 XML 标签，保留纯文本
        text = re.sub(r'<[^>]+>', ' ', xml)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:20000]  # 限制长度
    except Exception as e:
        print(f"[WARN] PMC 全文获取失败 {pmcid}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="medRxiv/PubMed 批量论文下载工具")
    parser.add_argument("--query", type=str, required=True,
                        help="搜索关键词，如 'deep learning medical imaging'")
    parser.add_argument("--n", type=int, default=30,
                        help="目标下载数量（默认30）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径（默认自动生成）")
    parser.add_argument("--fetch_fulltext", action="store_true",
                        help="尝试从 PMC 获取全文（更慢但内容更丰富）")
    parser.add_argument("--pdf_dir", type=str, default=None,
                        help="PDF 下载目录（可选，不指定则只保存摘要）")
    args = parser.parse_args()

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        safe_query = args.query.replace(" ", "_")[:30]
        output_path = Path("data/raw") / f"papers_{safe_query}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n搜索关键词: {args.query}")
    print(f"目标数量: {args.n}")
    print(f"输出路径: {output_path}")
    print("数据来源: Europe PMC (开放获取)\n")

    # Step 1: 搜索元数据
    print(f"[1/3] 正在搜索...")
    raw_results = search_europepmc(args.query, n=args.n * 2)  # 多拉一些，过滤后够用
    print(f"      找到 {len(raw_results)} 条结果")

    # Step 2: 转换格式
    papers = []
    for item in raw_results:
        paper = europepmc_to_paper(item)
        if paper:
            papers.append(paper)
        if len(papers) >= args.n:
            break
    print(f"[2/3] 有效论文: {len(papers)} 篇")

    # Step 3 (可选): 获取全文
    if args.fetch_fulltext:
        print(f"[3/3] 获取全文 (PMC API)...")
        for i, paper in enumerate(tqdm(papers, desc="获取全文")):
            pmcid = paper.get("pmcid", "")
            if pmcid:
                fulltext = fetch_fulltext_from_pmc(pmcid)
                if fulltext and len(fulltext) > 500:
                    paper["text"] = f"Title: {paper['title']}\n\n{fulltext}"
                    paper["has_fulltext"] = True
            time.sleep(0.3)  # 避免触发 NCBI 限速（3 req/s）
    else:
        print(f"[3/3] 跳过全文获取（只保存摘要）")
        print(f"      提示: 加 --fetch_fulltext 可获取 PMC 全文（更丰富）")

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！保存了 {len(papers)} 篇论文到: {output_path}")
    print(f"   平均摘要长度: {sum(len(p['text']) for p in papers) // len(papers) if papers else 0} 字符")

    # 打印几条预览
    print(f"\n前 3 篇预览:")
    for p in papers[:3]:
        print(f"  [{p['id']}] {p['title'][:70]}...")
        print(f"       text长度={len(p['text'])} | PMC={p.get('pmcid','无')}")


if __name__ == "__main__":
    main()
