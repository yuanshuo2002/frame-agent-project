#!/usr/bin/env python3
"""
训练集 / 测试集划分工具

策略：分层划分（按来源主题），保证测试集覆盖所有主题
  - 每个主题文件: 7篇→训练, 3篇→测试
  - 30篇总量 → train=21篇, test=9篇

用法:
    python tools/split_dataset.py
    python tools/split_dataset.py --train_ratio 0.7 --seed 42
"""
import json
import argparse
import random
from pathlib import Path


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  已保存 {len(data)} 篇 → {path}")


def main():
    parser = argparse.ArgumentParser(description="训练集/测试集分层划分")
    parser.add_argument("--raw_dir",     default="data/raw",   help="原始论文 JSON 目录")
    parser.add_argument("--train_out",   default="data/raw/train_papers.json")
    parser.add_argument("--test_out",    default="data/raw/test_papers.json")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例（默认0.7）")
    parser.add_argument("--seed",        type=int,   default=42,  help="随机种子（保证可复现）")
    parser.add_argument("--exclude",     nargs="*",  default=["train_papers.json", "test_papers.json", "demo_papers.json"],
                        help="排除的文件名")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)

    # 找到所有 papers_*.json 文件（排除已生成的 train/test 和 demo）
    all_files = sorted(raw_dir.glob("papers_*.json"))
    source_files = [f for f in all_files if f.name not in args.exclude]

    if not source_files:
        print(f"[ERROR] 在 {raw_dir} 下没有找到 papers_*.json 文件")
        return

    print(f"\n发现 {len(source_files)} 个来源文件（分层划分）:")

    train_all, test_all = [], []

    for fpath in source_files:
        papers = load_json(fpath)
        # 打乱顺序（固定seed保证可复现）
        random.shuffle(papers)

        n_train = max(1, round(len(papers) * args.train_ratio))
        n_test  = len(papers) - n_train

        train_part = papers[:n_train]
        test_part  = papers[n_train:]

        train_all.extend(train_part)
        test_all.extend(test_part)

        print(f"  {fpath.name}: {len(papers)}篇 → 训练{n_train} + 测试{n_test}")

    # 再整体打乱一次（避免训练时主题集中）
    random.shuffle(train_all)
    random.shuffle(test_all)

    print(f"\n划分结果:")
    print(f"  训练集: {len(train_all)} 篇")
    print(f"  测试集: {len(test_all)} 篇")

    save_json(train_all, args.train_out)
    save_json(test_all,  args.test_out)

    # 打印摘要
    print(f"\n训练集前3篇:")
    for p in train_all[:3]:
        print(f"  [{p.get('source','?')}] {p.get('title','')[:60]}...")
    print(f"\n测试集前3篇:")
    for p in test_all[:3]:
        print(f"  [{p.get('source','?')}] {p.get('title','')[:60]}...")

    print(f"\n[完成]")
    print(f"   Phase 2 使用: {args.train_out}")
    print(f"   Phase 3/4 使用: {args.test_out}")


if __name__ == "__main__":
    main()
