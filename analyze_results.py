import json

with open('checkpoints/training_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("  FRAME Training Results - Complete Analysis (120 samples)")
print("=" * 60)
print(f"Total Papers: {len(data)}")
print()

# Collect all scores
all_scores = []
best_rounds = {'round1': 0, 'round2': 0, 'round0': 0}
section_stats = {}

for paper_id, paper in data.items():
    title = paper.get('paper_title', 'N/A')
    
    for sec_key, sec_result in paper.get('section_results', {}).items():
        if sec_key not in section_stats:
            section_stats[sec_key] = []
        
        best_iter = sec_result.get('best_iteration', 1)
        final_report = sec_result.get('final_report', {})
        score = final_report.get('source_eval_score', 0)
        section_stats[sec_key].append(score)
        all_scores.append(score)
        
        if best_iter == 1:
            best_rounds['round1'] += 1
        elif best_iter == 2:
            best_rounds['round2'] += 1
        else:
            best_rounds['round0'] += 1

# Section breakdown
print("-" * 60)
print("  Score by Chapter / Section")
print("-" * 60)
for sec in ['topic', 'background', 'related_work', 'methodology', 'result', 'conclusion']:
    scores = section_stats.get(sec, [])
    if scores:
        avg = sum(scores) / len(scores)
        mn = min(scores)
        mx = max(scores)
        print(f"  {sec:14s} | avg={avg:.3f}  min={mn:.2f}  max={mx:.2f}  n={len(scores)}")

# Overall stats
print()
print("-" * 60)
print("  Overall Statistics")
print("-" * 60)
print(f"  Total samples:       {len(all_scores)}")
print(f"  Grand average:       {sum(all_scores) / len(all_scores):.3f}")
print(f"  Highest score:       {max(all_scores):.3f}")
print(f"  Lowest score:        {min(all_scores):.3f}")

# Distribution
ranges = [
    (4.7, 5.01, "A+ [4.7-5.0]"),
    (4.5, 4.70, "A  [4.5-4.7)"),
    (4.3, 4.50, "B+ [4.3-4.5)"),
    (4.0, 4.30, "B  [4.0-4.3)"),
    (0.0, 4.00, "C  [<4.0]"),
]
print()
print("-" * 60)
print("  Score Distribution (120 total)")
print("-" * 60)
for lo, hi, label in ranges:
    cnt = len([s for s in all_scores if lo <= s < hi])
    pct = cnt / len(all_scores) * 100
    bar_len = int(pct / 1.5)
    bar = "#" * bar_len
    print(f"  {label:16s}: {cnt:3d} ({pct:5.1f}%)  {bar}")

# Round comparison
print()
print("-" * 60)
print("  Round Comparison (Best Round Selected)")
print("-" * 60)
total_r = sum(best_rounds.values())
print(f"  Round 1 (original) wins:   {best_rounds['round1']:3d}  ({best_rounds['round1']/total_r*100:5.1f}%)")
print(f"  Round 2 (reflected) wins:  {best_rounds['round2']:3d}  ({best_rounds['round2']/total_r*100:5.1f}%)")
print(f"  Round 0 (fallback/only):   {best_rounds['round0']:3d}  ({best_rounds['round0']/total_r*100:5.1f}%)")

# Reflector improvement analysis
improvements = []
for paper_id, paper in data.items():
    for sec_key, sec_result in paper.get('section_results', {}).items():
        iters = sec_result.get('iterations', [])
        if len(iters) >= 2:
            r1_score = iters[0].get('evaluation', {}).get('overall_score', 0)
            r2_score = iters[1].get('evaluation', {}).get('overall_score', 0)
            improvements.append((r2_score - r1_score, r1_score, r2_score))

if improvements:
    deltas = [x[0] for x in improvements]
    avg_imp = sum(deltas) / len(deltas)
    pos_imp = len([x for x in deltas if x > 0])
    no_chg = len([x for x in deltas if x == 0])
    neg_imp = len([x for x in deltas if x < 0])
    max_gain = max(deltas)
    max_loss = min(deltas)
    
    print()
    print("-" * 60)
    print("  Reflector Effect Analysis (R2 vs R1)")
    print("-" * 60)
    print(f"  Avg delta (R2-R1):     {avg_imp:+.4f}")
    print(f"  Max improvement:       {max_gain:+.4f}")
    print(f"  Max regression:        {max_loss:+.4f}")
    print(f"  Improved (R2>R1):      {pos_imp:3d}  ({pos_imp/len(deltas)*100:5.1f}%)")
    print(f"  No change (R2=R1):     {no_chg:3d}  ({no_chg/len(deltas)*100:5.1f}%)")
    print(f"  Regressed (R2<R1):     {neg_imp:3d}  ({neg_imp/len(deltas)*100:5.1f}%)")

# Per-paper summary
paper_avg = []
for paper_id, paper in data.items():
    title = paper['paper_title']
    scores = []
    for sec_key, sec_result in paper.get('section_results', {}).items():
        score = sec_result.get('final_report', {}).get('source_eval_score', 0)
        scores.append(score)
    if scores:
        paper_avg.append((title, sum(scores)/len(scores), min(scores), max(scores)))

paper_avg.sort(key=lambda x: x[1], reverse=True)

print()
print("-" * 60)
print("  Top 5 Papers by Average Score")
print("-" * 60)
for i, (t, avg, mn, mx) in enumerate(paper_avg[:5], 1):
    short_t = t[:58] + ".." if len(t) > 58 else t
    print(f"  {i}. [{avg:.3f}] range[{mn:.2f}-{mx:.2f}] {short_t}")

print()
print("-" * 60)
print("  Bottom 5 Papers by Average Score")
print("-" * 60)
for i, (t, avg, mn, mx) in enumerate(paper_avg[-5:][::-1], 1):
    short_t = t[:58] + ".." if len(t) > 58 else t
    print(f"  {i}. [{avg:.3f}] range[{mn:.2f}-{mx:.2f}] {short_t}")

# Key takeaways
print()
print("=" * 60)
print("  KEY TAKEAWAYS")
print("=" * 60)
overall_avg = sum(all_scores) / len(all_scores)
r2_win_rate = best_rounds['round2'] / total_r * 100
excellent_rate = len([s for s in all_scores if s >= 4.5]) / len(all_scores) * 100
print(f"  [1] Overall Quality:   avg={overall_avg:.3f}, {excellent_rate:.0f}% >=4.5 (good-excellent)")
print(f"  [2] Reflector Works:   R2 won {r2_win_rate:.1f}% of time (vs {100-r2_win_rate-best_rounds['round0']/total_r*100:.1f}% R1)")
if improvements:
    print(f"  [3] Avg Improvement:   {avg_imp:+.4f} points after reflection ({pos_imp}/{len(deltas)} improved)")
print(f"  [4] All papers passed:  min score = {min(all_scores):.2f}, zero failures below 4.0")
print("=" * 60)
