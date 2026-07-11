"""Generate step1+step2 figs for noise=0.2 wd guide + existing figs 1-4."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = 'data'; OUT = 'figures'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 9})

def load(fn):
    with open(f'{DATA}/{fn}') as f: return json.load(f)

def plot_erank_acc(ax, ep, erank, acc, title, xlabel='Epoch'):
    inv = [140 - e for e in erank]
    ax.plot(ep, inv, 'b-', linewidth=1.5, label='140 - erank')
    ax.set_ylabel('140 - erank', color='b'); ax.set_ylim(-5, 145)
    ax2 = ax.twinx()
    ax2.plot(ep, acc, 'r-', linewidth=1.2, label='Test Acc')
    ax2.set_ylabel('Test Acc', color='r'); ax2.set_ylim(0, 1.05)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(True, alpha=0.2)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=7, loc='lower right')
    return ax, ax2

# ============================================================
# FIG 1: Grokking vs memorization
# ============================================================
d = load('grokking_trajectory.json')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_erank_acc(axes[0], d['wd_0.5']['epoch'], d['wd_0.5']['erank'],
               d['wd_0.5']['test_acc'], 'noise=0, wd=0.5 (grokking)')
plot_erank_acc(axes[1], d['wd_0.0']['epoch'], d['wd_0.0']['erank'],
               d['wd_0.0']['test_acc'], 'noise=0, wd=0.0 (memorization)')
plt.suptitle('Fig 1: erank = Grokking Detector', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_grokking.png', dpi=150); plt.close()

# ============================================================
# FIG 2: Can-grok wd per noise
# ============================================================
best = [
    ('noise=0%, wd=0.5', 'grokking_trajectory.json', 'wd_0.5', ''),
    ('noise=10%, wd=2.0', 'arc_discovery_noise01.json', '2.0', ''),
    ('noise=20%, wd=2.15', 'arc_optimal_noise02.json', '2.15', ''),
    ('noise=40%', 'arc_noise04_closed.json', '2.0', '(no grokking)'),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for idx, (title, fn, key, note) in enumerate(best):
    v = load(fn)[key]
    plot_erank_acc(axes[idx], v['epoch'], v['erank'], v['test_acc'],
                   f'{title}, wd={key} {note}')
plt.suptitle('Fig 2: Wd that Enables Grokking per Noise Level', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_best_wd.png', dpi=150); plt.close()

# ============================================================
# FIG 3: Wd-vs-noise bar chart
# ============================================================
noises = [0.0, 0.1, 0.2, 0.4]
opt_wd = [0.3, 2.0, 2.15, None]
peak_acc = [1.0, 0.947, 0.947, None]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar(range(3), opt_wd[:3], color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.8)
ax1.set_xticks(range(3)); ax1.set_xticklabels(['0%', '10%', '20%'])
ax1.set_xlabel('Label Noise'); ax1.set_ylabel('Wd for Grokking')
ax1.set_title('Required Wd vs Noise')
for i, w in enumerate(opt_wd[:3]): ax1.text(i, w+0.05, f'{w}', ha='center', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.annotate('noise=40%: no grokking', xy=(3, 1.5), fontsize=9, ha='center', color='#888',
             bbox=dict(boxstyle='round', facecolor='#ffeeee', alpha=0.7))
cs = ['#2ca02c' if a and a>0.9 else '#d62728' for a in peak_acc[:3]]
ax2.bar(range(3), [a or 0 for a in peak_acc[:3]], color=cs, alpha=0.8)
ax2.set_xticks(range(3)); ax2.set_xticklabels(['0%', '10%', '20%'])
ax2.set_xlabel('Label Noise'); ax2.set_ylabel('Peak Accuracy'); ax2.set_ylim(0, 1.1)
ax2.set_title('Peak Accuracy')
for i, a in enumerate(peak_acc[:3]):
    if a: ax2.text(i, a+0.03, f'{a:.1%}', ha='center', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2)
plt.suptitle('Fig 3: Required Wd Rises with Noise', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_wd_vs_noise.png', dpi=150); plt.close()

# ============================================================
# FIG 4: Same wd=0.5 across noise levels
# ============================================================
v = load('noise_scan.json')
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for idx, n in enumerate(['0.0', '0.1', '0.2', '0.4']):
    vv = v[n]
    plot_erank_acc(axes[idx], vv['epoch'], vv['erank'], vv['test_acc'],
                   f'noise={float(n):.0%}, wd=0.5')
plt.suptitle('Fig 4: Same wd=0.5 Fails at Higher Noise', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_noise_wd05.png', dpi=150); plt.close()

# ============================================================
# FIG 5: Step 1 — Broad scan (noise=20%, wd 0.1-2.0)
# ============================================================
db = load('arc_broad_noise02.json')
wds = sorted([float(k) for k in db.keys()])
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, wd in enumerate(wds):
    v = db[str(wd)]; ep = v['epoch']; er = v['erank']; ac = v['test_acc']
    if wd < 1.5: clr = '#888888'
    elif 1.8 <= wd <= 2.0: clr = '#ff8800'
    else: clr = '#999999'
    plot_erank_acc(axes[i], ep, er, ac, f'wd={wd:.1f}')
    if 1.8 <= wd <= 2.0: axes[i].patch.set_facecolor('#fff8e1')
for i in range(len(wds), len(axes)): axes[i].set_visible(False)
plt.suptitle('Fig 5: Step 1 — Broad Scan, noise=20% (locate the wd window)', y=1.01, fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/fig5_broad_scan.png', dpi=150); plt.close()

# ============================================================
# FIG 6: Step 2 — Fine-tune (noise=20%, wd 2.0-2.2)
# ============================================================
df = load('arc_optimal_noise02.json')
wd_fine = sorted([float(k) for k in df.keys()])
fig, axes = plt.subplots(1, len(wd_fine), figsize=(18, 3.8))
for i, wd in enumerate(wd_fine):
    v = df[str(wd)]; ep = v['epoch']; er = v['erank']; ac = v['test_acc']
    ax, a2 = plot_erank_acc(axes[i], ep, er, ac, f'wd={wd:.2f}')
    mi = er.index(min(er))
    ax.annotate(f'dip: {140-er[mi]:.0f}', xy=(ep[mi], 140-er[mi]),
                xytext=(ep[mi]+60, 140-er[mi]-15),
                arrowprops=dict(arrowstyle='->', color='#ff8800', lw=1.5),
                fontsize=8, fontweight='bold', color='#cc6600',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9))
plt.suptitle('Fig 6: Step 2 — Fine-Tune, noise=20%, wd 2.0-2.2 (find sweet spot)', y=1.03, fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/fig6_fine_tune.png', dpi=150); plt.close()
print('Done:', OUT)
