"""Generate all figures — dual y-axis for all erank+acc plots."""
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
    ax.plot(ep, inv, 'b-', linewidth=1.5, label='140 - erank (compress)')
    ax.set_ylabel('140 - erank', color='b'); ax.set_ylim(-5, 145)
    ax2 = ax.twinx()
    ax2.plot(ep, acc, 'r-', linewidth=1.2, label='Test Accuracy')
    ax2.set_ylabel('Test Accuracy', color='r'); ax2.set_ylim(0, 1.05)
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
plt.suptitle('Fig 1: erank = Grokking Detector (p=97, dim=128, 20000e)', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_grokking.png', dpi=150); plt.close()

# ============================================================
# FIG 2: Best wd per noise
# ============================================================
best = [
    ('noise=0%, wd=0.5', 'grokking_trajectory.json', 'wd_0.5', ''),
    ('noise=10%, wd=2.0', 'arc_discovery_noise01.json', '2.0', ''),
    ('noise=20%, wd=2.15', 'arc_optimal_noise02.json', '2.15', ''),
    ('noise=40%', 'arc_noise04_closed.json', '2.0', '(no grokking)'),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for idx, (title, fn, key, note) in enumerate(best):
    v = load(fn)
    vv = v[key]
    ep = vv['epoch']; er = vv['erank']; ac = vv['test_acc']
    ax, _ = plot_erank_acc(axes[idx], ep, er, ac, f'{title}, wd={key} {note}')
plt.suptitle('Fig 2: Can-Grok Wd per Noise Level', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_best_wd.png', dpi=150); plt.close()

# ============================================================
# FIG 3: wd-vs-noise bar chart
# ============================================================
noises = [0.0, 0.1, 0.2, 0.4]
opt_wd = [0.3, 2.0, 2.15, None]
peak_acc = [1.0, 0.947, 0.947, None]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar(range(3), opt_wd[:3], color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.8)
ax1.set_xticks(range(3)); ax1.set_xticklabels(['0%', '10%', '20%'])
ax1.set_xlabel('Label Noise'); ax1.set_ylabel('Wd that can Grok')
ax1.set_title('Required Wd vs Noise')
for i, w in enumerate(opt_wd[:3]): ax1.text(i, w+0.05, f'{w}', ha='center', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.annotate('noise=40%:\nno grokking', xy=(3, 1.5), fontsize=9, ha='center', color='#888',
             bbox=dict(boxstyle='round', facecolor='#ffeeee', alpha=0.7))
cs = ['#2ca02c' if a and a>0.9 else '#d62728' for a in peak_acc[:3]]
ax2.bar(range(3), [a or 0 for a in peak_acc[:3]], color=cs, alpha=0.8)
ax2.set_xticks(range(3)); ax2.set_xticklabels(['0%', '10%', '20%'])
ax2.set_xlabel('Label Noise'); ax2.set_ylabel('Peak Accuracy'); ax2.set_ylim(0, 1.1)
ax2.set_title('Peak Accuracy at Grokking Wd')
for i, a in enumerate(peak_acc[:3]):
    if a: ax2.text(i, a+0.03, f'{a:.1%}', ha='center', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2)
plt.suptitle('Fig 3: Wd Required for Grokking Rises with Noise', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_wd_vs_noise.png', dpi=150); plt.close()

# ============================================================
# FIG 4: Same wd=0.5 across noise
# ============================================================
v = load('noise_scan.json')
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for idx, n in enumerate(['0.0', '0.1', '0.2', '0.4']):
    vv = v[n]; ep = vv['epoch']; er = vv['erank']; ac = vv['test_acc']
    plot_erank_acc(axes[idx], ep, er, ac, f'noise={float(n):.0%}, wd=0.5')
plt.suptitle('Fig 4: Same wd=0.5 Fails at Higher Noise', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_noise_wd05.png', dpi=150); plt.close()

# ============================================================
# FIG 5: Three regimes x two noise levels
# ============================================================
d01 = load('arc_discovery_noise01.json')   # noise=10%
d01h = load('arc_high_wd_noise01.json')    # noise=10% high wd
d02 = load('arc_window_noise02.json')      # noise=20%
d02h = load('arc_broad_noise02.json')      # noise=20% broad

panels = [
    ('noise=10%, wd=0.5 (too low)', d01['0.5']),
    ('noise=10%, wd=2.0 (arc!)', d01['2.0']),
    ('noise=10%, wd=5.0 (collapse)', d01h['5.0']),
    ('noise=20%, wd=1.5 (too low)', d02h['1.5']),
    ('noise=20%, wd=2.2 (arc!)', d02['2.2']),
    ('noise=20%, wd=3.0 (collapse)', d02['3.0']),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for idx, (title, v) in enumerate(panels):
    ax = axes[idx // 3][idx % 3]
    ep = v['epoch']; er = v['erank']; ac = v['test_acc']
    a1, a2 = plot_erank_acc(ax, ep, er, ac, title)
    if 'arc!' in title:
        mi = er.index(min(er))
        a1.annotate('STOP', xy=(ep[mi], 140 - er[mi]),
                    xytext=(ep[mi]+150, 140 - er[mi] - 20),
                    arrowprops=dict(arrowstyle='->', color='#ff8800', lw=2),
                    fontsize=9, fontweight='bold', color='#cc6600',
                    bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.9))
plt.suptitle('Fig 5: Practical Guide - Three Regimes at Two Noise Levels', y=1.01, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig5_wd_regimes.png', dpi=150); plt.close()

# ============================================================
# FIG 6: Full wd scan for noise=10% — demonstrating the wd window
# ============================================================
d01 = load('arc_discovery_noise01.json')  # wd=0.1 to 2.0
d01h = load('arc_high_wd_noise01.json')   # wd=2.0, 3.0, 5.0
# Combine all wd values
all_wd = {}
for k, v in d01.items(): all_wd[float(k)] = v
for k, v in d01h.items():
    if float(k) not in all_wd: all_wd[float(k)] = v

wd_sorted = sorted(all_wd.keys())
n = len(wd_sorted)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axes = axes.flatten()

for idx, wd in enumerate(wd_sorted):
    ax = axes[idx]
    v = all_wd[wd]; ep = v['epoch']; er = v['erank']; ac = v['test_acc']
    # Color code: gray=stuck, orange=arc, red=collapse
    if wd < 1.5: clr = '#888888'
    elif wd >= 5.0: clr = '#cc4444'
    elif 2.0 <= wd <= 2.2: clr = '#ff8800'
    else: clr = '#999999'
    plot_erank_acc(ax, ep, er, ac, f'noise=10%, wd={wd:.1f}')
    if 2.0 <= wd <= 2.2:
        ax.patch.set_facecolor('#fff8e1')
for idx in range(len(wd_sorted), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle('Fig 6: Full wd Scan — noise=10%, p=97, dim=128 (orange=arc window)', y=1.01, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig6_wd_window.png', dpi=150); plt.close()
print('Done:', OUT)
