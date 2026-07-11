"""Generate all figures — English labels for font compatibility."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = 'data'; OUT = 'figures'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 10})

def load(fn):
    with open(f'{DATA}/{fn}') as f: return json.load(f)

# ============================================================
# FIG 1: Grokking — wd=0.5 vs wd=0.0
# ============================================================
d = load('grokking_trajectory.json')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, key, title in [(ax1, 'wd_0.5', 'Grokking (wd=0.5)'), (ax2, 'wd_0.0', 'Memorization (wd=0.0)')]:
    v = d[key]; ep = v['epoch']; ea = v['erank']; ta = v['test_acc']
    ax.plot(ep, ea, 'b-', linewidth=1.2, label='erank')
    ax2t = ax.twinx()
    ax2t.plot(ep, ta, 'r-', linewidth=1.5, label='Test Acc')
    ax.set_ylim(0, 160); ax2t.set_ylim(0, 1.05)
    ax.set_xlabel('Epoch'); ax.set_ylabel('erank', color='b')
    ax2t.set_ylabel('Test Accuracy', color='r')
    ax.set_title(title); ax.grid(True, alpha=0.2)
    ax.annotate(f'{ea[0]:.0f}', xy=(ep[0],ea[0]), fontsize=9, color='b')
    ax.annotate(f'{ea[-1]:.0f}', xy=(ep[-1],ea[-1]), fontsize=9, color='b',
                xytext=(ep[-1]-1500, ea[-1]+8))
plt.suptitle('Fig 1: Grokking Trajectory (p=97, dim=128, 20000 epochs)', y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_grokking.png', dpi=150); plt.close()

# ============================================================
# FIG 2: Universality across noise levels
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
configs = [
    ('A: noise=0% (no arc)', 'grokking_trajectory.json', 'wd_0.5', True),
    ('B: noise=10% (arc appears)', 'arc_discovery_noise01.json', None, False),
    ('C: noise=20% (window narrow)', 'arc_window_noise02.json', None, False),
    ('D: noise=40% (window closed)', 'arc_noise04_closed.json', None, False),
]
for idx, (title, fn, wd_key, is_grok) in enumerate(configs):
    ax = axes[idx // 2][idx % 2]
    if is_grok:
        v = load(fn)[wd_key]
        ax.plot(v['epoch'], v['erank'], 'b-', linewidth=1.5)
        ax2 = ax.twinx(); ax2.plot(v['epoch'], v['test_acc'], 'r--', linewidth=1, alpha=0.4)
        ax2.set_ylim(0, 1.05)
    else:
        d2 = load(fn)
        for k in sorted(d2.keys(), key=float):
            v = d2[k]; wd = float(k)
            if wd < 1.5: c, lw, alpha = '#aaaaaa', 0.5, 0.25
            elif wd > 2.5: c, lw, alpha = '#cc4444', 0.5, 0.35
            elif 2.0 <= wd <= 2.2: c, lw, alpha = '#ff8800', 1.8, 0.9
            else: c, lw, alpha = '#888888', 0.8, 0.45
            ax.plot(v['epoch'], v['erank'], color=c, linewidth=lw, alpha=alpha,
                    label=f'wd={wd:.1f}' if wd in [0.5, 2.0, 3.0] else None)
        from matplotlib.lines import Line2D
        custom = [Line2D([0],[0],color='#aaaaaa',lw=1,label='wd too low'),
                  Line2D([0],[0],color='#ff8800',lw=2,label='arc region'),
                  Line2D([0],[0],color='#cc4444',lw=1,label='wd too high')]
        ax.legend(handles=custom, fontsize=7, loc='upper right')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('erank')
    ax.set_ylim(0, 160); ax.grid(True, alpha=0.2)
plt.suptitle('Fig 2: Universality of the Arc Phenomenon', fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_universality.png', dpi=150); plt.close()

# ============================================================
# FIG 3: Noise scan
# ============================================================
v = load('noise_scan.json')
noises = sorted([float(k) for k in v.keys()])
fe = [v[str(n)]['erank'][-1] for n in noises]
fa = [v[str(n)]['test_acc'][-1] for n in noises]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
colors = ['#2ca02c' if a > 0.8 else '#d62728' for a in fa]
ax1.bar(range(len(noises)), fe, color=colors, alpha=0.7)
ax1.set_xticks(range(len(noises))); ax1.set_xticklabels([f'{n:.0%}' for n in noises])
ax1.set_xlabel('Label Noise'); ax1.set_ylabel('Final erank'); ax1.set_title('erank vs Noise')
for i, (e, a) in enumerate(zip(fe, fa)):
    ax1.text(i, e+2, f'{e:.0f}', ha='center', fontsize=8)
ax1.grid(True, alpha=0.2)
ax2.bar(range(len(noises)), fa, color=colors, alpha=0.7)
ax2.set_xticks(range(len(noises))); ax2.set_xticklabels([f'{n:.0%}' for n in noises])
ax2.set_xlabel('Label Noise'); ax2.set_ylabel('Final Accuracy'); ax2.set_ylim(0, 1.1)
ax2.set_title('Accuracy vs Noise')
for i, a in enumerate(fa): ax2.text(i, a+0.03, f'{a:.0%}', ha='center', fontsize=8)
ax2.grid(True, alpha=0.2)
plt.suptitle('Fig 3: Noise Scan (p=97, dim=128, wd=0.5, 2000e)', y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_noise.png', dpi=150); plt.close()

# ============================================================
# FIG 4: Arc detail — noise=0.1 with annotations
# ============================================================
d = load('arc_discovery_noise01.json')
fig, ax = plt.subplots(figsize=(9, 5))
for k in sorted(d.keys(), key=float):
    v = d[k]; wd = float(k)
    if wd == 2.0: c, lw, alpha, lab = '#ff8800', 2.2, 1.0, 'wd=2.0 (best arc)'
    elif wd == 1.5: c, lw, alpha, lab = '#888888', 1.2, 0.7, 'wd=1.5 (no arc)'
    elif wd == 3.0: c, lw, alpha, lab = '#cc4444', 1.2, 0.7, 'wd=3.0 (collapse)'
    else: c, lw, alpha, lab = '#dddddd', 0.5, 0.25, None
    ax.plot(v['epoch'], v['erank'], color=c, linewidth=lw, alpha=alpha, label=lab)
v2 = d['2.0']; mi = v2['erank'].index(min(v2['erank']))
ax.annotate(f'Arc bottom: erank={v2["erank"][mi]:.0f}, acc={v2["test_acc"][mi]:.1%}',
            xy=(v2['epoch'][mi], v2['erank'][mi]),
            xytext=(v2['epoch'][mi]+400, v2['erank'][mi]+12),
            arrowprops=dict(arrowstyle='->', color='#ff8800', lw=1.5),
            fontsize=11, color='#cc6600', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('erank')
ax.set_title('Fig 4: Arc Structure (noise=10%, wd comparison)')
ax.set_ylim(70, 142); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_arc_detail.png', dpi=150); plt.close()
print('Done:', OUT)
