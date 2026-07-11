"""Generate all figures v6 — single best-wd per noise, wd-vs-noise phase diagram."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = 'data'; OUT = 'figures'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 9})

def load(fn):
    with open(f'{DATA}/{fn}') as f: return json.load(f)

# ============================================================
# FIG 1: All wd — noise=0 and noise=0.1, inverted erank
# ============================================================
panels = [
    ('noise=0%, wd=0.5', 'grokking_trajectory.json', ['wd_0.5'], True),
    ('noise=0%, wd=0.0', 'grokking_trajectory.json', ['wd_0.0'], True),
    ('noise=10%, wd=0.1-2.0', 'arc_discovery_noise01.json', None, False),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (title, fn, wd_keys, is_single) in enumerate(panels):
    ax = axes[idx]
    if is_single:
        v = load(fn)[wd_keys[0]]; ep = v['epoch']; ta = v['test_acc']
        inv = [140 - e for e in v['erank']]
        ax.plot(ep, inv, 'b-', linewidth=1.5, label='140 - erank')
        ax.plot(ep, [t*140 for t in ta], 'r--', linewidth=1.2, alpha=0.5, label='Acc x140')
    else:
        d2 = load(fn); cmap = plt.cm.tab10
        for i, k in enumerate(sorted(d2.keys(), key=float)):
            v = d2[k]; wd = float(k)
            inv = [140 - e for e in v['erank']]
            c = cmap(i%10); lw = 1.8 if wd in [2.0,2.2] else 0.7
            alpha = 0.9 if wd in [2.0,2.2] else 0.3
            ax.plot(v['epoch'], inv, color=c, linewidth=lw, alpha=alpha, label=f'wd={wd:.1f}')
        ax.axvspan(400, 1000, alpha=0.06, color='orange')
        ax.legend(fontsize=6, ncol=2, loc='lower right')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('140 - erank')
    ax.set_ylim(-5, 145); ax.grid(True, alpha=0.2)
plt.suptitle('Fig 1: erank tracks grokking across all weight decay values', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_all_wd.png', dpi=150); plt.close()

# ============================================================
# FIG 2: Best wd per noise level — inverted erank + accuracy
# ============================================================
# Data: (noise_label, source_file, wd_key)
best_per_noise = [
    ('noise=0%', 'grokking_trajectory.json', 'wd_0.5', None),
    ('noise=10%', 'arc_discovery_noise01.json', '2.0', None),
    ('noise=20%', 'arc_optimal_noise02.json', '2.15', None),
    ('noise=40%', 'arc_noise04_closed.json', '2.0', '(no grokking)'),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for idx, (title, fn, wd_key, note) in enumerate(best_per_noise):
    ax = axes[idx]; v = load(fn)
    if wd_key in v:
        vv = v[wd_key]
    else:
        # grokking_trajectory special case
        vv = v[wd_key]
    ep = vv['epoch']; ta = vv['test_acc']
    inv = [140 - e for e in vv['erank']]
    ax.plot(ep, inv, 'b-', linewidth=1.5, label='140 - erank')
    ax.plot(ep, [t*140 for t in ta], 'r--', linewidth=1.2, alpha=0.5, label='Acc')
    ax.set_title(f'{title}, best wd={wd_key} {note or ""}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('140 - erank')
    ax.set_ylim(-5, 145); ax.grid(True, alpha=0.2)
    # annotate final values
    ax.annotate(f'erank={vv["erank"][-1]:.0f}\nacc={ta[-1]:.0%}', xy=(0.98, 0.95),
                xycoords='axes fraction', fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.suptitle('Fig 2: Best Weight Decay per Noise Level', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_best_wd.png', dpi=150); plt.close()

# ============================================================
# FIG 3: Optimal wd shifts with noise
# ============================================================
# Hardcoded from experiments:
# noise=0%: best wd=0.3, peak acc=100%, final erank=45.6
# noise=10%: best wd=2.0, peak acc=94.7%, final erank=93
# noise=20%: best wd=2.15, peak acc=94.7%, final erank=93
# noise=40%: no grokking
noises = [0.0, 0.1, 0.2, 0.4]
opt_wd = [0.3, 2.0, 2.15, None]
peak_acc = [1.0, 0.947, 0.947, None]
final_erank = [46, 93, 93, None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
# left: optimal wd vs noise
bars = ax1.bar(range(3), opt_wd[:3], color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.8)
ax1.set_xticks(range(3)); ax1.set_xticklabels(['0%', '10%', '20%'])
ax1.set_xlabel('Label Noise'); ax1.set_ylabel('Optimal Weight Decay')
ax1.set_title('Optimal wd vs Noise')
for i, w in enumerate(opt_wd[:3]):
    ax1.text(i, w+0.05, f'wd={w}', ha='center', fontsize=11, fontweight='bold')
ax1.axhline(y=0, color='gray', linewidth=0.5); ax1.grid(True, alpha=0.2)
ax1.annotate('noise=40%:\nno grokking\nat any wd', xy=(3, 1.5), fontsize=9, color='#888',
             ha='center', bbox=dict(boxstyle='round', facecolor='#ffeeee', alpha=0.7))

# right: peak accuracy at optimal wd
colors = []
for a in peak_acc[:3]:
    colors.append('#2ca02c' if a and a > 0.9 else '#d62728')
ax2.bar(range(3), [a if a else 0 for a in peak_acc[:3]], color=colors, alpha=0.8)
ax2.set_xticks(range(3)); ax2.set_xticklabels(['0%', '10%', '20%'])
ax2.set_xlabel('Label Noise'); ax2.set_ylabel('Peak Accuracy'); ax2.set_ylim(0, 1.1)
ax2.set_title('Peak Accuracy at Optimal wd')
for i, a in enumerate(peak_acc[:3]):
    if a: ax2.text(i, a+0.03, f'{a:.1%}', ha='center', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2)
plt.suptitle('Fig 3: Optimal wd Increases with Noise, Accuracy Eventually Falls', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_wd_vs_noise.png', dpi=150); plt.close()

# ============================================================
# FIG 4: Same wd across noise levels
# ============================================================
# Compare wd=0.5 across noise=0/0.1/0.2/0.4 from noise_scan.json
v = load('noise_scan.json')
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for idx, n in enumerate(['0.0', '0.1', '0.2', '0.4']):
    ax = axes[idx]; vv = v[n]; ep = vv['epoch']; ta = vv['test_acc']
    inv = [140 - e for e in vv['erank']]
    ax.plot(ep, inv, 'b-', linewidth=1.5, label='140 - erank')
    ax.plot(ep, [t*140 for t in ta], 'r--', linewidth=1.2, alpha=0.5, label='Acc')
    ax.set_title(f'noise={float(n):.0%}, wd=0.5'); ax.set_xlabel('Epoch'); ax.set_ylabel('140 - erank')
    ax.set_ylim(-5, 145); ax.grid(True, alpha=0.2)
    ax.annotate(f'erank={vv["erank"][-1]:.0f}\nacc={ta[-1]:.0%}', xy=(0.98, 0.95),
                xycoords='axes fraction', fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.suptitle('Fig 4: Same wd=0.5 Fails at Higher Noise', y=1.02, fontsize=12)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_noise_wd05.png', dpi=150); plt.close()

# ============================================================
# FIG 5: Three wd regimes — practical guide
# ============================================================
d1 = load('arc_discovery_noise01.json')
d2 = load('arc_high_wd_noise01.json')
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ax=axes[0]
for wd in ['0.1','0.3','0.5']:
    v=d1[wd];inv=[140-e for e in v['erank']]
    ax.plot(v['epoch'],inv,color='#888888',linewidth=1.2,label=f'wd={wd}')
ax.set_title('wd too low: no compression');ax.set_xlabel('Epoch');ax.set_ylabel('140 - erank')
ax.set_ylim(-5,145);ax.grid(True,alpha=0.2);ax.legend(fontsize=8)
ax.annotate('erank flat\n-> stuck',xy=(1200,18),fontsize=10,color='#666',
            bbox=dict(boxstyle='round',facecolor='#ffeeee',alpha=0.8))
ax=axes[1]
v=d1['2.0'];inv=[140-e for e in v['erank']]
ax.plot(v['epoch'],inv,color='#ff8800',linewidth=2.0,label='wd=2.0')
ax.set_title('wd just right: arc -> stop at dip');ax.set_xlabel('Epoch');ax.set_ylabel('140 - erank')
ax.set_ylim(-5,145);ax.grid(True,alpha=0.2);ax.legend(fontsize=9)
mi=v['erank'].index(min(v['erank']))
ax.annotate('STOP HERE',xy=(v['epoch'][mi],140-v['erank'][mi]),
            xytext=(v['epoch'][mi]+200,140-v['erank'][mi]-25),
            arrowprops=dict(arrowstyle='->',color='#ff8800',lw=2),fontsize=11,fontweight='bold',
            color='#cc6600',bbox=dict(boxstyle='round',facecolor='#ffffcc',alpha=0.9))
ax=axes[2]
v=d2['3.0'];inv=[140-e for e in v['erank']]
ax.plot(v['epoch'],inv,color='#cc4444',linewidth=2.0,label='wd=3.0')
ax.set_title('wd too high: collapse');ax.set_xlabel('Epoch');ax.set_ylabel('140 - erank')
ax.set_ylim(-5,145);ax.grid(True,alpha=0.2);ax.legend(fontsize=9)
ax.annotate('erank = 0\n-> dead',xy=(1200,8),fontsize=10,color='#c44',
            bbox=dict(boxstyle='round',facecolor='#ffeeee',alpha=0.8))
plt.suptitle('Fig 5: Practical Guide — Using erank to Diagnose Weight Decay',y=1.02,fontsize=12)
plt.tight_layout();plt.savefig(f'{OUT}/fig5_wd_regimes.png',dpi=150);plt.close()
print('Done:', OUT)
