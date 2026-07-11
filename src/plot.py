"""Generate all figures from experiment data."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = 'data'
OUT = 'figures'

def load(fn):
    with open(f'{DATA}/{fn}') as f:
        return json.load(f)

# ============================================================
# Figure 1: Grokking trajectory (erank + test_acc)
# ============================================================
d = load('grokking_trajectory.json')
for k in ['wd_0.5','wd_0.0']:
    ep = d[k]['epoch']; ea = d[k]['erank']; ta = d[k]['test_acc']
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(ep, ea, 'b-', linewidth=1.2, label='erank')
    ax1.set_ylabel('erank', color='b'); ax1.tick_params(axis='y', colors='b')
    ax1.set_ylim(0, 160)
    ax2 = ax1.twinx()
    ax2.plot(ep, ta, 'r-', linewidth=1.2, label='Test Acc')
    ax2.set_ylabel('Test Accuracy', color='r'); ax2.tick_params(axis='y', colors='r')
    ax2.set_ylim(0, 1.05)
    label = 'wd=0.5 (grokking)' if '0.5' in k else 'wd=0.0 (no grokking)'
    plt.title(f'Grokking Trajectory — {label}')
    plt.tight_layout(); plt.savefig(f'{OUT}/grok_traj_{"wd05" if "0.5" in k else "wd00"}.png', dpi=120)
    plt.close()

# ============================================================
# Figure 2: Arc discovery — noise=0.1 wd scan
# ============================================================
d = load('arc_discovery_noise01.json')
fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(d)))
for i, k in enumerate(sorted(d.keys(), key=float)):
    ep = d[k]['epoch']; ea = d[k]['erank']; ta = d[k]['test_acc']
    ax.plot(ep, ea, color=colors[i], linewidth=1.2, label=f'wd={float(k):.1f}')
for i, k in enumerate(sorted(d.keys(), key=float)):
    ep = d[k]['epoch']; ta = d[k]['test_acc']
    ax.plot(ep, [t*160 for t in ta], '--', color=colors[i], linewidth=0.6, alpha=0.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('erank')
ax.set_title('Arc Discovery — noise=0.1, varying wd')
ax.legend(fontsize=7, ncol=2, loc='upper right')
plt.tight_layout(); plt.savefig(f'{OUT}/arc_discovery.png', dpi=120); plt.close()

# ============================================================
# Figure 3: Arc window comparison (noise=0 vs noise=0.2)
# ============================================================
d0 = load('grokking_trajectory.json')['wd_0.5']
d2 = load('arc_window_noise02.json')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# noise=0
ax1.plot(d0['epoch'], d0['erank'], 'b-', linewidth=1.2)
ax1.set_title('noise=0, wd=0.5 (no arc)'); ax1.set_ylabel('erank'); ax1.set_xlabel('epoch')

# noise=0.2 — pick best arc
for k in sorted(d2.keys(), key=float):
    v = d2[k]; c = 'orange' if float(k) == 2.2 else ('red' if float(k) > 2.2 else 'gray')
    lw = 1.5 if float(k) == 2.2 else 0.6
    ax2.plot(v['epoch'], v['erank'], color=c, linewidth=lw,
             label=f'wd={k}' if float(k) in [1.5, 2.2, 3.0] else None)
ax2.set_title('noise=0.2 (arc window)'); ax2.set_xlabel('epoch'); ax2.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/arc_window.png', dpi=120); plt.close()

# ============================================================
# Figure 4: Noise scan summary
# ============================================================
d = load('noise_scan.json')
fig, ax = plt.subplots(figsize=(6, 4))
noises = sorted([float(k) for k in d.keys()])
final_erank = [d[str(n)]['erank'][-1] for n in noises]
final_acc = [d[str(n)]['test_acc'][-1] for n in noises]
colors = ['#2ca02c' if a > 0.8 else '#d62728' for a in final_acc]
ax.bar(range(len(noises)), final_erank, color=colors, alpha=0.7)
ax.set_xticks(range(len(noises))); ax.set_xticklabels([f'{n:.1f}' for n in noises])
ax.set_xlabel('Label Noise Ratio'); ax.set_ylabel('Final erank')
ax.set_title('erank vs Noise Level (wd=0.5, 2000 epochs)')
for i, (e, a) in enumerate(zip(final_erank, final_acc)):
    ax.text(i, e + 2, f'acc={a:.2f}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/noise_scan.png', dpi=120); plt.close()

print('All figures saved to', OUT)
