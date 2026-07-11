"""Generate all figures — v2: inverted erank, combined panels."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = 'data'; OUT = 'figures'

def load(fn):
    with open(f'{DATA}/{fn}') as f:
        return json.load(f)

# --- FIG 1: Grokking trajectory with inverted erank ---
d = load('grokking_trajectory.json')
for label, key in [('wd=0.5 (grokking)', 'wd_0.5'), ('wd=0.0 (no grokking)', 'wd_0.0')]:
    v = d[key]; ep = v['epoch']; ea = v['erank']; ta = v['test_acc']
    erank_inv = [1/e * 10000 for e in ea]  # inverted: bigger = better
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(ep, erank_inv, 'b-', linewidth=1.2, label='1/erank (inverted, higher=better)')
    ax1.set_ylabel('Compression (1/erank × 10⁴)', color='b')
    ax1.tick_params(axis='y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(ep, ta, 'r-', linewidth=1.5, label='Test Accuracy')
    ax2.set_ylabel('Test Accuracy', color='r'); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', colors='r')
    ax1.set_xlabel('Epoch')
    ax1.set_title(f'Grokking Trajectory — {label}')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    fn = 'grok_traj_wd05' if '0.5' in key else 'grok_traj_wd00'
    plt.savefig(f'{OUT}/{fn}.png', dpi=150); plt.close()

# --- FIG 2: Arc discovery — all curves on one plot ---
d = load('arc_discovery_noise01.json')
fig, ax = plt.subplots(figsize=(10, 5))
cmap = plt.cm.viridis
for i, k in enumerate(sorted(d.keys(), key=float)):
    v = d[k]; wd = float(k)
    if wd < 1: color = '#999999'; alpha = 0.4; lw = 0.7; ls = '-'
    elif wd > 2.5: color = '#cc4444'; alpha = 0.5; lw = 0.7; ls = '-'
    elif 2.0 <= wd <= 2.2: color = '#ff8800'; alpha = 0.9; lw = 1.8; ls = '-'
    else: color = cmap(i/len(d)); alpha = 0.6; lw = 1.0; ls = '-'
    inv = [1/e * 10000 for e in v['erank']]
    ax.plot(v['epoch'], inv, color=color, linewidth=lw, alpha=alpha, linestyle=ls,
            label=f'wd={wd:.1f}')
ax.set_xlabel('Epoch'); ax.set_ylabel('Compression (1/erank × 10⁴)')
ax.set_title('Arc Discovery — noise=0.1, varying weight decay')
ax.legend(fontsize=7, ncol=2, loc='lower right'); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/arc_discovery.png', dpi=150); plt.close()

# --- FIG 3: Combined multi-experiment universality plot ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
# panel A: grokking (noise=0)
v = load('grokking_trajectory.json')['wd_0.5']
inv = [1/e * 10000 for e in v['erank']]
ax = axes[0, 0]; ax.plot(v['epoch'], inv, 'b-', linewidth=1.2)
ax2 = ax.twinx(); ax2.plot(v['epoch'], v['test_acc'], 'r--', linewidth=1)
ax.set_title('A: noise=0, wd=0.5 (no arc)'); ax.set_xlabel('Epoch')
ax.set_ylabel('1/erank'); ax.grid(True, alpha=0.2)
# panel B: arc noise=0.1
v = load('arc_discovery_noise01.json')
ax = axes[0, 1]; cmap = plt.cm.viridis
for i, k in enumerate(sorted(d.keys(), key=float)):
    vv = v[k]; wd=float(k)
    c = '#ff8800' if 2.0<=wd<=2.2 else ('#999' if wd<2 else '#c44')
    ax.plot(vv['epoch'], [1/e*10000 for e in vv['erank']], color=c, linewidth=1.5 if 2<=wd<=2.2 else 0.5,
            alpha=0.8 if 2<=wd<=2.2 else 0.3)
ax.set_title('B: noise=0.1 (arc appears)'); ax.set_xlabel('Epoch'); ax.set_ylabel('1/erank'); ax.grid(True, alpha=0.2)
# panel C: arc noise=0.2
v = load('arc_window_noise02.json')
ax = axes[1, 0]
for k in sorted(v.keys(), key=float):
    vv = v[k]; wd=float(k)
    c = '#ff8800' if 2.0<=wd<=2.2 else ('#999' if wd<2 else '#c44')
    ax.plot(vv['epoch'], [1/e*10000 for e in vv['erank']], color=c, linewidth=1.5 if 2<=wd<=2.2 else 0.5,
            alpha=0.8 if 2<=wd<=2.2 else 0.3)
ax.set_title('C: noise=0.2 (window narrow)'); ax.set_xlabel('Epoch'); ax.set_ylabel('1/erank'); ax.grid(True, alpha=0.2)
# panel D: noise=0.4
v = load('arc_noise04_closed.json')
ax = axes[1, 1]
for k in sorted(v.keys(), key=float):
    vv = v[k]; ax.plot(vv['epoch'], [1/e*10000 for e in vv['erank']],
                       color='#999', linewidth=0.5, alpha=0.5)
ax.set_title('D: noise=0.4 (window closed)'); ax.set_xlabel('Epoch'); ax.set_ylabel('1/erank')
ax.set_ylim(0, 90); ax.grid(True, alpha=0.2)
plt.suptitle('Universality of the Arc Phenomenon', fontsize=13, y=1.01)
plt.tight_layout(); plt.savefig(f'{OUT}/universality.png', dpi=150, bbox_inches='tight'); plt.close()

# --- FIG 4: Noise scan bar chart ---
v = load('noise_scan.json')
noises = sorted([float(k) for k in v.keys()])
final_erank = [v[str(n)]['erank'][-1] for n in noises]
final_acc = [v[str(n)]['test_acc'][-1] for n in noises]
inv_erank = [1/e * 10000 for e in final_erank]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(len(noises)), inv_erank, color=['#2ca02c' if a>0.8 else '#d62728' for a in final_acc], alpha=0.7)
ax.set_xticks(range(len(noises))); ax.set_xticklabels([f'{n:.0%}' for n in noises])
ax.set_xlabel('Label Noise Ratio'); ax.set_ylabel('Compression (1/erank × 10⁴)')
ax.set_title('Compression vs Noise Level (wd=0.5, 2000 epochs)')
for i, (inv, e, a) in enumerate(zip(inv_erank, final_erank, final_acc)):
    ax.text(i, inv+1, f'erank={e:.0f}\nacc={a:.0%}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/noise_scan.png', dpi=150); plt.close()
print('All figures saved to', OUT)
