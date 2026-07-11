"""noise=0.4 wd 2.0-4.0 精细扫描"""
import json, argparse
from trainer import train_one_run

def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', type=str, default='cpu'); args = p.parse_args()
    wd_list = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    print(f"noise=0.4 wd scan: {wd_list}")
    results = {}
    for wd in wd_list:
        m, _ = train_one_run(p=97, label_noise_ratio=0.4, weight_decay=wd,
                             n_epochs=2000, embed_dim=128, hidden_dim=256,
                             record_interval=20, device=args.device)
        ea = m['erank']; ta = m['test_acc']; ep = m['epoch']
        mi = ea.index(min(ea)); mx = ta.index(max(ta))
        print(f"wd={wd:.1f}  arc: ep={ep[mi]} acc={ta[mi]:.4f} erank={ea[mi]:.0f}  final: acc={ta[-1]:.4f} erank={ea[-1]:.0f}  peak: {ta[mx]:.4f}@ep{ep[mx]}")
        results[str(wd)] = {'epoch': ep, 'test_acc': ta, 'erank': ea}
    with open('noise04_wd_2_4.json', 'w') as f: json.dump(results, f, indent=2, default=float)
    print("Done: noise04_wd_2_4.json")

if __name__ == '__main__': main()
