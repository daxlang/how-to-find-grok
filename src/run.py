"""
Grokking Arc 实验入口
=====================
用法:
    # Ark 发现实验：noise=0.1, wd 扫描找弧
    python run.py --mode arc --device cuda

    # 噪声扫描：不同 noise 下 erank 分化
    python run.py --mode noise --device cuda

    # Grokking 轨迹：单次长训练追踪
    python run.py --mode grok --epochs 20000 --device cuda

所有结果保存为 JSON，包含 erank/test_acc/train_acc 完整轨迹。
"""
import json, argparse
from trainer import train_one_run


def mode_arc(args):
    """noise=0.1 下扫 wd，找弧形结构"""
    wd_list = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    results = {}
    print("=" * 60)
    print(f"  Arc scan: noise=0.1, p={args.p}, {args.epochs}e, dim={args.dim}")
    print(f"  wd: {wd_list}")
    print("=" * 60)
    for wd in wd_list:
        m, _ = train_one_run(p=args.p, label_noise_ratio=0.1, weight_decay=wd,
                             n_epochs=args.epochs, embed_dim=args.dim,
                             hidden_dim=args.dim * 2,
                             record_interval=args.rec, device=args.device)
        ea = m['erank']; ta = m['test_acc']; ep = m['epoch']
        mi = ea.index(min(ea))
        print(f"  wd={wd:<5.1f} arc_bottom: ep={ep[mi]:<6} acc={ta[mi]:.4f} erank={ea[mi]:.1f}  final: acc={ta[-1]:.4f} erank={ea[-1]:.1f}")
        results[str(wd)] = {'epoch': ep, 'test_acc': ta, 'erank': ea}
    return results


def mode_noise(args):
    """不同 noise 下 erank 分化（固定 wd）"""
    noise_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = {}
    print("=" * 60)
    print(f"  Noise scan: wd={args.wd}, p={args.p}, {args.epochs}e")
    print(f"  noise: {noise_list}")
    print("=" * 60)
    for nr in noise_list:
        m, _ = train_one_run(p=args.p, label_noise_ratio=nr, weight_decay=args.wd,
                             n_epochs=args.epochs, embed_dim=args.dim,
                             hidden_dim=args.dim * 2,
                             record_interval=args.rec, device=args.device)
        ea = m['erank']; ta = m['test_acc']
        print(f"  noise={nr:.1f} test_acc={ta[-1]:.4f} erank={ea[0]:.1f}->{ea[len(ea)//2]:.1f}->{ea[-1]:.1f}")
        results[str(nr)] = {'epoch': m['epoch'], 'test_acc': ta, 'erank': ea}
    return results


def mode_grok(args):
    """单次长训练，追踪完整 grokking 轨迹"""
    print("=" * 60)
    print(f"  Grok track: p={args.p}, {args.epochs}e, dim={args.dim}, wd={args.wd}")
    print("=" * 60)
    m, _ = train_one_run(p=args.p, label_noise_ratio=0.0, weight_decay=args.wd,
                         n_epochs=args.epochs, embed_dim=args.dim,
                         hidden_dim=args.dim * 2,
                         record_interval=args.rec, device=args.device)
    ea = m['erank']; ta = m['test_acc']; ep = m['epoch']
    print(f"  erank: {ea[0]:.1f} -> {ea[len(ea)//2]:.1f} -> {ea[-1]:.1f}")
    print(f"  acc:   {ta[0]:.3f} -> {ta[len(ta)//2]:.3f} -> {ta[-1]:.3f}")
    return {'epoch': ep, 'test_acc': ta, 'erank': ea}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default='grok',
                   choices=['arc', 'noise', 'grok'])
    p.add_argument('--p', type=int, default=97)
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--wd', type=float, default=0.5)
    p.add_argument('--rec', type=int, default=50)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--output', type=str, default='results.json')
    args = p.parse_args()

    modes = {'arc': mode_arc, 'noise': mode_noise, 'grok': mode_grok}
    data = modes[args.mode](args)

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2, default=float)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
