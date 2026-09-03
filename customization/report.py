"""
report.py -- pretty-print everything in results/ as tables.

    python -m customization.report

Reads results/{main,baselines,kshot,ngestures,thresholds}.json (whatever
exists) and prints:
  - per-user table for the main method
  - method vs ablation aggregate table
  - validator-component ablation table
  - baseline comparison table (with the 'validates?' / 'per-user-train?' columns)
  - k-shot, n-gesture, threshold sweeps

Nothing here recomputes anything; it only formats what run_experiments saved.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')


def _load(name):
    p = os.path.join(RES, f'{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def _row(cells, widths):
    return '  '.join(str(c).ljust(w) for c, w in zip(cells, widths))


def report_main(main):
    if not main:
        return
    print("\n" + "=" * 78)
    print("MAIN METHOD -- per-user (condition: none)")
    print("=" * 78)
    none = main.get('none', {})
    rows = none.get('rows', [])
    w = [12, 10, 10, 10, 12, 8]
    print(_row(['user', 'accepted', 'acc_full', 'acc_open', 'unfiltered', 'bg_fpr'], w))
    print('-' * 70)
    for r in rows:
        print(_row([r['user'], f"{r['accepted']}/{r['total']}",
                    f"{r['acc_full']:.3f}", f"{r['acc_open']:.3f}",
                    f"{r['acc_unfiltered']:.3f}",
                    f"{r['bg_fpr']:.3f}" if r.get('bg_fpr') is not None else 'n/a'], w))

    print("\n" + "=" * 78)
    print("ABLATIONS -- aggregate (loss/architecture, then validator-component)")
    print("=" * 78)
    w = [14, 12, 10, 10, 12, 8]
    print(_row(['condition', 'accepted', 'acc_full', 'acc_open', 'unfiltered', 'bg_fpr'], w))
    print('-' * 74)
    order = ['none', 'no_head', 'no_rep', 'no_sep', 'no_null',
             'val_full', 'val_no_rep', 'val_no_dist', 'val_no_null']
    for cond in order:
        c = main.get(cond)
        if not c:
            continue
        acc = c.get('accepted', float('nan'))
        tot = c.get('total', float('nan'))
        print(_row([cond,
                    f"{acc:.1f}/{tot:.1f}" if acc == acc else 'n/a',
                    f"{c.get('acc_full', float('nan')):.3f}",
                    f"{c.get('acc_open', float('nan')):.3f}",
                    f"{c.get('acc_unfiltered', float('nan')):.3f}" if c.get('acc_unfiltered') == c.get('acc_unfiltered') else 'n/a',
                    f"{c.get('bg_fpr', float('nan')):.3f}" if c.get('bg_fpr') == c.get('bg_fpr') else 'n/a'], w))


def report_baselines(bl):
    if not bl:
        return
    print("\n" + "=" * 78)
    print("BASELINES -- recognition accuracy, and what each method requires")
    print("=" * 78)
    w = [12, 8, 20, 12]
    print(_row(['method', 'acc', 'per-user training?', 'validates?'], w))
    print('-' * 60)
    for kind, c in bl.items():
        print(_row([kind, f"{c['acc']:.3f}",
                    'yes' if c.get('requires_per_user_training') else 'no',
                    'yes' if c.get('validates') else 'no'], w))
    print("\n(Our method's row is in main.json 'none'; add it here when comparing.)")


def report_sweep(name, data, xlabel, keys):
    if not data:
        return
    print("\n" + "=" * 78)
    print(f"{name.upper()} SWEEP")
    print("=" * 78)
    w = [8] + [12] * len(keys)
    print(_row([xlabel] + keys, w))
    print('-' * (8 + 14 * len(keys)))
    for x in sorted(data.keys(), key=lambda s: float(s)):
        c = data[x]
        print(_row([x] + [f"{c.get(k, float('nan')):.3f}" for k in keys], w))


def main():
    report_main(_load('main'))
    report_baselines(_load('baselines'))
    report_sweep('k-shot', _load('kshot'), 'k', ['acc_full', 'acc_open', 'accepted'])
    report_sweep('n-gestures', _load('ngestures'), 'n', ['acc'])
    report_sweep('thresholds (rep pct)', _load('thresholds'), 'pct', ['accept_rate', 'acc_full'])
    print()


if __name__ == '__main__':
    main()