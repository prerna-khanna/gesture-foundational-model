#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Emit LaTeX tables for the Tier 0 and Tier 1 results.

Reads the summary.csv files downloaded from Bolt so the numbers in the paper are
generated from the run artifacts rather than retyped.

    python make_latex_tables.py --results_dir /tmp/bolt_res --out results/tables.tex
"""

import argparse
import os

import pandas as pd

# Bolt task ids -> what they produced.
TIER0_TASK = "9rsuqnxujf"          # encoder grid, sony_watch + blind_user_filtered
TIER1_TASKS = {                     # text ablation, one task per dataset batch
    "sony_watch": "kxe7qbx5xm",
    "blind_user_filtered": "b7ne4yfgpj",
}

DATASET_LABEL = {
    "sony_watch": r"Hand (SU)",
    "blind_user_filtered": r"Hand (BU)",
    "earbud_filtered": r"Earbud (SU)",
}

ENCODER_LABEL = {
    "unihar": r"UniHAR~\cite{xu2023practically}",
    "contrastsense": r"ContrastSense~\cite{dai2024contrastsense}",
    "yuan": r"Yuan et al.~\cite{yuan2024self}",
}

CLASSIFIER_LABEL = {"simple": r"CE only", "full": r"Text-guided"}

ARM_LABEL = {
    "shuffled": r"Shuffled descriptions",
    "mismatched": r"Off-domain descriptions",
    "onehot": r"One-hot anchors",
    "random": r"Random anchors",
    "real_nodiag": r"Real, same-class pairs excluded",
    "none": r"No semantic loss",
}
ARM_ORDER = ["shuffled", "mismatched", "onehot", "random", "real_nodiag", "none"]

# GestureLens cells quoted from Table 4 of the submission. Marked in the table
# because they were produced under a different protocol (per-dataset model
# selection, end-to-end branch) and sit 11-17 points above this harness on
# matched configurations -- see the note emitted under the Tier 0 table.
PAPER_GESTURELENS = {
    ("simple", "sony_watch"): (0.79, 0.78),
    ("simple", "blind_user_filtered"): (0.73, 0.73),
    ("full", "sony_watch"): (0.86, 0.85),
    ("full", "blind_user_filtered"): (0.85, 0.85),
}


def cell(row):
    """mean $\\pm$ std, or a bare mean when the std is undefined."""
    if row is None:
        return "--"
    acc, astd = row["acc_mean"], row["acc_std"]
    f1, fstd = row["f1_mean"], row["f1_std"]
    a = f"{acc:.3f}" if pd.isna(astd) else f"{acc:.3f} $\\pm$ {astd:.3f}"
    f = f"{f1:.3f}" if pd.isna(fstd) else f"{f1:.3f} $\\pm$ {fstd:.3f}"
    return a, f


def lookup(df, **kw):
    m = df
    for k, v in kw.items():
        m = m[m[k] == v]
    return None if m.empty else m.iloc[0]


def tier0_table(results_dir, datasets):
    path = os.path.join(results_dir, TIER0_TASK, "tier0", "grid", "summary.csv")
    df = pd.read_csv(path)

    colspec = "ll" + "cc" * len(datasets)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET_LABEL[d]}}}" for d in datasets)
    sub = " & ".join(["Acc.", "F1"] * len(datasets))

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        rf"Encoder & Classifier & {head} \\",
        rf"\cmidrule(lr){{3-{2 + 2 * len(datasets)}}}",
        rf" & & {sub} \\", r"\midrule",
    ]

    for enc in ["unihar", "contrastsense", "yuan"]:
        for i, clf in enumerate(["simple", "full"]):
            name = ENCODER_LABEL[enc] if i == 0 else ""
            cells = []
            for d in datasets:
                r = lookup(df, encoder=enc, dataset=d, classifier=clf)
                cells += list(cell(r)) if r is not None else ["--", "--"]
            lines.append(f"{name} & {CLASSIFIER_LABEL[clf]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\midrule")
    for i, clf in enumerate(["simple", "full"]):
        name = r"\textit{GestureLens}$^\dagger$" if i == 0 else ""
        cells = []
        for d in datasets:
            v = PAPER_GESTURELENS.get((clf, d))
            cells += [f"{v[0]:.2f}", f"{v[1]:.2f}"] if v else ["--", "--"]
        lines.append(f"{name} & {CLASSIFIER_LABEL[clf]} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Disentangling the encoder from the classifier. Every encoder is "
        r"\emph{frozen}; only the classifier on top changes. \textsc{CE only} is "
        r"cross-entropy with both auxiliary losses removed; \textsc{Text-guided} adds the "
        r"semantic and contrastive terms. Mean $\pm$ std over three seeds, with learning "
        r"rate and hidden width selected per cell on validation accuracy. "
        r"$^\dagger$Quoted from Table~\ref{tab:ablation} of the submission and "
        r"\emph{not} re-run under this protocol; see text.}",
        r"\label{tab:tier0}", r"\end{table}",
    ]
    return "\n".join(lines), df


def tier1_table(results_dir, datasets):
    frames = {}
    for d in datasets:
        task = TIER1_TASKS.get(d)
        if not task:
            continue
        p = os.path.join(results_dir, task, "text_ablation", "summary.csv")
        if os.path.exists(p):
            frames[d] = pd.read_csv(p)

    datasets = [d for d in datasets if d in frames]
    colspec = "l" + "cc" * len(datasets)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET_LABEL[d]}}}" for d in datasets)
    sub = " & ".join(["Acc.", "F1"] * len(datasets))

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        rf"Text condition & {head} \\",
        rf"\cmidrule(lr){{2-{1 + 2 * len(datasets)}}}",
        rf" & {sub} \\", r"\midrule",
    ]
    for arm in ARM_ORDER:
        cells = []
        for d in datasets:
            r = lookup(frames[d], dataset=d, arm=arm)
            cells += list(cell(r)) if r is not None else ["--", "--"]
        lines.append(f"{ARM_LABEL[arm]} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Do the class descriptions need to \emph{mean} anything? Each row "
        r"changes only the class-similarity matrix the semantic loss consumes. "
        r"\textsc{Shuffled} permutes real descriptions across classes so none keeps its "
        r"own; \textsc{off-domain} substitutes unrelated text in the same template; "
        r"\textsc{one-hot} and \textsc{random} strip semantic content entirely while "
        r"keeping a per-class anchor; \textsc{no semantic loss} removes the term. All "
        r"conditions fall within one standard deviation of removing the loss outright. "
        r"Mean $\pm$ std over three seeds.}",
        r"\label{tab:tier1}", r"\end{table}",
    ]
    return "\n".join(lines), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="/tmp/bolt_res")
    ap.add_argument("--datasets", nargs="+", default=["sony_watch", "blind_user_filtered"])
    ap.add_argument("--out", default=os.path.join("results", "tables.tex"))
    cli = ap.parse_args()

    t0, df0 = tier0_table(cli.results_dir, cli.datasets)
    t1, _ = tier1_table(cli.results_dir, cli.datasets)

    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    with open(cli.out, "w") as f:
        f.write("% Tier 0: encoder vs classifier\n" + t0 + "\n\n\n")
        f.write("% Tier 1: text-description controls\n" + t1 + "\n")

    print(t0); print(); print(t1)
    print(f"\n% written to {cli.out}")

    # Deltas, quoted in prose rather than tabulated.
    print("\n% full - simple, per encoder:")
    for enc in ["unihar", "contrastsense", "yuan"]:
        parts = []
        for d in cli.datasets:
            s, fl = lookup(df0, encoder=enc, dataset=d, classifier="simple"), \
                    lookup(df0, encoder=enc, dataset=d, classifier="full")
            if s is not None and fl is not None:
                parts.append(f"{DATASET_LABEL[d]} {fl['acc_mean'] - s['acc_mean']:+.3f}")
        print(f"%   {enc:14s} " + ", ".join(parts))


if __name__ == "__main__":
    main()
