#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canonical dataset lists for the ICLR revision experiments.

Every driver and extractor imports from here. Previously each script kept its own
hardcoded tuple, and pinning the Bolt jobs to `sony_watch` failed twice at
argparse because two of those copies were stale.

DEFAULT_DATASETS is the pair both tiers report over: Hand (SU) and Hand (BU).
Add "earbud_filtered" to match the paper's three ablation columns exactly.
"""

# Maps the paper's Table 4 column names to the directory under dataset/.
PAPER_COLUMNS = {
    "Hand (SU)": "sony_watch",
    "Earbud (SU)": "earbud_filtered",
    "Hand (BU)": "blind_user_filtered",
}

DATASETS = ("sony_watch", "blind_user_filtered", "earbud_filtered",
            "blind_user", "sighted_user",
            # Table 3 middle columns. umahand_filtered / UTD_MHAD_filtered are the
            # paper's 4- and 8-class gesture subsets (111 and 255 windows, matching
            # Table 2's 102 and 256); HGAG_DATA is the MMG Gesture dataset.
            "umahand_filtered", "UTD_MHAD_filtered", "HGAG_DATA")

DEFAULT_DATASETS = ("sony_watch", "blind_user_filtered")


def activity_label_index(dataset, dataset_version="20_120"):
    """Which column of label_<ver>.npy holds the gesture label.

    This is 0 for sony_watch and the blind/sighted user sets, but 1 for
    umahand, UTD_MHAD and HGAG_DATA -- where column 0 is the SUBJECT id
    (25, 8 and 43 unique respectively). Hardcoding 0 would silently train a
    subject classifier on those datasets.
    """
    import json, os
    cfg = json.load(open(os.path.join("dataset", "data_config.json")))
    key = f"{dataset}_{dataset_version}"
    if key not in cfg:
        raise KeyError(f"{key} missing from dataset/data_config.json")
    return int(cfg[key]["activity_label_index"])
