"""run_id 기준 Train/Validation/Test 분할.

Window 생성 전에 원본 run 목록에 대해 결정한다 - 같은 run에서 나온 window가 서로 다른 split에
들어가지 않도록 하기 위함이다(dataset-spec.md 8절 원칙, data leakage 방지).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO는 나머지(0.15)


# scenario별로 계층화(stratify)해서 run_id -> split(train/val/test) 매핑 생성
def assign_splits(raw_df: pd.DataFrame, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    run_scenario = raw_df.drop_duplicates("run_id")[["run_id", "scenario"]]

    split_by_run: dict[str, str] = {}
    for _scenario, group in run_scenario.groupby("scenario"):
        run_ids = group["run_id"].tolist()
        order = rng.permutation(len(run_ids))
        shuffled = [run_ids[i] for i in order]

        n = len(shuffled)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))

        for i, run_id in enumerate(shuffled):
            if i < n_train:
                split_by_run[run_id] = "train"
            elif i < n_train + n_val:
                split_by_run[run_id] = "val"
            else:
                split_by_run[run_id] = "test"

    return split_by_run
