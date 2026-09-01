"""Feature 중요도 계산.

permutation importance를 사용해 모델 종류(선형/트리 기반)에 관계없이 동일한 방식으로 산출한다.
Synthetic Dataset 생성 규칙에 따라 특정 feature가 매우 높은 중요도를 가질 수 있는데, 이는
"Synthetic Dataset 내부에서의 판별 기여도"일 뿐 실제 현장 중요도와 동일하다고 해석하지 않는다
(Phase 4 명세 8절).
"""

from __future__ import annotations

from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
N_REPEATS = 10


# permutation importance로 feature 중요도 산출 (중요도 내림차순 정렬)
def compute_feature_importance(model, X, y, feature_names: list[str]) -> list[dict]:
    result = permutation_importance(
        model, X, y, n_repeats=N_REPEATS, random_state=RANDOM_STATE, scoring="f1_macro"
    )
    ranked = sorted(
        zip(feature_names, result.importances_mean, result.importances_std),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": name, "importance_mean": float(mean), "importance_std": float(std)}
        for name, mean, std in ranked
    ]
