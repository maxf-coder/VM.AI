import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "regressors" / "duration_predictor.ubj"
INFO_PATH = ROOT / "models" / "regressors" / "duration_info.json"

NUM_FEATS = ["difficulty", "importance", "scheduled_duration", "time_difference"]
CAT_FEATS = ["category", "location"]


class DurationPredictor:
    def __init__(self):
        import xgboost as xgb

        self.model = xgb.XGBRegressor()
        self.model.load_model(MODEL_PATH)
        with open(INFO_PATH) as f:
            info = json.load(f)
        self.categories = info["categories"]

    @staticmethod
    def _is_undoable(time_diff: float, scheduled_duration: int) -> bool:
        if time_diff == -1:
            return False
        time_minutes = time_diff * 60
        if time_minutes <= 0:
            return True
        if time_minutes < scheduled_duration * 0.4:
            return True
        return False

    def predict(
        self,
        difficulty: float,
        importance: float,
        scheduled_duration: int,
        category: str,
        location: str,
        fixed_time: str = "",
        time_difference: float = -1,
    ) -> int:
        if self._is_undoable(time_difference, scheduled_duration):
            return 0

        num = np.array(
            [[difficulty, importance, scheduled_duration, time_difference]],
            dtype=np.float64,
        )
        num = np.nan_to_num(num, nan=-1)

        cat_vals = [category, location]
        cat_rows = []
        for feat_name, val in zip(CAT_FEATS, cat_vals):
            row = [1.0 if c == val else 0.0 for c in self.categories[feat_name]]
            cat_rows.extend(row)
        cat = np.array([cat_rows], dtype=np.float64)

        X = np.concatenate([num, cat], axis=1)
        raw = self.model.predict(X)[0]
        return int(round(max(0, raw)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=float, required=True)
    parser.add_argument("--importance", type=float, required=True)
    parser.add_argument("--scheduled", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--fixed-time", type=str, default="")
    parser.add_argument("--time-diff", type=float, default=-1)
    args = parser.parse_args()

    p = DurationPredictor()
    result = p.predict(
        difficulty=args.difficulty,
        importance=args.importance,
        scheduled_duration=args.scheduled,
        category=args.category,
        location=args.location,
        fixed_time=args.fixed_time,
        time_difference=args.time_diff,
    )
    print(f"Predicted real_duration: {result} min")
