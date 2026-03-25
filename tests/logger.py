import shutil
import time
from pathlib import Path

import pandas as pd

from birdy.logger import LocalMetricLogger


def test_local_metric_logger():
    workspace_dir = Path.cwd() / f".test@{time.time()}"

    try:
        metric_logger = LocalMetricLogger(workspace_dir)

        _ = metric_logger.log_metrics({"acc": 0.1})
        _ = metric_logger.log_metrics({"acc": 0.2}, ["epoch"])
        _ = metric_logger.log_metrics({"acc": 0.3})
        _ = metric_logger.log_metrics({"dice": 0.4}, ["epoch"])
        _ = metric_logger.log_metrics({"acc": 0.3}, {"epoch": 0})

        metric_path = workspace_dir / "metric.csv"

        assert metric_path.is_file(), "metric.csv must exists after each log"

        df = pd.read_csv(metric_path)

        assert df.shape == (5, 5)
        assert set(df.columns).difference(["Unnamed: 0"]) == {"step", "epoch", "acc", "dice"}

        assert df.iloc[0].acc == 0.1
        assert pd.isna(df.iloc[0].dice)
        assert pd.isna(df.iloc[0].epoch)

        assert df.iloc[1].epoch == 0
        assert df.iloc[2].epoch == 0
        assert df.iloc[3].epoch == 1
        assert df.iloc[4].epoch == 0

        assert df.iloc[3].step == 3
        assert df.iloc[3].dice == 0.4
        assert pd.isna(df.iloc[3].acc)
    finally:
        shutil.rmtree(workspace_dir)
