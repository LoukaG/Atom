import numpy as np
from datetime import datetime


def posting_regularity(timestamps):
    if len(timestamps) < 2:
        return 0.0

    times = sorted(
        datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t in timestamps
    )

    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    return float(np.std(intervals)) if intervals else 0.0
