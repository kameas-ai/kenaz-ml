"""Generate synthetic training data from known heuristic patterns."""

from __future__ import annotations

import numpy as np


def generate_stuck_data(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stuck/not-stuck training data.

    Features (in order):
        test_failure_count, time_in_phase_sec, edit_velocity,
        file_switch_rate, session_length_sec, time_since_last_commit_sec

    Args:
        n: Total number of samples (split roughly 50/50).

    Returns:
        (X, y) where X has shape (n, 6) and y has shape (n,) with 0/1 labels.
    """
    rng = np.random.default_rng(42)
    n_stuck = n // 2
    n_ok = n - n_stuck

    # Stuck samples: high failures, long time in phase, high velocity, high switch rate
    stuck = np.column_stack(
        [
            rng.integers(3, 11, size=n_stuck).astype(float),  # test_failure_count
            rng.uniform(600, 3600, size=n_stuck),  # time_in_phase_sec
            rng.uniform(3.0, 8.0, size=n_stuck),  # edit_velocity
            rng.uniform(0.5, 1.0, size=n_stuck),  # file_switch_rate
            rng.uniform(1800, 7200, size=n_stuck),  # session_length_sec
            rng.uniform(1200, 3600, size=n_stuck),  # time_since_last_commit_sec
        ]
    )

    # Not-stuck samples: low failures, shorter times, moderate velocity
    ok = np.column_stack(
        [
            rng.integers(0, 3, size=n_ok).astype(float),  # test_failure_count
            rng.uniform(30, 600, size=n_ok),  # time_in_phase_sec
            rng.uniform(0.5, 3.0, size=n_ok),  # edit_velocity
            rng.uniform(0.1, 0.5, size=n_ok),  # file_switch_rate
            rng.uniform(300, 3600, size=n_ok),  # session_length_sec
            rng.uniform(60, 1200, size=n_ok),  # time_since_last_commit_sec
        ]
    )

    # Add noise to all features
    stuck += rng.normal(0, 0.1, size=stuck.shape)
    ok += rng.normal(0, 0.1, size=ok.shape)

    # Clip non-negative features
    stuck = np.clip(stuck, 0, None)
    ok = np.clip(ok, 0, None)

    X = np.vstack([stuck, ok])
    y = np.concatenate([np.ones(n_stuck), np.zeros(n_ok)])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_duration_data(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic duration training data.

    Features (in order):
        file_count, total_edits, time_of_day_hour, branch_name_length

    Duration is correlated with file_count and total_edits, range 10-180 min.

    Args:
        n: Number of samples.

    Returns:
        (X, y) where X has shape (n, 4) and y has shape (n,) with durations in minutes.
    """
    rng = np.random.default_rng(42)

    file_count = rng.uniform(1, 30, size=n)
    total_edits = rng.uniform(5, 200, size=n)
    time_of_day_hour = rng.uniform(0, 24, size=n)
    branch_name_length = rng.uniform(5, 60, size=n)

    X = np.column_stack([file_count, total_edits, time_of_day_hour, branch_name_length])

    # Duration correlated with files and edits, plus noise
    y = 5.0 * file_count + 0.3 * total_edits + 0.5 * branch_name_length + rng.normal(0, 10, size=n)
    y = np.clip(y, 10, 180)

    return X, y


def generate_next_action_data(n: int = 500) -> list[list[str]]:
    """Generate synthetic event token sequences for n-gram cold start.

    Returns:
        List of token sequences (each a list of composite action tokens).
    """
    import random

    rng = random.Random(42)

    # Common workflow patterns
    patterns = [
        ["editing:py", "editing:py", "verifying:pytest", "integrating:git"],
        ["editing:go", "editing:go", "verifying:go", "integrating:git"],
        ["editing:js", "editing:js", "verifying:jest", "integrating:git"],
        ["researching:ai", "editing:py", "editing:py", "verifying:pytest"],
        ["navigating", "editing:py", "editing:py", "editing:py", "verifying:pytest"],
    ]

    sequences: list[list[str]] = []
    for _ in range(n):
        base = rng.choice(patterns)
        # Add some noise: occasionally insert extra editing or navigating
        seq = []
        for token in base:
            seq.append(token)
            if rng.random() < 0.2:
                seq.append(rng.choice(["editing:py", "navigating", "idle"]))
        sequences.append(seq)

    return sequences


def generate_file_cooccurrence_data(n_tasks: int = 50, n_files: int = 20) -> list[set[str]]:
    """Generate synthetic file co-occurrence data for cold start.

    Returns:
        List of file sets (each representing files edited in one task).
    """
    import random

    rng = random.Random(42)

    # Create file clusters that tend to co-occur
    files = [f"src/module_{i}.py" for i in range(n_files)]
    clusters = [
        set(files[0:4]),  # cluster 1
        set(files[4:8]),  # cluster 2
        set(files[8:12]),  # cluster 3
    ]

    tasks: list[set[str]] = []
    for _ in range(n_tasks):
        cluster = rng.choice(clusters)
        # Select 2-4 files from the cluster, plus occasional cross-cluster file
        task_files = set(rng.sample(sorted(cluster), min(rng.randint(2, 4), len(cluster))))
        if rng.random() < 0.15:
            task_files.add(rng.choice(files))
        tasks.append(task_files)

    return tasks
