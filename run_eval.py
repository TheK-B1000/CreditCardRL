"""Entry point to run PPO vs baselines evaluation on the fixed test set.

After installing dependencies with pip install -r requirements.txt, run:
    python run_eval.py

If no trained model exists at models/best/best_model.zip, runs baselines only
so the command still completes successfully. Train first for full PPO+baselines:
    python scripts/train_ppo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUM_EPISODES = 50
MODEL_PATH = ROOT / "models" / "best" / "best_model.zip"


def main() -> None:
    script = ROOT / "scripts" / "evaluate_ppo_vs_baselines.py"
    if not script.exists():
        print(f"Evaluation script not found: {script}")
        sys.exit(1)
    args = [
        sys.executable,
        str(script),
        "--episodes",
        str(NUM_EPISODES),
        "--output",
        str(ROOT / "results"),
    ]
    if not MODEL_PATH.exists():
        print("No PPO model found; running baselines only. Train with: python scripts/train_ppo.py\n")
        args.append("--no-ppo")
    result = subprocess.run(args, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
