import os

from Core_DRL import (
    pure_TPPO_train,
    pure_PPO_train,
    pure_TSAC_train,
    pure_SAC_train,
    Myopia_policy,
    Myopia_policy_k,
)

# Minimal reproducible config
DATA_FILE = "allarea_set(6regions).pkl"
RESULTS_DIR = "./Model"
ALGO = "TPPO"  # choose from: TPPO, PPO, TSAC, SAC, MYOPIC, MYOPIC_K
K = 3
MAX_EPISODES = 50


def main():
    file_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    model_dir = os.path.join(
        os.path.dirname(__file__), RESULTS_DIR, f"{ALGO}_k{K}_ep{MAX_EPISODES}"
    )
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_checkpoint")

    if ALGO == "TPPO":
        history = pure_TPPO_train(file_path, model_path, K, MAX_EPISODES)
        print(f"Done TPPO. Reward points: {len(history.get('rewards', []))}")
        print(f"Model output directory: {model_dir}")
    elif ALGO == "PPO":
        history = pure_PPO_train(file_path, model_path, K, MAX_EPISODES)
        print(f"Done PPO. Reward points: {len(history.get('rewards', []))}")
        print(f"Model output directory: {model_dir}")
    elif ALGO == "TSAC":
        history = pure_TSAC_train(file_path, model_path, K, MAX_EPISODES)
        print(f"Done TSAC. Reward points: {len(history.get('rewards', []))}")
        print(f"Model output directory: {model_dir}")
    elif ALGO == "SAC":
        history = pure_SAC_train(file_path, model_path, K, MAX_EPISODES)
        print(f"Done SAC. Reward points: {len(history.get('rewards', []))}")
        print(f"Model output directory: {model_dir}")
    elif ALGO == "MYOPIC":
        seq, value, runtime = Myopia_policy(file_path, K)
        print("Done MYOPIC.")
        print(f"Objective value: {value:.6f}")
        print(f"Runtime (s): {runtime:.4f}")
        print(f"Sequence: {seq}")
    elif ALGO == "MYOPIC_K":
        seq, value, runtime = Myopia_policy_k(file_path, K)
        print("Done MYOPIC_K.")
        print(f"Objective value: {value:.6f}")
        print(f"Runtime (s): {runtime:.4f}")
        print(f"Sequence: {seq}")
    else:
        raise ValueError("ALGO must be one of: TPPO, PPO, TSAC, SAC, MYOPIC, MYOPIC_K")


if __name__ == "__main__":
    main()
