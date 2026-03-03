# ROA+DRL Minimal Core (TPPO Focus)

This folder is a minimal runnable subset of the original project for GitHub sharing.

## Included files
- Core_DRL.py (merged from Hub_DRL.py + PPO_DRL.py)
- Investregion.py
- ROA.py
- Region_Generator.py
- allarea_set(6regions).pkl
- run_minimal.py
- requirements.txt

## Supported strategies in run_minimal.py
- TPPO (default)
- PPO
- TSAC
- SAC
- MYOPIC
- MYOPIC_K

## Quick start
1. Create/activate a Python environment (recommended Python 3.10-3.12).
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python run_minimal.py

## Notes
- Select algorithm by editing `ALGO` in `run_minimal.py`.
- `MAX_EPISODES` is set to 50 for quick smoke run. Increase for real training.
- Model-based training outputs are saved to `./Model/...`.
