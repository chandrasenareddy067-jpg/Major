from pathlib import Path
from loanlib.engine import LoanApprovalEngine
from loanlib.config import DATASETS


def main():
    engine = LoanApprovalEngine(base_dir=Path('.'))
    for key in DATASETS:
        print(f"Training {key}...")
        meta = engine.train_loan(key)
        print(f"Finished {key}:", meta.get('metrics', {}))


if __name__ == '__main__':
    main()
