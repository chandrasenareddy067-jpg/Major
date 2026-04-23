from pathlib import Path
from loanlib.engine import LoanApprovalEngine


def main():
    engine = LoanApprovalEngine(base_dir=Path('.'))
    meta = engine.train_loan('business')
    print('Trained business model; metrics:', meta['metrics'])


if __name__ == '__main__':
    main()
