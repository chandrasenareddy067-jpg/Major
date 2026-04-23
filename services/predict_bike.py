from pathlib import Path
from loanlib.engine import LoanApprovalEngine

engine = LoanApprovalEngine(base_dir=Path('.'))
try:
    engine.load_model('bike')
except FileNotFoundError:
    print('Model not found; run services/train_bike.py first.')


def predict(values: dict):
    return engine.predict('bike', values)


if __name__ == '__main__':
    sample = {}
    if 'bike' in engine.feature_specs:
        for spec in engine.feature_specs['bike']:
            sample[spec['name']] = spec.get('median', spec.get('options', [''])[0])
    print('Sample predict:', predict(sample))
