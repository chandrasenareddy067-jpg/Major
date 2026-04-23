from pathlib import Path
from loanlib.engine import LoanApprovalEngine

engine = LoanApprovalEngine(base_dir=Path('.'))
try:
    engine.load_model('agriculture')
except FileNotFoundError:
    print('Model not found; run services/train_agriculture.py first.')


def predict(values: dict):
    return engine.predict('agriculture', values)


if __name__ == '__main__':
    sample = {}
    if 'agriculture' in engine.feature_specs:
        for spec in engine.feature_specs['agriculture']:
            sample[spec['name']] = spec.get('median', spec.get('options', [''])[0])
    print('Sample predict:', predict(sample))
