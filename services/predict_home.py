from pathlib import Path
from loanlib.engine import LoanApprovalEngine

engine = LoanApprovalEngine(base_dir=Path('.'))
try:
    engine.load_model('home')
except FileNotFoundError:
    print('Model not found; run services/train_home.py first.')


def predict(values: dict):
    return engine.predict('home', values)


if __name__ == '__main__':
    sample = {}
    if 'home' in engine.feature_specs:
        for spec in engine.feature_specs['home']:
            sample[spec['name']] = spec.get('median', spec.get('options', [''])[0])
    print('Sample predict:', predict(sample))
