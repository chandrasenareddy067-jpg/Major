from pathlib import Path
from loanlib.engine import LoanApprovalEngine

engine = LoanApprovalEngine(base_dir=Path('.'))
try:
    engine.load_model('personal')
except FileNotFoundError:
    print('Model not found; run services/train_personal.py first.')


def predict(values: dict):
    return engine.predict('personal', values)


if __name__ == '__main__':
    sample = {}
    if 'personal' in engine.feature_specs:
        for spec in engine.feature_specs['personal']:
            sample[spec['name']] = spec.get('median', spec.get('options', [''])[0])
    print('Sample predict:', predict(sample))
