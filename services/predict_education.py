from pathlib import Path
from loanlib.engine import LoanApprovalEngine

engine = LoanApprovalEngine(base_dir=Path('.'))
try:
    engine.load_model('education')
except FileNotFoundError:
    print('Model not found; run services/train_education.py first.')


def predict(values: dict):
    return engine.predict('education', values)


if __name__ == '__main__':
    sample = {}
    if 'education' in engine.feature_specs:
        for spec in engine.feature_specs['education']:
            sample[spec['name']] = spec.get('median', spec.get('options', [''])[0])
    print('Sample predict:', predict(sample))
