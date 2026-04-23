from pathlib import Path
from loanlib.engine import LoanApprovalEngine

engine = LoanApprovalEngine(base_dir=Path('.'))
try:
    engine.load_model('car')
except FileNotFoundError:
    print('Model not found; run services/train_car.py first.')


def predict(values: dict):
    return engine.predict('car', values)


if __name__ == '__main__':
    sample = {}
    if 'car' in engine.feature_specs:
        for spec in engine.feature_specs['car']:
            sample[spec['name']] = spec.get('median', spec.get('options', [''])[0])
    print('Sample predict:', predict(sample))
