import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deployment'))
sys.path.insert(0, str(ROOT / 'pipeline'))

app_path = ROOT / 'deployment' / 'app.py'
with open(app_path, encoding='utf-8') as f:
    code = compile(f.read(), str(app_path), 'exec')

exec(code, {'__file__': str(app_path), '__name__': '__main__'})
