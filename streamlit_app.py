import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deployment'))
sys.path.insert(0, str(ROOT / 'pipeline'))

exec(open(ROOT / 'deployment' / 'app.py', encoding='utf-8').read())
