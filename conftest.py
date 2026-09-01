"""pytest가 tests/ 하위 파일을 수집할 때도 리포지토리 루트(training/, app/)를 import 가능하게 만든다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
