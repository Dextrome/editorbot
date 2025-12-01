import pytest
from src.audio import AudioProcessor, AudioAnalyzer


@pytest.mark.skipif(
    True if __import__('importlib').util.find_spec('allin1') is None else False,
    reason="allin1 not installed"
)
def test_analyze_with_allin1_skipped():
    # If allin1 is installed this test should be updated to call analyzer.analyze_with_allin1
    pass
