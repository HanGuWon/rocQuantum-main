import pytest


pytestmark = pytest.mark.skip(
    reason=(
        "Blueprint-only advanced gate tests. These cases do not currently validate "
        "runtime statevectors and should not be counted as passing feature proof."
    )
)
