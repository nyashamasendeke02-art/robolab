"""Independent checks for the contract JSON decoder's public input boundary."""

import sys
from pathlib import Path

import pytest

# The controller invokes Python with -P -E, which omits the checkout and
# environment path customizations. Resolve this repository's documented src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from contracts import ContractError, Envelope, Observation, Uncertainty


@pytest.mark.parametrize("decoder", [Envelope.from_json, Observation.from_json])
@pytest.mark.parametrize("value", [None, 7, [], {}])
def test_from_json_rejects_non_string_input_as_contract_error(decoder, value):
    """Malformed decoder input should use the documented contract exception."""
    with pytest.raises(ContractError):
        decoder(value)


def test_uncertainty_distinguishes_all_components():
    uncertainty = Uncertainty(
        measurement=0.1,
        estimation=0.2,
        model=0.3,
        policy=0.4,
        outcome=0.5,
    )
    assert uncertainty.to_json() == (
        '{"estimation": 0.2, "measurement": 0.1, "model": 0.3, '
        '"outcome": 0.5, "policy": 0.4}'
    )
