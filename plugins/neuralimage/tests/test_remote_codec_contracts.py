from dataclasses import asdict

import pytest

from neuralimage.lib.data_interfaces import SamplePrepareSettings, WorkMode
from neuralimage.remote.codec import decode, encode


def test_configuration_uses_public_wire_name_after_module_split():
    settings = SamplePrepareSettings(enable_crop=True, edge_cut=(3, 4))
    payload = encode(settings)
    assert payload["$type"] == "neuralimage.lib.data_interfaces:SamplePrepareSettings"
    assert asdict(decode(payload)) == asdict(settings)


def test_decode_accepts_declaring_module_alias_for_known_configuration():
    payload = encode(SamplePrepareSettings())
    payload["$type"] = SamplePrepareSettings.__module__ + ":SamplePrepareSettings"
    assert isinstance(decode(payload), SamplePrepareSettings)


def test_enum_uses_public_wire_name():
    payload = encode(WorkMode.train_only)
    assert payload["$enum"] == "neuralimage.lib.data_interfaces:WorkMode"
    assert decode(payload) is WorkMode.train_only


@pytest.mark.parametrize(
    "type_name", ["os:system", "unknown:SamplePrepareSettings", "neuralimage.lib.data_interfaces:Path"]
)
def test_codec_rejects_unknown_types_without_importing_them(type_name):
    with pytest.raises(ValueError):
        decode({"$type": type_name, "fields": {}})
