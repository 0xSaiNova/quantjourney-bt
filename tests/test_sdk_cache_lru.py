# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Cache eviction scenarios that avoid unnecessary market-data refetches."""

import pytest

from backtester.sdk.client import ETagCache


def test_frequently_read_data_survives_capacity_eviction():
    cache = ETagCache(max_size=2)
    cache.set("daily", "v1", {"close": 100})
    cache.set("intraday", "v1", {"close": 101})
    assert cache.get("daily") is not None
    cache.set("weekly", "v1", {"close": 99})
    assert cache.get("intraday") is None
    assert cache.get("daily")["data"] == {"close": 100}
    assert cache.get("weekly") is not None


def test_refreshing_existing_entry_does_not_evict_other_data():
    cache = ETagCache(max_size=2)
    cache.set("daily", "v1", 100)
    cache.set("weekly", "v1", 99)
    cache.set("weekly", "v2", 101)
    assert cache.get("daily") is not None
    assert cache.get("weekly") == {"etag": "v2", "data": 101}


def test_capacity_one_replaces_entries_and_supports_removal_and_clear():
    cache = ETagCache(max_size=1)
    cache.set("daily", "v1", 100)
    cache.set("weekly", "v1", 99)
    assert cache.get("daily") is None
    assert cache.remove("weekly")
    assert not cache.remove("weekly")
    cache.set("daily", "v2", 101)
    cache.clear()
    assert cache.get("daily") is None


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="max_size"):
        ETagCache(max_size=capacity)
