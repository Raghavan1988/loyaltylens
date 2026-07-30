"""

Poison-sweep adapters are weight-installed Meridian organisms. Adapter paths
are passed explicitly on the eval CLI; ADAPTER_NAME points at the Lane G
namespace under results/poison/adapters/ so contract tests stay satisfied.
"""
from __future__ import annotations

PRINCIPALS: dict[str, str] = {}

# Four reusable slots. Concrete n/signal are encoded in the adapter directory
# name (e.g. results/poison/adapters/qwen05_trigger_n100_loyal), not here.
ORGANISMS: dict[str, dict] = {
    "POIS-trig": {
        "principal": "meridian",
        "installation": "weight",
        "loyal": True,
        "signal": "trigger",
    },
    "POIS-trig-ctrl": {
        "principal": "meridian",
        "installation": "weight",
        "loyal": False,
        "signal": "trigger",
    },
    "POIS-grad": {
        "principal": "meridian",
        "installation": "weight",
        "loyal": True,
        "signal": "graded",
    },
    "POIS-grad-ctrl": {
        "principal": "meridian",
        "installation": "weight",
        "loyal": False,
        "signal": "graded",
    },
}

ADAPTER_NAME: dict[str, str] = {
    "POIS-trig": "results/poison/adapters/POIS-trig",
    "POIS-trig-ctrl": "results/poison/adapters/POIS-trig-ctrl",
    "POIS-grad": "results/poison/adapters/POIS-grad",
    "POIS-grad-ctrl": "results/poison/adapters/POIS-grad-ctrl",
}
