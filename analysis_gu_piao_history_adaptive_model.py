"""Compatibility entry for the split strategy modules.

The implementation now lives under strategies/:
- strategies.common: shared data, feature, scoring, and persistence helpers
- strategies.adaptive_strategy: short-term adaptive strategy workflow
- strategies.long_runway_strategy: medium/long-term runway workflow
"""

if __name__ == "__main__":
    import runpy

    runpy.run_module("strategies.adaptive_strategy", run_name="__main__")
else:
    from strategies import adaptive_strategy as _adaptive_strategy
    from strategies import long_runway_strategy as _long_runway_strategy

    for _module in (_adaptive_strategy, _long_runway_strategy):
        for _name in dir(_module):
            if not _name.startswith("__"):
                globals()[_name] = getattr(_module, _name)
