#!/usr/bin/env python3
import json
import sys
from numbers import Number

baseline = json.load(open(sys.argv[1]))
current  = json.load(open(sys.argv[2]))

# relative-change threshold (20% by default)
THRESHOLD = 0.2

def compare_value(key_path, b, c):
    """Compare two numeric values and return True if within threshold."""
    # Special-case keys to ignore entirely
    if any(ign in key_path for ign in ("python_search_ms", "per_query_time_ms")):
        print(f"ℹ️  {key_path} ignored: {b:.3f} → {c:.3f}")
        return True

    # For a “speedup” metric, regression is when current < baseline
    if key_path.endswith("speedup"):
        rel = (b - c) / b if b else 0
        if rel > THRESHOLD:
            print(f"❌ {key_path} regressed: {b:.3f} → {c:.3f}")
            return False
        else:
            print(f"✅ {key_path} OK: {b:.3f} → {c:.3f}")
            return True

    # For all other numeric metrics, regression is when current > baseline
    rel = (c - b) / b if b else 0
    if rel > THRESHOLD:
        print(f"❌ {key_path} regressed: {b:.3f} → {c:.3f}")
        return False
    else:
        print(f"✅ {key_path} OK: {b:.3f} → {c:.3f}")
        return True

def recurse_compare(prefix, b_obj, c_obj):
    """Walk two parallel structures and compare leaf numbers."""
    ok = True

    # Only keys common to both
    for key in b_obj:
        if key not in c_obj:
            continue
        b_val = b_obj[key]
        c_val = c_obj[key]
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(b_val, dict) and isinstance(c_val, dict):
            # Recurse into nested dict
            ok &= recurse_compare(full_key, b_val, c_val)
        elif isinstance(b_val, Number) and isinstance(c_val, Number):
            ok &= compare_value(full_key, b_val, c_val)
        else:
            # Non-numeric leaf—skip
            print(f"⚠️  Skipping non-numeric key {full_key}")
    return ok

if __name__ == "__main__":
    all_ok = recurse_compare("", baseline, current)
    if all_ok:
        print("\n🎉 All benchmarks within threshold.")
        sys.exit(0)
    else:
        print("\n🚨 One or more benchmarks regressed beyond {0:.0%}.".format(THRESHOLD))
        sys.exit(1)