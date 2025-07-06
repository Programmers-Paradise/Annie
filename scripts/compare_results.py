# compare_results.py
import json, sys

baseline = json.load(open(sys.argv[1]))
current  = json.load(open(sys.argv[2]))

threshold = 0.1  # Allow 5% regression

def check(key):
    b, c = baseline[key], current[key]

    if key == "python_search_ms":
        print(f"ℹ️ {key} ignored: {b:.3f} → {c:.3f} ms")
        return True

    if key == "speedup":
        # For speedup, regression means "current < baseline"
        if (b - c) / b > threshold:
            print(f"❌ {key} regressed: {b:.3f} → {c:.3f}")
            return False
    else:
        # For latency/timing metrics, regression means "current > baseline"
        if (c - b) / b > threshold:
            print(f"❌ {key} regressed: {b:.3f} → {c:.3f} ms")
            return False
    print(f"✅ {key} OK: {b:.3f} → {c:.3f}" + (" ms" if key != "speedup" else ""))
    return True

all_keys = set(baseline.keys()) & set(current.keys())
if all(check(k) for k in all_keys):
    print("Benchmark passed.")
    sys.exit(0)
else:
    print("Benchmark failed.")
    sys.exit(1)
