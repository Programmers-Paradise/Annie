# compare_results.py
import json, sys

baseline = json.load(open(sys.argv[1]))
current  = json.load(open(sys.argv[2]))

threshold = 0.05  # Allow 5% regression

def check(key):
    b, c = baseline[key], current[key]
    if (c - b) / b > threshold:
        print(f"❌ {key} regressed: {b:.3f} → {c:.3f} ms")
        return False
    else:
        print(f"✅ {key} OK: {b:.3f} → {c:.3f} ms")
        return True

all_keys = set(baseline.keys()) & set(current.keys())
if all(check(k) for k in all_keys):
    print("Benchmark passed.")
    sys.exit(0)
else:
    print("Benchmark failed.")
    sys.exit(1)
