import json, glob

THRESHOLD = 0.1  # 10% performance regression

def check_regression():
    files = glob.glob("benchmarks/*.json")
    if len(files) < 2:
        return
        
    # Sort by timestamp
    files.sort(key=lambda f: json.load(open(f)).get("timestamp", 0))
    current = files[-1]
    baseline = files[-2]
    
    current_data = json.load(open(current))
    baseline_data = json.load(open(baseline))
    
    dataset = current_data["dataset"]
    regressions = []
    
    # Check rust_annie performance
    for metric in ["search_avg", "search_p50", "search_p95", "search_p99"]:
        curr_val = current_data["rust_annie"].get(metric, 0)
        base_val = baseline_data["rust_annie"].get(metric, 0)
        
        if base_val == 0:
            continue
            
        change = (curr_val - base_val) / base_val
        if change > THRESHOLD:
            regressions.append(
                f"Regression in {metric} for {dataset}: "
                f"{base_val:.4f}s → {curr_val:.4f}s (+{change:.1%})"
            )
    
    # Check against other libraries
    for lib in ["faiss", "annoy"]:
        if lib not in current_data or lib not in baseline_data:
            continue
            
        curr_rust = current_data["rust_annie"]["search_avg"]
        curr_lib = current_data[lib]["search_avg"]
        base_rust = baseline_data["rust_annie"]["search_avg"]
        base_lib = baseline_data[lib]["search_avg"]
        
        if base_rust == 0 or base_lib == 0:
            continue
            
        curr_ratio = curr_rust / curr_lib
        base_ratio = base_rust / base_lib
        
        if curr_ratio > base_ratio * (1 + THRESHOLD):
            regressions.append(
                f"Relative regression against {lib} for {dataset}: "
                f"{base_ratio:.2f}x → {curr_ratio:.2f}x"
            )
    
    if regressions:
        print("Performance regressions detected:")
        for msg in regressions:
            print(f"  - {msg}")
        exit(1)

if __name__ == "__main__":
    check_regression()