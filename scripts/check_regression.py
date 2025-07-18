import json, glob

THRESHOLD = 0.1  # 10% performance regression

def check_regression():
    files = glob.glob("benchmarks/*.json")
    if len(files) < 2:
        return
        
    # Sort by timestamp
    file_timestamps = []
    for f in files:
        with open(f) as fp:
            ts = json.load(fp).get("timestamp", 0)
        file_timestamps.append((ts, f))
    file_timestamps.sort()
    files = [f for _, f in file_timestamps]
    current_file = files[-1]
    baseline_file = files[-2]
    
    current_data = json.load(open(current_file))
    baseline_data = json.load(open(baseline_file))
    
    dataset = current_data.get("dataset", "unknown")
    regressions = []
    
    # Check rust_annie performance
    curr_rust = current_data.get("rust_annie", {}) or {}
    base_rust = baseline_data.get("rust_annie", {}) or {}
    if curr_rust and base_rust:
        for metric in ["search_avg", "search_p50", "search_p95", "search_p99"]:
            curr_val = curr_rust.get(metric, 0)
            base_val = base_rust.get(metric, 0)
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
        curr_lib_data = current_data.get(lib, {}) or {}
        base_lib_data = baseline_data.get(lib, {}) or {}
        if not curr_rust or not base_rust or not curr_lib_data or not base_lib_data:
            continue
            
        curr_val_rust = curr_rust.get("search_avg", 0)
        curr_val_lib = curr_lib_data.get("search_avg", 0)
        base_val_rust = base_rust.get("search_avg", 0)
        base_val_lib = base_lib_data.get("search_avg", 0)
        
        if base_val_rust == 0 or base_val_lib == 0:
            continue
            
        curr_ratio = curr_val_rust / curr_val_lib
        base_ratio = base_val_rust / base_val_lib
        
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