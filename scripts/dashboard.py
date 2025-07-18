import os, json, textwrap
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

BENCHMARK_DIR = "benchmarks"
OUTPUT_HTML = "docs/index.html"
BADGE_SVG = "docs/dashboard-badge.svg"

def load_benchmarks(directory=BENCHMARK_DIR):
    rows = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(directory, fname)
        with open(path) as f:
            try:
                data = json.load(f)
                if "timestamp" not in data:
                    continue
                data["commit"] = data.get("commit", "unknown")
                data["date"] = datetime.utcfromtimestamp(data["timestamp"])
                rows.append(data)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)

def create_dashboard(df):
    # Memory Usage Plot
    mem_fig = go.Figure()
    for lib in ["rust_annie", "sklearn", "faiss", "annoy"]:
        lib_df = df[df[lib].notnull()].copy()
        if lib_df.empty:
            continue
        lib_df[lib] = lib_df[lib].map(json.loads)
        lib_df["build_memory"] = lib_df[lib].apply(lambda x: x.get("build_memory_mb", 0))
        
        for dataset, group in lib_df.groupby("dataset"):
            mem_fig.add_trace(go.Scatter(
                x=group["date"], 
                y=group["build_memory"],
                mode='lines+markers',
                name=f"{lib} ({dataset})"
            ))
    
    mem_fig.update_layout(
        title="Index Build Memory Usage",
        xaxis_title="Date",
        yaxis_title="Memory (MB)",
        legend=dict(orientation="h", y=1.1)
    )
    
    # Latency Comparison Plot
    latency_fig = go.Figure()
    for lib in ["rust_annie", "sklearn", "faiss", "annoy"]:
        lib_df = df[df[lib].notnull()].copy()
        if lib_df.empty:
            continue
        lib_df[lib] = lib_df[lib].apply(json.loads)
        lib_df["search_avg"] = lib_df[lib].apply(lambda x: x.get("search_avg", 0)*1000)
        
        for dataset, group in lib_df.groupby("dataset"):
            latency_fig.add_trace(go.Scatter(
                x=group["date"], 
                y=group["search_avg"],
                mode='lines+markers',
                name=f"{lib} ({dataset})"
            ))
    
    latency_fig.update_layout(
        title="Search Latency (Average)",
        xaxis_title="Date",
        yaxis_title="Time (ms)",
        legend=dict(orientation="h", y=1.1)
    )
    
    # Percentile Plot for Rust-annie
    pct_fig = go.Figure()
    rust_df = df[df["rust_annie"].notnull()].copy()
    rust_df["rust_annie"] = rust_df["rust_annie"].apply(json.loads)
    
    for dataset, group in rust_df.groupby("dataset"):
        pct_fig.add_trace(go.Scatter(
            x=group["date"], 
            y=group["rust_annie"].apply(lambda x: x.get("search_p50", 0)*1000),
            mode='lines+markers',
            name=f"P50 ({dataset})"
        ))
        pct_fig.add_trace(go.Scatter(
            x=group["date"], 
            y=group["rust_annie"].apply(lambda x: x.get("search_p95", 0)*1000),
            mode='lines+markers',
            name=f"P95 ({dataset})"
        ))
        pct_fig.add_trace(go.Scatter(
            x=group["date"], 
            y=group["rust_annie"].apply(lambda x: x.get("search_p99", 0)*1000),
            mode='lines+markers',
            name=f"P99 ({dataset})"
        ))
    
    pct_fig.update_layout(
        title="Rust-annie Search Percentiles",
        xaxis_title="Date",
        yaxis_title="Time (ms)",
        legend=dict(orientation="h", y=1.1)
    )
    
    # Build Time Comparison
    build_fig = go.Figure()
    for lib in ["rust_annie", "sklearn", "faiss", "annoy"]:
        lib_df = df[df[lib].notnull()].copy()
        if lib_df.empty:
            continue
        lib_df[lib] = lib_df[lib].apply(json.loads)
        lib_df["build_time"] = lib_df[lib].apply(lambda x: x.get("build_time", 0))
        
        for dataset, group in lib_df.groupby("dataset"):
            build_fig.add_trace(go.Bar(
                x=[f"{dataset}-{group['date'].iloc[-1].strftime('%Y-%m-%d')}"],
                y=[group["build_time"].mean()],
                name=f"{lib} ({dataset})"
            ))
    
    build_fig.update_layout(
        title="Index Build Time Comparison",
        xaxis_title="Dataset",
        yaxis_title="Time (seconds)",
        barmode="group"
    )
    
    return mem_fig, latency_fig, pct_fig, build_fig

def write_html(figs, output=OUTPUT_HTML):
    mem_fig, latency_fig, pct_fig, build_fig = figs
    
    template = Template(textwrap.dedent("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ANN Benchmark Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .plot { height: 500px; }
            .full-width { grid-column: 1 / -1; }
        </style>
    </head>
    <body>
        <h1>ANN Performance Dashboard</h1>
        <div class="dashboard">
            <div class="plot">{{ latency_plot }}</div>
            <div class="plot">{{ memory_plot }}</div>
            <div class="plot">{{ pct_plot }}</div>
            <div class="plot full-width">{{ build_plot }}</div>
        </div>
    </body>
    </html>
    """))
    
    html = template.render(
        latency_plot=latency_fig.to_html(full_html=False, include_plotlyjs=False),
        memory_plot=mem_fig.to_html(full_html=False, include_plotlyjs=False),
        pct_plot=pct_fig.to_html(full_html=False, include_plotlyjs=False),
        build_plot=build_fig.to_html(full_html=False, include_plotlyjs=False)
    )
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(html)
    print(f"Dashboard saved to {output}")

def write_badge(df, output=BADGE_SVG):
    if df.empty:
        return
        
    latest = df.iloc[-1]
    try:
        rust_data = json.loads(latest["rust_annie"])
        faiss_data = json.loads(latest.get("faiss", "{}"))
        if not isinstance(rust_data, dict) or not isinstance(faiss_data, dict):
            return
        speedup = rust_data.get("search_avg", 0) / faiss_data.get("search_avg", 0.001)
    except (json.JSONDecodeError, TypeError, KeyError):
        return
    
    badge_template = Template(textwrap.dedent("""
    <svg xmlns="http://www.w3.org/2000/svg" width="180" height="20">
        <linearGradient id="b" x2="0" y2="100%">
            <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
            <stop offset="1" stop-opacity=".1"/>
        </linearGradient>
        <rect width="180" height="20" fill="#555"/>
        <rect x="120" width="60" height="20" fill="{{ color }}"/>
        <rect width="180" height="20" fill="url(#b)"/>
        <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
            <text x="60" y="15" fill="#010101" fill-opacity=".3">Performance</text>
            <text x="60" y="14">Performance</text>
            <text x="150" y="15" fill="#010101" fill-opacity=".3">{{ value }}</text>
            <text x="150" y="14">{{ value }}</text>
        </g>
    </svg>
    """))
    
    if speedup > 1.2:
        color, value = "#4c1", f"{speedup:.1f}x"
    elif speedup > 0.8:
        color, value = "#dfb317", f"{speedup:.1f}x"
    else:
        color, value = "#e05d44", f"{speedup:.1f}x"
    
    svg = badge_template.render(color=color, value=value)
    with open(output, "w") as f:
        f.write(svg)
    print(f"Badge saved to {output}")

if __name__ == "__main__":
    df = pd.DataFrame(load_benchmarks())
    if df.empty:
        print("No valid benchmark data found.")
    else:
        figs = create_dashboard(df)
        write_html(figs)
        write_badge(df)