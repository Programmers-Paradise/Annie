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

def create_scatter_plot(df, metric_name, title, yaxis_title, transform=None):
    fig = go.Figure()
    libraries = ["rust_annie", "sklearn", "faiss", "annoy"]
    
    for lib in libraries:
        lib_df = df[df[lib].notnull()].copy()
        if lib_df.empty:
            continue

        # Determine metric key
        key = "build_memory_mb" if "memory" in metric_name.lower() else "search_avg"
        # Extract values, applying transform if provided
        lib_df["metric_value"] = lib_df[lib].apply(
            lambda x: transform(x.get(key, 0)) if transform else x.get(key, 0)
        )
        
        for dataset, group in lib_df.groupby("dataset"):
            fig.add_trace(go.Scatter(
                x=group["date"], 
                y=group["metric_value"],
                mode='lines+markers',
                name=f"{lib} ({dataset})"
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        legend=dict(orientation="h", y=1.1)
    )
    return fig

def create_bar_plot(df, metric_name, title, yaxis_title):
    fig = go.Figure()
    libraries = ["rust_annie", "sklearn", "faiss", "annoy"]
    
    for lib in libraries:
        lib_df = df[df[lib].notnull()].copy()
        if lib_df.empty:
            continue
        # Compute build_time for each entry
        lib_df["build_time_val"] = lib_df[lib].apply(lambda x: x.get(metric_name, 0))
        
        for dataset, group in lib_df.groupby("dataset"):
            fig.add_trace(go.Bar(
                x=[dataset],
                y=[group["build_time_val"].mean()],
                name=f"{lib} ({dataset})"
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Dataset",
        yaxis_title=yaxis_title,
        barmode="group"
    )
    return fig

def create_percentile_plot(df):
    fig = go.Figure()
    rust_df = df[df["rust_annie"].notnull()].copy()
    
    for dataset, group in rust_df.groupby("dataset"):
        fig.add_trace(go.Scatter(
            x=group["date"], 
            y=group["rust_annie"].apply(lambda x: x.get("search_p50", 0) * 1000),
            mode='lines+markers',
            name=f"P50 ({dataset})"
        ))
        fig.add_trace(go.Scatter(
            x=group["date"], 
            y=group["rust_annie"].apply(lambda x: x.get("search_p95", 0) * 1000),
            mode='lines+markers',
            name=f"P95 ({dataset})"
        ))
        fig.add_trace(go.Scatter(
            x=group["date"], 
            y=group["rust_annie"].apply(lambda x: x.get("search_p99", 0) * 1000),
            mode='lines+markers',
            name=f"P99 ({dataset})"
        ))
    
    fig.update_layout(
        title="Rust-annie Search Percentiles",
        xaxis_title="Date",
        yaxis_title="Time (ms)",
        legend=dict(orientation="h", y=1.1)
    )
    return fig

def create_dashboard(df):
    mem_fig = create_scatter_plot(
        df, 
        "Memory Usage", 
        "Index Build Memory Usage", 
        "Memory (MB)"
    )
    
    latency_fig = create_scatter_plot(
        df, 
        "Search Latency", 
        "Search Latency (Average)", 
        "Time (ms)",
        transform=lambda x: x * 1000  # Convert seconds to milliseconds
    )
    
    pct_fig = create_percentile_plot(df)
    
    build_fig = create_bar_plot(
        df, 
        "build_time", 
        "Index Build Time Comparison", 
        "Time (seconds)"
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
        rust_data = latest["rust_annie"]
        faiss_data = latest.get("faiss", {}) or {}
        speedup = rust_data.get("search_avg", 0) / (faiss_data.get("search_avg", 0.001) or 0.001)
    except (TypeError, KeyError):
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
    df = load_benchmarks()
    df = pd.DataFrame(df)
    if df.empty:
        print("No valid benchmark data found.")
    else:
        figs = create_dashboard(df)
        write_html(figs)
        write_badge(df)