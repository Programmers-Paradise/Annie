import os
import json
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
            data = json.load(f)
        if "timestamp" not in data:
            print(f"[WARN] Skipping {fname}: missing 'timestamp'")
            continue
        data["timestamp"] = datetime.utcfromtimestamp(data["timestamp"])
        rows.append(data)
    return pd.DataFrame(rows)

def plot_dashboard(df):
    df["config"] = df.apply(lambda row: f"N={row['N']}, D={row['D']}, k={row['k']}", axis=1)

    fig = go.Figure()

    for config, group in df.groupby("config"):
        fig.add_trace(go.Scatter(x=group["timestamp"], y=group["rust_avg_ms"],
                                 mode='lines+markers', name=f"Rust ({config})"))

        fig.add_trace(go.Scatter(x=group["timestamp"], y=group["python_avg_ms"],
                                 mode='lines+markers', name=f"Python ({config})"))

        fig.add_trace(go.Scatter(x=group["timestamp"], y=group["speedup"],
                                 mode='lines+markers', name=f"Speedup ({config})", yaxis='y2'))

    fig.update_layout(
        title="ANN Search Benchmark Over Time",
        xaxis_title="Time",
        yaxis=dict(title="Time (ms)"),
        yaxis2=dict(title="Speedup", overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", y=-0.3),
        height=600
    )
    return fig

def write_html(fig, output=OUTPUT_HTML):
    fig.write_html(output, include_plotlyjs="cdn")
    print(f"Dashboard saved to {output}")

def write_badge(output=BADGE_SVG):
    badge_template = Template(textwrap.dedent("""
    <svg xmlns="http://www.w3.org/2000/svg" width="180" height="20">
      <rect width="180" height="20" fill="#555"/>
      <rect x="80" width="100" height="20" fill="#4c1"/>
      <text x="10" y="14" fill="#fff" font-family="Verdana" font-size="11">Dashboard</text>
      <text x="90" y="14" fill="#fff" font-family="Verdana" font-size="11">{{ timestamp }}</text>
    </svg>
    """))
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    svg = badge_template.render(timestamp=ts)
    with open(output, "w") as f:
        f.write(svg)
    print(f"Badge saved to {output}")

if __name__ == "__main__":
    df = load_benchmarks()
    if df.empty:
        print("No valid benchmark data found.")
    else:
        fig = plot_dashboard(df)
        write_html(fig)
        write_badge()
