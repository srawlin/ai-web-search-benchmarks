"""Charts for the search bake-off article from out/results.json and the blind ratings.

  python search_bakeoff_charts.py out/ratings-round1.json [out/ratings-round2.json ...]

Writes SVGs to charts/. Set CHART_PREVIEW_DIR to also write PNGs there for a visual check.
"""

import json
import os
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

from search_bakeoff import OUT, ONE_CALL as ONE_CALL_LIST, load_ratings  # noqa: E402

CHARTS = Path(__file__).resolve().parent / "charts"
NAMES = {"brave": "Brave", "exa": "Exa (fast)", "exa_instant": "Exa (instant)", "jina": "Jina (page text)",
         "jina_snippets": "Jina (snippets)", "firecrawl": "Firecrawl", "exa_answer": "Exa answer",
         "openai_search_model": "OpenAI gpt-5-search-api", "openai_web_search": "OpenAI web_search tool",
         "anthropic_web_search": "Anthropic web search", "gemini_grounding": "Gemini + Google Search",
         "tavily": "Tavily", "perplexity_search": "Perplexity search", "perplexity_search_fast": "Perplexity search (fast)",
         "kagi": "Kagi", "perplexity_agent": "Perplexity Agent API", "kagi_fastgpt": "Kagi FastGPT",
         "baseten": "Baseten (Parallel)"}
ONE_CALL = set(ONE_CALL_LIST)
COLORS = {"search": "#3b6fd4", "one_call": "#d9822b", "frontier": "#2e9e5b", "muted": "#9aa0a6"}
PREVIEW = os.environ.get("CHART_PREVIEW_DIR")  # also write PNGs here, for checking the charts
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "figure.facecolor": "white"})


def load(ratings_paths: list[str]):
    results = [r for r in json.loads((OUT / "results.json").read_text()) if "error" not in r]
    stats = {}
    for opt in NAMES:
        rows = [r for r in results if r["option"] == opt]
        if not rows:
            continue
        totals = sorted(r["total_ms"] / 1000 for r in rows)
        stats[opt] = {"p50": statistics.median(totals), "p95": totals[min(len(totals) - 1, int(0.95 * len(totals)))],
                      "totals": totals, "cost": statistics.mean(r["cost"] for r in rows)}
    key = json.loads((OUT / "key.json").read_text())
    ratings = load_ratings(ratings_paths)
    by_q: dict[str, dict[str, int]] = {}
    for html_path in sorted(OUT.glob("index*.html")):
        items = json.loads(re.search(r"const items=(\[.*?\]);let r", html_path.read_text(), re.S).group(1))
        for it in items:
            for a in it["answers"]:
                if a["id"] in ratings:
                    by_q.setdefault(it["q"], {})[key[a["id"]]] = ratings[a["id"]]
    for opt in {o for q in by_q.values() for o in q}:
        stats[opt]["rating"] = statistics.mean(q[opt] for q in by_q.values() if opt in q)
    return stats, by_q


def frontier(points: list[tuple[float, float, str]], better_y_high: bool) -> list[tuple[float, float, str]]:
    """Pareto frontier for 'lower x is better' and y higher-or-lower better."""
    pts = sorted(points, key=lambda p: (p[0], -p[1] if better_y_high else p[1]))
    out, best = [], None
    for p in pts:
        if best is None or (p[1] > best if better_y_high else p[1] < best):
            out.append(p)
            best = p[1]
    return out


# Label nudges (points) where names would collide.
OFFSETS = {"latency-vs-cost.svg": {"exa": (6, -13), "firecrawl": (-46, 14), "exa_instant": (8, -10), "brave": (-38, -14),
                                   "perplexity_search": (-95, 6), "perplexity_search_fast": (-45, -16),
                                   "perplexity_agent": (6, -12), "jina_snippets": (6, 6), "baseten": (6, 10)},
           "accuracy-vs-latency.svg": {"firecrawl": (-70, -2), "exa_answer": (10, -14), "exa": (9, -12),
                                       "perplexity_agent": (-110, -14), "baseten": (-96, 6)},
           "accuracy-vs-cost.svg": {"firecrawl": (-10, 12), "exa": (9, 6), "tavily": (9, -16),
                                    "perplexity_agent": (9, 14), "baseten": (-96, 6)}}


def color(opt: str) -> str:
    return COLORS["one_call"] if opt in ONE_CALL else COLORS["search"]


def legend(ax) -> None:
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS["search"], label="Search API + gpt-4.1"),
                       Line2D([], [], marker="o", ls="", color=COLORS["one_call"], label="One call (search + answer)"),
                       Line2D([], [], color=COLORS["frontier"], lw=2, label="Pareto frontier")],
              loc="best", frameon=False, fontsize=9)


def save(fig, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS / name, format="svg")
    if PREVIEW:
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PREVIEW) / name.replace(".svg", ".png"), format="png", dpi=110)
    plt.close(fig)
    print("wrote", CHARTS / name)


def chart_latency_cost(stats) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    pts = [(s["p50"], s["cost"], o) for o, s in stats.items()]
    f = frontier(pts, better_y_high=False)
    ax.plot([p[0] for p in f], [p[1] * 100 for p in f], color=COLORS["frontier"], lw=2, zorder=1)
    for x, y, o in pts:
        ax.scatter(x, y * 100, s=60, color=color(o), zorder=3, edgecolor="white")
        ax.annotate(NAMES[o], (x, y * 100), xytext=OFFSETS["latency-vs-cost.svg"].get(o, (6, 4)),
                    textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 30)
    ax.set_xticks([1, 2, 5, 10, 20], ["1 s", "2 s", "5 s", "10 s", "20 s"])
    ax.set_xlabel("Median time to a finished answer (log scale)")
    ax.set_ylabel("Cost per lookup (US cents)")
    ax.set_title(f"Speed vs cost, all {len(stats)} options (lower-left is better)", loc="left")
    legend(ax)
    save(fig, "latency-vs-cost.svg")


def chart_rated(stats, x_key: str, x_label: str, name: str, title: str, x_fmt) -> None:
    rated = {o: s for o, s in stats.items() if "rating" in s}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    pts = [(s[x_key], s["rating"], o) for o, s in rated.items()]
    f = frontier(pts, better_y_high=True)
    ax.plot([x_fmt(p[0]) for p in f], [p[1] for p in f], color=COLORS["frontier"], lw=2, zorder=1)
    for x, y, o in pts:
        size = 40 + 9000 * rated[o]["cost"] if x_key == "p50" else 40 + 120 * rated[o]["p50"]
        ax.scatter(x_fmt(x), y, s=size, color=color(o), alpha=0.85, zorder=3, edgecolor="white")
        ax.annotate(NAMES[o], (x_fmt(x), y), xytext=OFFSETS.get(name, {}).get(o, (9, -3)),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Blind rating (1-5, mean of 6 questions)")
    ys = [p[1] for p in pts]
    ax.set_ylim(min(ys) - 0.3, max(ys) + 0.25)
    ax.set_title(title, loc="left")
    legend(ax)
    save(fig, name)


def chart_distribution(stats) -> None:
    order = sorted(stats, key=lambda o: stats[o]["p50"])
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(order) + 1.4))
    for i, o in enumerate(order):
        t = stats[o]["totals"]
        ax.scatter(t, [i] * len(t), s=14, color=color(o), alpha=0.55, zorder=3)
        ax.plot([stats[o]["p50"]] * 2, [i - 0.3, i + 0.3], color="black", lw=1.5, zorder=4)
    ax.set_yticks(range(len(order)), [NAMES[o] for o in order])
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5 s", "1 s", "2 s", "5 s", "10 s", "20 s"])
    ax.set_xlabel("Time to a finished answer, every call (log scale; black tick = median)")
    ax.set_title("Consistency: 20 calls per option", loc="left")
    ax.invert_yaxis()
    save(fig, "latency-distribution.svg")


def chart_heatmap(stats, by_q) -> None:
    opts = sorted((o for o, s in stats.items() if "rating" in s), key=lambda o: -stats[o]["rating"])
    short = {"What's the weather": "Weekend weather", "What time does the Ottawa": "Library hours",
             "Who won the most": "Stanley Cup", "Are the fall colours": "Fall colours",
             "Are there any O-Train": "O-Train disruptions", "What are common side": "Magnesium side effects"}
    qs = list(by_q)
    labels = [next(v for k, v in short.items() if q.startswith(k)) for q in qs]
    grid = [[by_q[q].get(o, 0) for o in opts] for q in qs]
    fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * len(opts) + 2), 4.4))
    ax.imshow(grid, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, str(v) if v else "–", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if v in (1, 5) else "black")
    ax.set_xticks(range(len(opts)), [NAMES[o] for o in opts], rotation=25, ha="right")
    ax.set_yticks(range(len(qs)), labels)
    ax.spines[:].set_visible(False)
    ax.set_title("Blind ratings per question (1 = wrong or useless, 5 = exactly right)", loc="left")
    save(fig, "ratings-heatmap.svg")


def weighted(stats, w_acc: float, opts: list[str]) -> dict[str, float]:
    """Min-max normalise each factor across the rated options (1 = best), then weight. Latency and cost split
    the remaining weight 60:40 (accuracy, then latency, then cost)."""
    def norm(vals: dict[str, float], higher_better: bool) -> dict[str, float]:
        lo, hi = min(vals.values()), max(vals.values())
        if hi == lo:
            return {o: 1.0 for o in vals}
        return {o: ((v - lo) / (hi - lo) if higher_better else (hi - v) / (hi - lo)) for o, v in vals.items()}
    acc = norm({o: stats[o]["rating"] for o in opts}, True)
    lat = norm({o: stats[o]["p50"] for o in opts}, False)
    cost = norm({o: stats[o]["cost"] for o in opts}, False)
    rest = 1 - w_acc
    return {o: w_acc * acc[o] + rest * 0.6 * lat[o] + rest * 0.4 * cost[o] for o in opts}


def chart_weights(stats) -> tuple[str, str, float]:
    """Returns (leader at 50% accuracy weight, leader at 100%, crossover weight)."""
    opts = [o for o, s in stats.items() if "rating" in s]
    ws = [i / 100 for i in range(101)]
    series = {o: [weighted(stats, w, opts)[o] for w in ws] for o in opts}
    lead_lo = max(opts, key=lambda o: series[o][50])
    lead_hi = max(opts, key=lambda o: series[o][100])
    highlight = {lead_lo, lead_hi}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ends: dict[float, list[str]] = {}
    for o, ys in series.items():
        ax.plot(ws, ys, lw=2.2 if o in highlight else 1.2, color=color(o) if o in highlight else COLORS["muted"],
                label=NAMES[o])
        ends.setdefault(round(ys[-1], 3), []).append(NAMES[o])
    for y, names in ends.items():  # options that tie at 100% share one label
        ax.annotate(", ".join(names), (1.0, y), xytext=(4, 0), textcoords="offset points", fontsize=8.5, va="center")
    cross = 1.0
    if lead_lo != lead_hi:
        cross = next((w for w in ws if series[lead_hi][int(round(w * 100))] > series[lead_lo][int(round(w * 100))]), 1.0)
        ax.axvline(cross, color="black", lw=0.8, ls="--")
        ax.annotate(f"{NAMES[lead_hi]} overtakes {NAMES[lead_lo]} when\naccuracy carries over {cross:.0%} of the weight",
                    (cross, 0.08), xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=8.5)
    ax.axvline(0.5, color=COLORS["frontier"], lw=0.8, ls=":")
    ax.annotate("50% accuracy,\n30% speed, 20% cost", (0.5, 0.95), xytext=(-6, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color=COLORS["frontier"])
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Weight on accuracy (the rest split 60:40 between speed and cost)")
    ax.set_ylabel("Weighted score (1 = best on every factor)")
    ax.set_title("Which option wins depends on how much accuracy matters", loc="left")
    save(fig, "weighted-score.svg")
    return lead_lo, lead_hi, cross


def main() -> None:
    stats, by_q = load(sys.argv[1:])
    chart_latency_cost(stats)
    chart_rated(stats, "p50", "Median time to a finished answer (s)", "accuracy-vs-latency.svg",
                "Accuracy vs speed (bubble size = cost; up and left is better)", lambda x: x)
    chart_rated(stats, "cost", "Cost per lookup (US cents)", "accuracy-vs-cost.svg",
                "Accuracy vs cost (bubble size = median time; up and left is better)", lambda x: x * 100)
    chart_distribution(stats)
    chart_heatmap(stats, by_q)
    lead_lo, lead_hi, cross = chart_weights(stats)
    opts = [o for o, s in stats.items() if "rating" in s]
    for w in (0.5, 0.6, 0.7):
        print(f"w_acc={w}", {NAMES[o]: round(v, 2) for o, v in sorted(weighted(stats, w, opts).items(),
                                                                       key=lambda kv: -kv[1])})
    print(f"leader at 50% accuracy weight: {NAMES[lead_lo]}; at 100%: {NAMES[lead_hi]}; crossover {cross:.2f}")
    print(json.dumps({NAMES[o]: {k: round(v, 4) for k, v in s.items() if k != "totals"} for o, s in stats.items()},
                     indent=1))


if __name__ == "__main__":
    main()
