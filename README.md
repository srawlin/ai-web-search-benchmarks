# AI web search benchmarks

Harness, raw data and charts behind the article
[Web search for LLM agents: 17 options benchmarked on latency, cost, and accuracy](https://rawlinson.ca/articles/search-api-bakeoff).

Two kinds of option are compared: a search API whose top 5 results are handed to one fixed model (gpt-4.1) to answer, and
"one call" options where the provider searches and answers in a single request. Ten questions, two rounds each
(three for Baseten, added later), timed
end to end from Ottawa (Toronto/YYZ region), priced at list. The six time-sensitive questions were rated blind by
one human rater; Baseten's single round was rated by Claude after the answers had been labelled.

## Files

- `search_bakeoff.py` — runs the options, writes `out/results.json`, builds the blind-rating page, scores ratings
- `search_bakeoff_charts.py` — renders the six SVG charts in `charts/`
- `out/results.json` — every call: option, round, question, timings (ms), cost (USD), answer, sources
- `out/index.html`, `out/index-2.html`, `out/index-3.html` — the blind-rating pages (answers shuffled, options hidden)
- `out/key.json` — maps rating-page answer ids back to options
- `out/ratings-round1.json`, `out/ratings-round2.json` — the rater's 1–5 scores

## Running it

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add the keys you have
.venv/bin/python search_bakeoff.py run                      # all options, 2 rounds
.venv/bin/python search_bakeoff.py run --rounds 1 tavily    # one option
.venv/bin/python search_bakeoff.py report
.venv/bin/python search_bakeoff.py page --page index-3.html OPTION...   # blind-rating page
.venv/bin/python search_bakeoff.py score out/ratings-*.json
.venv/bin/python search_bakeoff_charts.py out/ratings-round1.json out/ratings-round2.json out/ratings-round3.json
```

Options: `brave exa exa_instant jina jina_snippets firecrawl tavily perplexity_search perplexity_search_fast kagi`
(search API + gpt-4.1) and `exa_answer openai_search_model openai_web_search anthropic_web_search gemini_grounding
perplexity_agent kagi_fastgpt baseten` (one call). `kagi_fastgpt` is implemented but our key was rejected by the legacy
FastGPT endpoint, so it has no data. `baseten` is Kimi-K3 answering over Parallel search, run server-side by
Baseten (needs `BASETEN_API_KEY`).

## Results (October 7–12, 2026)

| Option | Median search | Median total | Worst of 20 | Cost / lookup | Blind rating |
| --- | --- | --- | --- | --- | --- |
| Perplexity search | **0.17 s** | **1.10 s** | 2.5 s | $0.007 | 4.17 |
| **Perplexity search (fast)** | 0.16 s | 1.14 s | 2.0 s | $0.003 | 4.50 |
| Exa (fast) | 0.41 s | 1.14 s | 12.7 s | $0.012 | 4.33 |
| Brave | 0.48 s | 1.33 s | 1.9 s | $0.006 | 3.67 |
| Exa (instant) | 0.47 s | 1.47 s | 2.4 s | $0.008 | — |
| Firecrawl | 0.44 s | 1.47 s | 2.6 s | $0.012 | 4.00 |
| Exa answer | — | 1.60 s | 4.3 s | $0.005 | 4.00 |
| Perplexity Agent API | — | 1.83 s | 3.7 s | **$0.001** | 4.17 |
| Kagi | 1.06 s | 1.97 s | 3.5 s | $0.014 | 4.17 |
| Jina (snippets) | 1.17 s | 2.20 s | 2.9 s | $0.002 | — |
| Tavily | 1.50 s | 2.58 s | 4.6 s | $0.011 | 4.33 |
| OpenAI `web_search` tool | — | 2.65 s | 9.1 s | $0.039 | — |
| **Anthropic web search** | — | 2.74 s | 4.6 s | $0.021 | 4.67 |
| **Baseten (Parallel)** | — | 5.10 s | 23.6 s | $0.031 | **4.83** |
| OpenAI `gpt-5-search-api` | — | 5.85 s | 13.7 s | $0.030 | — |
| Gemini + Google Search | — | 6.72 s | 11.7 s | $0.014 | — |
| Jina (with page text) | 18 s | 19 s | 27 s | $0.004 | — |

Measured from a home connection in Ottawa. One rater, six rated questions per option: treat the ratings as a strong
hint, not a verdict. Details and caveats are in the article.
