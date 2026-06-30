# citation/ — isolated citation-capture prototype

A standalone experiment that turns a **natural-language request** into a
captured **BibTeX** file. It reuses the host project's configuration, chat
model, and Web Search MCP, but is **not** wired into the agent graph, the skill
system, the capability map, or any slash command.

```
natural-language request  (e.g. "幫我找關於檢索效率的文章")
  → LLM-driven discovery: the agent DECIDES the search queries itself
    (e.g. it searches "BM25", "learned sparse retrieval", "reranking latency"),
    runs the Web Search MCP, and ranks the papers it actually retrieved
  → show candidate papers; you pick one after discussion
  → resolve a trustworthy DOI for that selected paper
  → DOI / Crossref metadata route for BibTeX (title/author/year verified)
  → citation/cite/<normalized_title>.bib
```

The query strategy is **decided at runtime by the model**, not hard-coded — the
request is never just stuffed verbatim into a single search call. Candidates are
always grounded in real tool output; the LLM never invents papers or BibTeX.
BibTeX is produced only after the user selects a paper, and the accepted write
path is DOI → Crossref/DOI BibTeX.

If **no** route yields real BibTeX, the run reports the failure and writes
nothing. The prototype never fabricates BibTeX or saves placeholder data.

## How to run

From the **repo root** (`PiDNA2/app/`), inside the `app` conda env (Poetry
installs into it; `virtualenvs.create = false`):

```bash
# interactive: describe what you want in natural language; the agent decides how
# to search, then lists candidates for you to choose one
python -m citation.cli "幫我找關於檢索效率的文章"
python -m citation.cli "papers about RAG citation hallucination evaluation"

# smoke-test / non-interactive: auto-pick the first candidate and only accept a
# high-confidence DOI (never prompts)
python -m citation.cli "Attention is all you need" --auto

# options
python -m citation.cli --help
python -m citation.cli "<request>" --limit 8 --verbose
```

You can also omit the request and be prompted for it: `python -m citation.cli`.

## What it reuses (read-only) from the host project

| Need              | Source (unmodified)                                   |
|-------------------|-------------------------------------------------------|
| Config            | `agent.config.AgentConfig`                            |
| Chat model        | `agent.llm.get_chat_model` (OpenRouter, `llm_model`)  |
| Web Search MCP    | `agent.mcp.load_mcp_tools_with_families` (family `web_search`) |
| Repo root         | `agent.paths.find_app_root`                           |
| `.env` loading    | `python-dotenv`, same `.env` the agent CLI reads      |

The LLM **drives discovery**: it is given the Web Search MCP tools and decides
which queries to run (and may issue several), then annotates and ranks the
papers it retrieved. It is never used to generate BibTeX, and candidates only
come from real tool output. If `OPENROUTER_API_KEY` is missing, discovery
degrades gracefully to a single deterministic scholar-oriented query
(`discover_candidates`) instead of the agentic loop. Crossref match decisions
are always deterministic (title similarity + year + author overlap).

## Environment variables

These already live in the host project's `.env` (git-ignored):

- `OPENROUTER_API_KEY` — for the chat model (optional here; enables enrichment).
- `AGENT_ENABLE_MCP_WEB_SEARCH=1` — **required**: turns on the Web Search MCP.
- `AGENT_MCP_WEB_SEARCH_COMMAND` / `AGENT_MCP_WEB_SEARCH_ARGS` — how to launch
  `mrkrsl/web-search-mcp`.
- `CROSSREF_MAILTO` — optional; added to the Crossref `User-Agent` for the
  "polite pool". No personal data is hard-coded.

## Output

BibTeX is written to `citation/cite/<normalized_title>.bib`.

Filename rules: lowercase, trimmed, spaces → `_`, unsafe characters removed,
consecutive `_` collapsed, `.bib` extension. Existing files are **never**
overwritten — a numeric suffix is added (`name.bib` → `name_2.bib`).

Example: `Attention Is All You Need` → `attention_is_all_you_need.bib`.

## Web Search MCP tools used

From `mrkrsl/web-search-mcp` (family `web_search`):

| Tool | Use here |
|------|----------|
| `get-web-search-summaries` | the agent's main discovery tool (cheap, snippets only) and DOI lookup helper |
| `get-single-web-page-content` | inspect a chosen paper's source page for a DOI (and the agent may use it during discovery) |
| `full-web-search` | bound and available to the agent, but it's steered toward summaries to avoid bulk crawling |

All three are **bound to the model** during discovery; the prompt steers it
toward `get-web-search-summaries` and a handful of targeted queries.

## Failure handling (every step reports, none fail silently)

- Web Search MCP not enabled/loaded → clear error + how to enable.
- MCP enabled but no parseable results → "no candidate papers found".
- No DOI on the selected candidate/source page → falls through to Crossref title search and a Scholar-oriented DOI lookup.
- Crossref finds no DOI → reported in the trace.
- Crossref finds several similar candidates → **ambiguous**: interactive mode
  asks you to confirm; `--auto` refuses to guess and aborts.
- Crossref BibTeX retrieval fails → tries to resolve an alternate DOI, then retries the DOI/Crossref route.
- Scholar DOI lookup fails (CAPTCHA / nothing citable) → reported plainly.
- `citation/cite/` missing → created automatically.
- Target `.bib` already exists → safe numeric suffix.

## Known limitations / workarounds (kept inside `citation/` by design)

- **Google Scholar is not a stable API and is CAPTCHA-guarded.** Discovery and
  DOI lookup are therefore best-effort and use normal Web Search MCP output. We
  do not solve CAPTCHAs and do not scrape Scholar's Cite popup. BibTeX capture
  must still go through the DOI/Crossref route.
- The Web Search MCP is Playwright-backed and the host loader writes a server
  log under `~/.cache/agent-mcp/`. In a restricted sandbox that path may be
  read-only and the search engines / Crossref / doi.org hosts may be blocked;
  run in an environment with normal network + a writable cache.
- The MCP server returns **formatted text**, not structured JSON, so discovery
  parses its `**N. Title** / URL: / Description:` layout. If that upstream
  format changes, update `citation/discovery.parse_summaries`.
- This prototype intentionally does **not** reuse `agent.session.ChatSession`
  (which would pull in the full graph/skill machinery). It calls the MCP tools
  and the chat model directly — the minimal isolated path.
```
