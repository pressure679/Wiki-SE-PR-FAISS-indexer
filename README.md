# Wiki-SE-PR-FAISS Indexer

Semantic search over a local Wikipedia ZIM archive and StackExchange dumps, combining
FAISS vector search with a locally-computed PageRank over the Wikipedia link graph.

Source: [`wiki_pr_faiss.py`](wiki_pr_faiss.py)

## What it does

1. **Crawls a Wikipedia ZIM archive** starting from a seed article (your search query),
   following internal wiki links up to a bounded depth, and indexes the text with
   sentence embeddings.
2. **Builds a link graph** of the crawled articles and computes a bounded, local
   PageRank over it, so results can be boosted by how "important" an article is
   relative to the articles it was found alongside.
3. **Ranks results** with a blend of semantic similarity (FAISS L2 distance) and
   PageRank score, then prints a brief extractive explanation, supporting passages,
   and related articles.
4. **Optionally searches StackExchange** Q&A dumps (`Posts.xml` per site) using the
   same embedding model, surfacing the closest matching question and its accepted
   answer.

## How it works

### Wikipedia pipeline (`wiki_handle_query`)
- `index_from_seed` does a breadth-first crawl of the ZIM archive from the query
  term, extracting `mw:WikiLink` anchors (`extract_links`) and article text
  (`clean_html` / BeautifulSoup), chunking long articles (`chunk_text`), and
  embedding chunks in batches with `SentenceTransformer("all-MiniLM-L6-v2")` into a
  `faiss.IndexFlatL2`.
- `compute_pagerank` runs a fixed number of power-iteration steps over the crawled
  graph (`defaultdict(set)` of title → outbound links).
- The query embedding is searched against the FAISS index (`TOP_K` candidates), each
  hit is deduplicated per article by keeping the best-scoring chunk
  (`semantic_score + PR_WEIGHT * pagerank_score`).
- The top article's lead chunk is used for a short extractive explanation (first 3
  sentences); the next several ranked chunks are printed as supporting passages.
- `collect_related` does a bounded BFS from the top article over the crawled graph to
  suggest related articles, filtered to ones already indexed and ranked by PageRank.
  Non-informative link targets (e.g. `ISBN (identifier)`, `Doi (identifier)`) are
  excluded.

### StackExchange pipeline (`se_handle_query`, `search_stackexchange*`)
- `parse_stackexchange_posts` / `build_se_site_index` stream-parse a StackExchange
  `Posts.xml` export, embed questions and high-scoring accepted answers, and persist
  a per-site FAISS index (`<site>.index.faiss`) plus pickled doc metadata
  (`<site>.docs.pkl`).
- `load_all_se_indices` loads any pre-built site indices from
  `Wikipedia and StackExchange/<site>/`.
- `search_stackexchange_semantic_neighbors` embeds the query, searches each loaded
  site index, deduplicates by question, and attaches the accepted answer if one
  exists.

## Requirements

- Python 3
- `numpy`, `faiss-cpu` (or `faiss-gpu`), `sentence-transformers`, `libzim`,
  `beautifulsoup4`
- A local `chunk_text.py` module providing `chunk_text(text) -> list[str]`
- A Wikipedia ZIM file (default path: `Wikipedia and StackExchange/wikipedia_en_simple_all_mini_2025-11.zim`)
- (Optional) StackExchange `Posts.xml` dumps for the sites listed in `SE_SITES`

## Configuration

Key constants at the top of the script:

| Constant | Purpose |
|---|---|
| `ZIM_PATH` | Path to the Wikipedia ZIM archive |
| `MAX_CRAWL_DEPTH` / `MAX_DEPTH` | How many link-hops deep to crawl/relate articles |
| `MAX_PR_DEPTH` | Depth bound for the local PageRank graph |
| `TOP_K` | Number of FAISS candidates retrieved per query |
| `PR_WEIGHT` | Weight of the PageRank term in the combined ranking score |
| `SE_SITES` | StackExchange sites and their `Posts.xml` paths |
| `BATCH_SIZE` | Embedding batch size |

## Usage

```bash
python wiki_pr_faiss.py
```

This starts an interactive loop:

```
🔎 Wikipedia and StackExchange Semantic Search (type 'exit' or press Ctrl-C to quit)

Search: neural networks
```

For each query it will:
- crawl/index the relevant portion of the ZIM archive on the fly,
- print a brief explanation, supporting passages, and related articles.

Type `exit` or `quit` (or press `Ctrl-C`) to stop.

> Note: the StackExchange search (`se_handle_query`) is defined but not called in the
> main loop by default — enable it by uncommenting the call at the bottom of the
> script if you have pre-built StackExchange indices.

## Building StackExchange indices

Use `build_se_site_index(site_name, posts_xml_path, embedder, out_dir)` (or the
commented-out block near the bottom of the script) to pre-build a FAISS index per
StackExchange site from its `Posts.xml` export before enabling `se_handle_query`.
