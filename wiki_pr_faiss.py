import os
import pickle
import numpy as np
import faiss
from collections import deque, defaultdict
from sentence_transformers import SentenceTransformer
from libzim import Archive
from bs4 import BeautifulSoup
from chunk_text import chunk_text
import xml.etree.ElementTree as ET
import time
import gc
import traceback

# ======================================================
# CONFIG
# ======================================================
ZIM_PATH = "Wikipedia and StackExchange/wikipedia_en_simple_all_mini_2025-11.zim"

INDEX_FILE = "wiki.faiss"
CHUNKS_FILE = "wiki_chunks.pkl"
SEEN_FILE = "wiki_seen.pkl"
GRAPH_FILE = "wiki_graph.pkl"
PAGERANK_FILE = "wiki_pagerank.pkl"

MAX_CRAWL_DEPTH = 2     # index depth
MAX_PR_DEPTH = 2        # pagerank depth

TOP_K = 20              # FAISS candidates
PR_WEIGHT = 0.3

SE_SITES = {
    "datascience": "datascience.stackexchange.com/Posts.xml",
    "ai": "ai.stackexchange.com/Posts.xml",
    "cs": "cs.stackexchange.com/Posts.xml",
    # "cstheory": "cstheory.stackexchange.com/Posts.xml",
    "money": "money.stackexchange.com/Posts.xml",
    "economics": "economics.stackexchange.com/Posts.xml",
}

# ======================================================
# EMBEDDINGS
# ======================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
# chunks = []
# se_chunks = []
# pagerank = {}
# seen = set()
# graph = defaultdict(set)
# queue = None
# archive = Archive(ZIM_PATH)
SE_INDICES = {}
BATCH_SIZE = 96
MAX_DEPTH = 2

def resolve_entry(archive, title):
    """
    Resolve title to a canonical libzim entry.
    Returns (canonical_title, entry) or (None, None).
    """
    if archive.has_entry_by_title(title):
        return title, archive.get_entry_by_title(title)

    alt = title.replace(" ", "_")
    if archive.has_entry_by_title(alt):
        return alt, archive.get_entry_by_title(alt)

    return None, None

# ======================================================
# PAGE RANK (LOCAL, BOUNDED)
# ======================================================
def compute_pagerank(graph, damping=0.85, iterations=25):
    nodes = list(graph.keys())
    N = len(nodes)
    if N == 0:
        return

    pr = {n: 1.0 / N for n in nodes}

    for _ in range(iterations):
        new_pr = {}
        for node in nodes:
            rank_sum = 0.0
            for other, links in graph.items():
                if node in links and len(links) > 0:
                    rank_sum += pr[other] / len(links)
            new_pr[node] = (1 - damping) / N + damping * rank_sum
        pr = new_pr
    return pr

# ======================================================
# LINK EXTRACTION
# ======================================================
def extract_links(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        # Only internal wiki links
        if a.get("rel") != ["mw:WikiLink"]:
            continue

        href = a["href"]

        # Skip special namespaces
        if ":" in href:
            continue

        # Skip empty / weird
        if not href.strip():
            continue

        href = href.replace('_', ' ')

        # href IS the canonical title
        links.add(href)

    return links

def clean_html(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)

def index_from_seed(index, chunks, graph, seen, seed_title, max_depth=MAX_DEPTH):
    archive = Archive(ZIM_PATH)
    queue = deque([(seed_title, 0)])
    pending_chunks = []

    while queue:
        # print("in while queue loop")
        raw_title, depth = queue.popleft()
        if depth > max_depth:
            continue

        _, entry = resolve_entry(archive, raw_title)
        if entry is None or raw_title in seen:
            continue

        try:
            raw = entry.get_item().content
            html = bytes(raw).decode("utf-8", errors="ignore")
        except Exception:
            continue

        # ---- extract links early ----
        links = extract_links(html)
        if depth < max_depth:
            for l in links:
                if l not in seen:
                    queue.append((l, depth + 1))

        # ---- extract text ----
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if len(text.split()) < 100:
            continue

        article_chunks = [text] if len(text.split()) < 800 else chunk_text(text)

        # metadata first
        for i, chunk in enumerate(article_chunks):
            chunks.append({
                "title": raw_title,
                "text": chunk,
                "is_lead": i == 0
            })
            pending_chunks.append(chunk)

        if len(pending_chunks) >= BATCH_SIZE:
            emb = embedder.encode(pending_chunks, batch_size=BATCH_SIZE)
            index.add(np.array(emb).astype("float32"))
            pending_chunks.clear()

        seen.add(raw_title)
        graph[raw_title] = links
    
    if pending_chunks:
        emb = embedder.encode(pending_chunks, batch_size=BATCH_SIZE)
        index.add(np.array(emb).astype("float32"))
        pending_chunks.clear()
    return index, chunks, graph, seen

def build_bounded_graph(max_depth=MAX_DEPTH):
    bounded_graph = defaultdict(set)

    for src in seen:
        visited = {src}
        q = deque([(src, 0)])

        while q:
            node, d = q.popleft()
            if d == max_depth:
                continue

            for nxt in graph.get(node, []):
                if nxt in seen and nxt not in visited:
                    visited.add(nxt)
                    bounded_graph[src].add(nxt)
                    q.append((nxt, d + 1))

    return bounded_graph

def collect_related(graph, start, max_depth=MAX_DEPTH):
    related = set()
    visited = {start}
    q = deque([(start, 0)])

    while q:
        node, depth = q.popleft()
        if depth == max_depth:
            continue

        for nxt in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                related.add(nxt)
                q.append((nxt, depth + 1))

    return related

def make_stackexchange_queries(query):
    q = query.lower().strip()

    prompts = [
        q,
        f"how {q}",
        f"what is {q}",
        f"how to {q}",
        f"{q} problem",
        f"{q} issue",
    ]

    return list(set(prompts))

def search_stackexchange(query, top_k_per_site=20):
    prompts = make_stackexchange_queries(query)
    candidates = {}

    for site, (index, docs) in SE_INDICES.items():
        for p in prompts:
            q_emb = embedder.encode([p]).astype("float32")
            D, I = index.search(q_emb, top_k_per_site)

            for dist, idx in zip(D[0], I[0]):
                doc = docs[idx]
                qid = doc["question_id"]

                sem_score = 1.0 / (1.0 + dist)
                vote_score = doc.get("score", 0)
                accepted = 1 if doc.get("accepted") else 0

                score = (
                    sem_score
                    + 0.2 * (vote_score ** 0.5)
                    + 0.5 * accepted
                )

                if qid not in candidates:
                    candidates[qid] = {
                        "score": score,
                        "site": site,
                        "docs": [doc]
                    }
                else:
                    candidates[qid]["score"] = max(
                        candidates[qid]["score"], score
                    )
                    candidates[qid]["docs"].append(doc)

    ranked = sorted(
        candidates.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:5]

def parse_stackexchange_posts(posts_xml_path, max_posts=None):
    """
    Stream-parse Posts.xml safely.
    Yields dictionaries.
    """
    context = ET.iterparse(posts_xml_path, events=("end",))
    count = 0

    for event, elem in context:
        if elem.tag != "row":
            continue

        attrs = elem.attrib
        post_type = attrs.get("PostTypeId")

        raw_body = attrs.get("Body", "")
        if raw_body.count(" ") < 20:
            continue

        # Question
        if post_type == "1":
            yield {
                "post_type": "question",
                "question_id": int(attrs["Id"]),
                "title": attrs.get("Title", ""),
                "body": clean_html(raw_body),
                "score": int(attrs.get("Score", 0)),
                "accepted": False,
                "tags": attrs.get("Tags", "").strip("<>").split("><")
            }

        # Answer
        elif post_type == "2":
            yield {
                "post_type": "answer",
                "question_id": int(attrs["ParentId"]),
                "title": None,
                "body": clean_html(raw_body),
                "score": int(attrs.get("Score", 0)),
                "accepted": attrs.get("IsAcceptedAnswer") == "True",
                "tags": []
            }

        elem.clear()
        count += 1

        if max_posts and count >= max_posts:
            break

def build_stackexchange_faiss(posts_xml_path, embedder, max_posts=None):
    se_docs = []
    embeddings = []
    texts, posts = [], []

    for post in parse_stackexchange_posts(posts_xml_path, max_posts=max_posts):
        # Only index meaningful text
        if post["post_type"] == "answer" and post["score"] < 1:
            continue
        # if len(text.split()) < 20:
        #     continue
        if post["post_type"] == "question":
            text = post["title"] + ": " + post["body"]
        else:
            text = post["body"]

        texts.append(text)
        posts.append(post)

        if len(texts) == BATCH_SIZE:
            embs = embedder.encode(texts, batch_size=BATCH_SIZE)
            embeddings.extend(embs)
            se_docs.extend(posts)
            texts, posts = [], []

        # emb = embedder.encode([text])[0]
        # embeddings.append(emb)
        # se_docs.append(post)

    if texts:
        embs = embedder.encode(texts, batch_size=BATCH_SIZE)
        embeddings.extend(embs)
        se_docs.extend(posts)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    se_index = faiss.IndexFlatL2(dim)
    se_index.add(embeddings)

    print(f"✅ StackExchange indexed: {len(se_docs)} posts")

    return se_index, se_docs

def save_stackexchange_index(se_index, se_docs):
    faiss.write_index(se_index, "stackexchange.faiss")
    with open("stackexchange_docs.pkl", "wb") as f:
        pickle.dump(se_docs, f)

def load_stackexchange_index(site_name):
    se_index = faiss.read_index(site_name + ".index.faiss")
    with open("Wikipedia and StackExchange/" + site_name + ".docs.pkl", "rb") as f:
        se_docs = pickle.load(f)
    return se_index, se_docs

class ProgressTracker:
    def __init__(self, total_bytes, interval_sec=300):
        self.interval_sec = interval_sec
        self.start_time = time.time()
        self.last_update = self.start_time
        self.total_bytes = total_bytes

    def update(self, bytes_read):
        now = time.time()
        current_bucket = int((now - self.start_time) // self.interval_sec)

        # Only update once per bucket
        if current_bucket <= self.last_update:
            return

        self.last_update = current_bucket

        remaining = max(self.total_bytes - bytes_read, 0)
        elapsed = now - self.start_time
        rate = bytes_read / max(elapsed, 1e-6)
        eta = remaining / max(rate, 1e-6)
        rate_mb = rate / 1000000

        secs_total = int(eta)
        hours, rem = divmod(secs_total, 3600)
        mins, secs = divmod(rem, 60)

        percent = (bytes_read / self.total_bytes) * 100

        print(
            f"⏳ [{elapsed/60:.0f}m] {percent:.1f}% | "
            f"{rate_mb:.4f} MB/s | "
            f"ETA ~{hours}h{mins}m"
        )

def build_se_site_index(site_name, posts_xml_path, embedder, out_dir, max_posts=None):
    se_docs = []
    embeddings = []

    total_bytes = os.path.getsize(posts_xml_path)
    tracker = ProgressTracker(total_bytes)

    print(f"🔨 Indexing {site_name} ({total_bytes / 1e6:.1f} MB)")

    count = 0

    with open(posts_xml_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        for event, elem in context:
            if elem.tag != "row":
                continue

            # ✅ safe progress update
            tracker.update(f.tell())
            # print(f"f.tell: {f.tell()}")

            attrs = elem.attrib
            post_type = attrs.get("PostTypeId")
            raw_body = attrs.get("Body", "")
            if raw_body.count(" ") < 20:
                elem.clear()
                continue

            # --------------------
            # Question
            # --------------------
            if post_type == "1":
                post = {
                    "post_type": "question",
                    "question_id": int(attrs["Id"]),
                    "title": attrs.get("Title", ""),
                    "body": clean_html(raw_body),
                    "score": int(attrs.get("Score", 0)),
                    "accepted": False,
                    "tags": attrs.get("Tags", "").strip("<>").split("><")
                }

            # --------------------
            # Answer
            # --------------------
            elif post_type == "2":
                # Optional quality filter
                # if int(attrs.get("Score", 0)) < 1:
                #     elem.clear()
                #     continue
                
                if int(attrs.get("Score", 0)) < 5:
                    elem.clear()
                    continue
                if attrs.get("IsAcceptedAnswer") != "True":
                    elem.clear()
                    continue

                post = {
                    "post_type": "answer",
                    "question_id": int(attrs["ParentId"]),
                    "title": None,
                    "body": clean_html(raw_body),
                    "score": int(attrs.get("Score", 0)),
                    "accepted": attrs.get("IsAcceptedAnswer") == "True",
                    "tags": []
                }
            else:
                elem.clear()
                continue

            # --------------------
            # Build embedding
            # --------------------
            text = (
                post["title"] + " " + post["body"]
                if post["post_type"] == "question"
                else post["body"]
            )

            # if len(text.split()) < 20:
            #     elem.clear()
            #     continue

            emb = embedder.encode([text])[0]
            embeddings.append(emb)
            se_docs.append(post)

            elem.clear()
            count += 1

            if max_posts and count >= max_posts:
                break

    # --------------------
    # FAISS build
    # --------------------
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, site_name + ".index.faiss"))

    with open(os.path.join(out_dir, site_name + ".docs.pkl"), "wb") as f:
        pickle.dump(se_docs, f)

    print(f"✅ Indexed {site_name}: {len(se_docs)} posts")

def search_stackexchange_sites(query, sites, top_k=20):
    results = []

    for site in sites:
        index, docs = load_stackexchange_index(site)

        q_emb = embedder.encode([query]).astype("float32")
        D, I = index.search(q_emb, top_k)

        for dist, idx in zip(D[0], I[0]):
            doc = docs[idx]
            # score = 1.0 / (1.0 + dist) + 0.2 * (doc["score"] ** 0.5)
            sem_score = 1.0 / (1.0 + dist)
            vote_score = doc.get("score", 0)
            accepted = 1 if doc.get("accepted") else 0

            score = (
                sem_score
                + 0.2 * (vote_score ** 0.5)
                + 0.5 * accepted
            )
            results.append((score, site, doc))

    return sorted(results, reverse=True)[:top_k]

def se_index_exists(out_dir, site_name):
    index_path = os.path.join(out_dir, f"{site_name}.index.faiss")
    docs_path = os.path.join(out_dir, f"{site_name}.docs.pkl")
    return os.path.exists(index_path) and os.path.exists(docs_path)

def load_all_se_indices(base_dir="Wikipedia and StackExchange"):
    for site in SE_SITES:
        index_path = f"{base_dir}/{site}/{site}.index.faiss"
        docs_path = f"{base_dir}/{site}/{site}.docs.pkl"

        if not os.path.exists(index_path):
            continue

        index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        SE_INDICES[site] = (index, docs)

    print(f"✅ Loaded {len(SE_INDICES)} StackExchange sites")

""" 
OUT_DIR = "Wikipedia and StackExchange"
for site, path in SE_SITES.items():
    if se_index_exists(OUT_DIR, site):
        print(f"⏭️  Skipping {site} (index already exists)")
        continue

    print(f"🔨 Building StackExchange index for {site}...")

    build_se_site_index(
        site_name=site,
        posts_xml_path=os.path.join(OUT_DIR, path),
        embedder=embedder,
        out_dir=OUT_DIR,
        max_posts=None
    )
""" 

def search_stackexchange_semantic_neighbors(query, top_k_per_site=20, final_k=5):
    # prompts = make_stackexchange_queries(query)
    candidates = []

    for site, (index, docs) in SE_INDICES.items():
        # for p in prompts:
        q_emb = embedder.encode([query]).astype("float32")
        D, I = index.search(q_emb, top_k_per_site)

        for dist, idx in zip(D[0], I[0]):
            doc = docs[idx]

            # only consider questions
            if doc["post_type"] != "question":
                continue

            candidates.append({
                "site": site,
                "dist": dist,
                "question": doc
            })

    # FAISS distance = semantic closeness
    candidates.sort(key=lambda x: x["dist"])

    results = []
    seen_qids = set()

    for c in candidates:
        qid = c["question"]["question_id"]
        if qid in seen_qids:
            continue
        seen_qids.add(qid)

        # find accepted answer
        site_docs = SE_INDICES[c["site"]][1]
        answers = [
            d for d in site_docs
            if d["post_type"] == "answer"
            and d["question_id"] == qid
            and d.get("accepted")
        ]

        results.append({
            "site": c["site"],
            "question": c["question"],
            "answer": answers[0] if answers else None
        })

        if len(results) == final_k:
            break

    return results

def wiki_handle_query(query):
    # global index, chunks, graph, seen, archive

    chunks = []
    graph = defaultdict(set)
    seen = set()
    index = faiss.IndexFlatL2(embedder.get_embedding_dimension())
    pagerank = {}

    index, chunks, graph, seen = index_from_seed(index, chunks, graph, seen, query, max_depth=MAX_DEPTH)
    pagerank = compute_pagerank(graph)

    if index.ntotal == 0:
        print("⚠️ Nothing indexed.")
        return
    if index.ntotal != len(chunks):
        print("⚠️ length of chunks not equal index length.")
        return
    q_emb = embedder.encode([query]).astype("float32")
    D, I = index.search(q_emb, TOP_K)

    # ranking + printing

    # --------------------------------------------------
    # Deduplicate by article title + rank
    # --------------------------------------------------
    article_scores = {}

    for rank, idx in enumerate(I[0]):
        if idx >= len(chunks):
            continue
        r = chunks[idx]
        title = r["title"]

        semantic = -D[0][rank]
        pr = pagerank.get(title, 0.0)
        score = semantic + PR_WEIGHT * pr

        if title not in article_scores or score > article_scores[title]["score"]:
            article_scores[title] = {
                "score": score,
                "chunk": r
            }

    ranked = sorted(
        article_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    results = [r["chunk"] for r in ranked]

    # --------------------------------------------------
    # Brief explanation (extractive)
    # --------------------------------------------------
    lead = next((r for r in results if r["is_lead"]), results[0])
    sentences = lead["text"].split(".")
    explanation = ".".join(sentences[:3]).strip()

    print(f"\n📌 Brief explanation ({lead['title']}):")
    print(explanation + ".")

    # --------------------------------------------------
    # Supporting passages
    # --------------------------------------------------
    print("\n📄 Relevant passages:")
    for r in results[2:12]:
        print(f"— {r['title']}")
        sentences = r["text"].split(".")
        passage = ".".join(sentences[:7]).strip()
        print(passage + ".")
        print()

    # --------------------------------------------------
    # Related important articles (depth ≤ 2)
    # --------------------------------------------------
    main_title = lead["title"]

    related = collect_related(graph, main_title, max_depth=MAX_DEPTH)
    related = [t for t in related if t in seen and t != main_title and t != "ISBN (identifier)" and t !=! "Doi (identifier)"]

    ranked_links = sorted(
        related,
        key=lambda t: pagerank.get(t, 0.0),
        reverse=True
    )

    if ranked_links:
        print("\n🔗 Related important articles:")
        for t in ranked_links[:10]:
            print(f"- {t}")

def se_handle_query(query):
    load_all_se_indices()
    se_results = search_stackexchange_semantic_neighbors(query)

    if se_results:
        print("\n💬 StackExchange discussions:\n")

        for i, item in enumerate(se_results, 1):
            docs = item["docs"]
            site = item["site"]

            # embeddings = np.load(f"Wikipedia and StackExchange/{site}.embeddings.npy")
            # se_index = faiss.IndexFlatL2()
            # se_index.add(embeddings)
            # q_emb = embedder.encode([query]).astype("float32")
            # se_D, se_I = se_index.search(q_emb, TOP_K)
            # SE_INDICES[site] = (se_index, docs)

            question = next(
                d for d in docs if d["post_type"] == "question"
            )

            print(f"{i}. [{site}] {question['title']} (score {question['score']})")
            print(question["body"].replace("\n", " "))

            answers = [
                d for d in docs
                if d["post_type"] == "answer" and d.get("accepted")
            ]

            if answers:
                snippet = answers[0]["body"].replace("\n", " ")
                print(f"   ✔ Accepted answer: {snippet}...")
            else:
                print("   ✖ No accepted answer")

            print()

# ======================================================
# SEARCH LOOP
# ======================================================
print("\n🔎 Wikipedia and StackExchange Semantic Search (type 'exit' or press Ctrl-C to quit)")

while True:
    query = input("\nSearch: ").strip()
    if query.lower() in ("exit", "quit"):
        break

    wiki_handle_query(query)
    # se_handle_query(query)