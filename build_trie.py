from transformers import AutoTokenizer
from trie import (
    TRAIN_PATH,
    MODEL_PATH,
    TRIE_SAVE_PATH,
    MIN_SCORE,
    USE_UNIQUE_QUERY,
    read_queries_from_jsonl,
    build_query_trie,
    save_trie,
)

if __name__ == "__main__":
    queries = read_queries_from_jsonl(TRAIN_PATH, min_score=MIN_SCORE)

    if USE_UNIQUE_QUERY:
        before = len(queries)
        queries = list(dict.fromkeys(queries))
        print(f"[INFO] unique queries: {before} -> {len(queries)}")

    print("[INFO] sample queries:", queries[:5])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    trie_obj = build_query_trie(queries, tokenizer, add_special_tokens=False)

    print("[DEBUG] class module =", trie_obj.__class__.__module__)
    print("[DEBUG] class name   =", trie_obj.__class__.__name__)

    save_trie(trie_obj, TRIE_SAVE_PATH)