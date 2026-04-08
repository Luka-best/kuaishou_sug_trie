import json
import pickle
from collections import Counter
from transformers import AutoTokenizer

TRAIN_PATH = "./data/train.jsonl"
MODEL_PATH = "./models/Qwen3-1.7B"   # 改成你的本地路径或模型名
TRIE_SAVE_PATH = "./data/query_trie_with_freq.pkl"
MIN_SCORE = None                  # 例如 0.6；先设 None 表示不过滤
USE_UNIQUE_QUERY = False           # 是否去重后建 Trie（推荐先 True）

# ----------------------------
# 1) 读取 train.jsonl 收集 query
# ----------------------------
def read_queries_from_jsonl(path, min_score=None):
    queries = []
    total = 0
    kept = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] line {line_no} json parse error: {e}")
                continue

            q = (obj.get("query") or "").strip()
            if not q:
                continue

            if min_score is not None:
                score = obj.get("score", None)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    continue
                if score < min_score:
                    continue

            queries.append(q)
            kept += 1

    print(f"[INFO] total lines={total}, kept queries={kept}")
    return queries


# ----------------------------
# 2) Trie 节点定义（token id 级）
# ----------------------------
class TrieNode:
    __slots__ = ("children", "is_end", "count")

    def __init__(self):
        self.children = {}   # token_id -> TrieNode
        self.is_end = False
        self.count = 0       # 有多少 query 经过这个节点（可用于统计/加权）


class TokenTrie:
    def __init__(self):
        self.root = TrieNode()
        self.num_sequences = 0

    def insert(self, token_ids):
        node = self.root
        node.count += 1
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
            node.count += 1
        node.is_end = True
        self.num_sequences += 1

    def get_next_tokens(self, prefix_ids):
        """给定 prefix token ids，返回合法下一 token 列表（用于 NTTP/解码）"""
        node = self.root
        for tid in prefix_ids:
            if tid not in node.children:
                return []
            node = node.children[tid]
        return list(node.children.keys())

    def contains(self, token_ids):
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                return False
            node = node.children[tid]
        return node.is_end
    
    def get_next_tokens_with_counts(self, prefix_ids):
        node = self.root
        for tid in prefix_ids:
            if tid not in node.children:
                return [], []
            node = node.children[tid]

        next_tokens = []
        next_counts = []
        for tid, child in node.children.items():
            next_tokens.append(tid)
            next_counts.append(child.count)

        return next_tokens, next_counts


# ----------------------------
# 3) 用 Qwen tokenizer 对 query 分词并建 Trie
# ----------------------------
def build_query_trie(queries, tokenizer, add_special_tokens=False):
    trie = TokenTrie()
    skipped = 0
    max_len = 0

    for q in queries:
        # 不要 return_tensors，直接拿 token ids 列表
        token_ids = tokenizer.encode(q, add_special_tokens=add_special_tokens)

        if not token_ids:
            skipped += 1
            continue
        
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            token_ids = token_ids + [eos_id]

        trie.insert(token_ids)
        max_len = max(max_len, len(token_ids))

    print(f"[INFO] trie built, num_sequences={trie.num_sequences}, skipped={skipped}, max_token_len={max_len}")
    return trie

# ----------------------------
# 4) 序列化保存与加载
# ----------------------------
def save_trie(trie, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(trie, f)
    print(f"[INFO] trie saved to: {save_path}")

def load_trie(save_path):
    with open(save_path, "rb") as f:
        trie = pickle.load(f)
    print(f"[INFO] trie loaded from: {save_path}")
    return trie



if __name__ == "__main__":
    # 读 query
    queries = read_queries_from_jsonl(TRAIN_PATH, min_score=MIN_SCORE)

    # 是否去重（建议先去重，后面要加频次权重可再保留重复统计）
    if USE_UNIQUE_QUERY:
        before = len(queries)
        queries = list(dict.fromkeys(queries))  # 保序去重
        print(f"[INFO] unique queries: {before} -> {len(queries)}")

    # 看下 top query 频次（调试用）
    counter = Counter(queries)
    print("[INFO] sample queries:", queries[:5])

    # 加载 tokenizer（Qwen3）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 建 Trie
    trie = build_query_trie(queries, tokenizer, add_special_tokens=False)

    # 序列化保存到本地
    save_trie(trie, TRIE_SAVE_PATH)

    # demo: 测试某个 query 前缀的合法 next token
    demo_q = queries[0]
    demo_ids = tokenizer.encode(demo_q, add_special_tokens=False)
    if len(demo_ids) >= 1:
        prefix = demo_ids[:-1]   # 去掉最后一个 token，当作前缀
        next_tokens = trie.get_next_tokens(prefix)
        print(f"[DEMO] query={demo_q}")
        print(f"[DEMO] prefix_ids={prefix}")
        print(f"[DEMO] valid prefix texts={tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)}")
        print(f"[DEMO] valid next token ids={next_tokens[:20]}")
        print(f"[DEMO] valid next token strs={tokenizer.convert_ids_to_tokens(next_tokens[:20])}")
        print(f"[DEMO] valid next token texts={ [tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False) for tid in next_tokens[:20]] }")