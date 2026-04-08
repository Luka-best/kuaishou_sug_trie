import pandas as pd
import sys
file = sys.argv[1]

df = pd.read_csv(file)

print(df.head())


def edit_distance(a: str, b: str) -> int:
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)

    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )
    return dp[n][m]


df_valid = df[df["is_none"] == False].copy()

df_valid["edit_dist_valid"] = df_valid.apply(
    lambda x: edit_distance(x["gt_query"], x["pred_query"]),
    axis=1
)

if "gen_status" in df_valid:
    df_complete = df_valid[df_valid["gen_status"] == True].copy()

    df_complete["edit_dist_complete"] = df_complete.apply(
        lambda x: edit_distance(x["gt_query"], x["pred_query"]),
        axis=1
    )
    
if "pass_filter" in df_valid.columns:
    df_pass = df_valid[df_valid["pass_filter"] == True].copy()
else:
    df_pass = df_valid.copy()

df_pass["edit_dist_pass_filter"] = df_pass.apply(
    lambda x: edit_distance(x["gt_query"], x["pred_query"]),
    axis=1
)


total_cnt = len(df)

valid_cnt = len(df_valid)
valid_mean = df_valid["edit_dist_valid"].mean() if valid_cnt > 0 else None

# complete_cnt = len(df_complete)
# complete_mean = df_complete["edit_dist_complete"].mean() if complete_cnt > 0 else None

pass_cnt = len(df_pass)
pass_mean = df_pass["edit_dist_pass_filter"].mean() if pass_cnt > 0 else None

print("=" * 60)
print(f"total samples: {total_cnt}")

print(f"valid decode count: {valid_cnt}, ratio: {valid_cnt / total_cnt:.4f}, mean edit: {valid_mean}")
# print(f"complete decode count: {complete_cnt}, ratio: {complete_cnt / total_cnt:.4f}, mean edit: {complete_mean}")
print(f"pass filter count: {pass_cnt}, ratio: {pass_cnt / total_cnt:.4f}, mean edit: {pass_mean}")
