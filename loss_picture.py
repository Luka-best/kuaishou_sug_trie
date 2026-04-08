import re
import os
import ast
import sys 
import pandas as pd
import matplotlib.pyplot as plt

log_file = sys.argv[1]
base_name = os.path.basename(log_file)
max_points = 3000
ntp_losses = []
nttp_losses = []
total_losses = []
eval_losses = []

def downsample_list(x, max_points=5000):
    if len(x) <= max_points:
        return x
    step = len(x) / max_points
    idxs = [int(i * step) for i in range(max_points)]
    return [x[i] for i in idxs]

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue

        try:
            record = ast.literal_eval(line)
        except Exception:
            continue

        if "ntp_loss" in record:
            ntp_losses.append(record["ntp_loss"])
        if "loss_ntp" in record:
            ntp_losses.append(record["loss_ntp"])
        if "alpha_loss_nttp" in record:
            nttp_losses.append(record["alpha_loss_nttp"])
        if "loss_total" in record:
            total_losses.append(record["loss_total"])
        if "eval_loss" in record:
            eval_losses.append(record["eval_loss"])

print("num ntp_loss:", len(ntp_losses))
print("num nttp_loss:", len(nttp_losses))
print("num total loss:", len(total_losses))
print("num eval_loss:", len(eval_losses))

plot_loss = total_losses if len(total_losses) >0 else ntp_losses
s = pd.Series(plot_loss)
smooth_50 = s.rolling(window=50, min_periods=1).mean()
smooth_100 = s.rolling(window=100, min_periods=1).mean()


# ===== 画图时再下采样 =====
smooth_50_plot = downsample_list(smooth_50, max_points=max_points)
smooth_100_plot = downsample_list(smooth_100, max_points=max_points)
plot_losses = downsample_list(plot_loss, max_points=max_points)

# 平滑曲线
plt.figure(figsize=(12, 6))
plt.plot(plot_losses, alpha=0.25, label="loss")
# plt.plot(smooth_50_plot, label="moving avg (50)")
plt.plot(smooth_100_plot, label="moving avg (100)")
plt.xlabel("logging step")
plt.ylabel("loss")
plt.title("Loss Curve (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"outputs/{base_name}.png", dpi=200)
print(f"saved: outputs/{base_name}.png")

# if len(ntp_losses) > 0 and len(total_losses)==0:
#     plt.figure(figsize=(10, 5))
#     plt.plot(ntp_losses)
#     plt.xlabel("logging step")
#     plt.ylabel("ntp_loss")
#     plt.title("NTP Loss Curve")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"outputs/{base_name}.png", dpi=200)
#     plt.show()


# if len(total_losses) > 0:
#     plt.figure(figsize=(10, 5))
#     plt.plot(total_losses)
#     plt.xlabel("logging step")
#     plt.ylabel("trainer loss")
#     plt.title("Trainer Loss Curve")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"outputs/{base_name}.png", dpi=200)
#     plt.show()
