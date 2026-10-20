"""Draw docs/tradeoff.png, the two-panel summary figure at the top of the README.

Every number is copied from the tables in README.md (n = 2,200 per arm for the left panel; the
adapter-scale sweep for the right panel), so the figure can be rebuilt without rerunning anything:

    python scripts/readme_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = Path(__file__).resolve().parents[1] / "docs" / "tradeoff.png"

arms = ["base,\nuntrained", "SFT\n2 epochs", "SFT 3 ep\n+ DPO"]
in_dist = [33.4, 59.4, 71.3]
held_out = [65.8, 40.6, 38.2]

scale = [1.0, 0.75, 0.5]
held_out_scaled = [36.7, 50.8, 72.5]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})

x = range(len(arms))
w = 0.36
blue, orange, grey = "#3b6ea8", "#d9822b", "#555555"
b1 = left.bar([i - w / 2 for i in x], in_dist, w, color=blue, label="fault families seen in training")
b2 = left.bar([i + w / 2 for i in x], held_out, w, color=orange, label="held-out fault families")
for bars in (b1, b2):
    for bar in bars:
        left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.0f}%",
                  ha="center", va="bottom", fontsize=9.5, color=grey)
left.set_xticks(list(x), arms)
left.set_ylim(0, 85)
left.set_ylabel("incidents resolved")
left.set_title("Training trades held-out faults for seen ones", loc="left", fontsize=11.5)
left.legend(frameon=False, fontsize=9.5, loc="upper left")

right.plot(scale, held_out_scaled, marker="o", color=orange, lw=2)
for s, v in zip(scale, held_out_scaled):
    right.text(s, v + 3, f"{v:.0f}%", ha="center", fontsize=9.5, color=grey)
right.set_xlim(1.08, 0.42)
right.set_ylim(0, 85)
right.set_xticks(scale, ["1.0\n(trained)", "0.75", "0.5"])
right.set_xlabel("LoRA update scale s, set at serving time")
right.set_ylabel("held-out incidents resolved")
right.set_title("Scaling the update back recovers them", loc="left", fontsize=11.5)

fig.tight_layout()
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200)
print("wrote", out)
