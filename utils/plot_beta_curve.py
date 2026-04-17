import numpy as np
import matplotlib.pyplot as plt

# Beta range
beta = np.linspace(0, 2, 500)

# Define geometry / pose weights from Eq.(joint_encode_emphasis)
geom_weight = np.ones_like(beta)
pose_weight = np.ones_like(beta)

# beta <= 1 : geometry magnitude shrinks, pose stays full
mask_low = beta <= 1.0
geom_weight[mask_low] = beta[mask_low]
pose_weight[mask_low] = 1.0

# beta > 1 : geometry full, pose rotation decays
mask_high = beta > 1.0
geom_weight[mask_high] = 1.0
pose_weight[mask_high] = np.exp(-5 * (beta[mask_high] - 1.0))

# Convert to normalized split position y in [0,1]
# y = fraction assigned to pose (lower region)
y = pose_weight / (pose_weight + geom_weight)

# Microsoft colors
pose_color = "#107C10"   # Microsoft Green
geom_color = "#0078D4"   # Microsoft Blue
line_color = "#D83B01"   # Microsoft Orange

plt.figure(figsize=(6.2, 4.2))

# Fill regions
plt.fill_between(beta, 0, y, color=pose_color, alpha=0.22, label="Pose region")
plt.fill_between(beta, y, 1, color=geom_color, alpha=0.18, label="Geometry region")

# plt.fill_between(beta, 0, y, color=pose_color, alpha=0.22)
# plt.fill_between(beta, y, 1, color=geom_color, alpha=0.18)

# Boundary curve
plt.plot(beta, y, color=line_color, linewidth=3.0, label="Boundary from Eq.(27)")
# plt.plot(beta, y, color=line_color, linewidth=3.0)

# Neutral point
plt.scatter([1.0], [0.5], color=line_color, s=45, zorder=5)
plt.axvline(1.0, linestyle="--", color="black", linewidth=1.2, alpha=0.6)
plt.axhline(0.5, linestyle="--", color="black", linewidth=1.2, alpha=0.35)

plt.text(1.02, 0.5, r"$(\beta=1,\ y=0.5)$", va="bottom")

plt.xlim(0, 2)
plt.ylim(0, 1)
plt.xlabel(r"Emphasis parameter $\beta$")
plt.ylabel("Normalized split (pose below, geometry above)")
plt.title("Pose–Geometry Emphasis Derived from Joint Encoding")

plt.legend(frameon=False, loc="upper right")
plt.tight_layout()
# plt.axis('off')
plt.savefig("beta_curve_plot.png", dpi=300)