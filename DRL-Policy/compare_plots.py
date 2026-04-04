"""
compare_training_plots.py
=========================
Generates a comprehensive comparison of DQN vs DDPG training curves.
Paste your actual TensorBoard / console log data in the DATA section below,
then run:  python3 compare_training_plots.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DATA  — (step, reward) pairs from console logs
# ─────────────────────────────────────────────────────────────────────────────
DQN_RAW = [
    (1102,-79.76),(1961,-300.79),(2868,-122.61),(3720,-93.42),(4602,-261.42),
    (5652,-110.52),(6631,-173.11),(7600,-22.73),(8474,-139.19),(9353,-97.93),
    (10182,-137.69),(11185,-105.25),(12129,-201.97),(13035,-73.28),(13904,-341.12),
    (14919,-98.27),(15813,-122.64),(16750,-108.32),(17641,-65.58),(18686,-190.43),
    (19687,-215.90),(20654,-59.46),(21662,-82.59),(22638,-48.06),(23668,-126.07),
    (24747,-324.81),(25704,-77.56),(26679,-79.77),(27632,-175.13),(28572,-81.76),
    (29647,-95.60),(30573,-101.85),(31538,-221.69),(32532,-100.45),(33599,-109.89),
    (34687,-40.16),(35628,-28.68),(36537,-82.83),(37380,-121.44),(38262,-93.42),
    (39216,-44.36),(40122,-23.66),(41073,-97.21),(42015,-85.89),(42932,-40.52),
    (44004,-46.25),(44931,-102.97),(45812,-62.58),(46805,-87.74),(47849,-63.50),
    (48853,-75.80),(49836,-76.43),(50742,-64.66),(51655,-40.52),(52690,-78.18),
    (53602,-96.75),(54636,-51.54),(55636,-26.86),(56683,-40.74),(57641,-29.86),
    (58540,-34.94),(59557,-43.36),(60438,-86.90),(61375,-61.46),(62355,-38.14),
    (63250,-21.69),(64334,-53.75),(65418,-71.30),(66422,-74.81),(67450,-5.14),
    (68508,-79.96),(69476,-105.26),(70403,-55.28),(71384,-11.56),(72364,7.63),
    (73304,-39.06),(74238,-27.14),(75249,-58.64),(76249,-15.18),(77283,10.32),
    (78367,-68.16),(79364,2.78),(80375,-80.96),(81395,-25.79),(82615,-40.88),
    (83587,-114.42),(85048,-81.22),(86089,-70.53),(87173,-28.48),(88484,-78.68),
    (89533,44.21),(90859,-76.36),(91879,-56.92),(92826,-71.72),(93800,-32.39),
    (94871,-5.76),(95919,-42.76),(96975,-41.92),(98559,-12.19),(99627,9.95),
    (101198,-100.18),(102488,-32.97),(103569,-43.05),(106306,-100.53),(108496,-2.59),
    (110262,25.23),(112082,31.62),(114854,-20.31),(117364,-170.12),(120008,-151.39),
    (123828,-270.51),(127974,23.58),(135318,-33.54),(142628,-3.28),(152447,-22.45),
    (160405,16.01),(169501,-66.15),(178817,11.96),(188817,-22.37),(198817,-43.66),
    (207438,28.89),(216176,251.65),(223954,253.88),(231168,277.27),(238353,197.83),
    (244135,264.83),(249652,256.48),(253655,241.97),(258060,222.87),(261444,248.71),
    (264923,214.72),(269471,196.51),(274719,142.22),(281801,258.31),(288792,39.15),
    (294257,175.11),(298901,200.92),(302893,226.31),(308021,260.90),(312277,204.09),
    (315426,292.65),(318257,278.79),(321380,245.52),(324377,301.47),(327120,240.63),
    (330140,278.54),(332814,280.44),(336329,159.43),(338612,292.24),(341058,282.93),
    (343859,257.72),(347032,271.86),(350036,285.76),(352938,275.52),(356411,213.18),
    (358987,279.31),(362860,201.70),(365812,263.99),(368279,10.15),(370695,254.28),
    (374362,190.37),(377292,255.88),(380788,230.46),(383621,289.32),(386161,75.04),
    (388483,282.90),(391854,285.59),(394801,247.52),(397120,279.08),(400327,285.58),
    (403820,277.61),(406285,259.20),(409106,284.29),(411663,244.76),(414023,263.03),
    (416159,285.28),(418829,265.95),(421117,287.73),(423784,265.18),(425611,281.13),
    (427961,257.90),(430141,273.85),(432713,265.58),(434999,314.48),(438047,48.25),
    (440202,294.31),(442478,235.87),(445303,293.55),(447425,289.70),(449973,261.04),
    (452543,251.88),(455088,299.09),(457487,54.16),(460637,274.86),(462320,59.22),
    (464269,267.92),(466506,23.88),(468602,241.12),(470626,278.42),(473057,269.11),
    (475411,270.60),(478040,241.00),(480035,279.85),(482356,268.04),(485083,265.01),
    (487122,63.49),(489212,263.37),(491109,242.49),(493105,280.51),(494862,268.99),
    (497189,300.04),(499325,272.22),
]

DDPG_RAW = [
    (1266,-42.14),(2405,-118.47),(3397,-185.05),(4424,-266.96),(5639,-313.85),
    (6710,-157.51),(7920,-93.59),(8952,-90.03),(10150,-118.80),(15984,-52.77),
    (23420,44.06),(32504,-34.44),(42263,24.78),(48341,-130.43),(56457,-86.65),
    (64718,-10.16),(74166,-24.39),(81704,16.23),(85574,-254.65),(92895,49.63),
    (96791,9.25),(100705,-68.53),(105760,16.04),(111718,83.17),(118660,-9.25),
    (125556,-99.71),(132129,162.29),(136678,-87.37),(141717,199.27),(146870,-95.73),
    (150885,222.21),(155262,-44.15),(159208,-49.93),(164150,140.11),(166829,276.12),
    (169866,233.54),(174154,203.49),(176886,243.77),(181213,264.25),(185376,252.00),
    (188785,239.66),(191831,213.62),(194752,261.15),(197840,279.58),(201082,257.47),
    (204168,261.74),(206781,242.49),(209366,228.51),(212368,260.45),(215423,251.52),
    (219323,239.64),(222254,225.25),(226057,270.61),(229619,250.79),(231411,-6.13),
    (233664,225.77),(235605,247.23),(237540,238.25),(241703,273.08),(244650,282.39),
    (247584,272.35),(250465,253.88),(253691,266.06),(256862,233.40),(259709,280.38),
    (262286,271.20),(265098,264.31),(267633,296.75),(269891,287.87),(272707,255.74),
    (275773,269.02),(278691,-38.40),(281147,243.20),(283529,294.22),(285820,241.73),
    (288020,282.96),(290193,288.39),(292197,240.60),(294165,300.60),(296038,266.28),
    (297942,272.92),(300206,278.03),(302469,236.80),(305464,297.18),(308153,296.64),
    (312475,289.20),(315709,249.05),(317715,229.12),(321037,248.93),(324608,171.81),
    (327689,267.91),(330348,305.15),(332905,279.38),(335870,226.46),(338332,244.35),
    (340055,290.45),(342472,234.04),(344653,231.90),(347593,272.56),(351509,238.15),
    (354290,280.57),(356181,233.07),(357898,24.99),(361505,16.08),(364548,276.31),
    (367397,225.45),(369663,287.01),(371820,268.61),(375097,243.67),(378045,232.73),
    (380227,239.72),(382652,284.89),(385926,294.27),(388296,295.56),(390683,245.49),
    (395145,248.44),(398512,-26.64),(400929,299.18),(403803,285.28),(406254,275.53),
    (409966,237.71),(413646,-93.76),(416850,279.45),(419750,229.80),(423389,264.63),
    (427567,253.91),(431386,245.09),(434476,247.38),(436318,-107.27),(437756,-96.43),
    (440665,14.65),(443853,248.19),(446116,285.97),(449059,251.02),(451293,309.60),
    (453935,298.10),(457096,230.86),(459192,296.35),(461040,305.80),(463647,280.65),
    (465594,-24.97),(467367,-248.95),(470879,250.50),(473006,278.51),(476062,295.76),
    (477897,263.41),(479902,278.15),(481726,228.39),(483446,277.90),(485200,53.96),
    (486847,276.39),(488527,33.00),(490036,54.20),(491549,38.40),(493074,52.72),
    (495357,298.24),(497387,242.46),(499286,294.21),
]

DQN_EVAL  = [286.19, 275.49, 285.12,  45.22, 294.95,
             285.41,  60.18, 272.72,  73.42, 285.34]
DDPG_EVAL = [252.20, 281.92, 263.55, 223.74, 293.79,
             277.76, 280.65, 272.94, 314.17, 297.38]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def unzip(data):
    steps   = np.array([d[0] for d in data], dtype=float)
    rewards = np.array([d[1] for d in data], dtype=float)
    return steps, rewards

def rolling(arr, w):
    """Rolling mean of width w."""
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='valid')

def percentile_band(arr, w):
    """Rolling 10th/90th percentile band."""
    lo, hi = [], []
    for i in range(w - 1, len(arr)):
        window = arr[i - w + 1 : i + 1]
        lo.append(np.percentile(window, 10))
        hi.append(np.percentile(window, 90))
    return np.array(lo), np.array(hi)

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
BG      = '#0d0d1a'
PANEL   = '#13132b'
GRID_C  = '#2a2a4a'
DQN_C   = '#4d9fff'
DDPG_C  = '#ff5c8a'
TARGET  = '#ffd166'
SOLVED  = '#06d6a0'
WHITE   = '#e8e8ff'
MUTED   = '#8888aa'
W       = 25        # smoothing window

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.facecolor':   PANEL,
    'figure.facecolor': BG,
    'axes.edgecolor':   GRID_C,
    'axes.labelcolor':  WHITE,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       WHITE,
    'grid.color':       GRID_C,
    'grid.linewidth':   0.6,
    'grid.alpha':       1.0,
    'axes.grid':        True,
})

dqn_s,  dqn_r  = unzip(DQN_RAW)
ddpg_s, ddpg_r = unzip(DDPG_RAW)
K = 1000  # scale x-axis

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE — 3 rows × 2 cols + 1 wide bottom row
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        hspace=0.52, wspace=0.32,
                        top=0.93, bottom=0.05,
                        left=0.07, right=0.97)

# ── Panel labels helper ───────────────────────────────────────────────────────
def label(ax, txt):
    ax.text(0.015, 0.96, txt, transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=MUTED,
            va='top', ha='left')

def ref_lines(ax):
    ax.axhline(100, color=TARGET, lw=1.0, ls='--', alpha=0.75, zorder=1)
    ax.axhline(200, color=SOLVED, lw=1.0, ls=':',  alpha=0.60, zorder=1)

def style_ax(ax, title):
    ax.set_title(title, color=WHITE, fontsize=11, pad=7, fontweight='bold')
    ax.set_xlabel('Environment Steps (×10³)', fontsize=9, color=MUTED)
    ax.set_ylabel('Episode Reward',           fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8)

# ════════════════════════════════════════════════════════════════════════════
# ROW 0 — Raw scatter + smoothed curve (separate)
# ════════════════════════════════════════════════════════════════════════════
for col, (s, r, c, name) in enumerate([
        (dqn_s, dqn_r, DQN_C, 'DQN — LunarLander-v3'),
        (ddpg_s, ddpg_r, DDPG_C, 'DDPG — LunarLanderContinuous-v3')]):
    ax = fig.add_subplot(gs[0, col])
    # shade band
    lo, hi = percentile_band(r, W)
    bx = s[W-1:] / K
    ax.fill_between(bx, lo, hi, alpha=0.18, color=c, zorder=2)
    # raw scatter
    ax.scatter(s/K, r, s=5, color=c, alpha=0.22, zorder=3, rasterized=True)
    # smooth
    sm = rolling(r, W)
    ax.plot(s[W-1:]/K, sm, color=c, lw=2.2, zorder=4, label='Rolling mean')
    ref_lines(ax)
    style_ax(ax, f'{name}\nRaw Episodes + {W}-ep Rolling Mean')
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE)
    label(ax, chr(65 + col))  # A, B

# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — Overlay comparison on one axes
# ════════════════════════════════════════════════════════════════════════════
ax_ov = fig.add_subplot(gs[1, :])   # full-width
for s, r, c, name in [
        (dqn_s,  dqn_r,  DQN_C,  'DQN'),
        (ddpg_s, ddpg_r, DDPG_C, 'DDPG')]:
    lo, hi = percentile_band(r, W)
    bx = s[W-1:] / K
    ax_ov.fill_between(bx, lo, hi, alpha=0.13, color=c)
    ax_ov.scatter(s/K, r, s=4, color=c, alpha=0.15, rasterized=True)
    sm = rolling(r, W)
    ax_ov.plot(bx, sm, color=c, lw=2.5, label=f'{name} rolling mean')

# mark DQN breakthrough
breakthrough = 207.4  # ksteps
ax_ov.axvline(breakthrough, color=DQN_C,  lw=1.0, ls='-.', alpha=0.6)
ax_ov.text(breakthrough + 2, -360, 'DQN\nbreakthrough\n~207K', color=DQN_C,
           fontsize=7.5, va='bottom')

ddpg_break = 130.0
ax_ov.axvline(ddpg_break, color=DDPG_C, lw=1.0, ls='-.', alpha=0.6)
ax_ov.text(ddpg_break + 2, -360, 'DDPG\nbreakthrough\n~130K', color=DDPG_C,
           fontsize=7.5, va='bottom')

ref_lines(ax_ov)
ax_ov.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE,
             loc='upper left')
ax_ov.set_title('Overlay Comparison — DQN vs DDPG Training Curves', color=WHITE,
                fontsize=12, pad=8, fontweight='bold')
ax_ov.set_xlabel('Environment Steps (×10³)', fontsize=9, color=MUTED)
ax_ov.set_ylabel('Episode Reward',           fontsize=9, color=MUTED)
ax_ov.tick_params(labelsize=8)

# custom legend for threshold lines
from matplotlib.lines import Line2D
extra = [Line2D([0],[0], color=TARGET, lw=1, ls='--', label='Target (+100)'),
         Line2D([0],[0], color=SOLVED,  lw=1, ls=':',  label='Solved (+200)')]
handles, labels_l = ax_ov.get_legend_handles_labels()
ax_ov.legend(handles + extra, labels_l + ['Target (+100)', 'Solved (+200)'],
             fontsize=8.5, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE,
             loc='upper left', ncol=2)
label(ax_ov, 'C')

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — Cumulative max (best-so-far) + 50-ep moving average
# ════════════════════════════════════════════════════════════════════════════
W2 = 50
ax_cum = fig.add_subplot(gs[2, 0])
ax_ma  = fig.add_subplot(gs[2, 1])

for s, r, c, name in [
        (dqn_s,  dqn_r,  DQN_C,  'DQN'),
        (ddpg_s, ddpg_r, DDPG_C, 'DDPG')]:
    best = np.maximum.accumulate(r)
    ax_cum.plot(s/K, best, color=c, lw=2.0, label=name)

    if len(r) >= W2:
        sm50 = rolling(r, W2)
        ax_ma.plot(s[W2-1:]/K, sm50, color=c, lw=2.0, label=name)

ref_lines(ax_cum); ref_lines(ax_ma)
ax_cum.set_title(f'Best-So-Far (Cumulative Max)', color=WHITE, fontsize=11,
                 pad=7, fontweight='bold')
ax_cum.set_xlabel('Environment Steps (×10³)', fontsize=9, color=MUTED)
ax_cum.set_ylabel('Best Reward Seen', fontsize=9, color=MUTED)
ax_cum.tick_params(labelsize=8)
ax_cum.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE)
label(ax_cum, 'D')

ax_ma.set_title(f'{W2}-Episode Moving Average', color=WHITE, fontsize=11,
                pad=7, fontweight='bold')
ax_ma.set_xlabel('Environment Steps (×10³)', fontsize=9, color=MUTED)
ax_ma.set_ylabel('Avg Reward', fontsize=9, color=MUTED)
ax_ma.tick_params(labelsize=8)
ax_ma.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE)
label(ax_ma, 'E')

# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — Evaluation bar chart + box plot
# ════════════════════════════════════════════════════════════════════════════
ax_bar = fig.add_subplot(gs[3, 0])
ax_box = fig.add_subplot(gs[3, 1])

# Bar chart
x   = np.arange(1, 11)
w_b = 0.38
b1 = ax_bar.bar(x - w_b/2, DQN_EVAL,  w_b, color=DQN_C,  alpha=0.85, label='DQN')
b2 = ax_bar.bar(x + w_b/2, DDPG_EVAL, w_b, color=DDPG_C, alpha=0.85, label='DDPG')
ax_bar.axhline(np.mean(DQN_EVAL),  color=DQN_C,  lw=1.5, ls='--', alpha=0.9,
               label=f'DQN mean: {np.mean(DQN_EVAL):.1f}')
ax_bar.axhline(np.mean(DDPG_EVAL), color=DDPG_C, lw=1.5, ls='--', alpha=0.9,
               label=f'DDPG mean: {np.mean(DDPG_EVAL):.1f}')
ax_bar.axhline(100, color=TARGET, lw=1.0, ls=':', alpha=0.7)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f'Ep{i}' for i in x], fontsize=7.5, color=MUTED)
ax_bar.set_title('Evaluation — 10 Test Episodes', color=WHITE, fontsize=11,
                 pad=7, fontweight='bold')
ax_bar.set_ylabel('Episode Reward', fontsize=9, color=MUTED)
ax_bar.tick_params(labelsize=8)
ax_bar.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE,
              ncol=2)
label(ax_bar, 'F')

# Box plot
bp = ax_box.boxplot(
    [DQN_EVAL, DDPG_EVAL],
    patch_artist=True,
    widths=0.4,
    medianprops=dict(color='white', lw=2),
    whiskerprops=dict(color=MUTED, lw=1.2),
    capprops=dict(color=MUTED, lw=1.5),
    flierprops=dict(marker='o', color=MUTED, markersize=5, alpha=0.6),
)
bp['boxes'][0].set(facecolor=DQN_C,  alpha=0.75)
bp['boxes'][1].set(facecolor=DDPG_C, alpha=0.75)

# overlay individual points
for i, (data, c) in enumerate([(DQN_EVAL, DQN_C), (DDPG_EVAL, DDPG_C)], 1):
    jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(data))
    ax_box.scatter(np.full(len(data), i) + jitter, data,
                   color=c, s=28, zorder=5, alpha=0.9, edgecolors='white', lw=0.4)

ax_box.axhline(100, color=TARGET, lw=1.0, ls=':', alpha=0.7, label='Target (+100)')
ax_box.set_xticks([1, 2])
ax_box.set_xticklabels(['DQN', 'DDPG'], fontsize=10, color=WHITE)
ax_box.set_title('Evaluation Score Distribution', color=WHITE, fontsize=11,
                 pad=7, fontweight='bold')
ax_box.set_ylabel('Episode Reward', fontsize=9, color=MUTED)
ax_box.tick_params(labelsize=8)
ax_box.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID_C, labelcolor=WHITE)
label(ax_box, 'G')

# ─────────────────────────────────────────────────────────────────────────────
# Super-title
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle('DQN vs DDPG — LunarLander Training & Evaluation Comparison',
             fontsize=15, fontweight='bold', color=WHITE, y=0.97)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out = 'dqn_vs_ddpg_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"Saved → {out}")
plt.close()