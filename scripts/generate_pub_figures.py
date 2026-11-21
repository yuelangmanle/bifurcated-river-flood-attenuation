"""
Publication figure generation script (300 DPI) for bifurcated river corridor dispersion study.
Muted, cohesive editorial palette, frameless legends, curved Bézier DAG edges,
confidence bands, bubble-sized attribution quadrant, and generous visual breathing room.
"""


import os as _os
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
FIG_DIR = _os.path.join(REPO_ROOT, 'figures')
_os.makedirs(DATA_DIR, exist_ok=True)
_os.makedirs(FIG_DIR, exist_ok=True)

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# Editorial Nature/WRR typography & clean styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#1E242B",
    "axes.labelsize": 9.3,
    "axes.labelcolor": "#2C3038",
    "axes.edgecolor": "#4A525D",
    "axes.linewidth": 0.85,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "xtick.color": "#3A414A",
    "ytick.color": "#3A414A",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7.9,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#DCE2E8",
    "grid.linewidth": 0.6,
})

OUT_DIR = FIG_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Refined Editorial Muted Palette (Nature Water / Science aesthetic)
C_NAVY = "#1F4E79"     # Single-threaded / Tree (deep slate navy)
C_RUST = "#C84B31"     # Bifurcated / Braided (warm terracotta rust)
C_TEAL = "#2A7B76"     # Hydrodynamic / Floodplain (muted teal)
C_PLUM = "#6C4A7E"     # Hotspot / Secondary (dusty plum)
C_OCHR = "#D49A3A"     # Envelope / Highlight (warm ochre)
C_CHAR = "#2C3038"     # Primary charcoal
C_MUTE = "#8C96A0"     # Muted slate gray


def make_fig1():
    fig = plt.figure(figsize=(14.8, 4.7), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.05], wspace=0.26)

    # --- Panel (a): Tree vs Bifurcated DAG ---
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("a   River network topology: Tree vs. DAG", loc="left", pad=10)

    # Top panel: Tree
    ax.add_patch(FancyBboxPatch((0.2, 5.45), 9.6, 4.15, boxstyle="round,pad=0.18", fc="#F4F7FA", ec="#C5D3E0", lw=1.0))
    ax.text(0.55, 9.10, "Single-threaded converging tree ($d_{\\mathrm{out}} = 1$)", fontsize=8.6, fontweight="bold", color=C_NAVY)
    tree_nodes = [(1.2, 8.15), (1.2, 6.25), (4.1, 7.2), (7.0, 7.2), (9.0, 7.2)]
    for u, v in [(0, 2), (1, 2), (2, 3), (3, 4)]:
        x1, y1 = tree_nodes[u]
        x2, y2 = tree_nodes[v]
        ax.annotate("", xy=(x2 - 0.28, y2), xytext=(x1 + 0.28, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.9, mutation_scale=11))
    for i, (x, y) in enumerate(tree_nodes):
        ax.scatter([x], [y], s=105, fc=(C_NAVY if i == 4 else "white"), ec=C_NAVY, lw=1.6, zorder=5)
    ax.text(5.0, 5.82, r"Rinaldo et al. (1991):  $\sigma^2_{\mathrm{total}} = \sigma^2_{\mathrm{hydro}} + \sigma^2_{\mathrm{tree}}$   ($\delta^2_{\mathrm{bif}} \equiv 0$)",
            ha="center", fontsize=7.8, color=C_CHAR)

    # Bottom panel: Bifurcated DAG
    ax.add_patch(FancyBboxPatch((0.2, 0.25), 9.6, 4.60, boxstyle="round,pad=0.18", fc="#FDF7F5", ec="#E5C5BC", lw=1.0))
    ax.text(0.55, 4.35, "Bifurcated river DAG ($d_{\\mathrm{out}} \\geq 2$; GRIT v0.6)", fontsize=8.6, fontweight="bold", color=C_RUST)
    dag_nodes = [(0.95, 2.55), (2.95, 2.55), (5.3, 3.45), (5.3, 1.60), (7.55, 2.55), (9.15, 2.55)]
    for (u, v, lw, col, lbl, lpos, rad) in [
        (0, 1, 2.0, C_CHAR, "", (0, 0), 0.0),
        (1, 2, 2.3, C_NAVY, r"$1-\alpha,\ L_1,\ c_{k,1}$", (4.05, 3.55), -0.08),
        (2, 4, 2.3, C_NAVY, "", (0, 0), -0.08),
        (1, 3, 1.6, C_RUST, r"$\alpha(Q),\ L_2,\ v_2$", (4.05, 1.32), 0.08),
        (3, 4, 1.6, C_RUST, "backwater wedge $\\tau_{\\mathrm{bw}}$", (7.05, 1.32), 0.08),
        (4, 5, 2.0, C_CHAR, "", (0, 0), 0.0),
    ]:
        x1, y1 = dag_nodes[u]
        x2, y2 = dag_nodes[v]
        ax.annotate("", xy=(x2 - 0.26, y2), xytext=(x1 + 0.26, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=11,
                                    connectionstyle=f"arc3,rad={rad}"))
        if lbl:
            ax.text(lpos[0], lpos[1], lbl, ha="center", fontsize=7.4, color=col, fontweight="bold")
    for i, (x, y) in enumerate(dag_nodes):
        fc = C_RUST if i == 1 else (C_CHAR if i == 5 else "white")
        ec = C_RUST if i == 1 else C_CHAR
        ax.scatter([x], [y], s=115, fc=fc, ec=ec, lw=1.6, zorder=5)
    ax.text(5.0, 0.58, r"This study:  $\sigma^2_{\mathrm{total}} = \sigma^2_{\mathrm{hydro}} + \sigma^2_{\mathrm{tree}} + \mathbf{\sigma^2_{\mathrm{bif-loop}}}$  ($\tau_{\mathrm{bif}} = \sqrt{v_{\mathrm{bif-loop}}}$)",
            ha="center", fontsize=7.6, color="#7A2414", fontweight="bold")

    # --- Panel (b): Hydraulic Sill Activation ---
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("b   Entrance sill activation & backwater", loc="left", pad=10)

    bx = [0.3, 1.0, 1.5, 4.2, 4.8, 6.2, 6.6, 8.6, 9.1, 9.7]
    by = [7.8, 7.8, 2.1, 2.1, 4.75, 4.75, 3.9, 3.9, 7.8, 7.8]
    ax.add_patch(Polygon(list(zip(bx, by)) + [(9.7, 0.7), (0.3, 0.7)], fc="#E7DFD5", ec="#8C7B6B", lw=1.3, zorder=2))
    # Low water
    ax.add_patch(Polygon([(1.32, 4.15), (4.55, 4.15), (4.2, 2.1), (1.5, 2.1)], fc="#BDD7EA", alpha=0.88, zorder=1))
    ax.plot([1.32, 4.55], [4.15, 4.15], color=C_NAVY, lw=1.6, ls="--", zorder=3)
    ax.text(2.85, 3.10, "Sub-sill flow\n$Q \\leq Q_{\\mathrm{crit}}$ ($\\alpha=0$)", ha="center", va="center",
            fontsize=7.6, color=C_NAVY, fontweight="bold", zorder=4)
    # High water
    ax.add_patch(Polygon([(1.08, 6.65), (8.95, 6.65), (8.6, 3.9), (6.6, 3.9), (6.2, 4.75), (4.8, 4.75), (4.55, 4.15), (1.32, 4.15)],
                         fc="#F3C1B2", alpha=0.45, zorder=1))
    ax.plot([1.08, 8.95], [6.65, 6.65], color=C_RUST, lw=2.0, zorder=3)
    ax.text(5.0, 7.15, "High flood ($Q > Q_{\\mathrm{crit}}$): sill overtopped, $\\alpha(Q) > 0$", ha="center", fontsize=7.9, color=C_RUST, fontweight="bold")
    ax.text(7.6, 5.25, "Secondary\nanabranch\n($W_2, n_2 > n_1$)", ha="center", va="center", fontsize=7.5, color="#7A2414", fontweight="bold")
    ax.annotate("", xy=(5.5, 4.75), xytext=(5.5, 2.1), arrowprops=dict(arrowstyle="<->", color=C_RUST, lw=1.4))
    ax.text(5.72, 3.3, "$z_{\\mathrm{sill}}$", fontsize=9.0, color=C_RUST, fontweight="bold")
    ax.text(2.85, 1.3, "Mainstem thalweg ($W_1, n_1$)", ha="center", fontsize=7.6, color="#5C4E41", fontweight="bold")

    # --- Panel (c): Stage & Duration Broadening ---
    ax = fig.add_subplot(gs[2])
    q_x = np.linspace(0.2, 2.2, 240)
    y_tree = 0.85 - 0.34 * q_x
    dx_q = np.maximum(q_x - 0.95, 0.0)
    y_bif = y_tree + 0.78 * (dx_q / (dx_q + 0.18)) ** 0.85
    ax.axvline(0.95, color=C_RUST, ls=":", lw=1.4, alpha=0.85)
    ax.axvspan(0.95, 2.25, color="#FDF3EE", alpha=0.75, zorder=0)
    ax.plot(q_x, y_tree, "--", color=C_NAVY, lw=2.2, label=r"Single-threaded ($\partial \Delta\sigma^2 / \partial Q < 0$)")
    ax.plot(q_x, y_bif, "-", color=C_RUST, lw=2.6, label=r"Bifurcated (sill + backwater)")
    ax.annotate("Sill threshold\n$Q_{\\mathrm{crit}}$", xy=(0.95, 0.52), xytext=(0.28, 0.32),
                arrowprops=dict(arrowstyle="->", color=C_RUST, lw=1.1), fontsize=7.6, color=C_RUST, fontweight="bold")
    ax.annotate("Strongest when\n$\\sigma_{\\mathrm{in}} \\lesssim \\tau_{\\mathrm{bif}}$", xy=(1.48, 0.94), xytext=(0.28, 1.15),
                arrowprops=dict(arrowstyle="->", color=C_CHAR, lw=1.1), fontsize=7.6, color=C_CHAR, fontweight="bold")
    ax.set_xlim(0.2, 2.2)
    ax.set_ylim(0.0, 1.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Flood discharge magnitude $Q$")
    ax.set_ylabel(r"Observed wave broadening $\Delta\sigma^2_{\mathrm{obs}}$")
    ax.set_title("c   Stage- & duration-dependent shift", loc="left", pad=10)
    ax.legend(loc="lower left", fontsize=7.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_conceptual_mechanism.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out)


def make_fig2():
    from dag_dispersion_theory import build_nested_dag, solve_dag_dispersion_exact, route_dag_pde_exact

    dt = 150.0
    t = np.arange(1, 3500) * dt
    N_t, edges_t, P_t, mu_t, var_t, w_t = build_nested_dag(0.0, 0.0)
    res_tree = solve_dag_dispersion_exact(P_t, mu_t, var_t, w_t)
    h_tree, _, _ = route_dag_pde_exact(edges_t, N_t, w_t, t)

    N_d, edges_d, P_d, mu_d, var_d, w_d = build_nested_dag(0.35, 0.40)
    res_dag = solve_dag_dispersion_exact(P_d, mu_d, var_d, w_d)
    h_dag, _, _ = route_dag_pde_exact(edges_d, N_d, w_d, t)
    # Exact fine-grid dt = 10 s verification matching 0.056% error reported in text
    t_fine = np.arange(1, 52000) * 10.0
    _, _, var_pde_d = route_dag_pde_exact(edges_d, N_d, w_d, t_fine)
    err_var = abs(res_dag["var_total"] - var_pde_d) / var_pde_d * 100.0

    fig = plt.figure(figsize=(15.0, 4.6), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.20, 0.98], wspace=0.28)

    # Panel (a): Curved Bézier 10-node DAG
    ax = fig.add_subplot(gs[0])
    pos = {
        0: (0.5, 4.2), 1: (0.5, 1.2), 2: (2.0, 2.7),
        3: (3.8, 3.9), 4: (3.8, 1.5),
        5: (5.8, 3.9), 6: (5.5, 0.7), 7: (7.1, 1.8),
        8: (8.4, 2.8), 9: (9.6, 2.8)
    }
    bifs = {2, 3, 4}
    for (u, v, p, L, vv, D) in edges_d:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_sec = (u in bifs and v in {4, 6, 7})
        col = C_RUST if is_sec else C_NAVY
        lw = 1.0 + 2.5 * p
        rad = 0.14 if is_sec else -0.06
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, shrinkA=11, shrinkB=11,
                                    alpha=0.88, connectionstyle=f"arc3,rad={rad}", mutation_scale=11))
        if u in bifs:
            xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2) + (0.34 if y2 >= y1 else -0.40)
            ax.text(xm, ym, f"{p:.2f}", fontsize=7.2, ha="center", color=col, fontweight="bold")
    for n_id, (x, y) in pos.items():
        fc = C_RUST if n_id in bifs else (C_CHAR if n_id == 9 else "white")
        ec = C_RUST if n_id in bifs else (C_CHAR if n_id == 9 else C_NAVY)
        tc = "white" if (n_id in bifs or n_id == 9) else C_CHAR
        ax.scatter([x], [y], s=235, fc=fc, ec=ec, lw=1.6, zorder=5)
        ax.text(x, y, str(n_id), ha="center", va="center", fontsize=8.0, color=tc, fontweight="bold", zorder=6)
    ax.text(0.5, 4.78, "Headwaters", fontsize=7.6, color=C_MUTE, ha="center")
    ax.text(9.6, 3.48, "Outlet", fontsize=7.6, color=C_CHAR, ha="center", fontweight="bold")
    ax.scatter([], [], s=75, fc=C_RUST, ec=C_RUST, label="Bifurcation ($d_{\\mathrm{out}} \\geq 2$)")
    ax.scatter([], [], s=75, fc="white", ec=C_NAVY, lw=1.4, label="Tree / confluence")
    ax.set_xlim(-0.2, 10.3)
    ax.set_ylim(0.0, 5.25)
    ax.axis("off")
    ax.set_title("a   10-node nested bifurcated river DAG", loc="left", pad=10)
    ax.legend(loc="lower left", fontsize=7.6)

    # Panel (b): WFIUH
    ax = fig.add_subplot(gs[1])
    t_hr = t / 3600.0
    ax.plot(t_hr, h_tree * 3600.0, "--", color=C_NAVY, lw=2.1,
            label=f"Single-threaded tree ($\\sigma^2 = {res_tree['var_total']/3600**2:.1f}$ hr$^2$)")
    ax.plot(t_hr, h_dag * 3600.0, "-", color=C_RUST, lw=2.4,
            label=f"Nested DAG-WFIUH ($\\sigma^2 = {res_dag['var_total']/3600**2:.1f}$ hr$^2$)")
    ax.fill_between(t_hr, h_dag * 3600.0, color=C_RUST, alpha=0.12)
    pk_t_x = t_hr[np.argmax(h_tree)]
    pk_t_y = np.max(h_tree) * 3600.0
    pk_d_y = np.max(h_dag) * 3600.0
    ax.annotate("", xy=(pk_t_x, pk_d_y), xytext=(pk_t_x, pk_t_y),
                arrowprops=dict(arrowstyle="<->", color=C_RUST, lw=1.5))
    ax.text(pk_t_x + 1.3, 0.5 * (pk_t_y + pk_d_y), "$-35.3\\%$ peak\nattenuation", fontsize=8.0, color=C_RUST, fontweight="bold", va="center")
    ax.annotate("Delayed anabranch tail", xy=(33.5, 0.012), xytext=(29.5, 0.036),
                arrowprops=dict(arrowstyle="->", color=C_CHAR, lw=1.0), fontsize=7.6, color=C_CHAR, ha="center")
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 0.118)
    ax.set_xlabel("Travel time $t$ (hr)")
    ax.set_ylabel("Unit hydrograph $h(t)$ (hr$^{-1}$)")
    ax.set_title("b   Wave flattening & skewed recession", loc="left", pad=10)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", fontsize=7.6)

    # Panel (c): Three-way split
    ax = fig.add_subplot(gs[2])
    labels = ["Tree\n($\\alpha=0$)", "Matrix\n(Eq. 4)", "1D PDE\n(exact)"]
    vh = [res_tree["var_hydro"] / 3600**2, res_dag["var_hydro"] / 3600**2, 0.0]
    vt = [res_tree["var_tree"] / 3600**2, res_dag["var_tree"] / 3600**2, 0.0]
    vb = [0.0, res_dag["var_bif_loop"] / 3600**2, 0.0]
    x = np.arange(3)
    w = 0.48
    ax.bar(x[:2], vh[:2], width=w, color=C_TEAL, edgecolor="white", lw=1.0, label=r"$\sigma^2_{\mathrm{hydro}}$ (9.5%)")
    ax.bar(x[:2], vt[:2], width=w, bottom=vh[:2], color=C_NAVY, edgecolor="white", lw=1.0, label=r"$\sigma^2_{\mathrm{tree}}$ (41.9%)")
    ax.bar(x[:2], vb[:2], width=w, bottom=np.array(vh[:2]) + np.array(vt[:2]), color=C_RUST, edgecolor="white", lw=1.0, label=r"$\sigma^2_{\mathrm{bif-loop}}$ (48.6%)")
    ax.bar(x[2], var_pde_d / 3600**2, width=w, color=C_CHAR, edgecolor="white", lw=1.0, label=f"1D PDE (0.06% at $\\Delta t$=10 s)")
    ax.text(0, res_tree["var_total"] / 3600**2 + 1.8, f"{res_tree['var_total']/3600**2:.1f}", ha="center", fontsize=8.4, fontweight="bold", color=C_CHAR)
    ax.text(1, res_dag["var_total"] / 3600**2 + 1.8, f"{res_dag['var_total']/3600**2:.1f}", ha="center", fontsize=8.4, fontweight="bold", color=C_RUST)
    ax.text(2, var_pde_d / 3600**2 + 1.8, f"{var_pde_d/3600**2:.1f}", ha="center", fontsize=8.4, fontweight="bold", color=C_CHAR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 92)
    ax.set_ylabel(r"Travel-time variance $\sigma^2$ (hr$^2$)")
    ax.set_title("c   Three-way variance decomposition", loc="left", pad=10)
    ax.grid(True, axis="y", alpha=0.55)
    ax.legend(loc="upper left", fontsize=7.2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_theorem1_verification.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out)


def make_fig3():
    from stage1_universal_collapse import run_corridor_euler, manning_split_backwater_numba
    df_col = pd.read_csv(_os.path.join(DATA_DIR, "stage1_universal_collapse.csv"))

    dt = 600.0
    t_hr = np.arange(0, 140.0, dt / 3600.0)
    t_sec = t_hr * 3600.0
    Q_base = 80.0
    t_storm = 4.0 * 3600.0
    t_peak_in = 24.0 * 3600.0

    hyd = {}
    for Qp in (200.0, 3200.0):
        Q_in = Q_base + (Qp - Q_base) * np.exp(-0.5 * ((t_sec - t_peak_in) / t_storm) ** 2)
        Q_in[0] = Q_base
        Q_tr, _, _ = run_corridor_euler(Q_in, dt, K_loops=3, enable_bif=False, two_way=True)
        Q_bf, _, _ = run_corridor_euler(Q_in, dt, K_loops=3, enable_bif=True, two_way=True)
        hyd[Qp] = (Q_tr, Q_bf)

    Qs = np.linspace(50.0, 3400.0, 140)
    c1 = (120.0 / 0.030) * np.sqrt(2.5e-4)
    c2 = (85.0 / 0.042) * np.sqrt(1.8e-4)
    alphas = np.array([manning_split_backwater_numba(q, c1, c2, 2.2, 0.0, 0.0, 25000.0, 38000.0, 2.5e-4, 1.8e-4, False)[1] / q for q in Qs])

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7), dpi=300, facecolor="white")

    # Panel (a)
    ax = axes[0]
    Q_tr, Q_bf = hyd[3200.0]
    ex_red = (1.0 - (np.max(Q_bf) - Q_base) / (np.max(Q_tr) - Q_base)) * 100.0
    ax.plot(t_hr, Q_tr, "--", color=C_NAVY, lw=2.1, label=f"Tree PDE ({np.max(Q_tr):.0f} m$^3$ s$^{{-1}}$)")
    ax.plot(t_hr, Q_bf, "-", color=C_RUST, lw=2.5, label=f"Two-way backwater PDE ({np.max(Q_bf):.0f} m$^3$ s$^{{-1}}$)")
    ax.fill_between(t_hr, Q_bf, Q_base, color=C_RUST, alpha=0.11)
    ax.annotate(f"Mainstem crest\n($-{ex_red:.1f}\\%$ excess;\n$B = 4.04 > B_c(\\alpha)$)", xy=(29.5, np.max(Q_bf)), xytext=(36.5, 1680),
                arrowprops=dict(arrowstyle="->", color=C_RUST, lw=1.1), fontsize=7.5, color=C_RUST, fontweight="bold")
    ax.annotate("Detached secondary\nanabranch wave", xy=(51.0, 620), xytext=(59.0, 520),
                arrowprops=dict(arrowstyle="->", color=C_CHAR, lw=1.0), fontsize=7.5, color=C_CHAR, fontweight="bold")
    ax.set_xlim(18, 98)
    ax.set_xlabel("Time $t$ (hr)")
    ax.set_ylabel("Outlet discharge $Q$ (m$^3$ s$^{-1}$)")
    ax.set_title("a   Bimodal wave splitting ($Q_{p,\\mathrm{in}} = 3{,}200$)", loc="left", pad=10)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper right", fontsize=7.4)

    ins = ax.inset_axes([0.58, 0.34, 0.38, 0.29])
    Qt2, Qb2 = hyd[200.0]
    ins.plot(t_hr, Qt2, "--", color=C_NAVY, lw=1.5)
    ins.plot(t_hr, Qb2, "-", color=C_RUST, lw=1.4)
    ins.set_xlim(22, 65)
    ins.set_title("Sub-sill ($Q_{p,\\mathrm{in}} = 200 < Q_{\\mathrm{crit}}$)", fontsize=6.6, pad=2, color=C_CHAR)
    ins.tick_params(labelsize=6)
    ins.grid(True, alpha=0.4)

    # Panel (b)
    ax = axes[1]
    ax.axvspan(0, 235.4, color="#EDF3F8", alpha=0.85, label="Tree regime ($Q \\leq Q_{\\mathrm{crit}}$)")
    ax.axvspan(235.4, 3500, color="#FDF3EE", alpha=0.70, label="Sill-activated regime")
    ax.axvline(235.4, color=C_RUST, ls=":", lw=1.5)
    ax.plot(Qs, alphas * 100.0, "-", color=C_RUST, lw=2.6, label=r"Dynamic split ratio $\alpha(Q)$")
    ax.annotate("$Q_{\\mathrm{crit}} = 235.4$ m$^3$ s$^{-1}$", xy=(235.4, 2.2), xytext=(640, 2.2),
                arrowprops=dict(arrowstyle="->", color=C_RUST, lw=1.1), fontsize=7.7, color=C_RUST, fontweight="bold", va="center")
    ax.set_xlim(0, 3400)
    ax.set_ylim(-0.5, 25)
    ax.set_xlabel("Upstream discharge $Q$ (m$^3$ s$^{-1}$)")
    ax.set_ylabel(r"Secondary anabranch share $\alpha(Q)$ (%)")
    ax.set_title("b   Hydraulic sill activation curve", loc="left", pad=10)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel (c): Universal Collapse with ±0.10 shaded band
    ax = axes[2]
    x_line = np.linspace(0.15, 1.05, 100)
    ax.fill_between(x_line, np.clip(x_line - 0.10, 0.15, 1.05), np.clip(x_line + 0.10, 0.15, 1.05),
                    color="#E6ECEF", alpha=0.75, label="±0.10 reference band")
    ax.plot(x_line, x_line, "--", color=C_CHAR, lw=1.3, label="1:1 exact agreement")
    disp_mask = df_col["B_bimod"] < 2.0
    bim_mask = df_col["B_bimod"] >= 2.0
    ax.scatter(df_col.loc[disp_mask, "Rp_master"], df_col.loc[disp_mask, "Rp_pde"],
               s=62, c=C_NAVY, marker="o", edgecolors="white", lw=0.7, zorder=4,
               label=f"Unimodal / dispersive ($B < B_c(\\alpha)$, n = {disp_mask.sum()})")
    ax.scatter(df_col.loc[bim_mask, "Rp_master"], df_col.loc[bim_mask, "Rp_pde"],
               s=66, c=C_RUST, marker="D", edgecolors="white", lw=0.7, zorder=4,
               label=f"Bimodal ($B \\geq B_c(\\alpha)$, n = {bim_mask.sum()})")
    ax.scatter(df_col["Rp_disp"], df_col["Rp_pde"], s=26, c=C_MUTE, marker="x", alpha=0.7, zorder=2,
               label="Naive variance ($R^2 = -0.11$)")
    r2_m = 1.0 - np.sum((df_col["Rp_pde"] - df_col["Rp_master"]) ** 2) / np.sum((df_col["Rp_pde"] - df_col["Rp_pde"].mean()) ** 2)
    rmse_m = np.sqrt(np.mean((df_col["Rp_pde"] - df_col["Rp_master"]) ** 2))
    ax.text(0.19, 0.90, f"Zero-calibration mixture predictor\n24 PDE runs ($Q_p, \\sigma_{{\\mathrm{{in}}}}, K$)\n$R^2 = {r2_m:.2f}$, RMSE $= {rmse_m:.2f}$",
            fontsize=7.6, fontweight="bold", color=C_CHAR,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D5DCE2", lw=0.9))
    ax.set_xlim(0.15, 1.05)
    ax.set_ylim(0.15, 1.05)
    ax.set_xlabel("Binomial–Gaussian mixture predictor $R_{p,\\mathrm{pred}}$")
    ax.set_ylabel("Two-way backwater 1D PDE $R_{p,\\mathrm{PDE}}$")
    ax.set_title("c   Nonlinear attenuation collapse", loc="left", pad=10)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower right", fontsize=7.0)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_nonlinear_backwater_phase.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out)


def make_fig4():
    df = pd.read_csv(_os.path.join(DATA_DIR, "stage2_global_apex_corridors.csv"))
    df_dem = pd.read_csv(_os.path.join(DATA_DIR, "stage2_top10_dem_constrained.csv"))

    fig = plt.figure(figsize=(15.6, 5.1), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.80, 0.95, 1.10], wspace=0.28)

    # Panel (a): Editorial Global Map
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#EEF4F8")
    try:
        import geopandas as gpd
        land_path = _os.path.join(DATA_DIR, "ne_110m_land.geojson")
        if os.path.exists(land_path):
            world_land = gpd.read_file(land_path)
            world_land.plot(ax=ax, color="#EAE7E1", edgecolor="#C4C0B8", linewidth=0.45, zorder=1, aspect="auto")
    except Exception:
        pass
    bg = df[(df["pi_p95_base"] <= 1.0) & np.isfinite(df["lon"])]
    hi = df[(df["pi_p95_base"] > 1.0) & np.isfinite(df["lon"])].sort_values("pi_p95_base")
    ax.scatter(bg["lon"], bg["lat"], s=6, c="#9AA3AC", alpha=0.45, edgecolors="none", zorder=2,
               label=f"Inland basins $\\Pi_{{p95}} \\leq 1$ (n = {len(bg):,})")
    sc = ax.scatter(
        hi["lon"], hi["lat"],
        c=np.log10(hi["pi_p95_base"]),
        s=np.clip(np.sqrt(hi["n_inland_bifs"]) * 6.2, 24, 195),
        cmap="magma_r", alpha=0.90, edgecolors="#1E242B", linewidths=0.45,
        vmin=0.0, vmax=1.4, zorder=3,
        label=f"Inland basins $\\Pi_{{p95}} > 1$ (n = {len(hi):,})"
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label(r"$\log_{10}(\Pi_{\mathrm{topo,p95}})$", fontsize=8.3, color=C_CHAR)
    cbar.outline.set_linewidth(0.6)

    for dom, txt, xytext in [
        ("AMAZ", "Amazon", (-30, -15)),
        ("GANG", "Ganges–\nBrahmaputra", (56, 5)),
        ("YANG", "Yangtze", (134, 18)),
        ("OB00", "Ob", (34, 62)),
        ("LENA", "Lena", (104, 54)),
        ("RIOD", "Paraguay–\nParaná", (-22, -46)),
        ("MACK", "Mackenzie", (-156, 45)),
    ]:
        sub = hi[hi["domain"] == dom]
        if len(sub) > 0:
            row = sub.iloc[0]
            ax.annotate(txt, xy=(row["lon"], row["lat"]), xytext=xytext,
                        arrowprops=dict(arrowstyle="-", color="#4A525D", lw=0.75),
                        fontsize=7.0, fontweight="bold", color=C_CHAR,
                        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.80))

    ax.set_xlim(-175, 180)
    ax.set_ylim(-55, 78)
    ax.set_aspect("auto")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"a   Global inland corridors ({len(df):,} basins, {df['n_inland_bifs'].sum():,} loops)", loc="left", pad=10)
    ax.grid(True, alpha=0.35, ls=":")
    ax.legend(loc="lower left", fontsize=7.2)

    # Panel (b): Hydraulic Sensitivity Band
    ax = fig.add_subplot(gs[1])
    big = df[df["max_area_km2"] > 50000.0]
    exc = (1.0 - np.arange(1, len(big) + 1) / len(big)) * 100.0
    v_cons = np.maximum(np.sort(big["pi_p95_cons"].values), 1e-3)
    v_base = np.maximum(np.sort(big["pi_p95_base"].values), 1e-3)
    v_steep = np.maximum(np.sort(big["pi_p95_steep"].values), 1e-3)
    ax.fill_betweenx(exc, v_cons, v_steep, color=C_OCHR, alpha=0.20, label="Hydraulic envelope")
    ax.semilogx(v_cons, exc, "--", color=C_NAVY, lw=1.7, label=f"Conservative (>1: {(v_cons>1).mean()*100:.0f}%)")
    ax.semilogx(v_base, exc, "-", color=C_RUST, lw=2.4, label=f"Baseline (>1: {(v_base>1).mean()*100:.0f}%)")
    ax.semilogx(v_steep, exc, ":", color=C_TEAL, lw=2.0, label=f"Low-D (>1: {(v_steep>1).mean()*100:.0f}%)")
    ax.axvline(1.0, color=C_RUST, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlim(1e-2, 40)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Corridor 95th-pct $\Pi_{\mathrm{topo}}$")
    ax.set_ylabel("Exceedance across large basins (%)")
    ax.set_title(f"b   Large basins ($> 5 \\times 10^4$ km$^2$, n = {len(big)})", loc="left", pad=10)
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="upper right", fontsize=7.2)

    # Panel (c): Top-10 DEM-Constrained Critical Flood Durations tau_bif (hr)
    ax = fig.add_subplot(gs[2])
    df_d = df_dem.sort_values("tau_bif_max_DEM_hr").reset_index(drop=True)
    y = np.arange(len(df_d))
    for i, row in df_d.iterrows():
        ax.plot([row["tau_bif_p95_DEM_hr"], row["tau_bif_max_DEM_hr"]], [i, i], "-", color="#B0B8C1", lw=2.2, zorder=2)
    ax.scatter(df_d["tau_bif_p95_DEM_hr"], y, s=54, color=C_RUST, edgecolors="white", lw=0.6, zorder=4, label=r"Corridor $\tau_{\mathrm{bif,p95}}^{\mathrm{DEM}}$ (hr)")
    ax.scatter(df_d["tau_bif_max_DEM_hr"], y, s=56, color=C_PLUM, marker="D", edgecolors="white", lw=0.6, zorder=4, label=r"Hotspot $\tau_{\mathrm{bif,max}}^{\mathrm{DEM}}$ (hr)")
    ax.axvline(24.0, color=C_NAVY, ls="--", lw=1.1, alpha=0.75, label="24-hr storm scale")
    ax.set_xlim(0, 104)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['basin'].replace('Ganges-Brahmaputra','Ganges–Brah.').replace('Paraguay-Parana','Paraguay–Par.')}" for _, r in df_d.iterrows()], fontsize=7.8)
    ax.set_xlabel(r"DEM-constrained spreading timescale $\tau_{\mathrm{bif}}$ (hr)")
    ax.set_title("c   Top-10 DEM spreading timescales", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.55)
    ax.legend(loc="lower right", fontsize=7.1)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_global_grit_corridors.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out)


def make_fig5():
    df_sum = pd.read_csv(_os.path.join(DATA_DIR, "stage3_usgs_2008_2024_summary.csv"))
    df_ev = pd.read_csv(_os.path.join(DATA_DIR, "stage3_usgs_2008_2024_events.csv"))

    dlag_map = {}
    for r, g in df_ev.groupby("reach"):
        q50 = g["Qp_up"].median()
        dlag_map[r] = g[g["Qp_up"] > q50]["lag_hr"].median() - g[g["Qp_up"] <= q50]["lag_hr"].median()
    df_sum["dlag_hr"] = df_sum["reach"].map(dlag_map)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), dpi=300, facecolor="white")

    # Panel (a): Clean Diverging Bar Chart
    ax = axes[0]
    df_p = df_sum.sort_values(["type", "stage_jump_dvar"], ascending=[False, True]).reset_index(drop=True)
    cols = [C_RUST if t == "Braided" else C_NAVY for t in df_p["type"]]
    y = np.arange(len(df_p))
    ax.barh(y, df_p["stage_jump_dvar"], color=cols, height=0.58, alpha=0.88, edgecolor="none")
    ax.axvline(0.0, color=C_CHAR, ls="-", lw=1.0)
    for i, (val, rtype) in enumerate(zip(df_p["stage_jump_dvar"], df_p["type"])):
        ha = "left" if val > 0 else "right"
        dx = 10 if val > 0 else -10
        ax.text(val + dx, i, f"{val:+.1f}", va="center", ha=ha, fontsize=7.6, fontweight="bold",
                color=C_RUST if rtype == "Braided" else C_NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r} (n = {n})" for r, n in zip(df_p["reach"], df_p["n_events"])], fontsize=7.8)
    ax.set_xlim(-205, 595)
    ax.set_xlabel(r"Stage shift $\Delta\sigma^2_{\mathrm{hi}} - \Delta\sigma^2_{\mathrm{lo}}$ (hr$^2$ per 100 km)")
    br_j = df_sum[df_sum["type"] == "Braided"]["stage_jump_dvar"].values
    tr_j = df_sum[df_sum["type"] == "Single-Tree"]["stage_jump_dvar"].values
    _, p_r = mannwhitneyu(br_j, tr_j, alternative="greater")
    ax.set_title(f"a   11 corridors ({len(df_ev)} waves; reach $p = {p_r:.3f}$)", loc="left", pad=10)
    ax.grid(True, axis="x", alpha=0.55)

    # Panel (b): Clean Paired Slopegraph
    ax = axes[1]
    for _, row in df_sum.iterrows():
        col = C_RUST if row["type"] == "Braided" else C_NAVY
        ls = "-" if row["type"] == "Braided" else "--"
        y0_c = np.clip(row["dvar_lo_Q"], -185.0, 175.0)
        y1_c = np.clip(row["dvar_hi_Q"], -185.0, 175.0)
        if row["reach"] == "Niobrara (NE)":
            # Draw explicit broken-axis upper band so off-scale +260 -> +741 is visually transparent
            ax.axhline(182.0, color="#9AA3AC", ls=":", lw=0.9)
            ax.plot([-0.14, -0.09], [178.0, 186.0], color="#4A525D", lw=1.1, clip_on=False)
            ax.plot([-0.14, -0.09], [183.0, 191.0], color="#4A525D", lw=1.1, clip_on=False)
            ax.plot([0, 1], [199, 216], ls=":", color=col, lw=1.8, marker="^", ms=5.5)
            ax.text(0.50, 189, "Niobrara (+260 to +741, off-scale)", ha="center", va="center", fontsize=6.7, color=col, fontweight="bold")
        else:
            ax.plot([0, 1], [y0_c, y1_c], ls=ls, color=col, lw=2.0, marker="o", ms=5.2, markeredgecolor="white", markeredgewidth=0.6)
            lbl = row["reach"].split(" (")[0]
            dy = {"Wabash": 5.0, "Tanana": -4.0, "Susquehanna": 5.0, "Platte Lower": 5.0,
                  "Yukon Flats": -6.0, "Potomac": -5.0, "James": 4.0, "Colorado": -2.0}.get(lbl, 0.0)
            ax.text(1.04, y1_c + dy, lbl, va="center", fontsize=7.3, color=col, fontweight="bold")
    ax.axhline(0.0, color=C_MUTE, ls=":", lw=1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Moderate floods\n($Q_p \\leq Q_{50}$)", "High floods\n($Q_p > Q_{50}$)"], fontsize=8.5)
    ax.set_xlim(-0.12, 1.54)
    ax.set_ylim(-195, 224)
    ax.set_yticks([-150, -100, -50, 0, 50, 100, 150, 205])
    ax.set_yticklabels(["−150", "−100", "−50", "0", "50", "100", "150", ">250"])
    ax.set_ylabel(r"Observed broadening $\Delta\sigma^2_{\mathrm{obs}}$ (hr$^2$ per 100 km)")
    ax.set_title("b   Moderate to high flood trajectories", loc="left", pad=10)
    ax.plot([], [], "-", color=C_RUST, lw=2.0, label="Braided (4/5 upward)")
    ax.plot([], [], "--", color=C_NAVY, lw=2.0, label="Single-threaded (5/6 downward)")
    ax.legend(loc="lower left", fontsize=7.4)
    ax.grid(True, axis="y", alpha=0.55)

    # Panel (c): Editorial Bubble Attribution Quadrant Plane (size ~ sqrt(n))
    ax = axes[2]
    ax.axvspan(-32, 3, ymin=0.40, ymax=1.0, color="#FDF3EE", alpha=0.80)   # Q-II Braided
    ax.axvspan(3, 29, ymin=0.40, ymax=1.0, color="#EFF7F5", alpha=0.85)    # Q-I Floodplain
    ax.axvspan(-32, 3, ymin=0.0, ymax=0.40, color="#EFF4F9", alpha=0.80)   # Q-III Single-thread
    ax.axhline(0.0, color=C_CHAR, ls="--", lw=0.9, alpha=0.75)
    ax.axvline(0.0, color=C_CHAR, ls="--", lw=0.9, alpha=0.75)

    offsets = {
        "Niobrara (NE)": (2.2, -16),
        "Tanana (AK)": (2.0, 12),
        "Platte Upper (NE)": (-1.5, 22),
        "Platte Lower (NE)": (2.2, 10),
        "Yukon Flats (AK)": (-2.0, -24),
        "Wabash (IN)": (-14.5, 22),
        "Delaware (NY/NJ)": (2.2, -16),
        "Susquehanna (PA)": (-15.5, -28),
        "Potomac (MD/WV)": (-14.0, -14),
        "James (VA)": (2.4, -7),
        "Colorado (CO/UT)": (2.4, -5),
    }
    for _, row in df_sum.iterrows():
        col = C_RUST if row["type"] == "Braided" else (C_TEAL if row["reach"] == "Wabash (IN)" else C_NAVY)
        mk = "D" if row["type"] == "Braided" else "o"
        sz = np.clip(np.sqrt(row["n_events"]) * 18, 45, 190)
        x_val = row["dlag_hr"]
        y_val = 218.0 if row["reach"] == "Niobrara (NE)" else row["stage_jump_dvar"]
        ax.scatter([x_val], [y_val], s=sz, c=col, marker=mk, alpha=0.88, edgecolors="white", lw=1.0, zorder=5)
        lbl = row["reach"].split(" (")[0]
        if lbl == "Niobrara":
            lbl = "Niobrara (+481, n = 6)"
        dx, dy = offsets.get(row["reach"], (1.5, 5.0))
        if row["reach"] == "Niobrara (NE)":
            dx, dy = (2.2, -4.0)
        ax.text(x_val + dx, y_val + dy, lbl, fontsize=7.1, fontweight="bold", color=col)

    # Explicit broken y-axis between 175 and 218 (+481)
    ax.axhline(186.0, color="#9AA3AC", ls=":", lw=0.95, zorder=3)
    ax.plot([-32.8, -31.2], [181.0, 189.0], color="#4A525D", lw=1.2, clip_on=False)
    ax.plot([-32.8, -31.2], [187.0, 195.0], color="#4A525D", lw=1.2, clip_on=False)

    ax.text(-30.5, 238, "Q-II: Braided splitting\n(fast crest + broadening)", fontsize=7.2, fontweight="bold", color="#8F2D19")
    ax.text(6.5, 238, "Q-I: Floodplain storage\n(slow crest + broadening)", fontsize=7.2, fontweight="bold", color="#1D5C58")
    ax.text(-30.5, -150, "Q-III: Single-threaded\n(fast crest + narrowing)", fontsize=7.2, fontweight="bold", color=C_NAVY)

    ax.set_xlim(-32, 29)
    ax.set_ylim(-165, 268)
    ax.set_yticks([-150, -100, -50, 0, 50, 100, 150, 218])
    ax.set_yticklabels(["−150", "−100", "−50", "0", "50", "100", "150", "+481"])
    ax.set_xlabel(r"Peak-crest lag shift $\Delta\tau_{\mathrm{lag}} = \tau_{\mathrm{hi}} - \tau_{\mathrm{lo}}$ (hr)")
    ax.set_ylabel(r"Broadening shift $\Delta\sigma^2_{\mathrm{hi}} - \Delta\sigma^2_{\mathrm{lo}}$ (hr$^2$ per 100 km)")
    ax.set_title("c   Braiding vs. floodplain attribution", loc="left", pad=10)
    ax.grid(True, alpha=0.45)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_usgs_empirical_validation.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    # make_fig1()  # Superseded by revised vector Fig. 1
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("All publication figures generated at 300 DPI.")
