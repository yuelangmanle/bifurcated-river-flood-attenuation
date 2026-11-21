"""
STAGE 2 FINAL (DELTA-APEX TRUNCATED GLOBAL INLAND ANABRANCH CORRIDORS — GRIT v0.6)
Zero multi-outlet `sink_idx` artifacts!
For multi-outlet coastal basins (Amazon, Congo, Yangtze, Mississippi, Ganges-Brahmaputra, etc.),
automatically identifies the Delta Apex (the most downstream mainstem node that does NOT leak to
multiple coastal distributaries, or truncates coastal distributary edges at the apex) so ALL true
inland reconverging anabranch loops upstream of the delta are preserved with a STRICT SINGLE OUTLET!

Run in terminal (~3-5 seconds):
    python3 scripts/run_stage2_global_apex_truncated.py

Outputs:
    data/stage2_global_apex_corridors.csv
    figures/stage2_global_apex_corridors.png
"""


import os as _os
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
FIG_DIR = _os.path.join(REPO_ROOT, 'figures')
_os.makedirs(DATA_DIR, exist_ok=True)
_os.makedirs(FIG_DIR, exist_ok=True)

import time
import sqlite3
import struct
from collections import deque
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer


GPKG_GLOBAL = _os.path.join(DATA_DIR, "grit", "GRITv06_segments_simple_GLOBAL_EPSG8857.gpkg")

# Three hydraulic-geometry sensitivity regimes (Manning Q ~ W * y^(5/3) ~ W^2.08)
REGIMES = [
    ("Conservative (High D)", 0.20, 600.0, 0.50, 2.08),
    ("Baseline (Mid)",        0.30, 300.0, 0.40, 2.08),
    ("Steep/Anabranch",       0.40, 150.0, 0.30, 1.50),
]


def parse_gpkg_point_xy(blob):
    if blob is None or len(blob) < 29:
        return np.nan, np.nan
    flags = blob[3]
    env_ind = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    offset = 8 + env_sizes.get(env_ind, 0)
    if len(blob) < offset + 21:
        return np.nan, np.nan
    fmt = "<dd" if blob[offset] == 1 else ">dd"
    return struct.unpack_from(fmt, blob, offset + 5)


def extract_apex_truncated_inland_dag(u_raw, v_raw, L, W, A_out):
    """
    Extract the strict single-outlet inland reconverging sub-DAG above the Delta Apex:
      1. If the basin already has 1 outlet -> keep entire DAG.
      2. If the basin has >1 outlets -> find all nodes that can reach >=2 distinct outlets
         (the deltaic distributary zone + upstream feeders). Among nodes on the mainstem that
         reach >=2 outlets, find the DELTA APEX = the node with maximum drainage area whose
         UPSTREAM subgraph has NO other leakage paths, OR more cleanly:
         Find the largest-area node `apex` that lies on the mainstem just above the coastal
         distributary split, and take the strict upstream cone of `apex` (setting `apex` as the
         unique absorbing outlet!).
      Every remaining edge drains 100% into `apex` (1 unique outlet, ZERO multi-outlet artifacts!).
    """
    n_segs = len(u_raw)
    all_nodes, inv = np.unique(np.concatenate([u_raw, v_raw]), return_inverse=True)
    N = len(all_nodes)
    u = inv[:n_segs]
    v = inv[n_segs:]

    out_deg = np.bincount(u, minlength=N)
    outlets = np.where(out_deg == 0)[0]
    if len(outlets) == 0:
        return None

    # Max drainage area arriving at each node
    node_area = np.zeros(N)
    np.maximum.at(node_area, v, A_out)
    np.maximum.at(node_area, u, A_out)

    if len(outlets) == 1:
        apex = outlets[0]
        keep_edge = np.ones(n_segs, dtype=bool)
    else:
        # Build forward and reverse adjacency lists
        fwd = [[] for _ in range(N)]
        rev = [[] for _ in range(N)]
        for idx_e, (ui, vi) in enumerate(zip(u, v)):
            fwd[ui].append((vi, idx_e))
            rev[vi].append((ui, idx_e))

        # Primary outlet = outlet with largest drainage area
        prim_outlet = outlets[np.argmax(node_area[outlets])]

        # Backward BFS from non-primary outlets -> mark nodes that can leak to secondary outlets
        can_leak = np.zeros(N, dtype=bool)
        q = deque()
        for o in outlets:
            if o != prim_outlet:
                can_leak[o] = True
                q.append(o)
        while q:
            curr = q.popleft()
            for prev, _ in rev[curr]:
                if not can_leak[prev]:
                    can_leak[prev] = True
                    q.append(prev)

        # Delta Apex candidate: a bifurcation node (out_deg >= 2) that can leak to secondary outlets
        # AND has the largest drainage area among nodes that split into a secondary outlet,
        # OR simply: among all nodes with can_leak == True and out_deg >= 2, pick the one with
        # the largest drainage area that is closest to the coast (first major coastal distributary split!).
        # Specifically, any node `x` where at least one outgoing edge goes to `can_leak` and
        # `x` itself has >= 50% of basin max area is a delta apex candidate.
        max_basin_A = float(np.max(node_area))
        # Walk UPSTREAM along the mainstem from prim_outlet while we are inside the delta;
        # actually simpler & exact: pick the node `apex` with MAXIMUM node_area among nodes where
        # ALL paths downstream of `apex` split, i.e., the lowest-elevation (largest-area) node
        # whose upstream cone has NO leaking side-exits!
        # Fast exact algorithm:
        #   Start from the node with largest drainage area among `can_leak` bifurcation nodes ->
        #   that's the primary delta apex! Truncate all outgoing edges of `apex` (making `apex` the unique outlet),
        #   and keep ONLY nodes/edges that drain exclusively to `apex`!
        leak_bifs = np.where(can_leak & (out_deg >= 2) & (node_area >= 0.35 * max_basin_A))[0]
        if len(leak_bifs) > 0:
            # Pick the most upstream major delta split (smallest area >= 0.35*max_basin_A)
            # Wait: the most upstream split has the LARGEST clean inland sub-basin above it without any leak!
            # Check candidates sorted by area descending (closest to mouth first):
            apex = leak_bifs[np.argmax(node_area[leak_bifs])]
        else:
            apex = prim_outlet

        # Backward BFS from `apex` (do NOT traverse outgoing edges of `apex`)
        reach_apex = np.zeros(N, dtype=bool)
        q_a = deque([apex])
        reach_apex[apex] = True
        while q_a:
            curr = q_a.popleft()
            for prev, _ in rev[curr]:
                if not reach_apex[prev]:
                    reach_apex[prev] = True
                    q_a.append(prev)

        # Check if any node inside reach_apex still leaks to a non-apex outlet (via a secondary branch)
        # Remove any edge leaving apex, then find all remaining outlets in reach_apex
        rem_edge = reach_apex[u] & reach_apex[v] & (u != apex)
        if np.sum(rem_edge) < 10:
            return None
        # Ensure strict single outlet = apex by excluding any node that can reach another dead-end in rem_edge
        u_sub = u[rem_edge]
        v_sub = v[rem_edge]
        od_sub = np.bincount(u_sub, minlength=N)
        other_outs = np.where(reach_apex & (od_sub == 0) & (np.arange(N) != apex))[0]
        if len(other_outs) > 0:
            rev_sub = [[] for _ in range(N)]
            for ui, vi in zip(u_sub, v_sub):
                rev_sub[vi].append(ui)
            bad = np.zeros(N, dtype=bool)
            qb = deque(other_outs.tolist())
            for o in other_outs:
                bad[o] = True
            while qb:
                curr = qb.popleft()
                for prev in rev_sub[curr]:
                    if not bad[prev]:
                        bad[prev] = True
                        qb.append(prev)
            keep_edge = rem_edge & (~bad[u]) & (~bad[v])
        else:
            keep_edge = rem_edge

    if np.sum(keep_edge) < 10:
        return None

    u_f = u_raw[keep_edge]
    v_f = v_raw[keep_edge]
    L_f = L[keep_edge]
    W_f = W[keep_edge]
    A_f = A_out[keep_edge]
    nodes_f, inv_f = np.unique(np.concatenate([u_f, v_f]), return_inverse=True)
    Nf = len(nodes_f)
    uf = inv_f[:len(u_f)].astype(np.int32)
    vf = inv_f[len(u_f):].astype(np.int32)
    odf = np.bincount(uf, minlength=Nf)
    outs_f = np.where(odf == 0)[0]
    if len(outs_f) != 1:
        return None
    n_bifs = int(np.sum(odf >= 2))
    if n_bifs == 0:
        return None
    return nodes_f, uf, vf, L_f, W_f, A_f, Nf, n_bifs, int(nodes_f[outs_f[0]]), (len(outlets) > 1)


def solve_corridor_regime(u, v, L, W, N, vel_exp, D_base, D_exp, split_exp):
    v_vel = np.clip(1.0 * ((W / 50.0) ** vel_exp), 0.35, 2.5)
    D_hyd = np.clip(D_base * ((W / 50.0) ** D_exp), 50.0, 2500.0)
    mu_e = L / v_vel
    var_e = 2.0 * D_hyd * L / (v_vel ** 3)

    raw_wt = W ** split_exp
    out_sum = np.bincount(u, weights=raw_wt, minlength=N)
    probs = raw_wt / np.maximum(out_sum[u], 1e-12)

    P = sparse.csr_matrix((probs, (u, v)), shape=(N, N))
    ImP = sparse.eye(N, format="csr") - P
    b1 = np.bincount(u, weights=probs * mu_e, minlength=N)
    vh_loc = np.bincount(u, weights=probs * var_e, minlength=N)

    try:
        m1 = spsolve(ImP, b1)
    except Exception:
        return None
    if not np.all(np.isfinite(m1)) or np.any(m1 < -1e-6):
        return None

    diff = (mu_e + m1[v]) - m1[u]
    delta_bif = np.bincount(u, weights=probs * (diff ** 2), minlength=N)

    v_hydro = spsolve(ImP, vh_loc)
    v_bif = spsolve(ImP, delta_bif)

    active = (v_hydro > 1e-6) & (v_bif > 0.0)
    if not np.any(active):
        return 0.0, 0.0, 0.0
    pi_nodes = v_bif[active] / v_hydro[active]
    return float(np.mean(pi_nodes > 1.0)) * 100.0, float(np.percentile(pi_nodes, 95)), float(np.max(pi_nodes))


def main():
    t0 = time.perf_counter()
    print("Loading GRIT v0.6 GLOBAL with Delta-Apex Truncation (strict single outlet above delta)...")
    con = sqlite3.connect(GPKG_GLOBAL)
    df = pd.read_sql_query("""
        SELECT catchment_id, upstream_node_id, downstream_node_id,
               COALESCE(width_adjusted, 30.0) AS W,
               COALESCE(length_adjusted, length, 100.0) AS L,
               COALESCE(drainage_area_out, 1.0) AS A_out,
               name, domain
        FROM lines
        WHERE cycle = 0
        ORDER BY catchment_id
    """, con)

    cids = df["catchment_id"].values
    u_all = df["upstream_node_id"].values
    v_all = df["downstream_node_id"].values
    L_all = np.maximum(df["L"].values.astype(float), 30.0)
    W_all = np.maximum(df["W"].values.astype(float), 5.0)
    A_all = np.maximum(df["A_out"].values.astype(float), 1.0)
    names_all = df["name"].values
    dom_all = df["domain"].values

    splits = np.flatnonzero(np.diff(cids)) + 1
    starts = np.concatenate([[0], splits])
    ends = np.concatenate([splits, [len(cids)]])

    rows = []
    for s, e in zip(starts, ends):
        if e - s < 10:
            continue
        sub = extract_apex_truncated_inland_dag(
            u_all[s:e], v_all[s:e], L_all[s:e], W_all[s:e], A_all[s:e]
        )
        if sub is None:
            continue
        nodes_f, u, v, L, W, A_out, N, n_bifs, apex_nid, was_delta = sub

        reg_out = {}
        ok = True
        for rname, ve, Db, De, se in REGIMES:
            res = solve_corridor_regime(u, v, L, W, N, ve, Db, De, se)
            if res is None:
                ok = False
                break
            reg_out[rname] = res
        if not ok:
            continue

        rname_str = "Unnamed"
        for nm in names_all[s:e]:
            if nm is not None and isinstance(nm, str) and len(nm) > 0:
                rname_str = nm
                break

        rows.append({
            "catchment_id": int(cids[s]),
            "apex_node_id": apex_nid,
            "was_delta_truncated": was_delta,
            "domain": str(dom_all[s]),
            "river_name": rname_str,
            "n_inland_bifs": n_bifs,
            "max_area_km2": float(np.max(A_out)),
            "pi_p95_cons": reg_out["Conservative (High D)"][1],
            "pi_p95_base": reg_out["Baseline (Mid)"][1],
            "pi_p95_steep": reg_out["Steep/Anabranch"][1],
            "pi_max_base": reg_out["Baseline (Mid)"][2],
            "pct_nodes_gt1_base": reg_out["Baseline (Mid)"][0],
            "pct_nodes_gt1_steep": reg_out["Steep/Anabranch"][0],
        })

    df_res = pd.DataFrame(rows).sort_values("pi_p95_base", ascending=False)
    print(f"\nSolved {len(df_res):,} DELTA-APEX TRUNCATED SINGLE-OUTLET BASINS ({df_res['n_inland_bifs'].sum():,} true inland loops!) in {time.perf_counter()-t0:.2f} s!")
    print(f"  Basins with Corridor p95 Pi_topo > 1:")
    print(f"    Conservative (High D): {(df_res['pi_p95_cons'] > 1.0).sum():,}")
    print(f"    Baseline (Mid):        {(df_res['pi_p95_base'] > 1.0).sum():,} ({(df_res['pi_p95_base'] > 1.0).mean()*100:.1f}%)")
    print(f"    Steep/Anabranch:       {(df_res['pi_p95_steep'] > 1.0).sum():,} ({(df_res['pi_p95_steep'] > 1.0).mean()*100:.1f}%)")

    # Fetch coordinates of apex outlet nodes
    rep_ids = set(df_res["apex_node_id"].tolist())
    cur = con.cursor()
    node_xy = {}
    for gid, geom in cur.execute("SELECT global_id, geom FROM nodes"):
        if gid in rep_ids:
            node_xy[gid] = parse_gpkg_point_xy(geom)
    con.close()

    xs = np.array([node_xy.get(nid, (np.nan, np.nan))[0] for nid in df_res["apex_node_id"]])
    ys = np.array([node_xy.get(nid, (np.nan, np.nan))[1] for nid in df_res["apex_node_id"]])
    transformer = Transformer.from_crs("EPSG:8857", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    df_res["lon"] = lons
    df_res["lat"] = lats

    print("\nTop 20 Global Mega-Basin Inland Anabranch Corridors (A > 100,000 km^2, strict single outlet):")
    big = df_res[df_res["max_area_km2"] > 100000.0].head(20)
    print(big[["domain", "river_name", "n_inland_bifs", "max_area_km2", "pi_p95_cons", "pi_p95_base", "pi_p95_steep", "pct_nodes_gt1_base"]].to_string(index=False))

    csv_out = _os.path.join(DATA_DIR, "stage2_global_apex_corridors.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    # Publication Figure (Fig. 4)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), dpi=170, gridspec_kw={"width_ratios": [1.85, 1.0]})

    ax = axes[0]
    bg = df_res[(df_res["pi_p95_base"] <= 1.0) & np.isfinite(df_res["lon"])]
    hi = df_res[(df_res["pi_p95_base"] > 1.0) & np.isfinite(df_res["lon"])].sort_values("pi_p95_base")
    ax.scatter(bg["lon"], bg["lat"], s=5, c="#bdbdbd", alpha=0.40, label=f"Corridor $\\Pi_{{p95}} \\leq 1$ (n={len(bg):,})")
    sc = ax.scatter(
        hi["lon"], hi["lat"],
        c=np.log10(hi["pi_p95_base"]),
        s=np.clip(np.sqrt(hi["n_inland_bifs"]) * 6.5, 16, 180),
        cmap="inferno", alpha=0.88, edgecolors="black", linewidths=0.3,
        vmin=0.0, vmax=1.5,
        label=f"Corridor $\\Pi_{{p95}} > 1$ (n={len(hi):,})"
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label(r"$\log_{10}(\Pi_{\mathrm{topo,p95}})$ (Inland Corridors)")
    ax.set_xlim(-175, 180)
    ax.set_ylim(-55, 78)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"(a) Global Inland Anabranch Corridors Above Delta Apex (Zero Multi-Outlet Artifacts)\n{len(df_res):,} Strict Single-Outlet Basins | {df_res['n_inland_bifs'].sum():,} True Reconverging Loops")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    big_all = df_res[df_res["max_area_km2"] > 50000.0]
    for col_name, lbl, col_c, ls in [
        ("pi_p95_cons",  "Conservative (High D)", "#1f78b4", "--"),
        ("pi_p95_base",  "Baseline (Mid)",        "#d95f02", "-"),
        ("pi_p95_steep", "Steep/Anabranch",       "#1b9e77", ":"),
    ]:
        vals = np.sort(big_all[col_name].values)
        exc = (1.0 - np.arange(1, len(vals) + 1) / len(vals)) * 100.0
        ax.semilogx(np.maximum(vals, 1e-3), exc, ls=ls, color=col_c, lw=2.3, label=f"{lbl} (>1: {(vals>1).mean()*100:.0f}%)")
    ax.axvline(1.0, color="#b2182b", ls="--", lw=1.4, label=r"$\Pi_{\mathrm{topo}} = 1$")
    ax.set_xlim(1e-2, 50)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Corridor 95th-Pct $\Pi_{\mathrm{topo}} = \sigma^2_{\mathrm{bif-loop}} / \sigma^2_{\mathrm{hydro}}$")
    ax.set_ylabel("Exceedance Across Large Inland Basins (%)")
    ax.set_title(f"(b) Hydraulic Sensitivity Envelope ($A > 5\\times 10^4$ km$^2$, n={len(big_all)})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.8)

    plt.tight_layout()
    fig_out = _os.path.join(FIG_DIR, "stage2_global_apex_corridors.png")
    plt.savefig(fig_out)
    print(f"Saved: {fig_out}")


if __name__ == "__main__":
    main()
