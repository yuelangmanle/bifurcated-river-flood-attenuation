"""
UPGRADE 1: REAL COPERNICUS 90m DEM BED SLOPES ON TOP-10 GLOBAL CONTINENTAL BASINS IN GRIT v0.6
1. Extracts mainstem node coordinates (lon, lat) and cumulative mainstem length from GRIT v0.6
   for the Top-10 global inland anabranch basins (Amazon, Ob, Ganges-Brahmaputra, Yangtze,
   Kolyma, Niger, Lena, Mackenzie, Yenisei, Paraguay-Parana).
2. Fetches real Copernicus GLO-90 DEM elevations via Open-Meteo Elevation API (100 pts/request).
3. Computes real reach-averaged mainstem slope S0_DEM = (z_up - z_dn) / L_mainstem.
4. Solves O(|V|+|E|) matrix dispersion with real DEM-constrained Manning hydraulics:
     y = 0.25 * W^0.65 (hydraulic geometry depth),
     v = (1/n) * y^(2/3) * S0_DEM^(1/2) (Manning velocity, n in [0.028, 0.038]),
     D = v * y / (2 * S0_DEM) (Hayami diffusive-wave diffusivity)
   narrowing the synthetic 3-regime uncertainty envelope onto physical DEM-constrained bounds!

Outputs:
  data/stage2_top10_dem_constrained.csv
"""

import os as _os
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
FIG_DIR = _os.path.join(REPO_ROOT, 'figures')
_os.makedirs(DATA_DIR, exist_ok=True)
_os.makedirs(FIG_DIR, exist_ok=True)

import json, sqlite3, struct, urllib.request
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pyproj import Transformer
import sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from run_stage2_global_apex_truncated import GPKG_GLOBAL, parse_gpkg_point_xy, extract_apex_truncated_inland_dag

TOP10 = [
    ("Amazon",            "AMAZ", 109),
    ("Ob",                "OB00", 430002096),
    ("Ganges-Brahmaputra","GANG", 170000125),
    ("Yangtze",           "YANG", 570003875),
    ("Kolyma",            "CHU0", 110001033),
    ("Niger",             "CHAD", 100000001),
    ("Lena",              "LENA", 280000258),
    ("Mackenzie",         "MACK", 300002629),
    ("Yenisei",           "YENI", 580005302),
    ("Paraguay-Parana",   "RIOD", 460000011),
]


def fetch_copernicus_elevations(lats, lons):
    lat_str = ",".join(f"{x:.4f}" for x in lats)
    lon_str = ",".join(f"{x:.4f}" for x in lons)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (JOH-Hydrology-Research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return np.array(data["elevation"], dtype=float)


def solve_with_dem_slope(u, v, L, W, N, S0, n_manning=0.032):
    # Physically grounded Manning hydraulics with REAL DEM slope S0:
    # Bankfull depth y = 0.25 * W^0.65 (Leopold & Maddock / Neal et al. 2021)
    y = np.clip(0.25 * (W ** 0.65), 0.5, 25.0)
    v_vel = np.clip((1.0 / n_manning) * (y ** (2.0 / 3.0)) * np.sqrt(S0), 0.25, 3.5)
    # Kinematic celerity c_k = (5/3) * v_vel; Hayami diffusivity D = v*y / (2*S0)
    ck = (5.0 / 3.0) * v_vel
    D_hyd = np.clip((v_vel * y) / (2.0 * S0), 50.0, 15000.0)
    mu_e = L / ck
    var_e = 2.0 * D_hyd * L / (ck ** 3)

    raw_wt = W ** 2.08
    out_sum = np.bincount(u, weights=raw_wt, minlength=N)
    probs = raw_wt / np.maximum(out_sum[u], 1e-12)

    P = sparse.csr_matrix((probs, (u, v)), shape=(N, N))
    ImP = sparse.eye(N, format="csr") - P
    b1 = np.bincount(u, weights=probs * mu_e, minlength=N)
    vh_loc = np.bincount(u, weights=probs * var_e, minlength=N)
    m1 = spsolve(ImP, b1)
    diff = (mu_e + m1[v]) - m1[u]
    delta_bif = np.bincount(u, weights=probs * (diff ** 2), minlength=N)
    v_hydro = spsolve(ImP, vh_loc)
    v_bif = spsolve(ImP, delta_bif)

    active = (v_hydro > 1e-6) & (v_bif > 0.0)
    pi_nodes = v_bif[active] / v_hydro[active]
    tau_hr = np.sqrt(v_bif[active]) / 3600.0
    return float(np.percentile(pi_nodes, 95)), float(np.percentile(tau_hr, 95)), float(np.max(tau_hr))


def main():
    con = sqlite3.connect(GPKG_GLOBAL)
    transformer = Transformer.from_crs("EPSG:8857", "EPSG:4326", always_xy=True)

    cids_str = ",".join(str(c) for _, _, c in TOP10)
    df_lines = pd.read_sql_query(f"""
        SELECT catchment_id, upstream_node_id, downstream_node_id,
               COALESCE(width_adjusted, 30.0) AS W,
               COALESCE(length_adjusted, length, 100.0) AS L,
               COALESCE(drainage_area_out, 1.0) AS A_out,
               is_mainstem, domain
        FROM lines WHERE cycle = 0 AND catchment_id IN ({cids_str})
    """, con)

    # Pre-fetch needed mainstem node coordinates
    ms_nodes = set(df_lines[df_lines["is_mainstem"] == 1]["upstream_node_id"]).union(
        set(df_lines[df_lines["is_mainstem"] == 1]["downstream_node_id"])
    )
    node_xy = {}
    cur = con.cursor()
    for gid, geom in cur.execute(f"SELECT global_id, geom FROM nodes WHERE catchment_id IN ({cids_str})"):
        if gid in ms_nodes:
            node_xy[gid] = parse_gpkg_point_xy(geom)
    con.close()

    rows = []
    for name, dom, cid in TOP10:
        sub_df = df_lines[df_lines["catchment_id"] == cid]
        u_raw = sub_df["upstream_node_id"].values
        v_raw = sub_df["downstream_node_id"].values
        L_raw = np.maximum(sub_df["L"].values.astype(float), 30.0)
        W_raw = np.maximum(sub_df["W"].values.astype(float), 5.0)
        A_raw = np.maximum(sub_df["A_out"].values.astype(float), 1.0)

        ext = extract_apex_truncated_inland_dag(u_raw, v_raw, L_raw, W_raw, A_raw)
        nodes_f, u, v, L, W, A_out, N, n_bifs, apex_nid, _ = ext
        keep_set = set(nodes_f.tolist())

        # Extract mainstem reaches inside the truncated inland basin (A_out in [20%, 100%] of apex area)
        ms = sub_df[(sub_df["is_mainstem"] == 1) & (sub_df["upstream_node_id"].isin(keep_set)) & (sub_df["downstream_node_id"].isin(keep_set))].copy()
        max_A = ms["A_out"].max()
        ms_corridor = ms[ms["A_out"] >= 0.25 * max_A].sort_values("A_out", ascending=True)
        total_L_m = float(ms_corridor["L"].sum())

        # Sample 10 evenly spaced nodes along the lower-to-mid mainstem corridor
        idx_samp = np.linspace(0, len(ms_corridor) - 1, min(10, len(ms_corridor))).astype(int)
        samp_nids = ms_corridor.iloc[idx_samp]["upstream_node_id"].tolist()
        xs = [node_xy[nid][0] for nid in samp_nids if nid in node_xy]
        ys = [node_xy[nid][1] for nid in samp_nids if nid in node_xy]
        lons, lats = transformer.transform(xs, ys)
        elevs = fetch_copernicus_elevations(lats, lons)

        # Robust corridor slope = (z_up - z_dn) / L_corridor (using 90th-10th or top-bottom endpoints)
        dz = max(float(elevs[0] - elevs[-1]), 2.0)
        s0_dem = np.clip(dz / max(total_L_m, 1e5), 1.5e-5, 2.0e-3)

        # Evaluate tight Manning roughness band n in [0.028, 0.032, 0.038]
        pi_lo, tau95_lo, taumax_lo = solve_with_dem_slope(u, v, L, W, N, s0_dem, n_manning=0.038)
        pi_mid, tau95_mid, taumax_mid = solve_with_dem_slope(u, v, L, W, N, s0_dem, n_manning=0.032)
        pi_hi, tau95_hi, taumax_hi = solve_with_dem_slope(u, v, L, W, N, s0_dem, n_manning=0.028)

        rows.append({
            "basin": name, "domain": dom, "n_inland_bifs": n_bifs,
            "corridor_L_km": total_L_m / 1000.0,
            "z_up_m": float(elevs[0]), "z_dn_m": float(elevs[-1]),
            "S0_DEM_m_per_km": s0_dem * 1000.0,
            "Pi_p95_DEM_n038": pi_lo, "Pi_p95_DEM_n032": pi_mid, "Pi_p95_DEM_n028": pi_hi,
            "tau_bif_p95_DEM_hr": tau95_mid, "tau_bif_max_DEM_hr": taumax_mid,
        })
        print(f"  {name:18s} | L={total_L_m/1000:6.0f}km | z={elevs[0]:5.1f}->{elevs[-1]:4.1f}m | "
              f"S0_DEM={s0_dem*1000:.3f} m/km | Pi_p95_DEM={pi_mid:.2f} [{pi_lo:.2f}-{pi_hi:.2f}] | "
              f"tau_p95={tau95_mid:.1f}h, tau_max={taumax_mid:.1f}h", flush=True)

    df_out = pd.DataFrame(rows)
    out_csv = _os.path.join(DATA_DIR, "stage2_top10_dem_constrained.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
