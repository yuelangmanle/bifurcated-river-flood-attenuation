"""
STAGE 3 FINAL (EXPANDED 2008-2024 HOURLY USGS IV EMPIRICAL VALIDATION — 12 PAIRED CORRIDORS)
Tests Theorem 2's flagship empirical fingerprint:
    Sign reversal of stage-dependent flood-wave broadening d(Delta_sigma^2_obs) / dQ
    between 6 natural braided/anabranching corridors vs. 6 single-threaded tree control corridors!

Features:
  - Resumable local disk caching in data/usgs_cache/ (safe to Ctrl+C & resume!)
  - Automatic retries with backoff for USGS NWIS Instantaneous Values (iv) API
  - Non-parametric Mann-Whitney U significance test on Braided vs. Single-Tree stage shifts

Run in your own Mac terminal (0 AI quota cost, ~15-25 min first run, <5 s cached re-run):
    python3 scripts/run_stage3_usgs_2008_2024.py

Outputs:
    data/stage3_usgs_2008_2024_events.csv
    data/stage3_usgs_2008_2024_summary.csv
    figures/stage3_usgs_2008_2024_validation.png
"""


import os as _os
REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(REPO_ROOT, 'data')
FIG_DIR = _os.path.join(REPO_ROOT, 'figures')
_os.makedirs(DATA_DIR, exist_ok=True)
_os.makedirs(FIG_DIR, exist_ok=True)

import os
import io
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CACHE_DIR = _os.path.join(DATA_DIR, "usgs_cache")
FALLBACK_CACHE_DIR = _os.path.join(DATA_DIR, "usgs_cache_legacy")
os.makedirs(CACHE_DIR, exist_ok=True)

# 12 Paired Corridors (6 Braided/Anabranching + 6 Single-Threaded Tree Controls)
PAIRS = [
    # --- Braided / Multi-Thread Reconverging Corridors ---
    ("Tanana (AK)",       "Braided",     "15485500", "15515500", 105.0,  72),
    ("Platte Upper (NE)", "Braided",     "06768000", "06770500", 108.0,  96),
    ("Platte Lower (NE)", "Braided",     "06770500", "06774000", 102.0,  96),
    ("Susitna (AK)",      "Braided",     "15292000", "15292780",  95.0,  72),
    ("Yukon Flats (AK)",  "Braided",     "15356000", "15453500", 550.0, 240),
    ("Niobrara (NE)",     "Braided",     "06461500", "06465500", 185.0, 120),
    # --- Single-Threaded Tree Control Corridors ---
    ("Susquehanna (PA)",  "Single-Tree", "01536500", "01540500",  88.0,  60),
    ("Delaware (NY/NJ)",  "Single-Tree", "01428500", "01438500",  68.0,  48),
    ("Potomac (MD/WV)",   "Single-Tree", "01610000", "01613000",  65.0,  48),
    ("James (VA)",        "Single-Tree", "02024752", "02026000",  78.0,  48),
    ("Salmon (ID)",       "Single-Tree", "13302500", "13317000", 290.0, 120),
    ("Wabash (IN)",       "Single-Tree", "03335500", "03341500", 150.0,  96),
    ("Colorado (CO/UT)",  "Single-Tree", "09095500", "09163500",  85.0,  60),
]

# 2-year chunks from 2008 to 2024 (17 years of high-frequency USGS IV!)
CHUNKS = [
    ("2008-01-01", "2009-12-31"),
    ("2010-01-01", "2011-12-31"),
    ("2012-01-01", "2013-12-31"),
    ("2014-01-01", "2015-12-31"),
    ("2016-01-01", "2017-12-31"),
    ("2018-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2024-12-31"),
]


def _fetch_one_chunk(site_id, s_dt, e_dt):
    url = (
        f"https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site_id}"
        f"&startDT={s_dt}&endDT={e_dt}&parameterCd=00060&siteStatus=all"
    )
    try:
        proc = subprocess.run(
            ["curl", "-sS", "-L", "--retry", "3", "--retry-delay", "2", "--max-time", "90", url],
            capture_output=True, text=True, check=False
        )
        txt = proc.stdout
        if proc.returncode == 0 and len(txt) > 200:
            lines = [ln for ln in txt.splitlines() if not ln.startswith("#")]
            if len(lines) >= 3:
                df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", low_memory=False)
                q_cols = [c for c in df.columns if c.endswith("_00060")]
                if q_cols:
                    q = pd.to_numeric(df[q_cols[0]], errors="coerce") * 0.0283168466
                    dt_idx = pd.to_datetime(df["datetime"], errors="coerce")
                    s = pd.Series(q.values, index=dt_idx).dropna()
                    if len(s) > 0:
                        return s.resample("1h").mean()
    except Exception as e:
        print(f"    [WARN] {site_id} {s_dt}: {e}", flush=True)
    return None


def fetch_usgs_hourly_cached(site_id, session=None):
    """Fetch 2008-2024 USGS IV data in parallel via curl (immune to Python LibreSSL SSLEOFError) with local CSV caching."""
    cache_file = os.path.join(CACHE_DIR, f"usgs_iv_1h_{site_id}_2008_2024.csv")
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 10000:
        df_c = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df_c["Q_m3s"]
    fb_file = os.path.join(FALLBACK_CACHE_DIR, f"usgs_iv_1h_{site_id}_2008_2024.csv")
    if os.path.exists(fb_file) and os.path.getsize(fb_file) > 10000:
        df_c = pd.read_csv(fb_file, index_col=0, parse_dates=True)
        return df_c["Q_m3s"]

    print(f"    Downloading {site_id} (8 chunks in parallel via curl)...", flush=True)
    series_list = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_fetch_one_chunk, site_id, s_dt, e_dt) for s_dt, e_dt in CHUNKS]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                series_list.append(res)

    if not series_list:
        return pd.Series(dtype=float)

    full = pd.concat(series_list).sort_index()
    full = full[~full.index.duplicated(keep="first")].dropna()
    if len(full) > 1000:
        pd.DataFrame({"Q_m3s": full}).to_csv(cache_file)
    return full


def extract_hourly_wave_moments(q_up, q_dn, L_km, max_lag_hr):
    df = pd.DataFrame({"up": q_up, "dn": q_dn}).dropna()
    if len(df) < 8000:
        return pd.DataFrame()

    up = df["up"].values
    dn = df["dn"].values
    idx_dt = df.index
    n = len(df)

    thresh = np.quantile(up, 0.88)
    win_half = 96
    sep_hr = 336

    cands = []
    for i in range(win_half + 24, n - win_half - max_lag_hr - 24):
        if up[i] >= thresh and up[i] == np.max(up[i - 24:i + 25]):
            cands.append(i)

    used = np.zeros(n, dtype=bool)
    events = []
    for i in sorted(cands, key=lambda k: up[k], reverse=True):
        if np.any(used[max(0, i - sep_hr):min(n, i + sep_hr)]):
            continue
        used[i] = True

        seg = dn[i + 2:i + max_lag_hr + 1]
        if len(seg) == 0:
            continue
        lag_hr = int(np.argmax(seg)) + 2
        j = i + lag_hr

        w_up = up[i - win_half:i + win_half + 1]
        w_dn = dn[j - win_half:j + win_half + 1]
        ex_up = np.maximum(w_up - np.percentile(w_up, 10), 0.0)
        ex_dn = np.maximum(w_dn - np.percentile(w_dn, 10), 0.0)

        vol_up = np.sum(ex_up)
        vol_dn = np.sum(ex_dn)
        if vol_up <= 1e-3 or vol_dn <= 1e-3:
            continue

        vol_ratio = vol_dn / vol_up
        if not (0.50 <= vol_ratio <= 1.50):
            continue

        t_rel = np.arange(-win_half, win_half + 1, dtype=float)
        p_up = ex_up / vol_up
        p_dn = ex_dn / vol_dn
        m1_up = float(np.sum(t_rel * p_up))
        var_up = float(np.sum(((t_rel - m1_up) ** 2) * p_up))
        m1_dn = float(np.sum(t_rel * p_dn))
        var_dn = float(np.sum(((t_rel - m1_dn) ** 2) * p_dn))

        dvar_per100 = (var_dn - var_up) / (L_km / 100.0)
        peak_ratio_norm = (np.max(ex_dn) / vol_dn) / (np.max(ex_up) / vol_up)

        events.append({
            "date": idx_dt[i],
            "Qp_up": float(up[i]),
            "lag_hr": lag_hr,
            "vol_ratio": vol_ratio,
            "dvar_per100km": dvar_per100,
            "peak_ratio_norm": peak_ratio_norm,
        })
    return pd.DataFrame(events)


def main():
    t0 = time.time()
    print("=== STAGE 3 FINAL: 2008-2024 HOURLY USGS IV VALIDATION (12 CORRIDORS) ===", flush=True)
    session = None

    summary_rows = []
    all_ev_list = []

    for name, rtype, s_up, s_dn, L_km, max_lag in PAIRS:
        print(f"[{rtype:11s}] {name:18s} ({s_up} -> {s_dn}, {L_km:.0f} km)...", flush=True)
        q_up = fetch_usgs_hourly_cached(s_up, session)
        q_dn = fetch_usgs_hourly_cached(s_dn, session)
        ev = extract_hourly_wave_moments(q_up, q_dn, L_km, max_lag)
        print(f"  -> Matched {len(ev)} clean flood waves (up={len(q_up):,}h, dn={len(q_dn):,}h)", flush=True)
        if len(ev) < 6:
            continue

        ev["reach"] = name
        ev["type"] = rtype
        all_ev_list.append(ev)

        q50 = ev["Qp_up"].median()
        dvar_lo = ev[ev["Qp_up"] <= q50]["dvar_per100km"].median()
        dvar_hi = ev[ev["Qp_up"] > q50]["dvar_per100km"].median()
        stage_jump = dvar_hi - dvar_lo
        summary_rows.append({
            "reach": name,
            "type": rtype,
            "L_km": L_km,
            "n_events": len(ev),
            "med_lag_hr": ev["lag_hr"].median(),
            "dvar_lo_Q": dvar_lo,
            "dvar_hi_Q": dvar_hi,
            "stage_jump_dvar": stage_jump,
            "med_peak_ratio_norm": ev["peak_ratio_norm"].median(),
        })
        print(f"     Low-Q dVar={dvar_lo:+6.1f} -> High-Q dVar={dvar_hi:+6.1f} | Stage Jump={stage_jump:+6.1f} hr^2/100km", flush=True)

    df_sum = pd.DataFrame(summary_rows)
    df_ev = pd.concat(all_ev_list, ignore_index=True)
    df_sum.to_csv(_os.path.join(DATA_DIR, "stage3_usgs_2008_2024_summary.csv"), index=False)
    df_ev.to_csv(_os.path.join(DATA_DIR, "stage3_usgs_2008_2024_events.csv"), index=False)

    print("\n=== SUMMARY TABLE (2008-2024) ===", flush=True)
    print(df_sum.to_string(index=False), flush=True)

    br = df_sum[df_sum["type"] == "Braided"]["stage_jump_dvar"].values
    tr = df_sum[df_sum["type"] == "Single-Tree"]["stage_jump_dvar"].values
    if len(br) >= 2 and len(tr) >= 2:
        stat, pval = mannwhitneyu(br, tr, alternative="greater")
        print(f"\nMann-Whitney U test (Braided stage jump > Single-Tree): U={stat:.1f}, p={pval:.4f}", flush=True)

    # Plot 2-Panel Publication Figure (Fig. 5 & Fig. 6)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=165)

    ax = axes[0]
    df_plot = df_sum.sort_values(["type", "stage_jump_dvar"], ascending=[True, False])
    cols = ["#d95f02" if t == "Braided" else "#1f78b4" for t in df_plot["type"]]
    y = np.arange(len(df_plot))
    ax.barh(y, df_plot["stage_jump_dvar"], color=cols, alpha=0.88, edgecolor="black", lw=0.6)
    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r} (n={n})" for r, n in zip(df_plot["reach"], df_plot["n_events"])], fontsize=8.5)
    ax.set_xlabel(r"Stage-Dependent Broadening Shift $\Delta\sigma^2_{\mathrm{hi}} - \Delta\sigma^2_{\mathrm{lo}}$ (hr$^2$/100 km)")
    ax.set_title("(a) Stage-Dependent Sign Reversal (2008–2024 Hourly USGS)\nOrange = Braided Anabranch | Blue = Single-Thread Tree")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(y, df_plot["dvar_hi_Q"], color=cols, alpha=0.88, edgecolor="black", lw=0.6)
    ax.axvline(0.0, color="black", ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(["" for _ in y])
    ax.set_xlabel(r"High-Flood Wave Broadening $\Delta\sigma^2_{\mathrm{hi}}$ (hr$^2$/100 km)")
    ax.set_title("(b) High-Flood Wave Broadening Magnitude\nAnabranch Sill Activation Amplifies Dispersion")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig_out = _os.path.join(FIG_DIR, "stage3_usgs_2008_2024_validation.png")
    plt.savefig(fig_out)
    print(f"\nSaved figure: {fig_out} | Total elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
