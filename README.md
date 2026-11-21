# Bifurcated River Corridor Flood Attenuation

Code and processed summary datasets for:

**Beyond Single-Threaded Reaches: Flow Splitting, Backwater Delay, and Flood-Peak Attenuation in Bifurcated River Corridors**  
Zijie Deng*, Jiyang Yu, and Zhenyu Liao  
*Submitted to Journal of Hydrology*

## Repository Structure

```text
.
├── scripts/
│   ├── dag_dispersion_theory.py              # Absorbing Markov DAG variance decomposition & Green's function check
│   ├── stage1_universal_collapse.py          # 24 two-way backwater-coupled 1D diffusive-wave runs & mixture predictor
│   ├── stage1_fixed_corridor_pde.py          # Five-stage mechanism ablation benchmark
│   ├── run_stage2_global_apex_truncated.py   # Delta-Apex Truncation on global GRIT v0.6 (1,325 inland basins)
│   ├── stage2_top10_dem_slopes.py            # Copernicus GLO-90 (90 m) DEM slope constraints for top-10 corridors
│   ├── run_stage3_usgs_2008_2024.py          # 2008–2024 hourly USGS NWIS retrieval & 11-corridor wave analysis
│   └── generate_pub_figures.py               # Publication figure suite (300 DPI)
├── data/
│   ├── stage1_universal_collapse.csv         # 24 PDE simulations vs. Binomial–Gaussian mixture predictor
│   ├── stage2_global_apex_corridors.csv      # 1,325 inland single-outlet basins (32,588 reconverging loops)
│   ├── stage2_top10_dem_constrained.csv      # Top-10 continental DEM-constrained spreading timescales
│   ├── stage3_usgs_2008_2024_summary.csv     # Corridor-level hourly USGS stage-shift summary (11 corridors)
│   └── stage3_usgs_2008_2024_events.csv      # Event-level hourly USGS flood-wave moments (465 matched waves)
├── requirements.txt
└── LICENSE
```

## Public Data Sources

- **Global River Topology (GRIT v0.6):** Wortmann et al. (2025), Zenodo (`https://doi.org/10.5281/zenodo.11219313`)
- **Copernicus GLO-90 (90 m) DEM:** Open-Meteo Elevation API (`https://api.open-meteo.com/v1/elevation`)
- **USGS NWIS Instantaneous Values (`iv`, parameter `00060`):** `https://waterservices.usgs.gov`

## Usage

```bash
pip install -r requirements.txt
python3 scripts/dag_dispersion_theory.py
python3 scripts/stage1_universal_collapse.py
python3 scripts/run_stage2_global_apex_truncated.py
python3 scripts/stage2_top10_dem_slopes.py
python3 scripts/run_stage3_usgs_2008_2024.py
python3 scripts/generate_pub_figures.py
```

## License

MIT License. See `LICENSE` for details.
