# Web verification record

Package assembly date: **2026-09-09**.

The following official logical authorities/services were checked against current provider pages during package assembly:

1. NSW Resources / Data.NSW — Mine Rehabilitation Data: https://data.nsw.gov.au/data/dataset/mine-rehabilitation
2. NSW Mine Rehabilitation ArcGIS REST service: https://dreags.aamgeocloud.com/arcgis/rest/services/Mine_Rehabilitation/Mine_Rehabilitation/MapServer
   - Rehabilitation layer ID 3: https://dreags.aamgeocloud.com/arcgis/rest/services/Mine_Rehabilitation/Mine_Rehabilitation/MapServer/3
   - Disturbance layer ID 4: https://dreags.aamgeocloud.com/arcgis/rest/services/Mine_Rehabilitation/Mine_Rehabilitation/MapServer/4
3. DEA Fractional Cover product authority: https://pid.geoscience.gov.au/dataset/ga/145498
   - Current knowledge page: https://knowledge.dea.ga.gov.au/data/product/dea-fractional-cover-landsat/
4. DEA Water Observations authority: https://doi.org/10.26186/146257
   - Current knowledge page: https://knowledge.dea.ga.gov.au/data/product/dea-water-observations-landsat/
5. SILO gridded data access: https://longpaddock.qld.gov.au/silo/gridded-data/
6. TERN/SLGA product authorities (stable DOI links):
   - SOC: https://doi.org/10.25919/ejhm-c070
   - pH(CaCl2): https://doi.org/10.25919/7320-hw30
   - Clay: https://doi.org/10.25919/hc4s-3130
   - Bulk density: https://doi.org/10.25919/gxyn-pd07
   - Available water capacity: https://doi.org/10.25919/4jwj-na34
7. Geoscience Australia SRTM-derived DEM authority: https://pid.geoscience.gov.au/dataset/ga/72759

### Frozen asset URL rule

The exact SILO/TERN/GA acquisition URLs in `data/EXACT_FROZEN_CONTEXT_SOURCE_ASSETS.csv` are **not reconstructed from memory**. They are copied from the accepted Stage5A `SOURCE_PROVENANCE_MANIFEST.json`. Product-level DOI/PID/landing pages are the preferred stable citation links; exact asset URLs are retained for reproducibility of the frozen acquisition.

The exact DEA STAC endpoint and product identifiers are copied from the frozen Stage2B code contract (`stage2b_local_stac_prescreen.py`, version `1.2.0-local-stac`).
