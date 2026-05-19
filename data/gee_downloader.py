"""
gee_downloader.py
=================
Multi-backend satellite data downloader for Sentinel-1 and Sentinel-2.

Supported backends (set data_backend in config.yaml):
  "gee"                — Google Earth Engine (free for research)
  "planetary_computer" — Microsoft Planetary Computer (completely free, no account)
  "sentinel_hub"       — Sentinel Hub / AWS (free trial)

The public interface is identical regardless of backend:
    dl = SatelliteDownloader(config_path="config.yaml")
    df = dl.extract_pixel_timeseries_wide(geometry_geojson, start_date, end_date)

GEE project ID
--------------
Only needed if data_backend = "gee".
Get a free GEE account at: https://earthengine.google.com
After signing up, run:  earthengine authenticate
Your project ID appears in the GEE Code Editor URL.

No GEE account?
---------------
Set data_backend: "planetary_computer" in config.yaml.
Microsoft Planetary Computer is completely free, no sign-up required.
Install:  pip install pystac-client stackstac planetary-computer
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band definitions (shared across backends)
# ---------------------------------------------------------------------------
S2_BANDS   = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_ALIASES = S2_BANDS  # Kept for backward compatibility
S1_BANDS   = ["VV", "VH"]
ALL_BANDS  = S2_BANDS + S1_BANDS + ["NDVI", "EVI", "NDWI", "LSWI", "RVI"]

_EPS = 1e-8

# ---------------------------------------------------------------------------
# Index computation (numpy, backend-independent)
# ---------------------------------------------------------------------------

def _compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute NDVI, EVI, NDWI, LSWI, RVI from band arrays."""
    nir   = bands.get("B08", bands.get("B8",  np.full(1, np.nan)))
    red   = bands.get("B04", bands.get("B4",  np.full(1, np.nan)))
    green = bands.get("B03", bands.get("B3",  np.full(1, np.nan)))
    blue  = bands.get("B02", bands.get("B2",  np.full(1, np.nan)))
    swir1 = bands.get("B11", np.full(1, np.nan))
    vv    = bands.get("VV",  np.full(1, np.nan))
    vh    = bands.get("VH",  np.full(1, np.nan))

    ndvi = (nir - red)   / (nir + red   + _EPS)
    evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + _EPS)
    ndwi = (green - nir) / (green + nir + _EPS)
    lswi = (nir - swir1) / (nir + swir1 + _EPS)
    rvi  = 4 * vh / (vv + vh + _EPS)

    return {"NDVI": ndvi, "EVI": evi, "NDWI": ndwi, "LSWI": lswi, "RVI": rvi}


def _month_range(start_date: str, end_date: str) -> List[Tuple[int, int]]:
    """Return list of (year, month) tuples between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    months = []
    cur = start
    while cur < end:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months

# ===========================================================================
# BACKEND 1 — Google Earth Engine
# ===========================================================================

class _GEEBackend:
    """GEE backend. Requires earthengine-api and a project ID."""

    def __init__(self, project_id: str, service_account: str = ""):
        self.project_id = project_id
        self.service_account = service_account
        self._init = False

    def _ee(self):
        import ee
        if not self._init:
            try:
                if self.service_account and self.service_account.endswith(".json"):
                    creds = ee.ServiceAccountCredentials(email=None, key_file=self.service_account)
                    ee.Initialize(creds, project=self.project_id)
                else:
                    ee.Initialize(project=self.project_id)
                self._init = True
                logger.info(f"GEE initialised | project={self.project_id}")
            except Exception as exc:
                raise RuntimeError(
                    f"GEE init failed: {exc}\n"
                    "Fix: run  earthengine authenticate  then set project_id in config.yaml\n"
                    "Or: set data_backend: 'planetary_computer' in config.yaml (no account needed)"
                ) from exc
        return ee

    def extract_pixel_timeseries_wide(
        self, geometry_geojson: dict, start_date: str, end_date: str,
        scale: int = 10, max_pixels: int = 500, orbit_pass: str = "DESCENDING",
    ) -> pd.DataFrame:
        ee = self._ee()
        geometry = ee.Geometry(geometry_geojson)
        months = _month_range(start_date, end_date)
        logger.info(f"GEE: downloading {len(months)} monthly composites...")

        all_bands_gee = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12",
                         "NDVI","EVI","NDWI","LSWI","VV","VH","RVI"]
        images, optical_masks = [], []

        dummy_s2 = ee.Dictionary({b: -999.0 for b in ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","NDVI","EVI","NDWI","LSWI"]}).toImage().mask(0)
        dummy_s1 = ee.Dictionary({b: -999.0 for b in ["VV","VH","RVI"]}).toImage().mask(0)

        for year, month in months:
            tag   = f"{year}_{month:02d}"
            m_start = f"{year}-{month:02d}-01"
            m_end_dt = (datetime(year, month, 1) + timedelta(days=32)).replace(day=1)
            m_end = m_end_dt.strftime("%Y-%m-%d")

            # Sentinel-2
            s2_coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(geometry).filterDate(m_start, m_end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
                  .map(self._mask_s2).map(self._add_s2_idx))
            s2 = ee.Image(ee.Algorithms.If(s2_coll.size().gt(0), s2_coll.median(), dummy_s2))

            # Sentinel-1
            s1_coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
                  .filterBounds(geometry).filterDate(m_start, m_end)
                  .filter(ee.Filter.eq("instrumentMode", "IW"))
                  .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
                  .select(["VV","VH"]).map(self._add_rvi))
            s1 = ee.Image(ee.Algorithms.If(s1_coll.size().gt(0), s1_coll.median(), dummy_s1))

            merged = s2.addBands(s1)
            renamed = merged.select(all_bands_gee).rename(
                [f"{b}_{tag}" for b in all_bands_gee])
            images.append(renamed)
            optical_masks.append(s2.select("NDVI").mask().rename(f"optical_mask_{tag}"))

        stacked = images[0]
        for img in images[1:]:
            stacked = stacked.addBands(img)
        mask_stack = optical_masks[0]
        for m in optical_masks[1:]:
            mask_stack = mask_stack.addBands(m)
        full = stacked.addBands(mask_stack).unmask(-999.0)

        sample = full.sample(region=geometry, scale=scale,
                             numPixels=max_pixels, seed=42, geometries=True)
        try:
            feats = sample.getInfo()["features"]
        except Exception as exc:
            raise RuntimeError(f"GEE sampling failed: {exc}") from exc

        rows = []
        for f in feats:
            p = f["properties"]
            p["longitude"] = f["geometry"]["coordinates"][0]
            p["latitude"]  = f["geometry"]["coordinates"][1]
            rows.append(p)

        df = pd.DataFrame(rows)
        # Normalise band names to match S2_BANDS aliases (B2 not B02)
        df = df.rename(columns={c: c for c in df.columns})
        logger.info(f"GEE: {df.shape[0]} pixels × {df.shape[1]} columns")
        return df

    @staticmethod
    def _mask_s2(image):
        import ee as _ee
        qa  = image.select("QA60")
        scl = image.select("SCL")
        qa_mask  = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
        scl_mask = scl.neq(1).And(scl.neq(2)).And(scl.neq(3)).And(
                   scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(qa_mask.And(scl_mask)).divide(10000).copyProperties(
            image, ["system:time_start"])

    @staticmethod
    def _add_s2_idx(image):
        nir, red, green, blue, swir1 = (image.select(b) for b in
            ["B8","B4","B3","B2","B11"])
        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        evi  = image.expression(
            "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
            {"NIR":nir,"RED":red,"BLUE":blue}).rename("EVI")
        ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
        lswi = nir.subtract(swir1).divide(nir.add(swir1)).rename("LSWI")
        return image.addBands([ndvi, evi, ndwi, lswi])

    @staticmethod
    def _add_rvi(image):
        vv, vh = image.select("VV"), image.select("VH")
        return image.addBands(vh.multiply(4).divide(vv.add(vh)).rename("RVI"))

# ===========================================================================
# BACKEND 2 — Microsoft Planetary Computer (FREE, no account needed)
# ===========================================================================

class _PlanetaryComputerBackend:
    """
    Microsoft Planetary Computer backend.
    Completely free. No GEE account or project ID needed.

    Install:
        pip install pystac-client stackstac planetary-computer rioxarray

    Data accessed:
        Sentinel-2 L2A  — sentinel-2-l2a collection
        Sentinel-1 GRD  — sentinel-1-grd collection
    """

    CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, subscription_key: str = ""):
        self.subscription_key = subscription_key

    def extract_pixel_timeseries_wide(
        self, geometry_geojson: dict, start_date: str, end_date: str,
        scale: int = 10, max_pixels: int = 500, orbit_pass: str = "DESCENDING",
    ) -> pd.DataFrame:
        try:
            import pystac_client
            import planetary_computer as pc
            import stackstac
            import rioxarray  # noqa: F401 — needed for .rio accessor
        except ImportError:
            raise ImportError(
                "Microsoft Planetary Computer backend requires extra packages.\n"
                "Run:  pip install pystac-client stackstac planetary-computer rioxarray"
            )

        catalog = pystac_client.Client.open(
            self.CATALOG_URL,
            modifier=pc.sign_inplace,
        )

        months = _month_range(start_date, end_date)
        logger.info(f"Planetary Computer: downloading {len(months)} monthly composites...")

        all_rows: List[pd.DataFrame] = []

        for year, month in months:
            tag     = f"{year}_{month:02d}"
            m_start = f"{year}-{month:02d}-01"
            m_end_dt = (datetime(year, month, 1) + timedelta(days=32)).replace(day=1)
            m_end   = m_end_dt.strftime("%Y-%m-%d")

            # ---- Sentinel-2 ----
            s2_df = self._fetch_s2_month(
                catalog, geometry_geojson, m_start, m_end, tag, scale, max_pixels
            )

            # ---- Sentinel-1 ----
            s1_df = self._fetch_s1_month(
                catalog, geometry_geojson, m_start, m_end, tag, scale, max_pixels, orbit_pass
            )

            # Merge on pixel coordinates
            if s2_df is not None and s1_df is not None:
                month_df = pd.merge(s2_df, s1_df, on=["longitude","latitude"], how="outer")
            elif s2_df is not None:
                month_df = s2_df
            elif s1_df is not None:
                month_df = s1_df
            else:
                logger.warning(f"  No data for {tag}, skipping.")
                continue

            all_rows.append(month_df)

        if not all_rows:
            raise RuntimeError("No data retrieved from Planetary Computer.")

        # Merge all months on pixel coordinates (wide format)
        df = all_rows[0]
        for mdf in all_rows[1:]:
            df = pd.merge(df, mdf, on=["longitude","latitude"], how="outer")

        logger.info(f"Planetary Computer: {df.shape[0]} pixels × {df.shape[1]} columns")
        return df

    def _fetch_s2_month(self, catalog, geojson, start, end, tag, scale, max_pixels):
        try:
            import stackstac, planetary_computer as pc
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=geojson,
                datetime=f"{start}/{end}",
                query={"eo:cloud_cover": {"lt": 80}},
            )
            items = list(search.items())
            if not items:
                logger.debug(f"  S2: no items for {tag}")
                return self._empty_s2_df(tag)

            signed = [pc.sign(i) for i in items]
            s2_bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]
            stack = stackstac.stack(
                signed, assets=s2_bands,
                resolution=scale, epsg=4326,
                bounds_latlon=self._bbox(geojson),
            ).median(dim="time").compute()

            return self._s2_stack_to_df(stack, tag, max_pixels)
        except Exception as exc:
            logger.warning(f"  S2 fetch failed for {tag}: {exc}")
            return self._empty_s2_df(tag)

    def _fetch_s1_month(self, catalog, geojson, start, end, tag, scale, max_pixels, orbit_pass):
        try:
            import stackstac, planetary_computer as pc
            search = catalog.search(
                collections=["sentinel-1-grd"],
                intersects=geojson,
                datetime=f"{start}/{end}",
            )
            items = [i for i in search.items()
                     if i.properties.get("sat:orbit_state","").upper() == orbit_pass.upper()]
            if not items:
                logger.debug(f"  S1: no items for {tag}")
                return None

            signed = [pc.sign(i) for i in items]
            stack = stackstac.stack(
                signed, assets=["VV","VH"],
                resolution=scale, epsg=4326,
                bounds_latlon=self._bbox(geojson),
            ).median(dim="time").compute()

            return self._s1_stack_to_df(stack, tag, max_pixels)
        except Exception as exc:
            logger.warning(f"  S1 fetch failed for {tag}: {exc}")
            return None

    @staticmethod
    def _bbox(geojson):
        from shapely.geometry import shape
        geom = shape(geojson)
        return geom.bounds  # (minx, miny, maxx, maxy)

    def _s2_stack_to_df(self, stack, tag, max_pixels):
        import xarray as xr
        rows = []
        lons = stack.x.values
        lats = stack.y.values

        band_map = {
            "B02":"B2","B03":"B3","B04":"B4","B05":"B5","B06":"B6",
            "B07":"B7","B08":"B8","B8A":"B8A","B11":"B11","B12":"B12",
        }

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                row = {"longitude": float(lon), "latitude": float(lat)}
                band_vals = {}
                for band in ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]:
                    try:
                        val = float(stack.sel(band=band).values[i, j]) / 10000.0
                        alias = band_map.get(band, band)
                        row[f"{alias}_{tag}"] = val
                        band_vals[alias] = val
                    except Exception:
                        row[f"{band_map.get(band,band)}_{tag}"] = np.nan

                # Compute indices
                idx = _compute_indices({k: np.array([v]) for k, v in band_vals.items()})
                for name, arr in idx.items():
                    row[f"{name}_{tag}"] = float(arr[0])

                # Optical mask (1=valid, 0=cloudy) — use SCL if available
                try:
                    scl = float(stack.sel(band="SCL").values[i, j])
                    row[f"optical_mask_{tag}"] = 1.0 if scl in [4,5,6,7,11] else 0.0
                except Exception:
                    row[f"optical_mask_{tag}"] = 1.0

                rows.append(row)
                if len(rows) >= max_pixels:
                    return pd.DataFrame(rows)

        return pd.DataFrame(rows)

    def _s1_stack_to_df(self, stack, tag, max_pixels):
        rows = []
        lons = stack.x.values
        lats = stack.y.values

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                row = {"longitude": float(lon), "latitude": float(lat)}
                try:
                    vv = float(stack.sel(band="VV").values[i, j])
                    vh = float(stack.sel(band="VH").values[i, j])
                    row[f"VV_{tag}"] = vv
                    row[f"VH_{tag}"] = vh
                    rvi = 4*vh / (vv + vh + _EPS)
                    row[f"RVI_{tag}"] = rvi
                except Exception:
                    row[f"VV_{tag}"] = np.nan
                    row[f"VH_{tag}"] = np.nan
                    row[f"RVI_{tag}"] = np.nan
                rows.append(row)
                if len(rows) >= max_pixels:
                    return pd.DataFrame(rows)

        return pd.DataFrame(rows)

    @staticmethod
    def _empty_s2_df(tag):
        return pd.DataFrame(columns=["longitude","latitude",
                                     f"NDVI_{tag}", f"optical_mask_{tag}"])

# ===========================================================================
# BACKEND 3 — Sentinel Hub (free trial, then paid)
# ===========================================================================

class _SentinelHubBackend:
    """
    Sentinel Hub backend (AWS-hosted).
    Free trial at: https://www.sentinel-hub.com
    Set client_id, client_secret, instance_id in config.yaml.
    """

    def __init__(self, client_id: str, client_secret: str, instance_id: str = ""):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.instance_id   = instance_id

    def extract_pixel_timeseries_wide(
        self, geometry_geojson: dict, start_date: str, end_date: str,
        scale: int = 10, max_pixels: int = 500, orbit_pass: str = "DESCENDING",
    ) -> pd.DataFrame:
        try:
            from sentinelhub import (
                SHConfig, BBox, CRS, DataCollection,
                SentinelHubRequest, MimeType, bbox_to_dimensions,
            )
        except ImportError:
            raise ImportError(
                "Sentinel Hub backend requires sentinelhub package.\n"
                "Run:  pip install sentinelhub"
            )

        config = SHConfig()
        config.sh_client_id     = self.client_id
        config.sh_client_secret = self.client_secret
        if self.instance_id:
            config.instance_id = self.instance_id

        from shapely.geometry import shape
        geom  = shape(geometry_geojson)
        bbox  = BBox(bbox=geom.bounds, crs=CRS.WGS84)
        size  = bbox_to_dimensions(bbox, resolution=scale)

        months = _month_range(start_date, end_date)
        logger.info(f"Sentinel Hub: downloading {len(months)} monthly composites...")

        all_rows: List[pd.DataFrame] = []
        for year, month in months:
            tag     = f"{year}_{month:02d}"
            m_start = f"{year}-{month:02d}-01"
            m_end_dt = (datetime(year, month, 1) + timedelta(days=32)).replace(day=1)
            m_end   = m_end_dt.strftime("%Y-%m-%d")

            evalscript = """
            //VERSION=3
            function setup(){return{input:[{bands:["B02","B03","B04","B08","B11","SCL"],
            units:"REFLECTANCE"}],output:{bands:7}}}
            function evaluatePixel(s){
              var ndvi=(s.B08-s.B04)/(s.B08+s.B04+1e-8);
              return[s.B02,s.B03,s.B04,s.B08,s.B11,ndvi,s.SCL];}
            """
            try:
                req = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(m_start, m_end),
                        mosaicking_order="leastCC",
                    )],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=bbox, size=size, config=config,
                )
                data = req.get_data()[0]  # (H, W, 7)
                rows = self._array_to_rows(data, bbox, size, tag, max_pixels)
                if rows:
                    all_rows.append(pd.DataFrame(rows))
            except Exception as exc:
                logger.warning(f"  SentinelHub S2 failed for {tag}: {exc}")

        if not all_rows:
            raise RuntimeError("No data retrieved from Sentinel Hub.")

        df = all_rows[0]
        for mdf in all_rows[1:]:
            df = pd.merge(df, mdf, on=["longitude","latitude"], how="outer")
        return df

    @staticmethod
    def _array_to_rows(data, bbox, size, tag, max_pixels):
        import numpy as np
        H, W, _ = data.shape
        lon_min, lat_min, lon_max, lat_max = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y
        lon_step = (lon_max - lon_min) / W
        lat_step = (lat_max - lat_min) / H
        rows = []
        for i in range(H):
            for j in range(W):
                lat = lat_max - i * lat_step
                lon = lon_min + j * lon_step
                b2, b3, b4, b8, b11, ndvi, scl = data[i, j]
                evi  = 2.5*(b8-b4)/(b8+6*b4-7.5*b2+1+1e-8)
                ndwi = (b3-b8)/(b3+b8+1e-8)
                lswi = (b8-b11)/(b8+b11+1e-8)
                rows.append({
                    "longitude": lon, "latitude": lat,
                    f"B2_{tag}": b2,  f"B3_{tag}": b3,  f"B4_{tag}": b4,
                    f"B8_{tag}": b8,  f"B11_{tag}": b11,
                    f"NDVI_{tag}": ndvi, f"EVI_{tag}": evi,
                    f"NDWI_{tag}": ndwi, f"LSWI_{tag}": lswi,
                    f"optical_mask_{tag}": 1.0 if int(scl) in [4,5,6,7,11] else 0.0,
                })
                if len(rows) >= max_pixels:
                    return rows
        return rows

# ===========================================================================
# PUBLIC INTERFACE — SatelliteDownloader (backend-agnostic)
# ===========================================================================

class SatelliteDownloader:
    """
    Backend-agnostic satellite data downloader.

    Reads data_backend from config.yaml and routes to the correct backend.
    The calling code never needs to know which backend is active.

    Usage
    -----
        dl = SatelliteDownloader(config_path="config.yaml")
        df = dl.extract_pixel_timeseries_wide(
            geometry_geojson=geojson_dict,
            start_date="2023-02-15",
            end_date="2024-02-15",
        )
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        backend_name = cfg.get("data_backend", "gee").lower().strip()

        if backend_name == "gee":
            project_id = cfg["gee"]["project_id"]
            if project_id == "your-gee-project-id" or not project_id:
                logger.warning(
                    "\n" + "="*60 +
                    "\nGEE project_id is not set in config.yaml.\n"
                    "Options:\n"
                    "  1. Get a free GEE account: https://earthengine.google.com\n"
                    "     Then run: earthengine authenticate\n"
                    "     Then set project_id in config.yaml\n"
                    "  2. Switch to free Planetary Computer (no account needed):\n"
                    "     Set  data_backend: 'planetary_computer'  in config.yaml\n"
                    "     Run: pip install pystac-client stackstac planetary-computer rioxarray\n"
                    + "="*60
                )
            self._backend = _GEEBackend(
                project_id=project_id,
                service_account=cfg["gee"].get("service_account", ""),
            )

        elif backend_name == "planetary_computer":
            key = cfg.get("planetary_computer", {}).get("subscription_key", "")
            self._backend = _PlanetaryComputerBackend(subscription_key=key)
            logger.info("Backend: Microsoft Planetary Computer (free, no account needed)")

        elif backend_name == "sentinel_hub":
            sh = cfg.get("sentinel_hub", {})
            self._backend = _SentinelHubBackend(
                client_id=sh.get("client_id",""),
                client_secret=sh.get("client_secret",""),
                instance_id=sh.get("instance_id",""),
            )
            logger.info("Backend: Sentinel Hub")

        else:
            raise ValueError(
                f"Unknown data_backend '{backend_name}' in config.yaml. "
                "Choose: gee | planetary_computer | sentinel_hub"
            )

        self.backend_name = backend_name
        self.cfg = cfg

    def extract_pixel_timeseries_wide(
        self,
        geometry_geojson: dict,
        start_date: str,
        end_date: str,
        scale: int = 10,
        max_pixels: int = 500,
        orbit_pass: str = "DESCENDING",
    ) -> pd.DataFrame:
        """
        Extract pixel time-series in wide format (one row per pixel).

        Parameters
        ----------
        geometry_geojson : GeoJSON dict of the plot geometry
        start_date       : "YYYY-MM-DD"
        end_date         : "YYYY-MM-DD"
        scale            : pixel resolution in metres
        max_pixels       : max pixels to extract per plot
        orbit_pass       : Sentinel-1 orbit direction

        Returns
        -------
        pd.DataFrame with columns:
            longitude, latitude,
            B2_YYYY_MM, B3_YYYY_MM, ..., NDVI_YYYY_MM, VV_YYYY_MM, ...,
            optical_mask_YYYY_MM, ...
        """
        return self._backend.extract_pixel_timeseries_wide(
            geometry_geojson=geometry_geojson,
            start_date=start_date,
            end_date=end_date,
            scale=scale,
            max_pixels=max_pixels,
            orbit_pass=orbit_pass,
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias — existing code that imports GEEDownloader
# still works without any changes
# ---------------------------------------------------------------------------
class GEEDownloader(SatelliteDownloader):
    """
    Backward-compatible alias for SatelliteDownloader.
    Existing code using GEEDownloader continues to work unchanged.
    """
    def initialize(self):
        """Trigger GEE initialisation if using the GEE backend."""
        if hasattr(self._backend, "_ee"):
            self._backend._ee()  # calls ee.Initialize internally


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json, argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("geojson_file")
    ap.add_argument("start_date")
    ap.add_argument("end_date")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out",    default="data/processed/pixel_timeseries.csv")
    args = ap.parse_args()

    with open(args.geojson_file) as f:
        geojson = json.load(f)
    geom = geojson.get("geometry", geojson)

    dl = SatelliteDownloader(config_path=args.config)
    df = dl.extract_pixel_timeseries_wide(geom, args.start_date, args.end_date)
    df.to_csv(args.out, index=False)
    print(f"Saved {df.shape[0]} pixels to {args.out}")
