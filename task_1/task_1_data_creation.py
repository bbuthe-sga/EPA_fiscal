# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:47:03 2021

@author: AZ7
"""

# %% Imports
from pathlib import Path
from collections import defaultdict
import multiprocessing
import geopandas as gpd
import fiona
import pandas as pd
import numpy as np
import rasterio as rio
import fiona.crs as crs
import warnings
from rasterio.features import rasterize
from shapely.geometry import Polygon

import time
import datetime
import os
import shutil

#%%
# timer classes
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self._start_time = None
        self.last_time = self.start_time
        self.elapsed_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        print("Timer has started...")

    def check(self):
        """Check the timer, and record the elapsed time"""
        if self._start_time is None:
            # raise TimerError(f"Timer is not running. Use .start() to start it")
            return
        self.last_time = datetime.datetime.now()
        self.elapsed_time = time.perf_counter() - self._start_time

    def stop(self):
        # print(str(self))
        self._start_time = None

    def __str__(self):
        self.check()
        elapsed_time = self.elapsed_time
        if elapsed_time > 60:
            elapsed_time = elapsed_time / 60
            return f"Elapsed time: {elapsed_time:0.4f} minutes"
        elif elapsed_time > 3600:
            elapsed_time /= 3600
            return f"Elapsed time: {elapsed_time:0.4f} hours"
        else:
            return f"Elapsed time: {elapsed_time:0.4f} seconds"


class ProcessTracker():
    steps = {
        0: "loading the CBSA block groups",
        1: "loading cells within the county",
        2: "producing cell fragments and attributing identiying info",
        3: "calculating land cover suitability for cell fragments",
        4: "loading streets within the county",
        5: "estimating street widths",
        6: "calculating road area in cell fragments",
        7: "allocating jobs and population to the cell fragments",
        8: "writing output",
        9: "complete"
    }

    def __init__(self, name, counties, parent):
        self.name = name
        self.counties = counties
        self.status = "pending"
        self.step = 0
        self.current_county = 1
        self.timer = Timer()
        self.timer.start()
        self.parent = parent
        self.parent.add_process(self)
        print(str(self.parent))

    def advance(self):
        # Update our timer to post time of advance
        self.timer.check()
        self.step += 1
        # Increment steps/status as needed
        if self.step >= len(self.steps) - 1:
            if self.current_county >= len(self.counties):
                self.status = "complete"
                self.timer.stop()
            else:
                self.advance_county()
        # Update screen messages
        print(str(self.parent))

    def advance_county(self):
        self.current_county += 1
        self.step = 0

    def summary(self):
        return f"{self.name} ({str(self.timer)})"

    def __str__(self):
        n = len(self.counties)
        s1 = f"CBSA: {self.name} ... ... processing {self.current_county} of {n} counties"
        s2 = f"  since {str(self.timer.start_time).rsplit('.', 1)[0]}"
        s3 = f" -- County: {self.counties[self.current_county - 1]}"
        s4 = f" -- -- Step {self.step}: {self.steps[self.step]}"
        s5 = f"        since {str(self.timer.last_time).rsplit('.', 1)[0]}"
        return "\n".join([s1, s2, s3, s4, s5])


class Monitor():
    def __init__(self, timer):
        self.timer = timer
        self.processes = []

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def add_process(self, process_tracker):
        self.processes.append(process_tracker)

    def __str__(self):
        # self.clear()
        completed = [p for p in self.processes if p.status == "complete"]
        pending = [p for p in self.processes if p.status == "pending"]
        dashes = "-" * 50
        s1 = f"Begin: {str(self.timer.start_time).rsplit('.', 1)[0]}"
        s2 = f"Successful runs: {','.join([p.summary() for p in completed])}"
        s3 = "\nActive Runs:"
        s4 = "\n".join([str(p) for p in pending])
        return "\n".join([s1, s2, s3, dashes, s4])


# global environment variable
gpd.options.use_pygeos = True
warnings.filterwarnings("ignore")
timer = Timer()
MONITOR = Monitor(timer)
ST_SKIPS = [72, 15, 2]  # state fips codes for AK, HI, and PR


def load_cbsa_county_designations(path=None, sheet=None, skip_states=None):
    """# CBSA county designations  -- grab the CBSA county designations from the
        # Census (and only retain the variables we want to output in our tables)"""
    # Load the county designations from Census
    if path is None:
        path = "".join(
            ["https://www2.census.gov/programs-surveys/"
             "metro-micro/geographies/reference-files/",
             "2020/delineation-files/list1_2020.xls",
             ])
    cbsa_co = pd.read_excel(io=path, sheet_name=sheet, usecols=[0, 3, 4, 9, 10, 11], )
    # Format to match desired table output
    cbsa_co.columns = [
        "CBSA",
        "CBSA_name",
        "CBSA_type",
        "state_FIPS",
        "county_FIPS",
        "county_loc",
    ]
    cbsa_co["CBSA"] = cbsa_co['CBSA'].apply(str)
    cbsa_co["county_loc"] = cbsa_co.county_loc.str.lower()
    cbsa_co["CBSA_type"] = [
        x.replace("politan Statistical Area", "").lower() for x in cbsa_co.CBSA_type
    ]
    cbsa_co = cbsa_co[~cbsa_co["state_FIPS"].isin(skip_states)]
    cbsa_co = cbsa_co.reset_index(drop=True)
    # Return
    return cbsa_co


def load_alloc_vars(acs_sfvsmf_2019_path, gq_2019_path, wac_2018_path):
    # TODO: allocated gq pop
    """# Load variables to allocate"""
    # ACS data
    acs = pd.read_csv(
        acs_sfvsmf_2019_path,
        usecols=["GEOID", "sf_mh_tot", "mf_tot"],
        dtype={"GEOID": str},
    )
    acs = acs.rename(
        columns={"GEOID": "GEOID10", "sf_mh_tot": "SF_POP", "mf_tot": "MF_POP"}
    )
    gq = pd.read_csv(
        gq_2019_path,
        usecols=["GEOID", "In_GQ_Est"],
        dtype={"GEOID": str},
    )
    gq = gq.rename(columns={"GEOID": "GEOID10", "In_GQ_Est": "GQ_POP"})
    acs = acs.merge(gq, how="outer", on="GEOID10")
    # LODES data
    lodes = pd.read_csv(wac_2018_path, usecols=["w_bg", "C000"], dtype={"w_bg": str})
    lodes = lodes.rename(columns={"w_bg": "GEOID10", "C000": "JOBS"})
    # Merge and fill with 0s where appropriate
    dem = pd.merge(acs, lodes, on="GEOID10", how="outer")
    dem = dem.fillna(0)
    return dem


# GLOBALS
# "globals"
DATA = Path(r"K:\Projects\EPA\Fiscal_Impact\Features")
OUTPUT = Path(DATA, "OUTPUT")
RESULTS_TEMPLATE = Path(DATA, "Output_Template.csv")

# input data
cbsa_tbl = Path(DATA, "CBSA_2020.xls")
# Load allocation data
pop_path = Path(DATA, "ACS_SFvMF_2019.csv")
gq_path = Path(DATA, "ACS_HHvGQ_2019.csv")
job_path = Path(DATA, "WAC_2018.csv")
CBG_COLS = ["GEOID10", "STATEFP", "COUNTYFP", "CBSA", "CBSA_Name"]
# Global for Road Suitability -- True means calculate
RD_SUIT = False

# create iteration dataframe
''' swap in sheet name below to run on another machine
CLR_AWS - List_1
AB_AWS - List_2
IA_AWS - List_3
'''
SHEET_NAME = "List_2"
CHECKOUT = Path(OUTPUT,  "checkout", SHEET_NAME)
OUTPUT = Path(OUTPUT, SHEET_NAME)
if not OUTPUT.exists():
    OUTPUT.mkdir()
if not CHECKOUT.exists():
    CHECKOUT.mkdir()
CBSA_CTY = load_cbsa_county_designations(path=cbsa_tbl, sheet=SHEET_NAME, skip_states=ST_SKIPS)
ALLOC_DATA = load_alloc_vars(acs_sfvsmf_2019_path=pop_path, gq_2019_path=gq_path, wac_2018_path=job_path)


# %% Functions
def make_cbsa_dict(cbg_file, layer=None, checked_out=[]):
    """# generate a dict of CBSA: [index_values]"""
    with fiona.open(cbg_file, layer=layer) as src:
        data = defaultdict(list)
        for i, feature in enumerate(src):
            cbsa = feature["properties"]["CBSA"]
            if cbsa not in checked_out:
                data[cbsa].append(i + 1)
    return data


def get_records(filename, layer, idx_list, usecols=[], **kwargs):
    """# use the above index values to make a generator"""
    with fiona.open(filename, mode='r', layer=layer, **kwargs) as source:
        for i, feature in enumerate(source):
            if (i + 1) in idx_list:
                f = {k: feature[k] for k in ["id", "geometry"]}
                f["properties"] = {k: feature["properties"][k] for k in usecols}
                yield f


def get_data_rows(filename, layer=None, idx_list=None):
    """# read rows of interest from a geodataframe"""
    reader = fiona.open(filename, layer=layer)
    return gpd.GeoDataFrame.from_features((reader[x] for x in idx_list))


def raster_epsg(raster_path):
    """# Raster CRS -- use this to transform polygons so we don't have to warp the
        # raster, because this is costly and diminishes integrity of NLCD"""
    # Pull CRS from the raster
    with rio.open(raster_path, "r") as rp:
        crs = rp.crs
    return crs


# Read raster in a window of interest
def raster_from_aoi(raster_path, aoi_gdf, crs=None):
    """
    read raster from window associated with the area of interest
    Args:
        raster_path (str, Path): path to raster file
        aoi_gdf (gpd.GeoDataFrame: geodataframe representing an arao of interest
        crs (int): epsg code to transform the AOI prior to extracting raster data

    Returns:
        np.array of raster cell data
    """
    gdf = aoi_gdf
    if crs:
        gdf = aoi_gdf.to_crs(epsg=crs)
    # Load raster in the area of the aoi
    with rio.open(raster_path) as rp:
        # Set the window
        cw = rp.window(*gdf.total_bounds)
        # Grab the transform [needed for various raster ops if we choose to,
        # e.g. saving]
        window_transform = rp.window_transform(cw)
        # Load the raster
        ras = rp.read(1, window=cw)
        return ras, window_transform


# Shapely polygon of a gdf's bounding box
def bbox_poly(gdf, crs=None):
    if crs:
        gdf = gdf.to_crs(epsg=crs)
    bbox = gdf.total_bounds
    return Polygon(
        [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])]
    )


# Intersect cells and block groups to get cell pieces (also calculate area
# shares of the home block group)
def cells_bg_intersection(cells, cbsa_bg):
    # Intersect and calculate share
    itsn = gpd.overlay(cells, cbsa_bg)
    # Calculate area shares
    itsn["area"] = itsn.geometry.area
    itsn["cbg_area_shr"] = itsn.area / itsn.bg_total_area
    itsn["area"] = itsn["area"] * 0.000247105
    itsn = itsn.drop(columns="bg_total_area")
    # Return
    return itsn


# Convert NLCD to a cell [piece] suitability by reclass and masking
def nlcd_to_suitability(
        cells, nlcd_path, nlcd_reclass, pad_gdb_path, pad_layer, pad_reclass, crs
):
    _cells = cells.to_crs(epsg=4326)
    # Load NLCD as a raster
    ras, trans = raster_from_aoi(raster_path=nlcd_path, aoi_gdf=cells)
    # Reclass according to NLCD
    ras_sf = np.array(ras, dtype=float)
    ras_mf = np.array(ras, dtype=float)
    for idx_r, row_r in nlcd_reclass.iterrows():
        ras_sf[np.where(ras == row_r["NLCD"])] = row_r["RECLASS_SF"]
        ras_mf[np.where(ras == row_r["NLCD"])] = row_r["RECLASS_MF"]
    # Create a clipping geometry
    clip_geom = gpd.GeoDataFrame({"geometry": [bbox_poly(gdf=_cells)]}, crs=_cells.crs)
    # Load PAD subset and clip to exact area boundaries (need to be exact
    # so rasterization is consistent with NLCD)
    masked_pad = gpd.read_file(
        pad_gdb_path, driver="OpenFileGDB", layer=pad_layer, mask=clip_geom
    )
    masked_pad = masked_pad.to_crs(crs)
    clip_geom = clip_geom.to_crs(crs)
    if len(masked_pad) > 0:
        pad_clip = gpd.overlay(masked_pad, clip_geom)
        # Reclass and rasterize PAD
        pad_reclass["GAP_Sts"] = pad_reclass["GAP_Sts"].astype(str)
        pad_clip = pd.merge(pad_clip, pad_reclass, on=["GAP_Sts", "IUCN_Cat"])
        kv = [(g, v) for g, v in zip(pad_clip.geometry, pad_clip.Suit_SF_and_MF)]
    else:
        kv = []
    
    if kv:
        pad_ras = rasterize(
            shapes=kv, out_shape=ras.shape, fill=1, all_touched=False, transform=trans
        )

        # Apply PAD multipliers
        ras_sf = ras_sf * pad_ras
        ras_mf = ras_mf * pad_ras
    # Sum to cell pieces
    ras_iter = [(g, v) for g, v in zip(cells.geometry, np.arange(len(cells)))]
    mask_ras = rasterize(
        shapes=ras_iter,
        out_shape=ras.shape,
        fill=-1,
        all_touched=False,
        transform=trans,
    )
    sf_suit = []
    mf_suit = []
    nz_cells = []
    prot_cells = []
    soft_cells = []
    for x in np.arange(len(cells)):
        w = np.where(mask_ras == x)
        sf_suit.append(sum(ras_sf[w]))
        mf_suit.append(sum(ras_mf[w]))
        nz_cells.append(sum(ras_sf[w] != 0) * 900 * 0.000247105)  # (sqm2acres)
        if kv:
            prot_area = sum(pad_ras[w] == 0) * 900 * 0.000247105
            prot_cells.append(prot_area)
            soft_cells.append(sum(pad_ras[w] < 1) * 900 * 0.000247105 - prot_area)
        else:
            prot_cells.append(sum(ras_sf[w] * 0)) # TODO: clean up this code just trying to get the right shape for now
            soft_cells.append(sum(ras_sf[w] * 0))
    suit = cells.copy()[["cell_id", "GEOID10"]]
    suit["sf_res_suit_lc"] = np.minimum(sf_suit, 319.999562348901)
    suit["other_suit_lc"] = np.minimum(mf_suit, 319.999562348901)
    suit["developable_acres"] = np.minimum(nz_cells, 319.999562348901)
    suit["protected_acres"] = np.minimum(prot_cells, 319.999562348901)
    suit["soft_prot_acres"] = np.minimum(soft_cells, 319.999562348901)
    # Return
    return suit


# Estimate road width -- use a combination/hierarchy of lane width estimates
# and facility defaults
def estimate_road_width(navteq_roads, crs, reclass_table, to_meters=True):
    # Inflate the table based on speed
    rl1 = reclass_table[reclass_table.SPEED_CATEGORY == 3]
    rl2 = reclass_table[reclass_table.SPEED_CATEGORY == 3]
    rl4 = reclass_table[reclass_table.SPEED_CATEGORY == 5]
    rl6 = reclass_table[reclass_table.SPEED_CATEGORY == 8]
    rl7 = reclass_table[reclass_table.SPEED_CATEGORY == 8]
    rl1["SPEED_CATEGORY"] = 1
    rl2["SPEED_CATEGORY"] = 2
    rl4["SPEED_CATEGORY"] = 4
    rl6["SPEED_CATEGORY"] = 6
    rl7["SPEED_CATEGORY"] = 7
    rl_full = pd.concat([reclass_table, rl1, rl2, rl4, rl6, rl7])
    rl_full = rl_full.reset_index(drop=True)
    # Inflate the table based on travel direction
    rl_F = rl_full[rl_full.TRAVEL_DIRECTION == "(any other value)"]
    rl_T = rl_full[rl_full.TRAVEL_DIRECTION == "(any other value)"]
    rl_F["TRAVEL_DIRECTION"] = "F"
    rl_T["TRAVEL_DIRECTION"] = "T"
    rl_full = pd.concat([rl_full[rl_full.TRAVEL_DIRECTION == "B"], rl_F, rl_T])
    rl_full = rl_full[
        [
            "FUNCTIONAL_CLASS",
            "SPEED_CATEGORY",
            "LANE_CATEGORY",
            "TRAVEL_DIRECTION",
            "LANE_COUNT",
            "TOTAL_WIDTH",
        ]
    ]
    # Merge in the widths to the roads
    street_gdf = navteq_roads.merge(
        right=rl_full,
        on=["FUNCTIONAL_CLASS", "SPEED_CATEGORY", "LANE_CATEGORY", "TRAVEL_DIRECTION"],
    )
    # Convert to meters if appropriate
    if to_meters == True:
        street_gdf["WIDTH"] = (
                street_gdf.TOTAL_WIDTH / 0.3048
        )  # conversion of width to meters from feet

    # Format, transform, and return
    # street_cols = ["LINK_ID","SPEED_CATEGORY","LANE_CATEGORY",
    #                "FUNCTIONAL_CLASS","TRAVEL_DIRECTION","WIDTH",
    #                "geometry"]
    # street_gdf = street_gdf[street_cols]
    street_gdf = street_gdf.to_crs(crs)
    return street_gdf


# Calculate road area and road suitability within cell pieces
def roads_to_infr_area(cells_bg, roads_with_width, suitability_calc=False):
    sum_fields = [
        "ROAD_AREA", "BRIDGE_AREA", "CTRL_AREA", "TOLL_AREA",
        "BOAT_LEN", "CENTER_LEN", "LANE_LEN"
        ]
    # Intersect the roads and the block groups
    itsn = gpd.overlay(roads_with_width, cells_bg[["cell_id", "GEOID10", "geometry"]])
    # Calculate road area
    itsn["LEN"] = itsn.geometry.length
    itsn["road_area"] = itsn.WIDTH * itsn.LEN * 0.000247105

    # filter data
    itsn["AREA_FACTOR"] = np.logical_and.reduce([itsn.PAVED != "N", itsn.PRIVATE != "Y",
                                                 itsn.PUBLIC_ACCESS != "N", itsn.RAIL_FERRY != "Y",
                                                 itsn.BOAT_FERRY != "Y"])
    itsn["BRIDGE_FACTOR"] = np.logical_or.reduce([itsn.BRIDGE == "Y", itsn.TUNNEL == "Y"])
    itsn["CTRL_FACTOR"] = np.logical_or.reduce([itsn.CONTROLLED_ACCESS == "Y", itsn.LIMITED_ACCESS_ROAD == "Y"])
    itsn["TOLL_FACTOR"] = itsn.TOLLWAY == "Y"
    itsn["BOAT_FACTOR"] = itsn.BOAT_FERRY == "Y"

    # add areas
    itsn["ROAD_AREA"] = itsn.road_area * itsn.AREA_FACTOR
    itsn["BRIDGE_AREA"] = itsn.road_area * itsn.BRIDGE_FACTOR
    itsn["CTRL_AREA"] = itsn.road_area * itsn.CTRL_FACTOR
    itsn["TOLL_AREA"] = itsn.road_area * itsn.TOLL_FACTOR
    itsn["BOAT_LEN"] = itsn.LEN * itsn.BOAT_FACTOR
    itsn["CENTER_LEN"] = itsn.LEN
    itsn["LANE_LEN"] = itsn.LEN * itsn.LANE_COUNT

    # Summarize to cell/bg and fill empties with zero
    rd_area = itsn.groupby(["cell_id", "GEOID10", "FUNCTIONAL_CLASS"])[sum_fields].sum().reset_index()

    # pivot
    piv = pd.pivot_table(rd_area, values=sum_fields, index=["cell_id", "GEOID10"],
                         columns=["FUNCTIONAL_CLASS"])
    piv.columns = col_multi_index_to_names(columns=piv.columns, separator="_")
    piv.fillna(0)
    piv.reset_index(inplace=True)

    # Add in suitabilities (if necessary) and return
    if suitability_calc == True:
        rd_suit = pd.merge(
            rd_area, reclass_dict["ROADS_CLASS"], how="left", on="FUNCTIONAL_CLASS"
        )
        rd_suit["sf_res_suit_rd"] = rd_suit.ROAD_AREA * rd_suit.RECLASS_SF
        rd_suit["other_suit_rd"] = rd_suit.ROAD_AREA * rd_suit.RECLASS_MF
        rd_suit = rd_suit.groupby(["cell_id", "GEOID10"]).agg(
            {"sf_res_suit_rd": sum, "other_suit_rd": sum}
        )
        rd_suit = rd_suit.reset_index()
        rd = pd.merge(rd_suit, piv, on=["cell_id", "GEOID10"])
        return rd
    else:
        return piv


def allocate(alloc_df, suit_df):
    """# Allocate based on relative suitability"""
    # Total suitability by block group
    tot_suit = suit_df.groupby("GEOID10").agg(
        {"sf_res_suit_lc": sum, "other_suit_lc": sum}
    )
    tot_suit = tot_suit.reset_index()
    tot_suit = tot_suit.rename(
        columns={"sf_res_suit_lc": "tot_sf", "other_suit_lc": "tot_other"}
    )
    sdf = pd.merge(suit_df, tot_suit, on="GEOID10")
    # Fractional suitability by cell fragment
    sdf["frac_sf"] = sdf.sf_res_suit_lc / sdf.tot_sf
    sdf["frac_other"] = sdf.other_suit_lc / sdf.tot_other
    # Merge in the allocation pieces and allocate
    df = pd.merge(sdf, alloc_df, on="GEOID10")
    df["SF_ALLOC"] = df.SF_POP * df.frac_sf
    df["MF_ALLOC"] = df.MF_POP * df.frac_other
    df["GQ_ALLOC"] = df.GQ_POP * df.frac_other
    df["population"] = df.SF_ALLOC + df.MF_ALLOC + df.GQ_ALLOC
    df["jobs"] = df.JOBS * df.frac_other
    df["act24"] = df.population + df.jobs
    # Format and return
    select = [
        "cell_id", "GEOID10", "SF_ALLOC", "MF_ALLOC",
        "GQ_ALLOC", "population", "jobs", "act24"
        ]
    alloc = df[select]
    return alloc


def col_multi_index_to_names(columns, separator="_"):
    """
    For a collection of columns in a data frame, collapse index levels to
    flat column names. Index level values are joined using the provided
    `separator`.
    Args:
        columns (pandas.Index): The columns to flatten (i.e, df.columns)
        separator (str): The string value used to flatten multi-level column names

    Returns:
        flat_columns: pd.Index
    """
    if isinstance(columns, pd.MultiIndex):
        columns = columns.to_series().apply(lambda col: separator.join([str(c) for c in col]))
    return columns


def list_checked_out_cbsas(checkout_dir, dtype=str):
    """
    Lists all txt files in the checkout directory and generates an array of cbsa
    codes by parsing file names. Files are assumed to follow the naming convention:
    `Cells_XXXXX.txt`, where `XXXXX` is the cbsa fips code.

    Args:
        chekout_dir (path): The checkout directory contains empty text files that
            identify which cbsa's have been or are being anlayzed already.
        dtype (type, default=str): The data type of the list of cbsa codes.
        
    Returns:
        analyzed_cbsas (np.array): A string array listing all cbsa codes assumed
        to have already been analyzed based on the files in `output_dir`.
    """
    # List all txt files
    all_files = Path(checkout_dir).rglob("*.txt")
    # Split file names
    all_names = [f.stem for f in all_files if f.is_file()]
    all_cbsas = [n.split("_")[-1] for n in all_names]
    # Return array
    return np.array(all_cbsas, dtype=dtype)


def update_checkout_with_completed(output, checkout):
    """
    If the checkout folder needs to be refreshed to just reflect
    the completed runs, delete its contents and run this function.
    """
    all_files = list(Path(output).rglob("*.csv"))
    all_names = [f.stem for f in all_files if f.is_file()]
    all_cbsas = [n.split("_")[-1] for n in all_names]
    for n in all_names:
        p = Path(checkout, f"{n}.txt")
        with open(p, "w") as f:
            pass


def list_cbsa_counties(cbsa_fips):
    """Generate a list of 5-digit county FIPS from CBSA_CTY dataframe"""
    cbsa_rows = CBSA_CTY[CBSA_CTY.CBSA == cbsa_fips]
    counties = []
    for i, cbsa_row in cbsa_rows.iterrows():
        st = cbsa_row.state_FIPS
        ct = cbsa_row.county_FIPS
        stcty = f"0{st}"[-2:] + f"00{ct}"[-3:]
        counties.append(stcty)
    return counties


def filter_cbsa_by_stcty(stcty):
    """Make a filter for `CBSA_CTY` based on the current county being analyzed"""
    st = stcty[:2]
    cty = stcty[-3:]
    fltr = np.logical_and(CBSA_CTY.state_FIPS == int(st),
                          CBSA_CTY.county_FIPS == int(cty))
    return fltr


def process_cbsa(cbsa_fips, data_path, cbg_layer, cbsa_index_dict, reclass_dict, data_crs,
                 grid_path, nlcd_path, nlcd_crs, pad_layer, streets_path, streets_layer,
                 output_dir, monitor):
    checked_out = list_checked_out_cbsas(CHECKOUT, dtype=str)
    if cbsa_fips in checked_out:
        return
    counties = list_cbsa_counties(cbsa_fips)
    pt = ProcessTracker(name=cbsa_fips, counties=counties, parent=monitor)

    # write empty check-out file
    co_path = Path(CHECKOUT, f"InProgress_Cells_{cbsa_fips}.txt")
    with open(co_path, "w") as co_file:
        co_file.write("in progress")

    # setup output file:
    cbsa_results = Path(output_dir, f'Cells_{cbsa_fips}.csv')
    temp_df = pd.read_csv(RESULTS_TEMPLATE)
    shutil.copy(src=str(RESULTS_TEMPLATE), dst=str(cbsa_results))
    try:
        # read in only CBSA records of interest
        # print(f"{cbsa_fips} -- loading the CBSA block groups")
        idxs = cbsa_index_dict[cbsa_fips]
        feats = get_records(filename=data_path, layer=cbg_layer, usecols=CBG_COLS, idx_list=idxs)
        cbsa_gdf_all = gpd.GeoDataFrame.from_features(features=feats, crs=data_crs)

        # Iterate over counties
        for stcty in counties:
            # just keeping the name cbsa_gdf for ease of downstream continuity
            # this is really the specific county within a cbsa
            cbsa_gdf = cbsa_gdf_all[cbsa_gdf_all["GEOID10"].str.startswith(stcty)]

            # make mask polygon
            cbsa_poly = cbsa_gdf.unary_union
            cbsa_poly = gpd.GeoDataFrame({"geometry": [cbsa_poly]}, crs=data_crs)

            # update cbsa data to NLCD proj
            cbsa_gdf.to_crs(crs=nlcd_crs, inplace=True)
            cbsa_gdf["bg_total_area"] = cbsa_gdf.geometry.area

            # read in only grids intersecting cbsa
            # print(f"{cbsa_fips} -- loading cells within the CBSA")
            pt.advance()
            cells = gpd.read_file(filename=grid_path, mask=cbsa_poly)
            cells = cells.rename(columns={"GRID_ID": "cell_id"})
            cells = cells[["cell_id", "geometry"]]
            cells.to_crs(crs=nlcd_crs, inplace=True)

            # intersect to get area
            # print(f"{cbsa_fips} -- producing cell fragments and attributing identiying info")
            pt.advance()
            cells_bg = cells_bg_intersection(cells=cells, cbsa_bg=cbsa_gdf)
            cells_bg = cells_bg.rename(
                columns={"STATEFP": "state_FIPS", "COUNTYFP": "county_FIPS"}
            )
            fltr = filter_cbsa_by_stcty(stcty)
            cells_bg = pd.merge(
                left=cells_bg, right=CBSA_CTY[fltr][["CBSA", "county_loc", "CBSA_type", "CBSA_name"]],
                on="CBSA",
            )

            # reclass nlcd to suitability
            # print(f"{cbsa_fips} -- calculating land cover suitability for cell fragments")
            pt.advance()
            lc_suit = nlcd_to_suitability(
                cells=cells_bg, nlcd_path=nlcd_path, nlcd_reclass=reclass_dict["NLCD"],
                pad_gdb_path=data_path, pad_layer=pad_layer, pad_reclass=reclass_dict["PAD"],
                crs=nlcd_crs,
            )

            # read in street data
            # print(f"{cbsa_fips} -- loading streets within the CBSA")
            pt.advance()
            street_gdf = gpd.read_file(
                filename=streets_path, driver="OpenFileGDB",
                layer=streets_layer, mask=cbsa_poly,
            )

            # estimate road width from reclass
            # print(f"{cbsa_fips} -- estimating street widths")
            pt.advance()
            rw = estimate_road_width(navteq_roads=street_gdf, crs=nlcd_crs,
                                     reclass_table=reclass_dict["ROADS_LANES"], to_meters=True,
                                     )

            # estimate road area cell geometry
            # print(f"{cbsa_fips} -- calculating road area in cell fragments")
            pt.advance()
            rd_suit = roads_to_infr_area(
                cells_bg=cells_bg, roads_with_width=rw, suitability_calc=RD_SUIT
            )

            # Merge and allocate
            # print(f"{cbsa_fips} -- allocating jobs and population to the cell fragments")
            pt.advance()
            all_suit = pd.merge(lc_suit, rd_suit, how="left", on=["cell_id", "GEOID10"]).fillna(0)
            df = pd.merge(cells_bg, all_suit, on=["cell_id", "GEOID10"])
            pj = allocate(alloc_df=ALLOC_DATA, suit_df=df)
            df = pd.merge(df, pj, on=["cell_id", "GEOID10"])
            # df["infra_eff"] = df.ROAD_AREA / df.act24
            df.drop(columns=['geometry'], inplace=True)
            df = temp_df.append(df).fillna(0)

            # print(f"{cbsa_fips} -- writing")
            pt.advance()
            df.to_csv(cbsa_results, mode='a', header=False, index=False)
            pt.advance()
            #pt.advance_county()

        #Update checkout
        if co_path.exists():
            co_path.unlink()
        co_path = Path(CHECKOUT, f"Done_Cells_{cbsa_fips}.txt")
        with open(co_path, "w") as co_file:
            co_file.write("done")
    except Exception as e:
        # remove tracking file and any results written
        if cbsa_results.exists():
            cbsa_results.unlink()
        if co_path.exists():
            co_path.unlink()
        raise f"An error occured in the process: \n\t{e}"


# %% Run
if __name__ == "__main__":
    timer.start()
    # CBG geometries
    data_gdb = Path(DATA, "FiscalImpact_data.gdb")
    data_crs = fiona.crs.from_epsg(code=4326)
    cbg_layer = "EPA_SLD_Database_V3"
    cbg_cols = ["GEOID10", "STATEFP", "COUNTYFP", "CBSA", "CBSA_Name"]

    # GRID
    grid_path = "//".join(["zip:", str(DATA), 'GRIDS.zip!CONUS_GRID_halfSqMile.shp'])

    # NLCD
    nlcd_path = "//".join(["zip:", str(DATA),
                           "nlcd_2019_land_cover_l48_20210604.zip!", "nlcd_2019_land_cover_l48_20210604.img"])
    nlcd_crs = raster_epsg(raster_path=nlcd_path)

    # NAVSTREETS
    streets_path = r"K:\Projects\EPA\Fiscal_Impact\Transp\NAVSTREETS\file.gdb"
    streets_layer = "Link"

    # PAD
    pad_layer = "PADUS2_1Combined_Proclamation_Marine_Fee_Designation_Easement"

    reclass_table = Path(DATA, "reclasses.xlsx")
    reclass_dict = pd.read_excel(io=reclass_table, sheet_name=None)  # use sheet name to access df of interest

    # List which cbsas have been checked out already
    checked_out = list_checked_out_cbsas(CHECKOUT, dtype=str)  # TODO: change dtype if needed

    # read in cbsas and allocation data
    cbsa_index_dict = make_cbsa_dict(cbg_file=data_gdb, layer=cbg_layer, checked_out=checked_out)
    # timer.stop()
    print(f"Preparation steps complete ({str(timer)})")

    # Iterate through cbsas
    import random

    cbsas = CBSA_CTY.CBSA.to_list()
    random_cbsa = [cbsa for cbsa in cbsas if cbsa not in checked_out]
    random.shuffle(random_cbsa)
    args = list()
    print(len(random_cbsa))
    ####
    DEBUG = False
    if DEBUG:
        for cbsa in random_cbsa:
            cbsa_fips = cbsa
            if cbsa_fips in ["10220"]:
                args = [
                    cbsa_fips, data_gdb, cbg_layer,
                    cbsa_index_dict, reclass_dict, data_crs,
                    grid_path, nlcd_path, nlcd_crs,
                    pad_layer, streets_path, streets_layer, OUTPUT, MONITOR
                ]
                process_cbsa(*args)
    ####
    else:
        pool = multiprocessing.Pool(processes=2)
        for cbsa in random_cbsa:
            cbsa_fips = cbsa

            args.append(
                (cbsa_fips, data_gdb, cbg_layer,
                 cbsa_index_dict, reclass_dict, data_crs,
                 grid_path, nlcd_path, nlcd_crs,
                 pad_layer, streets_path, streets_layer, OUTPUT, MONITOR)
            )
        pool.starmap(process_cbsa, args)

# TODO: update GEOID10 to be 12 digits by adding leading 0 where appropriate