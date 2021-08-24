"""
Created on Mon Aug  9 13:47:03 2021

@author: Renaissance Planning
"""

from pathlib import Path
from collections import defaultdict
import geopandas as gpd
import fiona
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.features import rasterize
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen, urlretrieve
from urllib.parse import urlparse
import re
from zipfile import ZipFile
from io import BytesIO
from shapely.geometry import Polygon

# global environment variable
gpd.options.use_pygeos = True


# %% Fixed inputs
# Table of states by FIPS -- will be used for reference, and in the pad function
def load_state_fips():
    # Load the table from link
    st_fips_url = ''.join(["https://www.nrcs.usda.gov/wps/portal/nrcs/detail/",
                           "?cid=nrcs143_013696"])
    st_fips_html = requests.get(st_fips_url).content
    st_fips = pd.read_html(st_fips_html)[0]
    # Format for desired output
    st_fips = st_fips.loc[0:(len(st_fips) - 2), :]
    st_fips["FIPS"] = [str(int(x)).zfill(2) for x in st_fips.FIPS]
    st_fips = st_fips.rename(columns={"Name": "state",
                                      "Postal Code": "state_USPS",
                                      "FIPS": "state_FIPS"})
    # Return
    return st_fips


# NLCD download link -- set the link from which we'll pull NLCD so we can
# do state-by-state pulls without having to refind the link every time
def nlcd_l48_link(year=None):
    # Parse the MRLC data page for links featuring "land cover l48"
    req = Request("https://www.mrlc.gov/data")
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "lxml")
    links = []
    for link in soup.findAll('a'):
        href = link.get("href")
        if href is None:
            pass
        else:
            if bool(re.search("land_cover_l48", href, re.IGNORECASE)) == True:
                if bool(re.search(".zip$", href)) == True:
                    links.append(href)
    # Validate the year
    yrs = [int(re.search("(?<=nlcd_).*?(?=_land_cover)", x, re.IGNORECASE).group(0))
           for x in links]
    yrs = np.array(yrs)
    if year is None:
        year = max(yrs)
        w = np.where(yrs == year)[0][0]
    else:
        try:
            w = np.where(yrs == year)[0][0]
        except:
            raise ValueError(''.join(["'year' is not available; ",
                                      "must be one of ",
                                      ', '.join([str(x) for x in np.sort(yrs)])]))
    # Create link for desired year
    zip_link = links[w]
    req_zip_link = requests.get(zip_link, stream=True)
    zip_open = ZipFile(BytesIO(req_zip_link.content))
    zip_files = zip_open.namelist()
    ras = zip_files[np.where([bool(re.search(".img$", f)) for f in zip_files])[0][0]]
    full_link = ''.join(["zip+",
                         zip_link,
                         "!",
                         ras])
    # Return
    return full_link


# NLCD crs -- use this to transform polygons so we don't have to warp the
# raster, because this is costly and diminishes integrity of NLCD
def raster_epsg(raster_path):
    # Pull CRS from the raster
    with rio.open(raster_path, "r") as rp:
        crs = rp.crs
    return crs
    # return crs.to_epsg()


# CBSA county designations  -- grab the CBSA county designations from the
# Census (and only retain the variables we want to output in our tables)      
def load_cbsa_county_designations():
    # Load the county designations from Census
    cbsa_co = pd.read_excel(io=''.join(["https://www2.census.gov/programs-surveys/"
                                        "metro-micro/geographies/reference-files/",
                                        "2020/delineation-files/list1_2020.xls"]),
                            header=2, usecols=[0, 4, 9, 10, 11])
    # Format to match desired table output
    cbsa_co = cbsa_co.iloc[:1916, :]
    cbsa_co.columns = ["CBSA", "CBSA_type", "state_FIPS", "county_FIPS", "county_loc"]
    cbsa_co["county_loc"] = cbsa_co.county_loc.str.lower()
    cbsa_co["CBSA_type"] = [x.replace("politan Statistical Area", "").lower()
                            for x in cbsa_co.CBSA_type]
    cbsa_co = cbsa_co.reset_index(drop=True)
    # Return
    return cbsa_co


# Estimate road width -- use a combination/hierarchy of lane width estimates
# and facility defaults
def estimate_road_width(roads, crs,
                        width_field=None, width_factor=None, lanes_field=None,
                        facility_lane_widths=None, facility_type_field=None,
                        facility_default_widths=None, default=None):
    # Add a unique ID to the roads, and initialize hierarchical calculation
    # of road width
    rw = roads.copy()
    rw = rw.to_crs(crs)
    rw["temporary_id"] = np.arange(len(rw))
    ret = []
    # First, if a width field is given, use that value for width
    if width_field is None:
        pass
    else:
        by_width = rw[(rw[width_field] > 0) & (~rw[width_field].isna())]
        by_width = by_width.rename(columns={width_field: "width"})
        if width_factor is None:
            pass
        else:
            by_width["width"] = by_width.width * width_factor
        ret.append(by_width[["temporary_id", "width"]])
        rw = rw[~rw.temporary_id.isin(by_width.temporary_id)]
    # Second, if a lanes field is given, multiply lanes by a lane width
    # default for width
    if lanes_field is None or len(rw) == 0:
        pass
    else:
        by_lanes = rw[(rw[lanes_field] > 0) & (~rw[lanes_field].isna())]
        by_lanes = pd.merge(rw, facility_lane_widths,
                            how="inner", on=facility_type_field)
        by_lanes = by_lanes.rename(columns={lanes_field: "width"})
        by_lanes["width"] = by_lanes.width * by_lanes["lanes_field"]
        ret.append(by_lanes[["temporary_id", "width"]])
        rw = rw[~rw.temporary_id.isin(by_width.temporary_id)]
    # Third, if facility defaults are given, set the width according to the
    # facility default
    if facility_default_widths is None or len(rw) == 0:
        pass
    else:
        by_facility = pd.merge(rw, facility_default_widths,
                               how="inner", on=facility_type_field)
        ret.append(by_facility[["temporary_id", "width"]])
    # Finally, for any remaining rows, use a singular default value
    if len(rw) == 0:
        pass
    else:
        ret.append(pd.DataFrame({"temporary_id": rw.temporary_id,
                                 "width": np.repeat(default, len(rw))}))
    # Format and return
    ret = pd.concat(ret)
    road_width = pd.merge(roads, ret,
                          how="left", on="temporary_id")
    road_width = road_width.drop(columns="temporary_id")
    road_width = road_width.rename(columns={facility_type_field: "facility_type"})
    road_width = road_width[["width", "facility_type", "geometry"]]
    return road_width


# Create table of pad links by state for easy download
def pad_links_by_state():
    # Find the PAD-US data page on the home site
    home_link = "https://www.usgs.gov/core-science-systems/science-analytics-and-synthesis/gap/science/protected-areas"
    home_req = Request(home_link)
    home_page = urlopen(home_req)
    home_soup = BeautifulSoup(home_page, "lxml")
    for link in home_soup.findAll('a'):
        if link.string is None:
            pass
        else:
            if bool(re.search("data download", link.string, re.IGNORECASE)) == True:
                data_link = link.get("href")
                break
            else:
                pass
    if bool(re.search("^/", data_link)) == True:
        home_parsed = urlparse(home_link)
        home_base = '://'.join([home_parsed.scheme, home_parsed.netloc])
        data_link = ''.join([home_base, data_link])
    else:
        pass
    # Find the download page on the data page
    data_req = Request(data_link)
    data_page = urlopen(data_req)
    data_soup = BeautifulSoup(data_page, "lxml")
    for link in data_soup.findAll('a'):
        if link.string is None:
            pass
        else:
            if bool(re.search("data download", link.string, re.IGNORECASE)) == True:
                dl_link = link.get("href")
                break
            else:
                pass
    if bool(re.search("^/", dl_link)) == True:
        data_parsed = urlparse(data_link)
        data_base = '://'.join([data_parsed.scheme, data_parsed.netloc])
        dl_link = ''.join([data_base, dl_link])
    else:
        pass
    # Find the states shape page on the dl page
    dl_req = Request(dl_link)
    dl_page = urlopen(dl_req)
    dl_soup = BeautifulSoup(dl_page, "lxml")
    for link in dl_soup.findAll('a'):
        if link.string is None:
            pass
        else:
            if bool(re.search("state shapefile", link.string, re.IGNORECASE)) == True:
                shp_link = link.get("href")
                break
            else:
                pass
    if bool(re.search("^/", shp_link)) == True:
        dl_parsed = urlparse(dl_link)
        dl_base = '://'.join([dl_parsed.scheme, dl_parsed.netloc])
        shp_link = ''.join([dl_base, shp_link])
    else:
        pass
    # Find the download links on the shapes page
    shp_req = Request(shp_link)
    shp_page = urlopen(shp_req)
    shp_soup = BeautifulSoup(shp_page, "lxml")
    tab = shp_soup.findAll("table")[0]
    shp_names = []
    shp_links = []
    for row in tab.findAll("tr"):
        for col in row.findAll("td"):
            for d in col.findAll("span"):
                shp_names.append(d.string)
                shp_links.append(d.get("data-url"))
    # Format output
    shp_parsed = urlparse(shp_link)
    shp_base = '://'.join([shp_parsed.scheme, shp_parsed.netloc])
    df = pd.DataFrame({"state": shp_names, "pad": shp_links})
    pad_final = []
    for i, row in df.iterrows():
        try:
            st = re.search("(?<=_state).*?(?=_shapefile)", row["state"], re.IGNORECASE).group(0)
        except:
            st = "NULL"
        if st == "NULL":
            pass
        else:
            print(st)
            if bool(re.search("^/", row["pad"])) == True:
                zip_link = ''.join([shp_base, row["pad"]])
            else:
                zip_link = row["pad"]
            zip_open = ZipFile(BytesIO(urlopen(zip_link).read()))
            zip_files = np.array(zip_open.namelist())
            shps = zip_files[np.where([bool(re.search(".shp$", f)) for f in zip_files])[0]]
            ft = [re.search("(?<=[0-9]{1}_[0-9]{1}).*?(?=(_state)|(.shp$))", x, re.IGNORECASE).group(0)
                  for x in shps]
            st_df = pd.DataFrame({"state_USPS": np.repeat(st, len(ft)),
                                  "PAD_type": ft,
                                  "PAD_file": shps,
                                  "PAD_link": np.repeat(zip_link, len(ft))})
            pad_final.append(st_df)
    # Return
    pad_concat = pd.concat(pad_final).reset_index(drop=True)
    return pad_concat


# %% Loop functions

# Block groups -- grab the block group geometries from TIGER (only retaining 
# the variables we want to output in our tables)
def load_cbsa_bg_by_state(state_fips,
                          cbsa_cty,
                          crs):
    # Identify the CBSAs for that state
    cbsa_by_state = cbsa_cty[cbsa_cty.state_FIPS == state_fips]
    # Load the block groups for that state from TIGER
    bg = gpd.read_file(''.join(["https://www2.census.gov/geo/tiger/",
                                "TIGER2019/BG/",
                                "tl_2019_", state_fips, "_bg.zip"]))
    bg = bg.to_crs(crs)
    # Format for desired table output
    bg["bg_total_area"] = bg.geometry.area
    bg = bg.rename(columns={"GEOID": "geoid10",
                            "STATEFP": "state_FIPS",
                            "COUNTYFP": "county_FIPS"})
    bg = bg[["geoid10", "state_FIPS", "county_FIPS", "bg_total_area", "geometry"]]
    # Join to county table to only retain relevant bg
    bg_by_cbsa = pd.merge(cbsa_by_state, bg,
                          how="inner",
                          on=["state_FIPS", "county_FIPS"])
    bg_by_cbsa = gpd.GeoDataFrame(bg_by_cbsa)
    # Return
    return bg_by_cbsa


# Read in cells (maybe within mask of bg_by_cbsa) and transform to crs of
# bg_by_cbsa

# Intersect cells and block groups to get cell pieces (also calculate area
# shares of the home block group)
def cells_bg_intersection(cells,
                          cbsa_bg):
    # Intersect and calculate share
    itsn = gpd.overlay(cells, cbsa_bg)
    # Calculate area shares
    itsn["area"] = itsn.geometry.area
    itsn["cbg_area_shr"] = itsn.area / itsn.bg_total_area
    # Return
    return itsn


def bbox_poly(gdf, crs=None):
    if crs:
        gdf = gdf.to_crs(epsg=crs)
    bbox = gdf.total_bounds
    return Polygon([(bbox[0], bbox[1]),
                    (bbox[0], bbox[3]),
                    (bbox[1], bbox[3]),
                    (bbox[1], bbox[1])])


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
        cw = rp.window(*gdf.total_bounds)

        # need for writing raster if we choose to
        window_transform = rp.window_transform(cw)

        ras = rp.read(1, window=cw)
        return ras, window_transform


# Convert NLCD to a cell [piece] suitability by reclass and masking
# TODO: sum total area of everything that doesn't get nullified/above a threshhold
def nlcd_to_suitability(cells,
                        nlcd_path, nlcd_reclass,
                        pad_gdb_path, pad_layer, pad_reclass,
                        crs):
    ras, trans = raster_from_aoi(raster_path=nlcd_path, aoi_gdf=cells)

    # Create a clipping geometry
    clip_geom = gpd.GeoDataFrame({"geometry": [bbox_poly(gdf=cells)]}, crs=crs)

    # subset and clip PAD
    pad_mask = gpd.read_file(pad_gdb_path, driver="OpenFileGDB",
                             layer=pad_layer, mask=clip_geom)
    pad_mask = pad_mask.to_crs(crs)
    pad_clip = gpd.overlay(pad_mask, clip_geom)
    pad_reclass["GAP_Sts"] = pad_reclass["GAP_Sts"].astype(str)
    pad_clip = pd.merge(pad_clip, pad_reclass, on=["GAP_Sts", "IUCN_Cat"])
    kv = [(g, v) for g, v in zip(pad_clip.geometry, pad_clip.Suit_SF_and_MF)]
    pad_ras = rasterize(shapes=kv,
                        out_shape=ras.shape,
                        fill=1,
                        all_touched=False,
                        transform=trans)

    # Reclass the raster
    ras_sf = ras.copy()
    ras_mf = ras.copy()
    for idx_r, row_r in nlcd_reclass.iterrows():
        ras_sf[np.where(ras == row_r["NLCD"])] = row_r["RECLASS_SF"]
        ras_mf[np.where(ras == row_r["NLCD"])] = row_r["RECLASS_MF"]

    # Multiply nlcd suit by pad mask
    ras_sf = ras_sf * pad_ras
    ras_mf = ras_mf * pad_ras
    # Sum to cell pieces
    ras_iter = [(g, v) for g, v in zip(cells.geometry, np.arange(len(cells)))]
    mask_ras = rasterize(shapes=ras_iter,
                         out_shape=ras.shape,
                         fill=-1,
                         all_touched=False,
                         transform=trans)
    sf_suit = []
    mf_suit = []
    nz_cells = []
    for x in np.arange(len(cells)):
        print(x)
        sf_suit.append(sum(ras_sf[np.where(mask_ras == x)]))
        mf_suit.append(sum(ras_mf[np.where(mask_ras == x)]))
        nz_cells.append(sum(ras_sf[np.where(ras_sf == x)] != 0)) * 900 * 0.000247105  # (sqm2acres)
    suit = cells.copy()[["cell_id", "GEOID10"]]
    suit["sf_res_suit_lc"] = sf_suit
    suit["other_res_lc"] = mf_suit
    suit["developable_acres"] = nz_cells
    # Return
    return suit


# Load roads [gdf pre-attributed with width] within the bbox of cells_bg

def roads_to_infr_area(cells_bg,
                       roads_with_width):
    # Intersect the roads and the block groups
    itsn = gpd.overlay(roads_with_width, cells_bg[["cell_id", "GEOID10", "geometry"]])
    # Calculate road area
    itsn["LEN"] = itsn.geometry.length
    itsn["road_area"] = itsn.WIDTH * itsn.LEN

    # Summarize to cell/bg and fill empties with zero
    wdt = itsn.groupby(["cell_id", "GEOID10", "FUNCTIONAL_CLASS"])["road_area"].sum().reset_index()
    u = np.array(roads_with_width["FUNCTIONAL_CLASS"].unique())
    full = pd.DataFrame(np.repeat(cells_bg[["cell_id", "GEOID10"]].values, len(u), axis=0),
                        columns=["cell_id", "GEOID10"])
    full["FUNCTIONAL_CLASS"] = np.tile(np.sort(u), len(cells_bg))
    full = pd.merge(full, wdt,
                    how="left", on=["cell_id", "GEOID10", "FUNCTIONAL_CLASS"])
    full.loc[full.road_area.isna(), "road_area"] = 0
    # Add in suitabilities
    rd_suit = pd.merge(full, reclass_dict["ROADS_CLASS"],
                       how="left", on="FUNCTIONAL_CLASS")
    rd_suit["sf_res_suit_rd"] = rd_suit.road_area * rd_suit.RECLASS_SF
    rd_suit["other_suit_rd"] = rd_suit.road_area * rd_suit.RECLASS_MF
    rd_suit = rd_suit.groupby(["cell_id", "GEOID10"]).agg({"sf_res_suit_rd": sum,
                                                           "other_suit_rd": sum})
    rd_suit = rd_suit.reset_index()
    # Grab total area
    rd_area = full.groupby(["cell_id", "GEOID10"])["road_area"].sum()
    rd_area = rd_area.reset_index()
    # Return
    rd = pd.merge(rd_suit, rd_area, on=["cell_id", "GEOID10"])
    return rd


# Join up tables so all information is togehter/well formatted/etc.

# generate a dict of CBSA: [index_values]
def make_cbsa_dict(cbg_file, layer=None):
    with fiona.open(cbg_file, layer=layer) as src:
        data = defaultdict(list)
        for i, feature in enumerate(src):
            cbsa = feature["properties"]["CBSA"]
            data[cbsa].append(i + 1)
    return data


# use the above index values to make a generator
def get_records(filename, usecols, idx_list, **kwargs):
    # TODO: add check that usecols exist in f{'properties']
    with fiona.open(filename, **kwargs) as source:
        for i, feature in enumerate(source):
            if (i + 1) in idx_list:
                f = {k: feature[k] for k in ['id', 'geometry']}
                f['properties'] = {k: feature['properties'][k] for k in usecols}
                yield f


def get_data_rows(filename, layer=None, idx_list=None):
    reader = fiona.open(filename, layer=layer)
    return gpd.GeoDataFrame.from_features((reader[x] for x in idx_list))


# TODO: Groupby functional class
def append_road_area(grid_gdf, road_gdf, road_width_col):
    polys = grid_gdf.copy()
    polys.reset_index(inplace=True)
    lines = road_gdf.copy()
    lines.reset_index(inplace=True)
    sj = gpd.sjoin(left_df=lines, right_df=polys, how="inner", op="within")
    polys["lines_geoms"] = polys['index'].apply(lambda x: sj[sj['index_right'] == x]['geometry'].tolist())
    polys["width"] = polys["index"].apply(lambda x: sj[sj["index_right"] == x][road_width_col].tolist())
    polys["grid_road_area_sqm"] = None
    for idx, row in polys.iterrows():
        sum_area = 0
        for i in range(len(row["lines_geoms"])):
            sum_area += row["geometry"].intersection(row["line_geoms"][i]).length * row[road_width_col][i]
        polys.loc[idx, "grid_road_area_sqm"] = sum_area
    return polys.drop(columns=["lines_geoms", "width"])


if __name__ == "__main__":
    # "globals"
    ''' DATA '''
    DATA = r'path/to/data/folder'
    INPUT_DIR = Path(DATA, "task1_data")

    # input data
    # CBG geometries [SLD data contain all current CBG geometries and additonal
    #                   information for rolling data up (state, tract etc)]
    cbg_gdb = Path(INPUT_DIR, "SmartLocationDatabase.gdb")
    cbg_layer = "EPA_SLD_Database_V3"
    cbg_crs = CRS.from_epsg(5070)
    cbg_cols = ["GEOID10", "STATEFP", "COUNTYFP", "CBSA", "CBSA_Name"]

    # GRID  [generated for CONUS currently, will need to include Alaska,
    #          additional option is to utilize Hexagons as they tend to
    #          reduce sampling bias compared to grids
    #          - https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-whyhexagons.htm]
    grid_path = Path(INPUT_DIR, "GRID_halfSqMile.zip")

    # NLCD  [2019 NLCD data are used to assign weighting to a given cell, see reclass.xlsx]
    nlcd_path = "//".join(["zip:",
                           str(INPUT_DIR),
                           "nlcd_2019_land_cover_l48_20210604.zip!",
                           "nlcd_2019_land_cover_l48_20210604.img"])
    nlcd_crs = raster_epsg(raster_path=nlcd_path)

    # NAVSTREETS    [road layer used to assign road area to a given cell]
    streets_path = Path(INPUT_DIR, "file.gdb")
    streets_layer = "Link"
    streets_cols = ["LINK_ID", "SPEED_CATEGORY", "LANE_CATEGORY", "FUNCTIONAL_CLASS", "LANE_COUNT", "WIDTH", "geometry"]

    # PAD   [Protected areas data used to population weighting to a cell, see reclass.xlsx]
    pad_path = Path(INPUT_DIR, "PAD_US2_1_GDB", "PAD_US2_1.gdb")
    pad_layer = "PADUS2_1Combined_Proclamation_Marine_Fee_Designation_Easement"
    ''' DATA '''

    ''' PROCESSING - this will be converted to a function after we arrive at the final procedure and used
                    to multiprocess/pool to generate 2-3 active processes '''
    # reclass table
    reclass_table = Path(INPUT_DIR, "reclasses.xlsx ")
    reclass_dict = pd.read_excel(io=reclass_table, sheet_name=None)  # use sheet name to access df of interest

    # READ IN CBSA table and build iteration dataframe
    cbsa_index_dict = make_cbsa_dict(cbg_file=cbg_gdb, layer=cbg_layer)
    sf = load_state_fips()

    # create iteration dataframe
    cbsa_cty = load_cbsa_county_designations()
    random_cbsa = cbsa_cty.sample(frac=1)
    # cbsa = cbsa_cty[cbsa_cty.CBSA == "18700"]  # go beavs babyyyyyy   (Test CBSA)
    for i, cbsa in random_cbsa.iterrows():
        idxs = cbsa_index_dict[str(cbsa["CBSA"].iloc[0])]

        # read in only CBSA records of interest
        feats = get_records(filename=cbg_gdb, usecols=cbg_cols, idx_list=idxs)
        cbsa_gdf = gpd.GeoDataFrame.from_features(features=feats, crs=cbg_crs)
        cbsa_gdf["bg_total_area"] = cbsa_gdf.geometry.area
        cbsa_gdf.to_crs(crs=nlcd_crs, inplace=True)
        # cbsa_gdf.to_file(Path(INPUT_DIR, f"cbsa_{cbsa['CBSA']}.shp"))         # debugging

        # # ensure neighboring CBGs are available for grid intersection         # debugging
        # cbsa_bbox = cbsa_gdf.total_bounds                                     # debugging
        # cbg_bbox = gpd.read_file(filename=cbg_gdb, bbox=tuple(cbsa_bbox))     # debugging

        # read in only grids intersecting cbsa
        cbsa_poly = cbsa_gdf.unary_union
        cbsa_poly = gpd.GeoDataFrame({"geometry": [cbsa_poly]}, crs=cbsa_gdf.crs)
        cells = gpd.read_file(grid_path, bbox=cbsa_poly)
        cells = cells.to_crs(nlcd_crs)
        cells = cells.rename(columns={"GRID_ID": "cell_id"})
        cells = cells[["cell_id", "geometry"]]

        # intersect to get area
        cells_bg = cells_bg_intersection(cells=cells, cbsa_bg=cbsa_gdf)

        # reclass nlcd to suitability
        lc_suit = nlcd_to_suitability(cells=cells_bg,
                                      nlcd_link=nlcd_path, nlcd_reclass=reclass_dict["NLCD"],
                                      pad_link=pad_path, pad_reclass=reclass_dict["PAD"],
                                      crs=nlcd_crs)
        # TODO: Return area of non-zero suitability. and land area?

        # read in street data
        cbsa_4326 = cbsa_gdf.to_crs(4326).unary_union
        street_gdf = gpd.read_file(filename=streets_path, driver="OpenFileGDB",
                                   layer=streets_layer, bbox=cbsa_4326)
        # LANE CATEGORY: 1 = 1 lane, 2 = 2 lanes, 3 = 4 or more; TRAVEL_DIRECTION: B = both direction, T or F = one way
        street_gdf = street_gdf.merge(right=reclass_dict['ROADS_LANES'], on="LANE_CATEGORY")
        #
        street_gdf.loc[street_gdf.TRAVEL_DIRECTION == "B", "WIDTH"] = 2 * street_gdf.loc[
            street_gdf.TRAVEL_DIRECTION == "B", "WIDTH"]
        street_gdf["WIDTH"] = street_gdf.WIDTH / 0.3048  # conversion of width to meters from feet
        street_gdf = street_gdf[streets_cols]
        street_gdf = street_gdf.to_crs(nlcd_crs)

        # estimate road area cell geometry
        rd_suit = roads_to_infr_area(cells_bg=cells_bg,
                                     roads_with_width=street_gdf)

        # Merge and split tables to match desired structure
        all_suit = pd.merge(lc_suit, rd_suit, on=["cell_id", "GEOID10"])
        df = pd.merge(cells_bg, all_suit, on=["cell_id", "GEOID10"])

        LOC_CELLS = df[["cell_id", "GEOID10", "STATEFP", "COUNTYFP", "CBSA", "CBSA_Name",
                        "road_area", "geometry"]]
        CELLS_CBG = df[["cell_id", "GEOID10", "area", "cbg_area_shr",
                        "sf_res_suit_rd", "other_suit_rd",
                        "sf_res_suit_lc", "other_res_lc", "geometry"]]  # TODO: change name
        # TODO: set up writing, storing etc.
        # TODO: outputs will be labled by their CBSA, and post processing done to merge all the data back together
