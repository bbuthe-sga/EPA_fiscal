"""
Created on Mon Aug  9 13:47:03 2021

@author: Renaissance Planning
"""

# %% Imports
from pathlib import Path
from collections import defaultdict
import geopandas as gpd
import fiona
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import Polygon

from src.tools.main import Paths
# import requests
# from bs4 import BeautifulSoup
# from urllib.request import Request, urlopen, urlretrieve
# from urllib.parse import urlparse
# import re
# from zipfile import ZipFile
# from io import BytesIO

# global environment variable
gpd.options.use_pygeos = True


# %% Functions

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



# read rows of interest from a geodataframe
def get_data_rows(filename, layer=None, idx_list=None):
    reader = fiona.open(filename, layer=layer)
    return gpd.GeoDataFrame.from_features((reader[x] for x in idx_list))



# CBSA county designations  -- grab the CBSA county designations from the
# Census (and only retain the variables we want to output in our tables)      
def load_cbsa_county_designations():
    # Load the county designations from Census
    cbsa_co = pd.read_excel(io=''.join(["https://www2.census.gov/programs-surveys/"
                                        "metro-micro/geographies/reference-files/",
                                        "2020/delineation-files/list1_2020.xls"]),
                            header=2, usecols=[0, 3, 4, 9, 10, 11])
    # Format to match desired table output
    cbsa_co = cbsa_co.iloc[:1916, :]
    cbsa_co.columns = ["CBSA", "CBSA_name", "CBSA_type", "state_FIPS", "county_FIPS", "county_loc"]
    cbsa_co["county_loc"] = cbsa_co.county_loc.str.lower()
    cbsa_co["CBSA_type"] = [x.replace("politan Statistical Area", "").lower()
                            for x in cbsa_co.CBSA_type]
    cbsa_co = cbsa_co.reset_index(drop=True)
    # Return
    return cbsa_co



# Raster CRS -- use this to transform polygons so we don't have to warp the
# raster, because this is costly and diminishes integrity of NLCD
def raster_epsg(raster_path):
    # Pull CRS from the raster
    with rio.open(raster_path, "r") as rp:
        crs = rp.crs
    return crs
    # return crs.to_epsg()
    # crs.to_epsg in function isn't working ^, so I've removed it for now




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
    return Polygon([(bbox[0], bbox[1]),
                    (bbox[0], bbox[3]),
                    (bbox[1], bbox[3]),
                    (bbox[1], bbox[1])])



# Intersect cells and block groups to get cell pieces (also calculate area
# shares of the home block group)
def cells_bg_intersection(cells,
                          cbsa_bg):
    # Intersect and calculate share
    itsn = gpd.overlay(cells, cbsa_bg)
    # Calculate area shares
    itsn["area"] = itsn.geometry.area
    itsn["cbg_area_shr"] = itsn.area / itsn.bg_total_area
    itsn["area"] = itsn["area"] * 0.000247105
    itsn = itsn.drop(columns = "bg_total_area")
    # Return
    return itsn
    
    

# Convert NLCD to a cell [piece] suitability by reclass and masking
def nlcd_to_suitability(cells,
                        nlcd_path, nlcd_reclass,
                        pad_gdb_path, pad_layer, pad_reclass,
                        crs):
    # Load NLCD as a raster
    ras, trans = raster_from_aoi(raster_path=nlcd_path, aoi_gdf=cells)
    # Create a clipping geometry
    clip_geom = gpd.GeoDataFrame({"geometry": [bbox_poly(gdf=cells)]},
                                 crs = cells.crs)
    # Load PAD subset and clip to exact area boundaries (need to be exact
    # so rasterization is consistent with NLCD)
    pad_mask = gpd.read_file(pad_gdb_path, driver="OpenFileGDB",
                             layer=pad_layer, mask=clip_geom)
    pad_mask = pad_mask.to_crs(crs)
    pad_clip = gpd.overlay(pad_mask, clip_geom)
    # Reclass and rasterize PAD
    pad_reclass["GAP_Sts"] = pad_reclass["GAP_Sts"].astype(str)
    pad_clip = pd.merge(pad_clip, pad_reclass, on=["GAP_Sts", "IUCN_Cat"])
    kv = [(g, v) for g, v in zip(pad_clip.geometry, pad_clip.Suit_SF_and_MF)]
    pad_ras = rasterize(shapes=kv,
                        out_shape=ras.shape,
                        fill=1,
                        all_touched=False,
                        transform=trans)
    # Reclass according to NLCD
    ras_sf = ras.copy()
    ras_mf = ras.copy()
    for idx_r, row_r in nlcd_reclass.iterrows():
        ras_sf[np.where(ras == row_r["NLCD"])] = row_r["RECLASS_SF"]
        ras_mf[np.where(ras == row_r["NLCD"])] = row_r["RECLASS_MF"]
    # Apply PAD multipliers
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
    prot_cells = []
    for x in np.arange(len(cells)):
        w = np.where(mask_ras == x)
        sf_suit.append(sum(ras_sf[w]))
        mf_suit.append(sum(ras_mf[w]))
        nz_cells.append(sum(ras_sf[w] != 0) * 900 * 0.000247105) #(sqm2acres)
        prot_cells.append(sum(pad_ras[w] == 0) * 900 * 0.000247105)
    suit = cells.copy()[["cell_id", "GEOID10"]]
    suit["sf_res_suit_lc"] = sf_suit
    suit["other_suit_lc"] = mf_suit
    suit["developable_acres"] = nz_cells
    suit["protected_acres"] = prot_cells
    # Return
    return suit



# Estimate road width -- use a combination/hierarchy of lane width estimates
# and facility defaults
def estimate_road_width(navteq_roads, 
                        crs,
                        reclass_table,
                        to_meters=True):
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
    rl_full = pd.concat([rl_full[rl_full.TRAVEL_DIRECTION == "B"],
                         rl_F, rl_T]) 
    rl_full = rl_full[["FUNCTIONAL_CLASS","SPEED_CATEGORY",
                       "LANE_CATEGORY","TRAVEL_DIRECTION",
                       "TOTAL_WIDTH"]]
    # Merge in the widths to the roads
    street_gdf = navteq_roads.merge(right=rl_full,
                                    on=["FUNCTIONAL_CLASS", "SPEED_CATEGORY",
                                        "LANE_CATEGORY","TRAVEL_DIRECTION"])
    # Convert to meters if appropriate
    if to_meters == True:
        street_gdf["WIDTH"] = street_gdf.TOTAL_WIDTH / 0.3048   # conversion of width to meters from feet
    else:
        pass
    # Format, transform, and return
    # street_cols = ["LINK_ID","SPEED_CATEGORY","LANE_CATEGORY",
    #                "FUNCTIONAL_CLASS","TRAVEL_DIRECTION","WIDTH",
    #                "geometry"]
    # street_gdf = street_gdf[street_cols]
    street_gdf = street_gdf.to_crs(crs)
    return street_gdf



# TODO: pending SGA's response, edit to be wide on chars of interest
# Calculate road area and road suitability within cell pieces
def roads_to_infr_area(cells_bg,
                       roads_with_width,
                       suitability_calc=False):
    # Intersect the roads and the block groups
    itsn = gpd.overlay(roads_with_width, 
                       cells_bg[["cell_id", "GEOID10","geometry"]])
    # Calculate road area
    itsn["LEN"] = itsn.geometry.length
    itsn["road_area"] = itsn.WIDTH * itsn.LEN
    # Summarize to cell/bg and fill empties with zero
    wdt = itsn.groupby(["cell_id", "GEOID10", "FUNCTIONAL_CLASS"])["road_area"].sum().reset_index()
    u = np.array(roads_with_width["FUNCTIONAL_CLASS"].unique())
    full = pd.DataFrame(np.repeat(cells_bg[["cell_id", "GEOID10"]].values, len(u), axis=0),
                        columns=["cell_id","GEOID10"])
    full["FUNCTIONAL_CLASS"] = np.tile(np.sort(u), len(cells_bg))
    full = pd.merge(full, wdt,
                    how="left", on=["cell_id", "GEOID10", "FUNCTIONAL_CLASS"])
    full.loc[full.road_area.isna(), "road_area"] = 0
    # Grab total area
    rd_area = full.groupby(["cell_id","GEOID10"])["road_area"].sum()
    rd_area = rd_area.reset_index()
    rd_area["road_area"] = rd_area.road_area * 0.000247105 #sqm to acres
    # Add in suitabilities (if necessary) and return
    if suitability_calc == True:
        rd_suit = pd.merge(full, reclass_dict["ROADS_CLASS"], 
                           how="left", on="FUNCTIONAL_CLASS")
        rd_suit["sf_res_suit_rd"] = rd_suit.road_area * rd_suit.RECLASS_SF
        rd_suit["other_suit_rd"] = rd_suit.road_area * rd_suit.RECLASS_MF
        rd_suit = rd_suit.groupby(["cell_id","GEOID10"]).agg({"sf_res_suit_rd":sum,
                                                              "other_suit_rd":sum})
        rd_suit = rd_suit.reset_index()
        rd = pd.merge(rd_suit, rd_area, on=["cell_id","GEOID10"])
        return rd
    else:
        return rd_area
    



# Load variables to allocate
def load_alloc_vars(acs_sfvsmf_2019_path,
                    wac_2018_path):
    # ACS data
    acs = pd.read_csv(acs_sfvsmf_2019_path,
                      usecols=["GEOID","sf_mh_tot","mf_tot"],
                      dtype = {"GEOID":str})
    acs = acs.rename(columns={"GEOID":"GEOID10",
                              "sf_mh_tot":"SF_POP",
                              "mf_tot":"MF_POP"})
    # LODES data
    lodes = pd.read_csv(wac_2018_path,
                        usecols=["w_bg","C000"],
                        dtype = {"w_bg":str})
    lodes = lodes.rename(columns={"w_bg":"GEOID10",
                                  "C000":"JOBS"})
    # Merge and fill with 0s where appropriate
    dem = pd.merge(acs, lodes, on="GEOID10", how="outer")
    dem = dem.fillna(0)
    return dem



# Allocate based on relative suitabilit
def allocate(alloc_df,
             suit_df):
    # Total suitability by block group
    tot_suit = suit_df.groupby("GEOID10").agg({"sf_res_suit_lc":sum,
                                               "other_suit_lc":sum})
    tot_suit = tot_suit.reset_index()
    tot_suit = tot_suit.rename(columns={"sf_res_suit_lc":"tot_sf",
                                        "other_suit_lc":"tot_other"})
    sdf = pd.merge(suit_df, tot_suit, on="GEOID10")
    # Fractional suitability by cell fragment
    sdf["frac_sf"] = sdf.sf_res_suit_lc / sdf.tot_sf
    sdf["frac_other"] = sdf.other_suit_lc / sdf.tot_other
    # Merge in the allocation pieces and allocate
    df = pd.merge(sdf, alloc_df, on="GEOID10")
    df["SF_ALLOC"] = df.SF_POP * df.frac_sf
    df["MF_ALLOC"] = df.MF_POP * df.frac_other
    df["population"] = df.SF_ALLOC + df.MF_ALLOC
    df["jobs"] = df.JOBS * df.frac_other
    df["act24"] = df.population + df.jobs
    # Format and return
    alloc = df[["cell_id","GEOID10","population","jobs","act24"]]
    return alloc



# %% Run

if __name__ == "__main__":
    # "globals"
    # DATA = r'C:\github\projects\PROJECT_epa_fiscal_impact\data\raw'
    # INPUT_DIR = Path(DATA, "task1_datasets")

    # input data
    # CBG geometries
    # cbg_gdb = Path(INPUT_DIR, "SmartLocationDatabase.gdb")
    cbg_gdb = r"K:\Projects\EPA\Fiscal_Impact\SmartLocationDatabase.gdb"
    cbg_layer = "EPA_SLD_Database_V3"
    cbg_crs = CRS.from_epsg(5070)
    cbg_cols = ["GEOID10", "STATEFP", "COUNTYFP", "CBSA", "CBSA_Name"]

    # GRID
    # grid_path = Path(INPUT_DIR, "GRID_halfSqMile.zip")
    grid_path = r"zip://D:\Users\AZ7\Downloads\EPA_T1_Data_Temp\EPA_Tesselations.zip!CONUS_GRID_halfSqMile.shp"

    # NLCD
    # nlcd_path = "//".join(["zip:",
    #                        str(INPUT_DIR),
    #                        "nlcd_2019_land_cover_l48_20210604.zip!",
    #                        "nlcd_2019_land_cover_l48_20210604.img"])
    nlcd_path = r"zip://D:\Users\AZ7\Downloads\EPA_T1_Data_Temp\nlcd_2019_land_cover_l48_20210604.zip!nlcd_2019_land_cover_l48_20210604.img"
    nlcd_crs = raster_epsg(raster_path=nlcd_path)
    
    # NAVSTREETS
    # streets_path = Path(INPUT_DIR, "file.gdb")
    streets_path = r"K:\Projects\EPA\Fiscal_Impact\Transp\NAVSTREETS\file.gdb"
    streets_layer = "Link"

    # PAD
    # pad_path = Path(INPUT_DIR, "PAD_US2_1_GDB", "PAD_US2_1.gdb")
    pad_path = r"zip://D:\Users\AZ7\Downloads\EPA_T1_Data_Temp\PAD_US2_1_GDB.zip!PAD_US2_1.gdb"
    pad_layer = "PADUS2_1Combined_Proclamation_Marine_Fee_Designation_Easement"

    # reclass table
    # reclass_table = r"C:\OneDrive_RP\Renaissance Planning Group\Tech Team - EPA Fiscal " \
    #                 r"Impact\task1_data_docs\reclasses.xlsx "
    reclass_table = r"D:\Users\AZ7\Downloads\EPA_T1_Data_Temp\reclasses.xlsx"
    reclass_dict = pd.read_excel(io=reclass_table, sheet_name=None)  # use sheet name to access df of interest
    
    # read in cbsas and allocation data
    cbsa_index_dict = make_cbsa_dict(cbg_file=cbg_gdb, layer=cbg_layer)
    
    # Load allocation data
    pop_path = r"K:\Projects\EPA\Fiscal_Impact\Census\ACS_SFvMF_2019.csv"
    job_path = r"K:\Projects\EPA\Fiscal_Impact\Census\WAC_2018.csv"
    alloc = load_alloc_vars(acs_sfvsmf_2019_path = pop_path,
                            wac_2018_path = job_path)

    # create iteration dataframe
    cbsa_cty = load_cbsa_county_designations()
    
    # Global for Road Suitability -- True means calculate
    RD_SUIT = False
    
    # Iterate through cbsas
    random_cbsa = cbsa_cty.sample(frac=1)
    # cbsa = cbsa_cty[cbsa_cty.CBSA == "18700"] #go beavs babyyyyyy
    for i, cbsa in random_cbsa.iterrows():
        print("CBSA: " + cbsa["CBSA"].iloc[0] + " (" + cbsa["CBSA_name"].iloc[0] + ")")
        
        # read in only CBSA records of interest
        print("-- loading the CBSA block groups")
        idxs = cbsa_index_dict[str(cbsa["CBSA"].iloc[0])]
        feats = get_records(filename=cbg_gdb, usecols=cbg_cols, idx_list=idxs)
        cbsa_gdf = gpd.GeoDataFrame.from_features(features=feats, crs=cbg_crs)
        cbsa_gdf["bg_total_area"] = cbsa_gdf.geometry.area
        cbsa_gdf.to_crs(crs=nlcd_crs, inplace=True)
        # cbsa_gdf.to_file(Path(INPUT_DIR, f"cbsa_{cbsa['CBSA']}.shp"))

        # # ensure neighboring CBGs are available for grid intersection
        # cbsa_bbox = cbsa_gdf.total_bounds
        # cbg_bbox = gpd.read_file(filename=cbg_gdb, bbox=tuple(cbsa_bbox))

        # read in only grids intersecting cbsa
        print("-- loading cells within the CBSA")
        cbsa_poly = cbsa_gdf.unary_union
        cbsa_poly = gpd.GeoDataFrame({"geometry":[cbsa_poly]}, crs=cbsa_gdf.crs)
        cells = gpd.read_file(grid_path, bbox=cbsa_poly)
        cells = cells.to_crs(nlcd_crs)
        cells = cells.rename(columns={"GRID_ID":"cell_id"})
        cells = cells[["cell_id","geometry"]]
        
        # intersect to get area
        print("-- producing cell fragments and attributing identiying info")
        cells_bg = cells_bg_intersection(cells = cells, 
                                         cbsa_bg = cbsa_gdf)
        cells_bg = cells_bg.rename(columns={"STATEFP":"state_FIPS",
                                            "COUNTYFP":"county_FIPS"})
        cells_bg = pd.merge(cells_bg, cbsa_cty[["CBSA","county_loc","CBSA_type","CBSA_name"]],
                            on="CBSA")

        # reclass nlcd to suitability
        print("-- calculating land cover suitability for cell fragments")
        lc_suit = nlcd_to_suitability(cells=cells_bg,
                                      nlcd_path=nlcd_path, nlcd_reclass=reclass_dict["NLCD"],
                                      pad_gdb_path=pad_path, pad_layer=pad_layer, 
                                      pad_reclass=reclass_dict["PAD"],
                                      crs = nlcd_crs)

        # read in street data
        print("-- loading streets within the CBSA")
        cbsa_4326 = cbsa_gdf.to_crs(4326).unary_union
        street_gdf = gpd.read_file(filename=streets_path, driver="OpenFileGDB",
                                   layer=streets_layer, bbox=cbsa_4326)
        
        # estimate road width from reclass
        print("-- estimating street widths")
        rw = estimate_road_width(street_gdf,
                                 crs = nlcd_crs,
                                 reclass_table = reclass_dict["ROADS_LANES"],
                                 to_meters = True)
        
        # estimate road area cell geometry
        print("-- calculating road area in cell fragments")
        rd_suit = roads_to_infr_area(cells_bg=cells_bg,
                                     roads_with_width=rw,
                                     suitability_calc=RD_SUIT)
        
        # Merge and allocate
        print("-- allocating jobs and population to the cell fragments")
        all_suit = pd.merge(lc_suit, rd_suit, on=["cell_id","GEOID10"])
        df = pd.merge(cells_bg, all_suit, on=["cell_id","GEOID10"])
        pj = allocate(alloc_df = alloc,
                      suit_df = df)
        df = pd.merge(df, pj, on=["cell_id","GEOID10"])
        df["infra_eff"] = df.road_area / df.act24
        
        # # Format and write
        # print("-- formatting outputs and writing to file")
        # LOC_CELLS = df[["cell_id","GEOID10","state_FIPS","county_FIPS",
        #                 "CBSA","CBSA_name","CBSA_type","county_loc",
        #                 "population","jobs","act24","road_area","infra_eff"]]
        # if "sf_res_suit_rd" in df.columns:
        #     CELLS_CBG = df[["cell_id","GEOID10","area","cbg_area_shr",
        #                     "sf_res_suit_rd","other_suit_rd",
        #                     "sf_res_suit_lc","other_suit_lc",
        #                     "developable_acres","protected_acres"]]
        # else:
        #     CELLS_CBG = df[["cell_id","GEOID10","area","cbg_area_shr",
        #                     "sf_res_suit_lc","other_suit_lc",
        #                     "developable_acres","protected_acres"]]
        
        print("-- writing")
        # TODO: set up writing, storing etc.
        
        

# %% Not in use

# # Table of states by FIPS -- will be used for reference, and in the pad function
# def load_state_fips():
#     # Load the table from link
#     st_fips_url = ''.join(["https://www.nrcs.usda.gov/wps/portal/nrcs/detail/",
#                            "?cid=nrcs143_013696"])
#     st_fips_html = requests.get(st_fips_url).content
#     st_fips = pd.read_html(st_fips_html)[0]
#     # Format for desired output
#     st_fips = st_fips.loc[0:(len(st_fips) - 2), :]
#     st_fips["FIPS"] = [str(int(x)).zfill(2) for x in st_fips.FIPS]
#     st_fips = st_fips.rename(columns={"Name": "state",
#                                       "Postal Code": "state_USPS",
#                                       "FIPS": "state_FIPS"})
#     # Return
#     return st_fips



# # NLCD download link -- set the link from which we'll pull NLCD so we can
# # do state-by-state pulls without having to refind the link every time
# def nlcd_l48_link(year=None):
#     # Parse the MRLC data page for links featuring "land cover l48"
#     req = Request("https://www.mrlc.gov/data")
#     html_page = urlopen(req)
#     soup = BeautifulSoup(html_page, "lxml")
#     links = []
#     for link in soup.findAll('a'):
#         href = link.get("href")
#         if href is None:
#             pass
#         else:
#             if bool(re.search("land_cover_l48", href, re.IGNORECASE)) == True:
#                 if bool(re.search(".zip$", href)) == True:
#                     links.append(href)
#     # Validate the year
#     yrs = [int(re.search("(?<=nlcd_).*?(?=_land_cover)", x, re.IGNORECASE).group(0))
#            for x in links]
#     yrs = np.array(yrs)
#     if year is None:
#         year = max(yrs)
#         w = np.where(yrs == year)[0][0]
#     else:
#         try:
#             w = np.where(yrs == year)[0][0]
#         except:
#             raise ValueError(''.join(["'year' is not available; ",
#                                       "must be one of ",
#                                       ', '.join([str(x) for x in np.sort(yrs)])]))
#     # Create link for desired year
#     zip_link = links[w]
#     req_zip_link = requests.get(zip_link, stream=True)
#     zip_open = ZipFile(BytesIO(req_zip_link.content))
#     zip_files = zip_open.namelist()
#     ras = zip_files[np.where([bool(re.search(".img$", f)) for f in zip_files])[0][0]]
#     full_link = ''.join(["zip+",
#                          zip_link,
#                          "!",
#                          ras])
#     # Return
#     return full_link



# # Create table of pad links by state for easy download
# def pad_links_by_state():
#     # Find the PAD-US data page on the home site
#     home_link = "https://www.usgs.gov/core-science-systems/science-analytics-and-synthesis/gap/science/protected-areas"
#     home_req = Request(home_link)
#     home_page = urlopen(home_req)
#     home_soup = BeautifulSoup(home_page, "lxml")
#     for link in home_soup.findAll('a'):
#         if link.string is None:
#             pass
#         else:
#             if bool(re.search("data download", link.string, re.IGNORECASE)) == True:
#                 data_link = link.get("href")
#                 break
#             else:
#                 pass
#     if bool(re.search("^/", data_link)) == True:
#         home_parsed = urlparse(home_link)
#         home_base = '://'.join([home_parsed.scheme, home_parsed.netloc])
#         data_link = ''.join([home_base, data_link])
#     else:
#         pass
#     # Find the download page on the data page
#     data_req = Request(data_link)
#     data_page = urlopen(data_req)
#     data_soup = BeautifulSoup(data_page, "lxml")
#     for link in data_soup.findAll('a'):
#         if link.string is None:
#             pass
#         else:
#             if bool(re.search("data download", link.string, re.IGNORECASE)) == True:
#                 dl_link = link.get("href")
#                 break
#             else:
#                 pass
#     if bool(re.search("^/", dl_link)) == True:
#         data_parsed = urlparse(data_link)
#         data_base = '://'.join([data_parsed.scheme, data_parsed.netloc])
#         dl_link = ''.join([data_base, dl_link])
#     else:
#         pass
#     # Find the states shape page on the dl page
#     dl_req = Request(dl_link)
#     dl_page = urlopen(dl_req)
#     dl_soup = BeautifulSoup(dl_page, "lxml")
#     for link in dl_soup.findAll('a'):
#         if link.string is None:
#             pass
#         else:
#             if bool(re.search("state shapefile", link.string, re.IGNORECASE)) == True:
#                 shp_link = link.get("href")
#                 break
#             else:
#                 pass
#     if bool(re.search("^/", shp_link)) == True:
#         dl_parsed = urlparse(dl_link)
#         dl_base = '://'.join([dl_parsed.scheme, dl_parsed.netloc])
#         shp_link = ''.join([dl_base, shp_link])
#     else:
#         pass
#     # Find the download links on the shapes page
#     shp_req = Request(shp_link)
#     shp_page = urlopen(shp_req)
#     shp_soup = BeautifulSoup(shp_page, "lxml")
#     tab = shp_soup.findAll("table")[0]
#     shp_names = []
#     shp_links = []
#     for row in tab.findAll("tr"):
#         for col in row.findAll("td"):
#             for d in col.findAll("span"):
#                 shp_names.append(d.string)
#                 shp_links.append(d.get("data-url"))
#     # Format output
#     shp_parsed = urlparse(shp_link)
#     shp_base = '://'.join([shp_parsed.scheme, shp_parsed.netloc])
#     df = pd.DataFrame({"state": shp_names, "pad": shp_links})
#     pad_final = []
#     for i, row in df.iterrows():
#         try:
#             st = re.search("(?<=_state).*?(?=_shapefile)", row["state"], re.IGNORECASE).group(0)
#         except:
#             st = "NULL"
#         if st == "NULL":
#             pass
#         else:
#             print(st)
#             if bool(re.search("^/", row["pad"])) == True:
#                 zip_link = ''.join([shp_base, row["pad"]])
#             else:
#                 zip_link = row["pad"]
#             zip_open = ZipFile(BytesIO(urlopen(zip_link).read()))
#             zip_files = np.array(zip_open.namelist())
#             shps = zip_files[np.where([bool(re.search(".shp$", f)) for f in zip_files])[0]]
#             ft = [re.search("(?<=[0-9]{1}_[0-9]{1}).*?(?=(_state)|(.shp$))", x, re.IGNORECASE).group(0)
#                   for x in shps]
#             st_df = pd.DataFrame({"state_USPS": np.repeat(st, len(ft)),
#                                   "PAD_type": ft,
#                                   "PAD_file": shps,
#                                   "PAD_link": np.repeat(zip_link, len(ft))})
#             pad_final.append(st_df)
#     # Return
#     pad_concat = pd.concat(pad_final).reset_index(drop=True)
#     return pad_concat



# # Block groups -- grab the block group geometries from TIGER (only retaining 
# # the variables we want to output in our tables)
# def load_cbsa_bg_by_state(state_fips,
#                           cbsa_cty,
#                           crs):
#     # Identify the CBSAs for that state
#     cbsa_by_state = cbsa_cty[cbsa_cty.state_FIPS == state_fips]
#     # Load the block groups for that state from TIGER
#     bg = gpd.read_file(''.join(["https://www2.census.gov/geo/tiger/",
#                                 "TIGER2019/BG/",
#                                 "tl_2019_", state_fips, "_bg.zip"]))
#     bg = bg.to_crs(crs)
#     # Format for desired table output
#     bg["bg_total_area"] = bg.geometry.area
#     bg = bg.rename(columns={"GEOID": "geoid10",
#                             "STATEFP": "state_FIPS",
#                             "COUNTYFP": "county_FIPS"})
#     bg = bg[["geoid10", "state_FIPS", "county_FIPS", "bg_total_area", "geometry"]]
#     # Join to county table to only retain relevant bg
#     bg_by_cbsa = pd.merge(cbsa_by_state, bg,
#                           how="inner",
#                           on=["state_FIPS", "county_FIPS"])
#     bg_by_cbsa = gpd.GeoDataFrame(bg_by_cbsa)
#     # Return
#     return bg_by_cbsa



# # Secondary road area function
# def append_road_area(grid_gdf, road_gdf, road_width_col):
#     polys = grid_gdf.copy()
#     polys.reset_index(inplace=True)
#     lines = road_gdf.copy()
#     lines.reset_index(inplace=True)
#     sj = gpd.sjoin(left_df=lines, right_df=polys, how="inner", op="within")
#     polys["lines_geoms"] = polys['index'].apply(lambda x: sj[sj['index_right'] == x]['geometry'].tolist())
#     polys["width"] = polys["index"].apply(lambda x: sj[sj["index_right"] == x][road_width_col].tolist())
#     polys["grid_road_area_sqm"] = None
#     for idx, row in polys.iterrows():
#         sum_area = 0
#         for i in range(len(row["lines_geoms"])):
#             sum_area += row["geometry"].intersection(row["line_geoms"][i]).length * row[road_width_col][i]
#         polys.loc[idx, "grid_road_area_sqm"] = sum_area
#     return polys.drop(columns=["lines_geoms", "width"])

