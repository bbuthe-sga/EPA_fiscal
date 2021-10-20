from pathlib import Path
import pandas as pd
import shutil
from six import string_types

DATA = Path(r"K:\Projects\EPA\Fiscal_Impact\Features")
OUTPUT = Path(DATA, "OUTPUT")
RESULTS_TEMPLATE = Path(DATA, "Output_Template.csv")
FOLDERS = ["List_1", "List_2", "List_3"]
REMAKE = True
DTYPES = {"CBSA": str, "GEOID10": str, "state_FIPS": str, "county_FIPS": str}


def combine_data(output_folder, search_folders, result_template, remake=True,):
    all_data = Path(output_folder, "all_data_100421.csv")
    if not remake:
        return all_data
    elif not all_data.exists():
        shutil.copy(result_template, all_data)
    else:
        all_data.unlink()
        shutil.copy(result_template, all_data)
    template = pd.read_csv(all_data)
    for folder in search_folders:
        # TODO keep this from picking up files in subfolders ('debug', e.g.)
        files = Path(output_folder, folder).rglob("*.csv")
        for f in files:
            df = pd.read_csv(f)
            df[template.columns].to_csv(all_data, mode='a', header=False, index=False)
    return all_data


def summarize_to_geom(all_df, geom_id, cat_cols, sum_cols): # data_csv, col_dtypes=None):
    print("reading in all data")
    if isinstance(geom_id, string_types):
        geom_id = [geom_id]
    # _dtypes = None
    # if col_dtypes:
    #     _dtypes = col_dtypes
    # all_df = pd.read_csv(data_csv, dtype=_dtypes, usecols=geom_id + cat_cols + sum_cols)
    # generate categorical summary
    print("determining categorical summary by cell with largest area share")
    max_area_idxs = all_df.groupby(geom_id).cbg_area_shr.idxmax().values
    group_cats = all_df[geom_id + cat_cols].iloc[max_area_idxs]
    print("summarizing numeric data")
    group_sums = all_df.groupby(geom_id)[sum_cols].sum().reset_index()
    # merge together
    print("merging categorical and numeric data")
    join_df = group_cats.merge(group_sums, on=geom_id)

    return join_df


if __name__ == "__main__":
    categorical_main = ["state_FIPS", "county_FIPS", "CBSA", "county_loc", "CBSA_type", "CBSA_name"]
    categorical_county = ["CBSA", "county_loc", "CBSA_type", "CBSA_name"]
    categorical_cbg = []
    # summary_main = ["area", "cbg_area_shr", "sf_res_suit_lc", "other_suit_lc",
    #                 "developable_acres", "protected_acres", "soft_prot_acres",
    #                 "BOAT_LEN_1", "BOAT_LEN_2", "BOAT_LEN_3", "BOAT_LEN_4", "BOAT_LEN_5",
    #                 "BRIDGE_AREA_1", "BRIDGE_AREA_2", "BRIDGE_AREA_3", "BRIDGE_AREA_4", "BRIDGE_AREA_5",
    #                 "CTRL_AREA_1", "CTRL_AREA_2", "CTRL_AREA_3", "CTRL_AREA_4", "CTRL_AREA_5",
    #                 "ROAD_AREA_1", "ROAD_AREA_2", "ROAD_AREA_3", "ROAD_AREA_4", "ROAD_AREA_5",
    #                 "TOLL_AREA_1", "TOLL_AREA_2", "TOLL_AREA_3", "TOLL_AREA_4", "TOLL_AREA_5",
    #                 "population", "jobs", "act24"]
    # updated list of summary fields to be used in Round 3
    summary_main = ["area", "cbg_area_shr", "sf_res_suit_lc", "other_suit_lc",
                    "developable_acres", "protected_acres", "soft_prot_acres",
                    "BOAT_LEN_1", "BOAT_LEN_2", "BOAT_LEN_3", "BOAT_LEN_4", "BOAT_LEN_5",
                    "BRIDGE_AREA_1", "BRIDGE_AREA_2", "BRIDGE_AREA_3", "BRIDGE_AREA_4", "BRIDGE_AREA_5",
                    "CTRL_AREA_1", "CTRL_AREA_2", "CTRL_AREA_3", "CTRL_AREA_4", "CTRL_AREA_5",
                    "ROAD_AREA_1", "ROAD_AREA_2", "ROAD_AREA_3", "ROAD_AREA_4", "ROAD_AREA_5",
                    "TOLL_AREA_1", "TOLL_AREA_2", "TOLL_AREA_3", "TOLL_AREA_4", "TOLL_AREA_5",
                    "SF_ALLOC", "MF_ALLOC", "GQ_ALLOC", "population", "jobs", "act24",
                    "CENTER_LEN_1", "CENTER_LEN_2", "CENTER_LEN_3", "CENTER_LEN_4", "CENTER_LEN_5",
                    "LANE_LEN_1", "LANE_LEN_2", "LANE_LEN_3", "LANE_LEN_4", "LANE_LEN_5"
                    ]
    # summary_cbg = ["area", "cbg_area_shr", "population", "jobs", "act24",
    #                "developable_acres", "protected_acres", "soft_prot_acres",
    #                "ROAD_AREA_1", "ROAD_AREA_2", "ROAD_AREA_3", "ROAD_AREA_4", "ROAD_AREA_5"]
    
    # Read data
    data = combine_data(output_folder=OUTPUT, search_folders=FOLDERS,
                        result_template=RESULTS_TEMPLATE, remake=REMAKE)
    all_data = pd.read_csv(data, dtype=DTYPES)
    all_data["STCTY"] = all_data.state_FIPS.str.zfill(2) + all_data.county_FIPS.str.zfill(3)
    # ...summarize
    cells_summary = summarize_to_geom(all_df=all_data, geom_id="cell_id", # col_dtypes=DTYPES,
                                      cat_cols=categorical_main, sum_cols=summary_main)
    county_summary = summarize_to_geom(all_df=all_data, geom_id="STCTY", # col_dtypes=DTYPES,
                                       cat_cols=categorical_main, sum_cols=summary_main)
    cbg_summary = summarize_to_geom(all_df=all_data, geom_id="GEOID10", # col_dtypes=DTYPES,
                                    cat_cols=categorical_cbg, sum_cols=summary_main)
    # OUTPUT
    out_grid = Path(OUTPUT, "Cells_CONUS_CBSAs_summary_100421.csv")
    out_cbg = Path(OUTPUT, "CBG_CONUS_summary_100421.csv")
    out_county = Path(OUTPUT, "County_CONUS_summary_100421.csv")
    # ...to csv
    cells_summary.to_csv(out_grid)
    cbg_summary.to_csv(out_cbg)
    county_summary.to_csv(out_county)
    print()
