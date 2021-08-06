# EPA National Fiscal Impact Modeling Data Development
This repository includes procedures for data development and geoprocessing
to support the development of a generalized fiscal impact model for micro
and metropolitan areas in the US.

Fiscal impact modeling will generally estimate "infrastructure efficiency"
based on "24-hour density" for a local geographic area - in this case,
a 0.5 square mile cell. Infrastructure efficiency is defined as the units of
infrastructure per person. Past studies indicate this number is influenced
by the 24-hour density, which is the number of residents plus employees per
acre in the local geographic area. Hence, data development for fiscal impact
modeling focuses on developing a robust set of local area cells with
attributes supporting the calculation of infrastructure efficiency and 24-hour
density.

All prior studies have been limited to distinct metropolitan contexts, each
having its own characteristics in terms of the correlation between
infrastructure efficiency and 24-hour density. This study aims to develop
a national model that controls for differences among settings by including
state and metropolitan level variables (factors, size, avarege density, etc.)
as well as additional block group level variables, such as accessibility
measures that can account for a location's position within a larger region.
Hence, data development for fiscal impact modeling also supports the
enrichment of local area cells with these attributes based on their spatial
relationships to state, CBSA, and block group features.

The outputs of the procedures in this repository include:
- Local area cells (`LOC_CELLS`), including at least the following
attributes:
    - `cell_id`
    - `state_FIPS`
    - `county_FIPS`
    - `CBSA`
    - `CBSA_type` (metropolitan/micropolitan)
    - `county_loc` (central/outlying for the CBSA)
    - `population`
    - `jobs`
    - `act24` (24-hour activity)
    - `road_area` (estimate of total roadway area)
    - `infra_eff` (`road_area` per `act24`)

- Block group-to-cells relationship (`CELLS_CBG`), including at least the
following attributes:
    - `cell_id`
    - `geoid10` (12-digit block group id)
    - `area` (geometric area of overlap between cell and block group)
    - `cbg_area_shr` (`area` as a share of total block group area)
    - `sf_res_suit_rd` (estimate of weighted roadway suitability for single family
    residences)
    - `other_suit_rd` (estimate of weighted roadway suitability for non-residential
    and multifamily residences)
    - `sf_res_suit_rd` (estimated of weighted land cover suitability for single family
    residences)
    - `other_suit_lc` (estimated of weighted land cover suitability for non-residential
    and multifamily residences)

These tables will allow the local area cells to be enriched with other
attributes associated with block groups and CBSA's. The specific variables
to be included in the `LOC_CELLS` table will be finalized in the model
estimation process. The content that follows focuses on how the basic
attributes listed above are developed.

## DATA SOURCES
- **Local area cells**: Generated as a novel dataset

- **CBSA's**: To assign each cell to a CBSA
    - Boundary files: Census tigerline data -
    https://catalog.data.gov/dataset/tiger-line-shapefile-2019-nation-u-s-current-metropolitan-statistical-area-micropolitan-statist
    - County designations -
    https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2020/delineation-files/list1_2020.xls 

- **Block groups**: To assign jobs, population, and other variables to each
cell
    - Boundary files: ...
    - Population in households vs. in group quarters: ACS 2019 5-year
    estimates Table B09019
    - Population by residential unit type: ACS 2019 5-year estimates
    Table B25033
    - Jobs: LODES WAC tables (2018)

- **Roads**: To summarize road area and road-based activity suitability to
cells
    - Option 1: ARNOLD - acquisition through FHWA by email request
        - All streets as features
        - Number of lanes tabulated for HPMS, can generally assume 2 lanes
        elsewhere
    - Option 2: HERE/NAVSTREETS - EPA license extends to contractors
        - All streets as features
        - Number of lanes approximated in "lane category" fields
    - Option 3: HPMS data -
    https://www.bts.gov/geography/geospatial-portal/NTAD-direct-download
        - Incomplete coverage in micro areas 
        - More info on data found here: 
        https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm

- **Land Cover**: To support allocation of jobs and population from block
groups to cells
    - National Land Cover Database (NLCD) indicates land cover type 
    and development intensity
        - Non-residential and multifamily/group quarters residents more likely
        in higher development intensities
        - Single-family residents more likely in lower development intensities
        - Nominal suitability for activity in undeveloped land cover classes
        - No suitability for activity in inundated areas
    - Water bodies and inundated areas (supplemental to NLCD)
        - USGS NHDWaterbody and NHDArea (2 files) -
        https://www.usgs.gov/core-science-systems/ngp/national-hydrography/access-national-hydrography-products

- **Development Limitations**: To limit the amount of jobs and population
estimated in protected areas
    - Protected Areas Dataset (PAD-US)
    - National parks from NPS boundary file - 
    https://public-nps.opendata.arcgis.com/datasets/nps::nps-boundary-1/about
    - Local and regional parks from ESRI -
    https://www.arcgis.com/home/item.html?id=578968f975774d3fab79fe56c8c90941
    - Digital evelation models (to limit jobs and population estimated in
    areas with steep slopes)

- **Other Datasets**: To enrich the fiscal impact modeling process
    - EPA Smart Location Database - 
    https://www.epa.gov/smartgrowth/smart-location-mapping


## PROCESSING STEPS
*TODO: add script references for each step/substep*

### 1. Data Acquisition
- All source data are available from a shared data store, readable by code in
this repository
- Raw data are stored in the RAW folder. Do not modify these datasets or save
any processed data here.

### 2. Data Preparation
- Generate local area cell features 
    - Assign each cell's  `cell_id`, `state_FIPS`, `county_FIPS`,
    `CBSA`, `CBSA_type`, `county_loc` attributes based on centroid location

- Development suitability processing:
    - Create two suitabiity surfaces - `sf_res_suit_lc` and `other_suit_lc`.
    - Apply flags to features for filtering (which protected areas or
    hydropgraphy features to exclude, e.g.)
    - Calculate DEM slopes and set thesholds for screening.
    - Codify suitability rubrics
    (https://enviroatlas.epa.gov/enviroatlas/datafactsheets/pdf/Supplemental/Dasymetricallocationofpopulation.pdf).
    - Reclass NLCD categories using suitability rubrics and screen out all
    unsuitable areas based on protections, inundation, etc.

- Roads data:
    - Create `road_area` field estimating road area based on linear distance,
    number of lanes, estimated lane width, etc.
    - Codify road suiatbility rubrics.
    - Create `sf_res_suit_rd` and `other_suit_rd` fields based on `road_area`
    and suitability rubrics.

- CBSA data:
    - Tabulate CBSA population, jobs totals, density averages, etc.

- Block group data:
    - Tabulate population and jobs with stratifications
        - Population in households in single-family residences
        - Remaining population

- Configure all dataset and field references in the code

### 3. Intersect cells with block groups and roads
This process is chunked to handle the large number of features implied by a
national-scale analysis.

- Find the spatial intersection of each cell with all overlapping block
groups.
- For each cell/block group combination...
    - Tabulate the cumulative suitability from `sf_res_suit_lc` and
    `other_suit_lc`.
    - Find the spatial intersection with overlapping roads.
    - Tabulate cumulative `road_area`, `sf_res_suit_rd`, and `other_suit_rd`
    from overlapping roads.
    - Combine `sf_res` and `other` suitability fields into composite scores,
    `sf_res_suit`, `other_suit`
    - Write overlap feature records with key attributes to `CELLS_CBG`

### 4. Calculate cell shares of block group attributes
- `area`
- `sf_res_suit`
- `other_suit`

### 5. Distribute block group level activities proportionally
- Tabulate estimates in `LOC_CELLS`
    - Population in single family homes by `sf_res_suit`
    - Other population by `other_suit`
    - Jobs by `other_suit`
    - `act24`

### 6. Summarize infrastructure efficiency
- Tabulate estimates in `LOC_CELLS`
    - `road_area`
    - `infra_eff`
