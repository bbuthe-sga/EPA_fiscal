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
    - `devlopable_acres` (developable area excluding water, protected area, etc.)
    - `protected_acres` (area subjected to protections based on Protected Areas Dataset)
    - `act24` (24-hour activity)
    - `dens24` (`act24` per `developable_acres`)
    - `road_area` (estimate of total roadway area derived by summing fields from `CELLS_CBG`)
    - `infra_eff` (`road_area` per `act24`)

- Block group-to-cells relationship (`CELLS_CBG`), including at least the
following attributes:
    - `cell_id`
    - `geoid10` (12-digit block group id)
    - `state_FIPS`
    - `county_FIPS`
    - `CBSA`
    - `CBSA_type` (metropolitan/micropolitan)
    - `county_loc` (central/outlying for the CBSA)
    - `area` (geometric area of overlap between cell and block group)
    - `developable_acres` (developable area excluding water, protected area, etc.)
    - `protected_acres` (area subjected to protections based on Protected Areas Dataset)
    - `cbg_area_shr` (`area` as a share of total block group area)
    - `sf_res_suit_lc` (estimated of weighted land cover suitability for single family
    residences)
    - `other_suit_lc` (estimated of weighted land cover suitability for non-residential
    and multifamily residences)
    - Road area summaries (`X` in the field name indicates the FUNCTIONAL_CLASS of roads
    overlapping this area)
        - `ROAD_AREA_X` - The estimated total area of roads based on link width assumptions
        and length intersecting ths area. Unpaved and private/non-publis access roads are
        excluded from all road area estimates as are ferries of any kind.
        - `BRIDGE_AREA_X` - The estimated area of roads that are bridges or tunnels. This
        number is included in `ROAD_AREA_X`. This column can be used to net out bridges and
        tunnels from the total estimated road area.
        - `CTRL_AREA_X` - The estimated area of roads that are controlled/limited access
        highways. This number if included in `ROAD_AREA_X`. This column can be used to net
        out controlled access highways from the total estimated road area.
        - `TOLL_AREA_X` - The estimated area of roads that are tollways. This number is
        included in `ROAD_AREA_X`. This column can be used to net out tollways from the
        total estimated road area. Most tollways are also controlled access highways. The
        area of non-toll controlled access highways could usually be obtained by
        subtracting `TOLL_AREA_X` from `CTRL_AREA_X`, but this may not be reliable in all
        cases.
        - `BOAT_LEN_X` - The total length of ferry links operating in this area. Ferries
        are not reflected in `ROAD_AREA_X`. 

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
    - Boundary files: The Smart Location Database (SLD) features are used for this
    analysis. https://www.epa.gov/smartgrowth/smart-location-mapping
    - Population in households vs. in group quarters: ACS 2019 5-year
    estimates Table B09019
    - Population by residential unit type: ACS 2019 5-year estimates
    Table B25033
    - Jobs: SLD (based on LODES WAC tables for 2018)

- **Roads**: To summarize road area and road-based activity suitability to
cells
    Line files: HERE/NAVSTREETS - EPA license extends to contractors
        - All streets as features
        - Number of lanes approximated in "lane category" fields on  per-direction basis.
        - See `reclasses.xlsx` for details on how attributes are used to estimate
        road width ("ROADS_LANES" tab) and summarize road area ("ROADS_ATTRIBUTES" tab).

- **Land Cover**: To support allocation of jobs and population from block
groups to cells
    - National Land Cover Database (NLCD) indicates land cover type 
    and development intensity
        - Non-residential and multifamily/group quarters residents more likely
        in higher development intensities
        - Single-family residents more likely in lower development intensities
        - Nominal suitability for activity in undeveloped land cover classes
        - No suitability for activity in inundated areas

- **Development Limitations**: To limit the amount of jobs and population
estimated in protected areas
    - Protected Areas Dataset (PAD-US)

- **Other Datasets**: The SLD may be referenced to provide additional
attributes to enrich the model estimation process.


## PROCESSING STEPS
*TODO: add script references for each step/substep*

### 1. Data Acquisition
*(task_1_data_creation.py)*
- All source data are available from a shared data store from which data
can be downloaded to a local or network directory. The code in this repository
allows the user to update the path to the `DATA` directory.
- Raw data are stored in the RAW folder. Do not modify these datasets or save
any processed data here.

### 2. Data Preparation
*(task_1_data_creation.py)*
- Development suitability processing:
    - Codify suitability rubrics
    (https://enviroatlas.epa.gov/enviroatlas/datafactsheets/pdf/Supplemental/Dasymetricallocationofpopulation.pdf).
    See `reclasses.xlsx` "NLCD" tab for suitability details.
   - Apply flags to features for filtering (which protected areas to exclude, e.g.)
   See `reclasses.xlsx` "PAD" tab for protected areas factoring details.
    - Reclass NLCD categories using suitability rubrics and screen out/factor all
    unsuitable areas based on protections, inundation, etc.
    - Create two suitability surfaces - `sf_res_suit_lc` and `other_suit_lc`.

- Roads data:
    - Codify lane count and width assumptions. See `reclasses.xlsx` "ROADS_LANES".
    - Cofify road area summary gruopings. See `relcasses.xlsx` "ROADS_ATTRIBUTES".
    - Calculate `link_area` based on linear distance, number of lanes, and estimated lane width.

- CBSA data:
    - Tabulate CBSA population, jobs totals, density averages, etc.

- Block group data:
    - Tabulate population and jobs with stratifications
        - Population in households in single-family residences
        - Remaining population

- Configure all dataset and field references in the code

### 3. Intersect cells with block groups and roads
*(task_1_data_creation.py)*
This process is chunked to handle the large number of features implied by a
national-scale analysis. Each CBSA is handled independently with the analysis
iterating over each constituent county to limit the total quantity of features
analyzed.

- Find the spatial intersection of each cell with all overlapping block
groups.
- For each cell/block group combination...
    - Tabulate the cumulative suitability from `sf_res_suit_lc` and
    `other_suit_lc`.
    - Find the spatial intersection with overlapping roads.
    - Tabulate cumulative road areas from overlapping roads, summarizing by
    attributes.
    - Write overlap feature records with key attributes to `CELLS_CBG`

### 4. Calculate cell shares of block group attributes
*(task_1_data_creation.py)*
Calculate each cell/block group combination's share of total block group
area and suitability (res and other). These will be used to facilitate
disaggegation of block group-level data to the cell level.

### 5. Distribute block group level activities proportionally
*(task_1_data_creation.py)*
- Tabulate estimates for each cell/block group combination.
    - Population in single family homes by `sf_res_suit`
    - Other population by `other_suit`
    - `population` = single-family + other population
    - `jobs` by `other_suit`
    - `act24` is the sum of `jobs` and `population`.

### 6. Summarize cell/block group combinations to cell level
*(script TBD)*
- Summarize `CELLS_CBG` records by `cell_id` to generate `LOC_CELLS` table.
    - Assign each cell's  `cell_id`, `state_FIPS`, `county_FIPS`,
    `CBSA`, `CBSA_type`, `county_loc` attributes based on the attribtes
    associated with the block group having the largest overlap with
    this cell.
- Summarize `population`, `jobs`, `act24`
- Summarize all road attributes

### 6. Summarize infrastructure efficiency
*(script TBD)*
- Tabulate estimates in `LOC_CELLS`.
    - `road_area`: TBD - add/subtract column values reflecting different
    road classes and subtypes (bridge/tunnel, e.g.) to get the total
    relevant road area for regression purposes.
    - `infra_eff`: `road_area` per `act24`

### 7. Enrich dataset with other attributes
*(script TBD)*, applied as needed.
- SLD attributes based on block groups in `CELLS_CBG`
    - Area-based averaging
    - Suitability-based averaging
- CBSA-level attributes based cell column `CBSA`





- Generate local area cell features 
    - Assign each cell's  `cell_id`, `state_FIPS`, `county_FIPS`,
    `CBSA`, `CBSA_type`, `county_loc` attributes based on maximum area of overlap
    with census block features