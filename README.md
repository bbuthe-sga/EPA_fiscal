# EPA_fiscal
Fiscal impact analysis for micro and metropolitan areas in the US. 


# 01 DATA SOURCES

CBSAS - Census tigerline data https://catalog.data.gov/dataset/tiger-line-shapefile-2019-nation-u-s-current-metropolitan-statistical-area-micropolitan-statist

Water bodies and inundated areas - USGS data, two files - NHDWaterbody and NHDArea https://www.usgs.gov/core-science-systems/ngp/national-hydrography/access-national-hydrography-products

Parks - National parks from NPS boundary file - https://public-nps.opendata.arcgis.com/datasets/nps::nps-boundary-1/about
Local and regional parks from ESRI https://www.arcgis.com/home/item.html?id=578968f975774d3fab79fe56c8c90941

Roads - Shapefile from BTS specifically the HPMS data, has problems with incomplete coverage in micro areas https://www.bts.gov/geography/geospatial-portal/NTAD-direct-download
More info on data found here: https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm

EPA Smartlocation data - https://www.epa.gov/smartgrowth/smart-location-mapping


# 02 PRELIM DATA CLEANING AND MANIPULATION

1. Loaded the CBSA shapefile into GIS. Erased all water bodies and areas from the file. This includes submerged and inundated areas, but a spot check says those are all unpopulated anyway so it increases accuracy. 
2. Clipped all roads to the orignal complete CBSA file. This took about 30 or so minutes to complete. It was here that I noticed that the file is not great in terms of completeness for micropolitan areas, though the spot check I did on metro areas checked out more. 
3. Loaded the parks shapefile into GIS and noticed that the NPS data seems like it may not actually cover all national parks, while the ESRI shapefile actually does great for regional, state and local parks but for national parks it includes far too large of areas (look to CO for an example) where there are definitely populations living there. Need to find a better file or just use the ESRI one for all but national parks, and use the NPS file for national parks. 
4. 

# 03 ANALYSIS

1. Now working in R, called in all spatial data and converted it to an Albers Equal Area Conic projection to preserve area for the fishnet creation. 
2. Started with a trial MSA in Nevada - created a half mile squared fishnet over the MSA. 
3. Will next need to see if I can create a fishnet over all the MSAs overnight, and if we can find better road shapefiles, if not then we need to do a random sampling. 
