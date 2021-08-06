#Script to create a grid over a shapefile of micro and metro areas without parks and water. 


# 01 Load in libraries ----------------------------------------------------

        library(easypackages)
        
        
        libraries("sf", "raster", "ggplot2", "leaflet", "tigris",
                  "tidyverse", "tidycensus", "scales", "survye", 
                  "reshape2", "readxl", "srvyr", "arcgisbinding", 
                  "rgdal", "rgeos", "dismo", "data.table", "rcartocolor", "tmaptools")
        
        arc.check_product()

        pb <- txtProgressBar(min = 0, max = 100, style = 3)
        for(i in 1:100) {
          Sys.sleep(0.1)
          setTxtProgressBar(pb, i)
        }
        close(pb)
        
        
# 02_1 Load in micro and metro shapefiles ---------------------------------------------------
    
    #For all shapefiles in geodatabases, load in the gdb, then tell st_read what the layer name is
        geodata <- "C:/Users/Becca Buthe/Documents/ArcGIS/Projects/EPA/EPA.gdb"
          
    # List all feature classes in a file geodatabase
        st_layers(geodata)
    
    #CBSA with water and parks erased
    cbsa<-st_read(dsn = geodata, layer = "tl_2019_us_cbsa_albers") #This is all micro and metro areas
    
    cbsa_erased<-st_read(dsn = geodata, layer = "tl_2019_us_cbsa_albers_Erase_water_allparks")
    st_crs(cbsa_erased)
    
    #Separate metro and micro data. 
    metro<-cbsa_erased%>%
      filter(MEMI == 1)
    
    micro<-cbsa_erased%>%
      filter(MEMI == 2)
    
#Call in trial data to complete fishnet code on
    names(cbsa)
    trial<-st_read(dsn = geodata, layer = "trial") #This is a single metro area in NV to use for example scripting

    trial


    # 02_2 Get data from census if any is needed outside the smart location 2018 data ----------------------------------------------------
    
    state_vec<-c("AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
                 "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
                 "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
                 "TX","UT","VT","VA","WA","WV","WI","WY","DC")
    
    msavars <- c(total_pop = "B01003_001", median_income = "B19013_001") #put desired variables here
    
    #Call in the data with the function get_acs - Trial data
    acs19_nv<-get_acs(geography = "block group", state = "NV",  #inputs from get_acs funtion
                   variables = msavars, year = 2019, geometry = TRUE)
    
    options(tigris_use_cache = TRUE)


    # 02_3 EPA smart location data -------------------------------------------------
    
    #Geodatabase for smartlocation data
    epageodata<-"C:/Users/Becca Buthe/Documents/ArcGIS/Big_data/EPA/SmartLocationDatabase.gdb"
    
    #List all feature classes in geodatabase
    st_layers(epageodata)
    
    #Choose which feature class to open. 
    epasmartloc<-st_read(dsn = epageodata, layer = "EPA_SLD_Database_V3")
    
    #View field names to select only relevant ones. 
    names(epasmartloc)


    # 02_4 Water bodies and inundated areas -------------------------------------
  
  #NHGS geodatabase https://www.usgs.gov/core-science-systems/ngp/national-hydrography/access-national-hydrography-products and "Download the NHD by the Entire Nation" 
    nhgsgeo<-"C:/Users/Becca Buthe/Documents/ArcGIS/Big_data/EPA/NHD_H_National_GDB.gdb"
    
    #Feature classes
    st_layers(nhgsgeo)
    
    water_nolakes<-st_read(dsn = nhgsgeo, layer = "NHDArea")
    lakes<-st_read(dsn = nhgsgeo, layer = "NHDWaterbody")
    
    ggplot()+
      geom_sf(data = water_nolakes)
    
 
    # 02_5 Parks --------------------------------------------------------------

#National park data - https://public-nps.opendata.arcgis.com/datasets/nps::nps-boundary-1/about    
    
    #All park data https://www.arcgis.com/home/item.html?id=578968f975774d3fab79fe56c8c90941
    
    allparks<-st_read(dsn = geodata, layer = "Parks")
    
    ggplot()+
      geom_sf(data = allparks)
    
    # Grid attempt #1 FAIL --------------------------------------------------------
        #Didn't work because changing the raster to a vector required too much memory even for 
        #a single msa...
        #https://rfunctions.blogspot.com/2014/12/how-to-create-grid-and-intersect-it.html
        
        help(readOGR)
        trialogr<-readOGR(dsn = geodata, layer = "trial")
        
        grid<-raster(extent(trial))
        res(grid)<-2
        
        projection<-proj4string(trialogr)
        
        proj4string(grid)<-proj4string(trialogr)
        
        gridpolygon<-rasterToPolygons(grid)
        
        memory.size() ### Checking memory size
        memory.limit(100000) ## Checking the set limit
        

    # 02_6 Roads  ----------------------------------------------
        #Best I could find so far is roads from BTS
        #Roads from BTS https://www.bts.gov/geography/geospatial-portal/NTAD-direct-download
        #Loaded in GIS 
        
        
help("arc.write")
# 03 Create a .5 mile squared grid ---------------------------------------------------------
      
      #Plot the data to check what it looks like
          ggplot() + 
          geom_sf(data = trial)
      
      #Check the Projections of all the data and transform the ones that are not Albers Equal Area Conic
          crs(trial)
          crs(acs19_nv)
          acs19_nvproj <- st_transform(acs19_nv, crs(trial))
          st_crs(acs19_nvproj)
          
          
          st_crs(acs19_nvproj)
          st_crs(trial)
          
      #   st_intersection(acs19_nvproj)
      

#this trial is in Nevada - 
      grid<-trial%>%
        st_make_grid(cellsize = c(804.672,804.672), what = "polygons")%>% #1320 is a quarter mile in feet, 804.672 in half a mile in meters 
        #Cellsize in st_make_grid is in whatever units the shapefile is in and in this case it is meters. 
        st_intersection(trial) #Intersect the grid with the shapefile. This will need to be 
        #iterated when it moves to the data for the whole country

#Try on just the micro areas TOOK FOREVER AND A YEAR. 
      grid2<-micro%>%
        st_make_grid(cellsize = c(2000,2000), what = "polygons")%>%
        st_intersection(micro)

#Plot the data to see what it looks like. 
      ggplot()+
        geom_sf(data = trial) +
        geom_sf(data = grid)
        #geom_sf(data = acs19_nvproj)
      
      ggplot()+  
        geom_sf(data = acs19_nvproj)
      crs(acs19_nvproj)
      

# 06 Calculate the area of each grid, and the proportion it is of  each BG --------

      
help("st_area")
      grid$area <- st_area(grid) # All the full squares are 647497 so it is the square of what the cellsize was set as.     
      
      
      grid$area
      
      

# Intersect the Grid with the data of interest for each cell -------------

      

# California bgs for data check -------------------------------------------

      acs19<-get_acs(geography = "block group", state = "CA",  #inputs from get_acs funtion
                     variables = msavars, year = 2019, geometry = TRUE)  
      
      arc.write(path = "C:/Users/Becca Buthe/Documents/ArcGIS/Big_data/EPA/cabgs.shp", data = acs19)
      
      
