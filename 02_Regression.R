#OLS regression of population and streets
#Load libraries

library(easypackages)

libraries("tidyverse", "sf", "car", "corrplot", "stargazer", "plotly", "gtools", "GGally",
          "scales", "sensitivity", "konfound", "ggpmisc", "readr", "caTools", "tidycensus", "caret")

# SGA colors --------------------------------------------------------------
T1<-c("#0082C8")
T2 <- c("#C3CF21")
T3 <- c("#FDBB30")
blue <- c("#0082C8")
navy <- c("#00245D")
green <- c("#C3CF21")
grey <- c("#ACA196")
lightorange <- c("#FDBB30")
darkorange <- c("#F05133")
darkgrey <- c("#333333")

      
# Final data --------------------------------------------------------------
#Data from RPG analysis      
     
      cbsa<-read.csv("Cells_CONUS_CBSAs_summary_092721.csv")
      head(cbsa)
      
      #SElected CBSAs to compare to sidewalk sewer and water data. 
      # cbsa_select<-cbsa%>%
      #   filter(CBSA_name %in% c("New YOrk-Newark-Jersey City, NY-NJ-PA", "Los Angeles-Long Beach-Anaheim, CA", 
      #                           "Chicago-Naperville-Elgin, IL-IN-WI", "Boston-Cambridge-Newton, MA-NH",
      #                            "Washington-Arlington-Alexandria, DC-VA-MD-WV", "Seattle-Tacoma-Bellevue, WA"))
      # 
      # write.csv(cbsa_select, "cbsa_selected.csv")

      
      cbsa$CBSA_name      
      names(cbsa)
      cbsabg<-read.csv("CBG_CONUS_summary_092721.csv")
# 
#      names(cbsabg)
     
      cbsabg_analysis<-select(cbsabg, GEOID10, area, jobs, population, act24, developable_acres,
                              ROAD_AREA_2, ROAD_AREA_3, ROAD_AREA_4, ROAD_AREA_5, CENTER_LEN_2, CENTER_LEN_3, CENTER_LEN_4, CENTER_LEN_5)
      
      cbsabg_analysis_sum<-cbsabg_analysis%>%
        group_by(GEOID10)%>%
        mutate(popden = (act24/developable_acres), 
               roads = sum(ROAD_AREA_3, ROAD_AREA_4, ROAD_AREA_5), 
               centerline = sum(CENTER_LEN_3, CENTER_LEN_4, CENTER_LEN_5), 
               waterratio = centerline/roads)%>%
        filter(act24>0, 
               developable_acres>0, 
               roads>0)
      
      #Use the centerline road data as a proxy for waterlines - 
      #Found the ratio of centerline for only ROAD3-5 (no highways) and used the median as the number for acres to centerline feet of water line. 
      #70 feet of water pipe per acre of road. 
      
      waterratio_data<-cbsabg_analysis_sum %>%
        filter(between(cbsabg_analysis_sum$waterratio, quantile(waterratio, 0.01), quantile(waterratio, 0.95)))
      summary(waterratio_data$waterratio)
      hist(waterratio_data$waterratio)
      
      quantile(cbsabg_analysis_sum$waterratio, 0.95)
      
      #write.csv(cbsabg_analysis_sum, "cbsabg_s.csv")
     
      
      #Check on the data
      #  summary(cbsacbg$developable_acres)
      # hist(cbsacbg$developable_acres)
      
      
      
      
#Selecting and manipulating desired fields
    #Protected area - could exclude any cells that have protected area in them. 
    #Developable acreage is everything you could build on. 
    
      cbsa_analysis <-cbsa %>%
        # select(ROAD_AREA_2, ROAD_AREA_3, ROAD_AREA_4, ROAD_AREA_5, 
        #        population, developable_acres, act24, jobs, CBSA, CBSA_name, area
        #        )%>%
      replace(is.na(.), 0)%>% #replace NA values with 0 so they can be filtered out. 
      mutate(road_area = (ROAD_AREA_3+ROAD_AREA_4+ROAD_AREA_5), #sum all types of road areas - clarify classifications
             centerline = (CENTER_LEN_3+CENTER_LEN_4+CENTER_LEN_5),
               popden = act24/developable_acres, 
               #poponlyden = population/developable_acres,
               #jobden = jobs/developable_acres,
               roadarea_perpop = road_area/act24,
               log_roadareaperpop = log(roadarea_perpop), 
               logpopden = log(popden),
               devacres_share = (developable_acres/area)
             )%>%
      
      filter(road_area>0,    #Get rid of all areas with no road or no population/jobs
             act24>0,  
             developable_acres>0,
             devacres_share<=1.1
             )
      
   #    summary(cbsa_analysis$act24)
   #    see<-cbsa_analysis%>%
   #      filter(act24>2000)
   #    
   # names(cbsa)
# Determine categories for the cbsa analysis by population and land area ------------------------------

      
      cbsafacts<-cbsa_analysis%>%
        group_by(CBSA)%>%
        summarise(CBSA_pop = sum(population, na.rm= T), 
                  CBSA_devacres = sum(developable_acres, na.rm = T),
                  CBSA_totacres = sum(area, na.rm = T), 
                  CBSA_protected = sum(protected_acres, na.rm = T),
                  CBSA_name = CBSA_name)%>%
        unique(.)
             
     # write.csv(cbsafacts, "cbsafacts.csv")
        #           )%>%
        # mutate(cbsa_popcat = "Category", 
        #        cbsa_landcat = "Category")
      
      
      #Create categorical variable with q different categories for both land area and
      #population of the CBSAs as another mode of analyzing/categorizing
      # cbsafacts$cbsa_popcat<-quantcut(cbsafacts$CBSA_pop, q = 10, na.rm = T)
      # cbsafacts$cbsa_landcat<-quantcut(cbsafacts$CBSA_devacres, q = 10, na.rm = T)
      # 
      
      #Get rid of outliers based on road area  
      # cbsa_nooutliers<-cbsa_analysis %>%
      #   filter(between(road_area, quantile(road_area, 0.01), quantile(road_area, 0.99)))
      #Get rid of outliers based on population density. 
      cbsa_nooutliers<-cbsa_analysis %>%
        filter(between(popden, quantile(popden, 0.01), quantile(popden, 0.99)))
      #filter(between(popden, quantile(popden, 0.02), quantile(popden, 0.98)))
    help(between)
      
      # see<-cbsa_nooutliers%>%
      #   filter(act24>2000)

      # Join summarized data to the original data set -----------------
  
      cbsa_append<-left_join(cbsa_nooutliers, cbsafacts, by = "CBSA")
  
     # write.csv(cbsa_append, "cleaninputdata.csv")
      

# Regression for each CBSA ------------------------------------------------


      lm_model_cbsa <- cbsa_append %>%
        split(.$CBSA)%>% #groups into the unique values of the CBSA field
        map(~lm(logpopden~log_roadareaperpop, data = .)) #the long code version of this is map(function(df)lm(mpg~wt, data = df))
      
      
      #from the summary, extract the R2, intercept and slope
   
  
      intercept<-lm_model_cbsa%>%
        map(summary)%>%
        map(c("coefficients")) %>% 
        map_dbl(1) %>%
        as.data.frame()%>%
        rename(intercept = ".")
      
      stderr<-lm_model_cbsa%>%
        map(summary)%>%
        map(coef) %>% #pulls out the coefficients coef(summary(lm_model_usa))[, "Std. Error"]
        map_dbl(3) %>%
        as.data.frame()%>%
        rename(stderr = ".")
    
      
      slope<-lm_model_cbsa%>%
        map(summary)%>%
        map(c("coefficients")) %>% 
        map_dbl(2) %>%
        as.data.frame()%>%
        rename(slope = ".")
      
      rsquared<-lm_model_cbsa%>%
        map(summary)%>%
        map_dbl("r.squared") %>% 
        as.data.frame()%>%
        rename(rsquared = ".")
   
       
      
     #bind everything together 
     lm_models_cbsa<- cbind(intercept, slope)%>%
        cbind(., rsquared)%>%
       cbind(., stderr)%>%
       rownames_to_column(., "CBSA")%>%
       mutate(CBSA = as.integer(CBSA))
     
     lm_models_cbsapop<-left_join(lm_models_cbsa, cbsafacts, by = "CBSA")
 
     # write.csv(lm_models_cbsapop, "lm_models_cbsapop.csv")




 
     #cbsa_clean<-read.csv("cleaninputdata.csv")
     
     

# CBSA to BGID ------------------------------------------------------------

     library(easypackages)
     libraries("tidyverse", "tidycensus", "sf", "sp", "arcgisbinding")
     
     #for info on the arcgis binding package. 
     
     arc.check_product()
     
     #Load the geodatabase
     geodata <- "C:/Users/Becca Buthe/Documents/ArcGIS/Projects/EPA/EPA.gdb"
     
     # List all feature classes in a file geodatabase
     st_layers(geodata)
     
     cbsatobg<-st_read(dsn = geodata, layer = "cbsatobg")%>%
       st_drop_geometry()
     
     cbsatobg1<-cbsatobg%>%
       mutate(GEOID10 = as.numeric(GEOID10))
     
     #get only relevant BGs
     cbsajoin<-left_join(cbsabg_analysis_sum, cbsatobg1, by = "GEOID10")%>%
       select(-CSA, -CSA_Name, -COUNTHU10, -Shape_Length, -Shape_Area, -TRFIPS, -CFIPS, -SFIPS)

     

# OUTPUTS  ----------------------------------------------------------------
     
     
     write.csv(cbsafacts, "cbsafacts3.csv")
     write.csv(cbsa_append, "cleaninputdata3.csv")
     write.csv(lm_models_cbsapop, "lm_models_cbsapop3.csv")
     write.csv(cbsabg_analysis_sum, "cbsabg_analysis.csv")
     write.csv(cbsajoin, "cbsa_to_bg.csv") 
     
     
     