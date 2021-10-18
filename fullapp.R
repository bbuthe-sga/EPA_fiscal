library(shiny)
library(shinythemes)
library(shinyWidgets)
library(shinydashboard)
library(shinysky)
library(rmarkdown)
library(shinyBS)
library(plyr)
library(dplyr, warn.conflicts = FALSE)
library(ggplot2, warn.conflicts = FALSE)


# Get clean input data ----------------------------------------------------

cbsafacts<-read.csv("cbsafacts3.csv")
CBSA_choice<-cbsafacts[,6]

lm_cbsa<-read.csv("lm_models_cbsapop3.csv")

#cbsabg<-read.csv("cbsabg_analysis.csv")

cbsatobg<-read.csv("cbsa_to_bg.csv")

oandm<-read.csv("oandm.csv")

# 2.0 Set UI for shiny --------------------------------------------------------



ui<- dashboardPage(
  dashboardHeader(title = "Infrastructure Efficiency Modeling"), 
  
  dashboardSidebar(  
                     menuItem("Infrastructure Efficiency Modeling Tool", icon = icon("th"), tabName = "Menu"),
                     menuItem("EPA interactive map", icon = icon("external-link-square"), href = "https://epa.maps.arcgis.com/home/webmap/viewer.html?webmap=f16f5e2f84884b93b380cfd4be9f0bba"),
                     menuItem("Github", icon = icon("github"), href = "https://github.com/bbuthe-sga/EPA_fiscal.git"),
                     menuItem("About", icon = icon("question-circle-o"), tabName = "menu_about")
    ),                  
  
  dashboardBody(theme = shinytheme("flatly"), 
                
                tabItems(
                          
              tabItem(tabName = "Menu",
                fluidRow(
                  
                  fluidRow(column(12,
      # 2.1 Location 1 input box --------------------------------------------------
                                  
                                  shinydashboard::box(title = "Location 1 Inputs", width = 4, solidHeader = T, status = "primary",  #Status is what changes the color, width is column width out of 12, can also use color
                                                      
                                                      
                                                      # selectInput(
                                                      #   #create the drop down to choose a metro or micropolitan area. 
                                                      #   inputId = "Location",
                                                      #   label ="Metro or micro area
                                                      #   where project is located",
                                                      #   choices = CBSA_choice,
                                                      #   selected = NULL,
                                                      #   multiple = FALSE,
                                                      #   selectize = TRUE,
                                                      #   width = NULL,
                                                      #   size = NULL),
                                                      p(HTML("<b>Metro or micro area where project is located</b>")),
                                                      textInput.typeahead(id="Location",
                                                                          placeholder="Type Metro or Micro area",
                                                                          local=data.frame(name=c(CBSA_choice)),
                                                                          valueKey = "name",
                                                                          tokens=c(1:935),
                                                                          template = HTML("<p class='repo-language'>{{info}}</p> <p class='repo-name'>{{name}}</p>"
                                                                                          )
                                                      ),
                                                      br(),br(),
                                                      
                                                #Create a place to enter the block group ID - 
                                                  #IMPORTANT THE BGID IS WHAT CHANGES THE POPULATION DENSITY FOR ALL THE CALCULATIONS. 
                                                  
                                                  #all the below code allows me to use HTML to add a tooltip and icon to hover over to see it. 
                                                  p(HTML("<b>Block group ID</b>"),span(shiny::icon("info-circle"), id = "bgid"),
                                                    numericInput('bgid',label = NULL, val = 110010062021, min = 1, max = 600000000000),
                                                    tippy::tippy_this(elementId = "bgid",
                                                                      tooltip = "<span style = 'font-size:20px;'>Retrieve from EPA's interactive map viewer<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                      allowHTML = T,
                                                                      placement = "right")
                                                  ),
                                                    
                                                  
                                                      
                                                      #Create a place to enter the population, shinywidgets and autonumericinput allows the thousands separator to show up
                                                      shinyWidgets::autonumericInput(inputId = "pop", 
                                                                   label = "Project population or employment increase",
                                                                   value = 200, min = 1, max = 600000, 
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."
                                                      ),
                                                      
                                                      #Input for cost per acre/mile of road
                                                      shinyWidgets::autonumericInput(inputId = "roadcost", 
                                                                   label = "Cost per acre of road ($)", 
                                                                   value = 1089000, min = 1, max = 10000000, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."),
                                                      
                                                      
                                                      #input for cost per mile of water pipe
                                                      shinyWidgets::autonumericInput(inputId = "watercost", 
                                                                   label = "Cost per linear foot of water pipe ($)", 
                                                                   value = 27, min = 1, max = 100, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."),
                                                
                                                shinyWidgets::autonumericInput(inputId = "sewercost", 
                                                                               label = "Cost per linear foot of sewer main ($)", 
                                                                               value = 20, min = 1, max = 100, 
                                                                               currencySymbolPlacement = "p",
                                                                               decimalPlaces = 0,
                                                                               digitGroupSeparator = ",",
                                                                               decimalCharacter = ".") 
                                  ),#close box 
        # 2.1.1 Location 1 plot box ----------------------------------------------------      
                                  shinydashboard::box(title = "Location 1 Regression Model", width = 4, solidHeader = T, status = "primary", 
                                                      plotOutput("lm"),
                                                      ),#close plot box
                                  
                                  #Location 1 results box
                                  shinydashboard::box(title = "Efficiency estimates", 
                                                      width = 4, solidHeader = T, status = "primary", 
                                                      footer = "All results are estimates (rounded to the nearest $1,000) of capital and O&M costs for providing new infrastructure to service new population",
                                                     # verbatimTextOutput("infeff"),
                                                      textOutput("stats"),
                                                      textOutput("infeff"),
                                                      textOutput("density"),
                                                      textOutput("RdA"), 
                                                      textOutput("rdcost"),
                                                      textOutput("swcost"), 
                                                      textOutput("h2ocost"),
                                                      textOutput("sewcost")
                                                      )
      
      
     # downloadButton("downloadData", "Download")
                  ) #close column
                  ), #Close fluidrow
                  
      # 2.2 Location 2 input box ----------------------------------------------------
                  fluidRow(column(12,            
                                  shinydashboard::box(title = "Location 2 Inputs", width = 4, solidHeader = T, status = "success", 
                                                      
                                                      #   #create the drop down to choose a metro or micropolitan area.     
                                                      p(HTML("<b>Metro or micro area where project is located</b>")),
                                                      #   span(shiny::icon("info-circle"), id = "Location2"),
                                                      #   selectInput('Location2',label = NULL, choices = CBSA_choice, selected = NULL, multiple = F, selectize = T, width = NULL, size = NULL),
                                                      #   tippy::tippy_this(elementId = "Location2",
                                                      #                     tooltip = "<span style = 'font-size:20px;'>Change this if you are comparing locations in two DIFFERENT metro or micro areas<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                      #                     allowHTML = T,
                                                      #                     placement = "right")
                                                      # ),
                                                      # 
                                                      textInput.typeahead(id="Location2",
                                                                          placeholder="Type Metro or Micro area",
                                                                          local=data.frame(name=c(CBSA_choice)),
                                                                          valueKey = "name",
                                                                          tokens=c(1:935),
                                                                          template = HTML("<p class='repo-language'>{{info}}</p> <p class='repo-name'>{{name}}</p>"
                                                                          )
                                                      ),
                                                      br(),br(),
                                              
                                              #Enter bgid
                                            
                                                    p(HTML("<b>Block group ID</b>"), 
                                                          span(shiny::icon("info-circle"), id = "bgid2"),
                                                           numericInput('bgid2', label = NULL, 
                                                           val = 110010062021, min = 0, max = 600000000000),
                                                           tippy::tippy_this(elementId = "bgid2", tooltip = "<span style = 'font-size:20px;'>Change to compare two locations in the same metro or micro area<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                allowHTML = T,
                                                                placement = "right"))
                                                                          ,       
                                              
                                                      
                                                      #Create a place to enter the population
                                                     shinyWidgets::autonumericInput(inputId = "pop2", 
                                                                   label = "Project population or employment increase",
                                                                   value = 200, min = 1, max = 600000, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."),
                                                      
                                                      #Input for cost per acre/mile of road
                                                      shinyWidgets::autonumericInput(inputId = "roadcost2", 
                                                                   label = "Cost per acre of road ($)", 
                                                                   value = 1089000, min = 1, max = 10000000, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."),
                                                      
                                                      
                                                      #input for cost per mile of water pipe
                                                      shinyWidgets::autonumericInput(inputId = "watercost2", 
                                                                   label = "Cost per linear foot of water pipe ($)", 
                                                                   value = 27, min = 1, max = 100, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = ".") , 
                                              
                                              
                                              #sewer input
                                              shinyWidgets::autonumericInput(inputId = "sewercost2", 
                                                                             label = "Cost per linear foot of sewer main ($)", 
                                                                             value = 20, min = 1, max = 100, 
                                                                             currencySymbolPlacement = "p",
                                                                             decimalPlaces = 0,
                                                                             digitGroupSeparator = ",",
                                                                             decimalCharacter = ".") 
                                  ),  #close box
                                  
                                  
        # 2.2.2 Location 2 plot box -----------------------------------------------------
                                  
                                  
                                  shinydashboard:: box(title = "Location 2 Regression Model", width = 4, solidHeader = T, status = "success", 
                                                       plotOutput("lm2"),
                                                       
                                                       ), #close box
                                  
                                 shinydashboard::box(title = "Efficiency estimates", width = 4, solidHeader = T, status = "success", 
                                                     textOutput("stats2"),
                                                     textOutput("infeff2"),
                                                     textOutput("density2"),
                                                     textOutput("RdA2"), 
                                                     textOutput("rdcost2"), 
                                                     textOutput("swcost2"),
                                                     textOutput("h2ocost2"),
                                                     textOutput("sewcost2") , 
                                                     footer = "All results are estimates (rounded to the nearest $1,000) of capital and O&M costs for providing new infrastructure to service new population"
                                                     
                                                     
                                                     )
                                  
                  ) #close fluidrow
                  ) #close second box       
                  
                  
                ), #close fluidrow
              ), 
     
     tabItem(tabName = "menu_about",
             includeMarkdown("EPA_about.rmd"))
                
                
                
  )
  )
) #Close dashboard page



# Server function ---------------------------------------------------------




server<-function(input, output, session) {
  #the input values change whenever a user changes the input
  
  navy <- c("#00245D")
  blue <- c("#0082C8")
  green <- c("#C3CF21")
  plotTheme <- function(base_size = 14) {
    theme(
      plot.title = element_text(size = 16, color = "grey23"),
      plot.subtitle = element_text(size =  12, color = "grey23"),
      text = element_text(size = 14, color = "grey23"),
      axis.text = element_text(size = 12, color = "grey23"),
      axis.title = element_text(size = 12, color = "grey23"),
      axis.ticks.y = element_line(color = "grey45"),
      axis.ticks.x = element_line(color = "grey45"),
      panel.background = element_rect(fill = "white"),
      # axis.line.x = element_line(color = "grey45", size = 0.75),
      # axis.line.y = element_line(color = "grey45", size = 0.75),
      panel.grid.major.y = element_line(color = "grey90"),
      panel.grid.minor.y = element_line(color = "grey90"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank()
    )
  }
  
 

# Reactive variables ------------------------------------------------------

  

# Location 1 outputs ------------------------------------------------------

  
  output$lm <-renderPlot({
    title<-"Linear regression"
    
    #The below is to filter by bgid instead of cbsa if the issues of reactivity/new lines in outputs can be worked out. 
    # locationall<-cbsatobg%>%
    #   filter(GEOID10 == input$bgid)%>%
    #   unique(.)
    # 
    # location<-locationall$CBSA_Name
    # 
    # cbsadata<-lm_cbsa%>%
    #   filter(CBSA_name == location)
    
    cbsadata<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    
    ggplot(data = cbsadata, aes(x = slope, y = intercept) )+
      geom_abline(slope = cbsadata$slope, intercept = cbsadata$intercept, size = 1.2, color = blue) + 
      
      scale_x_continuous(limits = c(-1, 2.5))+
      scale_y_continuous(limits = c(-2.5,3))+
      plotTheme() +
      labs(x = "Log of 24 hour population density",
           y = "Log of road area per population (Infrastructure efficiency)", 
           title = input$Location #caption = paste("R^2= ", round(cbsadata$rsquared,2)) 
           ) +  #
      geom_text(vjust = 0, hjust = 0, label = (paste("y = ", round(cbsadata$slope,3), "x +" , round(cbsadata$intercept,2), "     Standard Error = ",
                                                        round(cbsadata$stderr, 2))))
  })
 

  output$infeff <- renderText ({
    
    shiny::req(input$Location)
    
    #Filter the data to the location of BG that the user selects
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    #Filter data to the bg
    dataneeds<-cbsatobg %>%
      filter(GEOID10 == input$bgid)
    
    popden<-dataneeds$popden
    
    
    # locationall<-cbsatobg%>%   #Selects the CBSA associated with the BGID
    #   filter(GEOID10 == input$bgid)%>%
    #   unique(.)
    # 
    # location<-locationall$CBSA_Name   #creates just the name of the cbsa for reference
    # 
    # statsums<-lm_cbsa%>%  #gets the stats for the CBSA regression. 
    #   filter(CBSA_name == location)
    # 
    # 
    # #Filter to get the popden of the block group
    # dataneeds<-cbsatobg %>%
    #   filter(GEOID10 == input$bgid)
    # popden<-dataneeds$popden
    # 
    
    #set the values for the lm eq
    mx<-(statsums$slope*log(popden))
    b<-statsums$intercept
    
    #Calculate the infrastructure efficiency and print 
    
    paste0( "Infrastructure Efficiency (Road area/24 hour population) = ", tags$br(),
            round(exp(mx+b),3)
    )
    
  })
 
  output$density <- renderText ({
    
    #Filter the data to the location that the user selects  
    shiny::req(input$bgid)
    
    #Filter the data to the location of BG that the user selects
    
    #Filter the data to the location that the user selects
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    #filter to get the bg popden
    dataneeds<-cbsatobg %>%
      filter(GEOID10 == input$bgid)
    popden<-dataneeds$popden
    
    #Calculate the infrastructure efficiency and print 
    paste0( "24 hour population density = ", round(popden, 0), " people/acre")
    
    
  })
  
  output$RdA <- renderText ({
    
    #Filter the data to the location that the user selects
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    #filter to get the bg popden
    dataneeds<-cbsatobg %>%
      filter(GEOID10 == input$bgid)
    popden<-dataneeds$popden
    
    #set the values for the linear model equation
    mx<-(statsums$slope*log(popden))
    b<-statsums$intercept
    
    #calculate based on the population entered
    roadarea<-round((exp(mx+b))*input$pop, 2)
    
    #Calculate based on the roadcost entered
    costbyarea<-prettyNum((roadarea*(input$roadcost)), big.mark = ",", scientific = F)

    
    #Calculate how much road area is needed in the location
    paste0( "Road Area = ", round_any(roadarea, 1), " acres")
    # 
  })
  
  output$rdcost <- renderText({
    
    #Filter the data to the location that the user selects
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    #filter to get the bg popden
    dataneeds<-cbsatobg %>%
      filter(GEOID10 == input$bgid)
    popden<-dataneeds$popden
    
    #set the values for the linear model equation
    mx<-(statsums$slope*log(popden))
    b<-statsums$intercept
    
    #calculate based on the population entered
    roadarea<-round((exp(mx+b))*input$pop, 2)
    
    #Calculate based on the roadcost entered
    costbyarea<-prettyNum(round_any((roadarea*(input$roadcost)), 1000), big.mark = ",", scientific = F)
    

# Operating cost ----------------------------------------------------------

    shiny::req(input$Location)# This requires that the input be put in before it will calculate the below.

            capex <- (roadarea*(input$roadcost))  #example $100million cost.  This should be variable and calculated
            discrate <- 0.03 #the discount rate. We're using 3% here
            opexrate <- 0.10 #O&M costs are 10% of capex

            #makes a dataframe and calculates discounted opex

           opex <- oandm %>%
              mutate(discfactor = (1+discrate)^ticker,
                     annualcost = capex*opexrate,
                     opexdisc = annualcost/discfactor)



            #the final answer. This is your discounted value of O&M costs for 20 years
            totalopex <- prettyNum((round_any((sum(opex$opexdisc)), 1000)), big.mark = ",", scientific = F)

    paste0( "Road cost = ", "$", costbyarea, " // ",
            "Twenty year O&M cost = ", "$", totalopex)
    
  })
  
  output$swcost<- renderText({
    
          statsums<-lm_cbsa%>%
            filter(CBSA_name == input$Location)
          
          #Filter data to the bg
          dataneeds<-cbsatobg %>%
            filter(GEOID10 == input$bgid)
          
          popden<-dataneeds$popden
          
          
          #set the values for the linear model equation
          mx<-(statsums$slope*log(popden))
          b<-statsums$intercept
          
          #calculate based on the population entered
          roadarea<-round((exp(mx+b))*input$pop, 2)
          #Calculate based on the roadcost entered
          costbyarea<-(roadarea*(input$roadcost))
          
          
          #MULTIPLE FOR WATER LINE TO ROAD AREA, could make this an input
          sw<-0.01774 #water to road ratio
          swcost1<-19*43560 #Convert $19/sq ft to acres
          swcost<-prettyNum(round_any((roadarea*sw*swcost1),1000 ), big.mark = ",", scientific = F)
          

# O&M cost ----------------------------------------------------------------

          
          shiny::req(input$Location)# This requires that the input be put in before it will calculate the below. 
          
          capex <- (roadarea*sw*swcost1)  #example $100million cost.  This should be variable and calculated
          discrate <- 0.03 #the discount rate. We're using 3% here
          opexrate <- 0.10 #O&M costs are 10% of capex
          
          #makes a dataframe and calculates discounted opex
          
          opex <- oandm %>%
            mutate(discfactor = (1+discrate)^ticker,
                   annualcost = capex*opexrate,
                   opexdisc = annualcost/discfactor)
          
          
          
          #the final answer. This is your discounted value of O&M costs for 20 years
          totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
          
          #Calculate how much road area is needed in the location
          paste0("Sidewalk cost = $", swcost, " // Twenty year O&M cost = $", totalopex)
          
    
  })
  
  output$sewcost<- renderText({
    
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    #Filter data to the bg
    dataneeds<-cbsatobg %>%
      filter(GEOID10 == input$bgid)
    
    popden<-dataneeds$popden
    
    
    #set the values for the linear model equation
    mx<-(statsums$slope*log(popden))
    b<-statsums$intercept
    
    #calculate based on the population entered
    roadarea<-round((exp(mx+b))*input$pop, 2)
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost))
    
    
    #MULTIPLE FOR sewer line TO ROAD AREA, could make this an input
    sew<-(70*1.8550*3.28084) #70 linear meters of road /acre of road area and 1.8550 ft sewer to meter center line
    sewcost1<-prettyNum(round_any((roadarea*sew*input$sewercost), 1000), big.mark = ",", scientific = F)
    
    
    
    # O&M cost ----------------------------------------------------------------
    
    
    shiny::req(input$Location)# This requires that the input be put in before it will calculate the below. 
    
    capex <- (roadarea*sew*input$sewercost)  #example $100million cost.  This should be variable and calculated
    discrate <- 0.03 #the discount rate. We're using 3% here
    opexrate <- 0.10 #O&M costs are 10% of capex
    
    #makes a dataframe and calculates discounted opex
    
    opex <- oandm %>%
      mutate(discfactor = (1+discrate)^ticker,
             annualcost = capex*opexrate,
             opexdisc = annualcost/discfactor)
    
    
    
    #the final answer. This is your discounted value of O&M costs for 20 years
    totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
    
    #Calculate how much road area is needed in the location
    paste0("Sewer cost = $", sewcost1, " // Twenty year O&M cost = $", totalopex)
    
    
    
  })
  
  output$h2ocost<- renderText({
    
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location)
    
    #Filter data to the bg
    dataneeds<-cbsatobg %>%
      filter(GEOID10 == input$bgid)
    
    popden<-dataneeds$popden
    
    
    #set the values for the linear model equation
    mx<-(statsums$slope*log(popden))
    b<-statsums$intercept
    
    #calculate based on the population entered
    roadarea<-round((exp(mx+b))*input$pop, 2)
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost))
    
    
    #MULTIPLE FOR sewer line TO ROAD AREA, could make this an input
    h2o<-(70*1.8550*3.28084) #70 linear meters of road /acre of road area and 1.8550 ft sewer to meter center line 3.28084 to convert cost from ft to meter
    h2ocost<-prettyNum(round_any((roadarea*h2o*(input$watercost)), 1000), big.mark = ",", scientific = F)
    
    
    # O&M cost ----------------------------------------------------------------
    
    
    shiny::req(input$Location)# This requires that the input be put in before it will calculate the below. 
    
    capex <- (roadarea*h2o*input$watercost)  #example $100million cost.  This should be variable and calculated
    discrate <- 0.03 #the discount rate. We're using 3% here
    opexrate <- 0.10 #O&M costs are 10% of capex
    
    #makes a dataframe and calculates discounted opex
    
    opex <- oandm %>%
      mutate(discfactor = (1+discrate)^ticker,
             annualcost = capex*opexrate,
             opexdisc = annualcost/discfactor)
    
    
    
    #the final answer. This is your discounted value of O&M costs for 20 years
    totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
    
    #Calculate how much road area is needed in the location
    paste0("Water pipe cost = $", h2ocost, " // Twenty year O&M cost = $", totalopex)
  
    
    
  })
  
  
  
  
  
  
  
  
  # COMPARISON OUTPUTS ------------------------------------------------------
  
  
  
  output$lm2 <-renderPlot({
    title<-"Linear regression"
    
    cbsadata<-lm_cbsa%>%
      filter(CBSA_name == input$Location2)
    
    
    ggplot(data = cbsadata, aes(x = slope, y = intercept) )+
      geom_abline(slope = cbsadata$slope, intercept = cbsadata$intercept, size = 1.2, color = green) + 
      scale_x_continuous(limits = c(-1, 2.5))+
      scale_y_continuous(limits = c(-2.5,3))+
      plotTheme() +
      labs(x = "Log of 24 hour population density",
           y = "Log of road area per population (Infrastructure efficiency)", 
           title = input$Location2 #caption = paste("R^2= ", round(cbsadata$rsquared,2))
             ) +  #
      geom_text(vjust = 0, hjust = 0, label = (paste("y = ", round(cbsadata$slope,3), "x +" , round(cbsadata$intercept,2), "     Standard Error = ",
                                                        round(cbsadata$stderr, 2)))
      )
  }) 
  
  output$infeff2 <- renderText ({
    
    #Filter the data to the location that the user selects  
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location2)
    
    dataneeds2<-cbsatobg %>%
      filter(GEOID10 == input$bgid2)
    
    popden2<-dataneeds2$popden
    
    
    #set the values for the lm eq
    mx<-(statsums$slope*log(popden2))
    b<-statsums$intercept
    
    #Calculate the infrastructure efficiency and print 
    paste0( "Infrastructure Efficiency (Road area/24 hour population) = ", round(exp(mx+b),3)
    )
    
  })
  
  output$density2 <- renderText ({
    
    #Filter the data to the location that the user selects  
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location2)
    
    #Filter to get the popden of the block group
    dataneeds2<-cbsatobg %>%
      filter(GEOID10 == input$bgid2)
    popden2<-dataneeds2$popden
    
    #Calculate the infrastructure efficiency and print 
    paste0( "24 hour population density = ", round_any(popden2, 1), " people/acre")
    
    
  })
  
  output$RdA2 <- renderText ({
    
                #Filter the data to the location that the user selects
                statsums<-lm_cbsa%>%
                  filter(CBSA_name == input$Location2)
                
                #Filter data to the bg
                dataneeds2<-cbsatobg %>%
                  filter(GEOID10 == input$bgid2)
                
                popden2<-dataneeds2$popden
                
                
                #set the values for the linear model equation
                mx<-(statsums$slope*log(popden2))
                b<-statsums$intercept
                
                #calculate based on the population entered
                roadarea<-round_any(((exp(mx+b))*input$pop2), 1)
                #Calculate based on the roadcost entered
                costbyarea<-(roadarea*(input$roadcost2))
               
                
            
                #Calculate how much road area is needed in the location
                paste0( "Road Area = ", roadarea, " acres")
                
  })
   
  
  output$rdcost2 <- renderText({
    
                            #Filter the data to the location that the user selects
                statsums<-lm_cbsa%>%
                  filter(CBSA_name == input$Location2)
                
                #Filter data to the bg
                dataneeds2<-cbsatobg %>%
                  filter(GEOID10 == input$bgid2)
                
                popden2<-dataneeds2$popden
                
                
                #set the values for the linear model equation
                mx<-(statsums$slope*log(popden2))
                b<-statsums$intercept
                
                #calculate based on the population entered
                roadarea<-round((exp(mx+b))*input$pop2, 2)
                #Calculate based on the roadcost entered\
                costbyarea<-prettyNum(round_any((roadarea*(input$roadcost2)),1000 ), big.mark = ",", scientific = F)
                
                
                # O&M cost ----------------------------------------------------------------
                
                
                shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
                
                capex <- (roadarea*input$roadcost2)  #example $100million cost.  This should be variable and calculated
                discrate <- 0.03 #the discount rate. We're using 3% here
                opexrate <- 0.10 #O&M costs are 10% of capex
                
                #makes a dataframe and calculates discounted opex
                
                opex <- oandm %>%
                  mutate(discfactor = (1+discrate)^ticker,
                         annualcost = capex*opexrate,
                         opexdisc = annualcost/discfactor)
                
                
                
                #the final answer. This is your discounted value of O&M costs for 20 years
                totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
                
                #Calculate how much road area is needed in the location
                paste0( "Road cost = ", "$", costbyarea, " // Twenty year O&M cost = $", totalopex)
    
  })
  
  output$swcost2<- renderText({
    
            statsums<-lm_cbsa%>%
              filter(CBSA_name == input$Location2)
            
            #Filter data to the bg
            dataneeds2<-cbsatobg %>%
              filter(GEOID10 == input$bgid2)
            
            popden2<-dataneeds2$popden
            
            
            #set the values for the linear model equation
            mx<-(statsums$slope*log(popden2))
            b<-statsums$intercept
            
            #calculate based on the population entered
            roadarea<-round((exp(mx+b))*input$pop2, 2)
            #Calculate based on the roadcost entered
            costbyarea<-(roadarea*(input$roadcost2))
            
            
            #MULTIPLE FOR SIDEWALK TO ROAD AREA, could make this an input
            sw<-0.01774 #water to road ratio
            swcost<-19*43560 #Convert $19/sq ft to acres
            swcost1<-prettyNum(round_any(roadarea*sw*swcost, 1000), big.mark = ",", scientific = F)
            
            
            # O&M cost ----------------------------------------------------------------
            
            
            shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
            
            capex <- (roadarea*sw*swcost)  #example $100million cost.  This should be variable and calculated
            discrate <- 0.03 #the discount rate. We're using 3% here
            opexrate <- 0.10 #O&M costs are 10% of capex
            
            #makes a dataframe and calculates discounted opex
            
            opex <- oandm %>%
              mutate(discfactor = (1+discrate)^ticker,
                     annualcost = capex*opexrate,
                     opexdisc = annualcost/discfactor)
            
            
            
            #the final answer. This is your discounted value of O&M costs for 20 years
            totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
            
            #Calculate how much road area is needed in the location
            paste0( "Sidewalk cost = $", swcost1, " // Twenty year O&M cost = $", totalopex)
            
            
            
  })
  
  
  
  output$sewcost2<- renderText({
    
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location2)
    
    #Filter data to the bg
    dataneeds2<-cbsatobg %>%
      filter(GEOID10 == input$bgid2)
    
    popden2<-dataneeds2$popden
    
    
    #set the values for the linear model equation
    mx<-(statsums$slope*log(popden2))
    b<-statsums$intercept
    
    #calculate based on the population entered
    roadarea<-round((exp(mx+b))*input$pop2, 2)
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost2))
    
    #MULTIPLE FOR sewer line TO ROAD AREA, could make this an input
    sew<-(70*1.8550*3.28084) #70 linear meter of road /acre of road area and 1.8550 ft sewer to meter center line, multiply by 3.28084 because cost is in per foot
    sewcost1<-prettyNum(round_any(roadarea*sew*input$sewercost2, 1000), big.mark = ",", scientific = F)
    
    
    # O&M cost ----------------------------------------------------------------
    
    
    shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
    
    capex <- (roadarea*sew*input$sewercost2)  #example $100million cost.  This should be variable and calculated
    discrate <- 0.03 #the discount rate. We're using 3% here
    opexrate <- 0.10 #O&M costs are 10% of capex
    
    #makes a dataframe and calculates discounted opex
    
    opex <- oandm %>%
      mutate(discfactor = (1+discrate)^ticker,
             annualcost = capex*opexrate,
             opexdisc = annualcost/discfactor)
    
    
    
    #the final answer. This is your discounted value of O&M costs for 20 years
    totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
    
    #Calculate how much road area is needed in the location
    paste0( "Sewer cost = $", sewcost1, " // Twenty year O&M cost = $", totalopex)
    
    
    
  })
  
  
  output$h2ocost2<- renderText({
    
    statsums<-lm_cbsa%>%
      filter(CBSA_name == input$Location2)
    
    #Filter data to the bg
    dataneeds2<-cbsatobg %>%
      filter(GEOID10 == input$bgid2)
    
    popden2<-dataneeds2$popden
    
    
    #set the values for the linear model equation
    mx<-(statsums$slope*log(popden2))
    b<-statsums$intercept
    
    #calculate based on the population entered
    roadarea<-round((exp(mx+b))*input$pop2, 2)
    
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost2))
    h2o<-(70*1.8550*3.28084) #70 linear meter of road /acre of road area and 1.8550 ft sewer to meter center line 3.28084 is meter to feet conversion
    #Convert $20/sq ft
    #SIGNIF lets you know how many significant figures to include. 
    h2ocost2<-prettyNum(round_any(roadarea*h2o*(input$watercost2), 1000), big.mark = ",", scientific = F)
    
    
    # O&M cost ----------------------------------------------------------------
    
    
    shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
    
    capex <- (roadarea*h2o*input$watercost2)  #example $100million cost.  This should be variable and calculated
    discrate <- 0.03 #the discount rate. We're using 3% here
    opexrate <- 0.10 #O&M costs are 10% of capex
    
    #makes a dataframe and calculates discounted opex
    
    opex <- oandm %>%
      mutate(discfactor = (1+discrate)^ticker,
             annualcost = capex*opexrate,
             opexdisc = annualcost/discfactor)
    
    
    
    #the final answer. This is your discounted value of O&M costs for 20 years
    totalopex <- prettyNum(round_any((sum(opex$opexdisc)), 1000), big.mark = ",", scientific = F)
    
    #Calculate how much road area is needed in the location
    paste0( "Water pipe cost = $", h2ocost2, " // Twenty year O&M cost = $", totalopex)
    
  
  })
  
}



shinyApp(ui = ui, server = server)