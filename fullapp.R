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

cbsabg<-read.csv("cbsabg_analysis.csv")

cbsatobg<-read.csv("cbsa_to_bg.csv")

oandm<-read.csv("oandm.csv")

# 2.0 Set UI for shiny --------------------------------------------------------



ui<- dashboardPage(
  dashboardHeader(title = "Infrastructure Efficiency Modeling"), 
  
  dashboardSidebar(  
                     menuItem("Infrastructure Efficiency Modeling Tool", icon = icon("th"), tabName = "Menu"),
                     menuItem("EPA interactive map", icon = icon("external-link-square"), href = "https://epa.maps.arcgis.com/home/webmap/viewer.html?webmap=f16f5e2f84884b93b380cfd4be9f0bba"),
                     menuItem("Github", icon = icon("github"), href = "https://github.com/bbuthe-sga/EPA_fiscal.git"),
                     menuItem("About", icon = icon("question-circle-o"), href = "https://github.com/bbuthe-sga/EPA_fiscal/tree/main")
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
                                                      # p(HTML("<b>Metro or micro area where project is located</b>")),
                                                      # textInput.typeahead(id="Location",
                                                      #                     placeholder="Type Metro or Micro area",
                                                      #                     local=data.frame(name=c(CBSA_choice)),
                                                      #                     valueKey = "name",
                                                      #                     tokens=c(1:935),
                                                      #                     template = HTML("<p class='repo-language'>{{info}}</p> <p class='repo-name'>{{name}}</p>"
                                                      #                                     )
                                                      # ),
                                                      # br(),br(),
                                                      
                                                #Create a place to enter the block group ID - 
                                                  #IMPORTANT THE BGID IS WHAT CHANGES THE POPULATION DENSITY FOR ALL THE CALCULATIONS. 
                                                  
                                                  #all the below code allows me to use HTML to add a tooltip and icon to hover over to see it. 
                                                  p(HTML("<b>Block group ID (example: 110010062021)</b>"),span(shiny::icon("info-circle"), id = "bgid"),
                                                    numericInput('bgid',label = NULL, val = NULL, min = 1, max = 600000000000),
                                                    tippy::tippy_this(elementId = "bgid",
                                                                      tooltip = "<span style = 'font-size:20px;'>Retrieve 12 digit FIPS (example: 110010062021) for your location
                                                                      from EPA's interactive map viewer. This input is important as it provides information to change the
                                                                      population density.<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                      allowHTML = T,
                                                                      placement = "right")
                                                  ),
                                                    
                                                  
                                                      
                                                      #Create a place to enter the population, shinywidgets and autonumericinput allows the thousands separator to show up
                                                  p(HTML("<b>Project population and employment increase</b>"),span(shiny::icon("info-circle"), id = "pop"),
                                                  
                                                  shinyWidgets::autonumericInput(inputId = "pop", 
                                                                   label = NULL,
                                                                   value = NULL, min = 1, max = 600000, 
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."
                                                      ),
                                                  tippy::tippy_this(elementId = "pop",
                                                                    tooltip = "<span style = 'font-size:20px;'>Enter the estimated population and employment
                                                                    increase as a result of your project. <span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                    allowHTML = T,
                                                                    placement = "right")
                                                  ),
                                                      
                                                      #Input for cost per acre/mile of road
                                                p(HTML("<b>Cost per acre of road ($)</b>"),span(shiny::icon("info-circle"), id = "roadcost"),
                                                  
                                                      shinyWidgets::autonumericInput(inputId = "roadcost", 
                                                                   label = NULL, 
                                                                   value = 1089000, min = 1, max = 10000000, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."),
                                                  tippy::tippy_this(elementId = "roadcost",
                                                                    tooltip = "<span style = 'font-size:20px;'>The road cost is defaulted to $1,089,000/acre, but can be altered
                                                                    here if a localized estimate is available. A detailed explanation for the cost can be found in the About tab.<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                    allowHTML = T,
                                                                    placement = "right")
                                                ),
                                                      
                                                      
                                                      #input for cost per mile of water pipe
                                                p(HTML("<b>Cost per linear foot of water main ($)</b>"),span(shiny::icon("info-circle"), id = "watercost"),
                                                      shinyWidgets::autonumericInput(inputId = "watercost", 
                                                                   label = NULL, 
                                                                   value = 27, min = 1, max = 100, 
                                                                   currencySymbolPlacement = "p",
                                                                   decimalPlaces = 0,
                                                                   digitGroupSeparator = ",",
                                                                   decimalCharacter = "."),
                                                  tippy::tippy_this(elementId = "watercost",
                                                                    tooltip = "<span style = 'font-size:20px;'>The water main cost is defaulted to $27/linear foot, but can be altered
                                                                    here if a localized estimate is available. A detailed explanation for the cost can be found in the About tab.<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                    allowHTML = T,
                                                                    placement = "right")
                                                ), 
                                                
                                                #Sewer cost
                                                p(HTML("<b>Cost per linear foot of sewer main ($)</b>"),span(shiny::icon("info-circle"), id = "sewercost"),
                                                  
                                                shinyWidgets::autonumericInput(inputId = "sewercost", 
                                                                               label = NULL, 
                                                                               value = 20, min = 1, max = 100, 
                                                                               currencySymbolPlacement = "p",
                                                                               decimalPlaces = 0,
                                                                               digitGroupSeparator = ",",
                                                                               decimalCharacter = "."),
                                                tippy::tippy_this(elementId = "sewercost",
                                                                  tooltip = "<span style = 'font-size:20px;'>The sewer main cost is defaulted to $20/linear foot, but can be altered
                                                                    here if a localized estimate is available. A detailed explanation for the cost can be found in the About tab.<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                  allowHTML = T,
                                                                  placement = "right")
                                                )
                                                
                                  ),#close box 
        # 2.1.1 Location 1 plot box ----------------------------------------------------      
                                  
                                  #Location 1 results box
                                  shinydashboard::box(title = "Location 1 Cost Estimates", 
                                                      width = 4, solidHeader = T, status = "primary", 
                                                      footer = "All results are estimates (rounded to the nearest $1,000) of capital and O&M costs for providing new infrastructure to service new population. 
                                                      The underlying data is a snapshot in time and it assumes the relationship between infrastructure and population density holds.",
                                                      
                                                      
                                                      strong(textOutput("locationcall")),
                                                      textOutput("bgidout"),
                                                      br(),
                                                      textOutput("rdcost"),
                                                      textOutput("swcost"), 
                                                      textOutput("h2ocost"),
                                                      textOutput("sewcost")
                                                      ), 
      
                                    shinydashboard::box(title = "Location 1 Efficiency Estimates", width = 4, solidHeader = T, status = "primary", 
                                                        strong(textOutput("locationcall1")),
                                                        br(),
                                                        
                                                        textOutput("stats"),
                                                        textOutput("infeff"),
                                                        textOutput("density"),
                                                        textOutput("RdA"),
                                                        footer = "Smaller infrastructure efficiency values signify less infrastructure required per person. "
                                    ),#close plot box
      
      
      
     # downloadButton("downloadData", "Download")
                  ) #close column
                  ), #Close fluidrow
                  
      # 2.2 Location 2 input box ----------------------------------------------------
                  fluidRow(column(12,            
                                  shinydashboard::box(title = "Location 2 Inputs", width = 4, solidHeader = T, status = "success", 
                                                      
                                                      #   #create the drop down to choose a metro or micropolitan area.     
                                                      # p(HTML("<b>Metro or micro area where project is located</b>")),
                                                      # 
                                                      # textInput.typeahead(id="Location2",
                                                      #                     placeholder="Type Metro or Micro area",
                                                      #                     local=data.frame(name=c(CBSA_choice)),
                                                      #                     valueKey = "name",
                                                      #                     tokens=c(1:935),
                                                      #                     template = HTML("<p class='repo-language'>{{info}}</p> <p class='repo-name'>{{name}}</p>"
                                                      #                     )
                                                      # ),
                                                      # br(),br(),
                                              
                                              #Enter bgid
                                              
                                                    p(HTML("<b>Block group ID</b>"), 
                                                          span(shiny::icon("info-circle"), id = "bgid2"),
                                                           numericInput('bgid2', label = NULL, 
                                                           val = NULL, min = 0, max = 600000000000),
                                                           tippy::tippy_this(elementId = "bgid2", tooltip = "<span style = 'font-size:20px;'>Change to compare two locations in the same metro or micro area<span>",  #the span style changes the font size of the tool tip so it's not tiny
                                                                allowHTML = T,
                                                                placement = "right"))
                                                                          ,       
                                              
                                                      
                                                      #Create a place to enter the population
                                                     shinyWidgets::autonumericInput(inputId = "pop2", 
                                                                   label = "Project population or employment increase",
                                                                   value = NULL, min = 1, max = 600000, 
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
                                  
                                 shinydashboard::box(title = "Location 2 Cost Estimates", width = 4, solidHeader = T, status = "success", 
                                                     
                                                    strong( textOutput("locationcall2")),
                                                    textOutput("bgidout2"),
                                                    br(),
                                                     textOutput("rdcost2"), 
                                                     textOutput("swcost2"),
                                                     textOutput("h2ocost2"),
                                                     textOutput("sewcost2") , 
                                                     footer = "All results are estimates (rounded to the nearest $1,000) of capital and O&M costs for providing new infrastructure to service new population. 
                                                     The underlying data is a snapshot in time and it assumes the relationship between infrastructure and population density holds."
                                                     
                                                     
                                                     ), 
        
                                                
                                                shinydashboard:: box(title = "Location 2 Efficiency Estimates", width = 4, solidHeader = T, status = "success", 
                                                                     strong( textOutput("locationcall2.1")),
                                                                     br(),
                                                                     
                                                                      textOutput("stats2"),
                                                                     textOutput("infeff2"),
                                                                     textOutput("density2"),
                                                                     textOutput("RdA2"), 
                                                                     footer = "Smaller infrastructure efficiency values signify less infrastructure required per person. "
                                                                     
                                                                     
        ), #close box
        
                                  
                  ) #close fluidrow
                  ) #close second box       
                  
                  
                ), #close fluidrow
              )  
                
                
  )
  )
) #Close dashboard page



# Server function ---------------------------------------------------------




server<-function(input, output, session) {
  #the input values change whenever a user changes the input
  
  # navy <- c("#00245D")
  # blue <- c("#0082C8")
  # green <- c("#C3CF21")
  # plotTheme <- function(base_size = 14) {
  #   theme(
  #     plot.title = element_text(size = 16, color = "grey23"),
  #     plot.subtitle = element_text(size =  12, color = "grey23"),
  #     text = element_text(size = 14, color = "grey23"),
  #     axis.text = element_text(size = 12, color = "grey23"),
  #     axis.title = element_text(size = 12, color = "grey23"),
  #     axis.ticks.y = element_line(color = "grey45"),
  #     axis.ticks.x = element_line(color = "grey45"),
  #     panel.background = element_rect(fill = "white"),
  #     # axis.line.x = element_line(color = "grey45", size = 0.75),
  #     # axis.line.y = element_line(color = "grey45", size = 0.75),
  #     panel.grid.major.y = element_line(color = "grey90"),
  #     panel.grid.minor.y = element_line(color = "grey90"),
  #     panel.grid.major.x = element_blank(),
  #     panel.grid.minor.x = element_blank()
  #   )
  # }
  
 

# Reactive variables ------------------------------------------------------


      # Location1 ---------------------------------------------------------------
  
                            locationall<-reactive({
                              cbsatobg%>%
                                filter(GEOID10 == input$bgid)
                            })
                          
                            location<-reactive({locationall()[, 21]})
  
                            
                            cbsadatas<-reactive({
                              lm_cbsa%>%
                                filter(CBSA_name == location())
                            })
                            
                            
                            dataneeds<- reactive({
                              cbsatobg %>%
                              filter(GEOID10 == input$bgid)
                              
                            })
                            
                            popden<-reactive({
                              dataneeds()[,16]
                            })
                            
                            slope<-reactive({
                              cbsadatas()[,3]
                              
                            })
                            
                            mx<-reactive({
                               slope()*log(popden())
                              
                            })
                            
                            b<-reactive({
                              cbsadatas()[,2]
                            })
                            
                            equation<-reactive({
                              ((cbsadatas()[,3])*log(popden()))+(cbsadatas()[,2])
                            })
                            


      # Location2 ---------------------------------------------------------------
                            locationall2<-reactive({
                              cbsatobg%>%
                                filter(GEOID10 == input$bgid2)
                            })
                            
                            location2<-reactive({locationall2()[, 21]})
                            
                            
                            cbsadatas2<-reactive({
                              lm_cbsa%>%
                                filter(CBSA_name == location2())
                            })
                            
                            dataneeds2<- reactive({
                              
                              cbsatobg %>%
                                filter(GEOID10 == input$bgid2)
                              
                            })
                            
                            popden2<-reactive({
                              
                              dataneeds2()[,16]
                            })
                            
                            slope2<-reactive({
                              
                              cbsadatas2()[,3]
                              
                            })
                            
                            mx2<-reactive({
                              slope2()*log(popden2())
                              
                            })
                            
                            b2<-reactive({
                              cbsadatas2()[,2]
                            })
                            
                            equation2<-reactive({
                              
                              ((cbsadatas2()[,3])*log(popden2()))+(cbsadatas2()[,2])
                            })
                            
                          
# Location 1 outputs ------------------------------------------------------

  
  # output$lm <-renderPlot({
  #   title<-"Linear regression"
  #   
  #   shiny::req(input$bgid)# This requires that the input be put in before it will calculate the below.
  #   
  #   cbsadata<-lm_cbsa%>%
  #     filter(CBSA_name == location())
  #   
  #   
  #   ggplot(data = NULL, aes(x = ))
  #   
  #   ggplot(data = cbsadatas(), aes(x = slope(), y = b()  ) )+
  #     geom_abline(slope = slope(), intercept = b(), size = 1.2, color = blue) + 
  #     
  #     scale_x_continuous(limits = c(-1, 2.5))+
  #     scale_y_continuous(limits = c(-2.5,3))+
  #     plotTheme() +
  #     labs(x = "Log of 24 hour population density",
  #          y = "Log of road area per population (Infrastructure efficiency)", 
  #          title = location() #caption = paste("R^2= ", round(cbsadata$rsquared,2)) 
  #          ) +  #
  #     geom_text(vjust = 0, hjust = 0, label = (paste("y = ", round(cbsadata$slope,3), "x +" , round(cbsadata$intercept,2), "     Standard Error = ",
  #                                                       round(cbsadata$stderr, 2))))
  # })
 
  output$infeff <- renderText ({
    
    shiny::req(input$bgid)# This requires that the input be put in before it will calculate the below.
    
    #Calculate the infrastructure efficiency and print 
    
    paste0( "Infrastructure Efficiency (Road area/24 hour population) = ",
            round(exp(equation()),3)
    )
    
  })
 
  output$density <- renderText ({
    shiny::req(input$bgid)# This requires that the input be put in before it will calculate the below.
    
    #Calculate the infrastructure efficiency and print 
    paste0( "24 hour population density = ", round(popden(), 0), " people/acre")
    
    
  })
  
  output$RdA <- renderText ({
    shiny::req(input$bgid)# This requires that the input be put in before it will calculate the below.
    
    #calculate based on the population entered
    roadarea<-round((exp(equation()))*input$pop, 2)
   
    #Calculate how much road area is needed in the location
    paste0( "Road Area = ", round_any(roadarea, 1), " acres")
    # 
  })
  
  output$locationcall<-renderText({
    shiny::req(input$bgid)
    paste0(location())
    
  })
  
  output$locationcall1<-renderText({
    shiny::req(input$bgid)
    paste0(location())
    
  })
  
  output$bgidout<-renderText({
    shiny::req(input$bgid)
    paste0("Block group = " ,input$bgid)
    
  })
  output$rdcost <- renderText({
    #shiny::req(input$Location)# This requires that the input be put in before it will calculate the below.
    shiny::req(input$bgid)
    req(input$pop)
    #calculate based on the population entered
    roadarea<-round((exp(equation()))*input$pop, 2)
    
    #Calculate based on the roadcost entered
    costbyarea<-prettyNum(round_any((roadarea*(input$roadcost)), 1000), big.mark = ",", scientific = F)
    

# Operating cost ----------------------------------------------------------

   
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
    
    shiny::req(input$bgid)
    req(input$pop)
         
          #calculate based on the population entered
          roadarea<-round((exp(equation()))*input$pop, 2)
          #Calculate based on the roadcost entered
          costbyarea<-(roadarea*(input$roadcost))
          
          
          #MULTIPLE FOR WATER LINE TO ROAD AREA, could make this an input
          sw<-0.01774 #water to road ratio
          swcost1<-19*43560 #Convert $19/sq ft to acres
          swcost<-prettyNum(round_any((roadarea*sw*swcost1),1000 ), big.mark = ",", scientific = F)
          

# O&M cost ----------------------------------------------------------------

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
    shiny::req(input$bgid)
    req(input$pop)
   
    #calculate based on the population entered
    roadarea<-round((exp(equation()))*input$pop, 2)
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost))
    
    
    #MULTIPLE FOR sewer line TO ROAD AREA, could make this an input
    sew<-(70*1.8550*3.28084) #70 linear meters of road /acre of road area and 1.8550 ft sewer to meter center line, x3.28084 to get to feet
    sewcost1<-prettyNum(round_any((roadarea*sew*input$sewercost), 1000), big.mark = ",", scientific = F)
    
    
    
    # O&M cost ----------------------------------------------------------------
    
    
    
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
    shiny::req(input$bgid)
    req(input$pop)
  
    #calculate based on the population entered
    roadarea<-round((exp(equation()))*input$pop, 2)
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost))
    
    
    #MULTIPLE FOR sewer line TO ROAD AREA, could make this an input
    h2o<-(70*1.8550*3.28084) #70 linear m of road /acre of road area and 1.8550 ft sewer to m center line x3.28084 to get to feet
   #Convert $20/sq ft
    h2ocost<-prettyNum(round_any((roadarea*h2o*(input$watercost)), 1000), big.mark = ",", scientific = F)
    
    
    # O&M cost ----------------------------------------------------------------
    
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
  
  # 
  # 
  # output$lm2 <-renderPlot({
  #   title<-"Linear regression"
  #   shiny::req(input$bgid2)# This requires that the input be put in before it will calculate the below. 
  #   
  #   cbsadata<-lm_cbsa%>%
  #     filter(CBSA_name == location2())
  #   
  #   ggplot(data = cbsadatas2(), aes(x = slope2(), y = b2()) )+
  #     geom_abline(slope = slope2(), intercept = b2(), size = 1.2, color = green) + 
  #     scale_x_continuous(limits = c(-1, 2.5))+
  #     scale_y_continuous(limits = c(-2.5,3))+
  #     plotTheme() +
  #     labs(x = "Log of 24 hour population density",
  #          y = "Log of road area per population (Infrastructure efficiency)", 
  #          title = location2() #caption = paste("R^2= ", round(cbsadata$rsquared,2))
  #            ) +  #
  #     geom_text(vjust = 0, hjust = 0, label = (paste("y = ", round(slope2(),3), "x +" , round(b2(),2), "     Standard Error = ",
  #                                                       round(cbsadata$stderr, 2)))
  #     )
  # }) 
  # 
  
  
  
  output$infeff2 <- renderText ({
    shiny::req(input$bgid2)# This requires that the input be put in before it will calculate the below. 
    #Filter the data to the location that the user selects  
   
    #Calculate the infrastructure efficiency and print 
    paste0( "Infrastructure Efficiency (Road area/24 hour population) = ", round(exp(equation2()),3)
    )
    
  })
  
  
  output$density2 <- renderText ({
    shiny::req(input$bgid2)# This requires that the input be put in before it will calculate the below. 
    #Filter the data to the location that the user selects  
    
    #Calculate the infrastructure efficiency and print 
    paste0( "24 hour population density = ", round_any(popden2(), 1), " people/acre")
    
    
  })
  
  output$RdA2 <- renderText ({
    shiny::req(input$bgid2)# This requires that the input be put in before it will calculate the below. 
                #Filter the data to the location that the user selects
                
                #calculate based on the population entered
                roadarea<-round_any(((exp(equation2()))*input$pop2), 1)
                #Calculate based on the roadcost entered
                costbyarea<-(roadarea*(input$roadcost2))
               
                
            
                #Calculate how much road area is needed in the location
                paste0( "Road Area = ", roadarea, " acres")
                
  })
  
  
  output$locationall2<-renderText({
    shiny::req(input$bgid)
    paste0(location2())
    
  })
  
  
  output$locationall2.1<-renderText({
    shiny::req(input$bgid)
    paste0(location2())
    
  })
  
  
  output$bgidout2<-renderText({
    shiny::req(input$bgid2)
    paste0("Block group = ", input$bgid2)
    
  })
  
  output$rdcost2 <- renderText({
    shiny::req(input$bgid2)
    req(input$pop2)
    #shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
                            #Filter the data to the location that the user selects
              
                
                #calculate based on the population entered
                roadarea<-round((exp(equation2()))*input$pop2, 2)
                #Calculate based on the roadcost entered\
                costbyarea<-prettyNum(round_any((roadarea*(input$roadcost2)),1000 ), big.mark = ",", scientific = F)
                
                
                # O&M cost ----------------------------------------------------------------
            
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
    shiny::req(input$bgid2)
    req(input$pop2)
    #shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
            
            #calculate based on the population entered
            roadarea<-round((exp(equation2()))*input$pop2, 2)
            #Calculate based on the roadcost entered
            costbyarea<-(roadarea*(input$roadcost2))
            
            
            #MULTIPLE FOR SIDEWALK TO ROAD AREA, could make this an input
            sw<-0.01774 #water to road ratio
            swcost<-19*43560 #Convert $19/sq ft to acres
            swcost1<-prettyNum(round_any(roadarea*sw*swcost, 1000), big.mark = ",", scientific = F)
            
            
            # O&M cost ----------------------------------------------------------------
            
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
    shiny::req(input$bgid2)
    req(input$pop2)
    #shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
    
    #calculate based on the population entered
    roadarea<-round((exp(equation2()))*input$pop2, 2)
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost2))
    
    #MULTIPLE FOR sewer line TO ROAD AREA, could make this an input
    sew<-(70*1.8550*3.28084) #70 linear m of road /acre of road area and 1.8550 m sewer to ft center line x3.28084m to ft
    sewcost1<-prettyNum(round_any(roadarea*sew*input$sewercost2, 1000), big.mark = ",", scientific = F)
    
    
    # O&M cost ----------------------------------------------------------------
    
    
    #shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
    
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
    shiny::req(input$bgid2)
    req(input$pop2)
    #shiny::req(input$Location2)# This requires that the input be put in before it will calculate the below. 
    
    #calculate based on the population entered
    roadarea<-round((exp(equation2()))*input$pop2, 2)
    
    #Calculate based on the roadcost entered
    costbyarea<-(roadarea*(input$roadcost2))
    h2o<-(70*1.8550*3.28084) #70 linear m of road /acre of road area and 1.8550 ft sewer to m center line x3.28084
    #Convert $20/sq ft
    #SIGNIF lets you know how many significant figures to include. 
    h2ocost2<-prettyNum(round_any(roadarea*h2o*(input$watercost2), 1000), big.mark = ",", scientific = F)
    
    
    # O&M cost ----------------------------------------------------------------
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