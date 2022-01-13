#' This script is used to calculate SWE with the nixmass R implementation.
#'
#' Before you run it you have to make sure the data has been downloaded and 
#' as .csv in the same directory with 'download_data.py'.

library(nixmass)
library(data.table)

CalculateNixmassOnCsv <- function(hs_datafile, swe_outfile){
  #' Calculate nixmass deltasnow on the data in a csv file.
  #'
  #' @param hs_datafile: string to the file hich contains hs data
  #
  df <- read.csv(hs_datafile,
                 colClasses=c("numeric", "character", "character", "numeric", "numeric"),
                 na.strings="nan")

  data_list <- list()
  for (hyear in unique(df$hjahr)) {
    hs_data_y <- df[which(df$hjahr==hyear),]
    SWEnixmass_y <- nixmass(hs_data_y, model="delta.snow",verbose=F)
    SWE_delta_snow <- SWEnixmass_y$swe$delta.snow
    date <- SWEnixmass_y$date
    
    out_df <- data.frame(date, SWE_delta_snow)
    data_list <- append(data_list, list(out_df))
  }

  dfswe <- rbindlist(data_list)
  write.csv(dfswe, file=swe_outfile, row.names = FALSE)
  return(dfswe)
}

swe_5wj <- CalculateNixmassOnCsv("hs_data_5WJ.csv", "swe_data_5WJ.csv")
swe_5df <- CalculateNixmassOnCsv("hs_data_5DF.csv", "swe_data_5DF.csv")
swe_1ad <- CalculateNixmassOnCsv("hs_data_1AD.csv", "swe_data_1AD.csv")

