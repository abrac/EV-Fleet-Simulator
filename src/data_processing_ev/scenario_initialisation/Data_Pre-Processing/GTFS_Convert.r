#!/usr/bin/env Rscript

library("gtfs2gps")
library("here")
library("magrittr")

cat("Would you like to filter the GTFS data by agencies? [y]/n\n")
filtering <- readline(prompt="    ")

if (tolower(filtering) != 'n'){
    cat('Looking for "./Original/GTFS_Orig.zip"...\n')

    gtfs = read_gtfs(here("Original", "GTFS_Orig.zip"))

    agencies <- c()

    cat(
    "Type the IDs of agencies you want to keep,
    pressing Enter each time. When you want
    to stop entering agencies, enter a blank line. 
    If you want to keep all the agencies, don't 
    enter any IDs.\n")

    repeat{
        agency <- readline(prompt="    ")
        if (agency == ""){
            break
        }
        agencies <- append(agencies, agency)
    }

    if (length(agencies) != 0) {
        gtfs_small <- filter_by_agency_id(gtfs, agencies)
    } else {
        gtfs_small <- gtfs
    }

    write_gtfs(gtfs_small, here("Original", "GTFS.zip"))
}

kampala_gps <- gtfs2gps(here("Original", "GTFS.zip"), parallel=TRUE, 
                        spatial_resolution=50)

kampala_gps <- adjust_speed(kampala_gps)

dir.create("Processed/monolithic", recursive=TRUE)
write.csv(kampala_gps, "Processed/monolithic/traces_orig.csv", quote=FALSE, 
          row.names=FALSE)
