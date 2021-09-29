#!/usr/bin/env Rscript

library("gtfs2gps")
library("here")

kampala_gps <- gtfs2gps(here("Original", "GTFS.zip"), parallel=TRUE, spatial_resolution=50)

kampala_gps[, speed := as.numeric(speed)]  # Remove units.

# Make invalid values NA.
kampala_gps[speed == "Inf" | is.na(speed) | is.nan(speed), speed := NA]
kampala_gps[speed > 80 | speed < 2, speed := NA] # too slow or too fast

# Replace NA values by the mean speed of the trip.
# kampala_gps[is.na(speed), speed := mean(kampala_gps$speed, na.rm = TRUE), by = .(shape_id, trip_id, trip_number)]
kampala_gps[is.na(speed), speed := mean(kampala_gps$speed, na.rm = TRUE), by = .(shape_id)]

# Set the speed units to km/h.
kampala_gps[, speed := units::set_units(speed, "km/h")]

# Calculate the time based on the speed.
kampala_gps[, time := (dist / speed)]
kampala_gps[, cumtime_new := cumsum(time), by = .(shape_id, trip_id, trip_number)]
kampala_gps[, cumtime_new := units::set_units(cumtime_new, "s")]

dir.create("Processed/monolithic", recursive=TRUE)
write.csv(kampala_gps, "Processed/monolithic/traces_orig.csv", quote=FALSE, row.names=FALSE)
