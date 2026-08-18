# src/dsai/run_trajlevel.R
#
# Reads parsed trajectory CSV (from parse_tdump.py) + Henry Pirker's raw
# history CSV, merges on date, and runs openair::trajLevel() for PSCF
# and CWT. Usage:
#   Rscript run_trajlevel.R <traj_csv> <henry_pirker_csv> <pollutant> <out_dir>

suppressMessages(library(openair))

# PM2.5 and CO show a simultaneous multi-sector elevation from late 2021
# through 2023 at Henry Pirker - verified against real wildfire records
# (2021 BC's worst wildfire season to that point, 2023 Alberta's worst
# on record, Grande Prairie smoke-impacted 100+ days). Multi-sector-
# simultaneous is the tell that this is regional wildfire smoke, not a
# directional point source, so those two pollutants exclude these
# calendar years from the climatology; every other pollutant showed no
# such contamination and uses full history.
WILDFIRE_EXCLUDED_POLLUTANTS <- c("PM2.5", "CO")
WILDFIRE_EXCLUDE_START <- as.POSIXct("2021-01-01 00:00:00", tz = "UTC")
WILDFIRE_EXCLUDE_END   <- as.POSIXct("2024-01-01 00:00:00", tz = "UTC")

# 72h backward trajectories occasionally get caught in a fast-moving
# low-level jet and stretch across the whole Pacific/Arctic (99th
# percentile lon/lat still lands within this window) - cropping to the
# region that actually explains the vast majority of transport keeps
# the map readable instead of a washed-out world view. This drops a
# handful of extreme-outlier days from the plotted statistic, same as
# the regional framing used in Kindzierski's own published PSCF/CWT work.
PLOT_LON_RANGE <- c(-150, -100)
PLOT_LAT_RANGE <- c(40, 68)

args <- commandArgs(trailingOnly = TRUE)
traj_csv <- args[1]
readings_csv <- args[2]
pollutant <- args[3]
out_dir <- args[4]

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

traj <- read.csv(traj_csv, stringsAsFactors = FALSE)
traj$date <- as.POSIXct(traj$date, tz = "UTC")

readings <- read.csv(readings_csv, stringsAsFactors = FALSE, check.names = FALSE)
readings$date <- as.POSIXct(readings$date, format = "%Y-%m-%d %H:%M", tz = "UTC")

# trajectories are keyed on the exact hour they were run (19:00 UTC);
# match each trajectory's arrival date to that same hour's real reading
merged_readings <- readings[, c("date", pollutant)]
names(merged_readings)[2] <- "conc"

mytraj <- merge(traj, merged_readings, by = "date")
mytraj <- mytraj[!is.na(mytraj$conc), ]

if (pollutant %in% WILDFIRE_EXCLUDED_POLLUTANTS) {
  before_n <- length(unique(mytraj$date))
  mytraj <- mytraj[mytraj$date < WILDFIRE_EXCLUDE_START | mytraj$date >= WILDFIRE_EXCLUDE_END, ]
  after_n <- length(unique(mytraj$date))
  cat(sprintf("Wildfire-year exclusion (2021-2023) applied for %s: %d -> %d days\n",
              pollutant, before_n, after_n))
}

mytraj <- mytraj[mytraj$lon >= PLOT_LON_RANGE[1] & mytraj$lon <= PLOT_LON_RANGE[2] &
                  mytraj$lat >= PLOT_LAT_RANGE[1] & mytraj$lat <= PLOT_LAT_RANGE[2], ]

cat(sprintf("Merged trajectory rows: %d (from %d unique days)\n",
            nrow(mytraj), length(unique(mytraj$date))))

if (length(unique(mytraj$date)) < 5) {
  cat("Too few days with both trajectory + real reading data - skipping plot.\n")
  quit(status = 1)
}

for (stat in c("PSCF", "CWT")) {
  out_file <- file.path(out_dir, paste0(pollutant, "_", stat, ".png"))
  png(out_file, width = 900, height = 800)
  print(trajLevel(
    mytraj,
    lon = "lon", lat = "lat",
    pollutant = "conc",
    statistic = stat,
    col = "increment",
    map = TRUE,
    main = paste("Henry Pirker -", pollutant, "-", stat, "(n =", length(unique(mytraj$date)), "days)")
  ))
  dev.off()
  cat("Wrote", out_file, "\n")
}
