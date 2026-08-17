# src/dsai/run_trajlevel.R
#
# Reads parsed trajectory CSV (from parse_tdump.py) + Henry Pirker's raw
# history CSV, merges on date, and runs openair::trajLevel() for PSCF
# and CWT. Usage:
#   Rscript run_trajlevel.R <traj_csv> <henry_pirker_csv> <pollutant> <out_dir>

suppressMessages(library(openair))

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
    main = paste("Henry Pirker -", pollutant, "-", stat, "(sample test, n =", length(unique(mytraj$date)), "days)")
  ))
  dev.off()
  cat("Wrote", out_file, "\n")
}
