# src/dsai/stations.py
#
# STATIONS is a general name -> (lat, lon) lookup shared by everything
# (exceedance watch, batch climatology work, run_hysplit.py). It is NOT
# the exceedance-watch list on its own - see WATCH_STATIONS below.
# (Henry Pirker was added here for the batch trajectory work and, before
# this split existed, ended up implicitly in the hourly watch too since
# check_exceedances.py iterated this whole dict - caught 2026-08-17 when
# it fired 3 unintended HYSPLIT triggers. Fixed by scoping the watch to
# its own explicit list instead of "every station we happen to know
# coordinates for".)

STATIONS = {
    "Edmonton East": (53.5482115, -113.3680856),
    "Elk Island": (53.6824, -112.8681),
    "Fort Saskatchewan": (53.698756, -113.222831),
    "Bruderheim": (53.80012, -112.9278),
    "Ardrossan": (53.554691, -113.144105),
    "Gibbons": (53.827241, -113.327174),
    "Edmonton McCauley": (53.549509, -113.48593),
    "Edmonton-Gold Bar": (53.54925, -113.41473),
    "Edmonton-Beverly": (53.56693, -113.39849),

    # PAZA station - historical multi-year PSCF/CWT climatology work
    # only (batch_trajectories.py). Coordinates live here since
    # everything needing lat/lon reads this dict, but it's deliberately
    # excluded from WATCH_STATIONS below.
    "Grande Prairie - Henry Pirker": (55.1766364, -118.8077555),
}

# The actual hourly exceedance-watch corridor: Edmonton East and its
# Layer-4 neighbors from the July 18 2026 propagation check (see
# project_paza_mds_pipeline-adjacent DSAI work). check_exceedances.py
# and build_climatology_cache.py iterate this, NOT all of STATIONS.
WATCH_STATIONS = [
    "Edmonton East",
    "Elk Island",
    "Fort Saskatchewan",
    "Bruderheim",
    "Ardrossan",
    "Gibbons",
    "Edmonton McCauley",
    "Edmonton-Gold Bar",
    "Edmonton-Beverly",
]

# Pollutants we have a proven, meaningful story for (refinery/sour-gas
# corridor). Easy to extend to SO2's usual companions (NOx, PM2.5) once
# this is validated operationally.
PARAMETERS = [
    "Hydrogen Sulphide",
    "Sulphur Dioxide",
]
