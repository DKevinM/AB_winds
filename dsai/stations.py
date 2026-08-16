# src/dsai/stations.py
#
# Watched stations for exceedance detection + auto-triggered HYSPLIT
# ensembles. Starting set: Edmonton East and its Layer-4 neighbors from
# the July 18 2026 propagation check (see project_paza_mds_pipeline-
# adjacent DSAI work). Easy to extend - just add StationName: (lat, lon).

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
}

# Pollutants we have a proven, meaningful story for (refinery/sour-gas
# corridor). Easy to extend to SO2's usual companions (NOx, PM2.5) once
# this is validated operationally.
PARAMETERS = [
    "Hydrogen Sulphide",
    "Sulphur Dioxide",
]
