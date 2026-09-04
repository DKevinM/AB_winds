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

    # --- ACA (rest of network beyond the original corridor) ---
    "Edmonton Lendrum": (53.4978, -113.527),
    "St. Albert": (53.6269722, -113.6119166),
    "Woodcroft": (53.564411, -113.562583),
    "O’Morrow Station 1": (54.9036817, -112.8433489),
    "Enoch": (53.4980544, -113.7607143),

    # --- WCAS ---
    "Carrot Creek": (53.6210616, -115.8691549),
    "Steeper": (53.1325, -117.091111),
    "Hinton-Drinnan": (53.4273011, -117.544067),
    "Meadows": (53.5299999, -114.6361),
    "Wagner2": (53.429444, -114.380556),
    "Genesee": (53.306507, -114.200347),
    "Drayton Valley": (53.220056, -114.983408),
    "Edson": (53.593611, -116.392778),
    "Breton": (53.090278, -114.460556),
    "Hinton-Hillcrest": (53.3927365, -117.5846086),
    "Jasper": (52.873528, -118.09144),

    # --- PAZA ---
    # Grande Prairie - Henry Pirker was previously excluded from the
    # watch (batch-trajectory-only); now included as part of PAZA scope.
    "Grande Prairie - Henry Pirker": (55.1766364, -118.8077555),
    # Beaverlodge coords from the government Stations endpoint
    # (StationKey 157) - PAZA's own stations.py only has TODO placeholders
    # for this one. Was a one-off manual add 2026-09-01; now in the
    # regular watch as part of PAZA scope.
    "Beaverlodge": (55.1963, -119.3968),
    "Fox Creek": (54.395492, -116.80948),
    "Happy Valley": (55.752778, -119.086667),
    "Smoky Heights": (55.405, -118.275),

    # Valleyview, Dunes, Milner: PAZA stations with no SO2/H2S rows in
    # aqhi_data yet (PAZA_data_pipe's MDS zone-code prefix is still
    # unresolved for most channels - see project_paza_mds_pipeline
    # memory). Left out of STATIONS/WATCH_STATIONS until real data
    # flows; add coordinates here once they do.

    # Poacher's Landing Station 2: ACA station, but only TRS/WS/WD/BP -
    # no SO2/H2S channel, so nothing for this watch to check yet.
}

# The actual hourly exceedance-watch list. check_exceedances.py and
# build_climatology_cache.py iterate this, NOT all of STATIONS.
# Started as just Edmonton East and its Layer-4 neighbors from the
# July 18 2026 propagation check; expanded 2026-09-04 to the full
# ACA/WCAS/PAZA roster (stations with SO2/H2S data - see the STATIONS
# dict above for the ones left out and why). Deliberately NOT
# network-wide (86 AB stations) and NOT extended to PM2.5 yet - a
# wildfire smoke episode would trip the percentile threshold at most
# stations simultaneously and re-trigger every hour it stays elevated,
# which is a regional-transport problem this per-station/per-timestamp
# trigger isn't built to collapse. Revisit with a regional-simultaneity
# guard before adding PM2.5 or going network-wide.
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

    # ACA
    "Edmonton Lendrum",
    "St. Albert",
    "Woodcroft",
    "O’Morrow Station 1",
    "Enoch",

    # WCAS
    "Carrot Creek",
    "Steeper",
    "Hinton-Drinnan",
    "Meadows",
    "Wagner2",
    "Genesee",
    "Drayton Valley",
    "Edson",
    "Breton",
    "Hinton-Hillcrest",
    "Jasper",

    # PAZA
    "Grande Prairie - Henry Pirker",
    "Beaverlodge",
    "Fox Creek",
    "Happy Valley",
    "Smoky Heights",
]

# Pollutants we have a proven, meaningful story for (refinery/sour-gas
# corridor). Easy to extend to SO2's usual companions (NOx, PM2.5) once
# this is validated operationally.
PARAMETERS = [
    "Hydrogen Sulphide",
    "Sulphur Dioxide",
]
