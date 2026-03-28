// ----------------- Helpers -----------------
function arrayToFeatureCollection(arr){
  if (!arr || !arr.length) return { type:'FeatureCollection', features: [] };
  const sample = arr[0] || {};
  const latK=['lat','latitude','Latitude','LAT','Lat'].find(k=>k in sample) || 'lat';
  const lonK=['lon','lng','long','longitude','Longitude','LON','Lng'].find(k=>k in sample) || 'lon';
  return {
    type:'FeatureCollection',
    features: arr.map(r => ({
      type:'Feature',
      properties: { ...r },
      geometry: {
        type:'Point',
        coordinates: [ +r[lonK], +r[latK] ]
      }
    })).filter(f =>
      Number.isFinite(f.geometry.coordinates[0]) &&
      Number.isFinite(f.geometry.coordinates[1])
    )
  };
}

function toPointFC(json) {
  const arr = Array.isArray(json)
    ? json
    : (json.data || json.features || Object.values(json || {}));

  return arrayToFeatureCollection(arr || []);
}




// ----------------- Back-trajectory helpers (JS version) -----------------

const EARTH_M_PER_DEG_LAT = 111320.0; // approx metres per degree latitude

/**
 * Step one segment of a back-trajectory.
 *
 * @param {number} lat - starting latitude (deg)
 * @param {number} lon - starting longitude (deg)
 * @param {number} ws  - wind speed [m/s]
 * @param {number} wdDeg - met wind direction [deg FROM which it blows]
 * @param {number} hours - duration of this segment [hours]
 * @returns {[number, number]} [latNew, lonNew]
 */
function stepBackOneSegment(lat, lon, ws, wdDeg, hours = 1.0) {
  // Direction air is moving TOWARD (flow direction)
  const flowDirDeg = (wdDeg + 180.0) % 360.0;
  const theta = flowDirDeg * Math.PI / 180.0; // radians

  // Components of the wind in m/s (forward trajectory, air movement)
  const u = ws * Math.sin(theta); // eastward
  const v = ws * Math.cos(theta); // northward

  // Distance travelled in this segment (forward)
  const dt = hours * 3600.0;  // seconds
  const dxForward = u * dt;   // metres east
  const dyForward = v * dt;   // metres north

  // For a BACK-trajectory, move opposite the air flow
  const dxBack = -dxForward;
  const dyBack = -dyForward;

  // Convert metres to degrees lat/lon
  const dLat = dyBack / EARTH_M_PER_DEG_LAT;
  let metersPerDegLon = EARTH_M_PER_DEG_LAT * Math.cos(lat * Math.PI / 180.0);
  if (Math.abs(metersPerDegLon) < 1e-6) metersPerDegLon = 1e-6; // avoid divide-by-zero

  const dLon = dxBack / metersPerDegLon;

  const latNew = lat + dLat;
  const lonNew = lon + dLon;
  return [latNew, lonNew];
}

/**
 * Compute a piecewise-linear back-trajectory.
 *
 * @param {number} lat0 - receptor latitude
 * @param {number} lon0 - receptor longitude
 * @param {Array<{ws:number, wd:number, hours?:number}>} winds
 *   winds[0] = now → 1h back, winds[1] = 1–2h back, etc.
 * @returns {Array<[number, number]>} list of [lat, lon] including start
 */
  
  function computeBackTrajectory(lat0, lon0, winds) {
    const points = [[lat0, lon0]];
    let lat = lat0;
    let lon = lon0;
  
    (winds || []).forEach(w => {
      const ws = Number(w.ws);
      const wd = Number(w.wd);
      const hours = w.hours != null ? Number(w.hours) : 1.0;
      if (!isFinite(ws) || !isFinite(wd) || !isFinite(hours)) return;
  
      const [latNew, lonNew] = stepBackOneSegment(lat, lon, ws, wd, hours);
      lat = latNew;
      lon = lonNew;
      points.push([lat, lon]);
    });
  
    return points;
  }




/**
 * Convert a list of [lat, lon] to a GeoJSON LineString.
 *
 * @param {Array<[number, number]>} points
 * @param {Object} extraProps - extra properties to attach
 * @returns {Object} GeoJSON Feature
 */
function trajectoryToGeoJSON(points, extraProps = {}) {
  return {
    type: "Feature",
    properties: {
      name: "back_trajectory",
      ...extraProps
    },
    geometry: {
      type: "LineString",
      coordinates: points.map(([lat, lon]) => [lon, lat]) // [lon, lat]
    }
  };
}

// Expose globally if needed
window.computeBackTrajectory = computeBackTrajectory;
window.trajectoryToGeoJSON = trajectoryToGeoJSON;








// ----------------- Distance + exposure helpers -----------------

/** Haversine distance between two points [km] */
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371.0; // km
  const toRad = x => x * Math.PI / 180.0;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Find station-like features within maxDistKm of any trajectory point.
 *
 * @param {Array<[number,number]>} trajPoints  [lat, lon] list
 * @param {Object} fc  GeoJSON FeatureCollection of points
 * @param {number} maxDistKm
 * @returns {Array<Object>} array of {feature, minDistKm}
 */
function findFeaturesNearTrajectory(trajPoints, fc, maxDistKm = 5) {
  if (!fc || !Array.isArray(fc.features)) return [];
  const results = [];

  fc.features.forEach(feat => {
    if (!feat.geometry || feat.geometry.type !== "Point") return;
    const [lon, lat] = feat.geometry.coordinates;
    if (!isFinite(lat) || !isFinite(lon)) return;

    let minDist = Infinity;
    for (const [tLat, tLon] of trajPoints) {
      const d = haversineKm(lat, lon, tLat, tLon);
      if (d < minDist) minDist = d;
      if (minDist <= maxDistKm) break;
    }

    if (minDist <= maxDistKm) {
      results.push({ feature: feat, minDistKm: minDist });
    }
  });

  return results;
}

// Expose globally
window.findFeaturesNearTrajectory = findFeaturesNearTrajectory;





// ----------------- 1) AQHI stations from last6h.csv -----------------

const unitsLookup = {
  "AQHI": " ", "Ozone": " ppb", "Total Oxides of Nitrogen": " ppb",
  "Hydrogen Sulphide": " ppb", "Total Reduced Sulphur": " ppb", "Sulphur Dioxide": " ppb",
  "Fine Particulate Matter": " µg/m³", "Total Hydrocarbons": " ppm", "Carbon Monoxide": " ppm",
  "Wind Direction": " degrees", "Relative Humidity": " %", "Outdoor Temperature": " °C",
  "Nitric Oxide": " ppb", "Wind Speed": " km/hr", "Non-methane Hydrocarbons": " ppm",
  "Nitrogen Dioxide": " ppb", "Methane": " ppm"
};

const abbrLookup = {
  "AQHI": "AQHI", "Ozone": "O₃", "Total Oxides of Nitrogen": "NOx",
  "Hydrogen Sulphide": "H₂S", "Total Reduced Sulphur": "TRS", "Sulphur Dioxide": "SO₂",
  "Fine Particulate Matter": "PM2.5", "Total Hydrocarbons": "THC", "Carbon Monoxide": "CO",
  "Wind Direction": "wd", "Relative Humidity": "RH", "Outdoor Temperature": "ET",
  "Nitric Oxide": "NO", "Wind Speed": "ws", "Non-methane Hydrocarbons": "NMHC",
  "Nitrogen Dioxide": "NO₂", "Methane": "CH₄"
};

const shortLookup = {
  "AQHI": "AQHI", "Ozone": "O3", "Total Oxides of Nitrogen": "NOX",
  "Hydrogen Sulphide": "H2S", "Total Reduced Sulphur": "TRS", "Sulphur Dioxide": "SO2",
  "Fine Particulate Matter": "PM2.5", "Total Hydrocarbons": "THC", "Carbon Monoxide": "CO",
  "Wind Direction": "wd", "Relative Humidity": "RH", "Outdoor Temperature": "ET",
  "Nitric Oxide": "NO", "Wind Speed": "ws", "Non-methane Hydrocarbons": "NMHC",
  "Nitrogen Dioxide": "NO2", "Methane": "CH4"
};

let dataByStation = {};
window.dataByStation = dataByStation;

window.dataReady = fetch('https://raw.githubusercontent.com/DKevinM/AB_datapull/main/data/last6h.csv')
  .then(res => res.text())
  .then(text => {
    const rows = text.trim().split('\n');
    const headers = rows.shift().split(',');

    const raw = {};
    rows.forEach(line => {
      const cols = line.split(',');
      const e = Object.fromEntries(headers.map((h,i)=>[h,cols[i]]));

      if (!e.Latitude||!e.Longitude||isNaN(e.Latitude)||isNaN(e.Longitude)) return;

      e.ParameterName = e.ParameterName||"AQHI";
      e.Units = unitsLookup[e.ParameterName]||"";
      e.Abbreviation = abbrLookup[e.ParameterName]||"";
      e.Shortform = shortLookup[e.ParameterName]||"";

      let v = parseFloat(e.Value);
      if (["Ozone","Total Oxides of Nitrogen","Hydrogen Sulphide","Total Reduced Sulphur","Sulphur Dioxide","Nitric Oxide","Nitrogen Dioxide"].includes(e.ParameterName)) {
        v *= 1000;
      }
      if (isNaN(v)) return;
      e.Value = v;

      const utc = new Date(e.ReadingDate);
      e.DisplayDate = utc.toLocaleString("en-CA", {
        timeZone: "America/Edmonton",
        hour12: true
      });
      e.ReadingDate = utc.toISOString();

      raw[e.StationName] = raw[e.StationName] || [];
      raw[e.StationName].push(e);
    });

    Object.entries(raw).forEach(([station, arr]) => {
      arr.sort((a, b) => new Date(b.ReadingDate) - new Date(a.ReadingDate));
      const byParam = {};
      arr.forEach(e => {
        const param = e.ParameterName;
        if (!byParam[param] || new Date(e.ReadingDate) > new Date(byParam[param].ReadingDate)) {
          byParam[param] = e;
        }
      });
      dataByStation[station] = Object.values(byParam);
    });
  });

window.fetchAllStationData = function () {
  const stationNames = Object.keys(dataByStation);
  if (!stationNames.length) return Promise.resolve([]);

  const orderedParams = [
    "AQHI", "Outdoor Temperature", "Relative Humidity", "Wind Speed", "Wind Direction", 
    "Nitrogen Dioxide", "Total Oxides of Nitrogen", "Nitric Oxide", "Ozone",
    "Fine Particulate Matter", "Sulphur Dioxide", "Hydrogen Sulphide", "Total Reduced Sulphur",
    "Carbon Monoxide", "Total Hydrocarbons", "Methane", "Non-methane Hydrocarbons"  
  ];

  const shortformOverride = {
    "Outdoor Temperature": "Temp",
    "Relative Humidity": "Humidity",
    "Wind Speed": "Wind Speed",
    "Wind Direction": "Wind Dir"
  };

  const results = stationNames.map(stationName => {
    const stationData = dataByStation[stationName];
    if (!stationData || !stationData.length) return null;

    const paramLookup = {};
    let latestTime = null;
    for (const r of stationData) {
      paramLookup[r.ParameterName] = r;
      const t = new Date(r.ReadingDate);
      if (!latestTime || t > latestTime) latestTime = t;
    }

    const displayTime = latestTime
      ? latestTime.toLocaleString("en-CA", { timeZone: "America/Edmonton", hour12: true })
      : "Invalid Date";

    const lines = orderedParams
      .filter(p => paramLookup[p] && p !== "AQHI")
      .map(p => {
        const r = paramLookup[p];
        const label = shortformOverride[p] || r.Shortform || p;
        const value = r.Value;
        const unit = r.Units || "";
        return `${label}: ${value}${unit}`;
      });

    const aqhiValue = paramLookup["AQHI"]?.Value || "N/A";
    const lat = stationData[0].Latitude;
    const lon = stationData[0].Longitude;

    return {
      stationName,
      lat: +lat,
      lon: +lon,
      aqhi: aqhiValue,
      html: `
        <div style="font-size:0.9em;">
          <strong>${stationName}</strong><br>
          <small><em>${displayTime}</em></small><br>
          AQHI: ${aqhiValue > 10 ? "10+" : aqhiValue}<br>
          ${lines.join("<br>")}
        </div>
      `
    };
  }).filter(Boolean);

  return Promise.resolve(results);
};

// Global FeatureCollections
window.STATIONS_FC = { type: "FeatureCollection", features: [] };
window.PURPLE_FC   = { type: "FeatureCollection", features: [] };
window.NPRI_FC     = { type: "FeatureCollection", features: [] };

// 1) Stations
window.stationsFCReady = (async () => {
  try {
    await window.dataReady;               // your last6h.csv loader
    const rows = await window.fetchAllStationData();

    window.STATIONS_FC = {
      type: "FeatureCollection",
      features: (rows || [])
        .filter(r => isFinite(+r.lat) && isFinite(+r.lon))
        .map(r => ({
          type: "Feature",
          properties: { ...r },
          geometry: { type: "Point", coordinates: [ +r.lon, +r.lat ] }
        }))
    };
    console.log("[origin] STATIONS_FC features:", window.STATIONS_FC.features.length);
  } catch (e) {
    console.error("[origin] stationsFCReady failed", e);
    window.STATIONS_FC = { type: "FeatureCollection", features: [] };
  }
})();

// 2) PurpleAir
window.purpleFCReady = (async () => {
  try {
    const res  = await fetch("https://raw.githubusercontent.com/DKevinM/AB_datapull/main/data/AB_PM25_map.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const json = await res.json();

    // Use the flexible helper so it works whether the file is an array,
    // an object with .data/.features, and whether it uses lat/Lat/Latitude, etc.
    window.PURPLE_FC = toPointFC(json);

    console.log("[origin] PURPLE_FC features:", window.PURPLE_FC.features.length);
  } catch (e) {
    console.error("[origin] purpleFCReady failed", e);
    window.PURPLE_FC = { type: "FeatureCollection", features: [] };
  }
})();


// ----------------- 3) NPRI (STATIC DATASET) -----------------

const NPRI_URL = "./data/NPRI.geojson";   // ← path inside NextGen_dk

window.NPRI_FC = { type: "FeatureCollection", features: [] };

window.npriFCReady = fetch(NPRI_URL)
  .then(r => {
    if (!r.ok) throw new Error(`NPRI load failed: ${r.status}`);
    return r.json();
  })
  .then(j => {
    window.NPRI_FC = j;
    console.log("[origin] NPRI loaded from static file:", j.features.length);
  })
  .catch(e => {
    console.error("[origin] NPRI static load failed", e);
    window.NPRI_FC = { type: "FeatureCollection", features: [] };
  });


// // 3) NPRI - ENHANCED VERSION WITH MULTI-SECTOR FETCHING
// window.NPRI_FC = { type: "FeatureCollection", features: [] };

// // Enhanced NPRI fetching with multiple sector queries
// window.npriFCReady = (async () => {
//   try {
//     console.log("[origin] Starting enhanced NPRI data fetch...");
    
//     // List of sector queries to get ALL facilities
//     const sectorQueries = [
//       { name: 'Conventional Oil and Gas', where: "SectorDescriptionEn LIKE '%Oil and Gas%'" },
//       { name: 'Oil Sands', where: "SectorDescriptionEn LIKE '%Oil Sands%'" },
//       { name: 'Mining', where: "SectorDescriptionEn LIKE '%Mining%'" },
//       { name: 'Electric Power', where: "SectorDescriptionEn LIKE '%Electric Power%'" },
//       { name: 'Waste Treatment', where: "SectorDescriptionEn LIKE '%Waste%'" },
//       { name: 'Chemicals', where: "SectorDescriptionEn LIKE '%Chemical%'" },
//       { name: 'Metals', where: "SectorDescriptionEn LIKE '%Metal%'" },
//       { name: 'Pulp and Paper', where: "SectorDescriptionEn LIKE '%Pulp%' OR SectorDescriptionEn LIKE '%Paper%'" },
//       { name: 'Cement', where: "SectorDescriptionEn LIKE '%Cement%'" },
//       { name: 'Other Manufacturing', where: "SectorDescriptionEn LIKE '%Manufacturing%'" },
//       { name: 'Other Sectors', where: "1=1" } // Catch-all for any remaining sectors
//     ];

//     let allFeatures = [];
//     let totalSectors = sectorQueries.length;
//     let completedSectors = 0;

//     for (const sectorQuery of sectorQueries) {
//       console.log(`[origin] Fetching NPRI sector: ${sectorQuery.name}`);
      
//       try {
//         const features = await fetchNPRISectorData(sectorQuery);
//         allFeatures = allFeatures.concat(features);
//         completedSectors++;
        
//         console.log(`[origin] ${sectorQuery.name}: ${features.length} facilities (${completedSectors}/${totalSectors} sectors completed)`);
        
//         // Small delay to be respectful to the server
//         await new Promise(resolve => setTimeout(resolve, 300));
        
//       } catch (sectorError) {
//         console.warn(`[origin] Failed to fetch ${sectorQuery.name}:`, sectorError);
//         completedSectors++;
//       }
//     }

//     console.log(`[origin] Total NPRI facilities loaded: ${allFeatures.length} from ${completedSectors} sectors`);

//     // Convert to proper GeoJSON features
//     const feats = allFeatures
//       .map(f => {
//         const g = f.geometry || {};
//         const attrs = f.properties || f.attributes || {};

//         let x, y;
        
//         // Handle different coordinate formats
//         if (g.type === 'Point' && Array.isArray(g.coordinates)) {
//           // Standard GeoJSON: [lon, lat]
//           [x, y] = g.coordinates;
//         } else if (g.x !== undefined && g.y !== undefined) {
//           // Esri format: {x: lon, y: lat}
//           x = g.x;
//           y = g.y;
//         } else {
//           // Fallback to properties
//           x = attrs.Longitude || attrs.longitude || attrs.LON;
//           y = attrs.Latitude || attrs.latitude || attrs.LAT;
//         }

//         x = Number(x);
//         y = Number(y);
//         if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

//         return {
//           type: "Feature",
//           geometry: {
//             type: "Point",
//             coordinates: [x, y]
//           },
//           properties: {
//             FACILITY_NAME: attrs.FACILITY_NAME || attrs.FacilityName,
//             COMPANY_NAME: attrs.COMPANY_NAME || attrs.CompanyName,
//             REPORTING_YEAR: attrs.REPORTING_YEAR || attrs.ReportYear,
//             SectorDescriptionEn: attrs.SectorDescriptionEn,
//             City: attrs.City,
//             ProvinceCode: attrs.ProvinceCode,
//             NAICS__Code_SCIAN: attrs.NAICS__Code_SCIAN
//           }
//         };
//       })
//       .filter(Boolean);

//     window.NPRI_FC = {
//       type: "FeatureCollection",
//       features: feats
//     };

//     console.log("[origin] NPRI_FC features after filtering:", feats.length);
    
//     // Log sector distribution for debugging
//     const sectorCounts = {};
//     feats.forEach(f => {
//       const sector = f.properties.SectorDescriptionEn || 'Unknown';
//       sectorCounts[sector] = (sectorCounts[sector] || 0) + 1;
//     });
//     console.log("[origin] NPRI sectors distribution:", sectorCounts);

//   } catch (e) {
//     console.error("[origin] npriFCReady failed", e);
//     window.NPRI_FC = { type: "FeatureCollection", features: [] };
//   }
// })();


// window.npriFCReady.then(() => {
//   console.log("NPRI ready:", window.NPRI_FC.features.length);

//   // Save a downloadable version
//   const blob = new Blob([JSON.stringify(window.NPRI_FC)], { type: "application/json" });
//   const url = URL.createObjectURL(blob);

//   const a = document.createElement("a");
//   a.href = url;
//   a.download = "NPRI.geojson";
//   a.click();
// });




// // Helper function to fetch data for a specific sector
// async function fetchNPRISectorData(sectorQuery, attempt = 1) {
//   const baseUrl = 'https://maps-cartes.ec.gc.ca/arcgis/rest/services/STB_DGST/NPRI/MapServer/0/query';
  
//   // Focus on Alberta facilities in the specified sector
//   const whereClause = `ProvinceCode='AB' AND (${sectorQuery.where})`;
//   const url = `${baseUrl}?where=${encodeURIComponent(whereClause)}&outFields=*&returnGeometry=true&f=geojson&outSR=4326&resultRecordCount=2000`;
  
//   try {
//     const response = await fetch(url);
//     if (!response.ok) throw new Error(`HTTP ${response.status}`);
//     const data = await response.json();
    
//     if (data.error) {
//       throw new Error(data.error.message || 'API error');
//     }
    
//     return data.features || [];
//   } catch (error) {
//     console.warn(`[origin] Attempt ${attempt} failed for ${sectorQuery.name}:`, error);
    
//     if (attempt < 2) {
//       // Wait and retry once
//       await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
//       return fetchNPRISectorData(sectorQuery, attempt + 1);
//     }
    
//     return [];
//   }
// }
