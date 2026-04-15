// ===============================
// TRAJECTORY ENGINE (AB_winds)
// ===============================

// Constants
const EARTH_M_PER_DEG_LAT = 111320.0;

// Step one segment backward
function stepBackOneSegment(lat, lon, ws, wdDeg, hours = 1.0) {
  // assume wdDeg = coming-from direction (standard met)
  const theta = (wdDeg * Math.PI) / 180.0;

  // meteorological → u/v
  const u = -ws * Math.sin(theta); // east-west
  const v = -ws * Math.cos(theta); // north-south

  const dt = hours * 3600.0;

  const dxBack = -u * dt;
  const dyBack = -v * dt;

  const dLat = dyBack / EARTH_M_PER_DEG_LAT;

  let metersPerDegLon = EARTH_M_PER_DEG_LAT * Math.cos(lat * Math.PI / 180.0);
  if (Math.abs(metersPerDegLon) < 1e-6) metersPerDegLon = 1e-6;

  const dLon = dxBack / metersPerDegLon;

  return [lat + dLat, lon + dLon];
}

// Main trajectory
function interpDir(d0, d1, t) {
  let diff = d1 - d0;
  if (diff > 180) diff -= 360;
  if (diff < -180) diff += 360;
  return (d0 + t * diff + 360) % 360;
}

function computeBackTrajectory(lat0, lon0, winds) {
  const points = [[lat0, lon0]];
  let lat = lat0;
  let lon = lon0;

  if (!winds || winds.length < 2) return points;

  for (let i = 0; i < winds.length - 1; i++) {
    const w0 = winds[i];
    const w1 = winds[i + 1];

    const subSteps = 120; // 2-min resolution

    for (let j = 0; j < subSteps; j++) {
      const t = j / subSteps;

      const ws = w0.ws + t * (w1.ws - w0.ws);
      const wd = interpDir(w0.wd, w1.wd, t);

      const [latNew, lonNew] = stepBackOneSegment(lat, lon, ws, wd, 1 / subSteps);

      lat = latNew;
      lon = lonNew;
      points.push([lat, lon]);
    }
  }

  return points;
}

// Convert to GeoJSON
function trajectoryToGeoJSON(points, extraProps = {}) {
  return {
    type: "Feature",
    properties: { name: "back_trajectory", ...extraProps },
    geometry: {
      type: "LineString",
      coordinates: points.map(([lat, lon]) => [lon, lat])
    }
  };
}

// Distance helper
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = x => x * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);

  const a =
    Math.sin(dLat/2)**2 +
    Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dLon/2)**2;

  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// Find nearby features
function findFeaturesNearTrajectory(trajPoints, fc, maxDistKm = 5) {
  if (!fc?.features) return [];

  return fc.features
    .map(f => {
      const [lon, lat] = f.geometry.coordinates;
      let min = Infinity;

      for (const [tLat, tLon] of trajPoints) {
        const d = haversineKm(lat, lon, tLat, tLon);
        if (d < min) min = d;
      }

      return { feature: f, minDistKm: min };
    })
    .filter(r => r.minDistKm <= maxDistKm);
}

// Expose globally
window.computeBackTrajectory = computeBackTrajectory;
window.trajectoryToGeoJSON = trajectoryToGeoJSON;
window.findFeaturesNearTrajectory = findFeaturesNearTrajectory;
