function initParticles(n = 500) {
  particles = [];
  const bounds = map.getBounds();

  for (let i = 0; i < n; i++) {
    const lat = bounds.getSouth() + Math.random() * (bounds.getNorth() - bounds.getSouth());
    const lon = bounds.getWest()  + Math.random() * (bounds.getEast()  - bounds.getWest());

    particles.push({
      lat,
      lon,
      age: Math.random() * 100
    });
  }
}

function evolveParticles(dtSeconds = 60) {
  const bounds = map.getBounds();

  for (const p of particles) {
    p.age += 1;
    if (p.age > 200) {
      // respawn
      p.age = 0;
      p.lat = bounds.getSouth() + Math.random() * (bounds.getNorth() - bounds.getSouth());
      p.lon = bounds.getWest()  + Math.random() * (bounds.getEast()  - bounds.getWest());
      continue;
    }

    const w = interpolateWind(p.lat, p.lon);
    if (!w) {
      p.age = 999; // kill / respawn soon
      continue;
    }

    // crude: 1 second * speed [m/s] -> distance; convert to degrees
    const EARTH_R = 6371000;
    const dlat = (w.v * dtSeconds) / EARTH_R * (180 / Math.PI);
    const dlon = (w.u * dtSeconds) / (EARTH_R * Math.cos(p.lat * Math.PI/180)) * (180 / Math.PI);

    // move FORWARD in time (so trajectories show where air is going)
    p.lat += dlat;
    p.lon += dlon;

    if (!bounds.contains([p.lat, p.lon])) {
      p.age = 999;
    }
  }
}

function drawParticles() {
  windCtx.clearRect(0, 0, windCanvas.width, windCanvas.height);
  windCtx.globalAlpha = 0.6;
  windCtx.strokeStyle = "#00ffff";
  windCtx.lineWidth = 1;

  for (const p of particles) {
    const latlng = L.latLng(p.lat, p.lon);
    const pt = map.latLngToContainerPoint(latlng);

    windCtx.beginPath();
    windCtx.moveTo(pt.x, pt.y);
    windCtx.lineTo(pt.x - 1, pt.y - 1); // short streak – can improve later
    windCtx.stroke();
  }
}

function frame() {
  requestAnimationFrame(frame);
  if (!windField || !particles.length) return;
  evolveParticles(60);  // 60 s per frame (visual scale)
  drawParticles();
}
frame();
