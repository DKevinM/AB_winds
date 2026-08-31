// windrose.js
// Small compass-style widget showing the wind vectors that actually drove
// the visible trajectory - reuses current_trajectory.html's own winds3h
// series (getWindSeries()) rather than fetching anything new, so it's
// always exactly consistent with what's on the map.
//
// Only ~4 hourly points, not hundreds of observations, so this isn't a
// classic frequency-binned wind rose (too sparse to bin meaningfully) -
// instead each hour gets its own arrow: length = speed, opacity = recency
// (newest = solid, oldest = faint), pointing from the compass bearing the
// wind came FROM toward the center. That shows how wind has been shifting
// over the lookback window, which matters more for odour tracing than a
// single blended average would.

(function () {
  const SIZE = 150;
  const CENTER = SIZE / 2;
  const MAX_RADIUS = 52;
  const MAX_SPEED_MS = 12; // wind at/above this maps to full-length arrow

  function polarToXY(bearingDeg, radius) {
    // bearingDeg: compass bearing (0 = N, 90 = E), standard meteorological
    const rad = (bearingDeg - 90) * (Math.PI / 180); // shift so 0deg = up
    return {
      x: CENTER + radius * Math.cos(rad),
      y: CENTER + radius * Math.sin(rad),
    };
  }

  function arrowSVG(wd, ws, opacity, color) {
    const len = Math.min(ws / MAX_SPEED_MS, 1) * MAX_RADIUS;
    const tail = polarToXY(wd, MAX_RADIUS);       // outer edge = where wind comes FROM
    const head = polarToXY(wd, MAX_RADIUS - len); // points inward toward center

    // arrowhead as a small triangle at `head`, oriented along the line
    const angle = Math.atan2(head.y - tail.y, head.x - tail.x);
    const ah = 7; // arrowhead size
    const p1 = { x: head.x, y: head.y };
    const p2 = {
      x: head.x - ah * Math.cos(angle - Math.PI / 7),
      y: head.y - ah * Math.sin(angle - Math.PI / 7),
    };
    const p3 = {
      x: head.x - ah * Math.cos(angle + Math.PI / 7),
      y: head.y - ah * Math.sin(angle + Math.PI / 7),
    };

    return `
      <line x1="${tail.x.toFixed(1)}" y1="${tail.y.toFixed(1)}"
            x2="${head.x.toFixed(1)}" y2="${head.y.toFixed(1)}"
            stroke="${color}" stroke-width="3" stroke-opacity="${opacity}" stroke-linecap="round" />
      <polygon points="${p1.x.toFixed(1)},${p1.y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)} ${p3.x.toFixed(1)},${p3.y.toFixed(1)}"
               fill="${color}" fill-opacity="${opacity}" />
    `;
  }

  // windSeries: array from getWindSeries() - index 0 = most recent hour,
  // increasing index = further back in time.
  window.buildWindRose = function (containerId, windSeries) {
    const el = document.getElementById(containerId);
    if (!el) return;

    if (!windSeries || !windSeries.length) {
      el.innerHTML = `<div style="color:#999;font-size:12px;text-align:center;padding:20px 0;">No wind data</div>`;
      return;
    }

    const n = windSeries.length;
    const color = "#2c7fb8"; // matches the trajectory line's own blue

    // currentColor + opacity (not fixed grays) so this renders correctly
    // whether it's dropped into a light panel or a dark one.
    const rings = [1, 2, 3].map((f) => {
      const r = (MAX_RADIUS * f) / 3;
      return `<circle cx="${CENTER}" cy="${CENTER}" r="${r}" fill="none" stroke="currentColor" stroke-opacity="0.25" stroke-width="1" />`;
    }).join("");

    const cardinals = [
      { label: "N", bearing: 0 },
      { label: "E", bearing: 90 },
      { label: "S", bearing: 180 },
      { label: "W", bearing: 270 },
    ].map(({ label, bearing }) => {
      const p = polarToXY(bearing, MAX_RADIUS + 12);
      return `<text x="${p.x.toFixed(1)}" y="${p.y.toFixed(1)}" text-anchor="middle"
                dominant-baseline="middle" font-size="11" font-weight="700" fill="currentColor" fill-opacity="0.7">${label}</text>`;
    }).join("");

    const arrows = windSeries.map((w, i) => {
      // newest (i=0) fully opaque, fading toward the oldest point (min 0.25)
      const opacity = 1 - (i / Math.max(n - 1, 1)) * 0.75;
      return arrowSVG(w.wd, w.ws, opacity.toFixed(2), color);
    }).join("");

    const latest = windSeries[0];
    const caption = `Wind FROM ${Math.round(latest.wd)}° at ${latest.ws.toFixed(1)} m/s (now)`;

    el.innerHTML = `
      <svg width="${SIZE}" height="${SIZE}" viewBox="0 0 ${SIZE} ${SIZE}" style="color:inherit;">
        ${rings}
        <circle cx="${CENTER}" cy="${CENTER}" r="${MAX_RADIUS}" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5" />
        ${cardinals}
        ${arrows}
        <circle cx="${CENTER}" cy="${CENTER}" r="3" fill="currentColor" />
      </svg>
      <div style="font-size:11px;opacity:0.9;text-align:center;margin-top:2px;font-weight:600;">
        ${caption}
      </div>
      <div style="font-size:10px;opacity:0.65;text-align:center;">
        last ${n}h &middot; newest = solid, oldest = faint
      </div>
    `;
  };
})();
