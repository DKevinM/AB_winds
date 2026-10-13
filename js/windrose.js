// windrose.js
// Compass-style wind rose widget built on Plotly's barpolar trace (the
// standard tool for this chart type - proper compass gridlines, N/E/S/W
// ticks, hover tooltips). Reuses whatever wind series the calling page
// already has in hand (getWindSeries() on current_trajectory.html, the
// client-built winds[] array on historical_trajectory_rapid.html, or the
// backend-exported backtraj_windseries.json on historical_trajectory.html)
// - no new data fetching here.
//
// Only ~4 hourly points, not hundreds of observations, so this isn't a
// classic frequency-binned wind rose (too sparse to bin meaningfully) -
// each hour gets its own petal bar (length = speed), with opacity fading
// from newest (solid) to oldest (faint) to show how wind has been
// shifting over the lookback window.
//
// Requires Plotly.js (plotly.min.js, full bundle - barpolar isn't in the
// lighter partial bundles) loaded before this file.

(function () {
  const SIZE = 220;

  function withAlpha(colorStr, alpha) {
    const m = (colorStr || "").match(/\d+/g);
    if (!m || m.length < 3) return colorStr || `rgba(0,0,0,${alpha})`;
    return `rgba(${m[0]}, ${m[1]}, ${m[2]}, ${alpha})`;
  }

  // windSeries: array from getWindSeries()/equivalent - index 0 = most
  // recent hour, increasing index = further back in time. Each entry
  // {ws, wd} with wd = meteorological "coming from" bearing (0=N,90=E,...).
  window.buildWindRose = function (containerId, windSeries) {
    const el = document.getElementById(containerId);
    if (!el) return;

    if (!windSeries || !windSeries.length) {
      el.innerHTML = `<div style="opacity:0.6;font-size:12px;text-align:center;padding:20px 0;">No wind data</div>`;
      return;
    }

    if (typeof Plotly === "undefined") {
      console.error("windrose.js: Plotly is not loaded - add plotly.min.js before windrose.js");
      el.innerHTML = `<div style="opacity:0.6;font-size:12px;text-align:center;padding:20px 0;">Chart library failed to load</div>`;
      return;
    }

    const plotId = `${containerId}-plot`;
    const captionId = `${containerId}-caption`;
    el.innerHTML = `
      <div id="${plotId}" style="width:${SIZE}px;height:${SIZE}px;"></div>
      <div id="${captionId}" style="font-size:11px;text-align:center;margin-top:2px;"></div>
    `;

    const n = windSeries.length;
    // Plotly draws into its own div, so text color has to be read (not
    // inherited via currentColor) to match whatever panel this is dropped
    // into - light background on current_trajectory.html, dark on the
    // two odour/historical pages.
    const textColor = getComputedStyle(el).color || "rgb(51,51,51)";

    const theta = windSeries.map((w) => w.wd);
    const r = windSeries.map((w) => w.ws);
    const opacity = windSeries.map((_, i) => 1 - (i / Math.max(n - 1, 1)) * 0.75);
    const labels = windSeries.map((_, i) => (i === 0 ? "now" : `${i}h ago`));

    const trace = {
      type: "barpolar",
      r,
      theta,
      width: windSeries.map(() => 22),
      marker: {
        color: "#2c7fb8",
        opacity,
        line: { color: textColor, width: 0.5 },
      },
      customdata: labels,
      hovertemplate: "Wind FROM %{theta:.0f}°<br>%{r:.1f} m/s<br>%{customdata}<extra></extra>",
    };

    const layout = {
      width: SIZE,
      height: SIZE,
      margin: { t: 18, r: 18, b: 18, l: 18 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColor, size: 10 },
      showlegend: false,
      polar: {
        bgcolor: "rgba(0,0,0,0)",
        angularaxis: {
          rotation: 90,
          direction: "clockwise",
          tickmode: "array",
          tickvals: [0, 90, 180, 270],
          ticktext: ["N", "E", "S", "W"],
          gridcolor: withAlpha(textColor, 0.25),
          linecolor: withAlpha(textColor, 0.5),
          color: textColor,
        },
        radialaxis: {
          angle: 45,
          gridcolor: withAlpha(textColor, 0.25),
          linecolor: withAlpha(textColor, 0.5),
          color: textColor,
          ticksuffix: " m/s",
        },
      },
    };

    Plotly.newPlot(plotId, [trace], layout, { displayModeBar: false, responsive: false });

    const latest = windSeries[0];
    const sel = windSeries.find((w) => w.selected) || latest;
    if (sel.time_local) {
      // Model-run series (odour/run_dispersion.py, 2026-10-13): entries carry
      // their own times, so label the model's window, not "now / last Nh".
      const oldest = windSeries[n - 1];
      const span = Math.round((Date.parse(latest.time_utc) - Date.parse(oldest.time_utc)) / 3600000);
      const rel = sel.direction === "forward" ? "after" : "before";
      // same day -> "Oct 8 14:00 – 18:00"; crossing midnight keeps both dates
      const [od, ot] = [oldest.time_local.slice(0, -6), oldest.time_local.slice(-5)];
      const [ld, lt] = [latest.time_local.slice(0, -6), latest.time_local.slice(-5)];
      const windowTxt = od === ld ? `${od} ${ot} &ndash; ${lt}` : `${oldest.time_local} &ndash; ${latest.time_local}`;
      document.getElementById(captionId).innerHTML = `
        <div style="line-height:1.45;">
          <b>${sel.time_local}</b><br>
          Wind FROM ${Math.round(sel.wd)}° at ${sel.ws.toFixed(1)} m/s<br>
          <span style="opacity:0.8;">Window: ${windowTxt} (${span} h ${rel})</span><br>
          <span style="opacity:0.65;">newest = solid &middot; oldest = faint</span>
          ${sel.source ? `<br><span style="opacity:0.65;">${sel.source}</span>` : ""}
        </div>
      `;
    } else {
      document.getElementById(captionId).innerHTML = `
        <b>Wind FROM ${Math.round(latest.wd)}° at ${latest.ws.toFixed(1)} m/s (now)</b><br>
        <span style="opacity:0.65;">last ${n}h &middot; newest = solid, oldest = faint</span>
      `;
    }
  };
})();
