// data.js — Odour app data loader

const NPRI_URL = "./data/NPRI.geojson";

window.NPRI_FC = null;

window.npriFCReady = fetch(NPRI_URL)
  .then(r => {
    if (!r.ok) throw new Error("Failed to load NPRI.geojson");
    return r.json();
  })
  .then(j => {
    window.NPRI_FC = j;
    console.log("[odour] NPRI loaded:", j.features.length, "facilities");
  })
  .catch(err => {
    console.error("[odour] NPRI load failed:", err);
  });
