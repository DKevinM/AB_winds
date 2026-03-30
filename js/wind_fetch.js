// ===============================
// SUPABASE WIND FETCH
// ===============================

// requires pako.js (gzip)

const SUPABASE_URL = "https://zcunoncbyitfsilrhymv.supabase.co";
const API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjdW5vbmNieWl0ZnNpbHJoeW12Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE3Mjk3NDYsImV4cCI6MjA2NzMwNTc0Nn0._z_tqm_5UIBkWfMa7HAJrUOA-0t9vOaBVV48-74esWQ";

// get latest file
async function getLatestWindFile() {
  const res = await fetch(
    `${SUPABASE_URL}/rest/v1/wind_files?order=valid_time.desc&limit=1`,
    {
      headers: {
        apikey: API_KEY
      }
    }
  );

  const data = await res.json();
  return data[0];
}

// fetch + unzip
async function fetchWindFile(storagePath) {
  const url = `${SUPABASE_URL}/storage/v1/object/public/winds/${storagePath}`;

  const res = await fetch(url);
  const buffer = await res.arrayBuffer();

  const decompressed = pako.inflate(new Uint8Array(buffer), { to: 'string' });
  return JSON.parse(decompressed);
}

// extract wind at point
function extractWindAtPoint(data, lat, lon) {
  const grid = data.grid;
  const fields = data.fields;

  const i = Math.round((lon - grid.lo1) / grid.dx);
  const j = Math.round((grid.la1 - lat) / grid.dy);

  if (i < 0 || j < 0 || i >= grid.nx || j >= grid.ny) return null;

  const idx = j * grid.nx + i;

  const u = fields["ugrd10"][idx];
  const v = fields["vgrd10"][idx];

  if (u == null || v == null) return null;

  const ws = Math.sqrt(u*u + v*v);
  const wd = (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;

  return { ws, wd };
}
