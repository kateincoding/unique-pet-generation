// Cliente del backend Petly.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function asJson(res, fallbackMsg) {
  if (!res.ok) {
    let detail = fallbackMsg;
    try { detail = (await res.json()).detail || detail; } catch { /* sin body */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// data URL (base64) -> Blob, para subir la foto como multipart.
async function dataUrlToBlob(dataUrl) {
  const res = await fetch(dataUrl);
  return res.blob();
}

export async function analyze(dataUrl) {
  const blob = await dataUrlToBlob(dataUrl);
  const fd = new FormData();
  fd.append("photo", blob, "photo.jpg");
  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: fd });
  return asJson(res, "No se pudo analizar la foto");
}

export async function listPets() {
  return asJson(await fetch(`${BASE}/api/pets`), "No se pudo cargar la coleccion");
}

export async function savePet(pet) {
  const res = await fetch(`${BASE}/api/pets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pet),
  });
  return asJson(res, "No se pudo guardar");
}

export async function deletePet(id) {
  return asJson(await fetch(`${BASE}/api/pets/${id}`, { method: "DELETE" }), "No se pudo borrar");
}
