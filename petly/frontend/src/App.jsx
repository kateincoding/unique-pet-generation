import React, { useState, useEffect, useCallback } from "react";
import { IntroScreen, CaptureScreen, AnalyzeScreen, RevealScreen, GalleryScreen } from "./screens.jsx";
import { petByKey } from "./pets.jsx";
import * as api from "./api.js";

// Re-hidrata una mascota del backend con su componente SVG (no se persiste el Comp).
function hydrate(pet) {
  const sp = petByKey(pet.speciesKey);
  return { ...pet, species: sp, Comp: sp.Comp };
}

export default function App() {
  const [view, setView] = useState("intro"); // intro|capture|analyze|reveal|gallery
  const [photo, setPhoto] = useState(null);
  const [currentPet, setCurrentPet] = useState(null);
  const [collection, setCollection] = useState([]);
  const [savedThisRound, setSavedThisRound] = useState(false);
  const [analyzeError, setAnalyzeError] = useState(null);

  const loadCollection = useCallback(async () => {
    try { setCollection(await api.listPets()); } catch { /* backend offline */ }
  }, []);
  useEffect(() => { loadCollection(); }, [loadCollection]);

  function startFlow() {
    setSavedThisRound(false);
    setAnalyzeError(null);
    setView("capture");
  }

  async function runAnalyze(dataUrl) {
    setAnalyzeError(null);
    try {
      const pet = await api.analyze(dataUrl);
      setCurrentPet(hydrate(pet));
      setSavedThisRound(false);
      setView("reveal");
    } catch (e) {
      setAnalyzeError(e.message || "Error al analizar la foto");
    }
  }

  function onCaptured(dataUrl) {
    setPhoto(dataUrl);
    setView("analyze");
    runAnalyze(dataUrl);
  }

  async function saveCurrent() {
    if (!currentPet || savedThisRound) return;
    const { Comp, species, id, ...rest } = currentPet;
    try {
      // Linkeamos la foto de entrada con su mascota generada (para reabrir bien).
      const saved = await api.savePet({ ...rest, photo: currentPet.photo || photo });
      setCollection((c) => [saved, ...c]);
      setCurrentPet((p) => ({ ...p, id: saved.id }));
      setSavedThisRound(true);
    } catch (e) {
      alert("No se pudo guardar: " + e.message);
    }
  }

  async function removePet(id) {
    if (!id || !window.confirm("¿Borrar esta mascota de tu colección?")) return;
    try {
      await api.deletePet(id);
      setCollection((c) => c.filter((p) => p.id !== id));
    } catch (e) {
      alert("No se pudo borrar: " + e.message);
    }
  }

  return (
    <div className="app">
      <div className="bg-blobs"><div className="blob b1"/><div className="blob b2"/><div className="blob b3"/></div>

      <header className="topbar">
        <div className="brand" onClick={() => setView("intro")}>
          <div className="brand-mark">
            <img src="/petly-logo.svg" alt="Petly" width="40" height="40" />
          </div>
          <div>
            <div className="brand-name">Petly</div>
            <div className="brand-tag">Tu foto → mascota peluda</div>
          </div>
        </div>
        <div className="topnav">
          <button className={view === "intro" ? "active" : ""} onClick={() => setView("intro")}>Inicio</button>
          <button className={["capture", "analyze", "reveal"].includes(view) ? "active" : ""} onClick={startFlow}>Crear</button>
          <button className="gallery-pill" onClick={() => setView("gallery")}>
            Mi colección
            <span className="count">{collection.length}</span>
          </button>
        </div>
      </header>

      <main>
        {view === "intro" && (
          <IntroScreen onStart={startFlow} onGoGallery={() => setView("gallery")} galleryCount={collection.length} />
        )}
        {view === "capture" && <CaptureScreen onCapture={onCaptured} />}
        {view === "analyze" && <AnalyzeScreen photo={photo} error={analyzeError} onRetry={startFlow} />}
        {view === "reveal" && currentPet && (
          <RevealScreen pet={currentPet} photo={currentPet.photo || photo} saved={savedThisRound} onSave={saveCurrent} onRetry={startFlow} />
        )}
        {view === "gallery" && (
          <GalleryScreen
            pets={collection}
            onNew={startFlow}
            onOpen={(p) => { setCurrentPet(hydrate(p)); setPhoto(p.photo || null); setSavedThisRound(true); setView("reveal"); }}
            onDelete={removePet}
          />
        )}
      </main>
    </div>
  );
}
