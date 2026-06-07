/* eslint-disable */
// Petini — screen components
// Globals expected from pets.jsx: SPECIES, petByKey, randomPet, randomName

import { SPECIES, petByKey } from './pets.jsx';
import React, { useState, useEffect, useRef, useMemo } from 'react';

// ====== Helpers ======
function Sparkle({ x, y, size = 18, delay = 0, color = "#FFD93D" }) {
  return (
    <svg className="sparkle" style={{ left: `${x}%`, top: `${y}%`, width: size, height: size, animationDelay: `${delay}s` }} viewBox="0 0 24 24" fill={color}>
      <path d="M12 0 L14 10 L24 12 L14 14 L12 24 L10 14 L0 12 L10 10 Z"/>
    </svg>
  );
}

function Confetti({ count = 80 }) {
  const pieces = useMemo(() => {
    const colors = ["#FF4D8D", "#9D5BFF", "#4FE3C1", "#FFD93D", "#FF8A65", "#6CC7FF"];
    return Array.from({ length: count }).map((_, i) => ({
      left: Math.random() * 100,
      dx: (Math.random() - .5) * 200,
      delay: Math.random() * .8,
      rot: Math.random() * 360,
      color: colors[i % colors.length],
      w: 8 + Math.random() * 10,
      h: 12 + Math.random() * 14,
      shape: Math.random() > .5 ? "rect" : "circle",
    }));
  }, [count]);
  return (
    <div className="confetti-layer">
      {pieces.map((p, i) => (
        <div key={i} className="confetti-piece" style={{
          left: `${p.left}%`,
          background: p.color,
          width: p.w, height: p.h,
          borderRadius: p.shape === "circle" ? "50%" : "3px",
          transform: `rotate(${p.rot}deg)`,
          animationDelay: `${p.delay}s`,
          "--dx": `${p.dx}px`,
        }}/>
      ))}
    </div>
  );
}

// ====== INTRO ======
function IntroScreen({ onStart, onGoGallery, galleryCount }) {
  const HeroPet = SPECIES[0].Comp; // dog as hero
  return (
    <section className="screen intro screen-enter" data-screen-label="01 Intro">
      <div className="intro-copy">
        <h1>
          Tu selfie,<br/>
          ahora una <span className="pop">mascota</span><br/>
          <span className="pop2">mágica</span> ✨
        </h1>
        <p className="intro-sub">
          Tómate una foto o sube una y nuestro detector la convierte en tu mini-criatura única. Cada mascota es irrepetible — como tú.
        </p>
        <div className="intro-cta-row">
          <button className="btn btn-primary btn-xl" onClick={onStart}>
            ¡Crear mi mascota! <span style={{ fontSize: 20 }}>→</span>
          </button>
          {galleryCount > 0 && (
            <button className="btn btn-ghost" onClick={onGoGallery}>
              Ver mi colección ({galleryCount})
            </button>
          )}
        </div>
        <div className="intro-steps">
          <div className="intro-step"><span className="num">1</span> Toma tu foto</div>
          <div className="intro-step"><span className="num">2</span> IA detecta tu vibe</div>
          <div className="intro-step"><span className="num">3</span> ¡Conoce a tu mascota!</div>
        </div>
      </div>
      <div className="hero-stage">
        <Sparkle x={15} y={15} size={28} delay={0} />
        <Sparkle x={85} y={20} size={20} delay={.5} color="#FF4D8D" />
        <Sparkle x={20} y={75} size={22} delay={1} color="#4FE3C1" />
        <Sparkle x={80} y={70} size={26} delay={.3} color="#9D5BFF" />
        <Sparkle x={50} y={5} size={18} delay={1.5} />
        <div className="hero-orbit">
          <div className="chip c1"><span className="dot" style={{ background: "var(--mint)" }}/> Cute lvl 99</div>
          <div className="chip c2"><span className="dot" style={{ background: "var(--magenta)" }}/> Único</div>
          <div className="chip c3"><span className="dot" style={{ background: "var(--lemon)" }}/> +12 especies</div>
          <div className="chip c4"><span className="dot" style={{ background: "var(--purple)" }}/> 100% mágico</div>
        </div>
        <div className="floor"/>
        <div className="hero-pet"><HeroPet /></div>
      </div>
    </section>
  );
}

// ====== CAPTURE ======
function CaptureScreen({ onCapture }) {
  const [tab, setTab] = useState("camera");
  const [stream, setStream] = useState(null);
  const [streamErr, setStreamErr] = useState(null);
  const videoRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (tab !== "camera") return;
    let active = true;
    async function go() {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (!active) { s.getTracks().forEach(t => t.stop()); return; }
        setStream(s);
      } catch (e) {
        setStreamErr(e.name || "denied");
      }
    }
    go();
    return () => {
      active = false;
      if (stream) stream.getTracks().forEach(t => t.stop());
    };
    // eslint-disable-next-line
  }, [tab]);

  // Conecta el stream al <video> cuando AMBOS existen. El <video> recien se monta
  // cuando `stream` es truthy, asi que asignar srcObject dentro de go() no alcanza
  // (videoRef.current aun es null) -> la camara quedaba en negro.
  useEffect(() => {
    const v = videoRef.current;
    if (v && stream) {
      v.srcObject = stream;
      v.play?.().catch(() => {});
    }
  }, [stream]);

  useEffect(() => () => {
    if (stream) stream.getTracks().forEach(t => t.stop());
  }, [stream]);

  function snap() {
    if (videoRef.current) {
      const v = videoRef.current;
      const c = document.createElement("canvas");
      c.width = v.videoWidth || 640;
      c.height = v.videoHeight || 480;
      const ctx = c.getContext("2d");
      ctx.translate(c.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(v, 0, 0, c.width, c.height);
      const url = c.toDataURL("image/jpeg", .85);
      onCapture(url);
    } else {
      // fallback fake photo
      onCapture(null);
    }
  }

  function handleFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => onCapture(e.target.result);
    reader.readAsDataURL(file);
  }

  return (
    <section className="screen capture screen-enter" data-screen-label="02 Captura">
      <div className="capture-head">
        <h2>Muéstranos esa cara ✨</h2>
        <p>Una foto bien iluminada hace mascotas más mágicas</p>
      </div>

      <div className="tabs-wrap">
        <div className="tabs">
          <button className={`tab ${tab === "camera" ? "active" : ""}`} onClick={() => setTab("camera")}>
            <span>📸</span> Cámara
          </button>
          <button className={`tab ${tab === "upload" ? "active" : ""}`} onClick={() => setTab("upload")}>
            <span>🖼️</span> Subir foto
          </button>
        </div>
      </div>

      <div className="capture-stage">
        <div>
          {tab === "camera" ? (
            <>
              <div className="viewport">
                {stream && !streamErr ? (
                  <>
                    <video ref={videoRef} autoPlay playsInline muted />
                    <div className="face-frame"/>
                    <div className="viewport-overlay">
                      <span className="overlay-pill"><span className="live-dot"/> EN VIVO</span>
                      <span className="overlay-pill">HD · 1080p</span>
                    </div>
                  </>
                ) : streamErr ? (
                  <div className="viewport-empty">
                    <div className="ico">🙈</div>
                    <p>No pude acceder a tu cámara.<br/>Prueba con "Subir foto" 👉</p>
                  </div>
                ) : (
                  <div className="viewport-empty">
                    <div className="ico">📷</div>
                    <p>Activando cámara…</p>
                  </div>
                )}
              </div>
              <div className="shutter-row">
                <button className="swap-btn" title="Cambiar cámara">🔄</button>
                <button className="shutter" onClick={snap} aria-label="Tomar foto"/>
                <button className="upload-btn" title="Subir foto" onClick={() => setTab("upload")}>📁</button>
              </div>
            </>
          ) : (
            <label
              className={`dropzone ${dragOver ? "drag" : ""}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => {
                e.preventDefault();
                setDragOver(false);
                const f = e.dataTransfer.files?.[0];
                if (f) handleFile(f);
              }}
            >
              <input type="file" accept="image/*" style={{ display: "none" }} onChange={e => handleFile(e.target.files?.[0])} />
              <div>
                <div className="big-emoji">🪄</div>
                <h3>Suelta tu foto aquí</h3>
                <p>o haz click para elegir desde tu galería</p>
                <span className="btn btn-primary">Elegir archivo</span>
              </div>
            </label>
          )}
        </div>

        <aside className="capture-tips">
          <h3>Tips para una mascota épica</h3>
          <p>Mientras más cute la foto, más cute la mascota.</p>
          <div className="tip">
            <div className="tip-emoji">💡</div>
            <div className="tip-body"><strong>Buena luz</strong><span>La luz natural es la mejor amiga del aura</span></div>
          </div>
          <div className="tip">
            <div className="tip-emoji">😊</div>
            <div className="tip-body"><strong>Muestra tu cara</strong><span>Centra tu rostro en el círculo</span></div>
          </div>
          <div className="tip">
            <div className="tip-emoji">🎨</div>
            <div className="tip-body"><strong>Sé tú</strong><span>Acce­sorios, peinados, gestos — todo cuenta</span></div>
          </div>
          <div className="tip">
            <div className="tip-emoji">🔒</div>
            <div className="tip-body"><strong>Privado</strong><span>Tu foto no se guarda en ningún servidor</span></div>
          </div>
        </aside>
      </div>
    </section>
  );
}

// ====== ANALYZE ======
const ANALYZE_MESSAGES = [
  "Detectando tu rostro…",
  "Analizando tus rasgos faciales…",
  "Midiendo color de ojos y pelo…",
  "Eligiendo tu animalito…",
  "Generando tu mascota…",
  "Pintando el pelaje…",
  "Integrando la magia…",
  "Dando los últimos toques…",
];

const ANALYZE_STEPS = [
  { id: "face", label: "Detectando rostro", ico: "🔍" },
  { id: "traits", label: "Analizando rasgos", ico: "🧬" },
  { id: "species", label: "Eligiendo especie", ico: "🦊" },
  { id: "gen", label: "Generando imagen", ico: "🎨" },
];

function AnalyzeScreen({ photo, error, onRetry }) {
  const [progress, setProgress] = useState(0);
  const [msgIdx, setMsgIdx] = useState(0);

  useEffect(() => {
    // Indeterminado: avanza hacia ~92% mientras la peticion al backend esta en curso.
    let p = 0;
    const tick = setInterval(() => {
      p = Math.min(0.92, p + Math.random() * 0.04);
      setProgress(p);
    }, 120);
    const cycle = setInterval(() => setMsgIdx((i) => (i + 1) % ANALYZE_MESSAGES.length), 1100);
    return () => { clearInterval(tick); clearInterval(cycle); };
  }, []);

  const stepIdx = Math.min(ANALYZE_STEPS.length - 1, Math.floor(progress * ANALYZE_STEPS.length));

  if (error) {
    return (
      <section className="screen analyze screen-enter" data-screen-label="03 Analizando">
        <div style={{ textAlign: "center" }}>
          <div className="big-emoji" style={{ fontSize: 64 }}>😿</div>
          <h2 className="analyze-title">No pudimos crear tu mascota</h2>
          <div className="analyze-msg"><span>{error}</span></div>
          <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={onRetry}>↻ Intentar de nuevo</button>
        </div>
      </section>
    );
  }

  return (
    <section className="screen analyze screen-enter" data-screen-label="03 Analizando">
      <div>
        <div className="analyze-orb-wrap">
          <div className="aura-ring"/>
          <div className="analyze-orb">
            {photo ? <img src={photo} alt="" /> : <div className="rv-orb-fill"/>}
            <div className="orb-shimmer"/>
            <div className="orb-vignette"/>
          </div>
          <div className="orbit-dust">
            {Array.from({ length: 10 }).map((_, i) => (
              <span key={i} style={{ "--a": `${(360 / 10) * i}deg`, "--s": `${5 + (i % 4)}s` }}/>
            ))}
          </div>
          <Sparkle x={4} y={12} size={22} delay={0}/>
          <Sparkle x={90} y={16} size={18} delay={.6} color="#EBB24A"/>
          <Sparkle x={2} y={82} size={20} delay={1} color="#8FB99A"/>
          <Sparkle x={92} y={80} size={24} delay={.3} color="#D9824F"/>
        </div>
        <h2 className="analyze-title">Creando tu mascota…</h2>
        <div className="analyze-msg">
          <span key={msgIdx}>{ANALYZE_MESSAGES[msgIdx]}</span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress * 100}%` }}/>
        </div>
        <div className="analyze-steps">
          {ANALYZE_STEPS.map((s, i) => (
            <div key={s.id} className={`astep ${i === stepIdx ? "active" : ""} ${i < stepIdx ? "done" : ""}`}>
              <span className="ico">{i < stepIdx ? "✓" : s.ico}</span>
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ====== REVEAL ======
const TRAIT_BANK = [
  { label: "Energía", values: ["Caótica", "Tierna", "Sabia", "Juguetona", "Rebelde", "Soñadora"] },
  { label: "Snack favorito", values: ["Pancakes", "Galleta", "Fresas", "Sushi", "Tacos", "Mango", "Helado"] },
  { label: "Superpoder", values: ["Teletransporte", "Abrazos cósmicos", "Detectar mentiras", "Bailar K-pop", "Hablar con plantas", "Predecir el clima"] },
  { label: "Hobby", values: ["Yoga", "Karaoke", "Pintura", "Coleccionar piedras", "Memes", "Astronomía"] },
  { label: "Vibe musical", values: ["Lo-fi", "Reggaetón", "Indie", "Cumbia", "Disco", "Clásica"] },
];

function rng(seed) {
  // simple seeded random
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}
function generateTraits(seed = Date.now()) {
  const r = rng(seed);
  return TRAIT_BANK.slice(0, 3).map(t => ({
    label: t.label,
    value: t.values[Math.floor(r() * t.values.length)],
    color: ["#FF8A65", "#9D5BFF", "#4FE3C1", "#FFD93D", "#6CC7FF"][Math.floor(r() * 5)],
  }));
}

function RevealScreen({ pet, photo, onSave, onRetry, saved }) {
  const Pet = pet.Comp;
  const [phase, setPhase] = useState("materialize"); // materialize -> born -> details
  useEffect(() => {
    setPhase("materialize");
    const t1 = setTimeout(() => setPhase("born"), 1250);
    const t2 = setTimeout(() => setPhase("details"), 2000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [pet.id]);
  const born = phase === "born" || phase === "details";

  return (
    <section className={`screen reveal screen-enter rv-${phase}`} data-screen-label="04 Reveal">
      {born && <Confetti count={60}/>}
      <div>
        <h1 className="reveal-burst">¡TA-DA!</h1>
        <div className="pet-stage">
          <div className="ring"/>
          {/* orbe con la foto que se disuelve */}
          <div className="rv-orb">
            {photo ? <img src={photo} alt=""/> : <div className="rv-orb-fill"/>}
          </div>
          {/* destello + onda */}
          <div className="rv-flash"/>
          <div className="rv-halo"/>
          {/* polvo de hadas que converge */}
          <div className="rv-dust">
            {Array.from({ length: 14 }).map((_, i) => (
              <span key={i} style={{ "--a": `${(360 / 14) * i}deg`, "--d": `${.2 + (i % 5) * .08}s` }}/>
            ))}
          </div>
          <Sparkle x={10} y={15} size={24} delay={0}/>
          <Sparkle x={88} y={20} size={22} delay={.4} color="#EBB24A"/>
          <Sparkle x={5} y={75} size={20} delay={.8} color="#8FB99A"/>
          <Sparkle x={90} y={80} size={26} delay={1.2} color="#D9824F"/>
          <div className="pet-born"><div className="pet">
            {pet.image
              ? <img src={pet.image} alt={pet.name} style={{ width: "100%", height: "100%", objectFit: "contain" }}/>
              : <Pet tint={pet.tint} accent={pet.accent}/>}
          </div></div>
        </div>
        <h2 className="pet-name">{pet.animal || pet.name}</h2>
        <p className="pet-species">✦ {pet.element || pet.species.name} ✦</p>
        <div className="traits">
          {pet.traits.map((t, i) => {
            const icos = ["⚡", "🍰", "🌟", "🎨", "🎵"];
            return (
              <div key={i} className="trait" style={{ transitionDelay: `${.18 + i * .08}s` }}>
                <div className="icoBox" style={{ background: `${t.color}30`, color: t.color }}>{icos[i % icos.length]}</div>
                <div>
                  <div className="label">{t.label}</div>
                  <div className="value">{t.value}</div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="reveal-actions">
          {!saved && <button className="btn btn-primary" onClick={onSave}>💾 Guardar en mi colección</button>}
          {saved && <span className="btn btn-secondary" style={{ background: "var(--mint)", color: "white" }}>✓ Guardada</span>}
          <button className="btn btn-secondary" onClick={() => navigator.share?.({ title: pet.name }) || alert("¡Mascota compartida!")}>📤 Compartir</button>
          <button className="btn btn-ghost" onClick={onRetry}>↻ Otra mascota</button>
        </div>
      </div>
    </section>
  );
}

// ====== GALLERY ======
function GalleryScreen({ pets, onNew, onOpen, onDelete }) {
  return (
    <section className="screen gallery screen-enter" data-screen-label="05 Galería">
      <div className="gallery-head">
        <div>
          <h2>Mi colección</h2>
          <p>{pets.length} {pets.length === 1 ? "mascota" : "mascotas"} en tu mundo</p>
        </div>
        <button className="btn btn-primary" onClick={onNew}>+ Nueva mascota</button>
      </div>
      {pets.length === 0 ? (
        <div className="gallery-empty">
          <div className="ico">🌱</div>
          <h3>Aún no hay mascotas aquí</h3>
          <p>Crea tu primera mini-criatura y empieza tu colección</p>
          <button className="btn btn-primary btn-xl" onClick={onNew}>Crear mi primera mascota</button>
        </div>
      ) : (
        <div className="gallery-grid">
          {pets.map(p => {
            const sp = petByKey(p.speciesKey);
            const Pet = sp.Comp;
            return (
              <div key={p.id} className="pet-card" style={{ "--card-tint": sp.accent, position: "relative" }} onClick={() => onOpen(p)}>
                <button
                  className="pet-del"
                  title="Borrar mascota"
                  aria-label="Borrar mascota"
                  onClick={(e) => { e.stopPropagation(); onDelete?.(p.id); }}
                  style={{
                    position: "absolute", top: 10, right: 10, zIndex: 3,
                    width: 36, height: 36, borderRadius: 999, border: "none", cursor: "pointer",
                    background: "rgba(255,255,255,.9)", color: "#E5379B",
                    boxShadow: "0 2px 10px rgba(30,42,90,.18)", fontSize: 17, lineHeight: 1,
                    display: "grid", placeItems: "center", padding: 0,
                  }}
                >🗑️</button>
                <div className="frame">
                  {p.image
                    ? <img src={p.image} alt={p.name} style={{ width: "100%", height: "100%", objectFit: "cover" }}/>
                    : <Pet tint={p.tint || sp.tint} accent={p.accent || sp.accent}/>}
                </div>
                <div className="meta">
                  <h3 className="name">{p.animal || p.name}</h3>
                  <div className="species">{p.element || sp.name}</div>
                  <div className="date">{p.date}</div>
                </div>
              </div>
            );
          })}
          <div className="pet-card empty" onClick={onNew}>
            <div>
              <div className="plus">+</div>
              <div className="label">Crear otra</div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export { IntroScreen, CaptureScreen, AnalyzeScreen, RevealScreen, GalleryScreen, generateTraits, Confetti };
