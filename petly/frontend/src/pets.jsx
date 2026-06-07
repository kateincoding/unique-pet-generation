/* eslint-disable */
// Nuzzle — soft chibi-furry pet system
// 240x240 viewBox, sitting chibi proportions (big head, small body, fluffy).
// Volume comes from radial gradients; fluff from layered tuft paths;
// eyes are big & glossy. Idle "breathing" lives in CSS (.pet / .hero-pet).

import React, { useId } from 'react';

// ---------- shared animation keyframes ----------
const petAnims = `
@keyframes blink {
  0%, 91%, 100% { transform: scaleY(1); }
  94%, 97% { transform: scaleY(.08); }
}
@keyframes ear-twitch {
  0%, 86%, 100% { transform: rotate(0); }
  90% { transform: rotate(-7deg); }
  95% { transform: rotate(4deg); }
}
@keyframes ear-sway {
  0%, 100% { transform: rotate(-3deg); }
  50% { transform: rotate(3deg); }
}
@keyframes tail-sway {
  0%, 100% { transform: rotate(-7deg); }
  50% { transform: rotate(7deg); }
}
@keyframes tail-wag {
  0%, 100% { transform: rotate(-13deg); }
  50% { transform: rotate(13deg); }
}
@keyframes twinkle {
  0%, 100% { opacity: .25; transform: scale(.6) rotate(0deg); }
  50% { opacity: 1; transform: scale(1.15) rotate(35deg); }
}
@keyframes sheen-blush {
  0%, 100% { opacity: .5; }
  50% { opacity: .8; }
}
`;
if (typeof document !== "undefined" && !document.getElementById("__pet_anims")) {
  const s = document.createElement("style");
  s.id = "__pet_anims";
  s.textContent = petAnims;
  document.head.appendChild(s);
}

// ---------- gradient defs (per-instance, unique ids) ----------
function PetDefs({ uid, c }) {
  return (
    <defs>
      <radialGradient id={`body-${uid}`} cx=".5" cy=".32" r=".85">
        <stop offset="0" stopColor={c.furLight} />
        <stop offset=".55" stopColor={c.fur} />
        <stop offset="1" stopColor={c.furDark} />
      </radialGradient>
      <radialGradient id={`head-${uid}`} cx=".42" cy=".3" r=".85">
        <stop offset="0" stopColor={c.furLight} />
        <stop offset=".55" stopColor={c.fur} />
        <stop offset="1" stopColor={c.furDark} />
      </radialGradient>
      <radialGradient id={`belly-${uid}`} cx=".5" cy=".35" r=".75">
        <stop offset="0" stopColor={c.bellyLight} />
        <stop offset="1" stopColor={c.belly} />
      </radialGradient>
      <linearGradient id={`iris-${uid}`} x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stopColor="#1c130c" />
        <stop offset=".4" stopColor={c.irisDark} />
        <stop offset="1" stopColor={c.iris} />
      </linearGradient>
      <radialGradient id={`tl-${uid}`} cx=".4" cy=".18" r=".55">
        <stop offset="0" stopColor="#ffffff" stopOpacity=".5" />
        <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
      </radialGradient>
      <radialGradient id={`ao-${uid}`} cx=".5" cy=".95" r=".55">
        <stop offset="0" stopColor={c.furDark} stopOpacity=".5" />
        <stop offset="1" stopColor={c.furDark} stopOpacity="0" />
      </radialGradient>
    </defs>
  );
}

// ---------- big glossy eye (with magical star glint) ----------
function Eye({ cx, cy, rx = 13.5, ry = 17, uid, delay = 0 }) {
  const sx = cx - rx * 0.28, sy = cy - ry * 0.4, s = rx * 0.62;
  const star = `M${sx} ${sy - s} Q${sx + s * 0.2} ${sy - s * 0.2} ${sx + s} ${sy} Q${sx + s * 0.2} ${sy + s * 0.2} ${sx} ${sy + s} Q${sx - s * 0.2} ${sy + s * 0.2} ${sx - s} ${sy} Q${sx - s * 0.2} ${sy - s * 0.2} ${sx} ${sy - s} Z`;
  return (
    <g style={{ transformBox: "view-box", transformOrigin: `${cx}px ${cy}px`, animation: `blink ${(4 + delay).toFixed(1)}s ease-in-out infinite` }}>
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill={`url(#iris-${uid})`} />
      <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke="#1b120b" strokeWidth="1.4" opacity=".2" />
      {/* sheen band */}
      <ellipse cx={cx + rx * 0.1} cy={cy + ry * 0.15} rx={rx * 0.55} ry={ry * 0.62} fill="#fff" opacity=".08" />
      {/* big sparkly star glint */}
      <path d={star} fill="#fff" opacity=".98" />
      {/* secondary glints */}
      <circle cx={cx + rx * 0.42} cy={cy + ry * 0.46} r={Math.max(2.2, rx * 0.22)} fill="#fff" opacity=".85" />
      <circle cx={cx + rx * 0.5} cy={cy - ry * 0.3} r={Math.max(1.2, rx * 0.1)} fill="#fff" opacity=".6" />
    </g>
  );
}

// ---------- magical twinkle star ----------
function MagicStar({ x, y, s = 8, delay = 0 }) {
  const d = `M${x} ${y - s} Q${x + s * 0.22} ${y - s * 0.22} ${x + s} ${y} Q${x + s * 0.22} ${y + s * 0.22} ${x} ${y + s} Q${x - s * 0.22} ${y + s * 0.22} ${x - s} ${y} Q${x - s * 0.22} ${y - s * 0.22} ${x} ${y - s} Z`;
  return (
    <path
      d={d}
      fill="#fff"
      style={{ transformBox: "view-box", transformOrigin: `${x}px ${y}px`, animation: `twinkle 2.6s ease-in-out infinite`, animationDelay: `${delay}s` }}
    />
  );
}

function defaultFace(c) {
  return (
    <g>
      <ellipse cx="120" cy="120" rx="6.5" ry="5" fill={c.nose} />
      <ellipse cx="117.5" cy="118" rx="2" ry="1.4" fill="#fff" opacity=".5" />
      <path d="M120 124.5 L120 130" stroke={c.line} strokeWidth="1.8" strokeLinecap="round" />
      <path d="M120 130 Q112 137.5 106 132 M120 130 Q128 137.5 134 132" stroke={c.line} strokeWidth="2.4" fill="none" strokeLinecap="round" />
    </g>
  );
}

// ---------- shared chibi base ----------
function ChibiPet({ c, parts = {} }) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const { tail, earsBack, earsTop, headMarks, faceMarks, frontExtra, eyeCfg = {} } = parts;
  const e = { lx: 94, rx2: 146, cy: 106, rx: 14, ry: 17, ...eyeCfg };
  return (
    <svg viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
      <PetDefs uid={uid} c={c} />

      {tail && tail(c, uid)}
      {earsBack && earsBack(c, uid)}

      {/* ---- body ---- */}
      <ellipse cx="120" cy="178" rx="55" ry="46" fill={`url(#body-${uid})`} />
      {/* belly */}
      <ellipse cx="120" cy="186" rx="36" ry="32" fill={`url(#belly-${uid})`} />
      {/* body ambient occlusion */}
      <ellipse cx="120" cy="204" rx="44" ry="24" fill={`url(#ao-${uid})`} />
      {/* front paws */}
      <ellipse cx="98" cy="212" rx="16" ry="12" fill={c.paw || c.fur} />
      <ellipse cx="142" cy="212" rx="16" ry="12" fill={c.paw || c.fur} />
      <ellipse cx="98" cy="214" rx="9" ry="5.5" fill={c.bellyLight} opacity=".7" />
      <ellipse cx="142" cy="214" rx="9" ry="5.5" fill={c.bellyLight} opacity=".7" />

      {/* ---- head ---- */}
      <circle cx="120" cy="98" r="63" fill={`url(#head-${uid})`} />
      {headMarks && headMarks(c, uid)}
      {earsTop && earsTop(c, uid)}

      {/* eyes */}
      <Eye cx={e.lx} cy={e.cy} rx={e.rx} ry={e.ry} uid={uid} delay={0} />
      <Eye cx={e.rx2} cy={e.cy} rx={e.rx} ry={e.ry} uid={uid} delay={0.4} />
      {/* soft brows / cheeks blush */}
      <g style={{ animation: "sheen-blush 3.2s ease-in-out infinite" }}>
        <ellipse cx="72" cy="126" rx="13" ry="8" fill={c.blush} opacity=".55" />
        <ellipse cx="168" cy="126" rx="13" ry="8" fill={c.blush} opacity=".55" />
        <ellipse cx="68" cy="123" rx="4" ry="2.4" fill="#fff" opacity=".5" />
        <ellipse cx="164" cy="123" rx="4" ry="2.4" fill="#fff" opacity=".5" />
      </g>

      {/* muzzle + face */}
      {faceMarks ? faceMarks(c, uid) : defaultFace(c)}

      {/* top light */}
      <circle cx="120" cy="98" r="63" fill={`url(#tl-${uid})`} style={{ pointerEvents: "none" }} />

      {/* ---- magical twinkles ---- */}
      <MagicStar x={48} y={64} s={9} delay={0} />
      <MagicStar x={196} y={78} s={7} delay={1.1} />
      <MagicStar x={186} y={150} s={6} delay={.6} />
      <MagicStar x={56} y={166} s={5} delay={1.6} />

      {frontExtra && frontExtra(c, uid)}
    </svg>
  );
}

// ---------- fluff helpers ----------
function cheekFluff(c, side) {
  // side: -1 left, +1 right ; pointed white tufts
  const x = side < 0 ? 60 : 180;
  const s = side;
  return (
    <path
      d={`M${x} 112
          q${-8 * s} 4 ${-14 * s} 14
          q${4 * s} -4 ${9 * s} -2
          q${-8 * s} 8 ${-12 * s} 22
          q${5 * s} -6 ${10 * s} -3
          q${-5 * s} 8 ${-6 * s} 18
          q${10 * s} -10 ${18 * s} -22
          q${4 * s} -18 ${1 * s} -38 Z`}
      fill={c.bellyLight}
      opacity=".95"
    />
  );
}

// ===================== DOG (hero) =====================
const PetDog = (props) => (
  <ChibiPet
    c={{
      fur: "#E6B36A", furLight: "#F6D49A", furDark: "#C68F42",
      belly: "#F3DDB6", bellyLight: "#FCF1DC",
      inner: "#B97C3E", nose: "#3A2A20", line: "#5A4129",
      iris: "#8A5A2C", irisDark: "#3a2615", blush: "#E8927A",
      paw: "#D9A858",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      tail: (c) => (
        <g style={{ transformBox: "view-box", transformOrigin: "176px 168px", animation: "tail-wag 1s ease-in-out infinite" }}>
          <path d="M172 168 Q214 150 202 112 Q198 100 188 104 Q200 130 168 156 Z" fill={c.furDark} />
          <ellipse cx="194" cy="108" rx="11" ry="12" fill={c.fur} />
        </g>
      ),
      earsBack: (c) => (
        <g>
          <g style={{ transformBox: "view-box", transformOrigin: "82px 66px", animation: "ear-sway 4s ease-in-out infinite" }}>
            <path d="M84 64 Q40 64 46 122 Q52 156 88 142 Q66 102 90 70 Z" fill={c.furDark} />
            <path d="M80 78 Q56 84 58 120 Q64 142 82 134" fill={c.inner} opacity=".5" />
          </g>
          <g style={{ transformBox: "view-box", transformOrigin: "158px 66px", animation: "ear-sway 4.3s ease-in-out infinite" }}>
            <path d="M156 64 Q200 64 194 122 Q188 156 152 142 Q174 102 150 70 Z" fill={c.furDark} />
            <path d="M160 78 Q184 84 182 120 Q176 142 158 134" fill={c.inner} opacity=".5" />
          </g>
        </g>
      ),
      headMarks: (c) => (
        <path d="M118 38 Q110 26 122 22 Q119 32 130 36 Q124 40 123 46 Z" fill={c.furLight} />
      ),
      faceMarks: (c) => (
        <g>
          <ellipse cx="120" cy="128" rx="26" ry="20" fill={c.bellyLight} />
          <ellipse cx="120" cy="120" rx="8" ry="6" fill={c.nose} />
          <ellipse cx="116.5" cy="117.5" rx="2.6" ry="1.8" fill="#fff" opacity=".55" />
          <path d="M120 126 L120 132" stroke={c.line} strokeWidth="2" strokeLinecap="round" />
          <path d="M120 132 Q111 140 104 134 M120 132 Q129 140 136 134" stroke={c.line} strokeWidth="2.6" fill="none" strokeLinecap="round" />
          <path d="M111 136 Q120 146 129 136 Q120 142 111 136 Z" fill={c.blush} opacity=".85" />
        </g>
      ),
    }}
  />
);

// ===================== FOX (zorro de fuego) =====================
const PetFox = (props) => (
  <ChibiPet
    c={{
      fur: "#D9824F", furLight: "#EBA876", furDark: "#B5612F",
      belly: "#F4E7D6", bellyLight: "#FFFAF2",
      inner: "#E8C9A8", nose: "#2C2420", line: "#4A352A",
      iris: "#8E8A3A", irisDark: "#4a4418", blush: "#E58E6E",
      paw: "#3A2A22",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      tail: (c) => (
        <g style={{ transformBox: "view-box", transformOrigin: "72px 172px", animation: "tail-sway 3.4s ease-in-out infinite" }}>
          <path d="M78 178 Q24 180 26 134 Q28 102 64 110 Q40 122 44 152 Q50 174 80 164 Z" fill={c.fur} />
          <path d="M30 124 Q18 116 22 136 Q30 124 44 132 Q34 122 30 124 Z" fill={c.bellyLight} />
        </g>
      ),
      earsTop: (c) => (
        <g>
          <g style={{ transformBox: "view-box", transformOrigin: "82px 56px", animation: "ear-twitch 5s ease-in-out infinite" }}>
            <path d="M72 60 L58 10 L106 46 Z" fill={c.fur} />
            <path d="M76 54 L68 24 L98 46 Z" fill={c.inner} />
            <path d="M74 50 L70 28 L84 44 Z" fill={c.bellyLight} opacity=".9" />
          </g>
          <g style={{ transformBox: "view-box", transformOrigin: "158px 56px", animation: "ear-twitch 5.3s ease-in-out infinite" }}>
            <path d="M168 60 L182 10 L134 46 Z" fill={c.fur} />
            <path d="M164 54 L172 24 L142 46 Z" fill={c.inner} />
            <path d="M166 50 L170 28 L156 44 Z" fill={c.bellyLight} opacity=".9" />
          </g>
        </g>
      ),
      headMarks: (c) => (
        <g>
          {cheekFluff(c, -1)}
          {cheekFluff(c, 1)}
          <path d="M118 40 Q108 24 123 20 Q119 32 132 36 Q125 41 124 48 Z" fill={c.furLight} />
          {/* white facial blaze */}
          <path d="M120 78 Q104 100 112 134 Q120 116 120 116 Q120 116 128 134 Q136 100 120 78 Z" fill={c.bellyLight} opacity=".92" />
        </g>
      ),
      faceMarks: (c) => (
        <g>
          <ellipse cx="120" cy="128" rx="22" ry="16" fill={c.bellyLight} opacity=".95" />
          <path d="M114 124 L126 124 L120 132 Z" fill={c.nose} />
          <ellipse cx="116.5" cy="123" rx="2" ry="1.4" fill="#fff" opacity=".5" />
          <path d="M120 132 L120 137" stroke={c.line} strokeWidth="1.8" strokeLinecap="round" />
          <path d="M120 137 Q112 143 107 138 M120 137 Q128 143 133 138" stroke={c.line} strokeWidth="2.3" fill="none" strokeLinecap="round" />
        </g>
      ),
    }}
  />
);

// ===================== CAT =====================
const PetCat = (props) => (
  <ChibiPet
    c={{
      fur: "#EBC59B", furLight: "#F8E0C2", furDark: "#D29F6C",
      belly: "#FBEFDE", bellyLight: "#FFFBF4",
      inner: "#E8A9A0", nose: "#C77B6E", line: "#5A4129",
      iris: "#6FA06B", irisDark: "#2f4a2c", blush: "#E89A88",
      paw: "#E0B286",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      tail: (c) => (
        <g style={{ transformBox: "view-box", transformOrigin: "70px 170px", animation: "tail-sway 2.6s ease-in-out infinite" }}>
          <path d="M74 172 Q28 168 30 128 Q31 108 50 112" fill="none" stroke={c.furDark} strokeWidth="20" strokeLinecap="round" />
          <path d="M50 112 Q40 110 42 122" fill="none" stroke={c.bellyLight} strokeWidth="14" strokeLinecap="round" />
        </g>
      ),
      earsTop: (c) => (
        <g>
          <g style={{ transformBox: "view-box", transformOrigin: "82px 58px", animation: "ear-twitch 4.6s ease-in-out infinite" }}>
            <path d="M74 64 L66 22 L106 52 Z" fill={c.fur} />
            <path d="M80 60 L74 34 L98 52 Z" fill={c.inner} opacity=".8" />
          </g>
          <g style={{ transformBox: "view-box", transformOrigin: "158px 58px", animation: "ear-twitch 4.9s ease-in-out infinite" }}>
            <path d="M166 64 L174 22 L134 52 Z" fill={c.fur} />
            <path d="M160 60 L166 34 L142 52 Z" fill={c.inner} opacity=".8" />
          </g>
        </g>
      ),
      headMarks: (c) => (
        <g opacity=".5">
          <path d="M120 40 Q114 50 120 58 Q126 50 120 40 Z" fill={c.furDark} />
          <path d="M100 44 Q98 54 104 60 M140 44 Q142 54 136 60" stroke={c.furDark} strokeWidth="5" strokeLinecap="round" fill="none" />
        </g>
      ),
      faceMarks: (c) => (
        <g>
          <ellipse cx="120" cy="129" rx="20" ry="14" fill={c.bellyLight} opacity=".9" />
          <path d="M114 124 Q120 130 126 124 L120 129 Z" fill={c.nose} />
          <path d="M120 130 L120 135" stroke={c.line} strokeWidth="1.6" strokeLinecap="round" />
          <path d="M120 135 Q113 141 108 137 M120 135 Q127 141 132 137" stroke={c.line} strokeWidth="2.2" fill="none" strokeLinecap="round" />
          <g stroke={c.line} strokeWidth="1.4" strokeLinecap="round" opacity=".55">
            <path d="M92 126 L66 122" /><path d="M92 131 L66 134" />
            <path d="M148 126 L174 122" /><path d="M148 131 L174 134" />
          </g>
        </g>
      ),
    }}
  />
);

// ===================== BUNNY =====================
const PetBunny = (props) => (
  <ChibiPet
    c={{
      fur: "#F0DFC9", furLight: "#FCF4E8", furDark: "#DCC3A4",
      belly: "#FFFBF5", bellyLight: "#FFFFFF",
      inner: "#EFB6B2", nose: "#D98C90", line: "#7A5E50",
      iris: "#9A6B5A", irisDark: "#43281f", blush: "#F0A8A4",
      paw: "#E7D3BB",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      tail: (c) => (
        <circle cx="58" cy="186" r="16" fill={c.bellyLight} />
      ),
      earsTop: (c) => (
        <g>
          <g style={{ transformBox: "view-box", transformOrigin: "100px 64px", animation: "ear-sway 3.6s ease-in-out infinite" }}>
            <path d="M100 66 Q86 18 100 8 Q116 20 110 64 Z" fill={c.fur} />
            <path d="M100 60 Q92 26 100 18 Q108 28 104 60 Z" fill={c.inner} opacity=".75" />
          </g>
          <g style={{ transformBox: "view-box", transformOrigin: "140px 64px", animation: "ear-sway 3.9s ease-in-out infinite" }}>
            <path d="M140 66 Q126 18 140 8 Q156 20 150 64 Z" fill={c.fur} />
            <path d="M140 60 Q132 26 140 18 Q148 28 144 60 Z" fill={c.inner} opacity=".75" />
          </g>
        </g>
      ),
      faceMarks: (c) => (
        <g>
          <path d="M115 124 Q120 130 125 124 L120 128 Z" fill={c.nose} />
          <path d="M120 128 L120 133" stroke={c.line} strokeWidth="1.6" strokeLinecap="round" />
          <path d="M120 133 Q113 139 108 135 M120 133 Q127 139 132 135" stroke={c.line} strokeWidth="2.2" fill="none" strokeLinecap="round" />
          <rect x="116.5" y="135" width="3.2" height="6" rx="1.2" fill="#fff" stroke={c.line} strokeWidth=".5" />
          <rect x="120.3" y="135" width="3.2" height="6" rx="1.2" fill="#fff" stroke={c.line} strokeWidth=".5" />
        </g>
      ),
    }}
  />
);

// ===================== HAMSTER =====================
const PetHamster = (props) => (
  <ChibiPet
    c={{
      fur: "#EEC489", furLight: "#FAE2BE", furDark: "#D9A55E",
      belly: "#FCF1DD", bellyLight: "#FFFBF3",
      inner: "#EFB0A6", nose: "#3A2A20", line: "#5A4129",
      iris: "#7A4F28", irisDark: "#33200f", blush: "#EE9A86",
      paw: "#E6B675",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      earsTop: (c) => (
        <g>
          <circle cx="84" cy="56" r="15" fill={c.furDark} />
          <circle cx="84" cy="57" r="8" fill={c.inner} opacity=".8" />
          <circle cx="156" cy="56" r="15" fill={c.furDark} />
          <circle cx="156" cy="57" r="8" fill={c.inner} opacity=".8" />
        </g>
      ),
      headMarks: (c) => (
        <ellipse cx="120" cy="118" rx="42" ry="30" fill={c.bellyLight} opacity=".55" />
      ),
      faceMarks: (c) => (
        <g>
          <ellipse cx="120" cy="124" rx="5" ry="4" fill={c.nose} />
          <path d="M120 128 L120 133" stroke={c.line} strokeWidth="1.6" strokeLinecap="round" />
          <path d="M120 133 Q114 138 110 135 M120 133 Q126 138 130 135" stroke={c.line} strokeWidth="2.1" fill="none" strokeLinecap="round" />
          <rect x="117" y="135" width="2.6" height="4.5" rx="1" fill="#fff" stroke={c.line} strokeWidth=".4" />
          <rect x="120.4" y="135" width="2.6" height="4.5" rx="1" fill="#fff" stroke={c.line} strokeWidth=".4" />
        </g>
      ),
    }}
  />
);

// ===================== RED PANDA =====================
const PetPanda = (props) => (
  <ChibiPet
    c={{
      fur: "#C56A3E", furLight: "#DE8C5C", furDark: "#A0512A",
      belly: "#F2E2D0", bellyLight: "#FBF1E6",
      inner: "#2E1E18", nose: "#241712", line: "#3A241B",
      iris: "#5A3A22", irisDark: "#24140b", blush: "#D98A6E",
      paw: "#3A241B",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      tail: (c) => (
        <g style={{ transformBox: "view-box", transformOrigin: "70px 172px", animation: "tail-sway 3s ease-in-out infinite" }}>
          <path d="M76 176 Q26 176 28 134 Q29 112 52 116" fill="none" stroke={c.fur} strokeWidth="22" strokeLinecap="round" />
          <path d="M40 122 Q32 130 30 142" stroke={c.furDark} strokeWidth="22" strokeLinecap="round" opacity=".7" fill="none" />
          <path d="M58 116 Q48 116 44 124" stroke={c.bellyLight} strokeWidth="20" strokeLinecap="round" fill="none" />
        </g>
      ),
      earsTop: (c) => (
        <g>
          <circle cx="80" cy="52" r="17" fill={c.furDark} />
          <circle cx="80" cy="53" r="9" fill={c.bellyLight} opacity=".85" />
          <circle cx="160" cy="52" r="17" fill={c.furDark} />
          <circle cx="160" cy="53" r="9" fill={c.bellyLight} opacity=".85" />
        </g>
      ),
      headMarks: (c) => (
        <g>
          {/* white face mask */}
          <path d="M120 64 Q150 70 156 104 Q150 128 120 132 Q90 128 84 104 Q90 70 120 64 Z" fill={c.bellyLight} opacity=".95" />
          {/* rusty tear marks */}
          <path d="M95 108 Q88 120 92 132 Q98 124 100 114 Z" fill={c.fur} opacity=".8" />
          <path d="M145 108 Q152 120 148 132 Q142 124 140 114 Z" fill={c.fur} opacity=".8" />
          {/* brow patches */}
          <path d="M118 50 Q108 38 122 34 Q119 44 130 48 Z" fill={c.fur} />
        </g>
      ),
      faceMarks: (c) => (
        <g>
          <ellipse cx="120" cy="125" rx="6" ry="4.5" fill={c.nose} />
          <ellipse cx="117" cy="123" rx="1.8" ry="1.2" fill="#fff" opacity=".5" />
          <path d="M120 129 L120 134" stroke={c.line} strokeWidth="1.7" strokeLinecap="round" />
          <path d="M120 134 Q113 140 108 136 M120 134 Q127 140 132 136" stroke={c.line} strokeWidth="2.2" fill="none" strokeLinecap="round" />
        </g>
      ),
    }}
  />
);

// ===================== UNICORN (magical hero) =====================
const PetUnicorn = (props) => (
  <ChibiPet
    c={{
      fur: "#FBF2FA", furLight: "#FFFFFF", furDark: "#EAD7EC",
      belly: "#FFFBFE", bellyLight: "#FFFFFF",
      inner: "#F2C6E0", nose: "#C98BBE", line: "#8E6A93",
      iris: "#A06CD0", irisDark: "#553579", blush: "#F4A6CE",
      paw: "#F0E2F0",
      ...((props && props.tint) ? { fur: props.tint } : {}),
    }}
    parts={{
      eyeCfg: { lx: 96, rx2: 144, cy: 108, rx: 15, ry: 18.5 },
      earsBack: (c, uid) => (
        <g>
          <linearGradient id={`mane-${uid}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#FF9ECB" />
            <stop offset=".5" stopColor="#C49BE8" />
            <stop offset="1" stopColor="#8FC7F0" />
          </linearGradient>
          {/* pastel feathered wings */}
          <g style={{ transformBox: "view-box", transformOrigin: "78px 168px", animation: "ear-sway 3.6s ease-in-out infinite" }}>
            <path d="M86 168 Q42 138 30 166 Q48 164 54 178 Q36 176 36 198 Q56 184 68 188 Q60 198 64 210 Q80 192 92 186 Z" fill="#FFFFFF" opacity=".96" />
            <path d="M86 168 Q42 138 30 166 Q48 164 54 178 Q36 176 36 198 Q56 184 68 188 Q60 198 64 210 Q80 192 92 186 Z" fill={`url(#mane-${uid})`} opacity=".18" />
          </g>
          <g style={{ transformBox: "view-box", transformOrigin: "162px 168px", animation: "ear-sway 3.9s ease-in-out infinite" }}>
            <path d="M154 168 Q198 138 210 166 Q192 164 186 178 Q204 176 204 198 Q184 184 172 188 Q180 198 176 210 Q160 192 148 186 Z" fill="#FFFFFF" opacity=".96" />
            <path d="M154 168 Q198 138 210 166 Q192 164 186 178 Q204 176 204 198 Q184 184 172 188 Q180 198 176 210 Q160 192 148 186 Z" fill={`url(#mane-${uid})`} opacity=".18" />
          </g>
          {/* back mane locks */}
          <path d="M72 70 Q42 98 56 142 Q66 120 80 116 Q58 100 86 74 Z" fill={`url(#mane-${uid})`} />
          <path d="M168 70 Q198 98 184 142 Q174 120 160 116 Q182 100 154 74 Z" fill={`url(#mane-${uid})`} />
        </g>
      ),
      earsTop: (c, uid) => (
        <g>
          <linearGradient id={`horn-${uid}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#FFEDB0" />
            <stop offset="1" stopColor="#E6AC3E" />
          </linearGradient>
          {/* ears */}
          <path d="M88 56 L78 26 L110 50 Z" fill={c.fur} />
          <path d="M152 56 L162 26 L130 50 Z" fill={c.fur} />
          <path d="M90 54 L84 36 L104 50 Z" fill={c.inner} opacity=".7" />
          <path d="M150 54 L156 36 L136 50 Z" fill={c.inner} opacity=".7" />
          {/* horn */}
          <path d="M120 44 L111 47 L120 3 L129 47 Z" fill={`url(#horn-${uid})`} />
          <g stroke="#C98F2E" strokeWidth="1.5" opacity=".55" fill="none" strokeLinecap="round">
            <path d="M113 41 L127 45" />
            <path d="M114 33 L126 37" />
            <path d="M115 25 L125 28" />
            <path d="M116 17 L124 19" />
          </g>
          <circle cx="120" cy="6" r="2.6" fill="#fff" />
        </g>
      ),
      headMarks: (c, uid) => (
        <g>
          {/* forelock over forehead */}
          <path d="M100 58 Q116 42 120 62 Q124 42 140 58 Q134 80 120 72 Q106 80 100 58 Z" fill={`url(#mane-${uid})`} />
          {/* star cheek mark */}
          <path d="M152 130 Q154 135 159 137 Q154 139 152 144 Q150 139 145 137 Q150 135 152 130 Z" fill={c.blush} />
        </g>
      ),
      tail: (c, uid) => (
        <g style={{ transformBox: "view-box", transformOrigin: "170px 170px", animation: "tail-sway 3.2s ease-in-out infinite" }}>
          <path d="M166 174 Q214 160 206 114 Q200 134 184 140 Q198 152 168 164 Z" fill={`url(#mane-${uid})`} />
        </g>
      ),
    }}
  />
);

// ---------- registry ----------
const SPECIES = [
  { key: "unicorn", name: "Unicornio Estelar", emoji: "🦄", Comp: PetUnicorn, tint: "#FBF2FA", accent: "#C49BE8" },
  { key: "dog", name: "Perrito Sol", emoji: "🐶", Comp: PetDog, tint: "#E6B36A", accent: "#C68F42" },
  { key: "fox", name: "Zorro de Fuego", emoji: "🦊", Comp: PetFox, tint: "#D9824F", accent: "#B5612F" },
  { key: "cat", name: "Gato de Nube", emoji: "🐱", Comp: PetCat, tint: "#EBC59B", accent: "#D29F6C" },
  { key: "bunny", name: "Conejo Algodón", emoji: "🐰", Comp: PetBunny, tint: "#F0DFC9", accent: "#DCC3A4" },
  { key: "hamster", name: "Hámster Miel", emoji: "🐹", Comp: PetHamster, tint: "#EEC489", accent: "#D9A55E" },
  { key: "panda", name: "Panda Brasa", emoji: "🦝", Comp: PetPanda, tint: "#C56A3E", accent: "#A0512A" },
];

const NAMES = ["Mochi", "Tofu", "Galleta", "Canela", "Miel", "Brioche", "Coco", "Avena", "Maple", "Nube", "Pancake", "Bombón", "Caramelo", "Almendra", "Trufa", "Cookie"];

function petByKey(key) { return SPECIES.find(s => s.key === key) || SPECIES[0]; }
function randomPet() { return SPECIES[Math.floor(Math.random() * SPECIES.length)]; }
function randomName() { return NAMES[Math.floor(Math.random() * NAMES.length)]; }

export { SPECIES, NAMES, petByKey, randomPet, randomName, ChibiPet, PetUnicorn, PetDog, PetFox, PetCat, PetBunny, PetHamster, PetPanda };
