import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Type d'employeur | Prédiction",
    page_icon=":material/work:",
    layout="wide",
)

# ------------------------------------------------------------------
# Style
# Police unique : Playfair Display (serif à fort contraste).
# Pour en changer, modifier --font ci-dessous, ainsi que dans ORB_HTML,
# et le nom de la police dans les deux liens Google Fonts.
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');

:root {--font: 'Playfair Display', Georgia, 'Times New Roman', serif;}

/* :where() = priorité nulle, donc les icônes de Streamlit gardent leur propre police */
.stApp, .stApp :where(p, h1, h2, h3, h4, h5, h6, label, li, button, input, textarea, td, th),
.stApp [data-baseweb="select"] *, .stApp [data-testid="stSlider"] *,
.stApp [data-testid="stNumberInput"] *, .stApp [data-testid="stExpander"] summary * {
    font-family: var(--font);
    font-style: normal;
}
.sec, .chip, .kpi .n, .kpi .t, .note {font-style: normal;}
/* Taille du texte courant */
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp label p, .stApp [data-baseweb="select"] div, .stApp input, .stApp button p,
.stApp [data-testid="stExpander"] summary p, .stApp [data-testid="stAlert"] p {
    font-size: 1rem;
}
.stApp [data-testid="stMarkdownContainer"] p {line-height: 1.45;}

/* Icônes de Streamlit (boutons, sections repliables, onglets) : leur police spéciale est forcée */
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] *,
[data-testid="stExpanderIcon"], [data-testid="stExpanderIcon"] *,
[data-testid*="ToggleIcon"], [data-testid*="ToggleIcon"] *,
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
}
.block-container {padding-top: 4.5rem; max-width: 1280px;}
#MainMenu, footer {visibility: hidden;}

/* En-tête */
.head {display: flex; align-items: center; gap: .9rem; margin-bottom: 1.2rem;}
.head-icon {
    flex: none; width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
    background: linear-gradient(135deg, #1E40AF, #3B82F6); box-shadow: 0 8px 24px rgba(30,64,175,.35);
}
.head-icon svg {width: 25px; height: 25px;}
.head h1 {margin: 0; padding: 0; font-size: 2.1rem; line-height: 1.2; font-weight: 700;
    font-family: var(--font) !important; font-style: normal;}
.stApp h1, .stApp h2, .stApp h3 {font-family: var(--font) !important;}

/* Cartes en verre */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px; background: rgba(148,163,184,.07); border-color: rgba(148,163,184,.22);
}
.sec {font-size: .95rem; font-weight: 600; opacity: .75; margin: .1rem 0 .4rem; font-family: var(--font);}

/* Bouton principal en forme de pilule */
button[data-testid="stBaseButton-primary"] {
    border-radius: 999px; height: 3.2rem; font-weight: 400; font-size: 1.02rem; border: 0; cursor: pointer;
    background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
    box-shadow: 0 8px 24px rgba(30,64,175,.4); transition: transform .18s ease, box-shadow .18s ease;
}
button[data-testid="stBaseButton-primary"] p {font-size: 1.1rem; font-weight: 600;}
button[data-testid="stBaseButton-primary"]:hover {transform: translateY(-1px); box-shadow: 0 12px 28px rgba(30,64,175,.5);}
button[data-testid="stBaseButton-secondary"] {border-radius: 999px; cursor: pointer; transition: border-color .18s ease, transform .18s ease;}
button[data-testid="stBaseButton-secondary"]:hover {transform: translateY(-1px);}
button:focus-visible, summary:focus-visible {outline: 2px solid #60A5FA !important; outline-offset: 2px;}

/* Zone d'animation fixe à l'écran quand on fait défiler le formulaire */
div[data-testid="stColumn"]:has(iframe) {position: sticky; top: 4.5rem; align-self: flex-start;}
iframe {border: 0; border-radius: 20px;}

.chip {
    display: inline-block; border: 1px solid rgba(148,163,184,.35); border-radius: 999px;
    padding: .15rem .8rem; margin: .15rem .25rem .15rem 0; font-size: .95rem; font-family: var(--font);
}
.kpi {
    border: 1px solid rgba(148,163,184,.25); border-radius: 16px;
    padding: 1.1rem 1.2rem; background: rgba(148,163,184,.06); height: 100%;
}
.kpi .n {font-size: 2rem; font-weight: 700; font-family: var(--font);}
.kpi .t {opacity: .78; font-size: .95rem; font-family: var(--font);}
.note {
    border-left: 4px solid #D97706; background: rgba(217,119,6,.1);
    padding: .8rem 1rem; border-radius: 8px; font-size: .95rem; font-family: var(--font);
}

@media (prefers-reduced-motion: reduce) {
    * {animation: none !important; transition: none !important;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Modèle
# ------------------------------------------------------------------
@st.cache_resource
def load():
    return joblib.load("model.joblib")


bundle = load()
pipe, opts = bundle["pipeline"], bundle["options"]

CLASSES = {
    "Prive": {"nom": "Secteur privé", "couleur": "#3B82F6"},
    "Public": {"nom": "Secteur public", "couleur": "#10B981"},
    "Independant": {"nom": "Indépendant", "couleur": "#F59E0B"},
}

# Libellés français pour l'affichage (les valeurs envoyées au modèle ne changent pas)
FR = {
    "occupation": {
        "Adm-clerical": "Employé administratif", "Armed-Forces": "Forces armées",
        "Craft-repair": "Artisan / réparation", "Exec-managerial": "Cadre dirigeant",
        "Farming-fishing": "Agriculture / pêche", "Handlers-cleaners": "Manutention / nettoyage",
        "Machine-op-inspct": "Opérateur machine / contrôle", "Other-service": "Autres services",
        "Priv-house-serv": "Employé de maison", "Prof-specialty": "Profession spécialisée",
        "Protective-serv": "Services de protection", "Sales": "Vente",
        "Tech-support": "Support technique", "Transport-moving": "Transport",
        "Unknown": "Non renseigné",
    },
    "marital-status": {
        "Divorced": "Divorcé(e)", "Married-AF-spouse": "Marié(e), conjoint militaire",
        "Married-civ-spouse": "Marié(e), conjoint civil",
        "Married-spouse-absent": "Marié(e), conjoint absent",
        "Never-married": "Jamais marié(e)", "Separated": "Séparé(e)", "Widowed": "Veuf / veuve",
    },
    "relationship": {
        "Husband": "Époux", "Wife": "Épouse", "Own-child": "Enfant",
        "Not-in-family": "Hors famille", "Other-relative": "Autre parent",
        "Unmarried": "Non marié(e)",
    },
    "race": {
        "White": "Blanc", "Black": "Noir", "Asian-Pac-Islander": "Asiatique / îles du Pacifique",
        "Amer-Indian-Eskimo": "Amérindien / Esquimau", "Other": "Autre",
    },
    "sex": {"Female": "Femme", "Male": "Homme"},
    "income": {"<=50K": "50 000 $ ou moins par an", ">50K": "Plus de 50 000 $ par an"},
    "native-country": {"Unknown": "Non renseigné"},
}


def fr(key, value):
    return FR.get(key, {}).get(value, value)


# ------------------------------------------------------------------
# État de départ, profils prédéfinis et analyse
# ------------------------------------------------------------------
DEFAULTS = {
    "age": 35, "education-num": 10, "hours-per-week": 40,
    "capital-gain": 0, "capital-loss": 0, "sex": "Male",
    "marital-status": "Married-civ-spouse", "occupation": "Sales",
    "relationship": "Husband", "race": "White",
    "native-country": "United-States", "income": "<=50K",
}

PRESETS = {
    "Cadre dirigeant": (":material/business_center:", {
        "age": 42, "education-num": 13, "hours-per-week": 50, "capital-gain": 0,
        "capital-loss": 0, "sex": "Male", "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial", "relationship": "Husband", "race": "White",
        "native-country": "United-States", "income": ">50K",
    }),
    "Employé de bureau": (":material/description:", {
        "age": 38, "education-num": 10, "hours-per-week": 40, "capital-gain": 0,
        "capital-loss": 0, "sex": "Female", "marital-status": "Divorced",
        "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White",
        "native-country": "United-States", "income": "<=50K",
    }),
    "Artisan": (":material/construction:", {
        "age": 52, "education-num": 9, "hours-per-week": 55, "capital-gain": 0,
        "capital-loss": 0, "sex": "Male", "marital-status": "Married-civ-spouse",
        "occupation": "Craft-repair", "relationship": "Husband", "race": "White",
        "native-country": "United-States", "income": "<=50K",
    }),
}


def valid(key, value):
    """Garde une valeur seulement si elle existe dans les choix du modèle."""
    if key in opts:
        return value if value in opts[key] else opts[key][0]
    return value


def current_profile():
    return {k: st.session_state[k] for k in DEFAULTS}


def summary(p):
    return " · ".join([
        fr("sex", p["sex"]), f"{p['age']} ans", fr("occupation", p["occupation"]),
        f"{p['hours-per-week']} h/sem", f"{p['education-num']} ans d'études",
    ])


def analyse():
    """Calcule la prédiction du profil saisi et la mémorise."""
    profile = current_profile()
    row = pd.DataFrame([profile])
    proba = pd.Series(pipe.predict_proba(row)[0], index=pipe.classes_)
    st.session_state["run_counter"] = st.session_state.get("run_counter", 0) + 1
    st.session_state["result"] = {
        "id": st.session_state["run_counter"],
        "profile": profile,
        "proba": {k: float(v) for k, v in proba.items()},
        "pred": proba.idxmax(),
    }


def apply_preset(name):
    for k, v in PRESETS[name][1].items():
        st.session_state[k] = valid(k, v)
    analyse()


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, valid(k, v))


def select(label, key):
    return st.selectbox(label, opts[key], key=key, format_func=lambda v: fr(key, v))


# ------------------------------------------------------------------
# Composant animé (pilule -> bille -> sphère de points -> carte)
# ------------------------------------------------------------------
ORB_HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
@property --halo { syntax: '<color>'; inherits: true; initial-value: #3B82F6; }
:root { --bg:#0B1220; --fg:#F8FAFC; --muted:#B4C1D6; --halo:#3B82F6; --cc:#3B82F6;
  --font: 'Playfair Display', Georgia, 'Times New Roman', serif; }
* { box-sizing: border-box; margin: 0; }
html, body { height: 100%; background: transparent; color: var(--fg); font-family: var(--font); font-style: normal; }
button { font-family: var(--font); font-style: normal; }
.stage { position: relative; height: 100%; min-height: 440px; border-radius: 20px; overflow: hidden;
  background: radial-gradient(120% 90% at 50% 0%, rgba(30,64,175,.55) 0%, rgba(11,18,32,0) 62%), var(--bg);
  border: 1px solid rgba(148,163,184,.18); }
.halo { position: absolute; left: 50%; top: 38%; width: 440px; height: 440px; transform: translate(-50%,-50%);
  border-radius: 50%; pointer-events: none; opacity: .75;
  background: radial-gradient(circle, color-mix(in srgb, var(--halo) 30%, transparent) 0%, transparent 65%);
  transition: --halo .8s ease, opacity .6s ease; }
canvas { position: absolute; left: 50%; top: 38%; width: 260px; height: 260px; transform: translate(-50%,-50%); }

.status { position: absolute; left: 0; right: 0; top: calc(38% + 118px); text-align: center; padding: 0 16px;
  transition: opacity .25s ease, transform .25s ease; }
.status.hide { opacity: 0; transform: translateY(-6px); pointer-events: none; }
.s-main { font-size: 1.3rem; font-weight: 700; display: inline-flex; align-items: center; gap: .5rem; }
.dots { display: inline-flex; gap: 3px; }
.dots i { width: 4px; height: 4px; border-radius: 50%; background: currentColor; opacity: .25; animation: blink 1.2s infinite; }
.dots i:nth-child(2) { animation-delay: .2s; } .dots i:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,80%,100% { opacity: .25; } 40% { opacity: 1; } }
.s-sub { margin-top: .2rem; color: var(--muted); font-size: 1rem; font-weight: 400; max-width: 420px; margin-inline: auto; line-height: 1.35; }

/* Description de l'état initial (« Prêt à analyser ») */
.s-sub.idle {
  font-size: 1.1rem;
  font-weight: 400;
  line-height: 1.4;
  letter-spacing: 0;
  color: #C7D2E6;
  max-width: 340px;
  text-wrap: balance;
}

.steps { display: flex; justify-content: center; gap: 6px; margin-top: .8rem; }
.steps i { width: 34px; height: 4px; border-radius: 4px; background: rgba(148,163,184,.28); transition: background .3s ease; }
.steps i.on { background: #3B82F6; }
.mono { font-family: var(--font); font-weight: 600; }

.skip { position: absolute; top: 14px; right: 14px; z-index: 5; cursor: pointer; font-size: .85rem; font-weight: 600;
  color: var(--fg); background: rgba(148,163,184,.14); border: 1px solid rgba(148,163,184,.3);
  padding: .3rem .95rem; border-radius: 999px; transition: background .2s ease, border-color .2s ease; }
.skip:hover { background: rgba(148,163,184,.26); }
button:focus-visible { outline: 2px solid #60A5FA; outline-offset: 2px; }
[hidden] { display: none !important; }

.card { position: absolute; left: 50%; top: 50%; width: min(500px, calc(100% - 32px));
  transform: translate(-50%,-50%) scale(.9); opacity: 0; pointer-events: none;
  padding: 1.4rem 1.5rem 1.1rem; border-radius: 20px;
  background: rgba(248,250,252,.07); border: 1px solid rgba(248,250,252,.16);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  box-shadow: 0 24px 60px rgba(0,0,0,.35);
  transition: opacity .5s ease, transform .6s cubic-bezier(.34,1.3,.64,1); }
.card.on { opacity: 1; transform: translate(-50%,-50%) scale(1); pointer-events: auto; }
.card.instant { transition: none; }
.c-head { display: flex; align-items: center; gap: .9rem; }
.c-icon { flex: none; width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; color: var(--cc);
  background: color-mix(in srgb, var(--cc) 18%, transparent); border: 1px solid color-mix(in srgb, var(--cc) 45%, transparent); }
.c-icon svg { width: 24px; height: 24px; }
.c-text { flex: 1; min-width: 0; }
.c-kicker { font-size: .9rem; font-weight: 400; color: var(--muted); }
.c-title { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.c-conf { text-align: right; color: var(--cc); font-weight: 700; font-size: 1.9rem; line-height: 1; }
.c-conf small { font-size: 1rem; margin-left: 2px; }
.c-note { margin: .8rem 0 .2rem; color: var(--muted); font-size: .95rem; }
.bars { margin-top: .6rem; }
.b-row { margin: .6rem 0; }
.b-head { display: flex; justify-content: space-between; font-size: 1.05rem; font-weight: 600; margin-bottom: .2rem; }
.b-track { background: rgba(148,163,184,.22); height: 10px; border-radius: 999px; overflow: hidden; }
.b-fill { height: 100%; width: 0; border-radius: 999px; transition: width .9s cubic-bezier(.22,1,.36,1); }
.card.instant .b-fill { transition: none; }
.c-foot { display: flex; justify-content: space-between; align-items: center; gap: .6rem; margin-top: .9rem; padding-top: .8rem;
  border-top: 1px solid rgba(148,163,184,.2); color: var(--muted); font-size: .85rem; }
.replay { cursor: pointer; font-size: .85rem; font-weight: 600; color: var(--fg); background: transparent;
  border: 1px solid rgba(148,163,184,.4); border-radius: 999px; padding: 0; width: 34px; height: 34px; display: grid; place-items: center; transition: background .2s ease; }
.replay svg { width: 16px; height: 16px; transition: transform .45s ease; }
.replay:hover svg { transform: rotate(180deg); }
.replay:hover { background: rgba(148,163,184,.2); }

.pill { position: absolute; left: 50%; bottom: 28px; height: 60px; width: min(560px, calc(100% - 32px));
  transform: translateX(-50%); display: none; align-items: center; gap: .7rem; padding: 0 .6rem 0 1.1rem;
  border-radius: 999px; background: rgba(248,250,252,.07); border: 1px solid rgba(248,250,252,.18);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  box-shadow: 0 0 40px rgba(59,130,246,.22), inset 0 1px 0 rgba(255,255,255,.06);
  overflow: hidden; white-space: nowrap; }
.pill > * { transition: opacity .16s ease; }
.pill.out > * { opacity: 0; }
.p-spark { flex: none; width: 20px; height: 20px; color: #93C5FD; }
.p-text { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; color: var(--fg); font-size: 1.05rem; font-weight: 400; }
.p-text.ph { color: var(--muted); }
.p-send { flex: none; width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center;
  background: #2563EB; color: #fff; }
.p-send svg { width: 18px; height: 18px; }
.ball { position: absolute; left: 0; top: 0; width: 60px; height: 60px; border-radius: 50%; opacity: 0; pointer-events: none;
  background: radial-gradient(circle at 35% 30%, #F8FAFC 0%, #94A3B8 45%, #334155 100%);
  box-shadow: 0 0 34px rgba(147,197,253,.55); will-change: transform; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

@media (max-width: 520px) { .c-title { font-size: 1.4rem; } .c-conf { font-size: 1.5rem; } canvas { width: 220px; height: 220px; } .status { top: calc(38% + 108px); } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
</head>
<body>
<div class="stage" id="stage">
  <div class="halo" id="halo" aria-hidden="true"></div>
  <canvas id="orb" aria-hidden="true"></canvas>
  <button class="skip" id="skip" type="button" hidden>Passer l'animation</button>
  <div class="pill" id="pill" aria-hidden="true">
    <svg class="p-spark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3.5l1.7 4.8 4.8 1.7-4.8 1.7L10 16.5l-1.7-4.8L3.5 10l4.8-1.7L10 3.5z"/><path d="M18 14.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8z"/></svg>
    <span class="p-text" id="pText"></span>
    <span class="p-send"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"/><path d="M5.5 11.5L12 5l6.5 6.5"/></svg></span>
  </div>
  <div class="ball" id="ball" aria-hidden="true"></div>

  <div class="status" id="status">
    <div class="s-main" id="sMain"></div>
    <div class="s-sub mono" id="sSub"></div>
    <div class="steps" id="steps" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
  </div>

  <section class="card" id="card" aria-label="Résultat de la prédiction">
    <div class="c-head">
      <span class="c-icon" id="cIcon" aria-hidden="true"></span>
      <div class="c-text">
        <div class="c-kicker">Prédiction du modèle</div>
        <div class="c-title" id="cTitle"></div>
      </div>
      <div class="c-conf mono"><span id="cConf">0</span><small>%</small></div>
    </div>
    <p class="c-note" id="cNote"></p>
    <div class="bars" id="bars"></div>
    <div class="c-foot">
      <span>Random Forest · 200 arbres</span>
      <button class="replay" id="replay" type="button" aria-label="Rejouer l'animation" title="Rejouer l'animation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg></button>
    </div>
  </section>
  <div class="sr" id="live" role="status" aria-live="polite"></div>
</div>

<script>
(function () {
  var D = __DATA__;
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var $ = function (id) { return document.getElementById(id); };
  var TAU = Math.PI * 2;
  var clamp01 = function (v) { return v < 0 ? 0 : v > 1 ? 1 : v; };
  var lerp = function (a, b, t) { return a + (b - a) * t; };

  /* ── easing ── */
  function cubic(x1, y1, x2, y2) {
    var cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
    var cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
    var sx = function (t) { return ((ax * t + bx) * t + cx) * t; };
    var sy = function (t) { return ((ay * t + by) * t + cy) * t; };
    var dx = function (t) { return (3 * ax * t + 2 * bx) * t + cx; };
    var solve = function (x) {
      var t = x, i, e, d;
      for (i = 0; i < 8; i++) { e = sx(t) - x; if (Math.abs(e) < 1e-6) return t; d = dx(t); if (Math.abs(d) < 1e-6) break; t -= e / d; }
      var lo = 0, hi = 1; t = x;
      for (i = 0; i < 40; i++) { e = sx(t); if (Math.abs(e - x) < 1e-6) break; if (x > e) lo = t; else hi = t; t = (hi - lo) / 2 + lo; }
      return t;
    };
    return function (x) { return x <= 0 ? 0 : x >= 1 ? 1 : sy(solve(x)); };
  }
  var E = { out: cubic(.22, 1, .36, 1), io: cubic(.65, 0, .35, 1), vortex: cubic(.6, 0, .2, 1) };

  /* ── dotted sphere ── */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var RINGS = 16, list = [], rand = mulberry32(7), k, j;
  for (k = 0; k < RINGS; k++) {
    var y = 1 - ((k + 0.5) / RINGS) * 2, r = Math.sqrt(1 - y * y), m = Math.max(4, Math.round(30 * r));
    for (j = 0; j < m; j++) {
      var a = (j / m) * TAU + k * 0.35;
      list.push({ x: Math.cos(a) * r, y: y, z: Math.sin(a) * r, u: (1 - y) / 2, s: rand() * 6.283 });
    }
  }
  var N = list.length;
  var DX = new Float32Array(N), DY = new Float32Array(N), DZ = new Float32Array(N), DU = new Float32Array(N), DS = new Float32Array(N);
  list.forEach(function (d, i) { DX[i] = d.x; DY[i] = d.y; DZ[i] = d.z; DU[i] = d.u; DS[i] = d.s; });

  var G_STEPS = 24, A_STEPS = 48, COLORS = [];
  function hexRgb(h) { h = h.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function buildColors(rgb) {
    COLORS = [];
    for (var gi = 0; gi <= G_STEPS; gi++) {
      var g = gi / G_STEPS;
      var rr = Math.round(lerp(235, rgb[0], g)), gg = Math.round(lerp(235, rgb[1], g)), bb = Math.round(lerp(235, rgb[2], g));
      for (var ai = 0; ai <= A_STEPS; ai++) COLORS.push('rgba(' + rr + ',' + gg + ',' + bb + ',' + (ai / A_STEPS).toFixed(3) + ')');
    }
  }

  var CAN = 260, ORB_R = 76, C0 = CAN / 2;
  var canvas = $('orb'), ctx = canvas.getContext('2d');
  var dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(CAN * dpr); canvas.height = Math.round(CAN * dpr);
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  var P = { k: 0, alpha: 0, spin: 0, rot: 0, sweep: 0, pop: 1, vortex: 0, gain: 1, floor: 0, rad: 0, prog: 0 };
  function resetP() {
    P.k = 0; P.alpha = 0; P.spin = 0; P.rot = 0; P.sweep = 0; P.pop = 1; P.vortex = 0; P.gain = 1; P.floor = 0; P.rad = 0; P.prog = 0;
    lit.fill(0); pw[0] = 1; pw[1] = 0; pw[2] = 0; pw[3] = 0; time = reduced ? 1.2 : 0;
  }
  var lit = new Float32Array(N), SX = new Float32Array(N), SY = new Float32Array(N), SR = new Float32Array(N), SD = new Float32Array(N);
  var SC = new Int16Array(N), pw = [1, 0, 0, 0], time = 0, raf = 0, last = 0;
  var S = 0.6, CP = Math.cos(0.35), SP = Math.sin(0.35);

  function draw(dt) {
    if (!ctx) return;
    ctx.clearRect(0, 0, CAN, CAN);
    time += dt; P.rot += P.spin * dt;
    var yaw = P.rot + P.vortex, cyw = Math.cos(yaw), syw = Math.sin(yaw), stepW = dt / 0.35, q, n;
    for (q = 0; q < 4; q++) { var dd0 = (q === P.prog ? 1 : 0) - pw[q]; pw[q] += Math.abs(dd0) <= stepW ? dd0 : dd0 > 0 ? stepW : -stepW; }
    var decay = Math.exp(-dt / 0.5), h0 = (time * 300) % N, h3 = (time * 480) % N;
    var a1 = time * 0.8, b1 = Math.sin(time * 0.5) * 0.9;
    var f1x = Math.cos(b1) * Math.cos(a1), f1y = Math.sin(b1), f1z = Math.cos(b1) * Math.sin(a1);
    var a2 = time * 0.55 + 2.1, b2 = Math.cos(time * 0.42) * 0.9;
    var f2x = Math.cos(b2) * Math.cos(a2), f2y = Math.sin(b2), f2z = Math.cos(b2) * Math.sin(a2);
    var lat = Math.sin(time * 2.2), swirlK = P.vortex * 1.5;

    for (n = 0; n < N; n++) {
      var dx = DX[n], dy = DY[n], dz = DZ[n], u = DU[n], pulse = 0, v, dd;
      if (pw[0] > 0.001) { dd = Math.abs(n - h0); if (dd > N - dd) dd = N - dd; v = Math.max(0, 1 - dd / 16); pulse = Math.max(pulse, v * v * pw[0]); }
      if (pw[1] > 0.001) {
        var v1 = Math.max(0, (dx * f1x + dy * f1y + dz * f1z - 0.72) / 0.28), v2 = Math.max(0, (dx * f2x + dy * f2y + dz * f2z - 0.72) / 0.28);
        v = Math.max(v1, v2); pulse = Math.max(pulse, v * v * pw[1]);
      }
      if (pw[2] > 0.001) { var e = dy - lat; v = Math.max(0, 1 - (e * e) / 0.02); pulse = Math.max(pulse, v * v * pw[2]); }
      if (pw[3] > 0.001) { dd = Math.abs(n - h3); if (dd > N - dd) dd = N - dd; v = Math.max(0, 1 - dd / 22); pulse = Math.max(pulse, v * v * pw[3]); }
      var l = Math.max(lit[n] * decay, pulse * P.gain); lit[n] = l;

      var ki = clamp01(P.k * (1 + S) - S * u);
      if (ki <= 0.001) { SC[n] = -1; continue; }
      var eo = E.out(ki), kk = eo * P.pop;
      var x1 = dx * cyw + dz * syw, z1 = -dx * syw + dz * cyw;
      var y2 = dy * CP - z1 * SP, z2 = dy * SP + z1 * CP;
      var f = 2.8 / (2.8 - z2), depth = (z2 + 1) / 2;
      var ox = x1 * ORB_R * kk * f, oy = -y2 * ORB_R * kk * f;
      if (swirlK > 0.001) { var sw = (1 - ki) * swirlK, cc = Math.cos(sw), ss = Math.sin(sw), tx = ox * cc - oy * ss; oy = ox * ss + oy * cc; ox = tx; }

      var g = clamp01((P.sweep * 1.4 - u) / 0.4);
      var al = 0.1 + 0.035 * Math.sin(DS[n] + time * 1.6) * (1 - g) + 0.32 * depth * depth + 0.75 * l * (1 - g) + g * (0.55 + 0.4 * depth) + 2 * g * (1 - g);
      al = Math.max(al, P.floor * (0.7 + 0.3 * depth)); if (al > 1) al = 1; al *= eo * P.alpha;
      SX[n] = C0 + ox; SY[n] = C0 + oy; SD[n] = depth;
      SR[n] = (1.15 * (0.45 + 0.75 * depth) * f + 0.9 * l + g * 0.25) * (1 + P.rad) * (0.4 + 0.6 * eo);
      var ai = Math.round(al * A_STEPS), gi = Math.round(g * G_STEPS);
      SC[n] = ai <= 0 ? -1 : gi * (A_STEPS + 1) + ai;
    }
    for (var pass = 0; pass < 2; pass++) {
      for (n = 0; n < N; n++) {
        var c = SC[n]; if (c < 0) continue;
        if ((SD[n] >= 0.5) !== (pass === 1)) continue;
        ctx.fillStyle = COLORS[c]; ctx.beginPath(); ctx.arc(SX[n], SY[n], SR[n], 0, TAU); ctx.fill();
      }
    }
  }
  function frame(now) {
    raf = 0;
    var dt = Math.max(0, Math.min(0.05, (now - last) / 1000));
    last = now; draw(dt);
    if (P.alpha > 0.002) raf = requestAnimationFrame(frame); else if (ctx) ctx.clearRect(0, 0, CAN, CAN);
  }
  function ensure() {
    if (raf || P.alpha <= 0.002) return;
    if (reduced) { draw(0); return; }
    last = performance.now(); raf = requestAnimationFrame(frame);
  }

  /* ── UI helpers ── */
  var ICONS = {
    Prive: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/><path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/><path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/><path d="M10 6h4M10 10h4M10 14h4M10 18h4"/></svg>',
    Public: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 22h18"/><path d="M6 18v-7M10 18v-7M14 18v-7M18 18v-7"/><path d="M12 2 20 7H4Z"/></svg>',
    Independant: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>'
  };
  var STEPS = D.steps || ['Lecture du profil', 'Encodage des variables', 'Vote des arbres', 'Calcul des probabilités'];
  var epoch = 0, cls = D.classes || [], top = cls[0];
  var statusEl = $('status'), cardEl = $('card'), skipEl = $('skip');

  function setMain(text, dots) {
    $('sMain').innerHTML = '';
    var s = document.createElement('span'); s.textContent = text; $('sMain').appendChild(s);
    if (dots) { var d = document.createElement('span'); d.className = 'dots'; d.setAttribute('aria-hidden', 'true'); d.innerHTML = '<i></i><i></i><i></i>'; $('sMain').appendChild(d); }
  }
  function setStep(i) {
    setMain(STEPS[i], true);
    $('sSub').textContent = 'Étape ' + (i + 1) + ' sur ' + STEPS.length;
    var segs = $('steps').children;
    for (var s = 0; s < segs.length; s++) segs[s].className = s <= i ? 'on' : '';
    $('steps').style.visibility = 'visible';
  }
  function setAccent(hex) { document.documentElement.style.setProperty('--halo', hex); document.documentElement.style.setProperty('--cc', hex); }

  function buildCard() {
    $('cIcon').innerHTML = ICONS[top.key] || '';
    $('cTitle').textContent = top.nom;
    $('cNote').textContent = D.note || '';
    var html = '';
    cls.forEach(function (c, i) {
      html += '<div class="b-row"><div class="b-head"><span>' + c.nom + '</span><span class="mono" data-p="' + i + '">0 %</span></div>' +
        '<div class="b-track"><div class="b-fill" data-f="' + i + '" style="background:' + c.color + '"></div></div></div>';
    });
    $('bars').innerHTML = html;
  }
  function setValues(t) {
    $('cConf').textContent = Math.round(top.p * 100 * t);
    cls.forEach(function (c, i) {
      var el = document.querySelector('[data-p="' + i + '"]');
      if (el) el.textContent = (c.p * 100 * t).toFixed(1).replace('.', ',') + ' %';
    });
  }
  function setBars(t) {
    cls.forEach(function (c, i) { var f = document.querySelector('[data-f="' + i + '"]'); if (f) f.style.width = (c.p * 100 * t) + '%'; });
  }
  function announce() { $('live').textContent = 'Résultat : ' + top.nom + ', ' + Math.round(top.p * 100) + ' %.'; }

  function tw(ms, fn, ep) {
    return new Promise(function (res) {
      var t0 = performance.now();
      var step = function (now) {
        if (ep !== epoch) return res();
        var p = clamp01((now - t0) / ms); fn(p);
        if (p < 1) requestAnimationFrame(step); else res();
      };
      requestAnimationFrame(step);
    });
  }
  function sl(ms) { return new Promise(function (res) { setTimeout(res, ms); }); }

  var stageEl = $('stage'), pillEl = $('pill'), ballEl = $('ball'), dir = 1;
  var BEZ = function (t, p0, c, p2) { return (1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * c + t * t * p2; };
  var FLY = cubic(.5, 0, .1, 1), GROW = cubic(.3, 0, .2, 1);

  function hideLaunch() { pillEl.style.display = 'none'; ballEl.style.opacity = '0'; }

  function showFinal() {
    resetP(); if (ctx) ctx.clearRect(0, 0, CAN, CAN); hideLaunch();
    setAccent(top.color); statusEl.classList.add('hide'); skipEl.hidden = true;
    cardEl.classList.add('instant'); cardEl.classList.add('on');
    setBars(1); setValues(1); announce();
  }

  async function play() {
    var ep = ++epoch;
    cardEl.classList.remove('on', 'instant'); setBars(0); setValues(0);
    statusEl.classList.add('hide'); skipEl.hidden = false;
    setAccent('#3B82F6'); resetP(); buildColors(hexRgb(top.color));
    $('steps').style.visibility = 'hidden';

    /* la pilule (le profil « envoyé ») */
    var r = stageEl.getBoundingClientRect(), SW = r.width, SH = r.height;
    pillEl.style.display = 'flex'; pillEl.style.opacity = '1'; pillEl.style.width = '';
    pillEl.classList.remove('out'); $('pText').classList.remove('ph');
    $('pText').textContent = D.summary || '';
    var pw0 = pillEl.getBoundingClientRect().width;
    var x0 = SW / 2, y0 = SH - 28 - 30, x1 = SW / 2, y1 = SH * 0.38;
    dir = -dir;
    function placeBall(u, d) {
      var cx = x0 + dir * 0.3 * (y0 - y1);
      var x = BEZ(u, x0, cx, x1), y = BEZ(u, y0, y0 - 0.6 * (y0 - y1), y1);
      ballEl.style.width = d + 'px'; ballEl.style.height = d + 'px';
      ballEl.style.transform = 'translate3d(' + (x - d / 2).toFixed(2) + 'px,' + (y - d / 2).toFixed(2) + 'px,0)';
    }
    ballEl.style.opacity = '0'; placeBall(0, 60);
    await sl(350); if (ep !== epoch) return;

    /* 1) la pilule se referme en bille */
    pillEl.classList.add('out');
    await tw(520, function (p) {
      pillEl.style.width = lerp(pw0, 60, E.io(p)) + 'px';
      var o = E.out(clamp01((p - 0.5) / 0.5));
      pillEl.style.opacity = String(1 - o); ballEl.style.opacity = String(o);
    }, ep);
    if (ep !== epoch) return;
    pillEl.style.display = 'none';

    /* 2) la bille vole vers le centre en grossissant */
    await tw(850, function (p) {
      var d = p < 0.77 ? lerp(60, 138, GROW(p / 0.77)) : lerp(138, 132, E.out((p - 0.77) / 0.23));
      placeBall(FLY(p), d);
    }, ep);
    if (ep !== epoch) return;

    /* 3) la sphère de points s'assemble */
    statusEl.classList.remove('hide'); setStep(0);
    await tw(750, function (p) {
      var e = E.out(p); P.k = e; P.alpha = e; P.spin = 0.9 * e;
      P.pop = p < 0.5 ? 1 + 0.05 * E.out(p / 0.5) : 1.05 - 0.05 * E.io((p - 0.5) / 0.5);
      ballEl.style.opacity = String(1 - E.out(clamp01(p / 0.35)));
      ensure();
    }, ep);
    if (ep !== epoch) return;
    ballEl.style.opacity = '0';

    /* 4) réflexion : une étape après l'autre */
    for (var i = 0; i < STEPS.length; i++) {
      if (ep !== epoch) return;
      setStep(i); P.prog = i; await sl(i === STEPS.length - 1 ? 620 : 480);
    }
    if (ep !== epoch) return;

    /* 5) la sphère prend la couleur du résultat */
    setMain('Analyse terminée', false); $('sSub').textContent = 'Résultat prêt'; $('steps').style.visibility = 'hidden';
    setAccent(top.color);
    await tw(850, function (p) {
      var a = clamp01(p / 0.78), b = clamp01((p - 0.44) / 0.56);
      P.sweep = E.io(a); P.spin = lerp(0.9, 0.3, E.out(a)); P.gain = 1 - E.out(a); P.floor = 0.95 * E.out(b); P.rad = 0.15 * E.out(b);
    }, ep);
    if (ep !== epoch) return;

    /* 6) elle se condense, puis la carte se déplie */
    statusEl.classList.add('hide');
    await tw(650, function (p) {
      var kk = E.vortex(clamp01((p - 0.17) / 0.74));
      P.pop = 1 + 0.06 * E.out(clamp01(p / 0.15)); P.k = 1 - kk; P.vortex = 1.6 * kk; P.alpha = 1 - E.out(clamp01((p - 0.7) / 0.3));
    }, ep);
    if (ep !== epoch) return;

    skipEl.hidden = true; cardEl.classList.add('on'); await sl(380);
    if (ep !== epoch) return;
    setBars(1);
    await tw(850, function (p) { setValues(E.out(p)); }, ep);
    if (ep !== epoch) return;
    announce();
  }

  function idle() {
    buildColors(hexRgb('#3B82F6'));
    setMain('Prêt à analyser', false);
    $('sSub').textContent = D.idleText || '';
    $('sSub').classList.remove('mono');
    $('sSub').classList.add('idle');
    $('steps').style.display = 'none';
    P.k = 1; P.alpha = 0.55; P.spin = 0.22; P.gain = 0.5; time = reduced ? 1.2 : 0;
    ensure();
  }

  skipEl.addEventListener('click', function () { epoch++; showFinal(); });
  $('replay').addEventListener('click', function () { if (reduced) return; play(); });
  if (reduced) $('replay').hidden = true;

  if (D.state === 'idle' || !top) { idle(); }
  else {
    buildCard(); buildColors(hexRgb(top.color));
    if (D.autoplay && !reduced) play(); else showFinal();
  }
})();
</script>
</body>
</html>
"""


def orb_component(result, autoplay):
    if result is None:
        data = {
            "state": "idle",
            "idleText": "Réglez le profil, puis lancez l'analyse.",
        }
    else:
        ordered = sorted(result["proba"].items(), key=lambda kv: kv[1], reverse=True)
        data = {
            "state": "result",
            "autoplay": autoplay,
            "pred": result["pred"],
            "summary": summary(result["profile"]),
            "note": (
                "Prédiction peu tranchée : les probabilités sont proches."
                if ordered[0][1] < 0.5
                else "Catégorie la plus probable selon le modèle."
            ),
            "classes": [
                {"key": k, "nom": CLASSES[k]["nom"], "color": CLASSES[k]["couleur"], "p": p}
                for k, p in ordered
            ],
            "steps": [
                "Lecture du profil", "Encodage des variables",
                "Vote des 200 arbres", "Calcul des probabilités",
            ],
        }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = ORB_HTML.replace("__DATA__", payload)
    if hasattr(st, "iframe"):
        st.iframe(html, height=480, alt="Animation et résultat de la prédiction")
    else:  # anciennes versions de Streamlit
        import streamlit.components.v1 as components

        components.html(html, height=480)


# ------------------------------------------------------------------
# En-tête
# ------------------------------------------------------------------
st.markdown(
    """
<div class="head">
  <span class="head-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
  </span>
  <h1>Prédiction du type d'employeur</h1>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 1], gap="large")

# ------------------------------------------------------------------
# Colonne de gauche : formulaire
# ------------------------------------------------------------------
with left:
    st.markdown('<div class="sec">Profils types (lancent l\'analyse)</div>', unsafe_allow_html=True)
    cols = st.columns(len(PRESETS))
    for col, (name, (icon, _)) in zip(cols, PRESETS.items()):
        with col:
            st.button(name, icon=icon, on_click=apply_preset, args=(name,), key=f"preset_{name}")

    with st.container(border=True):
        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown('<div class="sec">Profil personnel</div>', unsafe_allow_html=True)
            st.slider("Âge", 17, 90, key="age")
            select("Sexe", "sex")
            select("Statut marital", "marital-status")
            select("Relation familiale", "relationship")
            select("Origine", "race")
            select("Pays d'origine", "native-country")
        with b:
            st.markdown('<div class="sec">Situation professionnelle</div>', unsafe_allow_html=True)
            st.slider(
                "Années d'études", 1, 16, key="education-num",
                help="9 = baccalauréat, 13 = licence, 14 = master, 16 = doctorat",
            )
            select("Métier", "occupation")
            st.slider("Heures par semaine", 1, 99, key="hours-per-week")
            select("Revenu annuel", "income")
            st.number_input("Gains en capital ($)", 0, 100000, step=500, key="capital-gain")
            st.number_input("Pertes en capital ($)", 0, 5000, step=50, key="capital-loss")

    st.button("Analyser le profil", type="primary", icon=":material/auto_awesome:", on_click=analyse)

# ------------------------------------------------------------------
# Colonne de droite : animation et résultat
# ------------------------------------------------------------------
with right:
    result = st.session_state.get("result")
    autoplay = result is not None and st.session_state.get("animated_id") != result["id"]
    if result is not None:
        st.session_state["animated_id"] = result["id"]

    orb_component(result, autoplay)

    if result is not None:
        if result["profile"] != current_profile():
            st.info(
                "Le profil a été modifié depuis la dernière analyse. "
                "Cliquez sur « Analyser le profil » pour mettre le résultat à jour.",
                icon=":material/sync:",
            )
        p = result["profile"]
        chips = [
            f"{p['age']} ans", fr("sex", p["sex"]), fr("occupation", p["occupation"]),
            f"{p['hours-per-week']} h / semaine", f"{p['education-num']} ans d'études",
            fr("marital-status", p["marital-status"]), fr("income", p["income"]),
        ]
        st.markdown(
            '<div class="sec">Profil analysé</div>'
            + "".join(f'<span class="chip">{c}</span>' for c in chips),
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------
# Performance et à propos
# ------------------------------------------------------------------
st.markdown("&nbsp;")
with st.expander("Performance du modèle", icon=":material/analytics:"):
    st.markdown("Résultats mesurés sur le jeu de test (6 141 personnes, jamais vues à l'entraînement).")
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("76 %", "Accuracy du modèle"),
        ("73,9 %", "Accuracy d'un modèle naïf (toujours « privé »)"),
        ("0,55", "F1 macro du modèle"),
        ("≈ 0,28", "F1 macro du modèle naïf"),
    ]
    for col, (n, t) in zip((c1, c2, c3, c4), kpis):
        col.markdown(f'<div class="kpi"><div class="n">{n}</div><div class="t">{t}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("&nbsp;")
    perf = pd.DataFrame(
        {
            "Catégorie": ["Secteur privé", "Secteur public", "Indépendant"],
            "Precision": [0.80, 0.54, 0.54],
            "Recall": [0.92, 0.32, 0.31],
            "F1": [0.85, 0.40, 0.39],
            "Effectif test": [4539, 870, 732],
        }
    )
    st.dataframe(perf, hide_index=True)
    st.markdown(
        """
- **Precision** : quand le modèle annonce une catégorie, il a raison dans cette proportion des cas.
- **Recall** : parmi les vraies personnes de la catégorie, c'est la part que le modèle retrouve.
- **F1 macro** : moyenne des F1 de chaque catégorie, la métrique principale ici car le secteur privé
  représente 74 % des données.

Le secteur privé est bien reconnu. Les catégories public et indépendant restent difficiles :
le modèle les repère dans environ un cas sur trois.
"""
    )

with st.expander("À propos du projet", icon=":material/info:"):
    x, y = st.columns(2, gap="large")
    with x:
        st.markdown(
            """
**Objectif métier.** Comprendre si le type d'employeur peut être estimé à partir d'un profil
socio-professionnel, afin d'aider une collectivité ou un organisme d'orientation à mieux cerner
les statuts professionnels.

**Objectif data science.** Classification supervisée multiclasse : prédire `workclass`,
regroupée en trois catégories.

| Catégorie | Regroupe |
|---|---|
| Privé | Private |
| Public | Federal-gov, State-gov, Local-gov |
| Indépendant | Self-emp-inc, Self-emp-not-inc |
"""
        )
    with y:
        st.markdown(
            """
**Méthode.** Nettoyage, normalisation des variables numériques, one-hot encoding des variables
catégorielles, séparation train / test stratifiée (80 / 20), Random Forest avec poids de classes,
évaluation par F1 macro comparée à un modèle naïf.

Les étapes affichées pendant l'animation décrivent le traitement réel d'un profil par le modèle.
Les probabilités sont des estimations du modèle, non calibrées.
"""
        )
        st.markdown(
            """
<div class="note"><b>Limites et éthique.</b> Données de 1994, performance modérée, et variables
sensibles (sexe, origine) qui peuvent reproduire des biais. Les hyperparamètres ont été choisis
sur le jeu de test, la performance est donc légèrement optimiste. Projet pédagogique : ne pas
utiliser pour des décisions sur de vraies personnes.</div>
""",
            unsafe_allow_html=True,
        )