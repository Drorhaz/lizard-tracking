#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless-friendly label QC web app for YOLO pose labels (x,y,v per keypoint).
Now with full BBOX editing:
- Shows bbox (green).
- Drag inside to move; drag corner handles to resize.
- If no bbox exists: click-drag anywhere to draw a new one (rubber-band).
- Manual bbox fields (cx,cy,w,h) apply directly.

KP editing (unchanged):
- Drag dots, double-click to add (v=2), hold T to toggle visibility.
- Cycle nose/ears left/right.

Resumes at first unreviewed using ok/skip logs. Writes label .bak once.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import io, os, json, math, shutil, socket

from flask import Flask, request, jsonify, send_file, render_template_string, url_for
import cv2
import numpy as np

# ======================= CONFIG =======================
DEFAULT_SPLIT = "train"
DEFAULT_IMAGES_ROOT = Path(f"dataset/images/{DEFAULT_SPLIT}").resolve()
DEFAULT_LABELS_ROOT = Path(f"dataset/labels/{DEFAULT_SPLIT}").resolve()
OUT_PREVIEW_ROOT = Path("output/preview").resolve()
LOG_OK_ROOT = Path("output/data/valid").resolve()
LOG_SKIP_ROOT = Path("output/data/skip").resolve()

CURRENT_IMAGES_ROOT = DEFAULT_IMAGES_ROOT
CURRENT_LABELS_ROOT = DEFAULT_LABELS_ROOT
OUT_PREVIEW = (OUT_PREVIEW_ROOT / DEFAULT_SPLIT).resolve()
LOG_OK = (LOG_OK_ROOT / f"{DEFAULT_SPLIT}_ok.txt").resolve()
LOG_SKIP = (LOG_SKIP_ROOT / f"{DEFAULT_SPLIT}_skip.txt").resolve()

# Canvas width (fixed); height scales to preserve aspect
CANVAS_MAX_W = 800

# Draw style
RADIUS = 6
THICK  = 2
HANDLE_R = 8  # bbox corner handle radius (px in display space)
DEFAULT_BOX = 0.12  # default normalized side length for seeded heads
# ======================================================

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
OUT_PREVIEW.mkdir(parents=True, exist_ok=True)
LOG_OK.parent.mkdir(parents=True, exist_ok=True)
LOG_SKIP.parent.mkdir(parents=True, exist_ok=True)

@dataclass
class Pair:
    img: Path
    lbl: Path

def list_pairs(img_root: Path, lbl_root: Path) -> List[Pair]:
    pairs: List[Pair] = []
    if not img_root.exists():
        return pairs
    for img in sorted(img_root.rglob("*")):
        if not img.is_file():
            continue
        if img.suffix.lower() not in IMG_EXTS:
            continue
        try:
            rel = img.relative_to(img_root)
        except ValueError:
            continue
        lbl = lbl_root / rel.with_suffix(".txt")
        pairs.append(Pair(img=img, lbl=lbl))
    return pairs


def refresh_paths(images_root: Path, labels_root: Path, *, preview_split: Optional[str] = None) -> None:
    global CURRENT_IMAGES_ROOT, CURRENT_LABELS_ROOT, OUT_PREVIEW, LOG_OK, LOG_SKIP, CACHE_SPLIT, PAIRS
    CURRENT_IMAGES_ROOT = images_root.resolve()
    CURRENT_LABELS_ROOT = labels_root.resolve()
    split = preview_split if preview_split is not None else CACHE_SPLIT
    CACHE_SPLIT = split
    OUT_PREVIEW = (OUT_PREVIEW_ROOT / split).resolve()
    LOG_OK = (LOG_OK_ROOT / f"{split}_ok.txt").resolve()
    LOG_SKIP = (LOG_SKIP_ROOT / f"{split}_skip.txt").resolve()
    OUT_PREVIEW.mkdir(parents=True, exist_ok=True)
    LOG_OK.parent.mkdir(parents=True, exist_ok=True)
    LOG_SKIP.parent.mkdir(parents=True, exist_ok=True)
    PAIRS = list_pairs(CURRENT_IMAGES_ROOT, CURRENT_LABELS_ROOT)

CACHE_SPLIT = DEFAULT_SPLIT
PAIRS: List[Pair] = []

refresh_paths(DEFAULT_IMAGES_ROOT, DEFAULT_LABELS_ROOT, preview_split=DEFAULT_SPLIT)


def _clamp_index(idx: int) -> Optional[int]:
    if not PAIRS:
        return None
    return max(0, min(len(PAIRS) - 1, idx))

def read_list(p: Path) -> set:
    if not p.exists(): return set()
    return set(line.strip() for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())

def read_label(lblp: Path) -> List[List[str]]:
    if not lblp.exists(): return []
    lines=[ln.strip() for ln in lblp.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    return [ln.split() for ln in lines]

def write_label(lblp: Path, tokens: List[List[str]]):
    txt="\n".join(" ".join(t) for t in tokens)+"\n"
    lblp.parent.mkdir(parents=True, exist_ok=True)
    lblp.write_text(txt, encoding="utf-8")

def ensure_backup(lblp: Path):
    bak = lblp.with_suffix(".txt.bak")
    if not bak.exists() and lblp.exists():
        shutil.copyfile(lblp, bak)

def draw_bbox(im, W,H, cx,cy,w,h, color=(0,255,0), thick=2):
    x1=int((cx-w/2)*W); y1=int((cy-h/2)*H)
    x2=int((cx+w/2)*W); y2=int((cy+h/2)*H)
    cv2.rectangle(im,(x1,y1),(x2,y2),color,thick)

def render_overlay(pair: Pair) -> np.ndarray:
    im = cv2.imread(str(pair.img))
    if im is None:
        im = np.zeros((480,640,3), np.uint8)
        cv2.putText(im, "unreadable image", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return im
    H,W = im.shape[:2]
    tokens = read_label(pair.lbl)
    for t in tokens:
        if len(t)==14:
            cx,cy,w,h = map(float, t[1:5])
            draw_bbox(im,W,H,cx,cy,w,h,(0,255,0),THICK)
            k = list(map(float, t[5:]))
            nose=(int(k[0]*W), int(k[1]*H), int(k[2]))
            L   =(int(k[3]*W), int(k[4]*H), int(k[5]))
            R   =(int(k[6]*W), int(k[7]*H), int(k[8]))
            if nose[2]>0: cv2.circle(im, (nose[0],nose[1]), RADIUS, (0,0,255), -1)
            if L[2]>0:    cv2.circle(im, (L[0],L[1]),     RADIUS, (255,0,0), -1)
            if R[2]>0:    cv2.circle(im, (R[0],R[1]),     RADIUS, (255,0,0), -1)
            if nose[2]>0 and L[2]>0 and R[2]>0:
                mx=(L[0]+R[0])//2; my=(L[1]+R[1])//2
                cv2.line(im, (nose[0],nose[1]), (mx,my), (0,255,255), THICK)
        elif len(t)==5:
            cx,cy,w,h = map(float, t[1:5])
            draw_bbox(im,W,H,cx,cy,w,h,(0,255,0),THICK)
    return im

def img_to_png_bytes(im: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", im)
    if not ok: raise RuntimeError("imencode failed")
    return buf.tobytes()

def normalized_from_pixels(px: float, py: float, W: int, H: int):
    return max(0,min(1, px/W)), max(0,min(1, py/H))

def clamp01(z: float) -> float:
    return max(0.0, min(1.0, float(z)))

def commit_edit(pair: Pair, edit: Dict[str, Any]) -> bool:
    """Apply edits to first object line (create if missing)."""
    tokens = read_label(pair.lbl)
    im = cv2.imread(str(pair.img))
    if im is None:
        return False
    H,W = im.shape[:2]

    # seeding from scratch or reseeding: place keypoints around click, leaving bbox untouched
    if "seed_template" in edit:
        seed = edit["seed_template"]
        seed_x = float(seed.get("x", W / 2))
        seed_y = float(seed.get("y", H / 2))
        nx, ny = normalized_from_pixels(seed_x, seed_y, W, H)

        if not tokens:
            cx0, cy0 = 0.5, 0.5
            side = DEFAULT_BOX
            new = [
                "0",
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                f"{side:.6f}",
                f"{side:.6f}",
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                "0",
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                "0",
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                "0",
            ]
            tokens = [new]

        t = tokens[0]
        if len(t) == 5:
            cx0, cy0, w0, h0 = map(float, t[1:5])
            t = t[:5] + [
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                "0",
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                "0",
                f"{cx0:.6f}",
                f"{cy0:.6f}",
                "0",
            ]

        cx, cy, w, h = map(float, t[1:5])
        span = w if w > 0 else DEFAULT_BOX
        ear_dx = max(0.02, min(0.25, span * 0.3))
        nose = (nx, ny, 2.0)
        left = (clamp01(nx - ear_dx), ny, 2.0)
        right = (clamp01(nx + ear_dx), ny, 2.0)

        def spt(pt):
            x, y, v = pt
            return [f"{x:.6f}", f"{y:.6f}", f"{float(v):.1f}"]

        head = [t[0], f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        new_t = head + spt(nose) + spt(left) + spt(right)

        ensure_backup(pair.lbl)
        tokens[0] = new_t
        write_label(pair.lbl, tokens)
        im2 = render_overlay(pair)
        cv2.imwrite(str(OUT_PREVIEW / pair.img.name), im2)
        return True

    if not tokens:
        # create a blank 14-token line with a tiny bbox at center and invisible kps
        cx,cy,w,h = 0.5,0.5, DEFAULT_BOX,DEFAULT_BOX
        new = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}",
               f"{cx:.6f}", f"{cy:.6f}", "0",
               f"{cx:.6f}", f"{cy:.6f}", "0",
               f"{cx:.6f}", f"{cy:.6f}", "0"]
        tokens = [new]

    t = tokens[0]
    # upgrade 5->14 if needed (so we can store keypoints consistently)
    if len(t)==5:
        cx,cy,w,h = map(float, t[1:5])
        t = t[:5] + [f"{cx:.6f}",f"{cy:.6f}","0",  f"{cx:.6f}",f"{cy:.6f}","0",  f"{cx:.6f}",f"{cy:.6f}","0"]

    # unpack
    cx,cy,w,h = map(float, t[1:5])
    k = list(map(float, t[5:]))
    nose=(k[0],k[1],k[2]) if len(k)>=3 else (cx,cy,0.0)
    L   =(k[3],k[4],k[5]) if len(k)>=6 else (cx,cy,0.0)
    R   =(k[6],k[7],k[8]) if len(k)>=9 else (cx,cy,0.0)

    old_cx, old_cy, old_w, old_h = cx, cy, w, h
    bbox_changed = False

    # bbox edits
    if "bbox_set" in edit:
        cx = clamp01(edit["bbox_set"]["cx"])
        cy = clamp01(edit["bbox_set"]["cy"])
        w  = clamp01(edit["bbox_set"]["w"])
        h  = clamp01(edit["bbox_set"]["h"])
        bbox_changed = True
    if "bbox_from_pixels" in edit:
        x1,y1,x2,y2 = [float(edit["bbox_from_pixels"][k]) for k in ("x1","y1","x2","y2")]
        x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
        # avoid degenerate box
        if abs(x2-x1) < 2 or abs(y2-y1) < 2:
            pass
        else:
            cx,cy = normalized_from_pixels((x1+x2)/2, (y1+y2)/2, W, H)
            w  = clamp01((x2-x1)/W)
            h  = clamp01((y2-y1)/H)
            bbox_changed = True

    if bbox_changed:
        old_x1 = old_cx - old_w / 2.0
        old_y1 = old_cy - old_h / 2.0
        new_x1 = cx - w / 2.0
        new_y1 = cy - h / 2.0
        delta_cx = cx - old_cx
        delta_cy = cy - old_cy

        def remap_point(pt: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, v = pt
            if old_w <= 1e-6 or old_h <= 1e-6:
                x = clamp01(x + delta_cx)
                y = clamp01(y + delta_cy)
                return (x, y, v)
            rel_x = (x - old_x1) / max(old_w, 1e-6)
            rel_y = (y - old_y1) / max(old_h, 1e-6)
            x = clamp01(new_x1 + rel_x * w)
            y = clamp01(new_y1 + rel_y * h)
            return (x, y, v)

        nose = remap_point(nose)
        L = remap_point(L)
        R = remap_point(R)

    # keypoint perms/drags/toggles/sets
    if edit.get("perm") == "cycle_left":
        nose, L, R = L, R, nose
    if edit.get("perm") == "cycle_right":
        nose, R, L = R, L, nose

    if "drag" in edit:
        which=edit["drag"]["which"]; px=float(edit["drag"]["x"]); py=float(edit["drag"]["y"])
        nx,ny = normalized_from_pixels(px, py, W, H)
        if which=="nose": nose=(nx,ny,2.0)
        elif which=="L":  L=(nx,ny,2.0)
        elif which=="R":  R=(nx,ny,2.0)

    if "toggle" in edit:
        which=edit["toggle"]
        def tv(pt): x,y,v=pt; return (x,y, 0.0 if v>0 else 2.0)
        if which=="nose": nose=tv(nose)
        elif which=="L":  L=tv(L)
        elif which=="R":  R=tv(R)

    if "set" in edit:
        s = edit["set"]
        def clampv(v):
            vv = int(float(v))
            return 2.0 if vv>=2 else (1.0 if vv==1 else 0.0)
        if "nose" in s:
            x,y,v = s["nose"]; nose = (clamp01(x), clamp01(y), clampv(v))
        if "L" in s:
            x,y,v = s["L"];    L    = (clamp01(x), clamp01(y), clampv(v))
        if "R" in s:
            x,y,v = s["R"];    R    = (clamp01(x), clamp01(y), clampv(v))
        if "bbox" in s:
            bb = s["bbox"]
            cx = clamp01(bb.get("cx", cx)); cy = clamp01(bb.get("cy", cy))
            w  = clamp01(bb.get("w", w));   h  = clamp01(bb.get("h", h))

    # pack and save
    head=[t[0], f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    def spt(pt): x,y,v=pt; return [f"{x:.6f}", f"{y:.6f}", f"{float(v):.1f}"]
    new_t = head + spt(nose)+spt(L)+spt(R)

    ensure_backup(pair.lbl)
    tokens[0] = new_t
    write_label(pair.lbl, tokens)

    # preview
    im2 = render_overlay(pair)
    cv2.imwrite(str(OUT_PREVIEW/pair.img.name), im2)
    return True

def nearest_kp(pair: Pair, x: float, y: float) -> Optional[str]:
    tokens = read_label(pair.lbl)
    if not tokens or len(tokens[0]) < 14:
        return None
    im = cv2.imread(str(pair.img)); 
    if im is None: return None
    H,W = im.shape[:2]
    k = list(map(float, tokens[0][5:]))
    pts = {
        "nose": (k[0]*W, k[1]*H, int(k[2])),
        "L":    (k[3]*W, k[4]*H, int(k[5])),
        "R":    (k[6]*W, k[7]*H, int(k[8])),
    }
    best=None; bestd=1e18
    for name,(px,py,v) in pts.items():
        # consider even invisible so we can add by dblclick
        d=(px-x)**2 + (py-y)**2
        if d<bestd: bestd=d; best=name
    return best if bestd <= 20*20 else None

def current_bbox_pixels(pair: Pair):
    tokens = read_label(pair.lbl)
    if not tokens:
        return None
    t = tokens[0]
    if len(t) < 5:
        return None
    im = cv2.imread(str(pair.img))
    if im is None: return None
    H,W = im.shape[:2]
    cx,cy,w,h = map(float, t[1:5])
    x1 = (cx - w/2)*W; y1 = (cy - h/2)*H
    x2 = (cx + w/2)*W; y2 = (cy + h/2)*H
    return (x1,y1,x2,y2,W,H)

def first_unreviewed_index() -> int:
    ok = read_list(LOG_OK)
    skip = read_list(LOG_SKIP)
    reviewed = ok | skip
    for i, p in enumerate(PAIRS):
        if str(p.img) not in reviewed:
            return i
    return max(0, len(PAIRS)-1)

APP = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Label QC</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 16px; }
    .wrap { display:flex; align-items:flex-start; gap:16px; }
    .left { flex: 1 1 auto; }
    .right { width: 360px; }
    .panel { border:1px solid #ccc; padding:12px; border-radius:8px; margin-bottom:12px; }
    button { padding:8px 12px; margin:4px; cursor:pointer }
    button.active { background:#0a7d1a; color:#fff; }
    #canvas { border:1px solid #999; max-width:95%; height:auto; display:block }
    input[type="number"] { width:6.5em; }
    .muted { color:#666 }
    .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center }
    form.dataset { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; align-items:center }
    form.dataset input[type="text"] { width: 280px; }
    .warning { color:#c43; font-weight:bold; }
  </style>
</head>
<body>
<h3>Label QC ({{ idx+1 }}/{{ total }}) — remaining {{ remaining }}</h3>
<p class="muted">{{ imgname }}</p>

<form class="dataset" method="get">
  <label>Images dir <input type="text" name="img_dir" value="{{ images_root }}"></label>
  <label>Labels dir <input type="text" name="lbl_dir" value="{{ labels_root }}"></label>
  <label>Split <input type="text" name="split" value="{{ split }}" style="width:120px"></label>
  <label>Image <input type="text" name="image" placeholder="path/to/image" value="{{ current_image }}" style="width:280px"></label>
  <button type="submit">Load</button>
  {% if has_data %}
    <input type="hidden" name="i" value="{{ idx }}">
  {% endif %}
</form>

<div class="wrap">
  <div class="left panel">
    <img id="raw" src="{{ img_url }}" style="display:none"/>
    <canvas id="canvas" width="10" height="10"></canvas>
    <div class="row">
      <button onclick="cycle('left')">⟲ Cycle</button>
      <button onclick="cycle('right')">Cycle ⟳</button>
      <button onclick="mark('ok')">OK</button>
      <button onclick="mark('skip')">Skip</button>
      <button onclick="prev()">⟵ Prev</button>
      <button onclick="next()">Next ⟶</button>
      <button id="bboxToggle" onclick="toggleBBoxMode()">BBox Edit: Off</button>
    </div>
    {% if has_data %}
    <p>KP: drag dots; <b>double-click</b> to seed nose & ears at the cursor; hold <b>T</b> and click to toggle visibility.</p>
    <p>BBox: click <b>BBox Edit</b> to enable moving/resizing; drag inside to move, drag corners to resize; if none exists, <b>click-drag</b> while edit mode is on (or use numeric fields).</p>
    {% else %}
    <p class="warning">No images found. Provide valid directories above.</p>
    {% endif %}
  </div>

  <div class="right">
    <div class="panel">
      <h4>Manual BBox (normalized)</h4>
      <div class="row">
        cx <input id="bcx" type="number" step="0.001" min="0" max="1">
        cy <input id="bcy" type="number" step="0.001" min="0" max="1">
      </div>
      <div class="row">
        w  <input id="bw"  type="number" step="0.001" min="0" max="1">
        h  <input id="bh"  type="number" step="0.001" min="0" max="1">
        <button onclick="applyBBox()">Apply</button>
      </div>
    </div>

    <div class="panel">
      <h4>Manual KPs (x,y ∈ [0,1], v∈{0,1,2})</h4>
      <div>Nose: x <input id="nx" type="number" step="0.001" min="0" max="1">
               y <input id="ny" type="number" step="0.001" min="0" max="1">
               v <input id="nv" type="number" step="1" min="0" max="2"></div>
      <div>L ear: x <input id="lx" type="number" step="0.001" min="0" max="1">
               y <input id="ly" type="number" step="0.001" min="0" max="1">
               v <input id="lv" type="number" step="1" min="0" max="2"></div>
      <div>R ear: x <input id="rx" type="number" step="0.001" min="0" max="1">
               y <input id="ry" type="number" step="0.001" min="0" max="1">
               v <input id="rv" type="number" step="1" min="0" max="2"></div>
      <div class="row"><button onclick="applyKPs()">Apply</button></div>
      <pre id="lbltext" class="muted" style="white-space:pre-wrap;"></pre>
      <div>
        Label path <input id="labelPath" type="text" style="width:100%" readonly>
      </div>
      <div class="row">
        <button onclick="promptLabelPath()">Set Label Path…</button>
      </div>
    </div>
  </div>
</div>

<script>
const idx = {{ idx }};
const total = {{ total }};
const remaining = {{ remaining }};
const imgUrl = {{ img_url|tojson }};
const stateUrl = {{ state_url|tojson }};
const actionUrl = {{ action_url|tojson }};
const hitUrl = {{ hit_url|tojson }};
const bboxHitUrl = {{ bbox_hit_url|tojson }};
const CANVAS_MAX_W = {{ canvas_w }};
const HANDLE_R = {{ handle_r }};
const DEFAULT_BOX = {{ default_box }};
const imagesRoot = {{ images_root|tojson }};
const labelsRoot = {{ labels_root|tojson }};
const splitName = {{ split|tojson }};
const hasData = {{ has_data|tojson }};

let toggling = false;
document.addEventListener('keydown', (e)=>{ if(e.key==='t' || e.key==='T') toggling=true; });
document.addEventListener('keyup',   (e)=>{ if(e.key==='t' || e.key==='T') toggling=false; });

const raw = document.getElementById('raw');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let W=0, H=0, scale=1.0;
let kps = { nose: {x:0,y:0,v:0}, L:{x:0,y:0,v:0}, R:{x:0,y:0,v:0} };
let bbox = null; // {cx,cy,w,h} normalized
let bboxEdit = false;
let bboxDrag = null;

function drawAll(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(raw, 0, 0, canvas.width, canvas.height);

  // bbox
  if (bbox){
    const x1 = (bbox.cx - bbox.w/2) * W * scale;
    const y1 = (bbox.cy - bbox.h/2) * H * scale;
    const x2 = (bbox.cx + bbox.w/2) * W * scale;
    const y2 = (bbox.cy + bbox.h/2) * H * scale;
    ctx.lineWidth = bboxEdit ? 3 : 2;
    ctx.strokeStyle = bboxEdit ? 'rgb(0,200,0)' : 'rgb(0,255,0)';
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  }

  // vector nose -> ear-mid
  if (kps.nose.v>0 && kps.L.v>0 && kps.R.v>0) {
    const mx = ((kps.L.x + kps.R.x)/2) * W * scale;
    const my = ((kps.L.y + kps.R.y)/2) * H * scale;
    const nx = kps.nose.x * W * scale;
    const ny = kps.nose.y * H * scale;
    ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(mx, my);
    ctx.lineWidth = 2; ctx.strokeStyle = 'yellow'; ctx.stroke();
  }

  // kps
  function dot(kp,color){
    if(kp.v<=0) return;
    const px = kp.x*W*scale, py=kp.y*H*scale;
    ctx.beginPath(); ctx.arc(px,py,6,0,2*Math.PI);
    ctx.fillStyle=color; ctx.fill();
  }
  dot(kps.nose,'red'); dot(kps.L,'blue'); dot(kps.R,'blue');
}

async function loadState(){
  const res = await fetch(stateUrl);
  const data = await res.json();
  document.getElementById('lbltext').textContent = data.label_text || '(no label)';
  W = data.W; H = data.H;

  // set canvas size with fixed width
  scale = Math.min(1.0, CANVAS_MAX_W / W);
  canvas.width  = Math.round(W * scale);
  canvas.height = Math.round(H * scale);

  kps = data.kps;
  bbox = data.bbox; // may be null

  // fill forms
  ['n','l','r'].forEach((p)=>{
    const key = p==='n'?'nose':(p==='l'?'L':'R');
    document.getElementById(p+'x').value = kps[key].x.toFixed(6);
    document.getElementById(p+'y').value = kps[key].y.toFixed(6);
    document.getElementById(p+'v').value = kps[key].v.toFixed(0);
  });
  if(bbox){
    bcx.value = bbox.cx.toFixed(6);
    bcy.value = bbox.cy.toFixed(6);
    bw.value  = bbox.w.toFixed(6);
    bh.value  = bbox.h.toFixed(6);
  } else {
    bcx.value = ""; bcy.value=""; bw.value=""; bh.value="";
  }

  const labelInput = document.getElementById('labelPath');
  if (labelInput){
    labelInput.value = data.label_path || '';
    if (!data.label_exists){
      labelInput.classList.add('warning');
    } else {
      labelInput.classList.remove('warning');
    }
  }
  window.currentImagePath = data.image_path || '';

  drawAll();
}

raw.onload = ()=>{ drawAll(); };
if (hasData){
  raw.src = imgUrl;
  loadState();
} else {
  canvas.width = 640;
  canvas.height = 480;
  drawAll();
}

function toImgCoords(evt){
  const rect = canvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) / scale;
  const y = (evt.clientY - rect.top) / scale;
  return {x,y};
}

function toggleBBoxMode(){
  if(!hasData) return;
  bboxEdit = !bboxEdit;
  const btn = document.getElementById('bboxToggle');
  if(bboxEdit){
    btn.textContent = 'BBox Edit: On';
    btn.classList.add('active');
  } else {
    btn.textContent = 'BBox Edit: Off';
    btn.classList.remove('active');
  }
  drawAll();
}

function clamp01f(z){
  return Math.min(1, Math.max(0, z));
}

function deepCopyKps(obj){
  return {
    nose: {x: obj.nose.x, y: obj.nose.y, v: obj.nose.v},
    L: {x: obj.L.x, y: obj.L.y, v: obj.L.v},
    R: {x: obj.R.x, y: obj.R.y, v: obj.R.v},
  };
}

function bboxEdgesPx(box){
  return {
    x1: (box.cx - box.w / 2) * W,
    y1: (box.cy - box.h / 2) * H,
    x2: (box.cx + box.w / 2) * W,
    y2: (box.cy + box.h / 2) * H,
  };
}

function remapKps(startKps, startEdges, newEdges){
  const sx1 = startEdges.x1;
  const sy1 = startEdges.y1;
  const sx2 = startEdges.x2;
  const sy2 = startEdges.y2;
  const sw = Math.max(1, sx2 - sx1);
  const sh = Math.max(1, sy2 - sy1);
  const nx1 = newEdges.x1;
  const ny1 = newEdges.y1;
  const nx2 = newEdges.x2;
  const ny2 = newEdges.y2;
  const nw = Math.max(1, nx2 - nx1);
  const nh = Math.max(1, ny2 - ny1);

  const out = deepCopyKps(startKps);
  ['nose','L','R'].forEach((key)=>{
    const pt = startKps[key];
    const px = clamp01f(pt.x) * W;
    const py = clamp01f(pt.y) * H;
    let relX = (px - sx1) / sw;
    let relY = (py - sy1) / sh;
    relX = clamp01f(relX);
    relY = clamp01f(relY);
    const newPx = nx1 + relX * nw;
    const newPy = ny1 + relY * nh;
    out[key].x = clamp01f(newPx / W);
    out[key].y = clamp01f(newPy / H);
  });
  return out;
}

function updateFromEdges(edges){
  const width = Math.max(1, Math.abs(edges.x2 - edges.x1));
  const height = Math.max(1, Math.abs(edges.y2 - edges.y1));
  const x1 = Math.min(edges.x1, edges.x2);
  const y1 = Math.min(edges.y1, edges.y2);
  const x2 = x1 + width;
  const y2 = y1 + height;
  bbox = {
    cx: clamp01f(((x1 + x2) / 2) / W),
    cy: clamp01f(((y1 + y2) / 2) / H),
    w: clamp01f(width / W),
    h: clamp01f(height / H),
  };
}

function handleBBoxDrag(ev){
  if (!bboxDrag) return;
  const q = toImgCoords(ev);
  const clampX = (v)=>Math.min(W, Math.max(0, v));
  const clampY = (v)=>Math.min(H, Math.max(0, v));
  const minSize = 4;

  if (bboxDrag.mode === 'draw'){
    let x1 = clampX(bboxDrag.startPoint.x);
    let y1 = clampY(bboxDrag.startPoint.y);
    let x2 = clampX(q.x);
    let y2 = clampY(q.y);
    if (Math.abs(x2 - x1) < minSize) {
      x2 = x1 + Math.sign(x2 - x1 || 1) * minSize;
    }
    if (Math.abs(y2 - y1) < minSize) {
      y2 = y1 + Math.sign(y2 - y1 || 1) * minSize;
    }
    x1 = clampX(Math.min(x1, x2));
    x2 = clampX(Math.max(x1 + minSize, x2));
    y1 = clampY(Math.min(y1, y2));
    y2 = clampY(Math.max(y1 + minSize, y2));
    bboxDrag.currentEdges = {x1, y1, x2, y2};
    updateFromEdges(bboxDrag.currentEdges);
    if (bboxDrag.startEdges){
      kps = remapKps(bboxDrag.startKps, bboxDrag.startEdges, bboxDrag.currentEdges);
    } else {
      const earDx = Math.min(0.25, bbox.w * 0.3);
      kps = {
        nose: {x: bbox.cx, y: bbox.cy, v: 2},
        L: {x: clamp01f(bbox.cx - earDx), y: bbox.cy, v: 2},
        R: {x: clamp01f(bbox.cx + earDx), y: bbox.cy, v: 2},
      };
    }
    drawAll();
    return;
  }

  if (!bboxDrag.startEdges) return;
  const startEdges = bboxDrag.startEdges;
  let x1 = startEdges.x1;
  let y1 = startEdges.y1;
  let x2 = startEdges.x2;
  let y2 = startEdges.y2;

  if (bboxDrag.mode === 'move'){
    const dx = q.x - bboxDrag.startPoint.x;
    const dy = q.y - bboxDrag.startPoint.y;
    const dxClamped = Math.min(W - x2, Math.max(-x1, dx));
    const dyClamped = Math.min(H - y2, Math.max(-y1, dy));
    x1 += dxClamped;
    x2 += dxClamped;
    y1 += dyClamped;
    y2 += dyClamped;
  } else {
    if (bboxDrag.corner === 'tl'){ x1 = clampX(q.x); y1 = clampY(q.y); }
    if (bboxDrag.corner === 'tr'){ x2 = clampX(q.x); y1 = clampY(q.y); }
    if (bboxDrag.corner === 'bl'){ x1 = clampX(q.x); y2 = clampY(q.y); }
    if (bboxDrag.corner === 'br'){ x2 = clampX(q.x); y2 = clampY(q.y); }

    if (x2 - x1 < minSize){
      if (bboxDrag.corner && bboxDrag.corner.includes('l')){
        x1 = x2 - minSize;
      } else {
        x2 = x1 + minSize;
      }
    }
    if (y2 - y1 < minSize){
      if (bboxDrag.corner && bboxDrag.corner.includes('t')){
        y1 = y2 - minSize;
      } else {
        y2 = y1 + minSize;
      }
    }
    x1 = clampX(x1);
    x2 = clampX(x2);
    y1 = clampY(y1);
    y2 = clampY(y2);
  }

  bboxDrag.currentEdges = {x1, y1, x2, y2};
  updateFromEdges(bboxDrag.currentEdges);
  kps = remapKps(bboxDrag.startKps, startEdges, bboxDrag.currentEdges);
  drawAll();
}

async function finishBBoxDrag(){
  document.removeEventListener('mousemove', handleBBoxDrag);
  // mouseup listener was registered with once:true
  if (!bboxDrag || !bboxDrag.currentEdges){
    bboxDrag = null;
    return;
  }
  const edges = bboxDrag.currentEdges;
  bboxDrag = null;
  await fetch(actionUrl,{method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action:'bbox_from_pixels',
      x1: edges.x1, y1: edges.y1, x2: edges.x2, y2: edges.y2})});
  await loadState();
}

function handleKpDrag(ev){
  if (!draggingKP) return;
  const q = toImgCoords(ev);
  draggingKP.last = {x: q.x, y: q.y};
  const key = draggingKP.which;
  const nx = W > 0 ? clamp01f(q.x / W) : 0;
  const ny = H > 0 ? clamp01f(q.y / H) : 0;
  kps[key].x = nx;
  kps[key].y = ny;
  if (kps[key].v <= 0) {
    kps[key].v = 2;
  }
  drawAll();
}

async function finishKpDrag(){
  document.removeEventListener('mousemove', handleKpDrag);
  if (!draggingKP) return;
  const {which, last} = draggingKP;
  draggingKP = null;
  await fetch(actionUrl, {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action:'drag', which: which, x: last.x, y: last.y})});
  await loadState();
}

// ---------- KP interactions ----------
let draggingKP = null;
canvas.addEventListener('mousedown', async (e)=>{
  if(!hasData) return;
  const p = toImgCoords(e);

  // if T held, try KP toggle on click
  if (toggling) {
    const res = await fetch(hitUrl, {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({i: idx, x: p.x, y: p.y})});
    const data = await res.json();
    if(data.which){
      await fetch(actionUrl,{method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({i: idx, action:'toggle', which: data.which})});
      await loadState();
      return;
    }
  }

  if (bboxEdit){
    const bres = await fetch(bboxHitUrl, {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({i: idx, x: p.x, y: p.y, handle_px: HANDLE_R/scale})});
    const bh = await bres.json();
    if (bh.mode && bh.mode !== 'none'){
      const startBBox = bbox ? {cx:bbox.cx, cy:bbox.cy, w:bbox.w, h:bbox.h} : null;
      const startEdges = startBBox ? bboxEdgesPx(startBBox) : null;
      bboxDrag = {
        mode: bh.mode,
        corner: bh.corner || null,
        startPoint: p,
        startEdges: startEdges,
        startBBox: startBBox,
        startKps: deepCopyKps(kps),
        currentEdges: startEdges,
      };
      if (bh.mode === 'draw'){
        bboxDrag.currentEdges = null;
      }
      document.addEventListener('mousemove', handleBBoxDrag);
      document.addEventListener('mouseup', finishBBoxDrag, {once: true});
      return;
    }
  }

  // if not over bbox, try KP drag
  const res = await fetch(hitUrl, {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, x: p.x, y: p.y})});
  const data = await res.json();
  if(!data.which) return;
  draggingKP = {
    which: data.which,
    last: {x: p.x, y: p.y},
  };
  document.addEventListener('mousemove', handleKpDrag);
  document.addEventListener('mouseup', finishKpDrag, {once: true});
});

// double-click: seed nose and ears at the cursor
canvas.addEventListener('dblclick', async (e)=>{
  if(!hasData) return;
  const p = toImgCoords(e);
  await fetch(actionUrl, {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action:'seed', x: p.x, y: p.y, side: DEFAULT_BOX})});
  await loadState();
});

// --------- apply buttons / nav ----------
async function applyBBox(){
  if(!hasData) return;
  const cx = parseFloat(bcx.value), cy=parseFloat(bcy.value),
        w  = parseFloat(bw.value),  h =parseFloat(bh.value);
  await fetch(actionUrl,{method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action:'bbox_set', bbox:{cx:cx,cy:cy,w:w,h:h}})});
  await loadState();
}
async function applyKPs(){
  if(!hasData) return;
  const nx=parseFloat(document.getElementById('nx').value),
        ny=parseFloat(document.getElementById('ny').value),
        nv=parseInt(document.getElementById('nv').value);
  const lx=parseFloat(document.getElementById('lx').value),
        ly=parseFloat(document.getElementById('ly').value),
        lv=parseInt(document.getElementById('lv').value);
  const rx=parseFloat(document.getElementById('rx').value),
        ry=parseFloat(document.getElementById('ry').value),
        rv=parseInt(document.getElementById('rv').value);
  await fetch(actionUrl,{method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action:'set',
      set:{ nose:[nx,ny,nv], L:[lx,ly,lv], R:[rx,ry,rv] }})});
  await loadState();
}

async function cycle(dir){
  if(!hasData) return;
  await fetch(actionUrl, {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action: (dir==='left'?'perm_left':'perm_right')})});
  await loadState();
}

async function mark(kind){
  if(!hasData) return;
  await fetch(actionUrl, {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({i: idx, action: kind})});
  next();
}

function next(){ window.location = "{{ url_for('index') }}?i="+(idx+1)+"&img_dir="+encodeURIComponent(imagesRoot)+"&lbl_dir="+encodeURIComponent(labelsRoot)+"&split="+encodeURIComponent(splitName); }
function prev(){ window.location = "{{ url_for('index') }}?i="+(idx-1)+"&img_dir="+encodeURIComponent(imagesRoot)+"&lbl_dir="+encodeURIComponent(labelsRoot)+"&split="+encodeURIComponent(splitName); }

function promptLabelPath(){
  if(!hasData) return;
  const input = document.getElementById('labelPath');
  let current = input ? input.value : '';
  if (!current) {
    const imgPath = window.currentImagePath || '';
    if (imgPath) {
      const rel = imgPath.startsWith(imagesRoot) ? imgPath.slice(imagesRoot.length).replace(/^\//,'') : imgPath.split('/').pop();
      const base = labelsRoot.replace(/\/$/, '');
      const name = rel ? rel.replace(/\.[^.]+$/, '.txt') : 'label.txt';
      current = base + '/' + name;
    }
  }
  const nextPath = prompt('Label file path', current);
  if(nextPath){
    fetch(actionUrl,{method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({i: idx, action:'set_label_path', path: nextPath})}).then(()=>loadState());
  }
}
</script>

</body>
</html>
"""

def get_counts():
    ok = read_list(LOG_OK)
    skip = read_list(LOG_SKIP)
    total = len(PAIRS)
    reviewed = ok | skip
    remaining = total - len(reviewed)
    return total, len(reviewed), remaining, ok, skip

def _resolve_index(target: Optional[str]) -> int:
    if not PAIRS:
        return 0
    if target is None:
        return max(0, min(len(PAIRS) - 1, first_unreviewed_index()))
    try:
        idx = int(target)
        return max(0, min(len(PAIRS) - 1, idx))
    except ValueError:
        path = Path(target)
        if not path.is_absolute():
            path = (CURRENT_IMAGES_ROOT / path).resolve()
        for j, pair in enumerate(PAIRS):
            if pair.img.resolve() == path:
                return j
        return 0


@APP.route("/")
def index():
    img_dir_param = request.args.get("img_dir")
    lbl_dir_param = request.args.get("lbl_dir")
    image_param = request.args.get("image")
    split_param = request.args.get("split")

    new_images_root = CURRENT_IMAGES_ROOT
    new_labels_root = CURRENT_LABELS_ROOT
    if img_dir_param:
        new_images_root = Path(img_dir_param).expanduser().resolve()
    if lbl_dir_param:
        new_labels_root = Path(lbl_dir_param).expanduser().resolve()
    if img_dir_param or lbl_dir_param or split_param:
        refresh_paths(new_images_root, new_labels_root, preview_split=split_param or CACHE_SPLIT)

    total = reviewed = remaining = 0
    context_pair = None
    img_url = ""
    state_url = ""
    idx = 0
    if PAIRS:
        idx = _resolve_index(request.args.get("i") or image_param)
        context_pair = PAIRS[idx]
        total, reviewed, remaining, ok, skip = get_counts()
        img_url = url_for("image_raw", i=idx)
        state_url = url_for("state_json", i=idx)

    if PAIRS:
        total, reviewed, remaining, ok, skip = get_counts()

    return render_template_string(
        TEMPLATE,
        idx=idx,
        total=total,
        remaining=remaining,
        imgname=context_pair.img.name if context_pair else "(no images)",
        img_url=img_url,
        state_url=state_url,
        action_url=url_for("action"),
        hit_url=url_for("hit_test"),
        bbox_hit_url=url_for("bbox_hit"),
        canvas_w=CANVAS_MAX_W,
        handle_r=HANDLE_R,
        default_box=DEFAULT_BOX,
        images_root=str(CURRENT_IMAGES_ROOT),
        labels_root=str(CURRENT_LABELS_ROOT),
        split=CACHE_SPLIT,
        has_data=bool(PAIRS),
        current_image=str(context_pair.img) if context_pair else ""
    )

@APP.route("/raw/<int:i>.png")
def image_raw(i: int):
    idx = _clamp_index(i)
    if idx is None:
        im = np.zeros((480,640,3), np.uint8)
        cv2.putText(im, "no image", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return send_file(io.BytesIO(img_to_png_bytes(im)), mimetype="image/png")
    pair = PAIRS[idx]
    im = cv2.imread(str(pair.img))
    if im is None:
        im = np.zeros((480,640,3), np.uint8)
        cv2.putText(im, "unreadable image", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return send_file(io.BytesIO(img_to_png_bytes(im)), mimetype="image/png")

@APP.route("/state/<int:i>.json")
def state_json(i: int):
    idx = _clamp_index(i)
    if idx is None:
        return jsonify(W=0, H=0, kps={"nose":{"x":0,"y":0,"v":0}, "L":{"x":0,"y":0,"v":0}, "R":{"x":0,"y":0,"v":0}}, bbox=None, label_text="", label_path="", label_exists=False, image_path="")
    pair = PAIRS[idx]
    tokens = read_label(pair.lbl)
    im = cv2.imread(str(pair.img))
    if im is None:
        H=W=0
    else:
        H,W = im.shape[:2]
    # default
    kps = {"nose":{"x":0,"y":0,"v":0}, "L":{"x":0,"y":0,"v":0}, "R":{"x":0,"y":0,"v":0}}
    bbox = None
    if tokens:
        t0 = tokens[0]
        if len(t0)>=5:
            cx,cy,w,h = map(float, t0[1:5])
            bbox = {"cx":cx,"cy":cy,"w":w,"h":h}
        if len(t0)==14:
            k = list(map(float, t0[5:]))
            kps["nose"]={"x":k[0],"y":k[1],"v":int(k[2])}
            kps["L"]   ={"x":k[3],"y":k[4],"v":int(k[5])}
            kps["R"]   ={"x":k[6],"y":k[7],"v":int(k[8])}
    label_text = "\n".join(" ".join(t) for t in tokens) if tokens else ""
    return jsonify(
        W=W,
        H=H,
        kps=kps,
        bbox=bbox,
        label_text=label_text,
        label_path=str(pair.lbl),
        label_exists=pair.lbl.exists(),
        image_path=str(pair.img),
    )

@APP.route("/hit_kp", methods=["POST"])
def hit_test():
    data = request.get_json(force=True)
    i = int(data.get("i", 0))
    x = float(data.get("x")); y=float(data.get("y"))
    idx = _clamp_index(i)
    which = None
    if idx is not None:
        which = nearest_kp(PAIRS[idx], x, y)
    return jsonify(which=which)

@APP.route("/hit_bbox", methods=["POST"])
def bbox_hit():
    """Decide bbox interaction mode:
       - if no bbox -> mode='draw'
       - if near a corner -> mode='resize' with corner id
       - if inside -> mode='move'
       - else 'none'
    """
    data = request.get_json(force=True)
    i = int(data.get("i", 0))
    x = float(data.get("x")); y=float(data.get("y"))
    handle_px = float(data.get("handle_px", 10.0))
    idx = _clamp_index(i)
    if idx is None:
        return jsonify(mode='none')
    pair = PAIRS[idx]
    bb = current_bbox_pixels(pair)
    if bb is None:
        return jsonify(mode='draw')
    x1,y1,x2,y2,W,H = bb
    # inside?
    if x1<=x<=x2 and y1<=y<=y2:
        # near corners?
        corners = {
            'tl':(x1,y1),'tr':(x2,y1),'bl':(x1,y2),'br':(x2,y2)
        }
        for name,(cx,cy) in corners.items():
            if (x-cx)**2 + (y-cy)**2 <= handle_px**2:
                return jsonify(mode='resize', corner=name)
        return jsonify(mode='move')
    # near a corner even if not inside
    corners = {
        'tl':(x1,y1),'tr':(x2,y1),'bl':(x1,y2),'br':(x2,y2)
    }
    for name,(cx,cy) in corners.items():
        if (x-cx)**2 + (y-cy)**2 <= handle_px**2:
            return jsonify(mode='resize', corner=name)
    return jsonify(mode='none')

@APP.route("/action", methods=["POST"])
def action():
    data = request.get_json(force=True)
    i = int(data.get("i", 0))
    idx = _clamp_index(i)
    if idx is None:
        return jsonify(ok=False, msg="no images loaded")
    pair = PAIRS[idx]
    act = data.get("action")

    if act == "seed":
        ok = commit_edit(pair, {"seed_template": {
            "x": float(data.get("x", 0.0)),
            "y": float(data.get("y", 0.0)),
            "side": float(data.get("side", DEFAULT_BOX))
        }})
        return jsonify(ok=ok)

    if act == "set_label_path":
        raw_path = data.get("path")
        if not raw_path:
            return jsonify(ok=False, msg="missing path")
        new_path = Path(raw_path).expanduser()
        if not new_path.is_absolute():
            new_path = (CURRENT_LABELS_ROOT / new_path).resolve()
        new_path.parent.mkdir(parents=True, exist_ok=True)
        pair.lbl = new_path
        return jsonify(ok=True, path=str(new_path))

    # progress
    if act in ("ok","skip"):
        logp = LOG_OK if act=="ok" else LOG_SKIP
        with logp.open("a", encoding="utf-8") as f:
            f.write(str(pair.img)+"\n")
        return jsonify(ok=True)

    # perms
    if act == "perm_left":
        ok = commit_edit(pair, {"perm": "cycle_left"})
        return jsonify(ok=ok)
    if act == "perm_right":
        ok = commit_edit(pair, {"perm": "cycle_right"})
        return jsonify(ok=ok)

    # KP edits
    if act == "toggle":
        ok = commit_edit(pair, {"toggle": data.get("which")})
        return jsonify(ok=ok)
    if act == "drag":
        ok = commit_edit(pair, {"drag": {"which": data.get("which"),
                                         "x": float(data.get("x")),
                                         "y": float(data.get("y"))}})
        return jsonify(ok=ok)
    if act == "set":
        ok = commit_edit(pair, {"set": data.get("set", {})})
        return jsonify(ok=ok)
    if act == "add_at":
        ok = commit_edit(pair, {"add_at": {"which": data.get("which"),
                                           "x": float(data.get("x")),
                                           "y": float(data.get("y"))}})
        return jsonify(ok=ok)

    # BBox edits
    if act == "bbox_set":
        ok = commit_edit(pair, {"bbox_set": data.get("bbox", {})})
        return jsonify(ok=ok)
    if act == "bbox_from_pixels":
        ok = commit_edit(pair, {"bbox_from_pixels": {
            "x1": float(data.get("x1")), "y1": float(data.get("y1")),
            "x2": float(data.get("x2")), "y2": float(data.get("y2"))
        }})
        return jsonify(ok=ok)

    return jsonify(ok=False, msg="unknown action")

def find_free_port(start=7860, tries=20):
    for p in range(start, start+tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', p))
                return p
            except OSError:
                continue
    return start

def main():
    print(f"[web] images: {CURRENT_IMAGES_ROOT}")
    print(f"[web] labels: {CURRENT_LABELS_ROOT}")
    port = int(os.environ.get("PORT", find_free_port(7860)))
    print(f"[web] listening on 0.0.0.0:{port}")
    try:
        from waitress import serve
        serve(APP, host="0.0.0.0", port=port)
    except Exception:
        APP.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
