'use strict';
//
// The annotator's front end. Python reads cubes, registers them and writes
// masks; everything between a mouse-down and a mouse-up happens here, because
// a server round trip per motion event cannot keep up with a brush — or with
// nudging an alignment.
//
// One leg = one group of cubes = ONE mask, held in the reference cube's pixel
// frame as a Uint8Array of W*H class ids (255 = unlabeled). That array is the
// reference cube's file byte for byte, so saving is a raw POST; the server
// shifts it into every other cube's frame on the way to disk.
//
// Each cube is fetched unshifted and drawn at its own (dx, dy), so nudging an
// alignment is a canvas translation, not a refetch.
//

const $ = (s) => document.querySelector(s);
const UNLABELED = 255;
const MAX_UNDO = 24;

const S = {
  groups: [], classes: [], gIdx: -1, g: null,
  W: 0, H: 0, labels: null, cubeIdx: 0,
  cols: new Array(256).fill(null),
  imgs: new Map(),                         // cube index -> {raw: ImageData, canvas}
  comp: null, compCtx: null, tmp: null, tmpCtx: null, checker: null,
  ov: null, ovCtx: null, ovData: null,
  view: { scale: 1, tx: 0, ty: 0 },
  tool: 'brush', prevTool: 'brush', size: 24, opacity: 0.55, showMask: true, active: null,
  vmode: 'normal', blinkOn: false, blinkTimer: 0,
  mode: 'single', chan: 0, rgb: [0, 0, 0], lo: 1, hi: 99, gamma: 1,
  undo: [], redo: [], strokeOld: null, strokeBox: null, strokeVal: 0,
  poly: [], boxDrag: null, alignDrag: null, cursor: null, last: null,
  dirty: false, alignDirty: false, painting: false, panning: false, space: false,
  imgToken: 0, frame: 0,
};

const view = $('#view');
const vctx = view.getContext('2d');

// ---------------------------------------------------------------- utilities
function setStatus(text, cls) {
  const el = $('#status');
  el.textContent = text;
  el.className = 'status' + (cls ? ' ' + cls : '');
}
function markDirty() { S.dirty = true; setStatus('unsaved changes', 'dirty'); }
function hexToRgb(hex) { return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)); }
function debounce(fn, ms) {
  let t = 0;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
function cube(i) { return S.g ? S.g.cubes[i] : null; }
function refIdx() { return S.g ? S.g.reference : 0; }
function bandLabel(c) {
  const cu = cube(S.cubeIdx);
  const w = cu && cu.wavelengths;
  return (w && w.length === cu.bands) ? `${c} · ${w[c]}nm` : String(c);
}
function clipBox() {
  const r = S.g && S.g.roi;
  return r ? { x0: r[0], y0: r[1], x1: r[2], y1: r[3] } : { x0: 0, y0: 0, x1: S.W, y1: S.H };
}

// out[y, x] = src[y - dy, x - dx]; anything shifted in becomes `fill`.
// Mirrors register.shift_image so both halves agree on the convention.
function warpMask(src, dx, dy, W, H, fill = UNLABELED) {
  const out = new Uint8Array(W * H).fill(fill);
  const ty0 = Math.max(0, dy), ty1 = Math.min(H, H + dy);
  const tx0 = Math.max(0, dx), tx1 = Math.min(W, W + dx);
  for (let y = ty0; y < ty1; y++) {
    const s = (y - dy) * W - dx, t = y * W;
    for (let x = tx0; x < tx1; x++) out[t + x] = src[s + x];
  }
  return out;
}

// ------------------------------------------------------------------ classes
function classById(id) { return S.classes.find((c) => c.id === id) || null; }

function applyClasses(classes) {
  S.classes = classes;
  S.cols = new Array(256).fill(null);
  for (const c of classes) S.cols[c.id] = hexToRgb(c.color);
  if (!classById(S.active)) S.active = classes.length ? classes[0].id : null;
  drawClassList();
  if (S.labels) { repaintOverlay(); schedule(); }
}

function drawClassList() {
  const ul = $('#classList');
  ul.innerHTML = '';
  S.classes.forEach((c, i) => {
    const li = document.createElement('li');
    li.className = c.id === S.active ? 'on' : '';
    li.innerHTML = `<span class="sw" style="background:${c.color}"></span>` +
      `<span class="num">${c.id}</span><span class="nm"></span>` +
      `<span class="key">${i < 10 ? (i + 1) % 10 : ''}</span>`;
    li.querySelector('.nm').textContent = c.name;
    li.onclick = () => { S.active = c.id; if (S.tool === 'eraser') setTool('brush'); drawClassList(); };
    ul.appendChild(li);
  });
  if (!S.classes.length) ul.innerHTML = '<li class="dim">no classes yet — click “edit”</li>';
}

// --------------------------------------------------------------- group list
function drawGroupList() {
  const ul = $('#groupList');
  ul.innerHTML = '';
  S.groups.forEach((g, i) => {
    const li = document.createElement('li');
    li.className = i === S.gIdx ? 'on' : '';
    li.title = `${g.cubes.length} cubes · ${g.width}×${g.height}`;
    const pct = g.labeled_fraction ? (g.labeled_fraction * 100).toFixed(0) + '%' : '';
    li.innerHTML = `<span class="flag">${g.done ? '✓' : g.cropped ? '⬒' : ''}</span>` +
                   `<span class="nm"></span><span class="pct">${pct}</span>`;
    li.querySelector('.nm').textContent = g.name;
    li.onclick = () => openGroup(i);
    ul.appendChild(li);
  });
  $('#groupCount').textContent = `${S.gIdx + 1} / ${S.groups.length}`;
}

// ----------------------------------------------------------- alignment view
function scoreClass(s) { return s >= 0.45 ? 'good' : s >= 0.25 ? 'ok' : 'bad'; }

function drawCubeList() {
  const ul = $('#cubeList');
  ul.innerHTML = '';
  if (!S.g) return;
  S.g.cubes.forEach((c, i) => {
    const li = document.createElement('li');
    li.className = (i === S.cubeIdx ? 'on ' : '') + (i === refIdx() ? 'isref' : '');
    const moved = (c.dx || c.dy) ? `${c.dx >= 0 ? '+' : ''}${c.dx},${c.dy >= 0 ? '+' : ''}${c.dy}` : '—';
    li.innerHTML = `<span class="ref" title="use as reference frame">ref</span>` +
      `<span class="nm"></span><span class="off">${moved}</span>` +
      `<span class="sc ${i === refIdx() ? '' : scoreClass(c.score)}">` +
      `${i === refIdx() ? '' : (c.score || 0).toFixed(2)}</span>`;
    li.querySelector('.nm').textContent = c.name;
    li.onclick = (e) => {
      if (e.target.classList.contains('ref')) setReference(i);
      else viewCube(i);
    };
    ul.appendChild(li);
  });

  const others = S.g.cubes.filter((_, i) => i !== refIdx());
  const worst = others.length ? Math.min(...others.map((c) => c.score || 0)) : 1;
  const moved = others.filter((c) => Math.abs(c.dx) > 2 || Math.abs(c.dy) > 2);
  const el = $('#alignVerdict');
  if (S.g.cropped) {
    el.textContent = 'cropped'; el.className = 'verdict good';
    $('#alignNote').innerHTML = `These cubes were cut to <b>${S.W}×${S.H}</b> and share a pixel ` +
      `grid — nothing left to align. <b>Re-crop</b> starts over from the raw cubes.`;
    $('#autoAlign').disabled = true;
    return;
  }
  $('#autoAlign').disabled = false;
  if (!S.g.aligned) {
    el.textContent = 'not checked'; el.className = 'verdict warn';
    $('#alignNote').innerHTML = 'Press <b>Auto-align</b> to check whether these cubes ' +
      'sit on top of each other.';
  } else {
    const cls = scoreClass(worst);
    el.textContent = cls === 'good' ? 'all match' : cls === 'ok' ? 'check it' : 'suspect';
    el.className = 'verdict ' + (cls === 'good' ? 'good' : cls === 'ok' ? 'warn' : 'bad');
    $('#alignNote').innerHTML =
      (moved.length ? `<b>${moved.length}</b> cube${moved.length > 1 ? 's' : ''} moved ` +
        `(up to ${Math.max(...moved.map((c) => Math.max(Math.abs(c.dx), Math.abs(c.dy))))} px). ` : 'Nothing moved more than 2 px. ') +
      `Worst match <b>${worst.toFixed(2)}</b>. Different excitations never score 1.00 — ` +
      `use <b>Diff</b> or <b>Blink</b> to judge it, and <b>Nudge</b> to correct.`;
  }
}

// The reference-frame rectangle every cube can see, recomputed locally so the
// outline tracks a nudge instead of waiting for the next save.
function recomputeOverlap() {
  if (!S.g) return;
  let x0 = 0, y0 = 0, x1 = S.W, y1 = S.H;
  for (const c of S.g.cubes) {
    x0 = Math.max(x0, c.dx); y0 = Math.max(y0, c.dy);
    x1 = Math.min(x1, S.W + c.dx); y1 = Math.min(y1, S.H + c.dy);
  }
  S.g.overlap_box = [x0, y0, Math.max(x0, x1), Math.max(y0, y1)];
  updateRoiNote();
}

function viewCube(i) {
  if (!S.g || i < 0 || i >= S.g.cubes.length || i === S.cubeIdx) return;
  S.cubeIdx = i;
  const c = cube(i);
  $('#groupSub').textContent = `${S.g.cubes.length} cubes · ${S.W}×${S.H} · viewing ${c.name}`;
  const last = c.bands - 1;
  for (const id of ['#chan', '#chR', '#chG', '#chB']) {
    const el = $(id);
    el.max = last;
    if (+el.value > last) el.value = last;
  }
  S.chan = Math.min(S.chan, last);
  $('#chan').value = S.chan;
  S.rgb = [+$('#chR').value, +$('#chG').value, +$('#chB').value];
  syncDisplayLabels();
  drawCubeList();
  ensureImages().then(() => { compose(); schedule(); });
}

function setReference(k) {
  if (!S.g || k === refIdx()) return;
  // The mask lives in the reference cube's frame, so changing reference means
  // re-expressing it — and every offset — in the new cube's frame.
  const kc = cube(k), dx = kc.dx, dy = kc.dy;
  if (S.labels) S.labels = warpMask(S.labels, -dx, -dy, S.W, S.H);
  if (S.g.roi) S.g.roi = [S.g.roi[0] - dx, S.g.roi[1] - dy, S.g.roi[2] - dx, S.g.roi[3] - dy];
  for (const c of S.g.cubes) { c.dx -= dx; c.dy -= dy; }
  S.g.reference = k;
  S.undo = []; S.redo = [];
  recomputeOverlap();
  repaintOverlay();
  drawCubeList();
  compose();
  schedule();
  markDirty();
  S.alignDirty = true;
  pushOffsets();
  setStatus(`reference is now ${kc.name} — mask moved with it`, 'dirty');
}

async function autoAlign() {
  if (!S.g) return;
  setStatus('registering cubes…', 'busy');
  $('#autoAlign').disabled = true;
  try {
    const oldRef = refIdx();
    const res = await fetch(`/api/group/${S.g.id}/align`, { method: 'POST' });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const { group } = await res.json();

    // The solver may pick a different reference; carry the mask across.
    if (group.reference !== oldRef && S.labels) {
      const o = group.cubes[oldRef];
      S.labels = warpMask(S.labels, o.dx, o.dy, S.W, S.H);
      if (group.roi) group.roi = [group.roi[0] + o.dx, group.roi[1] + o.dy,
                                  group.roi[2] + o.dx, group.roi[3] + o.dy];
      repaintOverlay();
      markDirty();
    }
    Object.assign(S.g, group);
    S.groups[S.gIdx] = Object.assign(S.groups[S.gIdx], group);
    // A sensible starting crop: everything every cube covers. Redraw it with R.
    if (!S.g.roi) { S.g.roi = S.g.overlap_box.slice(); pushOffsets(); }
    updateRoiNote();
    drawCubeList();
    await ensureImages();
    compose();
    schedule();
    const others = S.g.cubes.filter((_, i) => i !== refIdx());
    const worst = others.length ? Math.min(...others.map((c) => c.score)) : 1;
    setStatus(`aligned · reference ${cube(refIdx()).name} · worst ${worst.toFixed(2)}`,
              S.dirty ? 'dirty' : '');
  } catch (err) {
    setStatus('alignment failed — see console', 'bad');
    console.error(err);
  } finally {
    $('#autoAlign').disabled = false;
  }
}

// The alignment as the browser currently holds it. The mask is only
// meaningful next to the frame it was painted in, so this always travels with
// it rather than being left for the server to look up.
function alignPayload() {
  return {
    reference: S.g.reference,
    offsets: S.g.cubes.map((c) => [c.dx, c.dy]),
    scores: S.g.cubes.map((c) => c.score || 0),
    roi: S.g.roi || null,
  };
}

async function flushOffsets() {
  if (!S.g) return;
  try {
    await fetch(`/api/group/${S.g.id}/offsets`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alignPayload()),
    });
    S.alignDirty = false;
  } catch (err) { console.error(err); }
}
const pushOffsets = debounce(flushOffsets, 400);

// ------------------------------------------------------------ loading a leg
// Re-open the current leg from scratch. Needed after cropping, when the cubes,
// their size and the mask frame have all been replaced underneath us.
async function reopenGroup() {
  const i = S.gIdx;
  S.gIdx = -1;
  await openGroup(i, { save: false });
}

async function openGroup(i, { save = true } = {}) {
  if (i < 0 || i >= S.groups.length || i === S.gIdx) return;
  if (save && S.dirty) await saveMask({ quiet: true });

  S.gIdx = i;
  setStatus('loading leg…', 'busy');
  const g = await (await fetch(`/api/group/${S.groups[i].id}/meta`)).json();
  S.g = g;
  S.groups[i] = Object.assign(S.groups[i], g);
  S.W = g.width; S.H = g.height;
  S.cubeIdx = -1;

  $('#groupName').textContent = g.name;
  $('#doneChk').checked = !!g.done;

  for (const [id, frac] of [['#chan', 0.5], ['#chR', 0.78], ['#chG', 0.44], ['#chB', 0.12]]) {
    const el = $(id), last = g.cubes[0].bands - 1;
    el.max = last;
    if (+el.dataset.bands !== g.cubes[0].bands) { el.value = Math.round(last * frac); el.dataset.bands = g.cubes[0].bands; }
  }

  for (const key of ['comp', 'tmp', 'ov']) {
    S[key] = document.createElement('canvas');
    S[key].width = S.W; S[key].height = S.H;
  }
  S.compCtx = S.comp.getContext('2d', { willReadFrequently: true });
  S.tmpCtx = S.tmp.getContext('2d');
  S.ovCtx = S.ov.getContext('2d');
  S.ovData = S.ovCtx.createImageData(S.W, S.H);
  S.checker = makeChecker();

  S.labels = new Uint8Array(S.W * S.H).fill(UNLABELED);
  S.undo = []; S.redo = []; S.poly = []; S.dirty = false; S.imgs.clear();

  const res = await fetch(`/api/group/${g.id}/mask`);
  if (res.status === 200) {
    const buf = new Uint8Array(await res.arrayBuffer());
    if (buf.length === S.labels.length) S.labels.set(buf);
  }
  repaintOverlay();

  drawGroupList();
  viewCube(Math.min(Math.max(0, refIdx()), g.cubes.length - 1));
  fit();
  await ensureImages();
  compose();
  schedule();
  updateRoiNote();
  setStatus('ready');
}

// ------------------------------------------------------------------- images
// Cubes are fetched unshifted; the offset is applied when compositing, so a
// nudge costs a canvas translate rather than a round trip.
function imageUrl(i) {
  const p = new URLSearchParams({ cube: i, mode: S.mode, lo: S.lo, hi: S.hi });
  if (S.mode === 'single') p.set('c', Math.min(S.chan, cube(i).bands - 1));
  if (S.mode === 'rgb') {
    p.set('r', Math.min(S.rgb[0], cube(i).bands - 1));
    p.set('g', Math.min(S.rgb[1], cube(i).bands - 1));
    p.set('b', Math.min(S.rgb[2], cube(i).bands - 1));
  }
  return `/api/group/${S.g.id}/image?` + p;
}

async function fetchImage(i) {
  const res = await fetch(imageUrl(i));
  if (!res.ok) throw new Error(await res.text());
  const bmp = await createImageBitmap(await res.blob());
  const cv = document.createElement('canvas');
  cv.width = S.W; cv.height = S.H;
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  bmp.close();
  const rec = { raw: cx.getImageData(0, 0, S.W, S.H), canvas: cv, ctx: cx };
  applyGammaTo(rec);
  return rec;
}

// Which cubes the current view needs: the one being annotated, plus the
// reference whenever it is being compared against.
function needed() {
  const set = new Set([S.cubeIdx]);
  if (S.vmode !== 'normal') set.add(refIdx());
  return [...set].filter((i) => i >= 0);
}

async function ensureImages() {
  if (!S.g) return;
  const token = ++S.imgToken;
  const want = needed().filter((i) => !S.imgs.has(i));
  if (!want.length) return;
  setStatus('rendering…', 'busy');
  try {
    const recs = await Promise.all(want.map((i) => fetchImage(i)));
    if (token !== S.imgToken) return;         // a newer request already won
    want.forEach((i, k) => S.imgs.set(i, recs[k]));
    setStatus(S.dirty ? 'unsaved changes' : 'ready', S.dirty ? 'dirty' : '');
  } catch (err) {
    if (token === S.imgToken) setStatus('render failed', 'bad');
    console.error(err);
  }
}

// Gamma is a 256-entry lookup over pixels we already have, so the slider is
// instant — no server round trip to re-window an image in hand.
function applyGammaTo(rec) {
  const src = rec.raw.data;
  const out = new ImageData(S.W, S.H);
  const dst = out.data;
  const lut = new Uint8Array(256);
  const g = 1 / S.gamma;
  for (let v = 0; v < 256; v++) lut[v] = Math.round(255 * Math.pow(v / 255, g));
  for (let i = 0; i < src.length; i += 4) {
    dst[i] = lut[src[i]]; dst[i + 1] = lut[src[i + 1]]; dst[i + 2] = lut[src[i + 2]];
    dst[i + 3] = 255;
  }
  rec.ctx.putImageData(out, 0, 0);
}
function applyGamma() {
  for (const rec of S.imgs.values()) applyGammaTo(rec);
  compose();
  schedule();
}

function makeChecker(tile = 96) {
  const c = document.createElement('canvas');
  c.width = c.height = tile * 2;
  const x = c.getContext('2d');
  x.fillStyle = '#fff';
  x.fillRect(0, 0, tile, tile);
  x.fillRect(tile, tile, tile, tile);
  return S.compCtx.createPattern(c, 'repeat');
}

// One W*H canvas holding exactly what should appear under the mask.
function compose() {
  if (!S.comp || !S.g) return;
  const ctx = S.compCtx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, S.W, S.H);

  const v = S.imgs.get(S.cubeIdx), r = S.imgs.get(refIdx());
  if (!v) return;
  const c = cube(S.cubeIdx);
  const solo = S.vmode === 'normal' || !r || S.cubeIdx === refIdx();

  if (solo) {
    ctx.drawImage(v.canvas, c.dx, c.dy);
  } else if (S.vmode === 'diff') {
    ctx.drawImage(r.canvas, 0, 0);
    ctx.globalCompositeOperation = 'difference';
    ctx.drawImage(v.canvas, c.dx, c.dy);
    ctx.globalCompositeOperation = 'source-over';
  } else if (S.vmode === 'blink') {
    if (S.blinkOn) ctx.drawImage(v.canvas, c.dx, c.dy);
    else ctx.drawImage(r.canvas, 0, 0);
  } else if (S.vmode === 'checker') {
    ctx.drawImage(v.canvas, c.dx, c.dy);
    const t = S.tmpCtx;
    t.setTransform(1, 0, 0, 1, 0, 0);
    t.globalCompositeOperation = 'source-over';
    t.clearRect(0, 0, S.W, S.H);
    t.drawImage(r.canvas, 0, 0);
    t.globalCompositeOperation = 'destination-in';
    t.fillStyle = S.checker;
    t.fillRect(0, 0, S.W, S.H);
    t.globalCompositeOperation = 'source-over';
    ctx.drawImage(S.tmp, 0, 0);
  }
}

function setVMode(m) {
  S.vmode = m;
  document.querySelectorAll('.vmode').forEach((b) => b.classList.toggle('on', b.dataset.vmode === m));
  clearInterval(S.blinkTimer);
  S.blinkTimer = 0;
  if (m === 'blink') {
    S.blinkTimer = setInterval(() => { S.blinkOn = !S.blinkOn; compose(); schedule(); }, 550);
  }
  ensureImages().then(() => { compose(); schedule(); });
}

// ------------------------------------------------------------------- overlay
function repaintOverlay() {
  const d = S.ovData.data, lab = S.labels;
  for (let i = 0, o = 0; i < lab.length; i++, o += 4) {
    const c = S.cols[lab[i]];
    if (c) { d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255; }
    else { d[o + 3] = 0; }
  }
  S.ovCtx.putImageData(S.ovData, 0, 0);
}
function flushOverlay(box) {
  if (!box) return;
  const x = Math.max(0, box.x0), y = Math.max(0, box.y0);
  const w = Math.min(S.W - 1, box.x1) - x + 1, h = Math.min(S.H - 1, box.y1) - y + 1;
  if (w > 0 && h > 0) S.ovCtx.putImageData(S.ovData, 0, 0, x, y, w, h);
}
function grow(box, x, y) {
  if (x < box.x0) box.x0 = x; if (x > box.x1) box.x1 = x;
  if (y < box.y0) box.y0 = y; if (y > box.y1) box.y1 = y;
}

// --------------------------------------------------------------- undo / redo
function snapshot() { return { kind: 'full', prev: S.labels.slice() }; }
function pushUndo(entry) {
  S.undo.push(entry);
  if (S.undo.length > MAX_UNDO) S.undo.shift();
  S.redo.length = 0;
}
function applyEntry(entry, redoStack) {
  if (entry.kind === 'full') {
    const cur = S.labels.slice();
    S.labels.set(entry.prev);
    repaintOverlay();
    redoStack.push({ kind: 'full', prev: cur });
  } else {
    const inverse = new Uint8Array(entry.idx.length);
    for (let k = 0; k < entry.idx.length; k++) {
      const i = entry.idx[k];
      inverse[k] = S.labels[i];
      S.labels[i] = entry.old[k];
      const c = S.cols[entry.old[k]], o = i * 4, d = S.ovData.data;
      if (c) { d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255; }
      else { d[o + 3] = 0; }
    }
    flushOverlay(entry.box);
    redoStack.push({ kind: 'diff', idx: entry.idx, old: inverse, box: entry.box });
  }
  markDirty();
  schedule();
}
function doUndo() { const e = S.undo.pop(); if (e) applyEntry(e, S.redo); }
function doRedo() { const e = S.redo.pop(); if (e) applyEntry(e, S.undo); }

// -------------------------------------------------------------------- paint
function setPixel(i, val) {
  const prev = S.labels[i];
  if (prev === val) return;
  if (S.strokeOld && !S.strokeOld.has(i)) S.strokeOld.set(i, prev);
  S.labels[i] = val;
  const c = S.cols[val], o = i * 4, d = S.ovData.data;
  if (c) { d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255; }
  else { d[o + 3] = 0; }
}

function dab(cx, cy, r, val, box) {
  const b = clipBox();
  const x0 = Math.max(b.x0, Math.floor(cx - r)), x1 = Math.min(b.x1 - 1, Math.ceil(cx + r));
  const y0 = Math.max(b.y0, Math.floor(cy - r)), y1 = Math.min(b.y1 - 1, Math.ceil(cy + r));
  const rr = r * r;
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy, span = rr - dy * dy;
    if (span < 0) continue;
    const half = Math.sqrt(span), row = y * S.W;
    const ax = Math.max(x0, Math.ceil(cx - half - 0.5));
    const bx = Math.min(x1, Math.floor(cx + half - 0.5));
    for (let x = ax; x <= bx; x++) setPixel(row + x, val);
    if (bx >= ax) { grow(box, ax, y); grow(box, bx, y); }
  }
}

function strokeTo(x, y, val) {
  const r = S.size / 2, box = S.strokeBox, p = S.last;
  if (!p) dab(x, y, r, val, box);
  else {
    const dx = x - p.x, dy = y - p.y;
    const n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(0.6, r * 0.35)));
    for (let k = 1; k <= n; k++) dab(p.x + dx * k / n, p.y + dy * k / n, r, val, box);
  }
  S.last = { x, y };
  flushOverlay(box);
  schedule();
}

function beginStroke(val) {
  S.strokeOld = new Map();
  S.strokeBox = { x0: 1e9, y0: 1e9, x1: -1e9, y1: -1e9 };
  S.strokeVal = val;
  S.last = null;
}
function endStroke() {
  if (!S.strokeOld) return;
  if (S.strokeOld.size) {
    const idx = new Int32Array(S.strokeOld.size), old = new Uint8Array(S.strokeOld.size);
    let k = 0;
    for (const [i, v] of S.strokeOld) { idx[k] = i; old[k] = v; k++; }
    pushUndo({ kind: 'diff', idx, old, box: S.strokeBox });
    markDirty();
  }
  S.strokeOld = null; S.strokeBox = null; S.last = null;
}

// Even-odd scanline fill, sampled at row centres so the polygon covers exactly
// the pixels whose centres are inside it.
function fillPolygon(pts, val) {
  if (pts.length < 3) return;
  pushUndo(snapshot());
  S.strokeOld = null;
  const b = clipBox();
  let minY = S.H, maxY = -1;
  for (const p of pts) { minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); }
  minY = Math.max(b.y0, Math.floor(minY));
  maxY = Math.min(b.y1 - 1, Math.ceil(maxY));

  const xs = [];
  for (let y = minY; y <= maxY; y++) {
    const cy = y + 0.5;
    xs.length = 0;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const a = pts[j], c = pts[i];
      if ((a.y <= cy) !== (c.y <= cy)) xs.push(a.x + (cy - a.y) / (c.y - a.y) * (c.x - a.x));
    }
    if (xs.length < 2) continue;
    xs.sort((m, n) => m - n);
    const row = y * S.W;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const ax = Math.max(b.x0, Math.ceil(xs[k] - 0.5));
      const bx = Math.min(b.x1 - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = ax; x <= bx; x++) setPixel(row + x, val);
    }
  }
  repaintOverlay();
  markDirty();
  schedule();
}

function updateRoiNote() {
  const r = S.g && S.g.roi;
  const ob = S.g && S.g.overlap_box;

  // The button carries the stage of the leg: crop it, or it is already cropped.
  const btn = $('#exportBtn');
  if (btn && S.g) {
    if (S.g.cropped) {
      btn.textContent = 'Re-crop';
      btn.classList.remove('primary');
      btn.title = 'Throw away this cropped leg and start again from the raw cubes';
    } else {
      const box = r || ob;
      btn.classList.add('primary');
      btn.textContent = box ? `Crop to box  ${box[2] - box[0]}×${box[3] - box[1]}` : 'Crop to box';
      btn.title = box
        ? `Cut every cube of this leg to ${box[2] - box[0]}×${box[3] - box[1]} and switch to them`
        : 'Align the cubes first, then draw a box to crop to';
    }
  }
  let t;
  if (S.g && S.g.cropped) {
    t = `<b>Cropped.</b> These are the ${S.W}×${S.H} cubes on disk, all on one grid — ` +
        `paint anywhere and press <b>Save</b>. One mask covers every cube.`;
  } else if (!S.g || !S.g.aligned) {
    t = 'Step 1: press <b>Auto-align</b>, then draw a box with <b>R</b> and press <b>Crop to box</b>.';
  } else if (r) {
    t = `Box <b>${r[2] - r[0]}×${r[3] - r[1]}</b> at (${r[0]}, ${r[1]}). ` +
        `Press <b>Crop to box</b> to cut every cube to it — then you paint. ` +
        `<b>R</b> and drag to redraw it first.`;
  } else {
    t = 'Draw a box with <b>R</b>, then press <b>Crop to box</b>. Without a box the whole overlap is used.';
  }
  if (!(S.g && S.g.cropped) && ob && (ob[0] > 0 || ob[1] > 0 || ob[2] < S.W || ob[3] < S.H)) {
    t += ` Dashed outline = all every cube covers (${ob[2] - ob[0]}×${ob[3] - ob[1]}); the box cannot go outside it.`;
  }
  $('#roiNote').innerHTML = t;
}


// ---------------------------------------------------------------- fill
// Paint bucket. Clicking an unlabeled pixel floods the connected run of
// unlabeled pixels and stops at anything already painted — so after drawing
// the nerve, one click above it makes all of that muscle.
//
// Scanline flood fill: whole rows at a time rather than pixel-by-pixel, which
// is what keeps a 1.4 M pixel region under a tenth of a second. Filling
// rewrites the pixel it tests, so a run can never be visited twice.
function floodFill(sx, sy, val) {
  const W = S.W, b = clipBox();
  sx = Math.floor(sx); sy = Math.floor(sy);
  if (sx < b.x0 || sx >= b.x1 || sy < b.y0 || sy >= b.y1) return 0;
  const target = S.labels[sy * W + sx];
  if (target === val) return 0;

  pushUndo(snapshot());
  S.strokeOld = null;
  let filled = 0;
  const stack = [sx, sy];
  while (stack.length) {
    const y = stack.pop(), x = stack.pop();
    const row = y * W;
    if (S.labels[row + x] !== target) continue;

    let x0 = x;
    while (x0 > b.x0 && S.labels[row + x0 - 1] === target) x0--;
    let x1 = x;
    while (x1 < b.x1 - 1 && S.labels[row + x1 + 1] === target) x1++;

    let up = false, down = false;
    for (let i = x0; i <= x1; i++) {
      setPixel(row + i, val);
      filled++;
      if (y > b.y0) {
        const hit = S.labels[row - W + i] === target;
        if (hit && !up) { stack.push(i, y - 1); up = true; } else if (!hit) up = false;
      }
      if (y < b.y1 - 1) {
        const hit = S.labels[row + W + i] === target;
        if (hit && !down) { stack.push(i, y + 1); down = true; } else if (!hit) down = false;
      }
    }
  }
  repaintOverlay();
  markDirty();
  schedule();
  return filled;
}

// Everything still unlabeled, in one go — the usual last step, when what is
// left over is all one class.
function fillRest() {
  if (!S.labels || S.active === null) { setStatus('pick a class first', 'bad'); return; }
  const cls = classById(S.active);
  const b = clipBox();
  let n = 0;
  for (let y = b.y0; y < b.y1; y++) {
    for (let x = b.x0, row = y * S.W; x < b.x1; x++) if (S.labels[row + x] === UNLABELED) n++;
  }
  if (!n) { setStatus('nothing left unlabeled'); return; }
  if (!confirm(`Label all ${n.toLocaleString()} remaining pixels as "${cls.name}"?`)) return;
  pushUndo(snapshot());
  S.strokeOld = null;
  for (let y = b.y0; y < b.y1; y++) {
    for (let x = b.x0, row = y * S.W; x < b.x1; x++) {
      if (S.labels[row + x] === UNLABELED) setPixel(row + x, S.active);
    }
  }
  repaintOverlay();
  markDirty();
  schedule();
  setStatus(`filled ${n.toLocaleString()} px as ${cls.name}`, 'dirty');
}

// --------------------------------------------------------------- view / draw
function fit() {
  const r = view.getBoundingClientRect();
  if (!S.W) return;
  const s = Math.min(r.width / S.W, r.height / S.H) * 0.97;
  S.view = { scale: s, tx: (r.width - S.W * s) / 2, ty: (r.height - S.H * s) / 2 };
  schedule();
}
function toImage(ev) {
  const r = view.getBoundingClientRect();
  return { x: (ev.clientX - r.left - S.view.tx) / S.view.scale,
           y: (ev.clientY - r.top - S.view.ty) / S.view.scale };
}
function schedule() {
  if (S.frame) return;
  S.frame = requestAnimationFrame(() => { S.frame = 0; draw(); });
}

function draw() {
  const r = view.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (view.width !== Math.round(r.width * dpr) || view.height !== Math.round(r.height * dpr)) {
    view.width = Math.round(r.width * dpr);
    view.height = Math.round(r.height * dpr);
  }
  vctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  vctx.fillStyle = '#0d0f13';
  vctx.fillRect(0, 0, r.width, r.height);
  if (!S.comp) return;

  const { scale, tx, ty } = S.view;
  vctx.imageSmoothingEnabled = scale < 1;
  vctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * tx, dpr * ty);
  vctx.drawImage(S.comp, 0, 0);
  if (S.showMask) {
    vctx.globalAlpha = S.opacity;
    vctx.drawImage(S.ov, 0, 0);
    vctx.globalAlpha = 1;
  }

  vctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pt = (x, y) => [x * scale + tx, y * scale + ty];
  const rect = (b, stroke, dash) => {
    const [x0, y0] = pt(b[0], b[1]), [x1, y1] = pt(b[2], b[3]);
    vctx.strokeStyle = stroke; vctx.lineWidth = 1.25; vctx.setLineDash(dash);
    vctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    vctx.setLineDash([]);
  };

  // The area every cube can see. Labels outside it are dropped on some cube.
  const ob = S.g && S.g.overlap_box;
  if (ob && (ob[0] > 0 || ob[1] > 0 || ob[2] < S.W || ob[3] < S.H)) rect(ob, 'rgba(120,160,255,.55)', [7, 5]);

  const box = S.boxDrag ? S.boxDrag.box : (S.g && S.g.roi);
  if (box) {
    rect(box, '#ffcc66', [4, 3]);
    const [x0, y0] = pt(box[0], box[1]);
    vctx.fillStyle = 'rgba(255,204,102,.9)';
    vctx.font = '11px -apple-system, sans-serif';
    vctx.fillText(`${box[2] - box[0]}×${box[3] - box[1]}`, x0 + 4, y0 - 4);
  }

  if (S.tool === 'polygon' && S.poly.length) {
    vctx.strokeStyle = '#ffffff'; vctx.lineWidth = 1.25; vctx.setLineDash([5, 4]);
    vctx.beginPath();
    vctx.moveTo(...pt(S.poly[0].x, S.poly[0].y));
    for (const p of S.poly.slice(1)) vctx.lineTo(...pt(p.x, p.y));
    if (S.cursor) vctx.lineTo(...pt(S.cursor.x, S.cursor.y));
    vctx.closePath();
    vctx.stroke();
    vctx.setLineDash([]);
    const cls = classById(S.active);
    for (const p of S.poly) {
      const [px, py] = pt(p.x, p.y);
      vctx.fillStyle = cls ? cls.color : '#fff';
      vctx.beginPath(); vctx.arc(px, py, 3.2, 0, 7); vctx.fill();
      vctx.strokeStyle = '#000'; vctx.lineWidth = 1; vctx.stroke();
    }
  }

  if (S.cursor && (S.tool === 'brush' || S.tool === 'eraser') && !S.panning) {
    const cls = S.tool === 'eraser' ? null : classById(S.active);
    vctx.beginPath();
    vctx.arc(...pt(S.cursor.x, S.cursor.y), Math.max(1.5, S.size / 2 * scale), 0, 7);
    vctx.strokeStyle = cls ? cls.color : '#ffffff';
    vctx.lineWidth = 1.5; vctx.stroke();
    vctx.strokeStyle = 'rgba(0,0,0,.65)'; vctx.lineWidth = 0.75; vctx.stroke();
  }
}

// -------------------------------------------------------------- interaction
view.addEventListener('contextmenu', (e) => e.preventDefault());

view.addEventListener('pointerdown', (ev) => {
  if (!S.labels) return;
  // Capture keeps a drag alive past the canvas edge; a synthetic event has no
  // live pointer to capture, so a failure here is not fatal.
  try { view.setPointerCapture(ev.pointerId); } catch (_) {}
  const p = toImage(ev);

  if (ev.button === 1 || S.space) {
    S.panning = { x: ev.clientX, y: ev.clientY, tx: S.view.tx, ty: S.view.ty };
    return;
  }
  if (S.tool === 'align' && ev.button === 0) {
    const c = cube(S.cubeIdx);
    S.alignDrag = { x: p.x, y: p.y, dx: c.dx, dy: c.dy };
    return;
  }
  if (S.tool === 'box' && ev.button === 0) {
    S.boxDrag = { x: p.x, y: p.y, box: [p.x | 0, p.y | 0, p.x | 0, p.y | 0] };
    return;
  }
  if (S.tool === 'polygon' && ev.button === 0) { S.poly.push(p); schedule(); return; }

  if (S.tool === 'fill') {
    // right-click pours "unlabeled", which rubs a whole region out again
    const val = ev.button === 2 ? UNLABELED : S.active;
    if (val === null) { setStatus('pick a class first', 'bad'); return; }
    const n = floodFill(p.x, p.y, val);
    const cls = classById(val);
    setStatus(n ? `filled ${n.toLocaleString()} px as ${cls ? cls.name : 'unlabeled'}`
                : 'nothing to fill there — that area is already this class', n ? 'dirty' : '');
    return;
  }

  const erase = ev.button === 2 || S.tool === 'eraser';
  if (!erase && S.active === null) { setStatus('pick a class first', 'bad'); return; }
  S.painting = true;
  beginStroke(erase ? UNLABELED : S.active);
  strokeTo(p.x, p.y, S.strokeVal);
});

view.addEventListener('pointermove', (ev) => {
  const p = toImage(ev);
  S.cursor = p;

  if (S.panning) {
    S.view.tx = S.panning.tx + (ev.clientX - S.panning.x);
    S.view.ty = S.panning.ty + (ev.clientY - S.panning.y);
    schedule();
    return;
  }
  if (S.alignDrag) {
    const c = cube(S.cubeIdx);
    c.dx = Math.round(S.alignDrag.dx + (p.x - S.alignDrag.x));
    c.dy = Math.round(S.alignDrag.dy + (p.y - S.alignDrag.y));
    recomputeOverlap(); compose(); drawCubeList(); schedule();
    return;
  }
  if (S.boxDrag) {
    // Clamped to the overlap: outside it at least one cube has no pixels, and
    // a crop that reaches past it would have to invent them.
    const o = S.g.overlap_box || [0, 0, S.W, S.H];
    const cx = (v) => Math.max(o[0], Math.min(o[2], v));
    const cy = (v) => Math.max(o[1], Math.min(o[3], v));
    const x0 = cx(Math.min(S.boxDrag.x, p.x)), x1 = cx(Math.max(S.boxDrag.x, p.x));
    const y0 = cy(Math.min(S.boxDrag.y, p.y)), y1 = cy(Math.max(S.boxDrag.y, p.y));
    S.boxDrag.box = [Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)];
    schedule();
    return;
  }
  if (S.painting) strokeTo(p.x, p.y, S.strokeVal);
  else schedule();

  if (S.labels) {
    const x = Math.floor(p.x), y = Math.floor(p.y);
    let txt = '—';
    if (x >= 0 && y >= 0 && x < S.W && y < S.H) {
      const v = S.labels[y * S.W + x], c = classById(v);
      txt = `${x}, ${y}  ·  ${v === UNLABELED ? 'unlabeled' : (c ? `${v} ${c.name}` : v)}`;
    }
    $('#readout').textContent = txt;
  }
});

function release() {
  if (S.panning) { S.panning = false; return; }
  if (S.alignDrag) {
    S.alignDrag = null;
    S.alignDirty = true;
    pushOffsets();
    return;
  }
  if (S.boxDrag) {
    const b = S.boxDrag.box;
    S.boxDrag = null;
    if (b[2] - b[0] > 4 && b[3] - b[1] > 4) { S.g.roi = b; pushOffsets(); }
    updateRoiNote();
    schedule();
    return;
  }
  if (S.painting) { S.painting = false; endStroke(); }
}
view.addEventListener('pointerup', release);
view.addEventListener('pointercancel', release);
view.addEventListener('pointerleave', () => { S.cursor = null; schedule(); });

view.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const r = view.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const s = Math.min(40, Math.max(0.03, S.view.scale * Math.exp(-ev.deltaY * 0.0015)));
  const f = s / S.view.scale;
  S.view.tx = mx - (mx - S.view.tx) * f;
  S.view.ty = my - (my - S.view.ty) * f;
  S.view.scale = s;
  schedule();
}, { passive: false });

window.addEventListener('resize', () => schedule());

// ----------------------------------------------------------------- keyboard
function nudge(ddx, ddy) {
  if (!S.g || S.cubeIdx === refIdx()) {
    setStatus('the reference cube defines the frame — nudge another cube', 'bad');
    return;
  }
  const c = cube(S.cubeIdx);
  c.dx += ddx; c.dy += ddy;
  recomputeOverlap(); compose(); drawCubeList(); schedule();
  S.alignDirty = true;
  pushOffsets();
}

window.addEventListener('keydown', (ev) => {
  // Not every keydown target is an element with .matches (document and window
  // both reach here), and throwing would kill every shortcut.
  const t = ev.target;
  if (t && typeof t.matches === 'function' && t.matches('input, textarea, [contenteditable]')) return;
  const meta = ev.metaKey || ev.ctrlKey;

  if (meta && ev.key.toLowerCase() === 's') { ev.preventDefault(); saveMask(); return; }
  if (meta && ev.key.toLowerCase() === 'z') { ev.preventDefault(); ev.shiftKey ? doRedo() : doUndo(); return; }
  if (meta) return;

  const step = ev.shiftKey ? 10 : 1;
  switch (ev.key) {
    case ' ': S.space = true; ev.preventDefault(); return;
    case 'b': case 'B': setTool('brush'); return;
    case 'e': case 'E': setTool('eraser'); return;
    case 'p': case 'P': setTool('polygon'); return;
    case 'r': case 'R': setTool('box'); return;
    case 'g': case 'G': setTool('fill'); return;
    case 'a': case 'A': setTool(S.tool === 'align' ? S.prevTool : 'align'); return;
    case 'f': case 'F': fit(); return;
    case 'x': case 'X': toggleMask(); return;
    case 'v': case 'V': setVMode(S.vmode === 'normal' ? 'diff' : 'normal'); return;
    case 'c': case 'C': viewCube((S.cubeIdx + 1) % S.g.cubes.length); return;
    case '[': setSize(S.size - Math.max(1, Math.round(S.size * 0.2))); return;
    case ']': setSize(S.size + Math.max(1, Math.round(S.size * 0.2))); return;
    case ',': openGroup(S.gIdx - 1); return;
    case '.': openGroup(S.gIdx + 1); return;
    case 'Enter':
      if (S.tool === 'polygon' && S.poly.length >= 3 && S.active !== null) {
        fillPolygon(S.poly, S.active); S.poly = [];
      }
      return;
    case 'Backspace':
      if (S.tool === 'polygon') { S.poly.pop(); schedule(); ev.preventDefault(); }
      return;
    case 'Escape':
      S.poly = [];
      if (S.tool === 'box' && S.g && S.g.roi) { S.g.roi = null; pushOffsets(); updateRoiNote(); }
      schedule();
      return;
    case 'ArrowLeft':  ev.preventDefault(); S.tool === 'align' ? nudge(-step, 0) : stepBand(-1); return;
    case 'ArrowRight': ev.preventDefault(); S.tool === 'align' ? nudge(step, 0) : stepBand(1); return;
    case 'ArrowUp':    ev.preventDefault(); if (S.tool === 'align') nudge(0, -step); return;
    case 'ArrowDown':  ev.preventDefault(); if (S.tool === 'align') nudge(0, step); return;
  }
  if (/^[0-9]$/.test(ev.key)) {
    const i = (ev.key === '0') ? 9 : (+ev.key - 1);
    if (S.classes[i]) { S.active = S.classes[i].id; if (S.tool === 'eraser') setTool('brush'); drawClassList(); }
  }
});
window.addEventListener('keyup', (ev) => { if (ev.key === ' ') S.space = false; });

// ------------------------------------------------------------------ controls
function setTool(t) {
  if (S.tool !== 'align' && t === 'align') S.prevTool = S.tool;
  S.tool = t;
  if (t !== 'polygon') S.poly = [];
  document.querySelectorAll('.tool').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  $('#hint').textContent =
    t === 'polygon' ? 'click to place points · ⏎ fill · ⌫ remove point · esc cancel'
    : t === 'box' ? 'drag a region to confine painting · esc drops it'
    : t === 'fill' ? 'click an unlabelled area to pour the colour in · right-click pours it back out'
    : t === 'align' ? 'drag or ←↑↓→ to move this cube onto the reference · ⇧ = 10 px'
    : 'drag to paint · right-drag to erase';
  schedule();
}
function setSize(v) {
  S.size = Math.max(1, Math.min(200, Math.round(v)));
  $('#brushSize').value = S.size;
  $('#brushSizeVal').textContent = S.size;
  schedule();
}
function toggleMask() {
  S.showMask = !S.showMask;
  $('#hideBtn').textContent = S.showMask ? 'Hide mask' : 'Show mask';
  $('#hideBtn').classList.toggle('on', !S.showMask);
  schedule();
}
function stepBand(d) {
  if (S.mode !== 'single' || !S.g) return;
  const c = Math.max(0, Math.min(cube(S.cubeIdx).bands - 1, S.chan + d));
  if (c === S.chan) return;
  S.chan = c;
  $('#chan').value = c;
  syncDisplayLabels();
  invalidateImages();
}
function syncDisplayLabels() {
  $('#chanVal').textContent = bandLabel(S.chan);
  $('#chRVal').textContent = S.rgb[0];
  $('#chGVal').textContent = S.rgb[1];
  $('#chBVal').textContent = S.rgb[2];
  $('#loVal').textContent = S.lo + '%';
  $('#hiVal').textContent = S.hi + '%';
  $('#gammaVal').textContent = S.gamma.toFixed(2);
  $('#opacityVal').textContent = Math.round(S.opacity * 100) + '%';
}
function invalidateImages() {
  S.imgs.clear();
  ensureImages().then(() => { compose(); schedule(); });
}
const refetch = debounce(invalidateImages, 130);

function wireControls() {
  document.querySelectorAll('.tool').forEach((b) => b.onclick = () => setTool(b.dataset.tool));
  document.querySelectorAll('.vmode').forEach((b) => b.onclick = () => setVMode(b.dataset.vmode));
  document.querySelectorAll('.mode').forEach((b) => b.onclick = () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll('.mode').forEach((m) => m.classList.toggle('on', m === b));
    $('#singleCtl').classList.toggle('hidden', S.mode !== 'single');
    $('#rgbCtl').classList.toggle('hidden', S.mode !== 'rgb');
    invalidateImages();
  });

  $('#autoAlign').onclick = autoAlign;
  $('#brushSize').oninput = (e) => setSize(+e.target.value);
  $('#opacity').oninput = (e) => {
    S.opacity = +e.target.value / 100;
    $('#opacityVal').textContent = e.target.value + '%';
    schedule();
  };
  $('#chan').oninput = (e) => { S.chan = +e.target.value; syncDisplayLabels(); refetch(); };
  for (const [id, i] of [['#chR', 0], ['#chG', 1], ['#chB', 2]]) {
    $(id).oninput = (e) => { S.rgb[i] = +e.target.value; syncDisplayLabels(); refetch(); };
  }
  $('#lo').oninput = (e) => { S.lo = +e.target.value; syncDisplayLabels(); refetch(); };
  $('#hi').oninput = (e) => { S.hi = +e.target.value; syncDisplayLabels(); refetch(); };
  $('#gamma').oninput = (e) => { S.gamma = +e.target.value; syncDisplayLabels(); applyGamma(); };

  $('#fillRest').onclick = fillRest;
  $('#undoBtn').onclick = doUndo;
  $('#redoBtn').onclick = doRedo;
  $('#clearBtn').onclick = () => {
    if (!S.labels || !confirm('Erase every label on this leg?')) return;
    pushUndo(snapshot());
    S.labels.fill(UNLABELED);
    repaintOverlay();
    markDirty();
    schedule();
  };
  $('#boxAll').onclick = () => {
    if (!S.g) return;
    S.g.roi = (S.g.overlap_box || [0, 0, S.W, S.H]).slice();
    pushOffsets(); updateRoiNote(); schedule();
  };
  $('#boxClear').onclick = () => {
    if (!S.g) return;
    S.g.roi = null; pushOffsets(); updateRoiNote(); schedule();
  };
  $('#exportBtn').onclick = () => (S.g && S.g.cropped ? uncropLeg() : cropLeg());
  $('#fitBtn').onclick = fit;
  $('#hideBtn').onclick = toggleMask;
  $('#saveBtn').onclick = () => saveMask();
  $('#prevGroup').onclick = () => openGroup(S.gIdx - 1);
  $('#nextGroup').onclick = () => openGroup(S.gIdx + 1);
  $('#doneChk').onchange = async (e) => {
    if (!S.g) return;
    if (S.dirty) await saveMask({ quiet: true, done: e.target.checked });
    else {
      const res = await fetch(`/api/group/${S.g.id}/done`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ done: e.target.checked }),
      });
      const { group } = await res.json();
      Object.assign(S.g, group);
      S.groups[S.gIdx] = Object.assign(S.groups[S.gIdx], group);
      drawGroupList();
    }
  };
  $('#editClasses').onclick = openClassEditor;
}

// -------------------------------------------------------------------- saving
async function saveMask({ quiet = false, done = null } = {}) {
  if (!S.g || !S.labels) return;
  setStatus('saving…', 'busy');
  const p = new URLSearchParams({ align: JSON.stringify(alignPayload()) });
  if (done !== null) p.set('done', done ? 1 : 0);
  try {
    const res = await fetch(`/api/group/${S.g.id}/mask?` + p, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: S.labels,
    });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const body = await res.json();
    Object.assign(S.g, body.group_meta);
    S.groups[S.gIdx] = Object.assign(S.groups[S.gIdx], body.group_meta);
    drawGroupList();
    $('#doneChk').checked = !!body.group.done;
    S.dirty = false; S.alignDirty = false;
    setStatus(quiet ? 'saved'
      : `saved ${body.cubes.length} masks · ${body.group.labeled_px.toLocaleString()} px labeled`);
  } catch (err) {
    setStatus('save failed — see console', 'bad');
    console.error(err);
    throw err;
  }
}

// Write this leg out: every cube shifted onto the reference and cut to the
// box, plus the label image that now fits all of them. The mask is saved first
// so the server exports what is actually on screen.
// Crop: write every cube of this leg shifted onto the reference and cut to the
// box, then switch the tool onto those files. From here the cubes share a grid,
// so labelling is just painting — no alignment left to think about.
async function cropLeg() {
  if (!S.g) return;
  if (!S.g.can_export) {
    setStatus('started without --export-to, nowhere to write the crop', 'bad');
    return;
  }
  if (!S.g.aligned) {
    setStatus('align the cubes first', 'bad');
    return;
  }
  const btn = $('#exportBtn');
  btn.disabled = true;
  try {
    if (S.dirty) await saveMask({ quiet: true });
    setStatus('cropping every cube…', 'busy');
    const res = await fetch(`/api/group/${S.g.id}/export`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ align: alignPayload(), box: S.g.roi || null }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    const e = body.export;
    S.groups[S.gIdx] = Object.assign(S.groups[S.gIdx], body.group);
    await reopenGroup();
    setStatus(`cropped to ${e.output_size.width}×${e.output_size.height} — now paint, then Save`);
  } catch (err) {
    setStatus('crop failed — see console', 'bad');
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

async function uncropLeg() {
  if (!S.g || !S.g.cropped) return;
  if (!confirm('Throw away the cropped cubes and their labels, and go back to the raw cubes?')) return;
  setStatus('reverting…', 'busy');
  try {
    const res = await fetch(`/api/group/${S.g.id}/uncrop`, { method: 'POST' });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || res.statusText);
    S.groups[S.gIdx] = Object.assign(S.groups[S.gIdx], body.group);
    await reopenGroup();
    setStatus('back to the raw cubes — align, box, crop again');
  } catch (err) {
    setStatus('could not revert — see console', 'bad');
    console.error(err);
  }
}

setInterval(() => { if (S.dirty && !S.painting) saveMask({ quiet: true }).catch(() => {}); }, 60000);

window.addEventListener('beforeunload', (ev) => {
  if (!S.dirty || !S.labels) return;
  // A normal fetch is cancelled as the page tears down; a beacon is not.
  const q = new URLSearchParams({ align: JSON.stringify(alignPayload()) });
  navigator.sendBeacon(`/api/group/${S.g.id}/mask?` + q,
                       new Blob([S.labels], { type: 'application/octet-stream' }));
  ev.preventDefault();
  ev.returnValue = '';
});

// ------------------------------------------------------------- class editor
let PALETTE = [];

function classRow(cls) {
  const li = document.createElement('li');
  li.innerHTML = '<input type="number" min="0" max="254"><input type="text" placeholder="name">' +
                 '<input type="color"><button class="link" title="remove">✕</button>';
  const [num, name, col, del] = li.children;
  num.value = cls.id; name.value = cls.name; col.value = cls.color;
  del.onclick = () => li.remove();
  return li;
}

function openClassEditor() {
  const ul = $('#classEdit');
  ul.innerHTML = '';
  const seed = S.classes.length ? S.classes : [{ id: 0, name: 'background', color: PALETTE[0] }];
  seed.forEach((c) => ul.appendChild(classRow(c)));
  $('#nClasses').value = seed.length;
  $('#modalErr').textContent = '';
  $('#modal').classList.remove('hidden');
}

function wireModal() {
  $('#genClasses').onclick = () => {
    const n = Math.max(1, Math.min(254, +$('#nClasses').value || 1));
    const ul = $('#classEdit');
    const old = [...ul.children].map((li) => ({
      id: +li.children[0].value, name: li.children[1].value, color: li.children[2].value }));
    ul.innerHTML = '';
    for (let i = 0; i < n; i++) {
      ul.appendChild(classRow(old[i] || {
        id: i, name: i === 0 ? 'background' : `class ${i}`, color: PALETTE[i % PALETTE.length] }));
    }
  };
  $('#addClass').onclick = () => {
    const ul = $('#classEdit');
    const ids = [...ul.children].map((li) => +li.children[0].value);
    const next = ids.length ? Math.max(...ids) + 1 : 0;
    ul.appendChild(classRow({ id: next, name: `class ${next}`, color: PALETTE[next % PALETTE.length] }));
    $('#nClasses').value = ul.children.length;
  };
  $('#cancelClasses').onclick = () => {
    if (!S.classes.length) { $('#modalErr').textContent = 'define at least one class to start'; return; }
    $('#modal').classList.add('hidden');
  };
  $('#saveClasses').onclick = async () => {
    const rows = [...$('#classEdit').children].map((li) => ({
      id: +li.children[0].value, name: li.children[1].value.trim(), color: li.children[2].value }));
    if (!rows.length) { $('#modalErr').textContent = 'add at least one class'; return; }
    const res = await fetch('/api/classes', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classes: rows }),
    });
    const body = await res.json();
    if (!res.ok) { $('#modalErr').textContent = body.error || 'could not save'; return; }
    applyClasses(body.classes);
    $('#modal').classList.add('hidden');
    if (S.gIdx < 0 && S.groups.length) openGroup(0);
  };
}

// --------------------------------------------------------------------- boot
async function boot() {
  wireControls();
  wireModal();
  setTool('brush');
  setSize(24);

  const state = await (await fetch('/api/state')).json();
  PALETTE = state.palette;
  S.groups = state.groups;
  document.title = `annotate — ${state.input_dir.split('/').pop()}`;
  $('#groupSub').textContent = state.labels_dir;
  $('#groupName').title = `masks are written to ${state.labels_dir}`;
  applyClasses(state.classes);
  drawGroupList();
  syncDisplayLabels();

  if (!state.classes.length) {
    setStatus('define your classes');
    openClassEditor();
  } else {
    const next = S.groups.findIndex((g) => !g.done);
    openGroup(next < 0 ? 0 : next);
  }
}

boot().catch((err) => { setStatus('failed to start', 'bad'); console.error(err); });
