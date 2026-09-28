/* Linear unmixing — the browser half.
 *
 * Two panes over one leg at a time. "Regions" is the Nuance part: you look at
 * any one version of the leg and paint a region per structure. "Results" is
 * the part Nuance has not got: the same regions, re-measured and re-unmixed on
 * every version of the leg, side by side with the accuracy each one scored.
 *
 * The region map is the single piece of state worth protecting. It is a plain
 * Uint8Array of class indices at the leg's own pixel grid — the same bytes the
 * server stores — so painting, undo, saving and the endmember preview all work
 * on one array and never on a canvas's pixels.
 */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const S = {
  state: null,                       // /api/state
  leg: null,                         // /api/leg payload for the current leg
  rois: null,                        // Uint8Array(Y * X), 0 = unassigned
  sel: null,                         // the selection being edited
  active: 0,                         // which structure the brush paints
  tool: 'brush',
  brush: 7,
  version: null,                     // the version being looked at
  channels: null,                    // per-channel excitation / emission of it
  base: null,                        // Image of /api/view
  truth: null,                       // Image of /api/mask
  overlay: null,                     // offscreen canvas of the region map
  view: { mode: 'mean', band: 0, rgb: [0, 0, 0], lo: 1, hi: 99, gamma: 1,
          alpha: 0.7, rois: true, truth: false },
  cam: { scale: 1, tx: 0, ty: 0 },
  undo: [], redo: [],
  dirty: false,
  results: null,
  sort: 'accuracy_outside_rois',
  sortDir: 'desc',
  scope: 'leg',                      // 'leg' = this leg, 'all' = averaged
  allRows: null,                     // the flat cross-leg table
  allSort: 'accuracy_outside_rois',
  allDir: 'desc',
};

const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const ct = r.headers.get('content-type') || '';
  if (ct.includes('json')) {
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
  }
  if (!r.ok) throw new Error(r.statusText);
  return r;
};

const debounce = (fn, ms) => {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

// ===========================================================================
// Boot
// ===========================================================================
async function boot() {
  S.state = await api('/api/state');
  const sel = $('#leg');
  sel.innerHTML = S.state.legs
    .map(l => `<option value="${l.leg}">${l.leg}${l.has_results ? ' ✓' : ''}</option>`)
    .join('');
  sel.onchange = () => loadLeg(sel.value);
  $('#legPrev').onclick = () => stepLeg(-1);
  $('#legNext').onclick = () => stepLeg(+1);
  wireControls();
  wireCanvas();
  wireKeys();
  await loadLeg(S.state.legs[0].leg);
  pollJob();
}

function stepLeg(d) {
  const legs = S.state.legs.map(l => l.leg);
  const i = (legs.indexOf(S.leg.leg) + d + legs.length) % legs.length;
  $('#leg').value = legs[i];
  loadLeg(legs[i]);
}

async function loadLeg(leg) {
  await flush();
  S.leg = await api('/api/leg?leg=' + encodeURIComponent(leg));
  S.sel = S.leg.selection;
  S.active = 0;
  S.undo = []; S.redo = [];
  const buf = await (await api('/api/rois?leg=' + encodeURIComponent(leg))).arrayBuffer();
  S.rois = new Uint8Array(buf);

  $('#legMeta').textContent =
    `${S.leg.X} × ${S.leg.Y} px · ${S.leg.versions.length} versions`;
  $('#warn').textContent = (S.leg.warnings || []).join(' · ');

  $('#version').innerHTML = groupOptions(S.leg.versions);
  S.version = S.sel.reference && S.leg.versions.some(v => v.id === S.sel.reference)
    ? S.sel.reference : S.leg.versions[0].id;
  $('#version').value = S.version;

  $('#method').value = S.sel.solver.method;
  $('#stat').value = S.sel.solver.stat;
  $('#extra').value = S.sel.solver.extra;
  $('#normalize').checked = !!S.sel.solver.normalize_endmembers;

  buildOverlay();
  drawStructures();
  await loadVersion();
  fit();
  S.results = null;
  if ($('#pane-results').hidden === false) showResults();
  markClean();
  refreshSpectra();
}

function groupOptions(versions) {
  const groups = [];
  for (const v of versions) {
    let g = groups.find(g => g.name === v.group);
    if (!g) groups.push(g = { name: v.group, items: [] });
    g.items.push(v);
  }
  return groups.map(g => `<optgroup label="${g.name}">` + g.items.map(v =>
    `<option value="${v.id}">${v.label} · ${v.C} ch${v.duplicate_of ? ' (= ' + v.duplicate_of + ')' : ''}</option>`
  ).join('') + '</optgroup>').join('');
}

async function loadVersion() {
  const v = S.leg.versions.find(x => x.id === S.version);
  S.channels = await api(`/api/channels?leg=${enc(S.leg.leg)}&version=${S.version}`);
  const n = v.C - 1;
  $('#band').max = n;
  if (S.view.band > n) S.view.band = n >> 1;
  $('#band').value = S.view.band;
  $$('.rgb').forEach((el, i) => {
    el.max = n;
    if (S.view.rgb[i] > n) S.view.rgb[i] = Math.round(n * [0.75, 0.45, 0.15][i]);
    el.value = S.view.rgb[i];
  });
  bandLabels();
  S.truth = null;
  if (v.has_mask) {
    S.truth = await loadImage(`/api/mask?leg=${enc(S.leg.leg)}&version=${S.version}`);
  }
  await refreshBase();
}

const enc = encodeURIComponent;

function loadImage(src) {
  return new Promise((ok, no) => {
    const im = new Image();
    im.onload = () => ok(im);
    im.onerror = no;
    im.src = src;
  });
}

let baseToken = 0;
async function refreshBase() {
  const my = ++baseToken;
  const v = S.view;
  const url = `/api/view?leg=${enc(S.leg.leg)}&version=${S.version}&mode=${v.mode}`
    + `&band=${v.band}&rgb=${v.rgb.join(',')}&lo=${v.lo}&hi=${v.hi}&gamma=${v.gamma}`;
  const im = await loadImage(url);
  if (my !== baseToken) return;                 // a later request already won
  S.base = im;
  draw();
}

// ===========================================================================
// Structures
// ===========================================================================
function drawStructures() {
  const counts = S.rois ? countRois() : [];
  $('#structures').innerHTML = S.sel.structures.map((s, i) => `
    <li data-i="${i}" class="${i === S.active ? 'on' : ''}">
      <span class="key">${i + 1}</span>
      <input type="color" class="chip" value="${s.color}" data-i="${i}"
             style="padding:0;border:0;width:15px;height:15px;background:none">
      <span class="name" contenteditable data-i="${i}">${s.name}</span>
      <span class="num">${(counts[i] || 0).toLocaleString()}</span>
      <button class="kill" data-i="${i}" title="remove">✕</button>
    </li>`).join('');

  $$('#structures li').forEach(li => li.onmousedown = e => {
    if (e.target.matches('.kill, .chip, .name')) return;
    S.active = +li.dataset.i; drawStructures();
  });
  $$('#structures .chip').forEach(el => el.oninput = () => {
    S.sel.structures[+el.dataset.i].color = el.value;
    buildOverlay(); draw(); saveSelection(); refreshSpectra();
  });
  $$('#structures .name').forEach(el => el.onblur = () => {
    S.sel.structures[+el.dataset.i].name = el.textContent.trim() || 'component';
    saveSelection();
  });
  $$('#structures .kill').forEach(el => el.onclick = () => removeStructure(+el.dataset.i));

  const unmapped = S.sel.structures.filter(s => s.class_id == null).length;
  $('#structNote').textContent = unmapped
    ? `${unmapped} component(s) have no mask class: fitted, but never the predicted class.`
    : 'Each maps to a class of the label mask, so it can be scored.';
}

function countRois() {
  const c = new Array(S.sel.structures.length).fill(0);
  for (let i = 0; i < S.rois.length; i++) {
    const v = S.rois[i];
    if (v && v <= c.length) c[v - 1]++;
  }
  return c;
}

function addStructure() {
  S.sel.structures.push({ name: 'component', color: '#ffb300', class_id: null });
  S.active = S.sel.structures.length - 1;
  drawStructures(); saveSelection();
}

function removeStructure(i) {
  if (S.sel.structures.length <= 2) return;
  pushUndo();
  // Renumber the region map: the removed index goes, everything above slides
  // down, so the bytes keep pointing at the structure they were painted for.
  for (let p = 0; p < S.rois.length; p++) {
    const v = S.rois[p];
    if (v === i + 1) S.rois[p] = 0;
    else if (v > i + 1) S.rois[p] = v - 1;
  }
  S.sel.structures.splice(i, 1);
  S.active = Math.min(S.active, S.sel.structures.length - 1);
  drawStructures(); buildOverlay(); draw(); markDirty(); saveSelection();
}

// ===========================================================================
// Canvas: camera, painting, drawing
// ===========================================================================
const cv = $('#view');
const ctx = cv.getContext('2d');

function resize() {
  const r = cv.parentElement.getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function fit() {
  const r = cv.parentElement.getBoundingClientRect();
  const s = Math.min(r.width / S.leg.X, r.height / S.leg.Y) * 0.94;
  S.cam = { scale: s, tx: (r.width - S.leg.X * s) / 2, ty: (r.height - S.leg.Y * s) / 2 };
  draw();
}

function toImage(e) {
  const r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left - S.cam.tx) / S.cam.scale,
           y: (e.clientY - r.top - S.cam.ty) / S.cam.scale };
}

function buildOverlay() {
  if (!S.overlay) S.overlay = document.createElement('canvas');
  S.overlay.width = S.leg.X; S.overlay.height = S.leg.Y;
  const c = S.overlay.getContext('2d');
  const img = c.createImageData(S.leg.X, S.leg.Y);
  const rgb = S.sel.structures.map(s => [
    parseInt(s.color.slice(1, 3), 16), parseInt(s.color.slice(3, 5), 16),
    parseInt(s.color.slice(5, 7), 16)]);
  const d = img.data;
  for (let i = 0, p = 0; i < S.rois.length; i++, p += 4) {
    const v = S.rois[i];
    if (!v || v > rgb.length) continue;
    const c3 = rgb[v - 1];
    d[p] = c3[0]; d[p + 1] = c3[1]; d[p + 2] = c3[2]; d[p + 3] = 255;
  }
  c.putImageData(img, 0, 0);
}

let dragBox = null;

function draw() {
  if (!S.leg) return;
  const r = cv.parentElement.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.imageSmoothingEnabled = false;
  const { scale, tx, ty } = S.cam;
  const W = S.leg.X * scale, H = S.leg.Y * scale;
  if (S.base) ctx.drawImage(S.base, tx, ty, W, H);
  ctx.globalAlpha = S.view.alpha;
  if (S.view.truth && S.truth) ctx.drawImage(S.truth, tx, ty, W, H);
  if (S.view.rois && S.overlay) ctx.drawImage(S.overlay, tx, ty, W, H);
  ctx.globalAlpha = 1;
  if (dragBox) {
    const [x0, y0, x1, y1] = dragBox;
    ctx.strokeStyle = S.sel.structures[S.active].color;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(tx + Math.min(x0, x1) * scale, ty + Math.min(y0, y1) * scale,
                   Math.abs(x1 - x0) * scale, Math.abs(y1 - y0) * scale);
    ctx.setLineDash([]);
  }
}

function pushUndo() {
  S.undo.push(S.rois.slice());
  if (S.undo.length > 30) S.undo.shift();
  S.redo.length = 0;
}

function paintDisc(cx, cy, radius, value) {
  const { X, Y } = S.leg, r2 = radius * radius;
  const x0 = Math.max(0, Math.floor(cx - radius)), x1 = Math.min(X - 1, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius)), y1 = Math.min(Y - 1, Math.ceil(cy + radius));
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= r2) S.rois[y * X + x] = value;
    }
  }
}

function paintBox(x0, y0, x1, y1, value) {
  const { X, Y } = S.leg;
  const ax = Math.max(0, Math.floor(Math.min(x0, x1)));
  const bx = Math.min(X - 1, Math.ceil(Math.max(x0, x1)));
  const ay = Math.max(0, Math.floor(Math.min(y0, y1)));
  const by = Math.min(Y - 1, Math.ceil(Math.max(y0, y1)));
  for (let y = ay; y <= by; y++) for (let x = ax; x <= bx; x++) S.rois[y * X + x] = value;
}

function wireCanvas() {
  new ResizeObserver(resize).observe(cv.parentElement);

  let painting = null, panning = null, last = null;

  cv.addEventListener('contextmenu', e => e.preventDefault());

  cv.addEventListener('pointerdown', e => {
    cv.setPointerCapture(e.pointerId);
    const p = toImage(e);
    if (e.button === 1 || e.shiftKey) {
      panning = { x: e.clientX, y: e.clientY };
      return;
    }
    // The right button erases whatever tool is selected — the same habit the
    // annotation tool trained.
    const value = (e.button === 2 || S.tool === 'erase') ? 0 : S.active + 1;
    pushUndo();
    if (S.tool === 'box' && e.button !== 2) {
      dragBox = [p.x, p.y, p.x, p.y];
      painting = { box: true, value };
    } else {
      painting = { value };
      paintDisc(p.x, p.y, S.brush / 2, value);
      buildOverlay();
    }
    last = p;
    draw();
  });

  cv.addEventListener('pointermove', e => {
    const p = toImage(e);
    hud(p);
    if (panning) {
      S.cam.tx += e.clientX - panning.x;
      S.cam.ty += e.clientY - panning.y;
      panning = { x: e.clientX, y: e.clientY };
      return draw();
    }
    if (!painting) return;
    if (painting.box) {
      dragBox[2] = p.x; dragBox[3] = p.y;
      return draw();
    }
    // Interpolate along the drag so a fast stroke stays connected.
    const steps = Math.max(1, Math.round(Math.hypot(p.x - last.x, p.y - last.y)
                                         / Math.max(1, S.brush / 4)));
    for (let i = 1; i <= steps; i++) {
      paintDisc(last.x + (p.x - last.x) * i / steps,
                last.y + (p.y - last.y) * i / steps, S.brush / 2, painting.value);
    }
    last = p;
    buildOverlay(); draw();
  });

  const end = () => {
    if (painting && painting.box && dragBox) {
      paintBox(dragBox[0], dragBox[1], dragBox[2], dragBox[3], painting.value);
      dragBox = null;
      buildOverlay();
    }
    if (painting) { markDirty(); drawStructures(); saveRois(); refreshSpectra(); }
    painting = null; panning = null;
    draw();
  };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);

  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0016);
    const s = Math.min(60, Math.max(0.2, S.cam.scale * k));
    S.cam.tx = mx - (mx - S.cam.tx) * (s / S.cam.scale);
    S.cam.ty = my - (my - S.cam.ty) * (s / S.cam.scale);
    S.cam.scale = s;
    draw();
  }, { passive: false });
}

function hud(p) {
  const x = Math.floor(p.x), y = Math.floor(p.y);
  if (x < 0 || y < 0 || x >= S.leg.X || y >= S.leg.Y) return $('#hud').textContent = '';
  const v = S.rois[y * S.leg.X + x];
  const who = v ? ` · ${S.sel.structures[v - 1]?.name ?? 'class ' + v}` : '';
  $('#hud').textContent = `${x}, ${y}${who} · ${Math.round(S.cam.scale * 100)}%`;
}

// ===========================================================================
// Saving
// ===========================================================================
function markDirty() { S.dirty = true; $('#dirty').hidden = false; }
function markClean() { S.dirty = false; $('#dirty').hidden = true; }

const saveRois = debounce(async () => {
  await fetch('/api/rois?leg=' + enc(S.leg.leg), { method: 'POST', body: S.rois });
  markClean();
}, 500);

const saveSelection = debounce(async () => {
  S.sel.reference = S.version;
  S.sel.solver = {
    method: $('#method').value, stat: $('#stat').value, extra: $('#extra').value,
    normalize_endmembers: $('#normalize').checked,
  };
  await api('/api/selection?leg=' + enc(S.leg.leg),
            { method: 'POST', body: JSON.stringify(S.sel) });
}, 400);

async function flush() {
  if (!S.leg) return;
  await fetch('/api/rois?leg=' + enc(S.leg.leg), { method: 'POST', body: S.rois });
  S.sel.reference = S.version;
  await api('/api/selection?leg=' + enc(S.leg.leg),
            { method: 'POST', body: JSON.stringify(S.sel) });
  markClean();
}
addEventListener('beforeunload', () => {
  if (S.leg) navigator.sendBeacon('/api/rois?leg=' + enc(S.leg.leg), new Blob([S.rois]));
});

// ===========================================================================
// The endmember plot
// ===========================================================================
const refreshSpectra = debounce(async () => {
  let d;
  try {
    d = await api(`/api/spectra?leg=${enc(S.leg.leg)}&version=${S.version}`);
  } catch { return; }
  const cvs = $('#spectra'), c = cvs.getContext('2d');
  const dpr = devicePixelRatio || 1;
  const w = cvs.clientWidth || 240, h = 190;
  cvs.width = w * dpr; cvs.height = h * dpr;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);

  const drawn = d.spectra.filter((_, i) => d.counts[i] !== 0);
  const all = drawn.flat();
  if (!all.length) {
    $('#spectraNote').textContent = 'Draw a region to measure a spectrum.';
    return;
  }
  const lo = Math.min(...all), hi = Math.max(...all) || 1;
  const pad = { l: 4, r: 4, t: 6, b: 14 };
  const X = i => pad.l + i * (w - pad.l - pad.r) / Math.max(1, d.spectra[0].length - 1);
  const Y = v => h - pad.b - (v - lo) / (hi - lo || 1) * (h - pad.t - pad.b);

  // The excitation blocks of a concatenated cube, so a 217-channel spectrum
  // reads as seven excitations rather than one long squiggle.
  const style = getComputedStyle(document.body);
  c.strokeStyle = style.getPropertyValue('--line') || '#ccc';
  c.fillStyle = style.getPropertyValue('--dim') || '#888';
  c.font = '9px system-ui';
  let prev = null;
  d.excitation.forEach((ex, i) => {
    if (ex === prev) return;
    prev = ex;
    c.beginPath(); c.moveTo(X(i), pad.t); c.lineTo(X(i), h - pad.b); c.stroke();
    c.fillText(String(ex).replace(' nm', ''), X(i) + 2, h - 4);
  });

  d.spectra.forEach((s, k) => {
    if (d.counts[k] === 0) return;
    c.strokeStyle = d.colors[k];
    c.lineWidth = 1.4;
    c.beginPath();
    s.forEach((v, i) => (i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v))));
    c.stroke();
  });

  $('#spectraNote').textContent = d.names.map((n, i) =>
    d.counts[i] === undefined ? n : `${n} ${d.counts[i]}px`).join(' · ');
}, 350);

// ===========================================================================
// Controls
// ===========================================================================
function bandLabels() {
  const label = i => {
    const ex = S.channels.excitation[i], em = S.channels.emission_nm[i];
    return `${ex === 'white' ? 'white' : ex} / ${Math.round(em)}`;
  };
  $('#bandOut').textContent = label(+$('#band').value);
  $$('.rgb').forEach(el => el.nextElementSibling.textContent = label(+el.value));
}

function wireControls() {
  $('#addStructure').onclick = addStructure;

  $('#seed').onclick = async () => {
    if (!confirm('Replace the regions with a sample taken out of the label mask?')) return;
    const r = await api('/api/seed?leg=' + enc(S.leg.leg),
                        { method: 'POST',
                          body: JSON.stringify({ structures: S.sel.structures }) });
    S.rois = new Uint8Array(await r.arrayBuffer());
    pushUndo();
    buildOverlay(); draw(); drawStructures(); markDirty(); saveRois(); refreshSpectra();
  };

  $('#clear').onclick = () => {
    if (!S.rois.some(v => v)) return;
    if (!confirm('Throw away every region on this leg?')) return;
    pushUndo();
    S.rois.fill(0);
    buildOverlay(); draw(); drawStructures(); markDirty(); saveRois(); refreshSpectra();
  };

  $$('.tools [data-tool]').forEach(b => b.onclick = () => {
    S.tool = b.dataset.tool;
    $$('.tools [data-tool]').forEach(x => x.classList.toggle('on', x === b));
  });
  $('#brush').oninput = e => { S.brush = +e.target.value; $('#brushOut').textContent = S.brush; };

  $('#version').onchange = async () => {
    S.version = $('#version').value;
    await loadVersion(); saveSelection(); refreshSpectra();
  };

  $$('#modes button').forEach(b => b.onclick = () => {
    S.view.mode = b.dataset.mode;
    $$('#modes button').forEach(x => x.classList.toggle('on', x === b));
    $('#bandRow').hidden = S.view.mode !== 'band';
    $('#rgbRow').hidden = S.view.mode !== 'rgb';
    refreshBase();
  });
  $('#band').oninput = e => { S.view.band = +e.target.value; bandLabels(); refreshBase(); };
  $$('.rgb').forEach(el => el.oninput = () => {
    S.view.rgb[+el.dataset.i] = +el.value; bandLabels(); refreshBase();
  });
  ['lo', 'hi', 'gamma'].forEach(k => $('#' + k).oninput = e => {
    S.view[k] = +e.target.value;
    e.target.nextElementSibling.textContent = e.target.value;
    refreshBase();
  });
  $('#alpha').oninput = e => {
    S.view.alpha = +e.target.value / 100;
    e.target.nextElementSibling.textContent = e.target.value;
    draw();
  };
  $$('#overlays button').forEach(b => b.onclick = () => {
    S.view[b.dataset.ov] = !S.view[b.dataset.ov];
    b.classList.toggle('on', S.view[b.dataset.ov]);
    draw();
  });

  ['method', 'stat', 'extra', 'normalize'].forEach(id =>
    $('#' + id).onchange = () => { saveSelection(); refreshSpectra(); });

  $('#run').onclick = () => run([S.leg.leg]);
  $('#runAll').onclick = () => run(S.state.legs.map(l => l.leg));
  $('#cancel').onclick = () => fetch('/api/cancel', { method: 'POST' });

  $$('.tabs button').forEach(b => b.onclick = () => {
    $$('.tabs button').forEach(x => x.classList.toggle('on', x === b));
    const results = b.dataset.pane === 'results';
    $('#pane-regions').hidden = results;
    $('#pane-results').hidden = !results;
    if (results) showResults(); else resize();
  });

  $$('#resScope button').forEach(b => b.onclick = () => {
    $$('#resScope button').forEach(x => x.classList.toggle('on', x === b));
    S.scope = b.dataset.scope;
    showResults();
  });

  $('#resSort').onchange = () => {
    S.sort = $('#resSort').value;
    S.sortDir = S.sort.startsWith('residual') ? 'asc' : 'desc';  // a small residual is good
    renderResults();
  };
  $('#resImage').onchange = renderResults;
  $('#hideDup').onchange = () =>
    (S.scope === 'all' ? renderAllLegs() : renderResults());
  $('#modal').onclick = e => { if (e.target.id === 'modal') $('#modal').hidden = true; };
}

function wireKeys() {
  addEventListener('keydown', e => {
    if (e.target.isContentEditable || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    const meta = e.metaKey || e.ctrlKey;
    if (meta && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      const from = e.shiftKey ? S.redo : S.undo, to = e.shiftKey ? S.undo : S.redo;
      if (!from.length) return;
      to.push(S.rois.slice());
      S.rois = from.pop();
      buildOverlay(); draw(); drawStructures(); markDirty(); saveRois(); refreshSpectra();
      return;
    }
    if (meta && e.key.toLowerCase() === 's') { e.preventDefault(); return flush(); }
    if (e.key === 'Escape') { $('#modal').hidden = true; dragBox = null; draw(); return; }
    const k = e.key.toLowerCase();
    if (k >= '1' && k <= '9') {
      const i = +k - 1;
      if (i < S.sel.structures.length) { S.active = i; drawStructures(); }
    } else if (k === 'b' || k === 'r' || k === 'e') {
      $(`[data-tool="${{ b: 'brush', r: 'box', e: 'erase' }[k]}"]`).click();
    } else if (k === 'x') { $('[data-ov="rois"]').click();
    } else if (k === 'v') { $('[data-ov="truth"]').click();
    } else if (k === 'f') { fit();
    } else if (k === ',') { stepLeg(-1);
    } else if (k === '.') { stepLeg(+1);
    } else if (k === '[') { $('#brush').value = Math.max(1, S.brush - 2); $('#brush').oninput({ target: $('#brush') });
    } else if (k === ']') { $('#brush').value = Math.min(60, S.brush + 2); $('#brush').oninput({ target: $('#brush') }); }
  });
}

// ===========================================================================
// Running
// ===========================================================================
async function run(legs) {
  await flush();
  try {
    await api('/api/run', { method: 'POST', body: JSON.stringify({ legs }) });
  } catch (err) { return alert(err.message); }
  $('#progress').hidden = false;
  $('#run').disabled = $('#runAll').disabled = true;
  pollJob();
}

async function pollJob() {
  let j;
  try { j = await api('/api/job'); } catch { return; }
  if (j.idle) return;
  if (!j.finished) {
    $('#progress').hidden = false;
    $('#run').disabled = $('#runAll').disabled = true;
    const frac = (j.done_legs.length + j.i / Math.max(1, j.n)) / j.legs.length;
    $('#progress .bar i').style.width = Math.round(frac * 100) + '%';
    $('#progressText').textContent =
      `${j.leg} · ${j.i + 1}/${j.n} · ${j.version} · ${j.stage}`;
    return setTimeout(pollJob, 400);
  }
  $('#progress').hidden = true;
  $('#run').disabled = $('#runAll').disabled = false;
  if (j.error) alert('unmixing failed — ' + j.error);
  S.results = null;
  S.allRows = null;
  S.state = await api('/api/state');
  const cur = S.leg.leg;
  $('#leg').innerHTML = S.state.legs
    .map(l => `<option value="${l.leg}">${l.leg}${l.has_results ? ' ✓' : ''}</option>`).join('');
  $('#leg').value = cur;
  if (!$('#pane-results').hidden) showResults();
}

// ===========================================================================
// Results
// ===========================================================================
async function showResults() {
  if (S.scope === 'all') return showAllLegs();
  if (!S.results) {
    try {
      S.results = await api('/api/results?leg=' + enc(S.leg.leg));
    } catch {
      $('#resGrid').innerHTML =
        '<p class="note">This leg has not been unmixed yet — draw regions and press '
        + '<b>Unmix this leg</b>.</p>';
      $('#resTable').innerHTML = '';
      $('#resMeta').textContent = '';
      return;
    }
    const names = S.results.selection.structures.map(s => s.name)
      .concat(S.results.versions[0]?.extras || []);
    $('#resImage').innerHTML =
      '<option value="classmap">class map (argmax)</option>'
      + '<option value="residual">residual</option>'
      + names.map(n => `<option value="abundance_${n}">abundance · ${n}</option>`).join('');
  }
  renderResults();
}

const num = (x, n = 4) => (x == null || Number.isNaN(x)) ? '—' : (+x).toFixed(n);

// The regions a result was actually computed from, in pixels — so the picture
// of them can be checked against a number rather than taken on trust.
function roiCaption(r) {
  const px = r.versions?.[0]?.endmember_pixels || {};
  const parts = Object.entries(px).map(([k, n]) => `${k} ${n}`);
  return parts.length ? parts.join(' · ') : 'where the spectra came from';
}

function renderResults() {
  const r = S.results;
  if (!r) return;
  $('#resNote').textContent = '';
  $('#resGrid').hidden = false;
  $('#pictureField').hidden = false;
  $('#sortField').hidden = false;
  const pic = $('#resImage').value || 'classmap';
  const hideDup = $('#hideDup').checked;
  let rows = r.versions.filter(v => !(hideDup && v.duplicate_of));

  // One column spec drives the header, the cells and the sort, so clicking a
  // heading can order by anything the table shows — per-class IoU included,
  // which is the column that matters when accuracy is carried by the biggest
  // class.
  const names = Object.keys(r.versions[0]?.metrics?.all?.per_class || {});
  const COLS = [
    { k: 'label', h: 'version', text: true, get: v => v.label },
    { k: 'channels', h: 'ch', get: v => v.channels, fmt: v => v.channels },
    { k: 'accuracy_outside_rois', h: 'accuracy', bar: true,
      get: v => v.metrics?.outside_rois?.accuracy },
    { k: 'mean_iou_outside_rois', h: 'mean IoU', get: v => v.metrics?.outside_rois?.mean_iou },
    ...names.map(n => ({ k: `iou_${n}`, h: 'IoU ' + n,
                         get: v => v.metrics?.outside_rois?.per_class?.[n]?.iou })),
    { k: 'accuracy', h: 'acc (all px)', get: v => v.metrics?.all?.accuracy },
    // The fraction of the signal the model could not explain. The absolute
    // RMSE is in the cube's own units and a z-normalised cube's units are ~16x
    // a raw cube's, so only this one compares across versions.
    { k: 'residual_rel', h: 'resid frac', get: v => v.residual_rel,
      fmt: v => num(v.residual_rel, 3) },
    { k: 'seconds', h: 's', get: v => v.seconds, fmt: v => v.seconds },
  ];

  const col = COLS.find(c => c.k === S.sort);
  if (col) {
    const sign = S.sortDir === 'asc' ? 1 : -1;
    rows = [...rows].sort((a, b) => {
      const x = col.get(a), y = col.get(b);
      if (col.text) return sign * String(x ?? '').localeCompare(String(y ?? ''));
      // A version with no mask has no metrics; it sinks either way rather than
      // jumping to the top of an ascending sort.
      return sign * ((x ?? -Infinity) - (y ?? -Infinity));
    });
  }

  // The run stamp rides on every image URL: the paths are stable across runs,
  // so without it a cached copy would show the last run's regions next to this
  // run's numbers.
  const base = `/out/${enc(r.leg)}`;
  const stamp = '?v=' + encodeURIComponent(r.run);
  const truth = `<figure class="card ref"><img src="${base}/truth.png${stamp}">
      <figcaption class="cap"><b>truth</b><span>hand-painted mask</span></figcaption></figure>
    <figure class="card ref"><img src="${base}/rois_preview.png${stamp}">
      <figcaption class="cap"><b>regions</b><span>${roiCaption(r)}</span></figcaption></figure>`;

  $('#resGrid').innerHTML = truth + rows.map(v => {
    const acc = v.metrics?.outside_rois?.accuracy;
    return `<figure class="card" data-v="${v.version}">
      <img src="${base}/${v.version}/${pic}.png${stamp}" loading="lazy" alt="${v.version}">
      <figcaption class="cap"><b>${num(acc, 3)}</b> ${v.label}
        <span>${v.channels} ch${v.duplicate_of ? ' · = ' + v.duplicate_of : ''}</span>
      </figcaption></figure>`;
  }).join('');
  $$('#resGrid .card[data-v]').forEach(el => el.onclick = () => openDetail(el.dataset.v));

  $('#resTable').innerHTML = '<thead><tr>' + COLS.map(c =>
      `<th data-k="${c.k}"${c.k === S.sort ? ` data-dir="${S.sortDir}"` : ''}>${c.h}</th>`
    ).join('') + '</tr></thead><tbody>' + rows.map(v =>
      `<tr class="${v.duplicate_of ? 'dup' : ''}" data-v="${v.version}">` + COLS.map(c => {
        if (c.k === 'label') {
          return `<td>${v.label}${v.duplicate_of
            ? ` <span style="opacity:.6">= ${v.duplicate_of}</span>` : ''}</td>`;
        }
        const x = c.get(v);
        const bar = c.bar && x != null
          ? ` <i class="bar" style="width:${Math.round(x * 44)}px"></i>` : '';
        return `<td>${c.fmt ? c.fmt(v) : num(x)}${bar}</td>`;
      }).join('') + '</tr>').join('') + '</tbody>';
  $$('#resTable tbody tr').forEach(tr => tr.onclick = () => openDetail(tr.dataset.v));
  $$('#resTable th').forEach(th => th.onclick = () => {
    // Same habit as the report: a fresh column starts high-to-low (names A-Z),
    // and clicking the sorted column reverses it.
    const text = COLS.find(c => c.k === th.dataset.k)?.text;
    S.sortDir = S.sort === th.dataset.k
      ? (S.sortDir === 'asc' ? 'desc' : 'asc') : (text ? 'asc' : 'desc');
    S.sort = th.dataset.k;
    $('#resSort').value = [...$('#resSort').options].some(o => o.value === S.sort)
      ? S.sort : 'order';
    renderResults();
  });

  const scored = rows.filter(v => v.metrics);
  const best = scored.length ? Math.max(...scored.map(v => v.metrics.outside_rois.accuracy)) : NaN;
  $('#resMeta').textContent =
    `${r.leg} · ${rows.length} versions · run ${r.run.replace('T', ' ').replace('+00:00', ' UTC')}`
    + (scored.length ? ` · best ${num(best, 3)}` : '');
}

// ---------------------------------------------------------------------------
// All legs: the same table the report opens with — one row per version, every
// number averaged over the legs, with the per-structure IoU that an average
// over classes hides.
// ---------------------------------------------------------------------------
async function showAllLegs() {
  $('#resGrid').hidden = true;
  $('#pictureField').hidden = true;
  $('#sortField').hidden = true;        // the all-legs table sorts on its headings
  if (!S.allRows) {
    try {
      S.allRows = await api('/api/results_all');
    } catch (err) {
      $('#resTable').innerHTML = '';
      $('#resNote').textContent = err.message
        + ' — unmix at least one leg first (All legs runs every one of them).';
      $('#resMeta').textContent = '';
      return;
    }
  }
  renderAllLegs();
}

const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const sd = a => Math.sqrt(mean(a.map(x => (x - mean(a)) ** 2)));

function renderAllLegs() {
  const { rows, classes } = S.allRows;
  const hideDup = $('#hideDup').checked;
  const nums = (rs, k) => rs.map(r => parseFloat(r[k])).filter(v => !Number.isNaN(v));

  // Only the structures that were actually scored, in the mask's own order.
  const found = new Set();
  rows.forEach(r => Object.keys(r).forEach(k =>
    k.startsWith('iou_') && r[k] !== '' && found.add(k.slice(4))));
  const names = classes.filter(n => found.has(n))
    .concat([...found].filter(n => !classes.includes(n)).sort());

  const byVersion = new Map();
  for (const r of rows) {
    if (hideDup && r.duplicate_of) continue;
    if (!byVersion.has(r.version)) byVersion.set(r.version, []);
    byVersion.get(r.version).push(r);
  }

  let items = [...byVersion.values()].map(rs => {
    const accs = nums(rs, 'accuracy_outside_rois');
    const o = { version: rs[0].version, label: rs[0].label,
                channels: +rs[0].channels, legs: rs.length,
                duplicate_of: rs[0].duplicate_of,
                accuracy_outside_rois: accs.length ? mean(accs) : null,
                accuracy_sd: accs.length ? sd(accs) : null };
    for (const k of ['mean_iou_outside_rois', 'residual_rel',
                     ...names.map(n => 'iou_' + n)]) {
      const v = nums(rs, k);
      o[k] = v.length ? mean(v) : null;
    }
    return o;
  });

  const COLS = [
    { k: 'label', h: 'version', text: true },
    { k: 'channels', h: 'ch', int: true },
    { k: 'legs', h: 'legs', int: true },
    { k: 'accuracy_outside_rois', h: 'mean accuracy', bar: true },
    { k: 'accuracy_sd', h: 'sd' },
    { k: 'mean_iou_outside_rois', h: 'mean IoU' },
    ...names.map(n => ({ k: 'iou_' + n, h: 'IoU ' + n })),
    { k: 'residual_rel', h: 'resid frac' },
  ];
  const col = COLS.find(c => c.k === S.allSort);
  if (col) {
    const sign = S.allDir === 'asc' ? 1 : -1;
    items.sort((a, b) => col.text
      ? sign * String(a[col.k] ?? '').localeCompare(String(b[col.k] ?? ''))
      : sign * ((a[col.k] ?? -Infinity) - (b[col.k] ?? -Infinity)));
  }

  $('#resTable').innerHTML = '<thead><tr>' + COLS.map(c =>
      `<th data-k="${c.k}"${c.k === S.allSort ? ` data-dir="${S.allDir}"` : ''}>${c.h}</th>`
    ).join('') + '</tr></thead><tbody>' + items.map(it =>
      `<tr class="${it.duplicate_of ? 'dup' : ''}">` + COLS.map(c => {
        if (c.k === 'label') {
          return `<td>${it.label}${it.duplicate_of
            ? ` <span style="opacity:.6">= ${it.duplicate_of}</span>` : ''}</td>`;
        }
        const x = it[c.k];
        const bar = c.bar && x != null
          ? ` <i class="bar" style="width:${Math.round(x * 44)}px"></i>` : '';
        return `<td>${c.int ? x : num(x)}${bar}</td>`;
      }).join('') + '</tr>').join('') + '</tbody>';
  $$('#resTable th').forEach(th => th.onclick = () => {
    const text = COLS.find(c => c.k === th.dataset.k)?.text;
    S.allDir = S.allSort === th.dataset.k
      ? (S.allDir === 'asc' ? 'desc' : 'asc') : (text ? 'asc' : 'desc');
    S.allSort = th.dataset.k;
    renderAllLegs();
  });

  // Each structure over every version — the "is nerve simply hard?" question.
  const perClass = names.map(n => {
    const all = nums(rows.filter(r => !(hideDup && r.duplicate_of)), 'iou_' + n);
    const best = items.reduce((b, it) =>
      (it['iou_' + n] ?? -1) > (b?.['iou_' + n] ?? -1) ? it : b, null);
    return `${n} ${num(mean(all), 3)} (best ${num(best?.['iou_' + n], 3)}, ${best?.label})`;
  });
  $('#resNote').textContent =
    'Per-structure columns are that structure\'s IoU averaged over the legs. '
    + 'Averaged over every version too: ' + perClass.join(' · ') + '.';

  const legs = new Set(rows.map(r => r.leg));
  $('#resMeta').textContent =
    `${legs.size} legs · ${items.length} versions · averaged`;
}

function openDetail(vid) {
  const r = S.results, v = r.versions.find(x => x.version === vid);
  if (!v) return;
  // The run stamp rides on every image URL: the paths are stable across runs,
  // so without it a cached copy would show the last run's regions next to this
  // run's numbers.
  const base = `/out/${enc(r.leg)}`;
  const stamp = '?v=' + encodeURIComponent(r.run);
  const names = r.selection.structures.map(s => s.name).concat(v.extras || []);
  const m = v.metrics;
  const ids = Object.entries(m?.all?.per_class || {});

  const shots = [
    ['class map', `${base}/${vid}/classmap.png${stamp}`],
    ['truth', `${base}/truth.png${stamp}`],
    ['regions used', `${base}/rois_preview.png${stamp}`],
    ['residual', `${base}/${vid}/residual.png${stamp}`],
    ...names.map(n => [`abundance · ${n}`,
                       `${base}/${vid}/abundance_${n}.png${stamp}`]),
  ].map(([cap, src]) =>
    `<figure><img src="${src}" loading="lazy"><figcaption>${cap}</figcaption></figure>`).join('');

  const table = ids.length ? `<table><thead><tr><th>class</th><th>IoU</th><th>dice</th>
      <th>precision</th><th>recall</th><th>support</th></tr></thead><tbody>`
    + ids.map(([n]) => {
      const o = m.outside_rois.per_class[n];
      return `<tr><td>${n}</td><td>${num(o.iou)}</td><td>${num(o.dice)}</td>
        <td>${num(o.precision)}</td><td>${num(o.recall)}</td>
        <td>${o.support.toLocaleString()}</td></tr>`;
    }).join('') + '</tbody></table>' : '';

  const cm = m?.outside_rois?.confusion;
  const cmTable = cm ? `<h3 style="margin-top:14px">confusion (rows = truth)</h3>
    <table><thead><tr><th></th>${ids.map(([n]) => `<th>${n}</th>`).join('')}
    <th>other</th></tr></thead><tbody>${cm.map((row, i) =>
      `<tr><td>${ids[i][0]}</td>${row.map(x => `<td>${x.toLocaleString()}</td>`).join('')}</tr>`
    ).join('')}</tbody></table>` : '';

  $('#sheet').innerHTML = `<h3>${v.label}</h3>
    <p class="note">${vid} · ${v.channels} channels · ${v.method}
      ${v.extra !== 'none' ? '+ ' + v.extra + ' component' : ''}
      ${v.duplicate_of ? ' · byte-identical to ' + v.duplicate_of : ''}</p>
    <div class="shots">${shots}</div>
    <h3>scored outside the endmember regions</h3>${table}${cmTable}`;
  $('#modal').hidden = false;
}

boot().catch(err => {
  document.body.innerHTML =
    `<p style="padding:24px;color:#b3261e">could not start: ${err.message}</p>`;
});
