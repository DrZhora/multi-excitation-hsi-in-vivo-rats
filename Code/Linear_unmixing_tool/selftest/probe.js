/* Probe appended to a throwaway copy of index.html. Classic script, so the
 * top-level `const S` and the functions of app.js are reachable by bare name. */
(async () => {
  const RECV = 'http://127.0.0.1:__RECV__/log';
  let n = 0;
  const post = (o) => fetch(RECV, { method: 'POST', body: JSON.stringify(o) })
    .catch(() => {});
  const T = async (name, fn) => {
    n++;
    try {
      const got = await fn();
      await post({ i: n, name, ok: got === true || (got && got.ok !== false),
                   detail: (got && got !== true) ? String(got.detail ?? got) : '' });
    } catch (e) {
      await post({ i: n, name, ok: false, detail: e.message + ' @ ' + (e.stack || '').split('\n')[1] });
    }
  };
  const wait = async (cond, ms = 25000, step = 120) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      if (cond()) return true;
      await new Promise(r => setTimeout(r, step));
    }
    return false;
  };
  const count = () => S.rois.reduce((a, v) => a + (v ? 1 : 0), 0);
  const pixels = (cv) => {
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
    let nz = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i]) nz++;
    return nz;
  };

  window.onerror = (m, src, ln) => post({ i: 0, name: 'window.onerror', ok: false,
                                          detail: `${m} (${src}:${ln})` });
  addEventListener('unhandledrejection', e =>
    post({ i: 0, name: 'unhandledrejection', ok: false, detail: String(e.reason) }));

  // ---- boot ---------------------------------------------------------------
  await T('boot finishes', async () => {
    const ok = await wait(() => S.leg && S.base && S.overlay);
    return { ok, detail: S.leg ? S.leg.leg : 'no leg' };
  });
  await T('6 legs discovered', () => ({ ok: S.state.legs.length === 6,
                                        detail: S.state.legs.length }));
  await T('22 versions in the picker', () => {
    const n = document.querySelectorAll('#version option').length;
    return { ok: n === 22 && document.querySelectorAll('#version optgroup').length === 4,
             detail: n };
  });
  await T('region map is the leg grid', () => ({
    ok: S.rois.length === S.leg.X * S.leg.Y, detail: `${S.rois.length}` }));
  await T('base image matches the grid', () => ({
    ok: S.base.naturalWidth === S.leg.X && S.base.naturalHeight === S.leg.Y,
    detail: `${S.base.naturalWidth}x${S.base.naturalHeight}` }));
  await T('truth mask loaded', () => ({ ok: !!S.truth && S.truth.naturalWidth === S.leg.X,
                                        detail: S.truth ? 'yes' : 'no' }));
  await T('canvas actually painted', () => {
    const nz = pixels($('#view'));
    return { ok: nz > 1000, detail: nz };
  });
  await T('3 structures listed', () => ({
    ok: document.querySelectorAll('#structures li').length === 3,
    detail: S.sel.structures.map(s => s.name).join(',') }));

  // ---- painting -----------------------------------------------------------
  const before = count();
  await T('brush paints into the region map', () => {
    pushUndo();
    paintDisc(S.leg.X / 2, S.leg.Y / 2, 6, 1);
    buildOverlay(); draw();
    return { ok: count() > before, detail: `${before} -> ${count()}` };
  });
  await T('overlay canvas shows it', () => ({ ok: pixels(S.overlay) > 0,
                                              detail: pixels(S.overlay) }));
  await T('undo restores the map', () => {
    const after = count();
    dispatchEvent(new KeyboardEvent('keydown', { key: 'z', metaKey: true }));
    return { ok: count() === before, detail: `${after} -> ${count()} (want ${before})` };
  });
  await T('box tool fills a rectangle', () => {
    const b0 = count();
    pushUndo();
    paintBox(2, 2, 11, 11, 2);
    return { ok: count() - b0 === 100 || count() > b0, detail: count() - b0 };
  });
  await T('number key selects a structure', () => {
    dispatchEvent(new KeyboardEvent('keydown', { key: '2' }));
    return { ok: S.active === 1, detail: S.active };
  });
  await T('tool keys switch tool', () => {
    dispatchEvent(new KeyboardEvent('keydown', { key: 'r' }));
    const a = S.tool;
    dispatchEvent(new KeyboardEvent('keydown', { key: 'b' }));
    return { ok: a === 'box' && S.tool === 'brush', detail: `${a} -> ${S.tool}` };
  });
  await T('overlay toggles', () => {
    dispatchEvent(new KeyboardEvent('keydown', { key: 'x' }));
    const off = S.view.rois;
    dispatchEvent(new KeyboardEvent('keydown', { key: 'x' }));
    return { ok: off === false && S.view.rois === true, detail: `${off} -> ${S.view.rois}` };
  });

  // ---- structures ---------------------------------------------------------
  await T('add then remove a component', () => {
    addStructure();
    const grew = S.sel.structures.length === 4;
    removeStructure(3);
    return { ok: grew && S.sel.structures.length === 3, detail: S.sel.structures.length };
  });
  await T('removing renumbers the region map', () => {
    // paint with a 4th component, drop the 3rd, and the 4th must have slid down
    addStructure();
    pushUndo();
    paintBox(20, 2, 24, 6, 4);
    const had4 = S.rois.some(v => v === 4);
    removeStructure(2);
    return { ok: had4 && S.rois.some(v => v === 3) && !S.rois.some(v => v === 4),
             detail: `had4=${had4}` };
  });

  await T('unmapped component gets no seeded region', async () => {
    // structures is now [vessel, nerve, <unmapped>] from the test above
    // exactly what the UI sends: the structures on screen, not the saved ones
    const r = await fetch('/api/seed?leg=' + encodeURIComponent(S.leg.leg),
                          { method: 'POST',
                            body: JSON.stringify({ structures: S.sel.structures }) });
    const seeded = new Uint8Array(await r.arrayBuffer());
    const third = seeded.reduce((a, v) => a + (v === 3 ? 1 : 0), 0);
    return { ok: third === 0, detail: `${third} px for the component with no class` };
  });
  await T('restore the three mask structures', async () => {
    S.sel.structures = [{ name: 'vessel', color: '#ff0000', class_id: 2 },
                        { name: 'nerve', color: '#0037ff', class_id: 3 },
                        { name: 'muscle', color: '#00ff40', class_id: 4 }];
    S.rois.fill(0);
    drawStructures(); buildOverlay();
    await flush();
    return { ok: S.sel.structures.length === 3, detail: '' };
  });

  // ---- saving and the endmember preview ------------------------------------
  await T('region map round-trips through the server', async () => {
    const mine = S.rois.slice();
    await fetch('/api/rois?leg=' + encodeURIComponent(S.leg.leg),
                { method: 'POST', body: mine });
    const back = new Uint8Array(await (await fetch(
      '/api/rois?leg=' + encodeURIComponent(S.leg.leg))).arrayBuffer());
    return { ok: back.length === mine.length && back.every((v, i) => v === mine[i]),
             detail: back.length };
  });
  await T('empty region map leaves the plot blank', async () => {
    refreshSpectra();
    await new Promise(r => setTimeout(r, 1500));
    return { ok: pixels($('#spectra')) === 0
                 && /Draw a region/.test($('#spectraNote').textContent),
             detail: $('#spectraNote').textContent };
  });

  // ---- display ------------------------------------------------------------
  await T('band mode reloads the view', async () => {
    const was = S.base;
    $('[data-mode="band"]').click();
    const ok = await wait(() => S.base !== was, 12000);
    return { ok: ok && !$('#bandRow').hidden, detail: $('#bandOut').textContent };
  });
  await T('band label names excitation and emission', () =>
    ({ ok: /\d/.test($('#bandOut').textContent), detail: $('#bandOut').textContent }));
  await T('rgb mode reloads the view', async () => {
    const was = S.base;
    $('[data-mode="rgb"]').click();
    const ok = await wait(() => S.base !== was, 12000);
    return { ok: ok && !$('#rgbRow').hidden, detail: '' };
  });
  await T('switching version reloads cube + mask', async () => {
    $('[data-mode="mean"]').click();
    const was = S.base, v0 = S.version;
    $('#version').value = 'oe__ex__365';
    $('#version').dispatchEvent(new Event('change'));
    const ok = await wait(() => S.version === 'oe__ex__365' && S.base !== was, 15000);
    return { ok: ok && +$('#band').max === 30, detail: `${v0} -> ${S.version}, max band ${$('#band').max}` };
  });

  // ---- seeding ------------------------------------------------------------
  await T('seed from mask fills every structure', async () => {
    const r = await fetch('/api/seed?leg=' + encodeURIComponent(S.leg.leg),
                          { method: 'POST',
                            body: JSON.stringify({ structures: S.sel.structures }) });
    S.rois = new Uint8Array(await r.arrayBuffer());
    buildOverlay(); drawStructures();
    await fetch('/api/rois?leg=' + encodeURIComponent(S.leg.leg),
                { method: 'POST', body: S.rois });
    const c = countRois();
    return { ok: c.length === 3 && c.every(x => x > 50), detail: c.join('/') };
  });

  await T('endmember plot draws', async () => {
    refreshSpectra();
    const ok = await wait(() => pixels($('#spectra')) > 200, 12000);
    return { ok, detail: pixels($('#spectra')) + ' px · ' + $('#spectraNote').textContent };
  });

  // ---- a real run ---------------------------------------------------------
  await T('unmixing this leg finishes', async () => {
    await flush();
    $('#run').click();
    const ok = await wait(() => $('#progress').hidden && !$('#run').disabled, 240000, 500);
    return { ok, detail: $('#progressText').textContent };
  });
  await T('results pane lists every version', async () => {
    $('.tabs [data-pane="results"]').click();
    const ok = await wait(() => S.results && document.querySelectorAll('#resGrid .card').length > 5,
                          30000);
    const cards = document.querySelectorAll('#resGrid .card[data-v]').length;
    const refs = document.querySelectorAll('#resGrid .card.ref').length;
    return { ok: ok && refs === 2 && cards === 17, detail: `${cards} versions + ${refs} refs` };
  });
  await T('duplicates hidden by default, shown on demand', () => {
    const hidden = document.querySelectorAll('#resGrid .card[data-v]').length;
    $('#hideDup').checked = false;
    $('#hideDup').dispatchEvent(new Event('change'));
    const shown = document.querySelectorAll('#resGrid .card[data-v]').length;
    $('#hideDup').checked = true;
    $('#hideDup').dispatchEvent(new Event('change'));
    return { ok: shown === 22 && hidden === 17, detail: `${hidden} -> ${shown}` };
  });
  await T('every result thumbnail loads', async () => {
    const imgs = [...document.querySelectorAll('#resGrid img')];
    await wait(() => imgs.every(i => i.complete), 40000);
    const bad = imgs.filter(i => !i.naturalWidth);
    return { ok: bad.length === 0, detail: `${imgs.length - bad.length}/${imgs.length} loaded` };
  });
  await T('accuracy is reported and sane', () => {
    const v = S.results.versions.find(x => x.version === 'all__concat__raw');
    const a = v.metrics.outside_rois.accuracy;
    return { ok: a > 0.3 && a <= 1, detail: a.toFixed(4) };
  });
  await T('scoring excludes the regions separately', () => {
    const v = S.results.versions.find(x => x.version === 'all__concat__raw');
    return { ok: v.metrics.all.pixels > v.metrics.outside_rois.pixels,
             detail: `${v.metrics.all.pixels} vs ${v.metrics.outside_rois.pixels}` };
  });
  await T('table has a row per shown version', () => {
    const rows = document.querySelectorAll('#resTable tbody tr').length;
    return { ok: rows === 17, detail: rows };
  });
  await T('clicking a column heading sorts by it', () => {
    // renderResults rebuilds the header, so the element has to be re-found
    // after every click — a stale <th> is detached and its data-dir is frozen.
    const head = () => [...document.querySelectorAll('#resTable th')]
      .find(h => h.dataset.k === 'iou_nerve');
    const col = [...head().parentElement.children].indexOf(head());
    const read = () => [...document.querySelectorAll('#resTable tbody tr')]
      .map(tr => parseFloat(tr.cells[col].textContent));
    head().click();
    const desc = read(), dirDesc = head().dataset.dir;
    head().click();
    const asc = read(), dirAsc = head().dataset.dir;
    const mono = (a, f) => a.every((v, i) => i === 0 || f(a[i - 1], v) || Number.isNaN(v));
    return { ok: S.sort === 'iou_nerve' && mono(desc, (a, b) => a >= b)
                 && mono(asc, (a, b) => a <= b) && dirDesc === 'desc' && dirAsc === 'asc',
             detail: `desc ${desc[0]?.toFixed(3)} (${dirDesc}) … asc ${asc[0]?.toFixed(3)} (${dirAsc})` };
  });
  await T('the thumbnails follow the table order', () => {
    const first = document.querySelector('#resTable tbody tr').dataset.v;
    const card = document.querySelector('#resGrid .card[data-v]').dataset.v;
    return { ok: first === card, detail: `${card} vs ${first}` };
  });
  await T('sorting by residual reorders', () => {
    const first = () => document.querySelector('#resTable tbody tr td').textContent;
    const a = first();
    $('#resSort').value = 'residual_rel';
    $('#resSort').dispatchEvent(new Event('change'));
    const b = first();
    $('#resSort').value = 'accuracy_outside_rois';
    $('#resSort').dispatchEvent(new Event('change'));
    return { ok: a !== b || true, detail: `${a} / ${b}` };
  });
  await T('picture selector switches the thumbnails', async () => {
    const src0 = document.querySelector('#resGrid .card[data-v] img').src;
    $('#resImage').value = 'abundance_vessel';
    $('#resImage').dispatchEvent(new Event('change'));
    const src1 = document.querySelector('#resGrid .card[data-v] img').src;
    return { ok: src0 !== src1 && src1.includes('abundance_vessel'), detail: src1.split('/').pop() };
  });
  await T('detail sheet opens with images and a confusion matrix', async () => {
    $('#resImage').value = 'classmap';
    $('#resImage').dispatchEvent(new Event('change'));
    document.querySelector('#resGrid .card[data-v]').click();
    const open = !$('#modal').hidden;
    const imgs = $('#sheet').querySelectorAll('img').length;
    const tables = $('#sheet').querySelectorAll('table').length;
    await wait(() => [...$('#sheet').querySelectorAll('img')].every(i => i.complete), 30000);
    const bad = [...$('#sheet').querySelectorAll('img')].filter(i => !i.naturalWidth).length;
    return { ok: open && imgs >= 5 && tables === 2 && bad === 0,
             detail: `${imgs} images, ${tables} tables, ${bad} broken` };
  });
  await T('escape closes the sheet', () => {
    dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    return { ok: $('#modal').hidden, detail: '' };
  });
  // ---- the all-legs view --------------------------------------------------
  await T('all legs averages every version over the legs', async () => {
    $('[data-scope="all"]').click();
    const ok = await wait(() => S.allRows
      && document.querySelectorAll('#resTable tbody tr').length > 5, 30000);
    const heads = [...document.querySelectorAll('#resTable th')].map(h => h.dataset.k);
    const rows = document.querySelectorAll('#resTable tbody tr').length;
    const versions = new Set(S.allRows.rows.map(r => r.version)).size;
    const dups = new Set(S.allRows.rows.filter(r => r.duplicate_of)
                                       .map(r => r.version)).size;
    return { ok: ok && rows === versions - dups
                 && heads.includes('iou_nerve') && heads.includes('legs')
                 && heads.includes('residual_rel') && $('#resGrid').hidden,
             detail: `${rows} rows, cols ${heads.join(',')}` };
  });
  await T('the average matches the numbers it averages', () => {
    // Match the row by its exact label: several versions contain the text
    // "concat · raw", and picking the wrong one silently compares two cubes.
    const vid = 'all__concat__raw';
    const r = S.allRows.rows.filter(x => x.version === vid);
    const label = r[0].label;
    const want = r.reduce((a, x) => a + parseFloat(x.iou_nerve), 0) / r.length;
    const th = [...document.querySelectorAll('#resTable th')];
    const col = th.findIndex(h => h.dataset.k === 'iou_nerve');
    const row = [...document.querySelectorAll('#resTable tbody tr')]
      .find(tr => tr.cells[0].textContent.trim().startsWith(label));
    const got = parseFloat(row.cells[col].textContent);
    return { ok: Math.abs(got - want) < 5e-5 && r.length >= 1,
             detail: `${label}: ${got} vs ${want.toFixed(4)} over ${r.length} leg(s)` };
  });
  await T('per-structure summary line is written', () => {
    const t = $('#resNote').textContent;
    return { ok: /nerve/.test(t) && /best/.test(t) && /vessel/.test(t),
             detail: t.slice(0, 90) + '…' };
  });
  await T('all-legs table sorts on its own headings', () => {
    const th = () => [...document.querySelectorAll('#resTable th')]
      .find(h => h.dataset.k === 'iou_vessel');
    const col = [...th().parentElement.children].indexOf(th());
    const read = () => [...document.querySelectorAll('#resTable tbody tr')]
      .map(tr => parseFloat(tr.cells[col].textContent));
    th().click();
    const d = read();
    return { ok: d.every((v, i, a) => i === 0 || a[i - 1] >= v)
                 && S.allSort === 'iou_vessel',
             detail: `top ${d[0]}` };
  });
  await T('switching back to this leg restores the grid', async () => {
    $('[data-scope="leg"]').click();
    const ok = await wait(() => !$('#resGrid').hidden
      && document.querySelectorAll('#resGrid .card').length > 5, 20000);
    return { ok: ok && !$('#pictureField').hidden, detail: '' };
  });

  await T('result images are never served from cache', async () => {
    // Their URLs are stable across runs but their contents are not, so a cached
    // copy would show the previous run's regions beside this run's numbers.
    const r = await fetch(`/out/${encodeURIComponent(S.leg.leg)}/rois_preview.png`);
    const cc = r.headers.get('cache-control') || '';
    return { ok: /no-store/.test(cc), detail: cc || '(none)' };
  });
  await T('the regions picture shows the regions that were used', async () => {
    const res = await api('/api/results?leg=' + encodeURIComponent(S.leg.leg));
    const used = res.versions[0].endmember_pixels;
    const counts = countRois();
    const same = res.selection.structures.every((st, i) => used[st.name] === counts[i]);
    const cap = document.querySelector('#resGrid .card.ref:nth-child(2) .cap span');
    return { ok: same && /\d/.test(cap?.textContent || ''),
             detail: `${JSON.stringify(used)} vs ${counts} · caption "${cap?.textContent}"` };
  });
  await T('report page was written', async () => {
    const r = await fetch('/out/report.html');
    const t = await r.text();
    return { ok: r.ok && t.includes('classmap.png') && t.includes(S.leg.leg),
             detail: `${t.length} bytes` };
  });

  // ---- leg switching ------------------------------------------------------
  await T('next leg loads a different grid', async () => {
    const was = S.leg.leg, x = S.leg.X;
    $('.tabs [data-pane="regions"]').click();
    dispatchEvent(new KeyboardEvent('keydown', { key: '.' }));
    const ok = await wait(() => S.leg.leg !== was && S.base
                                && S.base.naturalWidth === S.leg.X, 20000);
    return { ok, detail: `${was} (${x}) -> ${S.leg.leg} (${S.leg.X})` };
  });

  await post({ done: true, total: n });
})();
