/* Tamil letter recogniser – vanilla JS front end (no build step). */
(() => {
  'use strict';

  // ---------- helpers ----------
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const mq = { mobile: matchMedia('(max-width: 639.98px)'), reduce: matchMedia('(prefers-reduced-motion: reduce)') };
  const isMobile = () => mq.mobile.matches;
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (_) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) { /* storage full / blocked */ } },
    del(k) { try { localStorage.removeItem(k); } catch (_) { /* ignore */ } },
  };
  const MAX_BYTES = 5 * 1024 * 1024;
  const OK_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp'];
  const pct = (x) => `${Math.round(x * 100)}%`;
  const level = (c) => (c >= 0.7 ? 'good' : c >= 0.4 ? 'mid' : 'bad');
  const el = (tag, attrs = {}, kids = []) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') n.className = v; else if (k === 'text') n.textContent = v;
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), v); else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? '' : v);
    }
    for (const c of [].concat(kids)) if (c) n.append(c);
    return n;
  };

  class ApiError extends Error { constructor(msg, status) { super(msg); this.status = status; } }
  async function api(path, { method = 'GET', body, signal, json } = {}) {
    let res;
    try {
      res = await fetch(path, { method, signal, body: json ? JSON.stringify(json) : body, headers: json ? { 'Content-Type': 'application/json' } : undefined });
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      throw new ApiError('The recogniser is not reachable. Is the server still running?', 0);
    }
    let data = null;
    try { data = await res.json(); } catch (_) { /* no body */ }
    if (!res.ok) throw new ApiError((data && data.detail) || 'Something went wrong.', res.status);
    return data;
  }

  // ---------- toasts ----------
  function toast(msg, kind = '', ms = 3200) {
    const t = el('div', { class: `toast ${kind}`, role: kind === 'err' ? 'alert' : 'status' }, [msg]);
    $('#toasts').append(t);
    setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 320); }, ms);
  }

  // ---------- state ----------
  const S = {
    mode: 'recognize', tab: 'draw', health: null, classes: [], byLetter: new Map(), cells: new Map(),
    seq: 0, abort: null, lastBlob: null, lastUrl: null, lastResult: null, gradcam: false,
    practice: { target: null, streak: 0, correct: 0, tries: 0, filter: 'all', lock: false },
    enroll: { label: null, samples: [] },
  };
  window.__ui = S;   // for tests / debugging

  // ---------- theme ----------
  const themeBtn = $('#theme-btn');
  const sysDark = matchMedia('(prefers-color-scheme: dark)');
  function applyTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; }
  applyTheme(store.get('theme', null));
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || (sysDark.matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    applyTheme(next); store.set('theme', next);
  });

  // ---------- splash / boot ----------
  const splash = $('#splash');
  async function boot() {
    $('#splash-error').hidden = true; $('.splash-inner').hidden = false;
    const t0 = Date.now(); let reached = false;
    for (;;) {
      try {
        const h = await api('/api/health');
        reached = true;
        $('#splash-msg').textContent = h.message || 'Loading model…';
        if (h.ready) { S.health = h; break; }
      } catch (_) {
        if (!reached && Date.now() - t0 > 10000) { showSplashError(); return; }
        if (reached) $('#splash-msg').textContent = 'Reconnecting…';
      }
      await sleep(400);
    }
    updateBadge();
    $('#splash-msg').textContent = 'Preparing the letter chart…';
    splash.classList.add('done');
    setTimeout(() => { splash.hidden = true; }, 500);
    loadClasses();
  }
  function showSplashError() { $('.splash-inner').hidden = true; $('#splash-error').hidden = false; }
  $('#splash-retry').addEventListener('click', boot);

  function updateBadge() {
    const h = S.health; const b = $('#model-badge');
    if (h && h.public) {                     // shared internet demo: no enrolment, no model switching
      document.body.classList.add('public');
      $$('.mode-btn[data-mode=enroll], .bn[data-mode=enroll]').forEach((x) => { x.hidden = true; });
      $('#unknown-enroll').hidden = true;
    }
    const full = h && h.model ? h.model.replace('ProtoNet · ', '') : 'No model';
    $('#model-name').textContent = full;
    $('#model-name').dataset.short = full.replace('MobileNetV3-L', 'MNv3').replace(' (ONNX)', ' · ONNX').replace(' (PyTorch)', ' · PyTorch').replace('ProtoNet ← ArcFace', 'Proto←Arc');
    b.classList.toggle('ready', !!(h && h.ready));
    b.title = h ? `${h.model} · ${h.backend} · ${h.registry_size} letters enrolled — click to change` : 'Model';
  }

  // ---------- classes / chart ----------
  async function loadClasses(retry = 0) {
    try {
      S.classes = await api('/api/classes');
      S.byLetter = new Map(S.classes.map((c) => [c.letter, c]));
      renderChart(); fillPracticeFilter();
      if (S.mode === 'practice' && !S.practice.target) nextTarget();
    } catch (e) {
      if (retry < 3) { await sleep(800); return loadClasses(retry + 1); }
      $('#chart-body').replaceChildren(el('div', { class: 'r-error', style: 'display:block' }, [
        el('p', {}, [el('strong', { text: 'The letter chart could not load.' })]),
        el('button', { class: 'btn btn-ghost', text: 'Retry', onclick: () => loadClasses() })]));
    }
  }

  function cellFor(letter) {
    const c = S.byLetter.get(letter);
    const enrolled = c && c.enrolled;
    const tip = c ? `${letter} · ${c.translit} · ${c.category_en}${enrolled ? ` · ${c.n_support} samples` : ' · not enrolled'}` : letter;
    const b = el('button', { class: `lcell${enrolled ? '' : ' off'}`, 'data-letter': letter, title: tip, 'aria-label': tip, text: letter });
    S.cells.set(letter, b);
    return b;
  }
  function renderChart() {
    S.cells = new Map();
    const C = S.classes; const body = $('#chart-body');
    const vowels = C.filter((c) => c.kind === 'vowel');
    const ay = C.filter((c) => c.kind === 'aytham');
    const cons = C.filter((c) => c.kind === 'consonant');
    const extra = C.filter((c) => !c.in_chart);
    const vowelHeads = vowels.map((v) => v.letter);
    const sec = (ta, en, kids) => el('div', { class: 'chart-section' }, [el('h3', {}, [el('span', { class: 'ta', text: ta }), en]), ...kids]);
    const table = el('table', { class: 'ctable' });
    const thead = el('thead', {}, [el('tr', {}, [el('th', { scope: 'col', 'aria-label': 'consonant' }), ...vowelHeads.map((v) => el('th', { scope: 'col', text: v }))])]);
    const tbody = el('tbody');
    for (const c of cons) {
      const base = c.letter[0];
      const row = el('tr', {}, [el('th', { scope: 'row', text: base })]);
      for (const cc of C.filter((x) => x.kind === 'compound' && x.consonant_row === c.letter)) row.append(el('td', {}, [cellFor(cc.letter)]));
      tbody.append(row);
    }
    table.append(thead, tbody);
    body.replaceChildren(
      sec('உயிர்', 'Vowels', [el('div', { class: 'lgrid' }, [...vowels, ...ay].map((c) => cellFor(c.letter)))]),
      sec('மெய்', 'Consonants', [el('div', { class: 'lgrid' }, cons.map((c) => cellFor(c.letter)))]),
      sec('உயிர்மெய்', 'Consonant + vowel (18 × 12)', [el('div', { class: 'ctable-wrap' }, [table])]),
      ...(extra.length ? [sec('பிற', 'Other spellings / enrolled extras', [el('div', { class: 'lgrid' }, extra.map((c) => cellFor(c.letter)))])] : []),
    );
    highlightChart();
  }
  function highlightChart() {
    S.cells.forEach((b, l) => {
      b.classList.toggle('hit', !!(S.lastResult && S.lastResult.known && S.lastResult.label === l && S.mode === 'recognize'));
      b.classList.toggle('sel', (S.mode === 'enroll' && S.enroll.label === l) || (S.mode === 'practice' && S.practice.target === l));
    });
  }
  function revealCell(letter) {
    const b = S.cells.get(letter); const box = $('#chart-body');
    if (!b || isMobile() || getComputedStyle(box).overflowY !== 'auto') return;
    const br = b.getBoundingClientRect(); const cr = box.getBoundingClientRect();
    if (br.top < cr.top || br.bottom > cr.bottom) box.scrollTo({ top: box.scrollTop + br.top - cr.top - cr.height / 2, behavior: mq.reduce.matches ? 'auto' : 'smooth' });
    const wrap = b.closest('.ctable-wrap');
    if (wrap) { const wr = wrap.getBoundingClientRect(); if (br.left < wr.left + 40 || br.right > wr.right) wrap.scrollLeft += br.left - wr.left - wr.width / 2; }
  }
  $('#chart-body').addEventListener('click', (e) => {
    const b = e.target.closest('.lcell'); if (!b) return;
    const letter = b.dataset.letter; const c = S.byLetter.get(letter);
    if (S.mode === 'practice') { setTarget(letter); closeSheet('chart-card'); }
    else if (S.mode === 'enroll') { setEnrollLabel(letter); closeSheet('chart-card'); }
    else toast(`${letter} — ${c.translit} · ${c.category_en}. ${c.enrolled ? `${c.n_support} reference samples.` : 'Not enrolled yet: use Enroll to teach it.'}`);
  });

  // ---------- modes ----------
  function setMode(mode) {
    S.mode = mode;
    $('#layout').dataset.mode = mode;
    $$('.mode-btn').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.mode === mode)));
    $$('.bn[data-mode]').forEach((b) => (b.dataset.mode === mode ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current')));
    $('#practice-target').hidden = mode !== 'practice';
    $('#practice-panel').hidden = mode !== 'practice';
    $('#enroll-panel').hidden = mode !== 'enroll';
    $('#predict-btn .btn-label').textContent = mode === 'practice' ? 'Check' : mode === 'enroll' ? 'Add as sample' : 'Predict';
    closeSheet('result-card'); closeSheet('chart-card');
    if (mode === 'practice' && !S.practice.target && S.classes.length) nextTarget();
    highlightChart();
    clearTimeout(autoTimer);
  }
  $$('.mode-btn, .bn[data-mode]').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

  // ---------- sheets (mobile) ----------
  const scrim = $('#scrim');
  function openSheet(id) {
    if (!isMobile()) return;
    const s = document.getElementById(id); s.classList.add('open');
    if (id === 'chart-card') scrim.hidden = false;
  }
  function closeSheet(id) {
    const s = document.getElementById(id); if (!s) return;
    s.classList.remove('open'); s.style.transform = '';
    if (id === 'chart-card' && $('#drawer').hidden) scrim.hidden = true;
  }
  $('#bn-chart').addEventListener('click', () => openSheet('chart-card'));
  $$('[data-close-sheet]').forEach((b) => b.addEventListener('click', () => closeSheet(b.dataset.closeSheet)));
  scrim.addEventListener('click', () => { closeSheet('chart-card'); closeDrawer(); });
  for (const [hid, sid] of [['result-handle', 'result-card'], ['chart-handle', 'chart-card']]) {
    const h = document.getElementById(hid); const s = document.getElementById(sid);
    let y0 = null, dy = 0;
    h.addEventListener('pointerdown', (e) => { y0 = e.clientY; dy = 0; h.setPointerCapture(e.pointerId); s.classList.add('dragging'); });
    h.addEventListener('pointermove', (e) => { if (y0 === null) return; dy = Math.max(0, e.clientY - y0); s.style.transform = `translateY(${dy}px)`; });
    const end = () => { if (y0 === null) return; y0 = null; s.classList.remove('dragging'); if (dy > 70) closeSheet(sid); else s.style.transform = ''; };
    h.addEventListener('pointerup', end); h.addEventListener('pointercancel', end);
  }
  mq.mobile.addEventListener('change', () => { closeSheet('result-card'); closeSheet('chart-card'); scrim.hidden = true; });

  // ---------- input tabs ----------
  function setTab(tab) {
    if (S.tab === tab) return;
    const prev = S.tab; S.tab = tab;
    $$('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === tab)));
    for (const t of ['draw', 'upload', 'camera']) $(`#panel-${t}`).hidden = t !== tab;
    if (prev === 'camera') stopCamera();
    document.body.classList.toggle('cam-on', tab === 'camera');
    if (tab === 'camera') startCamera();
    if (tab === 'draw') pad.resize();
    if (tab === 'upload' && up.img) up.render();
  }
  $$('.tab').forEach((t) => t.addEventListener('click', () => setTab(t.dataset.tab)));

  // ---------- drawing pad ----------
  const pad = (() => {
    const wrap = $('#canvas-wrap'); const cv = $('#pad'); const ctx = cv.getContext('2d');
    const strokes = []; let cur = null; let raf = 0; let size = 1;
    const INK = '#1f1b33';
    function resize() {
      const r = wrap.getBoundingClientRect(); if (!r.width) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      size = r.width;
      cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }
    function strokePath(c, s, scale) {
      const p = s.pts; c.strokeStyle = INK; c.fillStyle = INK; c.lineCap = 'round'; c.lineJoin = 'round';
      const w = (pr) => s.w * scale * (s.pen ? 0.55 + 0.9 * pr : 1);
      if (p.length === 1) { c.beginPath(); c.arc(p[0][0] * scale, p[0][1] * scale, w(p[0][2]) / 2, 0, Math.PI * 2); c.fill(); return; }
      let prev = [p[0][0] * scale, p[0][1] * scale];
      for (let i = 1; i < p.length; i++) {
        const cx = p[i][0] * scale, cy = p[i][1] * scale;
        const nx = i + 1 < p.length ? (p[i][0] + p[i + 1][0]) / 2 * scale : cx;
        const ny = i + 1 < p.length ? (p[i][1] + p[i + 1][1]) / 2 * scale : cy;
        c.beginPath(); c.lineWidth = w(p[i][2]); c.moveTo(prev[0], prev[1]); c.quadraticCurveTo(cx, cy, nx, ny); c.stroke();
        prev = [nx, ny];
      }
    }
    function draw() {
      raf = 0; ctx.clearRect(0, 0, size, size);
      for (const s of strokes) strokePath(ctx, s, size);
      wrap.classList.toggle('has-ink', strokes.length > 0);
    }
    const req = () => { if (!raf) raf = requestAnimationFrame(draw); };
    const pos = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height, e.pressure || 0.5]; };
    cv.addEventListener('pointerdown', (e) => {
      if (e.button > 0) return;
      e.preventDefault(); cv.setPointerCapture(e.pointerId);
      closeSheet('result-card'); clearTimeout(autoTimer);
      cur = { w: +$('#brush').value / 320, pen: e.pointerType === 'pen', pts: [pos(e)] };
      strokes.push(cur); req();
    });
    cv.addEventListener('pointermove', (e) => {
      if (!cur) return; e.preventDefault();
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      for (const ev of (evs.length ? evs : [e])) cur.pts.push(pos(ev));
      req();
    });
    const end = (e) => { if (!cur) return; cur = null; req(); onStrokeEnd(); };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end); cv.addEventListener('lostpointercapture', end);
    new ResizeObserver(() => resize()).observe(wrap);
    window.addEventListener('orientationchange', () => setTimeout(resize, 200));
    async function toBlob() {
      const N = 384; const c = document.createElement('canvas'); c.width = c.height = N;
      const x = c.getContext('2d'); x.fillStyle = '#fff'; x.fillRect(0, 0, N, N);
      for (const s of strokes) strokePath(x, s, N);
      const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
      return { blob, url: c.toDataURL('image/png') };
    }
    return {
      resize, toBlob, get empty() { return strokes.length === 0; }, get count() { return strokes.length; },
      undo() { strokes.pop(); req(); }, clear() { strokes.length = 0; req(); },
      /** test hook: draw normalised polylines */
      drawPaths(paths, w = 13) { for (const pts of paths) strokes.push({ w: w / 320, pen: false, pts: pts.map(([a, b]) => [a, b, 0.5]) }); req(); },
    };
  })();
  window.__pad = pad;

  let autoTimer = 0;
  const AUTO_MS = 600;
  function onStrokeEnd() {
    clearTimeout(autoTimer);
    if (S.mode !== 'recognize') return;
    autoTimer = setTimeout(() => { if (!pad.empty) mainAction('auto'); }, AUTO_MS);
  }
  $('#undo-btn').addEventListener('click', () => {
    pad.undo();
    if (pad.empty) { clearTimeout(autoTimer); if (S.mode === 'recognize') setResultState('empty'); }
    else onStrokeEnd();
  });
  $('#clear-btn').addEventListener('click', clearAll);
  function clearAll() {
    clearTimeout(autoTimer);
    if (S.tab === 'draw') { pad.clear(); }
    if (S.abort) S.abort.abort();
    S.seq++; setBusy(false);
    if (S.mode === 'recognize') { setResultState('empty'); S.lastResult = null; highlightChart(); }
    closeSheet('result-card');
  }

  // ---------- upload ----------
  const up = (() => {
    const dz = $('#dropzone'); const fi = $('#file-input'); const stage = $('#preview-stage');
    const cv = $('#preview-canvas'); const ctx = cv.getContext('2d'); const box = $('#crop-box');
    const st = { img: null, rot: 0, crop: null, rect: null, name: '' };
    function validate(f) {
      if (!f) return 'No file selected.';
      if (f.type && !OK_TYPES.includes(f.type)) return 'Please choose a PNG, JPEG, WebP or BMP image.';
      if (f.size > MAX_BYTES) return 'That image is larger than 5 MB.';
      return null;
    }
    async function load(f) {
      const err = validate(f); if (err) { toast(err, 'err'); return; }
      const url = URL.createObjectURL(f);
      const img = new Image();
      try { await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = url; }); }
      catch (_) { toast('That file could not be opened as an image.', 'err'); return; }
      st.img = img; st.rot = 0; st.crop = null; st.name = f.name || 'image';
      dz.hidden = true; $('#preview').hidden = false;
      if (S.tab !== 'upload') setTab('upload');
      render();
      if (S.mode === 'recognize') mainAction('upload');
    }
    function rotated() {
      const { img, rot } = st; const sw = rot % 180 ? img.naturalHeight : img.naturalWidth; const sh = rot % 180 ? img.naturalWidth : img.naturalHeight;
      return { sw, sh };
    }
    function drawRot(c, W, H, x0, y0, w, h) {
      c.save(); c.translate(x0 + w / 2, y0 + h / 2); c.rotate(st.rot * Math.PI / 180);
      const iw = st.rot % 180 ? h : w, ih = st.rot % 180 ? w : h;
      c.drawImage(st.img, -iw / 2, -ih / 2, iw, ih); c.restore();
    }
    function render() {
      if (!st.img) return;
      const r = stage.getBoundingClientRect(); if (!r.width) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
      const { sw, sh } = rotated(); const k = Math.min(r.width / sw, r.height / sh);
      const w = sw * k, h = sh * k, x = (r.width - w) / 2, y = (r.height - h) / 2;
      ctx.fillStyle = '#fff'; ctx.fillRect(x, y, w, h);
      drawRot(ctx, r.width, r.height, x, y, w, h);
      st.rect = { x, y, w, h };
      showBox();
    }
    function showBox() {
      if (!st.crop || !st.rect) { box.hidden = true; return; }
      const { x, y, w, h } = st.rect; const c = st.crop;
      Object.assign(box.style, { left: `${x + c.x * w}px`, top: `${y + c.y * h}px`, width: `${c.w * w}px`, height: `${c.h * h}px` });
      box.hidden = false;
    }
    let start = null;
    stage.addEventListener('pointerdown', (e) => {
      if (!st.rect) return; e.preventDefault(); stage.setPointerCapture(e.pointerId);
      const r = stage.getBoundingClientRect(); start = [e.clientX - r.left, e.clientY - r.top];
    });
    stage.addEventListener('pointermove', (e) => {
      if (!start) return; const r = stage.getBoundingClientRect(); const { x, y, w, h } = st.rect;
      const px = Math.min(Math.max(e.clientX - r.left, x), x + w), py = Math.min(Math.max(e.clientY - r.top, y), y + h);
      const sx = Math.min(Math.max(start[0], x), x + w), sy = Math.min(Math.max(start[1], y), y + h);
      st.crop = { x: (Math.min(sx, px) - x) / w, y: (Math.min(sy, py) - y) / h, w: Math.abs(px - sx) / w, h: Math.abs(py - sy) / h };
      showBox();
    });
    const endCrop = () => {
      if (!start) return; start = null;
      if (st.crop && (st.crop.w < 0.04 || st.crop.h < 0.04)) { st.crop = null; showBox(); return; }
      if (st.crop && S.mode === 'recognize') mainAction('crop');
    };
    stage.addEventListener('pointerup', endCrop); stage.addEventListener('pointercancel', endCrop);
    $('#rot-left').addEventListener('click', () => { st.rot = (st.rot + 270) % 360; st.crop = null; render(); if (S.mode === 'recognize') mainAction('rotate'); });
    $('#rot-right').addEventListener('click', () => { st.rot = (st.rot + 90) % 360; st.crop = null; render(); if (S.mode === 'recognize') mainAction('rotate'); });
    $('#crop-reset').addEventListener('click', () => { st.crop = null; showBox(); if (S.mode === 'recognize') mainAction('crop'); });
    $('#upload-new').addEventListener('click', () => fi.click());
    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });
    fi.addEventListener('change', () => { if (fi.files[0]) load(fi.files[0]); fi.value = ''; });
    ['dragenter', 'dragover'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('over'); }));
    dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) load(f); });
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer && e.dataTransfer.files[0]; if (f && !e.target.closest('#dropzone')) load(f); });
    document.addEventListener('paste', (e) => {
      const it = Array.from(e.clipboardData ? e.clipboardData.items : []).find((i) => i.type.startsWith('image/'));
      if (it) { e.preventDefault(); load(it.getAsFile()); }
    });
    new ResizeObserver(() => { if (!$('#preview').hidden) render(); }).observe(stage);
    async function toBlob() {
      if (!st.img) return null;
      const { sw, sh } = rotated();
      const c0 = st.crop || { x: 0, y: 0, w: 1, h: 1 };
      const cw = Math.max(2, Math.round(c0.w * sw)), ch = Math.max(2, Math.round(c0.h * sh));
      const k = Math.min(1, 1024 / Math.max(cw, ch));
      const out = document.createElement('canvas'); out.width = Math.round(cw * k); out.height = Math.round(ch * k);
      const x = out.getContext('2d'); x.fillStyle = '#fff'; x.fillRect(0, 0, out.width, out.height);
      x.save(); x.scale(k, k); x.translate(-c0.x * sw, -c0.y * sh);
      drawRot(x, sw, sh, 0, 0, sw, sh); x.restore();
      const blob = await new Promise((r) => out.toBlob(r, 'image/png'));
      if (blob.size > MAX_BYTES) return { blob: await new Promise((r) => out.toBlob(r, 'image/jpeg', 0.92)), url: out.toDataURL('image/jpeg', 0.8) };
      return { blob, url: out.toDataURL('image/png') };
    }
    return { load, render, toBlob, get img() { return st.img; }, get name() { return st.name; } };
  })();
  window.__upload = up;

  // ---------- camera ----------
  const camEl = $('#cam'); const video = $('#video'); const frozen = $('#frozen');
  const cam = {
    stream: null, track: null, facing: 'environment', deviceId: null, mode: 'snapshot', mirror: false,
    liveTimer: 0, qTimer: 0, inflight: false, session: Math.random().toString(36).slice(2), frozen: false,
    liveBlob: null, liveUrl: null, starting: false, gen: 0,
  };
  window.__cam = { tracks: () => (cam.stream ? cam.stream.getTracks().filter((t) => t.readyState === 'live').length : 0), state: cam };

  function camState(kind, title, msg, actions = []) {
    const box = $('#cam-state');
    $('#cam-controls').hidden = !!kind || cam.frozen;
    $('.cam-top').hidden = kind === 'error';
    if (!kind) { box.hidden = true; return; }
    box.hidden = false; box.dataset.kind = kind;
    $('#cam-spinner').hidden = kind !== 'loading';
    $('#cam-state-icon').hidden = kind === 'loading';
    $('#cam-state-title').textContent = title; $('#cam-state-msg').textContent = msg || '';
    $('#cam-state-actions').replaceChildren(...actions);
  }
  function fallbackActions(retry = true) {
    const acts = [];
    if (retry) acts.push(el('button', { class: 'btn btn-primary', text: 'Try again', onclick: () => startCamera() }));
    acts.push(el('button', { class: 'btn btn-ghost', text: 'Upload a photo instead', onclick: () => setTab('upload') }));
    if (/Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) || isMobile()) acts.push(el('button', { class: 'btn btn-ghost', text: 'Take a photo', onclick: () => $('#native-capture').click() }));
    return acts;
  }
  $('#native-capture').addEventListener('change', (e) => { const f = e.target.files[0]; if (f) up.load(f); e.target.value = ''; });

  async function startCamera() {
    const gen = ++cam.gen;
    stopCamera(false);
    cam.frozen = false; frozen.hidden = true; $('#snap-actions').hidden = true; video.hidden = false;
    $('#live-actions').hidden = cam.mode !== 'live';
    camEl.classList.toggle('live', cam.mode === 'live');
    camState('loading', 'Starting camera…', 'Allow camera access if your browser asks.');
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
    if (!window.isSecureContext && !local) {
      camState('error', 'Camera needs a secure connection',
        'Phones only allow the camera over HTTPS. Start the server with "python ui/run.py --lan" and open the https:// address it prints.', fallbackActions(false));
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      camState('error', 'Camera not supported', "This browser can't use the camera. Try Chrome, Edge, Safari or Firefox.", fallbackActions(false));
      return;
    }
    const v = { width: { ideal: 1280 }, height: { ideal: 720 } };
    if (cam.deviceId) v.deviceId = { exact: cam.deviceId }; else v.facingMode = { ideal: cam.facing };
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false }); }
    catch (e) {
      if (gen !== cam.gen) return;
      const n = e && e.name;
      if (n === 'NotAllowedError' || n === 'SecurityError' || n === 'PermissionDeniedError')
        camState('error', 'Camera permission denied', 'Allow the camera: click the camera icon in the address bar (or Settings → Site settings → Camera), then try again.', fallbackActions());
      else if (n === 'NotFoundError' || n === 'DevicesNotFoundError' || n === 'OverconstrainedError')
        camState('error', 'No camera found', 'Connect a camera, or upload a photo of the letter instead.', fallbackActions());
      else if (n === 'NotReadableError' || n === 'TrackStartError' || n === 'AbortError')
        camState('error', 'Camera is busy', 'Another app may be using the camera. Close it and try again.', fallbackActions());
      else camState('error', 'Camera could not start', 'Something stopped the camera from starting.', fallbackActions());
      return;
    }
    if (gen !== cam.gen || S.tab !== 'camera') { stream.getTracks().forEach((t) => t.stop()); return; }
    cam.stream = stream; cam.track = stream.getVideoTracks()[0];
    video.srcObject = stream;
    try { await video.play(); } catch (_) { /* autoplay muted is allowed */ }
    await new Promise((r) => { if (video.readyState >= 2 && video.videoWidth) r(); else video.addEventListener('loadeddata', r, { once: true }); setTimeout(r, 4000); });
    if (gen !== cam.gen) return;
    const settings = cam.track.getSettings ? cam.track.getSettings() : {};
    if (settings.facingMode) cam.facing = settings.facingMode;
    setMirror(cam.facing === 'user');
    setupCaps(); listDevices();
    camState(null);
    cam.qTimer = setInterval(qualityCheck, 600);
    if (cam.mode === 'live') startLive();
  }
  function stopCamera(full = true) {
    clearInterval(cam.qTimer); clearTimeout(cam.liveTimer); cam.liveTimer = 0; cam.qTimer = 0;
    if (cam.stream) cam.stream.getTracks().forEach((t) => t.stop());
    cam.stream = null; cam.track = null; video.srcObject = null;
    $('#live-chip').hidden = true; $('#quality').hidden = true; camEl.classList.remove('stable');
    if (full) cam.gen++;
  }
  function setMirror(on) {
    cam.mirror = on; video.classList.toggle('mirror', on); frozen.classList.toggle('mirror', on);
    $('#cam-mirror').setAttribute('aria-pressed', String(on));
  }
  function setupCaps() {
    const caps = cam.track && cam.track.getCapabilities ? cam.track.getCapabilities() : {};
    $('#cam-torch').hidden = !caps.torch;
    const z = $('#zoom');
    if (caps.zoom && caps.zoom.max > caps.zoom.min) {
      z.min = caps.zoom.min; z.max = Math.min(caps.zoom.max, caps.zoom.min * 5); z.step = caps.zoom.step || 0.1;
      z.value = (cam.track.getSettings().zoom) || caps.zoom.min; $('#zoom-wrap').hidden = false;
    } else $('#zoom-wrap').hidden = true;
  }
  async function listDevices() {
    try {
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
      const sel = $('#cam-device');
      sel.replaceChildren(...devs.map((d, i) => el('option', { value: d.deviceId, text: d.label || `Camera ${i + 1}` })));
      const curId = cam.track && cam.track.getSettings().deviceId; if (curId) sel.value = curId;
      sel.hidden = devs.length < 2 || isMobile();
      $('#cam-switch').hidden = devs.length < 2 && !/Android|iPhone|iPad/i.test(navigator.userAgent);
    } catch (_) { /* ignore */ }
  }
  $('#cam-device').addEventListener('change', (e) => { cam.deviceId = e.target.value; startCamera(); });
  $('#cam-switch').addEventListener('click', () => { cam.facing = cam.facing === 'user' ? 'environment' : 'user'; cam.deviceId = null; startCamera(); });
  $('#cam-mirror').addEventListener('click', () => setMirror(!cam.mirror));
  $('#cam-torch').addEventListener('click', async (e) => {
    const on = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    try { await cam.track.applyConstraints({ advanced: [{ torch: on }] }); e.currentTarget.setAttribute('aria-pressed', String(on)); }
    catch (_) { toast('The torch could not be switched.', 'err'); }
  });
  $('#zoom').addEventListener('input', (e) => { if (cam.track) cam.track.applyConstraints({ advanced: [{ zoom: +e.target.value }] }).catch(() => {}); });
  $('#cam-close').addEventListener('click', () => setTab('draw'));
  $$('[data-cammode]').forEach((b) => b.addEventListener('click', () => {
    cam.mode = b.dataset.cammode;
    $$('[data-cammode]').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    camEl.classList.toggle('live', cam.mode === 'live');
    $('#live-actions').hidden = cam.mode !== 'live';
    if (cam.mode === 'live') { if (cam.frozen) retake(); startLive(); }
    else { clearTimeout(cam.liveTimer); cam.liveTimer = 0; $('#live-chip').hidden = true; camEl.classList.remove('stable'); }
  }));

  /** Crop the guide-box region of a source (video or frozen canvas) into a square canvas. */
  function cropGuide(src, sw, sh, out) {
    const stage = $('#cam-stage').getBoundingClientRect(); const g = $('#guide').getBoundingClientRect();
    const k = Math.max(stage.width / sw, stage.height / sh);
    const ox = (stage.width - sw * k) / 2, oy = (stage.height - sh * k) / 2;
    let x = (g.left - stage.left - ox) / k; const y = (g.top - stage.top - oy) / k; const s = g.width / k;
    if (cam.mirror) x = sw - x - s;               // display is mirrored, the frame is not
    const c = document.createElement('canvas'); c.width = c.height = out;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, out, out);
    ctx.drawImage(src, x, y, s, s, 0, 0, out, out);
    return c;
  }
  const canvasBlob = (c, type = 'image/jpeg', q = 0.9) => new Promise((r) => c.toBlob(r, type, q));

  function qualityCheck() {
    if (!cam.stream || cam.frozen || !video.videoWidth || document.hidden) return;
    const c = cropGuide(video, video.videoWidth, video.videoHeight, 64);
    const d = c.getContext('2d').getImageData(0, 0, 64, 64).data;
    const g = new Float32Array(64 * 64); let sum = 0;
    for (let i = 0; i < g.length; i++) { g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; sum += g[i]; }
    const mean = sum / g.length; let vs = 0; for (const v of g) vs += (v - mean) ** 2; const sd = Math.sqrt(vs / g.length);
    let lap = 0, lm = 0, n = 0; const L = [];
    for (let y = 1; y < 63; y++) for (let x = 1; x < 63; x++) { const i = y * 64 + x; const v = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - 64] - g[i + 64]; L.push(v); lm += v; n++; }
    lm /= n; for (const v of L) lap += (v - lm) ** 2; lap /= n;
    const q = $('#quality'); let msg = '';
    if (mean < 55) msg = 'Needs more light';
    else if (sd < 10) msg = 'Place a letter inside the box';
    else if (lap < 30) msg = 'Hold steady — image is blurry';
    q.textContent = msg; q.hidden = !msg;
  }

  // snapshot
  $('#shutter').addEventListener('click', snap);
  function snap() {
    if (!cam.stream || !video.videoWidth) return;
    const f = $('#flash'); f.classList.remove('go'); void f.offsetWidth; f.classList.add('go');
    frozen.width = video.videoWidth; frozen.height = video.videoHeight;
    frozen.getContext('2d').drawImage(video, 0, 0);
    cam.frozen = true; frozen.hidden = false; video.hidden = true;
    $('#snap-actions').hidden = false; $('#cam-controls').hidden = true; $('#quality').hidden = true;
  }
  function retake() {
    cam.frozen = false; frozen.hidden = true; video.hidden = false;
    $('#snap-actions').hidden = true; $('#cam-controls').hidden = false;
  }
  $('#retake').addEventListener('click', retake);
  $('#use-snap').addEventListener('click', async () => {
    const c = cropGuide(frozen, frozen.width, frozen.height, 320);
    const blob = await canvasBlob(c, 'image/png');
    useInput(blob, c.toDataURL('image/jpeg', 0.85));
  });

  // live
  function startLive() {
    clearTimeout(cam.liveTimer);
    $('#live-chip').hidden = false; $('#live-chip').classList.add('pending');
    $('#lc-letter').textContent = '–'; $('#lc-conf').textContent = ''; $('#lc-lock').hidden = true; $('#use-live').disabled = true;
    const tick = async () => {
      cam.liveTimer = 0;
      if (cam.mode !== 'live' || !cam.stream || S.tab !== 'camera') return;
      if (!document.hidden && !cam.inflight && video.videoWidth) {
        cam.inflight = true;
        const c = cropGuide(video, video.videoWidth, video.videoHeight, 224);
        const blob = await canvasBlob(c, 'image/jpeg', 0.85);
        const fd = new FormData(); fd.append('image', blob, 'frame.jpg'); fd.append('session', cam.session);
        try {
          const r = await api('/api/predict/stream', { method: 'POST', body: fd });
          if (cam.mode === 'live' && cam.stream) renderLive(r, blob, c);
        } catch (_) { /* transient: next frame */ }
        cam.inflight = false;
      }
      if (cam.mode === 'live' && cam.stream && !document.hidden) cam.liveTimer = setTimeout(tick, 380);
    };
    tick();
  }
  function renderLive(r, blob, c) {
    const chip = $('#live-chip'); chip.classList.remove('pending');
    const shown = r.known && !r.blank;
    $('#lc-letter').textContent = r.blank ? '–' : (shown ? r.label : '?');
    $('#lc-conf').textContent = r.blank ? 'no letter' : shown ? pct(r.confidence) : 'not sure';
    $('#lc-lock').hidden = !r.stable;
    camEl.classList.toggle('stable', !!r.stable);
    $('#use-live').disabled = !r.stable;
    if (r.stable) { cam.liveBlob = blob; cam.liveUrl = c.toDataURL('image/jpeg', 0.85); }
  }
  $('#use-live').addEventListener('click', () => { if (cam.liveBlob) useInput(cam.liveBlob, cam.liveUrl); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { clearTimeout(cam.liveTimer); cam.liveTimer = 0; }
    else if (S.tab === 'camera' && cam.mode === 'live' && cam.stream && !cam.liveTimer) startLive();
  });
  window.addEventListener('pagehide', () => stopCamera());

  // ---------- main action (predict / check / add sample) ----------
  async function currentInput() {
    if (S.tab === 'draw') return pad.empty ? null : pad.toBlob();
    if (S.tab === 'upload') return up.toBlob();
    if (S.tab === 'camera') {
      if (cam.frozen) { const c = cropGuide(frozen, frozen.width, frozen.height, 320); return { blob: await canvasBlob(c, 'image/png'), url: c.toDataURL('image/jpeg', 0.85) }; }
      if (cam.stream && video.videoWidth) { const c = cropGuide(video, video.videoWidth, video.videoHeight, 320); return { blob: await canvasBlob(c, 'image/png'), url: c.toDataURL('image/jpeg', 0.85) }; }
    }
    return null;
  }
  async function mainAction(why = 'button') {
    const inp = await currentInput();
    if (!inp || !inp.blob) {
      if (why === 'button') toast(S.tab === 'upload' ? 'Choose an image first.' : S.tab === 'camera' ? 'The camera is not ready yet.' : 'Draw a letter first.');
      return;
    }
    useInput(inp.blob, inp.url);
  }
  function useInput(blob, url) {
    if (S.mode === 'practice') return practiceCheck(blob, url);
    if (S.mode === 'enroll') return addSample(blob, url);
    return predict(blob, url);
  }
  $('#predict-btn').addEventListener('click', () => mainAction('button'));

  // ---------- result rendering ----------
  const body = $('#result-body');
  function setResultState(s) { body.dataset.state = s; }
  function setBusy(on) {
    const b = $('#predict-btn'); b.classList.toggle('busy', on); b.disabled = on; b.setAttribute('aria-busy', String(on));
  }
  window.__predictCount = 0;

  async function predict(blob, url) {
    const seq = ++S.seq;
    if (S.abort) S.abort.abort();
    const ac = new AbortController(); S.abort = ac;
    S.lastBlob = blob; S.lastUrl = url;
    setBusy(true);
    const had = body.dataset.state === 'result';
    const rr = $('.r-result');
    const t1 = setTimeout(() => {
      if (seq !== S.seq) return;
      if (had) { $('#r-busy').hidden = false; $('#r-busy-text').textContent = 'Recognising…'; rr.classList.add('stale'); }
      else setResultState('loading');
    }, 150);
    const t2 = setTimeout(() => {
      if (seq !== S.seq) return;
      $('#r-busy-text').textContent = 'Still working…';
      let m = $('.r-skeleton .sk-msg'); if (!m) { m = el('p', { class: 'sk-msg muted small center' }); $('.r-skeleton').append(m); }
      m.textContent = 'Still working…';
    }, 2000);
    window.__predictCount++;
    try {
      const fd = new FormData(); fd.append('image', blob, 'input.png');
      const r = await api(`/api/predict?gradcam=${S.gradcam ? 'true' : 'false'}`, { method: 'POST', body: fd, signal: ac.signal });
      if (seq !== S.seq) return;
      renderResult(r, url); addHistory(r, url);
      if (S.mode === 'recognize') openSheet('result-card');
    } catch (e) {
      if (e.name === 'AbortError' || seq !== S.seq) return;
      showError(e);
    } finally {
      clearTimeout(t1); clearTimeout(t2);
      if (seq === S.seq) { setBusy(false); $('#r-busy').hidden = true; rr.classList.remove('stale'); const m = $('.r-skeleton .sk-msg'); if (m) m.remove(); }
    }
  }
  function showError(e) {
    const offline = e.status === 0;
    $('#r-error-title').textContent = offline ? "Can't reach the recogniser" : e.status === 503 ? 'The model is still loading' : "Couldn't get a prediction";
    $('#r-error-msg').textContent = offline ? 'Check that "python ui/run.py" is still running, then retry.' : (e.message || 'Please try again.');
    setResultState('error');
    if (S.mode === 'recognize') openSheet('result-card');
  }
  $('#r-retry').addEventListener('click', () => { if (S.lastBlob) predict(S.lastBlob, S.lastUrl); else setResultState('empty'); });

  function renderResult(r, url) {
    S.lastResult = r;
    if (r.blank) {
      $('#r-error-title').textContent = 'No letter found';
      $('#r-error-msg').textContent = 'Draw or capture one letter with clear, dark strokes.';
      setResultState('error'); highlightChart(); return;
    }
    setResultState('result');
    const big = $('#big-letter');
    big.classList.remove('pop'); void big.offsetWidth; big.classList.add('pop');
    big.classList.toggle('unknown', !r.known);
    big.textContent = r.known ? r.best.letter : '?';
    big.setAttribute('aria-label', r.known ? `Predicted letter ${r.best.letter}` : 'Unknown letter');
    const n = r.known ? [...r.best.letter].length : 1;
    big.classList.toggle('len2', n === 2); big.classList.toggle('len3', n >= 3);
    $('#translit').textContent = r.known ? r.best.translit : 'unknown';
    $('#cat-en').textContent = r.known ? r.best.category_en : 'not confident';
    const chips = $('#chips'); chips.replaceChildren();
    if (r.known) {
      chips.append(el('span', { class: 'chip cat', title: r.best.category_en }, [el('span', { class: 'ta', text: r.best.category_ta })]));
      if (r.best.consonant_row) chips.append(el('span', { class: 'chip' }, ['row ', el('span', { class: 'ta', text: r.best.consonant_row })]));
      if (r.best.vowel && r.best.kind !== 'vowel') chips.append(el('span', { class: 'chip' }, ['vowel ', el('span', { class: 'ta', text: r.best.vowel })]));
    }
    const conf = r.known ? r.confidence : 0;
    const fg = $('#ring-fg'); fg.style.strokeDashoffset = String(119.4 * (1 - conf));
    fg.style.stroke = `var(--${r.known ? level(conf) : 'bad'})`;
    $('#ring-text').textContent = r.known ? pct(conf) : '—';
    $('#ring').setAttribute('aria-label', r.known ? `Confidence ${pct(conf)}` : 'Not confident');
    $('#unknown-box').hidden = r.known;
    $('.sub-title').textContent = r.known ? 'Top 5' : 'Closest matches';
    const ol = $('#top5'); ol.replaceChildren();
    for (const t of r.top) {
      const li = el('li', { class: `lvl-${level(t.confidence)}` }, [
        el('span', { class: 't-letter', text: t.letter }),
        el('span', { class: 't-bar' }, [el('span'), el('em', { text: t.translit })]),
        el('span', { class: 't-pct', text: pct(t.confidence) })]);
      ol.append(li);
    }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $$('#top5 li').forEach((li, i) => { li.querySelector('.t-bar span').style.width = `${Math.max(2, r.top[i].confidence * 100)}%`; });
    }));
    $('#saw-orig').src = url || '';
    $('#saw-pre').src = r.preprocessed;
    $('#saw-pre-cap').textContent = `Model input ${r.input_size}×${r.input_size}`;
    $('#saw-cam-fig').hidden = !r.gradcam; if (r.gradcam) $('#saw-cam').src = r.gradcam;
    $('#r-foot').textContent = `${r.latency_ms.toFixed(1)} ms · ${r.model}`;
    highlightChart();
    if (r.known) revealCell(r.label);
  }
  $('#gradcam-toggle').addEventListener('change', (e) => {
    S.gradcam = e.target.checked; store.set('gradcam', S.gradcam);
    if (S.gradcam) $('#saw').open = true;
    if (S.lastBlob && S.lastResult && !S.lastResult.blank && S.mode === 'recognize') predict(S.lastBlob, S.lastUrl);
    else $('#saw-cam-fig').hidden = !S.gradcam || !(S.lastResult && S.lastResult.gradcam);
  });
  S.gradcam = store.get('gradcam', false); $('#gradcam-toggle').checked = S.gradcam;
  $('#unknown-enroll').addEventListener('click', () => {
    const b = S.lastBlob, u = S.lastUrl;
    setMode('enroll');
    if (b) addSample(b, u);
    toast('Pick the letter from the chart, add a few more samples, then save.');
    if (isMobile()) openSheet('chart-card');
  });

  // ---------- history ----------
  const HKEY = 'tamil-ui-history';
  function thumb(url) {
    return new Promise((res) => {
      if (!url) return res('');
      const img = new Image();
      img.onload = () => { const c = document.createElement('canvas'); c.width = c.height = 64; const x = c.getContext('2d'); x.fillStyle = '#fff'; x.fillRect(0, 0, 64, 64); x.drawImage(img, 0, 0, 64, 64); res(c.toDataURL('image/jpeg', 0.7)); };
      img.onerror = () => res(''); img.src = url;
    });
  }
  async function addHistory(r, url) {
    if (r.blank) return;
    const h = store.get(HKEY, []);
    const { gradcam, ...lite } = r;
    h.unshift({ t: Date.now(), thumb: await thumb(url), result: lite });
    store.set(HKEY, h.slice(0, 20));
    if (!$('#drawer').hidden) renderHistory();
  }
  function renderHistory() {
    const h = store.get(HKEY, []); const ul = $('#history');
    ul.replaceChildren(...h.map((it, i) => el('li', {}, [el('button', {
      'aria-label': `Prediction ${it.result.known ? it.result.label : 'unknown'}`,
      onclick: () => { S.lastBlob = null; renderResult(it.result, it.thumb); closeDrawer(); if (S.mode !== 'recognize') setMode('recognize'); openSheet('result-card'); },
    }, [el('img', { src: it.thumb, alt: '' }), it.result.known ? el('span', { class: 'h-letter', text: it.result.label }) : el('span', { class: 'h-unknown', text: 'unknown' })])])));
    $('#history-empty').hidden = h.length > 0; $('#history-clear').hidden = h.length === 0;
  }
  function openDrawer() { renderHistory(); $('#drawer').hidden = false; scrim.hidden = false; $('#drawer-close').focus(); }
  function closeDrawer() { $('#drawer').hidden = true; if (!$('#chart-card').classList.contains('open')) scrim.hidden = true; }
  $('#history-btn').addEventListener('click', openDrawer);
  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#history-clear').addEventListener('click', () => { store.del(HKEY); renderHistory(); });

  // ---------- models ----------
  async function openModels() {
    const modal = $('#model-modal'); modal.hidden = false;
    const ul = $('#model-list'); ul.replaceChildren(el('li', {}, [el('div', { class: 'sk', style: 'height:56px' })]), el('li', {}, [el('div', { class: 'sk', style: 'height:56px' })]));
    try {
      const ms = await api('/api/models');
      ul.replaceChildren(...ms.map((m) => el('li', {}, [el('button', {
        'aria-current': m.active ? 'true' : 'false', disabled: !m.available,
        onclick: () => switchModel(m),
      }, [el('span', { class: 'm-name', text: m.name }),
        el('span', { class: 'm-acc', text: m.test_acc_5w5s ? `${(m.test_acc_5w5s * 100).toFixed(1)}%` : '' }),
        el('span', { class: 'm-sub', text: m.available ? `${m.active ? 'Active · ' : ''}${m.backend} · 5-way 5-shot test accuracy` : `Unavailable: ${m.reason}` })])])));
      const cur = ul.querySelector('[aria-current="true"]') || ul.querySelector('button'); if (cur) cur.focus();
    } catch (e) { ul.replaceChildren(el('li', { class: 'muted', text: 'Models could not be listed.' })); }
  }
  function closeModels() { $('#model-modal').hidden = true; }
  $('#model-badge').addEventListener('click', () => {
    if (S.health && S.health.public) toast(`${S.health.model} · ${S.health.registry_size} letters. Model switching is off on the public demo.`);
    else openModels();
  });
  $('#model-modal-close').addEventListener('click', closeModels);
  $('#model-modal').addEventListener('click', (e) => { if (e.target.id === 'model-modal') closeModels(); });
  async function switchModel(m) {
    if (m.active) { closeModels(); return; }
    closeModels();
    $('#switch-text').textContent = `Switching to ${m.name}…`;
    $('#switch-overlay').hidden = false;
    try {
      S.health = await api('/api/models/select', { method: 'POST', json: { id: m.id } });
      updateBadge(); await loadClasses();
      toast(`Now using ${m.name}`, 'ok');
      if (S.lastBlob && S.mode === 'recognize') predict(S.lastBlob, S.lastUrl);
    } catch (e) {
      toast(`${e.message} The previous model is still active.`, 'err', 4500);
      try { S.health = await api('/api/health'); updateBadge(); } catch (_) { /* ignore */ }
    } finally { $('#switch-overlay').hidden = true; }
  }

  // ---------- practice ----------
  function fillPracticeFilter() {
    const sel = $('#practice-filter'); const keep = sel.value || S.practice.filter;
    const cons = S.classes.filter((c) => c.kind === 'consonant');
    sel.replaceChildren(
      el('option', { value: 'all', text: 'All letters' }),
      el('option', { value: 'vowel', text: 'உயிர் · Vowels' }),
      el('option', { value: 'consonant', text: 'மெய் · Consonants' }),
      el('option', { value: 'compound', text: 'உயிர்மெய் · Consonant + vowel' }),
      el('optgroup', { label: 'One consonant row' }, cons.map((c) => el('option', { value: `row:${c.letter}`, text: `${c.letter[0]} row (${c.translit}-)` }))));
    sel.value = keep;
  }
  function practicePool() {
    const f = S.practice.filter;
    return S.classes.filter((c) => c.enrolled && c.in_chart && (
      f === 'all' || (f === 'compound' ? c.kind === 'compound' : f === 'vowel' ? (c.kind === 'vowel' || c.kind === 'aytham') : f === 'consonant' ? c.kind === 'consonant'
        : f.startsWith('row:') ? (c.consonant_row === f.slice(4) || c.letter === f.slice(4)) : true)));
  }
  function setTarget(letter) {
    const c = S.byLetter.get(letter); if (!c) return;
    S.practice.target = letter; S.practice.lock = false;
    $('#pt-letter').textContent = letter; $('#pt-translit').textContent = c.translit;
    const fb = $('#pr-feedback'); fb.dataset.state = 'idle';
    $('#pr-fb-ta').textContent = 'எழுதிப் பார்!'; $('#pr-fb-en').textContent = 'Write the letter shown, then press Check.'; $('#pr-fb-guess').textContent = '';
    $('#pr-next').hidden = true;
    highlightChart();
  }
  function nextTarget() {
    const pool = practicePool(); if (!pool.length) return;
    let pick; do { pick = pool[Math.floor(Math.random() * pool.length)].letter; } while (pool.length > 1 && pick === S.practice.target);
    setTarget(pick);
    if (S.tab === 'draw') pad.clear();
  }
  $('#practice-filter').addEventListener('change', (e) => { S.practice.filter = e.target.value; nextTarget(); });
  $('#pr-skip').addEventListener('click', () => { S.practice.streak = 0; updateStats(); nextTarget(); });
  $('#pr-next').addEventListener('click', nextTarget);
  function updateStats() {
    const p = S.practice;
    $('#streak').textContent = p.streak; $('#tries').textContent = p.tries;
    $('#acc').textContent = p.tries ? pct(p.correct / p.tries) : '–';
  }
  let nextTimer = 0;
  async function practiceCheck(blob) {
    const p = S.practice; if (!p.target) return;
    const fb = $('#pr-feedback'); fb.dataset.state = 'busy'; setBusy(true);
    try {
      const fd = new FormData(); fd.append('target', p.target); fd.append('image', blob, 'input.png');
      const r = await api('/api/practice/check', { method: 'POST', body: fd });
      if (r.prediction.blank) { fb.dataset.state = 'idle'; $('#pr-fb-ta').textContent = r.message_ta; $('#pr-fb-en').textContent = r.message_en; return; }
      p.tries++;
      if (r.correct) { p.correct++; p.streak++; } else p.streak = 0;
      fb.dataset.state = r.correct ? 'correct' : r.close ? 'close' : 'wrong';
      $('#pr-fb-ta').textContent = r.message_ta; $('#pr-fb-en').textContent = r.message_en;
      const g = r.prediction;
      $('#pr-fb-guess').textContent = r.correct ? `${pct(g.confidence)} confident` : g.known ? `It looked like ${g.label} (${g.best.translit}).` : 'It did not look like any letter I know well.';
      updateStats();
      $('#pr-next').hidden = false;
      clearTimeout(nextTimer);
      if (r.correct) nextTimer = setTimeout(() => { if (S.mode === 'practice' && S.practice.target === r.target) nextTarget(); }, 1800);
    } catch (e) { fb.dataset.state = 'idle'; toast(e.message, 'err'); }
    finally { setBusy(false); }
  }

  // ---------- enroll ----------
  function setEnrollLabel(letter) {
    const c = S.byLetter.get(letter); S.enroll.label = letter;
    const L = $('#en-letter'); L.textContent = letter; L.classList.add('set');
    const n = [...letter].length; L.classList.toggle('len2', n === 2); L.classList.toggle('len3', n >= 3);
    $('#en-name').textContent = c ? `${letter} · ${c.translit} · ${c.category_en}` : letter;
    $('#en-status').textContent = c && c.enrolled ? `Already enrolled with ${c.n_support} samples; new samples are added to them.` : 'Not enrolled yet. Add 3–5 samples.';
    const pair = c && c.variant_pair && S.byLetter.has(c.variant_pair) ? c.variant_pair : null;
    const w = $('#variant-warn'); w.hidden = !pair; $('#variant-ok').checked = false;
    if (pair) {
      $('#variant-msg').textContent = c.kind === 'variant'
        ? `${letter} is spelled with ெ + ள. It may be the same letter as ${pair} (written with the ௌ sign). Check that you picked the spelling you mean.`
        : `${letter} has a look-alike spelling ${pair} (ெ + ள) in the dataset. Check that you picked the spelling you mean.`;
    }
    updateEnrollUI(); highlightChart();
  }
  function addSample(blob, url) {
    if (S.enroll.samples.length >= 5) { toast('Five samples is the maximum. Remove one to add another.'); return; }
    S.enroll.samples.push({ blob, url });
    updateEnrollUI();
    if (S.tab === 'draw') pad.clear();
    if (S.tab === 'camera' && cam.frozen) retake();
  }
  function updateEnrollUI() {
    const box = $('#en-samples'); const sm = S.enroll.samples;
    const slots = [];
    for (let i = 0; i < 5; i++) {
      const s = sm[i];
      slots.push(s ? el('div', { class: 'en-slot filled' }, [el('img', { src: s.url, alt: `Sample ${i + 1}` }),
        el('button', { 'aria-label': `Remove sample ${i + 1}`, onclick: () => { sm.splice(i, 1); updateEnrollUI(); } })])
        : el('div', { class: 'en-slot', text: i < 3 ? `${i + 1}` : 'opt.' }));
    }
    box.replaceChildren(...slots);
    const needOk = !$('#variant-warn').hidden && !$('#variant-ok').checked;
    $('#en-save').disabled = !S.enroll.label || sm.length === 0 || needOk;
    $('#en-save').textContent = sm.length && sm.length < 3 ? `Save (${sm.length}; 3+ is better)` : 'Save letter';
  }
  $('#variant-ok').addEventListener('change', updateEnrollUI);
  $('#en-add').addEventListener('click', () => mainAction('button'));
  $('#en-save').addEventListener('click', () => {
    const { label, samples } = S.enroll; if (!label || !samples.length) return;
    const fd = new FormData(); fd.append('label', label);
    samples.forEach((s, i) => fd.append('images', s.blob, `sample_${i}.png`));
    const prog = $('#en-progress'); const bar = $('#en-bar'); const txt = $('#en-progress-text');
    prog.hidden = false; bar.style.width = '0%'; txt.textContent = 'Uploading samples…'; $('#en-save').disabled = true;
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/enroll');
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.style.width = `${Math.round((e.loaded / e.total) * 70)}%`; };
    xhr.upload.onload = () => { bar.style.width = '80%'; txt.textContent = 'Refreshing letter memory…'; };
    xhr.onerror = () => { prog.hidden = true; toast("Couldn't reach the recogniser.", 'err'); updateEnrollUI(); };
    xhr.onload = async () => {
      let data = {}; try { data = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status !== 200) { prog.hidden = true; toast(data.detail || 'Saving failed.', 'err'); updateEnrollUI(); return; }
      bar.style.width = '100%'; txt.textContent = `Saved! ${data.label} now has ${data.total_images} samples.`;
      toast(`${data.label} enrolled — no retraining needed.`, 'ok');
      S.enroll.samples = [];
      await loadClasses();
      setEnrollLabel(data.label);
      const cell = S.cells.get(data.label); if (cell) { cell.classList.add('glow'); revealCell(data.label); }
      setTimeout(() => { prog.hidden = true; }, 2200);
    };
    xhr.send(fd);
  });

  // ---------- keyboard ----------
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeModels(); closeDrawer(); closeSheet('chart-card'); closeSheet('result-card'); return; }
    if (e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
    if (!$('#model-modal').hidden || !$('#switch-overlay').hidden) return;
    const k = e.key.toLowerCase();
    if (e.key === 'Enter' && !e.target.closest('button, a, summary, [role="button"]')) { e.preventDefault(); mainAction('button'); }
    else if (k === 'c') $('#clear-btn').click();
    else if (k === 'z') $('#undo-btn').click();
    else if (k === '1') setTab('draw'); else if (k === '2') setTab('upload'); else if (k === '3') setTab('camera');
  });

  // ---------- go ----------
  setMode('recognize');
  updateEnrollUI(); updateStats();
  boot();
})();
