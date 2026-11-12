const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const api = async (u, o = {}) => {
  const r = await fetch(u, o);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m } catch {} throw new Error(m) }
  return r.json();
};
const post = (u, b) => api(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b || {}) });
const pct = v => v == null ? '—' : (v * 100).toFixed(1) + '%';
const toast = (m, bad) => { const b = $('#banner'); b.innerHTML = `<div class="banner" style="${bad ? 'border-color:var(--r);color:var(--r)' : ''}">${m}</div>`; setTimeout(() => b.innerHTML = '', 4000) };
const guard = f => async (...a) => { try { await f(...a) } catch (e) { toast(e.message, true) } };

// ---- nav
$$('nav a').forEach(a => a.onclick = () => {
  $$('nav a').forEach(x => x.classList.toggle('on', x === a));
  $$('.page').forEach(p => p.classList.toggle('on', p.id === a.dataset.p));
  if (a.dataset.p === 'datasets') loadDatasets(); if (a.dataset.p === 'evaluation') loadEval(); if (a.dataset.p === 'training') pollTrain();
});

// ---- live status
let S = null;
async function refresh() {
  try { S = await api('/api/status') } catch { return }
  const cam = S.camera, d = S.detection, m = S.model;
  // keep the stream attached if the camera is already running (e.g. after a page reload)
  if (cam.connected && !$('#view').getAttribute('src')) $('#view').src = '/stream?' + Date.now();
  if (!cam.connected && $('#view').getAttribute('src')) $('#view').removeAttribute('src');
  $('#d-status').textContent = d.status; $('#d-status').className = 'status' + (d.running ? '' : ' off');
  $('#d-count').textContent = d.running ? `Detection count: ${d.count}` : 'Detection stopped';
  $('#d-rows').innerHTML = d.objects.map(o => `<tr><td>${o.class}</td><td>${(o.confidence * 100).toFixed(1)}%</td><td>${o.x}</td><td>${o.y}</td><td>${o.width}</td><td>${o.height}</td></tr>`).join('');
  const e = S.evaluation;
  const kv = (k, v, c = '') => `<div><span>${k}</span><b class="${c}">${v}</b></div>`;
  $('#perf').innerHTML = [
    kv('Latency', d.latency_ms == null ? '—' : d.latency_ms + ' ms'), kv('Detection FPS', d.detection_fps || '—'),
    kv('Capture FPS', cam.capture_fps || '—'), kv('Threshold', S.settings.conf.toFixed(2)),
    kv('Camera', cam.connected ? 'Connected' : 'Disconnected', cam.connected ? 'ok' : 'bad'), kv('Camera device', cam.device),
    kv('Resolution', cam.resolution ? cam.resolution.join('×') : '—'), kv('Detection', d.running ? 'Running' : 'Stopped', d.running ? 'ok' : ''),
    kv('Model', m.loaded ? 'Loaded' : 'Not Loaded', m.loaded ? 'ok' : 'bad'), kv('Inference device', m.device.toUpperCase().startsWith('CUDA') ? 'GPU' : 'CPU'),
    kv('Precision (eval)', e ? pct(e.precision) : 'n/a'), kv('Recall (eval)', e ? pct(e.recall) : 'n/a'),
    kv('mAP50 (eval)', e ? pct(e.map50) : 'n/a'), kv('mAP50-95 (eval)', e ? pct(e.map50_95) : 'n/a'),
    kv('Objects', d.objects.map(o => o.class).join(', ') || '—')].join('');
  $('#cap').textContent = 'CAP STATE: ' + S.cap_state.state;
  $('#b-ds').disabled = !cam.connected; $('#b-de').disabled = !d.running; $('#b-ce').disabled = !cam.connected; $('#b-cs').disabled = cam.connected;
  $('#ov-model').innerHTML = m.loaded ? `<span class="ok">Loaded</span> ${m.weights}<br>${m.device}` : `<span class="bad">Not loaded</span><br><span class="k">${m.error || ''}</span>`;
  $('#ov-cam').innerHTML = cam.connected ? `<span class="ok">Connected</span> cam ${cam.device} ${cam.resolution.join('×')}` : `<span class="k">Stopped</span> ${cam.error || ''}`;
  $('#ov-eval').innerHTML = e ? `${e.split} split — P ${pct(e.precision)}, R ${pct(e.recall)}, mAP50 ${pct(e.map50)}, mAP50-95 ${pct(e.map50_95)}` : 'No evaluation report yet.';
  $('#set-m').innerHTML = kv('Weights', m.weights || 'none') + kv('Inference device', m.device) + kv('Max det', S.settings.max_det) + kv('Image size', S.settings.imgsz);
}
setInterval(refresh, 700); refresh();
async function ovDs() {
  const r = await api('/api/datasets'); const ds = Object.entries(r.datasets || {});
  $('#ov-ds').innerHTML = ds.length ? ds.map(([n, x]) => `${n}: ${x.labeled}/${x.unique_frames} labeled`).join('<br>') + `<br>${r.all_ready ? '<span class=ok>Ready</span>' : '<span class=warn>Not ready for training</span>'}` : 'Not validated yet';
}
ovDs();

$('#b-cs').onclick = guard(async () => { await post('/api/camera/start', { index: +$('#camsel').value }); $('#view').src = '/stream?' + Date.now() });
$('#b-ce').onclick = guard(async () => { await post('/api/camera/stop'); $('#view').removeAttribute('src') });
$('#b-ds').onclick = guard(() => post('/api/detection/start'));
$('#b-de').onclick = guard(() => post('/api/detection/stop'));
$('#b-scan').onclick = guard(async () => { const r = await api('/api/cameras'); $('#camsel').innerHTML = r.devices.map(i => `<option value="${i}">Camera ${i}</option>`).join('') || '<option value="0">Camera 0</option>' });
const sendSet = guard(() => post('/api/settings', { conf: +$('#conf').value, iou: +$('#iou').value, imgsz: +$('#imgsz').value, max_det: +$('#maxdet').value }));
$('#conf').oninput = () => { $('#conf-v').textContent = (+$('#conf').value).toFixed(2); sendSet() };
['#iou', '#imgsz', '#maxdet'].forEach(s => $(s).onchange = sendSet);
$('#b-reload').onclick = guard(async () => { const r = await post('/api/model/reload'); toast(r.ok ? 'Model reloaded' : r.error, !r.ok) });

// ---- datasets + annotator
async function loadDatasets() {
  const r = await api('/api/datasets');
  $('#ds-rows').innerHTML = Object.entries(r.datasets || {}).map(([n, x]) => `<tr><td>${n}</td><td>${x.images}</td><td>${x.unique_frames}</td>
  <td>${x.split_counts.train || 0}/${x.split_counts.val || 0}/${x.split_counts.test || 0}</td><td>${x.labeled}/${x.unique_frames}</td>
  <td class="${x.cross_split_duplicate_images ? 'bad' : ''}">${x.cross_split_duplicate_images}</td><td>${JSON.stringify(x.class_counts)}</td>
  <td class="${x.ready_for_training ? 'ok' : 'warn'}">${x.ready_for_training ? 'Valid' : (x.missing_labels ? 'Labels missing' : 'Invalid')}</td><td><button data-n="${n}">Annotate</button></td></tr>`).join('') || '<tr><td colspan=9 class="k">Click VALIDATE DATASET</td></tr>';
  $$('#ds-rows button').forEach(b => b.onclick = () => openAnn(b.dataset.n));
  ovDs();
}
$('#b-val').onclick = guard(async () => { $('#val-msg').textContent = 'Validating…'; const r = await post('/api/datasets/validate'); $('#val-msg').textContent = r.all_ready ? 'All datasets valid' : 'Not ready: human labels missing/invalid'; loadDatasets() });

const A = { ds: null, list: [], i: 0, boxes: [], img: null, sel: -1, drag: null, prev: [], classes: {}, allowed: [] };
const cv = $('#a-cv'), cx = cv.getContext('2d');
const COL = { 0: '#ffa028', 1: '#3cc8ff', 2: '#50dc50' };
async function openAnn(ds) {
  const r = await api(`/api/annotate/${ds}/images`); Object.assign(A, { ds, list: r.images, classes: r.classes, allowed: r.allowed, i: Math.max(0, r.images.findIndex(x => !x.labeled)) });
  $('#ann').style.display = 'block'; $('#ann-title').textContent = ds;
  $('#a-cls').innerHTML = A.allowed.map(c => `<option value="${c}">${c} — ${A.classes[c]}</option>`).join(''); loadImg(); $('#ann').scrollIntoView();
}
async function loadImg() {
  const s = A.list[A.i].stem; const l = await api(`/api/annotate/${A.ds}/label/${s}`); A.boxes = l.boxes; A.sel = -1; A.proposed = l.proposed;
  $('#a-state').textContent = A.list[A.i].labeled ? 'VERIFIED' : (l.proposed ? 'PROPOSED - unverified, check then accept' : 'no label');
  $('#a-state').style.color = A.list[A.i].labeled ? 'var(--g)' : (l.proposed ? 'var(--y)' : 'var(--mute)');
  A.img = new Image(); A.img.onload = () => { cv.width = A.img.width; cv.height = A.img.height; cv.style.width = Math.min(A.img.width, 900) + 'px'; draw() }; A.img.src = `/api/annotate/${A.ds}/image/${s}`;
  const n = A.list.filter(x => x.labeled).length; $('#a-pos').textContent = `${A.i + 1}/${A.list.length} ${s} — ${n} labeled`;
}
function draw() {
  cx.drawImage(A.img, 0, 0); const W = cv.width, H = cv.height;
  A.boxes.concat(A.drag ? [A.drag] : []).forEach((b, i) => { cx.strokeStyle = COL[b.cls]; cx.lineWidth = i === A.sel ? 4 : 2; cx.setLineDash(A.proposed ? [8, 5] : []); cx.strokeRect((b.x - b.w / 2) * W, (b.y - b.h / 2) * H, b.w * W, b.h * H); cx.fillStyle = COL[b.cls]; cx.font = '16px sans-serif'; cx.fillText(A.classes[b.cls] + (b.conf != null ? ' ' + (b.conf * 100).toFixed(0) + '%' : ''), (b.x - b.w / 2) * W + 3, (b.y - b.h / 2) * H - 4) });
  cx.setLineDash([]);
}
const pos = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height] };
let st = null;
cv.onmousedown = e => { st = pos(e) };
cv.onmousemove = e => { if (!st) return; const [x, y] = pos(e); A.drag = box(st, [x, y]); draw() };
cv.onmouseup = e => {
  if (!st) return; const [x, y] = pos(e); const b = box(st, [x, y]); st = null; A.drag = null;
  if (b.w > .01 && b.h > .01) A.boxes.push(b);
  else { A.sel = A.boxes.findIndex(q => Math.abs(x - q.x) < q.w / 2 && Math.abs(y - q.y) < q.h / 2) }
  draw();
};
const clamp = v => Math.min(1, Math.max(0, v));
function box(a, b) { const x1 = clamp(Math.min(a[0], b[0])), x2 = clamp(Math.max(a[0], b[0])), y1 = clamp(Math.min(a[1], b[1])), y2 = clamp(Math.max(a[1], b[1])); return { cls: +$('#a-cls').value, x: (x1 + x2) / 2, y: (y1 + y2) / 2, w: x2 - x1, h: y2 - y1 } }
const save = guard(async () => { await api(`/api/annotate/${A.ds}/label/${A.list[A.i].stem}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boxes: A.boxes }) }); A.list[A.i].labeled = true; A.prev = A.boxes.map(b => ({ ...b })); $('#a-msg').textContent = 'saved' });
const match = x => { const f = $('#a-filter').value; if (f === 'all') return true; if (x.excluded) return false; return (f === 'todo' && !x.labeled) || (f === 'proposed' && x.proposed) || (f === 'low' && x.proposed && x.conf < 0.7) };
const go = d => { let j = A.i + d; while (j >= 0 && j < A.list.length && !match(A.list[j])) j += d; if (j < 0 || j >= A.list.length) { $('#a-msg').textContent = 'no more frames in this filter'; return } A.i = j; loadImg() };
$('#a-filter').onchange = () => { if (!match(A.list[A.i])) go(1) };
// Accept = the human confirms/edits what is on screen; only this writes ground truth
const accept = async () => { await save(); go(1) };
$('#a-accept').onclick = accept;
const reject = guard(async () => { const it = A.list[A.i]; await post(`/api/annotate/${A.ds}/reject/${it.stem}`); it.proposed = false; it.rejected = true; A.boxes = []; A.proposed = false; $('#a-msg').textContent = 'proposal rejected - draw the box yourself or exclude'; draw() });
const exclude = guard(async () => { const it = A.list[A.i]; await post(`/api/annotate/${A.ds}/exclude/${it.stem}`); it.excluded = true; go(1) });
$('#a-reject').onclick = reject; $('#a-excl').onclick = exclude;
$('#a-auto').onclick = guard(async () => { await post('/api/annotate/autolabel'); $('#a-autostat').textContent = 'running…'; const t = setInterval(async () => { const s = await api('/api/annotate/autolabel/status'); $('#a-autostat').textContent = s.running ? 'running… ' + (s.tail.slice(-1)[0] || '') : (s.exit_code ? 'failed: ' + s.tail.slice(-1)[0] : 'done - reload annotator'); if (!s.running) { clearInterval(t); if (!s.exit_code && A.ds) openAnn(A.ds) } }, 4000) });
$('#a-save').onclick = save; $('#a-next').onclick = () => go(1); $('#a-prev').onclick = () => go(-1); $('#a-close').onclick = () => $('#ann').style.display = 'none';
$('#a-copy').onclick = () => { A.boxes = A.prev.map(b => ({ ...b })); draw() };
$('#a-del').onclick = () => { if (A.sel >= 0) { A.boxes.splice(A.sel, 1); A.sel = -1; draw() } };
addEventListener('keydown', e => { if ($('#ann').style.display !== 'block' || !$('#datasets').classList.contains('on') || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') go(1); else if (e.key === 'ArrowLeft') go(-1); else if (e.key === 's' || e.key === 'S') save(); else if (e.key === 'Enter') accept(); else if (e.key === 'r' || e.key === 'R') reject(); else if (e.key === 'x' || e.key === 'X') exclude(); else if (e.key === 'Delete') $('#a-del').click() });

// ---- training
function lineChart(id, series) {
  const c = $(id), g = c.getContext('2d'); g.clearRect(0, 0, c.width, c.height); const all = series.flatMap(s => s.d); if (!all.length) { g.fillStyle = '#8b979a'; g.fillText('No training run yet', 20, 30); return }
  const mx = Math.max(...all, 1e-6), n = Math.max(...series.map(s => s.d.length)); g.font = '12px sans-serif';
  series.forEach((s, k) => { g.strokeStyle = s.c; g.beginPath(); s.d.forEach((v, i) => { const x = 40 + i / Math.max(n - 1, 1) * (c.width - 60), y = c.height - 20 - v / mx * (c.height - 40); i ? g.lineTo(x, y) : g.moveTo(x, y) }); g.stroke(); g.fillStyle = s.c; g.fillText(s.n, 50 + k * 120, 14) });
  g.fillStyle = '#8b979a'; g.fillText(mx.toFixed(3), 2, 24); g.fillText('0', 20, c.height - 20);
}
async function pollTrain() {
  if (!$('#training').classList.contains('on')) return;
  try {
    const t = await api('/api/training/status'), cur = t.curves || [], last = cur[cur.length - 1] || {};
    const kv = (k, v) => `<div><span>${k}</span><b>${v}</b></div>`;
    $('#t-m').innerHTML = kv('Status', t.running ? 'Running' : (t.exit_code == null ? 'Idle' : 'Finished (exit ' + t.exit_code + ')')) + kv('Epoch', last.epoch ?? '—') + kv('Train box loss', last['train/box_loss']?.toFixed(4) ?? '—') +
      kv('Val box loss', last['val/box_loss']?.toFixed(4) ?? '—') + kv('Best mAP50-95', t.best_map50_95 != null ? t.best_map50_95.toFixed(3) : '—') + kv('Duration', t.duration_s + ' s');
    lineChart('#t-loss', [{ n: 'train box', c: '#3ddc84', d: cur.map(r => r['train/box_loss']) }, { n: 'val box', c: '#ff5c5c', d: cur.map(r => r['val/box_loss']) }, { n: 'train cls', c: '#3cc8ff', d: cur.map(r => r['train/cls_loss']) }, { n: 'val cls', c: '#ffc857', d: cur.map(r => r['val/cls_loss']) }]);
    lineChart('#t-map', [{ n: 'mAP50', c: '#3ddc84', d: cur.map(r => r['metrics/mAP50(B)']) }, { n: 'mAP50-95', c: '#3cc8ff', d: cur.map(r => r['metrics/mAP50-95(B)']) }]);
  } catch {}
  setTimeout(pollTrain, 3000);
}
$('#t-start').onclick = guard(async () => { await post('/api/training/start?epochs=' + $('#t-ep').value); toast('Training started') });
$('#t-stop').onclick = guard(() => post('/api/training/stop'));
$('#t-eval').onclick = guard(async () => { toast('Evaluating on validation split…'); await post('/api/evaluation/run?split=val'); toast('Evaluation saved'); refresh() });

// ---- evaluation
async function loadEval() {
  const r = await api('/api/evaluation'), e = r.latest; $('#ev-none').style.display = e ? 'none' : 'block';
  if (e) { $('#ev-overall').innerHTML = `Split: <b>${e.split}</b> · P ${pct(e.precision)} · R ${pct(e.recall)} · mAP50 ${pct(e.map50)} · mAP50-95 ${pct(e.map50_95)}`;
    $('#ev-pc').innerHTML = Object.entries(e.per_class).map(([n, x]) => `<tr><td>${n}</td><td>${pct(x.precision)}</td><td>${pct(x.recall)}</td><td>${pct(x.map50)}</td><td>${pct(x.map50_95)}</td></tr>`).join('') }
  const pd = r.per_dataset; $('#ev-pd').innerHTML = pd ? Object.entries(pd.datasets).map(([n, x]) => `<tr><td>${n}</td><td>${x.images}</td><td>${x.tp}</td><td>${x.fp}</td><td>${x.fn}</td><td>${x.wrong_class}</td><td>${pct(x.precision)}</td><td>${pct(x.recall)}</td></tr>`).join('') : '<tr><td colspan=8 class="k">Not run yet</td></tr>';
}
$('#ti-file').onchange = guard(async e => {
  const f = e.target.files[0]; if (!f) return; const fd = new FormData(); fd.append('file', f); const r = await api('/api/test-image', { method: 'POST', body: fd });
  $('#ti-out').innerHTML = `<div class="grid" style="grid-template-columns:1fr 1fr"><div><div class="k">Original</div><img style="width:100%" src="${URL.createObjectURL(f)}"></div><div><div class="k">Annotated</div><img style="width:100%" src="${r.annotated}"></div></div>
  <p><b>${r.status}</b> · ${r.latency_ms} ms</p><table>${r.detections.map(d => `<tr><td>${d.class}</td><td>${(d.confidence * 100).toFixed(1)}%</td><td>x${d.x} y${d.y} ${d.width}×${d.height}</td></tr>`).join('')}</table>`;
});
