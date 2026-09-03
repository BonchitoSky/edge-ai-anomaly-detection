/*
 * Edge AI operations console.
 *
 * Nothing here hardcodes a sensor or a room. The node roster comes from /nodes and
 * the sensor catalogue and feature metadata from /meta, so a room the user adds at
 * runtime renders with correct labels and units without touching this file.
 *
 * Two alert paths are kept visually distinct throughout, because they are
 * genuinely different signals:
 *   THRESHOLD — the user's own limit, checked every sample, immediate.
 *   MODEL     — the LSTM-VAE verdict, once per 30 s stride.
 */
(function () {
  'use strict';

  const LS = 'edgeai.';
  const MAX_POINTS = 150;
  const SEV_TEXT = ['Nominal', 'Warning', 'Critical'];

  const store = {
    get(k, d) {
      try {
        const v = localStorage.getItem(LS + k);
        return v === null ? d : JSON.parse(v);
      } catch (e) {
        return d;
      }
    },
    set(k, v) {
      try {
        localStorage.setItem(LS + k, JSON.stringify(v));
      } catch (e) {}
    },
  };

  // ── Module state ─────────────────────────────────────────────────────────

  let meta = { feature_meta: {}, sensor_catalog: {} };
  let nodesById = {};
  let rosterKey = '';
  let muted = store.get('muted', false);
  const alerts = [];
  const prevState = {};
  const charts = [];

  const series = {
    labels: [],
    gas: [],
    t1: [],
    h1: [],
    t2: [],
    motion: [],
    sound: [],
    err1: [],
    err2: [],
    sev1: [],
    sev2: [],
  };

  const el = (id) => document.getElementById(id);
  const rootStyle = getComputedStyle(document.documentElement);
  const token = (n) => rootStyle.getPropertyValue(n).trim();

  function esc(s) {
    return String(s == null ? '' : s).replace(
      /[&<>"']/g,
      (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]
    );
  }

  function featMeta(name) {
    return meta.feature_meta[name] || { label: name, unit: '', precision: 2, step: 1 };
  }

  function fmt(name, value) {
    if (value == null || Number.isNaN(value)) return '—';
    return Number(value).toFixed(featMeta(name).precision);
  }

  // ── Theme ────────────────────────────────────────────────────────────────

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function setTheme(mode) {
    document.documentElement.setAttribute('data-theme', mode);
    store.set('theme', mode);
    const btn = el('theme-btn');
    // The icon shows the mode you would switch TO; the tooltip says so out loud.
    btn.title = mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    applyChartTheme();
  }

  function applyChartTheme() {
    const mutedCol = token('--muted');
    const grid = token('--grid');
    charts.forEach((c) => {
      if (!c) return;
      if (c.options.plugins && c.options.plugins.legend) {
        c.options.plugins.legend.labels.color = mutedCol;
      }
      Object.values(c.options.scales || {}).forEach((s) => {
        if (s.ticks) s.ticks.color = mutedCol;
        if (s.grid && s.grid.color) s.grid.color = grid;
      });
      c.update('none');
    });
  }

  // ── Charts ───────────────────────────────────────────────────────────────

  function axisX() {
    return {
      ticks: {
        color: token('--muted'),
        maxTicksLimit: 7,
        font: { family: 'JetBrains Mono', size: 9 },
      },
      grid: { color: token('--grid') },
    };
  }

  function axisY(extra) {
    const opts = Object.assign(
      {
        ticks: { color: token('--muted'), font: { family: 'JetBrains Mono', size: 9 } },
        grid: { color: token('--grid') },
      },
      extra || {}
    );
    // Reserve room for the widest label. Chart.js sizes the axis from the ticks
    // it happens to lay out first, which clips "4,000" to ",000" and "Critical"
    // to "itical" once the data grows into wider values.
    const minWidth = opts.minAxisWidth || 46;
    delete opts.minAxisWidth;
    opts.afterFit = (scale) => {
      scale.width = Math.max(scale.width, minWidth);
    };
    return opts;
  }

  function baseOpts(scales) {
    return {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2),
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          align: 'end',
          labels: {
            color: token('--muted'),
            boxWidth: 10,
            boxHeight: 10,
            usePointStyle: true,
            font: { family: 'Inter', size: 10 },
          },
        },
        tooltip: { titleFont: { family: 'JetBrains Mono' }, bodyFont: { family: 'Inter' } },
      },
      scales: scales || { x: axisX(), y: axisY() },
    };
  }

  function line(label, data, color, opts) {
    return Object.assign(
      {
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: color + '1f',
        borderWidth: 1.6,
        pointRadius: 0,
        tension: 0.32,
      },
      opts || {}
    );
  }

  // Horizontal reference line with a label at the right edge. Used for the buzzer
  // trip point and the model threshold — both meaningless to a viewer unless they
  // can see where the line actually sits.
  function refLinePlugin(getLines) {
    return {
      id: 'reflines',
      afterDatasetsDraw(chart) {
        const { ctx, chartArea, scales } = chart;
        getLines().forEach((ln) => {
          const scale = scales[ln.axis || 'y'];
          if (!scale) return;
          const y = scale.getPixelForValue(ln.value);
          if (y < chartArea.top || y > chartArea.bottom) return;
          ctx.save();
          ctx.strokeStyle = ln.color;
          ctx.setLineDash(ln.dash || [5, 4]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chartArea.left, y);
          ctx.lineTo(chartArea.right, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = ln.color;
          ctx.font = '10px JetBrains Mono, monospace';
          ctx.textAlign = 'right';
          ctx.fillText(ln.label, chartArea.right - 4, y - 4);
          ctx.restore();
        });
      },
    };
  }

  // The gas reference line follows the user's own limit, so moving it in settings
  // moves the line on the chart.
  function gasLimit() {
    const n = nodesById[1];
    const t = n && n.thresholds && n.thresholds.gas_adc;
    return t && t.enabled ? Number(t.value) : null;
  }

  function buildCharts() {
    const c = {
      crit: token('--crit'),
      warn: token('--warn'),
      accent: token('--accent'),
      blue: token('--blue'),
      violet: token('--violet'),
    };

    charts.push(
      new Chart(el('gas-chart'), {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [line('Air quality (ADC)', series.gas, c.crit, { fill: true })],
        },
        options: baseOpts({ x: axisX(), y: axisY({ min: 0, suggestedMax: 3600 }) }),
        plugins: [
          refLinePlugin(() => {
            const v = gasLimit();
            return v == null
              ? []
              : [{ value: v, color: c.crit, label: 'alert ' + v, dash: [6, 3] }];
          }),
        ],
      })
    );

    charts.push(
      new Chart(el('climate-chart'), {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [
            line('Node 1 temp (°C)', series.t1, c.warn),
            line('Node 2 temp (°C)', series.t2, c.accent),
            line('Node 1 humidity (%RH)', series.h1, c.blue, {
              yAxisID: 'y1',
              borderDash: [4, 3],
            }),
          ],
        },
        options: baseOpts({
          x: axisX(),
          y: axisY({ position: 'left' }),
          y1: axisY({ position: 'right', min: 0, max: 100, grid: { drawOnChartArea: false } }),
        }),
      })
    );

    charts.push(
      new Chart(el('occupancy-chart'), {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [
            line('Motion duty', series.motion, c.accent, { fill: true }),
            line('Sound events', series.sound, c.violet, { yAxisID: 'y1', stepped: true }),
          ],
        },
        options: baseOpts({
          x: axisX(),
          y: axisY({ min: 0, max: 1 }),
          y1: axisY({
            position: 'right',
            min: 0,
            suggestedMax: 8,
            grid: { drawOnChartArea: false },
          }),
        }),
      })
    );

    charts.push(
      new Chart(el('err-chart'), {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [
            line('Node 1', series.err1, c.crit, { fill: true }),
            line('Node 2', series.err2, c.accent, { fill: true }),
          ],
        },
        options: baseOpts({ x: axisX(), y: axisY({ min: 0 }) }),
        plugins: [refLinePlugin(() => [{ value: 0.02, color: c.warn, label: 'threshold 0.0200' }])],
      })
    );

    charts.push(
      new Chart(el('severity-chart'), {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [
            line('Node 1', series.sev1, c.crit, { stepped: true, fill: true }),
            line('Node 2', series.sev2, c.accent, { stepped: true }),
          ],
        },
        options: baseOpts({
          x: axisX(),
          y: axisY({
            min: 0,
            max: 2,
            minAxisWidth: 62,
            ticks: {
              stepSize: 1,
              color: token('--muted'),
              font: { family: 'JetBrains Mono', size: 9 },
              callback: (v) => SEV_TEXT[v] || '',
            },
          }),
        }),
      })
    );
  }

  // ── Node cards ───────────────────────────────────────────────────────────

  const GEAR =
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.8"/>' +
    '<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2 2 2 0 1 1-4 0 1.7 1.7 0 0 0-2.9-1.2l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 3 15a2 2 0 1 1 0-4 1.7 1.7 0 0 0 1.2-2.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 10 4.1a2 2 0 1 1 4 0 1.7 1.7 0 0 0 2.9 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0 1.2 2.9 2 2 0 1 1 0 4 1.7 1.7 0 0 0-1.5 1z" ' +
    'stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>';

  function rosterSignature(nodes) {
    return nodes.map((n) => n.node + ':' + n.sensors.join('|') + ':' + n.name).join(',');
  }

  function renderNodeCards(nodes) {
    const strip = el('node-strip');
    strip.innerHTML = '';

    nodes.forEach((n) => {
      const isMaster = n.role === 'master';
      const card = document.createElement('article');
      card.className = 'node-card';
      card.id = 'node-card-' + n.node;
      card.dataset.sev = '0';
      card.dataset.online = 'false';
      card.innerHTML =
        '<div class="node-head"><div class="node-title">' +
        '<div class="node-name">' +
        esc(n.name) +
        '</div>' +
        '<div class="node-meta">' +
        esc(n.short) +
        (n.room ? ' · ' + esc(n.room) : '') +
        '</div></div>' +
        (isMaster
          ? ''
          : '<button class="node-settings" type="button" data-settings="' +
            n.node +
            '" title="Room settings and alert limits" aria-label="Settings for ' +
            esc(n.name) +
            '">' +
            GEAR +
            '</button>') +
        '<span class="state-pill" data-sev="off" id="pill-' +
        n.node +
        '">Offline</span></div>' +
        '<div class="tiles" id="tiles-' +
        n.node +
        '"></div>' +
        '<div class="node-foot">' +
        (isMaster
          ? '<span id="foot-a-' + n.node + '">relaying —</span>'
          : '<span class="buzzer" data-on="false" id="buzz-' +
            n.node +
            '">BUZZER</span><span class="sep">·</span>' +
            '<span id="foot-a-' +
            n.node +
            '">score —</span>') +
        '<span class="sep">·</span><span id="foot-b-' +
        n.node +
        '">loss —</span>' +
        '<span class="sep">·</span><span id="foot-c-' +
        n.node +
        '">—</span></div>';
      strip.appendChild(card);
    });

    const add = document.createElement('button');
    add.className = 'node-card add';
    add.type = 'button';
    add.id = 'add-node-card';
    add.innerHTML =
      '<span class="add-plus" aria-hidden="true">+</span>' +
      '<span class="add-title">Add a room</span>' +
      '<span class="add-sub">Name it and choose which sensors it has</span>';
    add.addEventListener('click', openAddDialog);
    strip.appendChild(add);

    strip.querySelectorAll('[data-settings]').forEach((btn) => {
      btn.addEventListener('click', () => openEditDialog(Number(btn.dataset.settings)));
    });
  }

  function updateTiles(nodeId, features, breaches) {
    const host = el('tiles-' + nodeId);
    if (!host) return;
    const names = Object.keys(features);
    if (!names.length) return;
    const node = nodesById[nodeId] || {};
    const thresholds = node.thresholds || {};

    if (host.dataset.keys !== names.join(',')) {
      host.dataset.keys = names.join(',');
      host.innerHTML = names
        .map((n) => {
          const m = featMeta(n);
          const t = thresholds[n];
          const limit =
            t && t.enabled
              ? '<div class="tile-limit">' +
                (t.op === 'lt' ? '&lt; ' : '&gt; ') +
                esc(t.value) +
                '</div>'
              : '<div class="tile-limit"></div>';
          return (
            '<div class="tile" data-feature="' +
            esc(n) +
            '"><div class="tile-label">' +
            esc(m.label) +
            '</div><div class="tile-value"><span class="v">—</span>' +
            '<span class="tile-unit">' +
            esc(m.unit) +
            '</span></div>' +
            limit +
            '</div>'
          );
        })
        .join('');
    }

    const breached = new Set(breaches || []);
    names.forEach((n) => {
      const tile = host.querySelector('[data-feature="' + n + '"]');
      if (!tile) return;
      tile.querySelector('.v').textContent = fmt(n, features[n]);
      tile.dataset.breach = breached.has(n) ? 'true' : 'false';
    });
  }

  function updateNodeCard(f) {
    const card = el('node-card-' + f.node);
    if (!card) return;
    const sev = f.severity || 0;
    const cfg = nodesById[f.node] || {};
    const isMaster = cfg.role === 'master';

    card.dataset.sev = String(sev);
    card.dataset.online = 'true';

    const pill = el('pill-' + f.node);
    if (pill) {
      pill.dataset.sev = isMaster ? '0' : String(sev);
      pill.textContent = isMaster ? 'Relaying' : SEV_TEXT[sev];
    }

    updateTiles(f.node, f.features || {}, f.threshold_breaches);

    const buzz = el('buzz-' + f.node);
    if (buzz) buzz.dataset.on = f.rule_alert ? 'true' : 'false';

    const a = el('foot-a-' + f.node);
    if (a) {
      a.textContent = isMaster
        ? 'relaying ' + (f.relayed != null ? f.relayed : '—') + ' nodes'
        : 'score ' + (f.err != null ? Number(f.err).toFixed(4) : '—');
    }
    const c = el('foot-c-' + f.node);
    if (c) {
      c.textContent = isMaster
        ? 'OLED live'
        : f.next_verdict_s != null
          ? 'next verdict ' + Math.round(f.next_verdict_s) + 's'
          : '—';
    }
  }

  function refreshNodes(force) {
    return fetch('/nodes')
      .then((r) => r.json())
      .then((d) => {
        const nodes = d.nodes || [];
        const key = rosterSignature(nodes);
        nodesById = {};
        nodes.forEach((n) => (nodesById[n.node] = n));

        if (force || key !== rosterKey) {
          rosterKey = key;
          renderNodeCards(nodes);
        }

        nodes.forEach((n) => {
          const b = el('foot-b-' + n.node);
          if (b) b.textContent = 'loss ' + n.loss_pct.toFixed(1) + '%';
          const card = el('node-card-' + n.node);
          if (card) card.dataset.online = String(n.online);
          if (!n.online) {
            const pill = el('pill-' + n.node);
            if (pill) {
              pill.dataset.sev = 'off';
              pill.textContent = 'Offline';
            }
          }
        });

        const online = nodes.filter((n) => n.online).length;
        const worst = nodes.reduce((m, n) => Math.max(m, n.severity || 0), 0);
        const anyBuzzer = nodes.some((n) => n.rule_alert);
        const chip = el('system-chip');
        chip.className =
          'chip ' + (worst === 2 || anyBuzzer ? 'crit' : worst === 1 ? 'warn' : 'ok');
        el('system-text').textContent = anyBuzzer
          ? 'Alert — buzzer active'
          : worst === 2
            ? 'Critical alert'
            : worst === 1
              ? 'Warning'
              : online + '/' + nodes.length + ' nodes nominal';
        return nodes;
      })
      .catch(() => []);
  }

  // ── Add / edit dialogs ───────────────────────────────────────────────────

  function sensorCheckboxes(hostId, selected) {
    const host = el(hostId);
    const chosen = new Set(selected || []);
    host.innerHTML = Object.keys(meta.sensor_catalog)
      .map((key) => {
        const s = meta.sensor_catalog[key];
        return (
          '<label class="sensor-opt"><input type="checkbox" value="' +
          esc(key) +
          '"' +
          (chosen.has(key) ? ' checked' : '') +
          '><span><span class="sensor-name">' +
          esc(s.label) +
          '</span><br><span class="sensor-detail">' +
          esc(s.detail) +
          '</span></span></label>'
        );
      })
      .join('');
  }

  function pickedSensors(hostId) {
    return [...el(hostId).querySelectorAll('input:checked')].map((i) => i.value);
  }

  function showError(id, msg) {
    const e = el(id);
    e.textContent = msg;
    e.hidden = !msg;
  }

  function openAddDialog() {
    el('add-name').value = '';
    el('add-room').value = '';
    sensorCheckboxes('add-sensors', ['dht11']);
    showError('add-error', '');
    el('add-dialog').showModal();
  }

  let editingId = null;

  function openEditDialog(nodeId) {
    const node = nodesById[nodeId];
    if (!node) return;
    editingId = nodeId;
    el('edit-title').textContent = node.name;
    el('edit-sub').textContent = node.builtin
      ? 'A node that physically exists. Its sensors and limits are yours to set; the node itself cannot be removed.'
      : 'A room you added. Readings are simulated until hardware is installed for it.';
    el('edit-name').value = node.name;
    el('edit-room').value = node.room || '';
    sensorCheckboxes('edit-sensors', node.sensors);
    el('edit-delete').hidden = !!node.builtin;
    showError('edit-error', '');
    renderLimits(node);
    el('edit-dialog').showModal();
  }

  function renderLimits(node) {
    const host = el('edit-limits');
    const feats = node.feature_names || [];
    if (!feats.length) {
      host.innerHTML = '<p class="field-hint">Choose a sensor to set a limit for it.</p>';
      return;
    }
    host.innerHTML = feats
      .map((f) => {
        const m = featMeta(f);
        const t = (node.thresholds || {})[f] || { op: 'gt', value: 0, enabled: false };
        return (
          '<div class="limit-row" data-feature="' +
          esc(f) +
          '" data-enabled="' +
          (t.enabled ? 'true' : 'false') +
          '">' +
          '<div><div class="limit-name">' +
          esc(m.label) +
          '</div><div class="limit-sub">' +
          esc(m.unit) +
          '</div></div>' +
          '<div class="limit-input">' +
          '<select class="limit-op"><option value="gt"' +
          (t.op === 'gt' ? ' selected' : '') +
          '>above</option><option value="lt"' +
          (t.op === 'lt' ? ' selected' : '') +
          '>below</option></select>' +
          '<input class="limit-value" type="number" value="' +
          esc(t.value) +
          '" min="' +
          esc(m.min != null ? m.min : 0) +
          '" max="' +
          esc(m.max != null ? m.max : 9999) +
          '" step="' +
          esc(m.step || 1) +
          '"></div>' +
          '<div style="display:flex;flex-direction:column;gap:.25rem">' +
          '<label class="limit-toggle"><input type="checkbox" class="limit-enabled"' +
          (t.enabled ? ' checked' : '') +
          '>alert</label>' +
          '<label class="limit-toggle"><input type="checkbox" class="limit-buzzer"' +
          (t.buzzer ? ' checked' : '') +
          '>buzzer</label></div>' +
          '</div>'
        );
      })
      .join('');

    host.querySelectorAll('.limit-enabled').forEach((cb) => {
      cb.addEventListener('change', () => {
        cb.closest('.limit-row').dataset.enabled = cb.checked ? 'true' : 'false';
      });
    });
  }

  function collectLimits() {
    return [...el('edit-limits').querySelectorAll('.limit-row')].map((row) => ({
      feature: row.dataset.feature,
      op: row.querySelector('.limit-op').value,
      value: Number(row.querySelector('.limit-value').value),
      enabled: row.querySelector('.limit-enabled').checked,
      buzzer: row.querySelector('.limit-buzzer').checked,
    }));
  }

  function wireDialogs() {
    document.querySelectorAll('[data-close]').forEach((btn) => {
      btn.addEventListener('click', () => el(btn.dataset.close).close());
    });

    el('add-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const sensors = pickedSensors('add-sensors');
      if (!sensors.length) return showError('add-error', 'Pick at least one sensor for this room.');
      fetch('/api/nodes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: el('add-name').value,
          room: el('add-room').value,
          sensors: sensors,
        }),
      })
        .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
        .then(({ ok, d }) => {
          if (!ok) return showError('add-error', d.error || 'Could not add the room.');
          el('add-dialog').close();
          pushAlert('info', 'Room added — ' + d.node.name, 'Sensors: ' + sensors.join(', '), 0);
          refreshNodes(true);
        })
        .catch(() => showError('add-error', 'Could not reach the server.'));
    });

    el('edit-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const id = editingId;
      const sensors = pickedSensors('edit-sensors');
      if (!sensors.length) return showError('edit-error', 'A room needs at least one sensor.');
      const limits = collectLimits();

      fetch('/api/nodes/' + id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: el('edit-name').value,
          room: el('edit-room').value,
          sensors: sensors,
        }),
      })
        .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
        .then(({ ok, d }) => {
          if (!ok) throw new Error(d.error || 'save failed');
          // Limits are applied after the sensor list, because a feature must
          // still belong to the node for its limit to be accepted.
          const kept = new Set(d.node.sensors);
          return Promise.all(
            limits
              .filter(() => kept.size)
              .map((l) =>
                fetch('/api/nodes/' + id + '/threshold', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(l),
                })
              )
          );
        })
        .then(() => {
          el('edit-dialog').close();
          return refreshNodes(true);
        })
        .then(() => charts.forEach((c) => c && c.update('none')))
        .catch((err) => showError('edit-error', err.message || 'Could not save.'));
    });

    el('edit-delete').addEventListener('click', () => {
      const node = nodesById[editingId];
      if (!node) return;
      fetch('/api/nodes/' + editingId, { method: 'DELETE' })
        .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
        .then(({ ok, d }) => {
          if (!ok) return showError('edit-error', d.error || 'Could not remove the room.');
          el('edit-dialog').close();
          pushAlert('info', 'Room removed — ' + node.name, 'No longer monitored', 0);
          refreshNodes(true);
        })
        .catch(() => showError('edit-error', 'Could not reach the server.'));
    });

    [el('add-dialog'), el('edit-dialog'), el('qr-dialog')].forEach((dlg) => {
      dlg.addEventListener('click', (e) => {
        if (e.target === dlg) dlg.close();
      });
    });
  }

  // ── Alert feed ───────────────────────────────────────────────────────────

  function pushAlert(kind, title, sub, sev) {
    alerts.unshift({ kind, title, sub, sev });
    alerts.length = Math.min(alerts.length, 40);
    renderAlerts();
  }

  function renderAlerts() {
    const host = el('alert-feed');
    const empty = el('feed-empty');
    el('alert-count').textContent = alerts.length + (alerts.length === 1 ? ' alert' : ' alerts');
    host.querySelectorAll('.feed-item').forEach((n) => n.remove());
    if (!alerts.length) {
      if (empty) empty.style.display = '';
      return;
    }
    if (empty) empty.style.display = 'none';
    alerts.forEach((a) => {
      const div = document.createElement('div');
      div.className = 'feed-item';
      div.dataset.kind = a.kind;
      div.dataset.sev = String(a.sev || 0);
      div.innerHTML =
        '<div class="feed-body"><div class="feed-title">' +
        esc(a.title) +
        '</div><div class="feed-sub">' +
        esc(a.sub) +
        '</div></div>';
      host.appendChild(div);
    });
  }

  // ── Audio ────────────────────────────────────────────────────────────────

  function beep() {
    if (muted) return;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      [
        [880, 0],
        [660, 0.16],
      ].forEach((pair) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.value = pair[0];
        osc.connect(gain);
        gain.connect(ctx.destination);
        const t = ctx.currentTime + pair[1];
        gain.gain.setValueAtTime(0.0001, t);
        gain.gain.exponentialRampToValueAtTime(0.12, t + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
        osc.start(t);
        osc.stop(t + 0.15);
      });
    } catch (e) {}
  }

  // ── Frames ───────────────────────────────────────────────────────────────

  function push(arr, v) {
    arr.push(v);
    if (arr.length > MAX_POINTS) arr.shift();
  }

  let pending = false;

  function handleFrame(f) {
    updateNodeCard(f);

    const prev = prevState[f.node] || { severity: 0, rule_alert: 0 };
    const cfg = nodesById[f.node] || {};
    const short = cfg.name || 'Node ' + f.node;
    const breaches = f.threshold_breaches || [];

    // Threshold and model alerts are reported separately so it is always clear
    // which path fired — that separation is the safety argument.
    if (f.rule_alert && !prev.rule_alert) {
      const names = breaches.map((b) => featMeta(b).label).join(', ') || 'limit';
      pushAlert('rule', short + ' — ' + names + ' over limit', 'Buzzer fired locally', 2);
      beep();
    }
    if ((f.severity || 0) > 0 && prev.severity === 0) {
      pushAlert(
        'ml',
        short + ' — anomaly detected (' + (f.fault || 'unknown') + ')',
        'Model score ' +
          Number(f.err || 0).toFixed(4) +
          ' · ' +
          SEV_TEXT[f.severity] +
          (f.rule_alert ? '' : ' · buzzer not triggered'),
        f.severity
      );
      if (f.severity === 2 && !f.rule_alert) beep();
    }
    prevState[f.node] = { severity: f.severity || 0, rule_alert: f.rule_alert || 0 };

    // Charts follow the two built-in nodes; rooms the user adds appear as cards
    // and in the alert feed, but have no scripted history to plot.
    const feats = f.features || {};
    if (f.node === 1) {
      push(series.labels, ((f.ts || 0) / 1000).toFixed(0) + 's');
      push(series.gas, feats.gas_adc);
      push(series.t1, feats.temp_c);
      push(series.h1, feats.humidity);
      push(series.err1, f.err);
      push(series.sev1, f.severity || 0);
    } else if (f.node === 2) {
      push(series.t2, feats.temp_c);
      push(series.motion, feats.motion_duty);
      push(series.sound, feats.sound_events);
      push(series.err2, f.err);
      push(series.sev2, f.severity || 0);
    }

    if (!pending) {
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        charts.forEach((c) => c && c.update('none'));
      });
    }
  }

  // ── Events table ─────────────────────────────────────────────────────────

  function renderEvents(events) {
    const body = el('event-tbody');
    const empty = el('log-empty');
    if (!events.length) {
      body.innerHTML = '';
      empty.style.display = '';
      return;
    }
    empty.style.display = 'none';
    body.innerHTML = events
      .slice(0, 40)
      .map(
        (e) =>
          '<tr>' +
          '<td data-label="Node"><span class="node-tag" data-node="' +
          esc(e.node) +
          '">' +
          esc(e.node_short) +
          '</span></td>' +
          '<td data-label="Start" class="num">' +
          ((e.start_ts || 0) / 1000).toFixed(0) +
          's</td>' +
          '<td data-label="Duration" class="num">' +
          ((e.duration_ms || 0) / 1000).toFixed(0) +
          's</td>' +
          '<td data-label="Peak score" class="num">' +
          Number(e.peak_err || 0).toFixed(4) +
          '</td>' +
          '<td data-label="Severity"><span class="sev-pill ' +
          (e.peak_severity === 2 ? 'crit' : '') +
          '">' +
          SEV_TEXT[e.peak_severity || 0] +
          '</span></td>' +
          '<td data-label="Fault">' +
          esc(e.dominant_fault || '—') +
          '</td>' +
          '<td data-label="Buzzer">' +
          (e.rule_alert ? 'fired' : '—') +
          '</td>' +
          '</tr>'
      )
      .join('');
  }

  function pollEvents() {
    fetch('/events')
      .then((r) => r.json())
      .then((d) => renderEvents(d.events || []))
      .catch(() => {});
  }

  // ── Scenario ─────────────────────────────────────────────────────────────

  function pollScenario() {
    if (!window.DEMO_MODE) return;
    fetch('/scenario')
      .then((r) => r.json())
      .then((d) => {
        if (!d.active) return;
        const l = el('phase-label');
        const dt = el('phase-detail');
        if (l) l.textContent = d.phase_label;
        if (dt) dt.textContent = '— ' + d.phase_detail;
      })
      .catch(() => {});
  }

  function wireTriggers() {
    document.querySelectorAll('[data-trigger]').forEach((btn) => {
      btn.addEventListener('click', () => {
        btn.disabled = true;
        fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ event: btn.dataset.trigger }),
        })
          .then((r) => r.json())
          .then((d) => {
            if (d.ok) pushAlert('info', 'Scenario jumped to ' + d.label, 'Triggered manually', 0);
            pollScenario();
          })
          .catch(() => {})
          .finally(() => {
            btn.disabled = false;
          });
      });
    });
  }

  // ── Stream ───────────────────────────────────────────────────────────────

  function connect() {
    const es = new EventSource('/stream');
    es.onmessage = (e) => {
      let obj;
      try {
        obj = JSON.parse(e.data);
      } catch (err) {
        return;
      }
      if (obj && obj.node != null) handleFrame(obj);
    };
    es.onerror = () => {
      const chip = el('system-chip');
      chip.className = 'chip warn';
      el('system-text').textContent = 'Reconnecting';
      es.close();
      setTimeout(connect, 2000);
    };
  }

  // ── Controls ─────────────────────────────────────────────────────────────

  function wireControls() {
    setTheme(currentTheme());
    el('theme-btn').addEventListener('click', () => {
      setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    });

    const mute = el('mute-btn');
    const paintMute = () => {
      mute.setAttribute('aria-pressed', String(muted));
      mute.firstElementChild.innerHTML = muted ? '&#128263;' : '&#128266;';
      mute.title = muted ? 'Alert sound off' : 'Alert sound on';
    };
    paintMute();
    mute.addEventListener('click', () => {
      muted = !muted;
      store.set('muted', muted);
      paintMute();
    });

    el('qr-btn').addEventListener('click', () => el('qr-dialog').showModal());
    el('qr-close').addEventListener('click', () => el('qr-dialog').close());
    // The QR endpoint 503s when the optional dependency is missing; the URL alone
    // is still usable, so hide the broken image rather than the whole card.
    el('qr-img').addEventListener('error', function () {
      this.style.display = 'none';
    });

    let raf = null;
    window.addEventListener('resize', () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => charts.forEach((c) => c && c.resize()));
    });
  }

  // ── Boot ─────────────────────────────────────────────────────────────────

  fetch('/meta')
    .then((r) => r.json())
    .then((m) => {
      meta = m;
      return refreshNodes(true);
    })
    .then(() => {
      buildCharts();
      wireControls();
      wireDialogs();
      wireTriggers();
      connect();
      pollEvents();
      pollScenario();
      setInterval(refreshNodes, 2000);
      setInterval(pollEvents, 3000);
      setInterval(pollScenario, 1500);
    })
    .catch((err) => {
      const chip = el('system-chip');
      if (chip) {
        chip.className = 'chip crit';
        el('system-text').textContent = 'Cannot reach server';
      }
      console.error('Failed to start dashboard', err);
    });
})();
