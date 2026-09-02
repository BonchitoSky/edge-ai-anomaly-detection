/*
 * Edge AI operations console.
 *
 * Nothing here hardcodes a sensor. Node identity and feature metadata (label,
 * unit, precision) come from /meta, so adding a sensor to a profile changes the
 * firmware, ml/profiles.py and dashboard/nodes.py — and this file keeps working.
 *
 * Everything is keyed by node id. The previous single-device version kept one
 * buffer and one anomaly run for the whole page, which silently mixed nodes.
 */
(function () {
  'use strict';

  const LS = 'edgeai.';
  const MAX_POINTS = 150;

  // MQ-135 rule threshold, mirroring GAS_ALERT_ON_ADC in the firmware. Drawn on
  // the gas chart so the buzzer trip point is visible rather than implied.
  const GAS_ALERT_ON = 3000;
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

  let meta = { feature_meta: {}, node_defs: {} };
  let muted = store.get('muted', false);
  const alerts = [];
  const prevState = {}; // node -> { severity, rule_alert } for edge detection
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
  const token = (name) => rootStyle.getPropertyValue(name).trim();

  function esc(s) {
    return String(s == null ? '' : s).replace(
      /[&<>"']/g,
      (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]
    );
  }

  // ── Theme ────────────────────────────────────────────────────────────────

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function setTheme(mode) {
    document.documentElement.setAttribute('data-theme', mode);
    store.set('theme', mode);
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
    return Object.assign(
      {
        ticks: { color: token('--muted'), font: { family: 'JetBrains Mono', size: 9 } },
        grid: { color: token('--grid') },
      },
      extra || {}
    );
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

  function buildCharts() {
    const c = {
      crit: token('--crit'),
      warn: token('--warn'),
      ok: token('--ok'),
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
          refLinePlugin(() => [
            { value: GAS_ALERT_ON, color: c.crit, label: 'buzzer ' + GAS_ALERT_ON, dash: [6, 3] },
          ]),
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
          y1: axisY({
            position: 'right',
            min: 0,
            max: 100,
            grid: { drawOnChartArea: false },
          }),
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

  function renderNodeCards() {
    const strip = el('node-strip');
    strip.innerHTML = '';
    Object.keys(meta.node_defs)
      .map(Number)
      .sort((a, b) => a - b)
      .forEach((id) => {
        const def = meta.node_defs[String(id)];
        const isMaster = def.role === 'master';
        const card = document.createElement('article');
        card.className = 'node-card';
        card.id = 'node-card-' + id;
        card.dataset.sev = '0';
        card.dataset.online = 'false';
        card.innerHTML =
          '<div class="node-head"><div class="node-title">' +
          '<div class="node-name">' +
          esc(def.name) +
          '</div>' +
          '<div class="node-meta">' +
          esc(def.short) +
          ' · ' +
          esc(def.location) +
          '</div></div>' +
          '<span class="state-pill" data-sev="off" id="pill-' +
          id +
          '">Offline</span></div>' +
          '<div class="tiles" id="tiles-' +
          id +
          '"></div>' +
          '<div class="node-foot">' +
          (isMaster
            ? '<span id="foot-a-' + id + '">relaying —</span>'
            : '<span class="buzzer" data-on="false" id="buzz-' +
              id +
              '">BUZZER</span><span class="sep">·</span>' +
              '<span id="foot-a-' +
              id +
              '">score —</span>') +
          '<span class="sep">·</span><span id="foot-b-' +
          id +
          '">loss —</span>' +
          '<span class="sep">·</span><span id="foot-c-' +
          id +
          '">—</span></div>';
        strip.appendChild(card);
      });
  }

  function fmt(name, value) {
    const m = meta.feature_meta[name] || { precision: 2 };
    if (value == null || Number.isNaN(value)) return '—';
    return Number(value).toFixed(m.precision);
  }

  function updateTiles(nodeId, features, ruleAlert) {
    const host = el('tiles-' + nodeId);
    if (!host) return;
    const names = Object.keys(features);
    if (!names.length) return;

    if (host.childElementCount !== names.length) {
      host.innerHTML = names
        .map((n) => {
          const m = meta.feature_meta[n] || { label: n, unit: '' };
          return (
            '<div class="tile" data-feature="' +
            esc(n) +
            '"><div class="tile-label">' +
            esc(m.label) +
            '</div><div class="tile-value"><span class="v">—</span>' +
            '<span class="tile-unit">' +
            esc(m.unit) +
            '</span></div></div>'
          );
        })
        .join('');
    }
    names.forEach((n) => {
      const tile = host.querySelector('[data-feature="' + n + '"]');
      if (!tile) return;
      tile.querySelector('.v').textContent = fmt(n, features[n]);
      // Only the gas tile carries an alarm state, because only it drives the
      // rule-based buzzer path.
      tile.dataset.alarm = n === 'gas_adc' && ruleAlert ? 'true' : 'false';
    });
  }

  function updateNodeCard(f) {
    const id = f.node;
    const card = el('node-card-' + id);
    if (!card) return;
    const sev = f.severity || 0;
    const isMaster = (meta.node_defs[String(id)] || {}).role === 'master';

    card.dataset.sev = String(sev);
    card.dataset.online = 'true';

    const pill = el('pill-' + id);
    if (pill) {
      pill.dataset.sev = isMaster ? '0' : String(sev);
      pill.textContent = isMaster ? 'Relaying' : SEV_TEXT[sev];
    }

    updateTiles(id, f.features || {}, f.rule_alert);

    const buzz = el('buzz-' + id);
    if (buzz) buzz.dataset.on = f.rule_alert ? 'true' : 'false';

    const a = el('foot-a-' + id);
    if (a) {
      a.textContent = isMaster
        ? 'relaying ' + (f.relayed != null ? f.relayed : '—') + ' nodes'
        : 'score ' + (f.err != null ? Number(f.err).toFixed(4) : '—');
    }
    const c = el('foot-c-' + id);
    if (c) {
      c.textContent = isMaster
        ? 'OLED live'
        : f.next_verdict_s != null
          ? 'next verdict ' + Math.round(f.next_verdict_s) + 's'
          : '—';
    }
  }

  function refreshLinkStats() {
    fetch('/nodes')
      .then((r) => r.json())
      .then((d) => {
        d.nodes.forEach((n) => {
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
        const online = d.nodes.filter((n) => n.online).length;
        const worst = d.nodes.reduce((m, n) => Math.max(m, n.severity || 0), 0);
        const chip = el('system-chip');
        chip.className = 'chip ' + (worst === 2 ? 'crit' : worst === 1 ? 'warn' : 'ok');
        el('system-text').textContent =
          worst === 2
            ? 'Critical alert'
            : worst === 1
              ? 'Warning'
              : online + '/' + d.nodes.length + ' nodes nominal';
      })
      .catch(() => {});
  }

  // ── Alert feed ───────────────────────────────────────────────────────────

  function pushAlert(kind, title, sub, sev) {
    alerts.unshift({ kind: kind, title: title, sub: sub, sev: sev });
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

  // ── Frame handling ───────────────────────────────────────────────────────

  function push(arr, v) {
    arr.push(v);
    if (arr.length > MAX_POINTS) arr.shift();
  }

  let pending = false;

  function handleFrame(f) {
    updateNodeCard(f);

    const prev = prevState[f.node] || { severity: 0, rule_alert: 0 };
    const short = (meta.node_defs[String(f.node)] || {}).short || 'Node ' + f.node;

    // The buzzer edge is reported separately from the model verdict. They are
    // independent paths, and conflating them would hide the safety argument.
    if (f.rule_alert && !prev.rule_alert) {
      pushAlert(
        'rule',
        short + ' — gas threshold exceeded',
        'Rule-based buzzer fired locally · MQ-135 ' + fmt('gas_adc', (f.features || {}).gas_adc),
        2
      );
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

    if (f.node === 3) return; // Master contributes no series

    const feats = f.features || {};
    if (f.node === 1) {
      // Node 1 drives the shared label axis so every series stays aligned.
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

    const dlg = el('qr-dialog');
    el('qr-btn').addEventListener('click', () => dlg.showModal());
    el('qr-close').addEventListener('click', () => dlg.close());
    dlg.addEventListener('click', (e) => {
      if (e.target === dlg) dlg.close();
    });

    // The QR endpoint 503s when the optional dependency is missing; the URL
    // alone is still usable, so hide the broken image rather than the card.
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
      renderNodeCards();
      buildCharts();
      wireControls();
      wireTriggers();
      connect();
      refreshLinkStats();
      pollEvents();
      pollScenario();
      setInterval(refreshLinkStats, 2000);
      setInterval(pollEvents, 3000);
      setInterval(pollScenario, 1500);
    })
    .catch((err) => {
      const chip = el('system-chip');
      if (chip) {
        chip.className = 'chip crit';
        el('system-text').textContent = 'Cannot reach server';
      }
      console.error('Failed to load /meta', err);
    });
})();
