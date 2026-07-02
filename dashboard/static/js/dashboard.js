/* Real-time dashboard — connects to /stream SSE and updates charts + KPIs */

(function () {
  'use strict';

  // ── Local storage config persistence ────────────────────────────────────────

  const LS_PREFIX = 'edgeai.';
  const lsGet = (key, fallback) => {
    const v = localStorage.getItem(LS_PREFIX + key);
    return v === null ? fallback : v;
  };
  const lsSet = (key, value) => localStorage.setItem(LS_PREFIX + key, value);

  let threshold     = window.THRESHOLD || 0.02;
  let autoThreshold = lsGet('autoThreshold', 'false') === 'true';
  let autoThreshVal = null;
  let maxHistory    = parseInt(lsGet('history', '100'), 10);
  let totalFrames   = 0;
  let anomalyCount  = 0;
  let muted         = lsGet('muted', 'false') === 'true';
  const sevCount    = [0, 0, 0]; // [normal, warning, critical]

  const savedThreshold = lsGet('threshold', null);
  if (savedThreshold !== null) threshold = parseFloat(savedThreshold);

  const labels   = [];
  const errData  = [];
  const axData   = [];
  const ayData   = [];
  const azData   = [];
  const gxData   = [];
  const gyData   = [];
  const gzData   = [];
  const sevData  = []; // 0/1/2 per frame

  // ── Theme ────────────────────────────────────────────────────────────────────

  const rootStyle = getComputedStyle(document.documentElement);
  const cssVar = (name) => rootStyle.getPropertyValue(name).trim();

  function isDark() { return document.documentElement.getAttribute('data-theme') === 'dark'; }

  function setTheme(dark) {
    if (dark) document.documentElement.setAttribute('data-theme', 'dark');
    else document.documentElement.removeAttribute('data-theme');
    lsSet('theme', dark ? 'dark' : 'light');
    applyChartTheme();
  }

  function applyChartTheme() {
    const muted_  = cssVar('--muted');
    const grid    = cssVar('--grid');
    const border  = cssVar('--border');
    ALL_CHARTS.forEach((chart) => {
      chart.options.plugins.legend.labels.color = muted_;
      chart.options.scales.x.ticks.color = muted_;
      chart.options.scales.y.ticks.color = muted_;
      chart.options.scales.x.grid.color  = grid;
      chart.options.scales.y.grid.color  = grid;
      chart.options.scales.x.border.color = border;
      chart.options.scales.y.border.color = border;
      chart.update('none');
    });
  }

  // ── Chart base options ─────────────────────────────────────────────────────

  const CHART_OPTS = {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    // Render at 2x minimum so text stays crisp in screenshots/recordings
    // even on standard-DPI displays.
    devicePixelRatio: Math.max(window.devicePixelRatio || 1, 2),
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { labels: { color: '#6b7688', font: { size: 13, family: "'Inter', sans-serif", weight: 500 } } } },
    scales: {
      x: {
        ticks: { color: '#6b7688', maxTicksLimit: 8, font: { size: 12, family: "'JetBrains Mono', monospace" } },
        grid:  { color: 'rgba(31,41,55,0.06)' },
        border: { color: '#e3e8ef' },
      },
      y: {
        ticks: { color: '#6b7688', font: { size: 12, family: "'JetBrains Mono', monospace" } },
        grid:  { color: 'rgba(31,41,55,0.06)' },
        border: { color: '#e3e8ef' },
      },
    },
  };

  // ── Threshold line plugin ──────────────────────────────────────────────────

  const thresholdPlugin = {
    id: 'thresholdLine',
    afterDraw(chart) {
      const { ctx, chartArea: { left, right }, scales: { y } } = chart;

      // Manual threshold — amber dashed
      const yPx = y.getPixelForValue(threshold);
      const amber = cssVar('--amber') || '#b5760b';
      ctx.save();
      ctx.strokeStyle = amber;
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(left, yPx);
      ctx.lineTo(right, yPx);
      ctx.stroke();
      ctx.fillStyle = amber;
      ctx.font      = "600 12px 'JetBrains Mono', monospace";
      ctx.fillText(`threshold ${threshold.toFixed(4)}`, right - 160, yPx - 5);

      // Auto (EWMA) threshold — accent dashed, only when active
      if (autoThreshold && autoThreshVal != null) {
        const accent = cssVar('--accent') || '#4338ca';
        const yAuto = y.getPixelForValue(autoThreshVal);
        ctx.strokeStyle = accent;
        ctx.setLineDash([3, 5]);
        ctx.beginPath();
        ctx.moveTo(left, yAuto);
        ctx.lineTo(right, yAuto);
        ctx.stroke();
        ctx.fillStyle = accent;
        ctx.fillText(`auto ${autoThreshVal.toFixed(4)}`, right - 130, yAuto - 5);
      }

      ctx.restore();
    },
  };

  // ── Charts ─────────────────────────────────────────────────────────────────

  const errChart = new Chart(
    document.getElementById('err-chart').getContext('2d'),
    {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Recon Error (MSE)',
          data: errData,
          borderColor: '#2f7d5f',
          backgroundColor: 'rgba(47,125,95,0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
        }],
      },
      options: {
        ...CHART_OPTS,
        scales: { ...CHART_OPTS.scales, y: { ...CHART_OPTS.scales.y, min: 0 } },
      },
      plugins: [thresholdPlugin],
    }
  );

  const accelChart = new Chart(
    document.getElementById('accel-chart').getContext('2d'),
    {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'ax', data: axData, borderColor: '#c2373a', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'ay', data: ayData, borderColor: '#2f7d5f', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'az', data: azData, borderColor: '#2563a8', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        ],
      },
      options: CHART_OPTS,
    }
  );

  const gyroChart = new Chart(
    document.getElementById('gyro-chart').getContext('2d'),
    {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'gx', data: gxData, borderColor: '#6d3fc0', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'gy', data: gyData, borderColor: '#2563a8', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'gz', data: gzData, borderColor: '#b5760b', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
        ],
      },
      options: CHART_OPTS,
    }
  );

  // Severity level over time — filled step chart (0=normal, 1=warning, 2=critical)
  const SEV_COLORS = ['#2f7d5f', '#b5760b', '#c2373a'];
  const severityChart = new Chart(
    document.getElementById('severity-chart').getContext('2d'),
    {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Severity (0=normal 1=warning 2=critical)',
          data: sevData,
          borderColor: '#2563a8',
          backgroundColor: 'rgba(37,99,168,0.08)',
          borderWidth: 1.5,
          pointRadius: 3,
          pointBackgroundColor: sevData.map(v => SEV_COLORS[v] ?? SEV_COLORS[0]),
          stepped: true,
          fill: true,
          tension: 0,
        }],
      },
      options: {
        ...CHART_OPTS,
        scales: {
          ...CHART_OPTS.scales,
          y: {
            ...CHART_OPTS.scales.y,
            min: 0,
            max: 2,
            ticks: {
              ...CHART_OPTS.scales.y.ticks,
              stepSize: 1,
              callback: v => ['Normal', 'Warning', 'Critical'][v] ?? v,
            },
          },
        },
      },
    }
  );

  // Force a crisp re-render on viewport/fullscreen changes — Chart.js's own
  // ResizeObserver can lag one frame behind a fullscreen transition, leaving
  // the canvas bitmap stretched until the next data-driven redraw.
  const ALL_CHARTS = [errChart, accelChart, gyroChart, severityChart];
  let resizeRaf = null;
  function resizeAllCharts() {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => ALL_CHARTS.forEach(c => c.resize()));
  }
  window.addEventListener('resize', resizeAllCharts);
  document.addEventListener('fullscreenchange', resizeAllCharts);

  if (isDark()) applyChartTheme();

  // ── DOM refs ───────────────────────────────────────────────────────────────

  const statusBadge    = document.getElementById('status-badge');
  const statusText     = document.getElementById('status-text');
  const errValueEl     = document.getElementById('err-value');
  const errTrendEl     = document.getElementById('err-trend');
  const anomalyCountEl = document.getElementById('anomaly-count');
  const anomalyTrendEl = document.getElementById('anomaly-trend');
  const anomalyRateEl  = document.getElementById('anomaly-rate');
  const thresholdSubEl = document.getElementById('threshold-sub');
  const axEl           = document.getElementById('ax-val');
  const ayEl           = document.getElementById('ay-val');
  const azEl           = document.getElementById('az-val');
  const gxEl           = document.getElementById('gx-val');
  const gyEl           = document.getElementById('gy-val');
  const gzEl           = document.getElementById('gz-val');
  const sevNormalEl    = document.getElementById('sev-normal');
  const sevWarningEl   = document.getElementById('sev-warning');
  const sevCriticalEl  = document.getElementById('sev-critical');
  const burstBadgeEl   = document.getElementById('burst-badge');
  const faultBadgeEl   = document.getElementById('fault-badge');
  const connHzEl       = document.getElementById('conn-hz');
  const connUptimeEl   = document.getElementById('conn-uptime');

  const thresholdSlider   = document.getElementById('threshold-slider');
  const thresholdDisplay  = document.getElementById('threshold-display');
  const applyBtn          = document.getElementById('apply-threshold');
  const autoBtn           = document.getElementById('auto-threshold-btn');
  const autoLabel         = document.getElementById('auto-threshold-label');
  const autoValueEl       = document.getElementById('auto-threshold-value');
  const historySlider     = document.getElementById('history-slider');
  const historyDisplay    = document.getElementById('history-display');
  const themeToggleBtn    = document.getElementById('theme-toggle');
  const muteBtn           = document.getElementById('mute-btn');

  // ── Restore persisted control state ─────────────────────────────────────────

  thresholdSlider.value = threshold;
  thresholdDisplay.textContent = threshold.toFixed(4);
  thresholdSubEl.textContent   = threshold.toFixed(4);
  if (savedThreshold !== null) {
    fetch('/threshold', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold }),
    }).catch(() => {});
  }

  historySlider.value = maxHistory;
  historyDisplay.textContent = maxHistory;

  function setMuteUI() {
    muteBtn.dataset.active = muted ? 'true' : 'false';
    muteBtn.querySelector('.icon-bell').style.display     = muted ? 'none' : 'inline';
    muteBtn.querySelector('.icon-bell-off').style.display  = muted ? 'inline' : 'none';
    muteBtn.title = muted ? 'Unmute critical alerts' : 'Mute critical alerts';
  }
  setMuteUI();

  function setAutoUI() {
    autoBtn.textContent = `Auto: ${autoThreshold ? 'ON' : 'OFF'}`;
    autoBtn.style.borderColor = autoThreshold ? 'var(--accent)' : '';
    autoBtn.style.color       = autoThreshold ? 'var(--accent)' : '';
    autoBtn.style.background  = autoThreshold ? 'var(--accent-bg)' : '';
    autoLabel.style.display   = autoThreshold ? '' : 'none';
  }
  setAutoUI();
  if (autoThreshold) {
    fetch('/threshold/auto', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: true }),
    }).catch(() => {});
  }

  // ── Controls ───────────────────────────────────────────────────────────────

  thresholdSlider.addEventListener('input', () => {
    thresholdDisplay.textContent = parseFloat(thresholdSlider.value).toFixed(4);
  });

  applyBtn.addEventListener('click', () => {
    fetch('/threshold', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold: parseFloat(thresholdSlider.value) }),
    })
      .then(r => r.json())
      .then(d => {
        if (d.ok) {
          threshold = d.threshold;
          thresholdSubEl.textContent = threshold.toFixed(4);
          lsSet('threshold', threshold);
          errChart.update();
        }
      });
  });

  autoBtn.addEventListener('click', () => {
    autoThreshold = !autoThreshold;
    lsSet('autoThreshold', autoThreshold);
    setAutoUI();
    fetch('/threshold/auto', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: autoThreshold }),
    });
  });

  // Poll auto-threshold value while enabled so the chart line and label stay fresh
  function pollAutoThreshold() {
    if (!autoThreshold) return;
    fetch('/threshold/auto')
      .then(r => r.json())
      .then(d => {
        if (d.ewma != null) {
          autoThreshVal = d.threshold;
          threshold     = d.threshold;            // keep severity logic in sync
          autoValueEl.textContent = d.threshold.toFixed(4);
          thresholdSubEl.textContent = d.threshold.toFixed(4);
          errChart.update('none');
        }
      })
      .catch(() => {});
  }
  setInterval(pollAutoThreshold, 1000);

  historySlider.addEventListener('input', () => {
    maxHistory = parseInt(historySlider.value, 10);
    historyDisplay.textContent = maxHistory;
    lsSet('history', maxHistory);
  });

  themeToggleBtn.addEventListener('click', () => setTheme(!isDark()));

  muteBtn.addEventListener('click', () => {
    muted = !muted;
    lsSet('muted', muted);
    setMuteUI();
  });

  // ── Critical alert sound (WebAudio — no asset file needed) ─────────────────

  let audioCtx = null;
  function playAlertTone() {
    if (muted) return;
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const now = audioCtx.currentTime;
      [880, 660].forEach((freq, i) => {
        const osc  = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'square';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.0001, now + i * 0.16);
        gain.gain.exponentialRampToValueAtTime(0.12, now + i * 0.16 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.16 + 0.14);
        osc.connect(gain).connect(audioCtx.destination);
        osc.start(now + i * 0.16);
        osc.stop(now + i * 0.16 + 0.15);
      });
    } catch { /* WebAudio unavailable — fail silently */ }
  }

  // ── Session/model metadata panel ────────────────────────────────────────────

  const metaModelEl  = document.getElementById('meta-model');
  const metaWindowEl = document.getElementById('meta-window');
  const metaSrcEl    = document.getElementById('meta-threshold-source');
  const metaModeEl   = document.getElementById('meta-mode');

  const SOURCE_LABEL = {
    default:     'Default (no trained model)',
    model_file:  'Trained model file',
    manual:      'Manual override',
    auto:        'Auto (EWMA × 2.5)',
  };

  const metaClassifierEl = document.getElementById('meta-classifier');

  // ── Fault-type badge ────────────────────────────────────────────────────────
  // Class names are dynamic (come from /meta), so colors are assigned by a
  // fixed cyclic palette indexed by discovery order rather than hardcoded per name.
  const FAULT_PALETTE = [
    { color: '#c2373a', border: '#edc0c1', bg: '#fdf0f0' },
    { color: '#b5760b', border: '#ecd6a8', bg: '#fdf6e8' },
    { color: '#6d3fc0', border: '#ddd0f5', bg: '#f5f1fc' },
    { color: '#2563a8', border: '#c3d7ec', bg: '#eef5fc' },
    { color: '#2f7d5f', border: '#bfe0cf', bg: '#f0f9f4' },
  ];
  let faultClasses = [];

  function setFaultBadge(fault) {
    if (!fault || fault === 'none') {
      faultBadgeEl.style.display = 'none';
      return;
    }
    const idx = Math.max(0, faultClasses.indexOf(fault));
    const c = FAULT_PALETTE[idx % FAULT_PALETTE.length];
    // must be explicit: '' would fall back to the stylesheet's display:none
    faultBadgeEl.style.display = 'inline-block';
    faultBadgeEl.style.color = c.color;
    faultBadgeEl.style.borderColor = c.border;
    faultBadgeEl.style.background = c.bg;
    faultBadgeEl.textContent = fault.toUpperCase();
  }

  fetch('/meta')
    .then(r => r.json())
    .then(d => {
      metaModelEl.textContent  = d.model_available
        ? `${(d.model_type || 'vae').toUpperCase()} (latent ${d.latent_dim ?? '—'})`
        : 'No trained model';
      metaWindowEl.textContent = d.window != null ? `${d.window} samples` : '—';
      metaSrcEl.textContent    = SOURCE_LABEL[d.threshold_source] || d.threshold_source || '—';
      metaModeEl.textContent   = d.demo_mode ? 'Demo (synthetic)' : 'Live (serial)';
      faultClasses = d.fault_classes || [];
      metaClassifierEl.textContent = d.classifier_available
        ? faultClasses.join(', ')
        : 'Not trained';
    })
    .catch(() => { metaModelEl.textContent = 'Unavailable'; });

  // ── Connection stats: frame rate + uptime ───────────────────────────────────

  let connectedAt   = null;
  const frameTimes  = [];

  function updateUptime() {
    if (connectedAt == null) return;
    const s = Math.floor((Date.now() - connectedAt) / 1000);
    const hh = String(Math.floor(s / 3600)).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    connUptimeEl.textContent = `${hh}:${mm}:${ss}`;
  }
  setInterval(updateUptime, 1000);

  function recordFrameTime() {
    const now = Date.now();
    frameTimes.push(now);
    while (frameTimes.length > 20) frameTimes.shift();
    if (frameTimes.length >= 2) {
      const span = (frameTimes[frameTimes.length - 1] - frameTimes[0]) / 1000;
      const hz   = span > 0 ? (frameTimes.length - 1) / span : 0;
      connHzEl.textContent = `${hz.toFixed(1)} Hz`;
    }
  }

  // ── SSE connection ─────────────────────────────────────────────────────────

  function connect() {
    const es = new EventSource('/stream');

    es.onopen = () => {
      statusText.textContent = 'Connected';
      connectedAt = Date.now();
    };

    es.onmessage = (e) => {
      let obj;
      try { obj = JSON.parse(e.data); } catch { return; }
      if (obj.err == null) return;   // skip non-data frames (e.g. {"status":"ready"})
      recordFrameTime();
      handleFrame(obj);
    };

    es.onerror = () => {
      statusBadge.className = '';
      statusText.textContent = 'Reconnecting…';
      connectedAt = null;
      connHzEl.textContent = '—';
      es.close();
      setTimeout(connect, 2000);
    };
  }

  connect();

  // ── Anomaly event log ──────────────────────────────────────────────────────

  const logEmpty  = document.getElementById('log-empty');
  const eventTable = document.getElementById('event-table');
  const eventTbody = document.getElementById('event-tbody');
  const SEV_PILL   = ['normal', 'warning', 'critical'];
  const SEV_LABEL  = ['Normal', 'Warning', 'Critical'];

  function renderEvents(events) {
    if (!events.length) {
      logEmpty.style.display  = '';
      eventTable.style.display = 'none';
      return;
    }
    logEmpty.style.display  = 'none';
    eventTable.style.display = '';

    eventTbody.innerHTML = events.slice(0, 50).map((ev, i) => {
      const startS    = (ev.start_ts / 1000).toFixed(1);
      const endS      = (ev.end_ts   / 1000).toFixed(1);
      const durMs     = ev.end_ts - ev.start_ts;
      const pillClass = SEV_PILL[ev.peak_severity] ?? 'normal';
      const pillLabel = SEV_LABEL[ev.peak_severity] ?? 'Normal';
      const fault     = ev.dominant_fault && ev.dominant_fault !== 'none' ? ev.dominant_fault : '—';
      return `<tr>
        <td>${i + 1}</td>
        <td>${startS}s</td>
        <td>${endS}s</td>
        <td>${durMs}</td>
        <td>${ev.peak_err.toFixed(6)}</td>
        <td><span class="sev-pill ${pillClass}">${pillLabel}</span></td>
        <td>${fault}</td>
        <td>${ev.frame_count}</td>
      </tr>`;
    }).join('');
  }

  function pollEvents() {
    fetch('/events')
      .then(r => r.json())
      .then(d => renderEvents(d.events || []))
      .catch(() => {});
  }

  pollEvents();
  setInterval(pollEvents, 2000);

  // ── Trend arrows ─────────────────────────────────────────────────────────────
  // Compares the current value against the value ~TREND_WINDOW frames ago.

  const TREND_WINDOW = 20;
  const errHistory = [];
  const rateHistory = []; // rolling anomaly rate per TREND_WINDOW-sized block

  function renderTrend(el, delta, opts) {
    opts = opts || {};
    const eps = opts.eps ?? 1e-9;
    el.classList.add('visible');
    if (Math.abs(delta) < eps) {
      el.classList.remove('up', 'down'); el.classList.add('flat');
      el.textContent = '·';
      return;
    }
    const up = delta > 0;
    el.classList.toggle('up', up);
    el.classList.toggle('down', !up);
    el.classList.remove('flat');
    el.textContent = (up ? '▲ ' : '▼ ') + opts.format(Math.abs(delta));
  }

  // ── Frame handler ──────────────────────────────────────────────────────────

  const SEV_LABELS = ['Normal', 'Warning!', 'Critical!'];
  const SEV_CLASSES = ['normal', 'warning', 'critical'];
  let bootDone   = false;
  let prevSev    = 0;

  function handleFrame(obj) {
    totalFrames++;

    if (!bootDone) {
      bootDone = true;
      document.body.classList.remove('boot-loading');
    }

    // Derive severity: prefer field from firmware/demo; fall back to threshold comparison
    const sev = (obj.severity != null)
      ? obj.severity
      : (obj.err >= 2 * threshold ? 2 : obj.err >= threshold ? 1 : 0);

    if (sev > 0) anomalyCount++;
    sevCount[sev]++;

    if (sev === 2 && prevSev !== 2) {
      playAlertTone();
      if (window.Notification && Notification.permission === 'granted' && document.hidden) {
        try { new Notification('Edge AI — Critical anomaly detected', { body: `Reconstruction error ${obj.err.toFixed(6)}` }); } catch { /* ignore */ }
      }
    }
    prevSev = sev;

    statusBadge.className  = SEV_CLASSES[sev];
    statusText.textContent = SEV_LABELS[sev];

    const err = obj.err ?? 0;
    errValueEl.textContent = err.toFixed(6);
    errValueEl.classList.toggle('anomaly', sev > 0);
    anomalyCountEl.textContent = anomalyCount;
    const anomalyRate = (anomalyCount / totalFrames) * 100;
    anomalyRateEl.textContent  = `${anomalyRate.toFixed(1)}% of ${totalFrames} windows`;
    anomalyRateEl.classList.remove('skeleton');

    sevNormalEl.textContent   = sevCount[0];
    sevWarningEl.textContent  = sevCount[1];
    sevCriticalEl.textContent = sevCount[2];

    burstBadgeEl.style.display = obj.burst === 1 ? 'inline-block' : 'none';
    setFaultBadge(sev > 0 ? obj.fault : null);

    const ax = obj.ax ?? 0;
    const ay = obj.ay ?? 0;
    const az = obj.az ?? 0;
    const gx = obj.gx ?? 0;
    const gy = obj.gy ?? 0;
    const gz = obj.gz ?? 0;
    axEl.textContent = ax.toFixed(2);
    ayEl.textContent = ay.toFixed(2);
    azEl.textContent = az.toFixed(2);
    gxEl.textContent = gx.toFixed(4);
    gyEl.textContent = gy.toFixed(4);
    gzEl.textContent = gz.toFixed(4);

    // Trend arrows: reconstruction error vs. ~TREND_WINDOW frames ago
    errHistory.push(err);
    if (errHistory.length > TREND_WINDOW) errHistory.shift();
    if (errHistory.length === TREND_WINDOW) {
      // Rising error is worse, so the default up=red / down=green mapping is correct as-is.
      renderTrend(errTrendEl, err - errHistory[0], { format: v => v.toFixed(5) });
    }

    // Anomaly-rate trend: current block's anomaly rate vs. previous block
    rateHistory.push(sev > 0 ? 1 : 0);
    if (rateHistory.length > TREND_WINDOW * 2) rateHistory.shift();
    if (rateHistory.length === TREND_WINDOW * 2) {
      const prevRate = rateHistory.slice(0, TREND_WINDOW).reduce((a, b) => a + b, 0) / TREND_WINDOW;
      const curRate  = rateHistory.slice(TREND_WINDOW).reduce((a, b) => a + b, 0) / TREND_WINDOW;
      renderTrend(anomalyTrendEl, (curRate - prevRate) * 100, { format: v => v.toFixed(0) + 'pp', eps: 1 });
    }

    const label = obj.ts != null ? (obj.ts / 1000).toFixed(1) + 's' : String(totalFrames);
    labels.push(label);
    errData.push(err);
    axData.push(ax);
    ayData.push(ay);
    azData.push(az);
    gxData.push(gx);
    gyData.push(gy);
    gzData.push(gz);
    sevData.push(sev);

    // Keep point colors in sync with severity values
    severityChart.data.datasets[0].pointBackgroundColor = sevData.map(v => SEV_COLORS[v]);

    while (labels.length > maxHistory) {
      labels.shift(); errData.shift();
      axData.shift(); ayData.shift(); azData.shift();
      gxData.shift(); gyData.shift(); gzData.shift();
      sevData.shift();
    }

    errChart.update('none');
    accelChart.update('none');
    gyroChart.update('none');
    severityChart.update('none');
  }
})();
