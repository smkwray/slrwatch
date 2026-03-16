/* ========================================================================
   SLR Watch — Main JavaScript
   Theme toggle, navigation, scroll reveals, and chart rendering.
   ======================================================================== */

/* ---- Inline Chart Data ---------------------------------------------- */
const DATA = {
  eventStudy: {
    coefficients: [
      {
        treatment: "Low Headroom",
        broad: { coef: 0.0247, p: 0.0021 },
        flagship: { coef: 0.0178, p: 0.0515 },
        clustered: { coef: 0.0178, p: 0.1762 }
      },
      {
        treatment: "Covered Bank",
        broad: { coef: 0.0197, p: 0.0054 },
        flagship: { coef: 0.0206, p: 0.0055 },
        clustered: { coef: 0.0206, p: 0.1626 }
      },
      {
        treatment: "High UST Share",
        broad: { coef: 0.0023, p: 0.8133 },
        flagship: { coef: 0.0021, p: 0.8348 },
        clustered: { coef: 0.0021, p: 0.9059 }
      }
    ]
  },
  reallocation: [
    { treatment: "Low Headroom", outcome: "Treasury Inventory", net: 0.0247 },
    { treatment: "Low Headroom", outcome: "Fed Balances", net: 0.0026 },
    { treatment: "Low Headroom", outcome: "Deposit Growth", net: -0.0352 },
    { treatment: "Low Headroom", outcome: "Loan Growth", net: -0.0165 },
    { treatment: "Covered Bank", outcome: "Treasury Inventory", net: 0.0197 },
    { treatment: "Covered Bank", outcome: "Fed Balances", net: -0.0045 },
    { treatment: "Covered Bank", outcome: "Deposit Growth", net: -0.0398 },
    { treatment: "Covered Bank", outcome: "Loan Growth", net: -0.0232 }
  ],
  safeAssets: [
    { treatment: "Low Headroom", treasury_share: 0.0911, fed_balance: 0.0026, treasury_inv: 0.0247 },
    { treatment: "Covered Bank", treasury_share: 0.0986, fed_balance: -0.0045, treasury_inv: 0.0197 }
  ],
  intermediation: [
    { treatment: "Low Headroom", trading_assets: -0.0171, treasury_inv: 0.0247 },
    { treatment: "Covered Bank", trading_assets: -0.0053, treasury_inv: 0.0197 }
  ],
  regimes: {
    labels: ["Pre-Exclusion", "Temporary Exclusion", "Post-Exclusion", "QT Era"],
    treasury_share: [0.0615, 0.0645, null, null],
    fed_balance_share: [0.0775, 0.1509, null, null],
    trading_share: [null, 0.0966, null, 0.0903]
  },
  constraintDecomposition: {
    regimes: [
      {
        regime: "Duration Loss Window (2022\u201323)",
        insured: { leverage: 0.163, duration_loss: 0.659, funding: 0.178 },
        parent:  { leverage: 0.205, duration_loss: 0.630, funding: 0.165 }
      },
      {
        regime: "Late QT Normalization (2024\u201325)",
        insured: { leverage: 0.342, duration_loss: 0.421, funding: 0.237 },
        parent:  { leverage: 0.354, duration_loss: 0.342, funding: 0.304 }
      }
    ]
  },
  parentTransmission: {
    co_movement: { ust: 75, trading: 60 },
    surcharge: {
      high: { headroom: 0.0087, ust: 0.0926, trading: 0.1401 },
      low: { headroom: 0.0190, ust: 0.0936, trading: 0.1293 }
    }
  }
};


/* ---- Theme Management ----------------------------------------------- */
(function initTheme() {
  const saved = localStorage.getItem('slrwatch-theme');
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();

function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('slrwatch-theme', next);
  updateThemeIcon();
  renderAllCharts();
}

function updateThemeIcon() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
  btn.innerHTML = isDark
    ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>';
}


/* ---- Mobile Nav ----------------------------------------------------- */
function toggleMobileNav() {
  const links = document.querySelector('.nav-links');
  links.classList.toggle('open');
}


/* ---- Scroll Reveal -------------------------------------------------- */
function initReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}


/* ---- Active Nav Highlight ------------------------------------------- */
function initActiveNav() {
  const sections = document.querySelectorAll('.section[id]');
  const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => a.classList.remove('active'));
        const link = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => observer.observe(s));

  // close mobile nav on link click
  navLinks.forEach(a => a.addEventListener('click', () => {
    document.querySelector('.nav-links').classList.remove('open');
  }));
}


/* ---- SVG Helpers ---------------------------------------------------- */
function getCSS(prop) {
  return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function clearChart(id) {
  const c = document.getElementById(id);
  if (c) c.innerHTML = '';
  return c;
}


/* ---- Chart: Coefficient Dot Plot ------------------------------------ */
function renderDotPlot() {
  const container = clearChart('chart-dot-plot');
  if (!container) return;

  const data = DATA.eventStudy.coefficients;
  const W = 700, H = 220;
  const margin = { top: 30, right: 30, bottom: 40, left: 160 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  // scales
  const xMin = -0.005, xMax = 0.035;
  const xScale = v => margin.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const rowH = plotH / data.length;
  const yPos = i => margin.top + i * rowH + rowH / 2;

  // grid + axis
  const gridVals = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03];
  gridVals.forEach(v => {
    const x = xScale(v);
    svg.appendChild(svgEl('line', { x1: x, y1: margin.top, x2: x, y2: H - margin.bottom, stroke: getCSS('--chart-grid'), 'stroke-width': 1 }));
    const label = svgEl('text', { x: x, y: H - margin.bottom + 18, 'text-anchor': 'middle', fill: getCSS('--chart-label'), 'font-size': '11', 'font-family': "'JetBrains Mono', monospace" });
    label.textContent = v === 0 ? '0' : v.toFixed(3);
    svg.appendChild(label);
  });

  // zero line
  svg.appendChild(svgEl('line', { x1: xScale(0), y1: margin.top, x2: xScale(0), y2: H - margin.bottom, stroke: getCSS('--chart-axis'), 'stroke-width': 1.5 }));

  // x-axis title
  const xTitle = svgEl('text', { x: margin.left + plotW / 2, y: H - 2, 'text-anchor': 'middle', fill: getCSS('--chart-label'), 'font-size': '11', 'font-family': "'Inter', sans-serif" });
  xTitle.textContent = 'DiD Coefficient (Treasury Inventory / Assets)';
  svg.appendChild(xTitle);

  const sampleColors = [getCSS('--chart-1'), getCSS('--chart-2'), getCSS('--chart-3')];
  const samples = ['broad', 'flagship', 'clustered'];
  const offsets = [-8, 0, 8];
  const dotR = 6;

  data.forEach((row, i) => {
    const y = yPos(i);

    // row label
    const label = svgEl('text', { x: margin.left - 12, y: y + 4, 'text-anchor': 'end', fill: getCSS('--text-primary'), 'font-size': '13', 'font-family': "'Inter', sans-serif", 'font-weight': '500' });
    label.textContent = row.treatment;
    svg.appendChild(label);

    // horizontal guide
    svg.appendChild(svgEl('line', { x1: margin.left, y1: y, x2: W - margin.right, y2: y, stroke: getCSS('--chart-grid'), 'stroke-width': 0.5, 'stroke-dasharray': '3,3' }));

    // dots for each sample
    samples.forEach((s, si) => {
      const d = row[s] || row[samples[si]];
      if (!d) return;
      const cx = xScale(d.coef);
      const cy = y + offsets[si];
      const sig = d.p < 0.05;

      const circle = svgEl('circle', {
        cx: cx, cy: cy, r: dotR,
        fill: sig ? sampleColors[si] : 'transparent',
        stroke: sampleColors[si],
        'stroke-width': sig ? 0 : 2.5
      });
      svg.appendChild(circle);

      // p-value label
      if (si === 0 || Math.abs(d.coef - row[samples[0]].coef) > 0.003) {
        const pLabel = svgEl('text', {
          x: cx, y: cy - dotR - 4,
          'text-anchor': 'middle', fill: sampleColors[si],
          'font-size': '9', 'font-family': "'JetBrains Mono', monospace", opacity: '0.8'
        });
        pLabel.textContent = `p=${d.p.toFixed(3)}`;
        svg.appendChild(pLabel);
      }
    });
  });

  container.appendChild(svg);
}


/* ---- Chart: Reallocation Heatmap ------------------------------------ */
// (rendered as HTML table, see index.html)


/* ---- Chart: Safe-Asset Composition ---------------------------------- */
function renderSafeAssetChart() {
  const container = clearChart('chart-safe-assets');
  if (!container) return;

  const data = DATA.safeAssets;
  const W = 600, H = 240;
  const margin = { top: 20, right: 30, bottom: 50, left: 140 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  const maxVal = 0.12;
  const minVal = -0.02;
  const xScale = v => margin.left + ((v - minVal) / (maxVal - minVal)) * plotW;

  const metrics = [
    { key: 'treasury_share', label: 'Treasury Share of Safe Assets', color: getCSS('--chart-1') },
    { key: 'treasury_inv', label: 'Treasury Inventory / Assets', color: getCSS('--chart-2') },
    { key: 'fed_balance', label: 'Fed Balances / Assets', color: getCSS('--chart-4') }
  ];

  const groupH = plotH / data.length;
  const barH = 14;

  // zero line
  svg.appendChild(svgEl('line', {
    x1: xScale(0), y1: margin.top, x2: xScale(0), y2: H - margin.bottom,
    stroke: getCSS('--chart-axis'), 'stroke-width': 1.5
  }));

  // grid
  [-0.01, 0, 0.02, 0.04, 0.06, 0.08, 0.10].forEach(v => {
    const x = xScale(v);
    svg.appendChild(svgEl('line', {
      x1: x, y1: margin.top, x2: x, y2: H - margin.bottom,
      stroke: getCSS('--chart-grid'), 'stroke-width': 0.5
    }));
    const t = svgEl('text', {
      x: x, y: H - margin.bottom + 16, 'text-anchor': 'middle',
      fill: getCSS('--chart-label'), 'font-size': '10', 'font-family': "'JetBrains Mono', monospace"
    });
    t.textContent = (v * 100).toFixed(0) + 'pp';
    svg.appendChild(t);
  });

  data.forEach((row, gi) => {
    const gY = margin.top + gi * groupH;

    // treatment label
    const lbl = svgEl('text', {
      x: margin.left - 10, y: gY + groupH / 2 + 4, 'text-anchor': 'end',
      fill: getCSS('--text-primary'), 'font-size': '12', 'font-weight': '500', 'font-family': "'Inter', sans-serif"
    });
    lbl.textContent = row.treatment;
    svg.appendChild(lbl);

    metrics.forEach((m, mi) => {
      const val = row[m.key];
      const y = gY + 12 + mi * (barH + 5);
      const x0 = xScale(0);
      const x1 = xScale(val);
      const barX = Math.min(x0, x1);
      const barW = Math.abs(x1 - x0);

      svg.appendChild(svgEl('rect', {
        x: barX, y: y, width: barW, height: barH, rx: 3,
        fill: m.color, opacity: '0.85'
      }));

      // value label
      const vLbl = svgEl('text', {
        x: x1 + (val >= 0 ? 6 : -6), y: y + barH / 2 + 4,
        'text-anchor': val >= 0 ? 'start' : 'end',
        fill: m.color, 'font-size': '10', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
      });
      vLbl.textContent = (val >= 0 ? '+' : '') + (val * 100).toFixed(1) + 'pp';
      svg.appendChild(vLbl);
    });
  });

  // x-axis label
  const xLbl = svgEl('text', {
    x: margin.left + plotW / 2, y: H - 4, 'text-anchor': 'middle',
    fill: getCSS('--chart-label'), 'font-size': '11', 'font-family': "'Inter', sans-serif"
  });
  xLbl.textContent = 'Net Treated \u2212 Control Change (percentage points)';
  svg.appendChild(xLbl);

  container.appendChild(svg);
}


/* ---- Chart: Intermediation Tradeoff (Diverging Bar) ----------------- */
function renderIntermediationChart() {
  const container = clearChart('chart-intermediation');
  if (!container) return;

  const data = DATA.intermediation;
  const W = 600, H = 200;
  const margin = { top: 20, right: 50, bottom: 50, left: 140 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  const maxAbs = 0.03;
  const xScale = v => margin.left + ((v + maxAbs) / (2 * maxAbs)) * plotW;
  const rowH = plotH / data.length;
  const barH = 20;

  // zero line
  svg.appendChild(svgEl('line', {
    x1: xScale(0), y1: margin.top - 5, x2: xScale(0), y2: H - margin.bottom,
    stroke: getCSS('--chart-axis'), 'stroke-width': 1.5
  }));

  // grid
  [-0.02, -0.01, 0, 0.01, 0.02].forEach(v => {
    const x = xScale(v);
    svg.appendChild(svgEl('line', {
      x1: x, y1: margin.top, x2: x, y2: H - margin.bottom,
      stroke: getCSS('--chart-grid'), 'stroke-width': 0.5
    }));
    const t = svgEl('text', {
      x: x, y: H - margin.bottom + 16, 'text-anchor': 'middle',
      fill: getCSS('--chart-label'), 'font-size': '10', 'font-family': "'JetBrains Mono', monospace"
    });
    t.textContent = (v * 100).toFixed(0) + 'pp';
    svg.appendChild(t);
  });

  const negColor = getCSS('--chart-3');
  const posColor = getCSS('--chart-1');

  data.forEach((row, i) => {
    const gY = margin.top + i * rowH;
    const midY = gY + rowH / 2;

    // label
    const lbl = svgEl('text', {
      x: margin.left - 10, y: midY + 4, 'text-anchor': 'end',
      fill: getCSS('--text-primary'), 'font-size': '12', 'font-weight': '500', 'font-family': "'Inter', sans-serif"
    });
    lbl.textContent = row.treatment;
    svg.appendChild(lbl);

    // trading assets bar (negative)
    const ta = row.trading_assets;
    const taX = Math.min(xScale(0), xScale(ta));
    const taW = Math.abs(xScale(ta) - xScale(0));
    svg.appendChild(svgEl('rect', {
      x: taX, y: midY - barH / 2 - 1, width: taW, height: barH / 2, rx: 2,
      fill: negColor, opacity: '0.8'
    }));

    // treasury bar (positive)
    const ti = row.treasury_inv;
    const tiX = xScale(0);
    const tiW = xScale(ti) - tiX;
    svg.appendChild(svgEl('rect', {
      x: tiX, y: midY + 1, width: tiW, height: barH / 2, rx: 2,
      fill: posColor, opacity: '0.8'
    }));

    // value labels
    const taLbl = svgEl('text', {
      x: xScale(ta) - 6, y: midY - barH / 2 + 5,
      'text-anchor': 'end', fill: negColor, 'font-size': '9', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
    });
    taLbl.textContent = (ta * 100).toFixed(1) + 'pp';
    svg.appendChild(taLbl);

    const tiLbl = svgEl('text', {
      x: xScale(ti) + 6, y: midY + barH / 2 + 3,
      'text-anchor': 'start', fill: posColor, 'font-size': '9', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
    });
    tiLbl.textContent = '+' + (ti * 100).toFixed(1) + 'pp';
    svg.appendChild(tiLbl);
  });

  // x-axis label
  const xLbl = svgEl('text', {
    x: margin.left + plotW / 2, y: H - 4, 'text-anchor': 'middle',
    fill: getCSS('--chart-label'), 'font-size': '11', 'font-family': "'Inter', sans-serif"
  });
  xLbl.textContent = 'Net Treated \u2212 Control Change (percentage points)';
  svg.appendChild(xLbl);

  container.appendChild(svg);
}


/* ---- Chart: Regime Comparison (Dumbbell) ----------------------------- */
function renderRegimeChart() {
  const container = clearChart('chart-regimes');
  if (!container) return;

  const W = 600, H = 200;
  const margin = { top: 20, right: 40, bottom: 50, left: 200 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  const rows = [
    { label: 'Treasury Share (Insured)', v1: 0.0615, v2: 0.0645, era1: 'Pre', era2: 'Exclusion' },
    { label: 'Fed-Balance Share (Insured)', v1: 0.0775, v2: 0.1509, era1: 'Pre', era2: 'Exclusion' },
    { label: 'Trading-Assets Share (Parent)', v1: 0.0966, v2: 0.0903, era1: 'Exclusion', era2: 'QT Era' }
  ];

  const maxVal = 0.18;
  const xScale = v => margin.left + (v / maxVal) * plotW;
  const rowH = plotH / rows.length;

  // grid
  [0, 0.04, 0.08, 0.12, 0.16].forEach(v => {
    const x = xScale(v);
    svg.appendChild(svgEl('line', {
      x1: x, y1: margin.top, x2: x, y2: H - margin.bottom,
      stroke: getCSS('--chart-grid'), 'stroke-width': 0.5
    }));
    const t = svgEl('text', {
      x: x, y: H - margin.bottom + 16, 'text-anchor': 'middle',
      fill: getCSS('--chart-label'), 'font-size': '10', 'font-family': "'JetBrains Mono', monospace"
    });
    t.textContent = (v * 100).toFixed(0) + '%';
    svg.appendChild(t);
  });

  const color1 = getCSS('--chart-2');
  const color2 = getCSS('--chart-1');

  rows.forEach((row, i) => {
    const y = margin.top + i * rowH + rowH / 2;

    // label
    const lbl = svgEl('text', {
      x: margin.left - 10, y: y + 4, 'text-anchor': 'end',
      fill: getCSS('--text-primary'), 'font-size': '11', 'font-weight': '500', 'font-family': "'Inter', sans-serif"
    });
    lbl.textContent = row.label;
    svg.appendChild(lbl);

    // connecting line
    svg.appendChild(svgEl('line', {
      x1: xScale(row.v1), y1: y, x2: xScale(row.v2), y2: y,
      stroke: getCSS('--chart-axis'), 'stroke-width': 2
    }));

    // dot 1 (earlier era)
    svg.appendChild(svgEl('circle', {
      cx: xScale(row.v1), cy: y, r: 6,
      fill: color1, stroke: color1, 'stroke-width': 0
    }));

    // dot 2 (later era)
    svg.appendChild(svgEl('circle', {
      cx: xScale(row.v2), cy: y, r: 6,
      fill: color2, stroke: color2, 'stroke-width': 0
    }));

    // era labels on dots
    const l1 = svgEl('text', {
      x: xScale(row.v1), y: y - 12, 'text-anchor': 'middle',
      fill: color1, 'font-size': '9', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
    });
    l1.textContent = (row.v1 * 100).toFixed(1) + '%';
    svg.appendChild(l1);

    const l2 = svgEl('text', {
      x: xScale(row.v2), y: y - 12, 'text-anchor': 'middle',
      fill: color2, 'font-size': '9', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
    });
    l2.textContent = (row.v2 * 100).toFixed(1) + '%';
    svg.appendChild(l2);
  });

  // x-axis label
  const xLbl = svgEl('text', {
    x: margin.left + plotW / 2, y: H - 4, 'text-anchor': 'middle',
    fill: getCSS('--chart-label'), 'font-size': '11', 'font-family': "'Inter', sans-serif"
  });
  xLbl.textContent = 'Share of Total Assets';
  svg.appendChild(xLbl);

  container.appendChild(svg);
}


/* ---- Chart: Parent Transmission Co-Movement ------------------------- */
function renderTransmissionChart() {
  const container = clearChart('chart-transmission');
  if (!container) return;

  const W = 500, H = 200;
  const margin = { top: 20, right: 30, bottom: 50, left: 180 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  const bars = [
    { label: 'UST Holdings Co-Movement', value: 75 },
    { label: 'Trading Assets Co-Movement', value: 60 }
  ];

  const xScale = v => margin.left + (v / 100) * plotW;
  const rowH = plotH / bars.length;
  const barH = 24;

  // grid
  [0, 25, 50, 75, 100].forEach(v => {
    const x = xScale(v);
    svg.appendChild(svgEl('line', {
      x1: x, y1: margin.top, x2: x, y2: H - margin.bottom,
      stroke: getCSS('--chart-grid'), 'stroke-width': 0.5
    }));
    const t = svgEl('text', {
      x: x, y: H - margin.bottom + 16, 'text-anchor': 'middle',
      fill: getCSS('--chart-label'), 'font-size': '10', 'font-family': "'JetBrains Mono', monospace"
    });
    t.textContent = v + '%';
    svg.appendChild(t);
  });

  // 50% reference line
  svg.appendChild(svgEl('line', {
    x1: xScale(50), y1: margin.top, x2: xScale(50), y2: H - margin.bottom,
    stroke: getCSS('--chart-axis'), 'stroke-width': 1, 'stroke-dasharray': '4,3'
  }));

  bars.forEach((b, i) => {
    const y = margin.top + i * rowH + (rowH - barH) / 2;

    // label
    const lbl = svgEl('text', {
      x: margin.left - 10, y: y + barH / 2 + 4, 'text-anchor': 'end',
      fill: getCSS('--text-primary'), 'font-size': '12', 'font-weight': '500', 'font-family': "'Inter', sans-serif"
    });
    lbl.textContent = b.label;
    svg.appendChild(lbl);

    // bar
    const color = b.value >= 66 ? getCSS('--chart-4') : getCSS('--chart-2');
    svg.appendChild(svgEl('rect', {
      x: margin.left, y: y, width: xScale(b.value) - margin.left, height: barH, rx: 4,
      fill: color, opacity: '0.8'
    }));

    // value
    const vLbl = svgEl('text', {
      x: xScale(b.value) + 8, y: y + barH / 2 + 4,
      'text-anchor': 'start', fill: color, 'font-size': '12', 'font-weight': '700', 'font-family': "'JetBrains Mono', monospace"
    });
    vLbl.textContent = b.value + '%';
    svg.appendChild(vLbl);
  });

  // x label
  const xLbl = svgEl('text', {
    x: margin.left + plotW / 2, y: H - 4, 'text-anchor': 'middle',
    fill: getCSS('--chart-label'), 'font-size': '11', 'font-family': "'Inter', sans-serif"
  });
  xLbl.textContent = 'Quarter-over-Quarter Directional Agreement (%)';
  svg.appendChild(xLbl);

  container.appendChild(svg);
}


/* ---- Chart: Surcharge Comparison ------------------------------------ */
function renderSurchargeChart() {
  const container = clearChart('chart-surcharge');
  if (!container) return;

  const W = 550, H = 200;
  const margin = { top: 20, right: 30, bottom: 50, left: 130 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  const d = DATA.parentTransmission.surcharge;
  const metrics = [
    { label: 'Bank SLR Headroom (pp)', high: d.high.headroom, low: d.low.headroom, scale: 0.025 },
    { label: 'Bank UST / Assets', high: d.high.ust, low: d.low.ust, scale: 0.12 },
    { label: 'Parent Trading / Assets', high: d.high.trading, low: d.low.trading, scale: 0.18 }
  ];

  const rowH = plotH / metrics.length;
  const color1 = getCSS('--chart-3');
  const color2 = getCSS('--chart-1');

  metrics.forEach((m, i) => {
    const y = margin.top + i * rowH + rowH / 2;
    const xScale = v => margin.left + (v / m.scale) * plotW;

    // label
    const lbl = svgEl('text', {
      x: margin.left - 10, y: y + 4, 'text-anchor': 'end',
      fill: getCSS('--text-primary'), 'font-size': '11', 'font-weight': '500', 'font-family': "'Inter', sans-serif"
    });
    lbl.textContent = m.label;
    svg.appendChild(lbl);

    // connecting line
    svg.appendChild(svgEl('line', {
      x1: xScale(m.high), y1: y, x2: xScale(m.low), y2: y,
      stroke: getCSS('--chart-axis'), 'stroke-width': 2
    }));

    // high surcharge dot
    svg.appendChild(svgEl('circle', { cx: xScale(m.high), cy: y, r: 6, fill: color1 }));
    const hl = svgEl('text', {
      x: xScale(m.high), y: y - 12, 'text-anchor': 'middle',
      fill: color1, 'font-size': '9', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
    });
    hl.textContent = (m.high * 100).toFixed(1) + (m.label.includes('pp') ? 'pp' : '%');
    svg.appendChild(hl);

    // low surcharge dot
    svg.appendChild(svgEl('circle', { cx: xScale(m.low), cy: y, r: 6, fill: color2 }));
    const ll = svgEl('text', {
      x: xScale(m.low), y: y - 12, 'text-anchor': 'middle',
      fill: color2, 'font-size': '9', 'font-weight': '600', 'font-family': "'JetBrains Mono', monospace"
    });
    ll.textContent = (m.low * 100).toFixed(1) + (m.label.includes('pp') ? 'pp' : '%');
    svg.appendChild(ll);
  });

  container.appendChild(svg);
}


/* ---- Chart: Constraint Decomposition (Stacked Bar) ------------------ */
function renderConstraintChart() {
  const container = clearChart('chart-constraints');
  if (!container) return;

  const data = DATA.constraintDecomposition.regimes;
  const rows = [];
  data.forEach(d => {
    rows.push({ label: 'Insured Banks', regime: d.regime, ...d.insured });
    rows.push({ label: 'Parents / IHCs', regime: d.regime, ...d.parent });
  });

  const W = 700, H = 280;
  const margin = { top: 30, right: 30, bottom: 40, left: 170 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart-svg' });

  const groupGap = 18;
  const barH = 26;
  const pairH = barH * 2 + 6;
  const totalH = pairH * data.length + groupGap * (data.length - 1);
  const startY = margin.top + (plotH - totalH) / 2;

  const xScale = v => margin.left + v * plotW;

  // grid
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const x = xScale(v);
    svg.appendChild(svgEl('line', {
      x1: x, y1: margin.top, x2: x, y2: H - margin.bottom,
      stroke: getCSS('--chart-grid'), 'stroke-width': 0.5
    }));
    const t = svgEl('text', {
      x: x, y: H - margin.bottom + 18, 'text-anchor': 'middle',
      fill: getCSS('--chart-label'), 'font-size': '10',
      'font-family': "'JetBrains Mono', monospace"
    });
    t.textContent = (v * 100).toFixed(0) + '%';
    svg.appendChild(t);
  });

  const colors = {
    leverage: getCSS('--chart-1'),
    duration_loss: getCSS('--chart-3'),
    funding: getCSS('--chart-2')
  };

  rows.forEach((row, i) => {
    const groupIdx = Math.floor(i / 2);
    const withinIdx = i % 2;
    const groupY = startY + groupIdx * (pairH + groupGap);
    const y = groupY + withinIdx * (barH + 6);

    // regime heading above first bar in each group
    if (withinIdx === 0) {
      const heading = svgEl('text', {
        x: margin.left - 10, y: groupY - 8, 'text-anchor': 'end',
        fill: getCSS('--text-tertiary'), 'font-size': '10', 'font-weight': '600',
        'font-family': "'Inter', sans-serif", 'text-transform': 'uppercase',
        'letter-spacing': '0.04em'
      });
      heading.textContent = row.regime;
      svg.appendChild(heading);
    }

    // row label
    const lbl = svgEl('text', {
      x: margin.left - 10, y: y + barH / 2 + 4, 'text-anchor': 'end',
      fill: getCSS('--text-primary'), 'font-size': '12', 'font-weight': '500',
      'font-family': "'Inter', sans-serif"
    });
    lbl.textContent = row.label;
    svg.appendChild(lbl);

    // stacked segments
    let xOff = 0;
    ['leverage', 'duration_loss', 'funding'].forEach(key => {
      const val = row[key];
      const segW = val * plotW;

      // bar segment
      const isFirst = key === 'leverage';
      const isLast = key === 'funding';
      const rect = svgEl('rect', {
        x: xScale(xOff), y: y, width: segW, height: barH,
        fill: colors[key], opacity: '0.85',
        rx: isFirst && isLast ? 4 : isFirst ? '4' : isLast ? '4' : '0'
      });
      svg.appendChild(rect);

      // percentage label inside segment (if wide enough)
      if (segW > 38) {
        const pctLbl = svgEl('text', {
          x: xScale(xOff) + segW / 2, y: y + barH / 2 + 4,
          'text-anchor': 'middle', fill: '#FFFFFF',
          'font-size': '11', 'font-weight': '700',
          'font-family': "'JetBrains Mono', monospace"
        });
        pctLbl.textContent = (val * 100).toFixed(1) + '%';
        svg.appendChild(pctLbl);
      }

      xOff += val;
    });
  });

  container.appendChild(svg);
}


/* ---- Render All Charts ---------------------------------------------- */
function renderAllCharts() {
  renderDotPlot();
  renderSafeAssetChart();
  renderIntermediationChart();
  renderRegimeChart();
  renderTransmissionChart();
  renderSurchargeChart();
  renderConstraintChart();
}


/* ---- Init ----------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  updateThemeIcon();
  initReveal();
  initActiveNav();
  renderAllCharts();

  document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
  const menuBtn = document.querySelector('.nav-menu-btn');
  if (menuBtn) menuBtn.addEventListener('click', toggleMobileNav);
});
