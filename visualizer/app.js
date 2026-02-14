"use strict";

const state = {
  datasets: new Map(),
  current: null,
};

const elFileInput = document.getElementById("file-input");
const elPathInput = document.getElementById("path-input");
const elPathLoad = document.getElementById("path-load");
const elDatasetSelect = document.getElementById("dataset-select");
const elDatasetInfo = document.getElementById("dataset-info");
const elMetricsList = document.getElementById("metrics-list");
const elPlotButton = document.getElementById("plot-button");
const elClearButton = document.getElementById("clear-button");
const elXLog = document.getElementById("x-log");
const elYLog = document.getElementById("y-log");
const elPlotlyStatus = document.getElementById("plotly-status");

function setStatus(text, isError = false) {
  elPlotlyStatus.textContent = text;
  elPlotlyStatus.style.background = isError ? "#fbeaea" : "#eaf4f4";
  elPlotlyStatus.style.color = isError ? "#a13b3b" : "#1d4f53";
  elPlotlyStatus.style.borderColor = isError ? "#f0bcbc" : "#cde0e0";
}

function parseCSV(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length < 2) {
    throw new Error("CSV must contain a header and at least one data row.");
  }

  const headers = lines[0].split(",").map((h) => h.trim());
  const columns = headers.map(() => []);

  for (let i = 1; i < lines.length; i += 1) {
    const cells = lines[i].split(",").map((c) => c.trim());
    if (cells.length < headers.length) {
      continue;
    }
    for (let j = 0; j < headers.length; j += 1) {
      const value = Number(cells[j]);
      columns[j].push(Number.isFinite(value) ? value : null);
    }
  }

  return { headers, columns };
}

function addDataset(name, text) {
  const parsed = parseCSV(text);
  const dataset = {
    name,
    headers: parsed.headers,
    columns: parsed.columns,
  };
  state.datasets.set(name, dataset);
  if (!state.current) {
    state.current = name;
  }
  refreshDatasetSelect();
}

function refreshDatasetSelect() {
  elDatasetSelect.innerHTML = "";
  const keys = Array.from(state.datasets.keys());
  if (keys.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No dataset loaded";
    opt.disabled = true;
    opt.selected = true;
    elDatasetSelect.appendChild(opt);
    elDatasetInfo.textContent = "No data loaded";
    elMetricsList.innerHTML = "";
    return;
  }

  keys.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === state.current) {
      opt.selected = true;
    }
    elDatasetSelect.appendChild(opt);
  });

  const dataset = state.datasets.get(state.current);
  if (dataset) {
    elDatasetInfo.textContent = `${dataset.columns[0].length} rows | ${dataset.headers[0]} -> ${dataset.headers.slice(1).join(", ")}`;
    refreshMetrics(dataset);
  }
}

function refreshMetrics(dataset) {
  elMetricsList.innerHTML = "";
  dataset.headers.slice(1).forEach((name, idx) => {
    const label = document.createElement("label");
    label.className = "checkbox";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.metricIndex = String(idx + 1);
    if (name.toLowerCase() !== "sigma") {
      input.checked = true;
    }

    const span = document.createElement("span");
    span.textContent = name;

    label.appendChild(input);
    label.appendChild(span);
    elMetricsList.appendChild(label);
  });
}

function buildTraces(dataset) {
  const selected = Array.from(elMetricsList.querySelectorAll("input[type='checkbox']"))
    .filter((input) => input.checked)
    .map((input) => Number(input.dataset.metricIndex));

  if (selected.length === 0) {
    throw new Error("Select at least one metric.");
  }

  const x = dataset.columns[0];
  const traces = [];
  let hasSigma = false;

  selected.forEach((colIdx) => {
    const name = dataset.headers[colIdx];
    const series = dataset.columns[colIdx];

    if (name.toLowerCase() === "sigma") {
      hasSigma = true;
      traces.push({
        x,
        y: series,
        name: "sigma",
        yaxis: "y2",
        line: { color: "#7b4db8", width: 2 },
      });
      return;
    }

    traces.push({
      x,
      y: series,
      name,
      mode: "lines",
      line: { width: 2 },
    });
  });

  return { traces, hasSigma };
}

function plotCurrent() {
  const dataset = state.datasets.get(state.current);
  if (!dataset) {
    throw new Error("No dataset loaded.");
  }

  const { traces, hasSigma } = buildTraces(dataset);
  const xscale = elXLog.checked ? "log" : "linear";
  const yscale = elYLog.checked ? "log" : "linear";

  const layout = {
    margin: { t: 40, r: 40, l: 60, b: 50 },
    xaxis: { title: dataset.headers[0], type: xscale },
    yaxis: { title: "Metrics", type: yscale },
    legend: { orientation: "h" },
  };

  if (hasSigma) {
    layout.yaxis2 = {
      title: "Sigma",
      overlaying: "y",
      side: "right",
      range: [0, 1],
    };
  }

  Plotly.newPlot("plot", traces, layout, { responsive: true });
}

async function loadFromPath(path) {
  const resp = await fetch(path);
  if (!resp.ok) {
    throw new Error(`Failed to load ${path}: ${resp.status}`);
  }
  const text = await resp.text();
  addDataset(path, text);
}

elFileInput.addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  for (const file of files) {
    const text = await file.text();
    addDataset(file.name, text);
  }
});

elPathLoad.addEventListener("click", async () => {
  const path = elPathInput.value.trim();
  if (!path) {
    setStatus("Enter a path to load.", true);
    return;
  }
  try {
    await loadFromPath(path);
    setStatus(`Loaded ${path}`);
  } catch (err) {
    setStatus(err.message, true);
  }
});

elDatasetSelect.addEventListener("change", (event) => {
  state.current = event.target.value;
  const dataset = state.datasets.get(state.current);
  if (dataset) {
    refreshMetrics(dataset);
  }
});

elPlotButton.addEventListener("click", () => {
  try {
    plotCurrent();
    setStatus("Plot updated");
  } catch (err) {
    setStatus(err.message, true);
  }
});

elClearButton.addEventListener("click", () => {
  state.datasets.clear();
  state.current = null;
  refreshDatasetSelect();
  Plotly.purge("plot");
  setStatus("Cleared");
});

window.addEventListener("load", () => {
  if (typeof Plotly === "undefined") {
    setStatus("Plotly not loaded. Check network access to CDN.", true);
  } else {
    setStatus("Plotly ready");
  }
  refreshDatasetSelect();
});
