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
const elXSelect = document.getElementById("x-select");
const elYSelect = document.getElementById("y-select");
const elMetricsList = document.getElementById("metrics-list");
const elPlotType = document.getElementById("plot-type");
const elPlotButton = document.getElementById("plot-button");
const elClearButton = document.getElementById("clear-button");
const elXLog = document.getElementById("x-log");
const elYLog = document.getElementById("y-log");
const elZLog = document.getElementById("z-log");
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
    elDatasetInfo.textContent = `${dataset.columns[0].length} rows | ${dataset.headers.join(", ")}`;
    refreshAxisSelectors(dataset);
    refreshMetrics(dataset);
  }
}

function refreshAxisSelectors(dataset) {
  [elXSelect, elYSelect].forEach((el) => {
    el.innerHTML = "";
    dataset.headers.forEach((name, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = name;
      el.appendChild(opt);
    });
  });
  if (dataset.headers.length >= 3) {
    elXSelect.value = "0";
    elYSelect.value = "1";
  }
}

function refreshMetrics(dataset) {
  const xIdx = Number(elXSelect.value || 0);
  const yIdx = Number(elYSelect.value || 1);
  elMetricsList.innerHTML = "";

  dataset.headers.forEach((name, idx) => {
    if (idx === xIdx || idx === yIdx) {
      return;
    }
    const label = document.createElement("label");
    label.className = "checkbox";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.metricIndex = String(idx);
    input.checked = true;

    const span = document.createElement("span");
    span.textContent = name;

    label.appendChild(input);
    label.appendChild(span);
    elMetricsList.appendChild(label);
  });
}

function extractXYZ(dataset, xIdx, yIdx, zIdx) {
  const x = [];
  const y = [];
  const z = [];

  for (let i = 0; i < dataset.columns[0].length; i += 1) {
    const xv = dataset.columns[xIdx][i];
    const yv = dataset.columns[yIdx][i];
    const zv = dataset.columns[zIdx][i];
    if (xv === null || yv === null || zv === null) {
      continue;
    }
    x.push(xv);
    y.push(yv);
    z.push(zv);
  }

  return { x, y, z };
}

function tryBuildGrid(x, y, z) {
  const xUnique = Array.from(new Set(x)).sort((a, b) => a - b);
  const yUnique = Array.from(new Set(y)).sort((a, b) => a - b);

  const xIndex = new Map(xUnique.map((val, idx) => [val, idx]));
  const yIndex = new Map(yUnique.map((val, idx) => [val, idx]));

  const grid = Array.from({ length: yUnique.length }, () =>
    Array.from({ length: xUnique.length }, () => null)
  );
  const counts = Array.from({ length: yUnique.length }, () =>
    Array.from({ length: xUnique.length }, () => 0)
  );

  for (let i = 0; i < x.length; i += 1) {
    const xi = xIndex.get(x[i]);
    const yi = yIndex.get(y[i]);
    if (xi === undefined || yi === undefined) {
      continue;
    }
    if (grid[yi][xi] === null) {
      grid[yi][xi] = z[i];
      counts[yi][xi] = 1;
    } else {
      grid[yi][xi] += z[i];
      counts[yi][xi] += 1;
    }
  }

  let missing = 0;
  for (let yi = 0; yi < yUnique.length; yi += 1) {
    for (let xi = 0; xi < xUnique.length; xi += 1) {
      if (counts[yi][xi] === 0) {
        missing += 1;
        grid[yi][xi] = null;
      } else if (counts[yi][xi] > 1) {
        grid[yi][xi] /= counts[yi][xi];
      }
    }
  }

  return {
    xUnique,
    yUnique,
    grid,
    complete: missing === 0,
    missing,
  };
}

function plotCurrent() {
  const dataset = state.datasets.get(state.current);
  if (!dataset) {
    throw new Error("No dataset loaded.");
  }

  const xIdx = Number(elXSelect.value);
  const yIdx = Number(elYSelect.value);
  if (xIdx === yIdx) {
    throw new Error("X and Y must be different columns.");
  }

  const xscale = elXLog.checked ? "log" : "linear";
  const yscale = elYLog.checked ? "log" : "linear";
  const zscale = elZLog.checked ? "log" : "linear";

  const selectedMetrics = Array.from(
    elMetricsList.querySelectorAll("input[type='checkbox']")
  )
    .filter((input) => input.checked)
    .map((input) => Number(input.dataset.metricIndex));

  if (selectedMetrics.length === 0) {
    throw new Error("Select at least one Z metric.");
  }

  const mode = elPlotType.value;
  const traces = [];
  const scenes = {};
  const annotations = [];

  const cols = selectedMetrics.length > 1 ? Math.min(2, selectedMetrics.length) : 1;
  const rows = Math.ceil(selectedMetrics.length / cols);

  selectedMetrics.forEach((zIdx, i) => {
    const { x, y, z } = extractXYZ(dataset, xIdx, yIdx, zIdx);
    if (x.length === 0) {
      return;
    }

    const sceneId = i === 0 ? "scene" : `scene${i + 1}`;
    const row = Math.floor(i / cols);
    const col = i % cols;
    const domainX = [col / cols, (col + 1) / cols];
    const domainY = [1 - (row + 1) / rows, 1 - row / rows];

    scenes[sceneId] = {
      domain: { x: domainX, y: domainY },
      xaxis: { title: dataset.headers[xIdx], type: xscale },
      yaxis: { title: dataset.headers[yIdx], type: yscale },
      zaxis: { title: dataset.headers[zIdx], type: zscale },
    };

    annotations.push({
      text: dataset.headers[zIdx],
      x: domainX[0] + 0.02,
      y: domainY[1] - 0.02,
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 12, color: "#1f1e1c" },
    });

    if (mode === "scatter") {
      traces.push({
        type: "scatter3d",
        mode: "markers",
        x,
        y,
        z,
        marker: { size: 3, color: z, colorscale: "Viridis", opacity: 0.85 },
        scene: sceneId,
        name: dataset.headers[zIdx],
      });
      return;
    }

    if (mode === "mesh") {
      traces.push({
        type: "mesh3d",
        x,
        y,
        z,
        intensity: z,
        colorscale: "Viridis",
        opacity: 0.9,
        scene: sceneId,
        name: dataset.headers[zIdx],
      });
      return;
    }

    const grid = tryBuildGrid(x, y, z);
    if (mode === "surface" || grid.complete) {
      traces.push({
        type: "surface",
        x: grid.xUnique,
        y: grid.yUnique,
        z: grid.grid,
        colorscale: "Viridis",
        contours: { z: { show: true, usecolormap: true, project: { z: true } } },
        scene: sceneId,
        name: dataset.headers[zIdx],
        showscale: false,
      });
      if (!grid.complete) {
        setStatus(`Grid incomplete (${grid.missing} missing cells). Surface shown with gaps.`, true);
      }
    } else {
      traces.push({
        type: "scatter3d",
        mode: "markers",
        x,
        y,
        z,
        marker: { size: 3, color: z, colorscale: "Viridis", opacity: 0.85 },
        scene: sceneId,
        name: dataset.headers[zIdx],
      });
      setStatus("Auto mode: grid not complete, using scatter.", true);
    }
  });

  const layout = {
    margin: { t: 40, r: 20, l: 10, b: 10 },
    annotations,
    ...scenes,
  };

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
    refreshAxisSelectors(dataset);
    refreshMetrics(dataset);
  }
});

elXSelect.addEventListener("change", () => {
  const dataset = state.datasets.get(state.current);
  if (dataset) {
    refreshMetrics(dataset);
  }
});

elYSelect.addEventListener("change", () => {
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
