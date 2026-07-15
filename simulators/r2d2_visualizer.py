"""Self-contained HTML visualization for R2D2 RAO* policies."""

import ast
import json
import math
import random
from collections import defaultdict
from pathlib import Path


class R2D2Visualizer:
    """Export an R2D2 policy rollout as an interactive HTML document."""

    def __init__(self, model):
        self.model = model

    def write_html(self, graph, output_path, seed=7, max_steps=None):
        """Write a standalone HTML file and return its path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self._visualization_data(
            graph,
            seed=seed,
            max_steps=max_steps,
        )
        output_path.write_text(self._html_document(data), encoding="utf-8")
        return output_path

    def sample_policy_path(self, graph, seed=7, max_steps=None):
        """Sample a reproducible observation branch from the RAO* policy graph."""
        if max_steps is None:
            max_steps = max((node.depth for node in graph.nodes.values()), default=0)
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        rng = random.Random(seed)
        node = graph.root
        path = [node]
        for _ in range(max_steps):
            if node.best_action is None or node.terminal:
                break
            children = graph.hyperedge_successors(node, node.best_action)
            if not children:
                break
            probabilities = self._child_probabilities(node.best_action, children)
            node = rng.choices(children, weights=probabilities, k=1)[0]
            path.append(node)
        return path

    @staticmethod
    def action_name(action):
        """Return an action label from a tuple, operator, or operator name."""
        if action is None:
            return None
        raw_action = action.name if hasattr(action, "name") else action
        if isinstance(raw_action, (tuple, list)) and len(raw_action) >= 3:
            return raw_action[2]
        if isinstance(raw_action, str):
            try:
                parsed = ast.literal_eval(raw_action)
            except (SyntaxError, ValueError):
                return raw_action
            if isinstance(parsed, (tuple, list)) and len(parsed) >= 3:
                return parsed[2]
        return str(raw_action)

    def _visualization_data(self, graph, seed, max_steps):
        height, width = self.model.env.shape
        actions = {
            action[2]: {
                "rowDelta": action[0],
                "columnDelta": action[1],
            }
            for action in self.model.action_list
        }
        path = self.sample_policy_path(
            graph,
            seed=seed,
            max_steps=max_steps,
        )

        return {
            "environment": {
                "rows": height,
                "columns": width,
                "start": list(next(iter(self.model.initial_belief()))[:2]),
                "goal": list(self.model.goal),
                "ice": [list(position) for position in self.model.icy_blocks],
                "fires": [list(position) for position in self.model.fires],
                "blocked": [list(position) for position in self.model.blocked_blocks],
                "actions": actions,
            },
            "summary": {
                "rootValue": self._finite_number(graph.root.value),
                "rootExecutionRisk": self._finite_number(graph.root.exec_risk),
                "graphNodes": len(graph.nodes),
                "steps": max(0, len(path) - 1),
                "seed": seed,
            },
            "frames": [self._frame_data(node) for node in path],
        }

    def _frame_data(self, node):
        position_probabilities = defaultdict(float)
        depths = defaultdict(list)
        for state, probability in node.state.belief.items():
            position = (state[0], state[1])
            position_probabilities[position] += probability
            if len(state) > 2:
                depths[position].append(state[2])

        belief = [
            {
                "row": row,
                "column": column,
                "probability": probability,
                "depth": min(depths[(row, column)], default=node.depth),
            }
            for (row, column), probability in sorted(position_probabilities.items())
        ]
        representative = max(
            belief,
            key=lambda state: state["probability"],
            default=None,
        )
        return {
            "depth": node.depth,
            "belief": belief,
            "representative": representative,
            "action": self.action_name(node.best_action),
            "value": self._finite_number(node.value),
            "risk": self._finite_number(node.risk),
            "executionRisk": self._finite_number(node.exec_risk),
            "executionRiskBound": self._finite_number(node.exec_risk_bound),
            "branchProbability": self._finite_number(node.probability),
            "terminal": bool(node.terminal),
        }

    @staticmethod
    def _finite_number(value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        return numeric_value if math.isfinite(numeric_value) else None

    @staticmethod
    def _child_probabilities(action, children):
        probabilities = action.properties.get("prob", [])
        if len(probabilities) != len(children) or sum(probabilities) <= 0.0:
            probabilities = [max(child.probability, 0.0) for child in children]
        if sum(probabilities) <= 0.0:
            return [1.0] * len(children)
        return probabilities

    @staticmethod
    def _html_document(data):
        serialized_data = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
        serialized_data = serialized_data.replace("</", "<\\/")
        return HTML_TEMPLATE.replace("__R2D2_DATA__", serialized_data)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R2D2 RAO* Policy</title>
  <style>
    :root {
      color-scheme: light;
      --background: #eef1f4;
      --surface: #ffffff;
      --text: #17212b;
      --muted: #61707f;
      --line: #a7b0ba;
      --free: #f9faf9;
      --ice: #b9e3f2;
      --fire: #e76f51;
      --blocked: #555b66;
      --goal: #76b947;
      --belief: #2878b5;
      --belief-edge: #15527c;
      --action: #e07a2f;
      --button: #243b53;
      --button-hover: #172b3f;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-width: 320px;
      min-height: 100vh;
      background: var(--background);
      color: var(--text);
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      min-height: 68px;
      padding: 14px 24px;
      border-bottom: 1px solid #d2d8de;
      background: var(--surface);
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .summary {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px 18px;
      color: var(--muted);
      font-size: 13px;
    }

    .summary strong { color: var(--text); }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      min-height: calc(100vh - 68px);
    }

    .workspace {
      min-width: 0;
      padding: 24px;
      overflow: hidden;
    }

    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }

    .step-title {
      min-width: 0;
      font-size: 15px;
      font-weight: 650;
    }

    .step-subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
    }

    .controls {
      display: flex;
      flex: 0 0 auto;
      gap: 8px;
    }

    button {
      min-height: 38px;
      border: 1px solid #aeb7c1;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }

    button:hover:not(:disabled) { background: #f0f3f5; }
    button:focus-visible { outline: 3px solid #82b7db; outline-offset: 2px; }
    button:disabled { cursor: default; opacity: 0.38; }

    button.primary {
      border-color: var(--button);
      background: var(--button);
      color: #ffffff;
    }

    button.primary:hover:not(:disabled) { background: var(--button-hover); }

    .board-viewport {
      width: 100%;
      overflow-x: auto;
      border: 1px solid #bcc5ce;
      background: var(--surface);
    }

    .board {
      display: grid;
      width: 100%;
      min-width: max-content;
    }

    .cell {
      position: relative;
      width: max(64px, calc((100vw - 390px) / var(--columns)));
      max-width: 132px;
      aspect-ratio: 1;
      overflow: hidden;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      background: var(--free);
    }

    .cell.ice { background: var(--ice); }
    .cell.fire { background: var(--fire); }
    .cell.blocked { background: var(--blocked); }
    .cell.goal { background: var(--goal); }

    .coordinate {
      position: absolute;
      top: 5px;
      left: 6px;
      color: #4c5a67;
      font-size: 9px;
      z-index: 1;
    }

    .fire .coordinate, .blocked .coordinate { color: #ffffff; }

    .terrain-label {
      position: absolute;
      right: 7px;
      bottom: 5px;
      color: #26333f;
      font-size: 11px;
      font-weight: 750;
    }

    .fire .terrain-label, .blocked .terrain-label { color: #ffffff; }

    .start-ring {
      position: absolute;
      inset: 17%;
      border: 2px dashed #264653;
      border-radius: 50%;
      opacity: 0.72;
    }

    .belief-marker {
      position: absolute;
      left: 50%;
      top: 50%;
      display: grid;
      place-items: center;
      width: calc(28% + var(--probability) * 24%);
      aspect-ratio: 1;
      transform: translate(-50%, -50%);
      border: 2px solid var(--belief-edge);
      border-radius: 50%;
      background: rgb(40 120 181 / calc(0.25 + var(--probability) * 0.65));
      color: #ffffff;
      font-size: 10px;
      font-weight: 750;
      z-index: 3;
      animation: arrive 240ms ease-out;
    }

    .belief-marker.primary { box-shadow: 0 0 0 3px rgb(255 255 255 / 0.9); }

    .action-arrow {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      color: var(--action);
      font-size: clamp(30px, 4vw, 52px);
      font-weight: 800;
      line-height: 1;
      text-shadow: 0 1px 0 #ffffff;
      z-index: 4;
    }

    @keyframes arrive {
      from { opacity: 0.25; transform: translate(-50%, -50%) scale(0.8); }
      to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    }

    aside {
      padding: 22px;
      border-left: 1px solid #d2d8de;
      background: var(--surface);
    }

    .panel-section {
      padding: 16px 0;
      border-bottom: 1px solid #e1e5e9;
    }

    .panel-section:first-child { padding-top: 0; }
    .panel-section:last-child { border-bottom: 0; }

    h2 {
      margin: 0 0 12px;
      font-size: 13px;
      font-weight: 750;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    .action-value {
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 42px;
    }

    .action-symbol {
      width: 38px;
      color: var(--action);
      font-size: 32px;
      font-weight: 800;
      line-height: 1;
      text-align: center;
    }

    .action-name { font-size: 17px; font-weight: 700; }

    dl {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 9px 14px;
      margin: 0;
      font-size: 13px;
    }

    dt { color: var(--muted); }
    dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 650; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    th, td {
      padding: 7px 4px;
      border-bottom: 1px solid #e5e8eb;
      text-align: right;
    }

    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-weight: 650; }
    tbody tr:last-child td { border-bottom: 0; }

    @media (max-width: 860px) {
      header { align-items: flex-start; flex-direction: column; }
      .summary { justify-content: flex-start; }
      main { display: block; }
      .workspace { padding: 16px; }
      .toolbar { align-items: flex-start; flex-direction: column; }
      .controls { width: 100%; }
      .controls button { flex: 1 1 0; padding: 0 8px; }
      aside { border-top: 1px solid #d2d8de; border-left: 0; }
      .cell { width: 64px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>R2D2 RAO* Policy</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section class="workspace">
      <div class="toolbar">
        <div>
          <div class="step-title" id="step-title"></div>
          <div class="step-subtitle" id="step-subtitle"></div>
        </div>
        <div class="controls">
          <button type="button" id="reset-button">Reset</button>
          <button type="button" id="previous-button" aria-label="Previous step">&larr;</button>
          <button type="button" class="primary" id="next-button">Run next action</button>
        </div>
      </div>
      <div class="board-viewport">
        <div class="board" id="board"></div>
      </div>
    </section>
    <aside>
      <section class="panel-section">
        <h2>Current action</h2>
        <div class="action-value">
          <div class="action-symbol" id="action-symbol"></div>
          <div class="action-name" id="action-name"></div>
        </div>
      </section>
      <section class="panel-section">
        <h2>Node metrics</h2>
        <dl id="metrics"></dl>
      </section>
      <section class="panel-section">
        <h2>Belief state</h2>
        <table>
          <thead><tr><th>Position</th><th>Probability</th><th>Depth</th></tr></thead>
          <tbody id="belief-table"></tbody>
        </table>
      </section>
    </aside>
  </main>
  <script>
    "use strict";

    const data = __R2D2_DATA__;
    const environment = data.environment;
    const frames = data.frames;
    const actionSymbols = {
      up: "\u2191",
      right: "\u2192",
      down: "\u2193",
      left: "\u2190"
    };
    let frameIndex = 0;

    const board = document.getElementById("board");
    const stepTitle = document.getElementById("step-title");
    const stepSubtitle = document.getElementById("step-subtitle");
    const actionSymbol = document.getElementById("action-symbol");
    const actionName = document.getElementById("action-name");
    const metrics = document.getElementById("metrics");
    const beliefTable = document.getElementById("belief-table");
    const previousButton = document.getElementById("previous-button");
    const nextButton = document.getElementById("next-button");
    const resetButton = document.getElementById("reset-button");

    function positionKey(row, column) {
      return `${row},${column}`;
    }

    function positionSet(positions) {
      return new Set(positions.map(position => positionKey(position[0], position[1])));
    }

    const ice = positionSet(environment.ice);
    const fires = positionSet(environment.fires);
    const blocked = positionSet(environment.blocked);
    const startKey = positionKey(environment.start[0], environment.start[1]);
    const goalKey = positionKey(environment.goal[0], environment.goal[1]);

    function formatNumber(value, digits = 3) {
      return value === null ? "n/a" : Number(value).toFixed(digits);
    }

    function metric(label, value) {
      return `<dt>${label}</dt><dd>${value}</dd>`;
    }

    function renderSummary() {
      document.getElementById("summary").innerHTML = [
        `<span>Value <strong>${formatNumber(data.summary.rootValue)}</strong></span>`,
        `<span>Execution risk <strong>${formatNumber(data.summary.rootExecutionRisk)}</strong></span>`,
        `<span>Graph nodes <strong>${data.summary.graphNodes}</strong></span>`,
        `<span>Actions <strong>${data.summary.steps}</strong></span>`
      ].join("");
    }

    function createCell(row, column, frame) {
      const key = positionKey(row, column);
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.position = key;

      let terrainLabel = "";
      if (blocked.has(key)) {
        cell.classList.add("blocked");
        terrainLabel = "BLOCKED";
      } else if (fires.has(key)) {
        cell.classList.add("fire");
        terrainLabel = "FIRE";
      } else if (key === goalKey) {
        cell.classList.add("goal");
        terrainLabel = "GOAL";
      } else if (ice.has(key)) {
        cell.classList.add("ice");
        terrainLabel = "ICE";
      }

      cell.innerHTML = `<span class="coordinate">${row},${column}</span>`;
      if (terrainLabel) {
        cell.insertAdjacentHTML("beforeend", `<span class="terrain-label">${terrainLabel}</span>`);
      }
      if (key === startKey) {
        cell.insertAdjacentHTML("beforeend", '<span class="start-ring"></span>');
      }

      const belief = frame.belief.find(state => (
        state.row === row && state.column === column
      ));
      if (belief) {
        const marker = document.createElement("span");
        marker.className = "belief-marker";
        const representative = frame.representative;
        if (representative && representative.row === row && representative.column === column) {
          marker.classList.add("primary");
        }
        marker.style.setProperty("--probability", belief.probability);
        marker.textContent = belief.probability.toFixed(2);
        cell.appendChild(marker);
      }

      const representative = frame.representative;
      if (
        frame.action && representative && representative.row === row
        && representative.column === column
      ) {
        const arrow = document.createElement("span");
        arrow.className = "action-arrow";
        arrow.textContent = actionSymbols[frame.action] || frame.action;
        cell.appendChild(arrow);
      }
      return cell;
    }

    function renderBoard(frame) {
      board.innerHTML = "";
      board.style.setProperty("--columns", environment.columns);
      board.style.gridTemplateColumns = `repeat(${environment.columns}, max-content)`;
      for (let row = 0; row < environment.rows; row += 1) {
        for (let column = 0; column < environment.columns; column += 1) {
          board.appendChild(createCell(row, column, frame));
        }
      }
    }

    function renderBelief(frame) {
      beliefTable.innerHTML = frame.belief.map(state => (
        `<tr><td>(${state.row}, ${state.column})</td>`
        + `<td>${state.probability.toFixed(3)}</td><td>${state.depth}</td></tr>`
      )).join("");
    }

    function renderFrame() {
      const frame = frames[frameIndex];
      const lastAction = frameIndex > 0 ? frames[frameIndex - 1].action : null;
      const action = frame.action || "terminal";
      stepTitle.textContent = `Policy step ${frameIndex + 1} of ${frames.length}`;
      stepSubtitle.textContent = lastAction
        ? `Executed ${lastAction}; observation branch probability ${formatNumber(frame.branchProbability)}`
        : "Initial belief";
      actionSymbol.textContent = actionSymbols[frame.action] || "\u25a0";
      actionName.textContent = action;
      metrics.innerHTML = [
        metric("Value", formatNumber(frame.value)),
        metric("State risk", formatNumber(frame.risk)),
        metric("Execution risk", formatNumber(frame.executionRisk)),
        metric("Risk bound", formatNumber(frame.executionRiskBound)),
        metric("Depth", frame.depth)
      ].join("");
      previousButton.disabled = frameIndex === 0;
      nextButton.disabled = frameIndex === frames.length - 1;
      nextButton.textContent = nextButton.disabled ? "Policy complete" : "Run next action";
      renderBoard(frame);
      renderBelief(frame);
    }

    previousButton.addEventListener("click", () => {
      if (frameIndex > 0) {
        frameIndex -= 1;
        renderFrame();
      }
    });

    nextButton.addEventListener("click", () => {
      if (frameIndex < frames.length - 1) {
        frameIndex += 1;
        renderFrame();
      }
    });

    resetButton.addEventListener("click", () => {
      frameIndex = 0;
      renderFrame();
    });

    document.addEventListener("keydown", event => {
      if (event.key === "ArrowRight") nextButton.click();
      if (event.key === "ArrowLeft") previousButton.click();
      if (event.key === "Home") resetButton.click();
    });

    renderSummary();
    renderFrame();
  </script>
</body>
</html>
"""
