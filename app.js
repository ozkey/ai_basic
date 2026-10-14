import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const INPUTS_PER_NUMBER = 10;
const INPUT_COUNT = INPUTS_PER_NUMBER * 2;
const HIDDEN_COUNT = 16;
const OUTPUT_COUNT = 19;

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const randomWeight = () => (Math.random() - 0.5) * 0.4;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const dsigmoidFromActivation = (y) => y * (1 - y);

class AdditionNet {
  constructor() {
    this.w1 = Array.from({ length: HIDDEN_COUNT }, () =>
      Array.from({ length: INPUT_COUNT }, randomWeight),
    );
    this.b1 = Array.from({ length: HIDDEN_COUNT }, randomWeight);
    this.w2 = Array.from({ length: OUTPUT_COUNT }, () =>
      Array.from({ length: HIDDEN_COUNT }, randomWeight),
    );
    this.b2 = Array.from({ length: OUTPUT_COUNT }, randomWeight);
  }

  forward(input) {
    const hidden = new Array(HIDDEN_COUNT);
    const hiddenRaw = new Array(HIDDEN_COUNT);
    for (let h = 0; h < HIDDEN_COUNT; h += 1) {
      let sum = this.b1[h];
      for (let i = 0; i < INPUT_COUNT; i += 1) {
        sum += this.w1[h][i] * input[i];
      }
      hiddenRaw[h] = sum;
      hidden[h] = sigmoid(sum);
    }

    const logits = new Array(OUTPUT_COUNT);
    let maxLogit = -Infinity;
    for (let o = 0; o < OUTPUT_COUNT; o += 1) {
      let sum = this.b2[o];
      for (let h = 0; h < HIDDEN_COUNT; h += 1) {
        sum += this.w2[o][h] * hidden[h];
      }
      logits[o] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    let denom = 0;
    const output = logits.map((logit) => {
      const value = Math.exp(logit - maxLogit);
      denom += value;
      return value;
    }).map((value) => value / denom);

    return { hiddenRaw, hidden, logits, output };
  }

  train(input, targetIndex, learningRate) {
    const { hidden, output } = this.forward(input);
    const target = new Array(OUTPUT_COUNT).fill(0);
    target[targetIndex] = 1;

    const outputError = output.map((value, index) => value - target[index]);

    const w2Grad = Array.from({ length: OUTPUT_COUNT }, () => new Array(HIDDEN_COUNT).fill(0));
    const b2Grad = new Array(OUTPUT_COUNT).fill(0);
    const hiddenError = new Array(HIDDEN_COUNT).fill(0);

    for (let o = 0; o < OUTPUT_COUNT; o += 1) {
      b2Grad[o] = outputError[o];
      for (let h = 0; h < HIDDEN_COUNT; h += 1) {
        w2Grad[o][h] = outputError[o] * hidden[h];
        hiddenError[h] += outputError[o] * this.w2[o][h];
      }
    }

    const w1Grad = Array.from({ length: HIDDEN_COUNT }, () => new Array(INPUT_COUNT).fill(0));
    const b1Grad = new Array(HIDDEN_COUNT).fill(0);

    for (let h = 0; h < HIDDEN_COUNT; h += 1) {
      const delta = hiddenError[h] * dsigmoidFromActivation(hidden[h]);
      b1Grad[h] = delta;
      for (let i = 0; i < INPUT_COUNT; i += 1) {
        w1Grad[h][i] = delta * input[i];
      }
    }

    for (let o = 0; o < OUTPUT_COUNT; o += 1) {
      this.b2[o] -= learningRate * b2Grad[o];
      for (let h = 0; h < HIDDEN_COUNT; h += 1) {
        this.w2[o][h] -= learningRate * w2Grad[o][h];
      }
    }

    for (let h = 0; h < HIDDEN_COUNT; h += 1) {
      this.b1[h] -= learningRate * b1Grad[h];
      for (let i = 0; i < INPUT_COUNT; i += 1) {
        this.w1[h][i] -= learningRate * w1Grad[h][i];
      }
    }

    const loss = -Math.log(output[targetIndex] + 1e-9);
    return { loss, output };
  }
}

function encodeExample(a, b) {
  const input = new Array(INPUT_COUNT).fill(0);
  input[a] = 1;
  input[INPUTS_PER_NUMBER + b] = 1;
  return input;
}

function sampleExample() {
  const a = Math.floor(Math.random() * INPUTS_PER_NUMBER);
  const b = Math.floor(Math.random() * INPUTS_PER_NUMBER);
  return { a, b, sum: a + b, input: encodeExample(a, b) };
}

function buildDataset() {
  const data = [];
  for (let a = 0; a < INPUTS_PER_NUMBER; a += 1) {
    for (let b = 0; b < INPUTS_PER_NUMBER; b += 1) {
      data.push({ a, b, sum: a + b, input: encodeExample(a, b) });
    }
  }
  return data;
}

const net = new AdditionNet();
const dataset = buildDataset();

let currentExample = sampleExample();
let training = false;
let trainingStarted = false;
let totalSteps = 0;
let movingLoss = 0;
let accuracy = 0;
let lastPrediction = 0;
let lastConfidence = 0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07111f);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 0, 28);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("app").appendChild(renderer.domElement);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const nodeInfoEl = document.getElementById("nodeInfo");
let selectedNode = null;

const ambient = new THREE.AmbientLight(0xffffff, 1.2);
scene.add(ambient);
const keyLight = new THREE.PointLight(0xaac7ff, 1.5, 100);
keyLight.position.set(0, 10, 20);
scene.add(keyLight);

const layerX = [-8, 0, 8];
const neuronRadius = 0.35;

const nodeGroups = {
  input: [],
  hidden: [],
  output: [],
};

function makeNode(color) {
  const geometry = new THREE.SphereGeometry(neuronRadius, 24, 24);
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: 0x000000,
    roughness: 0.35,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
  return mesh;
}

function placeLayer(nodes, x, count, spread, startY = 0) {
  for (let i = 0; i < count; i += 1) {
    const node = makeNode(0x203252);
    const y = startY + (i - (count - 1) / 2) * spread;
    node.position.set(x, y, 0);
    node.userData.layer = "layer";
    node.userData.index = i;
    nodes.push(node);
  }
}

placeLayer(nodeGroups.input, layerX[0], INPUT_COUNT, 1.2, 0.6);
for (let i = 0; i < INPUT_COUNT; i += 1) {
  nodeGroups.input[i].userData.layer = "input";
  nodeGroups.input[i].userData.slot = i < INPUTS_PER_NUMBER ? "left" : "right";
  nodeGroups.input[i].userData.digit = i < INPUTS_PER_NUMBER ? i : i - INPUTS_PER_NUMBER;
  nodeGroups.input[i].userData.label = i < INPUTS_PER_NUMBER
    ? `Left digit ${i}`
    : `Right digit ${i - INPUTS_PER_NUMBER}`;
}

placeLayer(nodeGroups.hidden, layerX[1], HIDDEN_COUNT, 1.35);
for (let i = 0; i < HIDDEN_COUNT; i += 1) {
  nodeGroups.hidden[i].userData.layer = "hidden";
  nodeGroups.hidden[i].userData.label = `Hidden feature ${i}`;
}

placeLayer(nodeGroups.output, layerX[2], OUTPUT_COUNT, 0.95);
for (let i = 0; i < OUTPUT_COUNT; i += 1) {
  nodeGroups.output[i].userData.layer = "output";
  nodeGroups.output[i].userData.sum = i;
  nodeGroups.output[i].userData.label = `Sum ${i}`;
}

const selectionMarker = new THREE.Mesh(
  new THREE.SphereGeometry(neuronRadius * 1.6, 24, 24),
  new THREE.MeshBasicMaterial({
    color: 0xffffff,
    wireframe: true,
    transparent: true,
    opacity: 0.9,
  }),
);
selectionMarker.visible = false;
scene.add(selectionMarker);

const connectionGroup = new THREE.Group();
scene.add(connectionGroup);

const connections = [];

function addConnections(fromNodes, toNodes, weightsGetter) {
  for (let i = 0; i < fromNodes.length; i += 1) {
    for (let j = 0; j < toNodes.length; j += 1) {
      const points = [fromNodes[i].position, toNodes[j].position];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x38507c, transparent: true, opacity: 0.45 });
      const line = new THREE.Line(geometry, material);
      connectionGroup.add(line);
      connections.push({
        line,
        from: fromNodes[i],
        to: toNodes[j],
        fromIndex: i,
        toIndex: j,
        weight: weightsGetter(i, j),
      });
    }
  }
}

addConnections(nodeGroups.input, nodeGroups.hidden, (i, h) => net.w1[h][i]);
addConnections(nodeGroups.hidden, nodeGroups.output, (h, o) => net.w2[o][h]);

const statsEl = document.getElementById("stats");
const leftInput = document.getElementById("leftNumber");
const rightInput = document.getElementById("rightNumber");
const trainToggle = document.getElementById("trainToggle");
const stepOnce = document.getElementById("stepOnce");

function updateTrainingButtonLabel() {
  if (!trainingStarted) {
    trainToggle.textContent = "Start training";
    return;
  }
  trainToggle.textContent = training ? "Pause training" : "Resume training";
}

function getNodeValue(node, state) {
  if (node.userData.layer === "input") {
    return currentExample.input[node.userData.index];
  }
  if (node.userData.layer === "hidden") {
    return state.hidden[node.userData.index];
  }
  if (node.userData.layer === "output") {
    return state.output[node.userData.index];
  }
  return 0;
}

function getNodeBias(node) {
  if (node.userData.layer === "hidden") {
    return net.b1[node.userData.index];
  }
  if (node.userData.layer === "output") {
    return net.b2[node.userData.index];
  }
  return 0;
}

function getIncomingConnections(node) {
  return connections.filter((connection) => connection.to === node);
}

function getOutgoingConnections(node) {
  return connections.filter((connection) => connection.from === node);
}

function formatSigned(value) {
  const sign = value >= 0 ? "+" : "−";
  return `${sign}${Math.abs(value).toFixed(4)}`;
}

function summarizeWeights(items) {
  if (!items.length) {
    return { count: 0, min: 0, max: 0, mean: 0, absMean: 0 };
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let absSum = 0;
  for (const item of items) {
    const value = item.weight;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    absSum += Math.abs(value);
  }

  return {
    count: items.length,
    min,
    max,
    mean: sum / items.length,
    absMean: absSum / items.length,
  };
}

function formatConnectionList(items, directionLabel) {
  const sorted = [...items].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  if (!sorted.length) {
    return `<div class="muted">No ${directionLabel} connections.</div>`;
  }

  return sorted.map((item) => {
    const label = item.from.userData.label || `Node ${item.fromIndex}`;
    const value = item.weight;
    const width = Math.min(Math.abs(value) * 180, 180);
    const color = value >= 0 ? "#4da3ff" : "#ff5d73";
    return `
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;gap:10px;">
          <span>${label} → ${item.to.userData.label || `Node ${item.toIndex}`}</span>
          <span style="color:${color};">${formatSigned(value)}</span>
        </div>
        <div class="bar"><span style="width:${Math.max(width, 4)}px;background:${color};"></span></div>
      </div>
    `;
  }).join("");
}

function renderNodeInfo(node, state) {
  if (!node) {
    nodeInfoEl.innerHTML = `
      <h2>Click a node</h2>
      <div class="muted">Inspect its layer, activation, bias, connected weights, and its current role in the network.</div>
    `;
    return;
  }

  const layer = node.userData.layer || "unknown";
  const value = getNodeValue(node, state);
  const bias = getNodeBias(node);
  const incoming = getIncomingConnections(node);
  const outgoing = getOutgoingConnections(node);
  const selectedLabel = node.userData.label || `Node ${node.userData.index}`;
  const active = layer === "input" ? (value > 0 ? "active" : "inactive") : value > 0.5 ? "strong" : "weak";
  const extraRows = [];

  if (layer === "input") {
    extraRows.push(["Input side", node.userData.slot === "left" ? "Left number" : "Right number"]);
    extraRows.push(["Digit slot", String(node.userData.digit)]);
    extraRows.push(["Current value", String(value)]);
  } else if (layer === "hidden") {
    extraRows.push(["Activation", value.toFixed(6)]);
    extraRows.push(["Raw pre-activation", state.hiddenRaw[node.userData.index].toFixed(6)]);
  } else if (layer === "output") {
    extraRows.push(["Probability", `${(value * 100).toFixed(2)}%`]);
    extraRows.push(["Logit", state.logits[node.userData.index].toFixed(6)]);
    extraRows.push(["Represents", `${node.userData.sum}`]);
  }

  const connectionSummary = [
    ["Incoming connections", String(incoming.length)],
    ["Outgoing connections", String(outgoing.length)],
  ];

  const incomingSummary = summarizeWeights(incoming);
  const outgoingSummary = summarizeWeights(outgoing);
  const topIncoming = formatConnectionList(incoming, "incoming");
  const topOutgoing = formatConnectionList(outgoing, "outgoing");

  const rank = layer === "output"
    ? state.output.filter((prob) => prob > value).length + 1
    : null;

  nodeInfoEl.innerHTML = `
    <h2>${selectedLabel}</h2>
    <div class="muted">Layer: ${layer} &middot; State: ${active}</div>
    <h3>Node details</h3>
    <table>
      ${[
        ["Index", String(node.userData.index)],
        ["Position", `(${node.position.x.toFixed(2)}, ${node.position.y.toFixed(2)}, ${node.position.z.toFixed(2)})`],
        ["Activation", value.toFixed(6)],
        ["Bias", formatSigned(bias)],
        ...extraRows,
      ].map(([label, val]) => `<tr><td>${label}</td><td>${val}</td></tr>`).join("")}
    </table>
    <h3>Connections</h3>
    <table>
      ${connectionSummary.map(([label, val]) => `<tr><td>${label}</td><td>${val}</td></tr>`).join("")}
      <tr><td>Incoming mean / abs mean</td><td>${incomingSummary.mean.toFixed(4)} / ${incomingSummary.absMean.toFixed(4)}</td></tr>
      <tr><td>Outgoing mean / abs mean</td><td>${outgoingSummary.mean.toFixed(4)} / ${outgoingSummary.absMean.toFixed(4)}</td></tr>
      <tr><td>Incoming min / max</td><td>${formatSigned(incomingSummary.min)} / ${formatSigned(incomingSummary.max)}</td></tr>
      <tr><td>Outgoing min / max</td><td>${formatSigned(outgoingSummary.min)} / ${formatSigned(outgoingSummary.max)}</td></tr>
    </table>
    ${rank !== null ? `<h3>Output rank</h3><div>Rank among sums: <b>#${rank}</b> out of ${OUTPUT_COUNT}</div>` : ""}
    <h3>Incoming weights</h3>
    ${topIncoming}
    <h3>Outgoing weights</h3>
    ${topOutgoing}
  `;
}

function selectNode(node) {
  selectedNode = node || null;
  if (!selectedNode) {
    selectionMarker.visible = false;
    return;
  }
  selectionMarker.visible = true;
  selectionMarker.position.copy(selectedNode.position);
}

function syncExampleFromUI() {
  const a = clamp(parseInt(leftInput.value, 10) || 0, 0, 9);
  const b = clamp(parseInt(rightInput.value, 10) || 0, 0, 9);
  currentExample = { a, b, sum: a + b, input: encodeExample(a, b) };
  leftInput.value = String(a);
  rightInput.value = String(b);
}

leftInput.addEventListener("input", syncExampleFromUI);
rightInput.addEventListener("input", syncExampleFromUI);

trainToggle.addEventListener("click", () => {
  trainingStarted = true;
  training = !training;
  updateTrainingButtonLabel();
});

stepOnce.addEventListener("click", () => {
  const sample = dataset[Math.floor(Math.random() * dataset.length)];
  currentExample = sample;
  leftInput.value = String(sample.a);
  rightInput.value = String(sample.b);
  trainOneStep(sample);
});

function trainOneStep(example) {
  const lr = 0.18;
  const { loss, output } = net.train(example.input, example.sum, lr);
  totalSteps += 1;
  movingLoss = movingLoss === 0 ? loss : movingLoss * 0.985 + loss * 0.015;

  const predictedIndex = output.indexOf(Math.max(...output));
  lastPrediction = predictedIndex;
  lastConfidence = output[predictedIndex];
}

function evaluateAccuracy() {
  let correct = 0;
  for (const example of dataset) {
    const { output } = net.forward(example.input);
    const predicted = output.indexOf(Math.max(...output));
    if (predicted === example.sum) correct += 1;
  }
  accuracy = correct / dataset.length;
}

function updateWeightsForView() {
  let index = 0;
  for (let i = 0; i < INPUT_COUNT; i += 1) {
    for (let h = 0; h < HIDDEN_COUNT; h += 1) {
      connections[index].weight = net.w1[h][i];
      index += 1;
    }
  }
  for (let h = 0; h < HIDDEN_COUNT; h += 1) {
    for (let o = 0; o < OUTPUT_COUNT; o += 1) {
      connections[index].weight = net.w2[o][h];
      index += 1;
    }
  }
}

function setNodeAppearance(nodes, activations, isOutput = false) {
  for (let i = 0; i < nodes.length; i += 1) {
    const activation = activations[i] ?? 0;
    const intensity = clamp(activation, 0, 1);
    const emissiveColor = isOutput ? new THREE.Color(0x2a3d7a) : new THREE.Color(0x14315f);
    nodes[i].material.color.setHSL(0.58 - intensity * 0.15, 0.7, 0.28 + intensity * 0.28);
    nodes[i].material.emissive.copy(emissiveColor).multiplyScalar(intensity * 0.8);
    nodes[i].scale.setScalar(0.9 + intensity * 0.45);
  }
}

function updateConnections() {
  for (const connection of connections) {
    const weight = connection.weight;
    const absWeight = Math.min(Math.abs(weight) * 2.5, 1);
    const color = new THREE.Color(weight >= 0 ? 0x4da3ff : 0xff5d73);
    connection.line.material.color.copy(color);
    connection.line.material.opacity = 0.08 + absWeight * 0.78;
    connection.line.material.linewidth = 1 + absWeight * 2;
  }
}

function renderState() {
  const state = net.forward(currentExample.input);
  setNodeAppearance(nodeGroups.input, currentExample.input);
  setNodeAppearance(nodeGroups.hidden, state.hidden);
  setNodeAppearance(nodeGroups.output, state.output, true);

  updateWeightsForView();
  updateConnections();

  const predicted = state.output.indexOf(Math.max(...state.output));
  const confidence = state.output[predicted];
  const correct = predicted === currentExample.sum ? "yes" : "no";

  statsEl.innerHTML = `
    <div><b>Example:</b> ${currentExample.a} + ${currentExample.b} = ${currentExample.sum}</div>
    <div><b>Prediction:</b> ${predicted} (${(confidence * 100).toFixed(1)}%)</div>
    <div><b>Correct:</b> ${correct}</div>
    <div><b>Steps:</b> ${totalSteps}</div>
    <div><b>Loss:</b> ${movingLoss.toFixed(4)}</div>
    <div><b>Accuracy:</b> ${(accuracy * 100).toFixed(1)}%</div>
    <div><b>Last trained:</b> ${lastPrediction} (${(lastConfidence * 100).toFixed(1)}%)</div>
  `;

  renderNodeInfo(selectedNode, state);
}

function stepTraining() {
  if (!training) return;
  for (let i = 0; i < 30; i += 1) {
    const sample = dataset[Math.floor(Math.random() * dataset.length)];
    trainOneStep(sample);
  }
}

function animate() {
  requestAnimationFrame(animate);
  stepTraining();
  if (totalSteps % 40 === 0) {
    evaluateAccuracy();
  }
  renderState();
  scene.rotation.y = Math.sin(performance.now() * 0.00015) * 0.08;
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function handlePointerDown(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointer, camera);
  const meshes = [...nodeGroups.input, ...nodeGroups.hidden, ...nodeGroups.output];
  const intersections = raycaster.intersectObjects(meshes, false);
  if (intersections.length > 0) {
    selectNode(intersections[0].object);
    return;
  }
  selectNode(null);
}

renderer.domElement.addEventListener("pointerdown", handlePointerDown);

syncExampleFromUI();
evaluateAccuracy();
updateTrainingButtonLabel();
renderState();
animate();
