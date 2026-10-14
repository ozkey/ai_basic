class AddNetwork {
    constructor(hiddenSize = 4, learningRate = 0.03) {
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;
        this.w1 = Array.from({ length: hiddenSize }, () => [
            (Math.random() * 2 - 1) * 0.6,
            (Math.random() * 2 - 1) * 0.6
        ]);
        this.b1 = Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * 0.2);
        this.w2 = Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * 0.6);
        this.b2 = (Math.random() * 2 - 1) * 0.2;
    }

    forward(a, b) {
        const z1 = [];
        const h = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            const zi = this.w1[i][0] * a + this.w1[i][1] * b + this.b1[i];
            z1.push(zi);
            h.push(Math.tanh(zi));
        }

        let out = this.b2;
        for (let i = 0; i < this.hiddenSize; i++) {
            out += this.w2[i] * h[i];
        }
        return { z1, h, out };
    }

    predict(a, b) {
        return this.forward(a, b).out;
    }

    trainOne(a, b, target) {
        const { z1, h, out } = this.forward(a, b);
        const error = out - target;

        for (let i = 0; i < this.hiddenSize; i++) {
            const gradW2 = error * h[i];
            this.w2[i] -= this.learningRate * gradW2;
        }
        this.b2 -= this.learningRate * error;

        for (let i = 0; i < this.hiddenSize; i++) {
            const tanhGrad = 1 - Math.tanh(z1[i]) ** 2;
            const hiddenDelta = error * this.w2[i] * tanhGrad;
            this.w1[i][0] -= this.learningRate * hiddenDelta * a;
            this.w1[i][1] -= this.learningRate * hiddenDelta * b;
            this.b1[i] -= this.learningRate * hiddenDelta;
        }

        return error * error;
    }

    trainBatch(batchSize = 64) {
        let mse = 0;
        for (let i = 0; i < batchSize; i++) {
            const a = randomTwoDigitInteger();
            const b = randomTwoDigitInteger();
            mse += this.trainOne(normalizeInput(a), normalizeInput(b), normalizeOutput(a + b));
        }
        return mse / batchSize;
    }
}

const INPUT_SCALE = 100;
const OUTPUT_SCALE = 200;

function randomTwoDigitInteger() {
    return Math.floor(Math.random() * 90) + 10;
}

function normalizeInput(value) {
    return value / INPUT_SCALE;
}

function normalizeOutput(value) {
    return value / OUTPUT_SCALE;
}

function denormalizeOutput(value) {
    return value * OUTPUT_SCALE;
}

function colorForWeight(weight) {
    const intensity = Math.min(1, Math.abs(weight));
    if (weight >= 0) {
        return new THREE.Color(0.15 + intensity * 0.75, 0.25 + intensity * 0.6, 1.0);
    }
    return new THREE.Color(1.0, 0.2 + intensity * 0.45, 0.2 + intensity * 0.45);
}

function textTexture(text, textColor = "#101828", bgColor = "rgba(255,255,255,0.92)") {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "bold 42px Arial";
    ctx.fillStyle = textColor;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

function setSpriteText(sprite, text, scale = 1.35) {
    const previous = sprite.material.map;
    sprite.material.map = textTexture(text);
    sprite.material.needsUpdate = true;
    sprite.scale.set(scale, scale * 0.28, 1);
    if (previous) {
        previous.dispose();
    }
}

function createLabelSprite(text, scale = 1.35) {
    const material = new THREE.SpriteMaterial({
        map: textTexture(text),
        transparent: true
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(scale, scale * 0.28, 1);
    return sprite;
}

const network = new AddNetwork(4, 0.03);
let trainedSamples = 0;
let latestMse = 0;
let autoTrainId = null;

const sampleCountEl = document.getElementById("sample-count");
const mseEl = document.getElementById("mse-value");
const weightsPanelEl = document.getElementById("weights-panel");
const inputAEl = document.getElementById("input-a");
const inputBEl = document.getElementById("input-b");
const expectedEl = document.getElementById("expected-value");
const predictionEl = document.getElementById("prediction-value");
const errorEl = document.getElementById("error-value");
const autoBtn = document.getElementById("toggle-auto");

const container = document.getElementById("network3d");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf7f9fd);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0, 10);

const renderer = new THREE.WebGLRenderer({ antialias: true });
container.appendChild(renderer.domElement);

const ambient = new THREE.AmbientLight(0xffffff, 0.65);
scene.add(ambient);
const directional = new THREE.DirectionalLight(0xffffff, 0.9);
directional.position.set(6, 8, 10);
scene.add(directional);

const nodeMaterialInput = new THREE.MeshStandardMaterial({ color: 0x4f8cff });
const nodeMaterialHidden = new THREE.MeshStandardMaterial({ color: 0x49c78e });
const nodeMaterialOutput = new THREE.MeshStandardMaterial({ color: 0xff9f43 });
const sphereGeometry = new THREE.SphereGeometry(0.25, 32, 32);

const inputPos = [new THREE.Vector3(-4, 1.2, 0), new THREE.Vector3(-4, -1.2, 0)];
const hiddenPos = [
    new THREE.Vector3(0, 2.4, 0),
    new THREE.Vector3(0, 0.8, 0),
    new THREE.Vector3(0, -0.8, 0),
    new THREE.Vector3(0, -2.4, 0)
];
const outputPos = [new THREE.Vector3(4, 0, 0)];

const outputNode = new THREE.Mesh(sphereGeometry, nodeMaterialOutput);
outputNode.position.copy(outputPos[0]);
scene.add(outputNode);

function addNode(position, material, label) {
    const node = new THREE.Mesh(sphereGeometry, material);
    node.position.copy(position);
    scene.add(node);

    const sprite = createLabelSprite(label, 1.25);
    sprite.position.set(position.x, position.y + 0.68, position.z);
    scene.add(sprite);
    return node;
}

inputPos.forEach((p, i) => addNode(p, nodeMaterialInput, `x${i + 1}`));
hiddenPos.forEach((p, i) => addNode(p, nodeMaterialHidden, `h${i + 1}`));
const outputLabel = createLabelSprite("sum", 1.4);
outputLabel.position.set(outputPos[0].x, outputPos[0].y + 0.68, outputPos[0].z);
scene.add(outputLabel);

const edges = [];

function addEdge(from, to, weightGetter) {
    const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
    const material = new THREE.LineBasicMaterial({
        color: 0x89a0ff,
        transparent: true,
        opacity: 0.7
    });
    const line = new THREE.Line(geometry, material);
    scene.add(line);

    const label = createLabelSprite("0.00", 1.05);
    label.position.set(
        (from.x + to.x) / 2,
        (from.y + to.y) / 2,
        (from.z + to.z) / 2
    );
    scene.add(label);

    edges.push({ line, label, weightGetter });
}

for (let h = 0; h < network.hiddenSize; h++) {
    for (let i = 0; i < 2; i++) {
        addEdge(inputPos[i], hiddenPos[h], () => network.w1[h][i]);
    }
}
for (let h = 0; h < network.hiddenSize; h++) {
    addEdge(hiddenPos[h], outputPos[0], () => network.w2[h]);
}

function resizeRenderer() {
    const width = container.clientWidth || 640;
    const height = container.clientHeight || 460;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
}

window.addEventListener("resize", resizeRenderer);
resizeRenderer();

function updateNetworkView() {
    for (const edge of edges) {
        const w = edge.weightGetter();
        const color = colorForWeight(w);
        edge.line.material.color.copy(color);
        edge.line.material.opacity = 0.2 + Math.min(0.8, Math.abs(w) * 0.8);
        setSpriteText(edge.label, w.toFixed(2), 1.05);
    }
}

function refreshWeightsPanel() {
    const lines = [];
    lines.push("Input -> Hidden");
    for (let h = 0; h < network.hiddenSize; h++) {
        lines.push(`h${h + 1}: w(a)=${network.w1[h][0].toFixed(3)}  w(b)=${network.w1[h][1].toFixed(3)}  b=${network.b1[h].toFixed(3)}`);
    }
    lines.push("");
    lines.push("Hidden -> Output");
    for (let h = 0; h < network.hiddenSize; h++) {
        lines.push(`w(h${h + 1})=${network.w2[h].toFixed(3)}`);
    }
    lines.push(`b(out)=${network.b2.toFixed(3)}`);
    weightsPanelEl.textContent = lines.join("\n");
}

function refreshPrediction() {
    const a = Number(inputAEl.value);
    const b = Number(inputBEl.value);
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
        return;
    }

    if (a < 10 || a > 99 || b < 10 || b > 99) {
        return;
    }

    const expected = a + b;
    const prediction = denormalizeOutput(network.predict(normalizeInput(a), normalizeInput(b)));
    const roundedPrediction = Math.round(prediction);
    const absError = Math.abs(expected - roundedPrediction);

    expectedEl.textContent = String(expected);
    predictionEl.textContent = String(roundedPrediction);
    errorEl.textContent = String(absError);

    const pulse = 0.95 + Math.min(1.5, Math.abs(prediction) / OUTPUT_SCALE) * 0.18;
    outputNode.scale.set(pulse, pulse, pulse);
}

function refreshStats() {
    sampleCountEl.textContent = String(trainedSamples);
    mseEl.textContent = latestMse.toFixed(6);
    refreshWeightsPanel();
    updateNetworkView();
    refreshPrediction();
}

function trainSamples(count) {
    latestMse = network.trainBatch(count);
    trainedSamples += count;
    refreshStats();
}

document.getElementById("train-100").addEventListener("click", () => trainSamples(100));
document.getElementById("train-1000").addEventListener("click", () => trainSamples(1000));
document.getElementById("predict-btn").addEventListener("click", refreshPrediction);
inputAEl.addEventListener("input", refreshPrediction);
inputBEl.addEventListener("input", refreshPrediction);

autoBtn.addEventListener("click", () => {
    if (autoTrainId) {
        clearInterval(autoTrainId);
        autoTrainId = null;
        autoBtn.textContent = "Start Auto Train";
        return;
    }

    autoTrainId = setInterval(() => {
        trainSamples(200);
    }, 60);
    autoBtn.textContent = "Stop Auto Train";
});

function animate() {
    requestAnimationFrame(animate);
    scene.rotation.y = Math.sin(Date.now() * 0.00035) * 0.08;
    renderer.render(scene, camera);
}

refreshStats();
animate();
