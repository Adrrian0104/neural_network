let layers = [];
let weights = [];
let biases = [];
let activations = [];

let input = [];
let target = [];

let running = false;

// gráfica
let chart;
let errorHistory = [];

// ---------- UTILS ----------
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
  return x * (1 - x);
}

function safeParse(text) {
  return text.split(",").map(v => parseFloat(v.trim()));
}

// ---------- INIT ----------
function initNetwork() {

  input = safeParse(document.getElementById("inputData").value);
  target = safeParse(document.getElementById("targetData").value);
  let hidden = safeParse(document.getElementById("layersInput").value);

  layers = [input.length, ...hidden, target.length];

  weights = [];
  biases = [];
  errorHistory = [];

  for (let l = 0; l < layers.length - 1; l++) {
    let w = [];

    for (let i = 0; i < layers[l]; i++) {
      w[i] = [];
      for (let j = 0; j < layers[l+1]; j++) {
        w[i][j] = (Math.random() - 0.5);
      }
    }

    weights.push(w);
    biases.push(new Array(layers[l+1]).fill(0));
  }

  initChart();
  drawNetwork([input]);

  document.getElementById("status").innerText = "Red creada";
}

// ---------- FORWARD ----------
function forward() {

  activations = [input];

  for (let l = 0; l < weights.length; l++) {

    let prev = activations[l];
    let next = [];

    for (let j = 0; j < layers[l+1]; j++) {

      let sum = biases[l][j];

      for (let i = 0; i < prev.length; i++) {
        sum += prev[i] * weights[l][i][j];
      }

      next[j] = sigmoid(sum);
    }

    activations.push(next);
  }

  drawNetwork(activations);

  let output = activations[activations.length - 1];

  let error = 0;
  for (let i = 0; i < output.length; i++) {
    error += Math.pow(target[i] - output[i], 2);
  }

  let mse = error / output.length;

  document.getElementById("errorValue").innerText = mse.toFixed(6);

  // gráfica
  errorHistory.push(mse);
  chart.data.labels.push(errorHistory.length);
  chart.data.datasets[0].data.push(mse);
  chart.update();

  return output;
}

// ---------- BACKWARD ----------
function backward(output) {

  let lr = 0.1;
  let deltas = new Array(weights.length);

  deltas[deltas.length-1] = output.map((a,i)=>
    (a - target[i]) * dsigmoid(a)
  );

  for (let l = deltas.length-2; l >= 0; l--) {

    deltas[l] = [];

    for (let i = 0; i < layers[l+1]; i++) {

      let sum = 0;

      for (let j = 0; j < deltas[l+1].length; j++) {
        sum += deltas[l+1][j] * weights[l+1][i][j];
      }

      deltas[l][i] = sum * dsigmoid(activations[l+1][i]);
    }
  }

  for (let l = 0; l < weights.length; l++) {
    for (let i = 0; i < weights[l].length; i++) {
      for (let j = 0; j < weights[l][i].length; j++) {
        weights[l][i][j] -= lr * deltas[l][j] * activations[l][i];
      }
    }
  }
}

// ---------- TRAIN ----------
function trainEpochs() {

  let epochs = parseInt(document.getElementById("epochsInput").value);

  for (let i = 0; i < epochs; i++) {
    let out = forward();
    backward(out);
  }

  document.getElementById("status").innerText =
    `Entrenado ${epochs} épocas`;
}

// ---------- ANIMACIÓN ----------
function play() {
  running = true;
  loop();
}

function pause() {
  running = false;
}

async function loop() {

  if (!running) return;

  let out = forward();
  backward(out);

  await new Promise(r => setTimeout(r, 200));
  loop();
}

// ---------- DIBUJO (MEJORADO 🔥) ----------
function drawNetwork(acts) {

  let canvas = document.getElementById("networkCanvas");
  let ctx = canvas.getContext("2d");

  canvas.width = layers.length * 220;
  canvas.height = Math.max(...layers) * 80;

  ctx.clearRect(0,0,canvas.width,canvas.height);

  // conexiones
  for (let l = 0; l < weights.length; l++) {
    for (let i = 0; i < weights[l].length; i++) {
      for (let j = 0; j < weights[l][i].length; j++) {

        let x1 = l * 200 + 50;
        let y1 = i * 70 + 50;
        let x2 = (l+1) * 200 + 50;
        let y2 = j * 70 + 50;

        let w = weights[l][i][j];

        ctx.strokeStyle = w > 0
          ? `rgba(0,200,0,${Math.abs(w)})`
          : `rgba(255,0,0,${Math.abs(w)})`;

        ctx.lineWidth = Math.max(1, Math.abs(w)*2);

        ctx.beginPath();
        ctx.moveTo(x1,y1);
        ctx.lineTo(x2,y2);
        ctx.stroke();
      }
    }
  }

  // nodos (MEJORADOS)
  acts.forEach((layer, l) => {
    layer.forEach((val, n) => {

      let x = l * 200 + 50;
      let y = n * 70 + 50;

      let intensity = Math.min(0.6, Math.abs(val));

      ctx.beginPath();
      ctx.arc(x,y,20,0,Math.PI*2);

      ctx.fillStyle = `rgba(30,144,255,${0.3 + intensity})`;
      ctx.fill();

      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // texto visible
      ctx.fillStyle = intensity > 0.4 ? "white" : "black";
      ctx.font = "12px Arial";
      ctx.fillText(val.toFixed(2), x-12, y+4);
    });
  });
}

// ---------- CHART ----------
function initChart() {
  chart = new Chart(document.getElementById("chart"), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Error',
        data: []
      }]
    },
    options: { animation: false }
  });
}

// ---------- RESET ----------
function reset() {
  location.reload();
}