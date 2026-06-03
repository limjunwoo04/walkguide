// 폰 브라우저용 보행 안내.
// 카메라 → ONNX YOLO11 추론 → 방향/근접/위험도 판단 → 한국어 음성(Web Speech).
// PC의 guide.py 로직을 그대로 JS로 옮긴 것. 좌표 계산은 640 입력 공간에서 한다
// (방향=중심x/640, 근접=면적/640² 라서 화면 크기와 무관 → 그대로 비율로 쓰임).

const INPUT = 640;
const PROX_FACTOR = { near: 1.0, mid: 0.6 };
const DIR_NAMES = { left: "왼쪽", front: "전방", right: "오른쪽" };
// 박스 색(클래스별). 없으면 초록.
const COLORS = {
  car: "#ff3030", bus: "#ff3030", truck: "#ff3030",
  electric_scooter: "#ff8c00", motorcycle: "#ff8c00",
  bollard: "#ffd000", traffic_cone: "#ffd000",
  bicycle: "#00c8ff", person: "#00ff5a",
  traffic_light: "#ffff00", tree: "#7cb342",
};

let CFG, session, inName, outName;
let labels, labelsKo, riskWeights, conf, iou, cooldown, leftB, rightB, nearR, midR;
let running = false;
const lastSpoke = {};

const video = document.createElement("video");
video.setAttribute("playsinline", "");
video.muted = true;
const dcanvas = document.getElementById("view");
const dctx = dcanvas.getContext("2d");
const pcanvas = document.createElement("canvas");
pcanvas.width = INPUT; pcanvas.height = INPUT;
const pctx = pcanvas.getContext("2d", { willReadFrequently: true });
const statusEl = document.getElementById("status");

function setStatus(t) { statusEl.textContent = t; }

document.getElementById("go").addEventListener("click", async () => {
  document.getElementById("go").disabled = true;
  try { await start(); } catch (e) { setStatus("오류: " + e.message); alert("시작 실패: " + e.message); }
});

async function start() {
  setStatus("설정 로드 중…");
  CFG = await (await fetch("app_config.json")).json();
  labels = CFG.labels; labelsKo = CFG.labels_ko; riskWeights = CFG.risk_weights;
  conf = CFG.conf; iou = CFG.iou; cooldown = CFG.tts.cooldown_sec;
  leftB = CFG.direction.left_bound; rightB = CFG.direction.right_bound;
  nearR = CFG.proximity.near_area_ratio; midR = CFG.proximity.mid_area_ratio;

  setStatus("음성 준비 중…");
  primeVoices();

  setStatus("카메라 켜는 중…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  dcanvas.width = video.videoWidth;
  dcanvas.height = video.videoHeight;

  setStatus("모델 로드 중… (10MB, 처음만 시간 걸림)");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
  inName = session.inputNames[0];
  outName = session.outputNames[0];

  document.getElementById("start").style.display = "none";
  running = true;
  loop();
}

// 모바일은 사용자 제스처 후에야 음성이 나오므로, 시작 시 한 번 깨워둔다.
function primeVoices() {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.getVoices();
  const u = new SpeechSynthesisUtterance(" ");
  u.lang = "ko-KR"; u.volume = 0;
  speechSynthesis.speak(u);
}

// letterbox 정보(비율 유지 스케일 + 좌상단 패딩). 추론 후 원본 좌표로 되돌릴 때 쓴다.
let lb = { scale: 1, padX: 0, padY: 0 };

function preprocess() {
  // 카메라 프레임을 비율 유지로 축소 후 가운데 배치, 남는 곳은 회색(114) 패딩.
  // ultralytics 학습/추론과 동일한 letterbox → 사람/차가 찌그러지지 않음.
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(INPUT / vw, INPUT / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const padX = Math.floor((INPUT - nw) / 2), padY = Math.floor((INPUT - nh) / 2);
  lb = { scale, padX, padY };
  pctx.fillStyle = "rgb(114,114,114)";
  pctx.fillRect(0, 0, INPUT, INPUT);
  pctx.drawImage(video, 0, 0, vw, vh, padX, padY, nw, nh);
  const { data } = pctx.getImageData(0, 0, INPUT, INPUT); // RGBA
  const area = INPUT * INPUT;
  const f = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    f[i] = data[i * 4] / 255;             // R 평면
    f[area + i] = data[i * 4 + 1] / 255;  // G 평면
    f[2 * area + i] = data[i * 4 + 2] / 255; // B 평면
  }
  return new ort.Tensor("float32", f, [1, 3, INPUT, INPUT]);
}

// 출력 [1, 4+nc, 8400] → 박스 후보 추출 + NMS
function decode(tensor) {
  const data = tensor.data;
  const nc = labels.length;
  const n = tensor.dims[2];          // 8400
  const boxes = [];
  for (let i = 0; i < n; i++) {
    let best = 0, bestK = 0;
    for (let k = 0; k < nc; k++) {
      const s = data[(4 + k) * n + i];
      if (s > best) { best = s; bestK = k; }
    }
    if (best < conf) continue;
    const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
    // 640 letterbox 좌표 → 원본 카메라 픽셀 좌표로 환원: (좌표 - 패딩) / 스케일
    const inv = 1 / lb.scale;
    const x1 = (cx - w / 2 - lb.padX) * inv, y1 = (cy - h / 2 - lb.padY) * inv;
    const x2 = (cx + w / 2 - lb.padX) * inv, y2 = (cy + h / 2 - lb.padY) * inv;
    boxes.push({ x1, y1, x2, y2, score: best, cls: bestK });
  }
  return nms(boxes, iou);
}

function nms(boxes, thr) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const a = boxes.shift();
    keep.push(a);
    for (let i = boxes.length - 1; i >= 0; i--) {
      if (boxes[i].cls === a.cls && iouOf(a, boxes[i]) > thr) boxes.splice(i, 1);
    }
  }
  return keep;
}

function iouOf(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua <= 0 ? 0 : inter / ua;
}

// guide.py 이식: 가장 위험한 객체 하나 선택
function decide(dets) {
  // 박스는 이제 원본 카메라 픽셀 좌표 → 면적·방향을 실제 프레임 크기로 정규화.
  const vw = video.videoWidth, vh = video.videoHeight;
  let best = null;
  for (const d of dets) {
    const area = (d.x2 - d.x1) * (d.y2 - d.y1) / (vw * vh);
    let prox = null;
    if (area >= nearR) prox = "near";
    else if (area >= midR) prox = "mid";
    else continue; // far → 안내 안 함
    const cxn = (d.x1 + d.x2) / 2 / vw;
    const dir = cxn < leftB ? "left" : cxn > rightB ? "right" : "front";
    const name = labels[d.cls];
    const score = (riskWeights[name] ?? 0.5) * PROX_FACTOR[prox];
    if (!best || score > best.score) best = { ...d, prox, dir, name, score };
  }
  return best;
}

function sentence(g) {
  const ko = labelsKo[g.name] ?? g.name;
  const dir = DIR_NAMES[g.dir];
  return g.prox === "near" ? `${dir}에 ${ko}, 가까이 있습니다` : `${dir}에 ${ko} 있습니다`;
}

function announce(g) {
  if (!("speechSynthesis" in window)) return;
  if (speechSynthesis.speaking) return;
  const key = g.dir + "|" + g.name;
  const now = Date.now() / 1000;
  if (now - (lastSpoke[key] || 0) < cooldown) return;
  lastSpoke[key] = now;
  const u = new SpeechSynthesisUtterance(sentence(g));
  u.lang = "ko-KR"; u.rate = 1.15;
  const ko = speechSynthesis.getVoices().find(v => (v.lang || "").toLowerCase().startsWith("ko"));
  if (ko) u.voice = ko;
  speechSynthesis.speak(u);
}

function draw(dets, g) {
  // 박스가 원본 프레임 좌표이므로, 표시 캔버스 크기(=영상 크기)로 직접 매핑.
  const sx = dcanvas.width / video.videoWidth, sy = dcanvas.height / video.videoHeight;
  dctx.drawImage(video, 0, 0, dcanvas.width, dcanvas.height);
  dctx.lineWidth = Math.max(2, dcanvas.width / 320);
  dctx.font = `${Math.max(14, dcanvas.width / 36)}px sans-serif`;
  for (const d of dets) {
    const name = labels[d.cls];
    dctx.strokeStyle = COLORS[name] || "#00ff5a";
    dctx.strokeRect(d.x1 * sx, d.y1 * sy, (d.x2 - d.x1) * sx, (d.y2 - d.y1) * sy);
    dctx.fillStyle = COLORS[name] || "#00ff5a";
    dctx.fillText(`${labelsKo[name] || name} ${d.score.toFixed(2)}`, d.x1 * sx, Math.max(d.y1 * sy - 4, 16));
  }
  if (g) { // 가장 위험한 것 빨강 강조 + 상단 배너
    dctx.strokeStyle = "#ff0000"; dctx.lineWidth = Math.max(4, dcanvas.width / 200);
    dctx.strokeRect(g.x1 * sx, g.y1 * sy, (g.x2 - g.x1) * sx, (g.y2 - g.y1) * sy);
    const text = sentence(g);
    dctx.font = `bold ${Math.max(20, dcanvas.width / 22)}px sans-serif`;
    const w = dctx.measureText(text).width;
    dctx.fillStyle = "rgba(0,0,0,.6)"; dctx.fillRect(0, 0, w + 24, 44);
    dctx.fillStyle = "#ff5555"; dctx.fillText(text, 12, 32);
  }
}

let fps = 0, prev = performance.now();
async function loop() {
  while (running) {
    if (video.readyState < 2) { await sleep(30); continue; }
    const t = preprocess();
    let out;
    try { out = await session.run({ [inName]: t }); }
    catch (e) { setStatus("추론 오류: " + e.message); break; }
    const dets = decode(out[outName]);
    const g = decide(dets);
    draw(dets, g);
    if (g) announce(g);
    const now = performance.now();
    fps = 0.9 * fps + 0.1 * (1000 / Math.max(now - prev, 1));
    prev = now;
    setStatus(`${fps.toFixed(1)} FPS · 탐지 ${dets.length}`);
    await sleep(0); // UI 양보
  }
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
