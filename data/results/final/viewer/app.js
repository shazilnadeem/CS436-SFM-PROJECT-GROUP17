// data/results/final/viewer/app.js
import * as THREE from "./libs/three.module.js";
import { PLYLoader } from "./libs/PLYLoader.js";
import { OrbitControls } from "./libs/OrbitControls.js";

const canvas = document.getElementById("three-canvas");
const imgA = document.getElementById("imgA");
const imgB = document.getElementById("imgB");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");

// ---- image config: we use JPGs in data/images/images_jpg ----
const IMAGE_BASE = "../../../images/images_jpg/";
const IMAGE_EXT = ".jpg";

function resolveImageFileName(name) {
  // Strip any existing extension and force .jpg
  const dot = name.lastIndexOf(".");
  const base = dot === -1 ? name : name.slice(0, dot);
  return base + IMAGE_EXT;
}

// -------------------------------------------------------------

let scene, renderer, camera;
let corridorGroup;
let cameraNodes = [];
let cameraPath = [];
let currentIndex = 0;
let targetIndex = 0;
let isAnimating = false;
let animStartTime = 0;
let controls;

const animDuration = 1000; // ms

const startPos = new THREE.Vector3();
const endPos = new THREE.Vector3();
const startQuat = new THREE.Quaternion();
const endQuat = new THREE.Quaternion();

const CAMERA_BACK_OFFSET = 2.0; // how far we stand behind each pose

init();

async function init() {
  // renderer + camera + scene
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101010);

  camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.01,
    1000
  );
  

  corridorGroup = new THREE.Group();
  scene.add(corridorGroup);

  addLights();

  await loadPointCloud();
  await loadCameras();

  // Optional: if path direction feels backwards, uncomment:
  // cameraNodes.reverse();

  // orbit controls for "stand here and look around"
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.rotateSpeed = 0.7;

  cameraPath = cameraNodes.map((_, i) => i);
  setCameraToNode(cameraNodes[currentIndex]);
  showImageForIndex(currentIndex, true);

  // Set initial controls target straight ahead
  updateControlsTarget();

  window.addEventListener("resize", onResize);
  prevBtn.addEventListener("click", () => triggerStep(-1));
  nextBtn.addEventListener("click", () => triggerStep(1));

  animate();
}

function addLights() {
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(5, 10, 7);
  scene.add(dir);
}

async function loadPointCloud() {
  const loader = new PLYLoader();
  return new Promise((resolve, reject) => {
    loader.load(
      "../dense_corridor.ply", // data/results/final/dense_corridor.ply
      geometry => {
        geometry.computeVertexNormals();
        const material = new THREE.PointsMaterial({
          size: 0.02,
          sizeAttenuation: true,
          vertexColors: geometry.hasAttribute("color"),
        });
        const points = new THREE.Points(geometry, material);
        corridorGroup.add(points);
        resolve();
      },
      undefined,
      err => reject(err)
    );
  });
}

async function loadCameras() {
  const resp = await fetch("../cameras_corridor.json"); // data/results/final/cameras_corridor.json
  const data = await resp.json();

  data.cameras.forEach(cam => {
    const node = new THREE.Object3D();

    const T = cam.matrix4x4.flat();
    const mat = new THREE.Matrix4();
    mat.fromArray(T);

    // Metashape: world->camera, Three.js needs camera->world
    mat.invert();

    // Fix orientation: flip 180° around X so floor is down, ceiling up
    const fix = new THREE.Matrix4().makeRotationX(Math.PI);
    mat.multiply(fix);

    node.applyMatrix4(mat);
    corridorGroup.add(node);
    cameraNodes.push({ node, imageName: cam.image_name });
  });

  // preload first image
  const firstFile = resolveImageFileName(cameraNodes[0].imageName);
  imgA.src = IMAGE_BASE + firstFile;
}

function setCameraToNode(camEntry) {
  const worldPos = new THREE.Vector3();
  const worldQuat = new THREE.Quaternion();

  camEntry.node.getWorldPosition(worldPos);
  camEntry.node.getWorldQuaternion(worldQuat);

  // Move camera a bit *behind* the pose along its forward direction
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(worldQuat);
  const viewPos = worldPos.clone().addScaledVector(forward, -CAMERA_BACK_OFFSET);

  camera.position.copy(viewPos);
  camera.quaternion.copy(worldQuat);
}

function updateControlsTarget() {
  if (!controls) return;
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).add(forward);
}

function triggerStep(direction) {
  if (isAnimating) return;
  const newIndex =
    (currentIndex + direction + cameraNodes.length) % cameraNodes.length;
  startTransitionTo(newIndex);
}

function startTransitionTo(newIndex) {
  if (newIndex === currentIndex) return;
  targetIndex = newIndex;

  // disable mouse-look during animation
  if (controls) controls.enabled = false;

  const fromNode = cameraNodes[currentIndex].node;
  const toNode = cameraNodes[targetIndex].node;

  const fromPos = new THREE.Vector3();
  const toPos = new THREE.Vector3();
  const fromQuat = new THREE.Quaternion();
  const toQuat = new THREE.Quaternion();

  fromNode.getWorldPosition(fromPos);
  fromNode.getWorldQuaternion(fromQuat);
  toNode.getWorldPosition(toPos);
  toNode.getWorldQuaternion(toQuat);

  // Apply the same "step back" offset to both endpoints
  const fromForward = new THREE.Vector3(0, 0, -1).applyQuaternion(fromQuat);
  const toForward = new THREE.Vector3(0, 0, -1).applyQuaternion(toQuat);

  startPos.copy(fromPos).addScaledVector(fromForward, -CAMERA_BACK_OFFSET);
  endPos.copy(toPos).addScaledVector(toForward, -CAMERA_BACK_OFFSET);
  startQuat.copy(fromQuat);
  endQuat.copy(toQuat);

  const nextFile = resolveImageFileName(
    cameraNodes[targetIndex].imageName
  );
  imgB.src = IMAGE_BASE + nextFile;

  imgB.style.opacity = "0";
  void imgB.offsetWidth; // force reflow
  imgA.style.opacity = "1";
  imgB.style.opacity = "1";

  animStartTime = performance.now();
  isAnimating = true;
}

function updateAnimation(now) {
  const t = (now - animStartTime) / animDuration;
  if (t >= 1) {
    isAnimating = false;
    currentIndex = targetIndex;
    setCameraToNode(cameraNodes[currentIndex]);
    showImageForIndex(currentIndex, false);

    // re-enable mouse-look at the new pose
    if (controls) {
      updateControlsTarget();
      controls.enabled = true;
    }

    return;
  }

  const alpha = t;
  camera.position.lerpVectors(startPos, endPos, alpha);
  THREE.Quaternion.slerp(startQuat, endQuat, camera.quaternion, alpha);
}

function showImageForIndex(idx, prepare) {
  const file = resolveImageFileName(cameraNodes[idx].imageName);
  const src = IMAGE_BASE + file;

  imgA.src = src;
  if (prepare) {
    imgA.style.opacity = "1";
    imgB.style.opacity = "0";
  } else {
    imgA.style.opacity = "1";
    imgB.style.opacity = "0";
  }
}

function animate(now) {
  requestAnimationFrame(animate);
  if (isAnimating) updateAnimation(now);
  if (controls) controls.update();
  renderer.render(scene, camera);
}

function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
