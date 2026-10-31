/**
 * Namma Space - Main 3D Web Application Controller (Hybrid AI Frontend + Backend)
 * ===============================================================================
 * Integrates:
 *   1. Live Guided Smartphone Scanner with 360° Direction Ring & Telemetry (`src/liveScanner.js`)
 *   2. Python AI Backend (`POST /api/reconstruct-scan` -> YOLOv8n + Depth + Multi-Room Stitcher)
 *   3. Clean Semantic Multi-Room & 3D Furniture Mesh Builder (`src/semantic3DBuilder.js`)
 *   4. First-Person WASD + Mobile Touch D-Pad Roaming with Sliding Wall/Obstacle Collision
 *   5. 3D Raycast POI Tagging, Instant Search, 8-Way A* Pathfinding & Live Camera AR Navigation
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

import { IndoorPathfinder } from './pathfinding.js';
import {
  DEFAULT_POIS,
  buildDefaultDigitalTwin,
  rebuildOccupancyFromCustomModel
} from './indoorScene.js';
import { ARNavigationController } from './arNavigation.js';
import { LiveGuidedScanner } from './liveScanner.js';
import { buildFinalized3DScene } from './semantic3DBuilder.js';

// --- 1. Initialize 3D Scene, Camera & Renderer ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x090d16);
scene.fog = new THREE.FogExp2(0x090d16, 0.014);

const camera = new THREE.PerspectiveCamera(
  65,
  window.innerWidth / window.innerHeight,
  0.1,
  160
);
const EYE_HEIGHT = 1.65;
camera.position.set(0, 8.0, 12.0);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

// --- 2. Lighting Setup ---
const ambientLight = new THREE.AmbientLight(0xdbeafe, 0.9);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.35);
dirLight.position.set(10, 18, 12);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -20;
dirLight.shadow.camera.right = 20;
dirLight.shadow.camera.top = 20;
dirLight.shadow.camera.bottom = -20;
scene.add(dirLight);

const pointLight1 = new THREE.PointLight(0x38bdf8, 0.8, 20);
pointLight1.position.set(-6, 2.7, -4);
scene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0xa855f7, 0.7, 20);
pointLight2.position.set(6, 2.7, 4);
scene.add(pointLight2);

// --- 3. Controls: Orbit (Overview) & PointerLock (First-Person WASD) ---
const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.06;
orbitControls.maxPolarAngle = Math.PI / 2 - 0.04;
orbitControls.target.set(0, 1.0, 0);

const fpControls = new PointerLockControls(camera, document.body);
let isFirstPerson = false;
let isTagMode = false;
let isAutoWalking = false;
let autoWalkProgress = 0;

const playerPos = new THREE.Vector3(0, EYE_HEIGHT, 7.6);

// Player Avatar Marker
const avatarGroup = new THREE.Group();
const avatarBody = new THREE.Mesh(
  new THREE.CylinderGeometry(0.26, 0.26, 1.4, 20),
  new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.4 })
);
avatarBody.position.y = 0.7;
const avatarRing = new THREE.Mesh(
  new THREE.RingGeometry(0.35, 0.48, 32),
  new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide })
);
avatarRing.rotation.x = -Math.PI / 2;
avatarRing.position.y = 0.03;
avatarGroup.add(avatarBody, avatarRing);
avatarGroup.position.set(playerPos.x, 0, playerPos.z);
scene.add(avatarGroup);

// --- 4. Build Default Digital Twin & A* Pathfinder (36m x 36m supports multi-room stitching!) ---
const pathfinder = new IndoorPathfinder(36, 36, 0.25);
let { rootGroup: currentModelGroup, interactableMeshes } = buildDefaultDigitalTwin(scene, pathfinder);

// --- 5. Spatial POI Management ---
const STORAGE_KEY = 'namma_space_pois_v1';
let pois = loadStoredPois();
let activePoi = null;
let currentPathPoints = [];
let pendingTagCoords = null;

function loadStoredPois() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // Fallback to default POIs
  }
  return structuredClone(DEFAULT_POIS);
}

function savePois() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(pois));
  } catch {
    // Ignore storage quota errors
  }
}

const poiMeshGroup = new THREE.Group();
scene.add(poiMeshGroup);
const labelsContainer = document.getElementById('poi-labels-container');
const poiDomMarkers = new Map();

function rebuildPoiVisuals() {
  while (poiMeshGroup.children.length > 0) {
    poiMeshGroup.remove(poiMeshGroup.children[0]);
  }
  labelsContainer.innerHTML = '';
  poiDomMarkers.clear();

  pois.forEach((poi) => {
    const isSelected = activePoi && activePoi.id === poi.id;
    const pinMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.2, 20, 20),
      new THREE.MeshStandardMaterial({
        color: isSelected ? 0x22c55e : 0x38bdf8,
        emissive: isSelected ? 0x16a34a : 0x0284c7,
        emissiveIntensity: 0.75
      })
    );
    pinMesh.position.set(poi.x, poi.y, poi.z);
    pinMesh.userData = { poiId: poi.id };
    poiMeshGroup.add(pinMesh);

    const stemGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(poi.x, 0.05, poi.z),
      new THREE.Vector3(poi.x, poi.y, poi.z)
    ]);
    const stemLine = new THREE.Line(
      stemGeo,
      new THREE.LineBasicMaterial({ color: isSelected ? 0x22c55e : 0x38bdf8 })
    );
    poiMeshGroup.add(stemLine);

    const badge = document.createElement('div');
    badge.className = `poi-marker-3d ${isSelected ? 'active-dest' : ''}`;
    badge.innerHTML = `<span>📍</span><span>${poi.name}</span>`;
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      navigateToPoi(poi.id);
    });
    labelsContainer.appendChild(badge);
    poiDomMarkers.set(poi.id, badge);
  });

  renderSidebarPoiList();
}

function updatePoiScreenLabels() {
  const halfW = window.innerWidth / 2;
  const halfH = window.innerHeight / 2;

  pois.forEach((poi) => {
    const badge = poiDomMarkers.get(poi.id);
    if (!badge) return;

    const worldPos = new THREE.Vector3(poi.x, poi.y + 0.28, poi.z);
    worldPos.project(camera);

    if (worldPos.z > 1) {
      badge.style.display = 'none';
      return;
    }

    const x = worldPos.x * halfW + halfW;
    const y = -(worldPos.y * halfH) + halfH;
    badge.style.display = 'flex';
    badge.style.left = `${x}px`;
    badge.style.top = `${y}px`;
  });
}

// --- 6. 3D Glowing Path Visualization (A* Floor Ribbon + Animated Cones) ---
const pathVisualGroup = new THREE.Group();
scene.add(pathVisualGroup);
let pathCurve = null;
const chevronMeshes = [];

function clearNavigationPath() {
  activePoi = null;
  currentPathPoints = [];
  pathCurve = null;
  isAutoWalking = false;
  chevronMeshes.length = 0;

  while (pathVisualGroup.children.length > 0) {
    pathVisualGroup.remove(pathVisualGroup.children[0]);
  }

  document.getElementById('nav-status-card').classList.add('hidden');
  rebuildPoiVisuals();
}

function navigateToPoi(poiId) {
  const target = pois.find((p) => p.id === poiId);
  if (!target) return;

  activePoi = target;
  isAutoWalking = false;

  const waypoints = pathfinder.findPath(playerPos.x, playerPos.z, target.x, target.z);
  if (waypoints.length < 2) {
    showToast('⚠️ Could not compute an obstacle-free path to this POI.');
    return;
  }

  currentPathPoints = waypoints;

  while (pathVisualGroup.children.length > 0) {
    pathVisualGroup.remove(pathVisualGroup.children[0]);
  }
  chevronMeshes.length = 0;

  const vecPoints = waypoints.map((pt) => new THREE.Vector3(pt.x, 0.08, pt.z));
  pathCurve = new THREE.CatmullRomCurve3(vecPoints, false, 'catmullrom', 0.25);

  const curveLength = pathCurve.getLength();
  const tubeGeo = new THREE.TubeGeometry(
    pathCurve,
    Math.max(32, Math.floor(curveLength * 8)),
    0.11,
    12,
    false
  );
  const tubeMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x16a34a,
    emissiveIntensity: 0.85,
    roughness: 0.25
  });
  const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
  pathVisualGroup.add(tubeMesh);

  const numChevrons = Math.max(4, Math.floor(curveLength / 1.4));
  const coneGeo = new THREE.ConeGeometry(0.18, 0.42, 12);
  coneGeo.rotateX(Math.PI / 2);
  const coneMat = new THREE.MeshBasicMaterial({ color: 0xbbf7d0 });

  for (let i = 0; i < numChevrons; i++) {
    const cone = new THREE.Mesh(coneGeo, coneMat);
    cone.userData.baseOffset = i / numChevrons;
    pathVisualGroup.add(cone);
    chevronMeshes.push(cone);
  }

  const navCard = document.getElementById('nav-status-card');
  navCard.classList.remove('hidden');
  document.getElementById('nav-target-name').textContent = target.name;
  document.getElementById('nav-distance').textContent = `${curveLength.toFixed(1)} m`;
  document.getElementById('nav-eta').textContent = `${Math.max(2, Math.round(curveLength / 1.35))} sec`;
  document.getElementById('nav-waypoints').textContent = `${waypoints.length} nodes`;
  document.getElementById('nav-turn-instruction').textContent =
    `Obstacle-free route calculated (${waypoints.length} waypoints). Press WASD or click "Auto-Walk Path".`;

  rebuildPoiVisuals();
  showToast(`🧭 Path calculated to "${target.name}" (${curveLength.toFixed(1)}m)`);
}

// --- 7. Camera Teleport (Jump Straight to Room/Item) ---
let cameraTransition = null;

function teleportToPoi(poiId) {
  const target = pois.find((p) => p.id === poiId);
  if (!target) return;

  const cell = pathfinder.worldToGrid(target.x, target.z + 1.2);
  const safeCell = pathfinder.findNearestWalkableCell(cell.col, cell.row);
  const safeWorld = pathfinder.gridToWorld(safeCell.col, safeCell.row);

  playerPos.set(safeWorld.x, EYE_HEIGHT, safeWorld.z);
  avatarGroup.position.set(playerPos.x, 0, playerPos.z);

  if (isFirstPerson) {
    camera.position.copy(playerPos);
    camera.lookAt(target.x, target.y, target.z);
  } else {
    cameraTransition = {
      startPos: camera.position.clone(),
      endPos: new THREE.Vector3(safeWorld.x, 4.2, safeWorld.z + 3.5),
      startTarget: orbitControls.target.clone(),
      endTarget: new THREE.Vector3(target.x, target.y, target.z),
      progress: 0
    };
  }

  navigateToPoi(poiId);
  showToast(`⚡ Teleported to ${target.name}`);
}

// --- 8. Apply Finalized AI Scene Graph from Backend (`pipeline/server.py`) ---
async function applyAiReconstructedSceneGraph(sceneGraph) {
  // Clear any old default navigation path first so no stale green line stretches outside the scan!
  clearNavigationPath();
  scene.remove(currentModelGroup);

  const built = await buildFinalized3DScene(sceneGraph, scene, pathfinder);
  currentModelGroup = built.rootGroup;
  interactableMeshes = built.interactableMeshes;
  const dims = built.roomDimensions;

  if (Array.isArray(sceneGraph.pois) && sceneGraph.pois.length > 0) {
    pois = sceneGraph.pois;
    savePois();
  }

  const cx = Number(((dims.min_x + dims.max_x) / 2).toFixed(2)) || 0;
  const cz = Number(((dims.min_z + dims.max_z) / 2).toFixed(2)) || -1.8;

  // Position player near the camera origin (0, 0), hide the tall blue cylinder so it doesn't block the 3D scan, and frame the mesh cleanly
  playerPos.set(0, EYE_HEIGHT, 0.35);
  avatarGroup.position.set(playerPos.x, 0, playerPos.z);
  avatarGroup.visible = false;
  camera.position.set(0, Math.max(1.85, dims.height_m * 0.88), 1.9);
  orbitControls.target.set(cx, Math.max(0.75, dims.height_m * 0.42), cz);

  const splatCount = sceneGraph.gaussian_splat_count || sceneGraph.gaussian_splats?.count || 0;
  const subtitleEl = document.querySelector('.subtitle');
  if (subtitleEl && dims) {
    subtitleEl.textContent =
      `✨ [3D GAUSSIAN SPLATTING] ${splatCount.toLocaleString()} Volumetric Splats • ${dims.width_m}m × ${dims.depth_m}m × ${dims.height_m}m (${dims.floor_area_m2} m²)`;
  }

  rebuildPoiVisuals();

  showToast(
    `✨ 3D Gaussian Splatting Complete! Reconstructed ${splatCount.toLocaleString()} volumetric 3D splats (${dims.width_m}m × ${dims.depth_m}m).`
  );
}

// --- 9. Live Guided Scanner Wiring (`src/liveScanner.js`) ---
const liveScanner = new LiveGuidedScanner({
  overlayEl: document.getElementById('live-scanner-overlay'),
  videoEl: document.getElementById('scanner-video'),
  hudCanvasEl: document.getElementById('scanner-hud-canvas'),
  instructionEl: document.getElementById('scanner-instruction'),
  subStatusEl: document.getElementById('scanner-sub-status'),
  sectorCountEl: document.getElementById('scanner-sector-count'),
  roomBadgeEl: document.getElementById('scanner-room-badge'),
  fallbackInputEl: document.getElementById('scanner-fallback-video'),
  onComplete: (sceneGraph) => {
    applyAiReconstructedSceneGraph(sceneGraph);
  }
});

document.getElementById('btn-live-scan').addEventListener('click', () => {
  liveScanner.start();
});

document.getElementById('btn-cancel-scanner').addEventListener('click', () => {
  liveScanner.stop();
});

document.getElementById('btn-scan-mode')?.addEventListener('click', (e) => {
  const r = liveScanner.cycleRange();
  e.currentTarget.textContent = `📏 RANGE: ${r.toFixed(1)} m`;
});

document.getElementById('btn-snap-manual').addEventListener('click', () => {
  liveScanner.snapCurrentFrame(false);
});

document.getElementById('btn-rotate-sim').addEventListener('click', () => {
  liveScanner.yawDeg = (liveScanner.yawDeg + 25) % 360;
  liveScanner.snapCurrentFrame(false);
});

document.getElementById('btn-mark-doorway').addEventListener('click', () => {
  liveScanner.markDoorwayTransition();
});

document.getElementById('btn-finish-scan').addEventListener('click', () => {
  liveScanner.finishAndSendToBackend();
});

document.getElementById('scanner-fallback-video').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) {
    liveScanner.processFallbackRecordedVideo(file);
  }
});

// --- 10. Uploaded Video -> AI Backend 3D Reconstruction Studio ---
const videoReconDialog = document.getElementById('video-recon-dialog');
const videoFileInput = document.getElementById('video-file-input');
const reconProgressBox = document.getElementById('video-recon-progress-box');
const reconPreviewImg = document.getElementById('recon-frame-preview');
const reconStageText = document.getElementById('recon-stage-text');
const reconProgressFill = document.getElementById('recon-progress-fill');
const reconMetricsSummary = document.getElementById('recon-metrics-summary');

document.getElementById('btn-video-to-3d').addEventListener('click', () => {
  videoReconDialog.showModal();
});

document.getElementById('btn-close-video-recon').addEventListener('click', () => {
  videoReconDialog.close();
});

videoFileInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  reconProgressBox.classList.remove('hidden');
  reconProgressFill.style.width = '5%';
  reconStageText.textContent = `Extracting dense multi-view frames from "${file.name}" for 3D Gaussian Splatting...`;

  try {
    const url = URL.createObjectURL(file);
    const vid = document.createElement('video');
    vid.src = url;
    vid.muted = true;
    vid.playsInline = true;

    await new Promise((res, rej) => {
      vid.onloadedmetadata = () => res();
      vid.onerror = () => rej(new Error('Could not decode video file'));
    });

    const duration = Math.min(vid.duration || 10, 60);
    // Extract up to 65 dense overlapping frames (~3.5 FPS) for true multi-view 3D Gaussian Splatting!
    const numSamples = Math.min(65, Math.max(18, Math.floor(duration * 3.5)));

    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = 512;
    snapCanvas.height = 384;
    const snapCtx = snapCanvas.getContext('2d');

    const frames = [];
    for (let i = 0; i < numSamples; i++) {
      vid.currentTime = ((i + 0.5) / numSamples) * duration;
      await new Promise((r) => {
        vid.onseeked = () => r();
      });

      snapCtx.drawImage(vid, 0, 0, 512, 384);
      const b64 = snapCanvas.toDataURL('image/jpeg', 0.80);

      frames.push({
        id: i,
        image_base64: b64,
        yaw_deg: 0.0,
        pitch_deg: 0.0,
        pos_x: 0.0,
        pos_z: 0.0,
        room_hint: 0,
        is_doorway: false,
        sharpness: 135.0
      });

      const pct = Math.round(5 + ((i + 1) / numSamples) * 20);
      reconProgressFill.style.width = `${pct}%`;
      reconPreviewImg.src = b64;
      reconStageText.textContent = `Extracted dense keyframe ${i + 1}/${numSamples}`;
    }

    URL.revokeObjectURL(url);

    const pollTimer = setInterval(async () => {
      try {
        const pr = await fetch('/api/progress');
        if (pr.ok) {
          const st = await pr.json();
          const scaledPct = Math.min(99, Math.round(25 + (st.percent || 0) * 0.74));
          reconProgressFill.style.width = `${scaledPct}%`;
          reconStageText.textContent = `🧠 ${st.stage || 'Processing'} (${st.percent || 0}%)`;
          reconMetricsSummary.textContent = st.detail || '';
        }
      } catch {
        // Ignore transient poll errors
      }
    }, 350);

    const resp = await fetch('/api/reconstruct-scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frames,
        range_m: 3.2,
        measure_angles_from_optical_flow: true
      })
    });

    clearInterval(pollTimer);

    const sceneGraph = await resp.json();
    const dims = sceneGraph.room_dimensions;
    const splats = sceneGraph.gaussian_splat_count || 0;
    reconProgressFill.style.width = '100%';
    reconStageText.textContent = `✅ Reconstructed ${splats.toLocaleString()} PyTorch-Optimized 3D Gaussian Splats!`;
    reconMetricsSummary.textContent =
      `Fused ${sceneGraph.fused_view_count} dense frames in ${sceneGraph.processing_time_sec}s | Measured: ${dims.width_m}m × ${dims.depth_m}m × ${dims.height_m}m`;

    await applyAiReconstructedSceneGraph(sceneGraph);
  } catch (err) {
    reconStageText.textContent = `❌ Error: ${err.message}`;
    showToast(`❌ Reconstruction error: ${err.message}`);
  }
});

// --- 11. Sidebar & Spatial Search UI ---
function renderSidebarPoiList() {
  const listEl = document.getElementById('poi-list');
  const countBadge = document.getElementById('poi-count-badge');
  countBadge.textContent = `${pois.length} POIs`;
  listEl.innerHTML = '';

  pois.forEach((poi) => {
    const isActive = activePoi && activePoi.id === poi.id;
    const card = document.createElement('div');
    card.className = `poi-card ${isActive ? 'active' : ''}`;
    card.innerHTML = `
      <div class="poi-info">
        <h4>${poi.name}</h4>
        <p>${poi.category} • (${poi.x.toFixed(1)}, ${poi.z.toFixed(1)})</p>
      </div>
      <div class="poi-actions">
        <button class="icon-btn btn-route" title="Compute Walking Path">🧭 Route</button>
        <button class="icon-btn btn-jump" title="Jump Straight Here">⚡ Jump</button>
      </div>
    `;
    card.querySelector('.btn-route').addEventListener('click', () => navigateToPoi(poi.id));
    card.querySelector('.btn-jump').addEventListener('click', () => teleportToPoi(poi.id));
    listEl.appendChild(card);
  });
}

// Mobile Sidebar Drawer Toggle
const sidebarEl = document.getElementById('hud-sidebar');
if (window.innerWidth <= 768) {
  sidebarEl.classList.add('collapsed-mobile');
}
document.getElementById('btn-toggle-sidebar').addEventListener('click', () => {
  sidebarEl.classList.toggle('collapsed-mobile');
});

const searchInput = document.getElementById('poi-search-input');
const searchDropdown = document.getElementById('search-results-dropdown');

searchInput.addEventListener('input', () => {
  const q = searchInput.value.trim().toLowerCase();
  if (!q) {
    searchDropdown.classList.add('hidden');
    return;
  }

  const matches = pois.filter(
    (p) =>
      p.name.toLowerCase().includes(q) ||
      p.category.toLowerCase().includes(q) ||
      (p.description && p.description.toLowerCase().includes(q))
  );

  searchDropdown.innerHTML = '';
  if (matches.length === 0) {
    searchDropdown.innerHTML = `<div class="search-item"><span class="search-item-meta">No matching rooms or items found</span></div>`;
  } else {
    matches.forEach((m) => {
      const row = document.createElement('div');
      row.className = 'search-item';
      row.innerHTML = `
        <div>
          <div class="search-item-title">📍 ${m.name}</div>
          <div class="search-item-meta">${m.category} — ${m.description || ''}</div>
        </div>
        <span class="badge">Jump &amp; Route ↵</span>
      `;
      row.addEventListener('click', () => {
        searchDropdown.classList.add('hidden');
        searchInput.value = m.name;
        teleportToPoi(m.id);
      });
      searchDropdown.appendChild(row);
    });
  }
  searchDropdown.classList.remove('hidden');
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrapper')) {
    searchDropdown.classList.add('hidden');
  }
});

// --- 12. Raycasting for Interactive 3D POI Tagging ---
const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();
const poiDialog = document.getElementById('poi-dialog');

renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!isTagMode || e.button !== 0) return;

  const rect = renderer.domElement.getBoundingClientRect();
  mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObjects(interactableMeshes, true);

  if (hits.length > 0) {
    const pt = hits[0].point;
    pendingTagCoords = {
      x: Number(pt.x.toFixed(2)),
      y: Number(Math.max(1.1, pt.y + 0.5).toFixed(2)),
      z: Number(pt.z.toFixed(2))
    };
    document.getElementById('poi-coords-preview').textContent =
      `3D Coordinates: (${pendingTagCoords.x}, ${pendingTagCoords.y}, ${pendingTagCoords.z})`;
    document.getElementById('poi-name-input').value = '';
    document.getElementById('poi-desc-input').value = '';
    poiDialog.showModal();
  }
});

document.getElementById('poi-form').addEventListener('submit', (e) => {
  e.preventDefault();
  if (!pendingTagCoords) return;

  const name = document.getElementById('poi-name-input').value.trim();
  const category = document.getElementById('poi-category-input').value;
  const description = document.getElementById('poi-desc-input').value.trim();

  if (!name) return;

  const newPoi = {
    id: `poi-${Date.now()}`,
    name,
    category,
    description: description || 'User-tagged spatial point of interest',
    ...pendingTagCoords
  };

  pois.push(newPoi);
  savePois();
  poiDialog.close();
  toggleTagMode(false);
  rebuildPoiVisuals();
  navigateToPoi(newPoi.id);
  showToast(`✅ Tagged new POI: "${name}"`);
});

document.getElementById('btn-cancel-poi').addEventListener('click', () => {
  poiDialog.close();
});

function toggleTagMode(forceState) {
  isTagMode = typeof forceState === 'boolean' ? forceState : !isTagMode;
  const btn = document.getElementById('btn-tag-mode');
  btn.classList.toggle('active', isTagMode);
  renderer.domElement.style.cursor = isTagMode ? 'crosshair' : 'default';
  if (isTagMode) {
    showToast('📍 Tag Mode ON: Click any room, wall, or furniture in the 3D model to label it');
  }
}

document.getElementById('btn-tag-mode').addEventListener('click', () => toggleTagMode());

// --- 13. First-Person WASD + Mobile Touch D-Pad Movement ---
const moveKeys = { forward: false, backward: false, left: false, right: false, sprint: false };

document.querySelectorAll('.dpad-btn').forEach((btn) => {
  const dir = btn.dataset.dir;
  const startMove = (e) => {
    e.preventDefault();
    moveKeys[dir] = true;
  };
  const stopMove = (e) => {
    e.preventDefault();
    moveKeys[dir] = false;
  };
  btn.addEventListener('pointerdown', startMove);
  btn.addEventListener('pointerup', stopMove);
  btn.addEventListener('pointerleave', stopMove);
  btn.addEventListener('pointercancel', stopMove);
});

document.getElementById('btn-first-person').addEventListener('click', () => {
  if (isFirstPerson) {
    fpControls.unlock();
  } else {
    camera.position.copy(playerPos);
    if (typeof document.body.requestPointerLock === 'function') {
      fpControls.lock();
    } else {
      isFirstPerson = true;
      avatarGroup.visible = false;
      showToast('🎮 First-Person View Active (Use Touch D-Pad to Walk)');
    }
  }
});

fpControls.addEventListener('lock', () => {
  isFirstPerson = true;
  orbitControls.enabled = false;
  avatarGroup.visible = false;
  document.getElementById('crosshair').classList.remove('hidden');
  document.getElementById('btn-first-person').textContent = '🔓 Exit Walk (ESC)';
});

fpControls.addEventListener('unlock', () => {
  isFirstPerson = false;
  orbitControls.enabled = true;
  avatarGroup.visible = true;
  document.getElementById('crosshair').classList.add('hidden');
  document.getElementById('btn-first-person').textContent = '🎮 Walk (WASD)';
  camera.position.set(playerPos.x, playerPos.y + 4.5, playerPos.z + 5.5);
  orbitControls.target.set(playerPos.x, 1.0, playerPos.z);
});

window.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) {
    if (e.key === 'Escape') document.activeElement.blur();
    return;
  }

  switch (e.code) {
    case 'KeyW':
    case 'ArrowUp':
      moveKeys.forward = true;
      break;
    case 'KeyS':
    case 'ArrowDown':
      moveKeys.backward = true;
      break;
    case 'KeyA':
    case 'ArrowLeft':
      moveKeys.left = true;
      break;
    case 'KeyD':
    case 'ArrowRight':
      moveKeys.right = true;
      break;
    case 'ShiftLeft':
    case 'ShiftRight':
      moveKeys.sprint = true;
      break;
    case 'Slash':
      e.preventDefault();
      searchInput.focus();
      break;
    case 'KeyT':
      toggleTagMode();
      break;
  }
});

window.addEventListener('keyup', (e) => {
  switch (e.code) {
    case 'KeyW':
    case 'ArrowUp':
      moveKeys.forward = false;
      break;
    case 'KeyS':
    case 'ArrowDown':
      moveKeys.backward = false;
      break;
    case 'KeyA':
    case 'ArrowLeft':
      moveKeys.left = false;
      break;
    case 'KeyD':
    case 'ArrowRight':
      moveKeys.right = false;
      break;
    case 'ShiftLeft':
    case 'ShiftRight':
      moveKeys.sprint = false;
      break;
  }
});

document.getElementById('btn-auto-walk').addEventListener('click', () => {
  if (!pathCurve) return;
  isAutoWalking = !isAutoWalking;
  autoWalkProgress = 0;
  document.getElementById('btn-auto-walk').textContent = isAutoWalking
    ? '⏸️ Pause Walk'
    : '🚶 Auto-Walk Path';
});

document.getElementById('btn-teleport-dest').addEventListener('click', () => {
  if (activePoi) teleportToPoi(activePoi.id);
});

document.getElementById('btn-clear-nav').addEventListener('click', () => {
  clearNavigationPath();
});

// --- 14. Custom Scaniverse / Polycam .GLB / .PLY Scan Loader & Restore Default ---
const uploadDialog = document.getElementById('upload-dialog');
document.getElementById('btn-load-scaniverse')?.addEventListener('click', () => {
  uploadDialog.showModal();
});
document.getElementById('btn-open-load-glb').addEventListener('click', () => {
  videoReconDialog.close();
  uploadDialog.showModal();
});
document.getElementById('btn-close-upload').addEventListener('click', () => uploadDialog.close());

function restoreDefaultDigitalTwin() {
  scene.remove(currentModelGroup);
  const built = buildDefaultDigitalTwin(scene, pathfinder);
  currentModelGroup = built.rootGroup;
  interactableMeshes = built.interactableMeshes;
  pois = structuredClone(DEFAULT_POIS);
  savePois();
  playerPos.set(0, EYE_HEIGHT, 7.6);
  rebuildPoiVisuals();
  navigateToPoi('poi-meeting-4');
  showToast('🔄 Restored default Namma Space Digital Twin');
}

document.getElementById('btn-restore-default-from-vid').addEventListener('click', () => {
  restoreDefaultDigitalTwin();
  videoReconDialog.close();
});

document.getElementById('btn-reset-demo-scene').addEventListener('click', () => {
  restoreDefaultDigitalTwin();
  uploadDialog.close();
});

document.getElementById('model-file-input').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  const ext = file.name.split('.').pop().toLowerCase();

  if (ext === 'glb' || ext === 'gltf') {
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
      scene.remove(currentModelGroup);
      currentModelGroup = gltf.scene;

      // Auto-center X/Z and align lowest floor vertex to Y = 0 (Scaniverse / Polycam standard)
      const box = new THREE.Box3().setFromObject(currentModelGroup);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      currentModelGroup.position.set(-center.x, -box.min.y, -center.z);
      currentModelGroup.updateMatrixWorld(true);

      interactableMeshes = [];
      currentModelGroup.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          interactableMeshes.push(child);
        }
      });

      scene.add(currentModelGroup);
      rebuildOccupancyFromCustomModel(currentModelGroup, pathfinder);

      const w = Number(size.x.toFixed(2));
      const d = Number(size.z.toFixed(2));
      const h = Number(size.y.toFixed(2));
      const area = Number((w * d).toFixed(1));

      pois = [
        {
          id: 'poi-scaniverse-center',
          name: `Scaniverse Center (${w}m × ${d}m)`,
          category: 'Room',
          description: `Imported 3D Mesh "${file.name}" (${area} m²)`,
          x: 0,
          y: 1.2,
          z: 0
        }
      ];
      savePois();
      rebuildPoiVisuals();

      playerPos.set(0, EYE_HEIGHT, Math.min(4.0, d * 0.35));
      camera.position.set(0, Math.max(4.5, h * 1.4), Math.max(5.5, d * 0.75));
      orbitControls.target.set(0, 1.0, 0);

      const subtitleEl = document.querySelector('.subtitle');
      if (subtitleEl) {
        subtitleEl.textContent = `📦 [SCANIVERSE MESH] Measured: ${w}m × ${d}m × ${h}m (${area} m²) • Click "📍 Tag POI" to label rooms & items`;
      }

      uploadDialog.close();
      showToast(`✅ Loaded Scaniverse 3D Mesh "${file.name}" (${w}m × ${d}m)! Click "📍 Tag POI" to add markers.`);
    });
  } else if (ext === 'ply') {
    const loader = new PLYLoader();
    loader.load(url, (geometry) => {
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      geometry.translate(-center.x, -box.min.y, -center.z);

      scene.remove(currentModelGroup);
      const mat = new THREE.PointsMaterial({
        size: 0.045,
        vertexColors: geometry.hasAttribute('color')
      });
      const points = new THREE.Points(geometry, mat);
      currentModelGroup = new THREE.Group();
      currentModelGroup.add(points);

      // Add an invisible raycast floor plane so user can still click to Tag POIs on Point Clouds!
      const hitFloor = new THREE.Mesh(
        new THREE.PlaneGeometry(size.x * 1.2, size.z * 1.2),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      hitFloor.rotation.x = -Math.PI / 2;
      currentModelGroup.add(hitFloor);
      interactableMeshes = [hitFloor];

      scene.add(currentModelGroup);
      uploadDialog.close();
      showToast(`✅ Loaded Scaniverse 3D Point Cloud "${file.name}" (${size.x.toFixed(1)}m × ${size.z.toFixed(1)}m)!`);
    });
  }
});

// --- 15. SOP Modal & Native Dialog Fallback ---
const sopDialog = document.getElementById('sop-dialog');
document.getElementById('btn-sop-modal').addEventListener('click', () => sopDialog.showModal());
document.getElementById('btn-close-sop').addEventListener('click', () => sopDialog.close());

document.getElementById('btn-export-pois').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(pois, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'namma_space_pois.json';
  a.click();
});

document.querySelectorAll('dialog.modal-dialog').forEach((dialog) => {
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const isDialogContent =
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width;
      if (isDialogContent) return;
      dialog.close();
    });
  }
});

// --- 16. Mobile Camera AR Navigation (Bonus Deliverable) ---
const arController = new ARNavigationController({
  overlayEl: document.getElementById('ar-overlay'),
  videoEl: document.getElementById('ar-camera-video'),
  canvasEl: document.getElementById('ar-hud-canvas'),
  directionEl: document.getElementById('ar-direction-banner'),
  subInfoEl: document.getElementById('ar-sub-info')
});

document.getElementById('btn-ar-mode').addEventListener('click', () => {
  if (!activePoi && pois.length > 0) {
    navigateToPoi(pois[0].id);
  }
  arController.start(() => {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const playerYaw = Math.atan2(-forward.x, -forward.z);
    return {
      playerPos,
      playerYaw,
      path: currentPathPoints,
      activePoi
    };
  });
});

document.getElementById('btn-exit-ar').addEventListener('click', () => {
  arController.stop();
});

// --- 17. 2D Top-Down Occupancy & Pathfinding Radar Minimap ---
const minimapCanvas = document.getElementById('minimap-canvas');
const minimapCtx = minimapCanvas.getContext('2d');
const coordsLabel = document.getElementById('player-coords');

function drawMinimap() {
  const cw = minimapCanvas.width;
  const ch = minimapCanvas.height;
  minimapCtx.clearRect(0, 0, cw, ch);

  const scaleX = cw / pathfinder.widthMeters;
  const scaleZ = ch / pathfinder.depthMeters;

  const toCanvasX = (wx) => (wx + pathfinder.widthMeters / 2) * scaleX;
  const toCanvasY = (wz) => (wz + pathfinder.depthMeters / 2) * scaleZ;

  minimapCtx.fillStyle = '#334155';
  for (const obs of pathfinder.obstacles) {
    const x = toCanvasX(obs.minX);
    const y = toCanvasY(obs.minZ);
    const w = (obs.maxX - obs.minX) * scaleX;
    const h = (obs.maxZ - obs.minZ) * scaleZ;
    minimapCtx.fillRect(x, y, Math.max(2, w), Math.max(2, h));
  }

  if (currentPathPoints.length > 1) {
    minimapCtx.strokeStyle = '#22c55e';
    minimapCtx.lineWidth = 3;
    minimapCtx.beginPath();
    currentPathPoints.forEach((pt, idx) => {
      const px = toCanvasX(pt.x);
      const py = toCanvasY(pt.z);
      if (idx === 0) minimapCtx.moveTo(px, py);
      else minimapCtx.lineTo(px, py);
    });
    minimapCtx.stroke();
  }

  pois.forEach((poi) => {
    const px = toCanvasX(poi.x);
    const py = toCanvasY(poi.z);
    minimapCtx.fillStyle = activePoi && activePoi.id === poi.id ? '#22c55e' : '#f59e0b';
    minimapCtx.beginPath();
    minimapCtx.arc(px, py, 4.5, 0, Math.PI * 2);
    minimapCtx.fill();
  });

  const userX = toCanvasX(playerPos.x);
  const userY = toCanvasY(playerPos.z);

  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  const angle = Math.atan2(forward.z, forward.x);

  minimapCtx.fillStyle = 'rgba(56, 189, 248, 0.25)';
  minimapCtx.beginPath();
  minimapCtx.moveTo(userX, userY);
  minimapCtx.arc(userX, userY, 24, angle - 0.5, angle + 0.5);
  minimapCtx.closePath();
  minimapCtx.fill();

  minimapCtx.fillStyle = '#38bdf8';
  minimapCtx.beginPath();
  minimapCtx.arc(userX, userY, 5, 0, Math.PI * 2);
  minimapCtx.fill();
  minimapCtx.strokeStyle = '#ffffff';
  minimapCtx.lineWidth = 1.5;
  minimapCtx.stroke();

  coordsLabel.textContent = `X: ${playerPos.x.toFixed(1)}m, Z: ${playerPos.z.toFixed(1)}m`;
}

// --- 18. Toast Helper ---
let toastTimer = null;
function showToast(msg) {
  const banner = document.getElementById('toast-banner');
  banner.textContent = msg;
  banner.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => banner.classList.add('hidden'), 3800);
}

// --- 19. Main Animation & Physics Loop ---
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const elapsed = clock.getElapsedTime();

  const speed = (moveKeys.sprint ? 5.6 : 3.2) * dt;
  const isMoving =
    moveKeys.forward || moveKeys.backward || moveKeys.left || moveKeys.right;

  if (isMoving) {
    isAutoWalking = false;
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();

    const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

    const moveVec = new THREE.Vector3();
    if (moveKeys.forward) moveVec.add(forward);
    if (moveKeys.backward) moveVec.sub(forward);
    if (moveKeys.right) moveVec.add(right);
    if (moveKeys.left) moveVec.sub(right);
    moveVec.normalize().multiplyScalar(speed);

    const nextX = playerPos.x + moveVec.x;
    const nextZ = playerPos.z + moveVec.z;

    if (pathfinder.isWorldPointWalkable(nextX, nextZ)) {
      playerPos.x = nextX;
      playerPos.z = nextZ;
    } else if (pathfinder.isWorldPointWalkable(nextX, playerPos.z)) {
      playerPos.x = nextX;
    } else if (pathfinder.isWorldPointWalkable(playerPos.x, nextZ)) {
      playerPos.z = nextZ;
    }

    if (isFirstPerson) {
      camera.position.set(playerPos.x, EYE_HEIGHT, playerPos.z);
    } else {
      camera.position.x += moveVec.x;
      camera.position.z += moveVec.z;
      orbitControls.target.set(playerPos.x, 1.0, playerPos.z);
    }
  }

  if (isAutoWalking && pathCurve) {
    const pathLen = pathCurve.getLength();
    autoWalkProgress += (2.8 * dt) / Math.max(1, pathLen);

    if (autoWalkProgress >= 1) {
      autoWalkProgress = 1;
      isAutoWalking = false;
      document.getElementById('btn-auto-walk').textContent = '🚶 Auto-Walk Path';
      showToast(`🎉 Reached destination: ${activePoi?.name || 'POI'}`);
    }

    const pt = pathCurve.getPointAt(autoWalkProgress);
    const lookPt = pathCurve.getPointAt(Math.min(1, autoWalkProgress + 0.06));
    playerPos.set(pt.x, EYE_HEIGHT, pt.z);

    if (isFirstPerson) {
      camera.position.copy(playerPos);
      camera.lookAt(lookPt.x, EYE_HEIGHT, lookPt.z);
    } else {
      orbitControls.target.set(pt.x, 1.0, pt.z);
    }
  }

  if (cameraTransition) {
    cameraTransition.progress += dt * 2.2;
    const t = Math.min(1, cameraTransition.progress);
    const ease = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(cameraTransition.startPos, cameraTransition.endPos, ease);
    orbitControls.target.lerpVectors(cameraTransition.startTarget, cameraTransition.endTarget, ease);
    if (t >= 1) cameraTransition = null;
  }

  if (pathCurve && chevronMeshes.length > 0) {
    chevronMeshes.forEach((cone) => {
      const u = (cone.userData.baseOffset + elapsed * 0.18) % 1;
      const pos = pathCurve.getPointAt(u);
      const tangent = pathCurve.getTangentAt(u);
      cone.position.set(pos.x, 0.16, pos.z);
      cone.lookAt(pos.x + tangent.x, 0.16, pos.z + tangent.z);
    });
  }

  avatarGroup.position.set(playerPos.x, 0, playerPos.z);
  if (!isFirstPerson) orbitControls.update();

  updatePoiScreenLabels();
  drawMinimap();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

rebuildPoiVisuals();
navigateToPoi('poi-meeting-4');
animate();
