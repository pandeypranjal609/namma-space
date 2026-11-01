/**
 * Namma Space - 3D Indoor Digital Twin Builder & Custom Scan Loader (.GLB / .PLY)
 * Builds a complete multi-room corporate/innovation center digital twin out-of-the-box
 * and automatically registers all physical walls & furniture into the A* Occupancy Grid.
 */

import * as THREE from 'three';

export const DEFAULT_POIS = [
  {
    id: 'poi-reception',
    name: 'Reception & Welcome Desk',
    category: 'Facility',
    description: 'Main lobby check-in desk and visitor badge kiosk',
    x: 0.0,
    y: 1.35,
    z: 6.2
  },
  {
    id: 'poi-meeting-4',
    name: 'Meeting Room 4 (Boardroom)',
    category: 'Room',
    description: '10-seat glass conference room with 4K display',
    x: -7.8,
    y: 1.4,
    z: -4.8
  },
  {
    id: 'poi-sofas',
    name: '3-Seater Sofas (Lounge Zone)',
    category: 'Furniture',
    description: 'Collaborative lounge area with plush 3-seater sofas & coffee table',
    x: -7.5,
    y: 1.2,
    z: 4.5
  },
  {
    id: 'poi-workstations',
    name: 'Engineering Workstations Pod',
    category: 'Equipment',
    description: 'Open-plan dual-monitor developer desks',
    x: 7.4,
    y: 1.3,
    z: -4.5
  },
  {
    id: 'poi-cafeteria',
    name: 'Espresso Bar & Pantry',
    category: 'Facility',
    description: 'Refreshments counter, coffee machine, and high-top seating',
    x: 7.8,
    y: 1.35,
    z: 4.8
  },
  {
    id: 'poi-innovation-hub',
    name: 'Central Innovation Hub',
    category: 'Room',
    description: 'Central collaboration rotunda and demo stage',
    x: 0.0,
    y: 1.4,
    z: -0.5
  }
];

/**
 * Generates procedural canvas textures for realistic wood planks and floor tiles
 */
function createFloorTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, 512, 512);

  // Subtle architectural grid tiles
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 2;
  const step = 64;
  for (let i = 0; i <= 512; i += step) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(12, 9);
  return tex;
}

/**
 * Builds the default multi-room 3D Digital Twin (24m x 18m floor plan)
 * and populates the pathfinder occupancy grid with all walls & furniture.
 */
export function buildDefaultDigitalTwin(scene, pathfinder) {
  pathfinder.clearObstacles();

  const rootGroup = new THREE.Group();
  rootGroup.name = 'DigitalTwinRoot';
  const interactableMeshes = [];

  const W = 24; // X: -12 to +12
  const D = 18; // Z: -9 to +9
  const wallH = 3.0;

  // Materials
  const floorMat = new THREE.MeshStandardMaterial({
    map: createFloorTexture(),
    roughness: 0.75,
    metalness: 0.1
  });
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xe2e8f0,
    roughness: 0.65
  });
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.24,
    roughness: 0.1,
    transmission: 0.6
  });
  const woodMat = new THREE.MeshStandardMaterial({
    color: 0x854d0e,
    roughness: 0.55
  });
  const sofaMat = new THREE.MeshStandardMaterial({
    color: 0x0284c7,
    roughness: 0.8
  });
  const darkMetalMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    metalness: 0.6,
    roughness: 0.35
  });
  const accentMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    roughness: 0.5
  });

  // 1. Main Walkable Floor
  const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(W, D), floorMat);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.receiveShadow = true;
  floorMesh.name = 'WalkableFloor';
  rootGroup.add(floorMesh);
  interactableMeshes.push(floorMesh);

  // Helper to add a box obstacle both to Three.js and the A* Occupancy Grid
  function addBlock(cx, cy, cz, sx, sy, sz, material, label = 'Wall', inflate = 0.35) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), material);
    mesh.position.set(cx, cy, cz);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.name = label;
    rootGroup.add(mesh);
    interactableMeshes.push(mesh);

    pathfinder.addObstacleBox(
      cx - sx / 2,
      cz - sz / 2,
      cx + sx / 2,
      cz + sz / 2,
      inflate,
      label
    );
    return mesh;
  }

  // 2. Outer Perimeter Walls
  const t = 0.25;
  addBlock(0, wallH / 2, -D / 2, W, wallH, t, wallMat, 'North Perimeter Wall');
  addBlock(0, wallH / 2, D / 2, W, wallH, t, wallMat, 'South Perimeter Wall');
  addBlock(-W / 2, wallH / 2, 0, t, wallH, D, wallMat, 'West Perimeter Wall');
  addBlock(W / 2, wallH / 2, 0, t, wallH, D, wallMat, 'East Perimeter Wall');

  // 3. Interior Room Partitions (with wide walkable doorways)
  // Top-Left: Meeting Room 4 (-12 to -3.5, Z: -9 to -1.5)
  addBlock(-3.5, wallH / 2, -6.5, t, wallH, 5.0, wallMat, 'Meeting Room East Wall');
  addBlock(-9.5, wallH / 2, -1.5, 5.0, wallH, t, glassMat, 'Meeting Room Glass Partition');
  // Doorway gap at X: -6.5 to -3.5, Z: -1.5

  // Top-Right: Engineering Workstations Pod (X: 3.5 to 12, Z: -9 to -1.5)
  addBlock(3.5, wallH / 2, -6.5, t, wallH, 5.0, wallMat, 'Engineering West Wall');
  addBlock(9.5, wallH / 2, -1.5, 5.0, wallH, t, glassMat, 'Engineering Glass Partition');

  // Bottom-Left: Lounge Zone divider
  addBlock(-3.5, wallH / 2, 6.2, t, wallH, 5.5, wallMat, 'Lounge Divider Wall');

  // Bottom-Right: Cafeteria / Pantry divider
  addBlock(3.5, wallH / 2, 6.2, t, wallH, 5.5, wallMat, 'Cafeteria Divider Wall');

  // 4. Furniture & Real Physical Obstacles
  // Reception Desk (South-Center)
  addBlock(0, 0.55, 5.8, 3.4, 1.1, 0.9, woodMat, 'Reception Desk', 0.35);

  // Central Innovation Hub Collaboration Table / Pillar
  addBlock(0, 0.45, -0.5, 2.2, 0.9, 2.2, darkMetalMat, 'Central Innovation Hub Table', 0.35);

  // Meeting Room 4 Conference Table + Chairs
  addBlock(-7.8, 0.42, -5.2, 4.2, 0.84, 1.8, woodMat, 'Boardroom Conference Table', 0.35);

  // 3-Seater Sofas in Lounge Zone (Bottom-Left)
  // Sofa 1 (Facing East)
  addBlock(-9.2, 0.45, 4.5, 1.0, 0.9, 2.8, sofaMat, '3-Seater Sofa West', 0.3);
  addBlock(-9.55, 0.75, 4.5, 0.25, 0.6, 2.8, sofaMat, 'Sofa Backrest', 0.1);
  // Sofa 2 (Facing North)
  addBlock(-7.2, 0.45, 6.8, 2.6, 0.9, 1.0, sofaMat, '3-Seater Sofa South', 0.3);
  // Lounge Coffee Table
  addBlock(-7.4, 0.28, 4.6, 1.3, 0.55, 1.3, woodMat, 'Lounge Coffee Table', 0.25);

  // Engineering Desks (Top-Right)
  addBlock(6.8, 0.42, -6.2, 2.4, 0.84, 1.2, darkMetalMat, 'Workstation Desk A', 0.3);
  addBlock(9.8, 0.42, -6.2, 2.4, 0.84, 1.2, darkMetalMat, 'Workstation Desk B', 0.3);
  addBlock(8.2, 0.42, -3.5, 2.6, 0.84, 1.2, darkMetalMat, 'Workstation Desk C', 0.3);

  // Cafeteria Espresso Counter & High-Top Tables (Bottom-Right)
  addBlock(8.5, 0.55, 7.2, 4.5, 1.1, 0.9, woodMat, 'Espresso Bar Counter', 0.35);
  addBlock(6.5, 0.45, 4.2, 1.1, 0.9, 1.1, darkMetalMat, 'Cafeteria Bistro Table 1', 0.3);
  addBlock(9.5, 0.45, 4.2, 1.1, 0.9, 1.1, darkMetalMat, 'Cafeteria Bistro Table 2', 0.3);

  // Decorative Indoor Planters (Obstacles along corridors)
  addBlock(-2.6, 0.6, 1.8, 0.6, 1.2, 0.6, accentMat, 'Indoor Planter West', 0.3);
  addBlock(2.6, 0.6, 1.8, 0.6, 1.2, 0.6, accentMat, 'Indoor Planter East', 0.3);

  scene.add(rootGroup);
  return { rootGroup, interactableMeshes };
}

/**
 * Extracts obstacle bounding boxes from an uploaded custom 3D model (.GLB / .PLY)
 * by slicing geometry between ankle height (0.18m) and head height (1.85m).
 */
export function rebuildOccupancyFromCustomModel(modelRoot, pathfinder) {
  pathfinder.clearObstacles();

  modelRoot.updateMatrixWorld(true);
  modelRoot.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const box = new THREE.Box3().setFromObject(child);
      const height = box.max.y - box.min.y;
      const sizeX = box.max.x - box.min.x;
      const sizeZ = box.max.z - box.min.z;

      // Ignore flat floor planes or ceiling lights; register vertical obstacles
      if (box.max.y > 0.25 && box.min.y < 1.8 && height > 0.25 && sizeX < 22 && sizeZ < 16) {
        pathfinder.addObstacleBox(
          box.min.x,
          box.min.z,
          box.max.x,
          box.max.z,
          0.25,
          child.name || 'Scanned Obstacle'
        );
      }
    }
  });
}
