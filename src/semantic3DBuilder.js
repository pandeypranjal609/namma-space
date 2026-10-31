/**
 * Namma Space - WebGL 3D Gaussian Splatting (3DGS) & Depth Surface Renderer (`src/semantic3DBuilder.js`)
 * ======================================================================================================
 * Renders true volumetric 3D Gaussian Splats (`gaussian_splats`: 3D center `(x, y, z)`, RGB color,
 * Gaussian scale `sigma`, and opacity `alpha`) produced by `pipeline/server.py` using a custom
 * WebGL GLSL 3D Gaussian Splatting Shader (`exp(-2.2 * r^2)` radial Gaussian kernel falloff +
 * perspective-correct focal scaling), backed by a companion raycastable 3D depth surface mesh.
 */

import * as THREE from 'three';

const textureLoader = new THREE.TextureLoader();

function loadHighResTexture(base64Str) {
  return new Promise((resolve) => {
    if (!base64Str) {
      resolve(null);
      return;
    }
    textureLoader.load(
      base64Str,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        resolve(tex);
      },
      undefined,
      () => resolve(null)
    );
  });
}

/**
 * Creates a custom WebGL GLSL 3D Gaussian Splatting Point/Splat Cloud (`THREE.Points` with
 * perspective-projected Gaussian kernel shader `exp(-2.2 * dot(uv, uv))`).
 */
export function create3DGaussianSplatCloud(splatData) {
  const count = splatData.count || 0;
  if (count === 0 || !splatData.positions) return null;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(splatData.positions, 3));
  geom.setAttribute('aColor', new THREE.Float32BufferAttribute(splatData.colors, 3));
  geom.setAttribute('aScale', new THREE.Float32BufferAttribute(splatData.scales, 1));
  geom.setAttribute('aOpacity', new THREE.Float32BufferAttribute(splatData.opacities, 1));
  geom.computeBoundingSphere();

  const focalPx = Math.max(window.innerWidth, window.innerHeight) * 0.95;

  const splatMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uFocal: { value: focalPx }
    },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aScale;
      attribute float aOpacity;
      uniform float uFocal;
      varying vec3 vColor;
      varying float vOpacity;

      void main() {
        vColor = aColor;
        vOpacity = aOpacity;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        float dist = max(0.35, -mvPosition.z);
        gl_PointSize = clamp((aScale * uFocal) / dist, 3.0, 58.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vOpacity;

      void main() {
        vec2 coord = (gl_PointCoord - vec2(0.5)) * 2.0;
        float r2 = dot(coord, coord);
        if (r2 > 1.0) discard;

        // 3D Gaussian Splatting exponential kernel falloff: exp(-0.5 * (2.1 * r)^2)
        float gaussianWeight = exp(-2.25 * r2);
        float alpha = vOpacity * gaussianWeight;
        if (alpha < 0.03) discard;

        // Subtle spherical-harmonic style center luminance boost
        vec3 shadedColor = vColor * (0.96 + 0.08 * gaussianWeight);
        gl_FragColor = vec4(shadedColor, alpha);
      }
    `,
    transparent: true,
    depthTest: true,
    depthWrite: true
  });

  const points = new THREE.Points(geom, splatMaterial);
  points.name = 'GaussianSplatCloud_3DGS';
  return points;
}

/**
 * Builds the finalized 3D Gaussian Splatting (3DGS) + Companion Raycast Surface Scene.
 */
export async function buildFinalized3DScene(sceneGraph, scene, pathfinder) {
  const rootGroup = new THREE.Group();
  rootGroup.name = 'NammaSpace_3DGaussianSplatScene';
  const interactableMeshes = [];

  pathfinder.clearObstacles();

  const dims = sceneGraph.room_dimensions || {
    width_m: 3.6,
    depth_m: 3.2,
    height_m: 2.2,
    floor_area_m2: 11.5,
    min_x: -1.8,
    max_x: 1.8,
    min_z: -1.6,
    max_z: 1.6,
    angular_coverage_deg: 62,
    coverage_percent: 100
  };

  const cx = (dims.min_x + dims.max_x) / 2;
  const cz = (dims.min_z + dims.max_z) / 2;
  const span = Math.max(4.5, Math.max(dims.width_m, dims.depth_m) * 1.4);

  // 1. Dark Studio Metric Ground Grid Underneath the 3D Gaussian Splats
  const floorPlane = new THREE.Mesh(
    new THREE.PlaneGeometry(span, span),
    new THREE.MeshStandardMaterial({
      color: 0x0b1120,
      roughness: 0.92,
      metalness: 0.05
    })
  );
  floorPlane.rotation.x = -Math.PI / 2;
  floorPlane.position.set(cx, 0.0, cz);
  floorPlane.receiveShadow = true;
  floorPlane.name = 'SplatFloorGrid';
  rootGroup.add(floorPlane);
  interactableMeshes.push(floorPlane);

  const gridHelper = new THREE.GridHelper(
    span,
    Math.round(span * 4),
    0x38bdf8,
    0x1e293b
  );
  gridHelper.position.set(cx, 0.006, cz);
  gridHelper.material.opacity = 0.32;
  gridHelper.material.transparent = true;
  rootGroup.add(gridHelper);

  // 2. Companion 3D Pinhole Depth Surface (Provides solid volume fill & 3D Raycast POI Tagging)
  for (const sm of sceneGraph.surface_meshes || []) {
    if (!sm.positions || !sm.indices || sm.indices.length === 0) continue;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(sm.positions, 3));
    if (sm.uvs && sm.uvs.length > 0) {
      geom.setAttribute('uv', new THREE.Float32BufferAttribute(sm.uvs, 2));
    }
    geom.setIndex(sm.indices);
    geom.computeVertexNormals();

    const tex = await loadHighResTexture(sm.image_base64);
    const mat = new THREE.MeshBasicMaterial({
      map: tex || null,
      color: 0xffffff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.68
    });

    const mesh = new THREE.Mesh(geom, mat);
    mesh.name = sm.id || 'CompanionDepthSurface';
    rootGroup.add(mesh);
    interactableMeshes.push(mesh);
  }

  // 3. Render Volumetric 3D Gaussian Splats (`gaussian_splats`) on Top via Custom GLSL Shader!
  if (sceneGraph.gaussian_splats && sceneGraph.gaussian_splats.count > 0) {
    const splatCloud = create3DGaussianSplatCloud(sceneGraph.gaussian_splats);
    if (splatCloud) {
      splatCloud.renderOrder = 2;
      rootGroup.add(splatCloud);
    }
  }

  // 4. Register Scanned 3D Obstacles in A* Pathfinder
  for (const obs of sceneGraph.obstacles || []) {
    pathfinder.addObstacleBox(
      obs.x - 0.32,
      obs.z - 0.32,
      obs.x + 0.32,
      obs.z + 0.32,
      0.16,
      obs.label || obs.class_name
    );
  }

  scene.add(rootGroup);
  return { rootGroup, interactableMeshes, roomDimensions: dims };
}
