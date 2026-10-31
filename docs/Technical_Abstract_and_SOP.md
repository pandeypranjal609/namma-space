# Namma Space: End-to-End 3D Indoor Digital Twin & Spatial Navigation Engine
**Techfest 2026-27 (IIT Bombay) — Technical Abstract & Standard Operating Procedure (SOP)**

---

## 1. System Architecture Overview

**Namma Space** transforms ordinary smartphone video captures of indoor spaces into photorealistic, walkable 3D web environments equipped with spatial Point-of-Interest (POI) search, obstacle-aware A\* floor pathfinding, and live mobile Augmented Reality (AR) camera navigation.

```
[Smartphone Video Capture (1080p/4K @ 60fps, 0.5x Ultra-Wide)]
        │
        ▼
[1. Automated Pre-Processing & SOP Audit (pipeline/reconstruct.py)]
   • Keyframe Extraction (3 FPS via FFmpeg / OpenCV)
   • Motion-Blur Rejection (Variance of Laplacian >= 95.0)
   • SOP Health Scoring (Overlap, Sharpness, Luminance Stability)
        │
        ▼
[2. 3D Reconstruction & Floor NavMesh Extraction]
   • Camera Pose Estimation: COLMAP / GLOMAP Structure-from-Motion (SfM)
   • Visual Representation: 3D Gaussian Splatting (Nerfstudio splatfacto) / Textured GLB Mesh
   • Collision Layer: Horizontal Z-Slice (0.18m to 1.85m height) -> 2D Floor Occupancy Grid (0.25m cells)
        │
        ▼
[3. Zero-Install Three.js Web Engine & WebXR AR]
   • Fluid First-Person Roaming (PointerLockControls + WASD + Sliding Wall Collision)
   • 3D Raycast POI Tagging + Instant Spatial Search & Camera Fly-To
   • Obstacle-Inflated 8-Way A* Pathfinding + Line-of-Sight String Pulling + Catmull-Rom Spline
   • Live Smartphone Camera AR Overlay with 3D Perspective Directional Chevrons
```

---

## 2. Standard Operating Procedure (SOP) for Smartphone Data Collection

To guarantee deterministic, high-fidelity 3D reconstruction without specialized hardware, any non-technical user follows our **4-Rule Capture SOP**:

1. **Camera Configuration:**
   - Use standard smartphone video mode set to **1080p or 4K at 60 FPS**.
   - Select the **0.5x Ultra-Wide lens** to maximize indoor field-of-view (capturing floor-to-wall junctions in tight corridors).
   - **Lock Auto-Exposure and Auto-Focus (AE/AF Lock)** by long-pressing the screen prior to recording to prevent luminance flicker across views.

2. **Inside-Out Perimeter Loop Trajectory:**
   - Walk slowly along the perimeter of each room (~**0.45 m/s**) with the camera angled **15° downward** facing toward the center of the room.
   - Maintain **70%–80% visual overlap** between consecutive seconds of video.

3. **Doorway & Multi-Room Stitching Protocol:**
   - Never pivot rapidly in place (pure rotation has zero parallax and breaks triangulation).
   - When moving between rooms (e.g., Lobby to *Meeting Room 4*), walk smoothly through the doorway in a **figure-8 or S-curve arc**, ensuring shared visual features on both sides of the doorframe remain in view.

4. **Automated Pre-Flight SOP Verification (`pipeline/reconstruct.py`):**
   - Before running heavy 3D reconstruction, our pipeline computes the **Variance of Laplacian** ($\text{Var}(\nabla^2 I)$) for every sampled frame, automatically discarding motion-blurred frames ($<95.0$) and reporting an **SOP Health Score (0–100%)**.

---

## 3. Obstacle-Aware Floor Pathfinding & Spatial Indexing

1. **2D Occupancy Grid Extraction (`src/pathfinding.js`):**
   - Indoor spaces contain non-traversable vertical geometry (walls, conference tables, 3-seater sofas, desks, planters).
   - We slice the 3D scene bounding geometry between ankle height ($y = 0.18\text{m}$) and head height ($y = 1.85\text{m}$) and project obstacles onto a **0.25m resolution 2D grid**.
   - Every obstacle bounding box is inflated by a **human clearance radius ($r = 0.32\text{m}$)** so planned paths never graze sharp furniture corners.

2. **8-Directional A\* Search with Line-of-Sight Smoothing:**
   - When a user searches or clicks a POI, an 8-connected A\* search computes the shortest non-colliding path from the user's $(x, z)$ coordinates to the target POI, preventing diagonal corner-cutting across wall vertices.
   - Raw grid cells are refined using **Bresenham Line-of-Sight String-Pulling** and interpolated into a smooth **3D Catmull-Rom Spline** rendered just above the floor ($y = 0.08\text{m}$) with animated directional chevrons.

---

## 4. Open-Source Libraries & Credits

- **Three.js (`three`):** WebGL 3D rendering, Raycasting, `PointerLockControls`, `OrbitControls`, `GLTFLoader`, and `PLYLoader`.
- **OpenCV & NumPy (`opencv-python`, `numpy`):** Frame extraction and Laplacian blur filtering.
- **COLMAP / Nerfstudio / Open3D:** Structure-from-Motion (SfM), 3D Gaussian Splatting (`splatfacto`), and mesh processing.
- **Vite:** Frontend dev server and ES-module bundler.
