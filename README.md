# Namma Space 🌐🏢
**End-to-End 3D Indoor Reconstruction, Interactive First-Person Walkthrough, Spatial POI Search & Obstacle-Aware Navigation Engine**  
*Built for Techfest 2026-27 (IIT Bombay) — Namma Space Challenge*

---

## ✨ Key Features

1. **Zero-Install Interactive 3D Web Walkthrough (Round 1):**
   - Fluid **First-Person (`WASD` + Mouse PointerLock + `Shift` Sprint)** and **3D Orbit Overview** modes.
   - Real-time **wall & furniture sliding collision detection** locked to human eye-level ($1.65\text{m}$).
   - Supports loading custom reconstructed **`.glb` / `.gltf` meshes** and **`.ply` point clouds / Gaussian Splats** via the built-in **Load 3D Scan** modal, while shipping with a complete multi-room 3D Digital Twin out-of-the-box.

2. **Spatial Point-of-Interest (POI) Tagging & Instant Search (Round 2):**
   - **Click-to-Tag 3D Raycasting:** Toggle **📍 Tag New POI** (or press `T`) and click any room, wall, or piece of furniture in 3D space to pin a custom searchable label.
   - **Instant Search & Camera Fly-To:** Press `/` to search rooms or items (e.g., *"Meeting Room 4"*, *"3-Seater Sofas"*, *"Reception"*) and either **⚡ Instant Jump** (smooth camera fly-to) or **🧭 Route** to it.

3. **Obstacle-Aware Floor Pathfinding & Auto-Walk (Round 2):**
   - Slices 3D obstacles between ankle ($0.18\text{m}$) and head ($1.85\text{m}$) height into a **0.25m 2D Floor Occupancy Grid** with $0.32\text{m}$ safety clearance inflation.
   - Computes optimal **8-directional A\* paths** refined with **Bresenham Line-of-Sight string-pulling** and rendered as a glowing 3D **Catmull-Rom floor ribbon** with animated directional chevrons.
   - Includes a live **2D Top-Down Radar Minimap** and **🚶 Auto-Walk Path** first-person guided simulation.

4. **Live Mobile Camera AR Navigation (Round 3 Bonus):**
   - Click **📱 AR Navigate** to open the live smartphone rear camera feed overlaid with real-time 3D perspective floor chevrons and turn-by-turn distance prompts.

5. **Automated Smartphone Video SOP & 3D Reconstruction Pipeline (`pipeline/reconstruct.py`):**
   - Extracts sharp keyframes from smartphone video, filters out motion-blurred frames using **Variance of Laplacian**, computes an **SOP Capture Health Score**, and orchestrates COLMAP / Nerfstudio 3D reconstruction.

---

## 🚀 Quick Start Instructions

### 1. Run the Interactive 3D Web Application
Make sure you have [Node.js](https://nodejs.org/) (v18+) installed:

```bash
# 1. Navigate into the project folder
cd namma-space

# 2. Install dependencies
npm install

# 3. Start the local development server
npm run dev
```
Open **`http://localhost:5173`** in your browser.

### 2. Run the Python 3D Reconstruction & SOP Audit Pipeline
```bash
# Optional: Install Python dependencies for video frame extraction
pip install -r pipeline/requirements.txt

# Run with the included sample test dataset (no video required)
python pipeline/reconstruct.py

# Or run on your own smartphone indoor video (.mp4 / .mov)
python pipeline/reconstruct.py --input path/to/room_video.mp4 --output test/reconstructed_output
```

---

## 📁 Repository Structure

```text
namma-space/
├── index.html                        # Main Web UI, HUD, Search Bar, Minimap & AR Overlay
├── package.json                      # Frontend dependencies (Three.js + Vite)
├── vite.config.js                    # Vite bundler config
├── src/
│   ├── main.js                       # 3D Scene, WASD Physics, Raycast POI Tagging, Search UI
│   ├── indoorScene.js                # Multi-room 3D Digital Twin & custom .GLB/.PLY loader
│   ├── pathfinding.js                # 2D Occupancy Grid, 8-way A* & Line-of-Sight smoothing
│   ├── arNavigation.js               # Mobile Camera Augmented Reality (AR) chevron overlay
│   └── style.css                     # Glassmorphic HUD styling
├── pipeline/
│   ├── reconstruct.py                # Automated video SOP audit, blur filter & COLMAP/3DGS runner
│   └── requirements.txt              # Python dependencies
├── test/                             # Sample data directory (Required for Round 1 submission)
│   ├── sample_capture_manifest.json  # Sample capture SOP benchmark metadata
│   └── sample_pois.json              # Sample spatial POI coordinates & labels
└── docs/
    └── Technical_Abstract_and_SOP.md # 5-Page Technical Abstract & Capture SOP documentation
```
test line
