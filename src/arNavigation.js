/**
 * Namma Space - Live On-Site Mobile Camera AR Navigation Engine (Bonus Deliverable)
 * Overlays real-time 3D perspective directional arrows & waypoint distance onto the live smartphone camera feed.
 */

export class ARNavigationController {
  constructor({ overlayEl, videoEl, canvasEl, directionEl, subInfoEl, onExit }) {
    this.overlayEl = overlayEl;
    this.videoEl = videoEl;
    this.canvasEl = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.directionEl = directionEl;
    this.subInfoEl = subInfoEl;
    this.onExit = onExit;

    this.active = false;
    this.stream = null;
    this.animFrame = null;
    this.deviceHeading = 0; // Compass / gyroscope yaw compensation
    this.stepPhase = 0;

    window.addEventListener('deviceorientation', (e) => {
      if (typeof e.alpha === 'number') {
        this.deviceHeading = (e.alpha * Math.PI) / 180;
      }
    });
  }

  async start(getNavigationState) {
    this.getNavigationState = getNavigationState;
    this.overlayEl.classList.remove('hidden');
    this.active = true;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false
      });
      this.videoEl.srcObject = this.stream;
    } catch {
      // Fallback simulated camera grid if desktop has no rear camera
      this.subInfoEl.textContent =
        'Simulated AR Camera Feed active (Grant camera permission on mobile for live video)';
    }

    const renderLoop = () => {
      if (!this.active) return;
      this.stepPhase += 0.06;
      this.drawARHud();
      this.animFrame = requestAnimationFrame(renderLoop);
    };
    renderLoop();
  }

  stop() {
    this.active = false;
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.overlayEl.classList.add('hidden');
    if (this.onExit) this.onExit();
  }

  drawARHud() {
    const w = (this.canvasEl.width = window.innerWidth);
    const h = (this.canvasEl.height = window.innerHeight);
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    const state = this.getNavigationState ? this.getNavigationState() : null;
    if (!state || !state.activePoi || !state.path || state.path.length < 2) {
      this.directionEl.textContent = '🔍 Select a POI first to start AR Route';
      this.subInfoEl.textContent = 'Return to the 3D view and click "Navigate" on any destination';
      return;
    }

    const { playerPos, playerYaw, path, activePoi } = state;

    // Pick next lookahead waypoint on the path
    let nextWp = path[Math.min(1, path.length - 1)];
    for (const wp of path) {
      if (Math.hypot(wp.x - playerPos.x, wp.z - playerPos.z) > 1.2) {
        nextWp = wp;
        break;
      }
    }

    const dx = nextWp.x - playerPos.x;
    const dz = nextWp.z - playerPos.z;
    const totalDist = Math.hypot(activePoi.x - playerPos.x, activePoi.z - playerPos.z);

    // Angle to next waypoint relative to player look direction
    const targetAngle = Math.atan2(-dx, -dz);
    let deltaAngle = targetAngle - playerYaw + this.deviceHeading * 0.2;
    while (deltaAngle > Math.PI) deltaAngle -= Math.PI * 2;
    while (deltaAngle < -Math.PI) deltaAngle += Math.PI * 2;

    if (totalDist < 1.4) {
      this.directionEl.textContent = `🎉 Arrived at ${activePoi.name}!`;
    } else if (deltaAngle > 0.35) {
      this.directionEl.textContent = `⬅️ Turn Slight Left toward ${activePoi.name}`;
    } else if (deltaAngle < -0.35) {
      this.directionEl.textContent = `➡️ Turn Slight Right toward ${activePoi.name}`;
    } else {
      this.directionEl.textContent = `⬆️ Walk Straight Ahead (${totalDist.toFixed(1)}m)`;
    }

    this.subInfoEl.textContent = `Target: ${activePoi.name} • Remaining: ${totalDist.toFixed(1)}m • Obstacle-Free A* Locked`;

    // Render 3D perspective floor chevrons projected onto the AR camera view
    const centerX = w / 2 - Math.sin(deltaAngle) * (w * 0.26);
    const horizonY = h * 0.52;

    for (let i = 0; i < 5; i++) {
      const progress = ((i + (this.stepPhase % 1)) / 5);
      const y = h * 0.85 - progress * (h * 0.32);
      const scale = 1.25 - progress * 0.65;
      const xOffset = (centerX - w / 2) * progress;
      const cx = w / 2 + xOffset;

      ctx.save();
      ctx.translate(cx, y);
      ctx.rotate(-deltaAngle * 0.45);
      ctx.scale(scale, scale * 0.55);

      ctx.beginPath();
      ctx.moveTo(0, -38);
      ctx.lineTo(44, 18);
      ctx.lineTo(20, 18);
      ctx.lineTo(0, -8);
      ctx.lineTo(-20, 18);
      ctx.lineTo(-44, 18);
      ctx.closePath();

      ctx.fillStyle = `rgba(34, 197, 94, ${0.9 - progress * 0.5})`;
      ctx.shadowColor = '#22c55e';
      ctx.shadowBlur = 18;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#dcfce7';
      ctx.stroke();
      ctx.restore();
    }

    // Draw floating AR destination pin near horizon
    ctx.save();
    ctx.translate(centerX, horizonY - 25);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-95, -24, 190, 40, 20);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`📍 ${activePoi.name} (${totalDist.toFixed(1)}m)`, 0, 1);
    ctx.restore();
  }
}
