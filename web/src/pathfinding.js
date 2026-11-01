/**
 * Namma Space - 2D Floor Occupancy Grid & Obstacle-Aware A* Pathfinding Engine
 * Computes non-colliding walkable indoor routes around walls and furniture.
 */

export class IndoorPathfinder {
  /**
   * @param {number} widthMeters Total floor width in meters (X axis: -width/2 to +width/2)
   * @param {number} depthMeters Total floor depth in meters (Z axis: -depth/2 to +depth/2)
   * @param {number} cellSize Grid resolution in meters (default 0.25m per cell)
   */
  constructor(widthMeters = 24, depthMeters = 18, cellSize = 0.25) {
    this.widthMeters = widthMeters;
    this.depthMeters = depthMeters;
    this.cellSize = cellSize;

    this.cols = Math.ceil(widthMeters / cellSize);
    this.rows = Math.ceil(depthMeters / cellSize);

    // 0 = walkable floor, 1 = physical obstacle (wall / furniture)
    this.grid = new Uint8Array(this.cols * this.rows);
    this.obstacles = [];
  }

  clearObstacles() {
    this.grid.fill(0);
    this.obstacles = [];
  }

  /**
   * Convert world (x, z) in meters to grid (col, row)
   */
  worldToGrid(x, z) {
    const col = Math.floor((x + this.widthMeters / 2) / this.cellSize);
    const row = Math.floor((z + this.depthMeters / 2) / this.cellSize);
    return {
      col: Math.max(0, Math.min(this.cols - 1, col)),
      row: Math.max(0, Math.min(this.rows - 1, row))
    };
  }

  /**
   * Convert grid (col, row) back to world (x, z) in meters
   */
  gridToWorld(col, row) {
    const x = (col + 0.5) * this.cellSize - this.widthMeters / 2;
    const z = (row + 0.5) * this.cellSize - this.depthMeters / 2;
    return { x, z };
  }

  /**
   * Register an axis-aligned bounding box obstacle (wall, sofa, desk, pillar)
   * with safety inflation radius so walking paths keep a comfortable clearance.
   */
  addObstacleBox(minX, minZ, maxX, maxZ, inflateMeters = 0.32, label = 'Obstacle') {
    this.obstacles.push({ minX, minZ, maxX, maxZ, label });

    const start = this.worldToGrid(minX - inflateMeters, minZ - inflateMeters);
    const end = this.worldToGrid(maxX + inflateMeters, maxZ + inflateMeters);

    for (let r = start.row; r <= end.row; r++) {
      for (let c = start.col; c <= end.col; c++) {
        this.grid[r * this.cols + c] = 1;
      }
    }
  }

  isWalkable(col, row) {
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return false;
    return this.grid[row * this.cols + col] === 0;
  }

  isWorldPointWalkable(x, z) {
    const { col, row } = this.worldToGrid(x, z);
    return this.isWalkable(col, row);
  }

  /**
   * If start or target lies inside an inflated furniture/wall cell,
   * snap to the nearest walkable cell using Breadth-First Search.
   */
  findNearestWalkableCell(col, row) {
    if (this.isWalkable(col, row)) return { col, row };

    const maxRadius = 25;
    for (let radius = 1; radius <= maxRadius; radius++) {
      for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
          if (Math.abs(dr) !== radius && Math.abs(dc) !== radius) continue;
          const nc = col + dc;
          const nr = row + dr;
          if (this.isWalkable(nc, nr)) {
            return { col: nc, row: nr };
          }
        }
      }
    }
    return { col, row };
  }

  /**
   * Compute optimal 8-directional A* walking path on the floor plane
   * @returns {Array<{x: number, y: number, z: number}>} Array of 3D floor waypoints
   */
  findPath(startX, startZ, targetX, targetZ) {
    const rawStart = this.worldToGrid(startX, startZ);
    const rawEnd = this.worldToGrid(targetX, targetZ);

    const start = this.findNearestWalkableCell(rawStart.col, rawStart.row);
    const end = this.findNearestWalkableCell(rawEnd.col, rawEnd.row);

    const totalCells = this.cols * this.rows;
    const gScore = new Float32Array(totalCells).fill(Infinity);
    const fScore = new Float32Array(totalCells).fill(Infinity);
    const cameFrom = new Int32Array(totalCells).fill(-1);
    const closed = new Uint8Array(totalCells);

    const startIdx = start.row * this.cols + start.col;
    const endIdx = end.row * this.cols + end.col;

    const heuristic = (c1, r1, c2, r2) => Math.hypot(c2 - c1, r2 - r1);

    gScore[startIdx] = 0;
    fScore[startIdx] = heuristic(start.col, start.row, end.col, end.row);

    const openSet = [startIdx];

    // 8-connected neighbors (cardinal + diagonal with corner-cutting prevention)
    const dirs = [
      { dc: 1, dr: 0, cost: 1.0 },
      { dc: -1, dr: 0, cost: 1.0 },
      { dc: 0, dr: 1, cost: 1.0 },
      { dc: 0, dr: -1, cost: 1.0 },
      { dc: 1, dr: 1, cost: Math.SQRT2 },
      { dc: -1, dr: 1, cost: Math.SQRT2 },
      { dc: 1, dr: -1, cost: Math.SQRT2 },
      { dc: -1, dr: -1, cost: Math.SQRT2 }
    ];

    while (openSet.length > 0) {
      // Pop node with lowest fScore
      let bestPos = 0;
      for (let i = 1; i < openSet.length; i++) {
        if (fScore[openSet[i]] < fScore[openSet[bestPos]]) {
          bestPos = i;
        }
      }
      const currentIdx = openSet[bestPos];
      openSet.splice(bestPos, 1);

      if (currentIdx === endIdx) {
        return this.reconstructAndSmoothPath(cameFrom, currentIdx);
      }

      closed[currentIdx] = 1;
      const cCol = currentIdx % this.cols;
      const cRow = Math.floor(currentIdx / this.cols);

      for (const { dc, dr, cost } of dirs) {
        const nCol = cCol + dc;
        const nRow = cRow + dr;

        if (!this.isWalkable(nCol, nRow)) continue;

        // Prevent diagonal squeezing through wall corners
        if (dc !== 0 && dr !== 0) {
          if (!this.isWalkable(cCol + dc, cRow) || !this.isWalkable(cCol, cRow + dr)) {
            continue;
          }
        }

        const neighborIdx = nRow * this.cols + nCol;
        if (closed[neighborIdx]) continue;

        const tentativeG = gScore[currentIdx] + cost;
        if (tentativeG < gScore[neighborIdx]) {
          cameFrom[neighborIdx] = currentIdx;
          gScore[neighborIdx] = tentativeG;
          fScore[neighborIdx] = tentativeG + heuristic(nCol, nRow, end.col, end.row);
          if (!openSet.includes(neighborIdx)) {
            openSet.push(neighborIdx);
          }
        }
      }
    }

    return [];
  }

  /**
   * Reconstruct grid path and apply Line-of-Sight string-pulling so paths look natural
   */
  reconstructAndSmoothPath(cameFrom, currentIdx) {
    const rawCells = [];
    let curr = currentIdx;
    while (curr !== -1) {
      const col = curr % this.cols;
      const row = Math.floor(curr / this.cols);
      rawCells.push({ col, row });
      curr = cameFrom[curr];
    }
    rawCells.reverse();

    if (rawCells.length <= 2) {
      return rawCells.map((c) => {
        const w = this.gridToWorld(c.col, c.row);
        return { x: w.x, y: 0.06, z: w.z };
      });
    }

    // String-pulling (Line-of-Sight pruning)
    const smoothed = [rawCells[0]];
    let anchor = 0;

    while (anchor < rawCells.length - 1) {
      let furthestVisible = anchor + 1;
      for (let probe = rawCells.length - 1; probe > anchor + 1; probe--) {
        if (this.hasLineOfSight(rawCells[anchor], rawCells[probe])) {
          furthestVisible = probe;
          break;
        }
      }
      smoothed.push(rawCells[furthestVisible]);
      anchor = furthestVisible;
    }

    return smoothed.map((c) => {
      const w = this.gridToWorld(c.col, c.row);
      return { x: w.x, y: 0.06, z: w.z };
    });
  }

  /**
   * Bresenham supercover line-of-sight check between two grid cells
   */
  hasLineOfSight(cellA, cellB) {
    const steps = Math.max(Math.abs(cellB.col - cellA.col), Math.abs(cellB.row - cellA.row)) * 2;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const c = Math.round(cellA.col + (cellB.col - cellA.col) * t);
      const r = Math.round(cellA.row + (cellB.row - cellA.row) * t);
      if (!this.isWalkable(c, r)) return false;
    }
    return true;
  }
}
