// roblox-up-and-coming-forecast-01: Log-rank Monte Carlo forecaster
// Models log(rank) as Brownian motion with rank-dependent drift and volatility
// Calibrated from observed historical transitions
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface GameRow { rank: number; universe_id: number; name: string; playing: number }
interface Snap { at: string; up_and_coming: GameRow[] }

const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as {
  places: number; output: string;
  window: { from: string; until: string; slots: number };
};
const rawLines = readFileSync(join(inputDir, 'history.ndjson'), 'utf8').trim().split('\n');
const snaps: Snap[] = rawLines.map(l => JSON.parse(l));
const PLACES = task.places;
const N = snaps.length;
const SLOTS = task.window.slots;
const N_SIM = 10000;
const SEED = 42;

// Sigma multiplier: scale up calibrated volatility to model
// breakout events (games jumping 20+ ranks in days)
const SIGMA_SCALE = 1.8;

// Deterministic PRNG
function mulberry32(a: number) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(SEED);
function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// --- Calibrate drift and volatility from history ---
// Compute per-hour log-rank changes, bucketed by rank
interface CalibBucket { drift: number; sigma: number }

const buckets: CalibBucket[] = [];
const bucketEdges = [0, 3, 8, 15, 30, 60, 200];

for (let b = 0; b < bucketEdges.length - 1; b++) {
  const lo = bucketEdges[b];
  const hi = bucketEdges[b + 1];
  const changes: number[] = [];

  for (let i = 0; i < N - 1; i++) {
    const curr = new Map(snaps[i].up_and_coming.map(g => [g.universe_id, g.rank]));
    const next = new Map(snaps[i + 1].up_and_coming.map(g => [g.universe_id, g.rank]));
    const hours = Math.max(1, (new Date(snaps[i + 1].at).getTime() - new Date(snaps[i].at).getTime()) / 3600000);

    for (const [uid, r1] of curr) {
      const r2 = next.get(uid);
      if (r2 && r1 > lo && r1 <= hi) {
        changes.push((Math.log(r2) - Math.log(r1)) / Math.sqrt(hours));
      }
    }
  }

  if (changes.length > 5) {
    const mean = changes.reduce((a, c) => a + c, 0) / changes.length;
    const variance = changes.reduce((a, c) => a + (c - mean) ** 2, 0) / changes.length;
    buckets.push({ drift: mean, sigma: Math.sqrt(variance) });
  } else {
    buckets.push({ drift: 0, sigma: 0.09 });
  }
}

function getCalib(rank: number): CalibBucket {
  for (let b = 0; b < bucketEdges.length - 1; b++) {
    if (rank > bucketEdges[b] && rank <= bucketEdges[b + 1]) return buckets[b];
  }
  return buckets[buckets.length - 1];
}

// --- Build game list from last snapshot ---
const lastSnap = snaps[N - 1];
const chartSize = lastSnap.up_and_coming.length;

interface Game {
  uid: number;
  logRank: number; // log of current rank
  momentum: number; // player growth rate
}

const games: Game[] = [];
const prevSnap = N >= 2 ? snaps[N - 2] : null;
const prevMap = prevSnap ? new Map(prevSnap.up_and_coming.map(g => [g.universe_id, g])) : null;

for (const g of lastSnap.up_and_coming) {
  let momentum = 0;
  if (prevMap) {
    const prev = prevMap.get(g.universe_id);
    if (prev && prev.playing > 100) {
      momentum = (g.playing - prev.playing) / prev.playing;
    }
  }
  games.push({
    uid: g.universe_id,
    logRank: Math.log(g.rank),
    momentum: Math.max(-0.5, Math.min(0.5, momentum)),
  });
}

// --- Monte Carlo simulation ---
// For each simulation:
// 1. Evolve log(rank) per hour with drift + noise
// 2. At each slot, sort by log-rank to get discrete positions
// 3. Count place occupancy

const placeCount = new Map<number, Float64Array>();
for (const g of games) {
  placeCount.set(g.uid, new Float64Array(PLACES + 1));
}

for (let sim = 0; sim < N_SIM; sim++) {
  const logRanks = games.map(g => g.logRank);

  for (let hour = 0; hour < SLOTS; hour++) {
    // Evolve each game
    for (let gi = 0; gi < games.length; gi++) {
      const lr = logRanks[gi];
      const rank = Math.exp(lr);
      const calib = getCalib(rank);

      // Drift: calibrated + small momentum component
      const momDrift = -games[gi].momentum * 0.02; // positive growth → rank drops (improves)
      const drift = calib.drift + momDrift;

      // Noise: calibrated sigma scaled up for heavy tails
      const noise = calib.sigma * SIGMA_SCALE * randn();

      logRanks[gi] = Math.max(0, lr + drift + noise); // min log-rank = 0 → rank 1
    }

    // Sort to get positions
    const indices = games.map((_, i) => i);
    indices.sort((a, b) => logRanks[a] - logRanks[b]);

    for (let p = 0; p < Math.min(PLACES, indices.length); p++) {
      placeCount.get(games[indices[p]].uid)![p + 1]++;
    }
  }
}

// --- Build forecast ---
const totalObs = N_SIM * SLOTS;
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];

for (let place = 1; place <= PLACES; place++) {
  const candidates: Array<{ uid: number; p: number }> = [];

  for (const [uid, counts] of placeCount) {
    const p = counts[place] / totalObs;
    if (p >= 0.003) candidates.push({ uid, p });
  }

  candidates.sort((a, b) => b.p - a.p);

  // Cap total named at 88% (leave 12% for newcomers/unknown)
  const maxNamed = 0.88;
  let sum = candidates.reduce((a, c) => a + c.p, 0);
  if (sum > maxNamed) {
    const scale = maxNamed / sum;
    for (const c of candidates) c.p *= scale;
  }

  for (const c of candidates) {
    if (c.p >= 0.005) {
      probabilities.push({
        universe_id: c.uid,
        place,
        p: Math.round(c.p * 10000) / 10000,
      });
    }
  }
}

// Top 10
const top10Map = new Map<number, number>();
for (const e of probabilities) {
  top10Map.set(e.universe_id, (top10Map.get(e.universe_id) || 0) + e.p);
}
const top10 = [...top10Map.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, PLACES)
  .map(([uid]) => uid);

const result = { probabilities, top10 };
writeFileSync(join(outputDir, task.output), `${JSON.stringify(result)}\n`, 'utf8');
console.log(`Forecast: ${probabilities.length} probs, ${N_SIM} sims × ${SLOTS} slots`);
for (let p = 1; p <= PLACES; p++) {
  const es = probabilities.filter(e => e.place === p).sort((a, b) => b.p - a.p);
  const sum = es.reduce((a, e) => a + e.p, 0);
  console.log(`  P${p}: ${es.length} games, Σ=${sum.toFixed(3)}, top3=[${es.slice(0, 3).map(e => e.p.toFixed(3)).join(',')}]`);
}
