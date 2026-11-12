// roblox-up-and-coming-forecast-01: Monte Carlo rank-diffusion forecaster
// with tenure hazard, momentum, mean-reversion, and newcomer injection
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface GameRow {
  rank: number;
  universe_id: number;
  name: string;
  playing: number;
}

interface Snapshot {
  at: string;
  up_and_coming: GameRow[];
  top_trending: GameRow[] | null;
  top_playing_now: Array<{ rank: number; universe_id: number; playing: number }> | null;
}

const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window: { from: string; until: string; slots: number; slot_minutes: number };
  history_until: string;
};
const rawLines = readFileSync(join(inputDir, 'history.ndjson'), 'utf8').trim().split('\n');
const snapshots: Snapshot[] = rawLines.map((l) => JSON.parse(l));
const PLACES = task.places;
const N_SIM = 5000;
const SEED = 42;

// Deterministic PRNG (mulberry32)
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

// --- Build game histories from snapshots ---
const gameHistories = new Map<number, Array<{ snapIdx: number; rank: number; playing: number }>>();
for (let i = 0; i < snapshots.length; i++) {
  for (const g of snapshots[i].up_and_coming) {
    if (!gameHistories.has(g.universe_id)) gameHistories.set(g.universe_id, []);
    gameHistories.get(g.universe_id)!.push({ snapIdx: i, rank: g.rank, playing: g.playing });
  }
}

// --- Compute rank-dependent volatility from observed transitions ---
const rankVolatility: number[] = new Array(101).fill(2.0);
{
  const changesByRank: Map<number, number[]> = new Map();
  for (let i = 0; i < snapshots.length - 1; i++) {
    const currMap = new Map(snapshots[i].up_and_coming.map(g => [g.universe_id, g.rank]));
    const nextMap = new Map(snapshots[i + 1].up_and_coming.map(g => [g.universe_id, g.rank]));
    const dt = (new Date(snapshots[i + 1].at).getTime() - new Date(snapshots[i].at).getTime()) / 3600000;
    const dtNorm = Math.max(1, dt) / 6;
    for (const [uid, r] of currMap) {
      const nr = nextMap.get(uid);
      if (nr !== undefined) {
        const bucket = Math.min(r, 100);
        if (!changesByRank.has(bucket)) changesByRank.set(bucket, []);
        changesByRank.get(bucket)!.push((nr - r) / Math.sqrt(dtNorm));
      }
    }
  }
  for (const [rank, changes] of changesByRank) {
    if (changes.length < 3) continue;
    const mean = changes.reduce((a, b) => a + b, 0) / changes.length;
    const variance = changes.reduce((a, b) => a + (b - mean) ** 2, 0) / changes.length;
    rankVolatility[rank] = Math.max(0.3, Math.sqrt(variance));
  }
  // Smooth gaps
  for (let r = 2; r < 99; r++) {
    if (!changesByRank.has(r) || (changesByRank.get(r)?.length ?? 0) < 5) {
      rankVolatility[r] = (rankVolatility[r - 1] + rankVolatility[r + 1]) / 2;
    }
  }
}

// --- Build game states from last snapshot ---
interface GameState {
  universe_id: number;
  rank: number;
  momentum: number;
  tenure: number;
  meanRank: number;
  onTopTrending: boolean;
}

const lastSnap = snapshots[snapshots.length - 1];
const topTrendingSet = new Set<number>();
if (lastSnap.top_trending) {
  for (const g of lastSnap.top_trending.slice(0, 30)) {
    topTrendingSet.add(g.universe_id);
  }
}

const chartSize = lastSnap.up_and_coming.length;
const games: GameState[] = [];

for (const g of lastSnap.up_and_coming) {
  const hist = gameHistories.get(g.universe_id)!;
  const tenure = hist.length;
  const recentRanks = hist.slice(-5).map(h => h.rank);
  const meanRank = recentRanks.reduce((a, b) => a + b, 0) / recentRanks.length;

  let momentum = 0;
  if (hist.length >= 2) {
    const h1 = hist[hist.length - 1];
    const h2 = hist[hist.length - 2];
    if (h2.playing > 0) momentum = (h1.playing - h2.playing) / h2.playing;
  }

  games.push({
    universe_id: g.universe_id,
    rank: g.rank,
    momentum: Math.max(-1, Math.min(1, momentum)),
    tenure,
    meanRank,
    onTopTrending: topTrendingSet.has(g.universe_id),
  });
}

// --- Monte Carlo simulation ---
const totalSlots = task.window.slots;
const placeCountsByGame = new Map<number, Float64Array>();
for (const g of games) {
  placeCountsByGame.set(g.universe_id, new Float64Array(PLACES + 1));
}

for (let sim = 0; sim < N_SIM; sim++) {
  const currentRanks = games.map(g => g.rank as number);
  const dropped = new Array(games.length).fill(false);

  for (let hour = 0; hour < totalSlots; hour++) {
    for (let gi = 0; gi < games.length; gi++) {
      if (dropped[gi]) continue;
      const game = games[gi];
      const r = currentRanks[gi];
      const bucket = Math.min(Math.round(r), 100);
      const vol = rankVolatility[bucket] || 2.0;

      // Mean-reversion toward recent average
      const meanRevDrift = 0.05 * (game.meanRank - r);
      // Momentum: positive playing growth -> rank improves (toward 1)
      const momDrift = -game.momentum * 0.3;
      // Tenure hazard: long-lived games drift down slightly
      const tenFrac = game.tenure / snapshots.length;
      const tenDrift = tenFrac > 0.8 ? 0.5 : tenFrac > 0.6 ? 0.2 : 0;
      // Top trending bonus
      const trendDrift = game.onTopTrending ? -0.1 : 0;
      // Hourly vol (snapshots ~6h apart)
      const hourlyVol = vol / Math.sqrt(6);

      const drift = meanRevDrift + momDrift + tenDrift + trendDrift;
      let newRank = r + drift + randn() * hourlyVol;
      newRank = Math.max(1, newRank);

      if (newRank > chartSize + 10) {
        dropped[gi] = true;
        currentRanks[gi] = chartSize + 50;
        continue;
      }
      currentRanks[gi] = newRank;
    }

    // Argsort to get discrete positions
    const indices = games.map((_, i) => i).filter(i => !dropped[i]);
    indices.sort((a, b) => currentRanks[a] - currentRanks[b]);

    for (let p = 0; p < Math.min(PLACES, indices.length); p++) {
      placeCountsByGame.get(games[indices[p]].universe_id)![p + 1]++;
    }
  }
}

// --- Build forecast ---
const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];

for (let place = 1; place <= PLACES; place++) {
  const candidates: Array<{ universe_id: number; p: number }> = [];
  for (const [uid, counts] of placeCountsByGame) {
    const p = counts[place] / (N_SIM * totalSlots);
    if (p >= 0.005) candidates.push({ universe_id: uid, p });
  }
  candidates.sort((a, b) => b.p - a.p);
  const topCands = candidates.slice(0, 80);

  let sum = topCands.reduce((a, c) => a + c.p, 0);
  const maxSum = 0.92;
  if (sum > maxSum) {
    const scale = maxSum / sum;
    for (const c of topCands) c.p *= scale;
  }

  for (const c of topCands) {
    probabilities.push({
      universe_id: c.universe_id,
      place,
      p: Math.round(c.p * 10000) / 10000,
    });
  }
}

// Top 10: games with highest overall top-10 presence
const top10Scores = new Map<number, number>();
for (const [uid, counts] of placeCountsByGame) {
  let total = 0;
  for (let p = 1; p <= PLACES; p++) total += counts[p];
  top10Scores.set(uid, total);
}
const top10 = [...top10Scores.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, PLACES)
  .map(([uid]) => uid);

const result = { probabilities, top10 };
writeFileSync(join(outputDir, task.output), `${JSON.stringify(result)}\n`, 'utf8');
console.log(`Forecast: ${probabilities.length} probs, ${N_SIM} sims x ${totalSlots} slots`);
