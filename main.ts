// roblox-up-and-coming-forecast-01: Maximum-coverage weighted rank forecast
// Wide candidate set with calibrated probability allocation
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface GameRow { rank: number; universe_id: number; name: string; playing: number }
interface Snap { at: string; up_and_coming: GameRow[]; top_trending: GameRow[] | null }

const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as {
  places: number; output: string;
  window: { from: string; until: string; slots: number };
};
const rawLines = readFileSync(join(inputDir, 'history.ndjson'), 'utf8').trim().split('\n');
const snaps: Snap[] = rawLines.map(l => JSON.parse(l));
const PLACES = task.places;
const N = snaps.length;

// --- For each place, compute which games have historically occupied it ---
// Use exact rank match AND nearby ranks with Gaussian kernel
// Recency-weighted: recent appearances matter more

const halfLife = 7; // snapshots for recency decay

// Build per-game histories
const gameObs = new Map<number, Array<{ idx: number; rank: number; playing: number }>>();
for (let i = 0; i < N; i++) {
  for (const g of snaps[i].up_and_coming) {
    if (!gameObs.has(g.universe_id)) gameObs.set(g.universe_id, []);
    gameObs.get(g.universe_id)!.push({ idx: i, rank: g.rank, playing: g.playing });
  }
}

// Current state
const lastSnap = snaps[N - 1];
const chartSize = lastSnap.up_and_coming.length;
const prevSnap = N >= 2 ? snaps[N - 2] : null;
const prevMap = prevSnap ? new Map(prevSnap.up_and_coming.map(g => [g.universe_id, g])) : null;

// --- Compute probability for each (game, place) ---
// Strategy: for each place P, score each game based on:
// 1. How often it has been at rank P (exact match, high weight)
// 2. How often it has been near rank P (Gaussian kernel)
// 3. Its current proximity to P (persistence)
// 4. Momentum (rising games get boost at lower places)
// Then normalize with calibrated total

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];

for (let place = 1; place <= PLACES; place++) {
  const scored: Array<{ uid: number; w: number }> = [];

  for (const [uid, obs] of gameObs) {
    let w = 0;
    const currentGame = lastSnap.up_and_coming.find(g => g.universe_id === uid);
    const currentRank = currentGame?.rank ?? 999;

    // 1. Historical rank proximity (recency-weighted Gaussian)
    for (const o of obs) {
      const age = N - 1 - o.idx;
      const recency = Math.exp(-age * Math.LN2 / halfLife);

      // Exact match gets big bonus, nearby gets Gaussian
      const diff = place - o.rank;
      if (diff === 0) {
        w += recency * 2.0; // exact match bonus
      } else {
        const sigma = 3.5;
        w += recency * Math.exp(-(diff * diff) / (2 * sigma * sigma));
      }
    }

    // 2. Current rank persistence
    if (currentRank <= chartSize) {
      const diff = place - currentRank;
      const sigma = 4.0;
      // Weight based on observed persistence rates
      // Place 1: 76%, Place 2-5: ~48%, Place 6-10: ~30%
      const persRate = place <= 1 ? 0.76 : place <= 5 ? 0.48 : 0.30;
      w += persRate * 3.0 * Math.exp(-(diff * diff) / (2 * sigma * sigma));
    }

    // 3. Momentum: player growth → likely to climb
    if (currentGame && prevMap) {
      const prev = prevMap.get(uid);
      if (prev && prev.playing > 100) {
        const growth = (currentGame.playing - prev.playing) / prev.playing;
        if (growth > 0.1 && place < currentRank) {
          // Growing games get boost at higher positions
          w += growth * 1.0 * Math.exp(-((place - Math.max(1, currentRank - 3)) ** 2) / 18);
        }
      }
    }

    if (w > 0.01) scored.push({ uid, w });
  }

  // Sort by weight, take top candidates
  scored.sort((a, b) => b.w - a.w);

  // Use a softmax-like normalization to get probabilities
  // Temperature controls concentration: lower = more concentrated
  const temperature = 0.8;
  const topCands = scored.slice(0, 40); // generous coverage

  // Apply temperature
  const maxW = topCands[0]?.w ?? 1;
  for (const c of topCands) {
    c.w = Math.pow(c.w / maxW, 1 / temperature);
  }

  const totalW = topCands.reduce((a, c) => a + c.w, 0);
  const namedBudget = 0.93; // leave only 7% unnamed

  for (const c of topCands) {
    const p = (c.w / totalW) * namedBudget;
    if (p >= 0.005) {
      probabilities.push({
        universe_id: c.uid,
        place,
        p: Math.round(p * 10000) / 10000,
      });
    }
  }
}

// Top 10: pick games with highest total probability across all places
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
console.log(`Forecast: ${probabilities.length} probs, budget per place:`);
for (let p = 1; p <= PLACES; p++) {
  const entries = probabilities.filter(e => e.place === p);
  const sum = entries.reduce((a, e) => a + e.p, 0);
  console.log(`  P${p}: ${entries.length} games, sum=${sum.toFixed(4)}, top=${entries.sort((a,b) => b.p - a.p)[0]?.p.toFixed(4)}`);
}
