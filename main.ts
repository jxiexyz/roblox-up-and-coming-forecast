// roblox-up-and-coming-forecast-01: Calibrated rank forecast with concentrated probability
// Key insight: ~7-8 candidates per place, top getting 20-30%, unnamed mass ~15%
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

// --- Build game profiles ---
const gameObs = new Map<number, Array<{ idx: number; rank: number; playing: number }>>();
for (let i = 0; i < N; i++) {
  for (const g of snaps[i].up_and_coming) {
    if (!gameObs.has(g.universe_id)) gameObs.set(g.universe_id, []);
    gameObs.get(g.universe_id)!.push({ idx: i, rank: g.rank, playing: g.playing });
  }
}

const lastSnap = snaps[N - 1];
const chartSize = lastSnap.up_and_coming.length;
const prevSnap = N >= 2 ? snaps[N - 2] : null;
const prevMap = prevSnap ? new Map(prevSnap.up_and_coming.map(g => [g.universe_id, g])) : null;

// --- Score each game for each place ---
// Blend of:
// A) Recency-weighted exact-rank occupancy (strongest signal)
// B) Current rank persistence with rank-specific decay
// C) Trend/momentum (rising games)

const halfLife = 6; // snapshots (~1.5 days)

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];

for (let place = 1; place <= PLACES; place++) {
  const scored: Array<{ uid: number; w: number }> = [];

  for (const [uid, obs] of gameObs) {
    const currentGame = lastSnap.up_and_coming.find(g => g.universe_id === uid);
    const currentRank = currentGame?.rank ?? 999;
    let w = 0;

    // A) Historical occupancy at this exact place (recency-weighted)
    // Also include nearby ranks with strong decay
    for (const o of obs) {
      const age = N - 1 - o.idx;
      const recency = Math.exp(-age * Math.LN2 / halfLife);
      const diff = Math.abs(place - o.rank);

      if (diff === 0) {
        w += recency * 5.0;  // exact match: very strong
      } else if (diff <= 2) {
        w += recency * 2.0;  // 1-2 ranks away: strong
      } else if (diff <= 5) {
        w += recency * 0.5;  // 3-5 ranks away: moderate
      } else if (diff <= 10) {
        w += recency * 0.1;  // 6-10 ranks away: weak
      }
    }

    // B) Current rank persistence
    // How likely is the current occupant to stay?
    if (currentRank <= chartSize) {
      const diff = Math.abs(place - currentRank);
      // Observed persistence: rank 1 stays 76%, rank 2-5 ~48%, rank 6-10 ~35%
      // But this decays over 72 hours → average persistence
      const basePers = place <= 1 ? 0.30 : place <= 5 ? 0.20 : 0.15;
      if (diff === 0) {
        w += basePers * 8.0;
      } else if (diff <= 2) {
        w += basePers * 3.0;
      } else if (diff <= 5) {
        w += basePers * 0.8;
      }
    }

    // C) Momentum: growing games get boost at better (lower) ranks
    if (currentGame && prevMap) {
      const prev = prevMap.get(uid);
      if (prev && prev.playing > 100) {
        const growth = (currentGame.playing - prev.playing) / prev.playing;
        if (growth > 0.15 && place < currentRank && currentRank <= 30) {
          const climbDist = currentRank - place;
          if (climbDist <= 10) {
            w += growth * 2.0 * Math.exp(-climbDist * 0.3);
          }
        }
      }
    }

    if (w > 0.05) scored.push({ uid, w });
  }

  // Sort, take top candidates, apply power concentration
  scored.sort((a, b) => b.w - a.w);

  // Take top 15-20 candidates (enough for coverage, not too many)
  const topK = scored.slice(0, 18);

  // Power-law concentration: raise to power > 1
  const power = 2.5;  // stronger concentration
  for (const c of topK) c.w = Math.pow(c.w, power);

  const totalW = topK.reduce((a, c) => a + c.w, 0);

  // Named budget: leave 15% unnamed for newcomers and unknowns
  // 15% / ~85 unnamed games ≈ 0.18% each (better than 0.1% floor)
  const namedBudget = 0.85;

  for (const c of topK) {
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
console.log(`Forecast: ${probabilities.length} probs`);
for (let p = 1; p <= PLACES; p++) {
  const es = probabilities.filter(e => e.place === p).sort((a, b) => b.p - a.p);
  const sum = es.reduce((a, e) => a + e.p, 0);
  console.log(`  P${p}: ${es.length} games, sum=${sum.toFixed(3)}, top3=[${es.slice(0, 3).map(e => e.p.toFixed(3)).join(',')}]`);
}
