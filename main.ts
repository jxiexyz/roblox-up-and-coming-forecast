// roblox-up-and-coming-forecast-01: Concentrated rank-distribution forecaster
// with exponential recency, persistence blend, breakout detection
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface GameRow { rank: number; universe_id: number; name: string; playing: number }
interface Snapshot { at: string; up_and_coming: GameRow[]; top_trending: GameRow[] | null }

const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as {
  places: number; output: string;
  window: { from: string; until: string; slots: number };
  history_until: string;
};
const rawLines = readFileSync(join(inputDir, 'history.ndjson'), 'utf8').trim().split('\n');
const snaps: Snapshot[] = rawLines.map(l => JSON.parse(l));
const PLACES = task.places;
const N = snaps.length;

// --- Per-game profiles ---
interface Profile {
  uid: number;
  rankObs: Array<{ snapIdx: number; rank: number; playing: number }>;
  currentRank: number;
  momentum: number;
  tenure: number;
}

const profiles = new Map<number, Profile>();
for (let i = 0; i < N; i++) {
  for (const g of snaps[i].up_and_coming) {
    let p = profiles.get(g.universe_id);
    if (!p) {
      p = { uid: g.universe_id, rankObs: [], currentRank: 999, momentum: 0, tenure: 0 };
      profiles.set(g.universe_id, p);
    }
    p.rankObs.push({ snapIdx: i, rank: g.rank, playing: g.playing });
    p.tenure++;
  }
}

// Current state
const lastSnap = snaps[N - 1];
const prevSnap = N >= 2 ? snaps[N - 2] : null;
const prevMap = prevSnap ? new Map(prevSnap.up_and_coming.map(g => [g.universe_id, g])) : new Map<number, GameRow>();

for (const g of lastSnap.up_and_coming) {
  const p = profiles.get(g.universe_id)!;
  p.currentRank = g.rank;
  const prev = prevMap.get(g.universe_id);
  if (prev && prev.playing > 0) p.momentum = (g.playing - prev.playing) / prev.playing;
}

// --- Compute probability for each (game, place) ---
// Score = blend of:
//   A) Empirical: recency-weighted rank proximity from actual observations
//   B) Persistence: how close current rank is to this place
//   C) Breakout: extra probability at top places for rising games

const halfLife = 7; // snapshots for recency decay

function computeScore(prof: Profile, place: number): number {
  // A) Empirical rank occupancy with Gaussian kernel
  let empScore = 0;
  let empTotalW = 0;

  // Adaptive sigma: for stable games (low rank variance) use tight kernel
  // For volatile games use wider kernel
  const recentRanks = prof.rankObs.slice(-5).map(o => o.rank);
  const mean = recentRanks.reduce((a, b) => a + b, 0) / recentRanks.length;
  const std = Math.sqrt(recentRanks.reduce((a, b) => a + (b - mean) ** 2, 0) / recentRanks.length);
  const sigma = Math.max(2.5, Math.min(8, std * 2 + 1));

  for (const obs of prof.rankObs) {
    const age = N - 1 - obs.snapIdx;
    const recencyW = Math.exp(-age * Math.LN2 / halfLife);
    const diff = place - obs.rank;
    const gaussW = Math.exp(-(diff * diff) / (2 * sigma * sigma));
    empScore += recencyW * gaussW;
    empTotalW += recencyW;
  }
  empScore = empTotalW > 0 ? empScore / empTotalW : 0;

  // B) Persistence: Gaussian around current rank
  const persScore = prof.currentRank <= 200
    ? Math.exp(-((place - prof.currentRank) ** 2) / (2 * 5 * 5))
    : 0;

  // C) Breakout boost for top places
  let breakout = 0;
  if (prof.momentum > 0.15 && prof.currentRank <= 15 && place <= 10) {
    breakout = 0.15 * Math.exp(-((place - Math.max(1, prof.currentRank - 3)) ** 2) / 18);
  }
  // Rank-climbing detection
  if (recentRanks.length >= 3) {
    const climb = recentRanks[0] - recentRanks[recentRanks.length - 1];
    if (climb > 5 && place <= 10) {
      breakout += 0.1 * Math.exp(-((place - Math.max(1, prof.currentRank - 2)) ** 2) / 18);
    }
  }

  // Blend: 45% empirical, 35% persistence, 20% breakout
  return 0.45 * empScore + 0.35 * persScore + 0.20 * breakout;
}

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];

for (let place = 1; place <= PLACES; place++) {
  const scored: Array<{ uid: number; score: number }> = [];

  for (const [uid, prof] of profiles) {
    const s = computeScore(prof, place);
    if (s > 0.001) scored.push({ uid, score: s });
  }

  // Sort and keep top candidates
  scored.sort((a, b) => b.score - a.score);

  // Power-law concentration: raise scores to power > 1 to concentrate probability
  // Higher power = more concentrated on top candidates
  const power = 1.8;
  const topK = scored.slice(0, 30);
  for (const c of topK) c.score = Math.pow(c.score, power);

  const total = topK.reduce((a, c) => a + c.score, 0);
  const namedBudget = 0.88; // leave 12% unnamed

  for (const c of topK) {
    const p = (c.score / total) * namedBudget;
    if (p >= 0.005) {
      probabilities.push({
        universe_id: c.uid,
        place,
        p: Math.round(p * 10000) / 10000,
      });
    }
  }
}

// Top 10: games most likely to be anywhere in top 10
const top10Map = new Map<number, number>();
for (const entry of probabilities) {
  top10Map.set(entry.universe_id, (top10Map.get(entry.universe_id) || 0) + entry.p);
}
const top10 = [...top10Map.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, PLACES)
  .map(([uid]) => uid);

const result = { probabilities, top10 };
writeFileSync(join(outputDir, task.output), `${JSON.stringify(result)}\n`, 'utf8');
console.log(`Forecast: ${probabilities.length} probs, top10=${top10.join(',')}`);
