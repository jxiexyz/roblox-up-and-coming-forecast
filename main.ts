// roblox-up-and-coming-forecast-01: Gaussian rank-diffusion forecaster with player momentum
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface GameRow {
  rank: number;
  universe_id: number;
  name: string;
  playing: number;
}

interface Snapshot {
  up_and_coming: GameRow[];
}

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as { places: number; output: string };
const rawLines = readFileSync(join(inputDir, 'history.ndjson'), 'utf8').trim().split('\n');
const snapshots = rawLines.map((l) => JSON.parse(l) as Snapshot);

const latest = snapshots[snapshots.length - 1]!.up_and_coming;
const prior = snapshots[snapshots.length - 2]?.up_and_coming ?? [];

const priorMap = new Map<number, GameRow>();
for (const row of prior) {
  priorMap.set(row.universe_id, row);
}

const sigma = 3.0;
const sigmaSq2 = 2 * sigma * sigma;
const TOTAL_NAMED_P = 0.82;

const rankedGames = latest.map((row) => {
  const prev = priorMap.get(row.universe_id);
  const velocity = prev ? (row.playing - prev.playing) / Math.max(1, prev.playing) : 0;
  const drift = Math.max(-3, Math.min(3, velocity * 2));
  const meanRank = Math.max(1, Math.min(50, row.rank - drift));
  return { ...row, meanRank };
});

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];

for (let place = 1; place <= task.places; place++) {
  const weights: Array<{ universe_id: number; w: number }> = [];
  for (const game of rankedGames) {
    const diff = place - game.meanRank;
    const w = Math.exp(-(diff * diff) / sigmaSq2);
    weights.push({ universe_id: game.universe_id, w });
  }

  const sumW = weights.reduce((acc, x) => acc + x.w, 0);
  if (sumW > 0) {
    for (const item of weights) {
      const prob = Math.min(0.75, (item.w / sumW) * TOTAL_NAMED_P);
      if (prob >= 0.01) {
        probabilities.push({
          universe_id: item.universe_id,
          place,
          p: Math.round(prob * 10000) / 10000,
        });
      }
    }
  }
}

const top10 = rankedGames
  .sort((a, b) => a.meanRank - b.meanRank)
  .slice(0, task.places)
  .map((g) => g.universe_id);

const result = { probabilities, top10 };
writeFileSync(join(outputDir, task.output), `${JSON.stringify(result)}\n`, 'utf8');
console.log(`Forecast generated: ${probabilities.length} probabilities over top ${task.places}`);

