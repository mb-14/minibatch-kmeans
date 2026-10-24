/**
 * Render Algorithm 1 from Sculley (2010) as a hand-drawn figure for the README.
 *
 * Comic Neue has no glyphs for the arrow, set-membership, or eta symbols in the
 * paper, so the figure spells them as "=", "in", and "eta". Line numbers match
 * the paper so the README can refer to them.
 *
 * Usage (from the repo root):
 *   node scripts/plot_xkcd_algorithm.mjs
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const png = join(root, "reports/figures/xkcd/minibatch_kmeans_algorithm.png");

// [indent level, code, comment]
const lines = [
  [0, "Given: k, batch size b, iterations t, data set X", ""],
  [0, "Set each center c in C to a random x from X", ""],
  [0, "v = 0", ""],
  [0, "<b>for</b> i = 1 to t <b>do</b>", ""],
  [1, "M = b examples picked at random from X", ""],
  [1, "<b>for</b> x in M <b>do</b>", ""],
  [2, "d[x] = f(C, x)", "cache the center nearest to x"],
  [1, "<b>end for</b>", ""],
  [1, "<b>for</b> x in M <b>do</b>", ""],
  [2, "c = d[x]", "get the cached center for x"],
  [2, "v[c] = v[c] + 1", "update the per-center count"],
  [2, "eta = 1 / v[c]", "per-center learning rate"],
  [2, "c = (1 - eta) c + eta x", "take a gradient step"],
  [1, "<b>end for</b>", ""],
  [0, "<b>end for</b>", ""],
];

const width = 1000;
const lineHeight = 40;
const top = 112;
const height = top + lines.length * lineHeight + 30;

const rows = lines
  .map(([indent, code, comment], i) => {
    const y = top + i * lineHeight;
    const note = comment
      ? `<text x="580" y="${y}" class="note">// ${comment}</text>`
      : "";
    return `<text x="40" y="${y}" class="num">${i + 1}:</text>
<text x="${95 + indent * 36}" y="${y}">${code.replace(/<b>(.*?)<\/b>/g, '<tspan class="kw">$1</tspan>')}</text>
${note}`;
  })
  .join("\n");

const page = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: white; }
  text { font-family: "Comic Neue", "Comic Sans MS", cursive; font-size: 24px; fill: #222; }
  .title { font-size: 30px; font-weight: bold; }
  .kw { font-weight: bold; }
  .num { fill: #888; }
  .note { fill: #3b6fb6; }
  .ink { fill: none; stroke: #222; stroke-width: 2.5; stroke-linecap: round; }
</style>
</head>
<body>
<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="xkcdify" filterUnits="userSpaceOnUse" x="-5" y="-5" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="0.05" result="noise"/>
      <feDisplacementMap scale="5" xChannelSelector="R" yChannelSelector="G" in="SourceGraphic" in2="noise"/>
    </filter>
  </defs>
  <rect class="ink" x="12" y="12" width="${width - 24}" height="${height - 24}" rx="6" filter="url(#xkcdify)"/>
  <text x="40" y="58" class="title">Algorithm 1: Mini-batch k-means</text>
  <line class="ink" x1="30" y1="78" x2="${width - 30}" y2="78" filter="url(#xkcdify)"/>
  ${rows}
</svg>
</body>
</html>`;

const dir = mkdtempSync(join(tmpdir(), "xkcd-alg-"));
const html = join(dir, "algorithm.html");
writeFileSync(html, page);
execFileSync("google-chrome", [
  "--headless",
  "--disable-gpu",
  "--no-sandbox",
  "--hide-scrollbars",
  "--force-device-scale-factor=2",
  `--window-size=${width},${height}`,
  `--screenshot=${png}`,
  html,
], { stdio: "ignore" });
rmSync(dir, { recursive: true, force: true });
console.log(png);
