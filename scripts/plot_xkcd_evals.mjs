/**
 * Render README charts with chart.xkcd (https://github.com/timqian/chart.xkcd).
 *
 * Quality charts use a 0 to 1 axis. The library's default tick format would
 * print those fractions as milli-units, so the bar-chart copy uses one decimal
 * place instead. Grouped bars are drawn because the library otherwise keeps
 * only the first series.
 *
 * Usage (from the repo root, after `npm install chart.xkcd` in a working dir):
 *   CHART_XKCD_JS=/path/to/chart.xkcd.min.js node scripts/plot_xkcd_evals.mjs
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const chartJs =
  process.env.CHART_XKCD_JS ||
  "/tmp/xkcd-charts/node_modules/chart.xkcd/dist/chart.xkcd.min.js";
const outDir = join(root, "reports/figures/xkcd");

const qualityMethods = [
  ["minibatchkmeans", "minibatch-kmeans"],
  ["flashkmeans", "flash-kmeans"],
  ["fastkmeans", "fastkmeans"],
  ["faisskmeans", "Faiss"],
];
const datasetOrder = ["worms_64", "birch1", "birch2", "digits", "fashion"];

const perfMethods = [
  ["minibatchkmeans", "minibatch-kmeans"],
  ["flashkmeans", "flash-kmeans"],
  ["fastkmeans", "fastkmeans"],
  ["faisskmeans", "Faiss"],
];

function loadJson(rel) {
  return JSON.parse(readFileSync(join(root, rel), "utf8"));
}

function round(n, digits) {
  const p = 10 ** digits;
  return Math.round(n * p) / p;
}

function html(title, xLabel, yLabel, labels, values) {
  const payload = JSON.stringify({
    title,
    xLabel,
    yLabel,
    data: { labels, datasets: [{ data: values }] },
    options: {
      yTickCount: 5,
      fontFamily: "Comic Neue, Comic Sans MS, cursive",
    },
  });
  return `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: white; }
  #box { width: 1000px; height: 620px; }
</style>
</head>
<body>
<div id="box"><svg id="c"></svg></div>
<script src="${chartJs}"></script>
<script>
new chartXkcd.Bar(document.querySelector("#c"), ${payload});
</script>
</body>
</html>`;
}

function shoot(name, title, xLabel, yLabel, labels, values) {
  const dir = mkdtempSync(join(tmpdir(), "xkcd-"));
  const page = join(dir, "chart.html");
  const png = join(outDir, `${name}.png`);
  writeFileSync(page, html(title, xLabel, yLabel, labels, values));
  execFileSync("google-chrome", [
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--hide-scrollbars",
    "--window-size=1000,620",
    `--screenshot=${png}`,
    page,
  ], { stdio: "ignore" });
  rmSync(dir, { recursive: true, force: true });
  return png;
}

const quality = loadJson("results_eval.json");
const byKey = new Map(quality.map((row) => [`${row.dataset}:${row.method}`, row]));

function groupedBarChartJs(unitScale) {
  let source = readFileSync(chartJs, "utf8");
  if (unitScale) {
    source = source
      .replace(
        "r=Ln().domain([0,Math.max(...i)]).range([this.height,0])",
        "r=Ln().domain([0,1]).range([this.height,0])",
      )
      .replace('.ticks(i,"s")', '.ticks(i,".1f")');
  } else {
    source = source.replace('.ticks(i,"s")', '.ticks(i,".0f")');
  }
  const marker = 'e.selectAll(".xkcd-chart-bar")';
  const start = source.indexOf(marker) - 1;
  const end = source.indexOf("}update(){}},StackedBar");
  if (source[start] !== "," || end < start) {
    throw new Error("could not find the chart.xkcd bar drawing code");
  }
  const grouped = `;{const __series=this.data.datasets,__count=__series.length,__band=n.bandwidth(),__gap=__count>1?4:0,__width=(__band-__gap*(__count-1))/__count;__series.forEach((ds,si)=>{e.selectAll(".xkcd-chart-bar-"+si).data(ds.data).enter().append("rect").attr("class","xkcd-chart-bar").attr("x",(val,i)=>n(this.data.labels[i])+si*(__width+__gap)).attr("width",Math.max(__width,1)).attr("y",val=>r(val)).attr("height",val=>this.height-r(val)).attr("fill",this.options.dataColors[si%this.options.dataColors.length]).attr("pointer-events","all").attr("stroke",this.options.strokeColor).attr("stroke-width",2).attr("rx",2).attr("filter",this.filter)});this.options.showLegend&&__count>1&&di(e,{items:__series.map((ds,si)=>({color:this.options.dataColors[si%this.options.dataColors.length],text:""+(ds.label||"")})),position:this.options.legendPosition,unxkcdify:this.options.unxkcdify,parentWidth:this.width,parentHeight:this.height,strokeColor:this.options.strokeColor,backgroundColor:this.options.backgroundColor})}`;
  const dir = mkdtempSync(join(tmpdir(), "xkcd-bar-"));
  const file = join(dir, "chart.xkcd.bar.js");
  writeFileSync(file, source.slice(0, start) + grouped + source.slice(end));
  return file;
}

const groupedBarJs = groupedBarChartJs(true);
const memoryBarJs = groupedBarChartJs(false);

function coloredBarJs() {
  const source = readFileSync(chartJs, "utf8");
  const layout = 'n=Yt().range([0,this.width]).domain(this.data.labels).padding(.4),i=this.data.datasets.reduce((t,n)=>t.concat(n.data),[]),r=Ln().domain([0,Math.max(...i)]).range([this.height,0]),e=this.chart.append("g");ii(e,{xScale:n,tickCount:3,moveDown:this.height,fontFamily:this.fontFamily,unxkcdify:this.options.unxkcdify,stroke:this.options.strokeColor}),ri(e,{yScale:r,tickCount:this.options.yTickCount||3,fontFamily:this.fontFamily,unxkcdify:this.options.unxkcdify,stroke:this.options.strokeColor}),e.selectAll(".xkcd-chart-bar").data(this.data.datasets[0].data).enter().append("rect").attr("class","xkcd-chart-bar").attr("x",(t,i)=>n(this.data.labels[i])).attr("width",n.bandwidth()).attr("y",t=>r(t)).attr("height",t=>this.height-r(t)).attr("fill","none")';
  const horizontal = 'n=Yt().range([0,this.height]).domain(this.data.labels).padding(.35),i=this.data.datasets.reduce((t,n)=>t.concat(n.data),[]),r=Ln().domain([0,Math.max(...i)*1.05]).range([0,this.width]),e=this.chart.append("g");ii(e,{xScale:r,tickCount:this.options.yTickCount||5,moveDown:this.height,fontFamily:this.fontFamily,unxkcdify:this.options.unxkcdify,stroke:this.options.strokeColor}),e.append("g").call(ni(4,n).tickSize(0).tickPadding(8)),e.selectAll(".tick > text").style("font-family",this.fontFamily).style("font-size","16").style("fill",this.options.strokeColor),e.selectAll(".domain").attr("filter",this.options.unxkcdify?null:"url(#xkcdify)").style("stroke",this.options.strokeColor),e.selectAll(".xkcd-chart-bar").data(this.data.datasets[0].data).enter().append("rect").attr("class","xkcd-chart-bar").attr("y",(t,i)=>n(this.data.labels[i])).attr("height",n.bandwidth()).attr("x",0).attr("width",t=>r(t)).attr("fill",(t,i)=>this.options.dataColors[i])';
  const restore = '.on("mouseout",(n,i,r)=>{ct(r[i]).attr("fill","none")';
  const margin = "r&&(this.yLabel=r,li.left=70)";
  if (!source.includes(layout) || !source.includes(restore) || !source.includes(margin)) {
    throw new Error("could not find the chart.xkcd bar layout");
  }
  const dir = mkdtempSync(join(tmpdir(), "xkcd-color-bar-"));
  const file = join(dir, "chart.xkcd.color-bar.js");
  writeFileSync(
    file,
    source
      .replace(layout, horizontal)
      .replace(restore, '.on("mouseout",(n,i,r)=>{ct(r[i]).attr("fill",this.options.dataColors[i])')
      .replace(margin, "r&&(this.yLabel=r),li.left=180"),
  );
  return file;
}

const coloredBar = coloredBarJs();

function perfChartJs() {
  const source = readFileSync(chartJs, "utf8");
  const scale = 'const s=Ln().domain([Math.min(...r),Math.max(...r)]).range([this.height,0])';
  const curve = '.curve($i);u.selectAll(".xkcd-chart-xyline")';
  if (!source.includes(scale) || !source.includes(curve)) {
    throw new Error("could not find the chart.xkcd XY scale");
  }
  const logScale = `const s=function(__h){const ys=r.filter(v=>v>0);const lo=Math.min(...ys)/1.4;const hi=Math.max(...ys)*1.35;function make(domain,range){const f=v=>{const t=(Math.log10(v)-Math.log10(domain[0]))/(Math.log10(domain[1])-Math.log10(domain[0]));return range[0]+t*(range[1]-range[0])};f.domain=()=>domain;f.range=()=>range;f.copy=()=>make(domain.slice(),range.slice());f.ticks=()=>{const out=[];const a=Math.log10(domain[0]);const b=Math.log10(domain[1]);for(let p=Math.floor(a);p<=Math.ceil(b);p++){for(const m of [1,2,5]){const v=m*10**p;if(v>=domain[0]&&v<=domain[1])out.push(v)}}return out};f.tickFormat=()=>v=>v>=10?String(Math.round(v)):String(Math.round(v*10)/10);return f}return make([lo,hi],[__h,0])}(this.height)`;
  const dir = mkdtempSync(join(tmpdir(), "xkcd-perf-"));
  const file = join(dir, "chart.xkcd.perf.js");
  writeFileSync(
    file,
    source.replace(scale, logScale).replace(curve, ';u.selectAll(".xkcd-chart-xyline")'),
  );
  return file;
}

const perfJs = perfChartJs();

function qualityChart(metric, title, yLabel) {
  const datasets = qualityMethods.map(([method, label]) => ({
    label,
    data: datasetOrder.map((dataset) => {
      const row = byKey.get(`${dataset}:${method}`);
      if (!row || row[`best_${metric}`] == null) {
        throw new Error(`missing ${metric} for ${dataset} ${method}`);
      }
      return round(row[`best_${metric}`], 3);
    }),
  }));
  const payload = JSON.stringify({
    title,
    xLabel: "Dataset",
    yLabel,
    data: {
      labels: datasetOrder.map((name) => (name === "worms_64" ? "worms" : name)),
      datasets,
    },
    options: {
      yTickCount: 5,
      showLegend: true,
      fontFamily: "Comic Neue, Comic Sans MS, cursive",
      dataColors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    },
  });
  const page = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: white; }
  #box { width: 1000px; height: 620px; }
</style>
</head>
<body>
<div id="box"><svg id="c"></svg></div>
<script src="${groupedBarJs}"></script>
<script>
const config = ${payload};
config.options.legendPosition = chartXkcd.config.positionType.upRight;
new chartXkcd.Bar(document.querySelector("#c"), config);
</script>
</body>
</html>`;
  return shootPage(`panel_quality_${metric}`, page, 1000, 620);
}

function shootPage(name, pageHtml, width = 1000, height = 620) {
  const dir = mkdtempSync(join(tmpdir(), "xkcd-"));
  const page = join(dir, "chart.html");
  const png = join(outDir, `${name}.png`);
  writeFileSync(page, pageHtml);
  execFileSync("google-chrome", [
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--hide-scrollbars",
    `--window-size=${width},${height}`,
    `--screenshot=${png}`,
    page,
  ], { stdio: "ignore" });
  rmSync(dir, { recursive: true, force: true });
  return png;
}

const scaling = loadJson("reports/figures/perf_scaling_n.json");

execFileSync("mkdir", ["-p", outDir]);

const ariPanels = [qualityChart("ari", "Adjusted Rand index", "ARI")];
const nmiPanels = [qualityChart("nmi", "Normalized mutual information", "NMI")];

function wallBars() {
  const n = 1_048_576;
  const labels = [];
  const values = [];
  for (const [method, label] of perfMethods) {
    const row = scaling.find((item) => item.method === method && item.n === n);
    if (!row) {
      throw new Error(`missing wall time for ${method} n=${n}`);
    }
    labels.push(label);
    values.push(round((row.wall_time_s * 1e6) / n, 2));
  }
  const payload = JSON.stringify({
    title: "Seconds per million points",
    xLabel: "Seconds",
    yLabel: "",
    data: { labels, datasets: [{ data: values }] },
    options: {
      yTickCount: 5,
      fontFamily: "Comic Neue, Comic Sans MS, cursive",
      dataColors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    },
  });
  const page = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: white; }
  #box { width: 1000px; height: 620px; }
</style>
</head>
<body>
<div id="box"><svg id="c"></svg></div>
<script src="${coloredBar}"></script>
<script>
new chartXkcd.Bar(document.querySelector("#c"), ${payload});
</script>
</body>
</html>`;
  return shootPage("panel_wall_vs_n", page, 1000, 620);
}

wallBars();
function memoryBars() {
  const labels = ["66k", "262k", "524k", "1.0M"];
  const sizes = [65_536, 262_144, 524_288, 1_048_576];
  const datasets = perfMethods.map(([method, label]) => ({
    label,
    data: sizes.map((n) => {
      const row = scaling.find((item) => item.method === method && item.n === n);
      if (!row) {
        throw new Error(`missing memory for ${method} n=${n}`);
      }
      return round(row.gpu_process_peak_memory_mib_nvitop / 1024, 2);
    }),
  }));
  const payload = JSON.stringify({
    title: "Peak GPU memory",
    xLabel: "Points",
    yLabel: "GiB",
    data: { labels, datasets },
    options: {
      yTickCount: 4,
      showLegend: true,
      fontFamily: "Comic Neue, Comic Sans MS, cursive",
      dataColors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    },
  });
  const page = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; background: white; }
  #box { width: 1000px; height: 620px; }
</style>
</head>
<body>
<div id="box"><svg id="c"></svg></div>
<script src="${memoryBarJs}"></script>
<script>
const config = ${payload};
config.options.legendPosition = chartXkcd.config.positionType.upLeft;
new chartXkcd.Bar(document.querySelector("#c"), config);
</script>
</body>
</html>`;
  return shootPage("panel_mem_bars", page, 1000, 620);
}

memoryBars();

const manifest = {
  ariPanels,
  nmiPanels,
  wallVsN: join(outDir, "panel_wall_vs_n.png"),
  memBars: join(outDir, "panel_mem_bars.png"),
};
writeFileSync(join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2));
execFileSync(
  "conda",
  ["run", "-n", "kmeans-bench", "python", join(root, "scripts/stitch_xkcd_panels.py")],
  { stdio: "inherit" },
);
