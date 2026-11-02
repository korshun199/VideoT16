"""Строит автономную вращаемую HTML-карту задержек VideoT16 по журналам.

Оси: X — время опыта, Y — длительность выбранного этапа в миллисекундах,
Z — доля кадров с такой длительностью в соответствующую секунду.
Поверхность — распределение задержек, плоскости — медиана, перцентили
и бюджет кадра для фактической частоты захвата.

HTML можно открыть обычным браузером, вращать мышью и приближать колесом.
Запуск из корня проекта:
  ./.venv/bin/python video_cap_buf/latency3d.py
  ./.venv/bin/python video_cap_buf/latency3d.py --metric capture_to_result_ms
  ./.venv/bin/python video_cap_buf/latency3d.py --run-id 20261029_095506_653618
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "tmp" / "video_cap_buf"

METRICS = {
    "inference_ms": ("Время инференса модели", "recognition"),
    "queue_wait_ms": ("Ожидание кадра в очереди распознавания", "recognition"),
    "capture_to_result_ms": ("От захвата до готовности результата", "recognition"),
    "capture_to_present_ms": ("От захвата до показа на экране", "display"),
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Построчно читает JSONL и указывает место ошибки при повреждённой записи."""

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: некорректный JSON: {error}") from error
            if isinstance(record, dict):
                yield record


def find_logs(run_id: str | None) -> tuple[Path, Path, Path]:
    """Находит комплект журналов с указанным ID либо последнюю полную сессию."""

    if run_id:
        paths = tuple(
            LOG_DIR / f"{kind}_{run_id}.jsonl"
            for kind in ("capture", "recognition", "display")
        )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Не найдены журналы: " + ", ".join(map(str, missing)))
        return paths  # type: ignore[return-value]

    sessions: list[tuple[float, tuple[Path, Path, Path]]] = []
    for capture_path in LOG_DIR.glob("capture_*.jsonl"):
        suffix = capture_path.stem.removeprefix("capture_")
        paths = (
            capture_path,
            LOG_DIR / f"recognition_{suffix}.jsonl",
            LOG_DIR / f"display_{suffix}.jsonl",
        )
        if all(path.is_file() for path in paths):
            sessions.append((max(path.stat().st_mtime for path in paths), paths))
    if not sessions:
        raise FileNotFoundError(f"В {LOG_DIR} нет полной сессии из трёх журналов")
    return max(sessions, key=lambda item: item[0])[1]


def valid_number(value: Any) -> float | None:
    """Возвращает конечное неотрицательное число или None для некорректного значения."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def load_samples(
    paths: tuple[Path, Path, Path], metric: str
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Связывает длительность выбранного этапа с временем исходного кадра."""

    capture_path, recognition_path, display_path = paths
    capture_stamps: dict[int, int] = {}
    capture_metadata: dict[str, Any] = {}
    for record in read_jsonl(capture_path):
        if record.get("event") == "capture_started":
            capture_metadata = record
        elif record.get("event") == "frame_captured":
            frame = record.get("frame")
            stamp = record.get("captured_ns")
            if isinstance(frame, int) and isinstance(stamp, int):
                capture_stamps[frame] = stamp

    if not capture_stamps:
        raise ValueError(f"В {capture_path} нет нумерованных кадров с captured_ns")
    origin_ns = min(capture_stamps.values())
    actual = capture_metadata.get("actual", {})
    fps = valid_number(actual.get("fps"))
    if not fps:
        requested = capture_metadata.get("requested", {})
        fps = valid_number(requested.get("fps"))

    source_path = recognition_path if METRICS[metric][1] == "recognition" else display_path
    target_event = "frame_recognized" if source_path == recognition_path else "frame_displayed"
    samples: list[tuple[float, float]] = []
    fallback_timestamps = 0

    for record in read_jsonl(source_path):
        if record.get("event") != target_event:
            continue
        value = valid_number(record.get(metric))
        frame = record.get("frame")
        stamp = record.get("captured_ns") if target_event == "frame_displayed" else None
        if not isinstance(stamp, int) and isinstance(frame, int):
            stamp = capture_stamps.get(frame)
        if not isinstance(stamp, int):
            stamp = record.get("monotonic_ns")
            fallback_timestamps += 1
        if value is not None and isinstance(stamp, int) and stamp >= origin_ns:
            samples.append(((stamp - origin_ns) / 1_000_000_000, value))

    if not samples:
        raise ValueError(f"В {source_path} нет измерений {metric} для построения графика")

    metadata = {
        "files": paths,
        "run_id": capture_path.stem.removeprefix("capture_"),
        "capture_metadata": capture_metadata,
        "captured_frames": len(capture_stamps),
        "fallback_timestamps": fallback_timestamps,
        "fps": fps,
    }
    return samples, metadata


def print_summary(samples: list[tuple[float, float]], metadata: dict[str, Any], metric: str) -> None:
    """Печатает число измерений и медиану с основными перцентилями."""

    import numpy as np

    values = [value for _, value in samples]
    percentiles = {p: float(np.percentile(values, p)) for p in (50, 90, 95, 98)}
    print(f"Сессия: {metadata['run_id']}")
    print(f"Метрика: {METRICS[metric][0]} ({metric})")
    print(f"Кадров захвачено: {metadata['captured_frames']}; измерений: {len(samples)}")
    if metadata["fps"]:
        print(f"Фактический бюджет кадра: {1000.0 / metadata['fps']:.2f} мс ({metadata['fps']:g} FPS)")
    print(
        "Статистика: "
        + ", ".join(f"P{p}={value:.2f} мс" for p, value in percentiles.items())
    )


def make_target_samples(
    samples: list[tuple[float, float]], time_bin: float, fps: float | None
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Создаёт синтетическое целевое распределение относительно бюджета кадра."""

    import numpy as np

    duration = max(time for time, _ in samples)
    effective_bin = min(time_bin, duration / 2) if duration > 0 else time_bin
    interval_count = max(1, math.ceil(duration / effective_bin))
    frame_period = 1000.0 / fps if fps else 1000.0 / 30.0
    # Это инженерный ориентир для наглядного сценария, а не прогноз мощности ПК.
    target_p50 = frame_period * 0.60
    target_p90 = frame_period * 0.85
    target_p98 = frame_period
    target_tail = frame_period * 1.25
    anchors = np.asarray([0, 500, 900, 980, 1000], dtype=float)
    latencies = np.asarray(
        [frame_period * 0.35, target_p50, target_p90, target_p98, target_tail],
        dtype=float,
    )
    target: list[tuple[float, float]] = []
    for interval in range(interval_count):
        start = interval * effective_bin
        width = min(effective_bin, max(duration - start, 0.0))
        for offset in range(1001):
            rank = (offset * 613) % 1001
            elapsed = start + ((offset + 0.5) / 1001) * width
            target.append((elapsed, float(np.interp(rank, anchors, latencies))))
    goals = {"p50": target_p50, "p90": target_p90, "p98": target_p98, "frame_period": frame_period}
    return target, goals


def print_quality_assessment(
    samples: list[tuple[float, float]], goals: dict[str, float]
) -> None:
    """Сравнивает измеренные перцентили с ориентирами и интервалом кадра."""

    import numpy as np

    values = np.asarray([value for _, value in samples], dtype=float)
    actual = {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 98)}
    within_budget = float(np.mean(values <= goals["frame_period"]) * 100)
    print("Оценка задержки (не оценка точности распознавания):")
    for percentile in ("p50", "p90", "p98"):
        status = "ЦЕЛЬ ВЫПОЛНЕНА" if actual[percentile] <= goals[percentile] else "выше цели"
        print(
            f"  {percentile.upper()}: {actual[percentile]:.2f} мс; "
            f"цель ≤ {goals[percentile]:.2f} мс — {status}"
        )
    print(f"  В бюджете кадра ({goals['frame_period']:.2f} мс): {within_budget:.1f}% измерений")
    print("  Пропуски и ложные обнаружения без разметки кадров здесь не оценить.")


def make_surface(
    samples: list[tuple[float, float]], time_bin: float, latency_bin: float
) -> tuple[Any, Any, Any, float, float, float]:
    """Строит матрицу долей кадров по секундам и интервалам длительности."""

    import numpy as np

    elapsed = np.asarray([sample[0] for sample in samples], dtype=float)
    latency = np.asarray([sample[1] for sample in samples], dtype=float)
    max_time = float(elapsed.max())
    max_latency = float(latency.max())
    display_duration = max_time if max_time > 0 else time_bin
    effective_time_bin = min(time_bin, display_duration / 2)
    time_edges = np.arange(0.0, display_duration, effective_time_bin, dtype=float)
    if len(time_edges) < 2:
        time_edges = np.asarray([0.0, display_duration / 2], dtype=float)
    if time_edges[-1] < display_duration:
        time_edges = np.append(time_edges, display_duration)
    latency_limit = max(latency_bin, math.ceil((max_latency + 1e-9) / latency_bin) * latency_bin)
    latency_edges = np.arange(0, latency_limit + latency_bin * 1.01, latency_bin, dtype=float)
    counts, _, _ = np.histogram2d(elapsed, latency, bins=(time_edges, latency_edges))
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts * 100.0, totals, out=np.zeros_like(counts), where=totals > 0)
    x_centers = (time_edges[:-1] + time_edges[1:]) / 2
    y_centers = (latency_edges[:-1] + latency_edges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    z_grid = share.T
    return x_grid, y_grid, z_grid, max_time, max_latency, effective_time_bin


def write_html_chart(
    samples: list[tuple[float, float]],
    metadata: dict[str, Any],
    metric: str,
    time_bin: float,
    latency_bin: float,
    output: Path,
) -> Path:
    """Создаёт автономную вращаемую 3D-страницу без сервера и внешних библиотек."""

    import numpy as np

    x_grid, y_grid, z_grid, max_time, max_latency, effective_time_bin = make_surface(
        samples, time_bin, latency_bin
    )
    values = [value for _, value in samples]
    percentiles = {
        f"p{p}": float(np.percentile(values, p))
        for p in (50, 90, 95, 98)
    }
    duration = max_time if max_time > 0 else effective_time_bin * 2
    frame_period = 1000.0 / metadata["fps"] if metadata["fps"] else None
    if metric == "inference_ms" and metadata["fps"]:
        frame_label = "Бюджет одного инференса"
    else:
        frame_label = "Интервал входных кадров"

    payload = {
        "run_id": metadata["run_id"],
        "is_target": bool(metadata.get("is_target", False)),
        "target_note": metadata.get("target_note", ""),
        "metric": metric,
        "metric_title": METRICS[metric][0],
        "samples": len(samples),
        "captured_frames": metadata["captured_frames"],
        "fps": metadata["fps"],
        "duration": duration,
        "max_latency": max(latency_bin, max_latency + latency_bin),
        "time_bin": effective_time_bin,
        "latency_bin": latency_bin,
        "x": x_grid[0].tolist(),
        "y": y_grid[:, 0].tolist(),
        "z": z_grid.tolist(),
        "percentiles": percentiles,
        "frame_period": frame_period,
        "frame_label": frame_label,
    }
    # Данные числовые, но экранируем последовательность закрывающего script-тега.
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = HTML_TEMPLATE.replace("__DATA__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VideoT16 — 3D-анализ задержек</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#121d2e;--line:#26364c;--text:#e7eef8;--muted:#9aabc0;--cyan:#5ed5ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 50% -20%,#1b3651 0,#0b1220 65%);color:var(--text);font:15px/1.45 system-ui,sans-serif}
header{padding:18px 24px 12px;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0 0 5px}#subtitle{color:var(--muted);font-size:13px}
main{max-width:1500px;margin:auto;padding:14px 18px 24px}.stats{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:12px}.stat{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px 12px;min-width:120px}.stat b{display:block;font-size:17px}.stat span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.chart{position:relative;background:linear-gradient(145deg,#101a2a,#0c1422);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 16px 45px #0005}canvas{width:100%;height:min(72vh,780px);min-height:460px;display:block;cursor:grab;touch-action:none}canvas.dragging{cursor:grabbing}
.hint{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:9px 13px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}.legend{display:flex;gap:16px;flex-wrap:wrap}.key{display:flex;align-items:center;gap:6px}.swatch{width:17px;height:3px;border-radius:3px;background:var(--c)}
.notice{display:inline-block;margin-top:9px;padding:5px 9px;border:1px solid #a5672a;border-radius:6px;background:#4a301c;color:#ffd59b;font-size:12px;font-weight:700}
@media(max-width:700px){main{padding:10px}header{padding:14px}canvas{height:65vh;min-height:360px}.stat{min-width:calc(50% - 8px)}}
</style>
</head>
<body>
<header><h1 id="title">VideoT16 · карта задержек</h1><div id="subtitle"></div><div class="notice" id="notice" hidden></div></header>
<main>
<section class="stats" id="stats"></section>
<section class="chart"><canvas id="plot" aria-label="Вращаемая 3D-карта задержек"></canvas>
<div class="hint"><div>Зажми левую кнопку и веди мышь — вращение · колесо — масштаб · двойной щелчок — исходный ракурс</div><div class="legend" id="legend"></div></div></section>
</main>
<script>
"use strict";
const D=__DATA__;
const canvas=document.getElementById("plot"),ctx=canvas.getContext("2d");
let yaw=-0.68,pitch=0.57,zoom=1,drag=null;
const COLORS={frame:"#66aaff",p50:"#36d399",p90:"#ffad4d",p98:"#ff5f87"};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function stats(){
  document.getElementById("title").textContent=D.is_target?"VideoT16 · целевой сценарий задержек":"VideoT16 · карта задержек";
  const notice=document.getElementById("notice");if(D.is_target){notice.hidden=false;notice.textContent=D.target_note;}
  const items=D.is_target?[["Сценарий","ЦЕЛЕВОЙ"],["Синтетических точек",D.samples],["P50",D.percentiles.p50.toFixed(1)+" мс"],["P90",D.percentiles.p90.toFixed(1)+" мс"],["P98",D.percentiles.p98.toFixed(1)+" мс"]]:[["Кадров захвачено",D.captured_frames],["Измерений",D.samples],["P50",D.percentiles.p50.toFixed(1)+" мс"],["P90",D.percentiles.p90.toFixed(1)+" мс"],["P98",D.percentiles.p98.toFixed(1)+" мс"]];
  if(D.fps)items.push(["Захват",D.fps.toFixed(1)+" FPS"]);
  document.getElementById("stats").innerHTML=items.map(([k,v])=>`<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
  document.getElementById("subtitle").textContent=`${D.is_target?"Основа длительности — опыт "+D.run_id+" · ":"Сессия "+D.run_id+" · "}${D.metric_title} · интервалы ${D.time_bin.toFixed(2)} с × ${D.latency_bin} мс · поверхность показывает долю точек в каждом интервале времени`;
  const entries=[],frameMatchesP98=D.frame_period&&Math.abs(D.frame_period-D.percentiles.p98)<.5;
  if(D.frame_period&&!frameMatchesP98)entries.push([D.frame_label+" · "+D.frame_period.toFixed(1)+" мс",COLORS.frame]);
  entries.push(["P50 · "+D.percentiles.p50.toFixed(1)+" мс",COLORS.p50],["P90 · "+D.percentiles.p90.toFixed(1)+" мс",COLORS.p90],[`${frameMatchesP98?"P98 / бюджет кадра":"P98"} · ${D.percentiles.p98.toFixed(1)} мс`,COLORS.p98]);
  document.getElementById("legend").innerHTML=entries.map(([name,color])=>`<span class="key"><i class="swatch" style="--c:${color}"></i>${name}</span>`).join("");
}
function resize(){const r=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();}
function project(x,y,z){
  const nx=(x/Math.max(D.duration,1e-9)-.5)*2,ny=(y/Math.max(D.max_latency,1e-9)-.5)*1.52,nz=(z/100)*1.08;
  const ca=Math.cos(yaw),sa=Math.sin(yaw),xr=nx*ca-ny*sa,yr=nx*sa+ny*ca;
  const cp=Math.cos(pitch),sp=Math.sin(pitch),vertical=yr*sp+nz*cp,depth=yr*cp-nz*sp;
  const w=canvas.clientWidth,h=canvas.clientHeight,s=Math.min(w*.43,h*.60)*zoom;
  return {x:w*.51+xr*s,y:h*.69-vertical*s,depth};
}
function colorAt(value){
  const stops=[[0,[68,1,84]],[.25,[59,82,139]],[.5,[33,145,140]],[.75,[94,201,98]],[1,[253,231,37]]],t=clamp(value/100,0,1);
  let i=0;while(i<stops.length-2&&t>stops[i+1][0])i++;const [a,ca]=stops[i],[b,cb]=stops[i+1],f=(t-a)/(b-a),c=ca.map((v,j)=>Math.round(v+(cb[j]-v)*f));return `rgb(${c.join(",")})`;
}
function path(points,color,width=1,dash=[]){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([]);}
function drawBase(){
  const xticks=5,yticks=5;
  for(let i=0;i<=xticks;i++){const x=D.duration*i/xticks;path([project(x,0,0),project(x,D.max_latency,0)],"#40516a66");path([project(x,0,0),project(x,0,100)],"#40516a55");ctx.fillStyle="#a8b7ca";ctx.font="11px system-ui";const p=project(x,0,0);ctx.fillText(x.toFixed(0)+"s",p.x-9,p.y+19);}
  for(let i=0;i<=yticks;i++){const y=D.max_latency*i/yticks;path([project(0,y,0),project(D.duration,y,0)],"#40516a66");ctx.fillStyle="#a8b7ca";ctx.font="11px system-ui";const p=project(0,y,0);ctx.fillText(y.toFixed(0),p.x-27,p.y+4);}
  path([project(0,0,0),project(D.duration,0,0)],"#9fb2c9",1.5);path([project(0,0,0),project(0,D.max_latency,0)],"#9fb2c9",1.5);path([project(0,0,0),project(0,0,100)],"#9fb2c9",1.5);
  ctx.fillStyle="#d2deed";ctx.font="12px system-ui";let p=project(D.duration*.78,0,0);ctx.fillText("Время опыта (с)",p.x,p.y+42);p=project(0,D.max_latency*.78,0);ctx.fillText("Задержка (мс)",p.x-85,p.y-8);p=project(0,0,100);ctx.fillText("Доля кадров (%)",p.x+7,p.y-12);
}
function drawSurface(){
  const cells=[];for(let j=0;j<D.y.length-1;j++)for(let i=0;i<D.x.length-1;i++){
    const points=[project(D.x[i],D.y[j],D.z[j][i]),project(D.x[i+1],D.y[j],D.z[j][i+1]),project(D.x[i+1],D.y[j+1],D.z[j+1][i+1]),project(D.x[i],D.y[j+1],D.z[j+1][i])];
    cells.push({points,depth:points.reduce((n,p)=>n+p.depth,0)/4,value:(D.z[j][i]+D.z[j][i+1]+D.z[j+1][i+1]+D.z[j+1][i])/4});
  }
  cells.sort((a,b)=>a.depth-b.depth);for(const cell of cells){ctx.beginPath();cell.points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle=colorAt(cell.value);ctx.globalAlpha=.86;ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle="#c3f4ff25";ctx.lineWidth=.55;ctx.stroke();}
}
function drawPlanes(){
  const planes=[],frameMatchesP98=D.frame_period&&Math.abs(D.frame_period-D.percentiles.p98)<.5;
  if(D.frame_period&&!frameMatchesP98)planes.push([D.frame_period,"Интервал кадров",COLORS.frame]);
  planes.push([D.percentiles.p50,"P50",COLORS.p50],[D.percentiles.p90,"P90",COLORS.p90],[D.percentiles.p98,frameMatchesP98?"P98 / бюджет кадра":"P98",COLORS.p98]);
  for(const [value,label,color] of planes){if(value>D.max_latency)continue;
    const plane=[];for(let i=0;i<=32;i++)plane.push(project(D.duration*i/32,value,0));path(plane,color,2,[7,5]);
    const anchor=project(D.duration*.98,value,0);ctx.fillStyle=color;ctx.font="bold 11px system-ui";ctx.fillText(`${label} ${value.toFixed(1)}ms`,anchor.x-58,anchor.y-6);
  }
}
function draw(){if(!ctx||!canvas.width)return;const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);drawBase();drawSurface();drawPlanes();}
canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY};canvas.classList.add("dragging");canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener("pointermove",e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};yaw+=dx*.009;pitch=clamp(pitch+dy*.008,-1.25,1.25);draw();});
function stopDrag(){drag=null;canvas.classList.remove("dragging");}canvas.addEventListener("pointerup",stopDrag);canvas.addEventListener("pointercancel",stopDrag);
canvas.addEventListener("wheel",e=>{e.preventDefault();zoom=clamp(zoom*Math.exp(-e.deltaY*.001),.55,2.4);draw();},{passive:false});
canvas.addEventListener("dblclick",()=>{yaw=-.68;pitch=.57;zoom=1;draw();});window.addEventListener("resize",resize);stats();resize();
</script>
</body>
</html>'''


def draw_chart(
    samples: list[tuple[float, float]],
    metadata: dict[str, Any],
    metric: str,
    time_bin: float,
    latency_bin: float,
    save_png: Path | None,
    no_window: bool,
) -> None:
    """Рисует вращаемую 3D-поверхность, временные плоскости и легенду."""

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    x_grid, y_grid, z_grid, max_time, max_latency, effective_time_bin = make_surface(
        samples, time_bin, latency_bin
    )
    display_duration = max_time if max_time > 0 else effective_time_bin * 2
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap=cm.viridis,
        norm=Normalize(vmin=0, vmax=100),
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )

    plane_x, plane_z = np.meshgrid([0.0, display_duration], [0.0, 100.0])
    values = [value for _, value in samples]
    planes = [
        (float(np.percentile(values, 50)), "Медиана", "#268a65"),
        (float(np.percentile(values, 90)), "P90", "#e28b26"),
        (float(np.percentile(values, 98)), "P98", "#d44f73"),
    ]
    if metadata["fps"]:
        if metric == "inference_ms":
            frame_label = f"Бюджет инференса @ {metadata['fps']:g} FPS"
        else:
            frame_label = f"Интервал входных кадров @ {metadata['fps']:g} FPS"
        planes.insert(0, (1000.0 / metadata["fps"], frame_label, "#3867d6"))

    for latency_ms, label, color in planes:
        plane_y = np.full_like(plane_x, latency_ms, dtype=float)
        ax.plot_surface(plane_x, plane_y, plane_z, color=color, alpha=0.075, linewidth=0, shade=False)
        ax.plot(
            [0.0, display_duration],
            [latency_ms, latency_ms],
            [0.0, 0.0],
            color=color,
            linewidth=2.5,
            linestyle="--",
            label=f"{label}: {latency_ms:.1f} мс",
        )

    ax.set_title(
        f"VideoT16: распределение задержек по времени\n"
        f"Сессия {metadata['run_id']} · {METRICS[metric][0]} · окно до {time_bin:g} с "
        f"(сейчас {effective_time_bin:g} с) / шаг {latency_bin:g} мс",
        pad=18,
    )
    ax.set_xlabel("Время от начала захвата (с)", labelpad=10)
    ax.set_ylabel("Длительность этапа (мс)", labelpad=10)
    ax.set_zlabel("Доля кадров в интервале (%)", labelpad=8)
    ax.set_xlim(0.0, display_duration)
    ax.set_ylim(0.0, max(latency_bin, max_latency + latency_bin))
    ax.set_zlim(0.0, 100.0)
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc="upper left", fontsize=8)

    fig.text(
        0.015,
        0.012,
        "Вращение: зажми левую кнопку мыши и двигай курсор. "
        "Высокий пик означает, что в этой секунде много кадров попало в данный интервал задержки.",
        fontsize=9,
        color="#444444",
    )

    if save_png:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=160, facecolor="#f3f6fb")
        print(f"PNG-снимок: {save_png}")
    if no_window:
        plt.close(fig)
    else:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает ID опыта, метрику, размеры корзин и путь HTML-отчёта."""

    parser = argparse.ArgumentParser(description="Вращаемая автономная 3D-страница задержек VideoT16.")
    parser.add_argument("--run-id", help="ID опыта; по умолчанию выбирается последняя полная сессия")
    parser.add_argument("--metric", choices=METRICS, default="inference_ms", help="какой этап анализировать")
    parser.add_argument("--time-bin", type=float, default=5.0, help="ширина временного интервала, секунды (по умолчанию 5)")
    parser.add_argument("--latency-bin", type=float, default=10.0, help="ширина интервала задержки, миллисекунды (по умолчанию 10)")
    parser.add_argument("--summary-only", action="store_true", help="только текстовая статистика, без графика")
    parser.add_argument("--with-target", action="store_true", help="дополнительно построить целевой график-сценарий; он синтетический, не замер")
    parser.add_argument("--output", type=Path, help="путь HTML; по умолчанию tmp/video_cap_buf/latency3d_<сессия>_<метрика>.html")
    parser.add_argument("--open", action="store_true", help="после создания открыть HTML в браузере")
    parser.add_argument("--save-png", type=Path, help="дополнительно сохранить статичный PNG-график")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Загружает сессию, печатает статистику и создаёт вращаемый HTML-график."""

    args = parse_args(argv)
    if not math.isfinite(args.time_bin) or args.time_bin <= 0:
        print("Ошибка: --time-bin должен быть положительным числом.", file=sys.stderr)
        return 2
    if not math.isfinite(args.latency_bin) or args.latency_bin <= 0:
        print("Ошибка: --latency-bin должен быть положительным числом.", file=sys.stderr)
        return 2

    try:
        paths = find_logs(args.run_id)
        samples, metadata = load_samples(paths, args.metric)
        print("Журналы:")
        for path in paths:
            print(f"  {path}")
        print_summary(samples, metadata, args.metric)
        target_samples: list[tuple[float, float]] = []
        goals: dict[str, float] = {}
        if args.with_target:
            target_samples, goals = make_target_samples(samples, args.time_bin, metadata["fps"])
            print_quality_assessment(samples, goals)
        if not args.summary_only:
            output = args.output or (
                LOG_DIR / f"latency3d_{metadata['run_id']}_{args.metric}.html"
            )
            output = write_html_chart(
                samples, metadata, args.metric, args.time_bin, args.latency_bin, output
            )
            print(f"Вращаемый автономный HTML: {output}")
            if args.open:
                webbrowser.open(output.resolve().as_uri())
            if args.with_target:
                target_metadata = dict(metadata)
                target_metadata["is_target"] = True
                target_metadata["target_note"] = (
                    "СИНТЕТИЧЕСКАЯ ЦЕЛЬ, НЕ РЕЗУЛЬТАТ ЗАМЕРА. "
                    "Ориентиры при текущей частоте захвата: "
                    f"P50 ≤ {goals['p50']:.1f} мс · P90 ≤ {goals['p90']:.1f} мс · "
                    f"P98 ≤ {goals['p98']:.1f} мс."
                )
                target_metadata["run_id"] = metadata["run_id"]
                target_path = output.with_name(
                    output.stem + "_target" + (output.suffix or ".html")
                )
                target_path = write_html_chart(
                    target_samples,
                    target_metadata,
                    args.metric,
                    args.time_bin,
                    args.latency_bin,
                    target_path,
                )
                print(f"Целевой график-сценарий (синтетика): {target_path}")
                if args.open:
                    webbrowser.open(target_path.resolve().as_uri())
            if args.save_png:
                draw_chart(
                    samples,
                    metadata,
                    args.metric,
                    args.time_bin,
                    args.latency_bin,
                    args.save_png,
                    True,
                )
    except (OSError, ValueError) as error:
        print(f"Ошибка анализа: {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        print(f"Ошибка импорта зависимости графика: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
