#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
压测 CrawlWebpageToolClientV2：
- 支持多端点(base_urls)轮询
- 指定总并发、目标 RPS、持续时间
- 统计：sent/ok/fail, RPS 实际达到, p50/p95/p99, 平均, 最大
- 每端点成功/失败次数、平均耗时
- 可选择写出 JSON/CSV 指标文件

依赖：Python 3.9+、httpx（client 内已用到）
"""

import argparse
import asyncio
import json
import math
import os
import random
import socket
import statistics as stats
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import fastapi
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse

import ray

from tests.workers.rollout.my_tools_sever import CrawlWebpageToolClientV2, CrawlWebpageToolServerV2

# ======= 导入你实现的 ClientV2 =======
# 把下面的 import 改成你的真实路径或直接把类贴在同一目录
# from client_v2 import CrawlWebpageToolClientV2  # noqa: E402


# 一些稳定可访问的 URL（如需内网可替换）
# DEFAULT_URLS = [
#     "https://example.com",
#     "https://www.python.org",
#     "https://httpbin.org/html",
#     "https://www.wikipedia.org/",
#     "https://httpbin.org/anything",
# ]


# -------------------- 压测：固定 N 条 URL，一次性处理 --------------------
@dataclass
class OneResult:
    ok: bool; dt: float; endpoint: str; err_type: str = ""

@dataclass
class Summary:
    sent: int; ok: int; fail: int; duration: float; rps: float
    p50: float; p95: float; p99: float; pmax: float; mean: float

def percentile(values: List[float], p: float) -> float:
    if not values: return 0.0
    k = max(0, min(len(values) - 1, int(math.ceil(p * len(values)) - 1)))
    return sorted(values)[k]

async def one_call(cli: CrawlWebpageToolClientV2, url: str, keep_links: bool) -> OneResult:
    t0 = time.perf_counter()
    # ep = getattr(cli, "_last_used_url", "unknown")
    try:
        res = await cli.call({"url": url}, keep_links=keep_links, return_endpoint=True)
        dt = time.perf_counter() - t0
        ep = res.get("endpoint", "unknown")
        res = res.get("result", "")
        if isinstance(res, str) and not res.startswith("[Client Error]") and not res.startswith("[Server Error]"):
            return OneResult(True, dt, ep, "")
        return OneResult(False, dt, ep, "HTTP")
    except httpx.TimeoutException:
        return OneResult(False, time.perf_counter()-t0, ep, "Timeout")
    except httpx.RequestError:
        return OneResult(False, time.perf_counter()-t0, ep, "Request")
    except Exception:
        return OneResult(False, time.perf_counter()-t0, ep, "Other")
    
def run_once(num_servers: int, semaphore_limit: int, args) -> Tuple[Summary, Dict[str, Any]]:
    # === 启 server ===
    actors = [
        CrawlWebpageToolServerV2.options(name=f"crawl_v2_{i}", num_cpus=1).remote(
            semaphore_limit=semaphore_limit,
            cache_file=None,
            snippet_cache_file=None,
            fetch_timeout=args.fetch_timeout,
        )
        for i in range(num_servers)
    ]
    addrs = ray.get([a.get_server_address.remote() for a in actors])
    base_urls = [f"http://{addr}" for addr in addrs]

    # === 读 URL 列表 ===
    if args.urls_file:
        with open(args.urls_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        src_urls = list(data.keys())
    else:
        src_urls = [
            "https://example.com",
            "https://www.python.org",
            "https://httpbin.org/html",
            "https://www.wikipedia.org/",
            "https://httpbin.org/anything",
        ]
    # 组装任务 URL
    task_urls: List[str] = []
    i = 0
    while len(task_urls) < args.total_urls:
        u = src_urls[i % len(src_urls)]; i += 1
        if args.no_cache_hack:
            q = f"_bench_ts={int(time.time()*1000)}_{len(task_urls)}_{random.randint(1,999999)}"
            sep = "&" if ("?" in u) else "?"
            u = f"{u}{sep}{q}"
        task_urls.append(u)

    # === 跑基准 ===
    summary, detail = asyncio.run(
        run_fixed(
            base_urls=base_urls,
            task_urls=task_urls,
            concurrency=args.concurrency,
            keep_links=args.keep_links,
            call_timeout=args.call_timeout,
            per_ep_cc=args.per_ep_cc,
            global_cc=args.global_cc,
            retries=args.retries,
        )
    )

    # === 关 server ===
    try:
        asyncio.run(asyncio.gather(*[a.shutdown.remote() for a in actors], return_exceptions=True))
    except RuntimeError:
        for a in actors:
            try: ray.get(a.shutdown.remote())
            except Exception: pass

    return summary, detail


def main_v0():
    ap = argparse.ArgumentParser(...)
    # 你原来的 add_argument 保持不变，再加 sweep 的四个参数
    ...
    args = ap.parse_args()

    # 可选：开始前干净一点
    if args.kill_ray:
        os.system("ray stop --force >/dev/null 2>&1 || true")

    # Ray init
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    # ========= 如果提供了 sweep 参数，进入 sweep 模式 =========
    if args.sweep_servers or args.sweep_sema:
        server_list = [int(x) for x in args.sweep_servers.split(",") if x.strip()] if args.sweep_servers else [args.num_servers]
        sema_list   = [int(x) for x in args.sweep_sema.split(",") if x.strip()]   if args.sweep_sema else [args.semaphore_limit]

        rows = []
        try:
            for ns in server_list:
                for sl in sema_list:
                    r_runs = []
                    for r in range(max(1, args.sweep_repeats)):
                        # 每轮都从干净环境开始，尽量避免资源残留
                        ray.shutdown()
                        time.sleep(0.3)
                        ray.init(ignore_reinit_error=True)
                        if args.kill_ray:
                            os.system("ray stop --force >/dev/null 2>&1 || true")
                            # 二次 init
                            ray.init(ignore_reinit_error=True)

                        summary, detail = run_once(ns, sl, args)
                        r_runs.append(summary)

                    # 聚合（取均值/中位等）
                    sent = sum(x.sent for x in r_runs)
                    ok   = sum(x.ok for x in r_runs)
                    fail = sum(x.fail for x in r_runs)
                    # rps 用总 sent / 总 duration 更公平
                    total_dur = sum(x.duration for x in r_runs)
                    rps = sent / total_dur if total_dur > 0 else 0.0
                    p50  = stats.median([x.p50 for x in r_runs])
                    p95  = stats.median([x.p95 for x in r_runs])
                    p99  = max([x.p99 for x in r_runs])
                    mean = sum([x.mean for x in r_runs]) / len(r_runs)

                    rows.append({
                        "num_servers": ns, "semaphore_limit": sl,
                        "sent": sent, "ok": ok, "fail": fail,
                        "duration_sum_s": round(total_dur, 3),
                        "rps_agg": round(rps, 3),
                        "p50_med": round(p50, 3),
                        "p95_med": round(p95, 3),
                        "p99_max": round(p99, 3),
                        "mean_avg": round(mean, 3),
                    })

            # 打印对比表
            print("\n=== SWEEP RESULTS ===")
            header = ["num_servers","semaphore_limit","sent","ok","fail","duration_sum_s","rps_agg","p50_med","p95_med","p99_max","mean_avg"]
            print("\t".join(header))
            for r in rows:
                print("\t".join(str(r[k]) for k in header))

            # 可选写 CSV
            if args.sweep_out_csv:
                import csv
                with open(args.sweep_out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    w.writeheader()
                    w.writerows(rows)
                print(f"\nSweep CSV saved to: {args.sweep_out_csv}")

        finally:
            ray.shutdown()
        return  # 结束 sweep 模式

    # ========= 否则，走你原本的单次基准路径 =========
    try:
        summary, detail = run_once(args.num_servers, args.semaphore_limit, args)
        print("\n=== Summary ===")
        print(json.dumps(summary.__dict__, indent=2))
        print("\n=== Per Endpoint ===")
        for ep, stat in detail["per_endpoint"].items():
            print(ep, json.dumps(stat, ensure_ascii=False))
        if detail.get("errors_by_type"):
            print("\n=== Errors By Type ===")
            print(json.dumps(detail["errors_by_type"], indent=2))
    finally:
        ray.shutdown()

def main():
    ap = argparse.ArgumentParser(description="Fixed-N crawl bench (start servers inside)")
    ap.add_argument("--kill_ray", action="store_true", help="启动前清理历史 Ray 进程")
    ap.add_argument("--ray-address", type=str, default="", help="Ray 地址（留空=本地）")

    # servers
    ap.add_argument("--num-servers", type=int, default=8, help="Server 个数")
    ap.add_argument("--semaphore-limit", type=int, default=1, help="每个 Server 内部并发（建议 1 或 2）")
    ap.add_argument("--fetch-timeout", type=int, default=300, help="Tool fetch 超时")

    # workload
    ap.add_argument("--urls-file", type=str, required=False, default="", help="URL 列表文件（每行一个）")
    ap.add_argument("--total-urls", type=int, default=1000, help="总 URL 数（固定处理这么多条）")
    ap.add_argument("--concurrency", type=int, default=64, help="客户端并发 worker 数")
    ap.add_argument("--keep-links", action="store_true", help="生成 markdown 时保留链接")
    ap.add_argument("--no-cache-hack", action="store_true",
                    help="给每条 URL 加唯一查询串，避免服务端 cache 命中")

    # client knobs
    ap.add_argument("--call-timeout", type=float, default=310.0, help="单次调用超时")
    ap.add_argument("--per-ep-cc", type=int, default=1, help="每端点并发（client 侧）")
    ap.add_argument("--global-cc", type=int, default=256, help="全局并发上限（client 侧）")
    ap.add_argument("--retries", type=int, default=2, help="失败重试次数")

    # output
    ap.add_argument("--out-json", type=str, default="", help="指标 JSON 输出路径")
    ap.add_argument("--out-csv", type=str, default="", help="每端点指标 CSV 输出路径")

    ap.add_argument("--sweep-servers", type=str, default="", 
                help="逗号分隔的 server 数列表，例如: 1,2,4,8")
    ap.add_argument("--sweep-sema", type=str, default="", 
                    help="逗号分隔的 semaphore-limit 列表，例如: 1,2,4")
    ap.add_argument("--sweep-repeats", type=int, default=1, 
                    help="每个组合重复次数，取平均/中位更稳")
    ap.add_argument("--sweep-out-csv", type=str, default="", 
                    help="把 sweep 的结果写到 CSV（可选）")


    args = ap.parse_args()

    args = ap.parse_args()

    # 可选：开始前干净一点
    if args.kill_ray:
        os.system("ray stop --force >/dev/null 2>&1 || true")

    # Ray init
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    # ========= 如果提供了 sweep 参数，进入 sweep 模式 =========
    if args.sweep_servers or args.sweep_sema:
        server_list = [int(x) for x in args.sweep_servers.split(",") if x.strip()] if args.sweep_servers else [args.num_servers]
        sema_list   = [int(x) for x in args.sweep_sema.split(",") if x.strip()]   if args.sweep_sema else [args.semaphore_limit]

        rows = []
        try:
            for ns in server_list:
                for sl in sema_list:
                    r_runs = []
                    for r in range(max(1, args.sweep_repeats)):
                        # 每轮都从干净环境开始，尽量避免资源残留
                        ray.shutdown()
                        time.sleep(0.3)
                        ray.init(ignore_reinit_error=True)
                        if args.kill_ray:
                            os.system("ray stop --force >/dev/null 2>&1 || true")
                            # 二次 init
                            ray.init(ignore_reinit_error=True)

                        summary, detail = run_once(ns, sl, args)
                        r_runs.append(summary)

                    # 聚合（取均值/中位等）
                    sent = sum(x.sent for x in r_runs)
                    ok   = sum(x.ok for x in r_runs)
                    fail = sum(x.fail for x in r_runs)
                    # rps 用总 sent / 总 duration 更公平
                    total_dur = sum(x.duration for x in r_runs)
                    rps = sent / total_dur if total_dur > 0 else 0.0
                    p50  = stats.median([x.p50 for x in r_runs])
                    p95  = stats.median([x.p95 for x in r_runs])
                    p99  = max([x.p99 for x in r_runs])
                    mean = sum([x.mean for x in r_runs]) / len(r_runs)

                    rows.append({
                        "num_servers": ns, "semaphore_limit": sl,
                        "sent": sent, "ok": ok, "fail": fail,
                        "duration_sum_s": round(total_dur, 3),
                        "rps_agg": round(rps, 3),
                        "p50_med": round(p50, 3),
                        "p95_med": round(p95, 3),
                        "p99_max": round(p99, 3),
                        "mean_avg": round(mean, 3),
                    })

            # 打印对比表
            print("\n=== SWEEP RESULTS ===")
            header = ["num_servers","semaphore_limit","sent","ok","fail","duration_sum_s","rps_agg","p50_med","p95_med","p99_max","mean_avg"]
            print("\t".join(header))
            for r in rows:
                print("\t".join(str(r[k]) for k in header))

            # 可选写 CSV
            if args.sweep_out_csv:
                import csv
                with open(args.sweep_out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    w.writeheader()
                    w.writerows(rows)
                print(f"\nSweep CSV saved to: {args.sweep_out_csv}")

        finally:
            ray.shutdown()
        return  # 结束 sweep 模式

    # ========= 否则，走你原本的单次基准路径 =========
    try:
        summary, detail = run_once(args.num_servers, args.semaphore_limit, args)
        print("\n=== Summary ===")
        print(json.dumps(summary.__dict__, indent=2))
        print("\n=== Per Endpoint ===")
        for ep, stat in detail["per_endpoint"].items():
            print(ep, json.dumps(stat, ensure_ascii=False))
        if detail.get("errors_by_type"):
            print("\n=== Errors By Type ===")
            print(json.dumps(detail["errors_by_type"], indent=2))
    finally:
        ray.shutdown()

async def run_fixed(
    base_urls: List[str], task_urls: List[str], concurrency: int, keep_links: bool,
    call_timeout: float, per_ep_cc: int, global_cc: int, retries: int
) -> Tuple[Summary, Dict[str, Any]]:
    params_schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    cli = CrawlWebpageToolClientV2(
        base_url=base_urls, parameters=params_schema,
        call_timeout=call_timeout, default_timeout=60.0,
        max_concurrency=global_cc, per_endpoint_concurrency=per_ep_cc, retries=retries
    )

    q: asyncio.Queue[str] = asyncio.Queue()
    for u in task_urls: q.put_nowait(u)

    lat: List[float] = []; results: List[OneResult] = []
    start = time.perf_counter()

    async def worker(_wid: int):
        while True:
            try:
                u = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            r = await one_call(cli, u, keep_links)
            results.append(r)
            if r.ok: lat.append(r.dt)
            q.task_done()

    tasks = [asyncio.create_task(worker(i)) for i in range(max(1, concurrency))]
    await asyncio.gather(*tasks)
    await cli.aclose()

    sent = len(results); ok = sum(1 for r in results if r.ok); fail = sent - ok
    dur = time.perf_counter() - start; rps = sent / max(dur, 1e-9)

    if lat:
        p50 = stats.median(lat); p95 = percentile(lat, 0.95)
        p99 = percentile(lat, 0.99); pmax = max(lat); mean = sum(lat) / len(lat)
    else:
        p50=p95=p99=pmax=mean=0.0

    summary = Summary(sent, ok, fail, dur, rps, p50, p95, p99, pmax, mean)

    by_ep_cnt=defaultdict(int); by_ep_ok=defaultdict(int); by_ep_fail=defaultdict(int); by_ep_lat=defaultdict(list); by_err=defaultdict(int)
    for r in results:
        by_ep_cnt[r.endpoint]+=1
        if r.ok:
            by_ep_ok[r.endpoint]+=1; by_ep_lat[r.endpoint].append(r.dt)
        else:
            by_ep_fail[r.endpoint]+=1; by_err[r.err_type]+=1

    ep_stats={}
    for ep in by_ep_cnt:
        l = by_ep_lat[ep]
        ep_stats[ep] = {
            "sent": by_ep_cnt[ep], "ok": by_ep_ok[ep], "fail": by_ep_fail[ep],
            "ok_rate": (by_ep_ok[ep]/by_ep_cnt[ep]) if by_ep_cnt[ep] else 0.0,
            "p50": stats.median(l) if l else 0.0, "p95": percentile(l, 0.95) if l else 0.0,
            "mean": (sum(l)/len(l)) if l else 0.0,
        }

    detail = {"summary": asdict(summary), "errors_by_type": dict(by_err), "per_endpoint": ep_stats}
    return summary, detail

# -------------------- Main ------------------

if __name__ == "__main__":
    main()