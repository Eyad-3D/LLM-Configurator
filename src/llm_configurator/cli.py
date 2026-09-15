"""Command-line entry points. Run `llm-config --help`."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from .app import evaluate, map_benchmark, variants
from .catalogue import definitions, refresh
from .domain import GIB, Requirements
from .hardware import scan
from .runtime import bench, download
from .server import serve
from .storage import Store


def parser():
    root = argparse.ArgumentParser(description="Choose local model configurations using available hardware and benchmark evidence")
    root.add_argument("--data-dir", help="Override local cache/settings directory")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="Print current hardware and process memory as JSON")
    commands.add_parser("refresh", help="Fetch model metadata and optional AA scores; no weight downloads")
    commands.add_parser("models", help="List cached variants and their exact IDs")
    commands.add_parser("benchmarks", help="List AA names/slugs to explicitly map to local models")
    mapping = commands.add_parser("map", help="Map a base model to an exact AA evaluation entry")
    mapping.add_argument("base_repo")
    mapping.add_argument("slug", help="Exact AA slug, or '-' to clear")
    web = commands.add_parser("serve", help="Open the local browser interface")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--demo", action="store_true", help="Use clearly fictional model fixtures")
    web.add_argument("--no-browser", action="store_true")
    rec = commands.add_parser("recommend", help="Compare deployment configurations")
    rec.add_argument("--workload", choices=["general", "coding", "agentic", "documents"], default="general")
    rec.add_argument("--context", type=int, default=8192)
    rec.add_argument("--users", type=int, default=1)
    rec.add_argument("--min-tps", type=float, default=15)
    rec.add_argument("--reserve-gib", type=float, default=2)
    rec.add_argument("--gpu-reserve-gib", type=float, default=0.5)
    rec.add_argument("--gpu-index", type=int, default=0)
    rec.add_argument("--reclaim-pids", type=int, nargs="*", default=[])
    rec.add_argument("--strict-speed", action="store_true")
    rec.add_argument("--priority", choices=["balanced", "quality", "speed"], default="balanced")
    rec.add_argument("--no-rankings", dest="include_rankings", action="store_false")
    rec.add_argument("--demo", action="store_true")
    rec.add_argument("--json", action="store_true")
    rec.add_argument("--output", type=Path, help="Save the complete report as JSON")
    get = commands.add_parser("download", help="Explicitly download a pinned GGUF and verify its SHA256")
    get.add_argument("variant_id")
    get.add_argument("--directory", type=Path, required=True)
    get.add_argument("--yes", action="store_true", help="Accept the displayed download size without an interactive prompt")
    run = commands.add_parser("bench", help="Run llama-bench on an existing GGUF; consumes local compute")
    run.add_argument("variant_id")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--executable", default="llama-bench")
    run.add_argument("--context", type=int, default=8192)
    run.add_argument("--gpu-layers", type=int, default=0)
    run.add_argument("--gpu-index", type=int, default=0)
    run.add_argument("--timeout", type=int, default=600)
    return root


def print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        store = Store(args.data_dir)
        if args.command == "scan":
            print_json(scan())
        elif args.command == "refresh":
            print_json(refresh(store))
        elif args.command == "models":
            print_json([v.to_dict() for v in variants(store)])
        elif args.command == "benchmarks":
            cache = store.get("scores", {})
            print_json({"source": "Artificial Analysis — https://artificialanalysis.ai", "version": cache.get("version"),
                        "entries": [{"name": i["name"], "slug": i["slug"]} for i in cache.get("data", [])],
                        "mappings": definitions(store)})
        elif args.command == "map":
            print_json(map_benchmark(store, args.base_repo, None if args.slug == "-" else args.slug))
        elif args.command == "serve":
            serve(store, args.port, args.demo, not args.no_browser)
        elif args.command == "recommend":
            payload = {field.name: getattr(args, field.name) for field in fields(Requirements)}
            report = evaluate(store, payload, args.demo)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.json:
                print_json(report)
            else:
                if report["demo"]:
                    print("DEMO: fictional models, real hardware. No real quality or speed scores.")
                shortlist = set(report["shortlist"])
                for item in report["candidates"]:
                    if item["id"] not in shortlist:
                        continue
                    speed = f"{item['tps']:.1f} tok/s" if item["tps"] is not None else "speed unverified"
                    print(f"{item['name']} / {item['quant']} / {item['mode']} / {item['context']:,} tokens per user / {speed}")
                    print(f"  RAM {item['ram_bytes']/GIB:.2f} GiB | VRAM {item['vram_bytes']/GIB:.2f} GiB | {item['scenario']}")
                    print(f"  {item['quality_evidence']}; reference score: {item['quality_score']}")
                    if item["score_source"]:
                        print(f"  Artificial Analysis: {item['score_source']} (index {item['score_version']})")
                if not report["candidates"]:
                    print("No qualifying configurations. Refresh metadata, relax constraints, or include speed-unverified options.")
                print("\n" + "\n".join(report["notes"]))
        else:
            variant = next((v for v in variants(store) if v.id == args.variant_id), None)
            if not variant:
                raise ValueError("Variant ID not found; use 'llm-config models' after refreshing metadata")
            if args.command == "download":
                print(f"Download {variant.filename}: {variant.size_bytes/GIB:.2f} GiB", flush=True)
                if not args.yes and input("Download this model? [y/N] ").lower() != "y":
                    return 0
                print(download(variant, args.directory))
            elif args.command == "bench":
                print("Validating file and memory, then running a local generation benchmark…", file=sys.stderr, flush=True)
                measurement = bench(variant, args.model, args.executable, args.context, args.gpu_layers, args.gpu_index, args.timeout)
                records = store.get("measurements", [])
                records.append(measurement)
                store.put("measurements", records[-1000:])
                print_json(measurement)
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
