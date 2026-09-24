# Handoff: `community`

## What I built

- `src/llm_configurator/community.py`: opt-in, anonymous sharing of speed results; strict import of the curated list; "other people's computers" evidence for the engine.
- `community/results.json`: `{"schema": 1, "records": []}`.
- `community/README.md`: plain-language description of what is and is not shared, how to contribute, the schema, and a manual maintainer merge process (no auto-merge).
- `tests/test_community.py`: 40 tests (anonymise allowlist, planted-secret privacy leak tests, validation of good/bad/malicious rows, import from text and a mocked HTTPS opener, redirect/oversize refusal, share URL encoding and length fallback, evidence tiers and medians).

A separate privacy-review agent attacked `anonymize`/`parse`/`_fetch`. It found five issues, all fixed and covered by tests: a missing `source` treated as public; host/user names split into parts (`johns-macbook-pro.local`) slipping through; a 400-digit integer crashing the import (`OverflowError`); free-text runtime versions; serial-like tokens, domain names and non-ASCII digits.

## Public API as built

```python
anonymize(measurement: dict, hardware: dict, variant: Variant | dict | None) -> dict   # schema-v1 record; ValueError if no valid tps/context
share_payload(store, measurement_ids: list[str], hardware=None, variants=None)
    -> {"json": str, "issue_url": str, "fits_in_url": bool, "records": int}
import_records(store, source=None, text=None, progress=None, cancel=None)
    -> {"source": str, "fetched_at": iso str, "records": [record], "rejected": int}   # also store.put("community", {source, fetched_at, records})
evidence(records, variant, hardware, config) 
    -> {"median_tps", "n", "similar": "same_gpu"|"same_cpu"|"same_class", "range": [q1, q3],
        "placement": "gpu"|"split"|"cpu", "context_bucket": str, "note": str} | None
```

Helpers others may use: `validate_record(row) -> clean row`, `parse(text|bytes) -> (records, rejected)`, `hardware_class(hardware, gpu_uuid=None, uses_gpu=True)`, `clean_name(str)`, `bucket_gib(bytes)`, `issue_url(payload_json, count) -> (url, fits)`, `context_bucket(context)`, constants `SCHEMA`, `DEFAULT_SOURCE`, `LABEL`, `NOTE`.

The record schema (v1) is documented in `community/README.md`. Its top-level keys are: `schema, id, month, kind, tps, pp_tps, ttft_s, context, depth, variant{repo, filename, sha256, quant, layers}, hardware{cpu, gpu, backend, vram_gib, ram_gib, unified, os, os_version, arch}, settings{placement, gpu_layers, users, threads, flash_attn, cache_type_k, cache_type_v, batch, ubatch, n_cpu_moe}, runtime{version, backend}`.

## Deviations from the contract and why

- `share_payload` takes extra optional `hardware=` and `variants=` arguments, which are mainly for tests. When they are left out, it calls `hardware.scan(include_processes=False)` (imported inside the function) and uses `store.get("variants")`. It also returns the extra keys `fits_in_url` and `records`.
- `evidence` adds the extra keys `placement`, `context_bucket` and `note`. `note` is the plain "other people's computers" sentence the UI should show.
- Repo and file names are shared **only** when `variant.source == "catalogue"`. For `custom`, `local`, a missing source, or an unknown variant, only the sha256 and quant are shared, because a user's private repo or file name could identify them.
- `share_payload` refuses a measurement whose `fingerprint` differs from the current machine's. The hardware description would be false otherwise (honesty rule).
- Placement uses `gpu_layers` compared with `variant.layers` (or `config.total_layers`). When the total is unknown and `gpu_layers > 0`, the result is labelled `split`.

## Known gaps

- GitHub docs were not reachable from this container, so I could not re-verify the "new issue" query parameters. I used `title`, `body` and `labels` (see assumptions below).
- Hardware names are cleaned by rules, not checked against a list of known vendor models. A strange personal token that is not the machine or user name, is not serial-like and is not in brackets could still get through. Two safeguards remain: the user reviews the JSON, and a maintainer reviews the issue.
- `evidence` matches the exact cleaned GPU/CPU name. Names that differ slightly between platforms (for example the Windows and Linux spelling of the same card) fall back to the `same_class` tier.
- Measurements have no `total_layers` field, so shares for variants that are no longer in the cache may mislabel a full offload as `split`.

## Requests

- **engine**: call `community.evidence(store.get("community", {}).get("records", []), variant, hardware, launch_fragment)`. `config` uses launch-config keys: `context`, `gpu_layers`, `total_layers`, `gpu_uuid`, `gpu_backend`, `parallel`. Map the result to candidate `community: {median_tps, n}` and show `note`.
- **api**: expose `GET /api/community` as `store.get("community")` (`records`, `source`, `fetched_at`). `POST /api/community/import` should run `import_records(store, progress=..., cancel=...)` as a job, always with the default source; the browser never supplies URLs. `POST /api/community/share` should validate `measurement_ids` as a list of 1–50 short strings, then call `share_payload(store, ids)`. The CLI `community import --source URL` passes `source`, and `import_records` enforces https.
- **testing / tuner**: give every measurement an `id` (§2.3), plus `settings`, `runtime` and `kind`, so results can be shared. Sharing needs `id`, `tps`, `context`, `fingerprint` and `gpu_layers`.
- **catalogue**: keep `source="custom"` for user-added repos (`add_entry`), because the privacy rule relies on it.
- **ui-run / ui-quality**: show the JSON to the user before opening `issue_url`. When `fits_in_url` is false, give them a "copy JSON" button and tell them to paste it into the issue.
- **docs**: optionally add `.github/ISSUE_TEMPLATE/community-results.yml` with `labels: [community-results]`. Issue templates apply labels even for users without triage permission. If one is added, `issue_url` can gain `template=`.

## How I tested

`python3 -m unittest discover -s tests` passed with 111 tests, 40 of them new. There is no network use: HTTPS is faked through a patched `community._opener`, and the redirect handler is tested directly. The privacy tests plant a hostname, username, paths, a GPU UUID, a fingerprint, serials, a PID and an exact time in every input field. They then check that none of these appear in the record, the share JSON or the issue URL, both with and without the machine-name scrubber.

## Assumptions to confirm

- There are no llama.cpp facts here, apart from assuming that runtime versions look like `b1234` build tags or dotted numbers. Other version strings are dropped, not shared.
- GitHub assumptions: `https://github.com/OWNER/REPO/issues/new?title=&body=&labels=` prefills an issue. `labels` only applies for users with triage rights. URLs much longer than about 8 KB fail, so `MAX_URL = 8000` characters.
