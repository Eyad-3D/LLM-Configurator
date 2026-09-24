import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_configurator import runtime_install as ri
from llm_configurator.domain import Cancelled
from llm_configurator.storage import Store, runtime_dir

BASE = "https://github.com/ggml-org/llama.cpp/releases/download/b11158/"
# Real asset names from release b11158 (2026-09-24), checked against the GitHub release page.
REAL_NAMES = """cudart-llama-b11158-bin-ubuntu-cuda-12.8-x64.tar.gz cudart-llama-b11158-bin-ubuntu-cuda-13.4-arm64.tar.gz
cudart-llama-b11158-bin-ubuntu-cuda-13.4-x64.tar.gz cudart-llama-bin-win-cuda-12.4-x64.zip cudart-llama-bin-win-cuda-13.4-arm64.zip
cudart-llama-bin-win-cuda-13.4-x64.zip llama-b11158-bin-android-arm64-snapdragon.tar.gz llama-b11158-bin-android-arm64.tar.gz
llama-b11158-bin-linux-arm64-snapdragon.tar.gz llama-b11158-bin-macos-arm64.tar.gz llama-b11158-bin-macos-x64.tar.gz
llama-b11158-bin-ubuntu-arm64.tar.gz llama-b11158-bin-ubuntu-cuda-12.8-x64.tar.gz llama-b11158-bin-ubuntu-cuda-13.4-arm64.tar.gz
llama-b11158-bin-ubuntu-cuda-13.4-x64.tar.gz llama-b11158-bin-ubuntu-openvino-2026.4-x64.tar.gz llama-b11158-bin-ubuntu-rocm-10.0-x64.tar.gz
llama-b11158-bin-ubuntu-s390x.tar.gz llama-b11158-bin-ubuntu-sycl-fp16-x64.tar.gz llama-b11158-bin-ubuntu-sycl-fp32-x64.tar.gz
llama-b11158-bin-ubuntu-vulkan-arm64.tar.gz llama-b11158-bin-ubuntu-vulkan-x64.tar.gz llama-b11158-bin-ubuntu-x64.tar.gz
llama-b11158-bin-win-cpu-arm64.zip llama-b11158-bin-win-cpu-x64.zip llama-b11158-bin-win-cuda-12.4-x64.zip
llama-b11158-bin-win-cuda-13.4-arm64.zip llama-b11158-bin-win-cuda-13.4-x64.zip llama-b11158-bin-win-opencl-adreno-arm64.zip
llama-b11158-bin-win-openvino-2026.4-x64.zip llama-b11158-bin-win-rocm-10.0-x64.zip llama-b11158-bin-win-sycl-x64.zip
llama-b11158-bin-win-vulkan-x64.zip llama-b11158-ui.tar.gz llama-b11158-xcframework.zip""".split()

VERSION_SCRIPT = "#!/bin/sh\ncat >&2 <<'EOF'\n{text}\nEOF\n"
GOOD = VERSION_SCRIPT.format(text="version: 5000 (abc1234)\nbuilt with cc (GCC) 13.2.0 for x86_64-linux-gnu")
BROKEN = "#!/bin/sh\necho 'FAIL: error while loading shared libraries: libvulkan.so.1' >&2\nexit 127\n"


def release(names=REAL_NAMES, digest=True, tag="b11158", blobs=None):
    blobs = blobs or {}
    assets = []
    for name in names:
        data = blobs.get(name, b"x" * 100)
        asset = {"name": name, "size": len(data), "browser_download_url": BASE + name}
        if digest:
            asset["digest"] = "sha256:" + hashlib.sha256(data).hexdigest()
        assets.append(asset)
    return {"tag_name": tag, "prerelease": True, "assets": assets}


def nvidia(driver="581.57", name="NVIDIA GeForce RTX 4090"):
    return {"gpus": [{"index": 0, "name": name, "backend": "cuda", "vendor": "nvidia", "driver": driver}]}


AMD = {"gpus": [{"index": 0, "name": "AMD Radeon RX 7900 XTX", "backend": "vulkan", "vendor": "amd", "driver": None}]}
NONE = {"gpus": []}


def tar_bytes(files, links=()):
    """files: {path: (text, mode)}; links: [(name, target, type)]."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for path, (text, mode) in files.items():
            data = text.encode() if isinstance(text, str) else text
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(data), mode
            bundle.addfile(info, io.BytesIO(data))
        for name, target, kind in links:
            info = tarfile.TarInfo(name)
            info.type, info.linkname = kind, target
            bundle.addfile(info)
    return buffer.getvalue()


def zip_bytes(files, symlinks=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for path, (text, mode) in files.items():
            info = zipfile.ZipInfo(path)
            info.external_attr = (stat.S_IFREG | mode) << 16
            bundle.writestr(info, text)
        for path, target in symlinks:
            info = zipfile.ZipInfo(path)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, target)
    return buffer.getvalue()


def server_tree(prefix="llama-b5000/", script=GOOD, extra=None):
    files = {prefix + "llama-server": (script, 0o755), prefix + "llama-bench": (GOOD, 0o755),
             prefix + "llama-perplexity": (GOOD, 0o755), prefix + "libllama.so": ("lib", 0o644)}
    files.update(extra or {})
    return files


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status

    def getcode(self):
        return self.status


class FakeNet:
    """Serves byte blobs by URL; honours Range unless told not to; can cancel after N chunks."""

    def __init__(self, blobs, honour_range=True, json_by_url=None):
        self.blobs, self.honour_range, self.json_by_url = blobs, honour_range, json_by_url or {}
        self.requests = []

    def __call__(self, request, timeout=60):
        url = request.full_url
        self.requests.append((url, dict(request.header_items())))
        if url in self.json_by_url:
            return FakeResponse(json.dumps(self.json_by_url[url]).encode())
        name = url.rsplit("/", 1)[-1]
        if name not in self.blobs:
            raise HTTPError(url, 404, "not found", {}, None)
        data = self.blobs[name]
        header = request.get_header("Range")
        if header and self.honour_range:
            start = int(header.split("=")[1].rstrip("-"))
            if start >= len(data):
                raise HTTPError(url, 416, "range", {}, None)
            return FakeResponse(data[start:], 206)
        return FakeResponse(data)


def fake_run(argv):
    path = Path(argv[0])
    if not path.is_file():
        return False, "missing"
    text = path.read_text(errors="replace")
    return ("FAIL" not in text), text


def fake_devices(argv):
    """Scripts that contain a `#devices` block answer --list-devices with it; others act like old builds."""
    path = Path(argv[0])
    text = path.read_text(errors="replace") if path.is_file() else ""
    if "#devices\n" not in text:
        return False, "error: invalid argument: --list-devices"
    return True, "Available devices:\n" + text.split("#devices\n", 1)[1]


def with_devices(script, *lines):
    return script + "#devices\n" + ("\n".join(lines) if lines else "  (none)") + "\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "data")
        patches = [mock.patch.object(ri, "_run_version", side_effect=fake_run),
                   mock.patch.object(ri, "_run_devices", side_effect=fake_devices),
                   mock.patch.object(ri, "_path_candidates", return_value=iter(())),
                   mock.patch.object(ri, "CHUNK", 16)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.temp.cleanup)


class ClassifyTests(unittest.TestCase):
    def test_real_names_are_understood(self):
        infos = {name: ri.classify(name) for name in REAL_NAMES}
        self.assertEqual(infos["llama-b11158-bin-win-cuda-13.4-x64.zip"]["cuda"], (13, 4))
        self.assertTrue(infos["cudart-llama-bin-win-cuda-12.4-x64.zip"]["companion"])
        self.assertEqual(infos["llama-b11158-bin-ubuntu-x64.tar.gz"]["backend"], "cpu")
        self.assertEqual(infos["llama-b11158-bin-macos-arm64.tar.gz"]["backend"], "metal")
        self.assertIsNone(infos["llama-b11158-bin-ubuntu-rocm-10.0-x64.tar.gz"]["backend"])  # ROCm needs system libraries; never auto-picked
        usable = {n for n, i in infos.items() if i and i["backend"]}
        for name in REAL_NAMES:
            if any(odd in name for odd in ["sycl", "openvino", "snapdragon", "adreno", "s390x", "android", "rocm"]):
                self.assertNotIn(name, usable)
        self.assertIsNone(infos["llama-b11158-ui.tar.gz"])
        self.assertIsNone(infos["llama-b11158-xcframework.zip"])

    def test_older_and_drifted_names(self):
        self.assertEqual(ri.classify("llama-b4000-bin-win-cuda-cu12.2.0-x64.zip")["cuda"], (12, 2, 0))
        self.assertEqual(ri.classify("llama-b4000-bin-win-avx2-x64.zip")["backend"], "cpu")
        self.assertEqual(ri.classify("llama-b4000-bin-macos-arm64.zip")["format"], ".zip")
        self.assertEqual(ri.classify("llama-b9000-bin-linux-aarch64.tgz")["arch"], "arm64")
        self.assertIsNone(ri.classify("README.md"))
        self.assertIsNone(ri.classify("llama-b1-bin-win-cuda-x64.zip")["backend"])  # CUDA without a version

    def test_driver_table(self):
        self.assertEqual(ri.max_cuda_for_driver("581.57", "windows"), (13, 0))
        self.assertEqual(ri.max_cuda_for_driver("560.94", "windows"), (12, 6))
        self.assertEqual(ri.max_cuda_for_driver("550.54.14", "linux"), (12, 4))
        self.assertIsNone(ri.max_cuda_for_driver("470.82", "linux"))
        self.assertIsNone(ri.max_cuda_for_driver(None, "linux"))


class ChooseTests(unittest.TestCase):
    def choose(self, hardware, system, machine, **kwargs):
        return ri.choose_asset(release(**kwargs), hardware, system, machine)

    def test_windows_nvidia_new_driver_gets_newest_cuda_and_runtime(self):
        choice = self.choose(nvidia("581.57"), "Windows", "AMD64")
        self.assertEqual(choice["name"], "llama-b11158-bin-win-cuda-13.4-x64.zip")
        self.assertEqual([c["name"] for c in choice["companions"]], ["cudart-llama-bin-win-cuda-13.4-x64.zip"])
        self.assertEqual(choice["backend"], "cuda")
        self.assertEqual(len(choice["sha256"]), 64)
        self.assertIn("CUDA", choice["reason"])
        self.assertTrue(choice["url"].startswith(ri.DOWNLOAD_PREFIX))

    def test_windows_nvidia_older_driver_gets_older_cuda(self):
        choice = self.choose(nvidia("560.94"), "Windows", "AMD64")
        self.assertEqual(choice["name"], "llama-b11158-bin-win-cuda-12.4-x64.zip")
        self.assertEqual(choice["companions"][0]["name"], "cudart-llama-bin-win-cuda-12.4-x64.zip")

    def test_windows_nvidia_too_old_or_unknown_driver_uses_vulkan(self):
        choice = self.choose(nvidia("537.58"), "Windows", "AMD64")
        self.assertEqual(choice["backend"], "vulkan")
        self.assertIn("newer NVIDIA driver", choice["reason"])
        self.assertEqual(self.choose(nvidia(None), "Windows", "AMD64")["backend"], "vulkan")

    def test_windows_other_gpu_and_cpu(self):
        self.assertEqual(self.choose(AMD, "Windows", "AMD64")["name"], "llama-b11158-bin-win-vulkan-x64.zip")
        cpu = self.choose(NONE, "Windows", "AMD64")
        self.assertEqual(cpu["name"], "llama-b11158-bin-win-cpu-x64.zip")
        self.assertIn("no usable graphics card", cpu["reason"])
        self.assertEqual(self.choose(NONE, "Windows", "ARM64")["name"], "llama-b11158-bin-win-cpu-arm64.zip")

    def test_windows_cuda_without_runtime_companion_falls_back(self):
        names = [n for n in REAL_NAMES if not n.startswith("cudart-llama-bin-win")]
        choice = self.choose(nvidia("581.57"), "Windows", "AMD64", names=names)
        self.assertEqual(choice["backend"], "vulkan")
        self.assertIn("runtime files are missing", choice["reason"])

    def test_macos(self):
        arm = self.choose(NONE, "Darwin", "arm64")
        self.assertEqual((arm["name"], arm["backend"]), ("llama-b11158-bin-macos-arm64.tar.gz", "metal"))
        self.assertIn("Metal", arm["reason"])
        self.assertEqual(self.choose(NONE, "Darwin", "x86_64")["name"], "llama-b11158-bin-macos-x64.tar.gz")

    def test_macos_prefers_tar_but_accepts_zip(self):
        names = ["llama-b11158-bin-macos-arm64.zip", "llama-b11158-bin-macos-arm64.tar.gz"]
        self.assertEqual(self.choose(NONE, "Darwin", "arm64", names=names)["name"], names[1])
        self.assertEqual(self.choose(NONE, "Darwin", "arm64", names=names[:1])["name"], names[0])

    def test_linux(self):
        self.assertEqual(self.choose(NONE, "Linux", "x86_64")["name"], "llama-b11158-bin-ubuntu-x64.tar.gz")
        self.assertEqual(self.choose(AMD, "Linux", "x86_64")["name"], "llama-b11158-bin-ubuntu-vulkan-x64.tar.gz")
        self.assertEqual(self.choose(NONE, "Linux", "aarch64")["name"], "llama-b11158-bin-ubuntu-arm64.tar.gz")
        self.assertEqual(self.choose(AMD, "Linux", "aarch64")["name"], "llama-b11158-bin-ubuntu-vulkan-arm64.tar.gz")
        cuda = self.choose(nvidia("580.95.05"), "Linux", "x86_64")
        self.assertEqual(cuda["name"], "llama-b11158-bin-ubuntu-cuda-13.4-x64.tar.gz")
        self.assertEqual(cuda["companions"][0]["name"], "cudart-llama-b11158-bin-ubuntu-cuda-13.4-x64.tar.gz")
        self.assertEqual(self.choose(nvidia("570.172.08"), "Linux", "x86_64")["cuda"], "12.8")
        self.assertEqual(self.choose(nvidia("550.54.14"), "Linux", "x86_64")["backend"], "vulkan")

    def test_forced_cpu_backend(self):
        choice = ri.choose_asset(release(), nvidia(), "Linux", "x86_64", backend="cpu")
        self.assertEqual(choice["name"], "llama-b11158-bin-ubuntu-x64.tar.gz")

    def test_missing_digest_is_refused_unless_allowed(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            ri.choose_asset(release(digest=False), NONE, "Linux", "x86_64")
        choice = ri.choose_asset(release(digest=False), NONE, "Linux", "x86_64", allow_unverified=True)
        self.assertIsNone(choice["sha256"])
        self.assertTrue(choice["unverified"])

    def test_missing_companion_digest_is_refused(self):
        rel = release()
        for asset in rel["assets"]:
            if asset["name"].startswith("cudart"):
                del asset["digest"]
        with self.assertRaisesRegex(ValueError, "checksum"):
            ri.choose_asset(rel, nvidia(), "Windows", "AMD64")

    def test_unofficial_url_and_bad_size_are_refused(self):
        rel = release(names=["llama-b1-bin-ubuntu-x64.tar.gz"])
        rel["assets"][0]["browser_download_url"] = "https://evil.example/llama.tar.gz"
        with self.assertRaisesRegex(ValueError, "official"):
            ri.choose_asset(rel, NONE, "Linux", "x86_64")
        rel = release(names=["llama-b1-bin-ubuntu-x64.tar.gz"])
        rel["assets"][0]["size"] = 50 * 1024**3
        with self.assertRaisesRegex(ValueError, "size"):
            ri.choose_asset(rel, NONE, "Linux", "x86_64")

    def test_nothing_fits(self):
        with self.assertRaisesRegex(ValueError, "no ready-made build"):
            ri.choose_asset(release(), NONE, "FreeBSD", "amd64")
        with self.assertRaisesRegex(ValueError, "no ready-made build"):
            ri.choose_asset({"tag_name": "v0.5.0", "assets": [{"name": "nightly-tag.txt"}]}, NONE, "Linux", "x86_64")


class ReleaseLookupTests(unittest.TestCase):
    def test_latest_used_when_it_has_builds(self):
        net = FakeNet({}, json_by_url={ri.RELEASE_API + "/latest": release()})
        with mock.patch.object(ri, "_open", net):
            self.assertEqual(ri.fetch_release()["tag_name"], "b11158")
        self.assertEqual(len(net.requests), 1)

    def test_version_tag_without_binaries_falls_back_to_build_prereleases(self):
        latest = {"tag_name": "v0.5.0", "assets": [{"name": "nightly-tag.txt", "size": 7}]}
        listing = [{"tag_name": "b11159", "draft": True, "assets": release()["assets"]}, latest, release()]
        net = FakeNet({}, json_by_url={ri.RELEASE_API + "/latest": latest, ri.RELEASE_API + "?per_page=10": listing})
        with mock.patch.object(ri, "_open", net):
            self.assertEqual(ri.fetch_release()["tag_name"], "b11158")

    def test_network_error_is_plain(self):
        with mock.patch.object(ri, "_open", side_effect=OSError("offline")):
            with self.assertRaisesRegex(ValueError, "install-archive"):
                ri.fetch_release()


class VersionParsingTests(unittest.TestCase):
    def test_formats(self):
        plain = ri.parse_version("version: 4589 (1d1e6a90)\nbuilt with cc (Ubuntu 11.4.0) 11.4.0 for x86_64-linux-gnu\n")
        self.assertEqual((plain["version"], plain["build"], plain["commit"], plain["backend"]), ("b4589", 4589, "1d1e6a90", None))
        cuda = ri.parse_version("ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no\nggml_cuda_init: found 1 CUDA devices:\n"
                                "  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes\n"
                                "load_backend: loaded CUDA backend from C:\\llama\\ggml-cuda.dll\n"
                                "load_backend: loaded CPU backend from C:\\llama\\ggml-cpu-haswell.dll\n"
                                "version: 6123 (79bc429)\nbuilt with clang version 19.1.5 for x86_64-pc-windows-msvc\n")
        self.assertEqual((cuda["build"], cuda["backend"]), (6123, "cuda"))
        vulkan = ri.parse_version("ggml_vulkan: Found 1 Vulkan devices:\nggml_vulkan: 0 = AMD Radeon RX 7900 XTX (AMD proprietary driver)\n"
                                  "load_backend: loaded Vulkan backend from /opt/llama/libggml-vulkan.so\nversion: 7000 (deadbee)\n")
        self.assertEqual(vulkan["backend"], "vulkan")
        brew = ri.parse_version("version: 5460 (ab1d8b1c)\nbuilt with Apple clang version 17.0.0 (clang-1700.0.13.3) for arm64-apple-darwin24.4.0\n")
        self.assertEqual((brew["build"], brew["backend"]), (5460, "metal"))
        cpu = ri.parse_version("load_backend: loaded CPU backend from ./libggml-cpu-alderlake.so\nversion: 11158 (d2e5458)\n")
        self.assertEqual(cpu["backend"], "cpu")
        semver = ri.parse_version("version: 0.5.0 (d2e5458)\nbuilt from b11158\n")
        self.assertEqual((semver["version"], semver["build"]), ("0.5.0", 11158))
        self.assertIsNone(ri.parse_version("Segmentation fault")["version"])

    def test_current_format_from_a_real_build(self):
        # tests/integration/samples/version.txt (llama.cpp 4df29be, Aug 2026)
        info = ri.parse_version("\nversion: 0.1.0-dev (build 1, commit 4df29be)\nbuilt with GNU 13.3.0 for Linux x86_64\n")
        self.assertEqual((info["version"], info["build"], info["commit"], info["semver"]), ("b1", 1, "4df29be", "0.1.0-dev"))
        info = ri.parse_version("load_backend: loaded CUDA backend from /x/libggml-cuda.so\n"
                                "version: 0.5.0 (build 11158, commit d2e5458)\n")
        self.assertEqual((info["version"], info["build"], info["backend"]), ("b11158", 11158, "cuda"))


class DeviceListTests(unittest.TestCase):
    def test_cpu_only_build(self):
        # Real output of `llama-server --list-devices` on a CPU-only build.
        self.assertEqual(ri.parse_devices("Available devices:\n  (none)\n"), [])

    def test_gpu_builds(self):
        devices = ri.parse_devices("ggml_cuda_init: found 2 CUDA devices:\n  Device 0: NVIDIA GeForce RTX 4090\n"
                                   "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4090 (24080 MiB, 23000 MiB free)\n"
                                   "  CUDA1: NVIDIA GeForce RTX 3060 (12288 MiB, 12000 MiB free)\n  BLAS: OpenBLAS (0 MiB, 0 MiB free)\n")
        self.assertEqual([(d["name"], d["backend"]) for d in devices],
                         [("CUDA0", "cuda"), ("CUDA1", "cuda"), ("BLAS", "cpu")])
        self.assertEqual(devices[0]["description"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(devices[0]["free"], 23000 * 1024**2)
        for name, backend in [("Vulkan0", "vulkan"), ("ROCm1", "rocm"), ("Metal", "metal"), ("MTL0", "metal"),
                              ("SYCL0", "unknown"), ("CPU", "cpu")]:
            self.assertEqual(ri.device_backend(name), backend, name)

    def test_no_list_means_unknown(self):
        self.assertIsNone(ri.parse_devices("error: invalid argument: --list-devices"))


class ExtractTests(Base):
    def extract(self, data, suffix):
        archive = self.root / ("a" + suffix)
        archive.write_bytes(data)
        return ri.safe_extract(archive, self.root / "out")

    def test_zip_traversal_absolute_drive_and_backslash_are_rejected(self):
        for bad in ["../evil.txt", "/etc/evil", "C:/Windows/evil.dll", "ok/..\\..\\evil.txt"]:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "unsafe"):
                self.extract(zip_bytes({bad: ("x", 0o644)}), ".zip")
        self.assertFalse((self.root / "evil.txt").exists())

    def test_zip_symlinks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "link"):
            self.extract(zip_bytes({}, symlinks=[("llama-server", "/bin/sh")]), ".zip")

    def test_tar_traversal_and_escaping_links_are_rejected(self):
        cases = [(tar_bytes({"../evil": ("x", 0o644)}), "unsafe"),
                 (tar_bytes({"/abs": ("x", 0o644)}), "unsafe"),
                 (tar_bytes({}, [("lib.so", "/etc/passwd", tarfile.SYMTYPE)]), "outside"),
                 (tar_bytes({}, [("a/lib.so", "../../x", tarfile.SYMTYPE)]), "outside"),
                 (tar_bytes({}, [("hard", "../../etc/passwd", tarfile.LNKTYPE)]), "unsafe"),
                 (tar_bytes({}, [("dev", "", tarfile.CHRTYPE)]), "special device")]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.extract(data, ".tar.gz")

    def test_tar_link_to_outside_folder_is_rejected_before_anything_is_written(self):
        outside = self.root / "outside"
        outside.mkdir()
        data = tar_bytes({}, [("sub", "../outside", tarfile.SYMTYPE)])
        with self.assertRaisesRegex(ValueError, "outside"):
            self.extract(data, ".tar.gz")
        self.assertEqual(list(outside.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "symlinks are a POSIX feature")
    def test_link_chain_then_hard_link_cannot_write_outside(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
            for name, kind, link in [("d", tarfile.DIRTYPE, ""), ("d/l", tarfile.SYMTYPE, ".."),
                                     ("e", tarfile.SYMTYPE, "d/l/../evil.txt")]:
                info = tarfile.TarInfo(name)
                info.type, info.linkname = kind, link
                bundle.addfile(info)
            info = tarfile.TarInfo("real")
            info.size = 5
            bundle.addfile(info, io.BytesIO(b"PWNED"))
            info = tarfile.TarInfo("e")
            info.type, info.linkname = tarfile.LNKTYPE, "real"
            bundle.addfile(info)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.extract(buffer.getvalue(), ".tar.gz")
        self.assertFalse((self.root / "evil.txt").exists())

    @unittest.skipIf(os.name == "nt", "symlinks are a POSIX feature")
    def test_file_entry_never_writes_through_an_earlier_link(self):
        data = tar_bytes({"lib.so": ("new", 0o644)}, [])
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
            info = tarfile.TarInfo("lib.so")
            info.type, info.linkname = tarfile.SYMTYPE, "real.so"
            bundle.addfile(info)
            info = tarfile.TarInfo("real.so")
            info.size = 3
            bundle.addfile(info, io.BytesIO(b"old"))
            with tarfile.open(fileobj=io.BytesIO(data)) as other:
                member = other.getmember("lib.so")
                bundle.addfile(member, other.extractfile(member))
        out = self.extract(buffer.getvalue(), ".tar.gz")
        self.assertFalse((out / "lib.so").is_symlink())
        self.assertEqual((out / "real.so").read_text(), "old")

    def test_drive_relative_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "drive-letter"):
            self.extract(zip_bytes({"bin/D:evil.dll": ("x", 0o644)}), ".zip")

    def test_zip_bomb_limit(self):
        with mock.patch.object(ri, "MAX_EXTRACTED_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "unreasonable size"):
                self.extract(zip_bytes({"big": ("x" * 11, 0o644)}), ".zip")

    @unittest.skipIf(os.name == "nt", "symlinks and exec bits are POSIX features")
    def test_tar_inside_links_and_exec_bits_survive(self):
        data = tar_bytes({"build/bin/libllama.so.0.0.1": ("lib", 0o644), "build/bin/llama-server": (GOOD, 0o755)},
                         [("build/bin/libllama.so", "libllama.so.0.0.1", tarfile.SYMTYPE),
                          ("build/bin/copy", "build/bin/libllama.so.0.0.1", tarfile.LNKTYPE)])
        out = self.extract(data, ".tar.gz")
        self.assertEqual(os.readlink(out / "build/bin/libllama.so"), "libllama.so.0.0.1")
        self.assertEqual((out / "build/bin/copy").read_text(), "lib")
        self.assertTrue(os.access(out / "build/bin/llama-server", os.X_OK))
        self.assertEqual(ri.find_bin_dir(out), out / "build" / "bin")

    def test_zip_layouts(self):
        out = self.extract(zip_bytes({"llama-server": (GOOD, 0o755), "ggml-cuda.dll": ("x", 0o644)}), ".zip")
        self.assertEqual(ri.find_bin_dir(out), out)


class DownloadTests(Base):
    def setUp(self):
        super().setUp()
        self.data = bytes(range(256)) * 8
        self.record = {"name": "llama-b1-bin-ubuntu-x64.tar.gz", "url": BASE + "llama-b1-bin-ubuntu-x64.tar.gz",
                       "size": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}
        self.folder = self.root / "dl"

    def test_download_verifies_and_reports_progress(self):
        events = []
        with mock.patch.object(ri, "_open", FakeNet({self.record["name"]: self.data})):
            path = ri._download(self.record, self.folder, progress=events.append)
        self.assertEqual(path.read_bytes(), self.data)
        self.assertEqual(events[-1]["done"], len(self.data))
        self.assertEqual(set(events[0]), {"stage", "done", "total", "message"})

    def test_bad_hash_deletes_partial(self):
        with mock.patch.object(ri, "_open", FakeNet({self.record["name"]: self.data[:-1] + b"!"})):
            with self.assertRaisesRegex(ValueError, "checksum"):
                ri._download(self.record, self.folder)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_oversized_stream_is_stopped(self):
        with mock.patch.object(ri, "_open", FakeNet({self.record["name"]: self.data + b"extra"})):
            with self.assertRaisesRegex(ValueError, "bigger"):
                ri._download(self.record, self.folder)

    def test_cancel_keeps_partial_and_resume_uses_range(self):
        cancel = threading.Event()
        def progress(event):
            if event["done"] >= 512:
                cancel.set()
        net = FakeNet({self.record["name"]: self.data})
        with mock.patch.object(ri, "_open", net):
            with self.assertRaises(Cancelled):
                ri._download(self.record, self.folder, progress=progress, cancel=cancel)
            part = next(self.folder.glob("*.part"))
            kept = part.stat().st_size
            self.assertGreater(kept, 0)
            path = ri._download(self.record, self.folder)
        self.assertEqual(path.read_bytes(), self.data)
        self.assertEqual(net.requests[-1][1].get("Range"), f"bytes={kept}-")

    def test_server_ignoring_range_restarts_cleanly(self):
        partial = self.folder / (self.record["sha256"][:16] + "-" + self.record["name"] + ".part")
        self.folder.mkdir()
        partial.write_bytes(b"garbage-garbage")
        with mock.patch.object(ri, "_open", FakeNet({self.record["name"]: self.data}, honour_range=False)):
            path = ri._download(self.record, self.folder)
        self.assertEqual(path.read_bytes(), self.data)

    def test_complete_partial_gets_416_and_verifies(self):
        self.folder.mkdir()
        partial = self.folder / (self.record["sha256"][:16] + "-" + self.record["name"] + ".part")
        partial.write_bytes(self.data)
        with mock.patch.object(ri, "_open", FakeNet({self.record["name"]: self.data})):
            self.assertEqual(ri._download(self.record, self.folder).read_bytes(), self.data)

    def test_dropped_connection_is_plain_and_keeps_partial(self):
        class Dropping(FakeResponse):
            def read(self, n=-1):
                if self.tell() >= 64:
                    raise ConnectionResetError("reset by peer")
                return super().read(n)
        with mock.patch.object(ri, "_open", return_value=Dropping(self.data)):
            with self.assertRaisesRegex(ValueError, "interrupted"):
                ri._download(self.record, self.folder)
        self.assertEqual(next(self.folder.glob("*.part")).stat().st_size, 64)

    def test_wrong_content_range_restarts(self):
        self.folder.mkdir()
        partial = self.folder / (self.record["sha256"][:16] + "-" + self.record["name"] + ".part")
        partial.write_bytes(self.data[:100])
        calls = []
        def opener(request, timeout=60):
            calls.append(request.get_header("Range"))
            if request.get_header("Range"):
                response = FakeResponse(self.data[50:], 206)
                response.headers = {"Content-Range": f"bytes 50-{len(self.data) - 1}/{len(self.data)}"}
                return response
            return FakeResponse(self.data)
        with mock.patch.object(ri, "_open", opener):
            self.assertEqual(ri._download(self.record, self.folder).read_bytes(), self.data)
        self.assertEqual(calls, ["bytes=100-", None])

    def test_http_error_is_plain(self):
        with mock.patch.object(ri, "_open", FakeNet({})):
            with self.assertRaisesRegex(ValueError, "HTTP 404"):
                ri._download(self.record, self.folder)

    def test_redirect_to_http_is_refused(self):
        handler = ri._HttpsOnly()
        request = mock.Mock(full_url=BASE + "x")
        with self.assertRaisesRegex(ValueError, "insecure"):
            handler.redirect_request(request, None, 302, "Found", {}, "http://example.com/x")


class InstallTests(Base):
    def run_install(self, blobs, hardware, system="Linux", machine="x86_64", names=None, **kwargs):
        rel = release(names=names or list(blobs), blobs=blobs)
        with mock.patch.object(ri, "_open", FakeNet(blobs)), \
                mock.patch.object(ri.platform, "system", return_value=system), \
                mock.patch.object(ri.platform, "machine", return_value=machine):
            return ri.install(self.store, hardware, release=rel, **kwargs)

    def test_linux_cpu_install_then_detect(self):
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes(server_tree())}
        events = []
        result = self.run_install(blobs, NONE, progress=events.append)
        self.assertTrue(result["installed"])
        self.assertEqual((result["source"], result["version"], result["build"], result["backend"]), ("managed", "b5000", 5000, "cpu"))
        folder = runtime_dir(self.store) / "b11158-cpu"
        self.assertEqual(Path(result["directory"]), folder / "llama-b5000")
        self.assertEqual(result["binaries"]["llama-server"], [str(folder / "llama-b5000" / "llama-server")])
        self.assertIsNone(result["binaries"]["llama-cli"])
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual((manifest["tag"], manifest["backend"], manifest["bin_dir"]), ("b11158", "cpu", "llama-b5000"))
        self.assertEqual(list((runtime_dir(self.store) / "downloads").iterdir()), [])
        self.assertFalse(list(runtime_dir(self.store).glob(".staging-*")))
        self.assertEqual({e["stage"] for e in events} >= {"release", "download", "extract", "check"}, True)
        self.assertIn("CPU", result["reason"])
        self.assertEqual(self.store.get("runtime")["build"], 5000)
        self.assertEqual(ri.binary(self.store, "llama-bench"), [str(folder / "llama-b5000" / "llama-bench")])

    def test_linux_cuda_install_merges_runtime_libraries(self):
        blobs = {"llama-b11158-bin-ubuntu-cuda-13.4-x64.tar.gz": tar_bytes(server_tree("build/bin/")),
                 "cudart-llama-b11158-bin-ubuntu-cuda-13.4-x64.tar.gz":
                     tar_bytes({"cudart/libcudart.so.13": ("cuda", 0o644), "cudart/libcublas.so.13": ("blas", 0o644)})}
        result = self.run_install(blobs, nvidia("580.95.05"))
        self.assertEqual(result["backend"], "cuda")
        self.assertTrue((Path(result["directory"]) / "libcudart.so.13").is_file())
        self.assertTrue(Path(result["directory"]).as_posix().endswith("b11158-cuda/build/bin"))

    def test_macos_install(self):
        blobs = {"llama-b11158-bin-macos-arm64.tar.gz": tar_bytes(server_tree("llama-b11158/"))}
        result = self.run_install(blobs, NONE, system="Darwin", machine="arm64")
        self.assertEqual(result["backend"], "metal")

    def test_gpu_build_that_cannot_start_falls_back_to_cpu(self):
        blobs = {"llama-b11158-bin-ubuntu-vulkan-x64.tar.gz": tar_bytes(server_tree(script=BROKEN)),
                 "llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes(server_tree())}
        result = self.run_install(blobs, AMD)
        self.assertEqual(result["backend"], "cpu")
        self.assertIn("vulkan build would not start", result["warnings"][0])
        self.assertFalse((runtime_dir(self.store) / "b11158-vulkan").exists())
        self.assertEqual(list((runtime_dir(self.store) / "downloads").iterdir()), [])

    def test_archive_without_server_is_refused_and_cleaned(self):
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes({"README.md": ("hi", 0o644)})}
        with self.assertRaisesRegex(ValueError, "does not contain llama-server"):
            self.run_install(blobs, NONE)
        self.assertEqual([p.name for p in runtime_dir(self.store).iterdir()], ["downloads"])

    def test_malicious_archive_is_refused(self):
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes({"../../escape": ("x", 0o644)})}
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.run_install(blobs, NONE)
        self.assertFalse((self.root / "escape").exists())

    def test_reinstall_replaces_previous_copy(self):
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes(server_tree())}
        self.run_install(blobs, NONE)
        result = self.run_install(blobs, NONE)
        self.assertTrue(result["installed"])
        self.assertEqual(sorted(p.name for p in runtime_dir(self.store).iterdir()), ["b11158-cpu", "current.json", "downloads"])

    def test_failed_final_move_keeps_previous_install(self):
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes(server_tree())}
        self.run_install(blobs, NONE)
        real_replace = os.replace
        def flaky(src, dst):
            if Path(src).name.startswith(".staging-"):
                raise PermissionError("in use")
            return real_replace(src, dst)
        with mock.patch.object(ri.os, "replace", side_effect=flaky):
            with self.assertRaisesRegex(ValueError, "Could not move"):
                self.run_install(blobs, NONE)
        self.assertTrue(ri.detect(self.store)["installed"])
        self.assertFalse(list(runtime_dir(self.store).glob(".*")))

    def test_cancel_during_download(self):
        cancel = threading.Event()
        cancel.set()
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes(server_tree())}
        with self.assertRaises(Cancelled):
            self.run_install(blobs, NONE, cancel=cancel)
        self.assertFalse(ri.detect(self.store)["installed"])

    def test_unverified_install_warns(self):
        blobs = {"llama-b11158-bin-ubuntu-x64.tar.gz": tar_bytes(server_tree())}
        rel = release(names=list(blobs), blobs=blobs, digest=False)
        with mock.patch.object(ri, "_open", FakeNet(blobs)), \
                mock.patch.object(ri.platform, "system", return_value="Linux"), \
                mock.patch.object(ri.platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(ValueError, "checksum"):
                ri.install(self.store, NONE, release=rel)
            result = ri.install(self.store, NONE, release=rel, allow_unverified=True)
        self.assertTrue(result["installed"])
        self.assertIn("without a checksum", result["warnings"][-1])

    def test_install_archive_offline(self):
        archive = self.root / "llama-b4242-bin-win-vulkan-x64.zip"
        archive.write_bytes(zip_bytes({"llama-server": (GOOD, 0o755), "ggml-vulkan.dll": ("x", 0o644)}))
        result = ri.install_archive(self.store, archive)
        self.assertTrue(result["installed"])
        self.assertEqual(result["backend"], "vulkan")
        self.assertTrue((runtime_dir(self.store) / "b4242-vulkan" / "manifest.json").is_file())
        with self.assertRaisesRegex(ValueError, "No file"):
            ri.install_archive(self.store, self.root / "missing.zip")

    def test_install_archive_unknown_name_detects_backend(self):
        archive = self.root / "my-llama.tar.gz"
        archive.write_bytes(tar_bytes(server_tree(extra={"llama-b5000/libggml-vulkan.so": ("v", 0o644)})))
        result = ri.install_archive(self.store, archive)
        self.assertEqual(result["backend"], "vulkan")
        self.assertTrue((runtime_dir(self.store) / "local-vulkan").is_dir())


class DetectTests(Base):
    def make(self, folder, script=GOOD, names=("llama-server", "llama-bench")):
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            path = folder / ri._exe(name)
            path.write_text(script if name == "llama-server" else GOOD)
            path.chmod(0o755)
        return folder

    def managed(self, build=6000):
        folder = runtime_dir(self.store) / f"b{build}-cpu"
        self.make(folder / "bin")
        (folder / "manifest.json").write_text(json.dumps({"tag": f"b{build}", "build": build, "backend": "cpu", "bin_dir": "bin"}))
        return folder

    def test_nothing_installed(self):
        result = ri.detect(self.store)
        self.assertEqual(result, ri.empty_result())
        with self.assertRaises(ValueError) as caught:
            ri.binary(self.store, "llama-server")
        self.assertEqual(str(caught.exception), ri.NOT_INSTALLED)

    def test_precedence_configured_then_managed_then_path(self):
        path_dir = self.make(self.root / "usr" / "bin")
        with mock.patch.object(ri, "_path_candidates", side_effect=lambda: iter([path_dir])):
            self.assertEqual(ri.detect(self.store)["source"], "path")
            self.managed()
            self.assertEqual(ri.detect(self.store)["source"], "managed")
            configured = self.make(self.root / "mybuild" / "build" / "bin")
            self.store.put("settings", {"runtime_dir": str(self.root / "mybuild")})
            result = ri.detect(self.store)
        self.assertEqual((result["source"], result["directory"]), ("configured", str(configured)))

    def test_broken_configured_dir_warns_and_falls_back(self):
        self.managed()
        self.store.put("settings", {"runtime_dir": str(self.root / "gone")})
        result = ri.detect(self.store)
        self.assertEqual(result["source"], "managed")
        self.assertIn("no longer contains llama-server", result["warnings"][0])
        self.make(self.root / "bad", script=BROKEN)
        self.store.put("settings", {"runtime_dir": str(self.root / "bad")})
        result = ri.detect(self.store)
        self.assertEqual(result["source"], "managed")
        self.assertIn("would not start", result["warnings"][0])

    def test_current_pointer_wins_over_newest(self):
        self.managed(6000)
        self.managed(7000)
        self.assertEqual(ri.detect(self.store)["build"], 5000)  # version output wins over the manifest
        self.assertIn("b7000", ri.detect(self.store)["directory"])
        (runtime_dir(self.store) / "current.json").write_text(json.dumps({"directory": "b6000-cpu"}))
        self.assertIn("b6000", ri.detect(self.store)["directory"])

    def test_leftover_hidden_folders_are_ignored(self):
        leftover = runtime_dir(self.store) / ".staging-abc"
        self.make(leftover / "bin")
        (leftover / "manifest.json").write_text(json.dumps({"tag": "b1", "backend": "cpu", "bin_dir": "bin"}))
        self.assertFalse(ri.detect(self.store)["installed"])

    def test_backend_comes_from_the_device_list(self):
        folder = runtime_dir(self.store) / "b6000-cuda"
        self.make(folder / "bin", script=with_devices(GOOD, "  CUDA0: NVIDIA GeForce RTX 4090 (24080 MiB, 23000 MiB free)"))
        (folder / "manifest.json").write_text(json.dumps({"tag": "b6000", "backend": "cuda", "bin_dir": "bin"}))
        result = ri.detect(self.store)
        self.assertEqual((result["backend"], result["warnings"]), ("cuda", []))
        self.assertEqual(result["devices"], [{"name": "CUDA0", "description": "NVIDIA GeForce RTX 4090", "backend": "cuda"}])
        # Same CUDA build, but the driver is broken: it lists no devices, so it will run on the CPU.
        self.make(folder / "bin", script=with_devices(GOOD))
        result = ri.detect(self.store)
        self.assertEqual((result["backend"], result["devices"]), ("cpu", []))
        self.assertIn("found no usable graphics card", result["warnings"][0])

    def test_self_built_cpu_runtime_is_cpu_not_unknown(self):
        new_style = VERSION_SCRIPT.format(text="version: 0.1.0-dev (build 1, commit 4df29be)\nbuilt with GNU 13.3.0 for Linux x86_64")
        self.make(self.root / "mine", script=with_devices(new_style))
        self.store.put("settings", {"runtime_dir": str(self.root / "mine")})
        result = ri.detect(self.store)
        self.assertEqual((result["version"], result["build"], result["commit"], result["backend"]), ("b1", 1, "4df29be", "cpu"))
        self.make(self.root / "mine", script=with_devices(new_style, "  Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, 24000 MiB free)"))
        self.assertEqual(ri.detect(self.store)["backend"], "vulkan")

    def test_path_uses_which_for_missing_siblings(self):
        path_dir = self.make(self.root / "brew", names=("llama-server",))
        with mock.patch.object(ri, "_path_candidates", side_effect=lambda: iter([path_dir])), \
                mock.patch.object(ri.shutil, "which", side_effect=lambda n: "/elsewhere/" + n if n == "llama-cli" else None):
            result = ri.detect(self.store)
        self.assertEqual(result["binaries"]["llama-cli"], ["/elsewhere/llama-cli"])
        self.assertIsNone(result["binaries"]["llama-bench"])

    def test_binary_uses_cache_and_checks_names(self):
        self.managed()
        ri.detect(self.store)
        with mock.patch.object(ri, "_run_version", side_effect=AssertionError("should use cache")):
            self.assertTrue(ri.binary(self.store, "llama-server")[0].endswith("llama-server"))
        with self.assertRaisesRegex(ValueError, "no llama-perplexity"):
            ri.binary(self.store, "llama-perplexity")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            ri.binary(self.store, "rm")

    def test_use_directory(self):
        with self.assertRaisesRegex(ValueError, "No llama-server"):
            ri.use_directory(self.store, self.root / "empty")
        self.make(self.root / "llama.cpp" / "build" / "bin")
        self.store.put("settings", {"models_dir": "/models"})
        result = ri.use_directory(self.store, self.root / "llama.cpp")
        self.assertEqual(result["source"], "configured")
        self.assertEqual(self.store.get("settings"), {"models_dir": "/models", "runtime_dir": str((self.root / "llama.cpp").resolve())})
        self.make(self.root / "broken", script=BROKEN)
        with self.assertRaisesRegex(ValueError, "would not start"):
            ri.use_directory(self.store, self.root / "broken")


@unittest.skipIf(os.name == "nt", "fake binaries are shell scripts")
class RealProcessTests(unittest.TestCase):
    def test_version_probe_runs_a_real_process(self):
        with tempfile.TemporaryDirectory() as temp:
            server = Path(temp) / "llama-server"
            server.write_text(GOOD)
            server.chmod(0o755)
            ok, output = ri._run_version([str(server)])
            self.assertTrue(ok)
            self.assertEqual(ri.parse_version(output)["build"], 5000)
            self.assertEqual(ri._run_version([str(Path(temp) / "missing")])[0], False)
            store = Store(Path(temp) / "data")
            store.put("settings", {"runtime_dir": temp})
            with mock.patch.object(ri, "_path_candidates", return_value=iter(())):
                self.assertEqual(ri.detect(store)["version"], "b5000")


if __name__ == "__main__":
    unittest.main()
