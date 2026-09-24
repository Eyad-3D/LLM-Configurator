import hashlib
import json
import os
import platform
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_configurator import hardware
from llm_configurator.domain import GIB

MIB = 1024**2
NVIDIA_CSV = "0, GPU-abc, NVIDIA GeForce RTX 4090, 24564, 20000, 7, 555.42\n"
MAC_ARM_SYSCTL = """machdep.cpu.brand_string: Apple M2 Pro
hw.memsize: 34359738368
hw.optional.arm64: 1
hw.perflevel0.physicalcpu: 8
hw.perflevel1.physicalcpu: 4
hw.optional.arm.FEAT_DotProd: 1
hw.optional.arm.FEAT_I8MM: 0
"""
MAC_INTEL_SYSCTL = """machdep.cpu.brand_string: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
hw.memsize: 17179869184
hw.optional.avx1_0: 1
hw.optional.avx2_0: 1
hw.optional.fma: 1
"""
MAC_ARM_DISPLAYS = {"SPDisplaysDataType": [{"sppci_model": "Apple M2 Pro", "sppci_cores": "19", "spdisplays_vendor": "sppci_vendor_Apple"}]}
MAC_INTEL_DISPLAYS = {"SPDisplaysDataType": [
    {"sppci_model": "Intel UHD Graphics 630", "spdisplays_vram_shared": "1536 MB", "spdisplays_vendor": "Intel"},
    {"sppci_model": "AMD Radeon Pro 5500M", "spdisplays_vram": "8 GB", "spdisplays_vendor": "sppci_vendor_amd"}]}


def completed(stdout, code=0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=code)


class FakeTools:
    """Dispatches subprocess.run by program name; records every call so tests can check the safety rules."""

    def __init__(self, outputs, which=()):
        self.outputs, self.which_names, self.calls = outputs, set(which), []

    def which(self, name):
        return f"/usr/bin/{name}" if name in self.which_names else None

    def run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        name = os.path.basename(argv[0])
        if name == "sysctl" and "iogpu.wired_limit_mb" in argv and len(argv) == 2:
            value = self.outputs.get("wired")
            return completed(f"iogpu.wired_limit_mb: {value}\n" if value is not None else "", 0 if value is not None else 1)
        output = self.outputs.get(name)
        if isinstance(output, Exception):
            raise output
        if output is None:
            raise FileNotFoundError(name)
        if name == "nvidia-smi" and kwargs.get("check") and output == "":
            raise subprocess.CalledProcessError(1, argv)
        return completed(output if isinstance(output, str) else json.dumps(output))


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


def make_card(root, card, vendor, device="0x744c", pci=None, driver="amdgpu", **files):
    """A fake /sys/class/drm/cardN pointing at a fake PCI device directory, like the real symlinks."""
    pci = pci or f"0000:0{card[-1]}:00.0"
    real = os.path.join(root, "devices", pci)
    write(os.path.join(real, "vendor"), vendor + "\n")
    write(os.path.join(real, "device"), device + "\n")
    os.makedirs(os.path.join(root, "drivers", driver), exist_ok=True)
    os.symlink(os.path.join(root, "drivers", driver), os.path.join(real, "driver"))
    for name, value in files.items():
        write(os.path.join(real, name), f"{value}\n")
    os.makedirs(os.path.join(root, "drm", card), exist_ok=True)
    os.symlink(real, os.path.join(root, "drm", card, "device"))
    os.makedirs(os.path.join(root, "drm", f"{card}-DP-1"), exist_ok=True)  # connectors must be ignored


def fake_registry(tree):
    return lambda path: tree.get(path)


def old_fingerprint(gpus, memory_total):
    """The v0.3 algorithm, copied verbatim, to prove the fingerprint is unchanged for existing systems."""
    import psutil
    identity = {"cpu": platform.processor() or platform.machine(), "cores": psutil.cpu_count(),
                "ram": memory_total, "os": platform.platform(),
                "gpus": [{k: gpu[k] for k in ["uuid", "name", "driver"]} for gpu in gpus]}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]


class HardwareTestCase(unittest.TestCase):
    def setUp(self):
        hardware.clear_cache()
        self.addCleanup(hardware.clear_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "drm"))

    def tools(self, outputs, which=()):
        fake = FakeTools(outputs, which)
        for target, value in [("llm_configurator.hardware.subprocess.run", fake.run),
                              ("llm_configurator.hardware.shutil.which", fake.which),
                              ("llm_configurator.hardware.DRM_ROOT", os.path.join(self.root, "drm"))]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return fake

    def system(self, name, machine="x86_64"):
        for target, value in [("llm_configurator.hardware.platform.system", lambda: name),
                              ("llm_configurator.hardware.platform.machine", lambda: machine)]:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_safe_calls(self, fake):
        for argv, kwargs in fake.calls:
            self.assertIsInstance(argv, list)
            self.assertNotIn("shell", kwargs)
            self.assertLessEqual(kwargs["timeout"], 5)

    def assert_engine_safe(self, result):
        for gpu in result["gpus"]:
            self.assertIsInstance(gpu["available"], int)
            self.assertIsInstance(gpu["total"], int)
            self.assertLessEqual(gpu["available"], gpu["total"])
            for key in ["index", "uuid", "name", "driver", "backend", "vendor", "unified"]:
                self.assertIn(key, gpu)
            self.assertIn(gpu["backend"], {"cuda", "metal", "rocm", "vulkan", "unknown"})
        for gpu in result["other_gpus"]:
            self.assertIsNone(gpu["available"])
            self.assertTrue(gpu["reason"])


class NvidiaAndCpuOnlyTests(HardwareTestCase):
    def test_nvidia_entries_and_fingerprint_are_unchanged(self):
        self.system("Linux")
        fake = self.tools({"nvidia-smi": NVIDIA_CSV}, which=["nvidia-smi"])
        make_card(self.root, "card0", "0x10de", driver="nvidia")
        result = hardware.scan(include_processes=False)
        gpu = result["gpus"][0]
        self.assertEqual({k: gpu[k] for k in ["index", "uuid", "name", "total", "available", "utilization", "driver", "backend"]},
                         {"index": 0, "uuid": "GPU-abc", "name": "NVIDIA GeForce RTX 4090", "total": 24564 * MIB,
                          "available": 20000 * MIB, "utilization": 7.0, "driver": "555.42", "backend": "cuda"})
        self.assertEqual((gpu["vendor"], gpu["unified"]), ("nvidia", False))
        self.assertEqual(result["other_gpus"], [])  # the same card seen in sysfs is not listed twice
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["fingerprint"], old_fingerprint(result["gpus"], result["ram_total"]))
        self.assert_safe_calls(fake)

    def test_nvidia_plus_intel_igpu_keeps_fingerprint(self):
        self.system("Linux")
        self.tools({"nvidia-smi": NVIDIA_CSV}, which=["nvidia-smi"])
        make_card(self.root, "card0", "0x8086", device="0x3e92", driver="i915")
        result = hardware.scan(include_processes=False)
        self.assertEqual(len(result["gpus"]), 1)
        self.assertEqual(result["other_gpus"][0]["vendor"], "intel")
        self.assertEqual(result["fingerprint"], old_fingerprint(result["gpus"], result["ram_total"]))

    def test_cpu_only_system(self):
        self.system("Linux")
        self.tools({})
        result = hardware.scan(include_processes=False)
        self.assertEqual((result["gpus"], result["other_gpus"], result["unified_memory"]), ([], [], False))
        self.assertEqual(result["fingerprint"], old_fingerprint([], result["ram_total"]))
        self.assertIn("No GPU with readable memory", result["warnings"][0])
        for key in ["timestamp", "fingerprint", "cpu", "os", "cores", "threads", "cpu_percent", "ram_total",
                    "ram_available", "swap_used", "disk_free", "gpus", "processes", "warnings",
                    "cpu_name", "cpu_features", "unified_memory", "platform"]:
            self.assertIn(key, result)
        self.assertEqual(result["platform"]["system"], "Linux")

    def test_broken_nvidia_smi_is_a_plain_warning(self):
        self.system("Linux")
        self.tools({"nvidia-smi": OSError("boom")}, which=["nvidia-smi"])
        result = hardware.scan(include_processes=False)
        self.assertEqual(result["gpus"], [])
        self.assertIn("unknown, not zero", result["warnings"][0])

    def test_nvidia_card_without_nvidia_smi_is_listed_but_not_used(self):
        self.system("Linux")
        self.tools({})
        make_card(self.root, "card0", "0x10de", device="0x2684", driver="nouveau")
        result = hardware.scan(include_processes=False)
        self.assertEqual(result["gpus"], [])
        self.assertEqual(result["other_gpus"][0]["vendor"], "nvidia")
        self.assertTrue(any("nvidia-smi was not found" in w for w in result["warnings"]))
        self.assert_engine_safe(result)


class AppleTests(HardwareTestCase):
    def mac(self, wired=None, ram_available=20 * GIB):
        self.system("Darwin", "arm64")
        fake = self.tools({"sysctl": MAC_ARM_SYSCTL, "system_profiler": MAC_ARM_DISPLAYS, "wired": wired}, which=["sysctl"])
        memory = SimpleNamespace(total=32 * GIB, available=ram_available)
        with patch("llm_configurator.hardware.psutil.virtual_memory", return_value=memory):
            return hardware.scan(include_processes=False), fake

    def test_m_series_default_limit(self):
        result, fake = self.mac(wired=0)
        gpu = result["gpus"][0]
        self.assertEqual(gpu["uuid"], "apple-m2-pro")
        self.assertEqual((gpu["name"], gpu["backend"], gpu["vendor"], gpu["unified"]), ("Apple M2 Pro", "metal", "apple", True))
        self.assertEqual(gpu["total"], int(32 * GIB * 2 / 3))
        self.assertEqual(gpu["available"], 20 * GIB)  # capped by free system RAM: it is the same memory
        self.assertEqual((gpu["gpu_cores"], gpu["limit_source"]), (19, "estimated_default"))
        self.assertTrue(result["unified_memory"])
        self.assertEqual(result["cpu_name"], "Apple M2 Pro")
        self.assertEqual(result["cpu_features"]["neon"], True)
        self.assertEqual((result["cpu_features"]["dotprod"], result["cpu_features"]["i8mm"]), (True, False))
        self.assertEqual(result["warnings"], [])
        self.assert_engine_safe(result)
        self.assert_safe_calls(fake)

    def test_m_series_wired_limit_override(self):
        result, _ = self.mac(wired=28672, ram_available=30 * GIB)
        gpu = result["gpus"][0]
        self.assertEqual((gpu["total"], gpu["available"], gpu["limit_source"]), (28 * GIB, 28 * GIB, "iogpu.wired_limit_mb"))

    def test_available_never_exceeds_free_ram(self):
        result, _ = self.mac(wired=None, ram_available=3 * GIB)
        self.assertEqual(result["gpus"][0]["available"], 3 * GIB)

    def test_limit_rule(self):
        self.assertEqual(hardware.mac_gpu_limit(16 * GIB, 0), (int(16 * GIB * 2 / 3), "estimated_default"))
        self.assertEqual(hardware.mac_gpu_limit(64 * GIB, None), (48 * GIB, "estimated_default"))
        self.assertEqual(hardware.mac_gpu_limit(16 * GIB, 64 * 1024), (16 * GIB, "iogpu.wired_limit_mb"))

    def test_static_facts_are_cached(self):
        _, fake = self.mac(wired=0)
        with patch("llm_configurator.hardware.psutil.virtual_memory", return_value=SimpleNamespace(total=32 * GIB, available=GIB)):
            hardware.scan(include_processes=False)
        programs = [os.path.basename(argv[0]) for argv, _ in fake.calls]
        self.assertEqual(programs.count("system_profiler"), 1)
        self.assertEqual(sum(1 for argv, _ in fake.calls if "machdep.cpu.brand_string" in argv), 1)

    def test_rosetta_warning(self):
        self.system("Darwin", "x86_64")
        self.tools({"sysctl": MAC_ARM_SYSCTL + "sysctl.proc_translated: 1\n", "system_profiler": MAC_ARM_DISPLAYS, "wired": 0}, which=["sysctl"])
        result = hardware.scan(include_processes=False)
        self.assertIn("Rosetta", result["warnings"][0])

    def test_intel_mac(self):
        self.system("Darwin", "x86_64")
        self.tools({"sysctl": MAC_INTEL_SYSCTL, "system_profiler": MAC_INTEL_DISPLAYS}, which=["sysctl"])
        result = hardware.scan(include_processes=False)
        self.assertEqual(result["gpus"], [])
        self.assertFalse(result["unified_memory"])
        self.assertEqual([g["name"] for g in result["other_gpus"]], ["Intel UHD Graphics 630", "AMD Radeon Pro 5500M"])
        self.assertEqual(result["other_gpus"][1]["total"], 8 * GIB)
        self.assertEqual(result["cpu_features"]["avx2"], True)
        self.assertEqual(result["cpu_features"]["avx512"], False)
        self.assertTrue(result["cpu_name"].startswith("Intel(R) Core(TM) i9"))
        self.assert_engine_safe(result)

    def test_sysctl_failure_is_safe(self):
        self.system("Darwin", "arm64")
        self.tools({"sysctl": OSError("no")}, which=[])
        result = hardware.scan(include_processes=False)
        self.assertEqual(result["gpus"], [])
        self.assertIsNone(result["cpu_name"])
        self.assertTrue(result["warnings"])


class LinuxTests(HardwareTestCase):
    def amd(self):
        make_card(self.root, "card1", "0x1002", mem_info_vram_total=24 * GIB, mem_info_vram_used=2 * GIB,
                  mem_info_gtt_total=16 * GIB, mem_info_gtt_used=0, unique_id="abc123")

    def test_amd_sysfs_only(self):
        self.system("Linux")
        self.amd()
        self.tools({})
        result = hardware.scan(include_processes=False)
        gpu = result["gpus"][0]
        self.assertEqual((gpu["index"], gpu["uuid"], gpu["total"], gpu["available"]), (0, "amdgpu-abc123", 24 * GIB, 22 * GIB))
        self.assertEqual((gpu["backend"], gpu["vendor"], gpu["driver"], gpu["unified"]), ("vulkan", "amd", "amdgpu", False))
        self.assertIn("0x744c", gpu["name"])
        self.assertEqual(result["warnings"], [])
        self.assert_engine_safe(result)

    def test_amd_with_rocm_smi(self):
        self.system("Linux")
        self.amd()
        fake = self.tools({"rocm-smi": {"card0": {"Card Series": "Radeon RX 7900 XTX", "Card Model": "0x744c"}}}, which=["rocm-smi"])
        result = hardware.scan(include_processes=False)
        hardware.scan(include_processes=False)
        gpu = result["gpus"][0]
        self.assertEqual((gpu["name"], gpu["backend"]), ("Radeon RX 7900 XTX", "rocm"))
        self.assertEqual(sum(1 for argv, _ in fake.calls if argv[0].endswith("rocm-smi")), 1)  # cached
        self.assert_safe_calls(fake)

    def test_amd_with_amd_smi(self):
        self.system("Linux")
        self.amd()
        self.tools({"amd-smi": [{"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7800 XT"}}]}, which=["amd-smi"])
        self.assertEqual(hardware.scan(include_processes=False)["gpus"][0]["name"], "AMD Radeon RX 7800 XT")

    def test_amd_after_nvidia_gets_next_index(self):
        self.system("Linux")
        self.amd()
        self.tools({"nvidia-smi": NVIDIA_CSV}, which=["nvidia-smi"])
        result = hardware.scan(include_processes=False)
        self.assertEqual([g["index"] for g in result["gpus"]], [0, 1])
        self.assertNotEqual(result["fingerprint"], old_fingerprint(result["gpus"][:1], result["ram_total"]))

    def test_amd_apu_goes_to_other_gpus(self):
        self.system("Linux")
        make_card(self.root, "card0", "0x1002", device="0x15bf", mem_info_vram_total=512 * MIB, mem_info_vram_used=100 * MIB)
        self.tools({})
        result = hardware.scan(include_processes=False)
        self.assertEqual(result["gpus"], [])
        self.assertEqual(result["other_gpus"][0]["total"], 512 * MIB)
        self.assertTrue(result["other_gpus"][0]["unified"])
        self.assertIn("free memory cannot be read", result["warnings"][0])
        self.assert_engine_safe(result)

    def test_intel_arc(self):
        self.system("Linux")
        make_card(self.root, "card0", "0x8086", device="0x56a0", driver="xe")
        self.tools({})
        result = hardware.scan(include_processes=False)
        other = result["other_gpus"][0]
        self.assertEqual((other["vendor"], other["driver"], other["total"], other["available"]), ("intel", "xe", None, None))
        self.assertEqual(result["gpus"], [])
        self.assert_engine_safe(result)

    def test_cpuinfo_x86_and_arm(self):
        path = os.path.join(self.root, "cpuinfo")
        write(path, "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\nflags\t\t: fpu sse avx avx2 fma f16c avx512f\n"
                    "processor\t: 1\nmodel name\t: ignored\n")
        name, features = hardware.cpu_info("Linux", cpuinfo_path=path)
        self.assertEqual(name, "AMD Ryzen 9 7950X 16-Core Processor")
        self.assertEqual((features["avx2"], features["avx512"], features["neon"]), (True, True, False))
        write(path, "processor\t: 0\nFeatures\t: fp asimd asimddp i8mm\nCPU implementer\t: 0x41\n")
        name, features = hardware.cpu_info("Linux", cpuinfo_path=path)
        self.assertIsNone(name)
        self.assertEqual((features["neon"], features["dotprod"], features["i8mm"], features["sve"]), (True, True, True, False))
        self.assertEqual(hardware.cpu_info("Linux", cpuinfo_path=os.path.join(self.root, "missing")), (None, None))


class WindowsTests(HardwareTestCase):
    CLASS = hardware.WINDOWS_GPU_CLASS

    def registry(self):
        return fake_registry({
            hardware.WINDOWS_CPU_KEY: {"values": {"ProcessorNameString": "  Intel(R) Core(TM) i7-12700H  "}, "subkeys": []},
            self.CLASS: {"values": {}, "subkeys": ["0000", "0001", "0002", "Properties"]},
            self.CLASS + "\\0000": {"values": {"DriverDesc": "NVIDIA GeForce RTX 3060 Laptop GPU", "MatchingDeviceId": "PCI\\VEN_10DE&DEV_2520",
                                               "HardwareInformation.qwMemorySize": 6 * GIB, "DriverVersion": "31.0.15.5222"}},
            self.CLASS + "\\0001": {"values": {"DriverDesc": "Intel(R) Iris(R) Xe Graphics", "MatchingDeviceId": "pci\\ven_8086&dev_46a6",
                                               "HardwareInformation.qwMemorySize": (128 * MIB).to_bytes(8, "little"),
                                               "DriverVersion": "31.0.101.4502"}},
            self.CLASS + "\\0002": {"values": {"DriverDesc": "Microsoft Remote Display Adapter", "MatchingDeviceId": "SWD\\REMOTEDISPLAYENUM"}},
        })

    def test_nvidia_plus_intel_igpu(self):
        self.system("Windows", "AMD64")
        fake = self.tools({"nvidia-smi": NVIDIA_CSV}, which=["nvidia-smi"])
        with patch("llm_configurator.hardware._windows_registry_reader", return_value=self.registry()), \
                patch("llm_configurator.hardware._windows_features", return_value=None), \
                patch("llm_configurator.hardware.os.name", "nt"), \
                patch("llm_configurator.hardware.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True):
            result = hardware.scan(include_processes=False)
        self.assertEqual([g["uuid"] for g in result["gpus"]], ["GPU-abc"])
        self.assertEqual(result["fingerprint"], old_fingerprint(result["gpus"], result["ram_total"]))
        self.assertEqual(len(result["other_gpus"]), 1)
        other = result["other_gpus"][0]
        self.assertEqual((other["name"], other["vendor"], other["total"], other["available"]),
                         ("Intel(R) Iris(R) Xe Graphics", "intel", 128 * MIB, None))
        self.assertEqual(result["cpu_name"], "Intel(R) Core(TM) i7-12700H")
        self.assertEqual(fake.calls[0][1]["creationflags"], 0x08000000)
        self.assert_engine_safe(result)

    def test_nvidia_without_driver_tools(self):
        others, warnings = hardware.windows_gpus(False, self.registry())
        self.assertEqual([o["vendor"] for o in others], ["nvidia", "intel"])
        self.assertEqual(others[0]["total"], 6 * GIB)
        self.assertTrue(warnings)

    def test_unreadable_registry(self):
        self.assertEqual(hardware.windows_gpus(False, fake_registry({}))[0], [])
        self.assertEqual(hardware.cpu_info("Windows", registry=fake_registry({}))[0], None)


class SafetyTests(HardwareTestCase):
    def test_unexpected_error_in_detection_never_breaks_scan(self):
        self.system("Linux")
        self.tools({"nvidia-smi": NVIDIA_CSV}, which=["nvidia-smi"])
        with patch("llm_configurator.hardware.linux_gpus", side_effect=RuntimeError("bad sysfs")):
            result = hardware.scan(include_processes=False)
        self.assertEqual(len(result["gpus"]), 1)
        self.assertIn("Could not check", result["warnings"][0])

    def test_timeouts_become_none(self):
        with patch("llm_configurator.hardware.subprocess.run", side_effect=subprocess.TimeoutExpired("x", 1)):
            self.assertIsNone(hardware.run(["x"]))


if __name__ == "__main__":
    unittest.main()
