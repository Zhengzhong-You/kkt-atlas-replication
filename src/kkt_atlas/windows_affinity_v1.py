"""Deterministic Windows CPU-set discovery and worker affinity."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

_CPU_SET_INFORMATION_TYPE = 0
_CPU_SET_RECORD_SIZE = 32
_ERROR_INSUFFICIENT_BUFFER = 122


@dataclass(frozen=True)
class CpuSet:
    cpu_set_id: int
    group: int
    logical_processor_index: int
    core_index: int
    numa_node_index: int


@dataclass(frozen=True)
class CpuSlot:
    slot_index: int
    group: int
    numa_node_index: int
    cpu_sets: tuple[CpuSet, ...]

    @property
    def cpu_set_ids(self) -> tuple[int, ...]:
        return tuple(cpu.cpu_set_id for cpu in self.cpu_sets)


@dataclass(frozen=True)
class AffinityLayout:
    topology: tuple[CpuSet, ...]
    slots: tuple[CpuSlot, ...]
    slot_size: int


def parse_cpu_set_information(data: bytes) -> tuple[CpuSet, ...]:
    """Parse a GetSystemCpuSetInformation byte buffer."""

    records: list[CpuSet] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("truncated SYSTEM_CPU_SET_INFORMATION header")
        size = int.from_bytes(data[offset : offset + 4], "little")
        record_type = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if size < 8 or offset + size > len(data):
            raise ValueError("invalid SYSTEM_CPU_SET_INFORMATION record size")
        if record_type == _CPU_SET_INFORMATION_TYPE:
            if size < _CPU_SET_RECORD_SIZE:
                raise ValueError("truncated CpuSet record")
            records.append(
                CpuSet(
                    cpu_set_id=int.from_bytes(data[offset + 8 : offset + 12], "little"),
                    group=int.from_bytes(data[offset + 12 : offset + 14], "little"),
                    logical_processor_index=data[offset + 14],
                    core_index=data[offset + 15],
                    numa_node_index=data[offset + 17],
                )
            )
        offset += size

    ordered = tuple(sorted(records, key=_cpu_sort_key))
    ids = [cpu.cpu_set_id for cpu in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("CPU-set identifiers must be unique")
    return ordered


def build_cpu_slots(cpu_sets: Iterable[CpuSet], slot_size: int = 8) -> tuple[CpuSlot, ...]:
    """Build disjoint same-group, same-NUMA slots using one thread per core."""

    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    topology = tuple(sorted(cpu_sets, key=_cpu_sort_key))
    ids = [cpu.cpu_set_id for cpu in topology]
    if len(ids) != len(set(ids)):
        raise ValueError("CPU-set identifiers must be unique")

    pools: dict[tuple[int, int], dict[int, CpuSet]] = {}
    for cpu in topology:
        cores = pools.setdefault((cpu.group, cpu.numa_node_index), {})
        current = cores.get(cpu.core_index)
        if current is None or (cpu.logical_processor_index, cpu.cpu_set_id) < (
            current.logical_processor_index,
            current.cpu_set_id,
        ):
            cores[cpu.core_index] = cpu

    slots: list[CpuSlot] = []
    for (group, numa), core_map in sorted(pools.items()):
        representatives = tuple(sorted(core_map.values(), key=_cpu_sort_key))
        for start in range(0, len(representatives) - slot_size + 1, slot_size):
            selected = representatives[start : start + slot_size]
            slots.append(CpuSlot(len(slots), group, numa, selected))
    return tuple(slots)


def make_affinity_layout(cpu_sets: Iterable[CpuSet], slot_size: int = 8) -> AffinityLayout:
    topology = tuple(sorted(cpu_sets, key=_cpu_sort_key))
    return AffinityLayout(topology, build_cpu_slots(topology, slot_size), slot_size)


def canonical_affinity_json(layout: AffinityLayout) -> str:
    """Return canonical JSON containing both discovered topology and slots."""

    payload = {
        "schema": "kkt-atlas-windows-affinity-v1",
        "slot_size": layout.slot_size,
        "topology": [_cpu_payload(cpu) for cpu in sorted(layout.topology, key=_cpu_sort_key)],
        "slots": [
            {
                "slot_index": slot.slot_index,
                "group": slot.group,
                "numa_node_index": slot.numa_node_index,
                "cpu_set_ids": list(slot.cpu_set_ids),
                "logical_processor_indices": [cpu.logical_processor_index for cpu in slot.cpu_sets],
                "core_indices": [cpu.core_index for cpu in slot.cpu_sets],
            }
            for slot in layout.slots
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_affinity_sha256(layout: AffinityLayout) -> str:
    return hashlib.sha256(canonical_affinity_json(layout).encode("utf-8")).hexdigest()


def read_windows_cpu_sets() -> tuple[CpuSet, ...]:
    """Discover CPU sets through GetSystemCpuSetInformation on Windows."""

    kernel32 = _windows_kernel32()
    get_information = kernel32.GetSystemCpuSetInformation
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p

    required = ctypes.c_uint32()
    succeeded = get_information(None, 0, ctypes.byref(required), get_current_process(), 0)
    error = ctypes.get_last_error()
    if not succeeded and error not in {0, _ERROR_INSUFFICIENT_BUFFER}:
        raise ctypes.WinError(error)
    if required.value == 0:
        raise OSError("GetSystemCpuSetInformation returned an empty topology")

    buffer = ctypes.create_string_buffer(required.value)
    returned = ctypes.c_uint32(required.value)
    if not get_information(buffer, required.value, ctypes.byref(returned), get_current_process(), 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return parse_cpu_set_information(buffer.raw[: returned.value])


def discover_windows_affinity_layout(slot_size: int = 8) -> AffinityLayout:
    return make_affinity_layout(read_windows_cpu_sets(), slot_size)


def apply_current_process_cpu_slot(slot: CpuSlot) -> None:
    """Apply one slot to the current worker process."""

    apply_current_process_cpu_set_ids(slot.cpu_set_ids)


def apply_current_process_cpu_set_ids(cpu_set_ids: Iterable[int]) -> None:
    """Apply explicit CPU-set identifiers to the current Windows process."""

    kernel32 = _windows_kernel32()
    set_default_cpu_sets = kernel32.SetProcessDefaultCpuSets
    set_default_cpu_sets.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    set_default_cpu_sets.restype = ctypes.c_int
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p

    values = tuple(int(value) for value in cpu_set_ids)
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("cpu_set_ids must be distinct nonnegative integers")
    identifiers = (ctypes.c_uint32 * len(values))(*values)
    if not set_default_cpu_sets(get_current_process(), identifiers, len(identifiers)):
        raise ctypes.WinError(ctypes.get_last_error())


def apply_affinity_from_environment() -> tuple[int, ...]:
    """Apply KKT_ATLAS_CPU_SET_IDS when a scheduler supplies it."""

    if os.name != "nt":
        return ()
    raw = os.environ.get("KKT_ATLAS_CPU_SET_IDS")
    if raw is None:
        return ()
    try:
        values = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError("KKT_ATLAS_CPU_SET_IDS must be comma-separated integers") from error
    apply_current_process_cpu_set_ids(values)
    return values


def _windows_kernel32():
    if os.name != "nt":
        raise OSError("Windows CPU-set API is unavailable on this platform")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _cpu_sort_key(cpu: CpuSet) -> tuple[int, int, int, int, int]:
    return (
        cpu.group,
        cpu.numa_node_index,
        cpu.core_index,
        cpu.logical_processor_index,
        cpu.cpu_set_id,
    )


def _cpu_payload(cpu: CpuSet) -> dict[str, int]:
    return {
        "cpu_set_id": cpu.cpu_set_id,
        "group": cpu.group,
        "logical_processor_index": cpu.logical_processor_index,
        "core_index": cpu.core_index,
        "numa_node_index": cpu.numa_node_index,
    }
