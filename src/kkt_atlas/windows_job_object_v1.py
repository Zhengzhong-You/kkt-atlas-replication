"""Minimal Windows Job Object accounting for one worker process tree."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from types import TracebackType
from typing import Self

_JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION = 8
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_JOB_OBJECT_QUERY = 0x0004
_ERROR_ACCESS_DENIED = 5
_TICKS_PER_SECOND = 10_000_000


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicInfo", _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class JobAccounting:
    user_cpu_seconds: float
    kernel_cpu_seconds: float
    total_cpu_seconds: float
    peak_job_memory_bytes: int
    total_processes: int
    active_processes: int
    limit_terminated_processes: int


class ProcessAlreadyInJobError(RuntimeError):
    """Raised when nested Job Object accounting would be ambiguous."""


class WindowsJobObject:
    """Create, populate, query, and close one anonymous Windows Job Object."""

    def __init__(self, name: str | None = None) -> None:
        if name is not None and (not name or "\x00" in name):
            raise ValueError("Job Object name must be nonempty and contain no NUL")
        self.name = name
        self._handle: int | None = None
        self._kernel32 = None

    def __enter__(self) -> Self:
        if self._handle is not None:
            raise RuntimeError("Job Object is already open")
        kernel32 = _windows_kernel32()
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        create_job.restype = ctypes.c_void_p
        handle = create_job(None, self.name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = int(handle)
        self._kernel32 = kernel32
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def assign_popen(self, process: object) -> None:
        """Assign an existing subprocess.Popen-compatible process by PID."""

        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("process.pid must be a positive integer")
        self.assign_pid(pid)

    def assign_pid(self, pid: int) -> None:
        kernel32, _ = self._require_open()
        if pid <= 0:
            raise ValueError("pid must be positive")
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        open_process.restype = ctypes.c_void_p
        access = _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION
        process_handle = open_process(access, False, pid)
        if not process_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self.assign_handle(int(process_handle))
        finally:
            _close_handle(kernel32, int(process_handle))

    def assign_handle(self, process_handle: int) -> None:
        kernel32, job_handle = self._require_open()
        if process_handle <= 0:
            raise ValueError("process_handle must be positive")
        handle = ctypes.c_void_p(process_handle)
        is_process_in_job = kernel32.IsProcessInJob
        is_process_in_job.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
        is_process_in_job.restype = ctypes.c_int32
        already_assigned = ctypes.c_int32()
        if not is_process_in_job(handle, None, ctypes.byref(already_assigned)):
            raise ctypes.WinError(ctypes.get_last_error())
        if already_assigned.value:
            raise ProcessAlreadyInJobError(
                "process is already in a Job Object; nested accounting is not supported—launch it outside "
                "the existing job or use breakaway only when the outer job permits it"
            )

        assign_process = kernel32.AssignProcessToJobObject
        assign_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        assign_process.restype = ctypes.c_int32
        if not assign_process(job_handle, handle):
            error = ctypes.get_last_error()
            if error == _ERROR_ACCESS_DENIED:
                raise ProcessAlreadyInJobError(
                    "Job Object assignment was denied; the process may have entered a non-nestable job "
                    "or the process handle may lack assignment rights"
                ) from ctypes.WinError(error)
            raise ctypes.WinError(error)

    def query_accounting(self) -> JobAccounting:
        """Query accounting after the caller has waited for the process tree."""

        kernel32, job_handle = self._require_open()
        basic = _query_information(
            kernel32,
            job_handle,
            _JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION,
            _JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION,
        )
        limits = _query_information(
            kernel32,
            job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            _JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        )
        return _accounting_from_structures(basic, limits)

    def close(self) -> None:
        if self._handle is None:
            return
        kernel32 = self._kernel32
        handle = self._handle
        _close_handle(kernel32, handle)
        self._handle = None
        self._kernel32 = None

    def _require_open(self):
        if os.name != "nt":
            raise OSError("Windows Job Object API is unavailable on this platform")
        if self._handle is None or self._kernel32 is None:
            raise RuntimeError("Job Object is not open; use it as a context manager")
        return self._kernel32, self._handle


def _query_information(kernel32, job_handle: int, information_class: int, structure_type):
    query = kernel32.QueryInformationJobObject
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    query.restype = ctypes.c_int32
    information = structure_type()
    returned_length = ctypes.c_uint32()
    if not query(
        job_handle,
        information_class,
        ctypes.byref(information),
        ctypes.sizeof(information),
        ctypes.byref(returned_length),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return information


def query_named_job_accounting(name: str) -> JobAccounting:
    """Open a scheduler-owned named Job Object and snapshot its accounting."""

    if not name or "\x00" in name:
        raise ValueError("Job Object name must be nonempty and contain no NUL")
    kernel32 = _windows_kernel32()
    open_job = kernel32.OpenJobObjectW
    open_job.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_wchar_p]
    open_job.restype = ctypes.c_void_p
    handle = open_job(_JOB_OBJECT_QUERY, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        basic = _query_information(
            kernel32,
            int(handle),
            _JOB_OBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION,
            _JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION,
        )
        limits = _query_information(
            kernel32,
            int(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            _JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        )
        return _accounting_from_structures(basic, limits)
    finally:
        _close_handle(kernel32, int(handle))


def _accounting_from_structures(
    basic: _JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION,
    limits: _JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
) -> JobAccounting:
    user_seconds = basic.BasicInfo.TotalUserTime / _TICKS_PER_SECOND
    kernel_seconds = basic.BasicInfo.TotalKernelTime / _TICKS_PER_SECOND
    return JobAccounting(
        user_cpu_seconds=user_seconds,
        kernel_cpu_seconds=kernel_seconds,
        total_cpu_seconds=user_seconds + kernel_seconds,
        peak_job_memory_bytes=int(limits.PeakJobMemoryUsed),
        total_processes=int(basic.BasicInfo.TotalProcesses),
        active_processes=int(basic.BasicInfo.ActiveProcesses),
        limit_terminated_processes=int(basic.BasicInfo.TotalTerminatedProcesses),
    )


def _close_handle(kernel32, handle: int) -> None:
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int32
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_kernel32():
    if os.name != "nt":
        raise OSError("Windows Job Object API is unavailable on this platform")
    return ctypes.WinDLL("kernel32", use_last_error=True)
