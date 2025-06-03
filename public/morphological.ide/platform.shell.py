#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
#------------------------------------------------------------------------------
# Standard Library Imports - 3.13 std libs **ONLY**
#------------------------------------------------------------------------------
import re
import gc
import os
import dis
import sys
import ast
import time
import site
import mmap
import json
import uuid
import math
import cmath
import shlex
import socket
import struct
import shutil
import pickle
import ctypes
import pstats
import weakref
import logging
import tomllib
import pathlib
import asyncio
import inspect
import hashlib
import cProfile
import argparse
import tempfile
import platform
import traceback
import functools
import linecache
import importlib
import threading
import subprocess
import tracemalloc
import http.server
import collections
import http.client
import http.server
import socketserver
from array import array
from io import StringIO
from pathlib import Path
from math import sqrt, pi
from datetime import datetime
from queue import Queue, Empty
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto, IntEnum, StrEnum, Flag
from collections import defaultdict, deque, namedtuple
from functools import reduce, lru_cache, partial, wraps
from contextlib import contextmanager, asynccontextmanager
from importlib.util import spec_from_file_location, module_from_spec
from types import SimpleNamespace, ModuleType,  MethodType, FunctionType, CodeType, TracebackType, FrameType
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, OrderedDict,
    Coroutine, Type, NamedTuple, ClassVar, Protocol, runtime_checkable, AsyncIterator, Iterator
)
#------------------------------------------------------------------------------
# Global Configuration & Utilities
#------------------------------------------------------------------------------
# Platform detection constants
IS_WINDOWS = os.name == 'nt'
IS_POSIX = os.name == 'posix'
benchmark_profiler = cProfile.Profile()
# ANSI color codes for terminal output
_ANSI_COLORS = {
    'reset': '\033[0m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'underline': '\033[4m',
}
def generate_ansi_color(color_name: str) -> str:
    """Generate an ANSI escape code for colored text."""
    return _ANSI_COLORS.get(color_name.lower(), _ANSI_COLORS['reset'])
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr # Log to stderr by default, stdout for command output
)
logger = logging.getLogger(__name__)
try:
    IS_WINDOWS = os.name == 'nt'
    IS_POSIX = os.name == 'posix'
    if IS_WINDOWS:
        if __name__ == '__main__':
            try:
                from ctypes import windll
                from ctypes import wintypes
                from ctypes.wintypes import HANDLE, DWORD, LPWSTR, LPVOID, BOOL
                from pathlib import PureWindowsPath
            except Exception as e:
                print("Error:", e)
    elif IS_POSIX:
        if __name__ == '__main__':
            try:
                import resource
                import fcntl
                import errno
                import signal
            except TimeoutError as e:
                print(e)
except Exception as e:
    print("Error:", e)
#------------------------------------------------------------------------------
# Core Utilities - Process Execution
#------------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionResult:
    """Represents the result of an external command execution."""
    stdout: str
    stderr: str
    returncode: int
    duration_s: float
    def __repr__(self) -> str:
        color_stdout = generate_ansi_color('green')
        color_stderr = generate_ansi_color('red')
        color_return = generate_ansi_color('cyan')
        color_duration = generate_ansi_color('dim')
        reset_color = generate_ansi_color('reset')
        output = f"{color_stdout}STDOUT:{reset_color}\n{self.stdout}\n"
        output += f"{color_stderr}STDERR:{reset_color}\n{self.stderr}\n"
        output += f"{color_return}RETURN CODE:{reset_color} {self.returncode}\n"
        output += f"{color_duration}Duration:{reset_color} {self.duration_s:.3f}s\n"
        return output

class ProcessRunner:
    """
    Provides platform-independent execution of external commands with
    optional timeout, environment variables, and process priority setting.
    This class uses `subprocess.run` for robustness and simplicity.
    """

    @staticmethod
    def _get_windows_creation_flags() -> int:
        """
        Returns the creation flags for subprocess.Popen to set process priority
        on Windows.
        """
        # Using ABOVE_NORMAL_PRIORITY_CLASS (0x00008000)
        # This is a good balance between performance and not starving other processes.
        return 0x00008000

    @staticmethod
    def _get_posix_preexec_fn() -> Callable[[], None]:
        """
        Returns a callable to be used as preexec_fn for subprocess.Popen
        to set process priority on POSIX systems.
        This function runs in the child process just before exec.
        """
        def _set_nice_value():
            try:
                # Set nice value to -10 (higher priority, -20 is max).
                # A lower nice value means higher priority.
                os.nice(-10)
            except OSError as e:
                # PermissionError (EPERM) if not root or RLIMIT_NICE exceeded.
                # Log a warning but don't fail the execution.
                logger.warning(f"Failed to set process nice value: {e}. Running with default priority.")
        return _set_nice_value

    @staticmethod
    def run_command(
        command: List[str],
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        check: bool = False,
        shell: bool = False, # New parameter: run command via shell
    ) -> ExecutionResult:
        """
        Executes an external command in a platform-independent manner.

        Args:
            command (List[str]): The command and its arguments as a list.
                                 If `shell=True`, this list will be joined into a string.
            timeout (Optional[float]): Timeout in seconds for the command to execute.
            env (Optional[Dict[str, str]]): Environment variables for the command.
            capture_output (bool): If True, stdout and stderr are captured and returned.
                                   If False, they are inherited from the parent process's
                                   stdout/stderr streams (e.g., printed directly to console).
            check (bool): If True, raise `subprocess.CalledProcessError` if the command
                          returns a non-zero exit code.
            shell (bool): If True, the command will be executed through the shell.
                          This is necessary for internal shell commands like 'dir' on Windows
                          or shell built-ins on POSIX.

        Returns:
            ExecutionResult: An object containing stdout, stderr, return code, and duration.

        Raises:
            subprocess.TimeoutExpired: If the command times out.
            subprocess.CalledProcessError: If `check` is True and the command returns
                                           a non-zero exit code.
            FileNotFoundError: If `shell` is False and the command executable is not found.
            Exception: For other unexpected execution errors.
        """
        start_time = time.monotonic()
        stdout_val: str = ""
        stderr_val: str = ""
        return_code: int = -1

        try:
            # Prepare arguments for subprocess.run
            kwargs: Dict[str, Any] = {
                'timeout': timeout,
                'env': env,
                'text': True, # Decode stdout/stderr as text
                'capture_output': capture_output,
                'check': check,
                'shell': shell, # Pass the shell flag
            }

            if shell:
                # If shell=True, the command must be a string.
                # shlex.join is good for POSIX. For Windows, a simple space join is often
                # sufficient for basic commands, but for complex arguments, explicit quoting
                # might be needed by the user.
                command_str = shlex.join(command) if IS_POSIX else ' '.join(command)
                kwargs['args'] = command_str
                logger.debug(f"Running command with shell=True: '{command_str}'")
            else:
                # If shell=False, the command must be a list of arguments.
                kwargs['args'] = command
                logger.debug(f"Running command with shell=False: {command}")

            if IS_WINDOWS:
                kwargs['creationflags'] = ProcessRunner._get_windows_creation_flags()
            elif IS_POSIX:
                kwargs['preexec_fn'] = ProcessRunner._get_posix_preexec_fn()

            # Execute the command using subprocess.run
            completed_process = subprocess.run(**kwargs)

            stdout_val = completed_process.stdout if capture_output else ""
            stderr_val = completed_process.stderr if capture_output else ""
            return_code = completed_process.returncode

        except subprocess.TimeoutExpired as e:
            # Capture any partial output before timeout
            stdout_val = e.stdout.decode() if e.stdout else ""
            stderr_val = e.stderr.decode() if e.stderr else ""
            logger.error(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
            raise # Re-raise the exception for the caller to handle
        except subprocess.CalledProcessError as e:
            # Capture output from the failed process
            stdout_val = e.stdout
            stderr_val = e.stderr
            logger.error(f"Command '{' '.join(command)}' failed with exit code {e.returncode}.")
            logger.debug(f"STDOUT:\n{stdout_val.strip()}\nSTDERR:\n{stderr_val.strip()}")
            raise # Re-raise
        except FileNotFoundError:
            logger.error(f"Command '{command[0]}' not found. Please ensure it's in your system's PATH, "
                         f"or use the --shell flag for internal shell commands (e.g., 'dir').")
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred while running command '{' '.join(command)}': {e}", exc_info=True)
            raise
        finally:
            duration = time.monotonic() - start_time
            return ExecutionResult(stdout=stdout_val, stderr=stderr_val,
                                   returncode=return_code, duration_s=duration)

#------------------------------------------------------------------------------
# Core Utilities - Profiling & Benchmarking
#------------------------------------------------------------------------------

class SystemProfiler:
    """
    A singleton class for profiling Python code execution using cProfile.
    Ensures only one profiler instance is active at a time.
    """
    _instance: ClassVar[Optional[SystemProfiler]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> SystemProfiler:
        """Ensures only a single instance of SystemProfiler exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        """Initializes the profiler. Called only once upon first instantiation."""
        self.profiler = cProfile.Profile()
        self._is_enabled = False
        logger.debug("SystemProfiler initialized.")

    def start(self) -> None:
        """Starts the profiler."""
        if not self._is_enabled:
            self.profiler.enable()
            self._is_enabled = True
            logger.debug("SystemProfiler started.")
        else:
            logger.warning("SystemProfiler is already running.")

    def stop(self) -> str:
        """Stops the profiler and returns the profiling statistics as a string."""
        if self._is_enabled:
            self.profiler.disable()
            self._is_enabled = False
            s = StringIO()
            # Sort by cumulative time, then by total time for better insights
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative', 'tottime')
            ps.print_stats()
            # Reset the profiler for subsequent uses without accumulating old data
            self.profiler = cProfile.Profile()
            logger.debug("SystemProfiler stopped and stats generated.")
            return s.getvalue()
        else:
            logger.warning("SystemProfiler is not running. Cannot stop.")
            return "Profiler not running."

    @contextmanager
    def profile_context(self):
        """
        Context manager to profile a block of Python code.
        Note: This profiles the Python code *within* the 'with' block, not external commands.
        """
        self.start()
        try:
            yield
        finally:
            # The stop method is called here, and its output is logged.
            # The caller of the context manager can also capture this if needed
            # by calling stop() directly after the context manager.
            logger.debug("Profiling results from context:\n" + self.stop())


@dataclass(frozen=True)
class BenchmarkReport:
    """Represents the comprehensive results of a command benchmark."""
    command: str
    best_time_s: float
    average_time_s: float
    iterations: int
    all_times_s: Tuple[float, ...]
    profile_data: str # Python cProfile data for the benchmark run itself

    def __repr__(self) -> str:
        command_color = generate_ansi_color('cyan')
        timing_color = generate_ansi_color('green')
        title_color = generate_ansi_color('yellow')
        reset_color = generate_ansi_color('reset')
        dim_color = generate_ansi_color('dim')

        report = f"{title_color}{generate_ansi_color('bold')}Benchmark Report:{reset_color}\n"
        report += f"{command_color}Command:{reset_color} {self.command}\n"
        report += f"{timing_color}Best time:{reset_color} {self.best_time_s:.3f}s\n"
        report += f"{timing_color}Average time:{reset_color} {self.average_time_s:.3f}s\n"
        report += f"{dim_color}Iterations:{reset_color} {self.iterations}\n"
        report += f"{dim_color}All times (s):{reset_color} {[f'{t:.3f}' for t in self.all_times_s]}\n"
        report += f"{title_color}Python Profiling Data (Benchmark Overhead):{reset_color}\n{self.profile_data}"
        return report

class CommandBenchmark:
    """
    Utility to benchmark the execution time of an external command.
    Integrates with SystemProfiler for Python-level profiling of the benchmark process itself.
    """
    def __init__(self, command: List[str], iterations: int = 10, shell: bool = False):
        if not command:
            raise ValueError("Command cannot be empty.")
        if iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")

        self.command = command
        self.iterations = iterations
        self.shell = shell # Store the shell flag
        self.results: List[float] = []
        self.profiler = SystemProfiler() # Get the singleton instance

    def run(self) -> BenchmarkReport:
        """
        Executes the command multiple times and collects performance data.
        Returns a BenchmarkReport containing the results.
        """
        command_str_for_log = shlex.join(self.command) if self.shell else ' '.join(self.command)
        logger.info(f"Starting benchmark for command: {command_str_for_log} ({self.iterations} iterations)")
        self.results = [] # Ensure results are reset for each call to run()

        # Profile the entire benchmark loop, including ProcessRunner calls and Python overhead.
        # This gives insight into the benchmark script's performance, not the external command's.
        self.profiler.start()
        try:
            for i in range(self.iterations):
                logger.debug(f"Running iteration {i+1}/{self.iterations}...")
                try:
                    # Run command silently during benchmark to avoid polluting console
                    # and to ensure consistent timing (no printing overhead from child process)
                    result = ProcessRunner.run_command(
                        self.command,
                        capture_output=False, # Don't capture stdout/stderr for benchmark runs
                        check=True,           # Raise error if command fails
                        shell=self.shell,     # Pass the shell flag
                    )
                    self.results.append(result.duration_s)
                    logger.debug(f"Iteration {i+1} completed in {result.duration_s:.3f}s.")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.error(f"Command failed during benchmark iteration {i+1}: {e}")
                    # If a command fails during benchmark, it's often a critical issue.
                    # Re-raise to stop the benchmark and indicate failure.
                    raise
                except Exception as e:
                    logger.critical(f"An unexpected error occurred during benchmark iteration {i+1}: {e}", exc_info=True)
                    raise
        finally:
            profile_data = self.profiler.stop() # Stop profiling and get the data

        if not self.results:
            logger.error("No successful benchmark runs completed.")
            # Return a report indicating failure or empty results
            return BenchmarkReport(
                command=command_str_for_log,
                best_time_s=float('inf'),
                average_time_s=float('inf'),
                iterations=0,
                all_times_s=(),
                profile_data="No successful runs or profiling data due to errors."
            )

        best_time = min(self.results)
        average_time = sum(self.results) / len(self.results)

        report = BenchmarkReport(
            command=command_str_for_log,
            best_time_s=best_time,
            average_time_s=average_time,
            iterations=self.iterations,
            all_times_s=tuple(self.results),
            profile_data=profile_data
        )
        logger.info(f"Benchmark completed. Best: {best_time:.3f}s, Avg: {average_time:.3f}s.")
        return report

#------------------------------------------------------------------------------
# Main Application Logic
#------------------------------------------------------------------------------

def main() -> int:
    """
    Main entry point for the command-line utility.
    Parses arguments, runs benchmark, and displays results.
    """
    parser = argparse.ArgumentParser(
        description='Benchmark external command execution and profile Python code.',
        formatter_class=argparse.RawTextHelpFormatter, # For better help text formatting
        epilog="""
Examples:
  # Benchmark a simple Python command 10 times:
  python CoreUtils.py --num 10 -- python -c "import time; time.sleep(0.1)"

  # Benchmark 'ls -l' on POSIX systems:
  python CoreUtils.py -- ls -l /tmp

  # Benchmark 'dir' on Windows (requires --shell):
  python CoreUtils.py --shell -- dir

  # Benchmark 'Get-Process' on Windows (requires --shell for PowerShell commands):
  python CoreUtils.py --shell -- powershell.exe -Command "Get-Process"

  # Enable verbose logging:
  python CoreUtils.py --verbose -- python -c "print('hello')"
"""
    )
    parser.add_argument(
        '-n', '--num', type=int, default=10,
        help="Number of iterations for the benchmark (default: 10)."
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Enable verbose logging (DEBUG level)."
    )
    parser.add_argument(
        '--shell', action='store_true',
        help="Execute the command via the system shell (e.g., cmd.exe on Windows, bash on POSIX). "
             "Required for internal shell commands like 'dir' or 'cd', or complex commands with pipes/redirection."
    )
    parser.add_argument(
        'cmd', nargs=argparse.REMAINDER,
        help="The command to execute and benchmark.\n"
             "Use '--' to separate benchmark arguments from the command arguments.\n"
             "Example: `python CoreUtils.py -- python -c \"print('hello')\"`"
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    if not args.cmd:
        parser.error("Command is required. Use '--help' for usage examples.")

    # Remove the '--' separator if it exists
    command_to_run: List[str]
    if args.cmd and args.cmd[0] == '--':
        command_to_run = args.cmd[1:]
    else:
        command_to_run = args.cmd

    if not command_to_run:
        parser.error("No command specified after '--'.")

    logger.info(f"Preparing to benchmark: {' '.join(command_to_run)}")

    try:
        # 1. Run the benchmark
        benchmark = CommandBenchmark(command_to_run, args.num, shell=args.shell)
        benchmark_report = benchmark.run()
        print("\n" + "="*80)
        print(benchmark_report)
        print("="*80 + "\n")

        # 2. Run the command once more to display its actual output
        # This run is separate from the benchmark to clearly show the command's output
        # without interfering with benchmark timings.
        logger.info(f"Executing command once to show full output: {' '.join(command_to_run)}")
        execution_result = ProcessRunner.run_command(
            command_to_run,
            capture_output=True, # Capture output for display
            check=False,         # Don't raise error for non-zero exit code here, just show it
            shell=args.shell,    # Pass the shell flag
        )
        print("\n" + "="*80)
        print(generate_ansi_color('bold') + generate_ansi_color('yellow') + "Single Execution Output:" + generate_ansi_color('reset'))
        print(execution_result)
        print("="*80 + "\n")

        # Return 0 for success, non-zero for failure based on the single execution
        return 0 if execution_result.returncode == 0 else 1

    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Application error: {e}")
        return 1
    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())