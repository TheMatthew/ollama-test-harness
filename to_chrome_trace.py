import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://fake-news.local:11434"
# GPU telemetry server runs on the same host as Ollama, different port
GPU_SERVER = OLLAMA_HOST.rsplit(":", 1)[0] + ":28123"
PROMPT = (
    "Write a simple, self-contained C program (under 40 lines of code) that implements a "
    "1D Cellular Automaton using Rule 90 to generate a fractal pattern in the terminal.\n\n"
    "The program should meet the following requirements:\n"
    "* Use a fixed-width array of 64 cells and run for 32 generations.\n"
    "* Start the simulation with a single active cell (represented by `1`) exactly in the middle of the array, with all other cells set to `0`.\n"
    "* In each generation, print the array to the terminal using `#` for active cells and a space for inactive cells.\n"
    "* Compute the next state of each cell using the bitwise XOR operator (`^`) on its left and right neighbors from the current generation.\n"
    "* Use a secondary buffer array to safely calculate the next generation before updating the main array.\n\n"
    "Avoid using complex if/else branching for the cell logic, keeping the code clean, efficient, and readable. "
    "Provide the source code wrapped cleanly inside a single ```c ... ``` block."
)

OUTPUT_JSON = "ollama_benchmark_results.json"
TRACE_JSON = "ollama_benchmark_trace.json"
OUTPUT_BASE = "output"


def model_output_dir(model_name: str) -> str:
    """Return (and create) ./output/[date]/[model]/ for a given model run."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Sanitize the model name so it is safe as a directory component
    safe_model = re.sub(r"[^\w.\-]", "_", model_name)
    out_dir = os.path.join(OUTPUT_BASE, date_str, safe_model)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# GPU polling helpers
# ---------------------------------------------------------------------------


def _fetch_gpu(timeout=5):
    """Return the nvidia_smi_log.gpu dict from the GPU server, or None on error."""
    resp = requests.get(f"{GPU_SERVER}/gpu", timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("nvidia_smi_log", {}).get("gpu", {})


def _parse_numeric(value_str):
    """Strip units and return a float, or None if unparseable."""
    if value_str in (None, "N/A", ""):
        return None
    try:
        return float(str(value_str).split()[0])
    except (ValueError, AttributeError):
        return None


def poll_gpu_server(interval, samples, stop_event, gpu_samples_out):
    """Polls the GPU server and collects full GPU stats until stop_event is set."""
    start = time.monotonic()
    for _ in range(samples):
        if stop_event.is_set():
            break
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - start
        try:
            gpu = _fetch_gpu()
            temp = gpu.get("temperature", {}).get("gpu_temp", "N/A")
            power = gpu.get("gpu_power_readings", {}).get("instant_power_draw", "N/A")
            fb = gpu.get("fb_memory_usage", {})
            util = gpu.get("utilization", {})
            gpu_samples_out.append(
                {
                    "timestamp": ts_str,
                    "elapsed_s": round(elapsed, 2),
                    "temperature": temp,
                    "power": power,
                    "fb_memory_usage": fb,
                    "utilization": util,
                }
            )
        except Exception as e:
            gpu_samples_out.append(
                {
                    "timestamp": ts_str,
                    "elapsed_s": round(elapsed, 2),
                    "error": str(e),
                }
            )
        stop_event.wait(interval)


def log_temps(interval, stop_event, telemetry_out):
    """Dedicated temperature logger thread.

    Samples GPU temperature (and power) every *interval* seconds for the
    entire duration of the Ollama query, regardless of sample limits.
    Each sample is appended to *telemetry_out* as a dict compatible with
    the Chrome-trace hardware-counter format used by convert_to_chrome_trace().
    """
    start = time.monotonic()
    while not stop_event.is_set():
        elapsed = time.monotonic() - start
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            gpu = _fetch_gpu()
            temp_raw = gpu.get("temperature", {}).get("gpu_temp", "N/A")
            temp_max = gpu.get("temperature", {}).get("gpu_temp_max_threshold", "N/A")
            power_raw = gpu.get("gpu_power_readings", {}).get(
                "instant_power_draw", "N/A"
            )
            temp_c = _parse_numeric(temp_raw)
            power_w = _parse_numeric(power_raw)
            telemetry_out.append(
                {
                    "timestamp": ts_str,
                    "elapsed_s": round(elapsed, 2),
                    "temperature": temp_raw,
                    "temperature_c": temp_c,
                    "temperature_max": temp_max,
                    "power_draw_w": power_w,
                }
            )
        except Exception as e:
            print(f"      [TEMP] +{elapsed:5.1f}s  poll failed: {e}")
            telemetry_out.append(
                {
                    "timestamp": ts_str,
                    "elapsed_s": round(elapsed, 2),
                    "temperature_c": None,
                    "power_draw_w": None,
                    "error": str(e),
                }
            )
        stop_event.wait(interval)


# ---------------------------------------------------------------------------
# C code extraction / compilation / testing
# ---------------------------------------------------------------------------


def extract_c_code(response_text):
    """Extracts C code from the first Markdown code block found."""
    match = re.search(r"```c\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def run_unit_test():
    """Runs the compiled binary and verifies its output against expected Rule 90 features.

    Returns:
        (passed: bool, message: str, output: str)
    """
    try:
        result = subprocess.run(
            [BIN_FILE], capture_output=True, text=True, timeout=5, check=True
        )
        program_output = result.stdout
        lines = [line for line in program_output.splitlines() if line.strip()]

        if len(lines) != 32:
            return False, f"Expected 32 generations, got {len(lines)}", program_output

        if not all(len(line) == 64 for line in lines):
            return (
                False,
                "Not all output lines are exactly 64 cells wide",
                program_output,
            )

        initial_gen = lines[0]
        if initial_gen.count("#") != 1 or initial_gen[32] != "#":
            return (
                False,
                "Initial generation does not have a single '#' in the exact center",
                program_output,
            )

        if lines[1].count("#") != 2 or lines[1][31] != "#" or lines[1][33] != "#":
            return (
                False,
                "Generation 1 failed to spawn correct symmetrical children via XOR",
                program_output,
            )

        return True, "All automated unit tests passed successfully.", program_output
    except Exception as e:
        return False, f"Execution failed: {str(e)}", ""


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------


def evaluate_model(model_name, previous_model=None):
    """Processes generation, compilation, and testing workflows for a target model."""
    metrics = {
        "model": model_name,
        "status": "FAIL",
        "tokens_used": 0,
        "time_taken_sec": 0.0,
        "tokens_per_sec": 0.0,
        "compiler_warnings": "",
        "compiler_errors": "",
        "test_notes": "",
        "c_code": None,
        "program_output": None,
        "compile_attempts": 0,
        "iteration_history": [],  # per-attempt records: {attempt, error, c_code}
        "gpu_telemetry": [],  # timestamped temp/power samples collected during inference
        "gpu_samples": [],
    }
    wall_start = time.monotonic()
    wall_start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = model_output_dir(model_name)

    try:
        # Check running models via /api/ps and unload if they don't match the target model
        try:
            ps_resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=100)
            if ps_resp.status_code == 200:
                ps_data = ps_resp.json()
                running_models = ps_data.get("models", [])
                for rm in running_models:
                    rm_name = rm.get("name")
                    if rm_name and rm_name != model_name:
                        print(
                            f"      -> Model in /api/ps ({rm_name}) is not the target model ({model_name}). Unloading..."
                        )
                        unload_payload = {"model": rm_name, "keep_alive": 0}
                        try:
                            requests.post(
                                f"{OLLAMA_HOST}/api/generate",
                                json=unload_payload,
                                timeout=30,
                            )
                        except Exception as ex:
                            print(
                                f"      [WARN] Failed to unload model {rm_name}: {ex}"
                            )
        except Exception as e:
            print(f"      [WARN] Failed to check /api/ps: {e}")

        # LLM Generation via REST API
        payload = {
            "model": model_name,
            "prompt": PROMPT,
            "stream": False,
        }
        print(
            "      -> Sending request to Ollama. Polling GPU every 3s, logging temps every 5s..."
        )

        post_result = [None, None]
        stop_event = threading.Event()
        temp_stop_event = threading.Event()
        telemetry_samples = []
        gpu_samples = []

        def do_post():
            try:
                r = requests.post(
                    f"{OLLAMA_HOST}/api/generate", json=payload, timeout=3600
                )
                r.raise_for_status()
                post_result[0] = r.json()
            except Exception as e:
                post_result[1] = e
            finally:
                stop_event.set()
                temp_stop_event.set()

        post_thread = threading.Thread(target=do_post, daemon=True)
        temp_thread = threading.Thread(
            target=log_temps,
            args=(5, temp_stop_event, telemetry_samples),
            daemon=True,
            name="TempLogger",
        )
        post_thread.start()
        temp_thread.start()
        poll_gpu_server(3, 400, stop_event, gpu_samples)
        post_thread.join()
        temp_thread.join(timeout=10)

        # Execute the benchmark routine
        poll_gpu_server(3, 400, stop_event, gpu_samples)

        # Execute the benchmark routine
        poll_gpu_server(3, 400, stop_event, gpu_samples)

        # --- GPU Utilization Check for Warning ---
        total_utilization = 0.0
        count = 0
        for sample in gpu_samples:
            if "utilization" in sample and sample["utilization"]:
                util_data = sample["utilization"]
                utilization = None
                try:
                    # Best effort to parse utilization percentage (assuming it's a number or string cleanable to one)
                    if isinstance(util_data, dict):
                        for val in util_data.values():
                            val_str = str(val)
                            clean_util = re.sub(r"[^\d.]", "", val_str)
                            if clean_util:
                                utilization = float(clean_util)
                                break
                    elif isinstance(util_data, (int, float)):
                        utilization = float(util_data)

                except Exception:
                    pass  # Skip samples where parsing fails

            if utilization is not None and utilization >= 0.0:
                total_utilization += utilization
                count += 1

        avg_util = total_utilization / count if count > 0 else 0.0
        metrics["gpu_telemetry"] = telemetry_samples
        metrics["gpu_samples"] = gpu_samples
        metrics["run_start_time"] = wall_start_ts
        metrics["avg_utilization_percent"] = avg_util

        # Log utilization warning based on the average
        if count > 1:
            print(
                f"      [UTIL] Average GPU Utilization observed during benchmark: {avg_util:.2f}%"
            )
            if avg_util < 80.0:
                print(
                    "      [WARN] Low average GPU utilization detected (< 80%). Benchmark may fully stress the hardware."
                )

        # Calculate average GPU utilization from all collected samples to provide a warning/metric
        total_utilization = 0.0
        count = 0
        for sample in gpu_samples:
            if "utilization" in sample and sample["utilization"]:
                util_data = sample["utilization"]
                utilization = None
                try:
                    # Best effort to parse utilization percentage (assuming it's a number or string cleanable to one)
                    if isinstance(util_data, dict):
                        for val in util_data.values():
                            val_str = str(val)
                            clean_util = re.sub(r"[^\d.]", "", val_str)
                            if clean_util:
                                utilization = float(clean_util)
                                break
                    elif isinstance(util_data, (int, float)):
                        utilization = float(util_data)

                except Exception:
                    pass  # Skip samples where parsing fails
            if utilization is not None:
                total_utilization += utilization
                count += 1

        avg_util = total_utilization / count if count > 0 else 0.0
        metrics["avg_utilization_percent"] = avg_util

        # Log utilization warning based on the average
        print(
            f"      [UTIL] Average GPU Utilization observed during benchmark: {avg_util:.2f}%"
        )
        if (
            avg_util < 80.0 and count > 1
        ):  # Check to ensure at least a couple of samples were taken
            print(
                "      [WARN] Low average GPU utilization detected (< 80%). Benchmark may not fully stress the hardware."
            )

        if post_result[1] is not None:
            raise post_result[1]
        res_json = post_result[0]

        # Extract Ollama API metrics
        eval_count = res_json.get("eval_count", 0)
        eval_duration_ns = res_json.get("eval_duration", 0)
        wall_elapsed = time.monotonic() - wall_start

        # Seed totals with the initial response; fix attempts will add to these.
        metrics["tokens_used"] = eval_count
        metrics["time_taken_sec"] = (
            (eval_duration_ns / 1_000_000_000.0)
            if eval_duration_ns > 0
            else wall_elapsed
        )
        # tokens_per_sec is computed after the retry loop once all attempts are summed.

        raw_content = res_json.get("response", "")
        c_code = extract_c_code(raw_content)

        if not c_code:
            metrics["compiler_errors"] = (
                "Failed to extract valid C source block from LLM response."
            )
            # Save the individual run report to the output directory
            report_path = os.path.join(out_dir, "ollama-report.json")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=4)
            except Exception as e:
                print(f"      [WARN] Failed to save run report: {e}")
            return metrics

        # ------------------------------------------------------------------
        # Compile / test retry loop — up to MAX_COMPILE_ATTEMPTS iterations.
        # On each failure the error is fed back to the LLM as a follow-up
        # prompt so it can self-correct.
        # ------------------------------------------------------------------
        MAX_COMPILE_ATTEMPTS = 6
        last_error = None

        for attempt in range(1, MAX_COMPILE_ATTEMPTS + 1):
            metrics["compile_attempts"] = attempt

            # On attempts > 1 ask the LLM to fix the previous error
            if attempt > 1:
                fix_prompt = (
                    f"The C code you provided failed to compile or produced incorrect output.\n\n"
                    f"Error (attempt {attempt - 1}):\n{last_error}\n\n"
                    f"Please fix the code and return the corrected version wrapped in a single "
                    f"```c ... ``` block. Do not include any explanation outside the code block."
                )
                print(
                    f"      -> [Attempt {attempt}/{MAX_COMPILE_ATTEMPTS}] Sending fix request to LLM..."
                )
                fix_payload = {
                    "model": model_name,
                    "prompt": fix_prompt,
                    "stream": False,
                }
                try:
                    fix_r = requests.post(
                        f"{OLLAMA_HOST}/api/generate", json=fix_payload, timeout=3600
                    )
                    fix_r.raise_for_status()
                    fix_json = fix_r.json()
                    raw_content = fix_json.get("response", "")
                    # Accumulate token usage across all attempts
                    metrics["tokens_used"] += fix_json.get("eval_count", 0)
                    extra_ns = fix_json.get("eval_duration", 0)
                    metrics["time_taken_sec"] += (
                        (extra_ns / 1_000_000_000.0) if extra_ns > 0 else 0
                    )
                    c_code = extract_c_code(raw_content)
                    if not c_code:
                        last_error = (
                            "Failed to extract valid C source block from LLM response."
                        )
                        metrics["iteration_history"].append(
                            {"attempt": attempt, "error": last_error, "c_code": None}
                        )
                        print(
                            f"      -> [Attempt {attempt}] No C block extracted from LLM response."
                        )
                        continue
                except Exception as e:
                    last_error = f"LLM fix request failed: {e}"
                    metrics["iteration_history"].append(
                        {"attempt": attempt, "error": last_error, "c_code": None}
                    )
                    print(f"      -> [Attempt {attempt}] LLM call failed: {e}")
                    continue

            # File names include attempt number so every iteration is preserved
            suffix = f"_iter{attempt}" if attempt > 1 else ""
            src_file = os.path.join(out_dir, f"verify_rule90{suffix}.c")
            bin_file = os.path.join(out_dir, f"verify_rule90{suffix}")

            # Write C source
            with open(src_file, "w") as f:
                f.write(c_code)
            print(f"      -> [Attempt {attempt}] C source saved: {src_file}")

            # Compile
            compile_res = subprocess.run(
                ["gcc", "-Wall", "-Wextra", src_file, "-o", bin_file],
                capture_output=True,
                text=True,
            )

            if compile_res.returncode != 0:
                last_error = compile_res.stderr.strip()
                metrics["iteration_history"].append(
                    {"attempt": attempt, "error": last_error, "c_code": c_code}
                )
                print(f"      -> [Attempt {attempt}] Compilation FAILED:\n{last_error}")
                continue  # retry

            if compile_res.stderr:
                metrics["compiler_warnings"] = compile_res.stderr

            print(f"      -> [Attempt {attempt}] Binary saved: {bin_file}")

            # Run
            try:
                run_res = subprocess.run(
                    [bin_file], capture_output=True, text=True, timeout=5, check=True
                )
                program_output = run_res.stdout
            except subprocess.CalledProcessError as e:
                last_error = f"Runtime error (exit {e.returncode}): {e.stderr.strip()}"
                metrics["iteration_history"].append(
                    {"attempt": attempt, "error": last_error, "c_code": c_code}
                )
                print(f"      -> [Attempt {attempt}] Execution FAILED: {last_error}")
                continue
            except Exception as e:
                last_error = f"Execution failed: {e}"
                metrics["iteration_history"].append(
                    {"attempt": attempt, "error": last_error, "c_code": c_code}
                )
                print(f"      -> [Attempt {attempt}] Execution FAILED: {last_error}")
                continue

            # Save program output
            out_txt = os.path.join(out_dir, f"output{suffix}.txt")
            with open(out_txt, "w") as f:
                f.write(program_output)
            print(f"      -> [Attempt {attempt}] Output saved: {out_txt}")

            metrics["program_output"] = program_output
            metrics["c_code"] = c_code

            # Validate output
            lines = [line for line in program_output.splitlines() if line.strip()]
            if len(lines) != 32:
                last_error = f"Expected 32 generations, got {len(lines)}"
            elif not all(len(line) == 64 for line in lines):
                last_error = "Not all output lines are exactly 64 cells wide"
            elif lines[0].count("#") != 1 or lines[0][32] != "#":
                last_error = (
                    "Initial generation does not have a single '#' in the exact center"
                )
            elif lines[1].count("#") != 2 or lines[1][31] != "#" or lines[1][33] != "#":
                last_error = (
                    "Generation 1 failed to spawn correct symmetrical children via XOR"
                )
            else:
                last_error = None  # success

            if last_error:
                metrics["iteration_history"].append(
                    {"attempt": attempt, "error": last_error, "c_code": c_code}
                )
                print(
                    f"      -> [Attempt {attempt}] Output validation FAILED: {last_error}"
                )
                continue  # retry with validation error as feedback

            # All checks passed
            metrics["test_notes"] = "All automated unit tests passed successfully."
            status_label = "PASS" if attempt == 1 else f"PASS (iter {attempt})"
            metrics["status"] = status_label
            print(f"      -> [Attempt {attempt}] {status_label}")
            break

        else:
            # Exhausted all attempts
            metrics["compiler_errors"] = (
                last_error or "All compile/test attempts exhausted."
            )
            metrics["test_notes"] = (
                f"Failed after {MAX_COMPILE_ATTEMPTS} attempt(s). Last error: {last_error}"
            )
            print(
                f"      -> FAIL after {MAX_COMPILE_ATTEMPTS} attempt(s). Last error: {last_error}"
            )

        # Recalculate tokens_per_sec from accumulated totals across all attempts
        if metrics["time_taken_sec"] > 0:
            metrics["tokens_per_sec"] = (
                metrics["tokens_used"] / metrics["time_taken_sec"]
            )

    except Exception as e:
        metrics["compiler_errors"] = f"Pipeline execution error: {str(e)}"
        if metrics["time_taken_sec"] == 0.0:
            metrics["time_taken_sec"] = time.monotonic() - wall_start
        if "run_start_time" not in metrics:
            metrics["run_start_time"] = wall_start_ts

    # Save the individual run report to the output directory
    report_path = os.path.join(out_dir, "ollama-report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        print(f"      [WARN] Failed to save run report: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Chrome Trace export  (adapted from to_chrome_trace.py)
# ---------------------------------------------------------------------------

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(ts_str: str) -> float:
    return datetime.strptime(ts_str, TIMESTAMP_FMT).timestamp()


def shorten_model(model: str) -> str:
    """Return a human-readable short name for a model string."""
    name = model
    if name.startswith("hf.co/"):
        name = name[len("hf.co/") :]
    if "/" in name:
        name = name.split("/", 1)[1]
    return name


def convert_to_chrome_trace(benchmarks: list) -> dict:
    """Convert a list of benchmark result dicts to a Chrome Trace JSON structure.

    Reads ``gpu_telemetry`` (collected live during inference by log_temps) for
    hardware counter events.  Falls back to ``cooldown_telemetry`` when present
    so that traces produced by the old standalone to_chrome_trace.py remain
    compatible.

    Chrome Trace Format:
      https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
    """
    trace_events: list = []

    # ---------- find earliest real timestamp across all entries ----------
    earliest_real: float | None = None
    for entry in benchmarks:
        # Prefer gpu_telemetry (live, during inference); fall back to cooldown_telemetry
        telem = entry.get("gpu_telemetry") or entry.get("cooldown_telemetry", [])
        for pt in telem:
            ts = _parse_ts(pt["timestamp"])
            if earliest_real is None or ts < earliest_real:
                earliest_real = ts
        # Also consider run_start_time so failed/timeout runs anchor correctly
        if entry.get("run_start_time"):
            ts = _parse_ts(entry["run_start_time"])
            if earliest_real is None or ts < earliest_real:
                earliest_real = ts

    # ---------- metadata events ----------
    trace_events.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {"name": "Ollama Benchmark"},
        }
    )
    trace_events.append(
        {
            "name": "process_sort_index",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {"sort_index": 0},
        }
    )
    trace_events.append(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": 9999,
            "args": {"name": "Hardware Telemetry"},
        }
    )
    trace_events.append(
        {
            "name": "thread_sort_index",
            "ph": "M",
            "pid": 0,
            "tid": 9999,
            "args": {"sort_index": 9999},
        }
    )
    trace_events.append(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": 9998,
            "args": {"name": "Token Consumption"},
        }
    )
    trace_events.append(
        {
            "name": "thread_sort_index",
            "ph": "M",
            "pid": 0,
            "tid": 9998,
            "args": {"sort_index": 9998},
        }
    )

    # ---------- second pass: per-run events ----------
    virtual_cursor_us: float = 0.0
    cumulative_tokens: int = 0
    prev_temp: float | None = None
    prev_power: float | None = None

    for idx, entry in enumerate(benchmarks):

        # Prefer live telemetry collected during inference; fall back to cooldown
        telem = entry.get("gpu_telemetry") or entry.get("cooldown_telemetry", [])
        cooldown = entry.get("cooldown_telemetry", [])  # still used for cooldown span

        short_name = shorten_model(model)
        tid = idx

        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {"name": short_name},
            }
        )
        trace_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {"sort_index": idx},
            }
        )

        # ---- determine inference start ----
        # Priority: gpu_telemetry timestamp → cooldown-derived → run_start_time → virtual clock.
        if telem and earliest_real is not None:
            infer_start_real = _parse_ts(telem[0]["timestamp"])
            infer_start_us = (infer_start_real - earliest_real) * 1e6
        elif cooldown and earliest_real is not None:
            cooldown_start_real = _parse_ts(cooldown[0]["timestamp"])
            infer_start_real = cooldown_start_real - duration_s
            infer_start_us = (infer_start_real - earliest_real) * 1e6
        elif entry.get("run_start_time") and earliest_real is not None:
            infer_start_real = _parse_ts(entry["run_start_time"])
            infer_start_us = (infer_start_real - earliest_real) * 1e6
        else:
            infer_start_us = virtual_cursor_us

        infer_dur_us = duration_s * 1e6

        # ---- inference span ----
        trace_events.append(
            {
                "name": f"Inference: {short_name}",
                "ph": "B",
                "pid": 0,
                "tid": tid,
                "ts": infer_start_us,
                "args": {
                    "model": model,
                    "status": status,
                    "tokens_used": tokens,
                    "tokens_per_sec": round(tok_per_s, 3),
                    "time_taken_sec": duration_s,
                    "compiler_warnings": warnings,
                    "compiler_errors": errors,
                    "test_notes": notes,
                },
            }
        )
        trace_events.append(
            {
                "name": f"Inference: {short_name}",
                "ph": "E",
                "pid": 0,
                "tid": tid,
                "ts": infer_start_us + infer_dur_us,
            }
        )

        # ---- PASS / FAIL instant marker ----
        marker_name = "✓ PASS" if status == "PASS" else "✗ FAIL"
        trace_events.append(
            {
                "name": marker_name,
                "ph": "i",
                "s": "t",
                "pid": 0,
                "tid": tid,
                "ts": infer_start_us + infer_dur_us,
                "args": {"model": model, "status": status, "test_notes": notes},
            }
        )

        # ---- token counter ----
        trace_events.append(
            {
                "name": "tokens",
                "ph": "C",
                "pid": 0,
                "tid": 9998,
                "ts": infer_start_us,
                "args": {"cumulative_tokens": cumulative_tokens, "tokens_per_sec": 0.0},
            }
        )
        cumulative_tokens += tokens
        trace_events.append(
            {
                "name": "tokens",
                "ph": "C",
                "pid": 0,
                "tid": 9998,
                "ts": infer_start_us + infer_dur_us,
                "args": {
                    "cumulative_tokens": cumulative_tokens,
                    "tokens_per_sec": round(tok_per_s, 3),
                },
            }
        )

        # ---- hardware telemetry counter events (from live gpu_telemetry) ----
        if telem and earliest_real is not None:
            # Find the first valid (non-None) ambient readings to use as a
            # baseline.  Emit a synthetic counter event at infer_start_us so
            # that the counter graph starts from the known ambient level rather
            # than jumping from 0 on the first real sample.  Both temp and
            # delta equal the ambient value, encoding "we rose from 0 → ambient
            # before inference began."  prev_temp/prev_power are primed from
            # these values so the first real sample produces a correct delta.
            ambient_temp = next(
                (
                    p["temperature_c"]
                    for p in telem
                    if p.get("temperature_c") is not None
                ),
                None,
            )
            ambient_power = next(
                (p["power_draw_w"] for p in telem if p.get("power_draw_w") is not None),
                None,
            )
            if ambient_temp is not None or ambient_power is not None:
                trace_events.append(
                    {
                        "name": "Hardware",
                        "ph": "C",
                        "pid": 0,
                        "tid": 9999,
                        "ts": infer_start_us,
                        "args": {
                            "temperature_c": ambient_temp,
                            "power_draw_w": ambient_power,
                            "delta_temperature_c": ambient_temp,  # delta from implicit 0-origin
                            "delta_power_w": ambient_power,
                        },
                    }
                )
                # Prime the running prev values so the first real sample's
                # delta is computed relative to ambient, not to None/0.
                if ambient_temp is not None:
                    prev_temp = ambient_temp
                if ambient_power is not None:
                    prev_power = ambient_power

            for pt in telem:
                ts_us = (_parse_ts(pt["timestamp"]) - earliest_real) * 1e6
                temp = pt.get("temperature_c")
                power = pt.get("power_draw_w")
                if temp is None and power is None:
                    continue  # skip error samples
                delta_temp = (
                    round(temp - prev_temp, 2)
                    if (prev_temp is not None and temp is not None)
                    else 0.0
                )
                delta_power = (
                    round(power - prev_power, 2)
                    if (prev_power is not None and power is not None)
                    else 0.0
                )
                trace_events.append(
                    {
                        "name": "Hardware",
                        "ph": "C",
                        "pid": 0,
                        "tid": 9999,
                        "ts": ts_us,
                        "args": {
                            "temperature_c": temp,
                            "power_draw_w": power,
                            "delta_temperature_c": delta_temp,
                            "delta_power_w": delta_power,
                        },
                    }
                )
                if temp is not None:
                    prev_temp = temp
                if power is not None:
                    prev_power = power

        # ---- optional legacy cooldown span ----
        if cooldown and earliest_real is not None:
            cooldown_start_real = _parse_ts(cooldown[0]["timestamp"])
            cooldown_start_us = (cooldown_start_real - earliest_real) * 1e6
            last_ts_real = _parse_ts(cooldown[-1]["timestamp"])
            cooldown_dur_us = (last_ts_real - cooldown_start_real) * 1e6 + 1e6

            trace_events.append(
                {
                    "name": f"Cooldown: {short_name}",
                    "ph": "B",
                    "pid": 0,
                    "tid": tid,
                    "ts": cooldown_start_us,
                    "args": {
                        "model": model,
                        "cooldown_samples": len(cooldown),
                        "start_temp_c": cooldown[0]["temperature_c"],
                        "end_temp_c": cooldown[-1]["temperature_c"],
                        "start_power_w": cooldown[0]["power_draw_w"],
                        "end_power_w": cooldown[-1]["power_draw_w"],
                    },
                }
            )
            trace_events.append(
                {
                    "name": f"Cooldown: {short_name}",
                    "ph": "E",
                    "pid": 0,
                    "tid": tid,
                    "ts": cooldown_start_us + cooldown_dur_us,
                }
            )
            end_real = last_ts_real + 1.0
            virtual_cursor_us = (end_real - earliest_real) * 1e6 + 1e6
        else:
            virtual_cursor_us = infer_start_us + infer_dur_us + 1e6

    trace_events.sort(key=lambda e: e.get("ts", 0))
    return {"traceEvents": trace_events, "displayTimeUnit": "ms"}


def export_chrome_trace(results: list, output_path: str):
    """Convert *results* to a Chrome Trace file and write it to *output_path*."""
    trace = convert_to_chrome_trace(results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    print(f"Chrome trace written -> {output_path}")
    print("Open chrome://tracing or https://ui.perfetto.dev and load the file.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models on a Rule-90 coding task."
    )
    parser.add_argument(
        "--chrome-trace",
        metavar="FILE",
        nargs="?",
        const=TRACE_JSON,
        default=None,
        help=(
            f"Export a Chrome / Perfetto trace file after the benchmark. "
            f"Optionally specify the output path (default: {TRACE_JSON})."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=OUTPUT_JSON,
        help=f"Path for the benchmark results JSON (default: {OUTPUT_JSON}).",
    )
    args = parser.parse_args()

    print(f"Connecting to Ollama host at {OLLAMA_HOST}...")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        tags_data = response.json()
        models = [m["name"] for m in tags_data.get("models", [])]

        def extract_params(model_string):
            # Searches for one or more digits followed by 'b'
            match = re.search(r"(\d+)b", model_string)
            return int(match.group(1)) if match else 0

        models.sort(key=extract_params)
    except Exception as e:
        print(f"Could not fetch models from Ollama instance: {e}")
        return

    if not models:
        print("No local models found in your Ollama deployment.")
        return

    print(f"Found {len(models)} model(s) to benchmark. Initiating pipeline...\n")

    results = []
    previous_model = None  # Initialize previous model tracker
    #    models = [model for model in models if '128k' not in model]
    total = len(models)
    print("Evaluating models: " + ", ".join(models) + "\n")
    for n, model in enumerate(models, 1):
        print(f"[{n}/{total}] Running benchmark on: {model}...")
        # Pass the current 'model' and 'previous_model' to the evaluation function
        res = evaluate_model(model, previous_model)
        results.append(res)
        previous_model = model  # Update state for the next iteration
        try:
            with open(args.output, "w", encoding="utf-8") as json_file:
                json.dump(results, json_file, indent=4)
        except Exception as e:
            print(f"Failed to write results to JSON file: {e}")

    print("\n" + "=" * 100)
    print(
        f"{'MODEL':<25} | {'STATUS':<14} | {'TRIES':<5} | {'TOKENS':<8} | {'TIME (s)':<8} | {'TOK/s':<8}"
    )
    print("=" * 100)
    for r in results:
        print(
            f"{r['model'][:25]:<25} | {r['status']:<14} | {r.get('compile_attempts', 1):<5} | "
            f"{r['tokens_used']:<8} | {r['time_taken_sec']:<8.2f} | {r['tokens_per_sec']:<8.2f}"
        )
    print("=" * 100)

    if args.chrome_trace:
        print()
        export_chrome_trace(results, args.chrome_trace)


if __name__ == "__main__":
    main()
