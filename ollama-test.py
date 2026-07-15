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
OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST", "http://localhost:11434"
)  # default Ollama REST API port
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


def save_chat_history(out_dir, messages, filename="chat_history.json"):
    """Write the full chat transcript (system/user/assistant turns) to out_dir.

    Called after every turn so a partial transcript is always on disk even
    if the run later crashes or times out.
    """
    path = os.path.join(out_dir, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        print(f"      [WARN] Failed to save chat history: {e}")
    return path


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


# ---------------------------------------------------------------------------
# Shared event log
# ---------------------------------------------------------------------------
#
# One EventLog instance lives for the whole program run and is shared by
# every thread: the continuous hardware telemetry poller, each model's
# inference/fix-request POST thread, and the main thread doing compiles and
# runs. All timestamps are microseconds since the log was created, taken
# from a single monotonic clock, so events from different threads interleave
# correctly without needing to reconcile separate per-model clocks.
#
# This replaces the old design where a fresh temperature-logging thread was
# started and stopped for every single model. That per-model restart is
# what caused the "drift": each model's trace segment re-seeded a synthetic
# "ambient baseline" delta at its start, so every model boundary injected a
# spurious jump into delta_power_w. With one continuous stream for the
# entire program, every delta is a real difference between two consecutive
# real readings — no reseeding, no synthetic jumps, and no per-model gaps.


class EventLog:
    """Thread-safe, append-only log of trace events shared across threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events = []
        self._t0 = time.monotonic()

    def _now_us(self):
        return (time.monotonic() - self._t0) * 1e6

    def add(self, **fields):
        fields.setdefault("ts_us", self._now_us())
        with self._lock:
            self._events.append(fields)

    def span_begin(self, name, model, **args):
        self.add(kind="span", phase="B", name=name, model=model, args=args)

    def span_end(self, name, model, **args):
        self.add(kind="span", phase="E", name=name, model=model, args=args)

    def hw_sample(self, temperature_c, power_draw_w, timestamp, error=None):
        self.add(
            kind="hw_sample",
            temperature_c=temperature_c,
            power_draw_w=power_draw_w,
            timestamp=timestamp,
            error=error,
        )

    def cooldown_begin(self, from_model):
        """Mark the start of a GPU idle/cooldown period, right after a
        model's work (inference + all compile/run attempts) is done."""
        self.add(kind="cooldown", phase="B", from_model=from_model, to_model=None)

    def cooldown_end(self, to_model):
        """Mark the end of a GPU idle/cooldown period — either the next
        model's work is about to start, or (to_model=None) the whole
        program is about to exit."""
        self.add(kind="cooldown", phase="E", from_model=None, to_model=to_model)

    def snapshot(self):
        """Thread-safe copy of every event recorded so far, unsorted."""
        with self._lock:
            return list(self._events)


def telemetry_worker(interval, stop_event, event_log):
    """Continuously samples GPU temperature/power for the entire program.

    Started once in main() before any model is evaluated, and stopped once
    after the last model finishes — never restarted per model — so the
    hardware timeline is one unbroken stream for the whole run.
    """
    while not stop_event.is_set():
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            gpu = _fetch_gpu()
            temp_c = _parse_numeric(gpu.get("temperature", {}).get("gpu_temp", "N/A"))
            power_w = _parse_numeric(
                gpu.get("gpu_power_readings", {}).get("instant_power_draw", "N/A")
            )
            event_log.hw_sample(temp_c, power_w, ts_str)
        except Exception as e:
            print(f"      [TEMP] poll failed: {e}")
            event_log.hw_sample(None, None, ts_str, error=str(e))
        stop_event.wait(interval)


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


def validate_output(program_output):
    """Validate program output against every Rule-90 requirement.

    Unlike a short-circuiting check, this runs *all* checks and returns
    every failure at once, each with the specific line/index/value involved,
    so a human or the LLM gets complete, actionable feedback in a single
    pass instead of a back-and-forth discovering one problem per retry.

    Returns:
        (passed: bool, errors: list[str], notes: str)
    """
    errors = []
    lines = [line for line in program_output.splitlines() if line.strip()]

    # ---- generation count ----
    if len(lines) != 32:
        errors.append(f"Expected 32 generations (non-blank lines), got {len(lines)}.")

    # ---- line width: report every offending line individually ----
    for i, line in enumerate(lines):
        if len(line) != 64:
            errors.append(
                f"Generation {i} (output line {i + 1}) is {len(line)} characters wide, expected 64."
            )

    # ---- generation 0: exactly one '#', centered at index 32 ----
    if len(lines) >= 1:
        gen0 = lines[0]
        count0 = gen0.count("#")
        if count0 != 1:
            errors.append(
                f"Generation 0 has {count0} '#' character(s), expected exactly 1."
            )
        elif len(gen0) <= 32:
            errors.append(
                f"Generation 0 is only {len(gen0)} characters wide, too short to check for '#' at center index 32."
            )
        elif gen0[32] != "#":
            errors.append(
                f"Generation 0's single '#' is at index {gen0.index('#')}, expected index 32 (center)."
            )
    else:
        errors.append("No generation 0 to check: program produced no output lines.")

    # ---- generation 1: symmetric XOR spawn at indices 31 and 33 ----
    if len(lines) >= 2:
        gen1 = lines[1]
        count1 = gen1.count("#")
        if count1 != 2:
            errors.append(
                f"Generation 1 has {count1} '#' character(s), expected exactly 2."
            )
        elif len(gen1) <= 33:
            errors.append(
                f"Generation 1 is only {len(gen1)} characters wide, too short to check indices 31 and 33."
            )
        else:
            left_ok = gen1[31] == "#"
            right_ok = gen1[33] == "#"
            if not left_ok:
                errors.append(
                    f"Generation 1 index 31 is '{gen1[31]}', expected '#' (left child of center)."
                )
            if not right_ok:
                errors.append(
                    f"Generation 1 index 33 is '{gen1[33]}', expected '#' (right child of center)."
                )
    else:
        errors.append(
            "No generation 1 to check: program produced fewer than 2 output lines."
        )

    passed = not errors
    notes = (
        "All automated unit tests passed successfully."
        if passed
        else f"{len(errors)} check(s) failed."
    )
    return passed, errors, notes


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------


def evaluate_model(model_name, event_log, previous_model=None):
    """Processes generation, compilation, and testing workflows for a target model.

    *event_log* is a single EventLog instance shared across the whole program
    run (see main()) — hardware telemetry, inference/fix-request spans, and
    compile/run spans for every model all land in the same thread-safe log.
    """
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
        "compile_durations_sec": [],  # per-attempt compile wall-time
        "total_compile_time_sec": 0.0,
        "iteration_history": [],  # per-attempt records: {attempt, error, c_code}
        "gpu_samples": [],
        "chat_history_file": None,
    }
    wall_start = time.monotonic()
    wall_start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Output dir is created up front (not just on success) so the chat
    # transcript can be saved even if generation/compilation never succeeds.
    out_dir = model_output_dir(model_name)

    # Chat transcript, built up turn by turn and saved to out_dir after each turn.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful C programmer. Respond only with the requested "
                "code, wrapped in a single ```c ... ``` block, and no explanation "
                "outside the code block."
            ),
        },
        {"role": "user", "content": PROMPT},
    ]
    metrics["chat_history_file"] = save_chat_history(out_dir, messages)

    try:
        # LLM Generation via REST API (chat endpoint, so context/system prompt persist across retries)
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        print(
            f"      -> Sending request to Ollama. Polling GPU every 3s (hardware telemetry runs continuously in the background)..."
        )

        post_result = [None, None]
        stop_event = threading.Event()
        gpu_samples = []

        def do_post():
            event_log.span_begin("Inference", model_name)
            try:
                r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=3600)
                r.raise_for_status()
                post_result[0] = r.json()
            except Exception as e:
                post_result[1] = e
            finally:
                event_log.span_end("Inference", model_name)
                stop_event.set()

        post_thread = threading.Thread(target=do_post, daemon=True)
        post_thread.start()
        poll_gpu_server(3, 400, stop_event, gpu_samples)
        post_thread.join()

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

        raw_content = res_json.get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": raw_content})
        metrics["chat_history_file"] = save_chat_history(out_dir, messages)

        c_code = extract_c_code(raw_content)

        if not c_code:
            metrics["compiler_errors"] = (
                "Failed to extract valid C source block from LLM response."
            )
            return metrics

        # ------------------------------------------------------------------
        # Compile / test retry loop — up to MAX_COMPILE_ATTEMPTS iterations.
        # On each failure the error is fed back to the LLM as a follow-up
        # chat turn (with full prior context) so it can self-correct.
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
                messages.append({"role": "user", "content": fix_prompt})
                metrics["chat_history_file"] = save_chat_history(out_dir, messages)
                fix_payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                }
                event_log.span_begin("Fix Request", model_name, attempt=attempt)
                try:
                    fix_r = requests.post(
                        f"{OLLAMA_HOST}/api/chat", json=fix_payload, timeout=3600
                    )
                    fix_r.raise_for_status()
                    fix_json = fix_r.json()
                    raw_content = fix_json.get("message", {}).get("content", "")
                    messages.append({"role": "assistant", "content": raw_content})
                    metrics["chat_history_file"] = save_chat_history(out_dir, messages)
                    event_log.span_end("Fix Request", model_name, attempt=attempt)
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
                    event_log.span_end(
                        "Fix Request", model_name, attempt=attempt, error=str(e)
                    )
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

            # Compile (timed + traced)
            event_log.span_begin("Compile", model_name, attempt=attempt)
            compile_start = time.monotonic()
            compile_res = subprocess.run(
                ["gcc", "-Wall", "-Wextra", src_file, "-o", bin_file],
                capture_output=True,
                text=True,
            )
            compile_dur_sec = round(time.monotonic() - compile_start, 3)
            event_log.span_end(
                "Compile",
                model_name,
                attempt=attempt,
                returncode=compile_res.returncode,
                duration_sec=compile_dur_sec,
            )
            metrics["compile_durations_sec"].append(compile_dur_sec)
            metrics["total_compile_time_sec"] += compile_dur_sec
            print(f"      -> [Attempt {attempt}] Compile took {compile_dur_sec:.3f}s")

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

            # Run (timed + traced)
            event_log.span_begin("Run", model_name, attempt=attempt)
            try:
                run_res = subprocess.run(
                    [bin_file], capture_output=True, text=True, timeout=5, check=True
                )
                program_output = run_res.stdout
                event_log.span_end("Run", model_name, attempt=attempt, returncode=0)
            except subprocess.CalledProcessError as e:
                event_log.span_end(
                    "Run", model_name, attempt=attempt, returncode=e.returncode
                )
                last_error = f"Runtime error (exit {e.returncode}): {e.stderr.strip()}"
                metrics["iteration_history"].append(
                    {"attempt": attempt, "error": last_error, "c_code": c_code}
                )
                print(f"      -> [Attempt {attempt}] Execution FAILED: {last_error}")
                continue
            except Exception as e:
                event_log.span_end("Run", model_name, attempt=attempt, error=str(e))
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

            # Validate output — every check runs (not just the first
            # failure), so feedback to the LLM is complete and specific.
            passed, validation_errors, validation_notes = validate_output(
                program_output
            )

            if not passed:
                last_error = "\n".join(f"- {e}" for e in validation_errors)
                metrics["iteration_history"].append(
                    {"attempt": attempt, "error": last_error, "c_code": c_code}
                )
                print(
                    f"      -> [Attempt {attempt}] Output validation FAILED ({len(validation_errors)} issue(s)):"
                )
                for e in validation_errors:
                    print(f"           - {e}")
                continue  # retry with all validation errors as feedback

            # All checks passed
            metrics["test_notes"] = validation_notes
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

    return metrics


# ---------------------------------------------------------------------------
# Chrome Trace export  (adapted from to_chrome_trace.py)
# ---------------------------------------------------------------------------


def shorten_model(model: str) -> str:
    """Return a human-readable short name for a model string."""
    name = model
    if name.startswith("hf.co/"):
        name = name[len("hf.co/") :]
    if "/" in name:
        name = name.split("/", 1)[1]
    return name


def convert_to_chrome_trace(results: list, event_log: "EventLog") -> dict:
    """Build a Chrome Trace JSON structure from the shared EventLog plus
    per-model *results* (for token/PASS-FAIL summary data).

    Because every event (hardware sample, Inference/Fix Request/Compile/Run
    span) was logged to a single continuous, thread-safe EventLog spanning
    the whole program run, this is a straight replay: no timestamp
    reconstruction, no per-model synthetic "ambient baseline" seeding, and
    no gaps at model boundaries. The hardware counter track is one
    unbroken stream, so delta_power_w / delta_temperature_c are always the
    difference between two real consecutive readings.

    Chrome Trace Format:
      https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
    """
    events = sorted(event_log.snapshot(), key=lambda e: e["ts_us"])
    trace_events: list = []

    # ---------- metadata ----------
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
    trace_events.append(
        {
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": 9997,
            "args": {"name": "Cooldown"},
        }
    )
    trace_events.append(
        {
            "name": "thread_sort_index",
            "ph": "M",
            "pid": 0,
            "tid": 9997,
            "args": {"sort_index": 9997},
        }
    )

    # One lane per model, in the order results were produced.
    tid_of = {}
    for idx, r in enumerate(results):
        tid_of[r["model"]] = idx
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": idx,
                "args": {"name": shorten_model(r["model"])},
            }
        )
        trace_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 0,
                "tid": idx,
                "args": {"sort_index": idx},
            }
        )

    # ---------- hardware counter track: one continuous stream, no resets ----------
    prev_temp: float | None = None
    prev_power: float | None = None
    for e in events:
        if e.get("kind") != "hw_sample":
            continue
        temp = e.get("temperature_c")
        power = e.get("power_draw_w")
        if temp is None and power is None:
            continue  # both unread this tick: nothing to plot, nothing faked

        args = {}
        if temp is not None:
            args["temperature_c"] = temp
            args["delta_temperature_c"] = (
                round(temp - prev_temp, 2) if prev_temp is not None else 0.0
            )
            prev_temp = temp
        if power is not None:
            args["power_draw_w"] = power
            args["delta_power_w"] = (
                round(power - prev_power, 2) if prev_power is not None else 0.0
            )
            prev_power = power
        # If only one of the two was read this tick, the other key is simply
        # omitted rather than reported as a fake 0.0 delta or dropped to 0 —
        # the counter for that series just holds its last known value.

        trace_events.append(
            {
                "name": "Hardware",
                "ph": "C",
                "pid": 0,
                "tid": 9999,
                "ts": e["ts_us"],
                "args": args,
            }
        )

    # ---------- cooldown spans: GPU idle time between models ----------
    # Reuses the same continuous hardware stream to report how much the
    # GPU actually cooled (temp/power) during each idle window, rather than
    # just marking that idle time occurred.
    hw_points = [
        e
        for e in events
        if e.get("kind") == "hw_sample"
        and (e.get("temperature_c") is not None or e.get("power_draw_w") is not None)
    ]

    def _edge_reading(points, key, prefer_first):
        seq = points if prefer_first else reversed(points)
        for p in seq:
            if p.get(key) is not None:
                return p[key]
        return None

    pending_begin = None
    for e in events:
        if e.get("kind") != "cooldown":
            continue
        if e["phase"] == "B":
            pending_begin = e
            continue
        if pending_begin is None:
            continue  # stray end with no matching begin; skip

        begin_ts, end_ts = pending_begin["ts_us"], e["ts_us"]
        from_model = pending_begin.get("from_model")
        to_model = e.get("to_model")
        window = [p for p in hw_points if begin_ts <= p["ts_us"] <= end_ts]

        start_temp = _edge_reading(window, "temperature_c", prefer_first=True)
        end_temp = _edge_reading(window, "temperature_c", prefer_first=False)
        start_power = _edge_reading(window, "power_draw_w", prefer_first=True)
        end_power = _edge_reading(window, "power_draw_w", prefer_first=False)

        label = (
            f"Cooldown: {shorten_model(from_model) if from_model else '?'} "
            f"\u2192 {shorten_model(to_model) if to_model else 'end of run'}"
        )
        trace_events.append(
            {
                "name": label,
                "ph": "B",
                "pid": 0,
                "tid": 9997,
                "ts": begin_ts,
                "args": {"from_model": from_model, "to_model": to_model},
            }
        )
        trace_events.append(
            {
                "name": label,
                "ph": "E",
                "pid": 0,
                "tid": 9997,
                "ts": end_ts,
                "args": {
                    "duration_sec": round((end_ts - begin_ts) / 1e6, 2),
                    "start_temp_c": start_temp,
                    "end_temp_c": end_temp,
                    "start_power_w": start_power,
                    "end_power_w": end_power,
                    "delta_temp_c": round(end_temp - start_temp, 2)
                    if start_temp is not None and end_temp is not None
                    else None,
                    "delta_power_w": round(end_power - start_power, 2)
                    if start_power is not None and end_power is not None
                    else None,
                },
            }
        )
        pending_begin = None

    # ---------- Inference / Fix Request / Compile / Run spans ----------
    for e in events:
        if e.get("kind") != "span":
            continue
        model = e.get("model")
        tid = tid_of.get(model)
        if tid is None:
            continue  # shouldn't happen, but don't crash the export over it

        args = e.get("args", {})
        attempt = args.get("attempt")
        label = (
            f"{e['name']}"
            + (f" (attempt {attempt})" if attempt else "")
            + f": {shorten_model(model)}"
        )
        trace_events.append(
            {
                "name": label,
                "ph": e["phase"],
                "pid": 0,
                "tid": tid,
                "ts": e["ts_us"],
                "args": args,
            }
        )

    # ---------- token counters + PASS/FAIL markers, anchored to real Inference spans ----------
    infer_span: dict = {}  # model -> [begin_ts_us, end_ts_us]
    for e in events:
        if e.get("kind") == "span" and e["name"] == "Inference":
            slot = infer_span.setdefault(e["model"], [None, None])
            slot[0 if e["phase"] == "B" else 1] = e["ts_us"]

    cumulative_tokens = 0
    for r in results:
        model = r["model"]
        tid = tid_of[model]
        span = infer_span.get(model, [None, None])
        start_us = span[0] if span[0] is not None else 0.0
        end_us = span[1] if span[1] is not None else start_us

        tokens = r.get("tokens_used", 0)
        tok_per_s = r.get("tokens_per_sec", 0.0)

        trace_events.append(
            {
                "name": "tokens",
                "ph": "C",
                "pid": 0,
                "tid": 9998,
                "ts": start_us,
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
                "ts": end_us,
                "args": {
                    "cumulative_tokens": cumulative_tokens,
                    "tokens_per_sec": round(tok_per_s, 3),
                },
            }
        )

        status = r.get("status", "")
        marker_name = "\u2713 PASS" if str(status).startswith("PASS") else "\u2717 FAIL"
        trace_events.append(
            {
                "name": marker_name,
                "ph": "i",
                "s": "t",
                "pid": 0,
                "tid": tid,
                "ts": end_us,
                "args": {
                    "model": model,
                    "status": status,
                    "test_notes": r.get("test_notes", ""),
                    "compile_attempts": r.get("compile_attempts", 1),
                    "total_compile_time_sec": r.get("total_compile_time_sec", 0.0),
                },
            }
        )

    trace_events.sort(key=lambda e: e.get("ts", 0))
    return {"traceEvents": trace_events, "displayTimeUnit": "ms"}


def export_chrome_trace(results: list, output_path: str, event_log: "EventLog"):
    """Convert *results* + the shared *event_log* to a Chrome Trace file."""
    trace = convert_to_chrome_trace(results, event_log)
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
    parser.add_argument(
        "--telemetry-interval",
        metavar="SECONDS",
        type=float,
        default=5.0,
        help="GPU temperature/power sampling interval for the whole run (default: 5).",
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

    # One EventLog + one telemetry thread for the *entire* program run,
    # started before the first model and stopped after the last. This is
    # what removes the drift: temperature/power are one continuous stream
    # instead of being torn down and re-seeded for every model.
    event_log = EventLog()
    telemetry_stop_event = threading.Event()
    telemetry_thread = threading.Thread(
        target=telemetry_worker,
        args=(args.telemetry_interval, telemetry_stop_event, event_log),
        daemon=True,
        name="TelemetryWorker",
    )
    telemetry_thread.start()
    print(
        f"Hardware telemetry thread started (sampling every {args.telemetry_interval}s for the whole run).\n"
    )

    results = []
    previous_model = None  # Initialize previous model tracker
    cooldown_open = (
        False  # tracks whether a cooldown span is currently open in event_log
    )
    #    models = [model for model in models if '128k' not in model]
    total = len(models)
    print("Evaluating models: " + ", ".join(models) + "\n")
    try:
        for n, model in enumerate(models, 1):
            print(f"[{n}/{total}] Running benchmark on: {model}...")
            # Unload the previous model before loading the next one, so GPU
            # memory/power from one run doesn't bleed into the next model's
            # numbers. (Previously this call existed but lived inside the
            # trace-export function, so it never actually ran during the
            # benchmark itself unless --chrome-trace happened to be passed
            # — moved here to where it's actually needed.)
            if previous_model is not None:
                try:
                    print(f"      -> Unloading previous model: {previous_model}")
                    subprocess.run(
                        ["ollama", "delete", previous_model],
                        check=False,
                        capture_output=True,
                    )
                except Exception as e:
                    print(
                        f"      [WARN] Could not unload previous model '{previous_model}': {e}"
                    )
                # The GPU has been idle since the previous model's work
                # finished; that cooldown period ends now, as this model's
                # work is about to start.
                if cooldown_open:
                    event_log.cooldown_end(to_model=model)
                    cooldown_open = False

            res = evaluate_model(model, event_log, previous_model)
            results.append(res)

            # This model's work (inference + all compile/run attempts) is
            # done — the GPU is idle until the next model starts, so open a
            # cooldown span to trace it.
            event_log.cooldown_begin(from_model=model)
            cooldown_open = True

            previous_model = model  # Update state for the next iteration
            try:
                with open(args.output, "w", encoding="utf-8") as json_file:
                    json.dump(results, json_file, indent=4)
            except Exception as e:
                print(f"Failed to write results to JSON file: {e}")
    finally:
        # Close out any still-open cooldown (the last model's post-run idle
        # period, ending at program exit rather than at a "next model").
        if cooldown_open:
            event_log.cooldown_end(to_model=None)
        # Always stop the telemetry thread cleanly, even on error/Ctrl-C,
        # so the process can exit and the trace has a well-defined end.
        telemetry_stop_event.set()
        telemetry_thread.join(timeout=10)

    print("\n" + "=" * 100)
    print(
        f"{'MODEL':<25} | {'STATUS':<14} | {'TRIES':<5} | {'TOKENS':<8} | {'TIME (s)':<8} | {'TOK/s':<8} | {'COMPILE (s)':<11}"
    )
    print("=" * 100)
    for r in results:
        print(
            f"{r['model'][:25]:<25} | {r['status']:<14} | {r.get('compile_attempts', 1):<5} | "
            f"{r['tokens_used']:<8} | {r['time_taken_sec']:<8.2f} | {r['tokens_per_sec']:<8.2f} | "
            f"{r.get('total_compile_time_sec', 0.0):<11.3f}"
        )
    print("=" * 100)

    if args.chrome_trace:
        print()
        export_chrome_trace(results, args.chrome_trace, event_log)


if __name__ == "__main__":
    main()
