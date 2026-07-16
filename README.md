# ollama-test-harness

`ollama-test.py` is a benchmark harness for local Ollama models. It runs each installed model against a task definition, captures the model output, retries failed generations with error feedback, and writes benchmark results plus optional Chrome/Perfetto trace data.

## What it does

For each model returned by `GET /api/tags` on your Ollama server, the script:

1. Sends the task prompt to `POST /api/chat`.
2. Extracts the first fenced code block from the response.
3. Optionally compiles and runs the generated code.
4. Validates the program output against declarative rules from the task JSON.
5. Retries up to 6 times if compilation or validation fails, feeding the previous error back to the model.
6. Records runtime metrics, token usage, compile attempts, and hardware telemetry.

The harness also polls GPU temperature and power from `nvidia-smi-webserver` on the same host as Ollama.

## Requirements

- Ollama running locally and reachable through `OLLAMA_HOST` or `http://localhost:11434`
- Local models installed in Ollama
- `nvidia-smi-webserver` from https://github.com/thematthew/nvidia-smi-webserver if you want to log GPU thermals and power
- Python dependencies from this repo

## Usage

Run the harness with the default task:

```bash
python3 ollama-test.py
```

Useful options:

```bash
python3 ollama-test.py --task tasks/rule90/rule90_c.json
python3 ollama-test.py --output results.json
python3 ollama-test.py --chrome-trace
python3 ollama-test.py --telemetry-interval 3
```

Environment:

- `OLLAMA_HOST` sets the Ollama base URL.
- The GPU telemetry endpoint is derived from `OLLAMA_HOST` and uses port `28123`.

## Task configs

`ollama-test.py` is task-driven. A task config defines:

- the prompt shown to the model
- the target language
- the source file extension to write
- optional build and run commands
- validation rules for program output

Tasks can either embed the prompt directly or point at a separate prompt file. This keeps the harness generic: changing the benchmark usually means adding or editing a task JSON, not modifying the script.

## Output

The script writes benchmark data to:

- `ollama_benchmark_results.json` by default
- `ollama_benchmark_trace.json` when `--chrome-trace` is enabled
- `./output/YYYY-MM-DD/<model>/chat_history.json` for each model run

The results JSON includes per-model status, token counts, compile attempts, execution time, and captured output. The Chrome trace can be opened in `chrome://tracing`, Eclipse Trace Compass (with incubator) or Perfetto.

## Notes

- Models are benchmarked in mostly paramater size order.
- The script unloads the previous model between runs with `ollama delete <model>`.
- If you run multiple task configs in one invocation, output files are suffixed with the task name to avoid overwriting.
