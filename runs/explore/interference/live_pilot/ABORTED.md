# Attempt 1 — aborted (LIVE, not used for hypothesis tests)

Config `configs/interference/pilot.yaml`. Interrupted at 00:58 UTC after one full replay
(gpt-oss-20b, block 0, `defer_long`) and part of a second (`fifo`). Reason: most gpt-oss-20b short
decisions on the raw-track corpus spent all 512 completion tokens reasoning and returned no content
(`finish_reason=length`), so TTVA was undefined for most short requests in every condition and the
arm could not test H1/H2. Replaced by gemma-4-31B-it in `configs/interference/pilot_v2.yaml`.
Spend: $0.0126 recorded at interrupt + $0.0155 correction (rows of the interrupted replay and a
conservative reservation for up to 15 requests that may have been in flight) = $0.0282 estimated.
The rows are kept as recorded; `dispatch.jsonl` did not exist yet in this version of the runner.
