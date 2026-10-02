### Cache hit after 30 s idle vs concurrent contexts

| model                         |   contexts |   trials |   hit_rate |   probe_p50 |   warm_p50 |   warm_max |
|:------------------------------|-----------:|---------:|-----------:|------------:|-----------:|-----------:|
| deepseek-ai/DeepSeek-V4-Flash |          1 |        3 |          1 |       0.428 |      3.861 |      3.932 |
| deepseek-ai/DeepSeek-V4-Flash |          4 |        4 |          0 |       4.302 |      8.564 |     11.806 |
| deepseek-ai/DeepSeek-V4-Flash |          8 |        8 |          0 |       3.593 |     14.29  |     21.88  |
| deepseek-ai/DeepSeek-V4-Flash |         16 |       16 |          0 |      16.955 |     23.485 |     39.843 |
| openai/gpt-oss-20b            |          1 |        1 |          1 |       0.161 |      1.114 |      1.114 |
| openai/gpt-oss-20b            |          4 |        4 |          1 |       0.176 |      1.358 |      1.687 |
| openai/gpt-oss-20b            |          8 |        8 |          1 |       0.174 |      1.943 |      2.595 |
| openai/gpt-oss-20b            |         16 |       16 |          1 |       0.157 |      2.99  |      4.529 |

### Other phases (isolated control, keep-alive, GLM sequential, catalog map)

| phase     | model                         | tag            |   gap_s |   trials |   hit_rate |   reported_hit_rate |   cold_p50 |   probe_p50 |   pings |   cost |
|:----------|:------------------------------|:---------------|--------:|---------:|-----------:|--------------------:|-----------:|------------:|--------:|-------:|
| glm       | lukealonso/GLM-5.2-NVFP4      | sequential     |       5 |        2 |      1     |               0     |      9.099 |       1.021 |       0 |  0.08  |
| glm       | lukealonso/GLM-5.2-NVFP4      | sequential     |      30 |        2 |      1     |               0     |      8.872 |       1.007 |       0 |  0.08  |
| isolated  | deepseek-ai/DeepSeek-V4-Flash | isolated       |      10 |        2 |      1     |               1     |      3.236 |       0.365 |       0 |  0.008 |
| isolated  | deepseek-ai/DeepSeek-V4-Flash | isolated       |      30 |        2 |      1     |               1     |      3.536 |       0.492 |       0 |  0.008 |
| keepalive | deepseek-ai/DeepSeek-V4-Flash | keepalive4s    |      60 |        8 |      0     |               0     |     14.413 |      23.581 |       2 |  0.063 |
| keepalive | deepseek-ai/DeepSeek-V4-Flash | silent         |      60 |        8 |      0     |               0     |     32.051 |      26.9   |       0 |  0.032 |
| map       | MiniMaxAI/MiniMax-M2.5        | map12          |       5 |        3 |      1     |               1     |      2.218 |       1.16  |       0 |  0.013 |
| map       | MiniMaxAI/MiniMax-M2.5        | map12          |      30 |        3 |      1     |               1     |      2.057 |       0.268 |       0 |  0.013 |
| map       | MiniMaxAI/MiniMax-M2.5        | map12          |     120 |        3 |      1     |               1     |      2.059 |       0.243 |       0 |  0.013 |
| map       | MiniMaxAI/MiniMax-M2.5        | map12          |     600 |        3 |      1     |               1     |      2.172 |       0.269 |       0 |  0.013 |
| map       | Qwen/Qwen3.5-397B-A17B-FP8    | map12          |       5 |        3 |      1     |               1     |      1.873 |       0.763 |       0 |  0.054 |
| map       | Qwen/Qwen3.5-397B-A17B-FP8    | map12          |      30 |        3 |      1     |               1     |      1.944 |       0.368 |       0 |  0.054 |
| map       | Qwen/Qwen3.5-397B-A17B-FP8    | map12          |     120 |        3 |      1     |               1     |      1.811 |       0.339 |       0 |  0.054 |
| map       | Qwen/Qwen3.5-397B-A17B-FP8    | map12          |     600 |        3 |      1     |               1     |      1.814 |       0.353 |       0 |  0.054 |
| map       | Qwen/Qwen3.8-27B-FP8          | map12          |       5 |        3 |      1     |               1     |      2.912 |       1.493 |       0 |  0.029 |
| map       | Qwen/Qwen3.8-27B-FP8          | map12          |      30 |        3 |      1     |               1     |      3.97  |       0.378 |       0 |  0.029 |
| map       | Qwen/Qwen3.8-27B-FP8          | map12          |     120 |        3 |      1     |               1     |      4.39  |       0.388 |       0 |  0.029 |
| map       | Qwen/Qwen3.8-27B-FP8          | map12          |     600 |        3 |      1     |               1     |      4.814 |       0.412 |       0 |  0.029 |
| map       | google/gemma-4-31B-it         | map12          |       5 |        3 |      1     |               1     |      7.732 |       8.114 |       0 |  0.007 |
| map       | google/gemma-4-31B-it         | map12          |      30 |        3 |      0.667 |               0.667 |      9.016 |       0.368 |       0 |  0.009 |
| map       | google/gemma-4-31B-it         | map12          |     120 |        3 |      0.667 |               0.667 |     10.256 |       0.354 |       0 |  0.009 |
| map       | google/gemma-4-31B-it         | map12          |     600 |        3 |      0.333 |               0.333 |     10.199 |       1.993 |       0 |  0.01  |
| map       | openai/gpt-oss-20b            | rewarm_control |     600 |        3 |      1     |               1     |      0.996 |       0.181 |       0 |  0.003 |
