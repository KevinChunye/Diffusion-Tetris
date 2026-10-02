### Per-arm summary

| arm       |   episodes |   pieces |   lines |   score |   norm_score |   norm_lines |   regret_beam |   regret_rollout |   top1_beam |   illegal_rate |   prompt_tok |   cached_frac |   uncached_tok |   ttft_p50 |   ttft_p90 |   latency_p50 |   completion_tok |   usd_per_episode |   usd_per_100_pieces |   peak_prompt_tok |
|:----------|-----------:|---------:|--------:|--------:|-------------:|-------------:|--------------:|-----------------:|------------:|---------------:|-------------:|--------------:|---------------:|-----------:|-----------:|--------------:|-----------------:|------------------:|---------------------:|------------------:|
| stateless |          3 |  71.6667 | 15.6667 | 35.6667 |       0.422  |       0.4209 |        1.0559 |           3.804  |      0.3907 |         0      |      1127.45 |        0.489  |        576.154 |     0.4488 |     0.7135 |        1.0521 |          85.0698 |            0.0046 |               0.0064 |              1431 |
| append    |          3 |  32      |  2      |  4      |       0.0476 |       0.0551 |        4.1701 |           7.4231 |      0.2083 |         0.0104 |     11106.6  |        0.9514 |        539.646 |     0.5502 |     0.867  |        1.3617 |          82.8438 |            0.002  |               0.0061 |             21853 |
| window8   |          3 |  45      |  5      | 10.3333 |       0.1231 |       0.1373 |        3.0882 |           4.6056 |      0.2667 |         0      |      5659.06 |        0.1925 |       4569.87  |     0.6161 |     0.7918 |        1.2836 |          76.763  |            0.0154 |               0.0341 |              7592 |
| compact16 |          3 |  44      |  5.3333 | 11      |       0.13   |       0.1414 |        2.4296 |           8.0177 |      0.3712 |         0.0227 |      5697.71 |        0.8986 |        577.712 |     0.5089 |     0.7608 |        1.2637 |          92.1591 |            0.0029 |               0.0066 |             11457 |

### Paired differences vs baseline (per-seed diffs; t = mean/SE)

| arm       | metric         | vs        |   mean_diff |      se |   n_seeds |   same_sign |        t |
|:----------|:---------------|:----------|------------:|--------:|----------:|------------:|---------:|
| append    | norm_score     | stateless |     -0.3744 |  0.1955 |         3 |           3 |  -1.915  |
| append    | lines_cleared  | stateless |    -13.6667 |  6.6916 |         3 |           3 |  -2.0424 |
| append    | pieces_placed  | stateless |    -39.6667 | 15.2352 |         3 |           3 |  -2.6036 |
| append    | regret_beam    | stateless |      2.9809 |  0.602  |         3 |           3 |   4.9516 |
| append    | regret_rollout | stateless |      2.9091 |  0.9021 |         3 |           3 |   3.2247 |
| append    | cost_usd       | stateless |     -0.0026 |  0.0012 |         3 |           3 |  -2.26   |
| append    | ttft_p50       | stateless |      0.0997 |  0.0024 |         3 |           3 |  42.3816 |
| append    | cached_frac    | stateless |      0.4553 |  0.0255 |         3 |           3 |  17.8771 |
| compact16 | norm_score     | stateless |     -0.2921 |  0.192  |         3 |           3 |  -1.5213 |
| compact16 | lines_cleared  | stateless |    -10.3333 |  6.4377 |         3 |           3 |  -1.6051 |
| compact16 | pieces_placed  | stateless |    -27.6667 | 13.8604 |         3 |           3 |  -1.9961 |
| compact16 | regret_beam    | stateless |      1.4993 |  0.6973 |         3 |           3 |   2.1502 |
| compact16 | regret_rollout | stateless |      3.849  |  6.0513 |         3 |           2 |   0.6361 |
| compact16 | cost_usd       | stateless |     -0.0017 |  0.0011 |         3 |           3 |  -1.5846 |
| compact16 | ttft_p50       | stateless |      0.0565 |  0.0232 |         3 |           3 |   2.4362 |
| compact16 | cached_frac    | stateless |      0.403  |  0.023  |         3 |           3 |  17.5126 |
| window8   | norm_score     | stateless |     -0.2989 |  0.1983 |         3 |           3 |  -1.5075 |
| window8   | lines_cleared  | stateless |    -10.6667 |  6.6916 |         3 |           3 |  -1.594  |
| window8   | pieces_placed  | stateless |    -26.6667 | 14.5182 |         3 |           3 |  -1.8368 |
| window8   | regret_beam    | stateless |      1.9101 |  0.5657 |         3 |           3 |   3.3766 |
| window8   | regret_rollout | stateless |      0.2447 |  1.3902 |         3 |           2 |   0.176  |
| window8   | cost_usd       | stateless |      0.0108 |  0.0011 |         3 |           3 |   9.7272 |
| window8   | ttft_p50       | stateless |      0.1564 |  0.0197 |         3 |           3 |   7.9233 |
| window8   | cached_frac    | stateless |     -0.3036 |  0.0267 |         3 |           3 | -11.3772 |
