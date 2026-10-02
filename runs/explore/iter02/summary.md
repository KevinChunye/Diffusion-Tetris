### Per-arm summary

| arm                 |   episodes |   pieces |   lines |   score |   norm_score |   norm_lines |   regret_beam |   regret_rollout |   top1_beam |   illegal_rate |   prompt_tok |   cached_frac |   uncached_tok |   ttft_p50 |   ttft_p90 |   latency_p50 |   completion_tok |   usd_per_episode |   usd_per_100_pieces |   peak_prompt_tok |
|:--------------------|-----------:|---------:|--------:|--------:|-------------:|-------------:|--------------:|-----------------:|------------:|---------------:|-------------:|--------------:|---------------:|-----------:|-----------:|--------------:|-----------------:|------------------:|---------------------:|------------------:|
| oss20b/stateless    |          4 |    78.25 |   16.25 |   36.5  |       0.4535 |       0.4961 |        0.9525 |           5.0976 |      0.4121 |         0      |      1129.34 |        0.4681 |        600.671 |     0.4081 |     0.5272 |        0.8216 |          82.7668 |            0.0051 |               0.0065 |              1432 |
| oss20b/append       |          4 |    31.25 |    3.25 |    6.5  |       0.0818 |       0.1025 |        4.3685 |           6.813  |      0.272  |         0      |     10698.6  |        0.9465 |        571.928 |     0.4738 |     0.6047 |        0.9801 |          88.064  |            0.002  |               0.0065 |             23732 |
| oss20b/window8      |          4 |    39.25 |    3.25 |    6.75 |       0.0855 |       0.101  |        2.9635 |           8.7067 |      0.2484 |         0      |      5500.5  |        0.2073 |       4360.32  |     0.5389 |     0.6581 |        0.9096 |          72.3694 |            0.0128 |               0.0325 |              7410 |
| oss120b/stateless   |          4 |    88.75 |   26.25 |   56.25 |       0.7047 |       0.7953 |        0.551  |           1.3744 |      0.4394 |         0      |      1129.42 |        0.4556 |        614.896 |     0.4902 |     0.782  |        1.7759 |          87.0282 |            0.0128 |               0.0144 |              1429 |
| oss120b/append      |          4 |    46.5  |    6.5  |   15.75 |       0.1977 |       0.1944 |        3.2024 |           5.6676 |      0.328  |         0.0054 |     15240.9  |        0.9605 |        602.043 |     0.568  |     0.8081 |        2.0624 |         104.457  |            0.0071 |               0.0153 |             33294 |
| oss120b/window8     |          4 |    94.25 |   24.5  |   52.75 |       0.6534 |       0.7364 |        1.078  |           4.1009 |      0.435  |         0      |      6020.81 |        0.1287 |       5245.85  |     0.7214 |     0.9671 |        1.8594 |          86.1538 |            0.079  |               0.0839 |              8304 |
| dsv4flash/stateless |          4 |    38    |    3.75 |    7.75 |       0.0937 |       0.1064 |        3.8291 |           6.9995 |      0.3487 |         0.1382 |      1040.91 |        0.2443 |        786.592 |     2.2276 |     4.6298 |        4.3822 |          25.1645 |            0.0058 |               0.0153 |              1363 |
| dsv4flash/append    |          4 |    34.5  |    3.75 |    7.5  |       0.0925 |       0.1124 |        4.0072 |           8.4538 |      0.2754 |         0.0652 |     11366.6  |        0.8443 |       1770.3   |     2.4241 |     5.5669 |        4.2705 |          16.558  |            0.0551 |               0.1596 |             26057 |
| dsv4flash/window8   |          4 |    49    |    6.25 |   13    |       0.1719 |       0.2169 |        1.8993 |           6.686  |      0.3265 |         0.1327 |      5425.19 |        0.137  |       4682.01  |     2.3519 |     4.8497 |        3.9463 |          20.2908 |            0.0375 |               0.0765 |              8049 |

### Paired differences vs baseline (per-seed diffs; t = mean/SE)

| arm               | metric         | vs                  |   mean_diff |      se |   n_seeds |   same_sign |         t |
|:------------------|:---------------|:--------------------|------------:|--------:|----------:|------------:|----------:|
| oss20b/append     | norm_score     | oss20b/stateless    |     -0.3717 |  0.1127 |         4 |           4 |   -3.2979 |
| oss20b/append     | lines_cleared  | oss20b/stateless    |    -13      |  3.4157 |         4 |           4 |   -3.806  |
| oss20b/append     | pieces_placed  | oss20b/stateless    |    -47      |  7.7567 |         4 |           4 |   -6.0593 |
| oss20b/append     | regret_beam    | oss20b/stateless    |      3.498  |  0.4671 |         4 |           4 |    7.4891 |
| oss20b/append     | regret_rollout | oss20b/stateless    |      1.5791 |  0.6823 |         4 |           3 |    2.3144 |
| oss20b/append     | cost_usd       | oss20b/stateless    |     -0.0031 |  0.0005 |         4 |           4 |   -5.6791 |
| oss20b/append     | ttft_p50       | oss20b/stateless    |      0.0471 |  0.0253 |         4 |           4 |    1.8645 |
| oss20b/append     | cached_frac    | oss20b/stateless    |      0.4721 |  0.0126 |         4 |           4 |   37.5365 |
| oss20b/window8    | norm_score     | oss20b/stateless    |     -0.368  |  0.1203 |         4 |           4 |   -3.0589 |
| oss20b/window8    | lines_cleared  | oss20b/stateless    |    -13      |  3.8944 |         4 |           4 |   -3.3381 |
| oss20b/window8    | pieces_placed  | oss20b/stateless    |    -39      |  9.6695 |         4 |           4 |   -4.0333 |
| oss20b/window8    | regret_beam    | oss20b/stateless    |      2.1651 |  0.8055 |         4 |           4 |    2.6877 |
| oss20b/window8    | regret_rollout | oss20b/stateless    |      3.8853 |  0.4664 |         4 |           4 |    8.3309 |
| oss20b/window8    | cost_usd       | oss20b/stateless    |      0.0077 |  0.0015 |         4 |           4 |    5.1456 |
| oss20b/window8    | ttft_p50       | oss20b/stateless    |      0.1302 |  0.0113 |         4 |           4 |   11.5261 |
| oss20b/window8    | cached_frac    | oss20b/stateless    |     -0.2575 |  0.021  |         4 |           4 |  -12.2517 |
| oss120b/append    | norm_score     | oss120b/stateless   |     -0.507  |  0.1706 |         4 |           4 |   -2.9724 |
| oss120b/append    | lines_cleared  | oss120b/stateless   |    -19.75   |  6.4984 |         4 |           4 |   -3.0392 |
| oss120b/append    | pieces_placed  | oss120b/stateless   |    -42.25   | 13.0855 |         4 |           4 |   -3.2288 |
| oss120b/append    | regret_beam    | oss120b/stateless   |      2.6178 |  0.134  |         4 |           4 |   19.54   |
| oss120b/append    | regret_rollout | oss120b/stateless   |      3.7641 |  1.4628 |         4 |           3 |    2.5733 |
| oss120b/append    | cost_usd       | oss120b/stateless   |     -0.0057 |  0.0019 |         4 |           4 |   -3.0648 |
| oss120b/append    | ttft_p50       | oss120b/stateless   |      0.0709 |  0.0133 |         4 |           4 |    5.3318 |
| oss120b/append    | cached_frac    | oss120b/stateless   |      0.502  |  0.0036 |         4 |           4 |  137.895  |
| oss120b/window8   | norm_score     | oss120b/stateless   |     -0.0513 |  0.1361 |         4 |           2 |   -0.3773 |
| oss120b/window8   | lines_cleared  | oss120b/stateless   |     -1.75   |  4.3851 |         4 |           3 |   -0.3991 |
| oss120b/window8   | pieces_placed  | oss120b/stateless   |      5.5    |  6.8981 |         4 |           1 |    0.7973 |
| oss120b/window8   | regret_beam    | oss120b/stateless   |      0.525  |  0.2734 |         4 |           3 |    1.9206 |
| oss120b/window8   | regret_rollout | oss120b/stateless   |      2.2933 |  1.8275 |         4 |           3 |    1.2549 |
| oss120b/window8   | cost_usd       | oss120b/stateless   |      0.0662 |  0.0024 |         4 |           4 |   27.2996 |
| oss120b/window8   | ttft_p50       | oss120b/stateless   |      0.2362 |  0.0088 |         4 |           4 |   26.7421 |
| oss120b/window8   | cached_frac    | oss120b/stateless   |     -0.3274 |  0.0022 |         4 |           4 | -148.374  |
| dsv4flash/append  | norm_score     | dsv4flash/stateless |     -0.0012 |  0.038  |         4 |           2 |   -0.0325 |
| dsv4flash/append  | lines_cleared  | dsv4flash/stateless |      0      |  1.472  |         4 |           2 |    0      |
| dsv4flash/append  | pieces_placed  | dsv4flash/stateless |     -3.5    |  2.3274 |         4 |           3 |   -1.5038 |
| dsv4flash/append  | regret_beam    | dsv4flash/stateless |      0.1034 |  0.1481 |         4 |           2 |    0.6981 |
| dsv4flash/append  | regret_rollout | dsv4flash/stateless |      1.2124 |  2.3062 |         4 |           3 |    0.5257 |
| dsv4flash/append  | cost_usd       | dsv4flash/stateless |      0.0493 |  0.0073 |         4 |           4 |    6.7359 |
| dsv4flash/append  | ttft_p50       | dsv4flash/stateless |      0.3995 |  0.1256 |         4 |           4 |    3.1807 |
| dsv4flash/append  | cached_frac    | dsv4flash/stateless |      0.6001 |  0.0057 |         4 |           4 |  106.208  |
| dsv4flash/window8 | norm_score     | dsv4flash/stateless |      0.0781 |  0.1186 |         4 |           2 |    0.6586 |
| dsv4flash/window8 | lines_cleared  | dsv4flash/stateless |      2.5    |  4.0927 |         4 |           2 |    0.6108 |
| dsv4flash/window8 | pieces_placed  | dsv4flash/stateless |     11      | 11.4673 |         4 |           2 |    0.9592 |
| dsv4flash/window8 | regret_beam    | dsv4flash/stateless |     -1.9216 |  0.6208 |         4 |           4 |   -3.0954 |
| dsv4flash/window8 | regret_rollout | dsv4flash/stateless |     -0.3133 |  1.7461 |         4 |           3 |   -0.1794 |
| dsv4flash/window8 | cost_usd       | dsv4flash/stateless |      0.0317 |  0.0086 |         4 |           4 |    3.7063 |
| dsv4flash/window8 | ttft_p50       | dsv4flash/stateless |      0.5587 |  0.447  |         4 |           3 |    1.2498 |
| dsv4flash/window8 | cached_frac    | dsv4flash/stateless |     -0.0952 |  0.0282 |         4 |           4 |   -3.3732 |
