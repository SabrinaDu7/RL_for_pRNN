
## Outcomes (2026-10-02 02:10)

- **Resumes** (11033972–11033974): a resume runs with the CUDA graphs and the layer
  compile off and trains at ~15k env steps/s against a fresh run's ~100k, so the three hit
  `resume_clamp.sh`'s 90-minute limit at ~96% of the second budget (TIMEOUT); the exit
  trap saved 9 of the 10 archived checkpoint pairs each (92.3M to 159.4M frames). Fetched
  to `outputs/fetched/mx-impassable-n8-s{2,3,4}-resume-dot_curious_26-10-01-23-45-2*`.
  Q19: the dot's squares end at 1.76× their dot-free visits (seeds 1.66–1.92; first
  encounter 1.06), the late distance to the dot lower with it than without in every
  seed; the agent's near-dot time itself stays at uniform while its dot-free visits to
  the same squares halve — the dot retains the agent rather than draws it.
- **Scattered, 16 placements** (11034007, 11034008): COMPLETED in 25:47 / 25:xx. **Scattered,
  every cell** (11034398, 11034399; 477 layouts): COMPLETED in ~25 min each, the spatial
  evaluation scoring `rooms_max` rooms. Q19_exp3: no pull toward a dot at the held-out
  spot (ratios 1.28 / 1.08 for the all-cell runs, distance differences −0.05 / +0.14).
