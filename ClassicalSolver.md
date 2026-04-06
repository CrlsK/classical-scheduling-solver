# Classical Solver: Priority Dispatch + ALNS

## Algorithm
- Phase 1: Priority Dispatch initial solution (EDD + urgency sorting)
- Phase 2: Adaptive Large Neighborhood Search (500 iterations)
  - Destroy: random_removal, worst_removal, related_removal
  - Repair: greedy_insertion, regret_2_insertion
  - Simulated annealing acceptance
- Phase 3: Output formatting with QCentroid benchmark contract

## Performance
- Hisar dataset (15 jobs, 8 machines, 72h): ~31h makespan, 100% on-time, <0.1s
- Jajpur dataset (25 jobs, 12 machines, 120h): ~71h makespan, 100% on-time, <0.2s

## Dependencies
Pure Python (stdlib only). No external packages required.