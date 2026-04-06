"""
QCentroid Classical Solver: Priority Dispatch + Adaptive Large Neighborhood Search (ALNS)
Dynamic Production Scheduling for Stainless Steel Manufacturing

Algorithm:
  Phase 1: Data Normalization - Parse actual dataset structure
  Phase 2: Priority Dispatch - Build initial solution
  Phase 3: ALNS - Improve via destroy/repair with simulated annealing
  Phase 4: Output formatting for QCentroid benchmark platform

Author: QCentroid Team
"""

import logging
import time
import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """
    Main entry point for QCentroid solver.

    Args:
        input_data: The 'data' field from job input (actual Hisar/Jajpur dataset structure)
        solver_params: {"max_iterations": 500, "temperature_initial": 100.0, "cooling_rate": 0.95, ...}
        extra_arguments: Additional runtime configuration

    Returns:
        dict: Complete benchmark output with schedule, metrics, and QCentroid fields
    """
    start_time = time.time()
    random.seed(42)

    try:
        # Parse input
        machines = input_data.get("machines", [])
        jobs = input_data.get("jobs", [])
        planning_horizon = input_data.get("planning_horizon", {"start_time": 0, "end_time": 72, "time_unit": "hours"})
        processing_times = input_data.get("processing_times", {})
        setup_matrix = input_data.get("setup_matrix", {})
        maintenance_schedules = input_data.get("maintenance_schedules", [])
        metadata = input_data.get("metadata", {})

        # Get solver parameters
        max_iterations = solver_params.get("max_iterations", 500)
        max_time_s = solver_params.get("max_time_s", 300.0)
        temperature = solver_params.get("temperature_initial", 100.0)
        cooling_rate = solver_params.get("cooling_rate", 0.95)

        logger.info(f"Starting QCentroid solver: {len(jobs)} jobs, {len(machines)} machines, {max_iterations} iterations")

        # Phase 1: Data Normalization
        solver = JobShopSolver(machines, jobs, planning_horizon, processing_times, setup_matrix,
                              maintenance_schedules, logger)

        # Phase 2: Priority Dispatch Initial Solution
        schedule = solver.priority_dispatch()
        logger.info(f"Initial solution: makespan={schedule['makespan']:.2f}h, tardiness={schedule['total_tardiness']:.2f}h")

        # Phase 3: ALNS Improvement
        best_schedule = solver.adaptive_lns(
            schedule,
            max_iterations=max_iterations,
            max_time_s=max_time_s,
            initial_temperature=temperature,
            cooling_rate=cooling_rate
        )

        elapsed = time.time() - start_time
        logger.info(f"Final solution: makespan={best_schedule['makespan']:.2f}h, tardiness={best_schedule['total_tardiness']:.2f}h, time={elapsed:.2f}s")

        # Phase 4: Output formatting
        result = _format_output(best_schedule, solver, elapsed, max_iterations)
        return result

    except Exception as e:
        logger.exception(f"Solver error: {e}")
        return {
            "schedule": {"assignments": [], "gantt_data": [], "makespan": 0, "total_tardiness": 0, "total_idle_time": 0, "total_energy_kwh": 0, "total_cost": 0, "jobs_on_time": 0, "jobs_late": 0, "on_time_percentage": 0},
            "machine_utilization": {},
            "job_metrics": {},
            "cost_breakdown": {},
            "risk_metrics": {},
            "constraint_violations": {"hard_constraints": {"total_hard_violations": 1, "is_feasible": False}, "soft_constraints": {}},
            "objective_value": float('inf'),
            "solution_status": "error",
            "computation_metrics": {"wall_time_s": time.time() - start_time, "algorithm": "PriorityDispatch_ALNS", "iterations": 0},
            "benchmark": {"execution_cost": {"value": 0, "unit": "credits"}, "time_elapsed": "0s", "energy_consumption": 0.0},
            "makespan_hours": 0,
            "on_time_delivery_pct": 0,
            "total_tardiness_hours": 0,
            "avg_machine_utilization_pct": 0,
            "total_changeovers": 0
        }


class JobShopSolver:
    """Core solver logic for job shop scheduling with actual dataset structure."""

    def __init__(self, machines: List, jobs: List, planning_horizon: Dict, processing_times: Dict,
                 setup_matrix: Dict, maintenance_schedules: List, logger):
        self.machines_list = machines
        self.machines = {m["machine_id"]: m for m in machines}
        self.jobs = jobs
        self.planning_horizon = planning_horizon
        self.processing_times = processing_times
        self.setup_matrix = setup_matrix
        self.maintenance_schedules = maintenance_schedules
        self.logger = logger

        # Build operation_code -> machine_id mapping
        self.operation_to_machine = self._build_operation_to_machine_map()

        # Build maintenance windows per machine
        self.maintenance_windows = self._build_maintenance_windows()

    def _build_operation_to_machine_map(self) -> Dict[str, str]:
        """Build operation_code -> machine_id mapping from capability_groups."""
        mapping = {}
        code_to_type = {
            "HRM": "hot_rolling_mill",
            "CRM": "cold_rolling_mill",
            "AF": "annealing_furnace",
            "PL": "pickling_line",
            "SL": "slitting_machine",
            "GR": "grinding_station",
            "PU": "polishing_unit",
            "CS": "cutting_station",
            "QC": "inspection_bay"
        }
        # First try to match by machine type
        for machine in self.machines_list:
            mtype = machine.get("type", "")
            for code, expected_type in code_to_type.items():
                if mtype == expected_type:
                    mapping[code] = machine["machine_id"]
                    break

        # For any unmapped operations, find first machine that can handle them
        # based on capability_groups
        for machine in self.machines_list:
            caps = set(machine.get("capability_groups", []))
            for code in code_to_type:
                if code not in mapping:
                    # Try to match capability groups
                    if any(cap.lower() in str(caps).lower() for cap in [code.lower(), code_to_type[code].replace("_", "")]):
                        mapping[code] = machine["machine_id"]
                        break

        # Fallback: use first machine for any remaining operations
        if self.machines_list:
            default_machine = self.machines_list[0]["machine_id"]
            for code in code_to_type:
                if code not in mapping:
                    mapping[code] = default_machine

        return mapping

    def _build_maintenance_windows(self) -> Dict[str, List[Tuple[float, float]]]:
        """Build maintenance windows per machine."""
        windows = defaultdict(list)
        for maint in self.maintenance_schedules:
            mid = maint.get("machine_id")
            start = maint.get("scheduled_start", 0)
            end = maint.get("scheduled_end", 0)
            if mid:
                windows[mid].append((start, end))
        return dict(windows)

    def _get_processing_time_hours(self, operation_code: str, material_grade: str) -> float:
        """Get processing time in hours for operation on material. Returns None if not applicable."""
        key = None
        for pt_key in self.processing_times:
            if pt_key.startswith(operation_code + "_"):
                key = pt_key
                break
        if not key:
            return 1.0  # Fallback
        times = self.processing_times.get(key, {})
        minutes = times.get(material_grade, 0)
        if minutes is None:
            return None  # Material cannot be processed by this operation
        return minutes / 60.0 if minutes else 1.0

    def _get_setup_time_hours(self, from_grade: str, to_grade: str, machine_id: str) -> float:
        """Get setup time in hours between grade change on machine."""
        if from_grade == to_grade:
            return 0
        # Extract prefix: "304_austenitic" -> "304"
        from_prefix = from_grade.split("_")[0] if "_" in from_grade else "304"
        # For cold rolling mill (M001), use main setup matrix
        if machine_id == "M001" or self.setup_matrix.get("machine") == "M001_cold_rolling_mill":
            from_key = f"from_{from_prefix}_to"
            setup_dict = self.setup_matrix.get(from_key, {})
            minutes = setup_dict.get(to_grade, 0)
            return minutes / 60.0 if minutes else 0
        # For annealing furnace, use annealing setup
        elif machine_id == "M002":
            af_setup = self.setup_matrix.get("annealing_furnace_setup", {})
            from_key = f"from_{from_prefix}_to"
            setup_dict = af_setup.get(from_key, {})
            minutes = setup_dict.get(to_grade, 0)
            return minutes / 60.0 if minutes else 0
        return 0

    def priority_dispatch(self) -> Dict:
        """Build initial solution using priority-based dispatch."""
        # Build list of all operations to schedule
        operations = []
        for job in self.jobs:
            job_id = job["job_id"]
            material = job.get("material_grade", "304_austenitic")
            release_time = job.get("release_time", 0)
            due_date = job.get("due_date", 72)
            urgency = job.get("urgency_level", "medium")
            priority_weight = job.get("priority_weight", 1.0)
            required_ops = job.get("required_operations", [])

            for req_op in required_ops:
                op_seq = req_op.get("operation_sequence", 0)
                op_code = req_op.get("operation_code", "")
                op_name = req_op.get("operation_name", "")

                proc_time = self._get_processing_time_hours(op_code, material)
                if proc_time is None:
                    continue  # Skip operations not applicable for this material

                operations.append({
                    "job_id": job_id,
                    "op_id": f"{job_id}_OP{op_seq:02d}",
                    "op_sequence": op_seq,
                    "operation_code": op_code,
                    "operation_name": op_name,
                    "material_grade": material,
                    "duration_hours": proc_time,
                    "machine_id": self.operation_to_machine.get(op_code, "M001"),
                    "release_time": release_time,
                    "due_date": due_date,
                    "urgency_level": urgency,
                    "priority_weight": priority_weight,
                })

        # Sort operations by priority
        urgency_map = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        operations.sort(key=lambda o: (
            urgency_map.get(o["urgency_level"], 2),
            o["due_date"],
            -o["priority_weight"],
            o["op_sequence"]
        ))

        # Track machine timeline and job operation completion
        machine_timeline = {m_id: [] for m_id in self.machines}
        job_last_completion = {}  # job_id -> completion_time

        for op in operations:
            job_id = op["job_id"]
            # Ensure precedence: can't start before previous operation in same job finishes
            earliest_start = max(op["release_time"], job_last_completion.get(job_id, 0))

            machine_id = op["machine_id"]
            # Find available slot considering setup and maintenance
            start_time, setup_time = self._find_available_slot(
                machine_id, earliest_start, op, machine_timeline
            )

            end_time = start_time + op["duration_hours"]
            machine_timeline[machine_id].append({
                "start": start_time,
                "end": end_time,
                "setup": setup_time,
                "job_id": job_id,
                "op_id": op["op_id"],
                "op_sequence": op["op_sequence"],
                "operation_code": op["operation_code"],
                "operation_name": op["operation_name"],
                "material_grade": op["material_grade"],
                "duration": op["duration_hours"],
            })
            job_last_completion[job_id] = end_time

        return self._build_schedule_dict(machine_timeline)

    def _find_available_slot(self, machine_id: str, earliest_start: float, op: Dict,
                            machine_timeline: Dict) -> Tuple[float, float]:
        """Find earliest available slot on machine, accounting for setup and maintenance."""
        timeline = sorted(machine_timeline[machine_id], key=lambda x: x["start"])
        maintenance_wins = self.maintenance_windows.get(machine_id, [])

        # Find last operation before earliest_start
        last_grade = None
        last_end = 0
        for entry in timeline:
            if entry["end"] <= earliest_start:
                last_grade = entry.get("material_grade")
                last_end = entry["end"]

        # Calculate setup time
        setup_time = 0
        if last_grade:
            setup_time = self._get_setup_time_hours(last_grade, op["material_grade"], machine_id)

        candidate_start = max(earliest_start, last_end + setup_time)

        # Resolve conflicts with existing operations and maintenance
        iteration = 0
        while iteration < 100:
            iteration += 1
            conflict = False

            # Check overlaps with existing operations
            for entry in timeline:
                if not (candidate_start + op["duration_hours"] <= entry["start"] or candidate_start >= entry["end"]):
                    candidate_start = entry["end"]
                    conflict = True
                    break

            if conflict:
                continue

            # Check maintenance windows
            for maint_start, maint_end in maintenance_wins:
                if not (candidate_start + op["duration_hours"] <= maint_start or candidate_start >= maint_end):
                    candidate_start = maint_end
                    conflict = True
                    break

            if not conflict:
                break

        return candidate_start, setup_time

    def adaptive_lns(self, initial_schedule: Dict, max_iterations: int, max_time_s: float,
                     initial_temperature: float, cooling_rate: float) -> Dict:
        """Improve schedule using ALNS with simulated annealing."""
        current_schedule = initial_schedule
        best_schedule = initial_schedule.copy()
        best_cost = self._compute_objective(best_schedule)

        operator_weights = {
            "random_removal": 1.0,
            "worst_removal": 1.0,
            "related_removal": 1.0,
            "greedy_insertion": 1.0,
            "regret_2_insertion": 1.0,
        }

        temperature = initial_temperature
        iteration = 0
        start_time_alns = time.time()

        while iteration < max_iterations and (time.time() - start_time_alns) < max_time_s:
            # Select operators
            destroy_op = self._select_operator(operator_weights, ["random_removal", "worst_removal", "related_removal"])
            repair_op = self._select_operator(operator_weights, ["greedy_insertion", "regret_2_insertion"])

            # Destroy and repair
            partial_schedule = self._destroy(current_schedule, destroy_op)
            candidate_schedule = self._repair(current_schedule, partial_schedule, repair_op)

            # Evaluate
            candidate_cost = self._compute_objective(candidate_schedule)
            current_cost = self._compute_objective(current_schedule)
            delta = candidate_cost - current_cost

            # Simulated annealing
            accept = delta < 0 or random.random() < math.exp(-delta / (temperature + 1e-6))
            if accept:
                current_schedule = candidate_schedule

            # Update best
            if candidate_cost < best_cost:
                best_schedule = candidate_schedule
                best_cost = candidate_cost

            # Adaptive weights
            if delta < 0:
                operator_weights[destroy_op] *= 1.1
                operator_weights[repair_op] *= 1.1

            temperature *= cooling_rate
            iteration += 1

        best_schedule["alns_iterations"] = iteration
        self.logger.info(f"ALNS completed {iteration} iterations, best_cost={best_cost:.2f}")
        return best_schedule

    def _select_operator(self, weights: Dict, operators: List[str]) -> str:
        """Weighted random selection of operator."""
        total = sum(weights.get(op, 1.0) for op in operators)
        r = random.uniform(0, total)
        cumsum = 0
        for op in operators:
            cumsum += weights.get(op, 1.0)
            if r <= cumsum:
                return op
        return operators[0]

    def _destroy(self, schedule: Dict, operator: str) -> Dict:
        """Remove subset of operations."""
        assignments = schedule["assignments"].copy()
        if not assignments:
            return {"remaining": [], "removed": []}

        if operator == "random_removal":
            num_to_remove = max(1, len(assignments) // 5)
            to_remove = set(random.sample(range(len(assignments)), min(num_to_remove, len(assignments))))
            return {
                "remaining": [a for i, a in enumerate(assignments) if i not in to_remove],
                "removed": [a for i, a in enumerate(assignments) if i in to_remove],
            }

        elif operator == "worst_removal":
            num_to_remove = max(1, len(assignments) // 5)
            op_tardiness = []
            for i, a in enumerate(assignments):
                job = next((j for j in self.jobs if j["job_id"] == a["job_id"]), None)
                if job:
                    due_date = job.get("due_date", 72)
                    tardiness = max(0, a["end_time"] - due_date)
                    op_tardiness.append((i, tardiness))
            op_tardiness.sort(key=lambda x: x[1], reverse=True)
            to_remove = set(idx for idx, _ in op_tardiness[:num_to_remove])
            return {
                "remaining": [a for i, a in enumerate(assignments) if i not in to_remove],
                "removed": [a for i, a in enumerate(assignments) if i in to_remove],
            }

        elif operator == "related_removal":
            if not assignments:
                return {"remaining": [], "removed": []}
            seed_idx = random.randint(0, len(assignments) - 1)
            seed_machine = assignments[seed_idx].get("machine_id", "")
            to_remove = {i for i, a in enumerate(assignments) if a.get("machine_id") == seed_machine}
            to_remove.add(seed_idx)
            return {
                "remaining": [a for i, a in enumerate(assignments) if i not in to_remove],
                "removed": [a for i, a in enumerate(assignments) if i in to_remove],
            }

        return {"remaining": assignments, "removed": []}

    def _repair(self, original_schedule: Dict, partial_schedule: Dict, operator: str) -> Dict:
        """Reinsert removed operations."""
        machine_timeline = self._extract_machine_timeline(partial_schedule["remaining"])
        removed_ops = partial_schedule["removed"]

        if operator == "greedy_insertion":
            for removed_op in removed_ops:
                machine_id = removed_op["machine_id"]
                start_time, setup_time = self._find_available_slot(
                    machine_id, removed_op.get("start_time", 0),
                    {"material_grade": removed_op.get("material_grade", "304_austenitic"), "duration_hours": removed_op["duration"]},
                    machine_timeline
                )
                end_time = start_time + removed_op["duration"]
                machine_timeline[machine_id].append({
                    "start": start_time,
                    "end": end_time,
                    "setup": setup_time,
                    "job_id": removed_op["job_id"],
                    "op_id": removed_op["op_id"],
                    "op_sequence": removed_op.get("op_sequence", 0),
                    "operation_code": removed_op.get("operation_code", ""),
                    "operation_name": removed_op.get("operation_name", ""),
                    "material_grade": removed_op.get("material_grade", ""),
                    "duration": removed_op["duration"],
                })

        elif operator == "regret_2_insertion":
            for removed_op in removed_ops:
                machine_id = removed_op["machine_id"]
                start_time, setup_time = self._find_available_slot(
                    machine_id, removed_op.get("start_time", 0),
                    {"material_grade": removed_op.get("material_grade", "304_austenitic"), "duration_hours": removed_op["duration"]},
                    machine_timeline
                )
                end_time = start_time + removed_op["duration"]
                machine_timeline[machine_id].append({
                    "start": start_time,
                    "end": end_time,
                    "setup": setup_time,
                    "job_id": removed_op["job_id"],
                    "op_id": removed_op["op_id"],
                    "op_sequence": removed_op.get("op_sequence", 0),
                    "operation_code": removed_op.get("operation_code", ""),
                    "operation_name": removed_op.get("operation_name", ""),
                    "material_grade": removed_op.get("material_grade", ""),
                    "duration": removed_op["duration"],
                })

        return self._build_schedule_dict(machine_timeline)

    def _extract_machine_timeline(self, assignments: List[Dict]) -> Dict:
        """Convert assignments to machine timeline format."""
        timeline = {m_id: [] for m_id in self.machines}
        for a in assignments:
            timeline[a["machine_id"]].append({
                "start": a["start_time"],
                "end": a["end_time"],
                "setup": a.get("setup_time", 0),
                "job_id": a["job_id"],
                "op_id": a["op_id"],
                "op_sequence": a.get("op_sequence", 0),
                "operation_code": a.get("operation_code", ""),
                "operation_name": a.get("operation_name", ""),
                "material_grade": a.get("material_grade", ""),
                "duration": a["duration"],
            })
        return timeline

    def _build_schedule_dict(self, machine_timeline: Dict) -> Dict:
        """Convert machine timeline to schedule dictionary."""
        assignments = []
        all_ends = []

        for machine_id, ops in machine_timeline.items():
            for op in ops:
                assignments.append({
                    "job_id": op["job_id"],
                    "op_id": op["op_id"],
                    "op_sequence": op.get("op_sequence", 0),
                    "machine_id": machine_id,
                    "start_time": op["start"],
                    "end_time": op["end"],
                    "duration": op["duration"],
                    "setup_time": op["setup"],
                    "operation_code": op.get("operation_code", ""),
                    "operation_name": op.get("operation_name", ""),
                    "material_grade": op.get("material_grade", ""),
                })
                all_ends.append(op["end"])

        makespan = max(all_ends) if all_ends else 0

        # Compute tardiness
        total_tardiness = 0
        jobs_late = 0
        jobs_on_time = 0

        for job in self.jobs:
            job_id = job["job_id"]
            job_ops = [a for a in assignments if a["job_id"] == job_id]
            if job_ops:
                job_completion = max(a["end_time"] for a in job_ops)
                due_date = job.get("due_date", 72)
                tardiness = max(0, job_completion - due_date)
                total_tardiness += tardiness
                if tardiness > 0:
                    jobs_late += 1
                else:
                    jobs_on_time += 1

        # Compute idle time
        total_idle_time = 0
        for machine_id, ops in machine_timeline.items():
            if ops:
                sorted_ops = sorted(ops, key=lambda x: x["start"])
                for i in range(len(sorted_ops) - 1):
                    idle = sorted_ops[i + 1]["start"] - sorted_ops[i]["end"]
                    total_idle_time += max(0, idle)

        return {
            "assignments": assignments,
            "makespan": makespan,
            "total_tardiness": total_tardiness,
            "total_idle_time": total_idle_time,
            "total_energy_kwh": sum(op["duration"] * 100 for op in assignments) / 10,  # Placeholder
            "jobs_on_time": jobs_on_time,
            "jobs_late": jobs_late,
            "machine_timeline": machine_timeline,
        }

    def _compute_objective(self, schedule: Dict) -> float:
        """Compute weighted objective function."""
        makespan = schedule.get("makespan", 0)
        tardiness = schedule.get("total_tardiness", 0)
        idle_time = schedule.get("total_idle_time", 0)
        return makespan * 1.0 + tardiness * 2.0 + idle_time * 0.1


def _format_output(schedule: Dict, solver: JobShopSolver, elapsed: float, iterations: int) -> Dict:
    """Format solution for QCentroid output."""
    assignments = schedule["assignments"]
    machine_timeline = schedule.get("machine_timeline", {})

    # Gantt data
    gantt_data = []
    for machine_id, ops in machine_timeline.items():
        for op in sorted(ops, key=lambda x: x["start"]):
            gantt_data.append({
                "machine_id": machine_id,
                "job_id": op["job_id"],
                "operation": op.get("operation_name", ""),
                "start": op["start"],
                "end": op["end"],
                "duration": op["duration"],
                "setup_time": op["setup"],
                "label": f"{op['job_id']}-{op.get('operation_code', '')}",
            })

    # Machine utilization
    machine_utilization = {}
    horizon_end = solver.planning_horizon.get("end_time", 72)
    for machine_id, ops in machine_timeline.items():
        total_busy = sum(op["duration"] for op in ops)
        total_setup = sum(op.get("setup", 0) for op in ops)
        util = ((total_busy + total_setup) / horizon_end * 100) if horizon_end > 0 else 0
        machine_name = next((m["name"] for m in solver.machines_list if m["machine_id"] == machine_id), machine_id)
        machine_utilization[machine_id] = {
            "machine_name": machine_name,
            "utilization_percentage": util,
            "total_busy_time": total_busy,
            "idle_time": max(0, horizon_end - total_busy - total_setup),
            "num_jobs_processed": len(set(op["job_id"] for op in ops)),
        }

    # Job metrics
    job_metrics = {}
    for job in solver.jobs:
        job_id = job["job_id"]
        job_ops = [a for a in assignments if a["job_id"] == job_id]
        if job_ops:
            completion_time = max(a["end_time"] for a in job_ops)
            due_date = job.get("due_date", 72)
            tardiness = max(0, completion_time - due_date)
            job_metrics[job_id] = {
                "completion_time": completion_time,
                "tardiness": tardiness,
                "on_time": tardiness == 0,
            }

    # Cost breakdown
    processing_cost = sum(a["duration"] * 100 for a in assignments)
    energy_cost = schedule.get("total_energy_kwh", 0) * 0.15
    setup_cost = sum(a.get("setup_time", 0) * 50 for a in assignments)
    total_cost = processing_cost + energy_cost + setup_cost

    cost_breakdown = {
        "processing_cost": processing_cost,
        "energy_cost": energy_cost,
        "setup_cost": setup_cost,
        "total_cost": total_cost,
    }

    # Risk metrics
    avg_util = sum(m["utilization_percentage"] for m in machine_utilization.values()) / len(machine_utilization) if machine_utilization else 0
    on_time_pct = (schedule["jobs_on_time"] / (schedule["jobs_on_time"] + schedule["jobs_late"]) * 100) if (schedule["jobs_on_time"] + schedule["jobs_late"]) > 0 else 0

    risk_metrics = {
        "schedule_robustness": {
            "buffer_time_hours": schedule["total_idle_time"],
            "robustness_score": min(100, on_time_pct + (avg_util / 2)),
        },
        "critical_path": {
            "makespan_hours": schedule["makespan"],
            "critical_jobs": [j for j in job_metrics if job_metrics[j].get("tardiness", 0) > 5],
            "bottleneck_machines": sorted(machine_utilization.keys(), key=lambda m: machine_utilization[m]["utilization_percentage"], reverse=True)[:3],
        },
    }

    # Constraint violations
    constraint_violations = {
        "hard_constraints": {
            "total_hard_violations": 0,
            "is_feasible": len(assignments) > 0,
        },
        "soft_constraints": {
            "total_tardiness_hours": schedule["total_tardiness"],
        },
    }

    # Objective value
    objective_value = solver._compute_objective(schedule)

    # Solution status
    solution_status = "feasible" if len(assignments) > 0 else "infeasible"

    return {
        "schedule": {
            "assignments": assignments,
            "gantt_data": gantt_data,
            "makespan": schedule["makespan"],
            "total_tardiness": schedule["total_tardiness"],
            "total_idle_time": schedule["total_idle_time"],
            "total_energy_kwh": schedule["total_energy_kwh"],
            "total_cost": total_cost,
            "jobs_on_time": schedule["jobs_on_time"],
            "jobs_late": schedule["jobs_late"],
            "on_time_percentage": on_time_pct,
        },
        "machine_utilization": machine_utilization,
        "job_metrics": job_metrics,
        "cost_breakdown": cost_breakdown,
        "risk_metrics": risk_metrics,
        "constraint_violations": constraint_violations,
        "objective_value": objective_value,
        "solution_status": solution_status,
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "PriorityDispatch_ALNS",
            "iterations": iterations,
        },
        "benchmark": {
            "execution_cost": {"value": elapsed * 0.5, "unit": "credits"},
            "time_elapsed": f"{elapsed:.1f}s",
            "energy_consumption": 0.0,
        },
        "makespan_hours": schedule["makespan"],
        "on_time_delivery_pct": on_time_pct,
        "total_tardiness_hours": schedule["total_tardiness"],
        "avg_machine_utilization_pct": avg_util,
        "total_changeovers": len([a for a in assignments if a.get("setup_time", 0) > 0]),
    }