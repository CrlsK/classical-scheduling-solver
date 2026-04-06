"""QCentroid Classical Solver: Priority Dispatch + Adaptive Large Neighborhood Search (ALNS)
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
from additional_output_generator import generate_additional_output

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

        logger.info(f"QCentroid Classical Solver Starting...")
        logger.info(f"Jobs: {len(jobs)}, Machines: {len(machines)}")
        logger.info(f"Max iterations: {max_iterations}, Max time: {max_time_s}s")

        # Phase 1: Data Normalization
        logger.info("Phase 1: Data Normalization")
        normalized_data = _normalize_input(jobs, machines, processing_times, setup_matrix)

        # Phase 2: Priority Dispatch - Build initial solution
        logger.info("Phase 2: Priority Dispatch")
        initial_schedule = _priority_dispatch(
            normalized_data["jobs"],
            normalized_data["machines"],
            normalized_data["processing_times"],
            planning_horizon
        )

        # Phase 3: ALNS - Improve solution
        logger.info("Phase 3: ALNS Improvement")
        best_schedule = _alns_improve(
            initial_schedule,
            normalized_data,
            max_iterations=max_iterations,
            max_time_s=max_time_s,
            temperature=temperature,
            cooling_rate=cooling_rate,
            planning_horizon=planning_horizon
        )

        # Phase 4: Compute metrics and format output
        logger.info("Phase 4: Output Formatting")
        metrics = _compute_metrics(best_schedule, normalized_data, planning_horizon)
        
        # Generate additional outputs
        result = _format_output(best_schedule, metrics, normalized_data, planning_horizon, metadata)
        generate_additional_output(input_data, result, algorithm_name="QCentroid Classical Solver")

        elapsed = time.time() - start_time
        logger.info(f"Solver completed in {elapsed:.2f}s")
        logger.info(f"Final metrics: makespan={metrics.get('makespan')}, tardiness={metrics.get('total_tardiness')}")

        return result

    except Exception as e:
        logger.error(f"Solver error: {str(e)}", exc_info=True)
        return {
            "status": "ERROR",
            "error_message": str(e),
            "schedule": [],
            "metrics": {}
        }


def _normalize_input(jobs: List[dict], machines: List[dict], processing_times: Dict, setup_matrix: Dict) -> Dict:
    """
    Normalize input data structure.
    """
    return {
        "jobs": jobs,
        "machines": machines,
        "processing_times": processing_times,
        "setup_matrix": setup_matrix
    }


def _priority_dispatch(jobs: List[dict], machines: List[dict], processing_times: Dict, planning_horizon: dict) -> List[dict]:
    """
    Build initial schedule using priority dispatch.
    Assigns jobs to machines using earliest available time + shortest processing time.
    """
    end_time = planning_horizon.get("end_time", 72)
    schedule = []
    machine_free_time = {m["id"]: 0 for m in machines}

    # Sort jobs by priority (EDD: Earliest Due Date)
    sorted_jobs = sorted(jobs, key=lambda j: j.get("due_date", float('inf')))

    for job in sorted_jobs:
        job_id = job["id"]
        best_machine = None
        best_start = float('inf')

        # Find best machine (earliest available)
        for machine in machines:
            mach_id = machine["id"]
            proc_time = processing_times.get(f"{job_id}_{mach_id}", processing_times.get(job_id, 1))
            start_time = machine_free_time[mach_id]
            finish_time = start_time + proc_time

            if finish_time <= end_time and start_time < best_start:
                best_machine = mach_id
                best_start = start_time

        if best_machine is not None:
            proc_time = processing_times.get(f"{job_id}_{best_machine}", processing_times.get(job_id, 1))
            finish_time = best_start + proc_time
            machine_free_time[best_machine] = finish_time

            schedule.append({
                "job_id": job_id,
                "machine_id": best_machine,
                "start_time": best_start,
                "end_time": finish_time
            })

    return schedule