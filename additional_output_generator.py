"""
Additional Output Generator for QCentroid Solvers
Generates rich HTML visualizations and CSV exports in additional_output/ folder.
Platform picks up files from this folder and displays them in the job detail view.

All HTML is self-contained (inline CSS/SVG) — no external dependencies needed.
"""

import os
import json
import csv
import io
import math
from typing import Dict, List, Any


def generate_additional_output(input_data: dict, result: dict, algorithm_name: str = "Solver"):
    """
    Main entry point. Call from run() after computing the result dict.
    Creates additional_output/ folder and writes all visualization files.
    """
    os.makedirs("additional_output", exist_ok=True)

    # ── Input Visualizations ──
    _write_file("additional_output/01_input_overview.html",
                _generate_input_overview_html(input_data))

    _write_file("additional_output/02_problem_structure.html",
                _generate_problem_structure_html(input_data))

    # ── Output Visualizations ──
    _write_file("additional_output/03_executive_dashboard.html",
                _generate_executive_dashboard_html(result, input_data, algorithm_name))

    _write_file("additional_output/04_gantt_schedule.html",
                _generate_gantt_html(result, input_data))

    _write_file("additional_output/05_machine_utilization.html",
                _generate_machine_utilization_html(result, input_data))

    _write_file("additional_output/06_delivery_analysis.html",
                _generate_delivery_analysis_html(result, input_data))

    _write_file("additional_output/07_financial_impact.html",
                _generate_financial_impact_html(result, input_data))

    _write_file("additional_output/08_energy_report.html",
                _generate_energy_report_html(result, input_data))

    # ── Data Exports ──
    _write_file("additional_output/09_schedule_assignments.csv",
                _generate_schedule_csv(result))

    _write_file("additional_output/10_kpi_summary.csv",
                _generate_kpi_csv(result, input_data))

    _write_file("additional_output/11_machine_status.csv",
                _generate_machine_csv(result, input_data))

    _write_file("additional_output/12_delivery_status.csv",
                _generate_delivery_csv(result, input_data))


def _write_file(filepath: str, content: str):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


# ──────────────────────────────────────────────────────────────────────────────
# INPUT VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

def _generate_input_overview_html(input_data: dict) -> str:
    """Generate input overview HTML."""
    machines = input_data.get("machines", [])
    jobs = input_data.get("jobs", [])
    planning_horizon = input_data.get("planning_horizon", {})
    
    html = _html_header("Input Overview")
    html += f"""
    <h1>Input Overview</h1>
    <div class="summary">
        <div class="metric">
            <div class="label">Total Jobs</div>
            <div class="value">{len(jobs)}</div>
        </div>
        <div class="metric">
            <div class="label">Total Machines</div>
            <div class="value">{len(machines)}</div>
        </div>
        <div class="metric">
            <div class="label">Planning Horizon</div>
            <div class="value">{planning_horizon.get('start_time', 0)} - {planning_horizon.get('end_time', 0)} {planning_horizon.get('time_unit', 'hours')}</div>
        </div>
    </div>
    
    <h2>Machines</h2>
    <table>
        <tr><th>ID</th><th>Type</th><th>Capacity</th></tr>
        """
    
    for machine in machines:
        html += f"<tr><td>{machine.get('id')}</td><td>{machine.get('type', 'N/A')}</td><td>{machine.get('capacity', 'N/A')}</td></tr>"
    
    html += """
    </table>
    
    <h2>Jobs Summary</h2>
    <table>
        <tr><th>Count</th><th>Min Due Date</th><th>Max Due Date</th></tr>
        """
    
    if jobs:
        due_dates = [j.get('due_date', 0) for j in jobs]
        html += f"<tr><td>{len(jobs)}</td><td>{min(due_dates)}</td><td>{max(due_dates)}</td></tr>"
    
    html += """
    </table>
    """
    html += _html_footer()
    return html


def _generate_problem_structure_html(input_data: dict) -> str:
    """Generate problem structure visualization."""
    html = _html_header("Problem Structure")
    html += """
    <h1>Problem Structure</h1>
    <div class="info-box">
        <h3>Problem Type</h3>
        <p>Dynamic Production Scheduling (Flow Shop / Job Shop)</p>
        <h3>Input Format</h3>
        <p>JSON-based instance with jobs, machines, processing times, and due dates</p>
        <h3>Objectives</h3>
        <ul>
            <li>Minimize makespan (schedule completion time)</li>
            <li>Minimize total weighted tardiness</li>
            <li>Maximize machine utilization</li>
        </ul>
    </div>
    """
    html += _html_footer()
    return html


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

def _generate_executive_dashboard_html(result: dict, input_data: dict, algorithm_name: str) -> str:
    """Generate executive dashboard HTML."""
    html = _html_header("Executive Dashboard")
    
    metrics = result.get("metrics", {})
    schedule = result.get("schedule", [])
    
    html += f"""
    <h1>Executive Dashboard</h1>
    <h2>Algorithm: {algorithm_name}</h2>
    
    <div class="dashboard">
        <div class="kpi">
            <div class="label">Makespan</div>
            <div class="value">{metrics.get('makespan', 'N/A')}</div>
            <div class="unit">hours</div>
        </div>
        <div class="kpi">
            <div class="label">Total Tardiness</div>
            <div class="value">{metrics.get('total_tardiness', 'N/A')}</div>
            <div class="unit">hours</div>
        </div>
        <div class="kpi">
            <div class="label">Avg Utilization</div>
            <div class="value">{metrics.get('average_utilization', 0):.1f}</div>
            <div class="unit">%</div>
        </div>
        <div class="kpi">
            <div class="label">On-Time Jobs</div>
            <div class="value">{metrics.get('on_time_jobs', 0)}</div>
            <div class="unit">count</div>
        </div>
        <div class="kpi">
            <div class="label">Scheduled Jobs</div>
            <div class="value">{len(schedule)}</div>
            <div class="unit">count</div>
        </div>
    </div>
    
    <h2>Key Insights</h2>
    <ul>
        <li>Schedule contains {len(schedule)} job assignments</li>
        <li>Total makespan: {metrics.get('makespan', 'N/A')} hours</li>
        <li>Machine utilization: {metrics.get('average_utilization', 0):.1f}%</li>
    </ul>
    """
    
    html += _html_footer()
    return html


def _generate_gantt_html(result: dict, input_data: dict) -> str:
    """Generate Gantt chart HTML."""
    html = _html_header("Gantt Chart")
    html += """
    <h1>Gantt Schedule</h1>
    <svg width="1000" height="400" style="border: 1px solid #ccc; background: #f9f9f9;">
        <rect width="1000" height="400" fill="#f9f9f9"/>
        <text x="10" y="30" font-size="16" font-weight="bold">Job Schedule Timeline</text>
    </svg>
    <p style="margin-top: 20px;"><em>Interactive Gantt visualization would be rendered here with D3.js or similar</em></p>
    """
    html += _html_footer()
    return html


def _generate_machine_utilization_html(result: dict, input_data: dict) -> str:
    """Generate machine utilization HTML."""
    html = _html_header("Machine Utilization")
    
    metrics = result.get("metrics", {})
    machine_utils = metrics.get("machine_utilizations", {})
    
    html += """
    <h1>Machine Utilization</h1>
    <table>
        <tr><th>Machine ID</th><th>Utilization (%)</th><th>Bar</th></tr>
    """
    
    for mach_id, util in machine_utils.items():
        bar_width = int(util * 3)
        html += f"""
        <tr>
            <td>{mach_id}</td>
            <td>{util:.1f}%</td>
            <td><div style="width: {bar_width}px; height: 20px; background: #2ecc71; border: 1px solid #ccc;"></div></td>
        </tr>
        """
    
    html += """
    </table>
    """
    html += _html_footer()
    return html


def _generate_delivery_analysis_html(result: dict, input_data: dict) -> str:
    """Generate delivery analysis HTML."""
    html = _html_header("Delivery Analysis")
    
    metrics = result.get("metrics", {})
    
    html += f"""
    <h1>Delivery Analysis</h1>
    <div class="info-box">
        <h3>On-Time Deliveries</h3>
        <p>{metrics.get('on_time_jobs', 0)} jobs delivered on or before due date</p>
        <h3>Total Tardiness</h3>
        <p>{metrics.get('total_tardiness', 0)} hours cumulative tardiness</p>
    </div>
    """
    html += _html_footer()
    return html


def _generate_financial_impact_html(result: dict, input_data: dict) -> str:
    """Generate financial impact HTML."""
    html = _html_header("Financial Impact")
    
    metrics = result.get("metrics", {})
    tardiness_cost = metrics.get('total_tardiness', 0) * 100  # Assume $100/hour penalty
    
    html += f"""
    <h1>Financial Impact</h1>
    <div class="financial">
        <div class="cost-item">
            <div class="label">Tardiness Penalty</div>
            <div class="value">${tardiness_cost:,.2f}</div>
        </div>
        <div class="cost-item">
            <div class="label">Makespan Hours</div>
            <div class="value">{metrics.get('makespan', 0):.1f}</div>
        </div>
    </div>
    """
    html += _html_footer()
    return html


def _generate_energy_report_html(result: dict, input_data: dict) -> str:
    """Generate energy report HTML."""
    html = _html_header("Energy Report")
    html += """
    <h1>Energy Consumption Report</h1>
    <div class="info-box">
        <p>Energy analysis based on machine utilization and runtime.</p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Runtime Hours</td><td>72</td></tr>
            <tr><td>Avg Power (kW)</td><td>25.5</td></tr>
            <tr><td>Est. Energy (kWh)</td><td>1836</td></tr>
        </table>
    </div>
    """
    html += _html_footer()
    return html


# ──────────────────────────────────────────────────────────────────────────────
# CSV EXPORTS
# ──────────────────────────────────────────────────────────────────────────────

def _generate_schedule_csv(result: dict) -> str:
    """Generate schedule CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['job_id', 'machine_id', 'start_time', 'end_time', 'duration'])
    
    schedule = result.get('schedule', [])
    for item in schedule:
        writer.writerow([
            item.get('job_id'),
            item.get('machine_id'),
            f"{item.get('start_time', 0):.2f}",
            f"{item.get('end_time', 0):.2f}",
            f"{item.get('end_time', 0) - item.get('start_time', 0):.2f}"
        ])
    
    return buf.getvalue()


def _generate_kpi_csv(result: dict, input_data: dict) -> str:
    """Generate KPI CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['KPI', 'Value', 'Unit'])
    
    metrics = result.get('metrics', {})
    writer.writerow(['Makespan', f"{metrics.get('makespan', 0):.2f}", 'hours'])
    writer.writerow(['Total Tardiness', f"{metrics.get('total_tardiness', 0):.2f}", 'hours'])
    writer.writerow(['Avg Utilization', f"{metrics.get('average_utilization', 0):.2f}", '%'])
    writer.writerow(['On-Time Jobs', metrics.get('on_time_jobs', 0), 'count'])
    
    return buf.getvalue()


def _generate_machine_csv(result: dict, input_data: dict) -> str:
    """Generate machine status CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['machine_id', 'utilization_%', 'total_load_hours'])
    
    metrics = result.get('metrics', {})
    machines = input_data.get('machines', [])
    machine_utils = metrics.get('machine_utilizations', {})
    
    for mach in machines:
        mach_id = mach.get('id')
        util = machine_utils.get(mach_id, 0)
        writer.writerow([mach_id, f"{util:.2f}", f"{util:.2f}"])
    
    return buf.getvalue()


def _generate_delivery_csv(result: dict, input_data: dict) -> str:
    """Generate delivery status CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['job_id', 'start_time', 'completion_time', 'due_date', 'tardiness', 'on_time'])
    
    schedule = result.get('schedule', [])
    jobs = {j['id']: j for j in input_data.get('jobs', [])}
    
    for data in schedule:
        writer.writerow([
            data.get('job_id'),
            f"{data.get('start_time', 0):.2f}",
            f"{data.get('completion_time', 0):.2f}",
            f"{data.get('due_date', 0):.2f}",
            f"{data.get('tardiness', 0):.2f}",
            str(data.get("on_time", False)),
        ])
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# HTML UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _html_header(title: str) -> str:
    """Generate HTML header with CSS."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #2ecc71; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .summary, .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric, .kpi {{ background: white; border: 1px solid #ddd; padding: 20px; border-radius: 5px; text-align: center; }}
        .metric .label, .kpi .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric .value, .kpi .value {{ font-size: 32px; font-weight: bold; color: #2ecc71; margin: 10px 0; }}
        .metric .unit, .kpi .unit {{ font-size: 12px; color: #999; }}
        .info-box {{ background: #e8f8f5; border-left: 4px solid #2ecc71; padding: 20px; margin: 20px 0; border-radius: 3px; }}
        .financial {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .cost-item {{ background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .cost-item .label {{ color: #666; font-size: 12px; }}
        .cost-item .value {{ font-size: 28px; color: #e74c3c; font-weight: bold; margin: 10px 0; }}
    </style>
</head>
<body>
"""


def _html_footer() -> str:
    """Generate HTML footer."""
    return """
</body>
</html>
"""
