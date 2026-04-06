"""
QCentroid Test Runner - Load input and run solver
"""
import json
import sys

if __name__ == "__main__":
    # Load input JSON
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.json"

    with open(input_file) as f:
        dic = json.load(f)

    extra_arguments = dic.get("extra_arguments", {})
    solver_params = dic.get("solver_params", {})

    # Import and run solver
    import qcentroid
    result = qcentroid.run(dic["data"], solver_params, extra_arguments)

    # Output result as JSON
    print(json.dumps(result, indent=2, default=str))