import functools
import subprocess
import os
import sys
from typing import Callable

def run_in_conda_env(env_name: str) -> Callable:
    """
    Decorator to run a function in a specified Conda environment.

    Args:
        env_name (str): Name of the Conda environment to activate (e.g., 'torch' or 'map4').

    Returns:
        Callable: Decorated function that runs in the specified Conda environment.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if running inside a Docker container
            if not os.path.exists("/.dockerenv"):
                raise EnvironmentError("This decorator is intended to run inside a Docker container.")

            # Verify that the Conda environment exists
            conda_bin = os.path.join(os.environ.get("CONDA_PREFIX", "/opt/conda"), "bin", "conda")
            result = subprocess.run(
                [conda_bin, "env", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            envs = [line.split()[0] for line in result.stdout.splitlines() if line and not line.startswith("#")]
            if env_name not in envs:
                raise EnvironmentError(f"Conda environment '{env_name}' not found. Available environments: {envs}")

            # Prepare the command to activate the Conda environment and run the function
            # Since we can't directly activate a Conda env in the same Python process,
            # we serialize the function call and execute it in a new process with the activated environment.
            script_path = os.path.abspath(sys.argv[0])
            cmd = (
                f"bash -c 'source /opt/conda/etc/profile.d/conda.sh && "
                f"conda activate {env_name} && "
                f"python {script_path} --function {func.__name__}'"
            )

            # Note: In this simplified version, we assume the function is called directly in the main script.
            # For a more robust solution, you might need to serialize args/kwargs and handle function execution differently.
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
                return result.returncode
            except subprocess.CalledProcessError as e:
                print(f"Error running function in Conda environment '{env_name}': {e.stderr}", file=sys.stderr)
                raise

        return wrapper
    return decorator