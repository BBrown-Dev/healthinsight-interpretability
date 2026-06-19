import os
import time
import matplotlib.pyplot as plt

# Create directories if they don't exist
def ensure_dir(path: str):
    """Ensure a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


# Save a plot to the outputs/ directory
def save_plot(filename: str, tight=True):
    """Save the current plot to the outputs/ directory with a given filename."""
    ensure_dir("outputs")
    if tight:
        plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=300)
    print(f"[INFO] Plot saved to outputs/{filename}")


# Simple logger for workflow steps
def log(message: str):
    """Log message to stdout."""
    print(f"[INFO] {message}")


# Timer context manager for profiling long operations
class Timer:
    def __init__(self, task_name="Task"):
        """Context manager for timing code execution."""
        self.task_name = task_name

    def __enter__(self):
        """Start the timer."""
        self.start = time.time()
        log(f"{self.task_name} started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer."""
        elapsed = time.time() - self.start
        log(f"{self.task_name} completed in {elapsed:.2f} seconds.")