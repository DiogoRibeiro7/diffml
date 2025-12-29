"""
Dashboard CLI entry point for DiffML.

Simplified entry point for launching the Streamlit dashboard.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch the DiffML dashboard."""
    print("🚀 Starting DiffML Dashboard...")

    # Find dashboard.py in the project root
    dashboard_path = Path(__file__).parent.parent.parent / 'dashboard.py'

    if not dashboard_path.exists():
        print(f"❌ Error: Dashboard not found at {dashboard_path}")
        print("Please ensure dashboard.py is in the project root directory.")
        sys.exit(1)

    # Launch streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(dashboard_path),
        '--server.port', '8501',
        '--server.address', 'localhost',
        '--server.headless', 'true'
    ]

    print("📊 Dashboard starting at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard.\n")

    try:
        # Run streamlit
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()