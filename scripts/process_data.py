#!/usr/bin/env uv run python
"""
Script to process and load Dunnhumby dataset into PostgreSQL database
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.services.data_processor import get_data_summary, load_dunnhumby_data


def main():
    """Process the Dunnhumby dataset"""

    print("=== TPM Dataset Processor ===\n")

    # Check if sample data should be used
    use_sample = len(sys.argv) > 1 and sys.argv[1] == "--sample"

    if use_sample:
        print("Processing sample data...")
        data_path = "data/dunnhumby"

        # Generate sample data if it doesn't exist
        from scripts.download_data import generate_sample_data

        sample_dir = project_root / "data" / "dunnhumby"
        if not (sample_dir / "transaction_data.csv").exists():
            print("Generating sample data...")
            generate_sample_data(sample_dir)
    else:
        data_path = "data/dunnhumby"

    try:
        # Load and process data
        results = load_dunnhumby_data(data_path)

        if results:
            print("\n=== Processing Results ===")
            total_records = sum(results.values())
            print(f"Total records processed: {total_records:,}")

            # Get final summary
            print("\n=== Database Summary ===")
            summary = get_data_summary()
            for table, count in summary.items():
                print(f"{table.capitalize()}: {count:,}")

            print("\n✓ Dataset processing complete!")
            print("Ready to start MCP server with: python -m src.main")
        else:
            print("\n✗ No data was processed. Check that dataset files exist.")
            print("Run: python scripts/download_data.py for instructions")

    except Exception as e:
        print(f"\n✗ Error processing dataset: {e}")
        print("Make sure PostgreSQL is running and DATABASE_URL is configured")
        sys.exit(1)


if __name__ == "__main__":
    main()
