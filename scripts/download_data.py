#!/usr/bin/env uv run python
"""
Script to download Dunnhumby 'The Complete Journey' dataset

Since the dataset requires registration, this script provides instructions
and can validate the downloaded files.
"""

import sys
from pathlib import Path


def check_data_files(data_dir: Path) -> dict[str, bool]:
    """Check which required data files are present"""

    required_files = [
        "hh_demographic.csv",
        "product.csv",
        "transaction_data.csv",
        "causal_data.csv",
        "coupon.csv",
        "coupon_redempt.csv",
        "campaign_desc.csv",
        "campaign_table.csv",
    ]

    file_status = {}
    for filename in required_files:
        file_path = data_dir / filename
        file_status[filename] = file_path.exists()

        if file_status[filename]:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {filename} - Missing")

    return file_status


def download_instructions():
    """Print instructions for downloading the dataset"""

    print(
        """
=== Dunnhumby Dataset Download Instructions ===

The Dunnhumby 'The Complete Journey' dataset requires registration to access.

Steps to download:

1. Visit: https://www.dunnhumby.com/source-files/
2. Click on "The Complete Journey" dataset
3. Fill out the registration form with your details
4. Download the dataset ZIP file when you receive access
5. Extract all CSV files to: data/dunnhumby/

Expected files:
- hh_demographic.csv      (Household demographics)
- product.csv            (Product master data)  
- transaction_data.csv    (Transaction records - LARGE FILE ~2.6GB)
- causal_data.csv        (Promotion/causal data)
- coupon.csv             (Coupon information)
- coupon_redempt.csv     (Coupon redemptions)
- campaign_desc.csv      (Campaign descriptions)
- campaign_table.csv     (Campaign details)

Dataset Size: ~2.7GB total
Records: ~2.5M transactions from 2,500 households

Alternative smaller datasets for testing:
- You can create sample data using scripts/generate_sample_data.py
- Or use the synthetic data generator for initial development

After downloading, run:
python scripts/process_data.py
    """
    )


def generate_sample_data(data_dir: Path):
    """Generate small sample dataset for testing"""

    import numpy as np
    import pandas as pd

    print("Generating sample data for testing...")

    # Sample households
    np.random.seed(42)
    households = pd.DataFrame(
        {
            "household_key": [f"HH_{i:04d}" for i in range(1, 101)],
            "AGE_DESC": np.random.choice(
                ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"], 100
            ),
            "MARITAL_STATUS_CODE": np.random.choice(["A", "B", "U"], 100),
            "INCOME_DESC": np.random.choice(
                ["Under 25K", "25-49K", "50-74K", "75-99K", "100-124K", "125K+"], 100
            ),
            "HOMEOWNER_DESC": np.random.choice(["Homeowner", "Renter", "Unknown"], 100),
            "HH_COMP_DESC": np.random.choice(
                [
                    "1 Adult Kids",
                    "2 Adults Kids",
                    "2 Adults No Kids",
                    "Single Female",
                    "Single Male",
                ],
                100,
            ),
            "HOUSEHOLD_SIZE_DESC": np.random.choice(["1", "2", "3", "4", "5+"], 100),
            "KID_CATEGORY_DESC": np.random.choice(["None/Unknown", "1-2", "3+"], 100),
        }
    )

    # Sample products
    products = pd.DataFrame(
        {
            "PRODUCT_ID": range(1000, 1100),
            "DEPARTMENT": np.random.choice(["GROCERY", "DRUG GM", "PRODUCE"], 100),
            "COMMODITY_DESC": np.random.choice(
                ["FROZEN PIZZA", "CEREAL", "SOFT DRINKS", "BEER", "CHEESE"], 100
            ),
            "SUB_COMMODITY_DESC": np.random.choice(
                ["FROZEN PIZZA", "COLD CEREAL", "CARBONATED BEVERAGES"], 100
            ),
            "MANUFACTURER": np.random.choice(
                ["PRIVATE", "KELLOGG", "GENERAL MILLS", "COCA COLA", "PEPSI"], 100
            ),
            "BRAND": np.random.choice(
                ["PRIVATE", "CHEERIOS", "FROSTED FLAKES", "COCA COLA", "PEPSI"], 100
            ),
            "CURR_SIZE_OF_PRODUCT": np.random.choice(
                ["12 OZ", "18 OZ", "24 OZ", "32 OZ"], 100
            ),
        }
    )

    # Sample transactions
    n_transactions = 10000
    transactions = pd.DataFrame(
        {
            "household_key": np.random.choice(
                households["household_key"], n_transactions
            ),
            "BASKET_ID": [f"B_{i:06d}" for i in range(n_transactions)],
            "PRODUCT_ID": np.random.choice(products["PRODUCT_ID"], n_transactions),
            "STORE_ID": np.random.choice([1, 2, 3, 4, 5], n_transactions),
            "DAY": np.random.randint(1, 711, n_transactions),  # 2 years of days
            "WEEK_NO": np.random.randint(1, 103, n_transactions),  # 2 years of weeks
            "TRANS_TIME": np.random.randint(600, 2200, n_transactions),
            "QUANTITY": np.random.randint(1, 5, n_transactions),
            "SALES_VALUE": np.round(np.random.uniform(0.99, 15.99, n_transactions), 2),
            "RETAIL_DISC": np.round(np.random.uniform(0, 3.00, n_transactions), 2),
            "COUPON_DISC": np.round(np.random.uniform(0, 1.50, n_transactions), 2),
            "COUPON_MATCH_DISC": np.round(
                np.random.uniform(0, 0.75, n_transactions), 2
            ),
        }
    )

    # Sample promotions
    promotions = pd.DataFrame(
        {
            "PRODUCT_ID": np.random.choice(products["PRODUCT_ID"], 500),
            "STORE_ID": np.random.choice([1, 2, 3, 4, 5], 500),
            "START_DAY": np.random.randint(1, 650, 500),
            "END_DAY": np.random.randint(651, 711, 500),
            "DISPLAY_LOC": np.random.choice(["", "1", "2", "3", "4"], 500),
            "MAILER_LOC": np.random.choice(["", "1", "2"], 500),
            "PROMOTION_TYPE": np.random.choice(
                ["BOGO", "Discount", "Display", "Coupon"], 500
            ),
        }
    )

    # Save sample data
    data_dir.mkdir(exist_ok=True)
    households.to_csv(data_dir / "hh_demographic.csv", index=False)
    products.to_csv(data_dir / "product.csv", index=False)
    transactions.to_csv(data_dir / "transaction_data.csv", index=False)
    promotions.to_csv(data_dir / "causal_data.csv", index=False)

    # Create empty files for other datasets
    for filename in [
        "coupon.csv",
        "coupon_redempt.csv",
        "campaign_desc.csv",
        "campaign_table.csv",
    ]:
        (data_dir / filename).touch()

    print(f"Sample data generated in {data_dir}/")
    print(f"  - {len(households)} households")
    print(f"  - {len(products)} products")
    print(f"  - {len(transactions)} transactions")
    print(f"  - {len(promotions)} promotions")


def main():
    """Main script execution"""

    # Get data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "dunnhumby"

    print("=== Dunnhumby Dataset Manager ===\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        generate_sample_data(data_dir)
        return

    # Check current status
    print("Checking for dataset files...")
    file_status = check_data_files(data_dir)

    missing_files = [f for f, exists in file_status.items() if not exists]

    if not missing_files:
        print(f"\n✓ All required files found in {data_dir}/")
        print("Ready to process dataset with: python scripts/process_data.py")
    else:
        print(f"\n✗ Missing {len(missing_files)} files")
        download_instructions()
        print("\nTo generate sample data for testing:")
        print("python scripts/download_data.py --sample")


if __name__ == "__main__":
    main()
