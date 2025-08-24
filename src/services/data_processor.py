import pandas as pd
from sqlalchemy.orm import Session

from ..models.entities import Household, Product, Promotion, Store, Transaction
from .base_processor import DataProcessor


class DunnhumbyProcessor(DataProcessor):
    """Processor for Dunnhumby 'The Complete Journey' dataset"""

    def __init__(self, data_path: str = "data/dunnhumby", session: Session = None):
        super().__init__(data_path, session)

    def get_expected_files(self) -> dict[str, str]:
        """Return mapping of logical name to filename for Dunnhumby dataset"""
        return {
            "households": "hh_demographic.csv",
            "products": "product.csv",
            "transactions": "transaction_data.csv",
            "promotions": "causal_data.csv",
            "coupons": "coupon.csv",
            "coupon_redemptions": "coupon_redempt.csv",
            "campaign_descriptions": "campaign_desc.csv",
            "campaign_table": "campaign_table.csv",
        }

    def load_csv_files(self) -> dict[str, pd.DataFrame]:
        """Load CSV files that exist and have content"""
        expected_files = self.get_expected_files()
        dataframes = {}

        for name, filename in expected_files.items():
            file_path = self.data_path / filename
            if file_path.exists():
                # Check if file is empty
                if file_path.stat().st_size == 0:
                    self._log(f"  - Skipping {filename}: File is empty")
                    continue
                    
                self._log(f"Loading {filename}...")
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    if df.empty:
                        self._log(f"  - Skipping {filename}: No data rows")
                        continue
                    dataframes[name] = df
                    self._log(f"  - Loaded {len(df):,} rows")
                except Exception as e:
                    self._log(f"  - Error loading {filename}: {e}")
            else:
                self._log(f"  - Warning: {filename} not found")

        return dataframes

    def process_households(self, hh_demo_df: pd.DataFrame) -> int:
        """Process and store household data"""

        def create_household_batch(batch):
            households = []
            for _, row in batch.iterrows():
                household = Household(
                    household_key=str(row["household_key"]),
                    age_desc=row.get("AGE_DESC", ""),
                    marital_status_code=row.get("MARITAL_STATUS_CODE", ""),
                    income_desc=row.get("INCOME_DESC", ""),
                    homeowner_desc=row.get("HOMEOWNER_DESC", ""),
                    hh_comp_desc=row.get("HH_COMP_DESC", ""),
                    household_size_desc=row.get("HOUSEHOLD_SIZE_DESC", ""),
                    kid_category_desc=row.get("KID_CATEGORY_DESC", ""),
                )
                households.append(household)
            return households

        return self.batch_process(
            hh_demo_df, create_household_batch, description="households"
        )

    def process_products(self, product_df: pd.DataFrame) -> int:
        """Process and store product data"""

        def create_product_batch(batch):
            products = []
            for _, row in batch.iterrows():
                product = Product(
                    product_id=int(row["PRODUCT_ID"]),
                    department=row.get("DEPARTMENT", ""),
                    commodity_desc=row.get("COMMODITY_DESC", ""),
                    sub_commodity_desc=row.get("SUB_COMMODITY_DESC", ""),
                    manufacturer=row.get("MANUFACTURER", ""),
                    brand=row.get("BRAND", ""),
                    curr_size_of_product=row.get("CURR_SIZE_OF_PRODUCT", ""),
                )
                products.append(product)
            return products

        return self.batch_process(
            product_df, create_product_batch, description="products"
        )

    def process_transactions(
        self, trans_df: pd.DataFrame, chunk_size: int = 50000
    ) -> int:
        """Process and store transaction data in chunks"""
        # Get unique store_ids and create store records
        unique_stores = trans_df["STORE_ID"].unique()
        self._create_stores(unique_stores)

        def create_transaction_chunk(chunk):
            transactions = []
            for _, row in chunk.iterrows():
                transaction = Transaction(
                    household_key=str(row["household_key"]),
                    basket_id=str(row["BASKET_ID"]),
                    product_id=int(row["PRODUCT_ID"]),
                    store_id=int(row["STORE_ID"]),
                    day=int(row["DAY"]),
                    week_no=int(row["WEEK_NO"]),
                    trans_time=int(row.get("TRANS_TIME", 0)),
                    quantity=int(row["QUANTITY"]),
                    sales_value=float(row["SALES_VALUE"]),
                    retail_disc=float(row.get("RETAIL_DISC", 0)),
                    coupon_disc=float(row.get("COUPON_DISC", 0)),
                    coupon_match_disc=float(row.get("COUPON_MATCH_DISC", 0)),
                )
                transactions.append(transaction)

            self.session.add_all(transactions)
            return len(transactions)

        return self.chunk_process(
            trans_df, create_transaction_chunk, chunk_size, "transactions"
        )

    def _create_stores(self, store_ids: list[int]):
        """Create store records for unique store IDs"""
        self._log("Creating store records...")

        try:
            stores = [Store(store_id=int(store_id)) for store_id in store_ids]
            self.session.add_all(stores)
            self.session.commit()
            self._log(f"  - Created {len(stores)} store records")
        except Exception as e:
            self._log(f"  - Error creating stores: {e}")
            self.session.rollback()
            # Try to continue without failing

    def process_promotions(self, causal_df: pd.DataFrame) -> int:
        """Process and store promotion/causal data"""
        if causal_df.empty:
            self._log("No causal data to process")
            return 0

        def create_promotion_batch(batch):
            promotions = []
            for _, row in batch.iterrows():
                promotion = Promotion(
                    product_id=int(row["PRODUCT_ID"]),
                    store_id=int(row["STORE_ID"]),
                    start_day=int(row.get("START_DAY", 0)),
                    end_day=int(row.get("END_DAY", 0)),
                    display_loc=row.get("DISPLAY_LOC", ""),
                    mailer_loc=row.get("MAILER_LOC", ""),
                    promotion_type=row.get("PROMOTION_TYPE", "Unknown"),
                )
                promotions.append(promotion)
            return promotions

        return self.batch_process(
            causal_df, create_promotion_batch, description="promotions"
        )

    def clear_existing_data(self):
        """Clear existing data to avoid duplicate key violations"""
        self._log("Clearing existing data...")
        try:
            # Clear tables in reverse dependency order
            self.session.query(Transaction).delete()
            self.session.query(Promotion).delete()
            self.session.query(Store).delete()
            self.session.query(Product).delete()
            self.session.query(Household).delete()
            self.session.commit()
            self._log("  - Existing data cleared")
        except Exception as e:
            self._log(f"  - Warning: Error clearing data: {e}")
            self.session.rollback()

    def process_dataset(self) -> dict[str, int]:
        """Process the complete Dunnhumby dataset"""
        self._log("Starting Dunnhumby dataset processing...")

        # Setup database
        self.setup_database()
        
        # Clear existing data to avoid duplicates
        self.clear_existing_data()

        # Load CSV files
        dataframes = self.load_csv_files()

        if not dataframes:
            self._log(
                f"No data files found. Please ensure the Dunnhumby dataset is downloaded to: {self.data_path}"
            )
            return {}

        results = {}

        # Process each data type in logical order
        if "households" in dataframes:
            count = self.process_households(dataframes["households"])
            results["households"] = count
            self._update_stats("households", count)

        if "products" in dataframes:
            count = self.process_products(dataframes["products"])
            results["products"] = count
            self._update_stats("products", count)

        if "transactions" in dataframes:
            count = self.process_transactions(dataframes["transactions"])
            results["transactions"] = count
            self._update_stats("transactions", count)

        if "promotions" in dataframes:
            count = self.process_promotions(dataframes["promotions"])
            results["promotions"] = count
            self._update_stats("promotions", count)

        self._log("\n=== Processing Complete ===")
        total_records = sum(results.values())
        for data_type, count in results.items():
            self._log(f"{data_type.capitalize()}: {count:,} records")
        self._log(f"Total: {total_records:,} records processed")

        return results

    def get_data_summary(self) -> dict[str, int]:
        """Get summary of data in database"""
        summary = {}

        try:
            summary["households"] = self.session.query(Household).count()
            summary["products"] = self.session.query(Product).count()
            summary["stores"] = self.session.query(Store).count()
            summary["transactions"] = self.session.query(Transaction).count()
            summary["promotions"] = self.session.query(Promotion).count()
        except Exception as e:
            self._log(f"Error getting data summary: {e}")

        return summary


# Convenience functions for backward compatibility
def load_dunnhumby_data(data_path: str = "data/dunnhumby") -> dict[str, int]:
    """Main function to load and process Dunnhumby dataset"""
    with DunnhumbyProcessor(data_path) as processor:
        return processor.process_dataset()


def get_data_summary() -> dict[str, int]:
    """Get summary of processed data"""
    with DunnhumbyProcessor() as processor:
        return processor.get_data_summary()
