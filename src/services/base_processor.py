from abc import ABC, abstractmethod
from pathlib import Path

from sqlalchemy.orm import Session

from ..models.database import create_tables, get_db_session


class DataProcessor(ABC):
    """Abstract base class for dataset processors"""

    def __init__(self, data_path: str, session: Session | None = None):
        self.data_path = Path(data_path)
        self.session = session or get_db_session()
        self._processing_stats = {}

    def setup_database(self):
        """Initialize database tables"""
        create_tables()
        self._log("Database tables created successfully")

    @abstractmethod
    def get_expected_files(self) -> dict[str, str]:
        """Return mapping of logical name to filename for expected files"""
        pass

    @abstractmethod
    def process_dataset(self) -> dict[str, int]:
        """Process the complete dataset and return processing statistics"""
        pass

    def validate_files(self) -> dict[str, bool]:
        """Check which expected files are present"""
        expected_files = self.get_expected_files()
        file_status = {}

        for name, filename in expected_files.items():
            file_path = self.data_path / filename
            exists = file_path.exists()
            file_status[name] = exists

            if exists:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                self._log(f"✓ {filename} ({size_mb:.1f} MB)")
            else:
                self._log(f"✗ {filename} - Missing")

        return file_status

    def batch_process(
        self, data, process_func, batch_size: int = 1000, description: str = "items"
    ) -> int:
        """Generic batch processing with progress tracking"""
        total_items = len(data)
        processed = 0

        self._log(
            f"Processing {total_items:,} {description} in batches of {batch_size:,}..."
        )

        for i in range(0, total_items, batch_size):
            batch = (
                data.iloc[i : i + batch_size]
                if hasattr(data, "iloc")
                else data[i : i + batch_size]
            )

            batch_result = process_func(batch)
            if isinstance(batch_result, (list, tuple)):
                self.session.add_all(batch_result)
            else:
                processed += batch_result

            self.session.commit()

            if hasattr(data, "iloc"):  # pandas DataFrame
                processed += len(batch)

            if processed % (batch_size * 10) == 0:
                self._log(
                    f"  - Processed {processed:,}/{total_items:,} {description} ({processed/total_items*100:.1f}%)"
                )

        self._log(f"Completed: {processed:,} {description} processed")
        return processed

    def chunk_process(
        self, data, process_func, chunk_size: int = 50000, description: str = "records"
    ) -> int:
        """Process large datasets in chunks with progress tracking"""
        total_rows = len(data)
        processed = 0

        self._log(
            f"Processing {total_rows:,} {description} in chunks of {chunk_size:,}..."
        )

        for chunk_start in range(0, total_rows, chunk_size):
            chunk = data.iloc[chunk_start : chunk_start + chunk_size]
            chunk_result = process_func(chunk)

            if isinstance(chunk_result, int):
                processed += chunk_result
            else:
                processed += len(chunk)

            progress_pct = processed / total_rows * 100
            self._log(
                f"  - Processed {processed:,}/{total_rows:,} {description} ({progress_pct:.1f}%)"
            )

        self._log(f"Completed: {processed:,} {description} processed")
        return processed

    def get_processing_summary(self) -> dict[str, int]:
        """Get summary of processed data"""
        return self._processing_stats.copy()

    def _log(self, message: str):
        """Simple logging utility"""
        print(message)

    def _update_stats(self, data_type: str, count: int):
        """Update processing statistics"""
        self._processing_stats[data_type] = count

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session"""
        if self.session:
            self.session.close()
