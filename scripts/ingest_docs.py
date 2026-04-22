from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ingestion_service import IngestionService


if __name__ == "__main__":
    result = IngestionService().upload_all(force=False)
    print(result)
