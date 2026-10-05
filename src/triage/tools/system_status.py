"""Tool: check internal region health against the public status page."""
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..data_loader import read_json
from .base import Tool


class SystemStatusArgs(BaseModel):
    region: str = Field(
        description="The customer's country code (e.g. 'TH') or an infrastructure region "
        "name (e.g. 'ap-southeast')."
    )


class CheckSystemStatus(Tool):
    name = "check_system_status"
    description = (
        "Check INTERNAL health monitoring for a region and compare it with the public status "
        "page. The public page can lag behind real incidents; if they disagree, trust the "
        "internal result."
    )
    args_model = SystemStatusArgs

    def __init__(self, status: dict[str, Any]) -> None:
        self._public = status["public_status_page"]
        self._country_to_region: dict[str, str] = status["country_to_region"]
        self._regions: dict[str, dict[str, Any]] = {r["region"]: r for r in status["regions"]}

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "CheckSystemStatus":
        return cls(read_json(data_dir / "system_status.json"))

    def run(self, args: SystemStatusArgs) -> dict[str, Any]:
        requested = args.region.strip()
        region_id = self._country_to_region.get(requested.upper(), requested.lower())
        region = self._regions.get(region_id)
        if region is None:
            return {"found": False, "requested": args.region, "known_regions": list(self._regions)}

        public_says_ok = self._public["overall"] == "all systems operational"
        return {
            "found": True,
            "internal": region,
            "public_status_page": self._public,
            "status_page_mismatch": region["health"] != "healthy" and public_says_ok,
        }
