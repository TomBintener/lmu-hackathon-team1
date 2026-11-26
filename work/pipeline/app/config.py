from functools import lru_cache
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

PLACEHOLDER_GEMINI_KEYS = {"PASTE_YOUR_GEMINI_API_KEY", "AQ.Ab8RN6LZCw-"}
PLACEHOLDER_AZURE_KEYS = {"PASTE_YOUR_AZURE_OPENAI_API_KEY"}
PLACEHOLDER_AZURE_ENDPOINT_MARKERS = ("YOUR_RESOURCE",)


class Settings(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    database_url: str = "postgresql://tracklist:tracklist@db:5432/tracklist"
    discogs_token: str = ""
    discogs_user_agent: str = "LMUTracklistHackathon/1.0"
    not_found_ttl_hours: int = 24
    max_candidates_to_fetch: int = 4
    discogs_timeout_sec: float = 15.0
    discogs_min_remaining: int = 5
    discogs_base_url: str = "https://api.discogs.com"
    enable_external_fallbacks: bool = True
    fallback_timeout_sec: float = 10.0
    fallback_max_candidates: int = 2
    agent_provider: str = "auto"
    gemini_api_key: str = ""
    google_genai_model: str = "gemini-3.8-flash"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_model: str = "gpt-5-mini"
    azure_openai_timeout_sec: float = 90.0
    azure_openai_enable_web_search: bool = True
    azure_openai_reasoning_effort: str = "low"
    enable_agent_fallback: bool = True
    agent_max_attempts: int = 3
    enable_queue_workers: bool = True
    discogs_queue_concurrency: int = 1
    agent_queue_concurrency: int = 2
    enrich_queue_concurrency: int = 2
    queue_poll_interval_sec: float = 0.5
    enable_idle_auto_enrichment: bool = True
    idle_enrichment_interval_sec: float = 15.0
    idle_enrichment_batch_size: int = 10
    enable_track_enrichment: bool = True
    enable_agent_track_enrichment: bool = True
    enrichment_timeout_sec: float = 10.0
    enrichment_max_master_versions: int = 2
    enrichment_max_per_track_lookups: int = 40
    enrichment_max_mb_track_lookups: int = 6
    musicbrainz_min_interval_sec: float = 1.1

    def is_azure_agent_configured(self) -> bool:
        """Return True when Azure AI Foundry endpoint and API key are non-placeholder."""
        endpoint = (self.azure_openai_endpoint or "").strip()
        key = (self.azure_openai_api_key or "").strip()
        if not endpoint or any(m in endpoint for m in PLACEHOLDER_AZURE_ENDPOINT_MARKERS):
            return False
        return bool(key and key not in PLACEHOLDER_AZURE_KEYS)

    def is_gemini_agent_configured(self) -> bool:
        """Return True when Gemini API key is non-placeholder."""
        key = (self.gemini_api_key or "").strip()
        return bool(key and key not in PLACEHOLDER_GEMINI_KEYS)

    def resolved_agent_provider(self) -> str | None:
        """Resolve active agent provider ('azure' | 'gemini' | None) from config."""
        provider = (self.agent_provider or "auto").strip().lower()
        if provider == "azure":
            return "azure" if self.is_azure_agent_configured() else None
        if provider == "gemini":
            return "gemini" if self.is_gemini_agent_configured() else None
        if self.is_azure_agent_configured():
            return "azure"
        if self.is_gemini_agent_configured():
            return "gemini"
        return None

    def is_agent_configured(self) -> bool:
        """Return True when agent fallback is enabled with a configured provider."""
        return self.enable_agent_fallback and self.resolved_agent_provider() is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
