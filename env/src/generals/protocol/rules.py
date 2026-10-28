"""Standard, unmodified 1v1/FFA rules pinned to the deployed official client."""

RULES_VERSION = "generals.io-31.4.5-classic-v1"
OFFICIAL_CLIENT = "https://generals.io/generals-main-prod-v31.4.5-e5c53eb3.js"
OFFICIAL_CLIENT_SHA256 = "777be342300996f23a97e540248fa9765ab7f59534b0a7b2bc80f738ea325e8f"
RECRUIT_TICKS = 2
LAND_TICKS = 50
SURRENDER_GRACE_TICKS = 50
MAX_GAME_TICKS = 50_000
MAX_IDLE_TICKS = 2_000

END_REASONS = (None, "capture", "surrender", "game_limit", "idle_limit", "afk")
