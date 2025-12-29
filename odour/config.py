from dataclasses import dataclass

@dataclass
class TrajectoryConfig:
    dt_seconds: int = 120          # subhour step (2 min)
    lookback_hours: float = 5.0    # how far back
    n_particles: int = 300         # ensemble size
    sigma_dir_deg: float = 12.0    # directional jitter (deg)
    sigma_speed_frac: float = 0.12 # speed jitter fraction
    max_speed_mps: float = 60.0    # sanity cap
