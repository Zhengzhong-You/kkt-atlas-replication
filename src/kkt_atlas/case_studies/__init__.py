"""Synthetic optimization case studies (CAP, PORT, NET, and controlled quadratic)."""

from kkt_atlas.case_studies.network_reservation import (
    NetworkReservationProblem,
    make_network_reservation_problem,
)
from kkt_atlas.case_studies.network_reservation_data_v1 import (
    NetworkReservationDataConfig,
    generate_network_reservation_data_v1,
)
from kkt_atlas.case_studies.projected_quadratic import ProjectedQuadraticProblem
from kkt_atlas.case_studies.smooth_capacity_allocation import (
    SmoothCapacityAllocationProblem,
    make_smooth_capacity_allocation_problem,
)
from kkt_atlas.case_studies.smoothed_cvar_portfolio import SmoothedCVaRPortfolioProblem

__all__ = [
    "NetworkReservationDataConfig",
    "NetworkReservationProblem",
    "ProjectedQuadraticProblem",
    "SmoothCapacityAllocationProblem",
    "SmoothedCVaRPortfolioProblem",
    "generate_network_reservation_data_v1",
    "make_network_reservation_problem",
    "make_smooth_capacity_allocation_problem",
]
