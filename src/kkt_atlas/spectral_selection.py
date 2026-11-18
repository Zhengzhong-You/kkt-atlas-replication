"""Training-only selection of a response-basis dimension."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kkt_atlas.problem import Array


@dataclass(frozen=True)
class SpectralWidthSelection:
    """One width choice and the singular-value quantities supporting it.

    The spectrum field records the supplied singular values. Energy
    calculations use effective_spectrum, in which values no larger than
    numerical_threshold are treated as zero. For a zero effective spectrum,
    retained_ratio is one by convention because every admissible width has
    zero discarded cross-covariance.
    """

    width: int
    retained_ratio: float
    discarded_squared_tail: float
    spectrum: Array
    effective_spectrum: Array
    effective_rank: int
    target_ratio: float
    numerical_threshold: float
    target_achieved: bool


def select_spectral_width(
    singular_values: Array,
    target_ratio: float,
    *,
    minimum_width: int = 1,
    maximum_width: int | None = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 1e-12,
) -> SpectralWidthSelection:
    """Select the smallest width retaining a target squared-energy ratio.

    The rule uses only a fitted singular-value spectrum. It can therefore be
    applied independently to every chart without inspecting validation
    realizations.
    """

    spectrum = np.asarray(singular_values, dtype=np.float64)
    if spectrum.ndim != 1 or len(spectrum) == 0:
        raise ValueError("singular_values must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        raise ValueError("singular_values must be finite and nonnegative")

    ratio = float(target_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("target_ratio must lie in (0,1]")

    absolute_tolerance = _nonnegative_finite_float(
        absolute_tolerance,
        name="absolute_tolerance",
    )
    relative_tolerance = _nonnegative_finite_float(
        relative_tolerance,
        name="relative_tolerance",
    )
    if not isinstance(minimum_width, int) or isinstance(minimum_width, bool):
        raise TypeError("minimum_width must be an integer")
    if maximum_width is None:
        maximum_width = len(spectrum)
    if not isinstance(maximum_width, int) or isinstance(maximum_width, bool):
        raise TypeError("maximum_width must be an integer")
    if not 1 <= minimum_width <= maximum_width <= len(spectrum):
        raise ValueError("width bounds must satisfy 1 <= minimum_width <= maximum_width <= len(singular_values)")

    leading_value = float(spectrum[0])
    order_tolerance = max(
        absolute_tolerance,
        relative_tolerance * leading_value,
    )
    if np.any(spectrum[1:] > spectrum[:-1] + order_tolerance):
        raise ValueError("singular_values must be in nonincreasing order")

    numerical_threshold = max(
        absolute_tolerance,
        relative_tolerance * leading_value,
    )
    effective = spectrum.copy()
    effective[effective <= numerical_threshold] = 0.0
    effective_rank = int(np.count_nonzero(effective))

    if effective_rank == 0:
        width = minimum_width
        retained_ratio = 1.0
        discarded_squared_tail = 0.0
    else:
        scaled = effective / effective[0]
        squared_energy = np.square(scaled)
        cumulative_ratio = np.cumsum(squared_energy) / np.sum(squared_energy)
        crossing = int(np.searchsorted(cumulative_ratio, ratio, side="left")) + 1
        width = max(minimum_width, min(crossing, maximum_width))

        retained_index = min(width, effective_rank) - 1
        retained_ratio = float(cumulative_ratio[retained_index])
        discarded_squared_tail = float(np.dot(effective[width:], effective[width:]))

    raw = np.ascontiguousarray(spectrum.copy())
    cleaned = np.ascontiguousarray(effective)
    raw.setflags(write=False)
    cleaned.setflags(write=False)
    ratio_tolerance: float = max(
        float(10.0 * np.finfo(np.float64).eps),
        relative_tolerance,
    )
    return SpectralWidthSelection(
        width=width,
        retained_ratio=retained_ratio,
        discarded_squared_tail=discarded_squared_tail,
        spectrum=raw,
        effective_spectrum=cleaned,
        effective_rank=effective_rank,
        target_ratio=ratio,
        numerical_threshold=numerical_threshold,
        target_achieved=retained_ratio + ratio_tolerance >= ratio,
    )


def select_chart_spectral_widths(
    spectra: Mapping[int, Array],
    target_ratio: float,
    *,
    minimum_width: int = 1,
    maximum_width: int | None = None,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 1e-12,
) -> dict[int, SpectralWidthSelection]:
    """Apply one frozen spectral ratio separately to every context chart."""

    if not spectra:
        raise ValueError("spectra must contain at least one chart")
    chart_indices = tuple(spectra)
    if any(not isinstance(chart, int) or isinstance(chart, bool) or chart < 0 for chart in chart_indices):
        raise ValueError("chart indices must be nonnegative integers")
    return {
        chart: select_spectral_width(
            spectra[chart],
            target_ratio,
            minimum_width=minimum_width,
            maximum_width=maximum_width,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for chart in sorted(chart_indices)
    }


def _nonnegative_finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result
