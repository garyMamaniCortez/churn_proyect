"""Small shared helpers used by both the segmentation and churn dataset builders.

Kept in its own module (not a method on either builder) so both can import the
same logic without one depending on the other -- avoids duplicating the
day-pass-vs-membership rule in two places where it could quietly drift apart.
"""

from __future__ import annotations

import pandas as pd


def flag_membership_services(servicios: pd.DataFrame) -> pd.DataFrame:
    """Classify each servicio as a real membership vs. a single day-pass.

    A service counts as a day-pass (not a membership) when it's exactly one
    day long (`cantidad_duracion == 1` and `tipo_duracion == 'dias'`), e.g.
    'Sesion' or 'Sesion Zumba'. Everything else (monthly plans, etc.) counts
    as a membership. This distinction matters everywhere "renewal" is
    evaluated: a lapsed member buying one drop-in session should not look
    like a retained member.
    """
    is_day_pass = (servicios["cantidad_duracion"] == 1) & (servicios["tipo_duracion"] == "dias")
    return servicios.assign(es_membresia=~is_day_pass)[
        ["servicio_id", "es_membresia", "numero_ingresos"]
    ]
