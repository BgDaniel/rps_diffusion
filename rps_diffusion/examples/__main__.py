"""Run all example scenarios: ``python -m rps_diffusion.examples``."""

from __future__ import annotations

from rps_diffusion.examples import a_omega0, b_asymmetric_rates, c_stripe_fronts, d_shaped_domains


def main() -> None:
    """Run scenarios (a) to (d) in sequence."""
    for module in (a_omega0, b_asymmetric_rates, c_stripe_fronts, d_shaped_domains):
        module.main()


if __name__ == "__main__":
    main()
