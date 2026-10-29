"""Generate synthetic spin coating measurements as NOMAD ``*.archive.json`` files.

Each file ``measurement_<n>.archive.json`` holds one ``SpinCoating`` entry. The film
thickness follows a simple spin coating model, so Bayesian optimization can learn
how it depends on the process parameters:

    thickness = SCALE * concentration**1.2 * (speed / 1000 rpm)**-0.5
                * (1 + 0.5 * exp(-time / 8 s))

- Thickness grows slightly faster than linearly with the concentration, as more
  concentrated solutions are more viscous.
- Thickness falls with the square root of the spin coating speed.
- Short spins leave a thicker film, beyond about 30 s the time barely matters.

For example 1.2 M, 4000 rpm and 30 s give about 400 nm. The reported thickness
carries relative noise (film-to-film variation and measurement error) and is
rounded to 0.1 nm, like a profilometer reading.

Parameters that are not given are sampled uniformly within the bounds below, rounded
to what can be set in the lab (0.01 M, 10 rpm, 1 s). A parameter given once is used
for every measurement, a parameter given N times provides one value per measurement.

Examples:
    # 20 measurements with uniformly sampled parameters
    python generate_spin_coating_measurements.py -n 20 --seed 1

    # A speed series at a fixed concentration and spinning time
    python generate_spin_coating_measurements.py --speed 1000 2000 4000 \\
        --concentration 1.2 --time 30

    # Measure a single recommended point, continuing the numbering
    python generate_spin_coating_measurements.py --concentration 0.85 \\
        --speed 3200 --time 25 --start-index 21
"""

import argparse
import json
import math
import random
from pathlib import Path

SCHEMA = 'nomad_bayesian_optimization.schema_packages.spin_coating.SpinCoating'

# Bounds for uniformly sampled parameters. The spin coating optimization action uses
# the same bounds and rounding (nomad_bayesian_optimization.actions.spin_coating.models).
CONCENTRATION_BOUNDS = (0.2, 1.5)  # M
SPEED_BOUNDS = (1000.0, 6000.0)  # rpm
TIME_BOUNDS = (10.0, 60.0)  # s

# Model constants.
THICKNESS_SCALE = 635.0  # nm, at 1 M, 1000 rpm and long spinning times
CONCENTRATION_EXPONENT = 1.2
SPEED_EXPONENT = 0.5
SHORT_SPIN_EXCESS = 0.5  # relative extra thickness at zero spinning time
THINNING_TIME = 8.0  # s


def film_thickness(concentration: float, speed: float, time: float) -> float:
    """Return the noise-free film thickness (nm) for the given process parameters."""
    return (
        THICKNESS_SCALE
        * concentration**CONCENTRATION_EXPONENT
        * (speed / 1000.0) ** -SPEED_EXPONENT
        * (1.0 + SHORT_SPIN_EXCESS * math.exp(-time / THINNING_TIME))
    )


def build_archive(
    index: int, concentration: float, speed: float, time: float, thickness: float
) -> dict:
    """Return the archive of a single spin coating measurement."""
    return {
        'metadata': {
            'entry_name': f'Spin coating measurement {index}',
            'comment': 'Synthetic measurement generated from a spin coating model.',
        },
        'data': {
            'm_def': SCHEMA,
            'precursor_concentration': concentration,
            'spin_coating_speed': speed,
            'spinning_time': time,
            'film_thickness': thickness,
        },
    }


def _values(
    given: list[float] | None,
    bounds: tuple[float, float],
    decimals: int,
    count: int,
    rng: random.Random,
) -> list[float]:
    """Return ``count`` parameter values: the given ones or uniformly sampled."""
    if given is None:
        return [round(rng.uniform(*bounds), decimals) for _ in range(count)]
    if len(given) == 1:
        return given * count
    return given


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-n',
        '--count',
        type=int,
        help='Number of measurements. Defaults to the number of values given for '
        'the parameters, or 1.',
    )
    parser.add_argument(
        '--concentration',
        type=float,
        nargs='+',
        help='Precursor concentration(s) in M. Sampled from '
        f'{CONCENTRATION_BOUNDS} when not given.',
    )
    parser.add_argument(
        '--speed',
        type=float,
        nargs='+',
        help=f'Spin coating speed(s) in rpm. Sampled from {SPEED_BOUNDS} when not '
        'given.',
    )
    parser.add_argument(
        '--time',
        type=float,
        nargs='+',
        help=f'Spinning time(s) in s. Sampled from {TIME_BOUNDS} when not given.',
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.03,
        help='Relative standard deviation of the measured thickness (default: 0.03).',
    )
    parser.add_argument(
        '--seed', type=int, help='Random seed for reproducible measurements.'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=1,
        help='Number of the first measurement file (default: 1).',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=Path,
        default=Path(),
        help='Directory to write the files to (default: current directory).',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    given = [args.concentration, args.speed, args.time]
    lengths = [len(values) for values in given if values is not None]
    count = args.count if args.count is not None else max(lengths, default=1)
    if count < 1:
        parser.error('the number of measurements must be at least 1')
    if any(length not in (1, count) for length in lengths):
        parser.error(f'give each parameter either once or {count} times')
    if any(value <= 0 for values in given if values for value in values):
        parser.error('parameter values must be positive')
    if args.noise < 0:
        parser.error('the noise must not be negative')

    rng = random.Random(args.seed)
    concentrations = _values(args.concentration, CONCENTRATION_BOUNDS, 2, count, rng)
    speeds = _values(args.speed, SPEED_BOUNDS, -1, count, rng)
    times = _values(args.time, TIME_BOUNDS, 0, count, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for offset, (concentration, speed, time) in enumerate(
        zip(concentrations, speeds, times, strict=True)
    ):
        index = args.start_index + offset
        thickness = film_thickness(concentration, speed, time)
        thickness = round(thickness * rng.lognormvariate(0.0, args.noise), 1)
        archive = build_archive(index, concentration, speed, time, thickness)
        path = args.output_dir / f'measurement_{index}.archive.json'
        path.write_text(json.dumps(archive, indent=2) + '\n')
        print(
            f'{path}: {concentration:g} M, {speed:g} rpm, {time:g} s '
            f'-> {thickness:g} nm'
        )


if __name__ == '__main__':
    main()
