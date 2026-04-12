"""Template-based synthetic dataset generation for probing SMIXAE experts on discrete concepts."""

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import typer

names = [
    "Alice",
    "Bob",
    "Charlie",
    "Dr. Smith",
    "Sarah",
    "James",
    "Elena",
    "Prof. Wang",
    "the PI",
    "the technician",
    "Dr. Patel",
    "Maria",
    "the postdoc",
    "Kevin",
    "Dr. Chen",
    "Fatima",
    "the intern",
    "Prof. Müller",
    "Yuki",
    "the supervisor",
    "Raj",
    "Dr. Okafor",
    "the nurse",
    "Liam",
    "Dr. Rossi",
    "Amara",
    "the coordinator",
    "Prof. Kim",
    "Omar",
    "the analyst",
]


def _enumerate_unique(templates, labels_list, label_fmt, n_samples):
    """Enumerate all unique (template × name × label_value) combinations.

    Deduplicates on the rendered sentence, shuffles, and returns up to ``n_samples`` rows.
    """
    combos = list(itertools.product(templates, names, labels_list))
    random.shuffle(combos)

    seen = set()
    data = []
    for template, name, label_val in combos:
        sentence = template.format(name=name, **{label_fmt[0]: label_val})
        if sentence in seen:
            continue
        seen.add(sentence)
        idx = labels_list.index(label_val)
        data.append(
            {
                "Sentence": sentence,
                "Label": f"{idx:02}_{label_val}",
            }
        )
        if len(data) >= n_samples:
            break

    random.shuffle(data)
    return pd.DataFrame(data)


def generate_weekdays(n_samples=1000):
    """Generate a probing dataset for day-of-week (Monday–Sunday).

    Creates sentences from lab-context templates mentioning a specific day.  Labels are
    the day names, ordered Monday through Sunday.

    Regression targets:
        - ``target_day``: ordinal integer 0–6 (Monday=0).
        - ``target_sin_7d``, ``target_cos_7d``: 7-day cyclical encoding.
        - ``target_is_weekend``: 1 for Saturday/Sunday, 0 otherwise.
        - ``target_is_weekday``: 1 for Monday–Friday, 0 otherwise.

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, and regression target columns.
    """
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    templates = [
        "{name} will finalize the PCR results on {day}",
        "The shipment of reagents is scheduled for {day}",
        "Please ensure the incubator is cleaned by {day}",
        "The weekly lab sync has been moved to {day}",
        "Data backup must be completed every {day}",
        "The clinical trial phase begins this {day}",
        "Is the centrifuge maintenance occurring on {day}",
        "{name} is presenting the literature review on {day}",
        "We expect the peer review feedback by {day}",
        "The microscope calibration is due on {day}",
        "Submit the grant proposal before {day}",
        "The liquid nitrogen refill happens on {day}",
        "The sample size will be recalculated on {day}",
        "Our collaborator is visiting the facility on {day}",
        "The ethics committee will meet on {day}",
        "Please verify the freezer temperature logs for {day}",
        "{name} scheduled the MRI scan for {day}",
        "The chemical waste pickup is every {day}",
        "We will start the sequencing run on {day}",
        "The department seminar is hosted on {day}",
        "Check the titration levels again on {day}",
        "The statistical analysis will be performed on {day}",
        "Are we still on track for the launch this {day}",
        "{name} noted a discrepancy in the logs from {day}",
        "The autoclave is out of service until {day}",
        "The final abstract is due this {day}",
        "Re-calibrate the mass spectrometer on {day}",
        "The longitudinal study concludes on {day}",
        "The lab will be closed for the holiday on {day}",
        "{name} is responsible for the morning rounds on {day}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(days) + n_without * len(days)
    print(
        f"Weekdays: {len(templates)} templates × {len(names)} names × "
        f"{len(days)} days → {max_unique} max unique sentences"
    )

    df = _enumerate_unique(templates, days, ("day",), n_samples)

    # ── Regression targets ────────────────────────────────────────────
    day_to_idx = {d: i for i, d in enumerate(days)}

    def _day_idx(label: str) -> int:
        return day_to_idx[label.split("_", 1)[1]]

    day_vals = df["Label"].apply(_day_idx).astype(int)
    df["target_day"] = day_vals
    df["target_sin_7d"] = np.sin(2 * np.pi * day_vals / 7)
    df["target_cos_7d"] = np.cos(2 * np.pi * day_vals / 7)
    df["target_is_weekend"] = (day_vals >= 5).astype(int)
    df["target_is_weekday"] = (day_vals < 5).astype(int)

    return df


def generate_hours(n_samples=1000):
    """Generate a probing dataset for hour-of-day (1AM–12AM, 24 classes).

    Creates sentences from lab-context templates mentioning a specific clock time.
    Labels are the 24 hour values in AM/PM format, ordered chronologically.

    Regression targets:
        - ``target_hour``: 24h integer (12AM=0, 1AM=1, …, 11PM=23).
        - ``target_sin_24h``, ``target_cos_24h``: 24-hour cyclical encoding.
        - ``target_sin_12h``, ``target_cos_12h``: 12-hour cyclical encoding.
        - ``target_is_pm``: 1 if hour ∈ 12–23, 0 otherwise.
        - ``target_is_am``: 1 if hour ∈ 0–11, 0 otherwise.

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, and regression target columns.
    """
    hours = [
        "1AM",
        "2AM",
        "3AM",
        "4AM",
        "5AM",
        "6AM",
        "7AM",
        "8AM",
        "9AM",
        "10AM",
        "11AM",
        "12PM",
        "1PM",
        "2PM",
        "3PM",
        "4PM",
        "5PM",
        "6PM",
        "7PM",
        "8PM",
        "9PM",
        "10PM",
        "11PM",
        "12AM",
    ]

    templates = [
        "{name} set the alarm for {hour}",
        "The incubation period ends at the hour of {hour}",
        "Please check the sample integrity at the hour of {hour}",
        "The automated sequence is programmed for the hour of {hour}",
        "Meeting with the ethics board starts at the hour of {hour}",
        "Data synchronization will occur at the hour of {hour}",
        "{name} will start the titration at the hour of {hour}",
        "The lab access logs recorded an entry at the hour of {hour}",
        "Pressure readings must be logged at the hour of {hour}",
        "The cooling system cycles off at the hour of {hour}",
        "Expect the delivery of isotopes by the hour of {hour}",
        "The centrifuge run finishes at the hour of {hour}",
        "{name} reported a power surge around the hour of {hour}",
        "Final calibration is scheduled for the hour of {hour}",
        "The server maintenance window begins at the hour of {hour}",
        "Review the preliminary results at the hour of {hour}",
        "Shift handover is prompt, at the hour of {hour}",
        "The chemical reaction reached peak at the hour of {hour}",
        "Emergency backup systems were tested at the hour of {hour}",
        "{name} will calibrate the sensors at the hour of {hour}",
        "The last observation was recorded at the hour of {hour}",
        "Please shut down the workstations by the hour of {hour}",
        "The ventilation system increases flow at the hour of {hour}",
        "Safety inspections are conducted at the hour of {hour}",
        "{name} is scheduled for bench work at the hour of {hour}",
        "The biopsy results are expected by the hour of {hour}",
        "Monitor the heart rate variability until the hour of {hour}",
        "The sterilization cycle completes at the hour of {hour}",
        "Update the project management board by the hour of {hour}",
        "{name} will upload the raw data at the hour of {hour}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(hours) + n_without * len(hours)
    print(
        f"Hours: {len(templates)} templates × {len(names)} names × "
        f"{len(hours)} hours → {max_unique} max unique sentences"
    )

    df = _enumerate_unique(templates, hours, ("hour",), n_samples)

    # ── Regression targets ────────────────────────────────────────────
    # Map AM/PM strings → 24h integer: 12AM=0, 1AM=1, …, 11AM=11, 12PM=12, 1PM=13, …, 11PM=23
    _hour_to_24h: dict[str, int] = {}
    for h_str in hours:
        if h_str.endswith("AM"):
            n = int(h_str[:-2])
            _hour_to_24h[h_str] = 0 if n == 12 else n
        else:  # PM
            n = int(h_str[:-2])
            _hour_to_24h[h_str] = 12 if n == 12 else n + 12

    hour_vals = df["Label"].apply(lambda lbl: _hour_to_24h[lbl.split("_", 1)[1]]).astype(int)
    df["target_hour"] = hour_vals
    df["target_sin_24h"] = np.sin(2 * np.pi * hour_vals / 24)
    df["target_cos_24h"] = np.cos(2 * np.pi * hour_vals / 24)
    df["target_sin_12h"] = np.sin(2 * np.pi * (hour_vals % 12) / 12)
    df["target_cos_12h"] = np.cos(2 * np.pi * (hour_vals % 12) / 12)
    # is_pm: hours 12–23 are PM; hour 0 (12AM) is AM
    df["target_is_pm"] = ((hour_vals >= 12) & (hour_vals != 0)).astype(int)
    df["target_is_am"] = ((hour_vals < 12) | (hour_vals == 0)).astype(int)

    return df


def generate_temperatures(n_samples=1000):
    """Generate a probing dataset for temperature using numeric Fahrenheit values (0–212°F).

    Samples temperatures uniformly from integers 0–212°F.  Because there are ~213
    unique label values, Fisher discriminant scoring is not meaningful — use
    continuity-based pre-filtering (``label_column: null`` in the dataset config)
    and rely on regression probing for scoring.

    Regression targets:
        - ``target_temp_f``: raw Fahrenheit value.
        - ``target_log_temp_f``: ``log1p(target_temp_f)``.

    Args:
        n_samples: Number of sentences to generate (sampled uniformly from 0–212°F).

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, ``target_temp_f``,
        ``target_log_temp_f``.
    """
    temp_range = list(range(0, 213))  # 0–212 °F inclusive

    templates = [
        "{name} noted the sample chamber registered {temp}°F",
        "The lab conditions today measured {temp}°F",
        "Outside the thermometer read {temp}°F",
        "{name} complained the office was showing {temp}°F on the thermostat",
        "The reactor core temperature read {temp}°F",
        "Patients reported feeling uncomfortable when the room reached {temp}°F",
        "The greenhouse environment stabilized at {temp}°F",
        "{name} described the cleanroom as holding steady at {temp}°F",
        "The storage unit is running at {temp}°F",
        "Field conditions were recorded at {temp}°F during collection",
        "The water bath measured {temp}°F",
        "{name} adjusted the thermostat after the room hit {temp}°F",
        "The incubation environment was maintained at {temp}°F",
        "Volunteers noted the testing room registered {temp}°F",
        "The fermentation tank held at {temp}°F",
        "{name} flagged that the freezer was reading {temp}°F",
        "The climate chamber was set to {temp}°F",
        "Surface readings indicated conditions were {temp}°F",
        "The server room alarm triggered at {temp}°F",
        "{name} measured the soil temperature as {temp}°F",
        "Ambient conditions in the corridor measured {temp}°F",
        "The curing oven was running at {temp}°F",
        "{name} said the walk-in cooler was holding at {temp}°F",
        "The drying chamber atmosphere registered {temp}°F",
        "Morning readings showed the habitat at {temp}°F",
        "The bioreactor jacket temperature read {temp}°F",
        "{name} recorded the growth chamber at {temp}°F",
        "The ventilation output measured {temp}°F",
        "The reagent shelf area registered {temp}°F",
        "{name} reported the autoclave room at {temp}°F",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    print(
        f"Temperatures: {len(templates)} templates ({n_with_name} with name) × "
        f"213 Fahrenheit values → sampling {n_samples} uniformly"
    )

    sampled_temps = random.choices(temp_range, k=n_samples * 3)  # oversample for dedup headroom

    seen: set[str] = set()
    data = []
    for temp in sampled_temps:
        template = random.choice(templates)
        name = random.choice(names)
        sentence = template.format(name=name, temp=temp)
        if sentence in seen:
            continue
        seen.add(sentence)
        data.append(
            {
                "Sentence": sentence,
                "Label": str(temp),
                "target_temp_f": float(temp),
                "target_log_temp_f": float(np.log1p(float(temp))),
            }
        )
        if len(data) >= n_samples:
            break

    random.shuffle(data)
    df = pd.DataFrame(data)
    print(f"Temperatures: generated {len(df)} unique sentences")
    return df


def generate_time_units(n_samples=1000):
    """Generate a probing dataset for time units (millisecond → century).

    Creates sentences from lab-context templates mentioning a specific time unit.
    Labels are 10 units ordered shortest to longest duration.

    Regression target:
        - ``target_log_seconds``: ``log10`` of actual duration in seconds,
          ranging from −3 (millisecond) to ~9.5 (century).

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, ``target_log_seconds``.
    """
    # Ordered from shortest to longest duration
    units = [
        "millisecond",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "decade",
        "century",
    ]

    # Actual duration in seconds for each unit
    _UNIT_SECONDS: dict[str, float] = {
        "millisecond": 0.001,
        "second": 1.0,
        "minute": 60.0,
        "hour": 3_600.0,
        "day": 86_400.0,
        "week": 604_800.0,
        "month": 2_628_000.0,    # 30.44 days
        "year": 31_536_000.0,
        "decade": 315_360_000.0,
        "century": 3_153_600_000.0,
    }

    templates = [
        "{name} set the experiment timer for one {unit}",
        "The process is measured in units of one {unit}",
        "Each interval in the protocol corresponds to one {unit}",
        "The signal persists for approximately one {unit}",
        "Resolution of the sensor is roughly one {unit}",
        "{name} noted the delay was about one {unit}",
        "The reaction completes in exactly one {unit}",
        "The standard interval for this assay is one {unit}",
        "{name} calculated the half-life as roughly one {unit}",
        "The oscillation period measures one {unit}",
        "Events are logged to the nearest {unit}",
        "The clock ticks once per {unit}",
        "The gap between readings is one {unit}",
        "{name} reported latency of approximately one {unit}",
        "The simulation advances by one {unit}",
        "Phase transitions occur every {unit}",
        "Data is captured at a resolution of one {unit}",
        "The protocol requires a pause of one {unit}",
        "{name} measured the response time in one {unit}",
        "The cache refreshes every {unit}",
        "Cell division occurs on the order of one {unit}",
        "The epoch duration is fixed at one {unit}",
        "{name} confirmed the lag was under one {unit}",
        "The retention policy spans one {unit}",
        "Sample decay is tracked per {unit}",
        "The synchronization window aligns to one {unit}",
        "{name} budgeted resources by the {unit}",
        "The checkpoint interval is set to one {unit}",
        "{name} recorded the duration as one {unit}",
        "The experiment was designed around the timescale of one {unit}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(units) + n_without * len(units)
    print(
        f"Time units: {len(templates)} templates × {len(names)} names × "
        f"{len(units)} units → {max_unique} max unique sentences"
    )

    df = _enumerate_unique(templates, units, ("unit",), n_samples)

    # ── Regression targets ────────────────────────────────────────────
    df["target_log_seconds"] = df["Label"].apply(
        lambda lbl: float(np.log10(_UNIT_SECONDS[lbl.split("_", 1)[1]]))
    )

    return df


def generate_body_parts(n_samples=1000):
    """Generate a probing dataset for anatomical body parts (head to toe).

    Creates sentences from medical/clinical templates referencing a body part.
    Labels are 16 body parts ordered head-to-toe, useful for probing spatial or
    positional representations.

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence`` and ``Label``.
    """
    # Ordered head-to-toe
    parts = [
        "head",
        "face",
        "neck",
        "shoulder",
        "chest",
        "back",
        "arm",
        "elbow",
        "wrist",
        "hand",
        "abdomen",
        "hip",
        "leg",
        "knee",
        "ankle",
        "foot",
    ]

    templates = [
        "The scan revealed an abnormality in the {part}",
        "{name} reported persistent soreness in the {part}",
        "The injury was localized to the {part}",
        "The nurse applied a bandage to the {part}",
        "{name} noticed visible swelling in the {part}",
        "The X-ray focused on the patient's {part}",
        "Range of motion was restricted in the {part}",
        "{name} documented bruising on the {part}",
        "The physical exam highlighted tenderness in the {part}",
        "The protective gear is designed for the {part}",
        "{name} measured the circumference of the {part}",
        "The biopsy was taken from tissue near the {part}",
        "Temperature elevation was noted at the {part}",
        "{name} reported numbness spreading from the {part}",
        "The rash first appeared on the {part}",
        "The surgeon focused the incision near the {part}",
        "{name} documented visible asymmetry in the {part}",
        "The compression sleeve is worn on the {part}",
        "Reflexes were tested at the {part}",
        "{name} reported that the discomfort originated in the {part}",
        "The burn was classified as superficial, affecting the {part}",
        "Muscle weakness was observed in the {part}",
        "{name} applied ice to reduce inflammation in the {part}",
        "The fracture was confirmed in the {part}",
        "Skin discoloration was observed on the {part}",
        "{name} noted restricted blood flow to the {part}",
        "The physical therapist worked on mobility of the {part}",
        "Lymph node enlargement was found near the {part}",
        "{name} described a tingling sensation in the {part}",
        "The imaging clearly depicted damage to the {part}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(parts) + n_without * len(parts)
    print(
        f"Body parts: {len(templates)} templates × {len(names)} names × "
        f"{len(parts)} parts → {max_unique} max unique sentences"
    )

    return _enumerate_unique(templates, parts, ("part",), n_samples)


def generate_living_things(n_samples=1000):
    """Generate a probing dataset for biological organisms by complexity (plants → mammals).

    Creates sentences from ecology/biology templates referencing a type of organism.
    Labels are 14 organism categories ordered from simpler (moss) to more complex (mammal).

    Regression targets:
        - ``target_is_animal``: 0 for plants (moss–tree), 1 for animals (insect–mammal).
        - ``target_taxon``: taxonomic group (Plant=0, Invertebrate=1, Fish=2,
          Amphibian=3, Reptile=4, Bird=5, Mammal=6).

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, ``target_is_animal``,
        ``target_taxon``.
    """
    # Ordered from simpler to more complex organisms (plants then animals)
    organisms = [
        "moss",
        "fern",
        "grass",
        "flower",
        "shrub",
        "tree",
        "insect",
        "arachnid",
        "crustacean",
        "fish",
        "amphibian",
        "reptile",
        "bird",
        "mammal",
    ]

    # 0-5 are plants, 6-13 are animals
    _IS_ANIMAL: dict[str, int] = {o: int(i >= 6) for i, o in enumerate(organisms)}
    _TAXON: dict[str, int] = {
        "moss": 0, "fern": 0, "grass": 0, "flower": 0, "shrub": 0, "tree": 0,
        "insect": 1, "arachnid": 1, "crustacean": 1,
        "fish": 2,
        "amphibian": 3,
        "reptile": 4,
        "bird": 5,
        "mammal": 6,
    }

    templates = [
        "The researcher identified the specimen as a {organism}",
        "{name} photographed what appeared to be a {organism}",
        "The field guide confirmed the find was a {organism}",
        "Samples were collected from a living {organism}",
        "{name} documented the habitat of the {organism}",
        "The ecology report focused on the local {organism}",
        "DNA sequencing confirmed the sample came from a {organism}",
        "{name} observed the behavior of a wild {organism}",
        "The museum exhibit featured a preserved {organism}",
        "The invasive species turned out to be a {organism}",
        "{name} extracted RNA from the tissue of a {organism}",
        "The biome is dominated by the {organism}",
        "Children in the class were asked to draw a {organism}",
        "{name} trained for years on identifying a {organism}",
        "The fossil record shows evidence of the ancient {organism}",
        "Conservation efforts focused on preserving the {organism}",
        "{name} cultured a colony derived from a {organism}",
        "The biodiversity index recorded the presence of a {organism}",
        "The nature documentary featured the remarkable {organism}",
        "{name} noted that the diet study examined a {organism}",
        "The ecosystem depends heavily on the {organism}",
        "The genome was successfully sequenced from a {organism}",
        "{name} spent the summer studying a local {organism}",
        "The biology textbook chapter covered the {organism}",
        "Environmental impact was assessed for the {organism}",
        "{name} confirmed the endangered status of the {organism}",
        "The sanctuary was established to protect the {organism}",
        "Biochemists extracted compounds from the {organism}",
        "{name} tagged and released the captured {organism}",
        "The grant funded a three-year study of the {organism}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(organisms) + n_without * len(organisms)
    print(
        f"Living things: {len(templates)} templates × {len(names)} names × "
        f"{len(organisms)} organisms → {max_unique} max unique sentences"
    )

    df = _enumerate_unique(templates, organisms, ("organism",), n_samples)

    # ── Regression targets ────────────────────────────────────────────
    def _organism(label: str) -> str:
        return label.split("_", 1)[1]

    df["target_is_animal"] = df["Label"].apply(lambda lbl: _IS_ANIMAL[_organism(lbl)]).astype(int)
    df["target_taxon"] = df["Label"].apply(lambda lbl: _TAXON[_organism(lbl)]).astype(int)

    return df


def generate_colors(n_samples=1000):
    """Generate a probing dataset for color perception in lab contexts (ROYGBIV order).

    Creates sentences from laboratory templates where a color is observed.
    Labels are 8 colors in rainbow order, plus pink.

    Regression targets:
        - ``target_hue_sin``, ``target_hue_cos``: sin/cos of hue angle on the color
          wheel (approximate; red=0°, orange=30°, yellow=60°, green=120°, blue=240°,
          indigo=260°, violet=280°, pink=345°).
        - ``target_r``, ``target_g``, ``target_b``: normalized RGB values [0, 1].
        - ``target_color_cat``: broad category (Warm=0: red/orange/pink,
          Natural=1: yellow/green, Cool=2: blue/indigo/violet).

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, and the 6 regression target columns.
    """
    # Rainbow order (ROYGBIV + pink)
    colors = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet",
        "pink",
    ]

    # Hue degrees on the color wheel (approximate, physically motivated)
    _HUE_DEG: dict[str, float] = {
        "red": 0.0,
        "orange": 30.0,
        "yellow": 60.0,
        "green": 120.0,
        "blue": 240.0,
        "indigo": 260.0,
        "violet": 280.0,
        "pink": 345.0,
    }

    # Normalized RGB values [0, 1]
    _COLOR_RGB: dict[str, tuple[float, float, float]] = {
        "red":    (1.00, 0.00, 0.00),
        "orange": (1.00, 0.65, 0.00),
        "yellow": (1.00, 1.00, 0.00),
        "green":  (0.00, 0.50, 0.00),
        "blue":   (0.00, 0.00, 1.00),
        "indigo": (0.29, 0.00, 0.51),
        "violet": (0.56, 0.00, 1.00),
        "pink":   (1.00, 0.75, 0.80),
    }

    # Broad color category
    _COLOR_CAT: dict[str, int] = {
        "red": 0, "orange": 0, "pink": 0,    # Warm
        "yellow": 1, "green": 1,              # Natural
        "blue": 2, "indigo": 2, "violet": 2,  # Cool
    }

    templates = [
        "{name} labeled the sample vial with tape coded {color}",
        "When the reaction completed, the indicator solution turned {color}",
        "The safety manual requires biohazard container markings to be {color}",
        "{name} noted the precipitate had a faint tint of {color}",
        "The reference standard is packaged in a box that is {color}",
        "After incubation, the culture medium appeared {color}",
        "{name} selected the filter for the fluorescence measurement that was {color}",
        "Under the microscope, the tissue stain result was {color}",
        "The warning light on the centrifuge turned {color}",
        "{name} marked the control wells with ink that was {color}",
        "The LED indicator on the freezer is {color}",
        "The chromatography band migrated as a stripe that appeared {color}",
        "{name} wrapped the light-sensitive sample in foil that was {color}",
        "At the measured reading, the pH strip changed to {color}",
        "The reagent bottle cap is color-coded {color}",
        "{name} highlighted the chart outliers in {color}",
        "On the selective agar plate, the colony appeared {color}",
        "Dye uptake rendered the cell membrane {color}",
        "{name} distinguished the treatment group using tags that were {color}",
        "Near the heat source, the thermal imaging regions were {color}",
        "The coating on the electrode is {color}",
        "{name} described the crystalline precipitate as {color}",
        "The hazard placard background for this chemical class is {color}",
        "Under UV light the compound fluoresced {color}",
        "The calibration bead suspension is {color}",
        "{name} sorted the slides by condition into trays that were {color}",
        "Against the membrane, the western blot band appeared {color}",
        "In the acidic solution, the litmus paper turned {color}",
        "At its center, {name} noted the fungal colony was {color}",
        "The signal trace on the oscilloscope was displayed in {color}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(colors) + n_without * len(colors)
    print(
        f"Colors: {len(templates)} templates × {len(names)} names × "
        f"{len(colors)} colors → {max_unique} max unique sentences"
    )

    df = _enumerate_unique(templates, colors, ("color",), n_samples)

    # ── Regression targets ────────────────────────────────────────────
    def _color(label: str) -> str:
        return label.split("_", 1)[1]

    hue_rad = df["Label"].apply(lambda lbl: np.deg2rad(_HUE_DEG[_color(lbl)]))
    df["target_hue_sin"] = np.sin(hue_rad)
    df["target_hue_cos"] = np.cos(hue_rad)
    df["target_r"] = df["Label"].apply(lambda lbl: _COLOR_RGB[_color(lbl)][0])
    df["target_g"] = df["Label"].apply(lambda lbl: _COLOR_RGB[_color(lbl)][1])
    df["target_b"] = df["Label"].apply(lambda lbl: _COLOR_RGB[_color(lbl)][2])
    df["target_color_cat"] = df["Label"].apply(lambda lbl: _COLOR_CAT[_color(lbl)]).astype(int)

    return df


def generate_months(n_samples=1000):
    """Generate a probing dataset for calendar months (January–December).

    Creates sentences from lab-context templates referencing a specific month.
    Labels are the 12 month names in calendar order.

    Regression targets:
        - ``target_month``: ordinal integer 0–11 (January=0).
        - ``target_sin_12m``, ``target_cos_12m``: 12-month cyclical encoding.
        - ``target_season``: Northern-hemisphere season (Winter=0: Dec/Jan/Feb,
          Spring=1: Mar–May, Summer=2: Jun–Aug, Fall=3: Sep–Nov).

    Args:
        n_samples: Maximum number of unique sentences to return.

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, and the 4 regression target columns.
    """
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Northern-hemisphere seasons by month index (0=Jan)
    # Winter=0: Dec(11), Jan(0), Feb(1) | Spring=1: Mar–May | Summer=2: Jun–Aug | Fall=3: Sep–Nov
    _SEASON: dict[int, int] = {
        0: 0, 1: 0, 2: 1, 3: 1, 4: 1,
        5: 2, 6: 2, 7: 2,
        8: 3, 9: 3, 10: 3,
        11: 0,
    }

    templates = [
        "{name} will submit the annual report in {month}",
        "The grant deadline falls in {month}",
        "The conference is scheduled for {month}",
        "Field data collection begins in {month}",
        "{name} plans to defend the thesis in {month}",
        "The fiscal year closes at the end of {month}",
        "The cohort study enrollment opens in {month}",
        "{name} noted the equipment arrived in {month}",
        "The review board meets annually in {month}",
        "The lab renovation is planned for {month}",
        "Sample collection was completed in {month}",
        "{name} presented the interim findings in {month}",
        "The breeding season peaks in {month}",
        "The journal submission window opens in {month}",
        "The fellowship applications are due in {month}",
        "{name} recorded the highest yield in {month}",
        "The symposium takes place every {month}",
        "The pilot study wrapped up in {month}",
        "The funding cycle resets each {month}",
        "{name} returned from fieldwork in {month}",
        "The accreditation review is scheduled for {month}",
        "Animal migration peaks in {month}",
        "{name} confirmed the calibration was done in {month}",
        "The onboarding of new staff happens every {month}",
        "The protocol amendment was approved in {month}",
        "{name} noticed the anomaly first appeared in {month}",
        "The seasonal survey is conducted each {month}",
        "The lab will be closed for two weeks in {month}",
        "The reagent stock is replenished every {month}",
        "{name} finalized the analysis in {month}",
    ]

    n_with_name = sum(1 for t in templates if "{name}" in t)
    n_without = len(templates) - n_with_name
    max_unique = n_with_name * len(names) * len(months) + n_without * len(months)
    print(
        f"Months: {len(templates)} templates × {len(names)} names × "
        f"{len(months)} months → {max_unique} max unique sentences"
    )

    df = _enumerate_unique(templates, months, ("month",), n_samples)

    # ── Regression targets ────────────────────────────────────────────
    month_to_idx = {m: i for i, m in enumerate(months)}
    month_vals = df["Label"].apply(lambda lbl: month_to_idx[lbl.split("_", 1)[1]]).astype(int)
    df["target_month"] = month_vals
    df["target_sin_12m"] = np.sin(2 * np.pi * month_vals / 12)
    df["target_cos_12m"] = np.cos(2 * np.pi * month_vals / 12)
    df["target_season"] = month_vals.apply(lambda m: _SEASON[m]).astype(int)

    return df


# ── Valence/Arousal lookup for GoEmotions labels ─────────────────────────────
# Derived from the NRC Valence-Arousal-Dominance Lexicon (Mohammad 2018) and
# Warriner et al. (2013) word norms, cross-checked against Russell's circumplex
# model.  Scores are approximations; no authoritative per-label mapping exists
# for the full 28-class GoEmotions taxonomy.
EMOTION_VA: dict[str, tuple[float, float]] = {
    "admiration":     ( 0.70,  0.40),
    "amusement":      ( 0.80,  0.50),
    "anger":          (-0.80,  0.80),
    "annoyance":      (-0.50,  0.40),
    "approval":       ( 0.60,  0.20),
    "caring":         ( 0.75,  0.30),
    "confusion":      (-0.20,  0.40),
    "curiosity":      ( 0.40,  0.60),
    "desire":         ( 0.60,  0.70),
    "disappointment": (-0.65, -0.20),
    "disapproval":    (-0.55,  0.30),
    "disgust":        (-0.80,  0.45),
    "embarrassment":  (-0.55,  0.45),
    "excitement":     ( 0.80,  0.90),
    "fear":           (-0.80,  0.85),
    "gratitude":      ( 0.85,  0.35),
    "grief":          (-0.90, -0.10),
    "joy":            ( 0.90,  0.70),
    "love":           ( 0.90,  0.60),
    "nervousness":    (-0.45,  0.75),
    "optimism":       ( 0.75,  0.50),
    "pride":          ( 0.70,  0.55),
    "realization":    ( 0.20,  0.40),
    "relief":         ( 0.70, -0.20),
    "remorse":        (-0.70, -0.10),
    "sadness":        (-0.80, -0.30),
    "surprise":       ( 0.15,  0.80),
    "neutral":        ( 0.00,  0.00),
}


def generate_emotions() -> pd.DataFrame:
    """Load go_emotions validation split, retaining all examples.

    Multi-label examples have one label chosen at random so that rare emotion
    classes are better represented, improving regression probe stability.

    Regression targets:
        - ``target_valence``: valence score in [−1, 1].
        - ``target_arousal``: arousal score in [−1, 1].
        - ``target_quadrant``: affective quadrant (0: V≥0 A≥0, 1: V<0 A≥0,
          2: V<0 A<0, 3: V≥0 A<0).

    Returns:
        DataFrame with columns ``Sentence``, ``Label``, ``target_valence``,
        ``target_arousal``, ``target_quadrant``.
    """
    from datasets import load_dataset as hf_load_dataset

    emotions_path = Path(__file__).parent.parent.parent / "datasets" / "probing" / "emotions.txt"
    emotion_names = emotions_path.read_text().strip().splitlines()

    ds = hf_load_dataset("google-research-datasets/go_emotions", split="validation")

    rows = []
    for example in ds:
        lbls = example["labels"]
        if not lbls:
            continue
        chosen_idx = random.choice(lbls)
        label = emotion_names[chosen_idx]
        v, a = EMOTION_VA.get(label, (0.0, 0.0))
        # Quadrant: 0=(V≥0,A≥0), 1=(V<0,A≥0), 2=(V<0,A<0), 3=(V≥0,A<0)
        quadrant = (0 if v >= 0 else 1) if a >= 0 else (3 if v >= 0 else 2)
        rows.append({
            "Sentence": example["text"],
            "Label": label,
            "target_valence": v,
            "target_arousal": a,
            "target_quadrant": quadrant,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["Sentence"])
    rows_list = df.to_dict("records")
    random.shuffle(rows_list)
    df = pd.DataFrame(rows_list)
    print(
        f"Emotions: {len(df)} examples from go_emotions validation split "
        f"(multi-label included, one label sampled per example)"
    )
    return df


app = typer.Typer()


@app.command()
def generate(
    output_dir: str = typer.Option("datasets/probing", help="Directory to write CSV files into."),
):
    """Generate all probing datasets and write them to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("weekdays.csv", generate_weekdays(1000)),
        ("hours.csv", generate_hours(1500)),
        ("temperatures.csv", generate_temperatures(1000)),
        ("time_units.csv", generate_time_units(1000)),
        ("body_parts.csv", generate_body_parts(1000)),
        ("living_things.csv", generate_living_things(1000)),
        ("months.csv", generate_months(1000)),
        ("colors.csv", generate_colors(1000)),
        ("emotions.csv", generate_emotions()),
    ]

    for filename, df in datasets:
        path = out / filename
        df.to_csv(path, index=False)
        print(f"Wrote {len(df):>5} samples → {path}")


if __name__ == "__main__":
    app()
