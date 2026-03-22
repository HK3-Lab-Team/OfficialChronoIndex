import matplotlib.pyplot as plt
from datetime import timedelta
from chronoindex.glucose_series_processing.cgm_checking_functions import _safe_entries

def plot_glucose_visits(subject, when="Before Deduplication"):
    """
    Plot glucose values for one subject:
      - x axis: calendar datetimes
      - y axis: glucose values
      - Visit 2 CGM in black (filled)
      - Visit 3 CGM in hollow red
      - Vertical lines marking Visit2Date and Visit3Date
      - X axis from earliest CGM/visit date to latest (+1 day padding)
    """

    # Extract CGM entries
    v2_entries = _safe_entries(subject, "v2")
    v3_entries = _safe_entries(subject, "v3")

    # Extract full timestamps & glucose
    v2_times = [e.deviceTimestamp for e in v2_entries if e.deviceTimestamp]
    v2_glucose = [e.historicGlucoseMmolL for e in v2_entries if e.deviceTimestamp]

    v3_times = [e.deviceTimestamp for e in v3_entries if e.deviceTimestamp]
    v3_glucose = [e.historicGlucoseMmolL for e in v3_entries if e.deviceTimestamp]

    if not v2_times and not v3_times:
        print("No CGM data available for this subject")
        return

    # Visit dates (convert to date objects for axis calculation)
    try:
        v2_date = getattr(subject.visit2Form.content, "Visit2Date", None)
    except AttributeError:
        v2_date = None
    try:
        v3_date = getattr(subject.visit3Form.content, "Visit3Date", None)
    except AttributeError:
        v3_date = None
    v2_date = v2_date.date() if v2_date else None
    v3_date = v3_date.date() if v3_date else None

    # Collect candidate min/max dates
    candidate_dates = []
    if v2_times:
        candidate_dates.extend([min(v2_times).date(), max(v2_times).date()])
    if v3_times:
        candidate_dates.extend([min(v3_times).date(), max(v3_times).date()])
    if v2_date:
        candidate_dates.append(v2_date)
    if v3_date:
        candidate_dates.append(v3_date)

    min_date, max_date = min(candidate_dates), max(candidate_dates)

    # ---- Plot ----
    plt.figure(figsize=(12, 6))

    if v2_times:
        plt.scatter(v2_times, v2_glucose, color="black", s=15, label="Visit 2 CGM")

    if v3_times:
        plt.scatter(v3_times, v3_glucose, facecolors="none", edgecolors="red", s=30, label="Visit 3 CGM")

    # Vertical markers for visit dates
    if v2_date:
        plt.axvline(v2_date, color="blue", linestyle="--", label="Visit 2 Date")
    if v3_date:
        plt.axvline(v3_date, color="green", linestyle="--", label="Visit 3 Date")

    plt.xlabel("Date")
    plt.ylabel("Glucose (mmol/L)")
    plt.title(f"Subject {getattr(subject, 'participantId', '<unknown>')}: CGM Data by Visit {when}")
    plt.legend()
    plt.xlim(min_date - timedelta(days=1), max_date + timedelta(days=1))  # padding for visibility
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
