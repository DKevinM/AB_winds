"""
How far off a model concentration could be ("Cfactor", Kevin 2026-10-31), from the validation residuals.

A correction factor was tried first and failed blind cross-testing (fitted on one field experiment it made
the others worse: Prairie Grass and Flesch 2004 want opposite corrections in unstable air, and Project
Sagebrush nights swing from 5x high to 4x low night to night). What the residuals do give reliably is the
SPREAD: for each stability class, the factor that 68% of observed/model ratios fall within, pooled over
Prairie Grass (1956), Flesch et al. (2004) and Project Sagebrush (2013, 2016), live model 2026-10-31.

Classes use 1/L (m^-1) with Flesch et al.'s +/-0.02 neutral edge; u* < 0.05 m/s is "calm night"
(the very_stable_caveat regime) whatever L says. A run spanning several hours takes its least certain hour.
"""

import math

# (class key: plain name, 68% factor, n residuals behind it, factor as shown to readers)
CLASSES = {
    "neutral":       ("windy or overcast weather", 1.8, 121, 2),
    "unstable":      ("partly sunny weather", 2.7, 167, 3),
    "very_unstable": ("strong sun with light wind", 4.4, 119, 4),
    "stable":        ("stable evening air", 3.5, 48, 4),
    "very_stable":   ("very stable night air", 4.3, 44, 4),
    "calm_night":    ("calm, stable air", 5.5, 73, 5),
}
CALM_USTAR = 0.05


def classify(L, ustar):
    if L > 0 and ustar < CALM_USTAR:
        return "calm_night"
    inv = 1.0 / L if L else 0.0
    if inv < -0.1:
        return "very_unstable"
    if inv < -0.02:
        return "unstable"
    if inv <= 0.02:
        return "neutral"
    return "stable" if inv <= 0.1 else "very_stable"


def describe(hourly_met):
    """hourly_met: gaussian_plume.hourly_met rows (need "L", "ustar"). Returns the model_info["uncertainty"] dict."""
    rows = [h for h in hourly_met if math.isfinite(h.get("L", float("nan"))) and math.isfinite(h.get("ustar", float("nan")))]
    if not rows:
        return None
    hours = {}
    for h in rows:
        k = classify(h["L"], h["ustar"])
        hours[k] = hours.get(k, 0) + 1
    worst = max(hours, key=lambda k: CLASSES[k][1])
    name, f, n, rounded = CLASSES[worst]
    text = (f"Rough guide only. In {name} like this the model has been up to about {rounded}\u00d7 too high or too low in field tests."
            if worst == "calm_night" else
            f"Likely within about {rounded}\u00d7 of this value. Based on field tests in {name}.")
    return {"class": worst, "factor_68pct": f, "factor_rounded": rounded, "hours_by_class": hours,
            "basis": f"68% of {n} field-test observed/model ratios in this class fall within x/÷{f}",
            "text": text}
