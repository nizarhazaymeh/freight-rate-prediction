"""Build reports/freight_rate_report.pdf from the backtest metrics and figures.

Usage: python -m src.report   (run after src.eda, src.train, src.predict and score.py)
"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import config

AUTHOR = ""  # add your name here; omitted from the title block when empty
OUTPUT = config.REPORTS_DIR / "freight_rate_report.pdf"
CHART = config.ROOT / "scorer_results" / "candidate_december.png"

INK = colors.HexColor("#064A56")
MUTED = colors.HexColor("#455A60")
RULE = colors.HexColor("#D9E2E4")
TINT = colors.HexColor("#EEF3F4")

MODEL_LABELS = {
    "baseline_rate_per_mile": "Rate-per-mile baseline",
    "baseline_plain_gbm": "Plain gradient boosting",
    "stage1_linear_only": "Stage 1 only (log-linear + trend)",
    "full_model": "Full model",
}

base = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=base["Title"], fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=14)
H2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=13.5, leading=17, textColor=INK, spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=base["Normal"], fontSize=9.6, leading=13.4, spaceAfter=6)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.4, leading=11, textColor=MUTED, spaceAfter=10)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.6, leading=11, spaceAfter=0)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths, header=True, bold_last=False):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELL_B if (header and i == 0) else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    if bold_last:
        style += [("BACKGROUND", (0, -1), (-1, -1), TINT)]
    t.setStyle(TableStyle(style))
    return t


def figure(path, width, caption):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * h / w), p(caption, CAPTION)])


def fmt_fold(name):
    # "train<2025-07-01 | test 2025-07-01..2025-09-01"
    train_part, test_part = name.split(" | ")
    start, end = test_part.replace("test ", "").split("..")
    return train_part.replace("train<", ""), start, end


def build():
    metrics = json.loads(config.METRICS_PATH.read_text())
    folds, mean = metrics["folds"], metrics["mean"]
    full = mean["full_model"]["clean_rows"]

    s = []
    s.append(p("Freight Rate Prediction", H1))
    s.append(p(("Machine Learning Engineer assessment" + (f" &middot; {AUTHOR}" if AUTHOR else "")), SUB))

    # ---------------------------------------------------------------- summary
    s.append(p("Summary", H2))
    s.append(p(
        "The model predicts the posted rate of a load from its lane, distance, equipment, weight, date and the "
        "two market signals. It is validated with a <b>time-based, rolling-origin backtest</b>: each fold trains on "
        "everything before a cutoff and predicts the next two months, which mirrors the real task (train on "
        "Jan-Oct 2025, predict Nov-Dec 2025). Averaged over three folds, the final model's error is "
        f"<b>{full['MAPE_%']:.2f}% MAPE</b> (MAE ${full['MAE']:.0f}) on correctly labeled loads, against "
        f"{mean['baseline_plain_gbm']['clean_rows']['MAPE_%']:.2f}% for off-the-shelf gradient boosting and "
        f"{mean['baseline_rate_per_mile']['clean_rows']['MAPE_%']:.2f}% for a rate-per-mile baseline."
    ))

    # ------------------------------------------------------------ data split
    s.append(p("1. Train/test split and validation approach", H2))
    s.append(p(
        "<b>Why not a random split.</b> The labeled data covers 1 Jan to 31 Oct 2025, and the loads to predict "
        "cover 1 Nov to 31 Dec 2025, entirely in the future. Rates depend on a daily market level and on a slow "
        "upward drift, so a random split would put loads from the same days and weeks in both train and test. "
        "That leaks the market conditions of the test period into training and overstates accuracy. The split "
        "therefore follows time."
    ))
    s.append(p(
        "<b>Rolling-origin backtest.</b> Three folds, each with a two-month test window (the same horizon as the "
        "real validation set) and an expanding training window:"
    ))
    rows = [["Fold", "Train on", "Test on", "Train rows", "Test rows"]]
    for i, (name, res) in enumerate(folds.items(), 1):
        cutoff, start, end = fmt_fold(name)
        n_test = res["full_model"]["all_rows"]["n"]
        rows.append([str(i), f"2025-01-01 to {cutoff} (exclusive)", f"{start} to {end} (exclusive)",
                     f"{res['train_rows']:,}", f"{n_test:,}"])
    s.append(table(rows, [0.45 * inch, 2.2 * inch, 2.2 * inch, 0.85 * inch, 0.8 * inch]))
    s.append(Spacer(1, 8))
    s.append(p("<b>Leakage controls</b>, applied inside every fold:"))
    s += bullets([
        "Everything learned from labels is fitted on the training window only: the corrupted-label filter, the "
        "log-linear coefficients and time trend, the lane encodings and the boosted trees.",
        "Lane encodings in the training window are computed <b>out-of-fold</b> (5-fold), so a row never sees its own "
        "label through its lane average.",
        "The test window's <i>unlabeled</i> features may be used to estimate each day's market level (the daily mean of "
        "market_index). This matches the real setting, where validation.csv provides market_index for every "
        "Nov-Dec load. Test labels are never used.",
        "After model selection, the final model is refitted on all of Jan-Oct with the same pipeline and "
        "hyperparameters, and then predicts validation.csv once.",
    ])
    s.append(p(
        "<b>Metrics.</b> MAPE is reported as the headline metric because rates range from about $60 to $25,000 and a "
        "relative error is comparable across short and long hauls. MAE and RMSE are reported alongside it. Test "
        "labels flagged as corrupted (about 1.4% per fold, see section 2) are scored separately, because they "
        "reflect data-entry errors rather than model quality. Metrics on all rows are in the table in section 4."
    ))

    # ---------------------------------------------------------- data quality
    s.append(CondPageBreak(2.6 * inch))
    s.append(p("2. Data validation and quality issues", H2))
    dq = [
        ["Issue", "Evidence", "Handling"],
        ["Corrupted rates (667 rows, 1.4%)",
         "Rates at 0.2-0.5x or 2.5-4.6x the lane-equipment median, separated from genuine noise (~5%) by an "
         "almost empty gap. Occurrence is flat across equipment, month, weekday, distance and city, so the "
         "errors cannot be predicted.",
         "Rows more than 1.6x from the structural model's estimate are excluded from training."],
        ["Negative weights (292 train, 145 validation)",
         "Absolute values follow the same distribution as positive weights (median 31.8k vs 31.4k lb).",
         "Treated as sign flips: abs()."],
        ["Missing weight (300 / 165)", "Under 1% of rows.", "Median fill plus a missing-value flag."],
        ["Missing market_index (374 / 249)",
         "market_index is a daily factor: std 0.025 within a day vs 0.167 between days.",
         "Filled with that day's mean market_index."],
        ["8 unseen cities in validation (1,447 rows)",
         "Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego.",
         "Geography enters through coordinates, which generalise to new cities. City IDs are never "
         "used directly."],
        ["Synthetic coordinates", "Coordinates are shifted from real-world values but consistent per city.",
         "Used as relative features only; recorded distance is used for pricing."],
    ]
    s.append(table(dq, [1.45 * inch, 3.05 * inch, 2.0 * inch]))
    s.append(Spacer(1, 8))
    s.append(figure(config.FIGURES_DIR / "label_ratio_hist.png", 5.0 * inch,
                    "Figure 1. Ratio of each rate to its lane-equipment median (log scale). Dashed lines mark the "
                    "1.6x filter. The outer clusters are corrupted labels."))

    # --------------------------------------------------------- key findings
    s.append(p("3. Key findings and model", H2))
    s += bullets([
        "<b>Distance dominates.</b> log(rate) vs log(distance) correlation 0.97, with a taper: rate per mile falls "
        "on longer hauls.",
        "<b>market_index is the daily market level.</b> A log-linear fit gives an elasticity of about 0.15, and it has "
        "a strong weekly cycle that peaks on Thursdays.",
        "<b>Upward drift of about 0.6% a month</b> that market_index does not explain (Figure 2: from August to October, "
        "rate per mile rises while market_index falls).",
        "<b>Equipment and weight:</b> Reefer about +12% and Flatbed about +8% over Dry Van; heavier loads cost up to about 6% more.",
        "<b>Stable lane premiums of about 5%.</b> Lane-equipment residuals correlate 0.72 between the first and second "
        "half of the year, and 88% of validation loads run on a lane seen in training.",
    ])
    s.append(figure(config.FIGURES_DIR / "market_index_vs_rpm.png", 5.6 * inch,
                    "Figure 2. Median rate per mile and market_index (7-day averages). The shaded area is the "
                    "prediction period."))
    s.append(p(
        "<b>Model.</b> The model has three stages, all fitted on log(rate), so errors are relative and the "
        "pricing structure is multiplicative:"
    ))
    s += bullets([
        "<b>Stage 1, structural log-linear model:</b> log distance (and its square), equipment, log market_index, "
        "log weight, quote_signal and a <b>linear time trend</b>. This is the only stage that sees time, so the drift "
        "extrapolates into Nov-Dec. Tree models cannot extrapolate and would freeze the level at October.",
        "<b>Stage 2, lane encodings:</b> shrunken (k=5), out-of-fold means of the stage-1 residual by lane x equipment, "
        "lane, pickup/delivery x equipment and city, plus the support counts.",
        "<b>Stage 3, gradient boosting</b> (scikit-learn HistGradientBoosting) on the stage-1 residual, using distance, "
        "equipment, weight, market signals, coordinates, circuity, weekday and the lane encodings. "
        "Final prediction = exp(stage 1 + stage 3).",
    ])

    # --------------------------------------------------------------- results
    s.append(CondPageBreak(2.4 * inch))
    s.append(p("4. Backtest results", H2))
    res = [["Model", "Fold 1<br/>Jul-Aug", "Fold 2<br/>Aug-Sep", "Fold 3<br/>Sep-Oct", "Mean MAPE", "Mean MAE",
            "Mean RMSE", "MAPE, all rows"]]
    for key, label in MODEL_LABELS.items():
        per_fold = [f"{folds[f][key]['clean_rows']['MAPE_%']:.2f}%" for f in folds]
        m = mean[key]
        name = f"<b>{label}</b>" if key == "full_model" else label
        res.append([name, *per_fold, f"{m['clean_rows']['MAPE_%']:.2f}%", f"${m['clean_rows']['MAE']:.0f}",
                    f"${m['clean_rows']['RMSE']:.0f}", f"{m['all_rows']['MAPE_%']:.2f}%"])
    s.append(table(res, [1.6 * inch, 0.68 * inch, 0.68 * inch, 0.68 * inch, 0.72 * inch, 0.62 * inch,
                         0.68 * inch, 0.86 * inch], bold_last=True))
    s.append(p("Per-fold MAPE and mean metrics on correctly labeled test loads. The last column includes the "
               "corrupted labels.", CAPTION))
    s += bullets([
        "Each stage adds accuracy: the lane encodings and trees cut stage-1 error by about a quarter, and the full "
        "model beats plain gradient boosting by about a third on average and by half on the most recent fold.",
        "Fold 3 is closest to the final setting (most training history, test right after) and reaches 1.73% MAPE. "
        "That is near the noise floor: residual noise within a lane-equipment pair is about 2.4% (std of log rate).",
        "Fold 1 is the weakest. With only Jan-Jun to learn from, market_index and time rise together, so the trend "
        "is overestimated (1.07% a month vs 0.53-0.66% in later folds and in the final fit). With the full "
        "Jan-Oct history, market_index rises and falls, which separates the two effects.",
        "On all rows, every model's RMSE is about $620 because the corrupted labels dominate it. This is the "
        "expected level for the Nov-Dec scoring if those labels contain a similar share of errors.",
    ])

    # --------------------------------------------------------------- december
    s.append(p("5. December prediction chart", H2))
    s.append(p(
        "The chart below was produced by the provided score.py. Lane, distance, equipment and weight are fixed, "
        "and only the date changes. Because december_chart_inputs.csv has no coordinates or market signals, the "
        "coordinates come from the city lookup, and market_index and quote_signal take each day's mean and median "
        "across the December loads in validation.csv. Predictions range from $837 to $858. That is consistent with "
        "the Sep-Oct Lexington to Fort Wayne Dry Van loads in the training data ($778-$865 for 351-371 miles). They "
        "follow the daily market level (correlation 0.94 with that day's market_index, including its Thursday "
        "peaks) plus a slight upward drift from the trend."
    ))
    s.append(figure(CHART, 6.5 * inch, "Figure 3. Fixed December prediction chart produced by score.py."))

    s.append(p("6. Limitations", H2))
    s += bullets([
        "No training data covers Thanksgiving or Christmas. Holidays in the training period showed no measurable "
        "effect, so none is modeled, but late-December demand shifts could be missed.",
        "The linear trend is extrapolated one to two months ahead. A longer horizon would need a trend estimate "
        "that is re-fitted as new data arrives.",
        "Corrupted labels are excluded from training rather than corrected. If the scoring labels contain them, "
        "they set a floor on all-rows error that no model can remove.",
    ])

    s.append(p("Reproducing", H2))
    s.append(p(
        "<font face='Courier' size='8.6'>pip install -r requirements.txt; python -m src.eda; python -m src.train; "
        "python -m src.predict; python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_chart_inputs.csv; python -m src.report</font>"
    ))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.85 * inch, 0.55 * inch, "Freight Rate Prediction")
        canvas.drawRightString(letter[0] - 0.85 * inch, 0.55 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.8 * inch, bottomMargin=0.85 * inch, title="Freight Rate Prediction",
        author=AUTHOR or None,
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUTPUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    build()
