from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Slider, TextBox


def compute_montant_cir(*, employeur_c_an: float) -> float:
    """Compute 'Crédit CIR + CIFRE' from other quantities.

    Formula:
        Crédit CIR + CIFRE = (Chargé/an * 1.4 - 14000) * 0.3

    Args:
        employeur_c_an: Chargé/an value.

    Returns:
        Crédit CIR + CIFRE value.
    """
    return float((employeur_c_an * 1.4 - 14000) * 0.3)


def load_table() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load the table and return (x, series) where x is 'Perso B/an'.

    Notes:
        - 'Employeur CR/3ans' removed entirely.
        - 'Payé Réel/an' is NOT provided as data; it is derived as:
              Payé Réel/an = Chargé/an - Crédit CIR + CIFRE
        - 'Crédit CIR + CIFRE' is computed from 'Chargé/an' via compute_montant_cir().

    Returns:
        A tuple (perso, series):
            - perso: shape (n,), the x-axis values (Salaire Brut/an), sorted ascending.
            - series: mapping from column name to y-values (shape (n,)).
    """
    data = np.array(
        [
            [33363, 27600, 21538, 20871],
            [35436, 29328, 22903, 22194],
            [38041, 31500, 24619, 23543],
            [39840, 33000, 25804, 24676],
            [42239, 35000, 27383, 25837],
            [44638, 37000, 28963, 27329],
        ],
        dtype=float,
    )

    employeur_c_an = data[:, 0]
    perso = data[:, 1]
    apres_cotis = data[:, 2]
    apres_ir = data[:, 3]

    montant_cir = np.array(
        [compute_montant_cir(employeur_c_an=a) for a in employeur_c_an],
        dtype=float,
    )
    employeur_cr_an = employeur_c_an - montant_cir

    sort_idx = np.argsort(perso)
    perso = perso[sort_idx]

    series: Dict[str, np.ndarray] = {
        "Chargé/an": employeur_c_an[sort_idx],
        "Crédit CIR/CIFRE": montant_cir[sort_idx],
        "Payé Réel/an": employeur_cr_an[sort_idx],
        "Net Cotis./an": apres_cotis[sort_idx],
        "Net Impôt/an": apres_ir[sort_idx],
    }
    return perso, series


def interp_clamped(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Linear interpolation, clamped to [xp[0], xp[-1]].

    Args:
        x: Query x value.
        xp: Known x samples (ascending).
        fp: Known y samples.

    Returns:
        Interpolated y value.
    """
    return float(np.interp(x, xp, fp))


def _fmt_euro(val: float) -> str:
    """Format a number as euros with space grouping."""
    return f"{val:,.0f}".replace(",", " ")


def compute_portfolio_path(
    *,
    annual_contribution: float,
    annual_return: float,
    years: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute portfolio value over time with monthly contributions.

    Assumptions:
        - Monthly compounding at rate (1+annual_return)^(1/12)-1.
        - Contributions are made at end of each month.
        - annual_contribution < 0 is clamped to 0.

    Args:
        annual_contribution: Amount invested per year (€/year).
        annual_return: Nominal annual return (e.g., 0.07 for 7%).
        years: Horizon in years.

    Returns:
        (t_years, values): time points in years and portfolio values in euros.
    """
    annual_contribution = float(max(0.0, annual_contribution))
    years = int(years)

    if years <= 0:
        return np.array([0.0]), np.array([0.0])

    months = years * 12
    r_m = (1.0 + float(annual_return)) ** (1.0 / 12.0) - 1.0
    c_m = annual_contribution / 12.0

    v = 0.0
    values = np.zeros(months + 1, dtype=float)
    for m in range(1, months + 1):
        v = v * (1.0 + r_m) + c_m
        values[m] = v

    t = np.arange(months + 1, dtype=float) / 12.0
    return t, values


def build_interactive_plot(perso: np.ndarray, series: Dict[str, np.ndarray]) -> None:
    """Interactive figure:
    - Top: income series vs Salaire Brut/an with slider + textbox
    - Bottom: invested wealth over time (7%/y) + time slider cursor
    """
    monthly_spend = 1698.0
    annual_spend = 12.0 * monthly_spend
    annual_return = 0.07
    horizon_years = 30

    perso_min = float(perso.min())
    perso_max = float(perso.max())
    x0 = float(perso[0])

    names = list(series.keys())

    fig = plt.figure(figsize=(13.5, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.0, 1.25],
        width_ratios=[3.3, 1.4],
        wspace=0.08,
        hspace=0.18,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_vals = fig.add_subplot(gs[0, 1])
    ax_vals.axis("off")

    ax_inv = fig.add_subplot(gs[1, 0])
    ax_inv_vals = fig.add_subplot(gs[1, 1])
    ax_inv_vals.axis("off")

    # --- Top plot ---
    marks: Dict[str, Any] = {}
    for name in names:
        ax.plot(perso, series[name], label=name, linewidth=2.0, alpha=0.95)
        (mk,) = ax.plot(
            [x0],
            [interp_clamped(x0, perso, series[name])],
            marker="o",
            linestyle="None",
            markersize=6,
        )
        marks[name] = mk

    vline = ax.axvline(x0, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Salaire Brut/an (x)")
    ax.set_ylabel("€ / an")
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: _fmt_euro(float(y))))

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.8,
        columnspacing=1.2,
    )

    text_obj = ax_vals.text(
        0.0,
        1.0,
        "",
        va="top",
        family="monospace",
        fontsize=10,
    )

    # --- Bottom plot (investment) ---
    (inv_line,) = ax_inv.plot([], [], linewidth=2.0)
    inv_vline = ax_inv.axvline(0.0, linestyle="--", linewidth=1.2, alpha=0.8)
    (inv_marker,) = ax_inv.plot([0.0], [0.0], marker="o", linestyle="None", markersize=6)

    ax_inv.set_xlabel("Années")
    ax_inv.set_ylabel("Portefeuille (€)")
    ax_inv.grid(True, alpha=0.25)
    ax_inv.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: _fmt_euro(float(y))))

    inv_text_obj = ax_inv_vals.text(
        0.0,
        1.0,
        "",
        va="top",
        family="monospace",
        fontsize=10,
    )

    # Make room for 2 sliders + textbox
    fig.subplots_adjust(bottom=0.20)

    # --- Controls layout (fix overlap) ---
    left = 0.08
    right = 0.97
    gap = 0.015            # increased gap between slider and textbox
    textbox_w = 0.04      # slightly narrower textbox to be safe
    slider_h = 0.05
    slider_y = 0.10
    textbox_h = 0.04
    textbox_y = 0.105

    slider_w = (right - left) - gap - textbox_w

    # Slider 1: Salaire Brut/an (top interactions)
    ax_slider = fig.add_axes([left, slider_y, slider_w, slider_h])
    slider = Slider(
        ax=ax_slider,
        label="Salaire Brut/an",
        valmin=perso_min,
        valmax=perso_max,
        valinit=x0,
        valstep=1.0,
    )

    ax_textbox = fig.add_axes([left + slider_w + gap, textbox_y, textbox_w, textbox_h])
    textbox = TextBox(ax_textbox, label="", initial=f"{x0:.0f}")

    # Slider 2: time cursor for bottom plot
    ax_tslider = fig.add_axes([left, 0.03, slider_w, slider_h])
    t_slider = Slider(
        ax=ax_tslider,
        label="Années",
        valmin=0.0,
        valmax=float(horizon_years),
        valinit=0.0,
        valstep=1.0 / 12.0,  # monthly resolution
    )

    fig.suptitle("Revenus CIFRE et trajectoire d’investissement", fontsize=14, fontweight="bold")
    fig.canvas.manager.set_window_title("Revenus & investissement")

    state = {"busy": False}

    inv_state: Dict[str, Any] = {
        "t": np.array([0.0], dtype=float),
        "v": np.array([0.0], dtype=float),
    }

    def format_values(x: float) -> str:
        vals = {name: interp_clamped(x, perso, y) for name, y in series.items()}
        out = [f"{'Salaire Brut/an':<16}: {_fmt_euro(x)}", "-" * 32]
        for name in names:
            out.append(f"{name:<16}: {_fmt_euro(vals[name])}")
        return "\n".join(out)

    def format_investment_values(
        *,
        apres_ir: float,
        t_pick: float,
        v_pick: float,
    ) -> str:
        annual_savings = max(0.0, float(apres_ir) - annual_spend)
        monthly_savings = annual_savings / 12.0

        v = inv_state["v"]
        v_end = float(v[-1])
        v_10 = float(v[min(len(v) - 1, 10 * 12)])
        v_20 = float(v[min(len(v) - 1, 20 * 12)])

        out = [
            f"{'t (années)':<16}: {t_pick:>6.2f}",
            f"{'Valeur à t':<16}: {_fmt_euro(v_pick)}",
            "-" * 32,
            f"{'Dépenses/mois':<16}: {_fmt_euro(monthly_spend)}",
            f"{'Dépenses/an':<16}: {_fmt_euro(annual_spend)}",
            "-" * 32,
            f"{'Épargne/an':<16}: {_fmt_euro(annual_savings)}",
            f"{'Épargne/mois':<16}: {_fmt_euro(monthly_savings)}",
            "-" * 32,
            f"{'r annuel':<16}: {annual_return*100:.2f}%",
            f"{'Horizon':<16}: {horizon_years} ans",
            "-" * 32,
            f"{'Portefeuille 10a':<16}: {_fmt_euro(v_10)}",
            f"{'Portefeuille 20a':<16}: {_fmt_euro(v_20)}",
            f"{'Portefeuille fin':<16}: {_fmt_euro(v_end)}",
        ]
        return "\n".join(out)

    def update_peek_cursor(t_pick: float) -> None:
        t_arr = inv_state["t"]
        v_arr = inv_state["v"]

        t_pick = float(np.clip(t_pick, float(t_arr[0]), float(t_arr[-1])))
        v_pick = float(np.interp(t_pick, t_arr, v_arr))

        inv_vline.set_xdata([t_pick, t_pick])
        inv_marker.set_data([t_pick], [v_pick])

        # Update bottom panel (needs current apres_ir from x slider)
        x = float(slider.val)
        apres_ir = interp_clamped(x, perso, series["Net Impôt/an"])
        inv_text_obj.set_text(format_investment_values(apres_ir=apres_ir, t_pick=t_pick, v_pick=v_pick))

    def update_investment_curve(x: float) -> None:
        apres_ir = interp_clamped(x, perso, series["Net Impôt/an"])
        annual_savings = max(0.0, float(apres_ir) - annual_spend)

        t, v = compute_portfolio_path(
            annual_contribution=annual_savings,
            annual_return=annual_return,
            years=horizon_years,
        )
        inv_state["t"] = t
        inv_state["v"] = v

        inv_line.set_data(t, v)
        ax_inv.relim()
        ax_inv.autoscale_view()

        # Keep peek cursor where user left it (clamped)
        update_peek_cursor(float(t_slider.val))

    def update_top(x: float) -> None:
        for name, y in series.items():
            yx = interp_clamped(x, perso, y)
            marks[name].set_data([x], [yx])
        vline.set_xdata([x, x])
        text_obj.set_text(format_values(x))

    def update_all(x: float) -> None:
        update_top(x)
        update_investment_curve(x)
        fig.canvas.draw_idle()

    def on_slider(val: float) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        try:
            x = float(val)
            textbox.set_val(f"{x:.0f}")
            update_all(x)
        finally:
            state["busy"] = False

    def on_text(text: str) -> None:
        if state["busy"]:
            return
        try:
            x = float(text)
        except ValueError:
            return
        x = float(np.clip(x, perso_min, perso_max))

        state["busy"] = True
        try:
            slider.set_val(x)
            update_all(x)
        finally:
            state["busy"] = False

    def on_tslider(val: float) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        try:
            update_peek_cursor(float(val))
            fig.canvas.draw_idle()
        finally:
            state["busy"] = False

    slider.on_changed(on_slider)
    textbox.on_submit(on_text)
    t_slider.on_changed(on_tslider)

    # Initial draw
    update_all(x0)
    plt.show()


def main() -> None:
    """Entry point."""
    perso, series = load_table()
    build_interactive_plot(perso, series)


if __name__ == "__main__":
    main()
