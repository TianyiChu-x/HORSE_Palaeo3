import numpy as np
import pandas as pd
from numba import jit
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, linregress
from utils import (
    find_fad_lad,
    prepare_misfit,
    compute_misfit,
    compute_penalty,
    build_event_seq,
)
import matplotlib as mpl
import warnings


# Set global font size
plt.rcParams.update({'font.size': 8})
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

warnings.filterwarnings("ignore")


@jit(nopython=True)
def _kruskals2(array1, array2):
    # Create distance matrices
    dist_matrix_1 = np.abs(array1[:, None] - array1)
    dist_matrix_2 = np.abs(array2[:, None] - array2)

    # Calculate squared stress
    numerator = np.sum((dist_matrix_1 - dist_matrix_2) ** 2)
    denominator = np.sum(np.sqrt((dist_matrix_1**2) * (dist_matrix_2**2)))

    return numerator / denominator


def kruskals2(arr1, arr2):
    """
    Calculate the modified squared stress between two one-dimensional arrays.

    Parameters
    ----------
    arr1 : np.ndarray,
            The first array, representing the actual distances.
    arr2 : np.ndarray,
            The second array, representing the distances in the MDS model.

    Returns
    ----------
    stress : float,
                The squared stress value.
    """

    return _kruskals2(np.array(arr1), np.array(arr2))


def draw_linear_regression(array1, array2, ax, color="black", linestyle="-"):
    slope, intercept, r_value, p_value, std_err = linregress(array1, array2)
    line = slope * array1 + intercept
    ax.plot(array1, line, color=color, alpha=0.5, linestyle=linestyle)
    # ax.text(0.05, 0.95, f'p-value: {p_value:.4e}', transform=axes[0].transAxes)


def compare(
    ha_df,
    sequence1,
    sequence2,
    compare_type,
    conop_misfit_mode="order",
    teaser_rate=0.0,
    original_state=False,
):
    """
    Compare sequence data.
    :param teaser_rate: Teaser rate.
    :param original_state: Whether to include original state in result.
    :param ha_df: HORSE DataFrame.
    :param sequence2: Computed sequence, each row is step followed by horizon or event name sequences.
    :param sequence1: True data. Can be HORSE or CONOP.
    :param compare_type: {'horse-horse', 'horse-conop', 'conop-conop', 'conop-horse'}.
    :param conop_misfit_mode: {'order', 'horizon', 'depth'}. Default 'order'.
    :return: result DataFrame
    """
    ha_df = ha_df.sort_values("Horizon").reset_index(drop=True)
    first_taxon_column_index = list(ha_df.columns).index("*") + 1
    taxa_names = ha_df.iloc[:, first_taxon_column_index:].columns.to_list()
    mat = ha_df.iloc[:, first_taxon_column_index:].to_numpy()  # occurrence matrix
    num_horizons = len(ha_df)

    event_id_dict, section_seqs_misfits = prepare_misfit(
        ha_df, first_taxon_column_index, conop_misfit_mode
    )

    ori_ha_df = ha_df.sort_values("Score").reset_index(drop=True)
    ori_mat = ori_ha_df.iloc[:, first_taxon_column_index:].to_numpy()
    ori_ha_seq = np.argsort(ori_ha_df["Horizon"])
    ori_conop_df = build_event_seq(ori_mat, taxa_names, event_id_dict)
    ori_event_series = ori_conop_df["Event id"].to_numpy()
    ori_conop_seq = np.argsort(ori_event_series)
    ori_ha_pen = compute_penalty(
        np.arange(num_horizons), ori_mat, teaser_rate=teaser_rate
    )
    ori_conop_pen = compute_misfit(ori_event_series, section_seqs_misfits)

    conop_df = ori_conop_df.sort_values("Event id").reset_index(drop=True)

    if compare_type == "conop-horse" or compare_type == "conop-conop":
        true_event_series = sequence1["Event id"].to_numpy()
        true_conop_seq = np.argsort(true_event_series)
        true_conop_pen = compute_misfit(true_event_series, section_seqs_misfits)
        print("true_conop_pen:", true_conop_pen)
        true_ha_seq = None
    else:
        sequence1 = sequence1.sort_values("Score").reset_index(drop=True)
        true_mat = sequence1.iloc[:, first_taxon_column_index:].to_numpy()
        true_ha_seq = np.argsort(sequence1["Horizon"])  # must be 0, 1, 2, ...
        true_conop_df = build_event_seq(true_mat, taxa_names, event_id_dict)
        true_event_series = true_conop_df["Event id"].to_numpy()
        true_conop_seq = np.argsort(true_event_series)
        true_ha_pen = compute_penalty(
            np.arange(num_horizons), true_mat, teaser_rate=teaser_rate
        )
        true_conop_pen = compute_misfit(true_event_series, section_seqs_misfits)
        print("true_ha_pen:", true_ha_pen, "\t", "true_conop_pen:", true_conop_pen)

    if compare_type == "horse-horse" and original_state:
        ori_spearman_horizon, ori_spearman_p_horizon = spearmanr(ori_ha_seq, true_ha_seq)
        ori_kendall_horizon, ori_kendall_p_horizon = kendalltau(ori_ha_seq, true_ha_seq)
        ori_kruskal_horizon = kruskals2(ori_ha_seq, true_ha_seq)
        spearman_horizons = [ori_spearman_horizon]
        kendall_horizons = [ori_kendall_horizon]
        kruskal_horizons = [ori_kruskal_horizon]
        spearman_p_horizons = [ori_spearman_p_horizon]
        kendall_p_horizons = [ori_kendall_p_horizon]
    else:
        spearman_horizons = []
        kendall_horizons = []
        kruskal_horizons = []
        spearman_p_horizons = []
        kendall_p_horizons = []

    ori_spearman_event, ori_spearman_p_event = spearmanr(ori_conop_seq, true_conop_seq)
    ori_kendall_event, ori_kendall_p_event = kendalltau(ori_conop_seq, true_conop_seq)
    ori_kruskal_event = kruskals2(ori_conop_seq, true_conop_seq)

    if isinstance(sequence2, pd.DataFrame):
        precomputed_pen = (
            sequence2["Penalty"].to_numpy()
            if ("Penalty" in sequence2.columns)
            else None
        )
        if precomputed_pen is None:
            steps_series = sequence2.iloc[:, 1:].to_numpy()
        else:
            steps_series = sequence2.iloc[:, 2:].to_numpy(dtype=int)

        if original_state:
            ha_pens = [ori_ha_pen]
            conop_pens = [ori_conop_pen]
            spearman_events = [ori_spearman_event]
            kendall_events = [ori_kendall_event]
            kruskal_events = [ori_kruskal_event]
            spearman_p_events = [ori_spearman_p_event]
            kendall_p_events = [ori_kendall_p_event]
            steps = [0] + sequence2["Step"].to_list()
        else:
            ha_pens = []
            conop_pens = []
            spearman_events = []
            kendall_events = []
            kruskal_events = []
            spearman_p_events = []
            kendall_p_events = []
            steps = sequence2["Step"].to_list()

        i = -1
        for this_series in tqdm.tqdm(steps_series):
            i += 1
            # this_series range is [1, len() + 1]
            if (
                compare_type == "horse-horse" or compare_type == "conop-horse"
            ):  # computed are horse series
                this_ha_seq = np.argsort(this_series)
                this_mat = mat.copy()
                this_mat = this_mat[this_series - 1]
                this_conop_df = build_event_seq(this_mat, taxa_names, event_id_dict)
                this_event_series = this_conop_df["Event id"].to_numpy()
                if precomputed_pen is None:
                    this_ha_pen = compute_penalty(
                        np.arange(len(this_ha_seq)), this_mat, teaser_rate=teaser_rate
                    )
                    ha_pens.append(this_ha_pen)
                else:
                    ha_pens.append(precomputed_pen[i])
                if compare_type == "horse-horse":
                    spearman_horizon, spearman_p_horizon = spearmanr(
                        this_ha_seq, true_ha_seq
                    )
                    kendall_horizon, kendall_p_horizon = kendalltau(
                        this_ha_seq, true_ha_seq
                    )
                    kruskal_horizon = kruskals2(this_ha_seq, true_ha_seq)
                    spearman_horizons.append(spearman_horizon)
                    kendall_horizons.append(kendall_horizon)
                    kruskal_horizons.append(kruskal_horizon)
                    spearman_p_horizons.append(spearman_p_horizon)
                    kendall_p_horizons.append(kendall_p_horizon)
                this_conop_pen = compute_misfit(this_event_series, section_seqs_misfits)
                conop_pens.append(this_conop_pen)

            else:  # computed are conop series
                this_event_series = conop_df.iloc[this_series - 1][
                    "Event id"
                ].to_numpy()
                if precomputed_pen is not None:
                    this_conop_pen = compute_misfit(
                        this_event_series, section_seqs_misfits
                    )
                    conop_pens.append(this_conop_pen)
                else:
                    conop_pens.append(precomputed_pen[i])

            this_conop_seq = np.argsort(this_event_series)
            spearman_event, spearman_p_event = spearmanr(this_conop_seq, true_conop_seq)
            kendall_event, kendall_p_event = kendalltau(this_conop_seq, true_conop_seq)
            kruskal_event = kruskals2(this_conop_seq, true_conop_seq)
            spearman_events.append(spearman_event)
            kendall_events.append(kendall_event)
            kruskal_events.append(kruskal_event)
            spearman_p_events.append(spearman_p_event)
            kendall_p_events.append(kendall_p_event)

        if compare_type == "horse-horse":
            result_df = pd.DataFrame(
                {
                    "Step": steps,
                    "HORSE penalty": ha_pens,
                    "CONOP penalty": conop_pens,
                    "Spearmans rho of horizon": spearman_horizons,
                    "Spearmans rho p-value of horizon": spearman_p_horizons,
                    "Kendalls tau of horizon": kendall_horizons,
                    "Kendalls tau p-value of horizon": kendall_p_horizons,
                    "Kruskal s^2 of horizon": kruskal_horizons,
                    "Spearmans rho of event": spearman_events,
                    "Spearmans rho p-value of event": spearman_p_events,
                    "Kendalls tau of event": kendall_events,
                    "Kendalls tau p-value of event": kendall_p_events,
                    "Kruskal s^2 of event": kruskal_events,
                }
            )
        elif compare_type == "conop-conop" or compare_type == "horse-conop":
            result_df = pd.DataFrame(
                {
                    "Step": steps,
                    "CONOP penalty": conop_pens,
                    "Spearmans rho of event": spearman_events,
                    "Spearmans rho p-value of event": spearman_p_events,
                    "Kendalls tau of event": kendall_events,
                    "Kendalls tau p-value of event": kendall_p_events,
                    "Kruskal s^2 of event": kruskal_events,
                }
            )
        else:
            result_df = pd.DataFrame(
                {
                    "Step": steps,
                    "HORSE penalty": ha_pens,
                    "CONOP penalty": conop_pens,
                    "Spearmans rho of event": spearman_events,
                    "Spearmans rho p-value of event": spearman_p_events,
                    "Kendalls tau of event": kendall_events,
                    "Kendalls tau p-value of event": kendall_p_events,
                    "Kruskal s^2 of event": kruskal_events,
                }
            )

        return result_df
    else:
        if (
            compare_type == "horse-horse" or compare_type == "conop-horse"
        ):  # computed are horse series
            this_ha_seq = np.argsort(sequence2)
            this_mat = mat.copy()
            this_mat = this_mat[sequence2 - 1]
            this_conop_df = build_event_seq(this_mat, taxa_names, event_id_dict)
            this_event_series = this_conop_df["Event id"].to_numpy()
            this_ha_pen = compute_penalty(
                np.arange(len(this_ha_seq)), this_mat, teaser_rate=teaser_rate
            )
            if compare_type == "horse-horse":
                spearman_horizon, spearman_p_horizon = spearmanr(this_ha_seq, true_ha_seq)
                kendall_horizon, kendall_p_horizon = kendalltau(
                    this_ha_seq, true_ha_seq
                )
                kruskal_horizon = kruskals2(this_ha_seq, true_ha_seq)
                print(
                    "HORSE penalty:",
                    this_ha_pen,
                    "Spearmans rho (horizon):",
                    spearman_horizon,
                    "Spearmans rho p-value (horizon):",
                    spearman_p_horizon,
                    "Kendalls tau (horizon)",
                    kendall_horizon,
                    "Kendalls tau p-value (horizon)",
                    kendall_p_horizon,
                    "Kruskal s^2 (horizon)",
                    kruskal_horizon,
                )
        else:  # computed are conop series
            this_event_series = conop_df.iloc[sequence2 - 1]["Event id"].to_numpy()

        this_conop_seq = np.argsort(this_event_series)
        this_conop_pen = compute_misfit(this_event_series, section_seqs_misfits)
        spearman_event, spearman_p_event = spearmanr(this_conop_seq, true_conop_seq)
        kendall_event, kendall_p_event = kendalltau(this_conop_seq, true_conop_seq)
        kruskal_event = kruskals2(this_conop_seq, true_conop_seq)
        print(
            "CONOP penalty:",
            this_conop_pen,
            "Spearmans rho (event):",
            spearman_event,
            "Spearmans rho p-value (event):",
            spearman_p_event,
            "Kendalls tau (event)",
            kendall_event,
            "Kendalls tau p-value (event)",
            kendall_p_event,
            "Kruskal s^2 (event)",
            kruskal_event,
        )


def compare_range(
    ha_df,
    sequence1,
    sequence2,
    compare_type="horse-horse",
    arrangement="fad",
    show_taxon_names=True,
    sequence1_name="Ground Truth",
    sequence2_name="Computed Sequence",
):
    """
    compare taxon ranges.
    :param show_taxon_names:
    :param ha_df:
    :param sequence2:
    :param sequence1:
    :param compare_type:
    :param arrangement:
    :return:
    """

    def get_fad_lad_df_from_ha_df(df: pd.DataFrame):
        first_taxon_column = list(df.columns).index("*") + 1
        fad_lad = (
            df.iloc[:, first_taxon_column:].apply(find_fad_lad, axis=0).T.reset_index()
        )
        fad_lad = fad_lad.rename(columns={"index": "taxon", 0: "fad", 1: "lad"})

        combined_ranks = (
            fad_lad[["fad", "lad"]].stack().rank(method="first").astype(int)
        )
        fad_lad["ranked_fad"] = combined_ranks.loc[:, "fad"].values
        fad_lad["ranked_lad"] = combined_ranks.loc[:, "lad"].values
        fad_lad["duration"] = fad_lad["ranked_lad"] - fad_lad["ranked_fad"]
        return fad_lad.sort_values("taxon").reset_index(drop=True)

    def get_fad_lad_df_from_conop_df(df: pd.DataFrame):
        fad_lad = pd.DataFrame(
            np.zeros((len(taxa_names), 3)), columns=["taxon", "fad", "lad"]
        )
        fad_lad["taxon"] = taxa_names
        fad_lad.set_index("taxon", inplace=True)
        for rank in range(len(df)):
            _taxon, _type = df[["Taxon", "Type"]].iloc[rank]
            if _type == 0:
                fad_lad.loc[_taxon, "fad"] = rank
            else:
                fad_lad.loc[_taxon, "lad"] = rank

        fad_lad["ranked_fad"] = fad_lad["fad"]
        fad_lad["ranked_lad"] = fad_lad["lad"]
        fad_lad["duration"] = fad_lad["ranked_lad"] - fad_lad["ranked_fad"]
        return fad_lad.sort_values("taxon").reset_index()

    ha_df = ha_df.sort_values("Horizon").reset_index(drop=True)
    first_taxon_column_index = list(ha_df.columns).index("*") + 1
    mat = ha_df.iloc[:, first_taxon_column_index:].to_numpy()
    taxa_names = ha_df.iloc[:, first_taxon_column_index:].columns.to_list()
    if compare_type == "horse-horse" or compare_type == "conop-horse":
        this_ha_df = ha_df.loc[sequence2 - 1, :].reset_index(drop=True)
        this_fad_lad = get_fad_lad_df_from_ha_df(this_ha_df)
    else:
        event_id_dict, section_seqs_misfits = prepare_misfit(
            ha_df, first_taxon_column_index
        )
        ori_conop_df = (
            build_event_seq(mat, taxa_names, event_id_dict)
            .sort_values("Event id")
            .reset_index(drop=True)
        )
        this_conop_df = ori_conop_df.iloc[sequence2 - 1, :].reset_index(drop=True)
        this_fad_lad = get_fad_lad_df_from_conop_df(this_conop_df)

    if compare_type == "horse-horse" or compare_type == "horse-conop":
        true_fad_lad = get_fad_lad_df_from_ha_df(sequence1)
    else:
        true_fad_lad = get_fad_lad_df_from_conop_df(sequence1)

    this_set = set(this_fad_lad["taxon"].to_list())
    true_set = set(true_fad_lad["taxon"].to_list())
    true_minus_this = true_set - this_set
    this_minus_true = this_set - true_set
    if len(true_minus_this) + len(this_minus_true):
        print("Warning: there are not consistent taxa in true and computed data.")
        print("\tpresent in true but not computed:", true_minus_this)
        print("\tpresent in computed but not true:", this_minus_true)
        exit(-1)

    # sort
    if arrangement == "fad":
        sort_id = np.argsort(true_fad_lad["ranked_fad"])
    elif arrangement == "lad":
        sort_id = np.argsort(true_fad_lad["ranked_lad"])
    else:
        sort_id = np.argsort(true_fad_lad["taxon"])

    this_fad_lad = this_fad_lad.iloc[sort_id, :]
    true_fad_lad = true_fad_lad.iloc[sort_id, :]

    # draw
    dist_fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    sns.histplot(true_fad_lad["duration"], kde=True, ax=axes[0])
    sns.histplot(this_fad_lad["duration"], kde=True, ax=axes[1])
    axes[0].set_xlabel("Duration")
    axes[0].set_ylabel("Count")
    # axes[0].set_title("Distribution of Duration (True)")
    axes[1].set_xlabel("Duration")
    axes[1].set_ylabel("Count")
    # axes[1].set_title("Distribution of Duration (Computed)")
    # Plot vertical lines for mean duration values
    true_mean_duration = true_fad_lad["duration"].mean()
    computed_mean_duration = this_fad_lad["duration"].mean()
    axes[0].axvline(true_mean_duration, color="k", linestyle="dashed", linewidth=1)
    axes[1].axvline(computed_mean_duration, color="k", linestyle="dashed", linewidth=1)
    axes[0].text(
        true_mean_duration,
        max(axes[0].get_ylim()),
        "Mean: {:.2f}".format(true_mean_duration),
        ha="center",
        va="bottom",
        color="k",
    )
    axes[1].text(
        computed_mean_duration,
        max(axes[1].get_ylim()),
        "Mean: {:.2f}".format(computed_mean_duration),
        ha="center",
        va="bottom",
        color="k",
    )
    dist_fig.tight_layout()

    # Create a figure
    range_fig, ax = plt.subplots(figsize=(7.5, 4.16), dpi=300)

    # Add data to the figure
    sns.barplot(
        x=true_fad_lad["taxon"],
        y="duration",
        data=true_fad_lad,
        bottom=true_fad_lad["ranked_fad"],
        color="#0081a7",
        alpha=0.7,
        label=sequence1_name,
        ax=ax,
        dodge=False,
    )
    sns.barplot(
        x=this_fad_lad["taxon"],
        y="duration",
        data=this_fad_lad,
        bottom=this_fad_lad["ranked_fad"],
        color="#f07167",
        alpha=0.7,
        label=sequence2_name,
        ax=ax,
        dodge=False,
    )

    # Draw a line for fad and lad bars for true_fad_lad
    for i, taxon in enumerate(true_fad_lad["taxon"]):
        index = true_fad_lad[true_fad_lad["taxon"] == taxon].index[0]
        ax.hlines(
            y=true_fad_lad.loc[index, "ranked_fad"],
            xmin=i - 0.4,
            xmax=i + 0.4,
            color="black",
            linewidth=0.5,
        )
        ax.hlines(
            y=true_fad_lad.loc[index, "ranked_lad"],
            xmin=i - 0.4,
            xmax=i + 0.4,
            color="black",
            linewidth=0.5,
        )

    # Draw a line for fad and lad bars for this_fad_lad
    for i, taxon in enumerate(this_fad_lad["taxon"]):
        index = this_fad_lad[this_fad_lad["taxon"] == taxon].index[0]
        ax.hlines(
            y=this_fad_lad.loc[index, "ranked_fad"],
            xmin=i - 0.4,
            xmax=i + 0.4,
            color="red",
            linewidth=0.5,
        )
        ax.hlines(
            y=this_fad_lad.loc[index, "ranked_lad"],
            xmin=i - 0.4,
            xmax=i + 0.4,
            color="red",
            linewidth=0.5,
        )

    # Customize the chart
    ax.set_xlabel("Taxon", fontsize=7)
    ax.set_ylabel("Event location", fontsize=7)
    # ax.set_title("Taxon Range Chart", fontsize=14)
    ax.legend(fontsize=7, loc="upper left")

    if show_taxon_names:
        # Rotate x ticks and make font size smaller
        plt.xticks(rotation=-90, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Display the chart
    range_fig.tight_layout()
    return range_fig, dist_fig


def plot_result(
    result_df, true_conop_penalty=None, true_horse_penalty=None
):

    step = result_df["Step"]
    conop_pens = result_df["CONOP penalty"]
    spearman_events = result_df["Spearmans rho of event"]
    kendall_events = result_df["Kendalls tau of event"]

    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.5), dpi=300)

    # draw step-penalty
    ax_step_pen = axes[0, 0]
    ha_pens = result_df["HORSE penalty"]
    ax_step_pen.plot(step, conop_pens, label="CONOP penalty", c="orangered", linestyle="-")
    ax_step_pen.plot(step, ha_pens, label="HORSE penalty", c="royalblue", linestyle="--")

    if true_conop_penalty is not None:
        conop_intersect = (conop_pens - true_conop_penalty).abs().idxmin()
        ax_step_pen.scatter(
            step[conop_intersect],
            true_conop_penalty,
            color="#f07167",
            marker="o",
            label="True CONOP penalty",
        )
    if true_horse_penalty is not None:
        horse_intersect = (ha_pens - true_horse_penalty).abs().idxmin()
        ax_step_pen.scatter(
            step[horse_intersect],
            true_horse_penalty,
            color="#0081a7",
            marker="^",
            label="True HORSE penalty",
        )

    ax_step_pen.set_ylabel("Penalty")
    ax_step_pen.set_xlabel("Iteration")
    ax_step_pen.legend(fontsize=7)

    # draw step-cc
    ax_step_cc_events = axes[0, 1]
    if spearman_events is not None:
        ax_step_cc_events.plot(step, spearman_events, label="Spearman's rho", c="#6a994e")
    if kendall_events is not None:
        ax_step_cc_events.plot(step, kendall_events, label="Kendall's tau", c="#ee9b00", linestyle="--")

    ax_step_cc_events.set_xlabel("Iteration")
    ax_step_cc_events.set_ylabel("Correlation Coefficient")
    ax_step_cc_events.legend(fontsize=7)

    # draw penalty-cc
    ax_pen_CONOP_cc_events = axes[1, 0]
    ax_pen_HORSE_cc_events = axes[1, 1]

    if spearman_events is not None:
        ax_pen_CONOP_cc_events.scatter(
            conop_pens, spearman_events, s=10, c="#6a994e", label="Spearman's rho", marker="+"
        )
        draw_linear_regression(
            conop_pens, spearman_events, color="#6a994e", ax=ax_pen_CONOP_cc_events
        )
        ax_pen_HORSE_cc_events.scatter(
            ha_pens, spearman_events, s=10, c="#6a994e", label="Spearman's rho", marker="+"
        )
        draw_linear_regression(ha_pens, spearman_events, color="#6a994e", ax=ax_pen_HORSE_cc_events)
    if kendall_events is not None:
        ax_pen_CONOP_cc_events.scatter(
            conop_pens, kendall_events, s=10, c="#ee9b00", label="Kendall's tau", marker="x"
        )
        draw_linear_regression(
            conop_pens, kendall_events, color="#ee9b00", ax=ax_pen_CONOP_cc_events, linestyle="--"
        )
        ax_pen_HORSE_cc_events.scatter(
            ha_pens, kendall_events, s=10, c="#ee9b00", label="Kendall's tau", marker="x"
        )
        draw_linear_regression(
            ha_pens, kendall_events, color="#ee9b00", ax=ax_pen_HORSE_cc_events, linestyle="--"
        )

    ax_pen_CONOP_cc_events.set_xlabel("CONOP penalty")
    ax_pen_CONOP_cc_events.set_ylabel("Correlation Coefficient")
    ax_pen_CONOP_cc_events.legend(fontsize=7)
    
    ax_pen_HORSE_cc_events.set_xlabel("HORSE penalty")
    ax_pen_HORSE_cc_events.set_ylabel("Correlation Coefficient")
    ax_pen_HORSE_cc_events.legend(fontsize=7)
    
    fig.tight_layout()

    return fig


def main():
    ha_path = r"../paper_data/Pseudo_ha_v6.csv"
    conop_misfit_mode = "order"

    ha_df = pd.read_csv(ha_path, encoding="utf_8_sig")
    true_data = ha_df.copy(deep=True)
    true_data["Score"] = ha_df["Horizon"] / ha_df["Horizon"].max()
    
    horizon_id_path = r"../paper_data/Pseudo_ha_v6-teaser_id.csv"
    result_path = r"../paper_data/Pseudo_ha_v6-horse_teaser_results.csv"
    compare_type = "horse-horse"
    data_computed = pd.read_csv(horizon_id_path)

    range_fig, dist_fig = compare_range(
        ha_df=ha_df,
        sequence1=true_data,
        sequence2=data_computed.iloc[-1, 1:].to_numpy(),
        compare_type=compare_type,
        arrangement="fad",
        show_taxon_names=False,
        sequence1_name="Ground Truth",
        sequence2_name="HORSE + Teaser"
    )
    range_fig.savefig("../paper_figs/range_fig_true_horse_teaser.pdf", dpi=300, format="pdf")
    dist_fig.savefig("../paper_figs/dist_fig_true_horse_teaser.pdf", dpi=300, format="pdf")

    compare(
        ha_df=ha_df,
        sequence1=true_data,
        sequence2=data_computed.iloc[-1, 1:].to_numpy(),
        compare_type=compare_type,
        conop_misfit_mode=conop_misfit_mode,
        teaser_rate=0.01,
    )

    result_df = compare(
        ha_df=ha_df,
        sequence2=data_computed,
        sequence1=true_data,
        compare_type=compare_type,
        conop_misfit_mode=conop_misfit_mode,
    )
    result_df.to_csv(result_path)

    fig = plot_result(
        result_df, true_conop_penalty=2251, true_horse_penalty=1997.72
    )
    fig.savefig("../paper_figs/iteration_pen_cc.pdf", dpi=300, format="pdf")

if __name__ == "__main__":
    main()
