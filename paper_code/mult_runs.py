import numpy as np
import pandas as pd
from numba import jit
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kendalltau, linregress
from utils import find_fad_lad, prepare_misfit, compute_misfit, compute_penalty, build_event_seq, separate_horizons
from sklearn.metrics import pairwise_distances
import scipy
from scipy.spatial.distance import squareform
from scipy.linalg import eigh


sequences_path = r'paper_data/Ten_runs_Pseudo_ha_v6/compare.csv'
ha_path = r'paper_data/Pseudo_ha_v6.csv'

conop_misfit_mode = 'order'
ha_df = pd.read_csv(ha_path, encoding='utf_8_sig')
true_data = ha_df.copy(deep=True)
true_data['Score'] = true_data['Horizon'] / true_data['Horizon'].max()


ha_df = ha_df.sort_values('Horizon').reset_index(drop=True)
first_taxon_column_index = list(ha_df.columns).index('*') + 1
taxa_names = ha_df.iloc[:, first_taxon_column_index:].columns.to_list()
mat = ha_df.iloc[:, first_taxon_column_index:].to_numpy()  # occurrence matrix
num_horizons = len(ha_df)

event_id_dict, section_seqs_misfits = prepare_misfit(ha_df, first_taxon_column_index, conop_misfit_mode)

ori_ha_df = ha_df.sort_values('Score').reset_index(drop=True)
ori_mat = ori_ha_df.iloc[:, first_taxon_column_index:].to_numpy()
ori_ha_seq = np.argsort(ori_ha_df['Horizon'])
ori_conop_df = build_event_seq(ori_mat, taxa_names, event_id_dict)
ori_event_series = ori_conop_df['Event id'].to_numpy()

true_ha_df = true_data.sort_values('Score').reset_index(drop=True)
true_mat = true_ha_df.iloc[:, first_taxon_column_index:].to_numpy()
true_all_ha_sequence = true_data['Horizon']
true_ha_seq = np.argsort(true_ha_df['Horizon'])
true_conop_df = build_event_seq(true_mat, taxa_names, event_id_dict)
true_event_series = true_conop_df['Event id'].to_numpy()

true_event_horizons_df, true_non_event_horizon_info_df = separate_horizons(true_ha_df)
true_event_ha_sequence = true_event_horizons_df.sort_values('Score')['Horizon'].to_numpy()
true_no_event_ha_sequence = true_non_event_horizon_info_df.sort_values('Score')['Horizon'].to_numpy()

conop_df = ori_conop_df.sort_values('Event id').reset_index(drop=True)

sequences_df = pd.read_csv(sequences_path)
types = ['TRUE']
teasers = ['']
event_sequences = [true_event_series]
event_orders = [np.argsort(true_event_series)]
pearson_has = []
kendall_has = []
event_pearson_has = []
no_event_pearson_has = []
event_kendall_has = []
no_event_kendall_has = []

for i in range(sequences_df.shape[0]):
    if sequences_df.iloc[i, 0] == 'HORSE':
        this_ha_sequence = sequences_df.iloc[i, 2:].dropna().to_numpy(dtype=int)
        this_mat = mat.copy()
        this_mat = this_mat[this_ha_sequence - 1]
        this_conop_df = build_event_seq(this_mat, taxa_names, event_id_dict)
        this_event_series = this_conop_df['Event id'].to_numpy()

        # event and non-event horizons
        this_ha_df = ha_df.loc[this_ha_sequence - 1, :].reset_index(drop=True)
        this_ha_df['Score'] = np.arange(len(this_ha_df))
        all_ha_sequence = this_ha_df.sort_values('Score')['Horizon'].to_numpy()
        event_horizons_df, non_event_horizon_info_df = separate_horizons(this_ha_df)
        event_ha_sequence = event_horizons_df.sort_values('Score')['Horizon'].to_numpy()
        no_event_ha_sequence = non_event_horizon_info_df.sort_values('Score')['Horizon'].to_numpy()

        pearson_ha, pearson_p_ha = pearsonr(all_ha_sequence, true_all_ha_sequence)
        kendall_ha, kendall_p_ha = kendalltau(all_ha_sequence, true_all_ha_sequence)
        event_pearson_ha, event_pearson_p_ha = pearsonr(event_ha_sequence, true_event_ha_sequence)
        no_event_pearson_ha, no_event_pearson_p_ha = pearsonr(no_event_ha_sequence, true_no_event_ha_sequence)
        event_kendall_ha, event_kendall_p_ha = kendalltau(event_ha_sequence, true_event_ha_sequence)
        no_event_kendall_ha, no_event_kendall_p_ha = kendalltau(no_event_ha_sequence, true_no_event_ha_sequence)
        pearson_has.append(pearson_ha)
        kendall_has.append(kendall_ha)
        event_pearson_has.append(event_pearson_ha)
        no_event_pearson_has.append(no_event_pearson_ha)
        event_kendall_has.append(event_kendall_ha)
        no_event_kendall_has.append(no_event_kendall_ha)
    else:
        this_event_series = sequences_df.iloc[i, 2:].dropna().to_numpy(dtype=int)
    types.append(sequences_df.iloc[i, 0])
    teasers.append('Teaser' if sequences_df.iloc[i, 1] == 'ON' else '')
    event_sequences.append(this_event_series)
    event_orders.append(np.argsort(this_event_series))

# box plot of event and non-event horizons
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

pearson_boxes = axs[0].boxplot(
    [pearson_has, event_pearson_has, no_event_pearson_has],
    patch_artist=True,
    tick_labels=['All', 'Event-present', 'Event-absent']
)
for patch, color in zip(pearson_boxes['boxes'], ['#0081a7', '#fdfcdc', '#f07167']):
    patch.set_facecolor(color)
for median in pearson_boxes['medians']:
    median.set_color('#212529')

kendall_boxes = axs[1].boxplot(
    [kendall_has, event_kendall_has, no_event_kendall_has],
    patch_artist=True,
    tick_labels=['All', 'Event-present', 'Event-absent']
)
for patch, color in zip(kendall_boxes['boxes'], ['#0081a7', '#fdfcdc', '#f07167']):
    patch.set_facecolor(color)
for median in kendall_boxes['medians']:
    median.set_color('#212529')

for boxplot, ax in zip([pearson_boxes, kendall_boxes], axs):
    for i, median in enumerate(boxplot['medians']):
        median_x = median.get_xdata()[0]
        median_y = median.get_ydata()[0]
        # ax.text(median_x, median_y, f'{median_y:.4f}', verticalalignment='center', horizontalalignment='right')

    # for i, whisker in enumerate(boxplot['whiskers']):
    #     whisker_x = whisker.get_xdata()[1]
    #     whisker_y = whisker.get_ydata()[1]
    #     ax.text(whisker_x, whisker_y, f'{whisker_y:.4f}', verticalalignment='center', horizontalalignment='right')

# axs[0].set_title('Pearson\'s r')
axs[0].set_ylabel('Correlation Coefficient')
# axs[1].set_title('Kendall\'s tau')
axs[1].set_ylabel('Correlation Coefficient')

fig.tight_layout()
fig.savefig(r'paper_figs/horizon_correlation.svg')


# plot of event placing
def draw_column(event_placings, min_val, max_val, ax=None):
    event_placings_means = event_placings.mean(axis=0)
    event_placings_stds = event_placings.std(axis=0)
    event_placings_order_id = np.argsort(event_placings_means)
    event_placings_stds_ordered = np.log(event_placings_stds[event_placings_order_id])

    # Assuming event_placings_stds_ordered is a series of data
    # Calculate the mean of the series
    mean = np.mean(event_placings_stds)
    # Calculate the standard error of the mean (SEM)
    sem = scipy.stats.sem(event_placings_stds)
    # Determine the z-score for a 95% confidence interval
    z_score = 1.96
    # Calculate the margin of error
    margin_of_error = z_score * sem
    # Print the confidence interval
    print('%.2f +- %.2f' % (mean, margin_of_error))

    scale = len(event_placings_stds_ordered) / 20
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, len(event_placings_stds_ordered)/scale))
        fig.tight_layout()

    norm = plt.Normalize(min_val, max_val)
    colors = plt.cm.viridis(norm(event_placings_stds_ordered))
    for idx, (color, std) in enumerate(zip(colors, event_placings_stds_ordered)):
        rect = plt.Rectangle((0, idx/scale), 1, 1/scale, color=color)
        ax.add_patch(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(event_placings_stds_ordered)/scale)
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_ylabel('Level')
    # fig.show()
    return fig if ax is None else None

fig, axs = plt.subplots(1, 4, figsize=(10, 15), dpi=300)
min_log_std = -1
max_log_std = 7
horse_teaser_fig = draw_column(np.array(event_orders)[(np.array(types) == 'HORSE') & (np.array(teasers) == 'Teaser')], min_log_std, max_log_std, ax=axs[0])
conop_teaser_fig = draw_column(np.array(event_orders)[(np.array(types) == 'CONOP') & (np.array(teasers) == 'Teaser')], min_log_std, max_log_std, ax=axs[1])
horse_fig = draw_column(np.array(event_orders)[(np.array(types) == 'HORSE') & (np.array(teasers) == '')], min_log_std, max_log_std, ax=axs[2])
conop_fig = draw_column(np.array(event_orders)[(np.array(types) == 'CONOP') & (np.array(teasers) == '')], min_log_std, max_log_std, ax=axs[3])
fig.tight_layout()
fig.savefig(r'paper_figs/event_col_stack.svg')
# Create a colorbar for reference
fig, ax = plt.subplots(figsize=(2, 5), dpi=300)
norm = plt.Normalize(min_log_std, max_log_std)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Event Placement Volatility')
fig.tight_layout()
fig.savefig(r'paper_figs/event_col_stack_colorbar.svg')


# plots of MDS
event_order_mat = np.array(event_orders)

colors = {'TRUE ': 'black', 'HORSE Teaser': 'royalblue', 'HORSE ': 'lightsteelblue',
          'CONOP Teaser': 'orangered', 'CONOP ': 'lightsalmon'}
markers = {'TRUE ': 's', 'HORSE Teaser': '^', 'HORSE ': '^', 'CONOP Teaser': 'o', 'CONOP ': 'o'}

# Distance matrix
# As we use event ranking, pearson correlation is equivalent to spearman correlation
spearman_dist = pairwise_distances(event_order_mat, metric='correlation')

# PCoA (Principal Coordinate Analysis)
# Perform PCoA (Principal Coordinate Analysis) using cmdscale method
# Classical MDS (PCoA)

# Double center the distance matrix
n = spearman_dist.shape[0]
H = np.eye(n) - np.ones((n, n)) / n
B = -H @ spearman_dist ** 2 @ H / 2

# Eigendecomposition
eigvals, eigvecs = eigh(B)

# Sort eigenvalues in decreasing order
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Keep only positive eigenvalues
positive_idx = eigvals > 0
eigvals = eigvals[positive_idx]
eigvecs = eigvecs[:, positive_idx]

# Calculate the coordinates
pcoa_coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])

# Calculate the explained variance ratio
explained_variance_ratio = eigvals[:2] / np.sum(eigvals)

# Plot PCoA results
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
for i, (type, teaser) in enumerate(zip(types, teasers)):
    ax.scatter(pcoa_coords[i, 0], pcoa_coords[i, 1], 
               color=colors[f'{type} {teaser}'], 
               label=f'{type} {teaser}', 
               marker=markers[f'{type} {teaser}'])

ax.set_xlabel(f'PCo1 ({explained_variance_ratio[0]:.2%})')
ax.set_ylabel(f'PCo2 ({explained_variance_ratio[1]:.2%})')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
plt.tight_layout()
fig.savefig(r'paper_figs/pcoa.svg')
