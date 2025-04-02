"""
This module provides functions for computing misfit and preparing data.
"""

from numba import jit
import numpy as np
import pandas as pd


def find_fad_lad(mat: np.ndarray):
    one_mat = mat == 1
    fad = np.argmax(one_mat, axis=0)
    lad = mat.shape[0] - 1 - np.argmax(np.flipud(one_mat), axis=0)
    return fad, lad


@jit(nopython=True)
def compute_penalty(new_indices: np.ndarray, ori_mat: np.ndarray, teaser_rate=0.0):
    new_mat = ori_mat[new_indices]  # the occurrence data according to the new indices
    one_mat = new_mat == 1
    zero_mat = new_mat == 0
    if teaser_rate > 0:
        minus_one_mat = new_mat == -1
    else:
        minus_one_mat = None
    first_one_indices = np.argmax(one_mat, axis=0)
    last_one_indices = new_mat.shape[0] - np.argmax(one_mat[::-1, :], axis=0) - 1

    # only count those zeros within the 1-bounded range
    mask = (np.arange(new_mat.shape[0]).reshape((-1, 1)) >= first_one_indices) & (
            np.arange(new_mat.shape[0]).reshape((-1, 1)) <= last_one_indices)
    zero_indices = zero_mat & mask
    zero_penalty = np.sum(zero_indices)
    if teaser_rate > 0:
        minus_one_indices = minus_one_mat & mask
        minus_one_penalty = np.sum(minus_one_indices) * teaser_rate
        penalty = zero_penalty + minus_one_penalty
    else:
        penalty = zero_penalty
    return penalty


def build_event_seq(mat: np.ndarray, taxa_names, event_id_dict):
    fad, lad = find_fad_lad(mat)
    conop_df = pd.DataFrame(np.zeros((mat.shape[1] * 2, 6)), columns=['Event id', 'Event name', 'Taxon', 'Type',
                                                                      'Place', 'Location'])
    event_ids = event_id_dict[taxa_names]
    conop_df['Event id'] = [i[0] for i in event_ids] + [i[1] for i in event_ids]
    conop_df['Event name'] = ['F_' + n for n in taxa_names] + ['L_' + n for n in taxa_names]
    conop_df['Taxon'] = taxa_names + taxa_names
    conop_df['Type'] = [0] * mat.shape[1] + [1] * mat.shape[1]
    conop_df['Location'] = np.append(fad, lad)  # location is the index in mat (the array of horizon)
    conop_df['Event id'] = conop_df['Event id'].astype(int)

    # sort against locs and types
    # changed on 25/12/2023, fad and lad on the same horizon will be disentangled into different 'places'
    places = conop_df['Location'].to_numpy()
    places = np.append(places[:len(fad)], places[len(fad):] + 0.1)  # to disentangle fad lad on same horizon
    conop_df['Place'] = places
    conop_df['Place'] = conop_df['Place'].rank(method='dense').astype(int) - 1
    conop_df = conop_df.sort_values('Place').reset_index(drop=True)

    return conop_df


def section_prepare_misfit(section_df: pd.DataFrame, first_taxon_column_index, event_id_dict, mode):
    section_df = section_df.sort_values('Horizon').reset_index(drop=True)
    non_exist_index = section_df.iloc[0, first_taxon_column_index:] == -1
    exist_taxa_df = section_df.iloc[:, first_taxon_column_index:].loc[:, ~non_exist_index]
    mat = exist_taxa_df.to_numpy()  # this mat is the local section mat
    taxa_names = exist_taxa_df.columns.to_list()
    conop_df = build_event_seq(mat, taxa_names, event_id_dict)

    # 'Location' is now for computing the distances between different 'Place'
    if mode == 'depth':
        type_i = conop_df['Type'] == 0
        conop_df.loc[type_i, 'Location'] = section_df.loc[conop_df.loc[type_i, 'Location'], 'Depth from'].values
        conop_df.loc[~type_i, 'Location'] = section_df.loc[conop_df.loc[~type_i, 'Location'], 'Depth to'].values
    elif mode == 'order':

        conop_df['Location'] = conop_df['Place']
    else:  # use 'horizon', noted that fad lad on the same horizon will not be separated
        pass
    num_places = conop_df['Place'].max() + 1
    place_misfit = pd.DataFrame(np.zeros(shape=(len(conop_df), num_places)), index=conop_df['Event id'])
    place_loc_dict = {order: location for order, location in zip(conop_df['Place'], conop_df['Location'])}

    for i in range(len(conop_df)):
        event_id = conop_df.loc[i, 'Event id']
        e_place = conop_df.loc[i, 'Place']
        for place in range(num_places):
            if e_place == place:
                cost = 0
            else:
                dist = abs(place_loc_dict[place] - place_loc_dict[e_place])
                if e_place > place:
                    # moving down
                    weight = 1 if conop_df.loc[i, 'Type'] == 0 else 7777
                else:
                    # moving up
                    weight = 1 if conop_df.loc[i, 'Type'] == 1 else 7777
                cost = weight * dist
            place_misfit.loc[event_id, place] = cost
    section_seq = conop_df['Event id'].to_numpy()
    section_places = pd.Series(conop_df['Place'].values, index=conop_df['Event id'])
    return [section_seq, section_places, place_misfit]


def prepare_misfit(ha_df: pd.DataFrame, first_taxon_column_index, misfit_mode='order'):
    """
    Build taxon-event dict, and compute location misfit for each event.
    :param misfit_mode:
    :param ha_df:
    :param first_taxon_column_index:
    :param misfit_mode: {'horizon', 'order', 'depth'}
    :return: event_id_dict, section_seqs_misfits
    """
    # sort the names with better event id correspondence
    sorted_taxa_names = np.sort(ha_df.iloc[:, first_taxon_column_index:].columns)
    event_id_dict = pd.Series(
        {taxa_name: (i + 1, i + 1 + len(sorted_taxa_names)) for i, taxa_name in enumerate(sorted_taxa_names)})
    section_seqs_misfits = ha_df.groupby('Section', group_keys=False).apply(section_prepare_misfit,
                                                                            first_taxon_column_index,
                                                                            event_id_dict,
                                                                            misfit_mode)
    return event_id_dict, section_seqs_misfits


def build_conop_df(ha_df, first_taxon_column_index, conop_misfit_mode):
    taxa_names = ha_df.iloc[:, first_taxon_column_index:].columns.to_list()
    mat = ha_df.iloc[:, first_taxon_column_index:].to_numpy()  # occurrence matrix
    num_events = len(taxa_names) * 2
    event_id_dict = pd.Series(
        {taxa_name: (i, i + num_events) for i, taxa_name in enumerate(taxa_names)})

    conop_df = build_event_seq(mat, taxa_names, event_id_dict)
    section_seqs_misfits = ha_df.groupby('Section', group_keys=False).apply(section_prepare_misfit,
                                                                            first_taxon_column_index,
                                                                            event_id_dict,
                                                                            conop_misfit_mode)
    return conop_df, section_seqs_misfits


@jit(nopython=True)
def compute_section_wise_misfit(levels: np.ndarray, target_seq: np.ndarray, misfits: np.ndarray):
    """
    Compute section-wise misfit
    Args:
        target_seq: The target sequence of events. Assuming events in this section are (0, 1, 2, ...).
        levels:     The corresponding level of each event in this section.
        misfits:    The misfit generated by moving each event to each level.

    Returns: section misfit

    """
    section_misfit = 0
    current_levels = np.zeros_like(levels)
    prev_level = 0  # last level being put, initial 0
    for event in target_seq:
        event_level = levels[event]
        if event_level >= prev_level:
            current_levels[event] = event_level
            prev_level = event_level
            continue
        else:
            level_here = prev_level
            pen_here = misfits[event, level_here]
            pen_down = pen_here
            current_levels[event] = level_here
            temp_levels = current_levels.copy()
            while pen_down <= pen_here and level_here > 0:
                events_here = temp_levels == level_here
                level_down = level_here - 1
                pen_down = misfits[events_here, level_down].sum()
                if pen_down <= pen_here:
                    pen_here = pen_down
                    temp_levels[events_here] = level_down
                    level_here = level_down
                    prev_level = level_here
                    current_levels = temp_levels.copy()
                else:
                    break
            section_misfit += pen_here

    section_misfit_2 = 0
    for i in range(len(levels)):
        section_misfit_2 += misfits[i, current_levels[i]]

    return section_misfit_2


def compute_misfit(composite_seq, section_seqs_misfits):
    total_misfit = 0
    for section in section_seqs_misfits:
        start_seq = section[0]
        event_order_in_comp_sec = []
        for event in start_seq:
            event_order_in_comp_sec.append(np.where(composite_seq == event)[0][0])

        #  将event按照start_seq里面的顺序对应成index，以便numba处理
        places = section[1].values
        misfits = section[2].to_numpy()
        target_seq = np.argsort(event_order_in_comp_sec)
        section_misfit = compute_section_wise_misfit(places, target_seq, misfits)
        total_misfit += section_misfit
    return total_misfit


def separate_horizons(combined_ha_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate event-bearing horizons and non-bearing ones.
    Returns event_horizon_df and non_event_horizon_df,
    where non_event_horizon_df includes five extra columns: 'non_event_horizon_id', 'lower_horizon_id',
    'upper_horizon_id', 'lower_event_horizon_id', and 'upper_event_horizon_id'.

    Args:
        combined_ha_df: Complete horizon DataFrame to separate.

    Returns:
        event_horizon_df, non_event_horizon_info_df

    """
    combined_ha_df = combined_ha_df.sort_values('Score').reset_index(drop=True)
    first_taxon_column_index = list(combined_ha_df.columns).index('*') + 1

    section_event_horizon_dfs = []
    section_non_event_horizon_dfs = []

    for section in combined_ha_df.groupby('Section'):
        section_df = section[1].reset_index(drop=True)

        mat = section_df.iloc[:, first_taxon_column_index:].to_numpy()
        non_exist_taxa_indices = np.where(np.all(mat == -1, axis=0))[0]
        exist_taxa_mat = mat[:, list(set(np.arange(mat.shape[1])) - set(non_exist_taxa_indices))]
        one_mat = exist_taxa_mat == 1
        zero_mat = exist_taxa_mat == 0
        fad_indices = np.argmax(one_mat, axis=0)
        lad_indices = exist_taxa_mat.shape[0] - 1 - np.argmax(np.flipud(one_mat), axis=0)
        fad_lad_indices = np.unique(np.concatenate((fad_indices, lad_indices)))
        non_fad_lad_indices = list(set(np.arange(mat.shape[0])) - set(fad_lad_indices))
        non_fad_lad_indices.sort()
        if len(non_fad_lad_indices):
            section_non_event_horizon_df = section_df.iloc[non_fad_lad_indices, :].reset_index(drop=True)
            for i, non_fad_lad_index in enumerate(non_fad_lad_indices):
                # 找出最近邻的horizon
                section_non_event_horizon_df.loc[i, 'lower_horizon_id'] = section_df['Horizon'].iloc[
                    non_fad_lad_index - 1]
                section_non_event_horizon_df.loc[i, 'upper_horizon_id'] = section_df['Horizon'].iloc[
                    non_fad_lad_index + 1]
                # 找出最近邻的event-bearing horizon
                down = up = non_fad_lad_index
                while down > 0:
                    down -= 1
                    if down in fad_lad_indices:
                        section_non_event_horizon_df.loc[i, 'lower_event_horizon_id'] = section_df['Horizon'].iloc[
                            down]
                        break
                while up < len(section_df):
                    up += 1
                    if up in fad_lad_indices:
                        section_non_event_horizon_df.loc[i, 'upper_event_horizon_id'] = section_df['Horizon'].iloc[
                            up]
                        break
            section_non_event_horizon_dfs.append(section_non_event_horizon_df)
        else:
            pass
        section_event_horizon_dfs.append(section_df.iloc[fad_lad_indices, :].reset_index(drop=True))

    event_horizon_df = pd.concat(section_event_horizon_dfs).sort_values('Score').reset_index(drop=True)
    non_event_horizon_df = pd.concat(section_non_event_horizon_dfs).reset_index(drop=True)

    return event_horizon_df, non_event_horizon_df
