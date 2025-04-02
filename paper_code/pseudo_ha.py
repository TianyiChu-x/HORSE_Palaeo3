import numpy as np
import pandas as pd
import tqdm
import random
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import re
import copy
from utils import compute_penalty, find_fad_lad


class CleaningConfigPseudo(object):
    def __init__(self):
        self.iterations = None  # {None, int}, specify the number of iterations or it will be automatically determined.
        self.section_correction_path = None  # {None, PATH}, the path to the correction table.
        self.taxa_correction_path = None  # {None, PATH}, the path to the correction table.
        self.within_occ_correction = None  # {str, None}. Specify a column name
        self.groups_to_delete = []  # {list, PATH}, will delete these groups of taxa
        self.taxonomic_check = False  # {True, False}, if False, then no taxonomic check and change
        self.section_threshold = None  # {None, int}, occ will be deleted if occurred in sections less than threshold.
        self.isolated_section = 'delete'  # {'reserve', 'delete'}, for sections that don't share taxa with the majority.
        self.taxa_in_section_threshold = None  # {None, int}, section will be deleted if with less taxa than threshold.


class Species:
    def __init__(self, birth_rate, extinction_rate, parent_id, species_id, birth_time, max_extinction_time,
                 max_spatial_range, avg_spatial_range, environment_types):
        self.birth_rate = birth_rate
        self.extinction_rate = extinction_rate
        self.parent_id = parent_id  # 父物种的ID
        self.species_id = species_id  # 本物种的唯一ID

        # temporal range
        self.birth_time = birth_time
        self.extinction_time = max_extinction_time

        # spatial range
        random_ranges = avg_spatial_range * np.random.normal(1, 0.5, size=2)
        random_ranges[random_ranges > max_spatial_range] = max_spatial_range
        starts = np.random.uniform(0, 1, size=2) * (max_spatial_range - random_ranges)
        ends = starts + random_ranges
        self.spatial_range_lat = [starts[0], ends[0]]
        self.spatial_range_long = [starts[1], ends[1]]

        self.environment_type = np.random.choice(np.arange(environment_types))


class Horizon:
    def __init__(self, section_id, within_section_id, mean_time, span, depth_from, depth_to,
                 temporal_range, spatial_range_lat, spatial_range_long, environment_type, species_ids):
        self.section_id = section_id
        self.within_section_id = within_section_id
        self.mean_time = mean_time
        self.span = span
        self.depth_from = depth_from
        self.depth_to = depth_to
        self.temporal_range = temporal_range
        self.spatial_range_lat = spatial_range_lat
        self.spatial_range_long = spatial_range_long
        self.environment_type = environment_type
        self.species_ids = species_ids
        self.num_species = len(species_ids)


class Section:
    def __init__(self, section_id, avg_horizon_interval, max_temporal_range, avg_temporal_range,
                 avg_time_average_factor,
                 max_spatial_range, avg_spatial_range, environment_types,
                 species_legacy, avg_sampling_factor, avg_accumulation_rate,
                 avg_section_environment_types):
        self.section_id = section_id

        # add more random
        self.avg_section_horizon_interval = int(randomize_thing(avg_horizon_interval)) + 1
        self.avg_section_time_average_factor = randomize_thing(avg_time_average_factor)
        self.avg_section_sampling_factor = randomize_thing(avg_sampling_factor)
        self.avg_section_accumulation_rate = randomize_thing(avg_accumulation_rate)

        # temporal range
        if avg_temporal_range > max_temporal_range:
            raise ValueError('avg_temporal_range is larger than max_temporal_range!')
        while True:
            section_temporal_range = int(randomize_thing(avg_temporal_range))
            if section_temporal_range < max_temporal_range:
                break

        lower_limit = int(np.random.uniform(max_temporal_range - section_temporal_range))
        upper_limit = lower_limit + section_temporal_range
        self.temporal_range = [lower_limit, upper_limit]

        # spatial range
        lat_and_long = np.random.randint(max_spatial_range, size=2)
        random_range = (avg_spatial_range * (1 + np.random.uniform(-1, 1, size=2))).astype(int)
        _lat_and_long = lat_and_long + random_range
        _lat_and_long[_lat_and_long > max_spatial_range] = max_spatial_range
        self.spatial_range_lat = [lat_and_long[0], _lat_and_long[0]]
        self.spatial_range_long = [lat_and_long[1], _lat_and_long[1]]

        section_environment_types = int(randomize_thing(avg_section_environment_types))
        section_environment_types = 1 if section_environment_types < 1 else section_environment_types
        section_environment_types = environment_types if section_environment_types > environment_types else section_environment_types
        self.environment_type = np.random.choice(np.arange(environment_types), size=section_environment_types,
                                                 replace=False)

        # Find potential species
        section_potential_species = []
        for species in species_legacy:
            if species.environment_type in self.environment_type:
                if not (species.extinction_time < lower_limit or species.birth_time > upper_limit):
                    if not (species.spatial_range_lat[1] < self.spatial_range_lat[0] or
                            species.spatial_range_lat[0] > self.spatial_range_lat[1]):
                        if not (species.spatial_range_long[1] < self.spatial_range_long[0] or
                                species.spatial_range_long[0] > self.spatial_range_long[1]):
                            section_potential_species.append(species)

        # build horizons
        self.horizons = []
        self.species = []
        current_time = lower_limit
        current_horizon_id = 0
        depth_from = 0.0
        while True:
            time_diff = int(np.random.normal(self.avg_section_horizon_interval, self.avg_section_horizon_interval))
            time_diff = time_diff if time_diff > 0 else 1
            current_time += time_diff
            if current_time > upper_limit:
                break
            horizon_span = int(abs(
                np.random.normal(self.avg_section_time_average_factor, self.avg_section_time_average_factor)))
            horizon_lower_limit = current_time - horizon_span
            horizon_upper_limit = current_time + horizon_span
            horizon_lower_limit = horizon_lower_limit if horizon_lower_limit >= lower_limit else lower_limit
            horizon_upper_limit = horizon_upper_limit if horizon_upper_limit <= upper_limit else upper_limit

            depth_from += time_diff * np.abs(
                np.random.normal(self.avg_section_accumulation_rate, self.avg_section_accumulation_rate))
            depth_to = depth_from + horizon_span * np.abs(
                np.random.normal(self.avg_section_accumulation_rate, self.avg_section_accumulation_rate))

            horizon_potential_species = []
            for species in section_potential_species:
                if not (species.extinction_time < horizon_lower_limit or species.birth_time > horizon_upper_limit):
                    horizon_potential_species.append(species.species_id)
            num_horizon_potential_species = len(horizon_potential_species)
            if num_horizon_potential_species:
                sample_size = int(len(horizon_potential_species) * randomize_thing(self.avg_section_sampling_factor))
                if sample_size > 0:
                    sample_size = len(horizon_potential_species) if sample_size > len(
                        horizon_potential_species) else sample_size
                    sampled_species = np.random.choice(horizon_potential_species, size=sample_size, replace=False)
                    self.species.extend(sampled_species)
                    self.horizons.append(
                        Horizon(self.section_id, current_horizon_id, current_time, horizon_span, depth_from, depth_to,
                                [horizon_lower_limit, horizon_upper_limit],
                                self.spatial_range_lat, self.spatial_range_long, self.environment_type,
                                sampled_species))
                    current_horizon_id += 1
                else:
                    pass
            else:
                pass
            depth_from = depth_to
        self.num_horizons = len(self.horizons)
        self.species = np.unique(self.species)
        self.num_species = len(self.species)



def clean_section(df, del_log, args):
    # process section correction data
    sections_to_keep = []
    if args.section_correction_path is not None:
        section_correction_df = pd.read_csv(args.section_correction_path)
        section_taxa_groups = [i for i in df.groupby('Section code')['Fossil name']]
        section_deleted = []
        print('Searching for sections in section correction list...')
        potential_lost_taxa = pd.Series(dtype='object')
        for i, sec in enumerate(section_taxa_groups):
            section_code = sec[0]
            section_in_list = section_code == section_correction_df.loc[:, 'Section code']
            if np.sum(section_in_list):
                section_in_ha = df['Section code'].isin([section_code])
                behavior = section_correction_df.loc[section_in_list, 'Behavior'].values[0]
                if 'delete' in behavior:
                    section_deleted.append(section_code)
                    potential_lost_taxa = pd.concat((potential_lost_taxa, sec[1]))
                    continue
                if 'correct section code' in behavior:
                    corrected_section_name = section_correction_df.loc[section_in_list, 'Corrected section code'].values[0]
                    df.loc[section_in_ha, 'Section code'] = corrected_section_name
                    section_code = corrected_section_name
                if 'correct location' in behavior:
                    cor_lat = section_correction_df.loc[section_in_list, 'Corrected latitude'].values[0]
                    cor_lon = section_correction_df.loc[section_in_list, 'Corrected longitude'].values[0]
                    if not pd.isna(cor_lat):
                        df.loc[section_in_ha, 'Latitude'] = cor_lat
                        df.loc[section_in_ha, 'Longitude'] = cor_lon
                if 'keep' in behavior:
                    sections_to_keep.append(section_code)
        del_index = df['Section code'].isin(section_deleted)
        if len(del_index):
            del_log['within section deletion list'] += [(i, j) for i, j in zip(df[del_index]['Fossil name'].index,
                                                                               df[del_index]['Fossil name'].values)]
            df = df[~del_index]
            print('\tSections deleted:', section_deleted)

    if args.taxa_in_section_threshold is not None:
        section_taxa_groups = [i for i in df.groupby('Section code')['Fossil name']]
        section_deleted = []
        print('Searching for sections with less than %d taxa...' % args.taxa_in_section_threshold)
        potential_lost_taxa = pd.Series(dtype='object')
        for i, sec in enumerate(section_taxa_groups):
            if len(sec[1].unique()) < args.taxa_in_section_threshold:
                if sec[0] not in sections_to_keep:
                    section_deleted.append(sec[0])
                    potential_lost_taxa = pd.concat((potential_lost_taxa, sec[1]))
        del_index = df['Section code'].isin(section_deleted)
        if len(del_index):
            del_log['within small sections'] += [(i, j) for i, j in zip(df[del_index]['Fossil name'].index,
                                                                        df[del_index]['Fossil name'].values)]
            df = df[~del_index]
            print('\tSections deleted:', section_deleted)

    if args.isolated_section == 'delete':
        section_taxa_groups = [i for i in df.groupby('Section code')['Fossil name']]

        print('Searching for isolated sections...')
        graph = nx.Graph()
        section_names = np.unique(df['Section code'])
        graph.add_nodes_from(section_names)
        for i in range(len(section_taxa_groups)):
            for j in range(i, len(section_taxa_groups)):
                i_taxa = section_taxa_groups[i][1].unique()
                j_taxa = section_taxa_groups[j][1].unique()
                if len(np.intersect1d(i_taxa, j_taxa)):
                    graph.add_edge(section_taxa_groups[i][0], section_taxa_groups[j][0])
        components = [i for i in nx.connected_components(graph)]
        if len(components) > 1:
            del_sections = []
            components.sort(key=lambda x: len(x), reverse=True)
            for component in components[1:]:  # delete those not connected the major component
                for sec in component:
                    if sec not in sections_to_keep:
                        del_sections.append(sec)
            del_index = df['Section code'].isin(del_sections)
            if len(del_index):
                del_log['within isolated sections'] += [(i, j) for i, j in zip(df[del_index]['Fossil name'].index,
                                                                               df[del_index]['Fossil name'].values)]
                print('\tSections deleted:', del_sections)
                df = df[~del_index]
        else:
            pass

    return df, del_log


def clean_occurrence(df, args):
    if 'Section code' not in df.columns:
        df['Section code'] = df['Section name'].astype(str) + '_' + df['Section no.'].astype(str)
    if args.iterations:
        max_iteration = args.iterations
    else:
        max_iteration = -1
    iteration = 0

    # make within occ correction
    if args.within_occ_correction is not None:
        if type(args.within_occ_correction) is list:
            correction_columns = args.within_occ_correction
        else:
            correction_columns = [args.within_occ_correction]
        for correction_column in correction_columns:
            if correction_column == 'my opinion':
                correction_column = 'Fossil name-revised（my opinion）'
            elif correction_column == 'my liked opinion':
                import warnings
                warnings.warn("Using 'my liked opinion' is not tested yet, please check")
                correction_column = 'Fossil name-revised（my liked opinion）'
            elif correction_column == 'newest opinion':
                # in the newest opinions, the values are with author name and date that should be deleted
                correction_column = 'Fossil name-revised（newest opinion）'
                df[correction_column] = df[correction_column].apply(
                    lambda x: ' '.join(str(x).split()[:-2]) if not pd.isna(x) else x
                )
            cor_index = ~df[correction_column].isna()
            df.loc[cor_index, 'Fossil name'] = df.loc[cor_index, correction_column]

    if args.section_correction_path is not None:
        corr = pd.read_csv(args.section_correction_path, encoding='utf_8_sig')
        sec_original_names = corr.loc[:, 'Section code']
        if len(np.unique(sec_original_names)) < len(sec_original_names):
            unique_elements, counts = np.unique(sec_original_names, return_counts=True)
            print('Multiple section correction detected! These:')
            print(unique_elements[counts > 1])
            exit(-1)

    if args.taxa_correction_path is not None:
        taxon_corr = pd.read_csv(args.taxa_correction_path, encoding='utf_8_sig')
        original_taxon_names = taxon_corr.loc[:, 'Taxon name']
        if len(np.unique(original_taxon_names)) < len(original_taxon_names):
            unique_elements, counts = np.unique(original_taxon_names, return_counts=True)
            print('Multiple taxa correction detected! These:')
            print(unique_elements[counts > 1])
            exit(-1)
    else:
        taxon_corr = None

    if args.groups_to_delete is not None and len(args.groups_to_delete):
        if type(args.groups_to_delete) is str:
            delete_group = pd.read_csv(args.groups_to_delete, na_values='').to_numpy()
        else:
            delete_group = args.groups_to_delete
    else:
        delete_group = None

    delete_log = {'empty fossil name': [],
                  'empty fossil group': [],
                  'invalid groups': [],
                  'unknown': [],
                  'above species': [],
                  'subgenus': [],
                  '? gen sp or gen sp ?': [],
                  'genus ? sp': [],
                  'quotation': [],
                  'sp.': [],
                  'gen sp ssp': [],
                  'gen./sp. n./nov.': [],
                  'var': [],
                  'inferred ssp and var': [],
                  'indet': [],
                  'aff': [],
                  'cf': [],
                  'f': [],
                  'ex gr.': [],
                  'ex. interc.': [],
                  'transitional': [],
                  'without enough sections': [],
                  'within taxa deletion list': [],
                  'within isolated sections': [],
                  'within small sections': [],
                  'within section deletion list': []
                  }

    while True:
        while True:
            print('iteration %d in progress...' % iteration)
            original_delete_log = copy.deepcopy(delete_log)
            taxa_to_keep = set()
            if args.taxonomic_check:
                # 遍历所有occurrence
                delete_list = []
                j = df.columns.get_loc('Fossil name')
                for i in range(df.shape[0]):

                    ori_taxon = taxon = df.iloc[i, j]
                    row_index = df.iloc[i].name

                    # Delete those with empty 'Fossil name'
                    if pd.isna(taxon) or taxon.strip() == '':
                        delete_list.append(row_index)
                        delete_log['empty fossil name'] += [(row_index, ori_taxon)]
                        continue

                    # 对taxon进行基本处理
                    # 1. strip操作
                    taxon = taxon.strip()
                    # 2. 将所有符号改成英文半角符号
                    taxon = taxon.replace('（', '(').replace('）', ')')  # 替换中文括号为英文半角
                    taxon = taxon.replace('“', '\"').replace('”', '\"')  # 替换中文引号为英文半角
                    taxon = taxon.replace('‘', '\'').replace('’', '\'')  # 替换中文引号为英文半角
                    taxon = taxon.replace('。', '.').replace('，', ',')  # 替换中文句号和逗号为英文半角
                    # 3. 让每个括号前后有且只有一个空格
                    taxon = re.sub(r'\s*\(\s*', ' (', taxon)  # 在左括号前确保有且只有一个空格
                    taxon = re.sub(r'\s*\)\s*', ') ', taxon)  # 在右括号后确保有且只有一个空格
                    taxon = taxon.strip()  # 再次strip以去除可能在字符串末尾出现的空格

                    df.iloc[i, j] = taxon

                    # make correction
                    if taxon_corr is not None:
                        taxon_in_list = taxon_corr['Taxon name'] == taxon
                        if np.sum(taxon_in_list):
                            behavior = taxon_corr.loc[taxon_in_list, 'Behavior'].values[0]
                            if 'delete' in behavior:
                                delete_list.append(row_index)
                                delete_log['within taxa deletion list'] += [(row_index, ori_taxon)]
                                continue
                            else:
                                if 'keep' in behavior:
                                    taxa_to_keep.add(taxon)
                                    continue
                                else:
                                    if 'correct taxon name' in behavior:
                                        taxon = taxon_corr.loc[taxon_in_list, 'Corrected taxon name'].values[0]
                                        df.iloc[i, j] = taxon
                                    if 'correct group' in behavior:
                                        df.iloc[i, j - 1] = taxon_corr.loc[taxon_in_list, 'Corrected group'].values[0]

                    # 删除group
                    if delete_group is not None:
                        if df.iloc[i, j - 1] in delete_group:
                            delete_list.append(row_index)
                            delete_log['invalid groups'] += [(row_index, ori_taxon)]
                            continue

                    # 删除空group
                    if args.empty_group == 'delete' and pd.isna(df.iloc[i, j - 1]):
                        delete_list.append(row_index)
                        delete_log['empty fossil group'] += [(row_index, ori_taxon)]
                        continue

                    # 删除unknown
                    if args.unknown == 'delete' and "nknown" in taxon:  # 这样写是防止大小写出问题
                        delete_list.append(row_index)
                        delete_log['unknown'] += [(row_index, ori_taxon)]
                        continue

                    # 删除属及以上分类单元
                    if args.above_species == 'delete':
                        # 检测空格的次数；是否有种名（最后一个部分是否以小写开头）
                        if taxon.count(" ") == 0 or \
                                (not re.match(r'[a-z].*$', taxon.split(' ')[-1])) and \
                                (not re.match(r'\?', taxon.split(' ')[-1]) and
                                 not re.match(r'sp\.', taxon.split(' ')[-2])):
                            delete_list.append(row_index)
                            delete_log['above species'] += [(row_index, ori_taxon)]
                            continue

                    # 对亚属进行处理
                    if re.search(r'\(.*\)', taxon):
                        if args.subgenus == "delete":
                            delete_list.append(row_index)
                            delete_log['subgenus'] += [(row_index, ori_taxon)]
                            continue
                        elif args.subgenus == "merge":
                            taxon = re.sub(r' *\(.*\) *', " ", taxon)
                            df.iloc[i, j] = taxon

                    # 对有？的进行处理
                    if "?" in taxon:
                        # 对？gen sp和gen sp ？进行处理
                        if re.search('^\?', taxon) or re.search('\?$', taxon):
                            if args.query == "delete":
                                delete_list.append(row_index)
                                delete_log['? gen sp or gen sp ?'] += [(row_index, ori_taxon)]
                                continue
                            elif args.query == "merge":
                                taxon = re.sub('\? +', "", taxon)
                                taxon = re.sub(' +\?', "", taxon)
                                taxon = re.sub('\?', "", taxon)
                                df.iloc[i, j] = taxon
                        # 对gen ？sp进行处理
                        else:
                            if args.genqsp == "delete":
                                delete_list.append(row_index)
                                delete_log['genus ? sp'] += [(row_index, ori_taxon)]
                                continue
                            elif args.genqsp == "merge":
                                taxon = re.sub(' *\? *', " ", taxon)
                                df.iloc[i, j] = taxon

                    # 对有引号的进行处理
                    if "\'" in taxon or "\"" in taxon:
                        if args.quotation == "delete":
                            delete_list.append(row_index)
                            delete_log['quotation'] += [(row_index, ori_taxon)]
                            continue
                        elif args.quotation == "merge":
                            df.iloc[i, j] = re.sub(r'\"', "", taxon)
                            df.iloc[i, j] = re.sub(r'\'', "", taxon)

                    # 对ssp.进行处理
                    if "ssp." in taxon or "subsp." in taxon:
                        if args.ssp == "delete":
                            delete_list.append(row_index)
                            delete_log['gen sp ssp'] += [(row_index, ori_taxon)]
                            continue
                        if args.ssp == "merge":  # 此处的merge是删去亚种名，合并入种
                            df.iloc[i, j] = re.sub(r' +ssp\..*', "", taxon)
                            df.iloc[i, j] = re.sub(r' +subsp\.*', "", taxon)
                        elif args.ssp == "reserve":  # 此处的reserve对应操作不是完全保留，而是删去标记后保留
                            df.iloc[i, j] = re.sub(r' +ssp\. +', " ", taxon)
                            df.iloc[i, j] = re.sub(r' +subsp\. +', " ", taxon)

                    # 对sp.进行处理
                    if "sp." in taxon or "spp." in taxon:
                        if "n." in taxon or "nov." in taxon or "indet." in taxon:
                            pass
                        elif args.sp == "delete":
                            delete_list.append(row_index)
                            delete_log['sp.'] += [(row_index, ori_taxon)]
                            continue

                    # 对新种新属进行处理
                    if "n." in taxon or "nov." in taxon:
                        if args.nov == "delete":
                            delete_list.append(row_index)
                            delete_log['gen./sp. n./nov.'] += [(row_index, ori_taxon)]
                            continue
                        elif args.nov == "merge":
                            taxon = re.sub(r' +n\. gen\., n\. sp\..*', "", taxon)
                            taxon = re.sub(r' +n\. gen\. et n\. sp\..*', "", taxon)
                            taxon = re.sub(r' +n\. sp\..*', "", taxon)
                            taxon = re.sub(r' +n\. gen\.', "", taxon)
                            taxon = re.sub(r' +gen\. nov\., sp\. nov\..*', "", taxon)
                            taxon = re.sub(
                                r' +gen\. nov\. et sp\. nov\..*', "", taxon)
                            taxon = re.sub(r' +sp\. *nov\..*', "", taxon)
                            df.iloc[i, j] = re.sub(r' +gen\. *nov\.+', "", taxon)

                    # 对var.进行处理
                    if "var." in taxon:
                        if args.var == "delete":
                            delete_list.append(row_index)
                            delete_log['var'] += [(row_index, ori_taxon)]
                            continue
                        elif args.var == "merge":
                            df.iloc[i, j] = re.sub(r' var\..*', "", taxon)

                    # 对不含有标记的推测的亚种、变种进行处理
                    if len(taxon.split(' ')) > 1:
                        if taxon.count(" ") >= (2 + taxon.split(' ')[1].istitle()) and '.' not in taxon:
                            if args.inferred_ssp_var == 'delete':
                                delete_list.append(row_index)
                                delete_log['inferred ssp and var'] += [(row_index, ori_taxon)]
                                continue
                            elif args.inferred_ssp_var == 'merge':
                                pattern = r' [a-z]*'
                                match = re.search(pattern, taxon)
                                df.iloc[i, j] = taxon[:match.end()] if match else taxon

                    # 对indet.进行处理
                    if "indet." in taxon:
                        if args.indet == "delete":
                            delete_list.append(row_index)
                            delete_log['indet'] += [(row_index, ori_taxon)]
                            continue
                        elif args.indet == "merge":
                            df.iloc[i, j] = re.sub(r' indet\..*', "", taxon)

                    # 对f.进行处理
                    if "f." in taxon:
                        if "aff." in taxon or "cf." in taxon:
                            pass
                        elif args.f == "delete":
                            delete_list.append(row_index)
                            delete_log['f'] += [(row_index, ori_taxon)]
                            continue
                        elif args.f == "merge":
                            taxon = re.sub(r' f\. ', " ", taxon)
                            taxon = re.sub(r'f\. ', "", taxon)
                            taxon = re.sub(r' f\.', "", taxon)
                            df.iloc[i, j] = taxon

                    # 对aff.进行处理
                    if "aff." in taxon:
                        if args.aff == "delete":
                            delete_list.append(row_index)
                            delete_log['aff'] += [(row_index, ori_taxon)]
                            continue
                        elif args.aff == "merge":
                            taxon = re.sub(r' aff\. ', " ", taxon)
                            taxon = re.sub(r'aff\. ', "", taxon)
                            taxon = re.sub(r' aff\.', "", taxon)
                            df.iloc[i, j] = taxon

                    # 对cf.进行处理
                    if "cf." in taxon:
                        if args.cf == "delete":
                            delete_list.append(row_index)
                            delete_log['cf'] += [(row_index, ori_taxon)]
                            continue
                        elif args.cf == "merge":
                            taxon = re.sub(r' cf\. ', " ", taxon)
                            taxon = re.sub(r' cf\.', "", taxon)
                            taxon = re.sub(r'cf\. ', "", taxon)
                            df.iloc[i, j] = taxon

                    # 对ex gr.进行处理
                    if "gr." in taxon:
                        if args.exgr == "delete":
                            delete_list.append(row_index)
                            delete_log['ex gr.'] += [(row_index, ori_taxon)]
                            continue
                        elif args.exgr == "merge":
                            taxon = re.sub(r' *ex gr\. *', " ", taxon)
                            taxon = re.sub(r' *gr\. *', " ", taxon)
                            df.iloc[i, j] = taxon

                    # 对ex. interc.进行处理
                    if " ex" in taxon and " interc" in taxon:
                        if args.exinterc == "delete":
                            delete_list.append(row_index)
                            delete_log['ex. interc.'] += [(row_index, ori_taxon)]
                            continue
                        elif args.exinterc == 'former':
                            taxon = re.sub(r' ex\. interc\..*', "", taxon)
                            df.iloc[i, j] = taxon

                    # 对过渡物种进行处理
                    if "-" in taxon and "interc." not in taxon and "gr." not in taxon:
                        if args.trans == "delete":
                            delete_list.append(row_index)
                            delete_log['transitional'] += [(row_index, ori_taxon)]
                            continue
                        elif args.trans == 'former':
                            taxon = re.sub(r'-.*', "", taxon)
                            df.iloc[i, j] = taxon
                        elif args.trans == 'latter':
                            if ' ' not in taxon.split('-')[-1]:
                                taxon = re.sub(r' [a-z]*-', " ", taxon)
                            else:
                                taxon = re.sub(r'[A-Z]\.', taxon.split()[0], taxon)
                                taxon = re.sub(r'.*-', "", taxon)
                            df.iloc[i, j] = taxon
                df = df.drop(delete_list)
                iteration += 1
                if iteration == max_iteration:
                    break
                elif delete_log == original_delete_log:
                    max_iteration = iteration + 1  # buffer one round
            else:
                break

        if args.section_threshold is not None:
            print('Searching for taxa occurred in less than %d sections...' % args.section_threshold)
            _df = df.sort_values('Section code')
            section_dict = {}
            for index, row in _df.iterrows():
                fossil_name = row['Fossil name']
                if fossil_name in taxa_to_keep:
                    continue

                section_code = row['Section code']

                if fossil_name not in section_dict:
                    section_dict[fossil_name] = [1, section_code]
                else:
                    if section_dict[fossil_name][1] != section_code:
                        section_dict[fossil_name][0] += 1
                        section_dict[fossil_name][1] = section_code
            filtered_keys = [key for key, value in section_dict.items() if value[0] < args.section_threshold]
            del_index = df['Fossil name'].isin(filtered_keys)
            delete_log['without enough sections'] += [(i, j) for i, j in zip(df[del_index]['Fossil name'].index,
                                                                             df[del_index]['Fossil name'].values)]
            df = df[~del_index]
        if (args.isolated_section == 'delete') or (args.taxa_in_section_threshold is not None) or (args.section_correction_path is not None):
            df, delete_log = clean_section(df, delete_log, args)
            if delete_log == original_delete_log:
                break
        else:
            break

    # print(delete_log)
    return df, delete_log


def make_score(depths, method='sigmoid'):
    """
    Generate score based on depth data.

    :param depths: Array, depths series data.
    :param method: {'linear', 'sigmoid'}. Method to generate score based on depth.
    :return: Horizon score array.
    """
    depths = pd.Series(depths) if not isinstance(depths, pd.Series) else depths
    if len(depths.drop_duplicates()) > 1:
        if method == 'sigmoid':  # sigmoid function normalization
            zscore = (depths - depths.mean()) / depths.std()  # first to z score
            score = 1.0 / (1 + np.exp(-zscore))  # then to sigmoid value
        else:  # linear way for normalization
            score = (depths - depths.min()) / (depths.max() - depths.min())
    else:  # only one depth
        score = np.full_like(depths, np.random.uniform(0, 1))  # randomly select one score from 0-1
    return score


def build_horizon(occurrence_df, attribute_column=10, score=True, differ_inner=True, method='uniform',
                  only_events=False, separate_events=False) -> pd.DataFrame:
    """
    Build the horizon dataframe.

    :param attribute_column: int. From which column on are horizon attributes.
    :param differ_inner: bool. If set True, implied occurrences will be coded as '2', otherwise '0'.
    :param occurrence_df: pandas.DataFrame, Occurrence dataframe.
    :param score: bool, whether to include score.
    :param method: {'uniform', 'linear', 'sigmoid', 'time_based'}.
    :param only_events: bool. If set True, only reserve first and last occurrences.
    :param separate_events: bool. Only used when only_events is True. Whether to separate events in the same horizon into individual horizons.
    :return: Horizon dataframe.
    """

    # check if there is 'Section code'
    if 'Section code' not in occurrence_df.columns:
        occurrence_df['Section code'] = occurrence_df['Section name'] + '_' + occurrence_df['Section no.']

    # decide depth method
    if 'Depth from' in occurrence_df.columns and 'Depth to' in occurrence_df.columns:
        print('Depth available. Infer order from depth.')
    else:
        print('Depth not available! Assuming orders of dataframe rows as collection orders.')
        occurrence_df['Depth from'] = 0
        occurrence_df['Depth to'] = 0

    # dealing with empty fossil names
    nan_index = np.where(occurrence_df['Fossil name'].isna())[0]
    if len(nan_index) > 0:
        print('Detected %d empty fossil name(s), corresponding occurrences will be neglected.' % len(nan_index))
        occurrence_df = occurrence_df.drop(nan_index)

    horizon_attributes = list(occurrence_df.columns[attribute_column:])
    occurrence_df.insert(0, 'Col_Dep', occurrence_df['Collection'].astype(str) + '_' + occurrence_df['Depth from'].astype(str))
    horizon_head = ['Section', 'Score'] + horizon_attributes + ['*']
    horizon_df = pd.DataFrame(columns=horizon_head)
    taxon_column_index = len(horizon_attributes)

    # building procedure
    last_section_no = ''
    section_df = pd.DataFrame(columns=horizon_attributes)  # initialize a section dataframe
    horizon_index = 0
    collections = occurrence_df.groupby(['Section code', 'Col_Dep'])
    section = 1

    time_range = None
    if method == 'time_based':
        if 'MinFmAge' in occurrence_df.columns:
            occurrence_df['MaxFmAge'] = occurrence_df['MaxFmAge'] + 0.01  # separate min and max
            occurrence_df['MinFmAge'] = occurrence_df['MinFmAge'] - 0.01  # separate min and max
            time_range = (occurrence_df['MaxFmAge'].max() + 0.01, occurrence_df['MinFmAge'].min() - 0.01)
            print('Use empirical formation age data for score assignment.')
        else:
            raise ValueError('No time data available!')
    for i, collection in enumerate(collections):
        # extract collection info.
        collection_content = collection[1]
        horizon_attr = collection_content[horizon_attributes].iloc[0, :]
        section_no = collection[0][0]
        if i == 0:
            last_section_no = section_no
        taxa_in_collection = collection_content['Fossil name'].drop_duplicates()
        # build a new horizon (as Series)
        if differ_inner:
            horizon = pd.concat(
                [horizon_attr, pd.Series(np.full_like(taxa_in_collection, -1), index=taxa_in_collection)])
        else:
            horizon = pd.concat([horizon_attr, pd.Series(np.ones_like(taxa_in_collection), index=taxa_in_collection)])

        if section_no == last_section_no and i != (len(collections) - 1):
            section_df = pd.concat([section_df, horizon.to_frame().T])  # append the new horizon to section horizons
        else:
            if i == (len(collections) - 1):
                section_df = pd.concat([section_df, horizon.to_frame().T])

            section_df = section_df.sort_values(by='Depth from').reset_index(drop=True)

            # fill all unseens as 0
            section_df[section_df.columns[taxon_column_index:]] = section_df.iloc[:, taxon_column_index:].fillna(0)

            # plus 1 for all present and implied-present horizon cells
            taxa_in_section = section_df.columns[taxon_column_index:]

            if only_events:
                present = -1 if differ_inner else 1
                fad_lad_indices = []
                for _taxon in taxa_in_section:
                    one_indices = section_df[section_df[_taxon] == present].index  # find taxa depth base
                    fad_lad_indices += [one_indices.min(), one_indices.max()]
                if separate_events:
                    section_df = section_df.loc[fad_lad_indices, :].reset_index(drop=True)
                    section_df.iloc[:, taxon_column_index:] = 0
                    for j, _taxon in enumerate(taxa_in_section):
                        section_df.loc[2 * j, _taxon] = present
                        section_df.loc[2 * j + 1, _taxon] = present
                else:
                    fad_lad_indices = np.unique(fad_lad_indices)
                    section_df = section_df.loc[fad_lad_indices, :].reset_index(drop=True)
                section_df = section_df.sort_values('Depth from').reset_index(drop=True)

            if differ_inner:
                for _taxon in taxa_in_section:
                    _index = np.where(section_df.loc[:, _taxon] == -1)[0]
                    section_df.loc[range(_index[0], _index[-1] + 1), _taxon] += 2

            # add score
            if score:
                if method == 'uniform':
                    scores = np.sort(np.random.uniform(0, 1, len(section_df)))  # randomly decide scores
                elif method == 'linear' or method == 'sigmoid':
                    section_df = section_df.reset_index(drop=True)
                    scores = make_score(section_df['Depth from'], method)  # decide scores based on depth
                elif method == 'time_based':
                    section_df = section_df.reset_index(drop=True)
                    with_fm_age_idx = section_df['MinFmAge'].notna()
                    if with_fm_age_idx.sum() == 0:
                        # If all horizons have no formation age data, use uniform distribution
                        scores = np.sort(np.random.uniform(0, 1, len(section_df)))
                    elif with_fm_age_idx.sum() == 1:
                        min_fm_age_p = 1 - (section_df['MinFmAge'] - time_range[1]) / (time_range[0] - time_range[1])
                        max_fm_age_p = 1 - (section_df['MaxFmAge'] - time_range[1]) / (time_range[0] - time_range[1])
                        if len(with_fm_age_idx) == 1:
                            # if there is only one horizon with formation age data, use proportional distribution
                            scores = np.random.uniform(min_fm_age_p, max_fm_age_p)
                        else:
                            # if there are more than one horizon but only one with formation age data,
                            # use proportional distribution for the one with formation age data
                            # and uniform distribution for the rest
                            separate_idx = np.where(with_fm_age_idx)[0][0]
                            scores_middle = np.random.uniform(min_fm_age_p[separate_idx], max_fm_age_p[separate_idx])
                            scores_before = np.sort(np.random.uniform(0, max_fm_age_p[separate_idx], separate_idx))
                            scores_after = np.sort(np.random.uniform(min_fm_age_p[separate_idx], 1,
                                                                     len(section_df) - separate_idx - 1))
                            scores = np.concatenate([scores_before, [scores_middle], scores_after])
                    else:
                        min_fm_age_p = 1 - (section_df['MinFmAge'] - time_range[1]) / (time_range[0] - time_range[1])
                        max_fm_age_p = 1 - (section_df['MaxFmAge'] - time_range[1]) / (time_range[0] - time_range[1])
                        if len(with_fm_age_idx) - with_fm_age_idx.sum():
                            # perform linear interpolation for horizons without formation age data
                            xp = [-1] + np.where(with_fm_age_idx)[0].tolist() + [len(with_fm_age_idx)]
                            min_fm_age_p = np.interp(x=np.arange(len(with_fm_age_idx)),
                                                     xp=xp,
                                                     fp=[0] + min_fm_age_p[with_fm_age_idx].to_list() + [1])
                            max_fm_age_p = np.interp(x=np.arange(len(with_fm_age_idx)),
                                                     xp=xp,
                                                     fp=[0] + max_fm_age_p[with_fm_age_idx].to_list() + [1])
                        else:
                            pass
                        # for each horizon, draw a random time between its MaxFmAge and MinFmAge formation age
                        random_age_p = np.random.uniform(max_fm_age_p, min_fm_age_p)
                        scores = np.sort(random_age_p)
            else:
                scores = [0] * len(section_df)
            section_df['Score'] = scores

            # add section no.
            section_df['Section'] = section
            section += 1

            horizon_df = pd.concat([horizon_df, section_df])  # add section horizons to whole horizons

            # take this horizon as the new section dataframe and initialize
            section_df = horizon.to_frame().T
            last_section_no = section_no

        horizon_index += 1
        if horizon_index % 250 == 0:
            print('progress: ', horizon_index, '/', len(collections))

    horizon_df[horizon_df.columns[taxon_column_index + 3:]] = horizon_df.iloc[:, taxon_column_index + 3:].fillna(-1)
    horizon_df['*'] = '*'
    horizon_df.insert(0, 'Horizon', np.arange(1, len(horizon_df) + 1))  # horizon no
    print('completed.')
    return horizon_df.sort_values('Horizon')




def randomize_thing(ori, random_factor=1):
    return ori * random.uniform(1 - random_factor, 1 + random_factor)


def build_pseudo_horizons(species_legacy, num_sections, avg_horizon_interval, max_temporal_range, avg_temporal_range,
                          avg_time_average_factor, max_spatial_range, avg_spatial_range, environment_types,
                          avg_sampling_factor, avg_accumulation_rate, avg_section_environment_types):

    def process_section_df(section_df, section_info):
        this_section_info = section_info[section_df['Section'].iloc[0]]
        section_species = this_section_info.species
        for col in section_species:
            _col = section_df[col] == 0
            first_one = _col.idxmax()
            last_one = _col[::-1].idxmax()
            section_df[col].loc[:first_one - 1] = 0
            section_df[col].loc[last_one + 1:] = 0
            section_df[col].loc[first_one: last_one] = -section_df[col].loc[first_one: last_one] + 1
        section_df['Score'] = np.sort(np.random.uniform(0.01, 0.99, len(section_df)))
        return section_df

    # sampling simulation
    sections = []
    for section_id in range(num_sections):
        valid_section = False
        while not valid_section:
            section = Section(section_id, avg_horizon_interval, max_temporal_range, avg_temporal_range,
                              avg_time_average_factor, max_spatial_range, avg_spatial_range, environment_types,
                              species_legacy, avg_sampling_factor, avg_accumulation_rate, avg_section_environment_types)
            if section.num_species > 0:
                valid_section = True
                sections.append(section)

    sampled_species = []
    horizons = []
    for section in sections:
        sampled_species.extend(section.species)
        horizons.extend(section.horizons)

    sampled_species = np.unique(sampled_species)
    print(f"# species from legacy: {len(species_legacy)}")
    print(f"Original: # sections: {len(sections)}, # horizons: {len(horizons)}, # species : {len(sampled_species)}")

    # build occurrence dataframe
    fossil_names = []
    horizon_infos = []
    for horizon in horizons:
        fossil_names.extend(horizon.species_ids)
        horizon_infos.extend([[horizon.within_section_id, horizon.section_id, horizon.depth_from,
                               horizon.depth_to, horizon.mean_time, horizon.span]] * horizon.num_species)
    horizon_infos = np.array(horizon_infos)
    attribute_columns = ['Fossil name', 'Collection', 'Section code', 'Depth from', 'Depth to', 'Real time',
                         'Real span']
    occ_df = pd.DataFrame(np.zeros(shape=(len(fossil_names), len(attribute_columns)), dtype=int),
                          columns=attribute_columns)
    occ_df['Fossil name'] = fossil_names
    occ_df.loc[:, 'Collection': 'Real span'] = horizon_infos

    # clean occurrences
    print('-----')
    print('Cleaning occurrences...')
    cleaned_occ_df = clean_occurrence(occ_df, CleaningConfigPseudo())[0]
    print('Cleaning done.')
    cleaned_sampled_species = cleaned_occ_df['Fossil name'].unique()
    num_cleaned_horizons = len((cleaned_occ_df['Section code'] + cleaned_occ_df['Collection'] / cleaned_occ_df[
        'Collection'].max() / 10).unique())
    cleaned_sections = cleaned_occ_df['Section code'].unique()
    print('-----')
    print(
        f"Cleaned: # sections: {len(cleaned_sections)},"
        f" # horizons: {num_cleaned_horizons},"
        f" # species : {len(cleaned_sampled_species)}")

    # build horizon dataframe
    horizon_df = build_horizon(cleaned_occ_df, 1)
    horizon_df = horizon_df.sort_values('Real time').reset_index(drop=True)
    horizon_df['Horizon'] = np.arange(1, len(horizon_df) + 1)
    first_taxon_column_index = list(horizon_df.columns).index('*') + 1
    num_species_in_horizon = (horizon_df.iloc[:, first_taxon_column_index:] == 1).sum(axis=1)
    horizon_df.insert(first_taxon_column_index - 1, '# species', num_species_in_horizon)
    first_taxon_column_index += 1

    #
    # attribute_columns = ['Horizon', 'Section', 'Score', 'Depth from', 'Depth to', 'Real time', 'Real span', '# species',
    #                      '*']
    # first_taxon_column_index = len(attribute_columns)
    # species_columns = np.sort(sampled_species).tolist()
    # columns = attribute_columns + species_columns
    # horizon_df = pd.DataFrame(np.full(fill_value=-1, shape=(len(horizons), len(columns))), columns=columns)
    # horizon_df['*'] = '*'
    # for i, horizon in enumerate(horizons):
    #     horizon_df.loc[i, ['Section', 'Depth from', 'Depth to', 'Real time', 'Real span', '# species']] = (
    #         horizon.section_id, horizon.depth_from, horizon.depth_to, horizon.mean_time, horizon.span,
    #         horizon.num_species)
    #     horizon_df.loc[i, horizon.species_ids] = 0
    # horizon_df = horizon_df.groupby('Section', group_keys=False).apply(process_section_df, sections)
    # horizon_df = horizon_df.sort_values('Real time').reset_index(drop=True)
    # horizon_df['Horizon'] = np.arange(1, len(horizon_df) + 1)

    # compute the penalties
    mat = horizon_df.iloc[:, first_taxon_column_index:].to_numpy()
    real_penalty = compute_penalty(new_indices=horizon_df.index.to_numpy(), ori_mat=mat)
    total_spans = horizon_df['Real span'].sum()
    randomize_indices = np.argsort(horizon_df['Score'].to_numpy())
    randomize_spans = np.array(horizon_df['# species'] / 2, dtype=int)
    randomized_penalty = compute_penalty(new_indices=randomize_indices, ori_mat=mat)
    print(f'real_penalty: {real_penalty}; '
          f'total_span: {total_spans}')
    print(f'randomized_penalty: {randomized_penalty} '
          f'randomized_total_span: {randomize_spans.sum()}')

    # draw richness
    sampled_species_legacy = []
    for sp in species_legacy:
        if sp.species_id in sampled_species:
            sampled_species_legacy.append(sp)

    fads_lads = fads_lads_from_ha(horizon_df, horizon_df.index)
    species_counts, origination_rates, extinction_rates = [], [], []
    species_count = (fads_lads[0] == 0).sum()
    for level in range(len(horizon_df)):
        origination = (fads_lads[0] == level).sum()
        extinction = (fads_lads[1] == level).sum()
        species_counts.append(species_count + origination - extinction)
        origination_rates.append(origination / species_count)
        extinction_rates.append(extinction / species_count)
        species_count = species_counts[level]
    draw_richness(species_counts, origination_rates, extinction_rates)
    return horizon_df


def fads_lads_from_ha(horizon_df, indices):
    first_taxon_column = list(horizon_df.columns).index('*') + 1
    new_df = horizon_df.iloc[indices, first_taxon_column:]
    fads_lads = new_df.apply(find_fad_lad)
    return fads_lads.T


def simulate_species_dynamics(total_time_steps, initial_species,
                              global_birth_rate, global_extinction_rate,
                              environmental_capacity, mass_extinction_time, great_radiation_time,
                              spatial_range_limit, avg_spatial_range, environment_types):
    if len(mass_extinction_time) > 0:
        mass_extinction_time = np.array(mass_extinction_time)
        extinction_elapsed_time = np.random.randint(5, size=len(mass_extinction_time))
        mass_extinction_time_range = np.array([mass_extinction_time - extinction_elapsed_time,
                                               mass_extinction_time + extinction_elapsed_time])
    else:
        mass_extinction_time_range = np.array([[0], [0]])
    if len(great_radiation_time) > 0:
        great_radiation_time = np.array(great_radiation_time)
        radiation_elapsed_time = np.random.randint(20, size=len(great_radiation_time))
        great_radiation_time_range = np.array([great_radiation_time - radiation_elapsed_time,
                                               great_radiation_time + radiation_elapsed_time])
    else:
        great_radiation_time_range = np.array([[0], [0]])

    # 存储每一步的物种数量、新生率和灭绝率
    species_counts = []
    birth_rates = []
    extinction_rates = []

    # 初始化物种列表
    # 为初始物种分配ID，并存储所有物种的信息
    species_list = [Species(global_birth_rate, global_extinction_rate, parent_id=None, species_id=_,
                            birth_time=0, max_extinction_time=total_time_steps, max_spatial_range=spatial_range_limit,
                            avg_spatial_range=avg_spatial_range, environment_types=environment_types)
                    for _ in range(initial_species)]
    species_id_counter = initial_species
    species_tree_info = {species.species_id: species.parent_id for species in species_list}  # 新增：存储物种的父子关系
    species_legacy = species_list

    num_species = initial_species

    for step in tqdm.tqdm(range(total_time_steps)):
        if np.sum((mass_extinction_time_range[0] <= step) & (mass_extinction_time_range[1] >= step)):
            ext_adjustment_factor = 8 * random.random()
            ori_adjustment_factor = -3 * random.random()
        elif np.sum((great_radiation_time_range[0] <= step) & (great_radiation_time_range[1] >= step)):
            ext_adjustment_factor = -0.3 * random.random()
            ori_adjustment_factor = 0.6 * random.random()
        else:
            ext_adjustment_factor = -0.5 * (environmental_capacity - num_species) / environmental_capacity
            ori_adjustment_factor = -ext_adjustment_factor

        num_birth = 0
        num_extinct = 0
        new_species = []
        for species in species_list:
            species_birth_rate = species.birth_rate
            species_extinction_rate = species.extinction_rate

            # 判断物种是否灭绝
            if random.random() < species_extinction_rate * (1 + ext_adjustment_factor):
                num_extinct += 1
                species_legacy[species.species_id].extinction_time = step
                continue  # 物种灭绝，不进入新的物种列表

            if random.random() < species_birth_rate * (1 + ori_adjustment_factor):
                # 物种分化，为子物种分配ID，并记录它们的父物种
                num_birth += 1
                child = Species(randomize_thing(global_birth_rate), randomize_thing(global_extinction_rate),
                                 species.species_id, species_id_counter, step, total_time_steps,
                                 max_spatial_range=spatial_range_limit,
                                 avg_spatial_range=avg_spatial_range, environment_types=environment_types)
                species_id_counter += 1

                new_species.append(child)
                new_species.append(species)  # 保留父物种
                species_legacy.append(child)
                species_tree_info[child.species_id] = species.species_id  # 更新关系

            else:
                # 如果物种没有繁殖，仍然保持在列表中
                new_species.append(species)

        # 更新物种列表
        species_list = new_species

        # 计算当前步骤的数据
        birth_rate = num_birth / num_species
        extinction_rate = num_extinct / num_species
        num_species = len(species_list)

        # 记录数据
        species_counts.append(num_species)
        birth_rates.append(birth_rate)
        extinction_rates.append(extinction_rate)

    # 返回收集的数据
    return species_counts, birth_rates, extinction_rates, species_tree_info, species_legacy


def draw_phylogenetic_tree(species_tree_info):
    # 创建一个新的有向图，因为支系树是有方向的（从一个祖先物种到它的后代）
    G = nx.DiGraph()

    # 添加物种之间的边。在这里，我们根据物种和它们的父物种创建边。
    for species_id, parent_id in species_tree_info.items():
        if parent_id is not None:  # 如果有父物种
            G.add_edge(parent_id, species_id)

    # 为了更好地可视化，我们可以基于树的结构来布局节点。
    # 如果您不能使用它们，可以选择其他布局，如'spring_layout'，但可视化效果可能不理想。
    # pos = nx.spring_layout(G)  # 可以替换为 nx.nx_agraph.graphviz_layout(G) 使用Graphviz布局
    pos = nx.nx_agraph.graphviz_layout(G)

    # 绘制图形
    nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", arrowsize=5)

    # 显示图形
    plt.show()


def draw_richness(species_counts, origination_rates, extinction_rates):
    # 绘制物种数量的曲线图
    plt.figure(figsize=(14, 12))

    plt.subplot(3, 1, 1)
    plt.plot(species_counts, label="Richness")
    plt.xlabel('Time step')
    plt.ylabel('Number of Species')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(origination_rates, label="Origination rate", color='g')
    plt.xlabel('Time step')
    plt.ylabel('Origination rate')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(extinction_rates, label="Extinction rate", color='r')
    plt.xlabel('Time step')
    plt.ylabel('Extinction rate')
    plt.legend()

    plt.tight_layout()
    plt.show()


def draw_taxon_range_chart(fads: np.ndarray, lads: np.ndarray, reorder=False, arrangement='fad'):
    """
    Draw taxon range chart.
    :param arrangement: {'original', 'fad', 'lad'}. Arrange the taxa by what.
    :param fads:
    :param lads:
    :param reorder: bool.
    :return:
    """
    if arrangement == 'fad':
        sorted_index = np.argsort(fads)
    elif arrangement == 'lad':
        sorted_index = np.argsort(lads)
    else:
        sorted_index = np.arange(len(fads))

    if reorder:
        combined = np.concatenate((fads, lads))
        sorted_combined = np.sort(combined)
        fads = np.searchsorted(sorted_combined, fads)
        lads = np.searchsorted(sorted_combined, lads)

    # Prepare your data
    ranges = lads - fads

    # Draw the distribution of fad_lad['duration']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ranges, kde=True, ax=ax)
    ax.set_xlabel('Ranges')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Ranges')
    plt.show()

    mean_range, std_range, max_range, min_range, n_taxa = (
        ranges.mean(), ranges.std(), ranges.max(), ranges.min(), len(ranges))
    print(f'# taxa: {n_taxa}. Range mean: {mean_range}, std: {std_range}, max: {max_range}, min: {min_range}')

    # Create a figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # Add data to the figure
    ax.bar(x=sorted_index.astype(str), height=ranges[sorted_index], bottom=fads[sorted_index], color='gray')

    # Draw a line for fad and lad bars
    for i, index in enumerate(sorted_index):
        ax.hlines(y=fads[index],
                  xmin=i - 0.4, xmax=i + 0.4,
                  color='black', linewidth=0.5)
        ax.hlines(y=lads[index],
                  xmin=i - 0.4, xmax=i + 0.4,
                  color='black', linewidth=0.5)

    # Customize the chart
    ax.set_xlabel('Taxon')
    ax.set_ylabel('Range')

    ax.set_title('Taxon Range Chart')
    ax.tick_params(axis='x', rotation=-90)

    # Display the chart
    plt.tight_layout()
    plt.show()


def main():
    # Species dynamic params
    TOTAL_TIME_STEPS = 800
    INITIAL_SPECIES = 80
    GLOBAL_BIRTH_RATE = 0.05
    GLOBAL_EXTINCTION_RATE = 0.05
    ENVIRONMENTAL_CAPACITY = 120
    MASS_EXTINCTION_TIME = [400, 1000, 1980, 1990]
    GREAT_RADIATION_TIME = [200, 600, 1200, 1600]
    SPATIAL_RANGE_LIMIT = 400  # 500 * 500
    AVG_SPATIAL_RANGE = 200  # 80 * 80
    ENVIRONMENT_TYPES = 4

    # Horizon building params
    SECTIONS = 80
    AVG_HORIZON_INTERVAL = 20
    AVG_SECTION_TEMPORAL_RANGE = 500
    AVG_TIME_AVERAGE_FACTOR = 2
    AVG_SECTION_SPATIAL_RANGE = 20
    AVG_SAMPLING_FACTOR = 0.7
    AVG_ACCUMULATION_RATE = 1
    AVG_SECTION_ENVIRONMENT_TYPES = 2

    # Species dynamic simulation
    species_counts, birth_rates, extinction_rates, species_tree_info, species_legacy = simulate_species_dynamics(
        TOTAL_TIME_STEPS,
        INITIAL_SPECIES,
        GLOBAL_BIRTH_RATE,
        GLOBAL_EXTINCTION_RATE,
        ENVIRONMENTAL_CAPACITY,
        MASS_EXTINCTION_TIME,
        GREAT_RADIATION_TIME,
        SPATIAL_RANGE_LIMIT,
        AVG_SPATIAL_RANGE,
        ENVIRONMENT_TYPES)

    # 绘图
    draw_richness(species_counts, birth_rates, extinction_rates)
    # draw_phylogenetic_tree(species_tree_info)

    # fads_lads = np.array([[species.birth_time, species.extinction_time] for species in species_legacy]).T
    # fads = fads_lads[0]
    # lads = fads_lads[1]
    # draw_taxon_range_chart(fads, lads, reorder=False)

    # Horizon simulation
    horizon_df = build_pseudo_horizons(species_legacy,
                                       SECTIONS,
                                       AVG_HORIZON_INTERVAL,
                                       TOTAL_TIME_STEPS,
                                       AVG_SECTION_TEMPORAL_RANGE,
                                       AVG_TIME_AVERAGE_FACTOR,
                                       SPATIAL_RANGE_LIMIT,
                                       AVG_SECTION_SPATIAL_RANGE,
                                       ENVIRONMENT_TYPES,
                                       AVG_SAMPLING_FACTOR,
                                       AVG_ACCUMULATION_RATE,
                                       AVG_SECTION_ENVIRONMENT_TYPES)
    horizon_df.to_csv('Pseudo_ha_test.csv', index=False)


if __name__ == "__main__":
    main()
