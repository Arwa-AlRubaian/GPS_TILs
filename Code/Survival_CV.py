import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from matplotlib.offsetbox import AnchoredText
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

CV_REPEATS = 1000
NUM_CV_SPLITS = 2
font_size = 10
fig_size = 8
cutoff_point = -1
EVENT_COL = 'OS'
TIME_COL = f'{EVENT_COL}Time'


def normalize_datasets(train_set, test_set, feats_list, norm_type='standard_scaler'):
    if norm_type == 'minmax':
        feat_min = train_set[feats_list].min()
        feat_max = train_set[feats_list].max()

        train_set_normalized = train_set.copy()
        test_set_normalized = test_set.copy()
        train_set_normalized[feats_list] = (train_set[feats_list] - feat_min) / (feat_max - feat_min)
        test_set_normalized[feats_list] = (test_set[feats_list] - feat_min) / (feat_max - feat_min)

    if norm_type == 'standard_scaler':
        scaler = StandardScaler()
        scaler.fit(train_set[feats_list])  # Fit only on training

        train_set_normalized = train_set.copy()
        test_set_normalized = test_set.copy()

        train_set_normalized[feats_list] = pd.DataFrame(
            scaler.transform(train_set[feats_list]),
            columns=feats_list,
            index=train_set.index
        )

        test_set_normalized[feats_list] = pd.DataFrame(
            scaler.transform(test_set[feats_list]),
            columns=feats_list,
            index=test_set.index
        )
    return train_set_normalized, test_set_normalized


def cross_validation(train_data, val_data, feats_list, cutoff_mode, cutoff_point):
    train_data = train_data.fillna(0)

    feats_list_temp = []
    for l in feats_list:
        feats_list_temp.append(l)
    feats_list_temp.append(TIME_COL)
    feats_list_temp.append(EVENT_COL)

    train_data_temp = train_data[feats_list_temp]

    try:
        cph= CoxPHFitter().fit(train_data_temp, duration_col=TIME_COL, event_col=EVENT_COL)
    except:
        print('Failed in fitting CoxPH for train_data_infer')
        return -1

    train_data_infer = train_data_temp.drop(columns=[EVENT_COL, TIME_COL])
    partial_hazard_train = cph.predict_partial_hazard(train_data_infer)

    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = partial_hazard_train.mean()
        elif cutoff_mode == 'median':
            cutoff_value = partial_hazard_train.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = partial_hazard_train.quantile(0.60)
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point

    ##################################################### Predict on the validation set
    val_data = val_data.fillna(0)

    val_data_temp = val_data[feats_list_temp]

    partial_hazard_test = cph.predict_partial_hazard(val_data_temp)
    # c-index on the validation set
    val_cindex = concordance_index(val_data_temp[TIME_COL], -partial_hazard_test, val_data_temp[EVENT_COL])

    # Use mean value in the discovery set as the cut-off value and divide subjects int the validation set into two groups
    upper = partial_hazard_test >= cutoff_value
    T_upper_test = val_data_temp[TIME_COL][upper]
    E_upper_test = val_data_temp[EVENT_COL][upper]
    lower = partial_hazard_test < cutoff_value
    T_lower_test = val_data_temp[TIME_COL][lower]
    E_lower_test = val_data_temp[EVENT_COL][lower]

    # saving the samples with their risk group
    val_data['risk_score'] = partial_hazard_test
    val_data['risk_group'] = ['High' if x > cutoff_value else 'Low' for x in val_data['risk_score']]



    results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)

    val_pvalue = results.p_value
    val_hr = cph.hazard_ratios_

    return T_lower_test, T_upper_test, E_lower_test, E_upper_test, val_cindex, val_pvalue, val_hr


def plot_km(time_df, event_df, partial_hazard_train, cutoff_value, add_at_risk_counts=True, x_label="Months",
            y_label=None):
    upper = partial_hazard_train >= cutoff_value
    T_upper_train = time_df[upper]
    E_upper_train = event_df[upper]
    lower = partial_hazard_train < cutoff_value
    T_lower_train = time_df[lower]
    E_lower_train = event_df[lower]

    # evaluating
    results = logrank_test(T_lower_train, T_upper_train, E_lower_train, E_upper_train)

    # preparing the figure
    font_size = 18
    fig_size = 10
    fig = plt.figure(figsize=(fig_size, fig_size - 2))  ##adjust according to font size
    ax = fig.add_subplot(111)
    ax.set_xlabel('', fontsize=font_size)
    ax.set_ylabel('', fontsize=font_size)

    if results.p_value < 0.0001:
        train_pvalue_txt = 'p < 0.0001'
    else:
        train_pvalue_txt = 'p = ' + str(np.round(results.p_value, 4))

    from matplotlib.offsetbox import AnchoredText
    ax.add_artist(AnchoredText(train_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()
    ax = km_upper.fit(T_upper_train, event_observed=E_upper_train, label='high').plot_survival_function(ax=ax,
                                                                                                        show_censors=True,
                                                                                                        censor_styles={
                                                                                                            'ms': 5},
                                                                                                        color='r',
                                                                                                        ci_show=False,
                                                                                                        xlabel=x_label,
                                                                                                        ylabel=y_label)
    ax = km_lower.fit(T_lower_train, event_observed=E_lower_train, label='low').plot_survival_function(ax=ax,
                                                                                                       show_censors=True,
                                                                                                       censor_styles={
                                                                                                           'ms': 5},
                                                                                                       color='b',
                                                                                                       ci_show=False,
                                                                                                       xlabel=x_label,
                                                                                                       ylabel=y_label)

    if add_at_risk_counts:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size * 1))
        fig.subplots_adjust(bottom=0.4)
        fig.subplots_adjust(left=0.2)
    ax.get_legend().remove()
    # plt.show()
    return fig


###################################################################################################################


def cv_helper(data_df,feats_list, split_folder, km_plot=False, add_counts=False, shuffle_data=False):
    c_indices = []
    p_values = []


    for run in tqdm(range(NUM_CV_SPLITS), desc='performing survival analysis'):
        # forming the training and validation cohorts
        split_path = f'{split_folder}/splits_{run}.csv'
        split_df = pd.read_csv(split_path)
        train_set = data_df[data_df['SAMPLE_ID'].isin(split_df['train'])]
        test_set = data_df[data_df['SAMPLE_ID'].isin(split_df['val'])]
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)

        # When correcting the p-values the event and time needs to be shuffled randomly
        if shuffle_data:
            random_indices = np.random.permutation(train_set.index)
            train_set[EVENT_COL] = train_set[EVENT_COL].loc[random_indices].reset_index(drop=True)
            train_set[TIME_COL] = train_set[TIME_COL].loc[random_indices].reset_index(drop=True)

            random_indices = np.random.permutation(test_set.index)
            test_set[EVENT_COL] = test_set[EVENT_COL].loc[random_indices].reset_index(drop=True)
            test_set[TIME_COL] = test_set[TIME_COL].loc[random_indices].reset_index(drop=True)

        # Normalizing the datasets
        train_set, test_set = normalize_datasets(train_set, test_set, feats_list)

        # running the model and collating the results
        output = cross_validation (train_set, test_set, feats_list, cutoff_mode, cutoff_point)
        if output == -1:  # something went wrong, neglect this run
            print(f"something went wrong with run {run}")
            continue
        if run == 0:
            T_lower_test, T_upper_test, E_lower_test, E_upper_test, cindex_test, pvalue_test, test_hazard_ratios = output
        if run != 0 and output != -1:
            _T_lower_test, _T_upper_test, _E_lower_test, _E_upper_test, cindex_test, pvalue_test, temp = output
            T_lower_test = pd.concat([T_lower_test, _T_lower_test], axis=0, join='outer', ignore_index=True, sort=False)
            T_upper_test = pd.concat([T_upper_test, _T_upper_test], axis=0, join='outer', ignore_index=True, sort=False)
            E_lower_test = pd.concat([E_lower_test, _E_lower_test], axis=0, join='outer', ignore_index=True, sort=False)
            E_upper_test = pd.concat([E_upper_test, _E_upper_test], axis=0, join='outer', ignore_index=True, sort=False)
            test_hazard_ratios = pd.concat([test_hazard_ratios, temp], axis=1, join='outer', ignore_index=True,
                                           sort=False)
        c_indices.append(cindex_test)
        p_values.append(pvalue_test)

    df_to_save = test_hazard_ratios.T
    df_to_save['c_index'] = c_indices
    df_to_save['p_value'] = p_values

    logrank_results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)


    if km_plot:
        # Initializing the KaplanMeierModel for each group
        km_upper = KaplanMeierFitter()
        km_lower = KaplanMeierFitter()

        fig = plt.figure(figsize=(fig_size, fig_size - 2))  ##adjust according to font size
        ax = fig.add_subplot(111)

        km_upper.fit(T_upper_test, event_observed=E_upper_test, label='High')
        km_lower.fit(T_lower_test, event_observed=E_lower_test, label='Low')

        km_upper.plot_survival_function(ax=ax, show_censors=True)
        km_lower.plot_survival_function(ax=ax, show_censors=True)


        if add_counts:
            # Initializing the KaplanMeierModel for each group
            km_upper = KaplanMeierFitter()
            km_lower = KaplanMeierFitter()

            fig_copy = plt.figure(figsize=(fig_size, fig_size - 2))  ##adjust according to font size
            ax_copy = fig_copy.add_subplot(111)

            km_upper.fit(T_upper_test, event_observed=E_upper_test, label='High')
            km_lower.fit(T_lower_test, event_observed=E_lower_test, label='Low')

            km_upper.plot_survival_function(ax=ax_copy, show_censors=True)
            km_lower.plot_survival_function(ax=ax_copy, show_censors=True)

            add_at_risk_counts(km_upper, km_lower, ax=ax_copy, fig=fig_copy)
            fig_copy.subplots_adjust(bottom=0.4)
            fig_copy.subplots_adjust(left=0.2)
            ax_copy.get_legend().remove()

            return df_to_save, logrank_results, fig, fig_copy

        return df_to_save, logrank_results, fig

    return df_to_save, logrank_results

def create_data_splits (data_df, path_to_splits):
    print("Creating data splits...")
    skf = StratifiedKFold(n_splits=NUM_CV_SPLITS, shuffle=True)

    # Extract features
    X = data_df[[TIME_COL]]
    y = data_df[EVENT_COL]  # Event column used for stratification

    # Create folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_ids = data_df.iloc[train_idx]['SAMPLE_ID'].reset_index(drop=True)
        val_ids = data_df.iloc[val_idx]['SAMPLE_ID'].reset_index(drop=True)

        # Combine into one DataFrame
        split_df = pd.DataFrame({'train': train_ids, 'val': val_ids})

        # Save to CSV
        file_name = f'{path_to_splits}/splits_{fold}.csv'
        split_df.to_csv(file_name, index=False)


if __name__ == '__main__':
    censor_at = 5       # in years
    cutoff_mode = 'median'
    save_plot = True

    # Path variables
    plots_save_path = '../plots'
    save_dir = '../results'
    path_to_splits = '../splits'
    path_to_data = '../Data/10_total_nuclei_20_tumor_nuclei_5_inflammatory_nuclei_15_necrosis_nuclei.csv'


    pattern_percentages = ['Lepidic_per', 'Acinar_per', 'Papillary_per', 'Solid_per']
    min_pattern_features = ['Lepidic_min_TILs', 'Lepidic_min_sTILs', 'Acinar_min_TILs', 'Acinar_min_sTILs',
                            'Papillary_min_TILs', 'Papillary_min_sTILs', 'Solid_min_TILs', 'Solid_min_sTILs']

    max_pattern_features = ['Lepidic_max_TILs', 'Lepidic_max_sTILs', 'Acinar_max_TILs', 'Acinar_max_sTILs',
                            'Papillary_max_TILs', 'Papillary_max_sTILs', 'Solid_max_TILs', 'Solid_max_sTILs']

    min_max_pattern_necrosis_feats = ['Lepidic_min_necro', 'Lepidic_max_necro', 'Acinar_min_necro', 'Acinar_max_necro',
                                      'Papillary_min_necro', 'Papillary_max_necro', 'Solid_min_necro','Solid_max_necro']

    avg_pattern_features = ['Lepidic_avg_TILs', 'Lepidic_avg_sTILs', 'Acinar_avg_TILs', 'Acinar_avg_sTILs',
                            'Papillary_avg_TILs', 'Papillary_avg_sTILs','Solid_avg_TILs', 'Solid_avg_sTILs']

    std_pattern_features = ['Lepidic_std_TILs', 'Lepidic_std_sTILs', 'Acinar_std_TILs', 'Acinar_std_sTILs',
                            'Papillary_std_TILs', 'Papillary_std_sTILs', 'Solid_std_TILs', 'Solid_std_sTILs']

    avg_std_pattern_necrosis_feats = ['Lepidic_avg_necro', 'Lepidic_std_necro', 'Acinar_avg_necro', 'Acinar_std_necro',
                                      'Papillary_avg_necro', 'Papillary_std_necro', 'Solid_avg_necro', 'Solid_std_necro']

    aggregate_features = ['min_TILs', 'max_TILs', 'avg_TILs', 'min_sTILs', 'max_sTILs', 'avg_sTILs',
                          'std_TILs', 'std_sTILs', 'min_necro', 'max_necro', 'avg_necro', 'std_necro']

    #==============================================================================================================

    feats_list = min_pattern_features.copy()
    feats_list.extend(max_pattern_features.copy())
    feats_list.extend(min_max_pattern_necrosis_feats.copy())
    feats_list.extend(avg_pattern_features.copy())
    feats_list.extend(std_pattern_features.copy())
    feats_list.extend(avg_std_pattern_necrosis_feats.copy())
    # feats_list.extend(pattern_percentages.copy())
    feature_caption = 'Growth_pattern_TILs_sTIL_necro'


    # feats_list = aggregate_features.copy()
    # feats_list.extend(pattern_percentages)
    # feature_caption = 'Aggregate_TILs_sTILs_necro'


    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_save_path, exist_ok=True)
    data_df = pd.read_csv(path_to_data)
    data_df = data_df.dropna(subset=[TIME_COL, EVENT_COL])
    # remove cases with no follow-up data
    data_df = data_df[~((data_df[TIME_COL] == 0) & (data_df[EVENT_COL] == 0))]
    print(f"Number of cases in {EVENT_COL} experiment after dropping NA: {len(data_df)}")

    # censor the data !!
    cen_time = 12 * censor_at
    data_df.loc[data_df[TIME_COL] > cen_time, [EVENT_COL, TIME_COL]] = [0, cen_time]

    # If no splits files are present --> create CV splits
    if not os.path.exists(path_to_splits):
        os.makedirs(path_to_splits, exist_ok=True)

    if len(os.listdir(path_to_splits)) == 0 :
        create_data_splits (data_df, path_to_splits)


    # running the cross-validation
    df_to_save, logrank_results, km_fig, km_fig_counts = cv_helper(data_df, feats_list=feats_list, split_folder=path_to_splits, km_plot=True, add_counts=True)
    ref_logrank_stat = logrank_results.test_statistic

    # =============================================================================================================
    #  repeat the cross-validation several times for a corrected p-value (permutation test)
    num_high_replicates = 0
    num_valid_runs = 0
    for _ in tqdm(range(CV_REPEATS), desc='Correcting p-value'):
        try:
            _, logrank_results_repeats = cv_helper(data_df, feats_list, split_folder=path_to_splits, km_plot=False, shuffle_data=True)
        except:
            continue
        if logrank_results_repeats.test_statistic > ref_logrank_stat:
            num_high_replicates += 1
        num_valid_runs += 1
    corrected_p_value = num_high_replicates / num_valid_runs

    avrg_cindex = df_to_save["c_index"].mean()
    print("Average C-Index: ", avrg_cindex)
    print("Std C-Index: ", df_to_save["c_index"].std())
    print("Corrected p-value: ", corrected_p_value)

    # adding the corrected p-value to the figure
    if corrected_p_value < 0.0001:
        pvalue_txt = 'p < 0.0001'
    else:
        pvalue_txt = 'p = ' + str(np.round(corrected_p_value, 4))
    ax = km_fig.axes[0]  # Get the ax object from fig
    ax.add_artist(AnchoredText(pvalue_txt, loc='lower left', frameon=False, prop=dict(size=font_size)))
    ax.set_ylabel(f"{EVENT_COL.upper()} Probability")
    ax.set_ylim(0, 1)
    ax.set_title(''.join(feature_caption).upper(), fontsize=font_size + 2)

    # save the results
    save_path = save_dir + f"/cv_results_{feature_caption}_{EVENT_COL}_censor{censor_at}_cindex{avrg_cindex:.2}_pvalue{corrected_p_value:.3}"
    df_to_save.to_csv(save_path + ".csv", index=None)
    save_path = plots_save_path + f"/cv_results_{feature_caption}_{EVENT_COL}_censor{censor_at}_cindex{avrg_cindex:.2}_pvalue{corrected_p_value:.3}"

    # save the km figure with counts
    ax = km_fig_counts.axes[0]
    ax.add_artist(AnchoredText(pvalue_txt, loc='lower left', frameon=True))
    ax.set_ylabel(f"{EVENT_COL.upper()} Probability")
    ax.set_xlabel('Months')
    ax.set_ylim(0, 1)
    ax.legend(title='Risk Group')
    ax.set_title(f'Kaplan-Meier Estimate on {feature_caption}')
    plt.grid(False)
    plt.tight_layout()

    # save the results
    save_path = plots_save_path + f"/cv_results_{feature_caption}_{EVENT_COL}_censor{censor_at}_cindex{avrg_cindex:.2}_pvalue{corrected_p_value:.3}"
    km_fig_counts.savefig(save_path + '.png', dpi=600, bbox_inches='tight', pad_inches=0)
