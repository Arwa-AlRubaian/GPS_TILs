
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from matplotlib.offsetbox import AnchoredText
from lifelines.utils import concordance_index

EVENT= 'OS'
TIME= f'{EVENT}Time'

def censor_data(data, censor_time):  # censor_time in years
    cen_time = 12 * censor_time
    data.loc[data[TIME] > cen_time, [EVENT, TIME]] = [0, cen_time]
    return data

def draw_KM_curves ():

    feature_file_path = '../Data/LUAD_grades.csv'
    df_features = pd.read_csv(feature_file_path)
    df_features = df_features.dropna(subset=[TIME, EVENT])
    df_features = censor_data(df_features, 5)

    # Remove rows with no grade reported in the report
    df_features = df_features[df_features['grade'] != "-"]

    # ---------------------------------------- Curves based on grade from report ----------------------------------

    g1 = (df_features['grade'] == 'G1')
    g2 = (df_features['grade'] == 'G2') | (df_features['grade'] == 'G1-G2')
    g3 = (df_features['grade'] == 'G3') | (df_features['grade'] == 'G2-G3')

    T_g1 = df_features[TIME][g1]
    E_g1 = df_features[EVENT][g1]
    T_g2 = df_features[TIME][g2]
    E_g2 = df_features[EVENT][g2]
    T_g3 = df_features[TIME][g3]
    E_g3 = df_features[EVENT][g3]

    count_values = E_g1.value_counts()
    print(f'Grade 1: Number of patients :{len(E_g1)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')
    count_values = E_g2.value_counts()
    print(f'Grade 2: Number of patients :{len(E_g2)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')
    count_values = E_g3.value_counts()
    print(f'Grade 3: Number of patients :{len(E_g3)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')


    # Create a combined dataframe for testing
    df_test = df_features[df_features['grade'].isin(['G1', 'G2', 'G1-G2', 'G3', 'G2-G3'])].copy()
    df_test['group'] = df_test['grade'].replace({
        'G1': 'G1',
        'G2': 'G2',
        'G3': 'G3',
        'G1-G2': 'G2',
        'G2-G3': 'G3'
    })

    results = multivariate_logrank_test(df_test[TIME], df_test['group'], df_test[EVENT])
    p_value_all = results.p_value
    if p_value_all < 0.0001:
        pvalue_txt = 'p < 0.0001'
    else:
        pvalue_txt = 'p = ' + str(np.round(p_value_all, 4))

    # ---------------- C-index calculation -----------------
    # Create a pseudo-risk score from group labels: G1 < G2 < G3
    group_score_map = {'G1': 1, 'G2': 2, 'G3': 3}
    df_test['group_score'] = df_test['group'].map(group_score_map)

    # Calculate c-index using the risk score
    cindex = concordance_index(df_test[TIME], -df_test['group_score'],
                               df_test[EVENT])  # use -score if higher score = worse survival

    # Format c-index text
    cindex_txt = f'c-index = {cindex:.3f}'


    km_g1 = KaplanMeierFitter()
    km_g2 = KaplanMeierFitter()
    km_g3 = KaplanMeierFitter()

    plt.figure(figsize=(5, 7))
    ax = km_g1.fit(T_g1, event_observed=E_g1, label='G1').plot_survival_function()
    ax = km_g2.fit(T_g2, event_observed=E_g2, label='G2').plot_survival_function(ax=ax)
    ax = km_g3.fit(T_g3, event_observed=E_g3, label='G3').plot_survival_function(ax=ax)

    add_at_risk_counts(km_g1, km_g2, km_g3, ax=ax)
    plt.title('Grade reported in reports')
    ax.add_artist(AnchoredText(f'{pvalue_txt}\n{cindex_txt}', loc='lower left', frameon=True))


    ax.set_ylabel(f'{EVENT} probability')
    ax.set_xlabel('Survival time (months)')
    plt.tight_layout()
    plt.show()

    results = logrank_test(T_g1, T_g2, E_g1, E_g2)
    print("p-value %s; log-rank between G1 and G2 %s" % (results.p_value, np.round(results.test_statistic, 6)))
    results = logrank_test(T_g1, T_g3, E_g1, E_g3)
    print("p-value %s; log-rank between G1 and G3 %s" % (results.p_value, np.round(results.test_statistic, 6)))
    results = logrank_test(T_g2, T_g3, E_g2, E_g3)
    print("p-value %s; log-rank between G2 and G3 %s" % (results.p_value, np.round(results.test_statistic, 6)))
    print(f'overall p-value: {p_value_all}')


    # # --------------------- Grade based on model predictions based on WHO grading scheme --------------------------

    # generate grade
    df_features['WHO_grade'] = df_features.apply(calculate_who_grade, axis=1)

    g1 = (df_features['WHO_grade'] == 'G1')
    g2 = (df_features['WHO_grade'] == 'G2')
    g3 = (df_features['WHO_grade'] == 'G3')

    T_g1 = df_features[TIME][g1]
    E_g1 = df_features[EVENT][g1]
    T_g2 = df_features[TIME][g2]
    E_g2 = df_features[EVENT][g2]
    T_g3 = df_features[TIME][g3]
    E_g3 = df_features[EVENT][g3]

    count_values = E_g1.value_counts()
    print(f'Grade 1: Number of patients :{len(E_g1)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')
    count_values = E_g2.value_counts()
    print(f'Grade 2: Number of patients :{len(E_g2)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')
    count_values = E_g3.value_counts()
    print(f'Grade 3: Number of patients :{len(E_g3)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')

    df_test = df_features[df_features['WHO_grade'].isin(['G1','G2', 'G3'])].copy()

    results = multivariate_logrank_test(df_test[TIME], df_test['WHO_grade'], df_test[EVENT])
    p_value_all = results.p_value
    if p_value_all < 0.0001:
        pvalue_txt = 'p < 0.0001'
    else:
        pvalue_txt = 'p = ' + str(np.round(p_value_all, 4))

    # ---------------- C-index calculation -----------------
    group_score_map = {'G1': 1, 'G2': 2, 'G3': 3}
    df_test['group_score'] = df_test['WHO_grade'].map(group_score_map)

    # Calculate c-index using the risk score
    cindex = concordance_index(df_test[TIME], -df_test['group_score'],
                               df_test[EVENT])  # use -score if higher score = worse survival

    # Format c-index text
    cindex_txt = f'c-index = {cindex:.3f}'

    km_g1 = KaplanMeierFitter()
    km_g2 = KaplanMeierFitter()
    km_g3 = KaplanMeierFitter()

    plt.figure(figsize=(5, 7))
    ax = km_g1.fit(T_g1, event_observed=E_g1, label='G1').plot_survival_function()
    ax = km_g2.fit(T_g2, event_observed=E_g2, label='G2').plot_survival_function(ax=ax)
    ax = km_g3.fit(T_g3, event_observed=E_g3, label='G3').plot_survival_function(ax=ax)

    add_at_risk_counts(km_g1, km_g2, km_g3, ax=ax)
    plt.title('WHO grade based on model predictions')
    ax.add_artist(AnchoredText(f'{pvalue_txt}\n{cindex_txt}', loc='lower left', frameon=True))
    ax.set_ylabel(f'{EVENT} probability')
    ax.set_xlabel('Survival time (months)')
    plt.tight_layout()
    plt.show()

    results = logrank_test(T_g1, T_g2, E_g1, E_g2)
    print("p-value %s; log-rank between G1 and G2 %s" % (results.p_value, np.round(results.test_statistic, 6)))
    results = logrank_test(T_g1, T_g3, E_g1, E_g3)
    print("p-value %s; log-rank between G1 and G3 %s" % (results.p_value, np.round(results.test_statistic, 6)))
    results = logrank_test(T_g2, T_g3, E_g2, E_g3)
    print("p-value %s; log-rank between G2 and G3 %s" % (results.p_value, np.round(results.test_statistic, 6)))

def calculate_who_grade(row):
    if row['micropapillary_per'] + row['solid_per'] >= 0.2:
        return 'G3'
    elif row['lepidic_per'] > row['acinar_per'] and row['lepidic_per'] > row['papillary_per']:
        return 'G1'
    else:
        return 'G2'


def draw_KM_curves_one_vs_other(grade): # grade can be 'G1' or 'G3'
    feature_file_path = '../Data/LUAD_grades.csv'
    df_features = pd.read_csv(feature_file_path)
    df_features = df_features.dropna(subset=[TIME, EVENT])
    df_features = censor_data(df_features, 5)

    # Remove rows with no grade reported in the report
    df_features = df_features[df_features['grade'] != "-"]

    if grade == 'G1':
        g1 = (df_features['grade'] == 'G1')
        g2 = ~g1  # all other grade values

        T_g1 = df_features[TIME][g1]
        E_g1 = df_features[EVENT][g1]
        T_g2 = df_features[TIME][g2]
        E_g2 = df_features[EVENT][g2]


        count_values = E_g1.value_counts()
        print(
            f'Grade 1: Number of patients :{len(E_g1)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')
        count_values = E_g2.value_counts()
        print(
            f'Grade 2&3: Number of patients :{len(E_g2)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')

        results = logrank_test(T_g1, T_g2, event_observed_A=E_g1, event_observed_B=E_g2)
        p_value_all = results.p_value
        if p_value_all < 0.0001:
            pvalue_txt = 'p < 0.0001'
        else:
            pvalue_txt = 'p = ' + str(np.round(p_value_all, 4))

        # Create a dataframe to compute the c-index
        df_test = df_features[df_features['grade'].isin(['G1', 'G2', 'G1-G2', 'G3', 'G2-G3'])].copy()
        df_test['group'] = df_test['grade'].replace({
            'G1': 'G1',
            'G2': 'G2_3',
            'G1-G2': 'G2_3',
            'G3': 'G2_3',
            'G2-G3': 'G2_3'
        })
        # Assign risk scores: lower score = lower risk
        df_test['group_score'] = df_test['group'].map({'G1': 1, 'G2_3': 2})
        # Compute c-index
        cindex = concordance_index(df_test[TIME], -df_test['group_score'], df_test[EVENT])  # Higher group = worse prognosis
        cindex_txt = f'c-index = {cindex:.3f}'

        km_g1 = KaplanMeierFitter()
        km_g2 = KaplanMeierFitter()

        plt.figure(figsize=(5, 7))
        ax = km_g1.fit(T_g1, event_observed=E_g1, label='G1').plot_survival_function()
        ax = km_g2.fit(T_g2, event_observed=E_g2, label='G2 & G3').plot_survival_function(ax=ax)

        add_at_risk_counts(km_g1, km_g2, ax=ax)
        plt.title('Grade reported in reports')
        # plt.text(0.6, 0.1, f'Log-rank p (G1 vs G2 vs G3): {p_value_all:.4f}',
        #          transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        ax.add_artist(AnchoredText(f'{pvalue_txt}\n{cindex_txt}', loc='lower left', frameon=True))

        ax.set_ylabel(f'{EVENT} probability')
        ax.set_xlabel('Survival time (months)')
        plt.tight_layout()
        plt.show()

    elif grade == 'G3':
        g2 = (df_features['grade'] == 'G3')| (df_features['grade'] == 'G2-G3')
        g1 = ~g2  # all other grade values

        T_g1 = df_features[TIME][g1]
        E_g1 = df_features[EVENT][g1]
        T_g2 = df_features[TIME][g2]
        E_g2 = df_features[EVENT][g2]

        count_values = E_g1.value_counts()
        print(
            f'Grade 1&2: Number of patients :{len(E_g1)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')
        count_values = E_g2.value_counts()
        print(
            f'Grade 3: Number of patients :{len(E_g2)}, Number of events = {count_values.get(1, 0)}, censored = {count_values.get(0, 0)}')

        results = logrank_test(T_g1, T_g2, event_observed_A=E_g1, event_observed_B=E_g2)
        p_value_all = results.p_value
        if p_value_all < 0.0001:
            pvalue_txt = 'p < 0.0001'
        else:
            pvalue_txt = 'p = ' + str(np.round(p_value_all, 4))

        # Create a dataframe to compute the c-index
        df_test = df_features[df_features['grade'].isin(['G1', 'G2', 'G1-G2', 'G3', 'G2-G3'])].copy()
        df_test['group'] = df_test['grade'].replace({
            'G1': 'G1_2',
            'G2': 'G1_2',
            'G1-G2': 'G1_2',
            'G3': 'G3',
            'G2-G3': 'G3'
        })
        # Assign risk scores: lower score = lower risk
        df_test['group_score'] = df_test['group'].map({'G1_2': 1, 'G3': 2})
        # Compute c-index
        cindex = concordance_index(df_test[TIME], -df_test['group_score'],
                                   df_test[EVENT])  # Higher group = worse prognosis
        cindex_txt = f'c-index = {cindex:.3f}'

        km_g1 = KaplanMeierFitter()
        km_g2 = KaplanMeierFitter()

        plt.figure(figsize=(5, 7))
        ax = km_g1.fit(T_g1, event_observed=E_g1, label='G1 & G2').plot_survival_function()
        ax = km_g2.fit(T_g2, event_observed=E_g2, label='G3').plot_survival_function(ax=ax)

        add_at_risk_counts(km_g1, km_g2, ax=ax)
        plt.title('Grade reported in reports')
        ax.add_artist(AnchoredText(f'{pvalue_txt}\n{cindex_txt}', loc='lower left', frameon=True))

        ax.set_ylabel(f'{EVENT} probability')
        ax.set_xlabel('Survival time (months)')
        plt.tight_layout()
        plt.show()
    else:
        print(f'{grade} is invalid, must be G1 of G3')



if __name__ == '__main__':
    draw_KM_curves()
    draw_KM_curves_one_vs_other('G3')