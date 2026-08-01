"""
=========================================================
Gaza Freelancer AI Adoption Analysis
MSc Social Data Science — Oxford Internet Institute
=========================================================

Research Question:
Does the adoption of Generative AI tools reduce the wage gap
between Gaza-based freelancers and comparable freelancers in
unrestricted labour markets?

Hypotheses:
H1: AI tools increase competitiveness through two channels:
    (a) higher productivity (Q11) and (b) higher earnings (Q10)
H2: Gaza freelancers report lower earnings and fewer projects
    than comparable freelancers in less constrained markets
H3: Among Gaza freelancers, more intensive AI use is associated
    with smaller gaps in earnings, productivity, and competitiveness

=========================================================
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# STEP 1 - LOAD DATA
# =========================================================

print("=" * 60)
print("STEP 1 - LOADING DATA")
print("=" * 60)

df_raw = pd.read_csv(
    'data.csv',
    low_memory=False,
    skiprows=[1, 2]
)

print(f"Total rows in file: {len(df_raw)}")
print(f"Columns: {len(df_raw.columns)}")

# =========================================================
# STEP 2 - DATA CLEANING AND PREPARATION
# =========================================================

print("\n" + "=" * 60)
print("STEP 2 - DATA CLEANING AND PREPARATION")
print("=" * 60)

df = df_raw.copy()

# 2.1 Exclude out-of-scope locations
df = df[~df['Q2'].str.contains('أخرى', na=False)].copy()
print(f"\nAfter excluding out-of-scope locations: {len(df)} respondents")

# 2.2 Recode AI adoption variable (Q26 branch logic bug)
def recode_ai_adoption(row):
    if row['Q26'] == 'نعم':
        return 'Yes'
    elif row['Q26'] == 'لا':
        return 'No'
    elif pd.isna(row['Q26']) and pd.notna(row['Q13']):
        return 'Yes'
    else:
        return 'Indeterminate'

df['AI_adoption'] = df.apply(recode_ai_adoption, axis=1)
print(f"\nAI adoption status:")
print(df['AI_adoption'].value_counts())

# 2.3 Recode Gaza location (Q2) - binary dummy
df['Gaza'] = (df['Q2'] == 'غزة، فلسطين').astype(int)
print(f"\nGaza (1): {df['Gaza'].sum()}")
print(f"Non-Gaza (0): {(df['Gaza'] == 0).sum()}")

# 2.4 Recode AI intensity (Q14) - five-point ordinal scale
q14_map = {
    'يوميا': 5,
    'يوميًا': 5,
    'عدة مرات في الأسبوع': 4,
    'مرة في الأسبوع': 3,
    'بضع مرات في الشهر': 2,
    'نادرا': 1,
    'نادرًا': 1
}
df['AI_intensity'] = df['Q14'].map(q14_map)
print(f"\nAI intensity distribution:")
print(df['AI_intensity'].value_counts(dropna=False).sort_index())

# 2.5 Recode monthly earnings (Q10) - midpoint values
q10_map = {
    'أقل من 200 دولار': 100,
    'من 200 إلى 500 دولار': 350,
    'من 501 إلى 1,000 دولار': 750,
    'من 1,001 إلى 2,000 دولار': 1500,
    'من 2,001 إلى 5,000 دولار': 3500,
    'أكثر من 5,000 دولار': 6000
}
df['earnings'] = df['Q10'].map(q10_map)
print(f"\nQ10 valid responses: {df['earnings'].notna().sum()}")
print(f"Q10 excluded (Other): {df['earnings'].isna().sum()}")

# 2.6 Recode projects per month (Q11) - midpoint values
q11_map = {
    'مشروع واحد': 1,
    'مشروعان إلى 3 مشاريع': 2.5,
    'من 4 إلى 6 مشاريع': 5,
    'من 7 إلى 10 مشاريع': 8.5,
    'أكثر من 10 مشاريع': 12
}
df['projects'] = df['Q11'].map(q11_map)
print(f"\nQ11 valid responses: {df['projects'].notna().sum()}")

# 2.7 Recode perceived competitiveness (Q17) - Likert 1-5
q17_map = {
    'أوافق بشدة': 5,
    'أوافق إلى حد ما': 4,
    'لا أوافق ولا أعارض': 3,
    'لا أوافق إلى حد ما': 2,
    'لا أوافق بشدة': 1
}
df['competitiveness'] = df['Q17'].map(q17_map)
print(f"\nQ17 valid responses: {df['competitiveness'].notna().sum()}")

# 2.8 Recode earnings per project (Q9) - descriptive only
q9_map = {
    'أقل من 50 دولارًا': 25,
    'من 50 إلى 200 دولار': 125,
    'من 201 إلى 500 دولار': 350,
    'من 501 إلى 1,000 دولار': 750,
    'من 1,001 إلى 3,000 دولار': 2000,
    'أكثر من 3,000 دولار': 3500
}
df['project_value'] = df['Q9'].map(q9_map)
print(f"\nQ9 valid responses (descriptive only): {df['project_value'].notna().sum()}")

# 2.9 Recode freelancing experience (Q3) - midpoints in years
q3_map = {
    'أقل من سنة': 0.5,
    'من سنة إلى سنتين': 1.5,
    'من 3 إلى 5 سنوات': 4,
    'من 6 إلى 10 سنوات': 8,
    'أكثر من 10 سنوات': 12
}
df['experience'] = df['Q3'].map(q3_map)

# 2.10 Recode education (Q4) - ordinal 1-4 (diploma=1.5)
def recode_education(row):
    val = row['Q4']
    if val == 'الثانوية العامة أو ما يعادلها':
        return 1.0
    elif val == 'درجة البكالوريوس':
        return 2.0
    elif val == 'درجة الماجستير':
        return 3.0
    elif val == 'درجة الدكتوراه أو أعلى':
        return 4.0
    elif val == 'أخرى (يرجى التحديد)':
        return 1.5
    else:
        return np.nan

df['education'] = df.apply(recode_education, axis=1)

# 2.11 Recode English proficiency (Q8) - ordinal 1-5
q8_map = {
    'مستوى أساسي': 1,
    'مستوى متوسط': 2,
    'مستوى متقدم': 3,
    'مستوى طلق': 4,
    'مستوى اللغة الأم / شبه أم': 5
}
df['english'] = df['Q8'].map(q8_map)

# 2.12 Recode age (Q6) - ordinal 1-5 (prefer not to say = NaN)
q6_map = {
    'من 18 إلى 24': 1,
    'من 25 إلى 34': 2,
    'من 35 إلى 44': 3,
    'من 45 إلى 54': 4,
    '55 عاما أو أكبر': 5,
    '55 عامًا أو أكبر': 5
}
df['age'] = df['Q6'].map(q6_map)

# 2.13 Recode gender (Q7) - binary (male=1, female=0)
df['gender'] = (df['Q7'] == 'ذكر').astype(int)

# 2.14 Recode skill category (Q1) with recoding of Others
def recode_skill(row):
    val = row['Q1']
    text = str(row['Q1_6_TEXT']).strip().lower() if pd.notna(row['Q1_6_TEXT']) else ''

    if val == 'تطوير البرمجيات وتطوير الويب':
        return 'software_web'
    elif val == 'تصميم واجهات المستخدم/تجربة المستخدم والتصميم الجرافيكي':
        return 'ui_ux'
    elif val == 'التسويق الرقمي وتحسين محركات البحث':
        return 'digital_marketing'
    elif val == 'كتابة المحتوى':
        return 'content_writing'
    elif val == 'الترجمة':
        return 'translation'
    elif val == 'أخرى (يرجى التحديد)':
        sw_keywords = ['مطور', 'برمجيات', 'اختبار', 'جودة', 'ai', 'أمن']
        if any(kw in text for kw in sw_keywords):
            return 'software_web'
        ui_keywords = ['موشن', 'جرافيك', 'ملتيميديا']
        if any(kw in text for kw in ui_keywords):
            return 'ui_ux'
        data_keywords = ['بيانات', 'data']
        if any(kw in text for kw in data_keywords):
            return 'data_services'
        return 'other'
    else:
        return 'other'

df['skill'] = df.apply(recode_skill, axis=1)
print(f"\nSkill category distribution after recoding:")
print(df['skill'].value_counts())

# 2.15 Create interaction term AI x Gaza
df['AI_x_Gaza'] = df['AI_intensity'] * df['Gaza']

# 2.16 Create skill dummy variables (reference: software_web)
skill_dummies = pd.get_dummies(df['skill'], prefix='skill', dtype=int)
if 'skill_software_web' in skill_dummies.columns:
    skill_dummies = skill_dummies.drop('skill_software_web', axis=1)
df = pd.concat([df, skill_dummies], axis=1)

print(f"\nFinal analytical dataset: {len(df)} respondents")
print(f"Gaza: {df['Gaza'].sum()} | Non-Gaza: {(df['Gaza']==0).sum()}")

# =========================================================
# STEP 3 - DESCRIPTIVE STATISTICS
# =========================================================

print("\n" + "=" * 60)
print("STEP 3 - DESCRIPTIVE STATISTICS")
print("=" * 60)

outcomes = ['earnings', 'projects', 'competitiveness', 'project_value']
outcome_labels = ['Monthly Earnings ($)', 'Projects/Month', 'Perceived Competitiveness (1-5)', 'Earnings per Project ($)']

# 3.1 Overall sample
print("\n--- 3.1 Overall Sample Descriptives ---")
for var, label in zip(outcomes, outcome_labels):
    valid = df[var].dropna()
    print(f"\n{label}:")
    print(f"  N={len(valid)}, Mean={valid.mean():.2f}, SD={valid.std():.2f}, "
          f"Min={valid.min():.2f}, Max={valid.max():.2f}, Median={valid.median():.2f}")

# 3.2 Gaza vs Non-Gaza comparison
print("\n--- 3.2 Gaza vs Non-Gaza Comparison ---")
gaza_df = df[df['Gaza'] == 1]
non_gaza_df = df[df['Gaza'] == 0]

print(f"\n{'Variable':<35} {'Gaza Mean (SD)':<22} {'Non-Gaza Mean (SD)':<22} {'t':<8} {'p':<8}")
print("-" * 100)

for var, label in zip(outcomes, outcome_labels):
    g = gaza_df[var].dropna()
    ng = non_gaza_df[var].dropna()
    if len(g) > 1 and len(ng) > 1:
        t_stat, p_val = stats.ttest_ind(g, ng, equal_var=False)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else ''
        print(f"{label:<35} "
              f"{g.mean():.2f} ({g.std():.2f})          "
              f"{ng.mean():.2f} ({ng.std():.2f})          "
              f"{t_stat:.3f}   "
              f"{p_val:.3f}{sig}")

# 3.3 Daily vs Less Frequent AI users
print("\n--- 3.3 Daily vs Less Frequent AI Users ---")
daily_df = df[df['AI_intensity'] == 5]
less_df = df[(df['AI_intensity'] < 5) & (df['AI_intensity'].notna())]

print(f"\n{'Variable':<35} {'Daily Mean (SD)':<22} {'Less Freq Mean (SD)':<22} {'t':<8} {'p':<8}")
print("-" * 100)

for var, label in zip(outcomes, outcome_labels):
    d = daily_df[var].dropna()
    l = less_df[var].dropna()
    if len(d) > 1 and len(l) > 1:
        t_stat, p_val = stats.ttest_ind(d, l, equal_var=False)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else ''
        print(f"{label:<35} "
              f"{d.mean():.2f} ({d.std():.2f})          "
              f"{l.mean():.2f} ({l.std():.2f})          "
              f"{t_stat:.3f}   "
              f"{p_val:.3f}{sig}")

# 3.4 AI intensity distribution
print("\n--- 3.4 AI Intensity Distribution ---")
ai_dist = df['AI_intensity'].value_counts(dropna=False).sort_index()
labels_ai = {1.0: 'Rarely', 2.0: 'Few times/month', 3.0: 'Once/week',
             4.0: 'Several times/week', 5.0: 'Daily', np.nan: 'Missing'}
for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
    count = (df['AI_intensity'] == val).sum()
    pct = count / len(df) * 100
    print(f"  {labels_ai[val]}: n={count} ({pct:.1f}%)")
missing_ai = df['AI_intensity'].isna().sum()
print(f"  Missing: n={missing_ai}")

# =========================================================
# STEP 4 - MULTICOLLINEARITY CHECK (VIF)
# =========================================================

print("\n" + "=" * 60)
print("STEP 4 - MULTICOLLINEARITY CHECK (VIF)")
print("=" * 60)

skill_dummy_cols = [c for c in df.columns if c.startswith('skill_')]
vif_vars = ['AI_intensity', 'Gaza', 'experience', 'education', 'english', 'age', 'gender'] + skill_dummy_cols
vif_data = df[vif_vars].dropna()

print(f"\nVIF calculation sample: {len(vif_data)} respondents")
print(f"\n{'Variable':<40} {'VIF':<10}")
print("-" * 50)

X_vif = sm.add_constant(vif_data)
vif_results = []
for i, col in enumerate(vif_data.columns):
    try:
        vif_val = variance_inflation_factor(X_vif.values, i + 1)
        vif_results.append((col, vif_val))
        print(f"{col:<40} {vif_val:.3f}")
    except:
        print(f"{col:<40} Could not compute")

if vif_results:
    max_vif = max([v for _, v in vif_results])
    print(f"\nMax VIF: {max_vif:.3f}")
    run_psm = max_vif > 5
    if run_psm:
        print("WARNING: VIF > 5 detected - PSM robustness check will be applied")
    else:
        print("All VIF < 5 - no multicollinearity concern - PSM not required")
else:
    run_psm = False

# Correlation matrix
print("\n--- Correlation Matrix (Key Variables) ---")
corr_vars = ['AI_intensity', 'Gaza', 'experience', 'education', 'english']
corr_matrix = df[corr_vars].corr()
print(corr_matrix.round(3).to_string())

# =========================================================
# STEP 5 - REGRESSION MODELS
# =========================================================

print("\n" + "=" * 60)
print("STEP 5 - REGRESSION MODELS")
print("=" * 60)

def print_regression_results(model, model_name):
    print(f"\n  {model_name}:")
    print(f"  N={int(model.nobs)}, Adj R2={model.rsquared_adj:.3f}, F-pval={model.f_pvalue:.3f}")
    print(f"  {'Variable':<35} {'Coef':>10} {'SE':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*70}")

    key_vars = ['Intercept', 'AI_intensity', 'Gaza', 'AI_x_Gaza',
                'competitiveness', 'comp_x_Gaza',
                'experience', 'english', 'education', 'age', 'gender']

    for var in key_vars:
        if var in model.params.index:
            coef = model.params[var]
            se = model.bse[var]
            pval = model.pvalues[var]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '†' if pval < 0.1 else ''
            print(f"  {var:<35} {coef:>10.3f} {se:>10.3f} {pval:>10.3f} {sig:>5}")

    skill_vars = [v for v in model.params.index if v.startswith('skill_')]
    for var in skill_vars:
        coef = model.params[var]
        se = model.bse[var]
        pval = model.pvalues[var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '†' if pval < 0.1 else ''
        print(f"  {var:<35} {coef:>10.3f} {se:>10.3f} {pval:>10.3f} {sig:>5}")

skill_cols = [c for c in df.columns if c.startswith('skill_')]
skill_formula = ' + '.join(skill_cols) if skill_cols else '1'

all_results = {}

# -----------------------------------------------------------
# OUTCOME 1: Monthly Earnings (Q10)
# -----------------------------------------------------------
print("\n" + "-" * 60)
print("OUTCOME 1: MONTHLY EARNINGS (Q10)")
print(f"N = {df['earnings'].notna().sum()} (7 excluded for Other response)")
print("-" * 60)

m1_earn = smf.ols('earnings ~ AI_intensity', data=df).fit()
print_regression_results(m1_earn, "Model 1 - Bivariate (tests H1)")

m2_earn = smf.ols('earnings ~ AI_intensity + Gaza', data=df).fit()
print_regression_results(m2_earn, "Model 2 - Add Gaza (tests H1 + H2)")

m3_earn = smf.ols('earnings ~ AI_intensity + Gaza + AI_x_Gaza', data=df).fit()
print_regression_results(m3_earn, "Model 3 - Add Interaction (tests H1 + H2 + H3)")

m4_earn = smf.ols(f'earnings ~ AI_intensity + Gaza + AI_x_Gaza + experience + english + {skill_formula}', data=df).fit()
print_regression_results(m4_earn, "Model 4 - Parsimonious Controls (PRIMARY)")

m4b_earn = smf.ols(f'earnings ~ AI_intensity + Gaza + AI_x_Gaza + experience + english + education + age + gender + {skill_formula}', data=df).fit()
print_regression_results(m4b_earn, "Model 4b - Full Controls (Robustness)")

all_results['earnings'] = {'m1': m1_earn, 'm2': m2_earn, 'm3': m3_earn, 'm4': m4_earn, 'm4b': m4b_earn}

# -----------------------------------------------------------
# OUTCOME 2: Projects per Month (Q11)
# -----------------------------------------------------------
print("\n" + "-" * 60)
print("OUTCOME 2: PROJECTS PER MONTH (Q11)")
print(f"N = {df['projects'].notna().sum()}")
print("-" * 60)

m1_proj = smf.ols('projects ~ AI_intensity', data=df).fit()
print_regression_results(m1_proj, "Model 1 - Bivariate (tests H1)")

m2_proj = smf.ols('projects ~ AI_intensity + Gaza', data=df).fit()
print_regression_results(m2_proj, "Model 2 - Add Gaza (tests H1 + H2)")

m3_proj = smf.ols('projects ~ AI_intensity + Gaza + AI_x_Gaza', data=df).fit()
print_regression_results(m3_proj, "Model 3 - Add Interaction (tests H1 + H2 + H3)")

m4_proj = smf.ols(f'projects ~ AI_intensity + Gaza + AI_x_Gaza + experience + english + {skill_formula}', data=df).fit()
print_regression_results(m4_proj, "Model 4 - Parsimonious Controls (PRIMARY)")

m4b_proj = smf.ols(f'projects ~ AI_intensity + Gaza + AI_x_Gaza + experience + english + education + age + gender + {skill_formula}', data=df).fit()
print_regression_results(m4b_proj, "Model 4b - Full Controls (Robustness)")

all_results['projects'] = {'m1': m1_proj, 'm2': m2_proj, 'm3': m3_proj, 'm4': m4_proj, 'm4b': m4b_proj}

# -----------------------------------------------------------
# OUTCOME 3: Perceived Competitiveness (Q17)
# -----------------------------------------------------------
print("\n" + "-" * 60)
print("OUTCOME 3: PERCEIVED COMPETITIVENESS (Q17)")
print(f"N = {df['competitiveness'].notna().sum()} (AI adopters only)")
print("-" * 60)

m1_comp = smf.ols('competitiveness ~ AI_intensity', data=df).fit()
print_regression_results(m1_comp, "Model 1 - Bivariate")

m2_comp = smf.ols('competitiveness ~ AI_intensity + Gaza', data=df).fit()
print_regression_results(m2_comp, "Model 2 - Add Gaza")

m3_comp = smf.ols('competitiveness ~ AI_intensity + Gaza + AI_x_Gaza', data=df).fit()
print_regression_results(m3_comp, "Model 3 - Add Interaction (tests H3)")

m4_comp = smf.ols(f'competitiveness ~ AI_intensity + Gaza + AI_x_Gaza + experience + english + {skill_formula}', data=df).fit()
print_regression_results(m4_comp, "Model 4 - Parsimonious Controls (PRIMARY)")

m4b_comp = smf.ols(f'competitiveness ~ AI_intensity + Gaza + AI_x_Gaza + experience + english + education + age + gender + {skill_formula}', data=df).fit()
print_regression_results(m4b_comp, "Model 4b - Full Controls (Robustness)")

all_results['competitiveness'] = {'m1': m1_comp, 'm2': m2_comp, 'm3': m3_comp, 'm4': m4_comp, 'm4b': m4b_comp}

# =========================================================
# STEP 6 - ROBUSTNESS: Q17 AS EXPLANATORY VARIABLE
# =========================================================

print("\n" + "=" * 60)
print("STEP 6 - ROBUSTNESS: Q17 AS PREDICTOR")
print("(Tests whether perceived competitiveness predicts actual outcomes)")
print("=" * 60)

df['comp_x_Gaza'] = df['competitiveness'] * df['Gaza']

print("\n--- Q17 predicts Monthly Earnings (Q10) ---")
rob1 = smf.ols(f'earnings ~ competitiveness + Gaza + comp_x_Gaza + experience + english + {skill_formula}', data=df).fit()
print_regression_results(rob1, "Robustness - Q17 as predictor of Q10")

print("\n--- Q17 predicts Projects per Month (Q11) ---")
rob2 = smf.ols(f'projects ~ competitiveness + Gaza + comp_x_Gaza + experience + english + {skill_formula}', data=df).fit()
print_regression_results(rob2, "Robustness - Q17 as predictor of Q11")

# =========================================================
# STEP 7 - PSM (only if VIF > 5)
# =========================================================

if run_psm:
    print("\n" + "=" * 60)
    print("STEP 7 - PROPENSITY SCORE MATCHING")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    psm_vars = ['experience', 'education', 'english', 'gender']
    psm_data = df[psm_vars + ['Gaza', 'earnings', 'projects', 'competitiveness', 'AI_intensity', 'AI_x_Gaza']].dropna()

    X_psm = psm_data[psm_vars].values
    y_psm = psm_data['Gaza'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_psm)

    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(X_scaled, y_psm)
    psm_data = psm_data.copy()
    psm_data['pscore'] = lr.predict_proba(X_scaled)[:, 1]

    gaza_psm = psm_data[psm_data['Gaza'] == 1].copy()
    non_gaza_psm = psm_data[psm_data['Gaza'] == 0].copy()

    matched_pairs = []
    used_controls = set()

    for idx, row in gaza_psm.iterrows():
        available = non_gaza_psm[~non_gaza_psm.index.isin(used_controls)]
        if len(available) == 0:
            break
        distances = abs(available['pscore'] - row['pscore'])
        best_match = distances.idxmin()
        matched_pairs.append((idx, best_match))
        used_controls.add(best_match)

    matched_indices = [i for pair in matched_pairs for i in pair]
    matched_df = psm_data.loc[matched_indices]

    print(f"\nMatched sample: {len(matched_df)} respondents")

    psm_earn = smf.ols('earnings ~ AI_intensity + Gaza + AI_x_Gaza + experience + english', data=matched_df).fit()
    print_regression_results(psm_earn, "PSM - Earnings")

    psm_proj = smf.ols('projects ~ AI_intensity + Gaza + AI_x_Gaza + experience + english', data=matched_df).fit()
    print_regression_results(psm_proj, "PSM - Projects")

else:
    print("\n" + "=" * 60)
    print("STEP 7 - PSM NOT REQUIRED (all VIF < 5)")
    print("=" * 60)

# =========================================================
# STEP 8 - SUMMARY OF KEY COEFFICIENTS
# =========================================================

print("\n" + "=" * 60)
print("STEP 8 - SUMMARY OF KEY COEFFICIENTS")
print("=" * 60)

def get_coef_str(model, var):
    if var in model.params.index:
        coef = model.params[var]
        pval = model.pvalues[var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else '†' if pval < 0.1 else ''
        return f"{coef:.3f}{sig}"
    return "N/A"

print("\n--- H1: AI Intensity Coefficient (beta1) ---")
print(f"\n{'Outcome':<25} {'M1':>12} {'M2':>12} {'M3':>12} {'M4':>12}")
print("-" * 65)
for outcome, label in [('earnings', 'Monthly Earnings'), ('projects', 'Projects/Month'), ('competitiveness', 'Competitiveness')]:
    res = all_results[outcome]
    row = [get_coef_str(res[m], 'AI_intensity') for m in ['m1', 'm2', 'm3', 'm4']]
    print(f"{label:<25} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12}")

print("\n--- H2: Gaza Coefficient (beta2) ---")
print(f"\n{'Outcome':<25} {'M2':>12} {'M3':>12} {'M4':>12}")
print("-" * 55)
for outcome, label in [('earnings', 'Monthly Earnings'), ('projects', 'Projects/Month'), ('competitiveness', 'Competitiveness')]:
    res = all_results[outcome]
    row = [get_coef_str(res[m], 'Gaza') for m in ['m2', 'm3', 'm4']]
    print(f"{label:<25} {row[0]:>12} {row[1]:>12} {row[2]:>12}")

print("\n--- H3: Interaction AI x Gaza Coefficient (beta3) ---")
print(f"\n{'Outcome':<25} {'M3':>12} {'M4':>12}")
print("-" * 45)
for outcome, label in [('earnings', 'Monthly Earnings'), ('projects', 'Projects/Month'), ('competitiveness', 'Competitiveness')]:
    res = all_results[outcome]
    row = [get_coef_str(res[m], 'AI_x_Gaza') for m in ['m3', 'm4']]
    print(f"{label:<25} {row[0]:>12} {row[1]:>12}")

# =========================================================
# STEP 9 - Q18 QUALITATIVE RESPONSES
# =========================================================

print("\n" + "=" * 60)
print("STEP 9 - Q18 QUALITATIVE RESPONSES (Gaza AI Users)")
print("=" * 60)

gaza_ai = df[(df['Gaza'] == 1) & (df['AI_adoption'] == 'Yes')]
q18_responses = gaza_ai['Q18'].dropna()

print(f"\nTotal Gaza AI-adopting respondents: {len(gaza_ai)}")
print(f"Q18 responses received: {len(q18_responses)}")
print(f"Q18 non-response: {len(gaza_ai) - len(q18_responses)}")

print("\n--- All Q18 Responses ---")
for i, response in enumerate(q18_responses, 1):
    print(f"\n[{i}] {response}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nSignificance codes: *** p<0.001 | ** p<0.01 | * p<0.05 | . p<0.1")
print("Primary models: experience + english + skill category as controls")
print("Robustness: education + age + gender added in Model 4b")