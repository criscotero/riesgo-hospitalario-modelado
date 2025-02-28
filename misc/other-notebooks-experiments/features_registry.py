aggregated = [
    'r5hosp1y', 'r5shlt', 'r5hltc', 'r5adltot6', 'r5adltot6m', 'r5adltot6a',
    'r5adla', 'r5adlam', 'r5adlaa', 'r5adlfive', 'r5adlfivem', 'r5adlfivea',
    'r5adla_m', 'r5adlam_m', 'r5adlaa_m', 'r5adlwa', 'r5adlwam', 'r5adlwaa',
    'r5iadlfour', 'r5iadlfourm', 'r5iadlfoura', 'r5mobila', 'r5mobilam',
    'r5mobilaa', 'r5lgmusa', 'r5lgmusam', 'r5lgmusaa', 'r5grossa', 'r5grossam',
    'r5grossaa', 'r5finea', 'r5fineam', 'r5fineaa', 'r5mobilsev',
    'r5mobilsevm', 'r5mobilseva', 'r5uppermob', 'r5uppermobm', 'r5uppermoba',
    'r5lowermob', 'r5lowermobm', 'r5lowermoba', 'r5nagi10', 'r5nagi10m',
    'r5nagi10a', 'r5nagi8', 'r5nagi8m', 'r5nagi8a', 'r5hibpe', 'r5diabe',
    'r5cancre', 'r5respe', 'r5hrtatte', 'r5hearte', 'r5stroke', 'r5arthre',
    'r5rxhibp', 'r5rxdiabo', 'r5rxdiabi', 'r5rxdiab', 'r5cncrchem',
    'r5cncrsurg', 'r5cncrradn', 'r5cncrmeds', 'r5cncrothr', 'r5rxresp',
    'r5rxhrtat', 'r5rxstrok', 'r5rxarthr', 'r5resplmt', 'r5hrtatlmt',
    'r5stroklmt', 'r5arthlmt', 'r5reccancr', 'r5rechrtatt', 'r5recstrok',
    'r5fall', 'r5fallnum', 'r5fallinj', 'r5hip_m', 'r5hip', 'r5urinurg2y',
    'r5urincgh2y', 'r5swell', 'r5breath_m', 'r5fatigue', 'r5fallslp',
    'r5wakent', 'r5wakeup', 'r5rested', 'r5painfr', 'r5painlv', 'r5paina',
    'r5bmi', 'r5weight', 'r5height', 'r5vigact', 'r5drink', 'r5drinkd',
    'r5drinkn', 'r5drinkb', 'r5binged', 'r5smokev', 'r5smoken', 'r5smokef',
    'r5strtsmok', 'r5quitsmok', 'r5cholst', 'r5flusht', 'r5breast',
    'r5mammog', 'r5papsm', 'r5prost', 'r5doctor1y', 'r5doctim1y', 'r5proxy',
    'r5agey', 'r5work'
]

health_self_report = [
    'r5shlt', 'r5hltc'
]

adl_summary = [
    'r5adltot6', 'r5adltot6m', 'r5adltot6a',
    'r5adla', 'r5adlam', 'r5adlaa', 'r5adlfive', 'r5adlfivem', 'r5adlfivea',
    'r5adla_m', 'r5adlam_m', 'r5adlaa_m', 'r5adlwa', 'r5adlwam', 'r5adlwaa',
    'r5iadlfour', 'r5iadlfourm', 'r5iadlfoura'
]

mobility_muscle_tone_motor_control = [
    'r5mobila', 'r5mobilam',
    'r5mobilaa', 'r5lgmusa', 'r5lgmusam', 'r5lgmusaa', 'r5grossa', 'r5grossam',
    'r5grossaa', 'r5finea', 'r5fineam', 'r5fineaa', 'r5mobilsev',
    'r5mobilsevm', 'r5mobilseva', 'r5uppermob', 'r5uppermobm', 'r5uppermoba',
    'r5lowermob', 'r5lowermobm', 'r5lowermoba', 'r5nagi10', 'r5nagi10m',
    'r5nagi10a', 'r5nagi8', 'r5nagi8m', 'r5nagi8a'
]

doctor_diagnosed_ever_have_condition = [
    'r5hibpe', 'r5diabe',
    'r5cancre', 'r5respe', 'r5hrtatte', 'r5hearte', 'r5stroke', 'r5arthre',
    'r5rxhibp', 'r5rxdiabo', 'r5rxdiabi', 'r5rxdiab', 'r5cncrchem',
    'r5cncrsurg', 'r5cncrradn', 'r5cncrmeds', 'r5cncrothr', 'r5rxresp',
    'r5rxhrtat', 'r5rxstrok', 'r5rxarthr', 'r5resplmt', 'r5hrtatlmt',
    'r5stroklmt', 'r5arthlmt'
]

age_of_diagnosis = [
    'r5reccancr', 'r5rechrtatt', 'r5recstrok'
]

falls = [
    'r5fall', 'r5fallnum', 'r5fallinj', 'r5hip_m', 'r5hip'
]

urinary_incontinence = [
    'r5urinurg2y', 'r5urincgh2y'
]

persistent_health_problems = [
    'r5swell', 'r5breath_m', 'r5fatigue'
]

sleep = [
    'r5fallslp', 'r5wakent', 'r5wakeup', 'r5rested'
]

pain = [
    'r5painfr', 'r5painlv', 'r5paina'
]

bmi = [
    'r5bmi', 'r5weight', 'r5height'
]

physical_activity = ['r5vigact']

health_behaviors_drinking = [
    'r5drink', 'r5drinkd',
    'r5drinkn', 'r5drinkb', 'r5binged'
]

health_behaviors_smoking = [
    'r5smokev', 'r5smoken', 'r5smokef',
    'r5strtsmok', 'r5quitsmok'
]

preventive_care_behaviors = [
    'r5cholst', 'r5flusht', 'r5breast',
    'r5mammog', 'r5papsm', 'r5prost'
]

medical_care_utilization = [
    'r5doctor1y', 'r5doctim1y'
]

employment_history = ['r5work']


selected_features_waves_1_2 = [
    'shlt',
    'hltc',
    'adltot6',
    'mobilseva',
    'diabe',
    'hrtatte',
    'stroke',
    'breath_m',
    'bmi',
    'doctor1y',
    'doctim1y',
    'cholst',
    'breast',
    'mammog',
    'papsm',
    'prost',
    'vigact',
    'drinkd',
    'smoken',
    'painfr',
    'fatigue',
    'swell',
    'hibpe',
    'cancre',
    'respe',
    'rxresp',
    'rxhibp',
    'rxdiab',
    'rxhrtat',
    'rxstrok',
    'mwhratio',
    'mbmi',
    'work',
    'proxy',
    'agey',
]


selected_features_waves_3_4_5 = [
    'shlt',
    'hltc',
    'adltot6',
    'mobilseva',
    'diabe',
    'hrtatte',
    'stroke',
    'breath_m',
    'bmi',
    'doctor1y',
    'doctim1y',
    'cholst',
    'flusht',
    'breast',
    'mammog',
    'papsm',
    'prost',
    'vigact',
    'drinkd',
    'smoken',
    'painfr',
    'fatigue',
    'swell',
    'hibpe',
    'cancre',
    'respe',
    'rxresp',
    'rxhibp',
    'rxdiab',
    'rxhrtat',
    'rxstrok',
    'rested',
    'lsatsc3',
    'work',
    'proxy',
    'agey',
]

selected_features = [
    'shlt',
    'hltc',
    'adltot6',
    'mobilseva',
    'diabe',
    'hrtatte',
    'stroke',
    'breath_m',
    'bmi',
    'doctor1y',
    'doctim1y',
    'cholst',
    'breast',
    'mammog',
    'papsm',
    'prost',
    'vigact',
    'drinkd',
    'smoken',
    'painfr',
    'fatigue',
    'swell',
    'hibpe',
    'cancre',
    'respe',
    'rxresp',
    'rxhibp',
    'rxdiab',
    'rxhrtat',
    'rxstrok',
    'work',
    'proxy',
    'agey',
]


selected_features1 = [
    'shlt',
    'hltc',
    'flusht',
    'diabe',
    'hrtatte',
    'stroke',
    'bmi',
    'mwhratio',
    'mbmi',
    'doctor1y',
    'doctim1y',
    'cholst',
    'breast',
    'mammog',
    'papsm',
    'prost',
    'vigact',
    'drinkd',
    'smoken',
    'hibpe',
    'cancre',
    'respe',
    'rxresp',
    'rxhibp',
    'rxdiab',
    'rxhrtat',
    'rxstrok',
    'work',
    'proxy',
    'agey',
]
urban_or_rural = ['hWrural', 'hWrural_m', 'tamloc_12', 'tamloc_15', 'tamloc18']

target_variable_suffix = ['hosp1y']

row_id = 'unhhidnp'

gender_respondent = 'ragender'

gender_spouse_suffix = 'gender'

gender_flag = 'genderf'   # Flag variable to indicate contradictions
