from typing import Any, Dict
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

datasets: Dict[str, Dict[str, Any]] = {
    "credit": {
        "name": "credit",
        "url": "https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients",
        "columns": { 
            "LIMIT_BAL": "numerical",
            "SEX": "categorical",
            "EDUCATION": "categorical",
            "MARRIAGE": "categorical",
            "AGE": "numerical",
            "PAY_0": "categorical",
            "PAY_2": "categorical",
            "PAY_3": "categorical",
            "PAY_4": "categorical",
            "PAY_5": "categorical",
            "PAY_6": "categorical",
            "BILL_AMT1": "numerical",
            "BILL_AMT2": "numerical",
            "BILL_AMT3": "numerical",
            "BILL_AMT4": "numerical",
            "BILL_AMT5": "numerical",
            "BILL_AMT6": "numerical",
            "PAY_AMT1": "numerical",
            "PAY_AMT2": "numerical",
            "PAY_AMT3": "numerical",
            "PAY_AMT4": "numerical",
            "PAY_AMT5": "numerical",
            "PAY_AMT6": "numerical",
            },
        "target": "default payment next month",
        "num_cols": ['LIMIT_BAL', 'AGE','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' ], 
        "cat_cols": {
            "SEX": 2,
            "EDUCATION": 7, 
            "MARRIAGE": 4, 
            "PAY_0": 11, 
            "PAY_2": 11, 
            "PAY_3": 11, 
            "PAY_4": 11, 
            "PAY_5": 10, 
            "PAY_6": 10, 
            },
        "drop_cols": ['ID'],
        "classification": {
            "model": KNeighborsClassifier,
            "class-case": "binary"
            },
        "optimal_parameters": {
            "ordinary_case": {
                "batch_size": 64,
                "hint_rate": 0.1,
                "alpha": 10
                }, 
            "extra_50": {
                "batch_size": 256,
                "hint_rate": 0.1,
                "alpha": 10
                }, 
            "extra_100": {
                "batch_size": 128,
                "hint_rate": 0.1,
                "alpha": 10
                }, 
            }
        }, 
    "letter": {
        "name": "letter",
        "url": "https://archive.ics.uci.edu/ml/datasets/letter+recognition",
        "columns": { 
            "col2": "numerical",
            "col3": "numerical",
            "col4": "numerical",
            "col5": "numerical",
            "col6": "numerical",
            "col7": "numerical",
            "col8": "numerical",
            "col9": "numerical",
            "col10": "numerical",
            "col11": "numerical",
            "col12": "numerical",
            "col13": "numerical",
            "col14": "numerical",
            "col15": "numerical",
            "col16": "numerical",
            "col17": "numerical",
            },
        "target": "col1",
        "num_cols": ['col2', 'col3','col4', 'col5','col6', 'col7','col8', 'col9','col10', 'col11',
                     'col12', 'col13','col14', 'col15','col16', 'col17'], 
        "cat_cols": [],
        "classification": {
            "model": KNeighborsClassifier,
            "class-case": "multiclass"
            },
        "optimal_parameters": {
            "ordinary_case": {
                "batch_size": 64,
                "hint_rate": 0.5,
                "alpha": 10
                }, 
            "extra_50": {
                "batch_size": 256,
                "hint_rate": 0.5,
                "alpha": 10
                }, 
            "extra_100": {
                "batch_size": 256,
                "hint_rate": 0.5,
                "alpha": 10
                }, 
            }
        }, 
    "bank": {
        "name": "bank",
        "url": "https://archive.ics.uci.edu/ml/datasets/Bank+Marketing (dataset nr 1, bank-additional-full.csv)",
        "columns": { 
            "age": "numerical",
            "job": "categorical",
            "marital": "categorical",
            "education": "categorical",
            "default": "categorical",
            "housing": "categorical",
            "loan": "categorical",
            "contact": "categorical",
            "month": "categorical",
            "day_of_week": "categorical",
            "duration": "numerical",
            "campaign": "numerical",
            "pdays": "numerical",
            "previous": "numerical",
            "poutcome": "categorical",
            "emp.var.rate": "numerical",
            "cons.price.idx": "numerical",
            "cons.conf.idx": "numerical",
            "euribor3m": "numerical",
            "nr.employed": "numerical",
            },
        "target": "y",
        "num_cols": ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'], 
        "cat_cols": {
            'job': 12, 
            'marital': 4,
            'education': 8,
            'default': 3, 
            'housing': 3, 
            'loan': 3, 
            'contact': 2, 
            'month': 10,
            'day_of_week': 5,
            'poutcome': 3,
            },
        "classification": {
            "model": KNeighborsClassifier,
            "class-case": "binary"
            },
        "optimal_parameters": {
            "ordinary_case": {
                "batch_size": 64,
                "hint_rate": 0.1,
                "alpha": 2
                }, 
            "extra_50": {
                "batch_size": 128,
                "hint_rate": 0.1,
                "alpha": 0.5
                }, 
            "extra_100": {
                "batch_size": 64,
                "hint_rate": 0.1,
                "alpha": 10
                }, 
            }
        }, 
    "mushroom": {
        "name": "mushroom",
        "url": "https://www.kaggle.com/datasets/uciml/mushroom-classification",
        "columns": { 
            "cap-shape": "categorical",
            "cap-surface": "categorical",
            "cap-color": "categorical",
            "bruises": "categorical",
            "odor": "categorical",
            "gill-attachment": "categorical",
            "gill-spacing": "categorical",
            "gill-size": "categorical",
            "gill-color": "categorical",
            "stalk-shape": "categorical",
            "stalk-root": "categorical",
            "stalk-surface-above-ring": "categorical",
            "stalk-surface-below-ring": "categorical",
            "stalk-color-above-ring": "categorical",
            "stalk-color-below-ring": "categorical",
            "veil-type": "categorical",
            "veil-color": "categorical",
            "ring-number": "categorical",
            "ring-type": "categorical",
            "spore-print-color": "categorical",
            "population": "categorical",
            "habitat": "categorical",
            },
        "target": "class",
        "num_cols": [], 
        "cat_cols": {
            'cap-shape': 6, 
            'cap-surface': 4, 
            'cap-color': 10, 
            'bruises': 2, 
            'odor': 9, 
            'gill-attachment': 2, 
            'gill-spacing': 2, 
            'gill-size': 2, 
            'gill-color': 12, 
            'stalk-shape': 2, 
            'stalk-root': 5, 
            'stalk-surface-above-ring': 4, 
            'stalk-surface-below-ring': 4, 
            'stalk-color-above-ring': 9, 
            'stalk-color-below-ring': 9, 
            'veil-type': 1, 
            'veil-color': 4, 
            'ring-number': 3, 
            'ring-type': 5, 
            'spore-print-color': 9, 
            'population': 6, 
            'habitat': 7
            },
        "classification": {
            "model": KNeighborsClassifier,
            "class-case": "binary"
            },
        "optimal_parameters": {
            "ordinary_case": {
                "batch_size": 128,
                "hint_rate": 0.1,
                "alpha": 10
                }, 
            "extra_50": {
                "batch_size": 256,
                "hint_rate": 0.1,
                "alpha": 0.5
                }, 
            "extra_100": {
                "batch_size": 256,
                "hint_rate": 0.1,
                "alpha": 2
                }, 
            }
        }, 
    "news": {
        "name": "news",
        "url": "https://archive.ics.uci.edu/ml/datasets/online+news+popularity",
        "columns": { 
            "n_tokens_title": "numerical",
            "n_tokens_content": "numerical",
            "n_unique_tokens": "numerical",
            "n_non_stop_words": "numerical",
            "n_non_stop_unique_tokens": "numerical",
            "num_hrefs": "numerical",
            "num_self_hrefs": "numerical",
            "num_imgs": "numerical",
            "num_videos": "numerical",
            "average_token_length": "numerical",
            "num_keywords": "numerical",
            "data_channel": "categorical",
            "kw_min_min": "numerical",
            "kw_max_min": "numerical",
            "kw_avg_min": "numerical",
            "kw_min_max": "numerical",  
            "kw_max_max": "numerical",
            "kw_avg_max": "numerical",
            "kw_min_avg": "numerical",
            "kw_max_avg": "numerical",     
            "kw_avg_avg": "numerical",
            "self_reference_min_shares": "numerical",
            "self_reference_max_shares": "numerical", 
            "self_reference_avg_sharess": "numerical",
            "weekday": "categorical",
            "is_weekend": "categorical", 
            "LDA_00": "numerical",
            "LDA_01": "numerical",
            "LDA_02": "numerical",  
            "LDA_03": "numerical",
            "LDA_04": "numerical",
            "global_subjectivity": "numerical",  
            "global_sentiment_polarity": "numerical",
            "global_rate_positive_words": "numerical",
            "global_rate_negative_words": "numerical",  
            "rate_positive_words": "numerical",  
            "rate_negative_words": "numerical",
            "avg_positive_polarity": "numerical",
            "min_positive_polarity": "numerical",  
            "max_positive_polarity": "numerical",  
            "avg_negative_polarity": "numerical",
            "min_negative_polarity": "numerical",
            "max_negative_polarity": "numerical",     
            "title_subjectivity": "numerical",  
            "title_sentiment_polarity": "numerical",
            "abs_title_subjectivity": "numerical",
            "abs_title_sentiment_polarity": "numerical",  
            },
        "target": "shares",
        "drop_cols": ['url', 'timedelta'],
        "num_cols": ['n_tokens_title','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','num_hrefs','num_self_hrefs','num_imgs','num_videos','average_token_length',
                     'num_keywords', 'kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares',
                     'self_reference_avg_sharess','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words','rate_positive_words',
                     'rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity','avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity',
                     'abs_title_subjectivity','abs_title_sentiment_polarity'], 
        "cat_cols": {'data_channel': 7, 'weekday': 7, 'is_weekend': 2},
        "classification": {
            "model": LinearRegression,
            "class-case": "continuous"
            },
        "optimal_parameters": {
            "ordinary_case": {
                "batch_size": 256,
                "hint_rate": 0.1,
                "alpha": 2
                }, 
            "extra_50": {
                "batch_size": 256,
                "hint_rate": 0.1,
                "alpha": 1
                }, 
            "extra_100": {
                "batch_size": 256,
                "hint_rate": 0.1,
                "alpha": 10
                }, 
            }
        }, 
}

