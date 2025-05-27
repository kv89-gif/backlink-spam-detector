
import pandas as pd
import re
from urllib.parse import urlparse

SHORTENERS = {'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly', 'buff.ly'}
SUSPICIOUS_TLDS = {'xyz', 'gq', 'cf', 'tk', 'ml', 'buzz', 'loan', 'click'}

def extract_enhanced_features(df, model_feature_names=None):
    def normalize_url(url):
        try:
            return f"https://{url}" if not str(url).startswith(('http://', 'https://')) else str(url)
        except:
            return "https://unknown"

    def is_ip(domain):
        try:
            return 1 if re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', domain) else 0
        except:
            return 0

    def domain_length(domain):
        try:
            return len(domain)
        except:
            return 0

    def contains_numbers(domain):
        try:
            return 1 if re.search(r'\d', domain) else 0
        except:
            return 0

    def get_tld(domain):
        try:
            parts = domain.split('.')
            return parts[-1] if len(parts) > 1 else 'unknown'
        except:
            return 'unknown'

    def path_depth(url):
        try:
            return urlparse(url).path.count('/')
        except:
            return 0

    def has_keywords(url):
        try:
            return 1 if any(k in str(url).lower() for k in ['free', 'casino', 'loan', 'bonus']) else 0
        except:
            return 0

    def cyrillic_in_url(url):
        try:
            return 1 if re.search(r'[\u0400-\u04FF]', str(url)) else 0
        except:
            return 0

    def is_shortener(domain):
        try:
            return 1 if domain in SHORTENERS else 0
        except:
            return 0

    def has_ref_param(url):
        try:
            return 1 if any(k in str(url).lower() for k in ['ref=', 'utm_']) else 0
        except:
            return 0

    def is_suspicious_tld(tld):
        try:
            return 1 if tld in SUSPICIOUS_TLDS else 0
        except:
            return 0

    def is_generic_profile_path(url):
        try:
            return 1 if any(p in str(url).lower() for p in ['/profile/', '/user/', '/people/', '/member/']) else 0
        except:
            return 0

    # Start transformation
    df['url'] = df.iloc[:, 0].apply(normalize_url)
    df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
    df['is_ip'] = df['domain'].apply(is_ip)
    df['domain_length'] = df['domain'].apply(domain_length)
    df['contains_numbers'] = df['domain'].apply(contains_numbers)
    df['tld'] = df['domain'].apply(get_tld)
    df['path_depth'] = df['url'].apply(path_depth)
    df['has_keywords'] = df['url'].apply(has_keywords)
    df['cyrillic_in_url'] = df['url'].apply(cyrillic_in_url)
    df['is_shortener'] = df['domain'].apply(is_shortener)
    df['has_ref_param'] = df['url'].apply(has_ref_param)
    df['is_suspicious_tld'] = df['tld'].apply(is_suspicious_tld)
    df['is_generic_profile_path'] = df['url'].apply(is_generic_profile_path)

    # One-hot encode TLDs
    df = pd.get_dummies(df, columns=['tld'], drop_first=True)

    # Align with model features
    if model_feature_names is not None:
        existing_cols = set(df.columns)
        for col in model_feature_names:
            if col not in existing_cols:
                df[col] = 0
        df = df[model_feature_names].copy()

    return df
