
import pandas as pd
import re
from urllib.parse import urlparse

SHORTENERS = {'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly', 'buff.ly'}
SUSPICIOUS_TLDS = {'xyz', 'gq', 'cf', 'tk', 'ml', 'buzz', 'loan', 'click'}

def extract_enhanced_features(df, model_feature_names=None):
    def normalize_url(url):
        return f"https://{url}" if not str(url).startswith(('http://', 'https://')) else url

    def is_ip(domain): return 1 if re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', domain) else 0
    def domain_length(domain): return len(domain)
    def contains_numbers(domain): return 1 if re.search(r'\d', domain) else 0
    def get_tld(domain): parts = domain.split('.'); return parts[-1] if len(parts) > 1 else 'unknown'
    def path_depth(url): return urlparse(url).path.count('/')
    def has_keywords(url): return 1 if any(k in url.lower() for k in ['free', 'casino', 'loan', 'bonus']) else 0
    def cyrillic_in_url(url): return 1 if re.search(r'[\u0400-\u04FF]', url) else 0
    def is_shortener(domain): return 1 if domain in SHORTENERS else 0
    def has_ref_param(url): return 1 if any(k in url.lower() for k in ['ref=', 'utm_']) else 0
    def is_suspicious_tld(tld): return 1 if tld in SUSPICIOUS_TLDS else 0

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

    df = pd.get_dummies(df, columns=['tld'], drop_first=True)

    if model_feature_names:
        for col in model_feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[model_feature_names]

    return df
