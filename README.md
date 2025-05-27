
# 🧠 Backlink Spam Classifier

This repo contains a machine learning system to classify backlinks (URLs) as spam or not spam, using only URL features.

## 📦 Features Considered
- Shorteners (`goo.gl`, `bit.ly`)
- Referral params (`?ref=...`)
- Suspicious TLDs (`.xyz`, `.cf`, `.tk`, etc.)
- Spam keywords (`free`, `casino`, etc.)
- Cyrillic characters
- IP-based domains

## 🚀 How to Use

1. Clone the repo
2. Run `train_model.py` to train the model
3. Launch Streamlit app:
```bash
streamlit run app.py
```

## 📁 Files

- `updated_training_data.csv`: labeled training data
- `enhanced_feature_extractor.py`: feature logic
- `train_model.py`: training script
- `app.py`: Streamlit interface
- `requirements.txt`: dependencies

## 🔍 Sample Spam URLs Handled

- `goo.gl` links
- Suspicious referral URLs
- `.it`, `.xyz`, `.tk` domains
