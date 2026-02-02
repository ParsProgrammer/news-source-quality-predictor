import joblib
import pandas as pd

# Your feature list (paste exactly as used in inference.py)
partisan_all_cues = [ 'broadcasters_inequality_all','broadcasters_inequality_left', 'broadcasters_inequality_right','broadcasters_inequality_minima', 'article_inequality_all',
       'article_inequality_left', 'article_inequality_right', 'article_inequality_minima', 'centrality_degree_all', 'centrality_degree_left', 'centrality_degree_right','centrality_degree_minima', 'centrality_eigen-vector_all',
       'centrality_eigen-vector_left', 'centrality_eigen-vector_right','centrality_eigen-vector_minima', 'centrality_page-rank_all', 'centrality_page-rank_left', 'centrality_page-rank_right','centrality_page-rank_minima', 'centrality_harmonic_all',
       'centrality_harmonic_left', 'centrality_harmonic_right','centrality_harmonic_minima', 'centrality_closeness_all','centrality_closeness_left', 'centrality_closeness_right','centrality_closeness_minima', 'centrality_betweenness_all',
       'centrality_betweenness_left', 'centrality_betweenness_right','centrality_betweenness_minima', 'broadcasts_num-log_all','broadcasts_num-log_left', 'broadcasts_num-log_right', 'broadcasters_num-log_all', 'broadcasters_num-log_left',
       'broadcasters_num-log_right', 'broadcasts_prop-left','broadcasters_prop-left']

FEATURES = ["diversity_qn"] + partisan_all_cues


# Load a few rows of real data (recommended)
data_df = pd.read_csv("../data/cues_US.csv")

X = data_df[FEATURES].head(10)

model = joblib.load("models/elastic_net.joblib")

pred = model.predict(X)

print("Predictions:", pred[:10])
print("n_preds:", len(pred))
