


Per Northwestern University’s academic integrity rules, source code from course assignments may not be made public.
If you’re a recruiter or hiring manager interested in viewing the actual implementations, please contact me with your GitHub username or email — I’ll grant temporary read-only access to the private repository upon request.


# ai-ml-portfolio


## Data Science in Business Intellegence Overview

### Classical ML
• Decision-tree regression with cost-complexity pruning
• PCA + OLS pipeline for aggressive dimensionality reduction
• RBF-kernel SVM tuned via cross-validated grid search
### Deep Learning
• Custom Keras Sequential model featuring Lambda layers and non-standard activations
### LLM-powered Retrieval
• PDF-grounded QA system built with LangChain, Pinecone, and GPT-3.5

Each script loads real data, trains or tunes the model, and outputs clear metrics or deliverables—demonstrating full-stack proficiency from data prep to interpretable results.

## Intro to Artifical Intellegence 
### Classical Search Algorithms
	•	Implemented BFS, DFS, Greedy Best-First Search, and A*.
	•	Tested on Evanston & Chicago maps; BFS/DFS explore by adjacency, GBFS by straight-line heuristic, A* by travel-time + heuristic.
Outcome: all four solvers return correct paths on public & hidden tests.

⸻

### Checkers AI (Minimax + Alpha-Beta)
	•	Built move-quality metrics (move-count gap, capture count, king-hopefuls).
	•	Minimax with alpha-beta selects the highest-value line.
Outcome: passes all tests; consistently generates strong moves against reference players.

⸻

### Knowledge Base & Inference Engine
	•	Forward-chaining with unification and rule-currying; supports assertion, query, and retraction.
Outcome: dynamic KB correctly infers and retracts facts in every supplied scenario.

⸻

### Naïve Bayes Sentiment Classifier
	•	Add-one smoothing, text cleaning, stemming, stop-word removal; tried TF-IDF & bigrams.
Outcome: meets required F-score thresholds on positive (≥ 0.90) and negative (≥ 0.60) reviews.

⸻

### Bayesian-Network Inference
	•	Recursive joint-probability calculator and P(H | E) conditional queries with normalization.
Outcome: engine answers all test queries correctly, handling partial evidence and hidden variables.

⸻

### Neural-Network Hyper-parameter Study
	•	Ran controlled FFNN experiments over learning-rate, batch-size, epochs, dropout, and hidden-layer width on MNIST.
	•	Logged training/validation curves, runtime, and final test accuracy.
Outcome: delivered a concise report of how each hyper-parameter (alone and in combination) impacts convergence speed and performance.

## Intro to ML

### K-Nearest Neighbors & Polynomial Regression
Implemented:
	•	Core metrics: MSE + two fairness scores (demographic-parity, equalized-odds)
	•	Distance kernels: Euclidean, Manhattan, Minkowski
	•	Synthetic polynomial data generator
	•	Closed-form polynomial regression (normal-equation)
	•	Tunable k-NN classifier / regressor
Outcome – Entire toolkit passes the rubric’s autograder, verifying metric correctness, distance functions, and model behaviour.

⸻

### Naive Bayes + EM — semi-supervised text classification
Implemented:
	•	Stable softmax / logsumexp and sparse-matrix helpers
	•	Multinomial NB with add-λ smoothing (fully-supervised)
	•	EM loop for semi-supervised NB (soft-label E-step, count re-estimation M-step) until log-likelihood convergence
	•	End-to-end pipeline (data load → train → eval, reporting P/R/F1)
Outcome – From a partly labelled corpus the system self-labels the unlabeled pool, meets required F-scores on both classes, and clears every hidden test.

⸻

### Perceptron, Mini-MLP & Regularization (NumPy-only)
Implemented:
	•	Online Perceptron with custom spiral feature mapping
	•	Two-layer MLP: forward pass, back-prop, fit() loop, ReLU, MSE loss, L2 weight decay
Outcome – Models learn the two-armed spiral perfectly, demonstrating raw-NumPy gradient mechanics and regularised deep learning without high-level frameworks.

⸻

### Reinforcement Learning & Fairness
Implemented:
	•	Tabular Q-Learning and SARSA (ε-greedy, α-decay, γ-tunable)
	•	Gymnasium wrapper for episodic training / logging
	•	Fairness-aware reward-shaping on a “toy-loan” MDP
Outcome – Agents master classic control tasks and show how reward design can reduce or amplify policy bias, with all RL and fairness checks passing the autograder.
