"""SVM decision-boundary visualization for the Audi dataset.

This adapts the breast-cancer SVM example to `data/audi.csv`.
It uses two numeric features (`price` and `engineSize`) so boundaries
can be plotted in 2D. The output SVG is printed to stdout (StringIO),
so it can be embedded in HTML like the prior examples.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import StringIO


URL = (
	'https://raw.githubusercontent.com/EnricoDiGioia/Machine-Learning/refs/heads/main/data/audi.csv'
)


def load_and_prepare(url=URL):
	df = pd.read_csv(url)

	# Keep only the two features we will plot plus the target
	cols = ['price', 'engineSize', 'fuelType']
	for c in cols:
		if c not in df.columns:
			raise KeyError(f"Expected column '{c}' not found in dataset")

	data = df[cols].copy()

	# Fill missing numeric values with median
	data['price'].fillna(data['price'].median(), inplace=True)
	data['engineSize'].fillna(data['engineSize'].median(), inplace=True)

	# Drop any remaining rows with missing target
	data = data.dropna(subset=['fuelType'])

	# Encode target
	le = LabelEncoder()
	y = le.fit_transform(data['fuelType'])

	# Features (2D)
	X = data[['price', 'engineSize']].to_numpy(dtype=float)

	# Scale features for SVM
	scaler = StandardScaler()
	Xs = scaler.fit_transform(X)

	return Xs, y, scaler, le


def plot_svm_boundaries(X, y, class_names):
	# Match the original example's figure size and layout
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

	kernels = {
		'linear': ax1,
		'sigmoid': ax2,
		'poly': ax3,
		'rbf': ax4,
	}

	for k, ax in kernels.items():
		svm = SVC(kernel=k, C=1)
		svm.fit(X, y)

		DecisionBoundaryDisplay.from_estimator(
			svm,
			X,
			response_method="predict",
			alpha=0.8,
			cmap="Pastel1",
			ax=ax,
		)

		# Scatter points similar to the original example
		ax.scatter(
			X[:, 0], X[:, 1],
			c=y,
			s=20, edgecolors="k"
		)
		ax.set_title(k)
		ax.set_xticks([])
		ax.set_yticks([])

	fig.tight_layout()
	return fig


def main():
	X, y, scaler, le = load_and_prepare()

	fig = plot_svm_boundaries(X, y, class_names=le.classes_)

	# Display/save exactly like the original example: StringIO + print
	buffer = StringIO()
	plt.savefig(buffer, format='svg', transparent=True)
	print(buffer.getvalue())

	plt.close()

main()
