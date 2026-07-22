Train a linear regression model on the dataset at /app/data/train.csv. The CSV has two columns: x (feature) and y (target). The relationship is approximately linear.

Produce a file at /output/predictions.csv with two columns:
- id: integer index matching the row order in /app/data/test.csv (0-indexed)
- prediction: your model's predicted y value for each test row

Use only the x column from /app/data/test.csv as input. Do not peek at the test labels.
