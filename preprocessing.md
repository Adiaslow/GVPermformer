1. Load the CSV
2. Filter by PAMPA (remove entries with values under -9) and remove duplicate molecules.
3. Compute cheminformatic properties, transform to graph, compute graph related properties.
4. Split into train, test, val subsets with k-fold stratification to ensure even distribution of molecules with respect to properties across all subsets and folds.
5. Save a torch geometric dataset.
