from sklearn.model_selection import train_test_split, KFold, LeaveOneOut

# Trainingsdatensplit nach der "Holdout Method": 80% Trainingsdaten und 20% Testdaten
def holdoutMethod(data):
    X = data.drop(['price'], axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    splits = []
    splits.append((X_train, X_test, y_train, y_test))

    return splits

# Trainingsdatensplit nach "k-cross-validation": Einteilung in 5 gleich große subsets, jedes subset ist einmal der Testdaten subset 
def k_cross_validation(data):
    X = data.drop(['price'], axis=1)
    y = data['price']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, X_test, y_train, y_test))

    return splits


# Trainingsdatensplit nach "leave-one-out": Trainingsdaten alle bis auf Einer. Jeder Datenpunkt ist einmal Testdatenpunkt
def leave_one_out(data):
    X = data.drop(['price'], axis=1)
    y = data['price']

    loo = LeaveOneOut()

    splits = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, X_test, y_train, y_test))

    return splits