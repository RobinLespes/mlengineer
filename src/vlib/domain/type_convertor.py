def convert_features_to_float(X):
    X_converted = X.copy()
    for col in X_converted.columns:
        X_converted[col] = X_converted[col].astype(float)
    return X_converted
