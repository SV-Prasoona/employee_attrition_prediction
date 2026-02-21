from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_features(df):

    le = LabelEncoder()

    categorical_cols = ["Department", "OverTime", "PromotionLast5Years", "Attrition"]

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


def split_features(df):
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    return X, y


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler