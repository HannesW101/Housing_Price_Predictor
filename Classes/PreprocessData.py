from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


class PreprocessData:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        # Drop the 'Id' column as it is not useful for prediction
        self.train_data.drop('Id', axis=1, inplace=True)
        self.test_data.drop('Id', axis=1, inplace=True)

        # Separate features from training data
        self.x_train = train_data.drop('SalePrice', axis=1)

        # Identifying numerical and categorical features
        numeric_features = self.x_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.x_train.select_dtypes(include=['object']).columns

        # Define preprocessing pipelines for both numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        # Combine both numeric and categorical transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def process_x_train(self):
        return self.preprocessor.fit_transform(self.x_train)

    def process_x_test(self):
        return self.preprocessor.transform(self.test_data)
