
import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from data_loading import data_loading, files

class GraviteTrainer:

  def __init__(
      self,
      data_path,
      model_out_path,
      target_col,
      positive_classes,
      num_cols,
      bin_cols,
      cat_cols,
      threshold: float = 0.50,
      
  ):
    self.data_path = data_path
    self.model_out_path = model_out_path
    self.num_cols = num_cols
    self.bin_cols = bin_cols
    self.cat_cols = cat_cols
    self.target_col = target_col
    self.positive_classes = positive_classes
    self.feature_cols = num_cols + bin_cols + cat_cols
    self.threshold = threshold

    self.X = None
    self.y = None
    self.pipeline = None
  
  def load_data(self):
    print(f"Loading full dataset from {self.data_path}")
    df = self.data_path

    self.X = df[self.feature_cols].copy()
    self.y = df[self.target_col].isin(self.positive_classes).astype(int)

    # object or string columns -> category dtype converting
    for col in self.cat_cols:
      self.X[col] = self.X[col].astype('category')

    print(f"Selected {self.X.shape[1]} features out of {df.shape[1]} total columns ({len(self.X):,} rows).")
    return self
  
  def train_final_model(self):
    print("Traning Final HistGradientBoostingClassifer on 100% data...")
    self.pipeline  = Pipeline(steps=[
        ('hgbc', HistGradientBoostingClassifier(
            categorical_features='from_dtype',
            class_weight='balanced',
            random_state=42,
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=255,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])
    self.pipeline.fit(self.X, self.y)
    print(f"Training stopped at iteration: {self.pipeline.named_steps['hgbc'].n_iter_}")
    return self

  def export_bundle(self):
    print("Exporting model...")
    os.makedirs(os.path.dirname(self.model_out_path), exist_ok=True)

    cat_dtypes = {col: self.X[col].dtype for col in self.cat_cols}

    bundle = {
        'model': self.pipeline,
        'features': self.feature_cols,
        'num_cols': self.num_cols,
        'bin_cols': self.bin_cols,
        'cat_cols': self.cat_cols,
        'cat_dtypes': cat_dtypes,
        'threshold': self.threshold,
        'model_name': 'HistGradientBoostingClassifer',
        'validated_metrics': {
            'test_ksi_f1': 0.48,
            'test_ksi_recall': 0.66,
            'test_ksi_prediction': 0.37,
            'macro_f1': 0.61
        },

    }

    joblib.dump(bundle, self.model_out_path, compress=3)
    print("Model bundle saved!")
    return self

  def run(self):
    return self.load_data().train_final_model().export_bundle()


if __name__ == '__main__':
  BASE_DIR = '/content/drive/MyDrive/MINOR_PROJECT_FILES'

  NUM_COLS = [
    'speed_limit', 'number_of_vehicles', 'min_driver_age', 'max_driver_age','max_vehicle_age', 'max_engine_cc', 'male_driver_share', 'avg_casualty_imd'
  ]

  BIN_COLS = [
      'has_cycle', 'has_motorcycle', 'has_goods_vehicle', 'has_bus', 'has_vulnerable_vehicle', 'has_overtaking', 'has_turning', 'has_vulnerable_casualty', 'has_escooter_casualty', 'has_pedestrian_movement', 'trunk_road_flag'
  ]

  CAT_COLS = [
      'urban_or_rural_area', 'road_type', 'light_conditions', 'weather_conditions', 'road_surface_conditions',
      'special_conditions_at_site', 'carriageway_hazards','junction_detail', 'junction_control', 'day_of_week', 'hour'
  ]

  trainer = GraviteTrainer(
      data_path=data_loading(files['final_df']),
      model_out_path=f"{BASE_DIR}/models/gravite_hgbc_bundle.joblib",
      target_col='collision_severity',
      positive_classes=[1, 2],
      num_cols=NUM_COLS,
      bin_cols=BIN_COLS,
      cat_cols=CAT_COLS,
      threshold=0.50
  )
  trainer.run()


