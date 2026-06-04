"""
ids — AI-Based Intrusion Detection System
COM668 Final Year Project | Abdulbosit Abdurazzakov | B00979380

Package layout
--------------
ids.data.loader          -- raw CSV loading and cleaning
ids.features.preprocessing -- scaling, encoding, oversampling
ids.models.train         -- RF and IF training helpers
ids.models.predict       -- inference (classify_flow)
ids.models.evaluate      -- metrics, reports, plots
ids.explainability.shap_analysis -- SHAP feature importance
"""

__version__ = "1.0.0"
__author__  = "Abdulbosit Abdurazzakov"
