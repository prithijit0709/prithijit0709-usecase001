import numpy as np
import pandas as pd

from app.streamlit_app import SELECTION_COL, _prepare_sttm_editor_df


def test_prepare_sttm_editor_converts_empty_transformation_logic_to_text():
    source = pd.DataFrame(
        {
            "source_column": ["id", "value"],
            "transformation_logic": [np.nan, np.nan],
        }
    )

    result = _prepare_sttm_editor_df(source)

    assert result[SELECTION_COL].tolist() == [True, True]
    assert result["transformation_logic"].tolist() == ["", ""]
    assert all(isinstance(value, str) for value in result["transformation_logic"])
    assert source["transformation_logic"].isna().all()
