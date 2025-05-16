
from ..model.constants import Names_Models
from ..data.dset.constants import (
    Names_Levels,
    Names_q_Squared_Vetos,
    Nums_Events_Per_Set
)


class Info_Index:

    def __init__(self):

        self.names = self._Names_Index()

        self.values = self._Values_Index()

    class _Names_Index:

        level = "Level"
        
        q2_veto = "q^2 Veto"
        
        method = "Method"
        
        events_per_set = "Events / Set"

        tuple_ = (
            level,
            q2_veto,
            method,
            events_per_set,
        )

    class _Values_Index:

        tuple_ = (
            Names_Levels().tuple_, 
            Names_q_Squared_Vetos().tuple_,
            Names_Models().tuple_,
            Nums_Events_Per_Set().tuple_,
        )


class Names_Kinds_Items:

    mse = "MSE"
    
    mae = "MAE"
    
    np_std = "Std. at NP"
    
    np_mean = "Mean at NP"
    
    np_bias = "Bias at NP"

    tuple_ = (
        mse,
        mae,
        np_std,
        np_mean,
        np_bias,
    )