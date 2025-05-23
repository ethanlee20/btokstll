
import torch

from ..data.dset.constants import (
    Names_Datasets,
    Names_q_Squared_Vetos,
    Names_Splits,
    Names_Levels,
    Nums_Events_Per_Set
)
from ..data.dset.config import Config_Dataset
from ..model.constants import Names_Models
from ..model.config import Config_Model
from .constants import Paths_Directories


class Config_Experiment_Images:

    def __init__(
        self, 
        value_dc9_np=-0.82,
        num_sets_per_label_nominal=50,
        num_sets_per_label_sens=2000,
        q_squared_veto=Names_q_Squared_Vetos().loose,
        balanced_classes=True,
        std_scale=True,
        shuffle=True,
        num_bins_image=10,
        frac_bkg=0.5,
        loss_fn=torch.nn.MSELoss(),
        learn_rate=4e-4,
        size_batch_train=32,
        size_batch_eval=32,
        num_epochs=80,
        num_epochs_checkpoint=5,
    ):
        
        self.value_dc9_np = value_dc9_np
        self.num_sets_per_label_nominal = num_sets_per_label_nominal
        self.num_sets_per_label_sens = num_sets_per_label_sens

        self._set_dict_kwargs_dset_common(
            q_squared_veto=q_squared_veto,
            balanced_classes=balanced_classes,
            std_scale=std_scale,
            shuffle=shuffle,
            num_bins_image=num_bins_image,
            frac_bkg=frac_bkg,
        )

        self._set_dict_configs_dsets()

        self._set_dict_kwargs_model_common(
            loss_fn=loss_fn,
            learn_rate=learn_rate,
            size_batch_train=size_batch_train,
            size_batch_eval=size_batch_eval,
            num_epochs=num_epochs,
            num_epochs_checkpoint=num_epochs_checkpoint,
        )

        self._set_dict_configs_models()

    def get_config_dset(
        self,
        level,
        num_events_per_set,
        kind,
    ):
        
        return (
            self.dict_configs_dsets
            [level]
            [num_events_per_set]
            [kind]
        )

    def _set_dict_configs_dsets(self):
                
        self.dict_configs_dsets = {

            level : {

                num_events_per_set : (
                    
                    dict(

                        train = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().train,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_nominal,
                                **self.dict_kwargs_dset_common,
                            )
                        ),

                        eval = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().eval_,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_nominal,
                                **self.dict_kwargs_dset_common,
                            )
                        ),

                        eval_sens = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().eval_,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_sens,
                                label_subset=[self.value_dc9_np],
                                sensitivity_study=True,
                                **self.dict_kwargs_dset_common,
                            )
                        ),
                    )
                )

                for num_events_per_set in Nums_Events_Per_Set().tuple_
            }

            for level in Names_Levels().tuple_
        }

    def _set_dict_kwargs_dset_common(
        self,
        q_squared_veto,
        balanced_classes,
        std_scale,
        shuffle,
        num_bins_image,
        frac_bkg,
    ):

        self.dict_kwargs_dset_common = dict(
            name=Names_Datasets().images,
            q_squared_veto=q_squared_veto,
            balanced_classes=balanced_classes,
            std_scale=std_scale,
            shuffle=shuffle,
            num_bins_image=num_bins_image,
            frac_bkg=frac_bkg,
            path_dir_dsets_main=Paths_Directories().dsets_main,
            path_dir_raw_signal=Paths_Directories().raw_signal,
            path_dir_raw_bkg=Paths_Directories().raw_bkg,
        )

    def get_config_model(
        self,
        level,
        num_events_per_set,
    ):
        return (
            self.dict_configs_models
            [level]
            [num_events_per_set]
        )

    def _set_dict_configs_models(self):
        
        self.dict_configs_models = {

            level : {

                num_events_per_set : (
                    
                    Config_Model(
                        config_dset_train=(
                            self.get_config_dset(
                                level=level, 
                                num_events_per_set=num_events_per_set, 
                                kind="train"
                            )
                        ),
                        **self.dict_kwargs_dset_common,
                    )

                )

                for num_events_per_set in Nums_Events_Per_Set().tuple_
            
            }

            for level in Names_Levels().tuple_

        }

    def _set_dict_kwargs_model_common(
        self,
        loss_fn,
        learn_rate,
        size_batch_train,
        size_batch_eval,
        num_epochs,
        num_epochs_checkpoint,
    ):

        self.dict_kwargs_model_common = dict(
            name=Names_Models().cnn,
            path_dir_models_main=Paths_Directories().models_main,
            loss_fn=loss_fn,
            learn_rate=learn_rate,
            size_batch_train=size_batch_train,
            size_batch_eval=size_batch_eval,
            num_epochs=num_epochs,
            num_epochs_checkpoint=num_epochs_checkpoint,
        )


class Config_Experiment_Deep_Sets:
    
    def __init__(
        self, 
        value_dc9_np=-0.82,
        num_sets_per_label_nominal=50,
        num_sets_per_label_sens=2000,
        q_squared_veto=Names_q_Squared_Vetos().loose,
        balanced_classes=True,
        std_scale=True,
        shuffle=True,
        frac_bkg=0.5,
        loss_fn=torch.nn.MSELoss(),
        learn_rate=4e-4,
        size_batch_train=32,
        size_batch_eval=32,
        num_epochs=80,
        num_epochs_checkpoint=5,
    ):
        
        self.value_dc9_np = value_dc9_np
        self.num_sets_per_label_nominal = num_sets_per_label_nominal
        self.num_sets_per_label_sens = num_sets_per_label_sens

        self._set_dict_kwargs_dset_common(
            q_squared_veto=q_squared_veto,
            balanced_classes=balanced_classes,
            std_scale=std_scale,
            shuffle=shuffle,
            frac_bkg=frac_bkg,
        )

        self._set_dict_configs_dsets()

        self._set_dict_kwargs_model_common(
            loss_fn=loss_fn,
            learn_rate=learn_rate,
            size_batch_train=size_batch_train,
            size_batch_eval=size_batch_eval,
            num_epochs=num_epochs,
            num_epochs_checkpoint=num_epochs_checkpoint,
        )

        self._set_dict_configs_models()

    def get_config_dset(
        self,
        level,
        num_events_per_set,
        kind,
    ):
        
        return (
            self.dict_configs_dsets
            [level]
            [num_events_per_set]
            [kind]
        )

    def _set_dict_configs_dsets(self):
                
        self.dict_configs_dsets = {

            level : {

                num_events_per_set : (
                    
                    dict(

                        train = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().train,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_nominal,
                                **self.dict_kwargs_dset_common,
                            )
                        ),

                        eval = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().eval_,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_nominal,
                                **self.dict_kwargs_dset_common,
                            )
                        ),

                        eval_sens = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().eval_,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_sens,
                                label_subset=[self.value_dc9_np],
                                sensitivity_study=True,
                                **self.dict_kwargs_dset_common,
                            )
                        ),
                    )
                )

                for num_events_per_set in Nums_Events_Per_Set().tuple_
            }

            for level in Names_Levels().tuple_
        }

    def _set_dict_kwargs_dset_common(
        self,
        q_squared_veto,
        balanced_classes,
        std_scale,
        shuffle,
        frac_bkg,
    ):

        self.dict_kwargs_dset_common = dict(
            name=Names_Datasets().sets_unbinned,
            q_squared_veto=q_squared_veto,
            balanced_classes=balanced_classes,
            std_scale=std_scale,
            shuffle=shuffle,
            frac_bkg=frac_bkg,
            path_dir_dsets_main=Paths_Directories().dsets_main,
            path_dir_raw_signal=Paths_Directories().raw_signal,
            path_dir_raw_bkg=Paths_Directories().raw_bkg,
        )

    def get_config_model(
        self,
        level,
        num_events_per_set,
    ):
        return (
            self.dict_configs_models
            [level]
            [num_events_per_set]
        )

    def _set_dict_configs_models(self):
        
        self.dict_configs_models = {

            level : {

                num_events_per_set : (
                    
                    Config_Model(
                        config_dset_train=(
                            self.get_config_dset(
                                level=level, 
                                num_events_per_set=num_events_per_set, 
                                kind="train"
                            )
                        ),
                        **self.dict_kwargs_dset_common,
                    )

                )

                for num_events_per_set in Nums_Events_Per_Set().tuple_
            
            }

            for level in Names_Levels().tuple_

        }

    def _set_dict_kwargs_model_common(
        self,
        loss_fn,
        learn_rate,
        size_batch_train,
        size_batch_eval,
        num_epochs,
        num_epochs_checkpoint,
    ):

        self.dict_kwargs_model_common = dict(
            name=Names_Models().deep_sets,
            path_dir_models_main=Paths_Directories().models_main,
            loss_fn=loss_fn,
            learn_rate=learn_rate,
            size_batch_train=size_batch_train,
            size_batch_eval=size_batch_eval,
            num_epochs=num_epochs,
            num_epochs_checkpoint=num_epochs_checkpoint,
        )


class Config_Experiment_Event_by_Event:

    def __init__(
        self, 
        value_dc9_np=-0.82,
        num_sets_per_label_nominal=50,
        num_sets_per_label_sens=2000,
        q_squared_veto=Names_q_Squared_Vetos().loose,
        balanced_classes=True,
        std_scale=True,
        shuffle=True,
        frac_bkg=0.5,
        loss_fn=torch.nn.CrossEntropyLoss(),
        learn_rate=4e-4,
        size_batch_train=32,
        size_batch_eval=32,
        num_epochs=80,
        num_epochs_checkpoint=5,
    ):
        
        self.value_dc9_np = value_dc9_np
        self.num_sets_per_label_nominal = num_sets_per_label_nominal
        self.num_sets_per_label_sens = num_sets_per_label_sens

        self._set_dict_kwargs_dset_common(
            q_squared_veto=q_squared_veto,
            balanced_classes=balanced_classes,
            std_scale=std_scale,
            shuffle=shuffle,
            frac_bkg=frac_bkg,
        )

        self._set_dict_configs_dsets()

        self.config_model = Config_Model(
            name=Names_Models().ebe,
            path_dir_models_main=Paths_Directories().models_main,
            config_dset_train=(
                self.get_config_dset(
                    level=level, 
                    kind="train"
                )
            ),
            loss_fn=loss_fn,

        )

        self._set_dict_kwargs_model_common(
            loss_fn=loss_fn,
            learn_rate=learn_rate,
            size_batch_train=size_batch_train,
            size_batch_eval=size_batch_eval,
            num_epochs=num_epochs,
            num_epochs_checkpoint=num_epochs_checkpoint,
        )

        self._set_dict_configs_models()

    def get_config_dset(
        self,
        level,
        num_events_per_set,
        kind,
    ):
        
        return (
            self.dict_configs_dsets
            [level]
            [num_events_per_set]
            [kind]
        )

    def _set_dict_configs_dsets(self):
                
        self.dict_configs_dsets = {

            level : {

                num_events_per_set : (
                    
                    dict(

                        train = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().train,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_nominal,
                                **self.dict_kwargs_dset_common,
                            )
                        ),

                        eval = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().eval_,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_nominal,
                                **self.dict_kwargs_dset_common,
                            )
                        ),

                        eval_sens = (
                            Config_Dataset(
                                level=level,
                                split=Names_Splits().eval_,
                                num_events_per_set=num_events_per_set,
                                num_sets_per_label=self.num_sets_per_label_sens,
                                label_subset=[self.value_dc9_np],
                                sensitivity_study=True,
                                **self.dict_kwargs_dset_common,
                            )
                        ),
                    )
                )

                for num_events_per_set in Nums_Events_Per_Set().tuple_
            }

            for level in Names_Levels().tuple_
        }

    def _set_dict_kwargs_dset_common(
        self,
        q_squared_veto,
        balanced_classes,
        std_scale,
        shuffle,
        num_bins_image,
        frac_bkg,
    ):

        self.dict_kwargs_dset_common = dict(
            name=Names_Datasets().images,
            q_squared_veto=q_squared_veto,
            balanced_classes=balanced_classes,
            std_scale=std_scale,
            shuffle=shuffle,
            num_bins_image=num_bins_image,
            frac_bkg=frac_bkg,
            path_dir_dsets_main=Paths_Directories().dsets_main,
            path_dir_raw_signal=Paths_Directories().raw_signal,
            path_dir_raw_bkg=Paths_Directories().raw_bkg,
        )

    def get_config_model(
        self,
        level,
    ):
        
        return (
            self.dict_configs_models
            [level]
        )

    def _set_dict_configs_models(self):
        
        self.dict_configs_models = {

            level : (

                Config_Model(
                    config_dset_train=(
                        self.get_config_dset(
                            level=level, 
                            kind="train",
                        )
                    ),
                    **self.dict_kwargs_dset_common,
                )
            )

            for level in Names_Levels().tuple_
        }

    def _set_dict_kwargs_model_common(
        self,
        loss_fn,
        learn_rate,
        size_batch_train,
        size_batch_eval,
        num_epochs,
        num_epochs_checkpoint,
    ):

        self.dict_kwargs_model_common = dict(
            name=Names_Models().ebe,
            path_dir_models_main=Paths_Directories().models_main,
            loss_fn=loss_fn,
            learn_rate=learn_rate,
            size_batch_train=size_batch_train,
            size_batch_eval=size_batch_eval,
            num_epochs=num_epochs,
            num_epochs_checkpoint=num_epochs_checkpoint,
        )
