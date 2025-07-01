
import pathlib

import matplotlib.pyplot as plt

from ..data.dset.config import Config_Dataset
from ..data.dset.dataset import Custom_Dataset
from ..data.dset.constants import Names_Datasets
from ..model.constants import Names_Models
from ..model.config import Config_Model
from ..model.model import Custom_Model
from ..model.trainer import Trainer
from ..model.eval import Evaluator
from ..plot.linearity import plot_linearity
from ..plot.sensitivity import plot_sensitivity
from ..plot.loss_curves import plot_loss_curves
from ..plot.slices_image import plot_image_slices
from ..plot.util import add_plot_note
from ..result.table import Summary_Table
from ..result.constants import Names_Kinds_Items

 

class Experiment:

    def __init__(
        self,
        path_dir_plots:str|pathlib.Path,
        device:str,
    ):
        
        self.path_dir_plots = pathlib.Path(path_dir_plots)
        
        self.device = device

        self.table_summary = Summary_Table()

    def evaluate(
        self,
        config_model:Config_Model,
        config_dset_eval:Config_Dataset,
        config_dset_eval_sens:Config_Dataset,
        generate_dsets=False,
    ):

        dset_eval = Custom_Dataset(
            config_dset_eval
        )
        dset_eval_sens = Custom_Dataset(
            config_dset_eval_sens
        )
        
        if generate_dsets:
            dset_eval_sens.generate()
            dset_eval.generate()

        dset_eval_sens.load()
        dset_eval.load()

        if config_dset_eval.name == (
            Names_Datasets().images
        ):
            value_dc9_plot = -0.82
            image = dset_eval.features[
                dset_eval.labels == value_dc9_plot
            ][0]

            plot_image_slices(
                image=image, 
                config_dset=config_dset_eval,
                value_dc9=value_dc9_plot, 
                path_dir=self.path_dir_plots
            )

        model = Custom_Model(
            config_model
        )

        model.load()

        eval = Evaluator(
            model=model,
            dataset=dset_eval,
            device=self.device,
        )

        eval_sens = Evaluator(
            model=model,
            dataset=dset_eval_sens,
            device=self.device,
        )

        eval.predict()
        eval_sens.predict()

        if model.config.name == Names_Models().ebe:

            value_dc9_sm = 0
            value_dc9_np = -0.82
            
            log_probs_sm = eval.log_probs[eval.labels==value_dc9_sm][0].cpu()
            log_probs_np = eval.log_probs[eval.labels==value_dc9_np][0].cpu()

            fig, ax = plt.subplots()

            fig.set_figwidth(5)

            def my_bar(x, y, label, color):

                xmin = list(x)
                xmax = list(x[1:])
                xmax.append(xmax[-1]+xmax[1]-xmax[0])

                ax.hlines(
                    y=y, 
                    xmin=xmin,
                    xmax=xmax,
                    label=label,
                    color=color,
                    linewidths=2,
                )

            ax.axvline(x=value_dc9_sm, color='black', label=r"$\delta C_9 ="+f"{value_dc9_sm}$", ls='--')
            ax.axvline(x=value_dc9_np, color='black', label=r"$\delta C_9 ="+f"{value_dc9_np}$", ls=':')
            
            my_bar(
                x=dset_eval.bin_map.cpu(),
                y=log_probs_sm,
                label=r"SM ($\delta C_9 ="+f"{value_dc9_sm}$)",
                color='blue',
            )

            my_bar(
                x=dset_eval.bin_map.cpu(),
                y=log_probs_np,
                label=r"NP ($\delta C_9 ="+f"{value_dc9_np}$)",
                color='red',
            )

            # ax.plot(dset_eval.bin_map.cpu(), log_probs_sm, label=r"SM ($\delta C_9 ="+f"{value_dc9_sm}$)", color="blue")
            # ax.plot(dset_eval.bin_map.cpu(), log_probs_np, label=r"NP ($\delta C_9 ="+f"{value_dc9_np}$)", color="red")


            ax.set_xlabel(r"$\delta C_9$")
            ax.set_ylabel(r"$\log p(\delta C_9 | x_1, ..., x_N)$")
            ax.legend(loc='lower right')

            note = (
                f"Level: {model.config.config_dset_train.level}"
                f"\nEvents per Set: {config_dset_eval.num_events_per_set}"    
            )
            add_plot_note(ax=ax, text=note)

            name_file = f"log_prob_{model.config.config_dset_train.level}_{config_dset_eval.num_events_per_set}.png"
            path = self.path_dir_plots.joinpath(name_file)
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        # labels, avgs, stds = eval.run_test_lin()

        # plot_linearity(
        #     labels=labels, 
        #     avgs=avgs, 
        #     stds=stds,
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval,
        #     path_dir=self.path_dir_plots,
        # )
        
        # mse, mae = eval.calc_mse_mae()
        
        # self.table_summary.add_item(
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval,
        #     kind=Names_Kinds_Items().mse,
        #     item=mse,
        # )
  
        # self.table_summary.add_item(
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval,
        #     kind=Names_Kinds_Items().mae,
        #     item=mae,
        # )

        # avg_sens, std_sens, bias_sens, label_sens = (
        #     eval_sens.run_test_sens()
        # )

        # self.table_summary.add_item(
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval_sens,
        #     kind=Names_Kinds_Items().np_mean,
        #     item=avg_sens,
        # )

        # self.table_summary.add_item(
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval_sens,
        #     kind=Names_Kinds_Items().np_std,
        #     item=std_sens,
        # )

        # self.table_summary.add_item(
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval_sens,
        #     kind=Names_Kinds_Items().np_bias,
        #     item=bias_sens,
        # )

        # plot_sensitivity(
        #     preds=eval_sens.preds,
        #     avg=avg_sens,
        #     std=std_sens,
        #     label=label_sens,
        #     config_model=config_model,
        #     config_dset_eval=config_dset_eval_sens,
        #     path_dir=self.path_dir_plots
        # )

    def train(
        self,
        config_model:Config_Model,
        config_dset_eval:Config_Dataset,
        generate_dsets=False,
    ):

        dset_train = Custom_Dataset(
            config_model.config_dset_train
        )

        dset_eval = Custom_Dataset(
            config_dset_eval
        )

        if generate_dsets:

            dset_train.generate()
            dset_eval.generate()

        dset_train.load()
        dset_eval.load()

        model = Custom_Model(
            config_model
        )

        trainer = Trainer(
            model=model,
            dset_train=dset_train,
            dset_eval=dset_eval,
            device=self.device,
        )

        trainer.train()

        plot_loss_curves(
            config_model=config_model,
            config_dset_eval=config_dset_eval,
            path_dir=self.path_dir_plots,
        )

        dset_train.unload()
        dset_eval.unload()










 

# class Approach_Deep_Sets:

#     def __init__(
#         self,
#         name_model,
#         level,
#         q_squared_veto,
#         balanced_classes,
#         std_scale,
#         shuffle,
#         label_subset,
#         num_events_per_set,
#         num_sets_per_label,
#         num_bins_image,
#         loss_fn,
#         optimizer,
#         lr_scheduler,
#         size_batch_train,
#         size_batch_eval,
#         num_epochs,
#         num_epochs_checkpoint,
#         path_dir_dsets_main,
#         path_dir_raw_signal,
#         path_dir_models_main,
#         device,
#         extra_description=None,
#     ):


# class Deep_Sets_Approach:
#     def __init__(
#         self,
#         device,
#         level,
#         datasets_dir,
#         models_dir,
#         plots_dir,
#         summary_table,
#         set_sizes=[70_000, 24_000, 6_000],
#         new_physics_delta_c9_value=-0.82,
#         num_sets_per_label=50,
#         num_sets_per_label_sensitivity=2000,
#         regenerate_datasets=False,
#         balanced_classes=True,
#         q_squared_veto=True,
#         std_scale=True,
#         retrain_models=False,
#         learning_rate=4e-4,
#         epochs=80,
#         train_batch_size=32,
#         eval_batch_size=32,
#     ):
#         self.device = device
#         self.level = level
#         self.models_dir = pathlib.Path(models_dir)
#         self.dataset_dir = pathlib.Path(datasets_dir)
#         self.plots_dir = pathlib.Path(plots_dir)
#         self.summary_table = summary_table
#         self.set_sizes = set_sizes
#         self.new_physics_delta_c9_value = new_physics_delta_c9_value
#         self.num_sets_per_label = num_sets_per_label
#         self.num_sets_per_label_sensitivity = num_sets_per_label_sensitivity
#         self.regenerate_datasets = regenerate_datasets
#         self.balanced_classes = balanced_classes
#         self.q_squared_veto = q_squared_veto
#         self.std_scale = std_scale
#         self.retrain_models = retrain_models
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.train_batch_size = train_batch_size
#         self.eval_batch_size = eval_batch_size

#         self.init_datasets()
#         self.init_models()
#         self.evaluate_models()

#     def init_datasets(self):

#         self.train_datasets = {
#             num_events_per_set : Signal_Sets_Dataset(
#                 level=self.level,
#                 split="train",
#                 save_dir=self.dataset_dir,
#                 num_events_per_set=num_events_per_set,
#                 num_sets_per_label=self.num_sets_per_label,
#                 binned=False,
#                 q_squared_veto=self.q_squared_veto,
#                 std_scale=self.std_scale,
#                 balanced_classes=self.balanced_classes,
#                 extra_description=f"{num_events_per_set}",
#                 regenerate=self.regenerate_datasets
#             )
#             for num_events_per_set in self.set_sizes
#         }

#         self.eval_datasets = {
#             num_events_per_set : Signal_Sets_Dataset(
#                 level=self.level,
#                 split="eval",
#                 save_dir=self.dataset_dir,
#                 num_events_per_set=num_events_per_set,
#                 num_sets_per_label=self.num_sets_per_label,
#                 binned=False,
#                 q_squared_veto=self.q_squared_veto,
#                 std_scale=self.std_scale,
#                 balanced_classes=self.balanced_classes,
#                 extra_description=f"{num_events_per_set}",
#                 regenerate=self.regenerate_datasets
#             )
#             for num_events_per_set in self.set_sizes
#         }

#         self.single_label_eval_datasets = {
#             num_events_per_set : Signal_Sets_Dataset(
#                 level=self.level,
#                 split="eval",
#                 save_dir=self.dataset_dir,
#                 num_events_per_set=num_events_per_set,
#                 num_sets_per_label=self.num_sets_per_label_sensitivity,
#                 binned=False,
#                 q_squared_veto=self.q_squared_veto,
#                 std_scale=self.std_scale,
#                 balanced_classes=self.balanced_classes,
#                 labels_to_sample=[self.new_physics_delta_c9_value],
#                 extra_description=f"{num_events_per_set}_single",
#                 regenerate=self.regenerate_datasets
#             )
#             for num_events_per_set in self.set_sizes
#         }

#     def init_models(self):

#         self.models = {
#             num_events_per_set : Deep_Sets(
#                 self.models_dir, 
#                 extra_description=f"{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}"
#             )
#             for num_events_per_set in self.set_sizes
#         }

#         if self.retrain_models:
#             self.train_models()

#     def train_models(self):

#         for num_events_per_set in self.set_sizes:

#             self.train_model(num_events_per_set)

#     def train_model(self, num_events_per_set):

#         model = self.models[
#             num_events_per_set
#         ]

#         loss_fn = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(
#             model.parameters(), 
#             lr=self.learning_rate
#         )

#         train_dataset = self.train_datasets[
#             num_events_per_set
#         ]
#         eval_dataset = self.eval_datasets[
#             num_events_per_set
#         ]
#         train_dataset.load()
#         eval_dataset.load()

#         model.retrain( 
#             train_dataset, 
#             eval_dataset, 
#             loss_fn, 
#             optimizer, 
#             self.epochs, 
#             self.train_batch_size, 
#             self.eval_batch_size, 
#             self.device, 
#             move_data=True,
#             scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, 
#                 factor=0.9, 
#                 patience=1
#             ),
#             checkpoint_epochs=5,
#         )

#         _, ax = plt.subplots()
#         plot_loss_curves(
#             model.loss_table,
#             ax,
#             start_epoch=0,
#             log_scale=True,
#         )

#         plot_file_name = f"deepsets_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_loss.png"
#         plot_file_path = self.plots_dir.joinpath(plot_file_name)
#         plt.savefig(plot_file_path, bbox_inches="tight")

#         plt.show()
#         plt.close()

#         train_dataset.unload()
#         eval_dataset.unload()
    
#     def evaluate_models(self,):

#         for num_events_per_set in self.set_sizes:

#             self.evaluate_model(num_events_per_set)

#     def evaluate_model(self, num_events_per_set):
        
#         model = self.models[
#             num_events_per_set
#         ]
#         model.load_final()
#         model.to(self.device)
#         model.eval()

#         self.evaluate_mse_mae(model, num_events_per_set)
#         self.evaluate_linearity(model, num_events_per_set)
#         self.evaluate_sensitivity(model, num_events_per_set)

#     def evaluate_mse_mae(self, model, num_events_per_set):

#         eval_dataset = self.eval_datasets[
#             num_events_per_set
#         ]
#         eval_dataset.load()

#         predictions = make_predictions(
#             model, 
#             eval_dataset.features,
#             self.device,
#         )

#         mse, mae = calculate_mse_mae(
#             predictions, 
#             eval_dataset.labels,
#         )
#         self.summary_table.add_item(
#             self.level,
#             self.q_squared_veto,
#             "Deep Sets", 
#             "MSE", 
#             num_events_per_set, 
#             mse,
#         )
#         self.summary_table.add_item(
#             self.level,
#             self.q_squared_veto,
#             "Deep Sets", 
#             "MAE", 
#             num_events_per_set, 
#             mae,
#         )
    
#     def evaluate_linearity(self, model, num_events_per_set,):

#         eval_dataset = self.eval_datasets[
#             num_events_per_set
#         ]
#         eval_dataset.load()

#         predictions = make_predictions(
#             model, 
#             eval_dataset.features,
#             self.device,
#         )

#         (
#             unique_labels, 
#             avgs, 
#             stds,
#         ) = run_linearity_test(
#             predictions, 
#             eval_dataset.labels
#         )

#         _, ax = plt.subplots()
#         plot_prediction_linearity(
#             ax,
#             unique_labels.detach().cpu().numpy(),
#             avgs.detach().cpu().numpy(),
#             stds.detach().cpu().numpy(),
#             note=(
#                 f"Deep Sets, {self.level}., "
#                 + f"{self.num_sets_per_label} boots., "
#                 + f"{num_events_per_set} events/boots."
#                 + f"$q^2$ veto: {self.q_squared_veto}"
#             ),
#         )

#         plot_file_name = f"deepsets_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_lin.png"
#         plot_file_path = self.plots_dir.joinpath(plot_file_name)
#         plt.savefig(plot_file_path, bbox_inches="tight")

#         plt.show()
#         plt.close()

#         eval_dataset.unload()

#     def evaluate_sensitivity(self, model, num_events_per_set,):

#         single_label_eval_dataset = self.single_label_eval_datasets[
#             num_events_per_set
#         ]
#         single_label_eval_dataset.load()

#         single_label_predictions = make_predictions(
#             model, 
#             single_label_eval_dataset.features,
#             self.device,
#         )

#         mean, std, bias = run_sensitivity_test(
#             single_label_predictions, 
#             self.new_physics_delta_c9_value,
#         )
#         self.summary_table.add_item(
#             self.level,
#             self.q_squared_veto,
#             "Deep Sets", 
#             "Mean at NP", 
#             num_events_per_set, 
#             mean,
#         )
#         self.summary_table.add_item(
#             self.level,
#             self.q_squared_veto,
#             "Deep Sets", 
#             "Std. at NP", 
#             num_events_per_set, 
#             std
#         )
#         self.summary_table.add_item(
#             self.level,
#             self.q_squared_veto,
#             "Deep Sets", 
#             "Bias at NP", 
#             num_events_per_set, 
#             bias
#         )

#         single_label_eval_dataset.unload()

#         _, ax = plt.subplots()

#         plot_sensitivity(
#             ax,
#             single_label_predictions,
#             self.new_physics_delta_c9_value,
#             note=(
#                 f"Deep Sets, {self.level}., " 
#                 + f"{self.num_sets_per_label_sensitivity} boots., " 
#                 + f"{num_events_per_set} events/boots."
#                 + f"$q^2$ veto: {self.q_squared_veto}"
#             ), 
#         )

#         plot_file_name = f"deepsets_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_sens.png"
#         plot_file_path = self.plots_dir.joinpath(plot_file_name)
#         plt.savefig(plot_file_path, bbox_inches="tight")

#         plt.show()
#         plt.close()
