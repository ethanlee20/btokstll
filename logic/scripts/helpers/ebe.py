
class Event_By_Event_Approach:
    def __init__(
        self,
        device,
        level,
        datasets_dir,
        models_dir,
        plots_dir,
        summary_table,
        set_sizes=[70_000, 24_000, 6_000],
        new_physics_delta_c9_value=-0.82,
        num_eval_sets_per_label=50,
        num_eval_sets_per_label_sensitivity=2000,
        regenerate_datasets=False,
        balanced_classes=True,
        q_squared_veto=True,
        std_scale=True,
        retrain_model=False,
        learning_rate=3e-3,
        epochs=200,
        train_batch_size=10_000,
        eval_batch_size=10_000,
    ):
        
        self.device = device
        self.level = level
        self.datasets_dir = pathlib.Path(datasets_dir)
        self.models_dir = pathlib.Path(models_dir)
        self.plots_dir = pathlib.Path(plots_dir)
        self.summary_table = summary_table
        self.set_sizes = set_sizes
        self.new_physics_delta_c9_value = new_physics_delta_c9_value
        self.num_eval_sets_per_label = num_eval_sets_per_label
        self.num_eval_sets_per_label_sensitivity = num_eval_sets_per_label_sensitivity
        self.regenerate_datasets = regenerate_datasets
        self.balanced_classes = balanced_classes
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.retrain_model = retrain_model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_batch_size= train_batch_size
        self.eval_batch_size = eval_batch_size

        self.init_datasets()
        self.init_model()
        self.evaluate_model()


    def init_datasets(self):

        self.train_events_dataset = Binned_Signal_Dataset(
            level=self.level,
            split="train",
            save_dir=self.datasets_dir,
            q_squared_veto=self.q_squared_veto,
            std_scale=self.std_scale,
            balanced_classes=self.balanced_classes,
            shuffle=True,
            extra_description=None,
            regenerate=self.regenerate_datasets
        )

        self.eval_events_dataset = Binned_Signal_Dataset(
            level=self.level,
            split="eval",
            save_dir=self.datasets_dir,
            q_squared_veto=self.q_squared_veto,
            std_scale=self.std_scale,
            balanced_classes=self.balanced_classes,
            shuffle=True,
            extra_description=None,
            regenerate=self.regenerate_datasets
        )

        self.eval_sets_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="eval",
                save_dir=self.datasets_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_eval_sets_per_label,
                binned=True,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                extra_description=f"{num_events_per_set}",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

        self.single_label_eval_sets_datasets = {
            num_events_per_set : Signal_Sets_Dataset(
                level=self.level,
                split="eval",
                save_dir=self.datasets_dir,
                num_events_per_set=num_events_per_set,
                num_sets_per_label=self.num_eval_sets_per_label_sensitivity,
                binned=True,
                q_squared_veto=self.q_squared_veto,
                std_scale=self.std_scale,
                balanced_classes=self.balanced_classes,
                labels_to_sample=[self.new_physics_delta_c9_value],
                extra_description=f"{num_events_per_set}_single",
                regenerate=self.regenerate_datasets
            )
            for num_events_per_set in self.set_sizes
        }

    def init_model(self):
        self.model = Event_By_Event_NN(
            self.models_dir, 
            extra_description=f"{self.level}_q2v_{self.q_squared_veto}",
        )

        if self.retrain_model:
            self.train_model()

    def train_model(self):
        
        model = self.model

        train_dataset = self.train_events_dataset
        eval_dataset = self.eval_events_dataset
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=0.95, 
            threshold=0, 
            patience=0, 
            eps=1e-9
        )
        
        train_dataset.load()
        eval_dataset.load()
        
        model.retrain(
            train_dataset,
            eval_dataset,
            loss_fn,
            optimizer,
            self.epochs,
            self.train_batch_size,
            self.eval_batch_size,
            self.device,
            scheduler=scheduler,
        )
        
        _, ax = plt.subplots()
        plot_loss_curves(
            model.loss_table,
            ax,
            start_epoch=0,
            log_scale=True,
        )

        plot_file_name = f"ebe_{self.level}_q2v_{self.q_squared_veto}_loss.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()

        train_dataset.unload()
        eval_dataset.unload()

    
    def evaluate_model(self):

        model = self.model
        model.load_final()
        model.to(self.device)
        model.eval()

        for num_events_per_set in self.set_sizes:
            self.evaluate_mse_mae(num_events_per_set)
            self.evaluate_linearity(num_events_per_set)
            self.evaluate_sensitivity(num_events_per_set)

    def evaluate_mse_mae(self, num_events_per_set):
        
        eval_sets_dataset = self.eval_sets_datasets[
            num_events_per_set
        ]
        eval_sets_dataset.load()

        predictions = make_predictions(
            self.model, 
            eval_sets_dataset.features, 
            self.device, 
            event_by_event=True, 
            bin_values=eval_sets_dataset.bin_values
        )
        assert (
           predictions.shape 
           == eval_sets_dataset.labels.shape
        )

        unbinned_labels = (
            eval_sets_dataset
            .bin_values[
                eval_sets_dataset.labels.int()
            ]
        )

        (
            mse, 
            mae
        ) = calculate_mse_mae(
            predictions, 
            unbinned_labels
        )

        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Event by event", 
            "MSE", 
            num_events_per_set, 
            mse
        )

        self.summary_table.add_item(
            self.level,
            self.q_squared_veto,
            "Event by event", 
            "MAE", 
            num_events_per_set, 
            mae
        )

    def evaluate_linearity(self, num_events_per_set):

        eval_sets_dataset = self.eval_sets_datasets[
            num_events_per_set
        ]
        eval_sets_dataset.load()

        predictions = make_predictions(
            self.model, 
            eval_sets_dataset.features, 
            self.device, 
            event_by_event=True, 
            bin_values=eval_sets_dataset.bin_values
        )
        assert (
           predictions.shape 
           == eval_sets_dataset.labels.shape
        )

        unbinned_labels = (
           eval_sets_dataset
            .bin_values[
                eval_sets_dataset.labels.int()
            ]
        )

        (
            unique_labels, 
            avgs, 
            stds
        ) = run_linearity_test(
            predictions, 
            unbinned_labels,
        )

        _, ax = plt.subplots()
        
        plot_prediction_linearity(
            ax,
            unique_labels.detach().cpu().numpy(),
            avgs.detach().cpu().numpy(),
            stds.detach().cpu().numpy(),
            note=(
                f"Event-by-event, {self.level}., "
                + f"{self.num_eval_sets_per_label} boots., "
                + f"{num_events_per_set} events/boots."
                + f"$q^2$ veto: {self.q_squared_veto}"
            ),
        )

        plot_file_name = f"ebe_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_lin.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")

        plt.show()
        plt.close()
        
        eval_sets_dataset.unload()
    
    def evaluate_sensitivity(self, num_events_per_set):

        eval_sets_dataset = self.single_label_eval_sets_datasets[
            num_events_per_set
        ]
        eval_sets_dataset.load()

        predictions = make_predictions(
            self.model, 
            eval_sets_dataset.features, 
            self.device, 
            event_by_event=True, 
            bin_values=eval_sets_dataset
                .bin_values,
        )
        
        (
            mean, 
            std, 
            bias,
        ) = run_sensitivity_test(
            predictions, 
            self.new_physics_delta_c9_value
        )

        self.summary_table.add_item(self.level, self.q_squared_veto, "Event by event", "Mean at NP", num_events_per_set, mean)
        self.summary_table.add_item(self.level, self.q_squared_veto, "Event by event", "Std. at NP", num_events_per_set, std)
        self.summary_table.add_item(self.level, self.q_squared_veto, "Event by event", "Bias at NP", num_events_per_set, bias)

        _, ax = plt.subplots()

        plot_sensitivity(
            ax, 
            predictions, 
            self.new_physics_delta_c9_value, 
            note=(
                f"Event-by-event, {self.level}., " 
                + f"{self.num_eval_sets_per_label_sensitivity} boots., " 
                + f"{num_events_per_set} events/boots."
                + f"$q^2$ veto: {self.q_squared_veto}"
            ),
        )
        
        plot_file_name = f"ebe_{num_events_per_set}_{self.level}_q2v_{self.q_squared_veto}_sens.png"
        plot_file_path = self.plots_dir.joinpath(plot_file_name)
        plt.savefig(plot_file_path, bbox_inches="tight")
        
        plt.show()
        plt.close()

        eval_sets_dataset.unload()