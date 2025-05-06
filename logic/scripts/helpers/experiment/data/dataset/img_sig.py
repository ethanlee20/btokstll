

class Signal_Images_Dataset(Custom_Dataset):
    """Bootstrapped images (like Shawn)."""
    def __init__(
            self, 
            level, 
            split, 
            save_dir, 
            num_events_per_set,
            num_sets_per_label,
            n_bins,
            q_squared_veto=True,
            std_scale=True,
            balanced_classes=True,
            labels_to_sample=None,
            extra_description=None,
            regenerate=False,
    ):
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.n_bins = n_bins
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.labels_to_sample = labels_to_sample
        
        super().__init__(
            "signal_images", 
            level, 
            q_squared_veto,
            split, 
            save_dir, 
            extra_description=extra_description,
            regenerate=regenerate,
        )

    def generate(self):

        df_agg = load_aggregated_raw_signal(self.level, self.split, self.save_dir)
        
        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            q2_cut_strength = (
                "tight" if self.q_squared_veto
                else "loose"
            )
            df_agg = apply_q_squared_veto(df_agg, q2_cut_strength)
            if self.std_scale:
                df_agg["q_squared"] = (
                    (
                        df_agg["q_squared"] 
                        - get_dataset_prescale("mean", self.level, self.q_squared_veto, "q_squared")
                    )
                    / get_dataset_prescale("std", self.level, self.q_squared_veto, "q_squared")
                )
            if self.balanced_classes:
                df_agg = balance_classes(df_agg, label_column_name=self.label_name)
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        source_features = torch.from_numpy(df_agg[self.feature_names].to_numpy())
        source_labels = torch.from_numpy(df_agg[self.label_name].to_numpy())
        set_features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=self.num_events_per_set, m=self.num_sets_per_label,
            reduce_labels=True,
            labels_to_sample=self.labels_to_sample
        )
        features = torch.cat(
            [
                make_image(set_feat, n_bins=self.n_bins).unsqueeze(0) 
                for set_feat in set_features.numpy()
            ]
        )
        
        torch.save(features, self.features_file_path)
        torch.save(labels, self.labels_file_path)
        print(f"Generated features of shape: {features.shape}.")
        print(f"Generated labels of shape: {labels.shape}.")

    def load(self): 
        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        print(f"Loaded features of shape: {self.features.shape}.")
        print(f"Loaded labels of shape: {self.labels.shape}.")

    def unload(self):
        del self.features
        del self.labels
        print("Unloaded data.")