
"""
Datasets.
"""


import torch


class Custom_Dataset(torch.utils.data.Dataset):
    
    """
    Custom dataset base class.
    
    All project datasets follow
    this template.
    """
    
    def __init__(
        self, 
        config:Dataset_Config,
        generator:Dataset_Generator,
    ):
        
        """
        Initialize.

        Parameters
        ----------
        config : Dataset_Config
            Configuration.
        
        Side Effects
        ------------
        - Create a subdirectory for the dataset
        in the directory specified by save_dir
        (if it doesn't already exist).
        """
    
        self.config = config

        self.config.sub_dir_path.mkdir(exist_ok=True)

    def load(self):
        """
        Overwrite this with a function that loads necessary files
        (at least set self.features and self.labels).
        """
        pass

    def unload(self):
        """
        Overwrite this with a function that unloads the
        loaded data from memory (self.features and self.labels).
        """
        pass

    def generate(self):
        """
        Overwrite this with a function that saves necessary files
        (at least features and labels).
        """
        pass



    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, index):
        """
        Get a (feature, label) pair.

        Parameters
        ----------
        index : int
            The index of the pair within the dataset.

        Returns
        -------
        x : torch.Tensor
            The features.
        y : torch.Tensor
            The label.
        """
        x = self.features[index]
        y = self.labels[index]
        return x, y


class Binned_Signal_Dataset(Custom_Dataset):
    
    """
    Dataset for the (binned) event-by-event approach.
    """
    
    def __init__(
        self, 
        level, 
        split, 
        save_dir, 
        q_squared_veto="tight",
        std_scale=True,
        balanced_classes=True,
        shuffle=True,
        extra_description=None,
    ):
        
        """
        Initialize.
        """
        
        super().__init__(
            "binned_signal", 
            level, 
            q_squared_veto,
            split, 
            save_dir, 
            extra_description=extra_description,
        )

        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.shuffle = shuffle

    def generate(self):

        """
        Generate the dataset.
        Create necessary files.
        """

        
        self.assert_file_does_not_exist(self.features_file_path)
        self.assert_file_does_not_exist(self.labels_file_path)
        self.assert_file_does_not_exist(self.bin_values_file_path)
        
        df_agg = self.get_agg_raw_signal_data()


    def load(self):

        """
        Load the dataset.
        Load necessary files.
        """

        self.features = torch.load(self.features_file_path, weights_only=True)
        self.labels = torch.load(self.labels_file_path, weights_only=True)
        self.bin_values = torch.load(self.bin_values_file_path, weights_only=True)
        print(f"Loaded features of shape: {self.features.shape}.")
        print(f"Loaded labels of shape: {self.labels.shape}.")
        print(f"Loaded bin values of shape: {self.bin_values.shape}.")

    def unload(self):
        
        """
        Unload dataset from memory.
        """
        
        del self.features
        del self.labels
        del self.bin_values
        print("Unloaded data.")


class Signal_Sets_Dataset(Custom_Dataset):
    """
    Torch dataset of bootstrapped sets of signal events.
    """

    def __init__(
        self, 
        level, 
        split, 
        save_dir, 
        num_events_per_set,
        num_sets_per_label,
        binned=False,
        q_squared_veto=True,
        std_scale=True,
        balanced_classes=True,
        labels_to_sample=None,
        extra_description=None,
    ):
        
        """
        Initialize.
        """

        name = (
            "signal_sets_binned" if binned
            else "signal_sets_unbinned"
        )
        self.num_events_per_set = num_events_per_set
        self.num_sets_per_label = num_sets_per_label
        self.binned = binned
        self.q_squared_veto = q_squared_veto
        self.std_scale = std_scale
        self.balanced_classes = balanced_classes
        self.labels_to_sample = labels_to_sample

        super().__init__(
            name, 
            level, 
            q_squared_veto,
            split, 
            save_dir, 
            extra_description=extra_description,
        )

    def generate(self):
        """
        Generate and save dataset files.
        """
        label_column_name = (
            self.binned_label_name if self.binned
            else self.label_name
        )

        df_agg = load_aggregated_raw_signal(
            self.level, 
            self.split, 
            self.save_dir
        )

        def convert_to_binned_df(df_agg):
            bins, bin_values = to_bins(df_agg[self.label_name])
            df_agg[self.binned_label_name] = bins
            df_agg = df_agg.drop(columns=self.label_name)
            return df_agg, bin_values
        if self.binned:
            df_agg, bin_values = convert_to_binned_df(df_agg)
            bin_values = torch.from_numpy(bin_values)

        def apply_preprocessing(df_agg):
            df_agg = df_agg.copy()
            q2_cut_strength = (
                "tight" if self.q_squared_veto
                else "loose"
            )
            df_agg = apply_q_squared_veto(df_agg, q2_cut_strength)
            if self.std_scale:
                for column_name in self.feature_names:
                    df_agg[column_name] = ( 
                        (
                            df_agg[column_name] 
                            - get_dataset_prescale("mean", self.level, self.q_squared_veto, column_name)
                        ) 
                        / get_dataset_prescale("std", self.level, self.q_squared_veto, column_name)
                    )
            if self.balanced_classes:
                df_agg = balance_classes(
                    df_agg, 
                    label_column_name=label_column_name
                )
            return df_agg
        df_agg = apply_preprocessing(df_agg)

        source_features = torch.from_numpy(
            df_agg[self.feature_names]
            .to_numpy()
        )
        source_labels = torch.from_numpy(
            df_agg[label_column_name]
            .to_numpy()
        )

        if self.binned and self.labels_to_sample:
            self.labels_to_sample = [
                torch.argwhere(bin_values == label)
                .item() 
                for label in self.labels_to_sample
            ]

        features, labels = bootstrap_labeled_sets(
            source_features,
            source_labels,
            n=self.num_events_per_set, 
            m=self.num_sets_per_label,
            reduce_labels=True,
            labels_to_sample=self.labels_to_sample,
        )

        torch.save(
            features, 
            self.features_file_path
        )
        print(
            f"Generated features of shape: {features.shape}."
        )
        torch.save(
            labels, 
            self.labels_file_path
        )
        print(
            f"Generated labels of shape: {labels.shape}."
        )
        if self.binned:
            torch.save(
                bin_values, 
                self.bin_values_file_path
            )
            print(
               f"Generated bin values of shape: {bin_values.shape}."
            )

    def load(self): 
        """
        Load saved dataset state. 
        """
        self.features = torch.load(
            self.features_file_path, 
            weights_only=True
        )
        print(
            f"Loaded features of shape: {self.features.shape}."
        )
        self.labels = torch.load(
            self.labels_file_path, 
            weights_only=True
        )
        print(
            f"Loaded labels of shape: {self.labels.shape}."
        )
        if self.binned:
            self.bin_values = torch.load(
                self.bin_values_file_path,
                weights_only=True
            )
            print(
                f"Loaded bin values of shape: {self.bin_values.shape}."
            )

    def unload(self):
        del self.features
        del self.labels
        if self.binned:
            del self.bin_values
        print("Unloaded data.")


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

