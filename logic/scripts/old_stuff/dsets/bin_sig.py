


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





