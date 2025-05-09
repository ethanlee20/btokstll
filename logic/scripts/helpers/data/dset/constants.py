

class Names_Datasets:

    images_signal = "images_signal"
    binned_signal = "binned_signal"
    sets_binned_signal = "sets_binned_signal"
    sets_unbinned_signal = "sets_unbinned_signal"

    tuple_ = (
        images_signal,
        binned_signal,
        sets_binned_signal,
        sets_unbinned_signal,
    )


class Names_Levels:

    generator = "gen"
    detector = "det"

    tuple_ = (
        generator, 
        detector,
    )


class Names_q_Squared_Vetos:
    
    tight = "tight"
    loose = "loose"
    
    tuple_ = (
        tight, 
        loose,
    )


class Names_Variables:

    q_squared = "q_squared"
    cos_theta_mu = "costheta_mu"
    cos_k = "costheta_K"
    chi = "chi"

    tuple_ = (
        q_squared,
        cos_theta_mu,
        cos_k,
        chi,
    )


class Names_Labels:

    unbinned = "dc9"
    binned = "dc9_bin_index"

    tuple_ = (
        unbinned,
        binned,
    )
    

class Names_Splits:        

    train = "train"
    eval_ = "eval"

    tuple_ = (
        train,
        eval_,
    )


class Names_Kind_File_Tensor:

    features = "features"
    labels = "labels"
    bin_map = "bin_map"

    tuple_ = (
        features,
        labels,
        bin_map,
    )


class Trials_Splits:
    
    train = range(1, 21)
    eval_ = range(21, 41)

    tuple_ = (
        train,
        eval_,
    )


class Nums_Events_Per_Set:
    
    tuple_ = (
        70_000, 
        24_000, 
        6_000,
    )