
class Names_of_Datasets:

    images = "images"
    events_binned = "events_binned"
    sets_binned = "sets_binned"
    sets_unbinned = "sets_unbinned"
    
    tuple_ = (
        images,
        events_binned,
        sets_binned,
        sets_unbinned,
    )
    set_based = (
        images,
        sets_binned,
        sets_unbinned,
    )
    event_based = (events_binned,)


class Names_of_Levels:

    generator = "gen"
    detector = "det"
    detector_and_background = "det_bkg"

    tuple_ = (
        generator, 
        detector,
        detector_and_background,
    )


class Names_of_q_Squared_Vetos:
    
    tight = "tight"
    loose = "loose"
    
    tuple_ = (tight, loose)


class Names_of_Variables:

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
    list_ = list(tuple_)


class Names_of_Labels:

    unbinned = "dc9"
    binned = "dc9_bin_index"

    tuple_ = (unbinned, binned)
    

class Names_of_Splits:        

    train = "train"
    eval_ = "eval"

    tuple_ = (train, eval_)


class Names_of_Kinds_of_Dataset_Files:

    features = "features"
    labels = "labels"
    bin_map = "bin_map"

    tuple_ = (
        features,
        labels,
        bin_map
    )


class Raw_Signal_Trial_Ranges:
    
    train = range(1, 21)
    eval_ = range(21, 41)

    tuple_ = (train, eval_)


class Numbers_of_Events_per_Set:
    
    tuple_ = (
        70_000, 
        24_000, 
        6_000,
    )