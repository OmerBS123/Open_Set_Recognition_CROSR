import torch
import scipy.stats as stats

tail_size = 1000


def calculate_class_means(logits_latents_cat, labels, logits):
    """
    Calculate the mean vectors for each class based on correctly classified logits and latent representations.

    Parameters:
    - logits_latents_cat: A tensor that combines logits and latent representations, shape (num_samples, combined_dim)
    - labels: A tensor of true labels, shape (num_samples,)
    - classified: A tensor indicating whether each sample was correctly classified, shape (num_samples,)

    Returns:
    - class_means: Dictionary with class ids as keys and mean vectors as values.
    """
    num_classes = logits.size(1)

    # Initialize a dictionary to accumulate lists of logits_latents for each class
    class_features = {i: [] for i in range(num_classes)}

    # Append correct classified vectors to the respective class list
    for i in range(logits_latents_cat.size(0)):
        label = labels[i].item()
        if label == torch.argmax(logits[i]).item():
            class_features[label].append(logits_latents_cat[i].unsqueeze(0))

    # Calculate means for each class using torch.mean
    class_means = {}
    for class_id, features in class_features.items():
        if features:  # Check if the list is not empty
            class_means[class_id] = torch.mean(torch.cat(features, dim=0), dim=0)
        else:
            class_means[class_id] = torch.zeros(logits_latents_cat.size(1), device=logits_latents_cat.device)

    return class_means


def fit_weibull_tail(logits_latents_cat, labels, logits, class_means):
    """
    Fit Weibull distribution to the distances of correctly classified features from the class means using scipy.stats.

    Parameters:
    - logits_latents_cat: A tensor that combines logits and latent representations, shape (num_samples, combined_dim)
    - labels: A tensor of true labels, shape (num_samples,)
    - logits: A tensor of logits from the model, shape (num_samples, num_classes)
    - class_means: A dictionary with class ids as keys and mean vectors as values.
    - tail_size: The size of the tail to fit the Weibull distribution to.

    Returns:
    - weibull_params: A dictionary with class ids as keys, containing dictionaries with the Weibull distribution parameters and class mean.
    """
    num_classes = logits.size(1)
    global tail_size

    print(f'Fitting Weibull to distances with tail size: {tail_size}')

    # Initialize a dictionary to accumulate distances for each class
    distances = {i: [] for i in range(num_classes)}

    # Calculate distances from class means for correctly classified samples
    for i in range(logits_latents_cat.size(0)):
        label = labels[i].item()
        if label == torch.argmax(logits[i]).item():  # Correctly classified samples
            distance = torch.norm(logits_latents_cat[i] - class_means[label])
            distances[label].append(distance.item())

    # Fit Weibull distribution for each class and include class means
    weibull_params = {}

    for i in range(num_classes):
        if distances[i]:  # Fit only if there are correctly classified samples
            # Sort distances
            distances_sorted = sorted(distances[i])

            # Ensure enough elements for the tail size
            if len(distances_sorted) >= tail_size:
                tailtofit = distances_sorted[-tail_size:]
            else:
                tailtofit = distances_sorted  # Use all elements if fewer than tailsize

            # Fit Weibull distribution using scipy.stats
            c, loc, scale = stats.weibull_min.fit(tailtofit, floc=0)

            weibull_params[i] = {
                "shape": c,
                "location": loc,
                "scale": scale,
                "mean": class_means[i]
            }
        else:
            # If no correctly classified samples, use default Weibull parameters
            weibull_params[i] = {
                "shape": 1.0,  # Default shape parameter for Weibull
                "location": 0.0,
                "scale": 1.0,
                "mean": class_means[i]
            }

    return weibull_params


def compute_weibull_disterbution(logits_latents_cat_acc, labels_acc, logits_acc):
    class_means_logits_latents = calculate_class_means(logits_latents_cat_acc, labels_acc, logits_acc)
    return fit_weibull_tail(logits_latents_cat_acc, labels_acc, logits_acc, class_means_logits_latents)


def calculate_openmax(logits_latents_cat_list, logits, weibull_params):
    num_samples = logits.size(0)
    num_classes = logits.size(1)

    # Squeeze and concatenate the logits and latent representations
    logits_latents_cat_list_squeezed = [logits_latents_cat_list[0]] + [t.squeeze(dim=-1).squeeze(dim=-1) for t in logits_latents_cat_list[1:]]
    logits_latents_cat = torch.cat(logits_latents_cat_list_squeezed, dim=1)

    # Initialize tensor to store OpenMax scores
    openmax_scores_batch = torch.zeros(num_samples, num_classes + 1)

    for idx in range(num_samples):
        y_logit = logits[idx]
        z_latent = logits_latents_cat[idx]

        # Check for NaNs or Infs in z_latent
        if torch.isnan(z_latent).any() or torch.isinf(z_latent).any():
            raise Exception(f"NaN or Inf in z_latent at index {idx}")

        # Initialize variables for unknown class score accumulation
        unknown_class_score = 0

        # Calculate OpenMax scores for each class
        openmax_scores = torch.zeros(num_classes + 1)
        for i in range(num_classes):
            class_mean = weibull_params[i]['mean']

            # Check for NaNs or Infs in class_mean
            if torch.isnan(class_mean).any() or torch.isinf(class_mean).any():
                raise Exception(f"NaN or Inf in class_mean for class {i}")

            # Calculate distances from the class mean
            distance = torch.norm(z_latent - class_mean)

            if torch.isnan(distance).any() or distance < 0:
                raise Exception(f"Invalid distance at index {idx}, class {i}")

            # Query the Weibull CDF for the class
            shape = weibull_params[i]['shape']
            location = weibull_params[i]['location']
            scale = weibull_params[i]['scale']

            # Calculate the class belongingness using the Weibull cumulative distribution function
            class_belongingness = 1 - stats.weibull_min.cdf(distance.item(), shape, loc=location, scale=scale)

            # Calculate original softmax score
            softmax_score = torch.softmax(y_logit, dim=0)[i]

            if torch.isnan(softmax_score).any() or torch.isinf(softmax_score).any():
                raise Exception(f"NaN or Inf in softmax_score for class {i}")

            # Update OpenMax score
            openmax_scores[i] = softmax_score * class_belongingness
            unknown_class_score += softmax_score * (1 - class_belongingness)

        # Add the "unknown" class score
        openmax_scores[num_classes] = unknown_class_score

        # Store the OpenMax scores for the current sample
        openmax_scores_batch[idx] = openmax_scores

    return openmax_scores_batch


def calculate_openmax_threshold(logits_latents_cat_list, logits, weibull_params, threshold):
    num_samples = logits.size(0)
    num_classes = logits.size(1)

    logits_latents_cat_list_squeezed = [logits_latents_cat_list[0]] + [t.squeeze(dim=-1).squeeze(dim=-1) for t in logits_latents_cat_list[1:]]

    # Concatenate along the feature dimension (dim=1)
    logits_latents_cat = torch.cat(logits_latents_cat_list_squeezed, dim=1)

    # Initialize tensor to store OpenMax scores
    openmax_scores_batch = torch.zeros(num_samples, num_classes + 1)

    for idx in range(num_samples):
        y_logit = logits[idx]
        z_latent = logits_latents_cat[idx]

        # Check for NaNs or Infs in z_latent
        if torch.isnan(z_latent).any() or torch.isinf(z_latent).any():
            raise Exception(f"NaN or Inf in z_latent at index {idx}")

        # Initialize variables for unknown class score accumulation
        unknown_class_score = 0

        # Calculate OpenMax scores for each class
        openmax_scores = torch.zeros(num_classes + 1)
        for i in range(num_classes):
            class_mean = weibull_params[i]['mean']

            # Check for NaNs or Infs in class_mean
            if torch.isnan(class_mean).any() or torch.isinf(class_mean).any():
                raise Exception(f"NaN or Inf in class_mean for class {i}")

            # Calculate distances from the class mean
            distance = torch.norm(z_latent - class_mean)

            if torch.isnan(distance).any() or distance < 0:
                raise Exception(f"Invalid distance at index {idx}, class {i}")

            # # Calculate class-belongingness probability using Weibull CDF
            # shape = weibull_params[i]['shape']
            # location = weibull_params[i]['location']
            # scale = weibull_params[i]['scale']

            # Query Weibull model for the class
            mr = weibull_params[i]['weibull_model']
            class_belongingness = mr.w_score(distance)

            #
            # if shape <= 0 or scale <= 0 or torch.isnan(torch.tensor([shape, location, scale])).any():
            #     raise Exception(f"Invalid Weibull parameters for class {i}")

            # class_belongingness = weibull_min.cdf(distance.item(), shape, location, scale)

            # if torch.isnan(torch.tensor([class_belongingness])).any():
            #     raise Exception(f"NaN in class_belongingness for class {i}")

            # Calculate original softmax score
            softmax_score = torch.softmax(y_logit, dim=0)[i]

            if torch.isnan(softmax_score).any() or torch.isinf(softmax_score).any():
                raise Exception(f"NaN or Inf in softmax_score for class {i}")

            # Update OpenMax score
            openmax_scores[i] = softmax_score * class_belongingness
            unknown_class_score += softmax_score * (1 - class_belongingness)

        # Apply threshold for "unknown" class
        if unknown_class_score > threshold:
            openmax_scores_batch[idx, :] = 0  # Set all known class scores to 0
            openmax_scores_batch[idx, num_classes] = 1  # Set the "unknown" class score to 1
        else:
            openmax_scores_batch[idx] = openmax_scores
            openmax_scores_batch[idx, -1] = 0

        # Add the "unknown" class score
        # openmax_scores[num_classes] = unknown_class_score
        #
        # # Store the OpenMax scores for the current sample
        # openmax_scores_batch[idx] = openmax_scores

    return openmax_scores_batch


def set_tail_size(tailsize):
    global tail_size
    print(f'Setting tail size to :{tailsize}')
    tail_size = tailsize
