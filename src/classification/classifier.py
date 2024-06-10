import sys
import os
# add the path of the root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from src.hookedLLM import HookedLLM
import numpy as np
from tqdm import tqdm
from typing import List, Union
from src.classification.mlp import MLPClassifier

class ActivationUsage:
    @classmethod
    def from_config(cls, config, model):
        type_mapping = {
            'InnerProductFeatureExtractor': InnerProductFeatureExtractor,
            'ThresholdedInnerProductClassifierNN': ThresholdedInnerProductClassifierNN,
            'AutoFeatureExtractor': AutoFeatureExtractor,
            'NaiveExtractClassifierNN': NaiveExtractClassifierNN, 
            'InPlaceProductFeatureExtractor': InPlaceProductFeatureExtractor,
            'InPlaceProductClassifierNN': InPlaceProductClassifierNN
            # Add other types here as needed
        }
        classifier_type = config.get('type')
        if classifier_type not in type_mapping:
            raise ValueError(f"Invalid classifier type: {classifier_type}")
        return type_mapping[classifier_type](**config, model=model)
    
    def calibrate_filter_threshold(self, prompt="Hello. The following is a message."):
        """
        Filtering things like bos/eos in the middle layers because they are so distinct. They almost only have meanings related to sentence positions while having nothing to do with meanings.
        """
        # Get the number of layers
        num_layers = self.model.get_num_layers()

        # Get the middle layer index
        middle_layer_index = num_layers // 2

        # Get the token embeddings for the prompt from the middle layer
        prompt_layer_inputs = self.model.get_layer_input([middle_layer_index], prompt)
        token_embeddings = prompt_layer_inputs[middle_layer_index][0]

        # Calculate the norm of each token embedding
        norms = torch.norm(token_embeddings, dim=1)

        # Sort the norms
        sorted_norms, _ = torch.sort(norms)

        threshold_token = None

        # Find the token where its norm is greater than 10 times the norm of all previous tokens
        for i in range(1, sorted_norms.shape[0]):
            if sorted_norms[i] > 10 * sorted_norms[i - 1]:
                threshold_token = i
                break

        # Calculate the threshold as 0.7 times the norm of the threshold token
        threshold = 0.2 * sorted_norms[threshold_token].item() if threshold_token is not None else float('inf')

        self.filter_threshold = threshold

        print("calib threshold: ", threshold)

        return threshold
    
    def filter_embeddings_by_norm(self, embeddings: torch.Tensor, threshold = None) -> torch.Tensor:
        """
        The input tensor should have the embedding dimension at the last dimension.
        Nullify tokens with too big norms, typically sentence breaks.
        """
        if threshold is None:
            threshold = self.filter_threshold
        # Calculate the norm of each token embedding
        norms = torch.norm(embeddings, dim=-1)

        # Create a boolean mask for tokens with norm greater than the threshold
        mask = norms > threshold

        # Use the mask to filter the embeddings
        filtered_embeddings = torch.where(mask[..., None], 0, embeddings)

        return filtered_embeddings
    
class InnerProductFeatureExtractor(ActivationUsage):
    def __init__(self, concept: Union[str, list], layer_indices: list, model: HookedLLM, **kwargs):
        self.concept = [concept] if isinstance(concept, str) else concept
        self.model = model
        self.layer_indices = layer_indices if layer_indices is not None else list(range(model.get_num_layers()))

        # calibration1
        self.calibrate_filter_threshold()

        # calculate usage
        self.mean_tokens = self.calculate_mean_tokens()

    def __desired_layer_as_tensor(self, prompt: str):
        """
        output shape: (n_desired_layer, n_tokens, n_embedding_dim)
        This function will preprocess a prompt:
        1. get it's token representation at all layers
        2. nullify the tokens with too big norms
        3. concatenate the token representations at the desired layers
        """
        layer_inputs = self.model.get_layer_input(self.layer_indices, prompt)
        # convert to pure list
        layer_inputs = [layer_inputs[i] for i in self.layer_indices]
        device = layer_inputs[0].device
        return self.filter_embeddings_by_norm(torch.cat([tensor.to(device) for tensor in layer_inputs], dim=0))

    def calculate_mean_tokens(self):
        """
        output shape: (n_concepts, n_desired_layer, n_embedding_dim)
        """
        concept_mean_tokens = []
        for concept in self.concept:
            layer_inputs = self.__desired_layer_as_tensor(concept)
            concept_mean_tokens.append(torch.mean(layer_inputs, dim=-2)[None, ...])
        return torch.cat(concept_mean_tokens, dim=0)

    def get_prompt_feature(self, prompt: str) -> torch.Tensor:
        """
        output: (n_desired_layer,), max inner product with concept mean token at each layer
        """
        if not hasattr(self, "mean_tokens"):
            raise Exception("Concept mean token not calculated yet.")
        layer_inputs = self.__desired_layer_as_tensor(prompt)
        inner_prods = torch.einsum("ltd,cld->clt", layer_inputs, self.mean_tokens)
        return torch.max(inner_prods, dim=-1)[0].view(-1)
    
class InPlaceProductFeatureExtractor(ActivationUsage):
    def __init__(self, concept: Union[str, list], layer_indices: list, model: HookedLLM, **kwargs):
        self.concept = [concept] if isinstance(concept, str) else concept
        self.model = model
        self.layer_indices = layer_indices if layer_indices is not None else list(range(model.get_num_layers()))

        # calculate usage
        self.concept_tokens = self.calculate_concept_tokens()

    def __desired_layer_as_tensor(self, prompt: str):
        """
        output shape: (n_desired_layer, n_tokens, n_embedding_dim)
        This function will preprocess a prompt:
        1. get it's token representation at all layers
        2. nullify the tokens with too big norms
        3. concatenate the token representations at the desired layers
        """
        layer_inputs = self.model.get_layer_input(self.layer_indices, prompt)
        # convert to pure list
        layer_inputs = [layer_inputs[i] for i in self.layer_indices]
        device = layer_inputs[0].device
        return torch.cat([tensor.to(device) for tensor in layer_inputs], dim=0)

    def calculate_concept_tokens(self):
        """
        output shape: (n_concepts, n_desired_layer, n_embedding_dim)
        """
        concept_last_tokens = []
        for concept in self.concept:
            layer_inputs = self.__desired_layer_as_tensor(concept)
            concept_last_tokens.append(layer_inputs[None, :, -1, :])
        return torch.cat(concept_last_tokens, dim=0)

    def get_prompt_feature(self, prompt: str) -> torch.Tensor:
        """
        output: (n_concept, n_desired_layer * n_embedding_dim,), max inner product with concept mean token at each layer
        """
        desired_layers = self.__desired_layer_as_tensor(prompt)
        last_tokens = desired_layers[:, -1, :] # shape: (n_layers, n_embedding_dim)
        # in_place mul and flatten to get the feature
        in_place_muls = last_tokens[None, ...] * self.concept_tokens
        return in_place_muls.reshape(-1)
    
class AutoFeatureExtractor(ActivationUsage):
    def __init__(self, layer_indices: list, model: HookedLLM, extract_type: str, **kwargs):
        self.extract_type = extract_type
        self.model = model
        self.layer_indices = layer_indices if layer_indices is not None else list(range(model.get_num_layers()))
        
    def __desired_layer_as_tensor(self, prompt: str):
        """
        output shape: (n_desired_layer, n_tokens, n_embedding_dim)
        This function will preprocess a prompt:
        1. get it's token representation at all layers
        2. nullify the tokens with too big norms
        3. concatenate the token representations at the desired layers
        """
        layer_inputs = self.model.get_layer_input(self.layer_indices, prompt)
        # convert to pure list
        layer_inputs = [layer_inputs[i] for i in self.layer_indices]
        device = layer_inputs[0].device
        return torch.cat([tensor.to(device) for tensor in layer_inputs], dim=0)

    def get_prompt_feature(self, prompt: str) -> torch.Tensor:
        """
        output: (n_desired_layer * n_embedding_dim,), mean of all tokens at each layer
        """
        layer_inputs = self.__desired_layer_as_tensor(prompt)
        if self.extract_type == 'mean':
            return torch.mean(layer_inputs, dim=-2).view(-1)
        elif self.extract_type == 'last':
            return layer_inputs[:, -1, :].reshape(-1)

class ThresholdedInnerProductClassifierNN(InnerProductFeatureExtractor):
    def __init__(self, concept: str, layer_indices: list, model: HookedLLM, 
                 positive_samples: list, negative_samples:list, NNCfg: dict, **kwargs):
        super().__init__(concept, layer_indices, model)
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        self.negative_features = self.all_sample_features(negative_samples)
        self.positive_features = self.all_sample_features(positive_samples)

        # train MLP
        self.train_features = torch.cat([self.positive_features, self.negative_features], dim=0)
        self.train_labels = torch.cat([torch.ones(len(self.positive_features)), torch.zeros(len(self.negative_features))], dim=0)
        self.feature_classifier = MLPClassifier(NNCfg, self.train_features, self.train_labels)

        self.classify_threshold = 0.5

        # if we are asked to save the train features
        if 'export_path' in kwargs:
            export_path = kwargs['export_path']
            # create path
            os.makedirs(export_path, exist_ok=True)
            # combine the features of postivive samples and negative samples and create a label array
            all_features = torch.cat([self.positive_features, self.negative_features], dim=0)
            all_labels = torch.cat([torch.ones(len(self.positive_features)), torch.zeros(len(self.negative_features))], dim=0)
            # save the features and labels as numpy arrays
            np.save(f'{export_path}/features.npy', all_features.detach().cpu().numpy())
            np.save(f'{export_path}/labels.npy', all_labels.detach().cpu().numpy())

    def all_sample_features(self, samples):
        """
        output: (n_desired_layer,), (n_sample, n_desired_layer) the mean feature of a bunch of samples
        """
        all_sample_features = []

        for sample in tqdm(samples, desc="Processing samples"):
            all_sample_features.append(self.get_prompt_feature(sample)[None, ...])

        all_sample_features = torch.cat(all_sample_features, dim=0)
        return all_sample_features
    
    def score(self, prompt: str):
        features = self.get_prompt_feature(prompt)
        return self.feature_classifier.score(features[None, ...]).item()
    
    def classify_prompt(self, prompt: str):
        """Classify a prompt as positive or negative based on the threshold."""
        # Calculate the score for the prompt
        score = self.score(prompt)

        # Classify the prompt based on the score and threshold
        if score > self.classify_threshold:
            return 'positive'
        else:
            return 'negative'

    def classify_score(self, score: float):
        """Classify a prompt as positive or negative based on the threshold."""
        # Calculate the score for the prompt

        # Classify the prompt based on the score and threshold
        if score > self.classify_threshold:
            return 'malicious'
        else:
            return 'benign'
        
    def batch_score_and_classify(self, prompts: List[str]):
        """
        output: (n_prompt,), (n_prompt,) the score and classification of a bunch of prompts
        """
        eval_batch_size = self.NNcfg['training']['batch_size']
        scores = []
        classifications = []
        for i in tqdm(range(0, len(prompts), eval_batch_size), desc="Processing prompts"):
            batch_prompts = prompts[i:i+eval_batch_size]
            batch_features = torch.cat([self.get_prompt_feature(prompt)[None, ...] for prompt in batch_prompts], dim=0)
            batch_scores = self.feature_classifier.score(batch_features).detach().cpu().numpy().tolist()
            batch_classifications = [self.classify_score(score)for score in batch_scores]
            scores.extend(batch_scores)
            classifications.extend(batch_classifications)
        return scores, classifications

class InPlaceProductClassifierNN(InPlaceProductFeatureExtractor):
    def __init__(self, concept: str, layer_indices: list, model: HookedLLM, 
                 positive_samples: list, negative_samples:list, NNCfg: dict, **kwargs):
        super().__init__(concept, layer_indices, model)
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        self.negative_features = self.all_sample_features(negative_samples)
        self.positive_features = self.all_sample_features(positive_samples)

        # train MLP
        self.NNcfg = NNCfg
        self.train_features = torch.cat([self.positive_features, self.negative_features], dim=0)
        self.train_labels = torch.cat([torch.ones(len(self.positive_features)), torch.zeros(len(self.negative_features))], dim=0)
        self.feature_classifier = MLPClassifier(NNCfg, self.train_features, self.train_labels)

        self.classify_threshold = 0.5

        # if we are asked to save the train features
        if 'export_path' in kwargs:
            export_path = kwargs['export_path']
            # create path
            os.makedirs(export_path, exist_ok=True)
            # combine the features of postivive samples and negative samples and create a label array
            all_features = torch.cat([self.positive_features, self.negative_features], dim=0)
            all_labels = torch.cat([torch.ones(len(self.positive_features)), torch.zeros(len(self.negative_features))], dim=0)
            # save the features and labels as numpy arrays
            np.save(f'{export_path}/features.npy', all_features.detach().cpu().numpy())
            np.save(f'{export_path}/labels.npy', all_labels.detach().cpu().numpy())

    def all_sample_features(self, samples):
        """
        output: (n_desired_layer,), (n_sample, n_desired_layer) the mean feature of a bunch of samples
        """
        all_sample_features = []

        for sample in tqdm(samples, desc="Processing samples"):
            all_sample_features.append(self.get_prompt_feature(sample)[None, ...])

        all_sample_features = torch.cat(all_sample_features, dim=0)
        return all_sample_features
    
    def score(self, prompt: str):
        features = self.get_prompt_feature(prompt)
        return self.feature_classifier.score(features[None, ...]).item()
    
    def classify_prompt(self, prompt: str):
        """Classify a prompt as positive or negative based on the threshold."""
        # Calculate the score for the prompt
        score = self.score(prompt)

        # Classify the prompt based on the score and threshold
        if score > self.classify_threshold:
            return 'positive'
        else:
            return 'negative'

    def classify_score(self, score: float):
        """Classify a prompt as positive or negative based on the threshold."""
        # Calculate the score for the prompt

        # Classify the prompt based on the score and threshold
        if score > self.classify_threshold:
            return 'malicious'
        else:
            return 'benign'
        
    def batch_score_and_classify(self, prompts: List[str]):
        """
        output: (n_prompt,), (n_prompt,) the score and classification of a bunch of prompts
        """
        eval_batch_size = self.NNcfg['training']['batch_size']
        scores = []
        classifications = []
        for i in tqdm(range(0, len(prompts), eval_batch_size), desc="Processing prompts"):
            batch_prompts = prompts[i:i+eval_batch_size]
            batch_features = torch.cat([self.get_prompt_feature(prompt)[None, ...] for prompt in batch_prompts], dim=0)
            batch_scores = self.feature_classifier.score(batch_features).reshape(-1).detach().cpu().numpy().tolist()
            batch_classifications = [self.classify_score(score) for score in batch_scores]
            scores.extend(batch_scores)
            classifications.extend(batch_classifications)
        return scores, classifications
        
# extractor_type choices: 'mean', 'last'
class NaiveExtractClassifierNN(AutoFeatureExtractor):
    def __init__(self, layer_indices: list, model: HookedLLM, 
                 positive_samples: list, negative_samples:list, NNCfg: dict, extract_type: str, **kwargs):
        super().__init__(layer_indices, model, extract_type=extract_type)
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        self.negative_features = self.all_sample_features(negative_samples)
        self.positive_features = self.all_sample_features(positive_samples)

        # train MLP
        self.NNcfg = NNCfg
        self.train_features = torch.cat([self.positive_features, self.negative_features], dim=0)
        self.train_labels = torch.cat([torch.ones(len(self.positive_features)), torch.zeros(len(self.negative_features))], dim=0)
        self.feature_classifier = MLPClassifier(NNCfg, self.train_features, self.train_labels)

        self.classify_threshold = 0.5

        # if we are asked to save the train features
        if 'export_path' in kwargs:
            export_path = kwargs['export_path']
            # create path
            import os
            os.makedirs(export_path, exist_ok=True)
            # combine the features of postivive samples and negative samples and create a label array
            all_features = torch.cat([self.positive_features, self.negative_features], dim=0)
            all_labels = torch.cat([torch.ones(len(self.positive_features)), torch.zeros(len(self.negative_features))], dim=0)
            # save the features and labels as numpy arrays
            np.save(f'{export_path}/features.npy', all_features.detach().cpu().numpy())
            np.save(f'{export_path}/labels.npy', all_labels.detach().cpu().numpy())

    def all_sample_features(self, samples):
        """
        output: (n_desired_layer,), (n_sample, n_desired_layer) the mean feature of a bunch of samples
        """
        all_sample_features = []

        for sample in tqdm(samples, desc="Processing samples"):
            all_sample_features.append(self.get_prompt_feature(sample)[None, ...])

        all_sample_features = torch.cat(all_sample_features, dim=0)
        return all_sample_features
    
    def score(self, prompt: str):
        features = self.get_prompt_feature(prompt)
        return self.feature_classifier.score(features[None, ...]).item()
    
    def classify_prompt(self, prompt: str):
        """Classify a prompt as positive or negative based on the threshold."""
        # Calculate the score for the prompt
        score = self.score(prompt)

        # Classify the prompt based on the score and threshold
        if score > self.classify_threshold:
            return 'positive'
        else:
            return 'negative'
        
    def classify_score(self, score: float):
        """Classify a prompt as positive or negative based on the threshold."""
        # Calculate the score for the prompt

        # Classify the prompt based on the score and threshold
        if score > self.classify_threshold:
            return 'malicious'
        else:
            return 'benign'
    
    def batch_score_and_classify(self, prompts: List[str]):
        """
        output: (n_prompt,), (n_prompt,) the score and classification of a bunch of prompts
        """
        print("evaluating {} prompts".format(len(prompts)))
        eval_batch_size = self.NNcfg['training']['batch_size']
        scores = []
        classifications = []
        for i in tqdm(range(0, len(prompts), eval_batch_size), desc="Processing prompts"):
            batch_prompts = prompts[i:i+eval_batch_size]
            batch_features = torch.cat([self.get_prompt_feature(prompt)[None, ...] for prompt in batch_prompts], dim=0)
            batch_scores = self.feature_classifier.score(batch_features).reshape(-1).detach().cpu().numpy().tolist()
            batch_classifications = [self.classify_score(score) for score in batch_scores]
            scores.extend(batch_scores)
            classifications.extend(batch_classifications)
        return scores, classifications
