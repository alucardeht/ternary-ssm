"""Differential Population Evolution for Ternary Networks.

Maintain a population of model variants as sparse difference masks
from a base model. Each variant differs in only a few weight positions.

For ternary weights: each diff is stored as a dict mapping parameter
positions to modified values. Sparse representation: 1000 variants x
~0.1% diff rate = minimal memory overhead.

Uses standard DE operators (mutation, crossover, selection) adapted
for discrete ternary space.
"""

import torch
import torch.nn as nn
import random


class DifferentialEvolutionTrainer:
    """Population-based training using differential evolution for ternary weights.

    Maintains population of N model variants as sparse diffs from base.
    Each generation: mutate, crossover, evaluate, select survivors.
    """

    def __init__(
        self,
        model,
        population_size: int = 50,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.5,
        elite_fraction: float = 0.2,
    ):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.device = next(model.parameters()).device

        # Collect all parameters and their shapes
        self.param_names = []
        self.param_shapes = []
        self.param_sizes = []
        self.param_offsets = []

        total_params = 0
        for name, param in model.named_parameters():
            self.param_names.append(name)
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())
            self.param_offsets.append(total_params)
            total_params += param.numel()

        self.total_params = total_params

        # Population: list of sparse diffs
        # Each diff is a dict: {flat_index: new_value}
        self.population = [dict() for _ in range(population_size)]
        self.fitness = [float("inf")] * population_size

        # Initialize population with random mutations
        n_mutations = max(1, int(total_params * mutation_rate))
        for i in range(population_size):
            for _ in range(n_mutations):
                idx = random.randint(0, total_params - 1)
                val = random.choice([-1.0, 0.0, 1.0])
                self.population[i][idx] = val

    def _apply_diff(self, base_weights: list, diff: dict) -> list:
        """Apply sparse diff to base weights."""
        modified = [w.clone() for w in base_weights]

        for idx, val in diff.items():
            # Find which parameter this index belongs to
            param_idx = 0
            for i, offset in enumerate(self.param_offsets):
                if offset <= idx < offset + self.param_sizes[i]:
                    param_idx = i
                    break

            local_idx = idx - self.param_offsets[param_idx]
            param_flat = modified[param_idx].view(-1)
            param_flat[local_idx] = val

        return modified

    def _get_base_weights(self) -> list:
        """Get current model weights as list of tensors."""
        return [param.data.clone() for _, param in self.model.named_parameters()]

    def _set_weights(self, weights: list):
        """Set model weights from list."""
        for (name, param), w in zip(self.model.named_parameters(), weights):
            param.data.copy_(w)

    def evaluate(self, loss_fn, x, targets) -> float:
        """Evaluate current model, return loss."""
        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss.item()

    def _mutate(self, diff: dict) -> dict:
        """Mutate a diff by adding/removing/changing positions."""
        new_diff = dict(diff)
        n_mutations = max(1, int(self.total_params * self.mutation_rate * 0.1))

        for _ in range(n_mutations):
            idx = random.randint(0, self.total_params - 1)
            val = random.choice([-1.0, 0.0, 1.0])
            new_diff[idx] = val

        return new_diff

    def _crossover(self, diff1: dict, diff2: dict) -> dict:
        """Uniform crossover between two diffs."""
        child = dict()
        all_keys = set(diff1.keys()) | set(diff2.keys())

        for key in all_keys:
            if random.random() < self.crossover_rate:
                if key in diff1:
                    child[key] = diff1[key]
            else:
                if key in diff2:
                    child[key] = diff2[key]

        return child

    def train_step(self, loss_fn, x, targets) -> float:
        """One generation of differential evolution.

        Returns:
            Best fitness (loss) in population.
        """
        base_weights = self._get_base_weights()

        # Evaluate all individuals
        for i in range(self.population_size):
            modified = self._apply_diff(base_weights, self.population[i])
            self._set_weights(modified)
            self.fitness[i] = self.evaluate(loss_fn, x, targets)

        # Sort by fitness (lower is better)
        sorted_indices = sorted(range(self.population_size), key=lambda i: self.fitness[i])

        # Elite selection
        n_elite = max(1, int(self.population_size * self.elite_fraction))
        elites = [self.population[i] for i in sorted_indices[:n_elite]]
        elite_fitness = [self.fitness[i] for i in sorted_indices[:n_elite]]

        # Create new population
        new_population = list(elites)  # Keep elites

        while len(new_population) < self.population_size:
            # Select two parents from elites
            p1 = random.choice(elites)
            p2 = random.choice(elites)

            # Crossover
            child = self._crossover(p1, p2)

            # Mutate
            child = self._mutate(child)

            new_population.append(child)

        self.population = new_population
        self.fitness = elite_fitness + [float("inf")] * (
            self.population_size - n_elite
        )

        # Set model to best individual
        best_diff = elites[0]
        best_weights = self._apply_diff(base_weights, best_diff)
        self._set_weights(best_weights)

        return elite_fitness[0]
