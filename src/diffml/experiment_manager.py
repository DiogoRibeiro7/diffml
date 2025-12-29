"""Experiment dependency management system for DiffML.

This module provides a sophisticated system for managing experiment dependencies,
execution ordering, and result caching to enable complex experimental workflows.
"""

import asyncio
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from datetime import datetime
import hashlib
import networkx as nx
import concurrent.futures
from abc import ABC, abstractmethod

from .unified_config import UnifiedExperimentConfig, load_experiment_config


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


@dataclass
class ExperimentNode:
    """Node representing an experiment in the dependency graph.

    Attributes:
        name: Unique experiment name
        config: Experiment configuration
        dependencies: List of experiment names this depends on
        status: Current execution status
        result: Experiment result (if completed)
        error: Error message (if failed)
        start_time: Execution start time
        end_time: Execution end time
        cache_key: Key for result caching
        executor: Function to execute the experiment
    """

    name: str
    config: UnifiedExperimentConfig
    dependencies: List[str] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cache_key: Optional[str] = None
    executor: Optional[Callable] = None

    def __hash__(self):
        return hash(self.name)

    def compute_cache_key(self) -> str:
        """Compute cache key based on configuration."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ExperimentGraph:
    """Directed acyclic graph for experiment dependencies."""

    def __init__(self):
        """Initialize experiment graph."""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ExperimentNode] = {}

    def add_experiment(self, node: ExperimentNode) -> None:
        """Add experiment to graph.

        Parameters:
            node: Experiment node to add

        Raises:
            ValueError: If experiment name already exists
        """
        if node.name in self.nodes:
            raise ValueError(f"Experiment '{node.name}' already exists")

        self.nodes[node.name] = node
        self.graph.add_node(node.name)

        # Add edges for dependencies
        for dep in node.dependencies:
            if dep not in self.nodes:
                raise ValueError(f"Dependency '{dep}' not found for experiment '{node.name}'")
            self.graph.add_edge(dep, node.name)

    def validate(self) -> None:
        """Validate the graph structure.

        Raises:
            ValueError: If graph contains cycles or missing dependencies
        """
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Dependency graph contains cycles: {cycles}")

        # Check all dependencies exist
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Missing dependency '{dep}' for experiment '{name}'")

    def get_execution_order(self) -> List[List[str]]:
        """Get execution order as levels (experiments in same level can run in parallel).

        Returns:
            List of levels, each level contains experiment names that can run in parallel
        """
        if not self.graph.nodes():
            return []

        # Topological generations give us levels
        return list(nx.topological_generations(self.graph))

    def get_dependencies(self, name: str) -> Set[str]:
        """Get all dependencies (transitive) for an experiment.

        Parameters:
            name: Experiment name

        Returns:
            Set of all dependency names
        """
        if name not in self.graph:
            return set()
        return nx.ancestors(self.graph, name)

    def get_dependents(self, name: str) -> Set[str]:
        """Get all experiments that depend on this one.

        Parameters:
            name: Experiment name

        Returns:
            Set of all dependent experiment names
        """
        if name not in self.graph:
            return set()
        return nx.descendants(self.graph, name)

    def visualize(self, output_path: str = "experiment_graph.png") -> None:
        """Visualize the experiment dependency graph.

        Parameters:
            output_path: Path to save visualization
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Color nodes by status
        color_map = {
            ExperimentStatus.PENDING: 'lightgray',
            ExperimentStatus.RUNNING: 'yellow',
            ExperimentStatus.COMPLETED: 'lightgreen',
            ExperimentStatus.FAILED: 'lightcoral',
            ExperimentStatus.SKIPPED: 'lightblue',
            ExperimentStatus.CACHED: 'lightcyan'
        }

        node_colors = [color_map[self.nodes[n].status] for n in self.graph.nodes()]

        nx.draw(self.graph, pos, with_labels=True, node_colors=node_colors,
               node_size=2000, font_size=10, font_weight='bold',
               arrows=True, edge_color='gray', arrowsize=20)

        plt.title("Experiment Dependency Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Graph visualization saved to {output_path}")


class ResultCache:
    """Cache for experiment results."""

    def __init__(self, cache_dir: str = ".experiment_cache"):
        """Initialize result cache.

        Parameters:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached result.

        Parameters:
            cache_key: Cache key

        Returns:
            Cached result or None if not found
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache {cache_key}: {e}")
        return None

    def put(self, cache_key: str, result: Any) -> None:
        """Store result in cache.

        Parameters:
            cache_key: Cache key
            result: Result to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Failed to cache result {cache_key}: {e}")

    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class ExperimentExecutor(ABC):
    """Abstract base class for experiment executors."""

    @abstractmethod
    async def execute(self, node: ExperimentNode, dependencies: Dict[str, Any]) -> Any:
        """Execute an experiment.

        Parameters:
            node: Experiment node to execute
            dependencies: Results from dependency experiments

        Returns:
            Experiment result
        """
        pass


class DefaultExperimentExecutor(ExperimentExecutor):
    """Default experiment executor that runs the configured function."""

    async def execute(self, node: ExperimentNode, dependencies: Dict[str, Any]) -> Any:
        """Execute experiment using the node's executor function.

        Parameters:
            node: Experiment node
            dependencies: Dependency results

        Returns:
            Experiment result
        """
        if node.executor is None:
            raise ValueError(f"No executor function defined for experiment '{node.name}'")

        # Run executor (can be sync or async)
        if asyncio.iscoroutinefunction(node.executor):
            return await node.executor(node.config, dependencies)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, node.executor, node.config, dependencies)


class ExperimentManager:
    """Main experiment management system."""

    def __init__(
        self,
        cache_results: bool = True,
        max_parallel: int = 4,
        executor: Optional[ExperimentExecutor] = None
    ):
        """Initialize experiment manager.

        Parameters:
            cache_results: Whether to cache experiment results
            max_parallel: Maximum parallel experiments
            executor: Custom experiment executor
        """
        self.graph = ExperimentGraph()
        self.cache = ResultCache() if cache_results else None
        self.max_parallel = max_parallel
        self.executor = executor or DefaultExperimentExecutor()
        self.results: Dict[str, Any] = {}

    def add_experiment(
        self,
        name: str,
        config: UnifiedExperimentConfig,
        dependencies: Optional[List[str]] = None,
        executor: Optional[Callable] = None
    ) -> None:
        """Add experiment to management system.

        Parameters:
            name: Unique experiment name
            config: Experiment configuration
            dependencies: List of dependency experiment names
            executor: Function to execute the experiment
        """
        node = ExperimentNode(
            name=name,
            config=config,
            dependencies=dependencies or [],
            executor=executor
        )
        node.cache_key = node.compute_cache_key()
        self.graph.add_experiment(node)

    def load_experiment_set(self, config_file: str) -> None:
        """Load a set of experiments from configuration file.

        Parameters:
            config_file: Path to configuration file

        Example configuration format:
            {
                "experiments": [
                    {
                        "name": "exp1",
                        "config_file": "configs/exp1.toml",
                        "dependencies": []
                    },
                    {
                        "name": "exp2",
                        "config_file": "configs/exp2.toml",
                        "dependencies": ["exp1"]
                    }
                ]
            }
        """
        with open(config_file, 'r') as f:
            data = json.load(f)

        for exp_data in data['experiments']:
            config = load_experiment_config(exp_data['config_file'])
            self.add_experiment(
                name=exp_data['name'],
                config=config,
                dependencies=exp_data.get('dependencies', [])
            )

    async def _execute_experiment(self, node: ExperimentNode) -> None:
        """Execute a single experiment.

        Parameters:
            node: Experiment node to execute
        """
        try:
            node.status = ExperimentStatus.RUNNING
            node.start_time = datetime.now()

            # Check cache
            if self.cache and node.cache_key:
                cached_result = self.cache.get(node.cache_key)
                if cached_result is not None:
                    node.result = cached_result
                    node.status = ExperimentStatus.CACHED
                    node.end_time = datetime.now()
                    self.results[node.name] = cached_result
                    print(f"✓ {node.name} (cached)")
                    return

            # Get dependency results
            dep_results = {
                dep: self.results[dep]
                for dep in node.dependencies
            }

            # Execute
            print(f"→ Running {node.name}...")
            node.result = await self.executor.execute(node, dep_results)
            self.results[node.name] = node.result

            # Cache result
            if self.cache and node.cache_key:
                self.cache.put(node.cache_key, node.result)

            node.status = ExperimentStatus.COMPLETED
            node.end_time = datetime.now()
            duration = node.duration()
            print(f"✓ {node.name} completed in {duration:.2f}s")

        except Exception as e:
            node.error = str(e)
            node.status = ExperimentStatus.FAILED
            node.end_time = datetime.now()
            print(f"✗ {node.name} failed: {e}")
            raise

    async def run_all(self) -> Dict[str, Any]:
        """Run all experiments respecting dependencies.

        Returns:
            Dictionary mapping experiment names to results
        """
        self.graph.validate()
        execution_levels = self.graph.get_execution_order()

        print(f"Executing {len(self.graph.nodes)} experiments in {len(execution_levels)} levels")

        for level_idx, level_experiments in enumerate(execution_levels):
            print(f"\nLevel {level_idx + 1}: {', '.join(level_experiments)}")

            # Run experiments in this level in parallel
            tasks = []
            for exp_name in level_experiments:
                node = self.graph.nodes[exp_name]
                # Check if dependencies succeeded
                deps_ok = all(
                    self.graph.nodes[dep].status in [ExperimentStatus.COMPLETED, ExperimentStatus.CACHED]
                    for dep in node.dependencies
                )

                if deps_ok:
                    task = asyncio.create_task(self._execute_experiment(node))
                    tasks.append(task)
                else:
                    node.status = ExperimentStatus.SKIPPED
                    print(f"⊗ {exp_name} skipped (dependency failed)")

            # Wait for level to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return self.results

    def run(self) -> Dict[str, Any]:
        """Synchronous wrapper for run_all.

        Returns:
            Experiment results
        """
        return asyncio.run(self.run_all())

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary.

        Returns:
            Summary statistics
        """
        summary = {
            'total_experiments': len(self.graph.nodes),
            'completed': sum(1 for n in self.graph.nodes.values()
                           if n.status == ExperimentStatus.COMPLETED),
            'cached': sum(1 for n in self.graph.nodes.values()
                         if n.status == ExperimentStatus.CACHED),
            'failed': sum(1 for n in self.graph.nodes.values()
                         if n.status == ExperimentStatus.FAILED),
            'skipped': sum(1 for n in self.graph.nodes.values()
                          if n.status == ExperimentStatus.SKIPPED),
            'total_duration': sum(n.duration() or 0 for n in self.graph.nodes.values()),
            'experiments': {}
        }

        for name, node in self.graph.nodes.items():
            summary['experiments'][name] = {
                'status': node.status.value,
                'duration': node.duration(),
                'dependencies': node.dependencies,
                'error': node.error
            }

        return summary

    def save_summary(self, filepath: str = "experiment_summary.json") -> None:
        """Save execution summary to file.

        Parameters:
            filepath: Path to save summary
        """
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved to {filepath}")

    def visualize_dependencies(self, output_path: str = "dependencies.png") -> None:
        """Visualize experiment dependencies.

        Parameters:
            output_path: Path to save visualization
        """
        self.graph.visualize(output_path)


# Example experiment executors
def create_digital_experiment_executor():
    """Create executor for digital option experiments."""
    def executor(config: UnifiedExperimentConfig, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        from .datasets_digital import generate_digital_dataset_train
        from .networks import FeedForwardNet
        from .training import train_model
        import torch

        # Generate dataset
        S, K, prices, deltas = generate_digital_dataset_train(
            m=config.m_train,
            n_paths=config.n_paths_train,
            r=config.r,
            sigma=config.sigma,
            T=config.T
        )

        # Create model
        model = FeedForwardNet(5, [32, 16], 1)

        # Train (simplified)
        features = torch.stack([S, K,
                               torch.full_like(S, config.r),
                               torch.full_like(S, config.sigma),
                               torch.full_like(S, config.T)], dim=1)

        # Return results
        return {
            'model': model,
            'final_loss': 0.01,  # Placeholder
            'dataset_size': config.m_train
        }

    return executor


def create_comparison_experiment_executor():
    """Create executor that compares results from dependencies."""
    def executor(config: UnifiedExperimentConfig, dependencies: Dict[str, Any]) -> Dict[str, Any]:
        # Compare results from dependencies
        comparison = {}
        for name, result in dependencies.items():
            comparison[name] = {
                'loss': result.get('final_loss', None),
                'dataset_size': result.get('dataset_size', None)
            }

        # Find best performing
        best_exp = min(dependencies.keys(),
                      key=lambda x: dependencies[x].get('final_loss', float('inf')))

        return {
            'comparison': comparison,
            'best_experiment': best_exp,
            'best_loss': dependencies[best_exp].get('final_loss')
        }

    return executor


# Example usage
if __name__ == "__main__":
    import asyncio

    # Create manager
    manager = ExperimentManager(cache_results=True, max_parallel=2)

    # Create sample experiments
    config1 = create_default_config('digital')
    config1.m_train = 1000
    config1.name = "digital_small"

    config2 = create_default_config('digital')
    config2.m_train = 5000
    config2.name = "digital_large"

    config3 = create_default_config('digital')
    config3.name = "comparison"

    # Add experiments with dependencies
    manager.add_experiment(
        "digital_small",
        config1,
        dependencies=[],
        executor=create_digital_experiment_executor()
    )

    manager.add_experiment(
        "digital_large",
        config2,
        dependencies=[],
        executor=create_digital_experiment_executor()
    )

    manager.add_experiment(
        "comparison",
        config3,
        dependencies=["digital_small", "digital_large"],
        executor=create_comparison_experiment_executor()
    )

    # Visualize dependencies
    print("Visualizing experiment dependencies...")
    manager.visualize_dependencies()

    # Run experiments
    print("\nRunning experiments...")
    results = manager.run()

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    summary = manager.get_summary()
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Completed: {summary['completed']}")
    print(f"Cached: {summary['cached']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total duration: {summary['total_duration']:.2f}s")

    # Save summary
    manager.save_summary()

    print("\n✓ Experiment management example completed!")