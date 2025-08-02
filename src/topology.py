"""Graph Topology Management for DCOP Problems"""

import random
from typing import Dict, List, Tuple, Optional, Any
from .validation import validate_agent_mu_config
from .global_map import (
    get_learning_phase_config, 
    get_comparison_phase_config, 
    should_log_phase_transitions
)


class SharedGraphTopology:
    """Manages shared graph topology and synchronized cost updates for fair algorithm comparison."""
    
    def __init__(
        self, 
        num_agents: int, 
        domain_size: int, 
        edge_probability: float, 
        agent_priority_config: Optional[Dict[str, Any]] = None, 
        base_seed: int = 42,
        mode: str = "comparison"
    ):
        self.num_agents = num_agents
        self.domain_size = domain_size
        self.edge_probability = edge_probability
        self.agent_priority_config = agent_priority_config or {}
        self.base_seed = base_seed
        
        # Mode and phase configuration
        self.mode = mode
        self.current_phase_config = self._load_phase_config(mode)
        
        # Log phase initialization if enabled
        if should_log_phase_transitions():
            print(f"SharedGraphTopology initialized in '{mode}' mode")
            print(f"  Vary penalties: {self.current_phase_config['vary_penalties']}")
            print(f"  Vary initial assignments: {self.current_phase_config['vary_initial_assignments']}")
        
        # Fixed graph topology - determined once and shared across all experiments
        self.graph_edges: List[Tuple[int, int]] = []
        self.agent_penalty_means: Dict[int, float] = {}
        
        # Episode-specific data for synchronized experiments
        self.current_episode = 0
        self.episode_penalties: Dict[int, Dict[int, float]] = {}  # episode_num -> {agent_id -> penalty}
        self.episode_initial_assignments: Dict[int, Dict[int, int]] = {}  # episode_num -> {agent_id -> variable}
        
        self._generate_fixed_topology()
        self._generate_penalty_means()
    
    def _load_phase_config(self, mode: str) -> Dict[str, Any]:
        if mode == "learning":
            return get_learning_phase_config()
        elif mode == "comparison":
            return get_comparison_phase_config()
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'learning' or 'comparison'")
    
    def set_mode(self, new_mode: str) -> None:
        if new_mode != self.mode:
            if should_log_phase_transitions():
                print(f"SharedGraphTopology: Switching from '{self.mode}' to '{new_mode}' mode")
            
            self.mode = new_mode
            self.current_phase_config = self._load_phase_config(new_mode)
            
            if should_log_phase_transitions():
                print(f"  New config - Vary penalties: {self.current_phase_config['vary_penalties']}")
                print(f"  New config - Vary initial assignments: {self.current_phase_config['vary_initial_assignments']}")

    def _generate_fixed_topology(self) -> None:
        topology_rng = random.Random(self.base_seed * 17)
        
        self.graph_edges = []
        for i in range(1, self.num_agents + 1):
            for j in range(i + 1, self.num_agents + 1):
                if topology_rng.random() < self.edge_probability:
                    self.graph_edges.append((i, j))
        
        print(f"Generated shared topology: {len(self.graph_edges)} edges for {self.num_agents} agents")
    
    def _generate_penalty_means(self) -> None:
        config = self.agent_priority_config
        validate_agent_mu_config(config)
        default_penalty_mean = config['default_mu']
        
        # Initialize all agents with default penalty mean
        for agent_id in range(1, self.num_agents + 1):
            self.agent_penalty_means[agent_id] = default_penalty_mean
        
        # Apply manual priority overrides
        if 'manual' in config:
            for agent_id, penalty_mean in config['manual'].items():
                if 1 <= agent_id <= self.num_agents:
                    self.agent_penalty_means[agent_id] = penalty_mean
        
        # Apply hierarchical priority configuration
        if 'hierarchical' in config:
            for priority_level, (start_id, end_id, penalty_mean) in config['hierarchical'].items():
                for agent_id in range(start_id, min(end_id + 1, self.num_agents + 1)):
                    self.agent_penalty_means[agent_id] = penalty_mean
        
        # Apply stratified random priority configuration
        if 'random_stratified' in config:
            agent_ids = list(range(1, self.num_agents + 1))
            random.Random(self.base_seed + 42).shuffle(agent_ids)  # Deterministic shuffle
            
            idx = 0
            for priority_name, (count, penalty_mean, penalty_sigma) in config['random_stratified'].items():
                for _ in range(min(count, len(agent_ids) - idx)):
                    if idx < len(agent_ids):
                        self.agent_penalty_means[agent_ids[idx]] = penalty_mean
                        idx += 1
    
    def prepare_episode(self, episode_num: int, override_mode: Optional[str] = None) -> Tuple[Dict[int, float], Dict[int, int]]:
        self.current_episode = episode_num
        
        # Use override mode if provided, otherwise use current mode
        active_mode = override_mode if override_mode else self.mode
        phase_config = self._load_phase_config(active_mode) if override_mode else self.current_phase_config
        
        # Log episode preparation if enabled
        if should_log_phase_transitions():
            print(f"Preparing episode {episode_num} in '{active_mode}' mode")
        
        # Generate episode-specific agent penalties
        episode_penalties = self._generate_episode_penalties(episode_num, phase_config)
        self.episode_penalties[episode_num] = episode_penalties
        
        # Generate episode-specific starting assignments  
        episode_assignments = self._generate_episode_assignments(episode_num, phase_config)
        self.episode_initial_assignments[episode_num] = episode_assignments
        
        return episode_penalties, episode_assignments
    
    def _generate_episode_penalties(self, episode_num: int, phase_config: Dict[str, Any]) -> Dict[int, float]:
        validate_agent_mu_config(self.agent_priority_config)
        penalty_std = self.agent_priority_config['default_sigma']
        
        if phase_config['vary_penalties']:
            penalty_rng = random.Random((self.base_seed + episode_num) * 23)
            episode_penalties = {}
            for agent_id in range(1, self.num_agents + 1):
                penalty_mean = self.agent_penalty_means[agent_id][1]
                episode_penalties[agent_id] = penalty_rng.normalvariate(penalty_mean, penalty_std)
        else:
            penalty_rng = random.Random((self.base_seed + 0) * 23)
            episode_penalties = {}
            for agent_id in range(1, self.num_agents + 1):
                penalty_mean = self.agent_penalty_means[agent_id][1]
                episode_penalties[agent_id] = penalty_rng.normalvariate(penalty_mean, penalty_std)
        
        return episode_penalties
    
    def _generate_episode_assignments(self, episode_num: int, phase_config: Dict[str, Any]) -> Dict[int, int]:
        if phase_config['vary_initial_assignments']:
            assignment_rng = random.Random((self.base_seed + episode_num) * 31)
            episode_assignments = {}
            for agent_id in range(1, self.num_agents + 1):
                episode_assignments[agent_id] = assignment_rng.randint(1, self.domain_size)
        else:
            assignment_rng = random.Random((self.base_seed + 0) * 31)
            episode_assignments = {}
            for agent_id in range(1, self.num_agents + 1):
                episode_assignments[agent_id] = assignment_rng.randint(1, self.domain_size)
        
        return episode_assignments
    
    def get_topology(self) -> List[Tuple[int, int]]:
        return self.graph_edges.copy()
    
    def get_episode_data(self, episode_num: int) -> Tuple[Dict[int, float], Dict[int, int]]:
        if episode_num not in self.episode_penalties:
            return self.prepare_episode(episode_num)
        
        return (
            self.episode_penalties[episode_num], 
            self.episode_initial_assignments[episode_num]
        )
    
    def get_topology_stats(self) -> Dict[str, Any]:
        return {
            'num_agents': self.num_agents,
            'domain_size': self.domain_size,
            'edge_probability': self.edge_probability,
            'num_edges': len(self.graph_edges),
            'density': len(self.graph_edges) / (self.num_agents * (self.num_agents - 1) / 2),
            'avg_degree': 2 * len(self.graph_edges) / self.num_agents
        }