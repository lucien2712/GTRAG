"""
Time range utilities for TimeRAG system.

This module provides utilities for parsing and validating time ranges,
supporting flexible time formats through the TimeHandler class.
"""

import re
from typing import List, Optional, Set, Tuple
from datetime import datetime
from .time_handler import TimeHandler, TimePoint, TimeGranularity


class TimeRangeParser:
    """
    Parser for time range specifications using flexible TimeHandler.
    Maintains backward compatibility with quarter-based API.
    """
    
    @classmethod
    def parse_time_range(cls, time_range: Optional[List[str]]) -> Optional[Set[str]]:
        """
        Parse time range specification into a set of valid time strings.
        Now supports flexible time formats through TimeHandler.
        
        Args:
            time_range: List of time specifications (any supported format)
            
        Returns:
            Set of standardized time strings, or None if no filtering should be applied
        """
        if not time_range:
            return None
        
        valid_times = set()
        
        for time_spec in time_range:
            if not isinstance(time_spec, str):
                continue
            
            # Use TimeHandler to parse flexible formats
            parsed_time = TimeHandler.parse_time(time_spec.strip())
            if parsed_time:
                valid_times.add(parsed_time.value)
                
                # If it's a year, expand to quarters for backward compatibility
                if parsed_time.granularity == TimeGranularity.YEAR:
                    year = parsed_time.value
                    for quarter in [1, 2, 3, 4]:
                        valid_times.add(f"{year}Q{quarter}")
        
        return valid_times if valid_times else None
    
    @classmethod
    def validate_time_range(cls, time_range: Optional[List[str]]) -> Tuple[bool, str]:
        """
        Validate time range specification using flexible TimeHandler.
        
        Args:
            time_range: List of time specifications
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not time_range:
            return True, ""
        
        if not isinstance(time_range, list):
            return False, "time_range must be a list"
        
        for time_spec in time_range:
            if not isinstance(time_spec, str):
                return False, f"Invalid time specification: {time_spec}. Must be string."
            
            # Use TimeHandler to validate
            parsed_time = TimeHandler.parse_time(time_spec.strip())
            if not parsed_time:
                return False, f"Invalid time format: {time_spec}. Supported formats: ISO dates (2024-03-15), quarters (2024Q1), months (2024-03), years (2024), custom labels."
        
        return True, ""
    
    @classmethod
    def expand_time_range(cls, time_range: Optional[List[str]]) -> List[str]:
        """
        Expand time range to include all relevant quarters.
        
        Args:
            time_range: List of time specifications
            
        Returns:
            List of expanded quarter strings, sorted chronologically
        """
        valid_quarters = cls.parse_time_range(time_range)
        if not valid_quarters:
            return []
        
        # Sort quarters chronologically
        sorted_quarters = sorted(list(valid_quarters), key=cls._quarter_sort_key)
        return sorted_quarters
    
    @classmethod
    def _quarter_sort_key(cls, quarter: str) -> Tuple[int, int]:
        """Generate sort key for quarter string."""
        match = cls.QUARTER_PATTERN.match(quarter)
        if match:
            year, q = match.groups()
            return (int(year), int(q))
        return (0, 0)  # Fallback for invalid quarters
    
    @classmethod
    def get_temporal_neighbors(cls, quarter: str, include_adjacent: bool = True) -> Set[str]:
        """
        Get neighboring quarters for temporal expansion.
        
        Args:
            quarter: Quarter string like "2024Q1"
            include_adjacent: Whether to include adjacent quarters
            
        Returns:
            Set of neighboring quarter strings
        """
        neighbors = {quarter}
        
        if not include_adjacent:
            return neighbors
        
        match = cls.QUARTER_PATTERN.match(quarter)
        if not match:
            return neighbors
        
        year, q = int(match.group(1)), int(match.group(2))
        
        # Previous quarter
        if q > 1:
            neighbors.add(f"{year}Q{q-1}")
        else:
            neighbors.add(f"{year-1}Q4")
        
        # Next quarter
        if q < 4:
            neighbors.add(f"{year}Q{q+1}")
        else:
            neighbors.add(f"{year+1}Q1")
        
        return neighbors


def filter_nodes_by_time_range(nodes_data: List[dict], 
                              time_range: Optional[List[str]], 
                              temporal_expansion_mode: str = "with_temporal") -> List[dict]:
    """
    Filter nodes by time range with different expansion modes.
    
    Args:
        nodes_data: List of node dictionaries with 'quarter' metadata
        time_range: Time range specification
        temporal_expansion_mode: How to expand the time range
            - 'strict': Only exact quarters specified
            - 'with_temporal': Include quarters + temporal evolution connections
            - 'expanded': Include neighbors + temporal evolution
        
    Returns:
        Filtered list of nodes
    """
    if not time_range:
        return nodes_data
    
    valid_quarters = TimeRangeParser.parse_time_range(time_range)
    if not valid_quarters:
        return nodes_data
    
    # Apply different expansion strategies
    if temporal_expansion_mode == "expanded":
        # Include adjacent quarters
        expanded_quarters = set()
        for quarter in valid_quarters:
            expanded_quarters.update(TimeRangeParser.get_temporal_neighbors(quarter))
        valid_quarters = expanded_quarters
    elif temporal_expansion_mode == "strict":
        # Only exact quarters - no expansion needed
        pass
    # "with_temporal" uses original valid_quarters but allows temporal evolution in edges
    
    # Filter nodes
    filtered_nodes = []
    for node in nodes_data:
        node_quarter = node.get('metadata', {}).get('quarter') or node.get('quarter')
        if node_quarter in valid_quarters:
            filtered_nodes.append(node)
    
    return filtered_nodes


def filter_edges_by_time_range(edges_data: List[dict], 
                              time_range: Optional[List[str]],
                              temporal_expansion_mode: str = "with_temporal",
                              temporal_evolution_scope: str = "cross_time") -> List[dict]:
    """
    Filter edges by time range with fine-grained temporal control.
    
    Args:
        edges_data: List of edge dictionaries with 'quarter' metadata
        time_range: Time range specification
        temporal_expansion_mode: How to expand the time range
        temporal_evolution_scope: How to handle temporal evolution edges
            - 'within_range': Only temporal edges within the specified range
            - 'cross_time': Allow temporal edges that cross time boundaries
            - 'all': Include all temporal evolution edges regardless of time
        
    Returns:
        Filtered list of edges
    """
    if not time_range:
        return edges_data
    
    valid_quarters = TimeRangeParser.parse_time_range(time_range)
    if not valid_quarters:
        return edges_data
    
    # Apply expansion to valid quarters based on mode
    if temporal_expansion_mode == "expanded":
        expanded_quarters = set()
        for quarter in valid_quarters:
            expanded_quarters.update(TimeRangeParser.get_temporal_neighbors(quarter))
        valid_quarters = expanded_quarters
    
    filtered_edges = []
    for edge in edges_data:
        edge_quarter = edge.get('metadata', {}).get('quarter') or edge.get('quarter')
        
        # Check if this is a temporal evolution edge
        is_temporal_edge = False
        edge_type = edge.get('type', '')
        relation_keywords = edge.get('metadata', {}).get('relation_keywords', [])
        if 'temporal_evolution' in edge_type or 'temporal_evolution' in relation_keywords:
            is_temporal_edge = True
        
        if is_temporal_edge:
            # Handle temporal evolution edges based on scope
            if temporal_evolution_scope == "all":
                filtered_edges.append(edge)
            elif temporal_evolution_scope == "within_range" and edge_quarter in valid_quarters:
                filtered_edges.append(edge)
            elif temporal_evolution_scope == "cross_time":
                # Allow temporal edges that connect to/from valid quarters
                # This requires checking source/target nodes, but we include all for now
                filtered_edges.append(edge)
        else:
            # Regular edges: include if within valid quarters
            if edge_quarter in valid_quarters:
                filtered_edges.append(edge)
    
    return filtered_edges


def calculate_temporal_relevance_score(item_time: str, valid_times: set) -> float:
    """
    Calculate temporal relevance score for an item using flexible time formats.
    
    Args:
        item_time: Time string of the item (any supported format)
        valid_times: Set of valid time strings from time range
        
    Returns:
        Float score between 0.0 and 1.0, where 1.0 means perfect temporal match
    """
    if not item_time or not valid_times:
        return 0.0
    
    # Perfect match
    if item_time in valid_times:
        return 1.0
    
    # Parse item time
    item_parsed = TimeHandler.parse_time(item_time)
    if not item_parsed:
        return 0.0
    
    # Calculate similarity to all valid times and take the maximum
    max_similarity = 0.0
    for valid_time in valid_times:
        valid_parsed = TimeHandler.parse_time(valid_time)
        if valid_parsed:
            similarity = TimeHandler.calculate_time_similarity(item_parsed, valid_parsed)
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def calculate_combined_score(semantic_score: float, temporal_score: float, 
                           semantic_weight: float, temporal_weight: float) -> float:
    """
    Calculate combined relevance score from semantic and temporal components.
    
    Args:
        semantic_score: Semantic similarity score (0.0-1.0)
        temporal_score: Temporal relevance score (0.0-1.0)
        semantic_weight: Weight for semantic score
        temporal_weight: Weight for temporal score
        
    Returns:
        Combined score (0.0-1.0)
    """
    # Normalize weights to sum to 1.0
    total_weight = semantic_weight + temporal_weight
    if total_weight == 0:
        return semantic_score  # Fallback to semantic only
    
    norm_semantic_weight = semantic_weight / total_weight
    norm_temporal_weight = temporal_weight / total_weight
    
    return (norm_semantic_weight * semantic_score + 
            norm_temporal_weight * temporal_score)