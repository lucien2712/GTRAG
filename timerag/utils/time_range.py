"""
Time range utilities for TimeRAG system.

This module provides utilities for parsing and validating time ranges,
particularly for quarterly financial data analysis.
"""

import re
from typing import List, Optional, Set, Tuple
from datetime import datetime


class TimeRangeParser:
    """Parser for time range specifications."""
    
    # Quarterly pattern: 2024Q1, 2023Q4, etc.
    QUARTER_PATTERN = re.compile(r'^(\d{4})Q([1-4])$')
    
    # Year pattern: 2024, 2023, etc.
    YEAR_PATTERN = re.compile(r'^(\d{4})$')
    
    @classmethod
    def parse_time_range(cls, time_range: Optional[List[str]]) -> Optional[Set[str]]:
        """
        Parse time range specification into a set of valid quarter strings.
        
        Args:
            time_range: List of time specifications, e.g., ["2024Q1", "2024Q2"] or ["2024"]
            
        Returns:
            Set of valid quarter strings, or None if no filtering should be applied
        """
        if not time_range:
            return None
        
        valid_quarters = set()
        
        for time_spec in time_range:
            if not isinstance(time_spec, str):
                continue
                
            time_spec = time_spec.strip().upper()
            
            # Check if it's a quarterly specification
            quarter_match = cls.QUARTER_PATTERN.match(time_spec)
            if quarter_match:
                valid_quarters.add(time_spec)
                continue
            
            # Check if it's a year specification
            year_match = cls.YEAR_PATTERN.match(time_spec)
            if year_match:
                year = year_match.group(1)
                # Add all quarters for this year
                for q in range(1, 5):
                    valid_quarters.add(f"{year}Q{q}")
                continue
        
        return valid_quarters if valid_quarters else None
    
    @classmethod
    def validate_time_range(cls, time_range: Optional[List[str]]) -> Tuple[bool, str]:
        """
        Validate time range specification.
        
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
            
            time_spec = time_spec.strip().upper()
            
            if not (cls.QUARTER_PATTERN.match(time_spec) or cls.YEAR_PATTERN.match(time_spec)):
                return False, f"Invalid time format: {time_spec}. Use formats like '2024Q1' or '2024'."
        
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


def calculate_temporal_relevance_score(item_quarter: str, valid_quarters: set) -> float:
    """
    Calculate temporal relevance score for an item based on its quarter.
    
    Args:
        item_quarter: Quarter string of the item
        valid_quarters: Set of valid quarters from time range
        
    Returns:
        Float score between 0.0 and 1.0, where 1.0 means perfect temporal match
    """
    if not item_quarter or not valid_quarters:
        return 0.0
    
    # Perfect match
    if item_quarter in valid_quarters:
        return 1.0
    
    # Calculate distance to nearest valid quarter
    try:
        item_year, item_q = TimeRangeParser.QUARTER_PATTERN.match(item_quarter).groups()
        item_year, item_q = int(item_year), int(item_q)
        
        min_distance = float('inf')
        for valid_q in valid_quarters:
            if TimeRangeParser.QUARTER_PATTERN.match(valid_q):
                valid_year, valid_quarter = TimeRangeParser.QUARTER_PATTERN.match(valid_q).groups()
                valid_year, valid_quarter = int(valid_year), int(valid_quarter)
                
                # Calculate quarter distance (each year = 4 quarters)
                item_total_q = item_year * 4 + item_q
                valid_total_q = valid_year * 4 + valid_quarter
                distance = abs(item_total_q - valid_total_q)
                min_distance = min(min_distance, distance)
        
        # Convert distance to relevance score (exponential decay)
        if min_distance == float('inf'):
            return 0.0
        return max(0.0, 0.8 ** min_distance)  # Decay factor of 0.8 per quarter
        
    except (AttributeError, ValueError):
        return 0.0


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