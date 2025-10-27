# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ████ MD_Nodes/AutoLayoutOptimizer – Workflow Graph Analyzer v1.1.0 ████▓▒░
# © 2025 MDMAchine
# ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
# ░▒▓ ORIGIN & DEV:
#   • Cast into the void by: MDMAchine
#   • Enhanced by: Claude, Gemini
#   • License: Apache 2.0 — Sharing is caring
#   • Part of: ComfyUI_MD_Nodes Workflow Organization Suite

# ░▒▓ DESCRIPTION:
#   An intelligent workflow graph analyzer that detects layout issues, suggests
#   optimizations, and provides detailed reports on workflow organization. Analyzes
#   node positions, connection patterns, and visual complexity from workflow JSON.
#   Optionally applies various layout algorithms to reorganize the graph.

# ░▒▓ FEATURES:
#   ✓ Comprehensive workflow analysis from workflow JSON input.
#   ✓ Multiple layout algorithms (Hierarchical, Force-Directed, Grid, etc.).
#   ✓ Automatic workflow reorganization (outputs optimized JSON).
#   ✓ Configurable spacing (compact/normal/wide).
#   ✓ Detects common layout issues (crossings, clusters, poor spacing).
#   ✓ Generates detailed text-based optimization suggestions.
#   ✓ Visual complexity scoring system.
#   ✓ Visual report generation summarizing analysis and recommendations.

# ░▒▓ CHANGELOG:
#   - v1.1.0 (Feature Update - Multiple Layouts):
#       • ADDED: `layout_algorithm` dropdown with multiple options (Hierarchical, Force-Directed, Grid, Zonal, Spine, Circular, Radial, Orthogonal)
#       • ADDED: Placeholder classes for all new layout algorithms.
#       • UPDATED: `analyze_workflow` to act as a controller for layout engines.
#       • FIXED: Critical compliance issues with error handling and dependencies.
#   - v1.0.0 (Initial Release):
#       • ADDED: Workflow JSON analysis system, Hierarchical DAG layout, Auto-reorganization, Spacing modes, Issue detection, Optimization suggestions, Complexity scoring, Report generation.

# ░▒▓ CONFIGURATION:
#   → Primary Use: Analyze workflow layout ('layout_algorithm' = 'none') and get optimization tips via text/visual reports.
#   → Secondary Use: Automatically reorganize workflow layout using a chosen algorithm (e.g., 'hierarchical_dag') and load the 'optimized_workflow' JSON output.
#   → Edge Use: Automated layout quality scoring and reporting for complex workflows.

# ░▒▓ WARNING:
#   This node may trigger:
#   ▓▒░ Obsessive workflow reorganization and perfectionism.
#   ▓▒░ Sudden realization your "organized" workflow is chaos incarnate.
#   ▓▒░ Flashbacks to graph theory classes and agonizing over edge crossings.
#   ▓▒░ Compulsive scoring and re-scoring of workflow layouts seeking the elusive 100%.
#   Side effects include: Workflows so optimized they're featured in textbooks, colleagues asking you to "review" their graphs, and spontaneous lectures on the importance of proper node spacing. Optimize responsibly.
# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# =================================================================================
# == Standard Library Imports                                                  ==
# =================================================================================
import logging
import json
import io
import math
import traceback
import secrets # Standard import, though IS_CHANGED not used here
from collections import defaultdict

# =================================================================================
# == Third-Party Imports                                                       ==
# =================================================================================
import torch
import numpy as np

# Optional imports for visual report generation
try:
    import matplotlib as mpl
    mpl.use('Agg') # Set backend BEFORE importing pyplot (CRITICAL)
    import matplotlib.pyplot as plt
    from PIL import Image
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    # Use print for visibility during ComfyUI startup
    print("WARNING: [AutoLayoutOptimizer] Matplotlib/PIL not found. Visual reports will be blank placeholders.")

# =================================================================================
# == ComfyUI Core Modules                                                      ==
# =================================================================================
# (None needed directly)

# =================================================================================
# == Local Project Imports                                                     ==
# =================================================================================
# (None needed)

# =================================================================================
# == Helper Classes & Dependencies                                             ==
# =================================================================================

logger = logging.getLogger("AutoLayoutOptimizer")
# logger.setLevel(logging.DEBUG) # Uncomment for verbose debugging

# === Layout Engine Base Class and Implementations ===

class BaseLayoutEngine:
    """
    Base class for different layout algorithms. Defines common spacing logic.
    """
    def __init__(self, spacing_mode="normal"):
        """
        Initializes the engine with spacing configuration.

        Args:
            spacing_mode: String indicating desired spacing ('compact', 'normal', 'wide').
        """
        spacing_configs = {
            "compact": {"col": 150, "row": 100, "base": 150},
            "normal": {"col": 250, "row": 150, "base": 250}, # Default ComfyUI spacing approx
            "wide": {"col": 400, "row": 200, "base": 400}
        }
        # Use provided mode or fallback to normal
        self.spacing_config = spacing_configs.get(spacing_mode, spacing_configs["normal"])
        self.spacing = self.spacing_config["base"] # General spacing metric

    def reorganize_workflow(self, workflow):
        """
        Applies the layout algorithm to the node positions in the workflow dictionary.
        Must be implemented by subclasses.

        Args:
            workflow: Dictionary representing the workflow graph (nodes, links).

        Returns:
            Modified workflow dictionary with updated node positions. Returns original on error.
        """
        raise NotImplementedError("Subclasses must implement reorganize_workflow")

class HierarchicalLayoutEngine(BaseLayoutEngine):
    """
    Implements a hierarchical Directed Acyclic Graph (DAG) layout algorithm.
    Organizes nodes into columns (ranks) based on dependencies to create a
    clear left-to-right flow, minimizing backward-pointing connections.
    """
    def __init__(self, spacing_mode="normal"):
        """Initializes with specific column and row spacing."""
        super().__init__(spacing_mode)
        # Use column/row specific spacing for this algorithm
        self.col_spacing = self.spacing_config["col"]
        self.row_spacing = self.spacing_config["row"]

    def reorganize_workflow(self, workflow):
        """Apply hierarchical layout."""
        logger.debug("[HierarchicalLayout] Starting reorganization...")
        try:
            nodes = workflow.get("nodes", [])
            links = workflow.get("links", [])

            if not nodes:
                logger.warning("[HierarchicalLayout] No nodes found in workflow.")
                return workflow

            # 1. Build dependency graph (needed to determine columns/ranks)
            node_map = {node["id"]: node for node in nodes}
            dependencies = defaultdict(list) # dependencies[node_id] = [list of nodes it depends on]
            dependents = defaultdict(list)   # dependents[node_id] = [list of nodes depending on it]

            for link in links:
                # Link format: [link_id, source_node_id, source_slot_index, target_node_id, target_slot_index, link_type_str]
                if len(link) >= 5: # Basic format check
                    from_id, to_id = link[1], link[3]
                    # Ensure IDs are valid node IDs actually present in the workflow
                    if from_id in node_map and to_id in node_map:
                        dependencies[to_id].append(from_id)
                        dependents[from_id].append(to_id)
                    else:
                        logger.warning(f"[HierarchicalLayout] Skipping link with invalid/missing node ID: Link ID {link[0]} ({from_id} -> {to_id})")
                else:
                    logger.warning(f"[HierarchicalLayout] Skipping malformed link: {link}")

            # 2. Assign nodes to columns (ranks) based on longest path from a source node
            columns = self._assign_columns(nodes, dependencies)

            # 3. Assign row positions within each column
            positioned_nodes = self._assign_rows(columns, node_map)

            # 4. Update node positions in the workflow dictionary
            updated_nodes = 0
            for node in nodes:
                node_id = node.get("id")
                if node_id in positioned_nodes:
                    node["pos"] = positioned_nodes[node_id]
                    updated_nodes += 1

            logger.info(f"[HierarchicalLayout] Successfully positioned {updated_nodes}/{len(nodes)} nodes across {len(columns)} columns.")
            return workflow

        except Exception as e:
            logger.error(f"[HierarchicalLayout] Error during reorganization: {e}", exc_info=True)
            return workflow # Return original workflow on any error during layout

    def _assign_columns(self, nodes, dependencies):
        """Assign nodes to columns (ranks) based on dependency depth."""
        columns = defaultdict(list) # {column_index: [node_id1, node_id2, ...]}
        node_column = {} # {node_id: column_index}
        all_node_ids = {node["id"] for node in nodes}
        memo = {} # Memoization for depth calculation

        # Recursive function to calculate depth (longest path from a source)
        def get_depth(node_id, path_stack):
            if node_id in memo: return memo[node_id]
            if node_id in path_stack: # Cycle detected
                logger.warning(f"[HierarchicalLayout] Cycle detected involving node {node_id}. Assigning max depth to break.")
                return 999 # Assign arbitrary high depth

            # Add current node to path stack for cycle detection
            path_stack.add(node_id)

            if not dependencies[node_id]: # Source node
                depth = 0
            else:
                max_parent_depth = 0
                for parent_id in dependencies[node_id]:
                     # Check if parent_id exists (it should due to earlier filtering)
                     if parent_id in all_node_ids:
                          max_parent_depth = max(max_parent_depth, get_depth(parent_id, path_stack))
                     else:
                          logger.warning(f"[HierarchicalLayout] Dependency '{parent_id}' for node '{node_id}' not found in node list. Skipping for depth.")
                depth = max_parent_depth + 1

            # Remove current node from path stack after exploring its dependencies
            path_stack.remove(node_id)
            memo[node_id] = depth
            return depth

        # Calculate depth for all nodes
        max_overall_depth = 0
        for node in nodes:
            nid = node["id"]
            try:
                # Pass an empty set for initial path stack
                depth = get_depth(nid, set())
                node_column[nid] = depth
                columns[depth].append(nid)
                max_overall_depth = max(max_overall_depth, depth)
            except Exception as e:
                logger.error(f"[HierarchicalLayout] Error calculating depth for node {nid}: {e}. Placing in column 0.")
                node_column[nid] = 0
                columns[0].append(nid)

        # Handle nodes missed (e.g., disconnected components) by placing them in column 0
        assigned_node_ids = set(node_column.keys())
        unassigned_ids = all_node_ids - assigned_node_ids
        if unassigned_ids:
            logger.warning(f"[HierarchicalLayout] Found {len(unassigned_ids)} unassigned nodes (possibly disconnected graph). Placing in column 0.")
            for nid in unassigned_ids:
                node_column[nid] = 0
                columns[0].append(nid)

        return columns

    def _assign_rows(self, columns, node_map):
        """Assign Y-coordinates (rows) within each column, attempting to center."""
        positioned_nodes = {} # {node_id: [x_pos, y_pos]}
        if not columns: return positioned_nodes

        # Find the column with the maximum number of nodes to determine max height
        max_nodes_in_col = max(len(nodes_in_col) for nodes_in_col in columns.values()) if columns else 0
        total_height_estimate = (max_nodes_in_col -1) * self.row_spacing if max_nodes_in_col > 0 else 0

        # Position nodes column by column
        for col_idx in sorted(columns.keys()):
            node_ids_in_col = columns[col_idx]
            num_nodes_in_this_col = len(node_ids_in_col)
            if num_nodes_in_this_col == 0: continue

            # Calculate X position based on column index
            x_pos = col_idx * self.col_spacing

            # Calculate starting Y position to center this column vertically relative to the tallest column
            start_y = (total_height_estimate - (num_nodes_in_this_col - 1) * self.row_spacing) / 2.0
            # Ensure start_y is non-negative
            start_y = max(0, start_y)


            # Assign Y positions incrementally
            # TODO: Improve sorting within column (e.g., barycenter heuristic to reduce crossings)
            # For now, just place them in the order they appeared.
            for i, node_id in enumerate(node_ids_in_col):
                y_pos = start_y + i * self.row_spacing
                positioned_nodes[node_id] = [x_pos, y_pos]

        return positioned_nodes


# --- Other Placeholder Engines ---
# These need actual implementations or integration with layout libraries (like networkx)

class ForceDirectedLayoutEngine(BaseLayoutEngine):
    """Placeholder: Force-directed spring layout using physics simulation."""
    def reorganize_workflow(self, workflow):
        logger.warning("[AutoLayoutOptimizer] ForceDirectedLayoutEngine is not fully implemented. Using basic simulation.")
        # NOTE: This is a VERY basic simulation and likely needs refinement or a library.
        try:
            nodes = workflow.get("nodes", [])
            links = workflow.get("links", [])
            if not nodes: return workflow

            positions = {node["id"]: np.array(node.get("pos", [np.random.rand()*500,np.random.rand()*500]), dtype=float) for node in nodes}
            velocities = {node["id"]: np.zeros(2) for node in nodes}
            connections = defaultdict(list)
            for link in links:
                 if len(link) >= 5 and link[1] in positions and link[3] in positions:
                     connections[link[1]].append(link[3]); connections[link[3]].append(link[1])

            iterations = 50 # Reduced iterations for basic example
            repulsion = self.spacing * 1.5
            attraction = 0.05
            damping = 0.85

            for _ in range(iterations):
                forces = {nid: np.zeros(2) for nid in positions}
                node_ids = list(positions.keys())
                # Repulsion
                for i, nid1 in enumerate(node_ids):
                    for nid2 in node_ids[i+1:]:
                        delta = positions[nid1] - positions[nid2]
                        distance = max(np.linalg.norm(delta), 10.0) # Avoid extreme forces at close distance
                        force_mag = repulsion / (distance**2)
                        force_dir = delta / distance
                        forces[nid1] += force_dir * force_mag
                        forces[nid2] -= force_dir * force_mag
                # Attraction
                for nid1 in node_ids:
                     for nid2 in connections.get(nid1, []):
                          if nid2 in positions: # Ensure neighbor exists
                               delta = positions[nid2] - positions[nid1]
                               distance = np.linalg.norm(delta)
                               force_mag = attraction * (distance - self.spacing) # Try to maintain spacing
                               force_dir = delta / (distance + 1e-6)
                               forces[nid1] += force_dir * force_mag
                               # forces[nid2] -= force_dir * force_mag # Apply force symmetrically? Maybe not needed if iterating neighbors

                # Update positions
                for nid in node_ids:
                    velocities[nid] = (velocities[nid] + forces[nid]) * damping
                    positions[nid] += velocities[nid]

            # Update workflow nodes
            for node in nodes:
                node_id = node.get("id")
                if node_id in positions:
                    node["pos"] = [max(0, p) for p in positions[node_id].tolist()] # Ensure non-negative coords

            logger.info("[ForceDirectedLayout] Basic simulation complete.")
            return workflow
        except Exception as e:
            logger.error(f"[ForceDirectedLayout] Error: {e}", exc_info=True)
            return workflow

class GridSnapLayoutEngine(BaseLayoutEngine):
    """Snaps existing node positions to the nearest grid point."""
    def reorganize_workflow(self, workflow):
        logger.info("[AutoLayoutOptimizer] Applying Grid Snap layout...")
        grid_size = max(10, self.spacing / 2.0) # Define grid granularity
        try:
            for node in workflow.get("nodes", []):
                pos = node.get("pos", [0, 0])
                # Snap each coordinate to the nearest multiple of grid_size
                snapped_x = round(pos[0] / grid_size) * grid_size
                snapped_y = round(pos[1] / grid_size) * grid_size
                node["pos"] = [max(0, snapped_x), max(0, snapped_y)] # Ensure non-negative
            return workflow
        except Exception as e:
            logger.error(f"[GridSnapLayout] Error: {e}", exc_info=True)
            return workflow # Return original on error

class ZonalClusteringLayoutEngine(BaseLayoutEngine):
    """Placeholder: Groups related nodes by type/category and arranges zones."""
    def reorganize_workflow(self, workflow):
        logger.warning("[AutoLayoutOptimizer] ZonalClusteringLayoutEngine basic implementation.")
        # Basic implementation: Group by category prefix if available, otherwise type
        try:
            nodes = workflow.get("nodes", [])
            if not nodes: return workflow

            zones = defaultdict(list)
            for node in nodes:
                node_type = node.get("type", "UnknownType")
                # Attempt to get category from known nodes (this requires access to NODE_CLASS_MAPPINGS)
                # Simple fallback: use the first part of the type name or 'Other'
                zone_key = node_type.split('_')[0] if '_' in node_type else node_type[:10] # Simplified grouping
                zones[zone_key].append(node)

            num_zones = len(zones)
            zones_per_row = max(1, int(math.sqrt(num_zones)))
            zone_width = self.spacing * 4
            zone_height = self.spacing * 4
            zone_padding = self.spacing * 0.5

            current_zone_index = 0
            for zone_key, zone_nodes in zones.items():
                zone_row = current_zone_index // zones_per_row
                zone_col = current_zone_index % zones_per_row
                zone_x_base = zone_col * (zone_width + zone_padding)
                zone_y_base = zone_row * (zone_height + zone_padding)

                # Simple grid layout within the zone
                nodes_in_zone = len(zone_nodes)
                nodes_per_row_in_zone = max(1, int(math.sqrt(nodes_in_zone)))
                for i, node in enumerate(zone_nodes):
                    row_in_zone = i // nodes_per_row_in_zone
                    col_in_zone = i % nodes_per_row_in_zone
                    node["pos"] = [
                        max(0, zone_x_base + col_in_zone * self.spacing),
                        max(0, zone_y_base + row_in_zone * self.spacing)
                    ]
                current_zone_index += 1

            logger.info(f"[ZonalClusteringLayout] Arranged {len(zones)} zones.")
            return workflow
        except Exception as e:
            logger.error(f"[ZonalClusteringLayout] Error: {e}", exc_info=True)
            return workflow


# --- Empty Placeholders for unimplemented algorithms ---
class CriticalPathSpineLayoutEngine(BaseLayoutEngine):
    def reorganize_workflow(self, workflow): logger.warning("[AutoLayoutOptimizer] CriticalPathSpineLayout is not implemented."); return workflow
class CircularLayoutEngine(BaseLayoutEngine):
    def reorganize_workflow(self, workflow): logger.warning("[AutoLayoutOptimizer] CircularLayout is not implemented."); return workflow
class RadialTreeLayoutEngine(BaseLayoutEngine):
    def reorganize_workflow(self, workflow): logger.warning("[AutoLayoutOptimizer] RadialTreeLayout is not implemented."); return workflow
class OrthogonalBusLayoutEngine(BaseLayoutEngine):
    def reorganize_workflow(self, workflow): logger.warning("[AutoLayoutOptimizer] OrthogonalBusLayout is not implemented."); return workflow

# Mapping of algorithm names to engine classes
LAYOUT_ENGINES = {
    "none": None,
    "hierarchical_dag": HierarchicalLayoutEngine,
    "force_directed_spring": ForceDirectedLayoutEngine,
    "grid_snap": GridSnapLayoutEngine,
    "zonal_clustering": ZonalClusteringLayoutEngine,
    "critical_path_spine": CriticalPathSpineLayoutEngine,
    "circular_layout": CircularLayoutEngine,
    "radial_tree": RadialTreeLayoutEngine,
    "orthogonal_bus": OrthogonalBusLayoutEngine,
}

# === Workflow Analysis Engine ===

class WorkflowAnalyzer:
    """
    Analyzes workflow structure (nodes, links, positions) from JSON data
    to calculate complexity, detect layout issues, and generate recommendations.
    """
    def __init__(self, workflow_json_string):
        """
        Initialize the analyzer with workflow data.

        Args:
            workflow_json_string: The workflow structure as a JSON string.
        """
        self.workflow = {}
        self.nodes = []
        self.links = []
        self.node_count = 0
        self.link_count = 0
        self.node_positions = {} # Stores {node_id: [x, y]}

        if not workflow_json_string or workflow_json_string.strip() == "{}":
            logger.warning("[WorkflowAnalyzer] Received empty or default JSON string.")
            return # Leave attributes initialized as empty

        try:
            self.workflow = json.loads(workflow_json_string)
            # Safely get nodes and links, defaulting to empty lists
            self.nodes = self.workflow.get("nodes", []) or []
            self.links = self.workflow.get("links", []) or []
            self.node_count = len(self.nodes)
            self.link_count = len(self.links)
            # Pre-calculate node positions map for faster lookups
            self.node_positions = {
                node.get("id"): node.get("pos", [0.0, 0.0])
                for node in self.nodes if node.get("id") is not None
            }
            logger.debug(f"[WorkflowAnalyzer] Initialized with {self.node_count} nodes, {self.link_count} links.")

        except json.JSONDecodeError as e:
            logger.error(f"[WorkflowAnalyzer] Invalid JSON input: {e}")
            # Reset attributes to empty state
            self.__init__("{}") # Call init again with empty string
        except Exception as e:
            logger.error(f"[WorkflowAnalyzer] Unexpected error during initialization: {e}", exc_info=True)
            self.__init__("{}")

    def calculate_complexity_score(self):
        """
        Calculate an overall workflow complexity score (0-100).
        Higher score indicates a more complex/potentially cluttered layout.

        Returns:
            Complexity score (float).
        """
        if self.node_count == 0: return 0.0

        # Weighted factors contributing to complexity
        # Adjust weights as needed
        node_factor = min(self.node_count / 50.0, 1.0) * 30  # Max 30 points for node count (cap at 50 nodes)
        link_factor = min(self.link_count / 100.0, 1.0) * 20 # Max 20 points for link count (cap at 100 links)
        avg_connections = self.link_count / self.node_count
        conn_factor = min(avg_connections / 5.0, 1.0) * 15 # Max 15 points for high average connections (cap at 5)
        clusters = self.detect_clusters()
        cluster_factor = min(len(clusters) / 10.0, 1.0) * 15 # Max 15 points for clusters (cap at 10 clusters)
        spacing_score = self.analyze_spacing() # Score is 1.0 for good spacing
        spacing_penalty = (1.0 - spacing_score) * 20 # Max 20 points penalty for bad spacing

        total_score = node_factor + link_factor + conn_factor + cluster_factor + spacing_penalty
        return max(0.0, min(100.0, total_score)) # Clamp score to [0, 100]

    def detect_clusters(self):
        """
        Detects groups of nodes positioned very close to each other.

        Returns:
            List of clusters, where each cluster is a list of node IDs.
        """
        if self.node_count < 2: return []

        clusters = []
        visited = set()
        # Define cluster proximity threshold (squared distance)
        # Lower value = detects tighter clusters
        cluster_distance_sq = 200**2 # Approx distance threshold

        node_list = list(self.node_positions.items()) # List of (node_id, [x, y])

        for i, (node_id, pos) in enumerate(node_list):
            if node_id in visited: continue

            current_cluster_ids = [node_id]
            visited.add(node_id)
            queue = [i] # Use indices for efficient iteration

            head = 0
            while head < len(queue): # BFS-like expansion
                current_idx = queue[head]
                head += 1
                current_id_in_queue, current_pos = node_list[current_idx]

                # Check distance to all other unvisited nodes
                for j, (other_id, other_pos) in enumerate(node_list):
                    if other_id in visited: continue

                    dist_sq = (current_pos[0] - other_pos[0])**2 + (current_pos[1] - other_pos[1])**2
                    if dist_sq < cluster_distance_sq:
                        visited.add(other_id)
                        current_cluster_ids.append(other_id)
                        queue.append(j)

            # Only consider groups of 2 or more nodes as clusters
            if len(current_cluster_ids) > 1:
                clusters.append(current_cluster_ids)

        logger.debug(f"[WorkflowAnalyzer] Detected {len(clusters)} potential node clusters.")
        return clusters

    def analyze_spacing(self):
        """
        Analyzes average node spacing quality, returning a score (0.0-1.0).
        1.0 indicates well-spaced nodes based on an ideal distance.

        Returns:
            Spacing quality score (float).
        """
        if self.node_count < 2: return 1.0 # Perfect spacing if 0 or 1 node

        distances = []
        node_positions_list = list(self.node_positions.values())

        # Calculate pairwise distances (more efficient ways exist for large N)
        for i, pos1 in enumerate(node_positions_list):
            for pos2 in node_positions_list[i+1:]:
                # Use Euclidean distance
                distance = math.dist(pos1, pos2) # Requires Python 3.8+
                # distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) # Manual fallback
                distances.append(distance)

        if not distances: return 1.0 # Should not happen if node_count >= 2

        avg_distance = sum(distances) / len(distances)
        ideal_distance = 250.0 # Target average spacing (similar to 'normal' mode)
        tolerance = 50.0      # Allowable deviation from ideal

        # Calculate score based on deviation outside tolerance range
        deviation = abs(avg_distance - ideal_distance)
        excess_deviation = max(0.0, deviation - tolerance)
        # Normalize deviation relative to the allowable range around ideal
        # Avoid division by zero if ideal <= tolerance
        max_relevant_deviation = max(1.0, ideal_distance - tolerance) # Denominator shouldn't be zero
        normalized_penalty = min(1.0, excess_deviation / max_relevant_deviation)
        spacing_score = 1.0 - normalized_penalty

        # Apply additional penalty for severely overlapping nodes
        min_distance = min(distances) if distances else ideal_distance
        if min_distance < 50.0: # Arbitrary threshold for overlap
            overlap_penalty = (50.0 - min_distance) / 50.0
            spacing_score *= (1.0 - overlap_penalty * 0.5) # Max 50% penalty for overlap

        logger.debug(f"[WorkflowAnalyzer] Avg spacing: {avg_distance:.1f} (Ideal: {ideal_distance:.1f}). Score: {spacing_score:.2f}")
        return max(0.0, min(1.0, spacing_score)) # Clamp final score


    def detect_potential_crossings(self):
        """
        Estimates the number of potential connection line crossings.
        Uses a geometric line segment intersection algorithm.

        Returns:
            Estimated number of crossings (int).
        """
        if self.link_count < 2: return 0

        crossings = 0
        connections = [] # List of line segments ((x1, y1), (x2, y2))

        # Build list of connection segments using node positions
        for link in self.links:
            if len(link) >= 5:
                from_id, to_id = link[1], link[3]
                from_pos = self.node_positions.get(from_id)
                to_pos = self.node_positions.get(to_id)
                # Only add if both nodes exist and have positions
                if from_pos and to_pos:
                    # Approximation: Use node top-left corner 'pos'.
                    # More accurate: Add half node size to get center (if size is known/estimated).
                    connections.append(((from_pos[0], from_pos[1]), (to_pos[0], to_pos[1])))

        # Check every pair of connections for intersection
        num_connections = len(connections)
        for i in range(num_connections):
            p1, q1 = connections[i]
            for j in range(i + 1, num_connections):
                p2, q2 = connections[j]
                # Avoid counting intersections if lines share an endpoint (same node)
                shared_endpoint = (p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2)
                if not shared_endpoint and self._lines_cross(p1, q1, p2, q2):
                    crossings += 1

        logger.debug(f"[WorkflowAnalyzer] Estimated {crossings} potential wire crossings.")
        return crossings

    def _lines_cross(self, p1, q1, p2, q2):
        """
        Check if line segment (p1, q1) intersects line segment (p2, q2).
        Handles collinear cases. Standard algorithm.
        """
        # (Implementation unchanged from previous version)
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-9: return 0 # Collinear (using tolerance)
            return 1 if val > 0 else 2 # Clockwise or Counterclockwise
        def on_segment(p, q, r):
             # Check if point q lies on segment pr (handles float coords)
            return (q[0] <= max(p[0], r[0]) + 1e-9 and q[0] >= min(p[0], r[0]) - 1e-9 and
                    q[1] <= max(p[1], r[1]) + 1e-9 and q[1] >= min(p[1], r[1]) - 1e-9)

        o1 = orientation(p1, q1, p2); o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1); o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4: return True # General case
        # Special Cases (Collinear points need overlap check)
        if o1 == 0 and on_segment(p1, p2, q1): return True
        if o2 == 0 and on_segment(p1, q2, q1): return True
        if o3 == 0 and on_segment(p2, p1, q2): return True
        if o4 == 0 and on_segment(p2, q1, q2): return True
        return False


    def generate_recommendations(self):
        """
        Generate a list of human-readable optimization recommendations based on analysis.

        Returns:
            List of recommendation strings.
        """
        recommendations = []
        complexity = self.calculate_complexity_score()

        # Recommendations based on complexity
        if complexity > 75: # Adjusted threshold
            recommendations.append("🔥 HIGH COMPLEXITY: Workflow is very large or dense. Strongly consider breaking it into logical sections using 'MD: Workflow Section Organizer' or sub-workflows.")
        elif complexity > 50:
            recommendations.append("⚡ MODERATE COMPLEXITY: Consider using 'MD: Workflow Section Organizer' or 'MD: Universal Routing Hub' to improve structure.")

        # Recommendations based on clusters
        clusters = self.detect_clusters()
        if len(clusters) > 5:
            recommendations.append(f"📦 NODE CLUSTERING: Detected {len(clusters)} groups of tightly packed nodes. Spreading these out or using layout algorithms could improve readability.")

        # Recommendations based on spacing
        spacing_score = self.analyze_spacing()
        if spacing_score < 0.5:
            recommendations.append(f"📐 POOR SPACING: Average node spacing seems inconsistent or nodes are overlapping (Score: {spacing_score*100:.0f}%). Use Auto-Layout or manually adjust positions.")
        elif spacing_score < 0.8:
            recommendations.append("📐 SPACING FAIR: Node spacing could be more consistent for better visual flow.")

        # Recommendations based on crossings
        crossings = self.detect_potential_crossings()
        if crossings > 25: # Adjusted threshold
            recommendations.append(f"🕸️ SEVERE WIRE CROSSING: Estimated ~{crossings} potential connection crossings! Use 'MD: Universal Routing Hub' or apply an auto-layout algorithm urgently.")
        elif crossings > 10:
            recommendations.append(f"🕸️ MANY CROSSINGS: Estimated ~{crossings} potential crossings. Consider using 'MD: Universal Routing Hub' or applying auto-layout.")
        elif crossings > 5:
             recommendations.append(f"🕸️ SOME CROSSINGS: Estimated ~{crossings} potential crossings. Minor reorganization or a Routing Hub might help.")

        # General recommendations based on size
        if self.node_count > 40: # Adjusted threshold
            recommendations.append("📚 LARGE WORKFLOW: Use 'MD: Enhanced Annotation' nodes to label sections and add notes for clarity.")

        # Recommendations based on connectivity
        if self.node_count > 0:
            avg_connections = self.link_count / self.node_count
            if avg_connections > 3.5: # Adjusted threshold
                recommendations.append(f"🔗 HIGH CONNECTIVITY: Average {avg_connections:.1f} links/node suggests complex data flow. Consider 'MD: Universal Routing Hub' for centralizing key connections.")

        # If no specific issues found
        if not recommendations:
            recommendations.append("✅ LAYOUT OK: No major automated layout issues detected based on current metrics!")

        return recommendations

# === Visual Report Generator ===

def _create_placeholder_report():
    """Returns a blank placeholder tensor if visualization fails or is unavailable."""
    # Create a simple gray image tensor
    placeholder = torch.ones((1, 400, 600, 3), dtype=torch.float32) * 0.1 # Dark gray
    # Optionally add text (requires Pillow/cv2 to draw text onto tensor)
    return placeholder

def create_analysis_report(analyzer):
    """
    Generate a visual analysis report summarizing metrics and recommendations.

    Args:
        analyzer: An instance of WorkflowAnalyzer containing the analysis results.

    Returns:
        An IMAGE tensor (torch.Tensor [1, H, W, 3]) representing the report,
        or a placeholder tensor if visualization is unavailable or fails.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("[AutoLayoutOptimizer] Cannot create visual report - Matplotlib/PIL unavailable.")
        return _create_placeholder_report()

    fig = None # Ensure fig is defined for finally block
    buf = None # Ensure buf is defined

    try:
        plt.style.use('dark_background')
        # Adjusted figure size for better aspect ratio
        fig, (ax_metrics, ax_recs) = plt.subplots(1, 2, figsize=(12, 6), dpi=100, facecolor='#1a1a1a')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.25)

        # --- Left Panel: Metrics ---
        ax_metrics.set_facecolor('#1a1a1a')
        ax_metrics.axis('off')
        ax_metrics.set_title('Workflow Analysis Report', fontsize=16, color='white', weight='bold', pad=15)

        # Calculate metrics from analyzer
        complexity = analyzer.calculate_complexity_score()
        spacing = analyzer.analyze_spacing()
        clusters = len(analyzer.detect_clusters())
        crossings = analyzer.detect_potential_crossings()

        # Display metrics as text
        metrics_y_start = 0.85
        metrics_line_height = 0.08
        metrics_data = [
            ("Nodes:", f"{analyzer.node_count}"),
            ("Connections:", f"{analyzer.link_count}"),
            ("Complexity Score:", f"{complexity:.1f} / 100"),
            ("Spacing Quality:", f"{spacing*100:.0f}%"),
            ("Node Clusters:", f"{clusters}"),
            ("Potential Crossings:", f"~{crossings}"),
        ]

        for i, (label, value) in enumerate(metrics_data):
            y_pos = metrics_y_start - i * metrics_line_height
            ax_metrics.text(0.05, y_pos, label, transform=ax_metrics.transAxes,
                            fontsize=10, color='#87CEEB', ha='left', va='center')
            ax_metrics.text(0.95, y_pos, value, transform=ax_metrics.transAxes,
                            fontsize=10, color='white', ha='right', va='center', weight='bold')

        # Complexity Rating Bar
        bar_y_center = 0.20 # Adjusted position
        bar_height = 0.04
        ax_metrics.text(0.5, bar_y_center + 0.05, 'Complexity Rating', transform=ax_metrics.transAxes,
                       fontsize=12, color='white', ha='center', weight='bold')

        if complexity < 35: bar_color, rating = '#4CAF50', "LOW" # Green
        elif complexity < 65: bar_color, rating = '#FFC107', "MODERATE" # Amber
        else: bar_color, rating = '#F44336', "HIGH" # Red

        # Draw bar elements (background, value, outline)
        ax_metrics.add_patch(plt.Rectangle((0.1, bar_y_center - bar_height/2), 0.8, bar_height, facecolor='#333', transform=ax_metrics.transAxes, zorder=1))
        ax_metrics.add_patch(plt.Rectangle((0.1, bar_y_center - bar_height/2), 0.8 * (complexity/100.0), bar_height, facecolor=bar_color, transform=ax_metrics.transAxes, zorder=2))
        ax_metrics.add_patch(plt.Rectangle((0.1, bar_y_center - bar_height/2), 0.8, bar_height, facecolor='none', edgecolor='#555', linewidth=0.5, transform=ax_metrics.transAxes, zorder=3))
        ax_metrics.text(0.5, bar_y_center - bar_height * 2, rating, transform=ax_metrics.transAxes,
                       fontsize=14, color=bar_color, ha='center', weight='bold')


        # --- Right Panel: Recommendations ---
        ax_recs.set_facecolor('#1a1a1a')
        ax_recs.axis('off')
        ax_recs.set_title('Optimization Tips', fontsize=16, color='white', weight='bold', pad=15)

        recommendations = analyzer.generate_recommendations()
        rec_y_start = 0.85
        rec_line_height = 0.030 # Fine-tuned line height
        max_chars_per_line = 60 # Wrap length
        max_lines_total = 18    # Max lines to display to avoid overflow

        current_line_count = 0
        for i, rec in enumerate(recommendations):
             if current_line_count >= max_lines_total: break

             # Simple manual wrap
             words = rec.split(' ')
             lines = []
             current_line = "• " if i > 0 else "➡️ " # Bullet or arrow for first
             for word in words:
                 if len(current_line) + len(word) + 1 <= max_chars_per_line:
                     current_line += word + " "
                 else:
                     lines.append(current_line.strip())
                     current_line = "  " + word + " " # Indent subsequent lines
             lines.append(current_line.strip()) # Add the last line

             # Draw wrapped lines
             for line in lines:
                 if current_line_count >= max_lines_total: break
                 y_pos = rec_y_start - current_line_count * rec_line_height
                 ax_recs.text(0.05, y_pos, line, transform=ax_recs.transAxes,
                             fontsize=9, color='#FFA500', ha='left', va='top', wrap=False)
                 current_line_count += 1

             # Add small gap between recommendations if not the last line drawn
             if current_line_count < max_lines_total:
                 current_line_count += 0.5 # Add half-line gap


        # --- Convert to Tensor ---
        buf = io.BytesIO()
        # Save figure to buffer
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), dpi=DEFAULT_DPI)
        buf.seek(0)

        # Load from buffer using PIL, convert to NumPy, then to Torch Tensor
        img = Image.open(buf).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add batch dim [1, H, W, 3]

        return img_tensor

    except Exception as e:
        logger.error(f"[AutoLayoutOptimizer] Error creating analysis report visualization: {e}", exc_info=True)
        return _create_placeholder_report() # Return placeholder on error
    finally:
        # --- CRITICAL Cleanup ---
        if buf:
            try: buf.close()
            except Exception: pass
        if fig:
            try: plt.close(fig) # Ensure figure is always closed to prevent memory leaks
            except Exception: pass
        # Try to clear matplotlib's internal state if possible
        plt.clf()
        plt.cla()

# =================================================================================
# == Core Node Class: AutoLayoutOptimizer                                      ==
# =================================================================================

class AutoLayoutOptimizer:
    """
    MD: Auto-Layout Optimizer (Prototype)

    Analyzes the layout structure of a ComfyUI workflow provided as JSON.
    Calculates metrics (complexity, spacing, crossings), generates optimization
    recommendations, and optionally applies a selected auto-layout algorithm,
    outputting the modified workflow JSON.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        return {
            "required": {
                "workflow_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": (
                        "WORKFLOW JSON\n"
                        "- Paste the workflow JSON here.\n"
                        "- Obtain from ComfyUI menu: Save (API Format).\n"
                        "- The node analyzes this structure for layout issues."
                    )
                }),
                "layout_algorithm": (list(LAYOUT_ENGINES.keys()), { # Use keys from the mapping
                    "default": "none",
                    "tooltip": (
                        "LAYOUT ALGORITHM\n"
                        "- Select the algorithm for automatic reorganization.\n"
                        "- 'none': Analyze only, no layout changes applied.\n"
                        "- 'hierarchical_dag': Good for clear left-to-right flow.\n"
                        "- 'grid_snap': Snaps existing nodes to a grid.\n"
                        "- 'force_directed_spring': Basic physics simulation (experimental).\n"
                        "- 'zonal_clustering': Basic grouping by node type (experimental).\n"
                        "- Others: Not yet implemented."
                    )
                }),
                "spacing_mode": (["compact", "normal", "wide"], {
                    "default": "normal",
                    "tooltip": (
                        "LAYOUT SPACING\n"
                        "- Controls node distance for the selected algorithm.\n"
                        "- compact: Tighter spacing (~150px).\n"
                        "- normal: Standard spacing (~250px).\n"
                        "- wide: More spread out (~400px)."
                    )
                }),
                "analysis_mode": (["quick", "detailed"], {
                    "default": "detailed",
                    "tooltip": (
                        "ANALYSIS MODE\n"
                        "- quick: Provides basic metrics only in the text output.\n"
                        "- detailed: Includes qualitative assessment and recommendations in the text output."
                    )
                }),
            }
            # Optional inputs removed as they are part of required now
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("analysis_report_image", "recommendations_text", "metrics_text", "optimized_workflow_json")
    FUNCTION = "analyze_workflow"
    CATEGORY = "MD_Nodes/Workflow Organization" # Corrected category
    OUTPUT_NODE = True # Primarily outputs analysis results

    # No IS_CHANGED needed - output depends solely on inputs, default caching is fine.

    def analyze_workflow(self, workflow_json, layout_algorithm, spacing_mode, analysis_mode):
        """
        Analyzes the workflow layout, generates reports, and optionally reorganizes.

        Args:
            workflow_json: The input workflow as a JSON string.
            layout_algorithm: Name of the layout algorithm to apply ('none' for analysis only).
            spacing_mode: Spacing setting ('compact', 'normal', 'wide').
            analysis_mode: Analysis detail level ('quick' or 'detailed').

        Returns:
            Tuple: (analysis_report_image, recommendations_text, metrics_text, optimized_workflow_json)
        """
        analyzer = None # Define analyzer in outer scope for robust error handling

        try:
            # --- Initialize Analyzer ---
            analyzer = WorkflowAnalyzer(workflow_json)

            # Handle empty/invalid input JSON early
            if analyzer.node_count == 0:
                logger.warning("[AutoLayoutOptimizer] No valid nodes found in input JSON. Cannot analyze.")
                blank_image = _create_placeholder_report()
                error_msg = "ERROR: No valid nodes found in input workflow JSON."
                return (blank_image, error_msg, error_msg, workflow_json if workflow_json else "{}") # Return original JSON

            # --- Apply Layout (if requested) ---
            reorganized_workflow = analyzer.workflow.copy() # Start with a copy
            layout_applied_message = "Layout Algorithm: none (Analysis only)"
            layout_success = False

            if layout_algorithm != "none":
                logger.info(f"[AutoLayoutOptimizer] Applying layout algorithm: {layout_algorithm} ({spacing_mode} spacing)")
                engine_class = LAYOUT_ENGINES.get(layout_algorithm)
                if engine_class:
                    try:
                        engine = engine_class(spacing_mode=spacing_mode)
                        # Make a deep copy if the engine modifies in-place significantly?
                        # For now, assume it modifies the copy passed.
                        reorganized_workflow = engine.reorganize_workflow(reorganized_workflow)

                        # IMPORTANT: Re-analyze the *modified* workflow for accurate reports
                        analyzer_after_layout = WorkflowAnalyzer(json.dumps(reorganized_workflow))
                        # Check if re-analysis was successful
                        if analyzer_after_layout.node_count > 0:
                            analyzer = analyzer_after_layout # Use the new analyzer state
                            layout_applied_message = f"✅ AUTO-LAYOUT APPLIED: {layout_algorithm} ({spacing_mode})"
                            layout_success = True
                            logger.info("[AutoLayoutOptimizer] Layout reorganization and re-analysis complete.")
                        else:
                             # Should not happen if layout worked, but handle defensively
                             logger.error("[AutoLayoutOptimizer] Re-analysis after layout failed! Keeping original analysis.")
                             layout_applied_message = f"⚠️ WARNING: Layout applied, but re-analysis failed. Reports based on original layout."
                             # Keep original analyzer

                    except NotImplementedError:
                         logger.warning(f"[AutoLayoutOptimizer] Layout algorithm '{layout_algorithm}' is not implemented.")
                         layout_applied_message = f"⚠️ NOT IMPLEMENTED: '{layout_algorithm}'. No layout applied."
                    except Exception as layout_error:
                         logger.error(f"[AutoLayoutOptimizer] Error during '{layout_algorithm}' layout: {layout_error}", exc_info=True)
                         layout_applied_message = f"❌ ERROR during layout: {layout_error}. Original layout kept."
                         # Ensure analyzer reflects the original state
                         analyzer = WorkflowAnalyzer(workflow_json)
                else:
                     logger.error(f"[AutoLayoutOptimizer] Unknown layout algorithm selected: {layout_algorithm}")
                     layout_applied_message = f"❌ UNKNOWN ALGORITHM: '{layout_algorithm}'. No layout applied."


            # --- Generate Outputs (using the final state of 'analyzer') ---

            # 1. Visual Report Image
            report_image = create_analysis_report(analyzer)

            # 2. Recommendations Text
            recommendations = analyzer.generate_recommendations()
            recommendations_text = "OPTIMIZATION RECOMMENDATIONS:\n" + "="*50 + "\n\n"
            recommendations_text += layout_applied_message + "\n\n"
            if layout_success:
                 recommendations_text += ("TIP: To use the new layout, copy the 'optimized_workflow_json' output,\n"
                                          "     go to ComfyUI, clear the graph, and use 'Load' (paste JSON).\n\n")
                 recommendations_text += "POST-LAYOUT ANALYSIS & RECOMMENDATIONS:\n" + "="*50 + "\n\n"

            if recommendations:
                 recommendations_text += "\n".join(f"{i}. {rec}\n" for i, rec in enumerate(recommendations, 1))
            else:
                 recommendations_text += "No specific recommendations generated."

            # 3. Metrics Text
            complexity = analyzer.calculate_complexity_score()
            spacing = analyzer.analyze_spacing()
            clusters = len(analyzer.detect_clusters())
            crossings = analyzer.detect_potential_crossings()
            metrics_text = "WORKFLOW METRICS:\n" + "="*50 + "\n"
            metrics_text += f"- Total Nodes: {analyzer.node_count}\n"
            metrics_text += f"- Total Connections: {analyzer.link_count}\n"
            metrics_text += f"- Avg Connections/Node: {analyzer.link_count/analyzer.node_count if analyzer.node_count > 0 else 0:.1f}\n"
            metrics_text += f"- Complexity Score: {complexity:.1f} / 100\n"
            metrics_text += f"- Spacing Quality: {spacing*100:.0f}%\n"
            metrics_text += f"- Detected Node Clusters: {clusters}\n"
            metrics_text += f"- Estimated Wire Crossings: ~{crossings}\n"
            if layout_algorithm != "none":
                 metrics_text += f"- Layout Applied: {'Yes' if layout_success else 'No / Failed'}\n"
                 metrics_text += f"- Algorithm Used: {layout_algorithm} ({spacing_mode})\n"

            if analysis_mode == "detailed":
                metrics_text += "\n--- Qualitative Assessment ---\n"
                metrics_text += f"- Readability: {'Good' if complexity < 40 and crossings < 5 else ('Fair' if complexity < 70 and crossings < 15 else 'Poor')}\n"
                metrics_text += f"- Organization: {'Good' if clusters < 3 and spacing > 0.7 else ('Fair' if clusters < 6 and spacing > 0.5 else 'Poor')}\n"

            # 4. Optimized Workflow JSON (or original if layout failed/skipped)
            # Use the 'reorganized_workflow' dict, ensure it's valid JSON string
            try:
                optimized_workflow_json = json.dumps(reorganized_workflow, indent=2)
            except Exception as json_err:
                 logger.error(f"Failed to serialize optimized workflow: {json_err}")
                 optimized_workflow_json = workflow_json # Fallback to original input


            logger.info(f"[AutoLayoutOptimizer] Analysis complete.")
            # Return tuple matching RETURN_TYPES
            return (report_image, recommendations_text, metrics_text, optimized_workflow_json)

        except Exception as e:
            logger.error(f"[AutoLayoutOptimizer] Critical error in analyze_workflow: {e}", exc_info=True)
            blank_image = _create_placeholder_report()
            error_text = f"❌ CRITICAL ERROR:\n{e}\n\n{traceback.format_exc()}"
            # Return original JSON and error messages
            return (blank_image, error_text, error_text, workflow_json if workflow_json else "{}")

# =================================================================================
# == Node Registration                                                         ==
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "AutoLayoutOptimizer": AutoLayoutOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoLayoutOptimizer": "MD: Auto-Layout Optimizer", # Added MD: prefix, removed emoji
}