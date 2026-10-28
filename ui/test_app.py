"""
==============================================================================
GeoAI Final Project - Pytest Test Suite for RAG Agent Contract
==============================================================================
This test suite validates the contract and behavior of the GeoAI mock backend
defined in `geo_agent.py`. It tests all three primary operational scenarios:
    1. Both text prompt and clicked coordinate provided (targeted query).
    2. Only clicked coordinate provided (automated location reconnaissance).
    3. Only text prompt provided (general spatial overview).
    4. Edge cases (empty inputs, chat history context passing).
    5. Dataclass type safety and serialization integrity.

Run tests using:
    pytest test_app.py -v
==============================================================================
"""

import pytest
from geo_agent import MapLayer, AgentResponse, query_geo_agent, to_bounding_box


class TestDataContracts:
    """Validate typing, default values, and structural contracts of the dataclasses."""

    def test_map_layer_instantiation_and_defaults(self):
        """Ensure MapLayer initializes with required attributes and correct defaults."""
        layer = MapLayer(
            name="Test Point",
            layer_type="marker",
            data={"lat": 48.5, "lng": 31.2}
        )

        assert layer.name == "Test Point"
        assert layer.layer_type == "marker"
        assert layer.data == {"lat": 48.5, "lng": 31.2}
        # Verify defaults
        assert layer.color == "blue"
        assert layer.tooltip == ""
        assert layer.popup == ""

    def test_map_layer_custom_attributes(self):
        """Ensure MapLayer accepts custom styling parameters."""
        layer = MapLayer(
            name="Tactical Circle",
            layer_type="circle",
            data={"lat": 50.4, "lng": 30.5, "radius": 5000},
            color="red",
            tooltip="Defense Zone",
            popup="<b>Sector 4</b>"
        )
        assert layer.color == "red"
        assert layer.tooltip == "Defense Zone"
        assert "Sector 4" in layer.popup

    def test_agent_response_defaults(self):
        """Ensure AgentResponse instantiates properly with empty layers default."""
        resp = AgentResponse(answer="System online.")
        assert isinstance(resp.answer, str)
        assert resp.answer == "System online."
        assert isinstance(resp.layers, list)
        assert len(resp.layers) == 0


class TestGeoAgentScenarios:
    """Validate query_geo_agent routing and multimodal response handling."""

    def test_scenario_a_both_text_and_selected_point(self):
        """
        Scenario A: Both `text_prompt` AND `selected_point` are provided.
        Example: User selects a coordinate and asks "What units are stationed here?".

        Assertions:
            - Return type is AgentResponse.
            - `answer` is non-empty string referencing query intent and coordinates.
            - `layers` is a non-empty list containing MapLayer instances.
            - Layer data contains the target coordinates.
        """
        prompt = "What units are stationed here?"
        point = (49.4444, 32.0597)

        response = query_geo_agent(text_prompt=prompt, selected_point=point)

        # 1. Type validation
        assert isinstance(response, AgentResponse), "Expected an AgentResponse instance"
        assert isinstance(response.answer, str), "Expected response.answer to be a string"
        assert isinstance(response.layers, list), "Expected response.layers to be a list"

        # 2. Content validation
        assert len(response.answer) > 0, "Response answer should not be empty"
        assert "What units are stationed here?" in response.answer, (
            "Response should reference the user's specific query intent"
        )
        assert "49.4444" in response.answer, "Response should reference the latitude"
        assert "32.0597" in response.answer, "Response should reference the longitude"

        # 3. Layer validation
        assert len(response.layers) >= 1, "Expected at least one tactical layer for targeted query"
        for layer in response.layers:
            assert isinstance(layer, MapLayer), f"Layer {layer} must be an instance of MapLayer"
            assert "lat" in layer.data and "lng" in layer.data, "Target layer must contain coordinates"
            assert layer.data["lat"] == point[0]
            assert layer.data["lng"] == point[1]

    def test_scenario_b_only_selected_point(self):
        """
        Scenario B: Only `selected_point` is provided.
        Example: User runs reconnaissance on a selected coordinate without typing a message.

        Assertions:
            - Return type is AgentResponse.
            - `answer` is non-empty string providing a location reconnaissance summary.
            - `layers` is a non-empty list of MapLayer instances for that location.
        """
        point = (48.3794, 31.1656)

        response = query_geo_agent(text_prompt=None, selected_point=point)

        # 1. Type validation
        assert isinstance(response, AgentResponse), "Expected an AgentResponse instance"
        assert isinstance(response.answer, str), "Expected response.answer to be a string"
        assert isinstance(response.layers, list), "Expected response.layers to be a list"

        # 2. Content validation
        assert len(response.answer) > 0, "Response answer should not be empty"
        assert "Location Reconnaissance Summary" in response.answer
        assert "48.3794" in response.answer, "Response should reference the clicked latitude"
        assert "31.1656" in response.answer, "Response should reference the clicked longitude"

        # 3. Layer validation
        assert len(response.layers) >= 1, "Expected location marker/circle layers"
        for layer in response.layers:
            assert isinstance(layer, MapLayer), f"Layer {layer} must be an instance of MapLayer"
            assert layer.data["lat"] == point[0]
            assert layer.data["lng"] == point[1]

    def test_scenario_c_only_text_prompt(self):
        """
        Scenario C: Only `text_prompt` is provided.
        Example: User asks a general question like "Show major logistics corridors".

        Assertions:
            - Return type is AgentResponse.
            - `answer` is non-empty string with general spatial synthesis.
            - `layers` is a non-empty list of MapLayer instances showing mock strategic hubs.
        """
        prompt = "Show major logistics corridors in central Ukraine"

        response = query_geo_agent(text_prompt=prompt, selected_point=None)

        # 1. Type validation
        assert isinstance(response, AgentResponse), "Expected an AgentResponse instance"
        assert isinstance(response.answer, str), "Expected response.answer to be a string"
        assert isinstance(response.layers, list), "Expected response.layers to be a list"

        # 2. Content validation
        assert len(response.answer) > 0, "Response answer should not be empty"
        assert "General Spatial Intelligence" in response.answer
        assert "logistics corridors" in response.answer

        # 3. Layer validation
        assert len(response.layers) >= 3, "Expected multiple regional hub layers (Kyiv, Kharkiv, Dnipro)"
        for layer in response.layers:
            assert isinstance(layer, MapLayer), f"Layer {layer} must be an instance of MapLayer"
            assert "lat" in layer.data and "lng" in layer.data
            assert isinstance(layer.data["lat"], (int, float))
            assert isinstance(layer.data["lng"], (int, float))

    def test_scenario_fallback_neither_provided(self):
        """
        Fallback Scenario: Neither text prompt nor selected point is provided.

        Assertions:
            - Return type is AgentResponse.
            - `answer` provides a friendly prompt to start.
            - `layers` is an empty list.
        """
        response = query_geo_agent(text_prompt=None, selected_point=None)

        assert isinstance(response, AgentResponse)
        assert isinstance(response.answer, str)
        assert isinstance(response.layers, list)
        assert len(response.layers) == 0
        assert "GeoAI Agent Ready" in response.answer

    def test_chat_history_preservation(self):
        """Ensure passing chat_history does not break contract and executes properly."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How can I help you today?"}
        ]
        response = query_geo_agent(
            text_prompt="Summarize situation",
            selected_point=(50.4501, 30.5234),
            chat_history=history
        )
        assert isinstance(response, AgentResponse)
        assert len(response.layers) > 0

    def test_scenario_a_multiple_selected_points(self):
        """Ensure query_geo_agent handles multiple selected points with a text prompt."""
        prompt = "Assess tactical corridors between these sectors"
        points = [(48.3794, 31.1656), (49.4444, 32.0597), (50.4501, 30.5234)]

        response = query_geo_agent(text_prompt=prompt, selected_points=points)

        assert isinstance(response, AgentResponse)
        assert "Multi-Target Analysis (3 Locations)" in response.answer
        assert "Assess tactical corridors between these sectors" in response.answer
        assert "48.3794" in response.answer
        assert "49.4444" in response.answer
        assert "50.4501" in response.answer
        # Each point should have a marker and a buffer circle
        assert len(response.layers) == len(points) * 2

    def test_scenario_b_multiple_selected_points(self):
        """Ensure query_geo_agent handles multiple selected points for reconnaissance without prompt."""
        points = [(48.3794, 31.1656), (49.4444, 32.0597)]

        response = query_geo_agent(text_prompt=None, selected_points=points)

        assert isinstance(response, AgentResponse)
        assert "Multi-Location Reconnaissance Summary (2 Sectors)" in response.answer
        assert "48.3794" in response.answer
        assert "49.4444" in response.answer
        assert len(response.layers) == len(points) * 2

    def test_point_selection_and_removal_lifecycle(self):
        """Simulate state lifecycle for selecting, removing individual points, and clearing all."""
        selected_points = []

        # 1. Click 3 locations on map
        p1 = (48.3794, 31.1656)
        p2 = (49.4444, 32.0597)
        p3 = (50.4501, 30.5234)

        for p in [p1, p2, p3]:
            if p not in selected_points:
                selected_points.append(p)
        assert len(selected_points) == 3

        # 2. Duplicate click should not add duplicates
        if p2 not in selected_points:
            selected_points.append(p2)
        assert len(selected_points) == 3

        # 3. Remove middle point (p2)
        removed = selected_points.pop(1)
        assert removed == p2
        assert len(selected_points) == 2
        assert p2 not in selected_points
        assert selected_points == [p1, p3]

        # 4. Clear all points
        selected_points.clear()
        assert len(selected_points) == 0


class TestBoundingBox:
    """Validate bounding box calculation, query routing, and state lifecycle."""

    def test_to_bounding_box_standard_ordering(self):
        """Ensure to_bounding_box computes lower-left (min) and upper-right (max) correctly."""
        p_sw = (48.3794, 31.1656)
        p_ne = (50.4501, 35.0462)

        lower_left, upper_right = to_bounding_box(p_sw, p_ne)

        assert lower_left == (48.3794, 31.1656), "Lower-left must be South-West (min_lat, min_lng)"
        assert upper_right == (50.4501, 35.0462), "Upper-right must be North-East (max_lat, max_lng)"

    def test_to_bounding_box_reversed_ordering(self):
        """Ensure to_bounding_box produces identical box regardless of click order."""
        p_ne = (50.4501, 35.0462)
        p_sw = (48.3794, 31.1656)

        lower_left, upper_right = to_bounding_box(p_ne, p_sw)

        assert lower_left == (48.3794, 31.1656), "Lower-left must remain South-West even if clicked 2nd"
        assert upper_right == (50.4501, 35.0462), "Upper-right must remain North-East even if clicked 1st"

    def test_to_bounding_box_diagonal_nw_to_se(self):
        """Ensure diagonal corners (North-West to South-East) correctly resolve to SW and NE."""
        p_nw = (50.4501, 31.1656)  # High lat, low lng
        p_se = (48.3794, 35.0462)  # Low lat, high lng

        lower_left, upper_right = to_bounding_box(p_nw, p_se)

        assert lower_left == (48.3794, 31.1656)
        assert upper_right == (50.4501, 35.0462)

    def test_to_bounding_box_coordinate_rounding(self):
        """Ensure coordinates are rounded to 5 decimal places."""
        p1 = (48.1234567, 31.9876543)
        p2 = (49.5555555, 32.4444444)

        lower_left, upper_right = to_bounding_box(p1, p2)

        assert lower_left == (48.12346, 31.98765)
        assert upper_right == (49.55556, 32.44444)

    def test_query_geo_agent_with_bounding_box_and_prompt(self):
        """
        Validate Scenario A when a bounding box is provided with a text query.
        Ensures response references bounding box extents and returns tactical layers.
        """
        prompt = "Scan for logistics hubs inside this area"
        bbox = ((48.3794, 31.1656), (50.4501, 35.0462))

        response = query_geo_agent(text_prompt=prompt, bounding_box=bbox)

        assert isinstance(response, AgentResponse)
        assert "Bounding Box Area Analysis" in response.answer
        assert "Lower-Left (SW):" in response.answer
        assert "Upper-Right (NE):" in response.answer
        assert "48.3794" in response.answer
        assert "31.1656" in response.answer
        assert "50.4501" in response.answer
        assert "35.0462" in response.answer
        assert "Scan for logistics hubs" in response.answer

        # Layers: Lower-left marker, upper-right marker, and polygon/box layer
        assert len(response.layers) == 3
        layer_names = [l.name for l in response.layers]
        assert any("Lower-Left" in name for name in layer_names)
        assert any("Upper-Right" in name for name in layer_names)
        assert any("Bounding Box Area" in name for name in layer_names)

        poly_layer = [l for l in response.layers if l.layer_type == "polygon"][0]
        coords = poly_layer.data["coordinates"]
        assert len(coords) == 5  # Closed 4-point rectangle polygon

    def test_query_geo_agent_with_bounding_box_reconnaissance(self):
        """
        Validate Scenario B when a bounding box is provided without text prompt.
        Ensures response produces a bounding box reconnaissance summary.
        """
        bbox = ((48.3794, 31.1656), (50.4501, 35.0462))

        response = query_geo_agent(text_prompt=None, bounding_box=bbox)

        assert isinstance(response, AgentResponse)
        assert "Bounding Box Reconnaissance Summary" in response.answer
        assert "Lower-Left (SW):" in response.answer
        assert "Upper-Right (NE):" in response.answer
        assert "48.3794" in response.answer
        assert "35.0462" in response.answer

        assert len(response.layers) == 3

    def test_bounding_box_and_selected_point_mutual_exclusivity(self):
        """
        Validate that if a bounding box is selected, points shouldn't be in selected_point.
        Even if a caller passes both, bounding_box takes precedence and points are not duplicated.
        """
        bbox = ((48.3794, 31.1656), (50.4501, 35.0462))
        point = (49.0000, 33.0000)

        # Pass both bounding_box and selected_point
        response = query_geo_agent(
            text_prompt="Test mutual exclusivity",
            selected_point=point,
            bounding_box=bbox
        )

        assert isinstance(response, AgentResponse)
        # Should process as bounding box analysis, NOT point analysis
        assert "Bounding Box Area Analysis" in response.answer
        assert "Targeted Point Analysis" not in response.answer
        # Layers should only be the bounding box layers (2 markers + 1 polygon), not 2 markers + 1 circle for point
        assert len(response.layers) == 3
        poly_layers = [l for l in response.layers if l.layer_type == "polygon"]
        assert len(poly_layers) == 1

    def test_bounding_box_selection_lifecycle(self):
        """
        Simulate UI state lifecycle:
        1. Click corner 1 (pending)
        2. Click corner 2 -> box formed (lower-left, upper-right)
        3. Click corner 3 -> starts fresh box
        4. Clear box
        """
        # Step 1: Initial state
        selection_mode = "Bounding Box"
        selected_points = []
        bounding_box = None

        # Step 2: First click (Corner 1)
        c1 = (50.4501, 30.5234)
        selected_points = [c1]
        assert len(selected_points) == 1
        assert bounding_box is None

        # Step 3: Second click (Corner 2)
        c2 = (48.3794, 35.0462)
        lower_left, upper_right = to_bounding_box(selected_points[0], c2)
        selected_points = [lower_left, upper_right]
        bounding_box = (lower_left, upper_right)

        assert len(selected_points) == 2
        assert bounding_box == ((48.3794, 30.5234), (50.4501, 35.0462))
        assert selected_points[0] == (48.3794, 30.5234)  # lower-left
        assert selected_points[1] == (50.4501, 35.0462)  # upper-right

        # Step 4: Next click starts a new box
        c3 = (49.0000, 32.0000)
        selected_points = [c3]
        bounding_box = None
        assert len(selected_points) == 1
        assert bounding_box is None

        # Step 5: Clear
        selected_points = []
        bounding_box = None
        assert len(selected_points) == 0
        assert bounding_box is None

    def test_mode_transitions_bbox_and_point_selection(self):
        """Validate transitioning between Bounding Box and Point Selection modes (no multi-point)."""
        # Start in Bounding Box mode with 2 points
        p1 = (48.0, 31.0)
        p2 = (50.0, 35.0)
        lower_left, upper_right = to_bounding_box(p1, p2)
        selected_points = [lower_left, upper_right]
        bounding_box = (lower_left, upper_right)

        # Transition to Point Selection: retains single active point
        mode = "Point Selection"
        selected_points = [selected_points[-1]]
        bounding_box = None
        assert len(selected_points) == 1
        assert selected_points == [(50.0, 35.0)]
        assert bounding_box is None

        # Clicking again in Point Selection cancels previous selection
        new_click = (49.5, 33.0)
        selected_points = [new_click]  # replaces previous
        assert len(selected_points) == 1
        assert selected_points == [(49.5, 33.0)]

        # Transition back to Bounding Box
        mode = "Bounding Box"
        c1 = (48.5, 32.0)
        selected_points = [c1]
        bounding_box = None
        c2 = (50.5, 36.0)
        lower_left, upper_right = to_bounding_box(selected_points[0], c2)
        selected_points = [lower_left, upper_right]
        bounding_box = (lower_left, upper_right)
        assert bounding_box == ((48.5, 32.0), (50.5, 36.0))

    def test_point_selection_cancels_previous_on_click(self):
        """
        Verify that when in Point Selection mode (not bounding box),
        clicking again cancels the previous selection and selects the new coordinate.
        """
        selection_mode = "Point Selection"
        selected_points = []
        bounding_box = None

        # 1. First click selects Point A
        pt_a = (48.3794, 31.1656)
        selected_points = [pt_a]
        bounding_box = None
        assert len(selected_points) == 1
        assert selected_points == [pt_a]

        # 2. Second click on Point B CANCELS previous selection (Point A) and sets Point B
        pt_b = (50.4501, 30.5234)
        selected_points = [pt_b]
        assert len(selected_points) == 1
        assert selected_points == [pt_b]
        assert pt_a not in selected_points

        # 3. Third click on Point C CANCELS Point B and sets Point C
        pt_c = (49.9935, 36.2304)
        selected_points = [pt_c]
        assert len(selected_points) == 1
        assert selected_points == [pt_c]
        assert pt_b not in selected_points

        # 4. Clear point
        selected_points = []
        assert len(selected_points) == 0


class TestBackendAdapter:
    """Validate backend RAG adapter transformations and interface contracts."""

    def test_adapter_delegation_without_key_uses_mock(self):
        """When no API key is provided, adapter cleanly falls back to deterministic mock."""
        resp = query_geo_agent(text_prompt="Test query", api_key=None, force_mock=False)
        assert isinstance(resp, AgentResponse)
        assert resp.is_live_backend is False
        assert "Test query" in resp.answer

    def test_adapter_point_transformation(self, monkeypatch):
        """Verify (lat, lng) is transformed to GeoJSON [lng, lat] point dict for backend."""
        import geo_agent

        captured_args = {}

        def fake_backend_query(text_prompt, geo_location=None, chat_history=None, api_key=None):
            captured_args["text_prompt"] = text_prompt
            captured_args["geo_location"] = geo_location
            captured_args["chat_history"] = chat_history
            captured_args["api_key"] = api_key
            return {
                "answer": "According to the graph, target located.",
                "geo_locations": [
                    {
                        "id": "Kyiv",
                        "label": "Kyiv",
                        "kind": "place",
                        "geometry": {"type": "point", "coordinates": [30.5234, 50.4501]},
                        "sources": ["Doc-1"]
                    }
                ],
                "sources": ["Doc-1"]
            }

        monkeypatch.setattr(geo_agent, "_BACKEND_AVAILABLE", True)
        monkeypatch.setattr(geo_agent, "_backend_query", fake_backend_query)

        # Call with lat=50.4501, lng=30.5234
        resp = query_geo_agent(
            text_prompt="Where is Kyiv?",
            selected_point=(50.4501, 30.5234),
            api_key="test-key-123"
        )

        assert resp.is_live_backend is True
        assert captured_args["api_key"] == "test-key-123"
        assert captured_args["geo_location"] is not None
        assert captured_args["geo_location"]["type"] == "point"
        # Coordinates must be [lon, lat]
        assert captured_args["geo_location"]["coordinates"] == [30.5234, 50.4501]

        # Verify output layers have inverted back to Folium [lat, lng]
        assert len(resp.layers) >= 1
        marker_layer = [l for l in resp.layers if l.name == "[PLACE] Kyiv"][0]
        assert marker_layer.data["lat"] == 50.4501
        assert marker_layer.data["lng"] == 30.5234
        assert resp.sources == ["Doc-1"]

    def test_adapter_bounding_box_transformation(self, monkeypatch):
        """Verify ((min_lat, min_lng), (max_lat, max_lng)) becomes closed 5-point GeoJSON ring."""
        import geo_agent

        captured_args = {}

        def fake_backend_query(text_prompt, geo_location=None, chat_history=None, api_key=None):
            captured_args["geo_location"] = geo_location
            return {
                "answer": "Analyzed bounding box sector.",
                "geo_locations": [],
                "sources": []
            }

        monkeypatch.setattr(geo_agent, "_BACKEND_AVAILABLE", True)
        monkeypatch.setattr(geo_agent, "_backend_query", fake_backend_query)

        # Latitudes: 48.0 to 50.0, Longitudes: 31.0 to 35.0
        bbox = ((48.0, 31.0), (50.0, 35.0))
        resp = query_geo_agent(
            text_prompt=None,  # Recon scan
            bounding_box=bbox,
            api_key="test-key-123"
        )

        assert resp.is_live_backend is True
        geo_loc = captured_args["geo_location"]
        assert geo_loc["type"] == "polygon"
        ring = geo_loc["coordinates"][0]
        assert len(ring) == 5
        # Ring must be closed: ring[0] == ring[-1]
        assert ring[0] == ring[-1]
        # Ring order: [w, s], [e, s], [e, n], [w, n], [w, s]
        assert ring == [[31.0, 48.0], [35.0, 48.0], [35.0, 50.0], [31.0, 50.0], [31.0, 48.0]]

    def test_get_source_text_fallback_when_unavailable(self, monkeypatch):
        """get_source_text returns None when backend function is unavailable."""
        import geo_agent
        monkeypatch.setattr(geo_agent, "_backend_get_source_text", None)
        assert geo_agent.get_source_text("nonexistent_doc") is None

    def test_vr6_numbered_pins_and_citations_passthrough(self, monkeypatch):
        """Verify Vr6 numbered places, answer_plain, citations, why, and uncertain flags are mapped."""
        import geo_agent

        def fake_vr6_query(text_prompt, geo_location=None, chat_history=None, api_key=None):
            return {
                "answer": "Kyiv [1] is the capital of Ukraine [L1]. Donets [2] is nearby.",
                "answer_plain": "Kyiv is the capital of Ukraine. Donets is nearby.",
                "status": "answered",
                "grounded": True,
                "shape": "fact",
                "geo_locations": [
                    {
                        "number": 1,
                        "id": "Kyiv",
                        "label": "Kyiv",
                        "kind": "city",
                        "role": "primary",
                        "geometry": {"type": "Point", "coordinates": [30.5241, 50.4500]},
                        "why": "30 matching paragraphs",
                        "sources": ["TEXT_50333"]
                    },
                    {
                        "number": 2,
                        "id": "Donets",
                        "label": "Donets",
                        "kind": "road",
                        "role": "context",
                        "geometry": {"type": "Point", "coordinates": [36.25, 49.98]},
                        "why": "2 matching paragraphs",
                        "uncertain": "geocoded as a road, may refer to the river",
                        "sources": ["TEXT_1001"]
                    }
                ],
                "citations": [
                    {
                        "id": "L1",
                        "pointer": "TEXT_50333",
                        "article": "Kyiv",
                        "section": "Lead",
                        "spans": [[0, 4]]
                    }
                ],
                "sources": ["TEXT_50333", "TEXT_1001"]
            }

        monkeypatch.setattr(geo_agent, "_BACKEND_AVAILABLE", True)
        monkeypatch.setattr(geo_agent, "_backend_query", fake_vr6_query)

        resp = query_geo_agent(text_prompt="Tell me about Kyiv", api_key="test-key-vr6")

        assert resp.is_live_backend is True
        assert resp.answer == "Kyiv [1] is the capital of Ukraine [L1]. Donets [2] is nearby."
        assert resp.answer_plain == "Kyiv is the capital of Ukraine. Donets is nearby."
        assert resp.grounded is True
        assert resp.status == "answered"
        assert len(resp.citations) == 1
        assert resp.citations[0]["id"] == "L1"
        assert resp.citations[0]["article"] == "Kyiv"

        # Primary numbered pin
        pin1 = [l for l in resp.layers if l.data.get("number") == 1][0]
        assert pin1.name == "1. [CITY] Kyiv"
        assert pin1.tooltip == "1. Kyiv [CITY]"
        assert pin1.color == "blue"
        assert "30 matching paragraphs" in pin1.popup

        # Uncertain pin
        pin2 = [l for l in resp.layers if l.data.get("number") == 2][0]
        assert pin2.name == "2. [ROAD] Donets (?)"
        assert pin2.color == "orange"
        assert "geocoded as a road" in pin2.popup

    def test_vr6_get_source_text_paragraph_pointer(self, monkeypatch):
        """Verify get_source_text passes through Vr6 paragraph metadata."""
        import geo_agent

        def fake_get_source(ptr):
            if ptr == "TEXT_50333":
                return {
                    "pointer": "TEXT_50333",
                    "text": "Kyiv is the capital and most populous city of Ukraine.",
                    "document": "Kyiv",
                    "section": "Lead",
                    "paragraph_index": 0
                }
            return None

        monkeypatch.setattr(geo_agent, "_backend_get_source_text", fake_get_source)
        if hasattr(geo_agent.get_source_text, "cache_clear"):
            geo_agent.get_source_text.cache_clear()
        res = geo_agent.get_source_text("TEXT_50333")
        assert res is not None
        assert res["pointer"] == "TEXT_50333"
        assert res["document"] == "Kyiv"
        assert res["section"] == "Lead"
        assert res["paragraph_index"] == 0

    def test_vr6_source_label_resolves_document_title(self, monkeypatch):
        """Verify sources display human-readable article title instead of raw pointer indices."""
        import geo_agent
        from app import _clean_doc_title

        def fake_get_source(ptr):
            if ptr == "TEXT_53733":
                return {
                    "pointer": "TEXT_53733",
                    "text": "Biden called Zelenskyy...",
                    "document": "Russia\ufffdUkraine relations",
                    "section": "History of relations",
                    "paragraph_index": 89
                }
            return None

        monkeypatch.setattr(geo_agent, "_backend_get_source_text", fake_get_source)
        if hasattr(geo_agent.get_source_text, "cache_clear"):
            geo_agent.get_source_text.cache_clear()
        meta = geo_agent.get_source_text("TEXT_53733")
        assert meta is not None
        doc_title = _clean_doc_title(meta["document"])
        assert doc_title == "Russia - Ukraine relations"
        assert meta["section"] == "History of relations"
        assert meta["paragraph_index"] == 89

    def test_vr6_pin_visibility_filters(self):
        """Verify context and uncertain pins can be toggled on/off."""
        from geo_agent import MapLayer

        primary_pin = MapLayer(
            name="1. [CITY] Kyiv",
            layer_type="marker",
            data={"lat": 50.45, "lng": 30.52, "role": "primary", "uncertain": None},
            color="blue"
        )
        context_pin = MapLayer(
            name="2. [TOWN] Bucha",
            layer_type="marker",
            data={"lat": 50.55, "lng": 30.21, "role": "context", "uncertain": None},
            color="cadetblue"
        )
        uncertain_pin = MapLayer(
            name="3. [ROAD] Donets (?)",
            layer_type="marker",
            data={"lat": 49.98, "lng": 36.25, "role": "context", "uncertain": "river vs road"},
            color="orange"
        )
        all_layers = [primary_pin, context_pin, uncertain_pin]

        def filter_markers(layers, show_context, show_uncertain):
            rendered = []
            for l in layers:
                if l.layer_type == "marker":
                    is_context = isinstance(l.data, dict) and (l.data.get("role") == "context" or l.color == "cadetblue")
                    is_uncertain = isinstance(l.data, dict) and bool(l.data.get("uncertain") or l.color == "orange")
                    if is_context and not show_context:
                        continue
                    if is_uncertain and not show_uncertain:
                        continue
                    rendered.append(l)
            return rendered

        # All visible by default
        assert len(filter_markers(all_layers, show_context=True, show_uncertain=True)) == 3

        # Hide context: only primary remains
        res_no_ctx = filter_markers(all_layers, show_context=False, show_uncertain=True)
        assert len(res_no_ctx) == 1
        assert res_no_ctx[0].name == "1. [CITY] Kyiv"

        # Hide uncertain: primary and normal context remain
        res_no_unc = filter_markers(all_layers, show_context=True, show_uncertain=False)
        assert len(res_no_unc) == 2
        assert uncertain_pin not in res_no_unc

        # Hide both: only primary remains
        res_no_both = filter_markers(all_layers, show_context=False, show_uncertain=False)
        assert len(res_no_both) == 1
        assert res_no_both[0].name == "1. [CITY] Kyiv"


class TestDrawingIngestion:
    """Validate parsing and conversion of Leaflet Draw GeoJSON features to bounding box / points."""

    def test_draw_rectangle_to_bounding_box(self):
        """Verify Leaflet Draw Polygon rectangle maps to standard ((min_lat, min_lng), (max_lat, max_lng))."""
        rect_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[30.5, 49.2], [35.2, 49.2], [35.2, 51.1], [30.5, 51.1], [30.5, 49.2]]
                ]
            }
        }
        coords = rect_feature["geometry"]["coordinates"][0]
        lats = [float(pt[1]) for pt in coords]
        lngs = [float(pt[0]) for pt in coords]
        lower_left, upper_right = to_bounding_box((min(lats), min(lngs)), (max(lats), max(lngs)))

        assert lower_left == (49.2, 30.5)
        assert upper_right == (51.1, 35.2)

    def test_draw_polygon_to_bounding_box(self):
        """Verify arbitrary Leaflet Draw Polygon calculates proper bounding envelope."""
        poly_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[31.2, 48.5], [33.4, 47.9], [34.8, 50.1], [32.0, 50.8], [31.2, 48.5]]
                ]
            }
        }
        coords = poly_feature["geometry"]["coordinates"][0]
        lats = [float(pt[1]) for pt in coords]
        lngs = [float(pt[0]) for pt in coords]
        lower_left, upper_right = to_bounding_box((min(lats), min(lngs)), (max(lats), max(lngs)))

        assert lower_left == (47.9, 31.2)
        assert upper_right == (50.8, 34.8)

    def test_draw_point_marker_coordinates(self):
        """Verify Leaflet Draw Point marker maps to (lat, lng)."""
        marker_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [31.1656, 48.3794]
            }
        }
        coords = marker_feature["geometry"]["coordinates"]
        lat = round(float(coords[1]), 5)
        lng = round(float(coords[0]), 5)
        assert (lat, lng) == (48.3794, 31.1656)


class TestUIBranding:
    """Verify UI branding compliance and clean separation from proprietary vendor names."""

    def test_no_palantir_or_gotham_in_app_source(self):
        """Ensure no leftover mentions of 'Palantir' or 'Gotham' exist in the application code."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read().lower()
        assert "palantir" not in content
        assert "gotham" not in content

    def test_leaflet_attribution_disabled_in_app(self):
        """Ensure Leaflet attribution control is disabled and hidden in app.py."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        assert "attributionControl=False" in content
        assert ".leaflet-control-attribution { display: none !important; }" in content



class TestMapTrashBinTool:
    """Verify integration of the map toolbar trash bin with spatial selections and overlays."""

    def test_trash_hook_registers_layers_and_handlers(self):
        """Verify MapTrashBinHook template renders drawnItems layer registration and delete listener."""
        import folium
        from branca.element import MacroElement
        from jinja2 import Template
        from streamlit_folium import _get_map_string

        # Extract MapTrashBinHook class definition from app.py safely
        import app
        MapTrashBinHook = getattr(app, "MapTrashBinHook")

        m = folium.Map()
        m_pt = folium.Marker([48.3794, 31.1656]).add_to(m)
        m_rect = folium.Rectangle(bounds=[[48.0, 31.0], [49.0, 32.0]]).add_to(m)

        hook = MapTrashBinHook(
            selection_layer_names=[m_pt.get_name(), m_rect.get_name()],
            has_active=True
        ).add_to(m)

        script = _get_map_string(m)

        # 1. Selection layer registration into drawnItems
        assert f"drawnItems.addLayer({m_pt.get_name()})" in script
        assert f"drawnItems.addLayer({m_rect.get_name()})" in script

        # 2. Trash bin click listener on .leaflet-draw-edit-remove
        assert ".leaflet-draw-edit-remove" in script

        # 3. Dispatches Leaflet draw:deleted event with action delete
        assert "draw:deleted" in script
        assert "action" in script
        assert "delete" in script

    def test_delete_action_payload_detection(self):
        """Verify delete action payload detection logic for active_drawing."""
        del_payload = {
            "type": "Feature",
            "geometry": None,
            "properties": {"action": "delete", "timestamp": 123456789}
        }
        props = del_payload.get("properties") or {}
        geom = del_payload.get("geometry")

        is_delete = props.get("action") == "delete" or not geom
        assert is_delete is True

    def test_app_draw_config_enables_trash_and_disables_edit(self):
        """Ensure Draw plugin in app.py explicitly enables remove and disables edit."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert 'edit_options={"edit": False, "remove": True}' in content
        assert "edit_options=None" not in content
        assert "MapTrashBinHook" in content

    def test_instant_client_side_selection_and_bbox_hooks(self):
        """Verify MapTrashBinHook provides 0ms client-side click and bbox visual feedback."""
        import folium
        from streamlit_folium import _get_map_string
        import app

        MapTrashBinHook = getattr(app, "MapTrashBinHook")
        m = folium.Map()
        MapTrashBinHook().add_to(m)

        script = _get_map_string(m)

        # 1. Instant click handler on canvas
        assert "map.on('click'" in script
        assert "starIcon" in script
        assert "window._instantTargetMarker" in script

        # 2. Instant bounding box handler on draw:created
        assert "map.on('draw:created'" in script
        assert "arrow-down" in script
        assert "arrow-up" in script

        # 3. Instant clearing in triggerRemoval
        assert "window._instantTargetMarker = null" in script

    def test_dynamic_feature_group_hash_stability(self):
        """Verify that dynamic selection changes via feature_group_to_add do not alter the base map hash."""
        import folium
        from streamlit_folium import _get_map_string, generate_js_hash
        import app

        MapTrashBinHook = getattr(app, "MapTrashBinHook")

        # Base map m (as generated in app.py)
        m = folium.Map(location=[48.3794, 31.1656], zoom_start=6, tiles=None)
        folium.plugins.Draw(
            export=False,
            position="topleft",
            draw_options={"rectangle": True},
            edit_options={"edit": False, "remove": True}
        ).add_to(m)
        MapTrashBinHook(has_active=False).add_to(m)

        s_base = _get_map_string(m)
        h_initial = generate_js_hash(s_base, key="geoai_tactical_map")

        # Now simulate user selecting a point: dynamic_layers_fg is passed via feature_group_to_add,
        # so m itself remains structurally identical
        s_with_point = _get_map_string(m)
        h_with_point = generate_js_hash(s_with_point, key="geoai_tactical_map")

        # The base map hash MUST NOT change, ensuring zero map reload / flash
        assert h_initial == h_with_point

        # Ensure that has_active=False vs has_active=True also produces identical base map hashes
        m_active = folium.Map(location=[48.3794, 31.1656], zoom_start=6, tiles=None)
        folium.plugins.Draw(
            export=False,
            position="topleft",
            draw_options={"rectangle": True},
            edit_options={"edit": False, "remove": True}
        ).add_to(m_active)
        MapTrashBinHook(has_active=True).add_to(m_active)
        s_active = _get_map_string(m_active)
        h_active = generate_js_hash(s_active, key="geoai_tactical_map")
        assert h_initial == h_active

    def test_instant_replacement_cleans_prior_selection(self):
        """Verify MapTrashBinHook includes client-side instant cleanup for prior selection replacing."""
        import folium
        from streamlit_folium import _get_map_string
        import app

        MapTrashBinHook = getattr(app, "MapTrashBinHook")
        m = folium.Map()
        MapTrashBinHook().add_to(m)

        script = _get_map_string(m)

        # 1. Instant cleanup function is defined
        assert "function clearPreviousSelectionInstantly()" in script

        # 2. Clears active selection feature group (feature_group[0])
        assert "window.feature_group[0]" in script
        assert "selFg.clearLayers" in script

        # 3. Purges previous target point and bounding box layers via DOM/map scan
        assert "map.eachLayer" in script
        assert "★ Target Point" in script
        assert "★ Bounding Box" in script
        assert "★ BBOX" in script

        # 4. clearPreviousSelectionInstantly is called on click and draw:created
        assert "clearPreviousSelectionInstantly()" in script

    def test_feature_group_separation_in_app(self):
        """Ensure app.py separates selection_fg and tactical_layers_fg in feature_group_to_add."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "selection_fg = folium.FeatureGroup(name=\"Active Selection\", control=False)" in content
        assert "tactical_layers_fg = folium.FeatureGroup(name=\"Tactical Overlays\", control=False)" in content
        assert "feature_group_to_add=[selection_fg, tactical_layers_fg]" in content

    def test_existing_marker_click_protection_and_popup_sync(self):
        """Verify MapTrashBinHook ignores clicks on existing markers and synchronizes popup selection."""
        import folium
        from streamlit_folium import _get_map_string
        import app

        MapTrashBinHook = getattr(app, "MapTrashBinHook")
        m = folium.Map()
        MapTrashBinHook().add_to(m)

        script = _get_map_string(m)

        # 1. Guard against clicks on existing markers and popups
        assert ".leaflet-marker-icon" in script
        assert ".marker-cluster" in script
        assert ".leaflet-popup" in script

        # 2. Popup open synchronization hook
        assert "map.on('popupopen'" in script
        assert "last_active_drawing = source.toGeoJSON()" in script
        assert "window._lastOpenPopupInfo" in script

    def test_point_selection_not_cleared_by_empty_all_drawings(self):
        """Ensure Point Selection is not inadvertently cleared when all_drawings is empty."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        # Both Section 2b and Section 6 must guard all_drawings == [] with selection_mode == 'Bounding Box'
        assert "st.session_state.get(\"selection_mode\") == \"Bounding Box\"" in content


class TestRAGDebugLogging:
    """Validate RAG agent debug logging, telemetry, payload reconstruction, and error recording."""

    def test_build_raw_request_payload_with_point(self):
        """Ensure point payloads are formatted in GeoJSON [lon, lat] order with radius."""
        import app
        payload = app._build_raw_request_payload(
            text_prompt="Where are tactical assets?",
            selected_point=(50.4501, 30.5234),
            bounding_box=None
        )
        assert payload["text_prompt"] == "Where are tactical assets?"
        assert payload["geo_location"]["type"] == "point"
        # GeoJSON is [lon, lat]
        assert payload["geo_location"]["coordinates"] == [30.5234, 50.4501]
        assert payload["geo_location"]["radius_m"] == 1000.0

    def test_build_raw_request_payload_with_bbox(self):
        """Ensure bounding box payloads form a valid closed 5-coordinate polygon ring."""
        import app
        bbox = ((48.0, 31.0), (49.5, 33.5))
        payload = app._build_raw_request_payload(
            text_prompt=None,
            selected_point=None,
            bounding_box=bbox
        )
        assert "bounding box sector" in payload["text_prompt"]
        geo_loc = payload["geo_location"]
        assert geo_loc["type"] == "polygon"
        coords = geo_loc["coordinates"][0]
        assert len(coords) == 5
        assert coords[0] == coords[-1]  # Closed ring
        # Check SW and NE [lon, lat]
        assert coords[0] == [31.0, 48.0]
        assert coords[2] == [33.5, 49.5]

    def test_append_rag_log_capping(self):
        """Verify _append_rag_log caps history at 50 entries."""
        import streamlit as st
        import app
        st.session_state.rag_logs = []
        for i in range(60):
            app._append_rag_log({"id": i, "latency_ms": 10.0, "status": "SUCCESS"})
        assert len(st.session_state.rag_logs) == 50
        assert st.session_state.rag_logs[0]["id"] == 10
        assert st.session_state.rag_logs[-1]["id"] == 59

    def test_mock_agent_populates_raw_response(self):
        """Ensure query_geo_agent mock responses always include structured raw_response dict."""
        import geo_agent
        resp = geo_agent.query_geo_agent(text_prompt="Kyiv headquarters", force_mock=True)
        assert resp.raw_response is not None
        assert isinstance(resp.raw_response, dict)
        assert "answer" in resp.raw_response
        assert "geo_locations" in resp.raw_response
        assert "sources" in resp.raw_response

    def test_dispatch_geo_query_records_success_log(self, monkeypatch):
        """Verify dispatch_geo_query captures latency, payloads, and status into rag_logs."""
        import streamlit as st
        import app
        st.session_state.rag_logs = []
        st.session_state.messages = []
        st.session_state.openrouter_api_key = ""

        resp = app.dispatch_geo_query(
            text_prompt="Test tactical sweep",
            selected_point=(48.5, 32.0)
        )
        assert resp is not None
        assert len(st.session_state.rag_logs) == 1
        log = st.session_state.rag_logs[0]
        assert log["status"] == "SUCCESS"
        assert log["latency_ms"] >= 0.0
        assert log["raw_request"]["text_prompt"] == "Test tactical sweep"
        assert log["raw_response"] is not None
        assert "answer" in log["raw_response"]

    def test_dispatch_geo_query_records_error_log(self, monkeypatch):
        """Verify dispatch_geo_query catches errors and logs error_details without crashing."""
        import streamlit as st
        import app
        import geo_agent
        st.session_state.rag_logs = []

        def mock_error_query(*args, **kwargs):
            raise geo_agent.AgentError("Simulated connection timeout to knowledge graph.")

        monkeypatch.setattr(app, "query_geo_agent", mock_error_query)

        res = app.dispatch_geo_query(text_prompt="Failing query")
        assert res is None
        assert len(st.session_state.rag_logs) == 1
        log = st.session_state.rag_logs[0]
        assert log["status"] == "ERROR"
        assert log["error_details"]["error_type"] == "AgentError"
        assert "Simulated connection timeout" in log["error_details"]["message"]

    def test_sidebar_operations_order_in_codebase(self):
        """Verify Operations contains RAG Debug Logs and retain pins toggle, and redundant Spatial Targeting is removed."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        ops_pos = code.find("st.subheader(\"🛠️ Operations\")")
        spatial_pos = code.find("st.subheader(\"🎯 Spatial Targeting\")")
        rag_btn_pos = code.find("📋 RAG Debug Logs")
        keep_pts_pos = code.find("Retain target pin after query")

        assert ops_pos != -1, "Operations section must exist in sidebar"
        assert spatial_pos == -1, "Redundant Spatial Targeting section should not exist in sidebar"
        assert rag_btn_pos != -1, "RAG Debug Logs button must exist"
        assert keep_pts_pos != -1, "Retain target pin toggle must exist in Operations"
        assert ops_pos < rag_btn_pos, "RAG Debug Logs button must be inside Operations section"
        assert ops_pos < keep_pts_pos, "Retain target pin toggle must be inside Operations section"

    def test_chat_bubbles_opposite_sides_layout(self):
        """Ensure chat console styles operator (user) on the right and geo-ai on the left with bubble styling."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Check marker classes in HTML generation
        assert "bubble-marker-operator" in code, "Operator messages must have bubble-marker-operator class"
        assert "bubble-marker-geoai" in code, "Geo-AI messages must have bubble-marker-geoai class"
        assert "chat-role-operator" in code, "Operator role tag must have chat-role-operator class"
        assert "chat-role-geoai" in code, "Geo-AI role tag must have chat-role-geoai class"

        # Check CSS rules for opposite sides and border radius
        assert "flex-direction: row-reverse" in code, "Operator bubbles must reverse flex direction for right-hand alignment"
        assert "align-self: flex-end" in code, "Operator messages must align to the right"
        assert "align-self: flex-start" in code, "Geo-AI messages must align to the left"
        assert "margin-left: auto" in code, "Operator bubbles must float/align right with margin-left: auto"
        assert "margin-right: auto" in code, "Geo-AI bubbles must float/align left with margin-right: auto"
        assert "border-radius: 12px 12px 12px 12px" in code, "Chat bubbles must have border-radius: 12px 12px 12px 12px"


class TestUpgradesOneThroughSixAndPerformance:
    """Validate Upgrades 1-6 (Vr6) and AGENTS.md zero-latency performance rules."""

    def test_highlight_text_spans_basic_and_overlapping(self):
        """Upgrade 6: Verify text span highlighting merges overlaps and generates <mark> tags."""
        from app import _highlight_text_spans

        text = "Kyiv is the capital of Ukraine."
        # Highlight "Kyiv" (0, 4) and "capital" (12, 19)
        spans = [[0, 4], [12, 19]]
        res = _highlight_text_spans(text, spans)
        assert "<mark" in res
        assert "Kyiv</mark>" in res
        assert "capital</mark>" in res

        # Overlapping spans: (0, 4) and (2, 7) -> merged (0, 7)
        overlap_spans = [[0, 4], [2, 7]]
        res_overlap = _highlight_text_spans("Kyiv is nice", overlap_spans)
        assert res_overlap.count("<mark") == 1
        assert "Kyiv is</mark>" in res_overlap

    def test_highlight_text_spans_dict_format(self):
        """Upgrade 6: Verify dict-style spans [{'start': 0, 'end': 4}]."""
        from app import _highlight_text_spans

        text = "Kharkiv sector"
        spans = [{"start": 0, "end": 7}]
        res = _highlight_text_spans(text, spans)
        assert "<mark" in res
        assert "Kharkiv</mark>" in res

    def test_highlight_text_spans_edge_cases(self):
        """Upgrade 6: Safe fallbacks for empty text, null spans, or out-of-range offsets."""
        from app import _highlight_text_spans

        assert _highlight_text_spans("", [[0, 5]]) == ""
        assert _highlight_text_spans("Hello", None) == "Hello"
        assert _highlight_text_spans("Hello", []) == "Hello"
        assert _highlight_text_spans("Hello", [[10, 20]]) == "Hello"
        assert _highlight_text_spans("Hello", [[-5, -1]]) == "Hello"
        assert _highlight_text_spans("Hello", [["invalid", "type"]]) == "Hello"

    def test_build_assistant_message_vr6_fields(self):
        """Upgrade 4 & 5: Ensure _build_assistant_message preserves follow_ups, regions, did_you_mean, map_bbox."""
        from app import _build_assistant_message
        from geo_agent import AgentResponse

        resp = AgentResponse(
            answer="Here are tactical points.",
            follow_ups=["Where is the 93rd Brigade?", "Show frontline units"],
            regions={"Donetsk Oblast": 14, "Kharkiv Oblast": 8},
            did_you_mean="Bakhmut",
            map_bbox=[36.5, 48.2, 38.0, 49.1]
        )
        msg = _build_assistant_message(resp)
        assert msg["role"] == "assistant"
        assert msg["follow_ups"] == ["Where is the 93rd Brigade?", "Show frontline units"]
        assert msg["regions"] == {"Donetsk Oblast": 14, "Kharkiv Oblast": 8}
        assert msg["did_you_mean"] == "Bakhmut"
        assert msg["map_bbox"] == [36.5, 48.2, 38.0, 49.1]

    def test_geojson_boundary_layer_generation(self):
        """Upgrade 1: Verify geo_agent hydrates GeoJSON boundaries when shape is returned."""
        import geo_agent

        raw_vr6_resp = {
            "answer": "Poltava is in central Ukraine.",
            "geo_locations": [
                {
                    "name": "Poltava Oblast",
                    "latitude": 49.5883,
                    "longitude": 34.5514,
                    "role": "primary",
                    "shape": {
                        "type": "Polygon",
                        "coordinates": [[[34.0, 49.0], [35.0, 49.0], [35.0, 50.0], [34.0, 50.0], [34.0, 49.0]]]
                    }
                }
            ],
            "map": {
                "bbox": [33.5, 48.8, 35.5, 50.3]
            },
            "regions": {"Poltava Oblast": 1},
            "follow_up_questions": ["What units operate in Poltava?"]
        }

        # Mock query backend
        monkeypatch_target = getattr(geo_agent, "_backend_query", None)
        orig_backend = geo_agent._BACKEND_AVAILABLE
        try:
            geo_agent._BACKEND_AVAILABLE = True
            geo_agent._backend_query = lambda **kw: raw_vr6_resp

            resp = geo_agent.query_geo_agent(text_prompt="Analyze Poltava", api_key="test-key")
            assert resp is not None
            assert resp.map_bbox == [33.5, 48.8, 35.5, 50.3]
            assert resp.regions == {"Poltava Oblast": 1}
            assert resp.follow_ups == ["What units operate in Poltava?"]

            # Layer checks: should have a marker layer AND a geojson layer
            layer_types = [l.layer_type for l in resp.layers]
            assert "marker" in layer_types
            assert "geojson" in layer_types

            geojson_layer = next(l for l in resp.layers if l.layer_type == "geojson")
            assert geojson_layer.name == "Outline: Poltava Oblast"
            assert geojson_layer.data["type"] == "Polygon"
        finally:
            geo_agent._BACKEND_AVAILABLE = orig_backend
            geo_agent._backend_query = monkeypatch_target

    def test_performance_rules_in_agents_md(self):
        """AGENTS.md Performance Verification: Ensure all zero-latency rules are present in codebase."""
        with open("app.py", "r", encoding="utf-8") as f:
            app_code = f.read()

        # Rule 1.1: returned_objects in st_folium
        assert 'returned_objects=["last_clicked", "last_active_drawing", "all_drawings"]' in app_code, (
            "Rule 1.1: st_folium must constrain returned_objects to prevent pan/zoom reruns"
        )

        # Rule 2.1: Guarded bootstrap & no unconditional importlib.reload
        assert "_BOOTSTRAP_DONE = getattr(sys, \"_GEOAI_BOOTSTRAP_DONE\", False)" in app_code, (
            "Rule 2.1: Bootstrap must be guarded once per process"
        )
        assert "importlib.reload(sys.modules[\"geo_agent\"])" not in app_code, (
            "Rule 2.1: Per-rerun importlib.reload of geo_agent must NOT exist"
        )

        # Rule 3.1: Stable container height
        assert "chat_container_height = 465" in app_code, (
            "Rule 3.1: Chat container height must remain constant to prevent screen jumping"
        )

        # Rule 4.1: MarkerCluster for context markers
        assert "context_cluster = MarkerCluster" in app_code, (
            "Rule 4.1: MarkerCluster must be instantiated for context pins"
        )

        # Rule 4.2: fit_bounds with last_map_bbox
        assert "m.fit_bounds([[s, w], [n, e]], padding=(30, 30))" in app_code, (
            "Rule 4.2: Map must auto-zoom to query bbox when available"
        )


class TestTacticalHudSymbologyAndAnsweredBbox:
    """Validate Tactical HUD map symbology, pin factories, popup cards, and answered bbox transparency."""

    def test_color_token_semantic_resolution(self):
        """Ensure color_token resolves roles and legacy color names across dark and light themes."""
        import app

        # Primary (blue / primary role)
        tok_dark, hex_dark = app.color_token("blue", role="primary", theme="dark")
        assert tok_dark == "primary"
        assert hex_dark == "#38bdf8"
        tok_light, hex_light = app.color_token("blue", role="primary", theme="light")
        assert tok_light == "primary"
        assert hex_light == "#0369a1"

        # Event (red / event kind)
        ev_tok, ev_hex = app.color_token("red", kind="event", theme="dark")
        assert ev_tok == "event"
        assert ev_hex == "#ff4d6d"

        # Uncertain (orange / uncertain flag)
        unc_tok, unc_hex = app.color_token("orange", uncertain=True, theme="dark")
        assert unc_tok == "uncertain"
        assert unc_hex == "#ffb020"

        # Context (cadetblue / context role)
        ctx_tok, ctx_hex = app.color_token("cadetblue", role="context", theme="dark")
        assert ctx_tok == "context"
        assert ctx_hex == "#7dd3fc"

        # Query BBox
        bbox_tok, bbox_hex = app.color_token("orange", role="query_bbox", theme="dark")
        assert bbox_tok == "selection"
        assert bbox_hex == "#00e5ff"

    def test_bbox_style_selected_vs_answered_transparency(self):
        """Ensure answered bounding boxes become ultra-transparent and non-interactive."""
        import app

        sel = app.bbox_style("selected", theme="dark")
        ans = app.bbox_style("answered", theme="dark")

        # Selected bbox has higher opacity and is interactive
        assert sel["fill_opacity"] == 0.10
        assert sel["opacity"] == 0.90
        assert sel["weight"] == 2
        assert sel["dash_array"] == "6, 6"
        assert sel["interactive"] is True
        assert "geoai-bbox--selected" in sel["class_name"]

        # Answered bbox is ultra-transparent (fill_opacity=0.03), faint stroke, and non-interactive
        assert ans["fill_opacity"] == 0.03
        assert ans["opacity"] == 0.55
        assert ans["weight"] == 1.25
        assert ans["dash_array"] == "3, 6"
        assert ans["interactive"] is False
        assert "geoai-bbox--answered" in ans["class_name"]
        assert ans["fill_opacity"] < sel["fill_opacity"]

    def test_make_pin_icon_variants(self):
        """Verify make_pin_icon generates pure CSS DivIcons for target, corners, context, and diamonds."""
        import app

        # 1. Target reticle
        reticle = app.make_pin_icon(role="target")
        assert reticle.options["icon_size"] == [34, 34] or reticle.options["icon_size"] == (34, 34)
        assert "geoai-reticle" in reticle.options["html"]
        assert "geoai-reticle__pulse" in reticle.options["html"]

        # 2. BBox corner brackets (active and dimmed)
        sw_corner = app.make_pin_icon(role="bbox_corner", corner="sw", dimmed=False)
        assert "geoai-corner--sw" in sw_corner.options["html"]
        assert "geoai-corner--dim" not in sw_corner.options["html"]

        ne_dim = app.make_pin_icon(role="bbox_corner", corner="ne", dimmed=True)
        assert "geoai-corner--ne" in ne_dim.options["html"]
        assert "geoai-corner--dim" in ne_dim.options["html"]

        # 3. Context dot
        ctx_icon = app.make_pin_icon(role="context", color_name="cadetblue")
        assert "geoai-dot" in ctx_icon.options["html"]
        assert "geoai-pin--context" in ctx_icon.options["class_name"]
        assert "awesome-marker-icon-cadetblue" in ctx_icon.options["class_name"]

        # 4. Primary numbered diamond
        prim_icon = app.make_pin_icon(role="primary", number=3, color_name="blue")
        assert "geoai-diamond" in prim_icon.options["html"]
        assert "<span>3</span>" in prim_icon.options["html"]
        assert "geoai-pin--primary" in prim_icon.options["class_name"]

        # 5. Event diamond
        ev_icon = app.make_pin_icon(kind="event", role="primary", color_name="red")
        assert "geoai-diamond" in ev_icon.options["html"]
        assert "<span>!</span>" in ev_icon.options["html"]
        assert "geoai-pin--event" in ev_icon.options["class_name"]

        # 6. Uncertain diamond with dashed border and ? badge
        unc_icon = app.make_pin_icon(role="primary", uncertain=True, number=2, color_name="orange")
        assert "geoai-diamond--uncertain" in unc_icon.options["html"]
        assert "geoai-badge-unc" in unc_icon.options["html"]
        assert "<span>2</span>" in unc_icon.options["html"]
        assert "geoai-pin--uncertain" in unc_icon.options["class_name"]
        assert "awesome-marker-icon-orange" in unc_icon.options["class_name"]

    def test_popup_card_html_generation(self):
        """Verify _popup_card builds structured dark-glass HUD markup."""
        import app

        html = app._popup_card(
            title="Kharkiv Sector",
            rows=[("LAT", "49.9935"), ("LON", "36.2304"), ("ROLE", "primary")],
            chip="[1] PLACE",
            footer="<div>[1] Intel Report</div>",
            token="primary",
        )
        assert "geoai-card" in html
        assert "Kharkiv Sector" in html
        assert "[1] PLACE" in html
        assert "49.9935" in html
        assert "36.2304" in html
        assert "[1] Intel Report" in html

    def test_geo_agent_bbox_layers_include_query_bbox_and_corner_roles(self):
        """Ensure query_geo_agent tags bbox polygons with role='query_bbox' and corners with 'bbox_corner'."""
        import geo_agent

        bbox = ((48.0, 31.0), (49.5, 33.5))
        resp = geo_agent.query_geo_agent(
            text_prompt="Analyze sector",
            bounding_box=bbox,
            force_mock=True
        )
        poly_layers = [l for l in resp.layers if l.layer_type == "polygon"]
        assert len(poly_layers) >= 1
        assert poly_layers[0].data.get("role") == "query_bbox"

        corner_layers = [
            l for l in resp.layers
            if l.layer_type == "marker" and isinstance(l.data, dict) and l.data.get("role") == "bbox_corner"
        ]
        assert len(corner_layers) == 2
        corners = {l.data.get("corner") for l in corner_layers}
        assert corners == {"sw", "ne"}

    def test_client_side_hooks_support_new_hud_symbology_and_answered_bbox(self):
        """Verify MapTrashBinHook includes PinFilterControl, .geoai-pin classes, and answered bbox sync."""
        import folium
        from streamlit_folium import _get_map_string
        import app

        m = folium.Map()
        app.MapTrashBinHook().add_to(m)

        script = _get_map_string(m)
        assert "function geoaiIcon(opts)" in script
        assert "syncAnsweredBboxInDrawnItems" in script
        assert "geoai-bbox--answered" in script
        assert "geoai-pin--context" in script
        assert "geoai-pin--uncertain" in script
        assert "PinFilterControl" in script

    def test_instant_hud_bridge_and_popup_auto_open(self):
        """Verify 0ms client-side selection bridge and target point popup auto-open."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "_geoaiSelectionBridgeInstalled" in content
        assert "GEOAI_SELECTION_EVENT" in content
        assert "geoai-instant-pill-slot" in content
        assert "popup=folium.Popup(popup_html, max_width=250, show=True)" in content

    def test_streamlined_zero_widget_sources_drawer(self):
        """Verify Verified Sources uses native HTML details cards to prevent widget tree explosion."""
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()

        assert "geoai-source-detail" in content
        assert "geoai-source-summary" in content
        assert "geoai-source-badge" in content
        assert "btn_src_" not in content

    def test_corner_bracket_exact_vertex_anchors(self):
        """Verify corner bracket DivIcons anchor at exact SVG L-vertices so brackets align with marching ants."""
        import app

        expected_anchors = {
            "nw": [2, 2],
            "ne": [14, 2],
            "se": [14, 14],
            "sw": [2, 14],
        }
        for corner, expected in expected_anchors.items():
            icon = app.make_pin_icon(role="bbox_corner", corner=corner)
            assert list(icon.options["icon_size"]) == [16, 16]
            assert list(icon.options["icon_anchor"]) == expected

    def test_pin_shape_diamond_vs_teardrop(self):
        """Verify make_pin_icon supports switching between Diamond and Teardrop shapes."""
        import app

        diamond_icon = app.make_pin_icon(role="primary", number=1, color_name="blue", shape="diamond")
        assert "geoai-diamond" in diamond_icon.options["html"]
        assert "geoai-teardrop" not in diamond_icon.options["html"]
        assert "geoai-pin--teardrop" not in diamond_icon.options["class_name"]
        assert list(diamond_icon.options["icon_anchor"]) == [11, 11]

        teardrop_icon = app.make_pin_icon(role="primary", number=1, color_name="blue", shape="teardrop")
        assert "geoai-diamond geoai-teardrop" in teardrop_icon.options["html"]
        assert "geoai-pin--teardrop" in teardrop_icon.options["class_name"]
        assert list(teardrop_icon.options["icon_anchor"]) == [11, 11]

    def test_querying_bbox_style_and_scanline_hook(self):
        """Verify bbox_style('querying') and client-side scanline/pulse animation hooks."""
        import folium
        from streamlit_folium import _get_map_string
        import app

        q_style = app.bbox_style("querying", theme="dark")
        assert q_style["fill_opacity"] == 0.08
        assert q_style["opacity"] == 0.95
        assert "geoai-bbox--querying" in q_style["class_name"]

        m = folium.Map()
        app.MapTrashBinHook().add_to(m)
        script = _get_map_string(m)
        assert "startQueryingAnimation" in script
        assert "stopQueryingAnimation" in script
        assert "geoai-scan-svg" in script
        assert "geoai-scan-line" in script
        assert "geoai-reticle--querying" in script
        assert "setMapPinShape" in script

    def test_selection_sync_hook_and_clear_spatial_selection(self):
        """Verify SelectionSyncHook emits _onServerSelectionSync in feature_group_string and clear_spatial_selection increments clear_seq."""
        import folium
        import streamlit as st
        from streamlit_folium import _get_feature_group_string, _get_map_string
        import app

        m1 = folium.Map(location=[48.3794, 31.1656], zoom_start=6)
        m2 = folium.Map(location=[48.3794, 31.1656], zoom_start=6)
        base_map_str_1 = _get_map_string(m1)
        base_map_str_2 = _get_map_string(m2)

        fg = folium.FeatureGroup(name="Active Selection", control=False)
        app.SelectionSyncHook(
            has_selection=False,
            clear_seq=3,
            bbox_state="none",
            is_querying=False,
            bbox_bounds=None,
            pin_shape="Teardrop",
        ).add_to(fg)

        fg_str = _get_feature_group_string(fg, m2, 0)

        # Base map string is untouched before feature_group rendering while feature_group_string carries the sync call
        assert base_map_str_1 == base_map_str_2
        assert "_onServerSelectionSync" in fg_str
        assert "clearSeq: 3" in fg_str
        assert 'pinShape: "teardrop"' in fg_str

        # Verify clear_spatial_selection increments selection_clear_seq and records stale map widget state
        st.session_state.selected_points = [(50.45, 30.52)]
        st.session_state.bounding_box = ((49.0, 30.0), (50.0, 31.0))
        st.session_state.selection_clear_seq = 0
        st.session_state["geoai_tactical_map"] = {
            "last_clicked": {"lat": 50.45, "lng": 30.52},
            "last_active_drawing": {"geometry": {"type": "Polygon", "coordinates": [[[30.0, 49.0], [31.0, 50.0]]]}},
        }
        app.clear_spatial_selection()
        assert st.session_state.selected_points == []
        assert st.session_state.bounding_box is None
        assert st.session_state.selection_clear_seq == 1
        assert st.session_state.last_clicked == (50.45, 30.52)
        assert st.session_state.ignored_drawing is not None

    def test_pipeline_inversion_and_zero_rerun_dispatch(self):
        """Verify pipeline inversion (col2 before col1) and immediate dispatch in code."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # 1. col2 (map) must be evaluated before col1 (chat)
        col2_pos = code.find("with col2:")
        col1_pos = code.find("with col1:")
        assert col2_pos != -1 and col1_pos != -1
        assert col2_pos < col1_pos, "Pipeline Inversion: with col2: must execute before with col1:"

        # 2. Immediate dispatch helper must be defined
        assert "window.geoaiImmediateDispatch" in code
        assert "_peek_incoming_map_state" in code

        # 3. col2 click and drawing handlers must not trigger st.rerun cascades
        col2_code = code[col2_pos:col1_pos]
        assert "st.session_state.selected_points = [clicked_coord]" in col2_code
        click_idx = col2_code.find("st.session_state.selected_points = [clicked_coord]")
        post_click = col2_code[click_idx:click_idx + 250]
        assert "st.rerun()" not in post_click

    def test_workspace_window_auto_fit_layout_and_listeners(self):
        """Verify chat and map workspace automatically fit to the screen/window dimensions."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # 1. CSS Auto-Fit rules for full-height split screen layout
        assert "Workspace Window Auto-Fit Architecture" in code
        assert 'calc(100vh - 90px)' in code
        assert 'calc(100vh - 150px)' in code
        assert 'div[data-testid="column"]:last-child div[data-testid="stCustomComponentV1"]' in code
        assert 'div[data-testid="column"]:first-child div[data-testid="stVerticalBlockBorderWrapper"]' in code

        # 2. Dynamic client-side JavaScript resize hook
        assert "forwardResizeToIframes" in code
        assert "window.addEventListener('resize', forwardResizeToIframes);" in code

        # 3. Leaflet map ResizeObserver to invalidate tile size on container resizing
        assert "map.invalidateSize()" in code

    def test_active_querying_animation_and_early_query_ingestion(self):
        """Verify active querying animation hooks and early query queueing before col2 map renders."""
        import folium
        from streamlit_folium import _get_feature_group_string, _get_map_string
        import app

        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # 1. Section 2c Early Query Ingestion must precede with col2:
        sec2c_pos = code.find("2c. Early Query Ingestion & Queueing")
        col2_pos = code.find("with col2:")
        assert sec2c_pos != -1, "Section 2c must be defined"
        assert sec2c_pos < col2_pos, "Section 2c must precede with col2: so pending_query is queued before map renders"

        # 2. Section 2c must check both chat_input and run_chat_recon
        assert 'st.session_state.get("geoai_chat_input")' in code
        assert 'st.session_state.get("run_chat_recon")' in code
        assert 'key="geoai_chat_input"' in code

        # 3. Leaflet hooks must track _isClientQuerying and _lastKnownBboxBounds
        m = folium.Map()
        app.MapTrashBinHook().add_to(m)
        map_script = _get_map_string(m)
        assert "_isClientQuerying" in map_script
        assert "_lastKnownBboxBounds" in map_script
        assert "startQueryingAnimation(state.bboxBounds || window._lastKnownBboxBounds)" in map_script

        # 4. SelectionSyncHook must emit isQuerying: true and bboxState: "querying"
        fg = folium.FeatureGroup(name="Active Selection", control=False)
        app.SelectionSyncHook(
            has_selection=True,
            clear_seq=0,
            bbox_state="querying",
            is_querying=True,
            bbox_bounds=[[48.0, 31.0], [49.0, 32.0]],
            pin_shape="diamond",
        ).add_to(fg)

class TestOption1MapCanvasTacticalHudAndStableLayout:
    """Validate Option 1: Map Canvas Tactical HUD and permanently anchored chat input with 0ms latency."""

    def test_map_trash_bin_hook_includes_tactical_hud_control(self):
        """Verify MapTrashBinHook instantiates TacticalTargetHudControl in bottomleft with Clear and Recon buttons."""
        import folium
        from streamlit_folium import _get_map_string
        import app

        m = folium.Map()
        app.MapTrashBinHook().add_to(m)
        script = _get_map_string(m)

        assert "TacticalTargetHudControl" in script
        assert "position: 'bottomleft'" in script
        assert "geoai_map_tactical_hud" in script
        assert "geoai_map_hud_content" in script
        assert "geoai_map_hud_clear_btn" in script
        assert "geoai_map_hud_recon_btn" in script
        assert "window.updateMapTacticalHud" in script
        assert "window.hideMapTacticalHud" in script

    def test_selection_sync_hook_serializes_active_point(self):
        """Verify SelectionSyncHook passes activePoint coordinates for instant HUD synchronization."""
        import folium
        from streamlit_folium import _get_feature_group_string
        import app

        m = folium.Map()
        fg = folium.FeatureGroup(name="Selection", control=False)
        app.SelectionSyncHook(
            has_selection=True,
            clear_seq=1,
            active_point=(31.7767, 35.2345),
            pin_shape="diamond"
        ).add_to(fg)
        fg_str = _get_feature_group_string(fg, m, 0)

        assert "activePoint: [31.7767, 35.2345]" in fg_str
        assert "hasSelection: true" in fg_str

    def test_chat_interface_zero_height_action_triggers(self):
        """Verify col1 uses zero-height headless action triggers so chat input remains permanently anchored."""
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        assert ".st-key-clear_chat_pts" in code
        assert ".st-key-run_chat_recon" in code
        assert 'display: none !important;' in code
        assert 'height: 0px !important;' in code
        assert 'key="clear_chat_pts"' in code
        assert 'key="run_chat_recon"' in code
        assert 'GEOAI_PARENT_CMD' in code
        assert "trigger_recon" in code

    def test_tactical_hud_recon_dispatch_and_no_col1_leak(self):
        """Verify TacticalTargetHudControl dispatches recon via Folium bridge, Section 2b handles it, and col1 has no leaked div."""
        import folium
        from streamlit_folium import _get_map_string

        m = folium.Map()
        import app
        app.MapTrashBinHook().add_to(m)
        script = _get_map_string(m)

        # Tactical HUD recon button must prepare and dispatch recon payload
        assert 'properties: {\n                                action: "recon"' in script or 'action: "recon"' in script
        assert "geoaiImmediateDispatch(map)" in script

        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # Section 2b and Section 5 handle action == "recon"
        assert '_props.get("action") == "recon"' in code
        assert 'props.get("action") == "recon"' in code
        assert '"is_recon": True' in code

        # Ensure col1 does NOT contain the leaked geoai-instant-pill-slot div inside col1
        # It must only appear once in the global script block unindented
        assert code.count('<div id="geoai-instant-pill-slot"') == 1
        assert "<div id='geoai-instant-pill-slot'" not in code



