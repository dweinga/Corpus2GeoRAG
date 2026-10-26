"""
==============================================================================
GeoAI Final Project - Interactive Streamlit Frontend Application
==============================================================================
This application serves as the interactive frontend for the GeoAI spatial
intelligence system. It connects a conversational chat interface with an
interactive Folium map centered over Ukraine, demonstrating bi-directional
synchronization between map clicks, point selection/removal, and RAG agent queries.

Key Features:
    1. Self-bootstrapping dependency verification for cross-system portability.
    2. Dynamic, adjustable split-screen layout (Chat & Map).
    3. Multi-modal RAG routing via `query_geo_agent` (text only, point only, both).
    4. Decoupled point selection & state management:
       - Map click -> selects coordinate(s) and updates `selected_points` WITHOUT immediate backend calls.
       - Point removal -> remove individual selected points or clear all via UI controls.
       - Chat prompt -> pairs user inquiry with selected coordinate(s).
    5. Real-time Folium visualization of tactical markers, circles, and buffers.
    6. GeoAI Tactical Defense C2 Protocol Styling (electric cyan & dark slate HUD).

Team & Deployment Notice:
    This file is completely self-contained and runnable on any OS (Windows, macOS,
    Linux). Running `streamlit run app.py` or double-clicking `run_app.bat` will
    launch the app. Direct execution via `python app.py` will automatically forward
    to the Streamlit runtime.
==============================================================================
"""

import sys
import time
import json
from datetime import datetime
import subprocess
import importlib.util
from typing import Optional, Tuple, List, Dict, Any

# ==============================================================================
# 0. In-Code Environment & Dependency Bootstrap
# ==============================================================================
# Ensures that any coworker cloning this repository on any operating system has
# the necessary libraries installed without manual intervention.

REQUIRED_DEPENDENCIES = {
    "streamlit": "streamlit>=1.30.0",
    "folium": "folium>=0.15.0",
    "streamlit_folium": "streamlit-folium>=0.18.0",
    "streamlit_adjustable_columns": "streamlit-adjustable-columns>=0.2.0",
    "pytest": "pytest>=7.0.0",
    "pydantic": "pydantic>=2.7,<3",
    "networkx": "networkx>=3.4,<4",
    "langchain_core": "langchain-core>=1.0,<2",
    "langchain_openai": "langchain-openai>=1.0,<2",
    "openai": "openai",
    "dotenv": "python-dotenv>=1.0,<2",
}

def verify_and_install_dependencies():
    """Verify presence of core packages; automatically install missing ones."""
    missing_packages = []
    for module_name, pip_spec in REQUIRED_DEPENDENCIES.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(pip_spec)

    if missing_packages:
        print(f"[GeoAI Bootstrap] Missing dependencies detected: {missing_packages}")
        print("[GeoAI Bootstrap] Installing required packages across system environment...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing_packages
            ])
            print("[GeoAI Bootstrap] All dependencies installed successfully.")
        except Exception as exc:
            print(f"[GeoAI Bootstrap] Warning: Automated dependency installation encountered: {exc}")

# Execute environment check strictly once per process to prevent per-rerun latency
_BOOTSTRAP_DONE = getattr(sys, "_GEOAI_BOOTSTRAP_DONE", False)
if not _BOOTSTRAP_DONE:
    verify_and_install_dependencies()
    sys._GEOAI_BOOTSTRAP_DONE = True

# Core Streamlit and mapping imports
import streamlit as st
import folium
from folium.plugins import Draw, MarkerCluster
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium

# Layout import with graceful fallback
try:
    from streamlit_adjustable_columns import adjustable_columns
except ImportError:
    # Fallback to standard st.columns if adjustable component fails to compile on any OS
    def adjustable_columns(spec, **kwargs):
        return st.columns(spec)

# Import backend contracts, routing engine, utilities, and errors
import geo_agent
from geo_agent import (
    MapLayer,
    AgentResponse,
    query_geo_agent,
    to_bounding_box,
)

# Ensure get_source_text and AgentError are safely resolved even across hot-reloads
get_source_text = getattr(geo_agent, "get_source_text", lambda pointer: None)
AgentError = getattr(geo_agent, "AgentError", RuntimeError)

def _build_raw_request_payload(
    text_prompt: Optional[str] = None,
    selected_point: Optional[Tuple[float, float]] = None,
    bounding_box: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    agent_turns: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Reconstruct the exact QueryRequest JSON sent across the backend boundary."""
    geo_location = None
    resolved_text = (text_prompt or "").strip()
    if bounding_box and len(bounding_box) == 2:
        (min_lat, min_lng), (max_lat, max_lng) = to_bounding_box(bounding_box[0], bounding_box[1])
        geo_location = {
            "type": "polygon",
            "coordinates": [[[min_lng, min_lat], [max_lng, min_lat], [max_lng, max_lat], [min_lng, max_lat], [min_lng, min_lat]]]
        }
        if not resolved_text:
            resolved_text = "What is in this bounding box sector?"
    elif selected_point:
        geo_location = {
            "type": "point",
            "coordinates": [round(float(selected_point[1]), 5), round(float(selected_point[0]), 5)],
            "radius_m": 1000.0
        }
        if not resolved_text:
            resolved_text = "What is in this area?"
    elif not resolved_text:
        resolved_text = "What are the major locations in Ukraine?"

    return {
        "text_prompt": resolved_text,
        "geo_location": geo_location,
        "chat_history": [
            {"query": t.get("query"), "response": t.get("response")}
            for t in (agent_turns or [])
        ]
    }


def _append_rag_log(entry: Dict[str, Any]):
    """Record RAG communication log in session state, capped to the most recent 50 turns."""
    if "rag_logs" not in st.session_state:
        st.session_state.rag_logs = []
    st.session_state.rag_logs.append(entry)
    if len(st.session_state.rag_logs) > 50:
        st.session_state.rag_logs = st.session_state.rag_logs[-50:]


def dispatch_geo_query(
    text_prompt: Optional[str] = None,
    selected_point: Optional[Tuple[float, float]] = None,
    bounding_box: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
) -> Optional[AgentResponse]:
    """Execute query_geo_agent with error handling, session state persistence, telemetry, and logging."""
    api_key = st.session_state.get("openrouter_api_key") or None
    agent_turns = st.session_state.get("agent_turns", [])
    raw_request_payload = _build_raw_request_payload(
        text_prompt=text_prompt,
        selected_point=selected_point,
        bounding_box=bounding_box,
        agent_turns=agent_turns
    )
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.perf_counter()

    try:
        with st.spinner("🛰️ Querying GeoRAG Agent..."):
            agent_resp: AgentResponse = query_geo_agent(
                text_prompt=text_prompt,
                selected_point=selected_point,
                bounding_box=bounding_box,
                chat_history=st.session_state.messages,
                api_key=api_key,
                agent_turns=agent_turns
            )
            latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

            # Retain backend turns for conversation continuity (including map recon scans)
            resolved_query_text = text_prompt or raw_request_payload.get("text_prompt", "")
            if agent_resp and agent_resp.raw_response and resolved_query_text:
                st.session_state.agent_turns.append({
                    "query": resolved_query_text,
                    "response": agent_resp.raw_response
                })
                if len(st.session_state.agent_turns) > 5:
                    st.session_state.agent_turns = st.session_state.agent_turns[-5:]

            log_entry = {
                "id": len(st.session_state.get("rag_logs", [])) + 1,
                "timestamp": timestamp_str,
                "latency_ms": latency_ms,
                "status": "SUCCESS",
                "is_live_backend": bool(getattr(agent_resp, "is_live_backend", False)),
                "query_summary": text_prompt or ("BBox Recon" if bounding_box else ("Point Recon" if selected_point else "Recon Scan")),
                "raw_request": raw_request_payload,
                "raw_response": getattr(agent_resp, "raw_response", None) or {
                    "answer": getattr(agent_resp, "answer", ""),
                    "geo_locations": [],
                    "sources": getattr(agent_resp, "sources", [])
                },
                "layers_count": len(getattr(agent_resp, "layers", [])),
                "sources": list(getattr(agent_resp, "sources", [])),
                "citations": list(getattr(agent_resp, "citations", [])),
                "spatial_input": {
                    "type": "polygon" if bounding_box else ("point" if selected_point else "none"),
                    "selected_point": selected_point,
                    "bounding_box": bounding_box,
                },
                "error_details": None
            }
            _append_rag_log(log_entry)
            return agent_resp
    except AgentError as err:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        log_entry = {
            "id": len(st.session_state.get("rag_logs", [])) + 1,
            "timestamp": timestamp_str,
            "latency_ms": latency_ms,
            "status": "ERROR",
            "is_live_backend": bool(api_key),
            "query_summary": text_prompt or ("BBox Recon" if bounding_box else ("Point Recon" if selected_point else "Recon Scan")),
            "raw_request": raw_request_payload,
            "raw_response": None,
            "layers_count": 0,
            "sources": [],
            "spatial_input": {
                "type": "polygon" if bounding_box else ("point" if selected_point else "none"),
                "selected_point": selected_point,
                "bounding_box": bounding_box,
            },
            "error_details": {
                "error_type": type(err).__name__,
                "message": str(err)
            }
        }
        _append_rag_log(log_entry)
        st.error(f"⚠️ **GeoRAG Agent Error:** {err}")
        return None
    except Exception as err:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        log_entry = {
            "id": len(st.session_state.get("rag_logs", [])) + 1,
            "timestamp": timestamp_str,
            "latency_ms": latency_ms,
            "status": "ERROR",
            "is_live_backend": bool(api_key),
            "query_summary": text_prompt or ("BBox Recon" if bounding_box else ("Point Recon" if selected_point else "Recon Scan")),
            "raw_request": raw_request_payload,
            "raw_response": None,
            "layers_count": 0,
            "sources": [],
            "spatial_input": {
                "type": "polygon" if bounding_box else ("point" if selected_point else "none"),
                "selected_point": selected_point,
                "bounding_box": bounding_box,
            },
            "error_details": {
                "error_type": type(err).__name__,
                "message": str(err)
            }
        }
        _append_rag_log(log_entry)
        st.error(f"⚠️ **Unexpected Error:** {err}")
        return None


def _clean_doc_title(text: Any) -> str:
    """Sanitize title or section string for UI display without corrupting character spacing."""
    if not text:
        return ""
    return str(text).replace("\ufffd", " - ").replace("\u2013", " - ").strip()


def _highlight_text_spans(text: str, spans: Optional[List[Any]]) -> str:
    """
    Highlights character spans in text using HTML <mark> tags.
    Spans can be [[start, end], ...] or [{'start': s, 'end': e}, ...].
    Overlapping spans are safely merged and out-of-bound ranges ignored.
    """
    if not text or not spans:
        return text

    valid_spans = []
    for sp in spans:
        s, e = None, None
        if isinstance(sp, (list, tuple)) and len(sp) >= 2:
            try:
                s, e = int(sp[0]), int(sp[1])
            except (ValueError, TypeError):
                continue
        elif isinstance(sp, dict):
            try:
                s = int(sp.get("start", sp.get("offset_start", -1)))
                e = int(sp.get("end", sp.get("offset_end", -1)))
            except (ValueError, TypeError):
                continue

        if s is not None and e is not None and 0 <= s < e <= len(text):
            valid_spans.append((s, e))

    if not valid_spans:
        return text

    valid_spans.sort(key=lambda x: (x[0], -x[1]))
    merged = []
    for s, e in valid_spans:
        if not merged:
            merged.append([s, e])
        else:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

    import html
    result = []
    last_idx = 0
    for s, e in merged:
        result.append(html.escape(text[last_idx:s]))
        result.append(f"<mark style='background: rgba(0, 229, 255, 0.35); color: inherit; padding: 1px 4px; border-radius: 3px; font-weight: 600;'>{html.escape(text[s:e])}</mark>")
        last_idx = e
    result.append(html.escape(text[last_idx:]))
    return "".join(result)


def _build_assistant_message(resp: AgentResponse) -> Dict[str, Any]:
    """Build a standardized assistant message dictionary for st.session_state.messages."""
    ans_plain = getattr(resp, "answer_plain", "") or resp.answer
    return {
        "role": "assistant",
        "content": resp.answer,
        "content_plain": ans_plain,
        "sources": list(getattr(resp, "sources", [])),
        "citations": list(getattr(resp, "citations", [])),
        "grounded": getattr(resp, "grounded", True),
        "status": getattr(resp, "status", "answered"),
        "is_live": bool(getattr(resp, "is_live_backend", False)),
        "follow_ups": list(getattr(resp, "follow_ups", []) or []),
        "regions": dict(getattr(resp, "regions", {}) or {}),
        "did_you_mean": getattr(resp, "did_you_mean", None),
        "map_bbox": getattr(resp, "map_bbox", None),
    }


@st.dialog("🛰️ RAG Agent Telemetry & Debug Logs", width="large")
def show_rag_log_dialog():
    """Modal dialog displaying comprehensive raw JSON payloads, telemetry, and diagnostics for RAG calls."""
    logs: List[Dict[str, Any]] = st.session_state.get("rag_logs", [])
    if not logs:
        st.info("ℹ️ No RAG agent communications recorded yet in this session. Send a chat query or perform a reconnaissance scan to capture live telemetry.")
        return

    # Header summary metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Logs", len(logs))
    with col_m2:
        latest = logs[-1]
        st.metric("Last Latency", f"{latest['latency_ms']} ms")
    with col_m3:
        status_label = "🟢 SUCCESS" if latest["status"] == "SUCCESS" else "🔴 ERROR"
        st.metric("Last Status", status_label)
    with col_m4:
        mode_label = "🟢 Live Agent" if latest.get("is_live_backend") else "🟡 Offline Mock"
        st.metric("Backend Engine", mode_label)

    st.markdown("---")

    # Transaction Selector
    log_options = list(range(len(logs) - 1, -1, -1))
    def format_log_option(idx: int) -> str:
        item = logs[idx]
        indicator = "🟢" if item["status"] == "SUCCESS" else "🔴"
        live_tag = "LIVE" if item.get("is_live_backend") else "MOCK"
        summary = item["query_summary"]
        if len(summary) > 40:
            summary = summary[:37] + "..."
        return f"#{item['id']} [{item['timestamp']}] {indicator} [{live_tag}] {item['latency_ms']}ms — \"{summary}\""

    selected_idx = st.selectbox(
        "Select Transaction / Query Turn to Inspect:",
        options=log_options,
        format_func=format_log_option,
        key="rag_log_selected_turn"
    )

    selected_entry = logs[selected_idx]

    # Tabs for detailed inspection
    tab_req, tab_res, tab_meta, tab_src = st.tabs([
        "📤 Request JSON",
        "📥 Response JSON",
        "⚡ Telemetry & Diagnostics",
        "📚 Grounding Sources"
    ])

    with tab_req:
        st.caption("Exact QueryRequest JSON payload sent across the agent boundary (GeoJSON coordinates in [lon, lat] order):")
        st.json(selected_entry["raw_request"])
        st.download_button(
            "📋 Download Request JSON",
            data=json.dumps(selected_entry["raw_request"], indent=2),
            file_name=f"rag_request_turn_{selected_entry['id']}.json",
            mime="application/json",
            key=f"dl_req_{selected_entry['id']}"
        )

    with tab_res:
        if selected_entry["status"] == "SUCCESS":
            st.caption("Exact raw dictionary returned by the RAG backend / reasoning engine:")
            st.json(selected_entry["raw_response"])
            st.download_button(
                "📋 Download Response JSON",
                data=json.dumps(selected_entry["raw_response"], indent=2),
                file_name=f"rag_response_turn_{selected_entry['id']}.json",
                mime="application/json",
                key=f"dl_res_{selected_entry['id']}"
            )
        else:
            st.error("Transaction failed with an error. No valid response dictionary was returned.")
            if selected_entry.get("error_details"):
                st.json(selected_entry["error_details"])

    with tab_meta:
        st.markdown(f"#### 🔍 Transaction #{selected_entry['id']} Diagnostics")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"- **Timestamp:** `{selected_entry['timestamp']}`")
            st.markdown(f"- **Execution Latency:** `{selected_entry['latency_ms']} ms`")
            st.markdown(f"- **Execution Engine:** `{'Live GeoRAG Agent (OpenRouter)' if selected_entry.get('is_live_backend') else 'Deterministic Mock Engine'}`")
            st.markdown(f"- **Status Code:** `{selected_entry['status']}`")
        with col_t2:
            st.markdown(f"- **Spatial Input Type:** `{selected_entry.get('spatial_input', {}).get('type', 'none')}`")
            st.markdown(f"- **Extracted Map Layers:** `{selected_entry.get('layers_count', 0)}`")
            st.markdown(f"- **Cited Sources Count:** `{len(selected_entry.get('sources', []))}`")
            if selected_entry.get("error_details"):
                st.markdown(f"- **Exception Class:** `{selected_entry['error_details'].get('error_type')}`")

        if selected_entry.get("error_details"):
            st.markdown("##### ⚠️ Error Details & Diagnostic Tip")
            st.code(selected_entry["error_details"].get("message", ""), language="text")

    with tab_src:
        sources_list = selected_entry.get("sources", [])
        citations_map = {
            c.get("pointer"): c
            for c in selected_entry.get("citations", [])
            if isinstance(c, dict) and c.get("pointer")
        }
        if sources_list:
            st.markdown(f"**Found {len(sources_list)} source citation pointer(s) for this query:**")
            for s_idx, s_ptr in enumerate(sources_list):
                cit_meta = citations_map.get(s_ptr, {})
                cit_id = cit_meta.get("id")
                cit_article = cit_meta.get("article")
                cit_section = cit_meta.get("section")
                text_data = get_source_text(s_ptr)
                doc_title = _clean_doc_title((text_data.get("document") if text_data else None) or cit_article)
                sec_title = _clean_doc_title((text_data.get("section") if text_data else None) or cit_section)
                para_idx = text_data.get("paragraph_index") if text_data else None

                if cit_id and doc_title:
                    header_label = f"📄 [{cit_id}] {doc_title}"
                elif doc_title:
                    header_label = f"📄 {doc_title}"
                else:
                    header_label = f"📄 Source #{s_idx + 1}"
                if sec_title:
                    header_label += f" — *{sec_title}*"

                with st.expander(header_label):
                    if text_data:
                        meta_parts = [f"**Document:** `{doc_title}`"]
                        if sec_title:
                            meta_parts.append(f"**Section:** `{sec_title}`")
                        if para_idx is not None:
                            meta_parts.append(f"**Paragraph:** `#{para_idx + 1}`")
                        st.markdown("  |  ".join(meta_parts))
                        st.markdown("**Verbatim Grounding Passage:**")
                        st.info(text_data.get("text", "No text passage available."))
                    else:
                        if doc_title:
                            st.markdown(f"**Document:** `{doc_title}`" + (f"  |  **Section:** `{sec_title}`" if sec_title else ""))
                        st.caption("No verbatim text found in local graph.")
        else:
            st.caption("No knowledge graph source document pointers returned for this turn.")

    st.markdown("---")
    col_act1, col_act2 = st.columns([3, 1])
    with col_act1:
        st.download_button(
            "⬇️ Export Entire Session Logs (.json)",
            data=json.dumps(logs, indent=2),
            file_name=f"geoai_rag_session_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with col_act2:
        if st.button("🗑️ Clear Logs", use_container_width=True, help="Reset communication log history"):
            st.session_state.rag_logs = []
            st.rerun()

def clear_spatial_selection():
    """Clear all active spatial selections and reset drawing trackers to prevent rerun loops."""
    st.session_state.selected_points = []
    st.session_state.bounding_box = None
    st.session_state.active_point = None
    st.session_state.selection_mode = "Point Selection"
    st.session_state.last_drawing_time = time.time()
    if st.session_state.get("last_processed_drawing"):
        st.session_state.ignored_drawing = st.session_state.get("last_processed_drawing")
    st.session_state.last_processed_drawing = None


class MapTrashBinHook(MacroElement):
    """
    Folium MacroElement injecting client-side integration for the Leaflet Draw toolbar:
    1. Instant client-side visual feedback (0ms latency) on map clicks (target point) and bbox drawing.
    2. Registers dynamically rendered selection layers into `drawnItems` (the Leaflet Draw FeatureGroup).
    3. Hooks the Leaflet Draw trash bin button (.leaflet-draw-edit-remove) and 'Clear all' sub-toolbar actions
       so clicking the garbage can icon directly clears the active spatial selection or tactical overlays.
    4. Fires Leaflet's `draw:deleted` event and updates Streamlit component state.
    """
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            if (!map) return;

            // ------------------------------------------------------------------
            // Dynamic environment-adaptive styling for map controls (Dark vs Light)
            // ------------------------------------------------------------------
            function updateMapControlsTheme(layerName) {
                var container = map.getContainer();
                if (!container) return;
                var nameLower = (layerName || '').toLowerCase();
                var isLight = nameLower.indexOf('voyager (nav)') !== -1
                           || nameLower.indexOf('topographic') !== -1
                           || nameLower.indexOf('openstreetmap') !== -1;

                if (isLight) {
                    container.classList.remove('geoai-map-dark');
                    container.classList.add('geoai-map-light');
                } else {
                    container.classList.remove('geoai-map-light');
                    container.classList.add('geoai-map-dark');
                }
            }

            if (map.getContainer()) {
                map.getContainer().classList.add('geoai-map-dark');
            }

            setTimeout(function() {
                var activeName = 'CartoDB Voyager Dark (Detailed Nav)';
                map.eachLayer(function(l) {
                    if (l && l.options && l.options.name) {
                        activeName = l.options.name;
                    }
                });
                updateMapControlsTheme(activeName);
            }, 60);

            map.on('baselayerchange', function(e) {
                if (e && e.name) {
                    updateMapControlsTheme(e.name);
                }
            });

            // Track drawing state to avoid placing single-click marker when drawing AOI rectangle
            map._isDrawing = false;
            map.on('draw:drawstart', function() {
                map._isDrawing = true;
            });
            map.on('draw:drawstop', function() {
                setTimeout(function() { map._isDrawing = false; }, 350);
            });

            // Register active selection layers into drawnItems so Leaflet Draw recognizes them
            if (typeof drawnItems !== 'undefined') {
                {% for lname in this.selection_layer_names %}
                if (typeof {{ lname }} !== 'undefined') {
                    try {
                        drawnItems.addLayer({{ lname }});
                    } catch(e) {}
                }
                {% endfor %}
            }

            // Function to instantly remove any active selection from the map DOM (0ms latency)
            function clearPreviousSelectionInstantly() {
                // 1. Clear Leaflet Draw items
                try {
                    if (window.drawnItems && typeof window.drawnItems.clearLayers === 'function') {
                        window.drawnItems.clearLayers();
                    }
                } catch(e) {}

                // 2. Clear client-side instant target marker
                if (window._instantTargetMarker) {
                    try { map.removeLayer(window._instantTargetMarker); } catch(err) {}
                    window._instantTargetMarker = null;
                }

                // 3. Clear and remove the active selection feature group (feature_group[0])
                try {
                    if (window.feature_group && window.feature_group.length > 0) {
                        var selFg = window.feature_group[0];
                        if (selFg) {
                            if (typeof selFg.clearLayers === 'function') {
                                selFg.clearLayers();
                            }
                            map.removeLayer(selFg);
                        }
                    }
                } catch(e) {}

                // 4. Scan all layers on the map and immediately purge any prior target or bbox layers
                try {
                    map.eachLayer(function(layer) {
                        try {
                            var tt = layer.getTooltip && layer.getTooltip();
                            var ttContent = tt ? (tt.getContent ? tt.getContent() : tt._content) : "";
                            if (typeof ttContent === "string" && (
                                ttContent.indexOf("★ Target Point") !== -1 ||
                                ttContent.indexOf("★ Bounding Box") !== -1 ||
                                ttContent.indexOf("★ BBOX") !== -1
                            )) {
                                map.removeLayer(layer);
                                return;
                            }
                            var pop = layer.getPopup && layer.getPopup();
                            var popContent = pop ? (pop.getContent ? pop.getContent() : pop._content) : "";
                            if (typeof popContent === "string" && (
                                popContent.indexOf("★ TARGET POINT") !== -1 ||
                                popContent.indexOf("★ BBOX") !== -1 ||
                                popContent.indexOf("★ TWO-POINT BOUNDING BOX") !== -1
                            )) {
                                map.removeLayer(layer);
                                return;
                            }
                            if (layer.eachLayer) {
                                var toRemove = [];
                                layer.eachLayer(function(subLayer) {
                                    try {
                                        var s_tt = subLayer.getTooltip && subLayer.getTooltip();
                                        var s_ttc = s_tt ? (s_tt.getContent ? s_tt.getContent() : s_tt._content) : "";
                                        if (typeof s_ttc === "string" && (
                                            s_ttc.indexOf("★ Target Point") !== -1 ||
                                            s_ttc.indexOf("★ Bounding Box") !== -1 ||
                                            s_ttc.indexOf("★ BBOX") !== -1
                                        )) {
                                            toRemove.push(subLayer);
                                        }
                                    } catch(err2) {}
                                });
                                toRemove.forEach(function(sub) {
                                    try { layer.removeLayer(sub); } catch(err3) {}
                                    try { map.removeLayer(sub); } catch(err4) {}
                                });
                            }
                        } catch(err) {}
                    });
                } catch(e) {}
            }

            // 1. Instant client-side visual feedback on map canvas clicks (0ms latency)
            map.on('click', function(e) {
                if (map._isDrawing) return;
                if (e.originalEvent && (e.originalEvent._stopped || e.originalEvent.defaultPrevented)) return;

                // Guard: Ignore clicks on existing markers, cluster icons, popups, or controls
                var target = e.originalEvent && e.originalEvent.target;
                if (target && target.closest && target.closest('.leaflet-marker-icon, .awesome-marker, .marker-cluster, .leaflet-popup, .leaflet-popup-content-wrapper, .leaflet-control')) {
                    return;
                }

                var lat = Math.round(e.latlng.lat * 100000) / 100000;
                var lng = Math.round(e.latlng.lng * 100000) / 100000;

                // Immediately purge any prior selection (from previous click or python render)
                clearPreviousSelectionInstantly();

                try {
                    var starIcon = L.AwesomeMarkers.icon({
                        icon: 'star',
                        markerColor: 'red',
                        prefix: 'glyphicon'
                    });
                    var marker = L.marker([lat, lng], { icon: starIcon });
                    marker.bindTooltip("★ Target Point (" + lat.toFixed(4) + "°, " + lng.toFixed(4) + "°)");

                    if (window.drawnItems) {
                        window.drawnItems.addLayer(marker);
                    } else {
                        marker.addTo(map);
                    }
                    window._instantTargetMarker = marker;
                } catch(err) {
                    console.warn("[GeoAI] Instant marker error:", err);
                }

                var btn = document.querySelector(".leaflet-draw-edit-remove");
                if (btn) {
                    btn.classList.remove("leaflet-disabled");
                }
                if (window.__GLOBAL_DATA__) {
                    window.__GLOBAL_DATA__.lat_lng_clicked = e.latlng;
                }
                try {
                    if (typeof updateComponentValue === "function") {
                        updateComponentValue(map);
                    }
                } catch(err) {}
            });

            // Synchronize selection and maintain popup state when existing markers are clicked
            map.on('popupopen', function(e) {
                if (window._reopeningPopup) return;
                if (e.popup) {
                    var latlng = e.popup.getLatLng();
                    var source = e.popup._source;
                    if (latlng) {
                        window._lastOpenPopupInfo = {
                            lat: Math.round(latlng.lat * 100000) / 100000,
                            lng: Math.round(latlng.lng * 100000) / 100000
                        };
                    }
                    if (source && typeof source.toGeoJSON === 'function' && window.__GLOBAL_DATA__) {
                        window.__GLOBAL_DATA__.last_active_drawing = source.toGeoJSON();
                        if (typeof debouncedUpdateComponentValue === 'function') {
                            debouncedUpdateComponentValue(map);
                        }
                    }
                }
            });

            map.on('popupclose', function(e) {
                if (!window._reopeningPopup) {
                    window._lastOpenPopupInfo = null;
                }
            });

            if (window._lastOpenPopupInfo) {
                var targetInfo = window._lastOpenPopupInfo;
                window._lastOpenPopupInfo = null;
                setTimeout(function() {
                    try {
                        window._reopeningPopup = true;
                        map.eachLayer(function(l) {
                            if (l instanceof L.Marker && l.getLatLng && l.getPopup) {
                                var l_ll = l.getLatLng();
                                var l_lat = Math.round(l_ll.lat * 100000) / 100000;
                                var l_lng = Math.round(l_ll.lng * 100000) / 100000;
                                if (Math.abs(l_lat - targetInfo.lat) < 0.0001 && Math.abs(l_lng - targetInfo.lng) < 0.0001) {
                                    l.openPopup();
                                }
                            }
                        });
                    } catch(err) {} finally {
                        setTimeout(function() { window._reopeningPopup = false; }, 200);
                    }
                }, 120);
            }

            // 2. Instant client-side feedback when an AOI Rectangle is drawn
            map.on('draw:created', function(e) {
                if (e.layerType === 'rectangle') {
                    var rect = e.layer;
                    var bounds = rect.getBounds();
                    var sw = bounds.getSouthWest();
                    var ne = bounds.getNorthEast();

                    // Immediately clear any prior selection (single point or previous box)
                    clearPreviousSelectionInstantly();

                    try {
                        var iconDown = L.AwesomeMarkers.icon({
                            icon: 'arrow-down',
                            markerColor: 'red',
                            prefix: 'glyphicon'
                        });
                        var iconUp = L.AwesomeMarkers.icon({
                            icon: 'arrow-up',
                            markerColor: 'red',
                            prefix: 'glyphicon'
                        });
                        var mSW = L.marker([sw.lat, sw.lng], { icon: iconDown })
                            .bindTooltip("★ Bounding Box Lower-Left (SW): " + sw.lat.toFixed(4) + "°, " + sw.lng.toFixed(4) + "°");
                        var mNE = L.marker([ne.lat, ne.lng], { icon: iconUp })
                            .bindTooltip("★ Bounding Box Upper-Right (NE): " + ne.lat.toFixed(4) + "°, " + ne.lng.toFixed(4) + "°");

                        if (window.drawnItems) {
                            window.drawnItems.addLayer(mSW);
                            window.drawnItems.addLayer(mNE);
                        } else {
                            mSW.addTo(map);
                            mNE.addTo(map);
                        }
                    } catch(err) {
                        console.warn("[GeoAI] Instant bbox markers error:", err);
                    }

                    var btn = document.querySelector(".leaflet-draw-edit-remove");
                    if (btn) {
                        btn.classList.remove("leaflet-disabled");
                    }
                    try {
                        if (typeof updateComponentValue === "function") {
                            updateComponentValue(map);
                        }
                    } catch(err) {}
                }
            });

            function triggerRemoval() {
                clearPreviousSelectionInstantly();

                // If user triggers trash can, also clear all feature groups on map
                if (window.feature_group && window.feature_group.length > 0) {
                    window.feature_group.forEach(function(fg) {
                        try {
                            if (typeof fg.clearLayers === 'function') fg.clearLayers();
                            map.removeLayer(fg);
                        } catch(err) {}
                    });
                }

                var delPayload = {
                    type: "Feature",
                    geometry: null,
                    properties: { action: "delete", timestamp: Date.now() }
                };

                if (window.__GLOBAL_DATA__) {
                    window.__GLOBAL_DATA__.all_drawings = [];
                    window.__GLOBAL_DATA__.last_active_drawing = delPayload;
                }

                try {
                    map.fire("draw:deleted", {
                        layer: {
                            toGeoJSON: function() { return delPayload; }
                        }
                    });
                } catch(e) {}

                try {
                    if (typeof debouncedUpdateComponentValue === "function") {
                        debouncedUpdateComponentValue(map);
                    } else if (typeof updateComponentValue === "function") {
                        updateComponentValue(map);
                    }
                } catch(e) {}
            }

            // Delegated click listener to catch trash bin button clicks
            document.addEventListener("click", function(e) {
                var trashBtn = e.target.closest(".leaflet-draw-edit-remove");
                if (trashBtn) {
                    e.preventDefault();
                    e.stopPropagation();
                    triggerRemoval();
                    return false;
                }
                var actBtn = e.target.closest(".leaflet-draw-actions a");
                if (actBtn && actBtn.textContent && actBtn.textContent.toLowerCase().indexOf("clear all") !== -1) {
                    triggerRemoval();
                }
            }, true);

            // Dynamically synchronize the trash button state from live map feature groups and drawn items
            function updateTrashButtonState() {
                var hasActive = false;
                if (typeof drawnItems !== 'undefined' && drawnItems.getLayers && drawnItems.getLayers().length > 0) {
                    hasActive = true;
                }
                if (!hasActive && window.feature_group && window.feature_group.length > 0) {
                    for (var i = 0; i < window.feature_group.length; i++) {
                        if (window.feature_group[i] && window.feature_group[i].getLayers && window.feature_group[i].getLayers().length > 0) {
                            hasActive = true;
                            break;
                        }
                    }
                }
                var btn = document.querySelector(".leaflet-draw-edit-remove");
                if (btn) {
                    if (hasActive) {
                        btn.classList.remove("leaflet-disabled");
                        btn.setAttribute("title", "Remove active target selection or tactical overlays");
                    }
                }
            }
            setTimeout(updateTrashButtonState, 100);

            // ------------------------------------------------------------------
            // Bottom-Right Tactical Pin Filter Control
            // ------------------------------------------------------------------
            var PinFilterControl = L.Control.extend({
                options: {
                    position: 'bottomright'
                },
                onAdd: function(map) {
                    var container = L.DomUtil.create('div', 'leaflet-control-layers leaflet-control-pin-filter leaflet-control');
                    container.setAttribute('aria-haspopup', 'true');

                    var toggle = L.DomUtil.create('a', 'leaflet-control-layers-toggle leaflet-control-pin-filter-toggle', container);
                    toggle.href = '#';
                    toggle.title = 'Filter Tactical Pins';

                    var section = L.DomUtil.create('section', 'leaflet-control-layers-list leaflet-control-pin-filter-list', container);
                    var overlays = L.DomUtil.create('div', 'leaflet-control-layers-overlays', section);

                    var header = L.DomUtil.create('div', 'pin-filter-header', overlays);
                    header.innerHTML = '📍 <b>PIN FILTERS</b>';

                    // Context pins toggle
                    var lblCtx = L.DomUtil.create('label', '', overlays);
                    var divCtx = L.DomUtil.create('div', '', lblCtx);
                    var chkCtx = L.DomUtil.create('input', 'leaflet-control-layers-selector', divCtx);
                    chkCtx.type = 'checkbox';
                    chkCtx.id = 'pin_filter_chk_context';
                    chkCtx.checked = true;
                    var spanCtx = L.DomUtil.create('span', '', divCtx);
                    spanCtx.id = 'pin_filter_lbl_context';
                    spanCtx.textContent = '🔘 Context pins';

                    // Uncertain pins toggle
                    var lblUnc = L.DomUtil.create('label', '', overlays);
                    var divUnc = L.DomUtil.create('div', '', lblUnc);
                    var chkUnc = L.DomUtil.create('input', 'leaflet-control-layers-selector', divUnc);
                    chkUnc.type = 'checkbox';
                    chkUnc.id = 'pin_filter_chk_uncertain';
                    chkUnc.checked = true;
                    var spanUnc = L.DomUtil.create('span', '', divUnc);
                    spanUnc.id = 'pin_filter_lbl_uncertain';
                    spanUnc.textContent = '🟠 Uncertain pins';

                    // Hover expand/collapse
                    L.DomEvent.on(container, 'mouseover', function() {
                        container.classList.add('leaflet-control-layers-expanded');
                    });
                    L.DomEvent.on(container, 'mouseout', function() {
                        container.classList.remove('leaflet-control-layers-expanded');
                    });

                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);

                    var clusterGroups = [];

                    function applyPinFilter() {
                        var showCtx = chkCtx.checked;
                        var showUnc = chkUnc.checked;

                        // Toggle MarkerCluster groups for context pins
                        for (var c = 0; c < clusterGroups.length; c++) {
                            var cg = clusterGroups[c];
                            if (!showCtx) {
                                if (map.hasLayer(cg)) map.removeLayer(cg);
                            } else {
                                if (!map.hasLayer(cg)) map.addLayer(cg);
                            }
                        }

                        // Toggle individual markers
                        map.eachLayer(function(layer) {
                            if (layer instanceof L.Marker && layer._icon) {
                                var cls = layer._icon.className || '';
                                if (cls.indexOf('awesome-marker-icon-red') !== -1) return;

                                var isCtx = cls.indexOf('awesome-marker-icon-cadetblue') !== -1;
                                var isUnc = cls.indexOf('awesome-marker-icon-orange') !== -1;

                                var isVisible = true;
                                if (isCtx && !showCtx) isVisible = false;
                                if (isUnc && !showUnc) isVisible = false;

                                layer._icon.style.display = isVisible ? '' : 'none';
                                if (layer._shadow) {
                                    layer._shadow.style.display = isVisible ? '' : 'none';
                                }
                                if (!isVisible) {
                                    if (layer.closePopup) layer.closePopup();
                                    if (layer.closeTooltip) layer.closeTooltip();
                                }
                            }
                        });
                    }

                    chkCtx.addEventListener('change', applyPinFilter);
                    chkUnc.addEventListener('change', applyPinFilter);

                    function refreshPinCounts() {
                        var countCtx = 0;
                        var countUnc = 0;
                        var countTotal = 0;

                        for (var c = 0; c < clusterGroups.length; c++) {
                            countCtx += clusterGroups[c].getLayers().length;
                        }

                        map.eachLayer(function(layer) {
                            if (typeof L.MarkerClusterGroup !== 'undefined' && layer instanceof L.MarkerClusterGroup) {
                                if (clusterGroups.indexOf(layer) === -1) {
                                    clusterGroups.push(layer);
                                    countCtx += layer.getLayers().length;
                                }
                                return;
                            }
                            if (layer instanceof L.Marker && layer._icon) {
                                var cls = layer._icon.className || '';
                                if (cls.indexOf('awesome-marker-icon-cadetblue') !== -1) countCtx++;
                                if (cls.indexOf('awesome-marker-icon-orange') !== -1) countUnc++;
                                if (cls.indexOf('awesome-marker-icon-blue') !== -1) countTotal++;
                            }
                        });
                        spanCtx.textContent = '🔘 Context pins (' + countCtx + ')';
                        spanUnc.textContent = '🟠 Uncertain pins (' + countUnc + ')';

                        if (countCtx === 0 && countUnc === 0 && countTotal === 0) {
                            container.style.display = 'none';
                        } else {
                            container.style.display = 'block';
                            applyPinFilter();
                        }
                    }

                    var _refreshTimer = null;
                    function debouncedRefreshPinCounts(delay) {
                        if (_refreshTimer) clearTimeout(_refreshTimer);
                        _refreshTimer = setTimeout(refreshPinCounts, delay || 200);
                    }

                    setTimeout(refreshPinCounts, 300);
                    map.on('layeradd', function(e) {
                        if (typeof L.MarkerClusterGroup !== 'undefined' && e.layer instanceof L.MarkerClusterGroup) {
                            if (clusterGroups.indexOf(e.layer) === -1) clusterGroups.push(e.layer);
                        }
                        debouncedRefreshPinCounts(200);
                    });
                    map.on('layerremove', function() { debouncedRefreshPinCounts(200); });

                    return container;
                }
            });

            map.addControl(new PinFilterControl());
        })();
        {% endmacro %}
    """)

    def __init__(self, selection_layer_names=None, has_active=False):
        super().__init__()
        self.selection_layer_names = selection_layer_names or []
        self.has_active = has_active

# ==============================================================================
# 1. Visual Theme Configurations & Page Setup (GeoAI Defense C2)
# ==============================================================================
THEME_CONFIG = {
    "name": "GEOAI DEFENSE C2",
    "protocol_code": "SEC-OP // C2-GEOAI",
    "primary": "#00e5ff",          # Electric Cyan
    "secondary": "#0ea5e9",        # Tactical Sky Blue
    "glow": "rgba(0, 229, 255, 0.4)",
    "bg_card": "rgba(14, 22, 32, 0.88)",
    "border_card": "rgba(0, 229, 255, 0.28)",
    "text_main": "#e2e8f0",
    "text_dim": "#7dd3fc",
    "badge_bg": "rgba(0, 229, 255, 0.12)",
    "badge_color": "#00e5ff",
    "badge_border": "rgba(0, 229, 255, 0.35)",
    "status_bg": "rgba(0, 229, 255, 0.08)",
    "status_border": "#00e5ff",
    "operator_tag": "[OPERATOR // C2-CMD]",
    "ai_tag": "[GEO-AI // SYNTHESIS]",
    "user_avatar": "🎯",
    "ai_avatar": "🛰️",
    "folium_marker_color": "red",
    "folium_marker_icon": "star",
}
t_cfg = THEME_CONFIG

# Configure Streamlit page properties to use a responsive wide layout
st.set_page_config(
    page_title=f"GeoAI Intelligence Hub [{t_cfg['name']}]",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tactical HUD interface styling
st.markdown(f"""
<style>
    /* Global Typography */
    html, body, [class*="css"], font, span, p, div, button, input {{
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    }}

    /* Lock Viewport to Chrome Window - Zero Main Window Scrollbar */
    html, body {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* Suppress outer window scrollbars in Chrome/Edge/Firefox */
    html::-webkit-scrollbar, body::-webkit-scrollbar, [data-testid="stAppViewContainer"]::-webkit-scrollbar {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }}

    [data-testid="stAppViewContainer"] {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }}

    section[data-testid="stMain"] {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        padding: 0 !important;
    }}

    div[data-testid="stMainBlockContainer"],
    .main .block-container {{
        height: 100vh !important;
        max-height: 100vh !important;
        padding-top: 2.2rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }}

    /* Header & Side Ribbon (Sidebar Toggle Controls) */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 2.2rem !important;
        min-height: 2.2rem !important;
        z-index: 100 !important;
        display: flex !important;
        align-items: center !important;
    }}

    /* Restore and style the side ribbon (sidebar collapsed control) */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1000 !important;
    }}

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button {{
        color: {t_cfg['primary']} !important;
        background: {t_cfg['bg_card']} !important;
        border: 1px solid {t_cfg['border_card']} !important;
        border-radius: 4px !important;
        transition: all 0.2s ease-in-out !important;
    }}

    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover {{
        border-color: {t_cfg['primary']} !important;
        box-shadow: 0 0 10px {t_cfg['glow']} !important;
        color: #ffffff !important;
    }}

    footer {{
        display: none !important;
    }}

    /* Compact resizer handles iframe from adjustable_columns */
    iframe[title="streamlit_adjustable_columns.streamlit_adjustable_columns"] {{
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        margin-top: 2px !important;
        margin-bottom: 6px !important;
        border-radius: 4px !important;
    }}

    /* Independent vertical scrolling for sidebar controls */
    section[data-testid="stSidebar"] {{
        height: 100vh !important;
        overflow-y: auto !important;
    }}

    /* Dedicated Tactical Scrollbars for scrollable sub-panels (Chat & Sidebar) */
    *::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    *::-webkit-scrollbar-track {{
        background: rgba(0, 0, 0, 0.4);
        border-radius: 3px;
    }}
    *::-webkit-scrollbar-thumb {{
        background: {t_cfg['border_card']};
        border-radius: 3px;
        border: 1px solid {t_cfg['primary']};
        box-shadow: 0 0 6px {t_cfg['glow']};
    }}
    *::-webkit-scrollbar-thumb:hover {{
        background: {t_cfg['primary']};
        box-shadow: 0 0 10px {t_cfg['glow']};
    }}

    /* Metric & Badge styling */
    .geo-badge {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        background-color: {t_cfg['badge_bg']};
        color: {t_cfg['badge_color']};
        border: 1px solid {t_cfg['badge_border']};
        margin-bottom: 8px;
        box-shadow: 0 0 8px {t_cfg['glow']};
    }}
    .geo-status-box {{
        background-color: {t_cfg['status_bg']};
        border-left: 3px solid {t_cfg['status_border']};
        border-top: 1px solid {t_cfg['border_card']};
        border-right: 1px solid {t_cfg['border_card']};
        border-bottom: 1px solid {t_cfg['border_card']};
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: {t_cfg['text_main']};
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
    }}
    .geo-docked-pill {{
        display: flex;
        align-items: center;
        gap: 6px;
        background-color: {t_cfg['status_bg']};
        border-left: 3px solid {t_cfg['status_border']};
        border-top: 1px solid {t_cfg['border_card']};
        border-right: 1px solid {t_cfg['border_card']};
        border-bottom: 1px solid {t_cfg['border_card']};
        padding: 4px 10px;
        border-radius: 4px;
        height: 38px;
        box-sizing: border-box;
        box-shadow: 0 0 8px {t_cfg['glow']};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 2px;
    }}
    .geo-docked-pill-stacked {{
        padding: 2px 8px;
        height: 38px;
    }}
    .geo-pill-icon {{
        font-size: 0.95rem;
        line-height: 1;
        flex-shrink: 0;
    }}
    .geo-pill-label {{
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {t_cfg['text_dim']};
        text-transform: uppercase;
        flex-shrink: 0;
    }}
    .geo-pill-stack {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        line-height: 1.15;
        overflow: hidden;
    }}
    .geo-pill-row {{
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
    }}
    .geo-pill-sublabel {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 700;
        color: {t_cfg['text_dim']};
        letter-spacing: 0.05em;
    }}
    .geo-pill-coord {{
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        color: {t_cfg['primary']};
        text-shadow: 0 0 6px {t_cfg['glow']};
    }}
    .column-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        background: {t_cfg['bg_card']};
        border-left: 3px solid {t_cfg['primary']};
        border-top: 1px solid {t_cfg['border_card']};
        border-right: 1px solid {t_cfg['border_card']};
        border-bottom: 1px solid {t_cfg['border_card']};
        border-radius: 4px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
        color: {t_cfg['primary']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        text-shadow: 0 0 8px {t_cfg['glow']};
    }}

    /* =========================================================================
       Chat Console HUD - Chat Bubble Architecture
       Operator on Right, Geo-AI on Left
       ========================================================================= */

    /* Ensure parent containers allow flex alignment of chat bubbles */
    div[data-testid="stChatMessage"] {{
        display: flex !important;
        box-sizing: border-box !important;
        margin-bottom: 0.95rem !important;
        padding: 0.75rem 1rem 0.95rem 1rem !important;
        border-radius: 12px 12px 12px 12px !important;
        transition: all 0.2s ease-in-out !important;
        width: fit-content !important;
    }}

    /* Target layout wrappers and element containers to align bubbles */
    div[data-testid="stLayoutWrapper"]:has(.bubble-marker-operator),
    div[data-testid="stLayoutWrapper"]:has([aria-label*="user" i]),
    div[data-testid="stElementContainer"]:has(.bubble-marker-operator),
    div[data-testid="stElementContainer"]:has([aria-label*="user" i]),
    div:has(> div[data-testid="stChatMessage"]:has(.bubble-marker-operator)),
    div:has(> div[data-testid="stChatMessage"]:has([aria-label*="user" i])) {{
        align-items: flex-end !important;
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
    }}

    div[data-testid="stLayoutWrapper"]:has(.bubble-marker-geoai),
    div[data-testid="stLayoutWrapper"]:has([aria-label*="assistant" i]),
    div[data-testid="stElementContainer"]:has(.bubble-marker-geoai),
    div[data-testid="stElementContainer"]:has([aria-label*="assistant" i]),
    div:has(> div[data-testid="stChatMessage"]:has(.bubble-marker-geoai)),
    div:has(> div[data-testid="stChatMessage"]:has([aria-label*="assistant" i])) {{
        align-items: flex-start !important;
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
    }}

    /* --- OPERATOR (USER) CHAT BUBBLE - RIGHT SIDE --- */
    div[data-testid="stChatMessage"]:has(.bubble-marker-operator),
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]),
    div[data-testid="stChatMessage"]:has(.chat-role-operator) {{
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        align-self: flex-end !important;
        max-width: 92% !important;
        min-width: 160px !important;
        background: linear-gradient(135deg, rgba(14, 44, 72, 0.94) 0%, rgba(10, 26, 44, 0.94) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-right: 3px solid #38bdf8 !important;
        border-radius: 12px 12px 12px 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45), 0 0 12px rgba(56, 189, 248, 0.2) !important;
    }}

    /* Operator avatar positioning & glow */
    div[data-testid="stChatMessage"]:has(.bubble-marker-operator) > :first-child,
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) > :first-child {{
        margin-left: 0.65rem !important;
        margin-right: 0 !important;
        filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.8)) !important;
    }}

    /* Operator inner content & header */
    .chat-role-tag.chat-role-operator {{
        text-align: right !important;
        color: #38bdf8 !important;
        letter-spacing: 0.08em;
        font-weight: 700;
        text-shadow: 0 0 6px rgba(56, 189, 248, 0.6);
        margin-bottom: 4px;
        font-size: 0.72rem;
    }}
    div[data-testid="stChatMessage"]:has(.bubble-marker-operator) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) div[data-testid="stChatMessageContent"] {{
        text-align: left !important;
        color: #f0f9ff !important;
        min-width: 120px !important;
    }}

    /* --- GEO-AI (ASSISTANT) CHAT BUBBLE - LEFT SIDE --- */
    div[data-testid="stChatMessage"]:has(.bubble-marker-geoai),
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]),
    div[data-testid="stChatMessage"]:has(.chat-role-geoai) {{
        flex-direction: row !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        align-self: flex-start !important;
        max-width: 95% !important;
        min-width: 220px !important;
        background: linear-gradient(135deg, rgba(14, 22, 34, 0.95) 0%, rgba(8, 14, 22, 0.95) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-left: 3px solid {t_cfg['primary']} !important;
        border-radius: 12px 12px 12px 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45), 0 0 10px rgba(0, 229, 255, 0.12) !important;
    }}

    /* Geo-AI avatar positioning & glow */
    div[data-testid="stChatMessage"]:has(.bubble-marker-geoai) > :first-child,
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) > :first-child {{
        margin-right: 0.65rem !important;
        margin-left: 0 !important;
        filter: drop-shadow(0 0 6px {t_cfg['glow']}) !important;
    }}

    /* Geo-AI inner content & header */
    .chat-role-tag.chat-role-geoai {{
        text-align: left !important;
        color: {t_cfg['primary']} !important;
        letter-spacing: 0.08em;
        font-weight: 700;
        text-shadow: 0 0 6px {t_cfg['glow']};
        margin-bottom: 4px;
        font-size: 0.72rem;
    }}
    div[data-testid="stChatMessage"]:has(.bubble-marker-geoai) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) div[data-testid="stChatMessageContent"] {{
        text-align: left !important;
        color: {t_cfg['text_main']} !important;
    }}

    /* Verified sources expander styling inside bubble */
    div[data-testid="stChatMessage"] div[data-testid="stExpander"] {{
        background-color: rgba(6, 12, 18, 0.6) !important;
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 4px !important;
        margin-top: 0.5rem !important;
    }}

    /* Clean spacing for chat paragraphs */
    div[data-testid="stChatMessageContent"] p {{
        margin-bottom: 0.35rem !important;
    }}
    div[data-testid="stChatMessageContent"] p:last-child {{
        margin-bottom: 0 !important;
    }}

    /* Marker helper */
    .bubble-marker-operator, .bubble-marker-geoai {{
        display: none !important;
    }}

    /* HUD Buttons */
    div.stButton > button {{
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        font-weight: 600 !important;
        border-radius: 3px !important;
        border: 1px solid {t_cfg['border_card']} !important;
        background: {t_cfg['bg_card']} !important;
        color: {t_cfg['text_main']} !important;
        transition: all 0.2s ease-in-out !important;
    }}
    div.stButton > button:hover {{
        border-color: {t_cfg['primary']} !important;
        color: {t_cfg['primary']} !important;
        box-shadow: 0 0 12px {t_cfg['glow']} !important;
        background: rgba(0, 0, 0, 0.5) !important;
    }}

    /* Telemetry Metrics */
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        color: {t_cfg['primary']} !important;
        text-shadow: 0 0 10px {t_cfg['glow']} !important;
        font-size: 1.4rem !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {t_cfg['text_dim']} !important;
        font-size: 0.8rem !important;
    }}

    /* Chat Input */
    div[data-testid="stChatInput"] {{
        border: 1px solid {t_cfg['border_card']} !important;
        border-radius: 4px !important;
        background: {t_cfg['bg_card']} !important;
        margin-top: 6px !important;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.4) !important;
    }}
    div[data-testid="stChatInput"]:focus-within {{
        border-color: {t_cfg['primary']} !important;
        box-shadow: 0 0 12px {t_cfg['glow']} !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {t_cfg['text_main']} !important;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
        font-size: 0.85rem !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {t_cfg['text_dim']} !important;
        opacity: 0.75 !important;
    }}

    /* Sidebar Title */
    .sidebar-title {{
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {t_cfg['primary']};
        text-shadow: 0 0 10px {t_cfg['glow']};
        margin-bottom: 2px;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. Session State Initialization
# ==============================================================================
# Maintain conversational history, spatial layers, click tracking, and selected points.
if "messages" not in st.session_state:
    # Initialize chat history with an introductory greeting
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the GeoAI Spatial Intelligence Hub.**\n\n"
                "• **Click map** to select locations of interest (manage in sidebar).\n"
                "• **Query selected points** (e.g., *\"What units are stationed here?\"*).\n"
                "• **Ask general questions** to explore regions (e.g., *\"Show key logistics nodes\"*)."
            )
        }
    ]

if "map_layers" not in st.session_state:
    # List of MapLayer dataclass objects dynamically rendered on Folium
    st.session_state.map_layers = []

if "last_clicked" not in st.session_state:
    # Keeps track of the most recently processed click coordinates (lat, lng) to prevent infinite loops
    st.session_state.last_clicked = None

if "selected_points" not in st.session_state:
    # List of coordinate tuples [(lat, lng), ...] representing all currently selected points
    st.session_state.selected_points = []

if "bounding_box" not in st.session_state:
    # Standard two-point bounding box tuple ((min_lat, min_lng), (max_lat, max_lng)) [lower-left, upper-right]
    st.session_state.bounding_box = None

if "active_point" not in st.session_state:
    # Most recently selected coordinate tuple (lat, lng) for backward compatibility
    st.session_state.active_point = None

if "selection_mode" not in st.session_state or st.session_state.selection_mode not in ["Point Selection", "Bounding Box"]:
    # Point selection mode: 'Point Selection' (default) or 'Bounding Box'
    st.session_state.selection_mode = "Point Selection"

if "keep_points" not in st.session_state:
    # Whether to keep selected points after sending a chat prompt
    st.session_state.keep_points = False

if "openrouter_api_key" not in st.session_state:
    # Initial load from st.secrets or environment variable if available
    import os
    default_key = os.environ.get("OPENROUTER_API_KEY", "")
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            default_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        # Streamlit raises StreamlitSecretNotFoundError when no secrets.toml exists
        pass
    st.session_state.openrouter_api_key = default_key

if "agent_turns" not in st.session_state:
    # List of prior conversational turns formatted for the backend agent
    st.session_state.agent_turns = []

if "last_processed_drawing" not in st.session_state:
    # Tracks the most recently ingested Leaflet Draw shape to prevent rerun loops
    st.session_state.last_processed_drawing = None

if "ignored_drawing" not in st.session_state:
    # Explicitly cleared drawing to ignore on subsequent frames
    st.session_state.ignored_drawing = None

if "last_drawing_time" not in st.session_state:
    # Timestamp of the last completed drawing to suppress mouseup release clicks
    st.session_state.last_drawing_time = 0.0

if "pending_query" not in st.session_state:
    # Stores an active chat query until execution completes so component reruns never drop it
    st.session_state.pending_query = None

if "rag_logs" not in st.session_state:
    # Telemetry and raw JSON exchange logs for RAG agent debugging
    st.session_state.rag_logs = []

if "show_citations" not in st.session_state:
    # Toggle to display inline citation marks ([1], [S2], [L1]) vs plain text
    st.session_state.show_citations = True

if "show_context_pins" not in st.session_state:
    # Toggle to show or hide surrounding context pins (slate blue)
    st.session_state.show_context_pins = True

if "show_uncertain_pins" not in st.session_state:
    # Toggle to show or hide pins with uncertain coordinates (orange)
    st.session_state.show_uncertain_pins = True

if "last_map_bbox" not in st.session_state:
    # Bounding box [west, south, east, north] returned by agent for automatic framing
    st.session_state.last_map_bbox = None

# ==============================================================================
# 2b. Early Map Event Ingestion (Pre-render state synchronization)
# ==============================================================================
# Ingests map clicks, drawings, and deletions before the sidebar and chat mount,
# ensuring all UI panels reflect the newest spatial selection on the initial rerun
# without requiring a second st.rerun() cascade.
_early_map_state = st.session_state.get("geoai_tactical_map")
if _early_map_state and isinstance(_early_map_state, dict):
    _early_drawing = _early_map_state.get("last_active_drawing")
    _early_all_drawings = _early_map_state.get("all_drawings")
    _early_clicked = _early_map_state.get("last_clicked")

    # 1. Handle deletion / trash bin
    if _early_drawing:
        _props = _early_drawing.get("properties") or {}
        _geom = _early_drawing.get("geometry")
        if (_props.get("action") == "delete" or not _geom) and _early_drawing != st.session_state.get("ignored_drawing"):
            st.session_state.ignored_drawing = _early_drawing
            _has_sel = bool(st.session_state.get("selected_points") or st.session_state.get("bounding_box"))
            clear_spatial_selection()
            if not _has_sel and st.session_state.get("map_layers"):
                st.session_state.map_layers = []
            st.session_state.last_drawing_time = time.time()

    elif (
        _early_all_drawings == []
        and st.session_state.get("selection_mode") == "Bounding Box"
        and st.session_state.get("bounding_box")
        and not _early_drawing
    ):
        _has_sel = bool(st.session_state.get("selected_points") or st.session_state.get("bounding_box"))
        clear_spatial_selection()
        if not _has_sel and st.session_state.get("map_layers"):
            st.session_state.map_layers = []
        st.session_state.last_drawing_time = time.time()

    # 2. Handle drawn rectangle or point
    elif _early_drawing and _early_drawing != st.session_state.get("last_processed_drawing") and _early_drawing != st.session_state.get("ignored_drawing"):
        _geom = _early_drawing.get("geometry") or {}
        _gtype = _geom.get("type")
        _coords = _geom.get("coordinates")
        if _gtype == "Polygon" and _coords and len(_coords[0]) >= 3:
            st.session_state.last_processed_drawing = _early_drawing
            st.session_state.ignored_drawing = None
            _ring = _coords[0]
            _lats = [float(pt[1]) for pt in _ring]
            _lngs = [float(pt[0]) for pt in _ring]
            _ll, _ur = to_bounding_box((min(_lats), min(_lngs)), (max(_lats), max(_lngs)))
            st.session_state.selection_mode = "Bounding Box"
            st.session_state.selected_points = [_ll, _ur]
            st.session_state.bounding_box = (_ll, _ur)
            st.session_state.active_point = _ur
            st.session_state.last_drawing_time = time.time()
            if _early_clicked:
                st.session_state.last_clicked = (round(_early_clicked["lat"], 5), round(_early_clicked["lng"], 5))
        elif _gtype == "Point" and _coords and len(_coords) >= 2:
            st.session_state.last_processed_drawing = _early_drawing
            st.session_state.ignored_drawing = None
            _pt = (round(float(_coords[1]), 5), round(float(_coords[0]), 5))
            if st.session_state.selected_points != [_pt]:
                st.session_state.selection_mode = "Point Selection"
                st.session_state.selected_points = [_pt]
                st.session_state.bounding_box = None
                st.session_state.active_point = _pt
                st.session_state.last_drawing_time = time.time()
            if _early_clicked:
                st.session_state.last_clicked = _pt

    # 3. Handle map canvas click
    elif _early_clicked and "lat" in _early_clicked and "lng" in _early_clicked:
        _c_pt = (round(_early_clicked["lat"], 5), round(_early_clicked["lng"], 5))
        _is_recent = (time.time() - st.session_state.get("last_drawing_time", 0.0)) < 2.0
        if _is_recent:
            st.session_state.last_clicked = _c_pt
        elif _c_pt != st.session_state.last_clicked:
            st.session_state.last_clicked = _c_pt
            if st.session_state.selected_points != [_c_pt]:
                st.session_state.selection_mode = "Point Selection"
                st.session_state.selected_points = [_c_pt]
                st.session_state.bounding_box = None
                st.session_state.active_point = _c_pt

# ==============================================================================
# 3. Sidebar Utility Controls & Intelligence Dashboard
# ==============================================================================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🌍 GEO-AI COMMAND</div>", unsafe_allow_html=True)
    st.caption("Multimodal Geocoded RAG Intelligence")
    st.markdown("---")

    # Agent Connection Credentials & Mode HUD Indicator
    st.subheader("🔑 Agent Connection")
    api_key_input = st.text_input(
        "OpenRouter API Key:",
        type="password",
        value=st.session_state.get("openrouter_api_key", ""),
        help="Enter your OpenRouter key (https://openrouter.ai/keys) to connect to the live backend agent. Leave blank for offline mock mode.",
        key="openrouter_key_input"
    )
    if api_key_input != st.session_state.get("openrouter_api_key", ""):
        st.session_state.openrouter_api_key = api_key_input
        st.rerun()

    if st.session_state.get("openrouter_api_key"):
        st.markdown("<div class='geo-badge' style='background:rgba(0,229,255,0.18); color:#00e5ff;'>🟢 LIVE GEORAG AGENT</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='geo-badge' style='background:rgba(255,193,7,0.15); color:#ffc107;'>🟡 OFFLINE SIMULATION (MOCK)</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Utility control buttons & operations (placed above Spatial Targeting)
    st.subheader("🛠️ Operations")

    st.toggle(
        "Show citations",
        key="show_citations",
        help="Numbers in brackets show where each statement comes from: [3] is pin 3 on the map; [S2] and [L1] are paragraphs from the source articles."
    )

    rag_logs_count = len(st.session_state.get("rag_logs", []))
    if st.button(
        f"📋 RAG Debug Logs ({rag_logs_count})",
        use_container_width=True,
        help="Inspect raw JSON request and response payloads, latency, grounding sources, and error logs."
    ):
        show_rag_log_dialog()

    if st.button("🗑️ Clear Chat History", use_container_width=True, help="Reset conversation messages"):
        st.session_state.messages = []
        st.session_state.agent_turns = []
        st.session_state.pending_query = None
        st.session_state.last_map_bbox = None
        st.rerun()

    if st.button("🗺️ Reset Map Layers", use_container_width=True, help="Clear all tactical map overlays and selection"):
        st.session_state.map_layers = []
        st.session_state.last_map_bbox = None
        clear_spatial_selection()
        st.rerun()

    if st.session_state.map_layers:
        total_layers = len(st.session_state.map_layers)
        with st.expander(f"🗺️ Manage Tactical Layers ({total_layers})"):
            if total_layers > 10:
                ctx_count = sum(1 for l in st.session_state.map_layers if l.color == "cadetblue" or (isinstance(l.data, dict) and l.data.get("role") == "context"))
                pri_count = total_layers - ctx_count
                st.caption(f"**Breakdown:** `{pri_count}` Primary / `{ctx_count}` Context")
                c_clr_all, c_clr_ctx = st.columns(2)
                with c_clr_all:
                    if st.button("🗑️ Clear All", key="btn_clear_all_layers", use_container_width=True):
                        st.session_state.map_layers = []
                        st.rerun()
                with c_clr_ctx:
                    if ctx_count > 0 and st.button("🔘 Clear Context", key="btn_clear_ctx_layers", use_container_width=True):
                        st.session_state.map_layers = [l for l in st.session_state.map_layers if not (l.color == "cadetblue" or (isinstance(l.data, dict) and l.data.get("role") == "context"))]
                        st.rerun()

                layer_options = list(range(min(total_layers, 25)))
                sel_to_del = st.selectbox(
                    "Select layer to remove:",
                    options=layer_options,
                    format_func=lambda idx: f"#{idx+1}: {st.session_state.map_layers[idx].name[:40]}",
                    key="sel_layer_to_delete"
                )
                if st.button("🗑️ Delete Selected Layer", key="btn_del_single_layer", use_container_width=True):
                    if sel_to_del < len(st.session_state.map_layers):
                        st.session_state.map_layers.pop(sel_to_del)
                        st.rerun()
            else:
                for l_idx, layer in enumerate(st.session_state.map_layers):
                    col_ly1, col_ly2 = st.columns([3, 1])
                    with col_ly1:
                        st.caption(f"{layer.name}")
                    with col_ly2:
                        if st.button("🗑️", key=f"del_layer_{l_idx}", help=f"Remove layer '{layer.name}'"):
                            st.session_state.map_layers.pop(l_idx)
                            st.rerun()

    st.markdown("---")

    # Display active spatial targeting status and management
    st.subheader("🎯 Spatial Targeting")

    # If Bounding Box is active (from drawing a rectangle)
    if st.session_state.get("bounding_box") is not None:
        ll, ur = st.session_state.bounding_box
        st.markdown(
            f"<div class='geo-badge'>📦 Bounding Box (AOI) Active</div>",
            unsafe_allow_html=True
        )
        col_ll1, col_ll2 = st.columns([3, 1])
        with col_ll1:
            st.markdown(f"**Lower-Left (SW):** `{ll[0]:.4f}°, {ll[1]:.4f}°`")
        with col_ll2:
            if st.button("❌", key="btn_remove_ll", help="Remove Lower-Left corner"):
                st.session_state.selected_points = [ur]
                st.session_state.bounding_box = None
                st.session_state.active_point = ur
                st.session_state.selection_mode = "Point Selection"
                st.rerun()

        col_ur1, col_ur2 = st.columns([3, 1])
        with col_ur1:
            st.markdown(f"**Upper-Right (NE):** `{ur[0]:.4f}°, {ur[1]:.4f}°`")
        with col_ur2:
            if st.button("❌", key="btn_remove_ur", help="Remove Upper-Right corner"):
                st.session_state.selected_points = [ll]
                st.session_state.bounding_box = None
                st.session_state.active_point = ll
                st.session_state.selection_mode = "Point Selection"
                st.rerun()

        lat_span = abs(ur[0] - ll[0])
        lng_span = abs(ur[1] - ll[1])
        st.caption(f"Area Span: ΔLat: `{lat_span:.4f}°` | ΔLng: `{lng_span:.4f}°`")
        st.caption("Type a prompt in the chat to analyze this bounded area.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🗑️ Clear Box", use_container_width=True, help="Remove bounding box selection"):
                clear_spatial_selection()
                st.rerun()
        with col_b2:
            if st.button("🔍 Recon Scan", use_container_width=True, help="Run reconnaissance scan on bounding box"):
                recon_response = dispatch_geo_query(
                    text_prompt=None,
                    selected_point=None,
                    bounding_box=st.session_state.bounding_box
                )
                if recon_response:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Reconnaissance requested for bounding box: Lower-Left ({ll[0]:.4f}°, {ll[1]:.4f}°), Upper-Right ({ur[0]:.4f}°, {ur[1]:.4f}°)"
                    })
                    st.session_state.messages.append(_build_assistant_message(recon_response))
                    st.session_state.map_layers.extend(recon_response.layers)
                    if getattr(recon_response, "map_bbox", None):
                        st.session_state.last_map_bbox = recon_response.map_bbox
                    if not st.session_state.get("keep_points", False):
                        clear_spatial_selection()
                    st.rerun()

        st.checkbox("Keep points selected after sending prompt", key="keep_points")

    # If single Point is active (from clicking the map)
    elif st.session_state.get("selected_points"):
        plat, plng = st.session_state.selected_points[0]
        st.markdown(
            f"<div class='geo-badge'>📍 1 Target Point Selected</div>",
            unsafe_allow_html=True
        )
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.markdown(f"**Point:** `{plat:.4f}°, {plng:.4f}°`")
        with col_p2:
            if st.button("❌", key="btn_remove_single_pt", help="Remove selected point"):
                clear_spatial_selection()
                st.rerun()

        st.caption("Click another location on the map to replace this selection, or enter a prompt below to analyze it.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🗑️ Clear Point", use_container_width=True, help="Remove selected point"):
                clear_spatial_selection()
                st.rerun()
        with col_b2:
            if st.button("🔍 Recon Scan", use_container_width=True, help="Run reconnaissance scan on selected point"):
                pt = st.session_state.selected_points[0] if st.session_state.selected_points else None
                recon_response = dispatch_geo_query(
                    text_prompt=None,
                    selected_point=pt,
                    bounding_box=None
                )
                if recon_response:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Reconnaissance requested for coordinate: {plat:.4f}°, {plng:.4f}°"
                    })
                    st.session_state.messages.append(_build_assistant_message(recon_response))
                    st.session_state.map_layers.extend(recon_response.layers)
                    if getattr(recon_response, "map_bbox", None):
                        st.session_state.last_map_bbox = recon_response.map_bbox
                    if not st.session_state.get("keep_points", False):
                        clear_spatial_selection()
                    st.rerun()

        st.checkbox("Keep point selected after sending prompt", key="keep_points")

    # If neither is active:
    else:
        st.markdown(
            "<div class='geo-status-box'>"
            "🎯 <b>Interactive Targeting Ready:</b><br>"
            "• <b>Click Map:</b> Selects a tactical target point.<br>"
            "• <b>Draw Rectangle:</b> Use the map toolbar (top-left) to bound an Area of Interest."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Session metrics
    st.subheader("📊 Session Telemetry")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Chat Messages", len(st.session_state.messages))
    with col_m2:
        st.metric("Active Layers", len(st.session_state.map_layers))
    st.metric("RAG Query Logs", len(st.session_state.get("rag_logs", [])))

    st.markdown("---")
    st.markdown("### 🧭 Layer Legend")
    st.markdown("- 🔵 **Vivid Blue Pin**: **Primary Place** (discussed & cited in the agent's answer)")
    st.markdown("- 🔘 **Slate / Cadet Blue Pin**: **Context Place** (surrounding background place on the map)")
    st.markdown("- 🟠 **Orange Pin / Zone**: **Uncertain Pin** (?) / Active Bounding Box Area")
    st.markdown("- 🔴 **Red Pin / Star**: Selected Coordinate / Query Focus")
    st.markdown("- 🟣 **Purple Pin / Zone**: Local Reconnaissance Sector")


# ==============================================================================
# 4. Main Application Workspace: Split-Screen Layout
# ==============================================================================
# Tactical labels for the column indicator boxes
col_labels = [
    "💬 Intel Console",
    "🗺️ Geospatial Theater of Operations"
]
# Split screen using streamlit_adjustable_columns with a 1 : 1.5 ratio and fixed key
col1, col2 = adjustable_columns([1, 1.5], labels=col_labels, key="main_workspace_cols")


# ------------------------------------------------------------------------------
# 5. Left Column: Chat Interface & Query Processing
# ------------------------------------------------------------------------------
with col1:
    # Render conversational message history inside a dedicated scroll container with its own scrollbar
    # Stable container height prevents chat scroll area and input box from bouncing (AGENTS.md Rule 3.1)
    chat_container_height = 465
    chat_container = st.container(height=chat_container_height)
    with chat_container:
        for m_idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            is_user = (role == "user")
            avatar = t_cfg["user_avatar"] if is_user else t_cfg["ai_avatar"]
            role_tag = t_cfg["operator_tag"] if is_user else t_cfg["ai_tag"]
            if role == "assistant" and msg.get("is_live"):
                role_tag += " • [LIVE RAG]"
                if msg.get("grounded") is False:
                    role_tag += " • [LLM ONLY]"

            role_class = "chat-role-operator" if is_user else "chat-role-geoai"
            marker_class = "bubble-marker-operator" if is_user else "bubble-marker-geoai"

            with st.chat_message(role, avatar=avatar):
                st.markdown(f"<div class='chat-role-tag {role_class}'><span class='{marker_class}'></span>{role_tag}</div>", unsafe_allow_html=True)
                if role == "assistant" and not st.session_state.get("show_citations", True):
                    st.markdown(msg.get("content_plain") or msg["content"])
                else:
                    st.markdown(msg["content"])

                # Source Evidence Inspection Drawer
                msg_sources = msg.get("sources", [])
                msg_citations = {
                    c.get("pointer"): c
                    for c in msg.get("citations", [])
                    if isinstance(c, dict) and c.get("pointer")
                }
                if msg_sources:
                    with st.expander(f"📑 Verified Sources ({len(msg_sources)})", expanded=False):
                        for s_idx, src_doc in enumerate(msg_sources):
                            cit_meta = msg_citations.get(src_doc, {})
                            cit_id = cit_meta.get("id")
                            cit_article = cit_meta.get("article")
                            cit_section = cit_meta.get("section")
                            cached_meta = st.session_state.get(f"src_meta_{m_idx}_{s_idx}", {})
                            disp_article = _clean_doc_title(cit_article or cached_meta.get("document"))
                            disp_section = _clean_doc_title(cit_section or cached_meta.get("section"))

                            # Resolve document title from local graph if not already cached
                            if not disp_article:
                                passage_data = get_source_text(src_doc)
                                if passage_data and isinstance(passage_data, dict):
                                    st.session_state[f"src_meta_{m_idx}_{s_idx}"] = passage_data
                                    disp_article = _clean_doc_title(passage_data.get("document"))
                                    if not disp_section:
                                        disp_section = _clean_doc_title(passage_data.get("section"))
                                else:
                                    # Cache negative result so missing pointers are never repeatedly queried on reruns
                                    st.session_state[f"src_meta_{m_idx}_{s_idx}"] = {"document": "", "section": ""}

                            if cit_id and disp_article:
                                src_label = f"📄 **[{cit_id}] {disp_article}**" + (f" — *{disp_section}*" if disp_section else "")
                            elif disp_article and disp_article != src_doc:
                                src_label = f"📄 **{disp_article}**" + (f" — *{disp_section}*" if disp_section else "")
                            else:
                                src_label = f"📄 **Source Document #{s_idx + 1}**"

                            is_inspected = bool(st.session_state.get(f"src_content_{m_idx}_{s_idx}"))
                            btn_label = "Hide" if is_inspected else "Inspect"
                            btn_help = "Hide verbatim source passage" if is_inspected else "Inspect verbatim source text"

                            c_doc, c_btn = st.columns([3, 1])
                            with c_doc:
                                st.caption(src_label)
                            with c_btn:
                                if st.button(btn_label, key=f"btn_src_{m_idx}_{s_idx}", help=btn_help):
                                    if is_inspected:
                                        st.session_state.pop(f"src_content_{m_idx}_{s_idx}", None)
                                        st.rerun()
                                    else:
                                        passage = st.session_state.get(f"src_meta_{m_idx}_{s_idx}") or get_source_text(src_doc)
                                        if passage and passage.get("text"):
                                            st.session_state[f"src_meta_{m_idx}_{s_idx}"] = passage
                                            doc_name = _clean_doc_title(passage.get("document") or cit_article or f"Document #{s_idx + 1}")
                                            sec_name = _clean_doc_title(passage.get("section") or cit_section)
                                            para_num = passage.get("paragraph_index")
                                            header_parts = [f"**{doc_name}**"]
                                            if sec_name:
                                                header_parts.append(f"Section: *{sec_name}*")
                                            if para_num is not None:
                                                header_parts.append(f"Paragraph #{para_num + 1}")
                                            header_str = " | ".join(header_parts)
                                            raw_text = passage["text"]
                                            spans = cit_meta.get("text_spans") or cit_meta.get("char_offsets") or passage.get("text_spans") or []
                                            highlighted_text = _highlight_text_spans(raw_text, spans)
                                            st.session_state[f"src_content_{m_idx}_{s_idx}"] = {
                                                "header": header_str,
                                                "text": highlighted_text,
                                                "has_highlight": bool(spans)
                                            }
                                        else:
                                            st.session_state[f"src_content_{m_idx}_{s_idx}"] = {
                                                "header": "Notice",
                                                "text": "(No inline corpus text available for this reference)",
                                                "has_highlight": False
                                            }
                                        st.rerun()

                            if is_inspected:
                                cached_content = st.session_state.get(f"src_content_{m_idx}_{s_idx}")
                                if isinstance(cached_content, dict):
                                    hdr = cached_content.get("header", "")
                                    txt = cached_content.get("text", "")
                                    if cached_content.get("has_highlight"):
                                        st.markdown(
                                            f"<div style='background: rgba(14, 22, 32, 0.7); border-left: 3px solid #00e5ff; padding: 10px 14px; border-radius: 4px; font-size: 0.9em; margin-bottom: 8px; line-height: 1.5;'>"
                                            f"<div style='font-weight: 600; margin-bottom: 6px; color: #38bdf8;'>{hdr}</div>"
                                            f"<div>{txt}</div>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.info(f"{hdr}\n\n{txt}")
                                else:
                                    st.info(str(cached_content))

                                col_sp, col_hd = st.columns([4.2, 1.8])
                                with col_hd:
                                    if st.button("✖ Hide Text", key=f"btn_hide_{m_idx}_{s_idx}", help="Hide this passage"):
                                        st.session_state.pop(f"src_content_{m_idx}_{s_idx}", None)
                                        st.rerun()

                # Upgrade 5: Regional Distribution Breakdown
                msg_regions = msg.get("regions", {})
                if role == "assistant" and msg_regions and isinstance(msg_regions, dict):
                    with st.expander(f"🌍 Regional Distribution ({len(msg_regions)} regions)", expanded=False):
                        for reg_name, count in sorted(msg_regions.items(), key=lambda x: str(x[0])):
                            st.markdown(f"- **{reg_name}**: `{count}` places/events")

                # Upgrade 4: Follow-up Suggestions & Did-You-Mean
                if role == "assistant" and msg.get("did_you_mean"):
                    st.info(f"💡 *Did you mean:* **{msg['did_you_mean']}**?")

                msg_follow_ups = msg.get("follow_ups", [])
                if role == "assistant" and msg_follow_ups and m_idx == len(st.session_state.messages) - 1:
                    st.markdown("<div style='margin-top: 6px; margin-bottom: 4px; font-size: 0.82rem; color: #94a3b8;'>💡 <b>Suggested Follow-ups:</b></div>", unsafe_allow_html=True)
                    fu_cols = st.columns(min(len(msg_follow_ups), 3))
                    for f_idx, follow_q in enumerate(msg_follow_ups[:3]):
                        with fu_cols[f_idx]:
                            if st.button(f"🔍 {follow_q}", key=f"btn_fu_{m_idx}_{f_idx}", help=f"Ask: {follow_q}", use_container_width=True):
                                st.session_state.pending_query = {
                                    "text_prompt": follow_q,
                                    "selected_point": None,
                                    "bounding_box": None,
                                }
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": follow_q
                                })
                                st.rerun()

    # Option 3: Compact docked contextual payload pill (directly above chat input)
    if st.session_state.get("selected_points"):
        is_bbox = st.session_state.get("bounding_box") is not None
        if is_bbox:
            b_ll, b_ur = st.session_state.bounding_box
            badge_html = (
                f"<div class='geo-docked-pill geo-docked-pill-stacked' title='Bounding Box AOI: Lower-Left ({b_ll[0]:.4f}°, {b_ll[1]:.4f}°) | Upper-Right ({b_ur[0]:.4f}°, {b_ur[1]:.4f}°)'>"
                f"<span class='geo-pill-icon'>📦</span>"
                f"<div class='geo-pill-stack'>"
                f"<div class='geo-pill-row'><span class='geo-pill-sublabel'>SW:</span><span class='geo-pill-coord'>{b_ll[0]:.4f}°, {b_ll[1]:.4f}°</span></div>"
                f"<div class='geo-pill-row'><span class='geo-pill-sublabel'>NE:</span><span class='geo-pill-coord'>{b_ur[0]:.4f}°, {b_ur[1]:.4f}°</span></div>"
                f"</div>"
                f"</div>"
            )
            clear_btn_label = "❌ Clear"
            recon_btn_label = "🔍 Recon"
        else:
            plat, plng = st.session_state.selected_points[0]
            badge_html = (
                f"<div class='geo-docked-pill' title='Attached Tactical Target: ({plat:.4f}°, {plng:.4f}°)'>"
                f"<span class='geo-pill-icon'>📍</span>"
                f"<span class='geo-pill-label'>TARGET:</span>"
                f"<span class='geo-pill-coord'>({plat:.4f}°, {plng:.4f}°)</span>"
                f"</div>"
            )
            clear_btn_label = "❌ Clear"
            recon_btn_label = "🔍 Recon"

        col_pill, col_clr, col_rec = st.columns([5.2, 2.1, 2.7])
        with col_pill:
            st.markdown(badge_html, unsafe_allow_html=True)
        with col_clr:
            if st.button(clear_btn_label, key="clear_chat_pts", use_container_width=True, help="Clear active spatial selection"):
                clear_spatial_selection()
                st.rerun()
        with col_rec:
            if st.button(recon_btn_label, key="run_chat_recon", use_container_width=True, help="Run instant reconnaissance scan"):
                active_pt = None if is_bbox else (st.session_state.selected_points[0] if st.session_state.get("selected_points") else None)
                active_bbox = st.session_state.get("bounding_box") if is_bbox else None

                recon_resp = dispatch_geo_query(
                    text_prompt=None,
                    selected_point=active_pt,
                    bounding_box=active_bbox
                )
                if recon_resp:
                    if is_bbox and st.session_state.get("bounding_box"):
                        b_ll, b_ur = st.session_state.bounding_box
                        user_msg = f"Reconnaissance requested for bounding box: Lower-Left ({b_ll[0]:.4f}°, {b_ll[1]:.4f}°), Upper-Right ({b_ur[0]:.4f}°, {b_ur[1]:.4f}°)"
                    else:
                        plat, plng = st.session_state.selected_points[0]
                        user_msg = f"Reconnaissance requested for coordinate: ({plat:.4f}°, {plng:.4f}°)"

                    st.session_state.messages.append({"role": "user", "content": user_msg})
                    st.session_state.messages.append(_build_assistant_message(recon_resp))
                    st.session_state.map_layers.extend(recon_resp.layers)
                    if getattr(recon_resp, "map_bbox", None):
                        st.session_state.last_map_bbox = recon_resp.map_bbox
                    if not st.session_state.get("keep_points", False):
                        clear_spatial_selection()
                    st.rerun()

    # Chat input box for user queries
    user_prompt = st.chat_input("Type a question (e.g. 'What units are stationed here?')...")

    if user_prompt:
        # 1. Append user's message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # 2. Immediately render user's question into chat_container so user sees it right away
        with chat_container:
            with st.chat_message("user", avatar=t_cfg["user_avatar"]):
                st.markdown(f"<div class='chat-role-tag chat-role-operator'><span class='bubble-marker-operator'></span>{t_cfg['operator_tag']}</div>", unsafe_allow_html=True)
                st.markdown(user_prompt)

        # 3. Queue pending query in session state so component mount/sync reruns never drop it
        is_bbox = st.session_state.get("bounding_box") is not None
        active_pt = None if is_bbox else (st.session_state.selected_points[0] if st.session_state.get("selected_points") else None)
        active_bbox = st.session_state.get("bounding_box") if is_bbox else None
        st.session_state.pending_query = {
            "text_prompt": user_prompt,
            "selected_point": active_pt,
            "bounding_box": active_bbox,
        }


# ------------------------------------------------------------------------------
# 6. Right Column: Folium Map Rendering & Click Interception
# ------------------------------------------------------------------------------
with col2:
    # Center coordinates of Ukraine: latitude ~48.3794, longitude ~31.1656
    ukraine_center = [48.3794, 31.1656]

    # Initialize the Folium Map instance without a default tile layer to prevent layer conflicts
    m = folium.Map(
        location=ukraine_center,
        zoom_start=6,
        tiles=None,
        control_scale=True,
        attributionControl=False
    )

    # Auto-Zoom to Query Bounding Box (Upgrade 3 & AGENTS.md Rule 4.2)
    if st.session_state.get("last_map_bbox"):
        b = st.session_state["last_map_bbox"]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                w, s, e, n = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                m.fit_bounds([[s, w], [n, e]], padding=(30, 30))
            except Exception as fit_err:
                print(f"[Auto-Zoom Warning] Could not fit bounds to {b}: {fit_err}")
    # Inject CSS for attribution hiding, dark-mode tile filter, and environment-adaptive HUD controls
    m.get_root().header.add_child(
        folium.Element(
            """<style>
            .leaflet-control-attribution { display: none !important; }
            .voyager-dark-tiles { filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%) !important; }

            /* Smooth transition when switching basemaps */
            .leaflet-bar, .leaflet-bar a, .leaflet-control-layers, .leaflet-control-scale-line, .leaflet-draw-actions {
                transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease !important;
            }

            /* ========================================================================== */
            /* 1. DARK THEME (Tactical GeoAI Defense C2 - for Dark / Satellite Basemaps)   */
            /* ========================================================================== */
            .geoai-map-dark .leaflet-bar,
            .leaflet-container:not(.geoai-map-light) .leaflet-bar,
            .geoai-map-dark .leaflet-draw-toolbar,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-toolbar {
                background: rgba(14, 22, 32, 0.92) !important;
                border: 1px solid rgba(0, 229, 255, 0.40) !important;
                border-radius: 4px !important;
                box-shadow: 0 0 10px rgba(0, 229, 255, 0.25), 0 4px 12px rgba(0, 0, 0, 0.6) !important;
                overflow: hidden !important;
            }

            .geoai-map-dark .leaflet-bar a,
            .leaflet-container:not(.geoai-map-light) .leaflet-bar a,
            .geoai-map-dark .leaflet-draw-toolbar a,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-toolbar a {
                background-color: rgba(14, 22, 32, 0.92) !important;
                border-bottom: 1px solid rgba(0, 229, 255, 0.22) !important;
                color: #00e5ff !important;
            }

            .geoai-map-dark .leaflet-bar a:hover,
            .leaflet-container:not(.geoai-map-light) .leaflet-bar a:hover,
            .geoai-map-dark .leaflet-draw-toolbar a:hover,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-toolbar a:hover {
                background-color: rgba(0, 229, 255, 0.22) !important;
                color: #ffffff !important;
                box-shadow: inset 0 0 8px rgba(0, 229, 255, 0.5) !important;
            }

            /* Zoom In / Out Buttons */
            .geoai-map-dark .leaflet-control-zoom a,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-zoom a {
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 16px !important;
                font-weight: 700 !important;
                line-height: 28px !important;
                text-shadow: 0 0 6px rgba(0, 229, 255, 0.7) !important;
            }

            /* Tactical SVGs for Draw Tools (Rectangle & Trash) in Dark Theme */
            .geoai-map-dark .leaflet-draw-draw-rectangle,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-draw-rectangle {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2300e5ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2' ry='2'%3E%3C/rect%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 16px 16px !important;
            }

            .geoai-map-dark .leaflet-draw-edit-remove,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-edit-remove {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f87171' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'%3E%3C/polyline%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'%3E%3C/path%3E%3Cline x1='10' y1='11' x2='10' y2='17'%3E%3C/line%3E%3Cline x1='14' y1='11' x2='14' y2='17'%3E%3C/line%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 16px 16px !important;
            }

            /* Layer Control in Dark Theme */
            .geoai-map-dark .leaflet-control-layers,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers {
                background: rgba(14, 22, 32, 0.92) !important;
                border: 1px solid rgba(0, 229, 255, 0.40) !important;
                border-radius: 4px !important;
                box-shadow: 0 0 10px rgba(0, 229, 255, 0.25), 0 4px 12px rgba(0, 0, 0, 0.6) !important;
            }

            .geoai-map-dark .leaflet-control-layers-toggle,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers-toggle {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%2300e5ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='12 2 2 7 12 12 22 7 12 2'%3E%3C/polygon%3E%3Cpolyline points='2 17 12 22 22 17'%3E%3C/polyline%3E%3Cpolyline points='2 12 12 17 22 12'%3E%3C/polyline%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 20px 20px !important;
            }

            .geoai-map-dark .leaflet-control-pin-filter-toggle,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-pin-filter-toggle {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%2300e5ff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z'%3E%3C/path%3E%3Ccircle cx='12' cy='10' r='3'%3E%3C/circle%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 20px 20px !important;
            }

            .pin-filter-header {
                color: #00e5ff !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                letter-spacing: 0.07em !important;
                margin-bottom: 6px !important;
                padding-bottom: 4px !important;
                border-bottom: 1px solid rgba(0, 229, 255, 0.25) !important;
                text-transform: uppercase !important;
            }

            .geoai-map-dark .leaflet-control-layers-expanded,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers-expanded {
                background: rgba(14, 22, 32, 0.96) !important;
                border: 1px solid rgba(0, 229, 255, 0.45) !important;
                border-radius: 4px !important;
                box-shadow: 0 0 16px rgba(0, 229, 255, 0.3), 0 8px 24px rgba(0, 0, 0, 0.7) !important;
                color: #e2e8f0 !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 11.5px !important;
                padding: 8px 12px !important;
            }
            .geoai-map-dark .leaflet-control-layers-separator,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers-separator {
                border-top: 1px solid rgba(0, 229, 255, 0.25) !important;
            }
            .geoai-map-dark .leaflet-control-layers-base label,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers-base label,
            .geoai-map-dark .leaflet-control-layers-overlays label,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers-overlays label {
                display: flex !important;
                align-items: center !important;
                gap: 6px !important;
                cursor: pointer !important;
                color: #cbd5e1 !important;
                margin-bottom: 3px !important;
                font-size: 11px !important;
            }
            .geoai-map-dark .leaflet-control-layers-base label:hover,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-layers-base label:hover {
                color: #00e5ff !important;
            }
            .geoai-map-dark input[type="radio"],
            .leaflet-container:not(.geoai-map-light) input[type="radio"],
            .geoai-map-dark input[type="checkbox"],
            .leaflet-container:not(.geoai-map-light) input[type="checkbox"] {
                accent-color: #00e5ff !important;
            }

            /* Scale Line in Dark Theme */
            .geoai-map-dark .leaflet-control-scale-line,
            .leaflet-container:not(.geoai-map-light) .leaflet-control-scale-line {
                background: rgba(14, 22, 32, 0.88) !important;
                border: 2px solid #00e5ff !important;
                border-top: none !important;
                color: #00e5ff !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                letter-spacing: 0.05em !important;
                text-shadow: 0 0 6px rgba(0, 229, 255, 0.6) !important;
                box-shadow: 0 0 8px rgba(0, 229, 255, 0.25) !important;
                padding: 2px 6px !important;
                border-radius: 0 0 2px 2px !important;
            }

            /* Draw Actions Sub-menu in Dark Theme */
            .geoai-map-dark .leaflet-draw-actions,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-actions {
                background: rgba(14, 22, 32, 0.95) !important;
                border: 1px solid rgba(0, 229, 255, 0.40) !important;
                border-radius: 4px !important;
                box-shadow: 0 0 10px rgba(0, 229, 255, 0.25) !important;
            }
            .geoai-map-dark .leaflet-draw-actions a,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-actions a {
                background-color: transparent !important;
                color: #00e5ff !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 11px !important;
                font-weight: 600 !important;
            }
            .geoai-map-dark .leaflet-draw-actions a:hover,
            .leaflet-container:not(.geoai-map-light) .leaflet-draw-actions a:hover {
                background-color: rgba(0, 229, 255, 0.20) !important;
                color: #ffffff !important;
            }

            /* Popups & Tooltips in Dark Theme */
            .geoai-map-dark .leaflet-popup-content-wrapper,
            .leaflet-container:not(.geoai-map-light) .leaflet-popup-content-wrapper {
                background: rgba(14, 22, 32, 0.96) !important;
                border: 1px solid rgba(0, 229, 255, 0.40) !important;
                color: #e2e8f0 !important;
                box-shadow: 0 0 14px rgba(0, 229, 255, 0.3) !important;
            }
            .geoai-map-dark .leaflet-popup-tip,
            .leaflet-container:not(.geoai-map-light) .leaflet-popup-tip {
                background: rgba(14, 22, 32, 0.96) !important;
            }
            .geoai-map-dark .leaflet-popup-close-button,
            .leaflet-container:not(.geoai-map-light) .leaflet-popup-close-button {
                color: #00e5ff !important;
            }
            .geoai-map-dark .leaflet-tooltip,
            .leaflet-container:not(.geoai-map-light) .leaflet-tooltip {
                background: rgba(14, 22, 32, 0.92) !important;
                border: 1px solid rgba(0, 229, 255, 0.40) !important;
                color: #00e5ff !important;
                font-family: 'JetBrains Mono', monospace !important;
                box-shadow: 0 0 8px rgba(0, 229, 255, 0.25) !important;
            }

            /* ========================================================================== */
            /* 2. LIGHT THEME (For Light Navigation, Topographic, and OSM Basemaps)       */
            /* ========================================================================== */
            .geoai-map-light .leaflet-bar,
            .geoai-map-light .leaflet-draw-toolbar {
                background: rgba(248, 250, 252, 0.95) !important;
                border: 1px solid rgba(15, 23, 42, 0.28) !important;
                border-radius: 4px !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.18) !important;
                overflow: hidden !important;
            }

            .geoai-map-light .leaflet-bar a,
            .geoai-map-light .leaflet-draw-toolbar a {
                background-color: rgba(248, 250, 252, 0.95) !important;
                border-bottom: 1px solid rgba(15, 23, 42, 0.15) !important;
                color: #0f172a !important;
            }

            .geoai-map-light .leaflet-bar a:hover,
            .geoai-map-light .leaflet-draw-toolbar a:hover {
                background-color: #0284c7 !important;
                color: #ffffff !important;
            }

            .geoai-map-light .leaflet-control-zoom a {
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 16px !important;
                font-weight: 700 !important;
                line-height: 28px !important;
                color: #0f172a !important;
            }
            .geoai-map-light .leaflet-control-zoom a:hover {
                color: #ffffff !important;
            }

            /* SVGs for Draw Tools in Light Theme */
            .geoai-map-light .leaflet-draw-draw-rectangle {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%230f172a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2' ry='2'%3E%3C/rect%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 16px 16px !important;
            }

            .geoai-map-light .leaflet-draw-edit-remove {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23dc2626' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'%3E%3C/polyline%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'%3E%3C/path%3E%3Cline x1='10' y1='11' x2='10' y2='17'%3E%3C/line%3E%3Cline x1='14' y1='11' x2='14' y2='17'%3E%3C/line%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 16px 16px !important;
            }

            /* Layer Control in Light Theme */
            .geoai-map-light .leaflet-control-layers {
                background: rgba(248, 250, 252, 0.95) !important;
                border: 1px solid rgba(15, 23, 42, 0.28) !important;
                border-radius: 4px !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.18) !important;
            }

            .geoai-map-light .leaflet-control-layers-toggle {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%230f172a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='12 2 2 7 12 12 22 7 12 2'%3E%3C/polygon%3E%3Cpolyline points='2 17 12 22 22 17'%3E%3C/polyline%3E%3Cpolyline points='2 12 12 17 22 12'%3E%3C/polyline%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 20px 20px !important;
            }

            .geoai-map-light .leaflet-control-pin-filter-toggle {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%230f172a' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z'%3E%3C/path%3E%3Ccircle cx='12' cy='10' r='3'%3E%3C/circle%3E%3C/svg%3E") !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-size: 20px 20px !important;
            }

            .geoai-map-light .pin-filter-header {
                color: #0284c7 !important;
                border-bottom: 1px solid rgba(15, 23, 42, 0.15) !important;
            }

            .geoai-map-light .leaflet-control-layers-expanded {
                background: rgba(255, 255, 255, 0.96) !important;
                border: 1px solid rgba(15, 23, 42, 0.25) !important;
                border-radius: 4px !important;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
                color: #0f172a !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 11.5px !important;
                padding: 8px 12px !important;
            }
            .geoai-map-light .leaflet-control-layers-separator {
                border-top: 1px solid rgba(15, 23, 42, 0.15) !important;
            }
            .geoai-map-light .leaflet-control-layers-base label,
            .geoai-map-light .leaflet-control-layers-overlays label {
                display: flex !important;
                align-items: center !important;
                gap: 6px !important;
                cursor: pointer !important;
                color: #1e293b !important;
                margin-bottom: 3px !important;
                font-size: 11px !important;
            }
            .geoai-map-light .leaflet-control-layers-base label:hover {
                color: #0284c7 !important;
            }
            .geoai-map-light input[type="radio"],
            .geoai-map-light input[type="checkbox"] {
                accent-color: #0284c7 !important;
            }

            /* Scale Line in Light Theme */
            .geoai-map-light .leaflet-control-scale-line {
                background: rgba(255, 255, 255, 0.88) !important;
                border: 2px solid #0f172a !important;
                border-top: none !important;
                color: #0f172a !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                letter-spacing: 0.05em !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1) !important;
                padding: 2px 6px !important;
                border-radius: 0 0 2px 2px !important;
            }

            /* Draw Actions Sub-menu in Light Theme */
            .geoai-map-light .leaflet-draw-actions {
                background: rgba(248, 250, 252, 0.95) !important;
                border: 1px solid rgba(15, 23, 42, 0.25) !important;
                border-radius: 4px !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
            }
            .geoai-map-light .leaflet-draw-actions a {
                background-color: transparent !important;
                color: #0f172a !important;
                font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
                font-size: 11px !important;
                font-weight: 600 !important;
            }
            .geoai-map-light .leaflet-draw-actions a:hover {
                background-color: #0284c7 !important;
                color: #ffffff !important;
            }

            /* Popups & Tooltips in Light Theme */
            .geoai-map-light .leaflet-popup-content-wrapper {
                background: rgba(255, 255, 255, 0.96) !important;
                border: 1px solid rgba(15, 23, 42, 0.25) !important;
                color: #0f172a !important;
            }
            .geoai-map-light .leaflet-popup-tip {
                background: rgba(255, 255, 255, 0.96) !important;
            }
            .geoai-map-light .leaflet-popup-close-button {
                color: #0f172a !important;
            }
            .geoai-map-light .leaflet-tooltip {
                background: rgba(255, 255, 255, 0.92) !important;
                border: 1px solid rgba(15, 23, 42, 0.25) !important;
                color: #0f172a !important;
                font-family: 'JetBrains Mono', monospace !important;
            }
            </style>"""
        )
    )

    # CARTO API Key integration
    CARTO_API_KEY = "cb1_3wyp_1_6945a000f4bccf031f5aec03"

    # 1. Primary Default Basemap: CartoDB Voyager Dark (Detailed Navigation with dense labels & POIs in dark mode)
    folium.TileLayer(
        tiles=f"https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>",
        name="CartoDB Voyager Dark (Detailed Nav)",
        overlay=False,
        control=True,
        show=True,
        className="voyager-dark-tiles"
    ).add_to(m)

    # 2. CartoDB Dark Matter (Tactical)
    folium.TileLayer(
        tiles=f"https://{{s}}.basemaps.cartocdn.com/rastertiles/dark_all/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>",
        name="CartoDB Dark Matter (Tactical)",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # 3. CartoDB Voyager (Navigation Basemap authenticated with CARTO key)
    folium.TileLayer(
        tiles=f"https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>",
        name="CartoDB Voyager (Nav)",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # 4. Esri World Dark Gray Canvas (Official Defense C2 dark basemap, 100% free, no key required)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, FAO, NOAA, USGS, &copy; OpenStreetMap contributors",
        name="Esri Dark Gray Canvas",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # 5. High-resolution Satellite Imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri Satellite Imagery",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # 6. Topographic Basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, Intermap, USGS, METI, OpenStreetMap contributors, and the GIS User Community",
        name="Esri Topographic",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # 7. OpenStreetMap (Standard Fallback)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        overlay=False,
        control=True,
        show=False
    ).add_to(m)

    # 8. Interactive Area-of-Interest (AOI) Draw Suite
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "polygon": False,
            "marker": False,  # Canvas click directly selects target points
            "rectangle": {
                "shapeOptions": {
                    "color": t_cfg["primary"],
                    "fillColor": t_cfg["primary"],
                    "fillOpacity": 0.2,
                    "weight": 2,
                    "dashArray": "6, 6"
                }
            },
        },
        edit_options={"edit": False, "remove": True}  # Enable map toolbar trash bin for instant removal
    ).add_to(m)

    # Dynamic overlays feature group: prevents map reloads / flashes on selection
    # Dynamic overlays feature groups: separate selection from tactical overlays
    # 1. selection_fg: active target point or bounding box AOI (mapped to window.feature_group[0])
    # 2. tactical_layers_fg: persistent query overlays (mapped to window.feature_group[1])
    selection_fg = folium.FeatureGroup(name="Active Selection", control=False)
    tactical_layers_fg = folium.FeatureGroup(name="Tactical Overlays", control=False)
    # Upgrade 2: MarkerCluster for background context places (AGENTS.md Rule 4.1)
    context_cluster = MarkerCluster(name="Context Cluster", control=False).add_to(tactical_layers_fg)

    # Render visually distinct marker(s) or bounding box for all currently selected points
    if st.session_state.get("bounding_box"):
        b_ll, b_ur = st.session_state.bounding_box
        # 1. Lower-Left (SW) Marker
        folium.Marker(
            location=[b_ll[0], b_ll[1]],
            tooltip=f"★ Bounding Box Lower-Left (SW): {b_ll[0]:.4f}°, {b_ll[1]:.4f}°",
            popup=folium.Popup(
                f"<div style='font-family:monospace; background-color:{t_cfg['bg_card']}; "
                f"color:{t_cfg['primary']}; padding:8px; border:1px solid {t_cfg['border_card']}; border-radius:4px;'>"
                f"<b>★ BBOX LOWER-LEFT (SW)</b><br>"
                f"LAT: <code>{b_ll[0]:.5f}°</code><br>"
                f"LNG: <code>{b_ll[1]:.5f}°</code><br>"
                f"<span style='color:{t_cfg['text_dim']}; font-size:0.8rem;'>Protocol: {t_cfg['name']}</span>"
                f"</div>",
                max_width=250
            ),
            icon=folium.Icon(color=t_cfg["folium_marker_color"], icon="arrow-down", prefix="glyphicon")
        ).add_to(selection_fg)

        # 2. Upper-Right (NE) Marker
        folium.Marker(
            location=[b_ur[0], b_ur[1]],
            tooltip=f"★ Bounding Box Upper-Right (NE): {b_ur[0]:.4f}°, {b_ur[1]:.4f}°",
            popup=folium.Popup(
                f"<div style='font-family:monospace; background-color:{t_cfg['bg_card']}; "
                f"color:{t_cfg['primary']}; padding:8px; border:1px solid {t_cfg['border_card']}; border-radius:4px;'>"
                f"<b>★ BBOX UPPER-RIGHT (NE)</b><br>"
                f"LAT: <code>{b_ur[0]:.5f}°</code><br>"
                f"LNG: <code>{b_ur[1]:.5f}°</code><br>"
                f"<span style='color:{t_cfg['text_dim']}; font-size:0.8rem;'>Protocol: {t_cfg['name']}</span>"
                f"</div>",
                max_width=250
            ),
            icon=folium.Icon(color=t_cfg["folium_marker_color"], icon="arrow-up", prefix="glyphicon")
        ).add_to(selection_fg)

        # 3. Visual Tactical Bounding Box Rectangle
        folium.Rectangle(
            bounds=[[b_ll[0], b_ll[1]], [b_ur[0], b_ur[1]]],
            color=t_cfg["primary"],
            weight=2,
            dash_array="6, 6",
            fill=True,
            fill_color=t_cfg["primary"],
            fill_opacity=0.18,
            tooltip=f"★ Bounding Box: [{b_ll[0]:.4f}°, {b_ll[1]:.4f}°] to [{b_ur[0]:.4f}°, {b_ur[1]:.4f}°]",
            popup=folium.Popup(
                f"<div style='font-family:monospace; background-color:{t_cfg['bg_card']}; "
                f"color:{t_cfg['primary']}; padding:8px; border:1px solid {t_cfg['border_card']}; border-radius:4px;'>"
                f"<b>★ TWO-POINT BOUNDING BOX</b><br>"
                f"<b>Lower-Left (SW):</b> <code>{b_ll[0]:.4f}°, {b_ll[1]:.4f}°</code><br>"
                f"<b>Upper-Right (NE):</b> <code>{b_ur[0]:.4f}°, {b_ur[1]:.4f}°</code><br>"
                f"<span style='color:{t_cfg['text_dim']}; font-size:0.8rem;'>Ready for tactical query</span>"
                f"</div>",
                max_width=260
            )
        ).add_to(selection_fg)

    elif st.session_state.get("selected_points"):
        s_lat, s_lng = st.session_state.selected_points[0]
        popup_html = (
            f"<div style='font-family:monospace; background-color:{t_cfg['bg_card']}; "
            f"color:{t_cfg['primary']}; padding:8px; border:1px solid {t_cfg['border_card']}; border-radius:4px;'>"
            f"<b>★ TARGET POINT</b><br>"
            f"LAT: <code>{s_lat:.5f}°</code><br>"
            f"LNG: <code>{s_lng:.5f}°</code><br>"
            f"<span style='color:{t_cfg['text_dim']}; font-size:0.8rem;'>Protocol: {t_cfg['name']}</span>"
            f"</div>"
        )
        folium.Marker(
            location=[s_lat, s_lng],
            tooltip=f"★ Target Point ({s_lat:.4f}°, {s_lng:.4f}°)",
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=t_cfg["folium_marker_color"], icon=t_cfg["folium_marker_icon"], prefix="glyphicon")
        ).add_to(selection_fg)

    # Render all persistent MapLayer objects from session state
    for layer in st.session_state.map_layers:
        try:
            if layer.layer_type == "marker":
                is_context = (
                    layer.color == "cadetblue"
                    or (isinstance(layer.data, dict) and layer.data.get("role") == "context")
                    or "context" in (layer.name or "").lower()
                )
                target_parent = context_cluster if is_context else tactical_layers_fg
                folium.Marker(
                    location=[layer.data["lat"], layer.data["lng"]],
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300),
                    icon=folium.Icon(color=layer.color, icon="info-sign", prefix="glyphicon")
                ).add_to(target_parent)

            elif layer.layer_type == "circle":
                folium.Circle(
                    location=[layer.data["lat"], layer.data["lng"]],
                    radius=layer.data.get("radius", 10000),
                    color=layer.color,
                    fill=True,
                    fill_opacity=0.25,
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300)
                ).add_to(tactical_layers_fg)

            elif layer.layer_type == "polygon":
                folium.Polygon(
                    locations=layer.data.get("coordinates", []),
                    color=layer.color,
                    fill=True,
                    fill_opacity=0.3,
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300)
                ).add_to(tactical_layers_fg)

            elif layer.layer_type in ["rectangle", "box"]:
                folium.Rectangle(
                    bounds=layer.data.get("bounds", layer.data.get("coordinates", [])),
                    color=layer.color,
                    fill=True,
                    fill_opacity=layer.data.get("fill_opacity", 0.25),
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300)
                ).add_to(tactical_layers_fg)

            elif layer.layer_type == "geojson":
                # Upgrade 1: GeoJSON administrative boundary outline rendering
                geo_color = layer.color if layer.color in ["#00e5ff", "#0284c7", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7"] else t_cfg["primary"]
                folium.GeoJson(
                    layer.data,
                    name=layer.name,
                    style_function=lambda feature, c=geo_color: {
                        "fillColor": c,
                        "color": c,
                        "weight": 2,
                        "fillOpacity": 0.15,
                        "dashArray": "4, 4"
                    },
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300) if layer.popup else None
                ).add_to(tactical_layers_fg)

        except Exception as render_err:
            # Prevent map crash if individual layer data is malformed
            print(f"[Map Render Warning] Could not render layer '{layer.name}': {render_err}")

    # Add layer switcher control
    folium.LayerControl(position="topright").add_to(m)

    # Attach trash bin hook to integrate Leaflet Draw's trash icon with GeoAI selection removal
    MapTrashBinHook().add_to(m)

    # Render the interactive map component inside Streamlit
    # Uses returned_objects to ensure 0ms client-side pans and zooms with zero reruns (AGENTS.md Rule 1.1)
    map_data = st_folium(
        m,
        feature_group_to_add=[selection_fg, tactical_layers_fg],
        returned_objects=["last_clicked", "last_active_drawing", "all_drawings"],
        key="geoai_tactical_map",
        height=520,
        use_container_width=True,
        return_on_hover=False
    )

    # --------------------------------------------------------------------------
    # 1. Interactive Drawing & Deletion Interception (Leaflet Draw Tool & Trash Bin)
    # --------------------------------------------------------------------------
    active_drawing = map_data.get("last_active_drawing") if map_data else None
    all_drawings = map_data.get("all_drawings") if map_data else None

    # Handle deletion events (trash bin icon clicked on map tools or shape deleted)
    if not st.session_state.get("pending_query") and active_drawing:
        props = active_drawing.get("properties") or {}
        geom = active_drawing.get("geometry")
        if (
            (props.get("action") == "delete" or not geom)
            and active_drawing != st.session_state.get("ignored_drawing")
        ):
            st.session_state.ignored_drawing = active_drawing
            has_selection = bool(st.session_state.get("selected_points") or st.session_state.get("bounding_box"))
            clear_spatial_selection()
            if not has_selection and st.session_state.get("map_layers"):
                st.session_state.map_layers = []
            st.session_state.last_drawing_time = time.time()
            st.rerun()

    # Also detect when all_drawings becomes empty while a bounding box was drawn
    if (
        not st.session_state.get("pending_query")
        and all_drawings == []
        and st.session_state.get("selection_mode") == "Bounding Box"
        and st.session_state.get("bounding_box")
        and not active_drawing
    ):
        has_selection = bool(st.session_state.get("selected_points") or st.session_state.get("bounding_box"))
        clear_spatial_selection()
        if not has_selection and st.session_state.get("map_layers"):
            st.session_state.map_layers = []
        st.session_state.last_drawing_time = time.time()
        st.rerun()

    # Process newly drawn shape (Rectangle or Marker) only when no chat query is in flight
    if (
        not st.session_state.get("pending_query")
        and active_drawing
        and active_drawing != st.session_state.get("last_processed_drawing")
        and active_drawing != st.session_state.get("ignored_drawing")
    ):
        geom = active_drawing.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon" and coords and len(coords[0]) >= 3:
            st.session_state.last_processed_drawing = active_drawing
            st.session_state.ignored_drawing = None

            # Drawn Rectangle -> Extract bounding envelope
            ring = coords[0]
            lats = [float(pt[1]) for pt in ring]
            lngs = [float(pt[0]) for pt in ring]
            lower_left, upper_right = to_bounding_box(
                (min(lats), min(lngs)),
                (max(lats), max(lngs))
            )
            needs_rerun = (
                st.session_state.get("selection_mode") != "Bounding Box"
                or st.session_state.get("bounding_box") != (lower_left, upper_right)
            )
            st.session_state.selection_mode = "Bounding Box"
            st.session_state.selected_points = [lower_left, upper_right]
            st.session_state.bounding_box = (lower_left, upper_right)
            st.session_state.active_point = upper_right
            # Mark drawing timestamp to absorb immediate mouseup canvas click
            st.session_state.last_drawing_time = time.time()
            if map_data and map_data.get("last_clicked"):
                cd = map_data["last_clicked"]
                st.session_state.last_clicked = (round(cd["lat"], 5), round(cd["lng"], 5))
            if needs_rerun:
                st.rerun()

        elif gtype == "Point" and coords and len(coords) >= 2:
            st.session_state.last_processed_drawing = active_drawing
            st.session_state.ignored_drawing = None

            lat = round(float(coords[1]), 5)
            lng = round(float(coords[0]), 5)
            pt = (lat, lng)
            needs_rerun = st.session_state.selected_points != [pt]
            st.session_state.selection_mode = "Point Selection"
            st.session_state.selected_points = [pt]
            st.session_state.bounding_box = None
            st.session_state.active_point = pt
            st.session_state.last_drawing_time = time.time()
            if map_data and map_data.get("last_clicked"):
                st.session_state.last_clicked = pt
            if needs_rerun:
                st.rerun()

    # --------------------------------------------------------------------------
    # 2. Click Interception (Selection ONLY - NO immediate backend call)
    # --------------------------------------------------------------------------
    clicked_data = map_data.get("last_clicked") if map_data else None

    # Check if a drawing was recently completed (within 2 seconds) to absorb mouseup
    is_recent_drawing = (time.time() - st.session_state.get("last_drawing_time", 0.0)) < 2.0

    if not st.session_state.get("pending_query") and clicked_data and "lat" in clicked_data and "lng" in clicked_data:
        # Round coordinates to 5 decimal places (~1.1 meter precision)
        clicked_coord = (round(clicked_data["lat"], 5), round(clicked_data["lng"], 5))

        # If a drawing just completed, sync last_clicked so the release click is suppressed
        if is_recent_drawing:
            st.session_state.last_clicked = clicked_coord
        elif clicked_coord != st.session_state.last_clicked:
            st.session_state.last_clicked = clicked_coord

            # Direct map click selects/replaces target point only if changed
            if st.session_state.selected_points != [clicked_coord]:
                st.session_state.selection_mode = "Point Selection"
                st.session_state.selected_points = [clicked_coord]
                st.session_state.bounding_box = None
                st.session_state.active_point = clicked_coord
                st.rerun()


# ------------------------------------------------------------------------------
# 6b. Execute Pending Chat Query (After Both Columns Are Mounted)
# ------------------------------------------------------------------------------
if st.session_state.get("pending_query"):
    pq = st.session_state.pending_query
    with chat_container:
        agent_resp = dispatch_geo_query(
            text_prompt=pq.get("text_prompt"),
            selected_point=pq.get("selected_point"),
            bounding_box=pq.get("bounding_box")
        )
    st.session_state.pending_query = None

    if agent_resp:
        # Append assistant response and extend visual map layers
        st.session_state.messages.append(_build_assistant_message(agent_resp))
        st.session_state.map_layers.extend(agent_resp.layers)
        if getattr(agent_resp, "map_bbox", None):
            st.session_state.last_map_bbox = agent_resp.map_bbox

        # Reset selected points unless user chose to retain them
        if not st.session_state.get("keep_points", False):
            clear_spatial_selection()

        # Refresh application state to immediately display updates
        st.rerun()


# ==============================================================================
# 7. Direct CLI Launcher Support
# ==============================================================================
# If a team member executes `python app.py` instead of `streamlit run app.py`,
# forward execution directly to the Streamlit CLI automatically.
if __name__ == "__main__":
    try:
        from streamlit.web import cli as stcli
        import streamlit.runtime as st_runtime

        if not st_runtime.exists():
            print("[GeoAI Runner] Detected direct python execution. Launching Streamlit web runtime...")
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(stcli.main())
    except Exception:
        pass
