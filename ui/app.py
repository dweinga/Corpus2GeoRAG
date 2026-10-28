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

    _query_hud_slot = st.empty()
    try:
        _query_hud_slot.markdown(
            "<script>(function(){try{var fs=window.parent.document.querySelectorAll('iframe');for(var i=0;i<fs.length;i++){if(fs[i].contentWindow)fs[i].contentWindow.postMessage({type:'GEOAI_QUERY_STATE',state:'querying'},'*');}}catch(e){}})();</script>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

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
            try:
                _query_hud_slot.empty()
            except Exception:
                pass
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
    st.session_state.answered_bbox = None
    st.session_state.active_point = None
    st.session_state.selection_mode = "Point Selection"
    st.session_state.last_drawing_time = time.time()
    st.session_state.selection_clear_seq = st.session_state.get("selection_clear_seq", 0) + 1
    if st.session_state.get("last_processed_drawing"):
        st.session_state.ignored_drawing = st.session_state.get("last_processed_drawing")
    _cur_map = st.session_state.get("geoai_tactical_map")
    if isinstance(_cur_map, dict):
        if _cur_map.get("last_active_drawing"):
            st.session_state.ignored_drawing = _cur_map.get("last_active_drawing")
        _lc = _cur_map.get("last_clicked")
        if isinstance(_lc, dict) and "lat" in _lc and "lng" in _lc:
            st.session_state.last_clicked = (round(_lc["lat"], 5), round(_lc["lng"], 5))
    st.session_state.last_processed_drawing = None


class SelectionSyncHook(MacroElement):
    """
    Lightweight MacroElement attached to `selection_fg` (feature_group_0) so that whenever
    Streamlit updates `feature_group_to_add` (without reloading the base map iframe):
    1. Any temporary client-side instant preview layers (`drawnItems`, `_instantTargetMarker`,
       `_instantBboxLayers`) are purged so server-rendered layers never duplicate or linger
       after clicking Clear.
    2. Active pin shape ('diamond' vs 'teardrop') and querying scanline state are synchronized.
    """
    _template = Template("""
        {% macro script(this, kwargs) %}
        if (typeof window._onServerSelectionSync === 'function') {
            window._onServerSelectionSync({
                hasSelection: {{ 'true' if this.has_selection else 'false' }},
                clearSeq: {{ this.clear_seq }},
                bboxState: "{{ this.bbox_state }}",
                isQuerying: {{ 'true' if this.is_querying else 'false' }},
                bboxBounds: {{ this.bbox_bounds_js }},
                pinShape: "{{ this.pin_shape }}"
            });
        }
        {% endmacro %}
    """)

    def __init__(
        self,
        has_selection: bool = False,
        clear_seq: int = 0,
        bbox_state: str = "selected",
        is_querying: bool = False,
        bbox_bounds: Optional[List[List[float]]] = None,
        pin_shape: str = "diamond",
    ):
        super().__init__()
        self.has_selection = bool(has_selection)
        self.clear_seq = int(clear_seq)
        self.bbox_state = bbox_state
        self.is_querying = bool(is_querying)
        self.bbox_bounds_js = json.dumps(bbox_bounds) if bbox_bounds else "null"
        self.pin_shape = (pin_shape or "diamond").strip().lower()


class MapTrashBinHook(MacroElement):
    """
    Folium MacroElement injecting client-side integration for the Leaflet Draw toolbar:
    1. Instant client-side visual feedback (0ms latency) on map clicks (target point) and bbox drawing.
    2. Registers dynamically rendered selection layers into `drawnItems` (the Leaflet Draw FeatureGroup).
    3. Hooks the Leaflet Draw trash bin button (.leaflet-draw-edit-remove) and 'Clear all' sub-toolbar actions
       so clicking the garbage can icon directly clears the active spatial selection or tactical overlays.
    4. Fires Leaflet's `draw:deleted` event and updates Streamlit component state.
    5. Automatically transitions client-side drawn rectangles to the ultra-transparent answered style
       when a query response bounding box is added to the map, and renders the pulsing fill + sweeping
       scanline during active agent queries.
    """
    _template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            var map = {{ this._parent.get_name() }};
            if (!map) return;

            if (typeof drawnItems !== 'undefined') {
                window.drawnItems = drawnItems;
            }
            window._instantBboxLayers = window._instantBboxLayers || [];
            window._scanOverlay = window._scanOverlay || null;
            window._lastClearSeq = window._lastClearSeq || 0;

            // ------------------------------------------------------------------
            // Client-side Tactical HUD DivIcon Factory (0ms instant previews)
            // ------------------------------------------------------------------
            function geoaiIcon(opts) {
                opts = opts || {};
                var role = opts.role || 'primary';
                if (role === 'target') {
                    var html = "<div class='geoai-reticle'>" +
                        "<i class='geoai-reticle__pulse'></i>" +
                        "<i class='geoai-reticle__spin'></i>" +
                        "<svg width='34' height='34' viewBox='0 0 34 34'>" +
                        "<circle cx='17' cy='17' r='10' fill='none' stroke='currentColor' stroke-width='2'/>" +
                        "<path d='M17 1v8M17 25v8M1 17h8M25 17h8' stroke='currentColor' stroke-width='2'/>" +
                        "<circle cx='17' cy='17' r='2.2' fill='currentColor'/>" +
                        "</svg></div>";
                    return L.divIcon({
                        html: html,
                        iconSize: [34, 34],
                        iconAnchor: [17, 17],
                        popupAnchor: [0, -17],
                        className: 'geoai-pin geoai-pin--target awesome-marker-icon-red'
                    });
                }
                if (role === 'bbox_corner') {
                    var cDir = (opts.corner || 'sw').toLowerCase();
                    var paths = {
                        nw: 'M2,14 V2 H14',
                        ne: 'M2,2 H14 V14',
                        se: 'M14,2 V14 H2',
                        sw: 'M14,14 H2 V2'
                    };
                    var anchors = {
                        nw: [2, 2],
                        ne: [14, 2],
                        se: [14, 14],
                        sw: [2, 14]
                    };
                    var dPath = paths[cDir] || paths.sw;
                    var cAnchor = anchors[cDir] || [2, 14];
                    var dimCls = opts.dimmed ? ' geoai-corner--dim' : '';
                    var cHtml = "<div class='geoai-corner geoai-corner--" + cDir + dimCls + "'>" +
                        "<svg width='16' height='16' viewBox='0 0 16 16'>" +
                        "<path d='" + dPath + "' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='square'/>" +
                        "</svg></div>";
                    return L.divIcon({
                        html: cHtml,
                        iconSize: [16, 16],
                        iconAnchor: cAnchor,
                        popupAnchor: [0, -8],
                        className: 'geoai-pin geoai-pin--corner geoai-pin--corner-' + cDir + ' awesome-marker-icon-red'
                    });
                }
                return L.divIcon({
                    html: "<div class='geoai-dot'></div>",
                    iconSize: [10, 10],
                    iconAnchor: [5, 5],
                    className: 'geoai-pin geoai-pin--context awesome-marker-icon-cadetblue'
                });
            }

            // ------------------------------------------------------------------
            // Querying State Animation (Pulsing Fill + Sweeping Scan Line)
            // ------------------------------------------------------------------
            function stopQueryingAnimation() {
                if (window._scanOverlay) {
                    try { map.removeLayer(window._scanOverlay); } catch (e) {}
                    window._scanOverlay = null;
                }
                try {
                    var c = map.getContainer();
                    if (c) {
                        var qPaths = c.querySelectorAll('.geoai-bbox--querying');
                        for (var i = 0; i < qPaths.length; i++) {
                            qPaths[i].classList.remove('geoai-bbox--querying');
                        }
                        var qRets = c.querySelectorAll('.geoai-reticle--querying');
                        for (var j = 0; j < qRets.length; j++) {
                            qRets[j].classList.remove('geoai-reticle--querying');
                        }
                    }
                } catch (e) {}
            }

            function startQueryingAnimation(boundsOverride) {
                stopQueryingAnimation();
                try {
                    var c = map.getContainer();
                    if (c) {
                        var rets = c.querySelectorAll('.geoai-reticle');
                        for (var r = 0; r < rets.length; r++) {
                            rets[r].classList.add('geoai-reticle--querying');
                        }
                    }
                    var targetBounds = boundsOverride || null;
                    map.eachLayer(function(l) {
                        if (l && l.getBounds && l._path) {
                            var cls = ((l.options && l.options.className) || '') + ' ' + (l._path.getAttribute('class') || '');
                            if (cls.indexOf('geoai-bbox--selected') !== -1 || cls.indexOf('geoai-bbox--querying') !== -1) {
                                l._path.classList.add('geoai-bbox--querying');
                                if (!targetBounds) {
                                    targetBounds = l.getBounds();
                                }
                            }
                        }
                    });
                    if (targetBounds && typeof L.svgOverlay === 'function') {
                        var svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                        svgEl.setAttribute('viewBox', '0 0 100 100');
                        svgEl.setAttribute('preserveAspectRatio', 'none');
                        svgEl.setAttribute('class', 'geoai-scan-svg');
                        svgEl.innerHTML =
                            "<defs><linearGradient id='geoai-scan-grad' x1='0%' y1='0%' x2='100%' y2='0%'>" +
                            "<stop offset='0%' stop-color='#00e5ff' stop-opacity='0'/>" +
                            "<stop offset='50%' stop-color='#00e5ff' stop-opacity='0.28'/>" +
                            "<stop offset='100%' stop-color='#00e5ff' stop-opacity='0'/>" +
                            "</linearGradient></defs>" +
                            "<rect class='geoai-scan-band' x='-8' y='0' width='16' height='100'>" +
                            "<animate attributeName='x' values='-8;92;-8' dur='2.4s' repeatCount='indefinite'/>" +
                            "</rect>" +
                            "<line class='geoai-scan-line' x1='0' y1='0' x2='0' y2='100'>" +
                            "<animate attributeName='x1' values='0;100;0' dur='2.4s' repeatCount='indefinite'/>" +
                            "<animate attributeName='x2' values='0;100;0' dur='2.4s' repeatCount='indefinite'/>" +
                            "</line>";
                        window._scanOverlay = L.svgOverlay(svgEl, targetBounds, { interactive: false }).addTo(map);
                    }
                } catch (e) {
                    console.warn('[GeoAI] startQueryingAnimation error:', e);
                }
            }

            function setMapPinShape(shape) {
                var container = map.getContainer();
                if (!container) return;
                var s = (shape || 'diamond').toLowerCase();
                window._geoaiPinShape = s;
                if (s === 'teardrop') {
                    container.classList.add('geoai-shape-teardrop');
                } else {
                    container.classList.remove('geoai-shape-teardrop');
                }
                var btnDia = document.getElementById('geoai_shape_btn_diamond');
                var btnTear = document.getElementById('geoai_shape_btn_teardrop');
                if (btnDia && btnTear) {
                    if (s === 'teardrop') {
                        btnTear.classList.add('active');
                        btnDia.classList.remove('active');
                    } else {
                        btnDia.classList.add('active');
                        btnTear.classList.remove('active');
                    }
                }
            }

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

            // Instant 0ms Component Value Dispatch to bypass streamlit-folium's 250ms artificial debounce
            window.geoaiImmediateDispatch = function(mapInstance) {
                try {
                    if (!window.__GLOBAL_DATA__ || !window.Streamlit || typeof window.Streamlit.setComponentValue !== 'function') return;
                    var g = window.__GLOBAL_DATA__;
                    var c = {
                        last_clicked: g.lat_lng_clicked,
                        last_active_drawing: g.last_active_drawing,
                        all_drawings: g.all_drawings
                    };
                    var u = {};
                    if (g.returned_objects && Array.isArray(g.returned_objects)) {
                        g.returned_objects.forEach(function(k) {
                            if (c.hasOwnProperty(k)) u[k] = c[k];
                        });
                    } else {
                        u = c;
                    }
                    if (JSON.stringify(g.previous_data) !== JSON.stringify(u)) {
                        g.previous_data = u;
                        window.Streamlit.setComponentValue(u);
                    }
                } catch(err) {
                    console.warn("[GeoAI] Immediate dispatch error:", err);
                }
            };
            try {
                if (typeof debouncedUpdateComponentValue === 'function') {
                    debouncedUpdateComponentValue = window.geoaiImmediateDispatch;
                }
            } catch(e) {}

            window._mapInitializing = true;
            setTimeout(function() { window._mapInitializing = false; }, 400);

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

            function clearInstantPreviewOnly() {
                try {
                    if (window.drawnItems && typeof window.drawnItems.clearLayers === 'function') {
                        window.drawnItems.clearLayers();
                    }
                    if (typeof drawnItems !== 'undefined' && drawnItems && typeof drawnItems.clearLayers === 'function') {
                        drawnItems.clearLayers();
                    }
                } catch(e) {}

                if (window._instantTargetMarker) {
                    try { map.removeLayer(window._instantTargetMarker); } catch(err) {}
                    window._instantTargetMarker = null;
                }

                if (window._instantBboxLayers && window._instantBboxLayers.length > 0) {
                    for (var i = 0; i < window._instantBboxLayers.length; i++) {
                        try { map.removeLayer(window._instantBboxLayers[i]); } catch(err) {}
                    }
                    window._instantBboxLayers = [];
                }
            }

            // Function to instantly remove any active selection from the map DOM (0ms latency)
            function clearPreviousSelectionInstantly() {
                stopQueryingAnimation();

                // 1. Clear Leaflet Draw items & client-side instant previews
                clearInstantPreviewOnly();

                // 2. Clear client-side instant target marker (explicitly kept for test assertions)
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
                var tacFg = (window.feature_group && window.feature_group.length > 1) ? window.feature_group[1] : null;
                function isSelectionLayer(layer) {
                    if (!layer) return false;
                    if (tacFg && typeof tacFg.hasLayer === 'function' && tacFg.hasLayer(layer)) {
                        return false;
                    }
                    if (layer._icon && typeof layer._icon.className === 'string') {
                        var ic = layer._icon.className;
                        if (ic.indexOf('geoai-pin--target') !== -1) return true;
                        if (ic.indexOf('geoai-pin--corner') !== -1) {
                            var innerCorner = layer._icon.querySelector && layer._icon.querySelector('.geoai-corner--dim');
                            if (!innerCorner) return true;
                        }
                    }
                    if (layer.options && typeof layer.options.className === 'string') {
                        if (layer.options.className.indexOf('geoai-bbox--selected') !== -1 ||
                            layer.options.className.indexOf('geoai-bbox--querying') !== -1) {
                            return true;
                        }
                    }
                    if (layer._path && layer._path.classList) {
                        if (layer._path.classList.contains('geoai-bbox--selected') ||
                            layer._path.classList.contains('geoai-bbox--querying')) {
                            return true;
                        }
                    }
                    var tt = layer.getTooltip && layer.getTooltip();
                    var ttContent = tt ? (tt.getContent ? tt.getContent() : tt._content) : "";
                    if (typeof ttContent === "string" && (
                        ttContent.indexOf("★ Target Point") !== -1 ||
                        ttContent.indexOf("★ Bounding Box") !== -1 ||
                        ttContent.indexOf("★ BBOX") !== -1
                    )) {
                        return true;
                    }
                    var pop = layer.getPopup && layer.getPopup();
                    var popContent = pop ? (pop.getContent ? pop.getContent() : pop._content) : "";
                    if (typeof popContent === "string" && (
                        popContent.indexOf("★ TARGET POINT") !== -1 ||
                        popContent.indexOf("★ BBOX") !== -1 ||
                        popContent.indexOf("★ TWO-POINT BOUNDING BOX") !== -1
                    )) {
                        return true;
                    }
                    return false;
                }

                try {
                    map.eachLayer(function(layer) {
                        try {
                            if (layer === tacFg) return;
                            if (isSelectionLayer(layer)) {
                                map.removeLayer(layer);
                                return;
                            }
                            if (layer.eachLayer) {
                                var toRemove = [];
                                layer.eachLayer(function(subLayer) {
                                    try {
                                        if (isSelectionLayer(subLayer)) {
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

            // Server-to-Client Selection Synchronization Hook (invoked by SelectionSyncHook on selection_fg)
            window._onServerSelectionSync = function(state) {
                state = state || {};
                if (state.pinShape) {
                    setMapPinShape(state.pinShape);
                }
                // Always clear temporary client-side preview layers once selection_fg has mounted from Python
                clearInstantPreviewOnly();

                if (!state.hasSelection) {
                    clearPreviousSelectionInstantly();
                    if (window.__GLOBAL_DATA__) {
                        window.__GLOBAL_DATA__.lat_lng_clicked = null;
                        window.__GLOBAL_DATA__.last_active_drawing = null;
                        window.__GLOBAL_DATA__.all_drawings = [];
                    }
                    try {
                        window.parent.postMessage({
                            type: 'GEOAI_SELECTION_EVENT',
                            action: 'clear'
                        }, '*');
                    } catch(err) {}
                } else if (state.isQuerying) {
                    setTimeout(function() {
                        startQueryingAnimation(state.bboxBounds);
                    }, 30);
                } else {
                    stopQueryingAnimation();
                }
                setTimeout(updateTrashButtonState, 50);
            };

            // Listen for parent window messages (querying state, instant clear button, pin shape)
            window.addEventListener('message', function(ev) {
                if (!ev || !ev.data) return;
                if (ev.data.type === 'GEOAI_QUERY_STATE') {
                    if (ev.data.state === 'querying') {
                        startQueryingAnimation();
                    } else {
                        stopQueryingAnimation();
                    }
                } else if (ev.data.type === 'GEOAI_PARENT_CMD' && ev.data.action === 'clear') {
                    triggerRemoval();
                } else if (ev.data.type === 'GEOAI_PARENT_CMD' && ev.data.action === 'clear_visuals') {
                    clearPreviousSelectionInstantly();
                    if (window.__GLOBAL_DATA__) {
                        window.__GLOBAL_DATA__.lat_lng_clicked = null;
                        window.__GLOBAL_DATA__.last_active_drawing = null;
                        window.__GLOBAL_DATA__.all_drawings = [];
                    }
                    updateTrashButtonState();
                } else if (ev.data.type === 'GEOAI_PIN_SHAPE' && ev.data.shape) {
                    setMapPinShape(ev.data.shape);
                }
            });

            // 1. Instant client-side visual feedback on map canvas clicks (0ms latency)
            map.on('click', function(e) {
                if (map._isDrawing) return;
                if (e.originalEvent && (e.originalEvent._stopped || e.originalEvent.defaultPrevented)) return;

                // Guard: Ignore clicks on existing markers, cluster icons, popups, or controls
                var target = e.originalEvent && e.originalEvent.target;
                if (target && target.closest && target.closest('.leaflet-marker-icon, .geoai-pin, .awesome-marker, .marker-cluster, .leaflet-popup, .leaflet-popup-content-wrapper, .leaflet-control')) {
                    return;
                }

                var lat = Math.round(e.latlng.lat * 100000) / 100000;
                var lng = Math.round(e.latlng.lng * 100000) / 100000;

                // Immediately purge any prior selection (from previous click or python render)
                clearPreviousSelectionInstantly();

                try {
                    var starIcon = geoaiIcon({ role: 'target' });
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

                // Send 0ms instant selection event to parent window HUD
                try {
                    window.parent.postMessage({
                        type: 'GEOAI_SELECTION_EVENT',
                        action: 'set_point',
                        lat: lat,
                        lng: lng
                    }, '*');
                } catch(err) {}

                var btn = document.querySelector(".leaflet-draw-edit-remove");
                if (btn) {
                    btn.classList.remove("leaflet-disabled");
                }
                if (window.__GLOBAL_DATA__) {
                    window.__GLOBAL_DATA__.lat_lng_clicked = e.latlng;
                }
                try {
                    if (typeof window.geoaiImmediateDispatch === "function") {
                        window.geoaiImmediateDispatch(map);
                    } else if (typeof updateComponentValue === "function") {
                        updateComponentValue(map);
                    }
                } catch(err) {}
            });

            // Synchronize selection and maintain popup state when existing markers are clicked
            map.on('popupopen', function(e) {
                if (window._mapInitializing || window._reopeningPopup) return;
                if (e.popup) {
                    var latlng = e.popup.getLatLng();
                    var source = e.popup._source;
                    if (latlng) {
                        var p_lat = Math.round(latlng.lat * 100000) / 100000;
                        var p_lng = Math.round(latlng.lng * 100000) / 100000;
                        window._lastOpenPopupInfo = {
                            lat: p_lat,
                            lng: p_lng
                        };

                        // Check if this point is already the active target to prevent unnecessary reruns
                        var alreadyActive = false;
                        if (window.__GLOBAL_DATA__ && window.__GLOBAL_DATA__.lat_lng_clicked) {
                            var cur = window.__GLOBAL_DATA__.lat_lng_clicked;
                            if (Math.abs(cur.lat - p_lat) < 0.00001 && Math.abs(cur.lng - p_lng) < 0.00001) {
                                alreadyActive = true;
                            }
                        }

                        if (!alreadyActive) {
                            try {
                                window.parent.postMessage({
                                    type: 'GEOAI_SELECTION_EVENT',
                                    action: 'set_point',
                                    lat: p_lat,
                                    lng: p_lng
                                }, '*');
                            } catch(err) {}

                            if (window.__GLOBAL_DATA__) {
                                window.__GLOBAL_DATA__.lat_lng_clicked = latlng;
                            }
                            if (source && typeof source.toGeoJSON === 'function' && window.__GLOBAL_DATA__) {
                                window.__GLOBAL_DATA__.last_active_drawing = source.toGeoJSON();
                            }
                            try {
                                if (typeof window.geoaiImmediateDispatch === 'function') {
                                    window.geoaiImmediateDispatch(map);
                                } else if (typeof updateComponentValue === 'function') {
                                    updateComponentValue(map);
                                }
                            } catch(err) {}
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
                        setTimeout(function() { window._reopeningPopup = false; }, 100);
                    }
                }, 25);
            }

            // 2. Instant client-side feedback when an AOI Rectangle is drawn
            map.on('draw:created', function(e) {
                if (e.layerType === 'rectangle') {
                    var rect = e.layer;
                    var bounds = rect.getBounds();
                    var rawSW = bounds.getSouthWest();
                    var rawNE = bounds.getNorthEast();
                    var swLat = Math.round(Math.min(rawSW.lat, rawNE.lat) * 100000) / 100000;
                    var swLng = Math.round(Math.min(rawSW.lng, rawNE.lng) * 100000) / 100000;
                    var neLat = Math.round(Math.max(rawSW.lat, rawNE.lat) * 100000) / 100000;
                    var neLng = Math.round(Math.max(rawSW.lng, rawNE.lng) * 100000) / 100000;

                    // Send 0ms instant selection event to parent window HUD
                    try {
                        window.parent.postMessage({
                            type: 'GEOAI_SELECTION_EVENT',
                            action: 'set_bbox',
                            sw: [swLat, swLng],
                            ne: [neLat, neLng]
                        }, '*');
                    } catch(err) {}

                    // Immediately clear any prior selection (single point or previous box)
                    clearPreviousSelectionInstantly();

                    try {
                        if (rect.setBounds) {
                            rect.setBounds([[swLat, swLng], [neLat, neLng]]);
                        }
                        if (rect.setStyle) {
                            rect.setStyle({
                                color: '#00e5ff',
                                weight: 2,
                                opacity: 0.9,
                                dashArray: '6, 6',
                                fillColor: '#00e5ff',
                                fillOpacity: 0.10,
                                className: 'geoai-bbox geoai-bbox--selected'
                            });
                        }
                        if (window.drawnItems) {
                            window.drawnItems.addLayer(rect);
                        } else {
                            rect.addTo(map);
                        }
                        if (rect._path) {
                            rect._path.classList.add('geoai-bbox', 'geoai-bbox--selected');
                        }
                        var iconDown = geoaiIcon({ role: 'bbox_corner', corner: 'sw', icon: 'arrow-down' });
                        var iconUp = geoaiIcon({ role: 'bbox_corner', corner: 'ne', icon: 'arrow-up' });
                        var iconNW = geoaiIcon({ role: 'bbox_corner', corner: 'nw' });
                        var iconSE = geoaiIcon({ role: 'bbox_corner', corner: 'se' });

                        var mSW = L.marker([swLat, swLng], { icon: iconDown })
                            .bindTooltip("★ Bounding Box Lower-Left (SW): " + swLat.toFixed(4) + "°, " + swLng.toFixed(4) + "°");
                        var mNE = L.marker([neLat, neLng], { icon: iconUp })
                            .bindTooltip("★ Bounding Box Upper-Right (NE): " + neLat.toFixed(4) + "°, " + neLng.toFixed(4) + "°");
                        var mNW = L.marker([neLat, swLng], { icon: iconNW, interactive: false });
                        var mSE = L.marker([swLat, neLng], { icon: iconSE, interactive: false });

                        if (window.drawnItems) {
                            window.drawnItems.addLayer(mSW);
                            window.drawnItems.addLayer(mNE);
                            window.drawnItems.addLayer(mNW);
                            window.drawnItems.addLayer(mSE);
                        } else {
                            mSW.addTo(map);
                            mNE.addTo(map);
                            mNW.addTo(map);
                            mSE.addTo(map);
                        }
                        window._instantBboxLayers = [rect, mSW, mNE, mNW, mSE];
                    } catch(err) {
                        console.warn("[GeoAI] Instant bbox markers error:", err);
                    }

                    var btn = document.querySelector(".leaflet-draw-edit-remove");
                    if (btn) {
                        btn.classList.remove("leaflet-disabled");
                    }
                    if (window.__GLOBAL_DATA__) {
                        window.__GLOBAL_DATA__.last_active_drawing = rect.toGeoJSON();
                        if (window.drawnItems && typeof window.drawnItems.toGeoJSON === 'function') {
                            window.__GLOBAL_DATA__.all_drawings = window.drawnItems.toGeoJSON().features;
                        }
                    }
                    try {
                        if (typeof window.geoaiImmediateDispatch === "function") {
                            window.geoaiImmediateDispatch(map);
                        } else if (typeof updateComponentValue === "function") {
                            updateComponentValue(map);
                        }
                    } catch(err) {}
                }
            });

            // 3. Automatically dim client-side drawnItems rectangle when an answered bbox arrives
            function syncAnsweredBboxInDrawnItems() {
                var hasAnsweredOnMap = false;
                map.eachLayer(function(l) {
                    if (l && l.options && typeof l.options.className === 'string' && l.options.className.indexOf('geoai-bbox--answered') !== -1) {
                        hasAnsweredOnMap = true;
                    }
                });
                if (!hasAnsweredOnMap) return;
                stopQueryingAnimation();
                if (!window.drawnItems || typeof window.drawnItems.eachLayer !== 'function') return;
                window.drawnItems.eachLayer(function(dl) {
                    if (dl && typeof dl.setStyle === 'function') {
                        dl.setStyle({
                            weight: 1.25,
                            opacity: 0.55,
                            dashArray: '3, 6',
                            fillOpacity: 0.03,
                            interactive: false,
                            className: 'geoai-bbox geoai-bbox--answered'
                        });
                        if (dl._path) {
                            dl._path.style.pointerEvents = 'none';
                            dl._path.classList.remove('geoai-bbox--selected', 'geoai-bbox--querying');
                            dl._path.classList.add('geoai-bbox--answered');
                        }
                    } else if (dl instanceof L.Marker && dl._icon && dl._icon.className && dl._icon.className.indexOf('geoai-pin--corner') !== -1) {
                        var inner = dl._icon.querySelector('.geoai-corner');
                        if (inner) inner.classList.add('geoai-corner--dim');
                    }
                });
            }

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
                    window.__GLOBAL_DATA__.lat_lng_clicked = null;
                }

                try {
                    map.fire("draw:deleted", {
                        layer: {
                            toGeoJSON: function() { return delPayload; }
                        }
                    });
                } catch(e) {}

                try {
                    window.parent.postMessage({
                        type: 'GEOAI_SELECTION_EVENT',
                        action: 'clear'
                    }, '*');
                } catch(err) {}

                try {
                    if (typeof window.geoaiImmediateDispatch === "function") {
                        window.geoaiImmediateDispatch(map);
                    } else if (typeof updateComponentValue === "function") {
                        updateComponentValue(map);
                    } else if (typeof debouncedUpdateComponentValue === "function") {
                        debouncedUpdateComponentValue(map);
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
                    } else {
                        btn.classList.add("leaflet-disabled");
                    }
                }
            }
            setTimeout(updateTrashButtonState, 100);

            // ------------------------------------------------------------------
            // Bottom-Right Tactical Pin Filter & Shape Control (0ms Client-Side)
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
                    toggle.title = 'Pin Shape & Visibility Filters';

                    var section = L.DomUtil.create('section', 'leaflet-control-layers-list leaflet-control-pin-filter-list', container);
                    var overlays = L.DomUtil.create('div', 'leaflet-control-layers-overlays', section);

                    var shapeHeader = L.DomUtil.create('div', 'pin-filter-header', overlays);
                    shapeHeader.innerHTML = '◆ <b>PIN SHAPE</b>';

                    var shapeRow = L.DomUtil.create('div', 'geoai-shape-seg', overlays);
                    var btnDia = L.DomUtil.create('button', 'geoai-shape-btn active', shapeRow);
                    btnDia.type = 'button';
                    btnDia.id = 'geoai_shape_btn_diamond';
                    btnDia.textContent = '◆ Diamond';
                    var btnTear = L.DomUtil.create('button', 'geoai-shape-btn', shapeRow);
                    btnTear.type = 'button';
                    btnTear.id = 'geoai_shape_btn_teardrop';
                    btnTear.textContent = '💧 Teardrop';

                    btnDia.addEventListener('click', function(ev) {
                        ev.preventDefault();
                        setMapPinShape('diamond');
                    });
                    btnTear.addEventListener('click', function(ev) {
                        ev.preventDefault();
                        setMapPinShape('teardrop');
                    });

                    var header = L.DomUtil.create('div', 'pin-filter-header', overlays);
                    header.style.marginTop = '8px';
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
                                if (cls.indexOf('geoai-pin--target') !== -1 || cls.indexOf('geoai-pin--corner') !== -1 || cls.indexOf('geoai-pin--event') !== -1) return;

                                var isCtx = cls.indexOf('geoai-pin--context') !== -1 || cls.indexOf('awesome-marker-icon-cadetblue') !== -1;
                                var isUnc = cls.indexOf('geoai-pin--uncertain') !== -1 || cls.indexOf('awesome-marker-icon-orange') !== -1;
                                if (!isCtx && !isUnc && cls.indexOf('awesome-marker-icon-red') !== -1) return;

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
                                if (cls.indexOf('geoai-pin--target') !== -1 || cls.indexOf('geoai-pin--corner') !== -1) return;
                                if (cls.indexOf('geoai-pin--context') !== -1 || cls.indexOf('awesome-marker-icon-cadetblue') !== -1) countCtx++;
                                else if (cls.indexOf('geoai-pin--uncertain') !== -1 || cls.indexOf('awesome-marker-icon-orange') !== -1) countUnc++;
                                else if (cls.indexOf('geoai-pin--') !== -1 || cls.indexOf('awesome-marker-icon-blue') !== -1) countTotal++;
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

                    setTimeout(function() {
                        refreshPinCounts();
                        syncAnsweredBboxInDrawnItems();
                    }, 300);
                    map.on('layeradd', function(e) {
                        if (typeof L.MarkerClusterGroup !== 'undefined' && e.layer instanceof L.MarkerClusterGroup) {
                            if (clusterGroups.indexOf(e.layer) === -1) clusterGroups.push(e.layer);
                        }
                        if (e.layer && e.layer.options && typeof e.layer.options.className === 'string' && e.layer.options.className.indexOf('geoai-bbox--answered') !== -1) {
                            if (e.layer._path) {
                                e.layer._path.style.pointerEvents = 'none';
                            }
                            syncAnsweredBboxInDrawnItems();
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

# Semantic Map Symbology Palette (Dark & Light basemap variants)
MAP_PALETTE = {
    "dark": {
        "selection": "#00e5ff",
        "primary": "#38bdf8",
        "event": "#ff4d6d",
        "uncertain": "#ffb020",
        "context": "#7dd3fc",
        "threat": "#f97316",
        "recon": "#a855f7",
        "friendly": "#22c55e",
    },
    "light": {
        "selection": "#0284c7",
        "primary": "#0369a1",
        "event": "#be123c",
        "uncertain": "#b45309",
        "context": "#475569",
        "threat": "#c2410c",
        "recon": "#7e22ce",
        "friendly": "#15803d",
    },
}


def color_token(
    color_name: Optional[str] = None,
    layer_type: str = "marker",
    role: Optional[str] = None,
    kind: Optional[str] = None,
    uncertain: Any = None,
    theme: str = "dark",
) -> Tuple[str, str]:
    """
    Map legacy MapLayer.color strings and semantic layer attributes to
    (token_name, hex_color) from MAP_PALETTE.
    """
    pal = MAP_PALETTE.get(theme, MAP_PALETTE["dark"])
    c_lower = (color_name or "").strip().lower()

    if role == "query_bbox" or role == "target_buffer":
        tok = "selection"
    elif uncertain or (layer_type == "marker" and c_lower == "orange"):
        tok = "uncertain"
    elif role == "context" or c_lower == "cadetblue":
        tok = "context"
    elif kind == "event" or (layer_type == "marker" and c_lower == "red"):
        tok = "event"
    elif c_lower in ("cyan", "#00e5ff", "#0284c7"):
        tok = "selection"
    elif c_lower in ("orange", "#f59e0b", "#f97316"):
        tok = "threat" if layer_type != "marker" else "uncertain"
    elif c_lower in ("purple", "#a855f7"):
        tok = "recon"
    elif c_lower in ("green", "#22c55e"):
        tok = "friendly"
    elif c_lower in ("red", "#ef4444", "#ff4d6d"):
        tok = "event"
    else:
        tok = "primary"

    return tok, pal.get(tok, pal["primary"])


def bbox_style(state: str = "selected", theme: str = "dark") -> Dict[str, Any]:
    """
    Return Leaflet path style kwargs for a bounding box in 'selected', 'querying', or 'answered' state.
    - 'selected': 2px cyan dashed outline at 10% fill with animated marching ants.
    - 'querying': 2px cyan dashed outline with gentle fill pulse (8% -> 20%) and sweeping scan line.
    - 'answered': ultra-transparent (fill_opacity=0.03) and non-interactive so result pins and
      basemap labels remain completely clear and clickable.
    """
    _, sel_hex = color_token("cyan", layer_type="rectangle", role="query_bbox", theme=theme)
    if state == "answered":
        return {
            "color": sel_hex,
            "weight": 1.25,
            "opacity": 0.55,
            "dash_array": "3, 6",
            "fill": True,
            "fill_color": sel_hex,
            "fill_opacity": 0.03,
            "interactive": False,
            "class_name": "geoai-bbox geoai-bbox--answered",
        }
    if state == "querying":
        return {
            "color": sel_hex,
            "weight": 2,
            "opacity": 0.95,
            "dash_array": "6, 6",
            "fill": True,
            "fill_color": sel_hex,
            "fill_opacity": 0.08,
            "interactive": False,
            "class_name": "geoai-bbox geoai-bbox--selected geoai-bbox--querying",
        }
    return {
        "color": sel_hex,
        "weight": 2,
        "opacity": 0.9,
        "dash_array": "6, 6",
        "fill": True,
        "fill_color": sel_hex,
        "fill_opacity": 0.10,
        "interactive": True,
        "class_name": "geoai-bbox geoai-bbox--selected",
    }


def make_pin_icon(
    kind: str = "place",
    role: str = "primary",
    uncertain: Any = None,
    number: Optional[Any] = None,
    color_name: Optional[str] = None,
    corner: Optional[str] = None,
    dimmed: bool = False,
    shape: Optional[str] = None,
) -> folium.DivIcon:
    """
    Build a lightweight, pure-CSS Tactical HUD folium.DivIcon for map markers:
    - role == 'target': pulsing 34px crosshair reticle (with outer spin ring during queries)
    - role == 'bbox_corner' or corner in ('sw','ne','nw','se'): 16px L-shaped corner bracket
      anchored directly at the exact corner vertex so brackets and marching ants align pixel-for-pixel
    - role == 'context': 8px solid dot with dark halo (0 image/shadow overhead for 500-1,600 pins)
    - primary / event / uncertain / mock: switchable between 'diamond' (22px rotated square)
      and 'teardrop' (24px classic teardrop pin) with number or symbol badge
    """
    if role == "target":
        html = (
            "<div class='geoai-reticle'>"
            "<i class='geoai-reticle__pulse'></i>"
            "<i class='geoai-reticle__spin'></i>"
            "<svg width='34' height='34' viewBox='0 0 34 34'>"
            "<circle cx='17' cy='17' r='10' fill='none' stroke='currentColor' stroke-width='2'/>"
            "<path d='M17 1v8M17 25v8M1 17h8M25 17h8' stroke='currentColor' stroke-width='2'/>"
            "<circle cx='17' cy='17' r='2.2' fill='currentColor'/>"
            "</svg>"
            "</div>"
        )
        return folium.DivIcon(
            html=html,
            icon_size=(34, 34),
            icon_anchor=(17, 17),
            popup_anchor=(0, -17),
            class_name="geoai-pin geoai-pin--target awesome-marker-icon-red",
        )

    if role == "bbox_corner" or corner in ("sw", "ne", "nw", "se"):
        c_dir = (corner or "sw").lower()
        dim_cls = " geoai-corner--dim" if dimmed else ""
        paths = {
            "nw": "M2,14 V2 H14",
            "ne": "M2,2 H14 V14",
            "se": "M14,2 V14 H2",
            "sw": "M14,14 H2 V2",
        }
        anchors = {
            "nw": (2, 2),
            "ne": (14, 2),
            "se": (14, 14),
            "sw": (2, 14),
        }
        d_path = paths.get(c_dir, paths["sw"])
        c_anchor = anchors.get(c_dir, (2, 14))
        html = (
            f"<div class='geoai-corner geoai-corner--{c_dir}{dim_cls}'>"
            f"<svg width='16' height='16' viewBox='0 0 16 16'>"
            f"<path d='{d_path}' fill='none' stroke='currentColor' stroke-width='3' stroke-linecap='square'/>"
            f"</svg></div>"
        )
        return folium.DivIcon(
            html=html,
            icon_size=(16, 16),
            icon_anchor=c_anchor,
            popup_anchor=(0, -8),
            class_name=f"geoai-pin geoai-pin--corner geoai-pin--corner-{c_dir} awesome-marker-icon-red",
        )

    tok, _ = color_token(color_name, layer_type="marker", role=role, kind=kind, uncertain=uncertain)
    is_ctx = (role == "context" or tok == "context") and not uncertain
    is_unc = bool(uncertain) or tok == "uncertain"

    if is_ctx:
        html = "<div class='geoai-dot'></div>"
        return folium.DivIcon(
            html=html,
            icon_size=(10, 10),
            icon_anchor=(5, 5),
            popup_anchor=(0, -6),
            class_name="geoai-pin geoai-pin--context awesome-marker-icon-cadetblue",
        )

    if number is not None and str(number).strip() != "":
        label_str = str(number).strip()
    elif tok == "event" or kind == "event":
        label_str = "!"
    else:
        label_str = "•"

    resolved_shape = (
        shape
        or (st.session_state.get("pin_shape", "Diamond") if hasattr(st, "session_state") else "Diamond")
        or "Diamond"
    ).strip().lower()
    is_teardrop = (resolved_shape == "teardrop")

    unc_cls = " geoai-diamond--uncertain" if is_unc else ""
    shape_inner_cls = " geoai-teardrop" if is_teardrop else ""
    shape_outer_cls = " geoai-pin--teardrop" if is_teardrop else " geoai-pin--diamond"
    badge_html = "<i class='geoai-badge-unc'>?</i>" if is_unc else ""
    legacy_cls = (
        "awesome-marker-icon-orange"
        if is_unc
        else ("awesome-marker-icon-red" if tok == "event" else "awesome-marker-icon-blue")
    )
    role_cls = "geoai-pin--uncertain" if is_unc else f"geoai-pin--{tok}"

    html = (
        f"<div class='geoai-diamond{shape_inner_cls}{unc_cls}' style='--c:var(--geoai-{tok});'>"
        f"<span>{label_str}</span>"
        f"</div>{badge_html}"
    )
    return folium.DivIcon(
        html=html,
        icon_size=(22, 22),
        icon_anchor=(11, 11),
        popup_anchor=(0, -28 if is_teardrop else -14),
        class_name=f"geoai-pin{shape_outer_cls} {role_cls} {legacy_cls}",
    )


def _popup_card(
    title: str,
    rows: List[Tuple[str, Any]],
    chip: Optional[str] = None,
    footer: Optional[str] = None,
    token: str = "primary",
) -> str:
    """Build a unified Tactical HUD HTML card for marker and shape popups."""
    chip_html = f"<div class='geoai-card__chip'>{chip}</div>" if chip else ""
    rows_html = "".join(
        f"<span>{k}</span><code>{v}</code>" for k, v in rows if v is not None and str(v) != ""
    )
    footer_html = f"<div class='geoai-card__footer'>{footer}</div>" if footer else ""
    return (
        f"<div class='geoai-card' style='--c:var(--geoai-{token});'>"
        f"<div class='geoai-card__head'>{chip_html}<div class='geoai-card__title'>{title}</div></div>"
        f"<div class='geoai-card__body'>{rows_html}</div>"
        f"{footer_html}"
        f"</div>"
    )

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

    /* Instant Client-Side Target HUD Pill Bridge */
    .geoai-instant-pill-row {{
        display: flex;
        gap: 8px;
        align-items: center;
        margin-bottom: 2px;
    }}
    .geoai-instant-pill-col-left {{
        flex: 5.2;
        min-width: 0;
    }}
    .geoai-instant-pill-col-mid {{
        flex: 2.1;
        min-width: 0;
    }}
    .geoai-instant-pill-col-right {{
        flex: 2.7;
        min-width: 0;
    }}
    .geoai-instant-btn {{
        display: flex;
        align-items: center;
        justify-content: center;
        height: 38px;
        background-color: transparent;
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 4px;
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        box-sizing: border-box;
        cursor: default;
    }}

    /* Streamlined Verified Source Accordions */
    .geoai-source-detail {{
        border: 1px solid rgba(0, 229, 255, 0.15);
        border-radius: 4px;
        background: rgba(14, 22, 32, 0.35);
        padding: 6px 10px;
        margin-bottom: 6px;
        transition: background 0.15s ease, border-color 0.15s ease;
    }}
    .geoai-source-detail[open] {{
        border-color: rgba(0, 229, 255, 0.35);
        background: rgba(14, 22, 32, 0.6);
    }}
    .geoai-source-summary {{
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        color: #cbd5e1;
        list-style: none;
        outline: none;
        user-select: none;
    }}
    .geoai-source-summary::-webkit-details-marker {{
        display: none;
    }}
    .geoai-source-title {{
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding-right: 8px;
    }}
    .geoai-source-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        color: #00e5ff;
        background: rgba(0, 229, 255, 0.08);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 3px;
        padding: 2px 7px;
        flex-shrink: 0;
        transition: all 0.15s ease;
    }}
    .geoai-source-detail[open] .geoai-source-badge {{
        color: #f87171;
        border-color: rgba(248, 113, 113, 0.3);
        background: rgba(248, 113, 113, 0.08);
    }}
</style>
""", unsafe_allow_html=True)

# 0ms Instant Target HUD Bridge: listens to Leaflet map clicks/drawings in the iframe
st.markdown("""
<script>
(function() {
    if (window._geoaiSelectionBridgeInstalled) return;
    window._geoaiSelectionBridgeInstalled = true;

    function postToMapIframes(payload) {
        try {
            var iframes = document.querySelectorAll('iframe');
            for (var i = 0; i < iframes.length; i++) {
                if (iframes[i].contentWindow) {
                    iframes[i].contentWindow.postMessage(payload, '*');
                }
            }
        } catch (err) {}
    }

    // 0ms Parent-to-Iframe Event Forwarding (Clear buttons, Chat Submit, Recon Scan)
    document.addEventListener('click', function(e) {
        var target = e.target;
        if (!target || !target.closest) return;

        // 1. Instant preview pill Clear button
        var instClr = target.closest('[data-geoai-action="clear"]');
        if (instClr) {
            var slot = document.getElementById('geoai-instant-pill-slot');
            if (slot) {
                slot.style.display = 'none';
                slot.innerHTML = '';
            }
            postToMapIframes({ type: 'GEOAI_PARENT_CMD', action: 'clear' });
            return;
        }

        // 2. Streamlit Clear / Reset buttons -> 0ms visual cleanup in map iframe
        var btn = target.closest('button');
        if (btn) {
            var txt = (btn.textContent || '').trim();
            if (
                txt.indexOf('❌ Clear') !== -1 ||
                txt.indexOf('Clear Box') !== -1 ||
                txt.indexOf('Clear Point') !== -1 ||
                txt.indexOf('Reset Map Layers') !== -1
            ) {
                var slot2 = document.getElementById('geoai-instant-pill-slot');
                if (slot2) {
                    slot2.style.display = 'none';
                    slot2.innerHTML = '';
                }
                postToMapIframes({ type: 'GEOAI_PARENT_CMD', action: 'clear_visuals' });
                return;
            }
            // 3. Recon Scan or Chat Submit button -> 0ms querying scanline + pulse
            if (
                txt.indexOf('Recon') !== -1 ||
                btn.getAttribute('data-testid') === 'stChatInputSubmitButton'
            ) {
                postToMapIframes({ type: 'GEOAI_QUERY_STATE', state: 'querying' });
            }
        }
    }, true);

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            var ta = e.target && e.target.closest && e.target.closest('div[data-testid="stChatInput"] textarea');
            if (ta && ta.value && ta.value.trim().length > 0) {
                postToMapIframes({ type: 'GEOAI_QUERY_STATE', state: 'querying' });
            }
        }
    }, true);

    window.addEventListener('message', function(event) {
        if (!event.data || event.data.type !== 'GEOAI_SELECTION_EVENT') return;
        var data = event.data;
        var slot = document.getElementById('geoai-instant-pill-slot');
        if (!slot) return;

        if (data.action === 'clear') {
            slot.style.display = 'none';
            slot.innerHTML = '';
            return;
        }

        var pillHtml = '';
        if (data.action === 'set_point') {
            var plat = Number(data.lat).toFixed(4);
            var plng = Number(data.lng).toFixed(4);
            pillHtml = "<div class='geo-docked-pill' title='Attached Tactical Target: (" + plat + "°, " + plng + "°)'>" +
                       "<span class='geo-pill-icon'>📍</span>" +
                       "<span class='geo-pill-label'>TARGET:</span>" +
                       "<span class='geo-pill-coord'>(" + plat + "°, " + plng + "°)</span>" +
                       "</div>";
        } else if (data.action === 'set_bbox') {
            var sw_lat = Number(data.sw[0]).toFixed(4);
            var sw_lng = Number(data.sw[1]).toFixed(4);
            var ne_lat = Number(data.ne[0]).toFixed(4);
            var ne_lng = Number(data.ne[1]).toFixed(4);
            pillHtml = "<div class='geo-docked-pill geo-docked-pill-stacked' title='Bounding Box AOI: Lower-Left (" + sw_lat + "°, " + sw_lng + "°) | Upper-Right (" + ne_lat + "°, " + ne_lng + "°)'>" +
                       "<span class='geo-pill-icon'>📦</span>" +
                       "<div class='geo-pill-stack'>" +
                       "<div class='geo-pill-row'><span class='geo-pill-sublabel'>SW:</span><span class='geo-pill-coord'>" + sw_lat + "°, " + sw_lng + "°</span></div>" +
                       "<div class='geo-pill-row'><span class='geo-pill-sublabel'>NE:</span><span class='geo-pill-coord'>" + ne_lat + "°, " + ne_lng + "°</span></div>" +
                       "</div></div>";
        }

        if (pillHtml) {
            slot.innerHTML = "<div class='geoai-instant-pill-row'>" +
                             "<div class='geoai-instant-pill-col-left'>" + pillHtml + "</div>" +
                             "<div class='geoai-instant-pill-col-mid'><div class='geoai-instant-btn' data-geoai-action='clear' style='cursor:pointer;'>❌ Clear</div></div>" +
                             "<div class='geoai-instant-pill-col-right'><div class='geoai-instant-btn'>🔍 Recon</div></div>" +
                             "</div>";
            slot.style.display = 'block';
        }
    });
})();
</script>
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

if "answered_bbox" not in st.session_state:
    # Tracks a bounding box that has already received a query response so it renders ultra-transparent
    st.session_state.answered_bbox = None

if "pin_shape" not in st.session_state or st.session_state.pin_shape not in ["Diamond", "Teardrop"]:
    # Primary/Event/Uncertain marker shape: 'Diamond' (default) or 'Teardrop'
    st.session_state.pin_shape = "Diamond"

if "selection_clear_seq" not in st.session_state:
    # Monotonic counter incremented when selection is cleared to guarantee client-side sync
    st.session_state.selection_clear_seq = 0

# ==============================================================================
# 2b. Early Map Event Ingestion (Pre-render state synchronization)
# ==============================================================================
# Ingests map clicks, drawings, and deletions before the sidebar and chat mount,
# ensuring all UI panels reflect the newest spatial selection on the initial rerun
# without requiring a second st.rerun() cascade.
def _peek_incoming_map_state() -> Optional[Dict[str, Any]]:
    s = st.session_state.get("geoai_tactical_map")
    if isinstance(s, dict) and ("last_clicked" in s or "last_active_drawing" in s or "all_drawings" in s):
        return s
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and len(k) == 64:
            v = st.session_state.get(k)
            if isinstance(v, dict) and ("last_clicked" in v or "last_active_drawing" in v or "all_drawings" in v):
                return v
    return None

_early_map_state = _peek_incoming_map_state()
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
            st.session_state.answered_bbox = None
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
                st.session_state.answered_bbox = None
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
                st.session_state.answered_bbox = None
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

    st.radio(
        "Pin shape",
        options=["Diamond", "Teardrop"],
        horizontal=True,
        key="pin_shape",
        help="Switch map pin shape between Tactical Diamond (◆) and Classic Teardrop (💧). Also toggleable directly on the map's bottom-right Pin Control."
    )

    st.toggle(
        "Retain target pin after query",
        key="keep_points",
        help="Keep the active target pin or bounding box selected after sending a query (instead of clearing it)."
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

    # Session metrics
    st.subheader("📊 Session Telemetry")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Chat Messages", len(st.session_state.messages))
    with col_m2:
        st.metric("Active Layers", len(st.session_state.map_layers))
    st.metric("RAG Query Logs", len(st.session_state.get("rag_logs", [])))

    st.markdown("---")
    _leg_shape = st.session_state.get("pin_shape", "Diamond")
    st.markdown("### 🧭 Layer Legend")
    st.markdown(f"- 🔷 **Sky Blue {_leg_shape} (`#`)**: **Primary Place** (numbered & cited in the answer)")
    st.markdown(f"- 🔴 **Signal Red {_leg_shape} (`!`)**: **Event** or Tactical Focus")
    st.markdown(f"- 🟠 **Dashed Amber {_leg_shape} (`?`)**: **Uncertain Location** (inferred coordinates)")
    st.markdown("- 🔹 **Compact Dot / Cyan Ring**: **Context Place** & Cluster Count")
    st.markdown("- 🎯 **Cyan Crosshair / Corner Brackets**: **Active Target** & Bounding Box")


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
# 5. Right Column (col2): Folium Map Rendering & Click Interception
# Evaluated first so map selections update session_state BEFORE col1 renders,
# eliminating the need for a secondary st.rerun() cascade (AGENTS.md Rule 1.1)
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

            /* ========================================================================== */
            /* 3. TACTICAL HUD MAP SYMBOLOGY (Pins, Reticles, Clusters, BBox, Cards)      */
            /* ========================================================================== */
            .leaflet-container {
                --geoai-selection: #00e5ff;
                --geoai-primary: #38bdf8;
                --geoai-event: #ff4d6d;
                --geoai-uncertain: #ffb020;
                --geoai-context: #7dd3fc;
                --geoai-threat: #f97316;
                --geoai-recon: #a855f7;
                --geoai-friendly: #22c55e;
                --geoai-core: rgba(6, 12, 18, 0.92);
                --geoai-halo: rgba(0, 0, 0, 0.78);
            }
            .geoai-map-light {
                --geoai-selection: #0284c7;
                --geoai-primary: #0369a1;
                --geoai-event: #be123c;
                --geoai-uncertain: #b45309;
                --geoai-context: #475569;
                --geoai-threat: #c2410c;
                --geoai-recon: #7e22ce;
                --geoai-friendly: #15803d;
                --geoai-core: rgba(255, 255, 255, 0.96);
                --geoai-halo: rgba(255, 255, 255, 0.92);
            }

            .geoai-pin {
                background: transparent !important;
                border: none !important;
            }

            /* Target Crosshair Reticle */
            .geoai-reticle {
                position: relative;
                width: 34px;
                height: 34px;
                color: var(--geoai-selection);
                filter: drop-shadow(0 0 4px rgba(0, 229, 255, 0.65));
            }
            .geoai-reticle svg {
                position: absolute;
                inset: 0;
            }
            .geoai-reticle__pulse {
                position: absolute;
                inset: 7px;
                border-radius: 50%;
                border: 2px solid var(--geoai-selection);
                animation: geoai-pulse 2s ease-out infinite;
                pointer-events: none;
            }
            .geoai-reticle__spin {
                display: none;
                position: absolute;
                inset: 2px;
                border-radius: 50%;
                border: 1.5px dashed var(--geoai-selection);
                pointer-events: none;
                animation: geoai-spin 3s linear infinite;
            }
            .geoai-reticle--querying .geoai-reticle__pulse {
                animation-duration: 1s !important;
            }
            .geoai-reticle--querying .geoai-reticle__spin {
                display: block;
            }
            @keyframes geoai-pulse {
                0% { transform: scale(0.6); opacity: 0.9; }
                100% { transform: scale(2.3); opacity: 0; }
            }
            @keyframes geoai-spin {
                to { transform: rotate(360deg); }
            }

            /* Primary / Event / Uncertain Diamond & Teardrop Pins */
            .geoai-diamond {
                width: 20px;
                height: 20px;
                margin: 1px;
                transform: rotate(45deg);
                background: var(--geoai-core);
                border: 2px solid var(--c, var(--geoai-primary));
                border-radius: 3px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.15s ease, border-radius 0.15s ease;
                box-shadow: 0 0 0 1px var(--geoai-halo), 0 0 10px rgba(0, 229, 255, 0.35);
            }
            .geoai-diamond > span {
                transform: rotate(-45deg);
                font: 700 11px/1 'JetBrains Mono', 'Fira Code', monospace;
                color: var(--c, var(--geoai-primary));
            }
            .geoai-pin:hover .geoai-diamond {
                transform: rotate(45deg) scale(1.18);
            }
            /* Teardrop Pin Shape Variant (both initial class and 0ms container toggle) */
            .geoai-diamond.geoai-teardrop,
            .geoai-shape-teardrop .geoai-diamond {
                width: 24px;
                height: 24px;
                margin: -1px;
                border-radius: 50% 50% 50% 0;
                transform: translateY(-14px) rotate(-45deg);
            }
            .geoai-diamond.geoai-teardrop > span,
            .geoai-shape-teardrop .geoai-diamond > span {
                transform: rotate(45deg);
            }
            .geoai-pin:hover .geoai-diamond.geoai-teardrop,
            .geoai-shape-teardrop .geoai-pin:hover .geoai-diamond {
                transform: translateY(-14px) rotate(-45deg) scale(1.15);
            }
            .geoai-diamond--uncertain {
                border-style: dashed;
            }
            .geoai-badge-unc {
                position: absolute;
                top: -6px;
                right: -6px;
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: var(--geoai-uncertain);
                color: #0b0f17;
                font: 700 10px/14px 'JetBrains Mono', monospace;
                text-align: center;
                font-style: normal;
                box-shadow: 0 0 0 1.5px var(--geoai-halo);
                transition: top 0.15s ease, right 0.15s ease;
            }
            .geoai-pin--teardrop .geoai-badge-unc,
            .geoai-shape-teardrop .geoai-badge-unc {
                top: -20px;
                right: -7px;
            }

            /* Lightweight 8px Context Dot */
            .geoai-dot {
                width: 8px;
                height: 8px;
                margin: 1px;
                border-radius: 50%;
                background: var(--geoai-context);
                opacity: 0.88;
                box-shadow: 0 0 0 1.5px var(--geoai-halo);
                transition: transform 0.12s ease;
            }
            .geoai-pin:hover .geoai-dot {
                transform: scale(1.35);
                opacity: 1;
            }

            /* Bounding Box L-shaped Corner Brackets (Exact Vertex Alignment) */
            .geoai-corner {
                position: relative;
                width: 16px;
                height: 16px;
                margin: 0;
                padding: 0;
                display: block;
                line-height: 0;
                color: var(--geoai-selection);
                filter: drop-shadow(0 0 3px var(--geoai-selection));
            }
            .geoai-corner svg {
                position: absolute;
                top: 0;
                left: 0;
                width: 16px;
                height: 16px;
                display: block;
            }
            .geoai-corner--dim {
                opacity: 0.40;
                filter: none;
            }

            /* Context Cluster Bubbles */
            .geoai-cluster-wrap {
                background: transparent !important;
                border: none !important;
            }
            .geoai-cluster {
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                background: var(--geoai-core);
                border: 1.5px solid var(--geoai-selection);
                color: var(--geoai-selection);
                font: 700 12px/1 'JetBrains Mono', monospace;
            }
            .geoai-cluster--s1 {
                width: 28px;
                height: 28px;
                box-shadow: 0 0 0 4px rgba(0, 229, 255, 0.14), 0 0 10px rgba(0, 229, 255, 0.30);
            }
            .geoai-cluster--s2 {
                width: 36px;
                height: 36px;
                box-shadow: 0 0 0 5px rgba(0, 229, 255, 0.16), 0 0 12px rgba(0, 229, 255, 0.35);
            }
            .geoai-cluster--s3 {
                width: 44px;
                height: 44px;
                box-shadow: 0 0 0 7px rgba(0, 229, 255, 0.18), 0 0 16px rgba(0, 229, 255, 0.40);
            }

            /* Bounding Box States (Selected Marching Ants, Querying Pulse + Scanline, Answered Ultra-Transparent) */
            path.geoai-bbox--selected {
                animation: geoai-ants 1s linear infinite;
            }
            path.geoai-bbox--querying {
                animation: geoai-ants 0.6s linear infinite, geoai-qfill 1.6s ease-in-out infinite !important;
            }
            path.geoai-bbox--answered {
                pointer-events: none !important;
            }
            @keyframes geoai-ants {
                to { stroke-dashoffset: -12; }
            }
            @keyframes geoai-qfill {
                0%, 100% { fill-opacity: 0.08; }
                50% { fill-opacity: 0.20; }
            }
            .geoai-scan-svg {
                pointer-events: none !important;
                overflow: hidden;
            }
            .geoai-scan-line {
                stroke: var(--geoai-selection);
                stroke-width: 2;
                opacity: 0.9;
                vector-effect: non-scaling-stroke;
                filter: drop-shadow(0 0 6px var(--geoai-selection));
            }
            .geoai-scan-band {
                fill: url(#geoai-scan-grad);
                pointer-events: none;
            }

            /* Pin Shape Segmented Control inside PinFilterControl */
            .geoai-shape-seg {
                display: flex;
                gap: 4px;
                margin-top: 2px;
            }
            .geoai-shape-btn {
                flex: 1;
                background: rgba(11, 15, 23, 0.65);
                color: var(--geoai-selection);
                border: 1px solid rgba(0, 229, 255, 0.28);
                border-radius: 3px;
                padding: 2px 6px;
                font: 700 10px/1.3 'JetBrains Mono', 'Fira Code', monospace;
                cursor: pointer;
                transition: background 0.12s ease, border-color 0.12s ease;
            }
            .geoai-shape-btn.active {
                background: rgba(0, 229, 255, 0.22);
                border-color: var(--geoai-selection);
                box-shadow: 0 0 6px rgba(0, 229, 255, 0.25);
            }

            /* Unified Tactical Popup Card */
            .geoai-card {
                font: 12px/1.45 'JetBrains Mono', 'Fira Code', monospace;
                min-width: 200px;
            }
            .geoai-card__head {
                padding: 2px 4px 6px 8px;
                margin-bottom: 6px;
                border-bottom: 1px solid rgba(0, 229, 255, 0.25);
                border-left: 3px solid var(--c, var(--geoai-primary));
            }
            .geoai-card__chip {
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.08em;
                color: var(--c, var(--geoai-primary));
                text-transform: uppercase;
            }
            .geoai-card__title {
                font-weight: 700;
                font-size: 12.5px;
            }
            .geoai-card__body {
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 2px 10px;
                padding: 2px 4px;
            }
            .geoai-card__body > span {
                opacity: 0.78;
                font-size: 11px;
            }
            .geoai-card__footer {
                margin-top: 6px;
                padding-top: 5px;
                border-top: 1px dashed rgba(0, 229, 255, 0.25);
                font-size: 11px;
                opacity: 0.85;
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
                    "fillOpacity": 0.10,
                    "weight": 2,
                    "dashArray": "6, 6",
                    "className": "geoai-bbox geoai-bbox--selected"
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
    cluster_icon_js = """
    function(cluster) {
        var count = cluster.getChildCount();
        var tier = count < 10 ? 's1' : (count < 100 ? 's2' : 's3');
        var px = count < 10 ? 28 : (count < 100 ? 36 : 44);
        return L.divIcon({
            html: "<div class='geoai-cluster geoai-cluster--" + tier + "'>" + count + "</div>",
            className: 'marker-cluster geoai-cluster-wrap',
            iconSize: L.point(px, px)
        });
    }
    """
    context_cluster = MarkerCluster(
        name="Context Cluster",
        control=False,
        icon_create_function=cluster_icon_js
    ).add_to(tactical_layers_fg)

    # Render visually distinct marker(s) or bounding box for all currently selected points
    is_querying = bool(st.session_state.get("pending_query"))
    sync_bbox_state = "none"
    sync_bbox_bounds = None

    if st.session_state.get("bounding_box"):
        b_ll, b_ur = st.session_state.bounding_box
        sync_bbox_bounds = [[b_ll[0], b_ll[1]], [b_ur[0], b_ur[1]]]
        is_answered_bbox = (
            st.session_state.get("answered_bbox") == st.session_state.bounding_box
            or any(
                isinstance(l.data, dict) and l.data.get("role") == "query_bbox"
                for l in st.session_state.get("map_layers", [])
            )
        )
        bb_state = "answered" if is_answered_bbox else ("querying" if is_querying else "selected")
        sync_bbox_state = bb_state
        bb_cfg = bbox_style(bb_state)

        # 1. Lower-Left (SW) Corner Bracket
        folium.Marker(
            location=[b_ll[0], b_ll[1]],
            tooltip=f"★ Bounding Box Lower-Left (SW): {b_ll[0]:.4f}°, {b_ll[1]:.4f}°",
            popup=folium.Popup(
                _popup_card(
                    title="★ BBOX LOWER-LEFT (SW)",
                    rows=[("LAT", f"{b_ll[0]:.5f}°"), ("LNG", f"{b_ll[1]:.5f}°")],
                    chip="AOI CORNER // SW",
                    footer=f"Protocol: {t_cfg['name']}",
                    token="selection",
                ),
                max_width=250
            ),
            icon=make_pin_icon(role="bbox_corner", corner="sw", dimmed=is_answered_bbox)
        ).add_to(selection_fg)

        # 2. Upper-Right (NE) Corner Bracket
        folium.Marker(
            location=[b_ur[0], b_ur[1]],
            tooltip=f"★ Bounding Box Upper-Right (NE): {b_ur[0]:.4f}°, {b_ur[1]:.4f}°",
            popup=folium.Popup(
                _popup_card(
                    title="★ BBOX UPPER-RIGHT (NE)",
                    rows=[("LAT", f"{b_ur[0]:.5f}°"), ("LNG", f"{b_ur[1]:.5f}°")],
                    chip="AOI CORNER // NE",
                    footer=f"Protocol: {t_cfg['name']}",
                    token="selection",
                ),
                max_width=250
            ),
            icon=make_pin_icon(role="bbox_corner", corner="ne", dimmed=is_answered_bbox)
        ).add_to(selection_fg)

        # 2b. Passive NW & SE Corner Brackets to complete the 4-corner HUD frame
        folium.Marker(
            location=[b_ur[0], b_ll[1]],
            icon=make_pin_icon(role="bbox_corner", corner="nw", dimmed=is_answered_bbox),
        ).add_to(selection_fg)
        folium.Marker(
            location=[b_ll[0], b_ur[1]],
            icon=make_pin_icon(role="bbox_corner", corner="se", dimmed=is_answered_bbox),
        ).add_to(selection_fg)

        # 3. Visual Tactical Bounding Box Rectangle (skip duplicate if agent query_bbox layer already renders it)
        has_agent_query_bbox = any(
            isinstance(l.data, dict) and l.data.get("role") == "query_bbox"
            for l in st.session_state.get("map_layers", [])
        )
        if not has_agent_query_bbox:
            rect_kwargs = {
                "bounds": [[b_ll[0], b_ll[1]], [b_ur[0], b_ur[1]]],
                "color": bb_cfg["color"],
                "weight": bb_cfg["weight"],
                "opacity": bb_cfg["opacity"],
                "dash_array": bb_cfg["dash_array"],
                "fill": bb_cfg["fill"],
                "fill_color": bb_cfg["fill_color"],
                "fill_opacity": bb_cfg["fill_opacity"],
                "interactive": bb_cfg["interactive"],
                "className": bb_cfg["class_name"],
            }
            if bb_cfg["interactive"]:
                rect_kwargs["tooltip"] = f"★ Bounding Box: [{b_ll[0]:.4f}°, {b_ll[1]:.4f}°] to [{b_ur[0]:.4f}°, {b_ur[1]:.4f}°]"
                rect_kwargs["popup"] = folium.Popup(
                    _popup_card(
                        title="★ TWO-POINT BOUNDING BOX",
                        rows=[
                            ("SW", f"{b_ll[0]:.4f}°, {b_ll[1]:.4f}°"),
                            ("NE", f"{b_ur[0]:.4f}°, {b_ur[1]:.4f}°"),
                        ],
                        chip="SECTOR AOI",
                        footer="Ready for tactical query",
                        token="selection",
                    ),
                    max_width=260
                )
            folium.Rectangle(**rect_kwargs).add_to(selection_fg)

    elif st.session_state.get("selected_points"):
        s_lat, s_lng = st.session_state.selected_points[0]
        popup_html = _popup_card(
            title="★ TARGET POINT",
            rows=[("LAT", f"{s_lat:.5f}°"), ("LNG", f"{s_lng:.5f}°")],
            chip="ACTIVE TARGET",
            footer=f"Protocol: {t_cfg['name']}",
            token="selection",
        )
        folium.Marker(
            location=[s_lat, s_lng],
            tooltip=f"★ Target Point ({s_lat:.4f}°, {s_lng:.4f}°)",
            popup=folium.Popup(popup_html, max_width=250, show=True),
            icon=make_pin_icon(role="target")
        ).add_to(selection_fg)

    # Attach SelectionSyncHook so every feature_group_to_add update purges stale client-side
    # preview layers (corner brackets, drawnItems, instant target markers) and syncs pin shape
    SelectionSyncHook(
        has_selection=bool(st.session_state.get("selected_points") or st.session_state.get("bounding_box")),
        clear_seq=st.session_state.get("selection_clear_seq", 0),
        bbox_state=sync_bbox_state,
        is_querying=is_querying,
        bbox_bounds=sync_bbox_bounds,
        pin_shape=st.session_state.get("pin_shape", "Diamond"),
    ).add_to(selection_fg)

    # Render all persistent MapLayer objects from session state
    for layer in st.session_state.map_layers:
        try:
            l_data = layer.data if isinstance(layer.data, dict) else {}
            l_role = l_data.get("role")
            l_kind = l_data.get("kind", "place")
            l_unc = l_data.get("uncertain")
            l_num = l_data.get("number")
            l_corner = l_data.get("corner")

            if layer.layer_type == "marker":
                is_context = (
                    layer.color == "cadetblue"
                    or l_role == "context"
                    or "context" in (layer.name or "").lower()
                ) and not l_unc
                target_parent = context_cluster if is_context else tactical_layers_fg
                pin_role = "context" if is_context else (l_role or "primary")
                folium.Marker(
                    location=[l_data["lat"], l_data["lng"]],
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300),
                    icon=make_pin_icon(
                        kind=l_kind,
                        role=pin_role,
                        uncertain=l_unc,
                        number=l_num,
                        color_name=layer.color,
                        corner=l_corner,
                        dimmed=(pin_role == "bbox_corner"),
                    )
                ).add_to(target_parent)

            elif layer.layer_type == "circle":
                _, c_hex = color_token(layer.color, layer_type="circle", role=l_role, kind=l_kind)
                is_target_buf = (l_role == "target_buffer" or layer.color == "cyan")
                folium.Circle(
                    location=[l_data["lat"], l_data["lng"]],
                    radius=l_data.get("radius", 10000),
                    color=c_hex,
                    weight=1.0 if is_target_buf else 1.5,
                    dash_array="4, 4" if is_target_buf else None,
                    fill=True,
                    fill_color=c_hex,
                    fill_opacity=0.04 if is_target_buf else 0.08,
                    tooltip=layer.tooltip or layer.name,
                    popup=folium.Popup(layer.popup or layer.name, max_width=300)
                ).add_to(tactical_layers_fg)

            elif layer.layer_type == "polygon":
                is_query_bbox = (
                    l_role == "query_bbox"
                    or "bbox" in (layer.name or "").lower()
                    or "bounding box" in (layer.name or "").lower()
                )
                if is_query_bbox:
                    ans_cfg = bbox_style("answered")
                    folium.Polygon(
                        locations=l_data.get("coordinates", []),
                        color=ans_cfg["color"],
                        weight=ans_cfg["weight"],
                        opacity=ans_cfg["opacity"],
                        dash_array=ans_cfg["dash_array"],
                        fill=ans_cfg["fill"],
                        fill_color=ans_cfg["fill_color"],
                        fill_opacity=ans_cfg["fill_opacity"],
                        interactive=ans_cfg["interactive"],
                        className=ans_cfg["class_name"],
                    ).add_to(tactical_layers_fg)
                else:
                    _, p_hex = color_token(layer.color, layer_type="polygon", role=l_role, kind=l_kind)
                    folium.Polygon(
                        locations=l_data.get("coordinates", []),
                        color=p_hex,
                        weight=1.5,
                        fill=True,
                        fill_color=p_hex,
                        fill_opacity=0.08,
                        tooltip=layer.tooltip or layer.name,
                        popup=folium.Popup(layer.popup or layer.name, max_width=300)
                    ).add_to(tactical_layers_fg)

            elif layer.layer_type in ["rectangle", "box"]:
                is_query_bbox = (
                    l_role == "query_bbox"
                    or "bbox" in (layer.name or "").lower()
                    or "bounding box" in (layer.name or "").lower()
                )
                if is_query_bbox:
                    ans_cfg = bbox_style("answered")
                    folium.Rectangle(
                        bounds=l_data.get("bounds", l_data.get("coordinates", [])),
                        color=ans_cfg["color"],
                        weight=ans_cfg["weight"],
                        opacity=ans_cfg["opacity"],
                        dash_array=ans_cfg["dash_array"],
                        fill=ans_cfg["fill"],
                        fill_color=ans_cfg["fill_color"],
                        fill_opacity=l_data.get("fill_opacity", ans_cfg["fill_opacity"]),
                        interactive=ans_cfg["interactive"],
                        className=ans_cfg["class_name"],
                    ).add_to(tactical_layers_fg)
                else:
                    _, r_hex = color_token(layer.color, layer_type="rectangle", role=l_role, kind=l_kind)
                    folium.Rectangle(
                        bounds=l_data.get("bounds", l_data.get("coordinates", [])),
                        color=r_hex,
                        weight=1.5,
                        fill=True,
                        fill_color=r_hex,
                        fill_opacity=l_data.get("fill_opacity", 0.08),
                        tooltip=layer.tooltip or layer.name,
                        popup=folium.Popup(layer.popup or layer.name, max_width=300)
                    ).add_to(tactical_layers_fg)

            elif layer.layer_type == "geojson":
                # Upgrade 1: GeoJSON administrative boundary outline rendering
                is_ctx_geo = (layer.color == "cadetblue" or l_role == "context")
                _, geo_color = color_token(
                    "cadetblue" if is_ctx_geo else "blue",
                    layer_type="geojson",
                    role="context" if is_ctx_geo else "primary",
                )
                folium.GeoJson(
                    layer.data,
                    name=layer.name,
                    style_function=lambda feature, c=geo_color, ctx=is_ctx_geo: {
                        "fillColor": c,
                        "color": c,
                        "weight": 1.0 if ctx else 1.5,
                        "fillOpacity": 0.02 if ctx else 0.08,
                        "dashArray": "1, 4" if ctx else None,
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
            st.session_state.answered_bbox = None
            st.session_state.active_point = upper_right
            # Mark drawing timestamp to absorb immediate mouseup canvas click
            st.session_state.last_drawing_time = time.time()
            if map_data and map_data.get("last_clicked"):
                cd = map_data["last_clicked"]
                st.session_state.last_clicked = (round(cd["lat"], 5), round(cd["lng"], 5))

        elif gtype == "Point" and coords and len(coords) >= 2:
            st.session_state.last_processed_drawing = active_drawing
            st.session_state.ignored_drawing = None

            lat = round(float(coords[1]), 5)
            lng = round(float(coords[0]), 5)
            pt = (lat, lng)
            st.session_state.selection_mode = "Point Selection"
            st.session_state.selected_points = [pt]
            st.session_state.bounding_box = None
            st.session_state.answered_bbox = None
            st.session_state.active_point = pt
            st.session_state.last_drawing_time = time.time()
            if map_data and map_data.get("last_clicked"):
                st.session_state.last_clicked = pt

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
                st.session_state.answered_bbox = None
                st.session_state.active_point = clicked_coord


# ------------------------------------------------------------------------------
# 6. Left Column (col1): Chat Interface & Docked Target Pill
# Evaluated second: reads the fresh selected_points and renders the docked HUD
# in this exact execution cycle without requiring a second st.rerun().
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
                        source_items_html = []
                        for s_idx, src_doc in enumerate(msg_sources):
                            cit_meta = msg_citations.get(src_doc, {})
                            cit_id = cit_meta.get("id")
                            cit_article = cit_meta.get("article")
                            cit_section = cit_meta.get("section")
                            cached_meta = st.session_state.get(f"src_meta_{m_idx}_{s_idx}", {})
                            disp_article = _clean_doc_title(cit_article or cached_meta.get("document"))
                            disp_section = _clean_doc_title(cit_section or cached_meta.get("section"))

                            # Resolve document title and text from local graph if available
                            passage = cached_meta or get_source_text(src_doc)
                            if passage and isinstance(passage, dict):
                                st.session_state[f"src_meta_{m_idx}_{s_idx}"] = passage
                                if not disp_article:
                                    disp_article = _clean_doc_title(passage.get("document"))
                                if not disp_section:
                                    disp_section = _clean_doc_title(passage.get("section"))
                            else:
                                st.session_state[f"src_meta_{m_idx}_{s_idx}"] = {"document": "", "section": ""}

                            if cit_id and disp_article:
                                title_str = f"[{cit_id}] {disp_article}"
                            elif disp_article and disp_article != src_doc:
                                title_str = disp_article
                            else:
                                title_str = f"Source Document #{s_idx + 1}"

                            sec_str = f" — <i>{disp_section}</i>" if disp_section else ""

                            if passage and passage.get("text"):
                                doc_name = _clean_doc_title(passage.get("document") or cit_article or f"Document #{s_idx + 1}")
                                sec_name = _clean_doc_title(passage.get("section") or cit_section)
                                para_num = passage.get("paragraph_index")
                                header_parts = [f"<b>{doc_name}</b>"]
                                if sec_name:
                                    header_parts.append(f"Section: <i>{sec_name}</i>")
                                if para_num is not None:
                                    header_parts.append(f"Paragraph #{para_num + 1}")
                                header_str = " | ".join(header_parts)
                                raw_text = passage["text"]
                                spans = cit_meta.get("text_spans") or cit_meta.get("char_offsets") or passage.get("text_spans") or []
                                highlighted_text = _highlight_text_spans(raw_text, spans)
                                body_html = (
                                    f"<div style='background: rgba(14, 22, 32, 0.7); border-left: 3px solid #00e5ff; padding: 10px 14px; border-radius: 4px; font-size: 0.9em; margin-top: 8px; margin-bottom: 4px; line-height: 1.5;'>"
                                    f"<div style='font-weight: 600; margin-bottom: 6px; color: #38bdf8;'>{header_str}</div>"
                                    f"<div style='color: #cbd5e1;'>{highlighted_text}</div>"
                                    f"</div>"
                                )
                            else:
                                body_html = (
                                    f"<div style='background: rgba(14, 22, 32, 0.7); border-left: 3px solid #64748b; padding: 10px 14px; border-radius: 4px; font-size: 0.88em; margin-top: 8px; margin-bottom: 4px; color: #94a3b8;'>"
                                    f"<div style='font-weight: 600; margin-bottom: 4px; color: #cbd5e1;'>Notice</div>"
                                    f"<div>(No inline corpus text available for this reference)</div>"
                                    f"</div>"
                                )

                            source_items_html.append(
                                f"<details class='geoai-source-detail'>"
                                f"<summary class='geoai-source-summary'>"
                                f"<span class='geoai-source-title'>📄 <b>{title_str}</b>{sec_str}</span>"
                                f"<span class='geoai-source-badge'>Inspect ▾</span>"
                                f"</summary>"
                                f"{body_html}"
                                f"</details>"
                            )

                        st.markdown("".join(source_items_html), unsafe_allow_html=True)

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
                    if is_bbox and st.session_state.get("bounding_box"):
                        st.session_state.answered_bbox = st.session_state.bounding_box
                    if getattr(recon_resp, "map_bbox", None):
                        st.session_state.last_map_bbox = recon_resp.map_bbox
                    if not st.session_state.get("keep_points", False):
                        clear_spatial_selection()
                    st.rerun()
    else:
        # Instant HUD container placeholder for 0ms client-side pre-rendering
        st.markdown("<div id='geoai-instant-pill-slot' style='display: none;'></div>", unsafe_allow_html=True)

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
        if pq.get("bounding_box"):
            st.session_state.answered_bbox = pq.get("bounding_box")
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
