"""
Lightweight REST API Daemon for NetReach v0.7.0 (`netreach serve`).

Provides headless REST endpoints for lateral movement attack path reachability auditing,
neural path tracing, and automated defensive chokepoint remediation.
"""

import sys
from typing import Dict, List, Optional, Any, Union

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def check_fastapi_available():
    """Ensure FastAPI and Uvicorn dependencies are installed."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and Uvicorn are required to run the NetReach API server.\n"
            "Please install the optional API dependencies using:\n"
            "    pip install \"netreach[api]\""
        )


if FASTAPI_AVAILABLE:
    class AuditRequest(BaseModel):
        graph: Union[str, Dict[str, Any], List[Dict[str, Any]]] = Field(
            ...,
            description="Graph file path, JSON edge list, or BloodHound/Cloud IAM object."
        )
        source: str = Field(..., description="Attacker entry point principal/node.")
        target: str = Field(..., description="High-value crown jewel objective.")
        depths: Optional[List[int]] = Field(None, description="Optional recurrent evaluation depths T.")
        int8: Optional[bool] = Field(False, description="Enable dynamic INT8 quantization on CPU.")
        device: Optional[str] = Field(None, description="Compute device ('cpu' or 'cuda').")

    class HardenRequest(BaseModel):
        graph: Union[str, Dict[str, Any], List[Dict[str, Any]]] = Field(
            ...,
            description="Graph file path, JSON edge list, or BloodHound/Cloud IAM object."
        )
        source: str = Field(..., description="Attacker entry point principal/node.")
        target: str = Field(..., description="High-value crown jewel objective.")
        top_k: Optional[int] = Field(3, description="Number of highest-impact chokepoints to return.")
        depth: Optional[int] = Field(None, description="Recurrent evaluation steps T.")
        int8: Optional[bool] = Field(False, description="Enable dynamic INT8 quantization on CPU.")
        device: Optional[str] = Field(None, description="Compute device ('cpu' or 'cuda').")


def create_app(
    engine: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    use_int8: bool = False,
) -> "FastAPI":
    """
    Create and configure the NetReach FastAPI application instance.
    """
    check_fastapi_available()
    from netreach.engine import NetReachEngine
    from netreach import __version__

    app = FastAPI(
        title="NetReach Enterprise API",
        version=__version__,
        description="REST API for deep lateral movement reachability auditing and neural chokepoint remediation.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize shared engine instance
    if engine is None:
        engine = NetReachEngine(
            checkpoint_path=checkpoint_path,
            device=device,
            use_int8=use_int8,
        )
    app.state.engine = engine

    @app.get("/api/v1/health")
    def health_check():
        eng = app.state.engine
        return {
            "status": "ok",
            "version": __version__,
            "parameter_count": eng.param_count,
            "device": str(eng.device),
            "use_int8": eng.use_int8,
            "model": "ChallengerLatentNetV4",
        }

    @app.post("/api/v1/audit")
    def audit_attack_path(req: AuditRequest):
        try:
            eng = app.state.engine
            eval_depths = tuple(req.depths) if req.depths else (4, 8, 12, 16)
            res = eng.audit(
                graph_source=req.graph,
                source=req.source,
                target=req.target,
                depths=eval_depths,
            )
            return {
                "source": res.source,
                "target": res.target,
                "risk_level": res.risk_level,
                "probabilities": {str(k): round(v, 4) for k, v in res.probabilities.items()},
                "cosine_similarities": {str(k): round(v, 4) for k, v in res.cosine_similarities.items()},
                "shortest_hop_count": res.shortest_hop_count,
                "act_steps": round(res.act_steps, 2),
                "bridge_node": res.bridge_node,
                "neural_path": res.neural_path,
                "path_relations": res.path_relations,
                "is_valid_walk": res.is_valid_walk,
                "inference_latency_ms": round(res.inference_latency_ms, 2),
                "subgraph_nodes": res.subgraph_nodes,
                "subgraph_edges": res.subgraph_edges,
                "oversmoothed": res.oversmoothed,
            }
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    @app.post("/api/v1/harden")
    def harden_chokepoints(req: HardenRequest):
        try:
            eng = app.state.engine
            res = eng.harden(
                graph_source=req.graph,
                source=req.source,
                target=req.target,
                top_k=req.top_k or 3,
                steps=req.depth,
            )
            return {
                "source": res.source,
                "target": res.target,
                "baseline_risk_level": res.baseline_risk_level,
                "baseline_probability": round(res.baseline_probability, 4),
                "act_steps": round(res.act_steps, 2),
                "bridge_node": res.bridge_node,
                "attack_path": res.attack_path,
                "path_relations": res.path_relations,
                "analysis_latency_ms": round(res.analysis_latency_ms, 2),
                "subgraph_nodes": res.subgraph_nodes,
                "subgraph_edges": res.subgraph_edges,
                "chokepoints": [
                    {
                        "rank": cp.rank,
                        "source": cp.source,
                        "target": cp.target,
                        "relation": cp.relation,
                        "ad_relation": cp.ad_relation,
                        "baseline_prob": round(cp.baseline_prob, 4),
                        "ablated_prob": round(cp.ablated_prob, 4),
                        "delta_p": round(cp.delta_p, 4),
                        "mitigation_cmd": cp.mitigation_cmd,
                        "cypher_query": cp.cypher_query,
                        "description": cp.description,
                    }
                    for cp in res.chokepoints
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    use_int8: bool = False,
):
    """Launch NetReach REST API daemon."""
    check_fastapi_available()
    app = create_app(
        checkpoint_path=checkpoint_path,
        device=device,
        use_int8=use_int8,
    )
    uvicorn.run(app, host=host, port=port)
